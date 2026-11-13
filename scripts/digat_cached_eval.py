"""Checkpoint-bound official DIGAT evaluation with reusable news/context/pair scores.

DIGAT user representations depend on each candidate. This cache never replaces
dual-graph interaction with an independent user vector/dot-product surrogate.
"""
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import scheduled_cooling as cooling
import time
import numpy as np
import torch
from naml_cached_eval import (_array_hash_into, _file_hash, _json_bytes, _save_json,
                              _save_npy, model_fingerprint, metrics_from_ranks, METRIC_NAMES)

ROOT = Path(__file__).resolve().parents[1]


def news_categories(corpus, data_root):
    result = np.full(len(corpus.news_ID_dict), len(corpus.category_dict), dtype=np.int64)
    for split in ("train", "dev", "test"):
        with (Path(data_root) / split / "news.tsv").open(encoding="utf-8") as stream:
            for line in stream:
                fields = line.rstrip("\r\n").split("\t")
                result[corpus.news_ID_dict[fields[0]]] = corpus.category_dict[fields[1]]
    if np.any(result[1:] >= len(corpus.category_dict)):
        raise ValueError("Incomplete news/category mapping")
    return result


def corpus_identity(corpus, categories):
    digest = hashlib.sha256()
    digest.update(_json_bytes({"news_ID_dict": corpus.news_ID_dict,
                              "category_dict": corpus.category_dict,
                              "history_length": corpus.max_history_num}))
    for name in ("news_title_text", "news_title_mask", "news_node_ID", "news_graph", "news_graph_mask"):
        _array_hash_into(digest, name, getattr(corpus, name))
    _array_hash_into(digest, "news_category_indices", categories)
    return digest.hexdigest()


def history_graph(history, categories, category_count):
    """Vectorized equivalent of official MIND_Corpus's news/topic graph loop."""
    history = np.asarray(history, dtype=np.int64)
    count = len(history)
    graph = np.eye(count + category_count, dtype=bool)
    mask = np.zeros(category_count + 1, dtype=bool)
    indices = np.full(count, category_count, dtype=np.int64)
    valid = np.flatnonzero(history)
    if len(valid):
        cats = categories[history[valid]]
        if np.any(cats >= category_count):
            raise ValueError("Unknown category for history")
        indices[valid] = cats
        present = np.unique(cats)
        mask[present] = True
        graph[valid, count + cats] = True
        graph[count + cats, valid] = True
        graph[np.ix_(valid, valid)] |= cats[:, None] == cats[None, :]
        graph[np.ix_(count + present, count + present)] = True
    return graph, mask, indices


class DigatCachedEvaluator:
    def __init__(self, model, corpus, categories, cache_dir, news_batch_size=256, score_batch_size=16, projection_cache=False, sparse_eval=False):
        if type(model.graph_encoder).__name__ != "DIGAT":
            raise ValueError("Requires the official DIGAT graph encoder")
        if next(model.parameters()).dtype != torch.float32:
            raise ValueError("This equivalence-checked path requires float32")
        self.model, self.corpus = model, corpus
        self.use_projection_cache = bool(projection_cache)
        self.use_sparse_eval = bool(sparse_eval)
        self.sparse_verified = False
        self.categories = np.asarray(categories, dtype=np.int64)
        self.device = next(model.parameters()).device
        self.history_length = int(corpus.max_history_num)
        self.category_count = len(corpus.category_dict)
        self.dimension = int(model.news_embedding_dim)
        self.news_batch_size, self.score_batch_size = int(news_batch_size), int(score_batch_size)
        if min(self.news_batch_size, self.score_batch_size) < 1:
            raise ValueError("Positive batch sizes required")
        self.model_hash = model_fingerprint(model)
        self.provenance = {
            "schema": 1, "model": model.model_name, "model_state_sha256": self.model_hash,
            "corpus_sha256": corpus_identity(corpus, self.categories),
            "dtype": "float32", "device": str(self.device), "torch_version": torch.__version__,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32, "pair_key": "(raw user id, last H news with tail padding, candidate news)",
            "source_sha256": {name: _file_hash(ROOT / "third_party" / "DIGAT" / name)
                              for name in ("model.py", "newsEncoders.py", "graphEncoders.py", "layers.py")},
            "cache_implementation_sha256": _file_hash(__file__),
            "cache_utilities_sha256": _file_hash(ROOT / "scripts/naml_cached_eval.py"),
            "projection_cache": self.use_projection_cache,
            "projection_implementation_sha256": _file_hash(ROOT / "scripts/digat_projection_cache.py") if self.use_projection_cache else None,
            "graph_backend": "sparse_csr" if self.use_sparse_eval else "dense",
            "sparse_score_batch_size": self.score_batch_size if self.use_sparse_eval else None,
            "sparse_implementation_sha256": _file_hash(ROOT / "scripts/digat_sparse_attention.py") if self.use_sparse_eval else None,
        }
        self.identity = hashlib.sha256(_json_bytes(self.provenance)).hexdigest()
        self.cache_dir = Path(cache_dir) / self.identity
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.user_keys = []
        self.user_lookup = {}
        self.pair_keys = np.zeros(0, dtype=np.int64)
        self.pair_scores = np.zeros(0, dtype=np.float32)
        self.timings = {"news_encoding_seconds": 0.0, "context_encoding_seconds": 0.0, "graph_scoring_seconds": 0.0}
        self.model.eval()
        self.cache_loaded = (self.cache_dir / "manifest.json").exists()
        if self.cache_loaded:
            meta = json.loads((self.cache_dir / "manifest.json").read_text(encoding="utf-8"))
            if meta["identity"] != self.identity or meta["provenance"] != self.provenance:
                raise ValueError("Cache identity mismatch")
            for name, expected in meta["files"].items():
                if _file_hash(self.cache_dir / name) != expected:
                    raise ValueError("Cache file changed: " + name)
            self.news = np.load(self.cache_dir / "news.npy", allow_pickle=False)
            self.context = np.load(self.cache_dir / "context.npy", allow_pickle=False)
            self.user_keys = json.loads((self.cache_dir / "user_keys.json").read_text(encoding="utf-8"))
            self.pair_keys = np.load(self.cache_dir / "pair_keys.npy", allow_pickle=False)
            self.pair_scores = np.load(self.cache_dir / "pair_scores.npy", allow_pickle=False)
        else:
            if any(self.cache_dir.iterdir()):
                raise ValueError("Incomplete cache; use a new cache root")
            self.news = self._encode_news()
            self.context = self._encode_context()
        expected = (len(corpus.news_ID_dict), self.dimension)
        if self.news.shape != expected or self.context.shape != expected:
            raise ValueError("News/context cache shape mismatch")
        if not np.isfinite(self.news).all() or not np.isfinite(self.context).all():
            raise ValueError("Nonfinite news/context cache")
        self.news_device = torch.as_tensor(self.news, device=self.device)
        self.context_device = torch.as_tensor(self.context, device=self.device)
        self._rebuild_users()
        if len(self.pair_keys) != len(self.pair_scores) or np.any(np.diff(self.pair_keys) <= 0):
            raise ValueError("Invalid sorted pair cache")
        if len(self.pair_keys) and (self.pair_keys[-1] >> 32) >= len(self.user_keys):
            raise ValueError("Pair cache user index out of range")
        if not np.isfinite(self.pair_scores).all():
            raise ValueError("Nonfinite pair cache")
        self.projections = None
        if self.use_projection_cache:
            from digat_projection_cache import DigatLayerZeroCache
            self.projections = DigatLayerZeroCache(model, self.news, self.cache_dir / "layer_zero", self.model_hash, self.device)
            self.timings["projection_cache_seconds"] = self.projections.elapsed_seconds
        self.sparse = None
        if self.use_sparse_eval:
            from digat_sparse_attention import SparseDIGATScore
            self.sparse = SparseDIGATScore(model, self.projections)
        self._save_cache()

    def _encode_news(self):
        begin = time.monotonic()
        count = len(self.corpus.news_ID_dict)
        result = np.empty((count, self.dimension), np.float32)
        with torch.inference_mode():
            for start in range(0, count, self.news_batch_size):
                end = min(count, start + self.news_batch_size)
                titles = torch.as_tensor(self.corpus.news_title_text[start:end], device=self.device).unsqueeze(1)
                masks = torch.as_tensor(self.corpus.news_title_mask[start:end], device=self.device).unsqueeze(1)
                result[start:end] = self.model.news_encoder(titles, masks).squeeze(1).cpu().numpy()
                cooling.safe_boundary()
        self.timings["news_encoding_seconds"] = time.monotonic() - begin
        return result

    def _encode_context(self):
        begin = time.monotonic()
        result = np.empty_like(self.news)
        with torch.inference_mode():
            for start in range(0, len(result), self.news_batch_size):
                end = min(len(result), start + self.news_batch_size)
                values = torch.as_tensor(self.news[self.corpus.news_node_ID[start:end]], device=self.device)
                masks = torch.as_tensor(self.corpus.news_graph_mask[start:end], device=self.device)
                result[start:end] = self.model.graph_encoder.compute_news_graph_context(values, masks).cpu().numpy()
                cooling.safe_boundary()
        self.timings["context_encoding_seconds"] = time.monotonic() - begin
        return result

    def _rebuild_users(self):
        self.user_lookup = {key: i for i, key in enumerate(self.user_keys)}
        if len(self.user_lookup) != len(self.user_keys):
            raise ValueError("Duplicate user-history keys")
        histories, graphs, masks, indices = [], [], [], []
        for key in self.user_keys:
            history = json.loads(key)[1]
            graph, mask, index = history_graph(history, self.categories, self.category_count)
            histories.append(history); graphs.append(graph); masks.append(mask); indices.append(index)
        self.histories = np.asarray(histories, dtype=np.int64).reshape(-1, self.history_length)
        size = self.history_length + self.category_count
        self.graphs = np.asarray(graphs, dtype=bool).reshape(-1, size, size)
        self.category_masks = np.asarray(masks, dtype=bool).reshape(-1, self.category_count + 1)
        self.category_indices = np.asarray(indices, dtype=np.int64).reshape(-1, self.history_length)

    def _ensure_users(self, keys):
        old_count = len(self.user_keys)
        for key in keys:
            if key not in self.user_lookup:
                self.user_lookup[key] = len(self.user_keys)
                self.user_keys.append(key)
        if len(self.user_keys) != old_count:
            self._rebuild_users()
        return np.asarray([self.user_lookup[key] for key in keys], dtype=np.int32)

    def _history(self, text):
        values = [self.corpus.news_ID_dict[item] for item in text.split()][-self.history_length:]
        return values + [0] * (self.history_length - len(values))

    @staticmethod
    def _key(user, history):
        return json.dumps([user, history], ensure_ascii=False, separators=(",", ":"))

    def _save_cache(self):
        for name, value in (("news.npy", self.news), ("context.npy", self.context)):
            if not (self.cache_dir / name).exists():
                _save_npy(self.cache_dir / name, value)
        _save_json(self.cache_dir / "user_keys.json", self.user_keys)
        _save_npy(self.cache_dir / "pair_keys.npy", self.pair_keys)
        _save_npy(self.cache_dir / "pair_scores.npy", self.pair_scores)
        names = ("news.npy", "context.npy", "user_keys.json", "pair_keys.npy", "pair_scores.npy")
        _save_json(self.cache_dir / "manifest.json", {
            "identity": self.identity, "provenance": self.provenance,
            "files": {name: _file_hash(self.cache_dir / name) for name in names},
        })

    def _assert_current(self):
        if model_fingerprint(self.model) != self.model_hash or corpus_identity(self.corpus, self.categories) != self.provenance["corpus_sha256"]:
            raise ValueError("Model/corpus changed; create a new checkpoint-bound evaluator")

    def _compute_pairs(self, user_indices, news_indices, backend=None, on_scored=None):
        if backend not in (None, "dense", "sparse") or (backend == "sparse" and self.sparse is None):
            raise ValueError("Invalid or unavailable DIGAT scoring backend")
        result = np.empty(len(news_indices), dtype=np.float32)
        score_started = time.monotonic()
        with torch.inference_mode():
            for start in range(0, len(result), self.score_batch_size):
                end = min(len(result), start + self.score_batch_size)
                users, news = user_indices[start:end], news_indices[start:end]
                history = torch.as_tensor(self.histories[users], device=self.device)
                nodes = torch.as_tensor(self.corpus.news_node_ID[news], dtype=torch.long, device=self.device)
                ids = torch.as_tensor(news, dtype=torch.long, device=self.device)
                user_graph = torch.as_tensor(self.graphs[users], device=self.device)
                category_mask = torch.as_tensor(self.category_masks[users], device=self.device)
                category_indices = torch.as_tensor(self.category_indices[users], device=self.device)
                news_graph = torch.as_tensor(self.corpus.news_graph[news], device=self.device)
                news_mask = torch.as_tensor(self.corpus.news_graph_mask[news], device=self.device)
                if self.sparse is not None and backend != "dense":
                    score = self.sparse.score(self.news_device, self.context_device, history, ids,
                        nodes, user_graph, category_mask, category_indices, news_graph, news_mask)
                elif self.projections is None:
                    score = self.model.inference(self.news_device[history], user_graph, category_mask,
                        category_indices, self.news_device[nodes], news_graph, news_mask, self.context_device[ids])
                else:
                    score = self.projections.score(self.news_device, self.context_device, history, ids,
                        nodes, user_graph, category_mask, category_indices, news_graph, news_mask)
                result[start:end] = score.cpu().numpy()
                cooling.safe_boundary()
                if on_scored is not None:
                    on_scored(end, len(result))
                if start == 0 or end == len(result) or (start // self.score_batch_size) % 5000 == 0:
                    if on_scored is not None:
                        from experiment_progress import clear_console_progress
                        clear_console_progress()
                    print(json.dumps({"stage": "digat_graph_score", "done": end, "total": len(result), "elapsed_seconds": time.monotonic() - score_started}), flush=True)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite DIGAT logits")
        return result

    def _scores(self, users, news, on_scored=None):
        combined = (users.astype(np.int64) << 32) | news.astype(np.int64)
        unique, inverse = np.unique(combined, return_inverse=True)
        positions = np.searchsorted(self.pair_keys, unique)
        found = positions < len(self.pair_keys)
        if len(self.pair_keys):
            found &= self.pair_keys[np.minimum(positions, len(self.pair_keys) - 1)] == unique
        missing = unique[~found]
        begin = time.monotonic()
        if len(missing):
            scored_users = (missing >> 32).astype(np.int64)
            scored_news = (missing & 0xffffffff).astype(np.int64)
            if on_scored is None:
                values = self._compute_pairs(scored_users, scored_news)
            else:
                values = self._compute_pairs(scored_users, scored_news, on_scored=on_scored)
            del scored_users, scored_news
            joined = np.concatenate((self.pair_keys, missing))
            order = np.argsort(joined)
            self.pair_keys = joined[order]
            self.pair_scores = np.concatenate((self.pair_scores, values))[order]
        self.timings["graph_scoring_seconds"] += time.monotonic() - begin
        scores = self.pair_scores[np.searchsorted(self.pair_keys, unique)][inverse]
        return scores, {"requested_pairs": len(combined), "unique_pairs": len(unique),
                        "new_pairs_computed": len(missing),
                        "reused_unique_pairs": len(unique) - len(missing),
                        "duplicates_within_file": len(combined) - len(unique),
                        "hit_rate_among_unique_pairs": float((len(unique) - len(missing)) / len(unique)) if len(unique) else 0.0,
                        "cached_pairs_total": len(self.pair_keys)}

    def evaluate_file(self, behavior_path, output_prefix, *, progress=None):
        self._assert_current(); self.model.eval()
        if self.use_sparse_eval and not self.sparse_verified:
            raise ValueError("Sparse DIGAT requires native eight-row and fixed 8192-pair dense score/rank verification first")
        if self.use_sparse_eval and self.sparse_verification["score_batch_size"] != self.score_batch_size:
            raise ValueError("Sparse batch size changed after checkpoint verification")
        if self.use_sparse_eval:
            self.sparse._validate()
            if self.score_batch_size != self.sparse_verification["score_batch_size"]:
                raise ValueError("Sparse score batch changed after checkpoint verification")
        cooling_start = cooling.marker()
        begin = time.monotonic()
        behavior_path, prefix = Path(behavior_path), Path(output_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        names = ("metrics.json", "scores.npy", "labels.npy", "offsets.npy", "news_indices.npy",
                 "user_indices.npy", "entry_metrics.npy", "records.jsonl", "predictions.txt", "truth.txt")
        paths = {name: Path(str(prefix) + "." + name) for name in names}
        if any(path.exists() for path in paths.values()):
            raise ValueError("Output artifacts already exist")
        records, keys, chunks, label_chunks, offsets = [], [], [], [], [0]
        with behavior_path.open(encoding="utf-8") as stream:
            for line in stream:
                impression, user, timestamp, history_text, candidate_text = line.rstrip("\r\n").split("\t")
                history = self._history(history_text)
                tokens = [item.rsplit("-", 1) for item in candidate_text.split()]
                if not tokens or any(len(item) != 2 or item[1] not in ("0", "1") for item in tokens):
                    raise ValueError("Invalid labeled candidates")
                news = np.asarray([self.corpus.news_ID_dict[item[0]] for item in tokens], dtype=np.int32)
                labels = np.asarray([int(item[1]) for item in tokens], dtype=np.int8)
                if set(labels.tolist()) != {0, 1} or len(set(news.tolist())) != len(news):
                    raise ValueError("Require both classes and unique candidates")
                keys.append(self._key(user, history))
                records.append({"impression_id": impression, "user_id": user, "time": timestamp,
                                "history_sha256": hashlib.sha256(history_text.encode("utf-8")).hexdigest(),
                                "positive_news": [item[0] for item in tokens if item[1] == "1"],
                                "candidate_count": len(news)})
                chunks.append(news); label_chunks.append(labels)
                offsets.append(offsets[-1] + len(news))
        if not records:
            raise ValueError("Empty behavior file")
        if progress is not None:
            progress(0, len(records))
        users = self._ensure_users(keys)
        news, labels = np.concatenate(chunks), np.concatenate(label_chunks)
        offsets = np.asarray(offsets, dtype=np.int64)
        del chunks, label_chunks, keys
        flat_users = np.repeat(users, np.diff(offsets))
        if progress is None:
            scores, reuse = self._scores(flat_users, news)
        else:
            def on_scored(done, total):
                progress(0, len(records), scored=done, score_total=total)
            scores, reuse = self._scores(flat_users, news, on_scored=on_scored)
        del flat_users
        entry_metrics = np.empty((len(records), 4), dtype=np.float64)
        with paths["predictions.txt"].open("x", encoding="utf-8", newline="\n") as pred, paths["truth.txt"].open("x", encoding="utf-8", newline="\n") as truth, paths["records.jsonl"].open("x", encoding="utf-8", newline="\n") as record:
            for row, metadata in enumerate(records):
                start, end = int(offsets[row]), int(offsets[row + 1])
                order = np.argsort(-scores[start:end], kind="stable")
                ranks = np.empty(end - start, dtype=np.int32); ranks[order] = np.arange(1, end - start + 1)
                entry_metrics[row] = metrics_from_ranks(labels[start:end], ranks)
                pred.write(str(row + 1) + " " + json.dumps(ranks.tolist(), separators=(",", ":")) + "\n")
                truth.write(str(row + 1) + " " + json.dumps(labels[start:end].tolist(), separators=(",", ":")) + "\n")
                record.write(json.dumps(metadata, ensure_ascii=False) + "\n")
                cooling.safe_boundary()
                if progress is not None and row + 1 < len(records):
                    progress(row + 1, len(records))
        for name, value in (("scores", scores), ("labels", labels), ("offsets", offsets),
                            ("news_indices", news), ("user_indices", users), ("entry_metrics", entry_metrics)):
            _save_npy(paths[name + ".npy"], value)
        self._save_cache()
        report = {"created_utc": datetime.now(timezone.utc).isoformat(),
                  "metrics": dict(zip(METRIC_NAMES, entry_metrics.mean(axis=0).tolist())),
                  "metric_order": list(METRIC_NAMES), "rows": len(records), "candidates": len(news),
                  "input": {"path": str(behavior_path.resolve()), "sha256": _file_hash(behavior_path)},
                  "cache_identity": self.identity, "cache_dir": str(self.cache_dir.resolve()),
                  "cache_loaded_from_disk": self.cache_loaded, "pair_cache": reuse,
                  "score_batch_size": self.score_batch_size,
                  "layer_zero_projection_cache": self.use_projection_cache,
                  "graph_backend": self.provenance["graph_backend"],
                  "sparse_verification": self.sparse_verification if self.use_sparse_eval else None,
                  "sparse_execution_stats_cumulative": copy.deepcopy(self.sparse.stats) if self.use_sparse_eval else None,
                  "projection_cache_bytes": self.projections.bytes if self.projections else 0,
                  "cache_timings_cumulative": dict(self.timings),
                  "elapsed_seconds": time.monotonic() - begin,
                  "tie_policy": "stable_original_candidate_order",
                  "files": {name: str(path.resolve()) for name, path in paths.items()},
                  "scope": "This checkpoint and input only; not historical paper reproduction"}
        report.update(cooling.duration_fields(report["elapsed_seconds"], cooling_start))
        _save_json(paths["metrics.json"], report)
        return report

    def verify_against_forward(self, behavior_path, max_entries=8, max_candidates=16, native_split="dev", atol=1e-4, rtol=1e-5):
        self._assert_current()
        if self.use_sparse_eval:
            self.sparse_verified = False
            if max_entries < 8 or max_candidates < 16 or native_split != "dev" or atol != 1e-4 or rtol != 1e-5:
                raise ValueError("Sparse native gate requires dev, eight rows, at least sixteen candidates per row, atol=1e-4 and rtol=1e-5")
        records, keys = [], []
        with Path(behavior_path).open(encoding="utf-8") as stream:
            for row, line in enumerate(stream):
                _, user, _, text, candidates = line.rstrip("\r\n").split("\t")
                history = self._history(text)
                news = [self.corpus.news_ID_dict[item.rsplit("-", 1)[0]] for item in candidates.split()[:max_candidates]]
                records.append((history, news)); keys.append(self._key(user, history))
                if len(records) >= max_entries: break
        users = self._ensure_users(keys)
        if self.use_sparse_eval and len(records) < 8:
            raise ValueError("Sparse native gate requires eight available dev rows")
        largest, compared, rank_disagreements = 0.0, 0, 0
        def reject_sparse_native(reason, row, direct, cached, allow_dense_fallback=True):
            from digat_sparse_attention import SparseEquivalenceError
            evidence = self.cache_dir / ("sparse_native_rejection_" + str(time.time_ns()) + ".json")
            _save_json(evidence, {"status": "rejected", "reason": reason, "row": row,
                "native_scores": direct.cpu().tolist(), "sparse_scores": cached.cpu().tolist(),
                "cache_identity": self.identity, "model_state_sha256": self.model_hash,
                "input_sha256": _file_hash(behavior_path), "provenance": self.provenance,
                "atol": atol, "rtol": rtol, "stable_ranks_required": True})
            if not allow_dense_fallback:
                raise ValueError(reason + ": " + str(evidence))
            raise SparseEquivalenceError(reason + ": " + str(evidence), evidence)
        with torch.inference_mode():
            for row, (history, news) in enumerate(records):
                ui = users[row]
                native_graph = getattr(self.corpus, native_split + "_user_history_graph")[row]
                native_mask = getattr(self.corpus, native_split + "_user_history_category_mask")[row]
                native_indices = getattr(self.corpus, native_split + "_user_history_category_indices")[row]
                if not (np.array_equal(self.graphs[ui], native_graph) and np.array_equal(self.category_masks[ui], native_mask) and np.array_equal(self.category_indices[ui], native_indices)):
                    raise ValueError("Reconstructed history graph differs from official preprocessed graph")
                def t(value): return torch.as_tensor(value, device=self.device).unsqueeze(0)
                nodes = self.corpus.news_node_ID[news]
                direct = self.model(
                    t(self.corpus.news_title_text[history]), t(self.corpus.news_title_mask[history]),
                    t(native_graph), t(native_mask), t(native_indices),
                    t(self.corpus.news_title_text[nodes]), t(self.corpus.news_title_mask[nodes]),
                    t(self.corpus.news_graph[news]), t(self.corpus.news_graph_mask[news])).flatten()
                cached = torch.as_tensor(self._compute_pairs(np.full(len(news), ui), np.asarray(news)), device=self.device)
                if not (torch.isfinite(direct).all() and torch.isfinite(cached).all()):
                    if self.use_sparse_eval:
                        reject_sparse_native("Nonfinite DIGAT native/sparse scores", row, direct, cached,
                                             allow_dense_fallback=False)
                    raise ValueError("Nonfinite DIGAT native/dense scores")
                difference = float((direct - cached).abs().max())
                largest = max(largest, difference); compared += len(news)
                if not torch.allclose(direct, cached, atol=atol, rtol=rtol):
                    if self.use_sparse_eval:
                        reject_sparse_native("DIGAT sparse/native score mismatch", row, direct, cached)
                    raise ValueError("DIGAT cached/native score mismatch: " + str(difference))
                if not np.array_equal(np.argsort(-direct.cpu().numpy(), kind="stable"),
                                      np.argsort(-cached.cpu().numpy(), kind="stable")):
                    rank_disagreements += 1
                    if self.use_sparse_eval:
                        reject_sparse_native("DIGAT sparse/native stable rank mismatch", row, direct, cached)
        verification = {"status": "passed", "rows": len(records), "candidate_scores": compared,
                "max_absolute_error": largest, "atol": atol, "rtol": rtol,
                "rows_with_rank_disagreement": rank_disagreements,
                "stable_ranks_required": self.use_sparse_eval,
                "native_history_graph_exact_match": True, "cache_identity": self.identity,
                "model_state_sha256": self.model_hash, "input_sha256": _file_hash(behavior_path)}
        if self.use_sparse_eval:
            from digat_sparse_attention import verify_checkpoint_gate
            self.sparse_verification = verify_checkpoint_gate(self, behavior_path, verification)
            self.sparse_verified = True
            verification["sparse_dense_gate"] = self.sparse_verification
        return verification
