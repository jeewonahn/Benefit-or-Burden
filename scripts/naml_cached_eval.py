"""Checkpoint-bound NAML/ATT evaluation with reusable news/history embeddings.

Only NAML + ATT + dot_product is supported. In particular, ATT's
unmasked attention over padded history is preserved, not silently repaired.
Caches and prediction arrays contain local research data and belong outside Git.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import scheduled_cooling as cooling
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
FEATURES = (
    'news_title_text', 'news_title_mask', 'news_title_entity',
    'news_abstract_text', 'news_abstract_mask', 'news_abstract_entity',
    'news_category', 'news_subCategory',
)
METRIC_NAMES = ('AUC', 'MRR', 'nDCG@5', 'nDCG@10')


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _array_hash_into(digest, name, value):
    value = np.ascontiguousarray(value)
    digest.update(_json_bytes([name, str(value.dtype), value.shape]))
    digest.update(memoryview(value).cast('B'))


def model_fingerprint(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        _array_hash_into(digest, name, value.detach().cpu().numpy())
    return digest.hexdigest()


def corpus_fingerprint(corpus):
    digest = hashlib.sha256()
    digest.update(_json_bytes({'news_ID_dict': corpus.news_ID_dict,
                               'max_history_num': int(corpus.max_history_num)}))
    for name in FEATURES:
        _array_hash_into(digest, name, getattr(corpus, name))
    return digest.hexdigest()


def _save_npy(path, value):
    temporary = Path(str(path) + '.tmp')
    with temporary.open('wb') as stream:
        np.save(stream, value, allow_pickle=False)
    os.replace(str(temporary), str(path))


def _save_json(path, value):
    temporary = Path(str(path) + '.tmp')
    temporary.write_bytes(json.dumps(value, indent=2, ensure_ascii=False).encode('utf-8') + b'\n')
    os.replace(str(temporary), str(path))


def metrics_from_ranks(labels, ranks):
    """Equivalent to evaluate.scoring's unique integer-rank convention.

    util.compute_scores breaks tied logits by input order. Preserve that same
    convention before calculating AUC, mean reciprocal positive rank and nDCG.
    """
    labels = np.asarray(labels)
    ranks = np.asarray(ranks)
    if not np.isin(labels, (0, 1)).all():
        raise ValueError('Only binary candidate labels are supported')
    positive = ranks[labels == 1].astype(np.float64)
    n_pos, n_neg = len(positive), len(labels) - len(positive)
    if not n_pos or not n_neg:
        raise ValueError('AUC requires both positive and negative candidates')
    auc = 1.0 - (positive.sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    mrr = np.mean(1.0 / positive)
    ndcg = []
    for k in (5, 10):
        actual = np.sum(1.0 / np.log2(positive[positive <= k] + 1.0))
        ideal = np.sum(1.0 / np.log2(np.arange(1, min(k, n_pos) + 1) + 1.0))
        ndcg.append(actual / ideal)
    return np.asarray([auc, mrr, *ndcg], dtype=np.float64)


class NamlCachedEvaluator:
    """Encode news once, encode each (raw user ID, padded history) once.

    A new instance is required after changing model weights. The full model
    state, corpus feature arrays, ID mapping and encoder source hashes identify
    each cache namespace. A matching namespace is validated before reuse.
    """

    def __init__(self, model, corpus, cache_dir, news_batch_size=256,
                 user_batch_size=256, score_batch_size=32768):
        if (type(model.news_encoder).__name__ != 'NAML' or
                type(model.user_encoder).__name__ != 'ATT' or
                model.click_predictor != 'dot_product' or model.use_user_embedding):
            raise ValueError('Cached evaluator supports only NAML + ATT + dot_product')
        if any(int(x) <= 0 for x in (news_batch_size, user_batch_size, score_batch_size)):
            raise ValueError('Batch sizes must be positive')
        if next(model.parameters()).dtype != torch.float32:
            raise ValueError('This validated cache path requires a float32 model')
        self.model, self.corpus = model, corpus
        self.device = next(model.parameters()).device
        self.news_batch_size, self.user_batch_size = int(news_batch_size), int(user_batch_size)
        self.score_batch_size = int(score_batch_size)
        self.history_length = int(corpus.max_history_num)
        self.dimension = int(model.news_embedding_dim)
        self.model_hash = model_fingerprint(model)
        self.provenance = {
            'schema': 1, 'model': 'NAML-ATT', 'click_predictor': 'dot_product',
            'model_state_sha256': self.model_hash,
            'corpus_sha256': corpus_fingerprint(corpus),
            'history_length': self.history_length, 'padding_policy': 'tail_zero_unmasked_ATT',
            'dtype': 'float32',
            'source_sha256': {name: _file_hash(ROOT / 'NNR-main' / name)
                              for name in ('model.py', 'newsEncoders.py', 'userEncoders.py', 'layers.py')},
            'cache_implementation_sha256': _file_hash(__file__),
        }
        self.identity = hashlib.sha256(_json_bytes(self.provenance)).hexdigest()
        self.cache_dir = Path(cache_dir) / self.identity
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_manifest = self.cache_dir / 'manifest.json'
        self.user_keys, self.user_lookup = [], {}
        self.users = np.zeros((0, self.dimension), dtype=np.float32)
        self.model.eval()
        self.timings = {'news_encoding_seconds': 0.0, 'user_encoding_seconds': 0.0}
        self.cache_loaded = False
        if self.cache_manifest.exists():
            self._load_cache()
            self.cache_loaded = True
        else:
            if any(self.cache_dir.iterdir()):
                raise ValueError('Incomplete cache directory; use a fresh cache root')
            self.news = self._encode_news()
            self._save_cache()
        self.news_device = torch.as_tensor(self.news, device=self.device)

    def _load_cache(self):
        metadata = json.loads(self.cache_manifest.read_text(encoding='utf-8'))
        if metadata['identity'] != self.identity or metadata['provenance'] != self.provenance:
            raise ValueError('Cache identity/provenance mismatch')
        for name, expected in metadata['files'].items():
            if _file_hash(self.cache_dir / name) != expected:
                raise ValueError('Cache integrity mismatch: ' + name)
        self.news = np.load(self.cache_dir / 'news_embeddings.npy', allow_pickle=False)
        self.users = np.load(self.cache_dir / 'user_embeddings.npy', allow_pickle=False)
        self.user_keys = json.loads((self.cache_dir / 'user_keys.json').read_text(encoding='utf-8'))
        self.user_lookup = {key: i for i, key in enumerate(self.user_keys)}
        if len(self.user_lookup) != len(self.user_keys):
            raise ValueError('Duplicate user-history cache keys')
        if self.news.shape != (len(self.corpus.news_title_text), self.dimension):
            raise ValueError('News cache shape mismatch')
        if self.users.shape != (len(self.user_keys), self.dimension):
            raise ValueError('User cache shape mismatch')
        if self.news.dtype != np.float32 or self.users.dtype != np.float32:
            raise ValueError('Cache dtype mismatch')
        if not np.isfinite(self.news).all() or not np.isfinite(self.users).all():
            raise ValueError('Nonfinite cached embeddings')

    def _save_cache(self):
        if not (self.cache_dir / 'news_embeddings.npy').exists():
            _save_npy(self.cache_dir / 'news_embeddings.npy', self.news)
        _save_npy(self.cache_dir / 'user_embeddings.npy', self.users)
        _save_json(self.cache_dir / 'user_keys.json', self.user_keys)
        names = ('news_embeddings.npy', 'user_embeddings.npy', 'user_keys.json')
        _save_json(self.cache_manifest, {
            'identity': self.identity, 'provenance': self.provenance,
            'files': {name: _file_hash(self.cache_dir / name) for name in names},
            'news_count': len(self.news), 'user_history_count': len(self.users),
        })

    def _encode_news(self):
        started = time.perf_counter()
        count = len(self.corpus.news_title_text)
        result = np.empty((count, self.dimension), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, count, self.news_batch_size):
                end = min(start + self.news_batch_size, count)
                arrays = [torch.as_tensor(np.asarray(getattr(self.corpus, name)[start:end]),
                                          device=self.device).unsqueeze(1) for name in FEATURES]
                vectors = self.model.news_encoder(*arrays, None).squeeze(1)
                result[start:end] = vectors.detach().cpu().numpy()
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite news embeddings')
        self.timings['news_encoding_seconds'] = time.perf_counter() - started
        return result

    def _history(self, raw_history):
        try:
            values = [self.corpus.news_ID_dict[x] for x in raw_history.split()]
        except KeyError as exc:
            raise ValueError('History news missing from the trained corpus mapping') from exc
        values = values[-self.history_length:]
        return values + [0] * (self.history_length - len(values))

    @staticmethod
    def _user_key(user, history):
        return json.dumps([user, history], ensure_ascii=False, separators=(',', ':'))

    def _ensure_users(self, keys):
        pending, seen = [], set()
        for key in keys:
            if key not in self.user_lookup and key not in seen:
                pending.append(key)
                seen.add(key)
        if pending:
            started = time.perf_counter()
            output = np.empty((len(pending), self.dimension), dtype=np.float32)
            with torch.inference_mode():
                for start in range(0, len(pending), self.user_batch_size):
                    batch = pending[start:start + self.user_batch_size]
                    histories = torch.as_tensor([json.loads(key)[1] for key in batch],
                                                dtype=torch.long, device=self.device)
                    # The original ATT intentionally supplies no padding mask.
                    vectors = self.model.user_encoder.attention(self.news_device[histories])
                    output[start:start + len(batch)] = vectors.detach().cpu().numpy()
            if not np.isfinite(output).all():
                raise ValueError('Nonfinite user embeddings')
            offset = len(self.user_keys)
            self.user_keys.extend(pending)
            self.user_lookup.update({key: offset + i for i, key in enumerate(pending)})
            self.users = np.concatenate([self.users, output], axis=0)
            self.timings['user_encoding_seconds'] += time.perf_counter() - started
            self._save_cache()
        return np.asarray([self.user_lookup[key] for key in keys], dtype=np.int32)

    def evaluate_file(self, behavior_path, output_prefix, *, progress=None):
        """Evaluate a TSV without adding millions of candidates to MIND_Corpus.

        Returns aggregate metrics and writes flat candidate arrays + row offsets,
        per-entry metrics, stable ranks, official-format truth and row metadata.
        No outputs are overwritten. Unknown news and invalid labels fail closed.
        """
        cooling_start = cooling.marker()
        started = time.perf_counter()
        if model_fingerprint(self.model) != self.model_hash:
            raise ValueError('Model changed after cache construction; create a new evaluator')
        if corpus_fingerprint(self.corpus) != self.provenance['corpus_sha256']:
            raise ValueError('Corpus changed after cache construction; create a new evaluator')
        self.model.eval()
        behavior_path, prefix = Path(behavior_path), Path(output_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        suffixes = ('metrics.json', 'scores.npy', 'labels.npy', 'offsets.npy', 'news_indices.npy',
                    'user_indices.npy', 'entry_metrics.npy', 'records.jsonl', 'predictions.txt', 'truth.txt')
        paths = {name: Path(str(prefix) + '.' + name) for name in suffixes}
        if any(path.exists() for path in paths.values()):
            raise ValueError('Output prefix already has artifacts; choose a new prefix')
        records, keys, news_chunks, label_chunks, offsets = [], [], [], [], [0]
        with behavior_path.open('r', encoding='utf-8') as stream:
            for row_index, line in enumerate(stream):
                parts = line.rstrip('\r\n').split('\t')
                if len(parts) != 5:
                    raise ValueError('Expected five behavior columns at row ' + str(row_index + 1))
                impression, user, timestamp, raw_history, raw_candidates = parts
                history = self._history(raw_history)
                tokens = [item.rsplit('-', 1) for item in raw_candidates.split()]
                if not tokens or any(len(item) != 2 or item[1] not in ('0', '1') for item in tokens):
                    raise ValueError('Invalid labeled candidates at row ' + str(row_index + 1))
                try:
                    news = np.asarray([self.corpus.news_ID_dict[item[0]] for item in tokens], dtype=np.int32)
                except KeyError as exc:
                    raise ValueError('Candidate news missing from the trained corpus mapping') from exc
                labels = np.asarray([int(item[1]) for item in tokens], dtype=np.int8)
                if labels.sum() == 0 or labels.sum() == len(labels):
                    raise ValueError('Every evaluated row needs positive and negative labels')
                if len(set(news.tolist())) != len(news):
                    raise ValueError('Duplicate candidate news in a behavior row')
                key = self._user_key(user, history)
                keys.append(key)
                records.append({'impression_id': impression, 'user_id': user, 'time': timestamp,
                                'history_sha256': hashlib.sha256(raw_history.encode('utf-8')).hexdigest(),
                                'positive_news': [item[0] for item in tokens if item[1] == '1'],
                                'candidate_count': len(news)})
                news_chunks.append(news)
                label_chunks.append(labels)
                offsets.append(offsets[-1] + len(news))
        if not records:
            raise ValueError('Empty evaluation behavior file')
        if progress is not None:
            progress(0, len(records))
        users = self._ensure_users(keys)
        news = np.concatenate(news_chunks)
        labels = np.concatenate(label_chunks)
        del news_chunks, label_chunks, keys
        offsets = np.asarray(offsets, dtype=np.int64)
        flat_users = np.repeat(users, np.diff(offsets))
        user_device = torch.as_tensor(self.users, device=self.device)
        scores = np.empty(len(news), dtype=np.float32)
        score_started = time.perf_counter()
        with torch.inference_mode():
            for start in range(0, len(news), self.score_batch_size):
                end = min(start + self.score_batch_size, len(news))
                ni = torch.as_tensor(news[start:end], dtype=torch.long, device=self.device)
                ui = torch.as_tensor(flat_users[start:end], dtype=torch.long, device=self.device)
                logits = (self.news_device[ni] * user_device[ui]).sum(dim=1)
                scores[start:end] = logits.detach().cpu().numpy()
                cooling.safe_boundary()
                if progress is not None:
                    progress(0, len(records), scored=end, score_total=len(news))
        if not np.isfinite(scores).all():
            raise ValueError('Nonfinite cached logits')
        dot_seconds = time.perf_counter() - score_started
        del flat_users, user_device
        entry_metrics = np.empty((len(records), len(METRIC_NAMES)), dtype=np.float64)
        with paths['predictions.txt'].open('x', encoding='utf-8', newline='\n') as prediction, \
             paths['truth.txt'].open('x', encoding='utf-8', newline='\n') as truth, \
             paths['records.jsonl'].open('x', encoding='utf-8', newline='\n') as row_file:
            for i, record in enumerate(records):
                start, end = int(offsets[i]), int(offsets[i + 1])
                order = np.argsort(-scores[start:end], kind='stable')
                ranks = np.empty(end - start, dtype=np.int32)
                ranks[order] = np.arange(1, end - start + 1)
                entry_metrics[i] = metrics_from_ranks(labels[start:end], ranks)
                prediction.write(str(i + 1) + ' ' + json.dumps(ranks.tolist(), separators=(',', ':')) + '\n')
                truth.write(str(i + 1) + ' ' + json.dumps(labels[start:end].tolist(), separators=(',', ':')) + '\n')
                row_file.write(json.dumps(record, ensure_ascii=False) + '\n')
                cooling.safe_boundary()
                if progress is not None and i + 1 < len(records):
                    progress(i + 1, len(records))
        for name, value in (('scores', scores), ('labels', labels), ('offsets', offsets),
                            ('news_indices', news), ('user_indices', users), ('entry_metrics', entry_metrics)):
            _save_npy(paths[name + '.npy'], value)
        report = {
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'metrics': dict(zip(METRIC_NAMES, entry_metrics.mean(axis=0).tolist())),
            'rows': len(records), 'candidates': len(news),
            'input': {'path': str(behavior_path.resolve()), 'sha256': _file_hash(behavior_path)},
            'cache_identity': self.identity, 'cache_dir': str(self.cache_dir.resolve()),
            'cache_loaded_from_disk': self.cache_loaded,
            'cached_news_count': len(self.news), 'cached_user_history_count': len(self.users),
            'elapsed_seconds': time.perf_counter() - started,
            'dot_product_seconds': dot_seconds, 'cache_timings_cumulative': dict(self.timings),
            'metric_order': list(METRIC_NAMES), 'tie_policy': 'stable_original_candidate_order',
            'files': {name: str(path.resolve()) for name, path in paths.items()},
            'scope': 'Evaluation of this checkpoint/input only; not historical paper reproduction',
        }
        report.update(cooling.duration_fields(report['elapsed_seconds'], cooling_start))
        _save_json(paths['metrics.json'], report)
        return report

    def verify_against_forward(self, behavior_path, max_entries=8, max_candidates=32,
                               atol=1e-4, rtol=1e-5):
        """Compare the actual model.forward and cached path on bounded real rows.

        This evaluates the supplied checkpoint, not a training update. Candidate
        order and padding are unchanged; only the number of inspected rows and
        candidates is bounded. Raises if numerical equivalence fails.
        """
        if model_fingerprint(self.model) != self.model_hash:
            raise ValueError('Model changed after cache construction; create a new evaluator')
        if corpus_fingerprint(self.corpus) != self.provenance['corpus_sha256']:
            raise ValueError('Corpus changed after cache construction; create a new evaluator')
        if max_entries <= 0 or max_candidates <= 0:
            raise ValueError('Verification limits must be positive')
        rows, keys = [], []
        with Path(behavior_path).open('r', encoding='utf-8') as stream:
            for line in stream:
                _, user, _, raw_history, raw_candidates = line.rstrip('\r\n').split('\t')
                history = self._history(raw_history)
                candidates = [self.corpus.news_ID_dict[item.rsplit('-', 1)[0]]
                              for item in raw_candidates.split()[:max_candidates]]
                if not candidates:
                    raise ValueError('No candidates to verify')
                rows.append((user, history, candidates, len(raw_history.split())))
                keys.append(self._user_key(user, history))
                if len(rows) == max_entries:
                    break
        if not rows:
            raise ValueError('No rows to verify')
        user_indices = self._ensure_users(keys)
        feature_order = ('news_category', 'news_subCategory', 'news_title_text', 'news_title_mask',
                         'news_title_entity', 'news_abstract_text', 'news_abstract_mask', 'news_abstract_entity')
        def features(indices):
            return [torch.as_tensor(np.asarray(getattr(self.corpus, name)[indices]),
                                    device=self.device).unsqueeze(0) for name in feature_order]
        largest, compared = 0.0, 0
        self.model.eval()
        with torch.inference_mode():
            for i, (user, history, candidates, original_history_length) in enumerate(rows):
                history_mask = torch.zeros((1, self.history_length), dtype=torch.bool, device=self.device)
                history_mask[:, :min(self.history_length, original_history_length)] = True
                user_id = getattr(self.corpus, 'user_ID_dict', {}).get(user, 0)
                ignored_graph = torch.zeros((1, self.history_length, self.history_length), device=self.device)
                ignored_category_mask = torch.zeros((1, 1), dtype=torch.bool, device=self.device)
                ignored_category_indices = torch.zeros((1, self.history_length), dtype=torch.long, device=self.device)
                direct = self.model(torch.tensor([user_id], device=self.device), *features(history),
                                    history_mask, ignored_graph, ignored_category_mask,
                                    ignored_category_indices, *features(candidates)).flatten()
                user_vector = torch.as_tensor(self.users[user_indices[i]], device=self.device)
                candidate_indices = torch.tensor(candidates, dtype=torch.long, device=self.device)
                cached = (self.news_device[candidate_indices] * user_vector).sum(dim=1)
                difference = float((direct - cached).abs().max().item())
                largest = max(largest, difference)
                compared += len(candidates)
                if not torch.allclose(direct, cached, atol=atol, rtol=rtol):
                    raise ValueError('Cached vs direct forward mismatch: max_abs=' + str(difference))
        return {'status': 'passed', 'rows': len(rows), 'candidate_scores': compared,
                'max_absolute_error': largest, 'atol': atol, 'rtol': rtol,
                'cache_identity': self.identity, 'model_state_sha256': self.model_hash,
                'input_sha256': _file_hash(behavior_path),
                'device': str(self.device), 'scope': 'Bounded direct-forward/cache equivalence'}
