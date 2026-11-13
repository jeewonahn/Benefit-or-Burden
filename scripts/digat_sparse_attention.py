"""Opt-in, evaluation-only sparse execution of official DIGAT graph attention.

Weights, edges, context updates, layer depth and residuals remain unchanged.
CSR reductions avoid atomic scatter summation. Floating-point reduction order
still differs from dense bmm and requires checkpoint-specific equivalence tests.
This module does not modify the vendor or install itself in an existing run.
Pair-score caches MUST use a distinct backend/code identity before integration.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import torch
import torch.nn.functional as F
from torch_scatter import segment_csr


class SparseEquivalenceError(ValueError):
    """Strict score/rank mismatch; callers may explicitly restart dense evaluation."""

    def __init__(self, message, evidence_path):
        super().__init__(message)
        self.evidence_path = str(Path(evidence_path).resolve())


@dataclass
class Topology:
    batch: int
    nodes: int
    batch_index: torch.Tensor
    row: torch.Tensor
    col: torch.Tensor
    flat_row: torch.Tensor
    indptr: torch.Tensor
    has_empty_row: bool


def topology(graph):
    if graph.ndim != 3 or graph.shape[1] != graph.shape[2]:
        raise ValueError("Require batched square graph masks")
    batch, nodes, _ = graph.shape
    if min(batch, nodes) < 1: raise ValueError("Empty graph dimensions")
    edge = (graph != 0).nonzero(as_tuple=False)  # Lexicographic (batch,row,column).
    b, row, col = edge.unbind(1)
    flat = b * nodes + row
    counts = torch.bincount(flat, minlength=batch * nodes)
    indptr = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    return Topology(batch, nodes, b, row, col, flat, indptr, bool((counts == 0).any().item()))


class SparseDIGATScore:
    def __init__(self, model, layer_zero_cache=None):
        self.model, self.graph = model, model.graph_encoder
        if type(self.graph).__name__ != "DIGAT": raise ValueError("Official DIGAT required")
        self.layer_zero_cache = layer_zero_cache
        self.bytes = layer_zero_cache.bytes if layer_zero_cache is not None else 0
        self.stats = {"sparse_layers": 0, "dense_fallback_layers": 0,
                      "valid_edges": 0, "dense_edges": 0, "fallback_reasons": {}}
        self.implementation_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self._validate()

    def _validate(self):
        if self.model.training or self.graph.training:
            raise ValueError("Sparse DIGAT is evaluation-only")
        parameter = next(self.graph.parameters())
        if parameter.dtype != torch.float32 or torch.is_autocast_enabled():
            raise ValueError("Strict float32 without autocast required")
        if parameter.device.type == "cuda" and (torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32):
            raise ValueError("Disable TF32 before sparse equivalence evaluation")

    def _dense(self, side, index, embeddings, graph, context, reason):
        self.stats["dense_fallback_layers"] += 1
        reasons = self.stats["fallback_reasons"]
        reasons[reason] = reasons.get(reason, 0) + 1
        return getattr(self.graph, "compute_" + side + "_graph_embeddings")(index, embeddings, graph, context)

    def layer(self, side, index, embeddings, graph, context, edges=None, node_ids=None):
        self._validate()
        g = self.graph
        if side not in ("news", "user") or not 0 <= index < g.graph_depth:
            raise ValueError("Invalid graph side or layer")
        edges = topology(graph) if edges is None else edges
        if embeddings.shape[:2] != (edges.batch, edges.nodes): raise ValueError("Graph/embedding shape mismatch")
        if edges.has_empty_row:
            # Dense masked_fill(-1e9) followed by softmax makes an empty row
            # uniform over ALL nodes, rather than a zero message.
            return self._dense(side, index, embeddings, graph, context, "all_masked_row")
        b, row, col = edges.batch_index, edges.row, edges.col
        if index == 0 and self.layer_zero_cache is not None and node_ids is not None:
            cache = self.layer_zero_cache.values
            h = cache[side + "_W"][node_ids]
            k1 = cache[side + "_ffn1"][node_ids[b, col]]
            k2 = cache[side + "_ffn2"][node_ids[b, row]]
        else:
            h = getattr(g, side + "_graph_attention_W")[index](embeddings)
            k1 = getattr(g, side + "_graph_attention_ffn1")[index](embeddings)[b, col]
            k2 = getattr(g, side + "_graph_attention_ffn2")[index](embeddings)[b, row]
        k3 = getattr(g, side + "_graph_attention_ffn3")[index](context)[b]
        # Preserve source addition order: (K3 + K1) + K2, then ReLU and a.
        logits = getattr(g, side + "_graph_attention_a")[index](F.relu(k3 + k1 + k2, inplace=True)).squeeze(1)
        logits = g.leaky_relu(logits)
        if not bool(torch.isfinite(logits).all().item()) or float(logits.min().item()) <= -1e8:
            # Removing masked entries is invalid when valid logits approach the
            # finite -1e9 masking floor. Preserve the dense source in this case.
            return self._dense(side, index, embeddings, graph, context, "mask_floor_or_nonfinite")
        maximum = segment_csr(logits, edges.indptr, reduce="max")
        numerator = (logits - maximum[edges.flat_row]).exp()
        denominator = segment_csr(numerator, edges.indptr, reduce="sum")
        alpha = numerator / denominator[edges.flat_row]
        aggregated = segment_csr(alpha.unsqueeze(1) * h[b, col], edges.indptr, reduce="sum")
        self.stats["sparse_layers"] += 1
        self.stats["valid_edges"] += len(logits)
        self.stats["dense_edges"] += edges.batch * edges.nodes * edges.nodes
        return F.relu(aggregated.reshape(edges.batch, edges.nodes, g.news_embedding_dim), inplace=True) + embeddings

    def score(self, news_vectors, contexts, histories, candidate_ids, news_nodes,
              user_graph, category_mask, category_indices, news_graph, news_graph_mask):
        self._validate()
        g = self.graph
        batch = len(candidate_ids)
        user = torch.cat((news_vectors[histories], g.topic_node_embedding.unsqueeze(0).expand(batch, -1, -1)), dim=1)
        news = news_vectors[news_nodes]
        news_context = contexts[candidate_ids]
        user_context = g.compute_user_graph_context(user, category_mask, category_indices, news_context)
        news_edges, user_edges = topology(news_graph), topology(user_graph)
        topics = torch.arange(g.category_num - 1, device=histories.device) + len(news_vectors)
        user_nodes = torch.cat((histories, topics.unsqueeze(0).expand(batch, -1)), dim=1)
        for index in range(g.graph_depth):
            news = self.layer("news", index, news, news_graph, user_context, news_edges, news_nodes)
            user = self.layer("user", index, user, user_graph, news_context, user_edges, user_nodes)
            news_context = news_context + g.compute_news_graph_context(news, news_graph_mask)
            user_context = user_context + g.compute_user_graph_context(user, category_mask, category_indices, news_context)
        return (news_context * user_context).sum(dim=1)


def fixed_dev_pairs(evaluator, behavior_path, count=8192):
    """First-seen unique (user/history, candidate) pairs in original row order."""
    import numpy as np
    keys, key_map, pairs, pair_map, groups = [], {}, [], {}, []
    with Path(behavior_path).open(encoding="utf-8") as stream:
        for line in stream:
            _, user, _, history, candidates = line.rstrip("\r\n").split("\t")
            key = evaluator._key(user, evaluator._history(history))
            if key not in key_map:
                key_map[key] = len(keys); keys.append(key)
            ui, group = key_map[key], []
            for item in candidates.split():
                ni = evaluator.corpus.news_ID_dict[item.rsplit("-", 1)[0]]
                pair = (ui, ni)
                if pair not in pair_map:
                    if len(pairs) >= count: break
                    pair_map[pair] = len(pairs); pairs.append(pair)
                group.append(pair_map[pair])
            if len(group) > 1: groups.append(group)
            if len(pairs) >= count: break
    if len(pairs) != count:
        raise ValueError("Sparse checkpoint gate requires exactly 8192 available unique dev pairs")
    mapped = evaluator._ensure_users(keys)
    return (np.asarray([mapped[ui] for ui, _ in pairs], dtype=np.int64),
            np.asarray([ni for _, ni in pairs], dtype=np.int64), groups)


def verify_checkpoint_gate(evaluator, behavior_path, native_verification):
    """Persist/reuse strict checkpoint evidence; leave fallback policy to caller."""
    import json
    import time
    import numpy as np
    from naml_cached_eval import _file_hash, _save_json, _save_npy, model_fingerprint
    if (native_verification.get("status") != "passed" or
            native_verification.get("rows", 0) < 8 or
            native_verification.get("rows_with_rank_disagreement") != 0 or
            native_verification.get("atol") != 1e-4 or native_verification.get("rtol") != 1e-5):
        raise ValueError("A strict native eight-row score/rank pass is required")
    evaluator._assert_current()
    if (native_verification.get("model_state_sha256") != evaluator.model_hash or
            native_verification.get("cache_identity") != evaluator.identity or
            native_verification.get("input_sha256") != _file_hash(behavior_path)):
        raise ValueError("Native evidence belongs to a different checkpoint, cache or input")
    provenance = {"schema": 1, "cache_identity": evaluator.identity,
        "model_state_sha256": evaluator.model_hash,
        "input_sha256": _file_hash(behavior_path), "unique_pairs": 8192,
        "sparse_batch_size": evaluator.score_batch_size, "dense_batch_size": 256,
        "atol": 1e-4, "rtol": 1e-5, "stable_ranks_required": True,
        "graph_backend": "sparse_csr", "dtype": "float32",
        "sparse_implementation_sha256": _file_hash(__file__),
        "evaluator_implementation_sha256": evaluator.provenance["cache_implementation_sha256"],
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}
    directory = evaluator.cache_dir / "sparse_checkpoint_gate"
    target = directory / "verification.json"
    if target.exists():
        report = json.loads(target.read_text(encoding="utf-8"))
        if report["status"] != "passed" or report["provenance"] != provenance:
            raise ValueError("Sparse checkpoint gate is rejected or its provenance changed")
        for name, expected in report["files"].items():
            if _file_hash(directory / name) != expected:
                raise ValueError("Sparse checkpoint gate evidence changed: " + name)
        return {"status": "passed", "reused": True, "report": str(target.resolve()),
                "report_sha256": _file_hash(target), "unique_pairs": 8192,
                "score_batch_size": evaluator.score_batch_size,
                "comparison": report["comparison"], "model_state_sha256": evaluator.model_hash}
    directory.mkdir(parents=True, exist_ok=False)
    users, news, groups = fixed_dev_pairs(evaluator, behavior_path)
    _save_npy(directory / "user_indices.npy", users)
    _save_npy(directory / "news_indices.npy", news)
    _save_json(directory / "rank_groups.json", groups)
    _save_json(directory / "user_keys.json", evaluator.user_keys)
    original_batch = evaluator.score_batch_size
    scores, timings = {}, {}
    try:
        for backend, size in (("dense", 256), ("sparse", original_batch)):
            evaluator.score_batch_size = size
            if evaluator.device.type == "cuda": torch.cuda.synchronize()
            started = time.monotonic()
            scores[backend] = evaluator._compute_pairs(users, news, backend=backend)
            if evaluator.device.type == "cuda": torch.cuda.synchronize()
            timings[backend] = time.monotonic() - started
            _save_npy(directory / (backend + "_scores.npy"), scores[backend])
    finally:
        evaluator.score_batch_size = original_batch
    dense, sparse = scores["dense"], scores["sparse"]
    finite = bool(np.isfinite(dense).all() and np.isfinite(sparse).all())
    allclose = bool(finite and np.allclose(sparse, dense, atol=1e-4, rtol=1e-5))
    bad_groups = [i for i, group in enumerate(groups)
                  if not np.array_equal(np.argsort(-sparse[group], kind="stable"),
                                        np.argsort(-dense[group], kind="stable"))]
    final_hash = model_fingerprint(evaluator.model)
    passed = allclose and not bad_groups and final_hash == evaluator.model_hash
    comparison = {"allclose": allclose, "finite": finite,
        "max_absolute_error": float(np.max(np.abs(sparse - dense))) if finite else None,
        "rows_with_rank_disagreement": len(bad_groups), "rank_disagreement_group_indices": bad_groups}
    names = ("user_indices.npy", "news_indices.npy", "rank_groups.json", "user_keys.json",
             "dense_scores.npy", "sparse_scores.npy")
    report = {"status": "passed" if passed else "rejected", "provenance": provenance,
        "native_verification": native_verification, "comparison": comparison,
        "timings_seconds": timings, "rank_groups": len(groups),
        "final_model_state_sha256": final_hash, "sparse_stats_cumulative": dict(evaluator.sparse.stats),
        "files": {name: _file_hash(directory / name) for name in names}}
    _save_json(target, report)
    if final_hash != evaluator.model_hash:
        raise ValueError("Checkpoint weights changed during sparse verification: " + str(target))
    if not finite:
        raise ValueError("Nonfinite checkpoint gate scores: " + str(target))
    if not passed:
        raise SparseEquivalenceError("Sparse checkpoint score/rank equivalence rejected: " + str(target), target)
    return {"status": "passed", "reused": False, "report": str(target.resolve()),
            "report_sha256": _file_hash(target), "unique_pairs": len(news),
            "score_batch_size": evaluator.score_batch_size,
            "comparison": comparison, "model_state_sha256": evaluator.model_hash}
