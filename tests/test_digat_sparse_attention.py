"""CPU-only official three-layer DIGAT comparisons in isolated child processes."""
import os
from pathlib import Path
import subprocess
import sys
import unittest
ROOT = Path(__file__).resolve().parents[1]


def exercise(case):
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "third_party/DIGAT"))
    import numpy as np
    import tempfile
    from types import SimpleNamespace
    import torch
    from graphEncoders import DIGAT
    from digat_cached_eval import history_graph
    from digat_projection_cache import DigatLayerZeroCache
    from digat_sparse_attention import SparseDIGATScore
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = SimpleNamespace(news_graph_size=4, max_history_num=4, category_num=3,
                          graph_depth=3, dropout_rate=0.2)
    torch.manual_seed(13)
    g = DIGAT(cfg, 16).cpu(); g.device = torch.device("cpu"); g.initialize(); g.eval()
    with torch.no_grad(): g.topic_node_embedding.normal_(0, 0.04)
    model = SimpleNamespace(graph_encoder=g, training=False)
    if case == "guards":
        g.train()
        try: SparseDIGATScore(model)
        except ValueError: pass
        else: raise AssertionError("Training accepted")
        g.eval(); g.double()
        try: SparseDIGATScore(model)
        except ValueError: pass
        else: raise AssertionError("Float64 accepted")
        return
    sparse = SparseDIGATScore(model)
    if case == "mask_floor":
        e = torch.randn(2, 4, 16); context = torch.randn(2, 16)
        graph = torch.eye(4, dtype=torch.bool).expand(2, -1, -1).clone()
        with torch.no_grad():
            g.news_graph_attention_ffn1[0].weight.zero_(); g.news_graph_attention_ffn2[0].weight.zero_()
            g.news_graph_attention_ffn3[0].weight.zero_(); g.news_graph_attention_ffn3[0].bias.fill_(1)
            g.news_graph_attention_a[0].weight.fill_(-1e9)
            expected = g.compute_news_graph_embeddings(0, e, graph, context)
            actual = sparse.layer("news", 0, e, graph, context)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert sparse.stats["fallback_reasons"]["mask_floor_or_nonfinite"] == 1
        return
    largest = 0.0
    for seed in (1, 7, 13, 42):
        torch.manual_seed(seed); rng = np.random.RandomState(seed)
        count, batch, history = 24, 16, 4
        vectors = torch.randn(count, 16) * 0.2
        nodes_all = torch.as_tensor(rng.randint(0, count, (count, 4)), dtype=torch.long)
        graph_all = torch.as_tensor(rng.rand(count, 4, 4) < 0.25)
        graph_all |= torch.eye(4, dtype=torch.bool).unsqueeze(0)
        mask_all = torch.as_tensor(rng.rand(count, 4) < 0.7); mask_all[:, 0] = True
        mask_all[0] = False  # Candidate-context all-masked semantics stay official.
        histories = torch.as_tensor(rng.randint(1, count, (batch, history)), dtype=torch.long)
        histories[0] = 0; histories[1] = torch.tensor([1, 0, 0, 0])
        histories[2] = torch.tensor([2, 2, 0, 0])
        # Groups of four candidates share one user's graph/history.
        for start in range(0, batch, 4): histories[start:start+4] = histories[start].clone()
        categories = np.asarray([3] + [i % 3 for i in range(1, count)])
        prepared = [history_graph(h.numpy(), categories, 3) for h in histories]
        ug = torch.as_tensor(np.stack([v[0] for v in prepared]))
        cm = torch.as_tensor(np.stack([v[1] for v in prepared]))
        ci = torch.as_tensor(np.stack([v[2] for v in prepared]))
        candidates = torch.as_tensor(rng.randint(0, count, batch), dtype=torch.long)
        nodes, ng, nm = nodes_all[candidates], graph_all[candidates], mask_all[candidates]
        if case == "empty_rows":
            ng[0, 1] = False; ug[1, 2] = False
            ng[2] = False; ug[3] = False
        with torch.inference_mode():
            contexts = g.compute_news_graph_context(vectors[nodes_all], mask_all)
            nr, ur = g(vectors[nodes], ng, nm, vectors[histories], ug, cm, ci)
            native = (nr * ur).sum(1)
            actual = sparse.score(vectors, contexts, histories, candidates, nodes, ug, cm, ci, ng, nm)
            torch.testing.assert_close(actual, native, atol=1e-4, rtol=1e-5)
            largest = max(largest, float((actual-native).abs().max()))
            for start in range(0, batch, 4):
                np.testing.assert_array_equal(np.argsort(-actual[start:start+4].numpy(), kind="stable"), np.argsort(-native[start:start+4].numpy(), kind="stable"))
            with tempfile.TemporaryDirectory() as directory:
                cache = DigatLayerZeroCache(model, vectors.numpy(), Path(directory), "synthetic-model", torch.device("cpu"))
                cached = SparseDIGATScore(model, cache)
                projected = cached.score(vectors, contexts, histories, candidates, nodes, ug, cm, ci, ng, nm)
                torch.testing.assert_close(projected, native, atol=1e-4, rtol=1e-5)
                for start in range(0, batch, 4):
                    np.testing.assert_array_equal(np.argsort(-projected[start:start+4].numpy(), kind="stable"), np.argsort(-native[start:start+4].numpy(), kind="stable"))
    if case == "empty_rows": assert sparse.stats["fallback_reasons"]["all_masked_row"] == 24
    else: assert sparse.stats["sparse_layers"] == 24 and sparse.stats["dense_fallback_layers"] == 0
    print({"case": case, "maximum_absolute_error": largest, "stats": sparse.stats})


class SparseCPUTests(unittest.TestCase):
    def run_case(self, case):
        environment = os.environ.copy(); environment["CUDA_VISIBLE_DEVICES"] = "-1"
        result = subprocess.run([sys.executable, __file__, "--cpu-case", case],
                                capture_output=True, text=True, env=environment, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    def test_three_layers_sparse_and_projection_cache(self): self.run_case("normal")
    def test_allmasked_rows_keep_dense_uniform_semantics(self): self.run_case("empty_rows")
    def test_finite_mask_floor_uses_dense_fallback(self): self.run_case("mask_floor")
    def test_training_and_non_float32_rejected(self): self.run_case("guards")

if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--cpu-case": exercise(sys.argv[2])
    else: unittest.main()
