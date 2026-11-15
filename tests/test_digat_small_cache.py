"""Real official DIGAT cache checks on synthetic TSVs; no external assets."""
import json
import importlib.util
import io
import os
from pathlib import Path
import pickle
import sys
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from digat_cached_eval import history_graph


def loop_graph(history, category, count):
    size = len(history)
    graph = np.eye(size + count, dtype=bool)
    mask = np.zeros(count + 1, dtype=bool)
    indices = np.full(size, count, dtype=np.int64)
    valid = [x for x in history if x]
    for i, news in enumerate(valid):
        c = category[news]; mask[c] = True; indices[i] = c
        graph[i, size + c] = graph[size + c, i] = True
        for j in range(i + 1, len(valid)):
            other = category[valid[j]]
            if c == other: graph[i, j] = graph[j, i] = True
            else: graph[size + c, size + other] = graph[size + other, size + c] = True
    return graph, mask, indices


class GraphTests(unittest.TestCase):
    def test_graph_matches_original_loop_for_padding_and_repeats(self):
        category = np.asarray([2, 0, 1, 0, 1, 0, 1])
        for history in ([0, 0, 0], [1, 0, 0], [1, 2, 3], [2, 2, 0], [6, 5, 4]):
            actual = history_graph(history, category, 2)
            expected = loop_graph(history, category, 2)
            for a, b in zip(actual, expected): np.testing.assert_array_equal(a, b)


class MetricTests(unittest.TestCase):
    def test_metrics_independently_match_official_scorer(self):
        from naml_cached_eval import metrics_from_ranks
        spec = importlib.util.spec_from_file_location("digat_official_scorer", ROOT / "DIGAT/evaluate.py")
        official = importlib.util.module_from_spec(spec); spec.loader.exec_module(official)
        rng = np.random.RandomState(42)
        for count in (2, 5, 10, 31, 91, 151):
            for repeat in range(20):
                labels = rng.randint(0, 2, count).astype(np.int8); labels[0] = 1; labels[1] = 0
                scores = rng.randint(0, 7, count).astype(np.float32)
                order = np.argsort(-scores, kind="stable")
                ranks = np.empty(count, dtype=np.int32); ranks[order] = np.arange(1, count + 1)
                actual = metrics_from_ranks(labels, ranks)
                truth = io.StringIO("1 " + json.dumps(labels.tolist(), separators=(",", ":")))
                prediction = io.StringIO("1 " + json.dumps(ranks.tolist(), separators=(",", ":")))
                expected = official.scoring(truth, prediction)
                np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


class CUDACacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        if os.environ.get("TORS_CACHE_TEST_DEVICE", "cuda") != "cpu" and not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA unavailable: actual DIGAT cache test did not execute")

    def test_official_graph_forward_cache_artifacts_and_reload(self):
        if os.environ.get("TORS_DIGAT_CACHE_CHILD") != "1":
            environment = os.environ.copy(); environment["TORS_DIGAT_CACHE_CHILD"] = "1"
            completed = subprocess.run([sys.executable, __file__, "CUDACacheTests.test_official_graph_forward_cache_artifacts_and_reload"],
                                       env=environment, capture_output=True, text=True, timeout=120)
            self.assertEqual(completed.returncode, 0, (completed.stdout or "") + (completed.stderr or ""))
            return
        import torch
        sys.path.insert(0, str(ROOT / "DIGAT"))
        from MIND_corpus import MIND_Corpus
        from model import Model
        from digat_cached_eval import DigatCachedEvaluator, news_categories
        torch.manual_seed(7); torch.cuda.manual_seed_all(7)
        before = Path.cwd()
        with tempfile.TemporaryDirectory(prefix="digat-cache-native-") as temp:
            root = Path(temp)
            try:
                os.chdir(root)
                cfg = SimpleNamespace(dataset="MIND-small", word_threshold=1, max_title_length=4,
                    word_embedding_dim=50, max_history_num=3, SAG_hops=1, SAG_neighbors=2,
                    news_graph_size=3, negative_sample_num=2, news_encoder="MSA",
                    graph_encoder="DIGAT", dropout_rate=0.0, MSA_head_num=2, MSA_head_dim=4,
                    attention_dim=8, graph_depth=3)
                ids = {"<PAD>": 0, **{"N" + str(i): i for i in range(1, 7)}}
                categories = {"a": 0, "b": 1}
                vocabulary = {"<PAD>": 0, "<UNK>": 1, "alpha": 2, "beta": 3}
                news = "".join("N%d\t%s\ts\t%s\tbody\turl\t[]\t[]\n" %
                    (i, "a" if i % 2 else "b", "alpha beta" if i % 2 else "beta alpha") for i in range(1, 7))
                behaviors = "1\tU1\ttime\tN1 N2\tN3-1 N4-0 N5-0\n2\tU2\ttime\tN5\tN1-1 N2-0\n"
                for split in ("train", "dev", "test"):
                    folder = root / split; folder.mkdir()
                    (folder / "news.tsv").write_text(news, encoding="utf-8")
                    (folder / "behaviors.tsv").write_text(behaviors, encoding="utf-8")
                    setattr(cfg, split + "_root", str(folder))
                files = {
                    "user_ID-MIND-small.json": {"<UNK>": 0, "U1": 1, "U2": 2},
                    "news_ID-MIND-small.json": ids, "category-MIND-small.json": categories,
                    "subCategory-MIND-small.json": {"s": 0}, "vocabulary-1-4-MIND-small.json": vocabulary,
                }
                for name, value in files.items(): (root / name).write_text(json.dumps(value), encoding="utf-8")
                with (root / "word_embedding-1-50-4-MIND-small.pkl").open("wb") as f:
                    pickle.dump(torch.randn(4, 50) * 0.1, f)
                node_ids = np.asarray([[0, 0, 0]] + [[i, i % 6 + 1, (i + 1) % 6 + 1] for i in range(1, 7)], dtype=np.int32)
                with (root / "news_graph-1-2-MIND-small.pkl").open("wb") as f:
                    pickle.dump({"news_node_ID": node_ids, "news_graph": np.ones((7, 3, 3), dtype=bool),
                                 "news_graph_mask": np.ones((7, 3), dtype=bool)}, f)
                MIND_Corpus.preprocess(cfg)
                corpus = MIND_Corpus(cfg)
                device = torch.device(os.environ.get("TORS_CACHE_TEST_DEVICE", "cuda"))
                model = Model(cfg).to(device)
                if device.type == "cpu":
                    for module in model.modules():
                        if hasattr(module, "device"): module.device = device
                model.initialize(); model.eval()
                with torch.no_grad(): model.graph_encoder.topic_node_embedding.normal_(0, 0.03)
                category = news_categories(corpus, root)
                cache_root = root / "cache"
                ev = DigatCachedEvaluator(model, corpus, category, cache_root, news_batch_size=2, score_batch_size=2)
                verification = ev.verify_against_forward(root / "dev/behaviors.tsv", max_entries=2)
                self.assertEqual(verification["status"], "passed")
                self.assertTrue(verification["native_history_graph_exact_match"])
                first = ev.evaluate_file(root / "dev/behaviors.tsv", root / "evaluation/dev")
                self.assertEqual(first["rows"], 2); self.assertEqual(first["candidates"], 5)
                self.assertEqual(first["pair_cache"]["new_pairs_computed"], 5)
                expected_dtypes = {"scores": "float32", "labels": "int8", "offsets": "int64",
                                   "news_indices": "int32", "user_indices": "int32", "entry_metrics": "float64"}
                for name, dtype in expected_dtypes.items():
                    self.assertEqual(str(np.load(first["files"][name + ".npy"]).dtype), dtype)
                old_scores = np.load(first["files"]["scores.npy"])
                del ev; torch.cuda.empty_cache()
                reloaded = DigatCachedEvaluator(model, corpus, category, cache_root, news_batch_size=2, score_batch_size=2)
                self.assertTrue(reloaded.cache_loaded)
                second = reloaded.evaluate_file(root / "test/behaviors.tsv", root / "evaluation/test")
                self.assertEqual(second["pair_cache"]["new_pairs_computed"], 0)
                self.assertEqual(second["pair_cache"]["reused_unique_pairs"], 5)
                self.assertEqual(second["pair_cache"]["hit_rate_among_unique_pairs"], 1.0)
                np.testing.assert_array_equal(old_scores, np.load(second["files"]["scores.npy"]))
                projected = DigatCachedEvaluator(model, corpus, category, cache_root, news_batch_size=2, score_batch_size=2, projection_cache=True)
                projected_verification = projected.verify_against_forward(root / "dev/behaviors.tsv", max_entries=2)
                self.assertEqual(projected_verification["status"], "passed")
                projected_result = projected.evaluate_file(root / "test/behaviors.tsv", root / "evaluation/projected")
                np.testing.assert_allclose(old_scores, np.load(projected_result["files"]["scores.npy"]), atol=1e-4, rtol=1e-5)
                self.assertNotEqual(projected.identity, reloaded.identity)
                self.assertGreater(projected.projections.bytes, 0)
                sparse = DigatCachedEvaluator(model, corpus, category, cache_root,
                    news_batch_size=2, score_batch_size=2, projection_cache=True, sparse_eval=True)
                self.assertNotEqual(sparse.identity, projected.identity)
                self.assertEqual(sparse.provenance["graph_backend"], "sparse_csr")
                self.assertEqual(len(sparse.pair_keys), 0)
                sparse._ensure_users(reloaded.user_keys)
                flat_users = np.repeat(np.load(first["files"]["user_indices.npy"]),
                                       np.diff(np.load(first["files"]["offsets.npy"])))
                flat_news = np.load(first["files"]["news_indices.npy"])
                for sparse_batch in (2, 3):
                    sparse.score_batch_size = sparse_batch
                    actual_sparse = sparse._compute_pairs(flat_users, flat_news)
                    np.testing.assert_allclose(actual_sparse, old_scores, atol=1e-4, rtol=1e-5)
                with self.assertRaisesRegex(ValueError, "verification first"):
                    sparse.evaluate_file(root / "test/behaviors.tsv", root / "evaluation/ungated_sparse")
                with self.assertRaisesRegex(ValueError, "eight available"):
                    sparse.verify_against_forward(root / "dev/behaviors.tsv")
                with torch.no_grad(): next(model.parameters()).add_(0.01)
                with self.assertRaisesRegex(ValueError, "changed"):
                    reloaded.evaluate_file(root / "test/behaviors.tsv", root / "evaluation/rejected")
            finally:
                os.chdir(before)


if __name__ == "__main__":
    unittest.main()
