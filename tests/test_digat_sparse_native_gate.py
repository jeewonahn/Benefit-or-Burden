"""Exercise native gate rejection branches without CUDA or model initialization."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from digat_cached_eval import DigatCachedEvaluator, history_graph
from digat_sparse_attention import SparseEquivalenceError


class NativeGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.behavior = root / "dev.tsv"
        candidates = " ".join("N%d-%d" % (i, i == 1) for i in range(1, 17))
        self.behavior.write_text("".join("%d\tU%d\tT\tN1 N2\t%s\n" % (i, i, candidates)
                                        for i in range(8)), encoding="utf-8")
        ev = self.ev = object.__new__(DigatCachedEvaluator)
        ev.cache_dir = root
        ev.identity = "native-test"
        ev.model_hash = "test-checkpoint"
        ev.provenance = {"graph_backend": "sparse_csr"}
        ev.device = torch.device("cpu")
        ev.use_sparse_eval = True
        ev.sparse_verified = False
        ev.user_keys, ev.user_lookup = [], {}
        ev.history_length, ev.category_count = 3, 2
        ev.categories = np.asarray([2] + [i % 2 for i in range(1, 17)])
        graph, mask, indices = history_graph([1, 2, 0], ev.categories, 2)
        ev.corpus = SimpleNamespace(news_ID_dict={"N" + str(i): i for i in range(1, 17)},
            news_title_text=np.zeros((17, 4), dtype=np.int64),
            news_title_mask=np.ones((17, 4), dtype=bool),
            news_node_ID=np.tile(np.arange(17)[:, None], (1, 2)),
            news_graph=np.ones((17, 2, 2), dtype=bool),
            news_graph_mask=np.ones((17, 2), dtype=bool),
            dev_user_history_graph=np.stack([graph] * 8),
            dev_user_history_category_mask=np.stack([mask] * 8),
            dev_user_history_category_indices=np.stack([indices] * 8))
        self.reference = np.asarray([1.0, 1.000001] + list(np.linspace(0.8, 0.1, 14)), dtype=np.float32)
        self.actual = self.reference.copy()
        ev.model = lambda *args: torch.as_tensor(self.reference).unsqueeze(0)
        ev._assert_current = lambda: None
        ev._compute_pairs = lambda users, news: self.actual.copy()

    def tearDown(self):
        self.temp.cleanup()

    def test_rank_only_native_rejection_persists_scores(self):
        self.actual[:2] = self.actual[:2][::-1]
        self.assertTrue(np.allclose(self.reference, self.actual, atol=1e-4, rtol=1e-5))
        with self.assertRaisesRegex(SparseEquivalenceError, "stable rank mismatch") as caught:
            self.ev.verify_against_forward(self.behavior)
        report = json.loads(Path(caught.exception.evidence_path).read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "rejected")
        self.assertEqual(report["row"], 0)
        self.assertEqual(len(report["native_scores"]), 16)
        self.assertFalse(self.ev.sparse_verified)

    def test_nonfinite_native_never_authorizes_fallback(self):
        self.reference[0] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite DIGAT") as caught:
            self.ev.verify_against_forward(self.behavior)
        self.assertNotIsInstance(caught.exception, SparseEquivalenceError)
        self.assertEqual(len(list(self.ev.cache_dir.glob("sparse_native_rejection_*.json"))), 1)

    def test_dense_records_tiny_rank_difference_without_sparse_rejection(self):
        self.ev.use_sparse_eval = False
        self.actual[:2] = self.actual[:2][::-1]
        result = self.ev.verify_against_forward(self.behavior)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["rows_with_rank_disagreement"], 8)
        self.assertFalse(result["stable_ranks_required"])

    def test_relaxed_tolerance_short_native_or_test_split_is_rejected(self):
        for options in ({"atol": 1e-3}, {"rtol": 1e-4}, {"max_entries": 7},
                        {"max_candidates": 15}, {"native_split": "test"}):
            with self.subTest(options=options):
                with self.assertRaisesRegex(ValueError, "Sparse native gate"):
                    self.ev.verify_against_forward(self.behavior, **options)


if __name__ == "__main__":
    unittest.main()
