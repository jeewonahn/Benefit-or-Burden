"""Returned evaluation metadata must retain its write-time diagnostic snapshot.

Runs the real artifact-writing method with a fixed two-candidate score fixture;
no model construction, checkpoint, CUDA operation or external data is needed.
"""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from digat_cached_eval import DigatCachedEvaluator


class MetadataSnapshotTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.behavior = self.root / "behaviors.tsv"
        self.behavior.write_text("1\tU1\tT\tN1\tN1-1 N2-0\n", encoding="utf-8")
        ev = self.ev = object.__new__(DigatCachedEvaluator)
        ev.model = SimpleNamespace(eval=lambda: None)
        ev._assert_current = lambda: None
        ev.use_sparse_eval = True
        ev.sparse_verified = True
        ev.score_batch_size = 2
        ev.sparse_verification = {"score_batch_size": 2, "status": "passed"}
        ev.sparse = SimpleNamespace(
            _validate=lambda: None,
            stats={"sparse_layers": 3, "dense_fallback_layers": 0,
                   "valid_edges": 8, "dense_edges": 16, "fallback_reasons": {}})
        ev.corpus = SimpleNamespace(news_ID_dict={"N1": 1, "N2": 2})
        ev.history_length = 1
        ev._ensure_users = lambda keys: np.zeros(len(keys), dtype=np.int32)
        ev._scores = lambda users, news: (np.asarray([.8, .2], dtype=np.float32), {"requested_pairs": 2})
        ev._save_cache = lambda: None
        ev.cache_dir = self.root / "cache"
        ev.identity = "synthetic-cache"
        ev.cache_loaded = False
        ev.use_projection_cache = False
        ev.projections = None
        ev.provenance = {"graph_backend": "sparse_csr"}
        ev.timings = {"graph_scoring_seconds": 1.0}

    def test_later_evaluation_and_nested_counter_changes_preserve_first_report(self):
        first = self.ev.evaluate_file(self.behavior, self.root / "first")
        first_snapshot = copy.deepcopy(first)
        first_json = Path(first["files"]["metrics.json"])
        stored_bytes = first_json.read_bytes()
        self.ev.sparse.stats["sparse_layers"] += 9
        self.ev.sparse.stats["valid_edges"] += 24
        self.ev.sparse.stats["dense_edges"] += 48
        self.ev.sparse.stats["fallback_reasons"]["synthetic_reason"] = 1
        self.ev.timings["graph_scoring_seconds"] += 2
        second = self.ev.evaluate_file(self.behavior, self.root / "second")
        self.assertEqual(first, first_snapshot)
        self.assertEqual(first, json.loads(stored_bytes))
        self.assertEqual(first_json.read_bytes(), stored_bytes)
        self.assertEqual(second["sparse_execution_stats_cumulative"], self.ev.sparse.stats)
        self.assertIsNot(second["sparse_execution_stats_cumulative"], self.ev.sparse.stats)
        self.assertIsNot(second["sparse_execution_stats_cumulative"]["fallback_reasons"],
                         self.ev.sparse.stats["fallback_reasons"])
        self.assertEqual(first["metrics"], second["metrics"])
        for name in ("scores.npy", "entry_metrics.npy"):
            np.testing.assert_array_equal(np.load(first["files"][name]), np.load(second["files"][name]))

    def test_dense_evaluation_keeps_no_sparse_diagnostic_object(self):
        self.ev.use_sparse_eval = False
        self.ev.sparse = None
        self.ev.provenance = {"graph_backend": "dense"}
        result = self.ev.evaluate_file(self.behavior, self.root / "dense")
        self.assertIsNone(result["sparse_execution_stats_cumulative"])
        self.assertIsNone(result["sparse_verification"])
        self.assertEqual(result, json.loads(Path(result["files"]["metrics.json"]).read_text(encoding="utf-8")))

    def test_protocol_progress_keeps_deduplicated_pair_scores_separate_from_request_rows(self):
        import torch
        from unittest.mock import patch
        from evaluation_progress import evaluate_test_file
        self.behavior.write_text("1\tU1\tT\tN1\tN1-1 N2-0\n2\tU1\tT\tN1\tN1-1 N2-0\n", encoding="utf-8")
        ev = self.ev
        del ev._scores
        ev.device = torch.device("cpu")
        ev.histories = ev.graphs = ev.category_masks = ev.category_indices = np.zeros((1, 1))
        ev.corpus.news_node_ID = ev.corpus.news_graph = ev.corpus.news_graph_mask = np.zeros((3, 1))
        ev.news_device = ev.context_device = torch.zeros(3, 1)
        ev.sparse.score = lambda _news, _ctx, _history, ids, *_: ids.float() / 10
        ev.pair_keys = np.array([1], dtype=np.int64)
        ev.pair_scores = np.array([.8], dtype=np.float32)
        with patch('evaluation_progress.test_progress') as display:
            result = evaluate_test_file(ev, self.behavior, self.root / "out", "total",
                                        start_row=2, total_rows=4)
        self.assertEqual((result['rows'], result['candidates'], result['pair_cache']['new_pairs_computed']),
                         (2, 4, 1))
        scored = [event for event in display.call_args_list if event.kwargs['scored'] is not None]
        self.assertEqual(len(scored), 1)
        self.assertEqual(scored[0].args, ("total", 2, 4))
        self.assertEqual(scored[0].kwargs, {"scored": 1, "score_total": 1})
        self.assertEqual(display.call_args_list[-1].args, ("total", 4, 4))


if __name__ == "__main__":
    unittest.main()
