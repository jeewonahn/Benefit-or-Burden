"""Checkpoint evidence acceptance/rejection contracts on CPU, with 8192 keys."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from digat_sparse_attention import SparseEquivalenceError, verify_checkpoint_gate
from naml_cached_eval import _file_hash, model_fingerprint


class FakeEvaluator:
    def __init__(self, directory, mismatch=False):
        self.cache_dir = directory / "cache"
        self.model = torch.nn.Linear(2, 2).eval()
        self.model_hash = model_fingerprint(self.model)
        self.identity = "isolated-sparse-test-identity"
        self.provenance = {"cache_implementation_sha256": "test-evaluator"}
        self.score_batch_size = 512
        self.device = torch.device("cpu")
        self.corpus = SimpleNamespace(news_ID_dict={"N" + str(i): i for i in range(64)})
        self.user_keys = []
        self.sparse = SimpleNamespace(stats={"sparse_layers": 1, "dense_fallback_layers": 0})
        self.calls = []
        self.mismatch = mismatch
    def _assert_current(self):
        if model_fingerprint(self.model) != self.model_hash: raise ValueError("Model changed")
    def _history(self, text): return text
    def _key(self, user, history): return user + ":" + history
    def _ensure_users(self, keys):
        for key in keys:
            if key not in self.user_keys: self.user_keys.append(key)
        return np.asarray([self.user_keys.index(key) for key in keys])
    def _compute_pairs(self, users, news, backend=None):
        self.calls.append((backend, self.score_batch_size, len(news)))
        scores = news.astype(np.float32)
        scores[0:2] = [1.0, 1.000001]
        if self.mismatch and backend == "sparse": scores[0:2] = scores[0:2][::-1]
        return scores


class SparseGateTests(unittest.TestCase):
    def setUp(self):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.behavior = self.directory / "dev.tsv"
        candidates = " ".join("N%d-%d" % (i, i == 0) for i in range(64))
        self.behavior.write_text("".join("%d\tU%d\tT\tH\t%s\n" % (i, i, candidates)
                                       for i in range(128)), encoding="utf-8")
        self.native = {"status": "passed", "rows": 8, "rows_with_rank_disagreement": 0,
                       "atol": 1e-4, "rtol": 1e-5}
    def tearDown(self): self.temp.cleanup()

    def native_for(self, ev):
        return dict(self.native, model_state_sha256=ev.model_hash,
                    cache_identity=ev.identity, input_sha256=_file_hash(self.behavior))

    def test_full_8192_gate_and_hash_bound_reuse(self):
        ev = FakeEvaluator(self.directory)
        first = verify_checkpoint_gate(ev, self.behavior, self.native_for(ev))
        self.assertFalse(first["reused"])
        self.assertEqual(first["unique_pairs"], 8192)
        self.assertEqual(ev.calls, [("dense", 256, 8192), ("sparse", 512, 8192)])
        second = verify_checkpoint_gate(ev, self.behavior, self.native_for(ev))
        self.assertTrue(second["reused"])
        self.assertEqual(len(ev.calls), 2)
        self.assertEqual(ev.score_batch_size, 512)
        report = json.loads(Path(first["report"]).read_text(encoding="utf-8"))
        self.assertEqual(report["final_model_state_sha256"], ev.model_hash)
        evidence = Path(first["report"]).parent / "sparse_scores.npy"
        evidence.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "evidence changed"):
            verify_checkpoint_gate(ev, self.behavior, self.native_for(ev))

    def test_rank_disagreement_rejected_even_when_scores_allclose(self):
        ev = FakeEvaluator(self.directory, mismatch=True)
        with self.assertRaisesRegex(SparseEquivalenceError, "equivalence rejected") as caught:
            verify_checkpoint_gate(ev, self.behavior, self.native_for(ev))
        self.assertTrue(Path(caught.exception.evidence_path).is_file())
        report = json.loads((ev.cache_dir / "sparse_checkpoint_gate/verification.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "rejected")
        self.assertTrue(report["comparison"]["allclose"])
        self.assertEqual(report["comparison"]["rows_with_rank_disagreement"], 1)
        with self.assertRaisesRegex(ValueError, "rejected or its provenance changed"):
            verify_checkpoint_gate(ev, self.behavior, self.native_for(ev))

    def test_inadequate_native_or_changed_batch_provenance_rejected(self):
        ev = FakeEvaluator(self.directory)
        with self.assertRaisesRegex(ValueError, "strict native"):
            verify_checkpoint_gate(ev, self.behavior, dict(self.native_for(ev), rows=2))
        verify_checkpoint_gate(ev, self.behavior, self.native_for(ev))
        ev.score_batch_size = 256
        with self.assertRaisesRegex(ValueError, "provenance changed"):
            verify_checkpoint_gate(ev, self.behavior, self.native_for(ev))

    def test_changed_checkpoint_cache_or_input_native_evidence_stops(self):
        ev = FakeEvaluator(self.directory)
        for key in ("model_state_sha256", "cache_identity", "input_sha256"):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "different checkpoint, cache or input"):
                    verify_checkpoint_gate(ev, self.behavior, dict(self.native_for(ev), **{key: "changed"}))
        self.assertFalse(ev.calls)

    def test_nonfinite_scores_are_not_an_allowed_fallback_rejection(self):
        ev = FakeEvaluator(self.directory)
        ev._compute_pairs = lambda users, news, backend=None: np.full(len(news), np.nan, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "Nonfinite checkpoint gate") as caught:
            verify_checkpoint_gate(ev, self.behavior, self.native_for(ev))
        self.assertNotIsInstance(caught.exception, SparseEquivalenceError)
        report = json.loads((ev.cache_dir / "sparse_checkpoint_gate/verification.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "rejected")
        self.assertFalse(report["comparison"]["finite"])


if __name__ == "__main__": unittest.main()
