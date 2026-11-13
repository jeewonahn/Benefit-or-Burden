"""CPU-only execution tests for the explicit sparse rejection policy."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from digat_sparse_attention import SparseEquivalenceError
from run_small_digat import verified_evaluator


class FallbackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.evidence = self.root / "rejected.json"
        self.evidence.write_text('{"status":"rejected"}', encoding="utf-8")
        self.record = self.root / "fallback.json"
        self.calls = []
        self.releases = []
        self.failure = SparseEquivalenceError("score/rank rejection", self.evidence)
        self.changed = False

    def tearDown(self):
        self.temp.cleanup()

    def factory(self, **options):
        self.calls.append(options)
        sparse = options["sparse_eval"]
        def verify(path):
            if sparse and self.failure is not None:
                raise self.failure
            return {"status": "passed", "rows": 8, "atol": 1e-4, "rtol": 1e-5}
        def current():
            if self.changed:
                raise ValueError("Checkpoint changed")
        return SimpleNamespace(identity="sparse" if sparse else "dense",
            model_hash="same-checkpoint", verify_against_forward=verify,
            _assert_current=current, use_sparse_eval=sparse,
            use_projection_cache=options["projection_cache"],
            score_batch_size=options["score_batch_size"])

    def run_policy(self, policy="dense256", batch=256, sparse=True):
        return verified_evaluator(self.factory, self.root / "dev.tsv",
            sparse_eval=sparse, projection_cache=sparse, score_batch_size=batch,
            rejection_policy=policy, fallback_path=self.record,
            release=lambda: self.releases.append(True))

    def test_only_explicit_rejection_uses_new_dense256_identity_same_checkpoint(self):
        for batch in (256, 512):
            with self.subTest(batch=batch):
                self.calls.clear()
                old_bytes = self.evidence.read_bytes()
                ev, verification, fallback = self.run_policy(batch=batch)
                self.assertEqual(self.calls, [
                    {"sparse_eval": True, "projection_cache": True, "score_batch_size": batch},
                    {"sparse_eval": False, "projection_cache": False, "score_batch_size": 256}])
                self.assertEqual(ev.model_hash, fallback["model_state_sha256"])
                self.assertEqual(ev.identity, "dense")
                self.assertEqual(verification["status"], "passed")
                self.assertEqual(fallback["status"], "dense_verified")
                self.assertFalse(fallback["retraining"])
                self.assertFalse(fallback["tolerance_changed"])
                self.assertEqual(self.evidence.read_bytes(), old_bytes)
                self.assertEqual(json.loads(self.record.read_text(encoding="utf-8")), fallback)

    def test_abort_retains_rejection_without_dense_retry(self):
        with self.assertRaises(SparseEquivalenceError):
            self.run_policy(policy="abort")
        self.assertEqual(len(self.calls), 1)
        self.assertFalse(self.record.exists())

    def test_runtime_integrity_nonfinite_or_missing_evidence_never_fall_back(self):
        for reason in ("Cache evidence changed", "Nonfinite scores", "CUDA out of memory"):
            with self.subTest(reason=reason):
                self.calls.clear()
                self.failure = ValueError(reason)
                with self.assertRaisesRegex(ValueError, reason):
                    self.run_policy()
                self.assertEqual(len(self.calls), 1)
                self.assertFalse(self.record.exists())
        self.failure = SparseEquivalenceError("score/rank rejection", self.evidence)
        self.evidence.unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_policy()
        self.assertFalse(self.record.exists())

    def test_checkpoint_change_stops_before_fallback(self):
        self.changed = True
        with self.assertRaisesRegex(ValueError, "Checkpoint changed"):
            self.run_policy()
        self.assertEqual(len(self.calls), 1)
        self.assertFalse(self.record.exists())

    def test_sparse_pass_or_default_dense_does_not_retry(self):
        self.failure = None
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                self.calls.clear()
                ev, result, fallback = self.run_policy(sparse=sparse)
                self.assertEqual(ev.use_sparse_eval, sparse)
                self.assertEqual(result["status"], "passed")
                self.assertIsNone(fallback)
                self.assertEqual(len(self.calls), 1)
                self.assertFalse(self.record.exists())
        self.assertFalse(self.releases)


if __name__ == "__main__":
    unittest.main()
