"""Synthetic protocol-policy checks; no dataset or model execution."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from training_validation import resolve_dev_path, resolve_dev_protocol, validate_candidate_profile


class TrainingValidationTests(unittest.TestCase):
    def test_automatic_protocol_matches_each_training_profile(self):
        for training in ("imp", "rs50"):
            for requested in (None, "auto", training):
                self.assertEqual(resolve_dev_protocol(training, requested), training)

    def test_imp_training_rejects_rands_development(self):
        with self.assertRaisesRegex(ValueError, "requires matching"):
            resolve_dev_protocol("imp", "rs50")

    def test_rands_training_rejects_imp_development(self):
        with self.assertRaisesRegex(ValueError, "requires matching"):
            resolve_dev_protocol("rs50", "imp")

    def test_unsupported_protocol_is_not_silently_reinterpreted(self):
        for training, requested in (("tt", "auto"), ("rs50", "rs30"), ("imp", "tt")):
            with self.assertRaises(ValueError):
                resolve_dev_protocol(training, requested)

    def test_matching_paths_stay_in_the_development_split(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.assertEqual(resolve_dev_path(root, "imp"), (root / "dev/behaviors.tsv").resolve())
            self.assertEqual(resolve_dev_path(root, "rs50"), (root / "dev/behaviors_rs50.tsv").resolve())



class CandidateCertificateTests(unittest.TestCase):
    def test_imp_does_not_require_or_read_a_rands_certificate(self):
        validator=Mock(side_effect=AssertionError("Must not read RandS inputs for IMP"))
        with patch.dict(sys.modules,{"prepare_development_candidates":
                                   SimpleNamespace(validate_matched_candidate_profile=validator)}):
            self.assertIsNone(validate_candidate_profile(Path("invented"),"imp"))
        validator.assert_not_called()

    def test_rands_preserves_the_validated_certificate_fingerprint(self):
        certificate={"path":"synthetic/matched_rands50_manifest.json","bytes":3,
                     "sha256":"invented-sha","profile":"request_time_history_and_current_positives_v1",
                     "seed":42,"outputs":{"train":{},"dev":{}}}
        validator=Mock(return_value=certificate)
        root=Path("synthetic")
        with patch.dict(sys.modules,{"prepare_development_candidates":
                                   SimpleNamespace(validate_matched_candidate_profile=validator)}):
            self.assertEqual(validate_candidate_profile(root,"rs50"),certificate)
        validator.assert_called_once_with(root)

    def test_invalid_or_old_training_profile_cannot_be_ignored(self):
        validator=Mock(side_effect=ValueError("Mismatched train/dev candidate profile"))
        with patch.dict(sys.modules,{"prepare_development_candidates":
                                   SimpleNamespace(validate_matched_candidate_profile=validator)}):
            with self.assertRaisesRegex(ValueError,"candidate profile"):
                validate_candidate_profile(Path("synthetic"),"rs50")


if __name__ == "__main__":
    unittest.main()

