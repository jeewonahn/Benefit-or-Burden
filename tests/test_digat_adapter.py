"""Behavioral tests for DIGAT staging/configuration, without model dependencies."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import digat_prepare as prepare
import digat_run as runner


class DIGATAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "source"
        news = "".join(f"N{i}\tworld\tworld\tTitle {i}\tAbstract {i}\turl\t[]\t[]\n" for i in range(1, 5))
        for split_index, split in enumerate(("train", "dev", "test")):
            directory = self.data / split
            directory.mkdir(parents=True)
            (directory / "news.tsv").write_text(news, encoding="utf-8")
            (directory / "behaviors.tsv").write_text(
                f"17\tU1\t11/01/2019 01:0{split_index}:00 AM\tN3 N4\tN1-1 N2-0\n",
                encoding="utf-8",
            )
            (directory / "behaviors_rs50.tsv").write_text(
                f"17\tU1\t11/01/2019 01:0{split_index}:00 AM\tN3 N4\tN2-0 N1-1 N4-0\n",
                encoding="utf-8",
            )
            (directory / "behaviors_tt.tsv").write_text(
                f"17\tU1\t11/01/2019 01:0{split_index}:00 AM\tN3 N4\tN4-0 N2-0 N1-1\n",
                encoding="utf-8",
            )
        self.prepared = self.root / "prepared"

    def candidate_profile_fixture(self):
        # This mocked certificate isolates byte-preserving adapter behavior.
        # Candidate-generation correctness is tested by its own synthetic suite.
        outputs = {}
        for split in ("train", "dev"):
            path = self.data / split / "behaviors_rs50.tsv"
            outputs[split] = {"path": split + "/behaviors_rs50.tsv",
                              "sha256": prepare.sha256_file(path), "bytes": path.stat().st_size}
        certificate = self.data / prepare.MATCHED_CERTIFICATE
        from prepare_development_candidates import RECIPE
        synthetic_evidence = {"bytes": 1, "sha256": "0" * 64}
        certificate.write_text(json.dumps({
            "schema": 1, "status": "complete", "profile": prepare.MATCHED_PROFILE,
            "seed": 42, "outputs": outputs, "sampling": dict(RECIPE),
            "publication_proxy": synthetic_evidence, "source_dataset_manifest": synthetic_evidence,
        }), encoding="utf-8")
        return {"path": str(certificate), "sha256": prepare.sha256_file(certificate),
                "profile": prepare.MATCHED_PROFILE, "seed": 42, "outputs": outputs}

    def stage(self, dataset="200k"):
        proof = self.candidate_profile_fixture()
        with patch.object(prepare, "matched_candidate_profile", return_value=proof):
            return prepare.prepare(dataset, self.data, "rs50", "auto", "tt", self.prepared)

    def args(self, **overrides):
        glove = self.root / "glove"
        glove.mkdir(exist_ok=True)
        (glove / "glove.840B.300d.txt").write_text("synthetic-not-a-model\n", encoding="utf-8")
        sag = self.root / "sag"
        sag.mkdir(exist_ok=True)
        (sag / "modules.json").write_text("[]", encoding="utf-8")
        values = dict(prepared=self.prepared, output=self.root / "run", seed=8,
                      epochs=9, dropout=0.17, settings=None, mode="train",
                      checkpoint=None, device_id=0, glove_dir=glove, sag_model=sag,
                      allow_model_download=False, acknowledge_unverified_profile=False,
                      check_only=True)
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_protocols_candidates_order_and_bytes_preserved(self):
        manifest = self.stage()
        self.assertEqual(manifest["protocols"], {"train": "rs50", "dev": "rs50", "test": "tt"})
        self.assertEqual(manifest["dataset_display_name"], "MIND-200K")
        for split, name in (("train", "behaviors_rs50.tsv"), ("dev", "behaviors_rs50.tsv"), ("test", "behaviors_tt.tsv")):
            self.assertEqual((self.data / split / name).read_bytes(),
                             (self.prepared / "dataset" / split / "behaviors.tsv").read_bytes())
        self.assertEqual(prepare.validate_prepared(self.prepared), manifest)

    def test_dev_test_overlap_rejected_despite_different_candidates_and_ids(self):
        row = (self.data / "dev" / "behaviors_rs50.tsv").read_text().rstrip("\n").split("\t")
        row[0] = "999"
        row[4] = "N4-0 N1-1"
        (self.data / "test" / "behaviors_tt.tsv").write_text("\t".join(row) + "\n")
        with self.assertRaisesRegex(ValueError, "Dev/test share"):
            self.stage()
        self.assertFalse(self.prepared.exists())

    def test_training_and_development_must_match_before_output(self):
        for train, dev in (("imp", "rs50"), ("rs50", "imp")):
            with self.assertRaisesRegex(ValueError, "matching"):
                prepare.prepare("small", self.data, train, dev, "tt", self.prepared)
            self.assertFalse(self.prepared.exists())

    def test_imp_training_automatically_uses_imp_dev_without_randS_certificate(self):
        manifest = prepare.prepare("small", self.data, "imp", "auto", "tt", self.prepared)
        self.assertEqual(manifest["protocols"]["dev"], "imp")
        self.assertNotIn("matched_candidate_profile", manifest)
        self.assertEqual((self.prepared / "dataset/dev/behaviors.tsv").read_bytes(),
                         (self.data / "dev/behaviors.tsv").read_bytes())

    def test_missing_randS_dev_does_not_fallback_to_imp(self):
        (self.data / "dev/behaviors_rs50.tsv").unlink()
        with self.assertRaises(FileNotFoundError):
            prepare.prepare("small", self.data, "rs50", "auto", "tt", self.prepared)
        self.assertFalse(self.prepared.exists())

    def test_old_mismatched_prepared_manifest_is_rejected(self):
        self.stage()
        path = self.prepared / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["protocols"]["dev"] = "imp"
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "matching"):
            prepare.validate_prepared(self.prepared)

    def test_real_candidate_preparer_integrates_without_a_model(self):
        # Reuse the independent preparer's invented data; never read MIND.
        from test_prepare_development_candidates import MatchedCandidateTests
        fixture = MatchedCandidateTests(methodName="test_both_splits_are_deterministic_and_pool_size_is_fifty")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        dataset, _ = fixture.assembled()
        for split in ("train", "dev", "test"):
            (dataset / split).mkdir(exist_ok=True)
            (dataset / split / "news.tsv").write_bytes((fixture.data / "train/news.tsv").read_bytes())
        (dataset / "test/behaviors_tt.tsv").write_text(
            "X1\\tU2\\t11/11/2019 01:00:00 PM\\tH\\tF-1 L-0\\n".replace("\\t", "\t").replace("\\n", "\n"),
            encoding="utf-8")
        target = fixture.root / "digat-staged"
        manifest = prepare.prepare("small", dataset, "rs50", "auto", "tt", target)
        self.assertEqual(manifest["protocols"]["dev"], "rs50")
        self.assertEqual(prepare.validate_prepared(target), manifest)
        self.assertEqual((target / "dataset/dev/behaviors.tsv").read_bytes(),
                         (dataset / "dev/behaviors_rs50.tsv").read_bytes())

    def test_copied_candidate_certificate_is_pinned(self):
        self.stage()
        certificate = self.prepared / prepare.MATCHED_CERTIFICATE
        certificate.write_text(certificate.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "certificate.*changed"):
            prepare.validate_prepared(self.prepared)

    def test_modified_prepared_data_rejected(self):
        self.stage()
        path = self.prepared / "dataset" / "test" / "behaviors.tsv"
        path.write_bytes(path.read_bytes().replace(b"N4-0 N2-0 N1-1", b"N1-1 N2-0 N4-0"))
        with self.assertRaisesRegex(ValueError, "modified"):
            prepare.validate_prepared(self.prepared)

    def test_no_overwrite_of_existing_output(self):
        self.stage()
        before = (self.prepared / "manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.stage()
        self.assertEqual((self.prepared / "manifest.json").read_bytes(), before)

    def test_missing_protocol_file_does_not_fallback(self):
        (self.data / "test" / "behaviors_tt.tsv").unlink()
        with self.assertRaises(FileNotFoundError):
            self.stage()
        self.assertFalse(self.prepared.exists())

    def test_unknown_news_rejected_before_output(self):
        (self.data / "dev" / "behaviors_rs50.tsv").write_text("1\tU1\ttime\tN3\tN999-1 N2-0\n")
        with self.assertRaisesRegex(ValueError, "unknown"):
            self.stage()
        self.assertFalse(self.prepared.exists())

    def test_train_no_negative_and_eval_single_class_rejected(self):
        for split, filename in (("train", "behaviors_rs50.tsv"), ("dev", "behaviors_rs50.tsv"), ("test", "behaviors_tt.tsv")):
            path = self.data / split / filename
            original = path.read_bytes()
            path.write_text("1\tU1\ttime\tN3\tN1-1\n")
            with self.assertRaisesRegex(ValueError, "both required"):
                self.stage()
            self.assertFalse(self.prepared.exists())
            path.write_bytes(original)

    def test_manifest_cannot_redirect_to_another_protocol_file(self):
        self.stage()
        path = self.prepared / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["files"]["test"]["behaviors"]["relative_path"] = "../source/test/behaviors.tsv"
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "fixed dataset"):
            prepare.validate_prepared(self.prepared)

    def test_truth_preserves_candidate_labels_and_row_order(self):
        self.stage()
        target = self.root / "truth.txt"
        runner.write_truth(self.prepared / "dataset" / "test" / "behaviors.tsv", target)
        self.assertEqual(target.read_text(), "1 [0,0,1]")

    def test_preflight_has_no_model_import_or_output_side_effect(self):
        self.stage()
        args = self.args()
        before = set(sys.modules)
        manifest, vendor, settings, glove, sag = runner.preflight(args)
        self.assertFalse(args.output.exists())
        self.assertEqual(manifest["dataset"], "200k")
        self.assertEqual(settings["epoch"], 9)
        self.assertEqual(settings["dropout_rate"], 0.17)
        self.assertEqual(settings["seed"], 8)
        self.assertNotIn("torch", set(sys.modules) - before)

    def test_negative_seed_and_missing_checkpoint_fail_closed(self):
        self.stage()
        with self.assertRaisesRegex(ValueError, "seed"):
            runner.preflight(self.args(seed=-1))
        with self.assertRaisesRegex(ValueError, "Standalone dev/test is disabled"):
            runner.preflight(self.args(mode="test"))

    def test_duplicate_candidate_id_rejected(self):
        path = self.data / "train" / "behaviors_rs50.tsv"
        path.write_text(path.read_text().replace("N4-0", "N2-0"))
        with self.assertRaisesRegex(ValueError, "duplicate candidate"):
            self.stage()
        self.assertFalse(self.prepared.exists())

    def test_standalone_evaluation_with_existing_checkpoint_is_also_rejected(self):
        self.stage()
        checkpoint = self.root / "model"
        checkpoint.write_bytes(b"synthetic-not-a-checkpoint")
        with self.assertRaisesRegex(ValueError, "training-time"):
            runner.preflight(self.args(mode="dev", checkpoint=checkpoint))

    def test_no_implicit_model_download(self):
        self.stage()
        args = self.args()
        args.sag_model = None
        with self.assertRaisesRegex(ValueError, "allow-model-download"):
            runner.preflight(args)

    def test_200k_config_preserves_identity_and_explicit_settings(self):
        manifest = self.stage()
        settings = runner.effective_settings(self.args())

        class StubConfig:
            def __init__(self):
                self.parse_argument()

        config_type = runner.make_config_class(StubConfig, self.prepared, settings, manifest)
        config = config_type()
        self.assertEqual(config.source_dataset, "200k")
        self.assertEqual(config.dataset, "MIND-small")
        self.assertEqual(config.source_protocols, {"train": "rs50", "dev": "rs50", "test": "tt"})
        self.assertEqual(config.epoch, 9)
        self.assertEqual(config.dropout_rate, 0.17)
        self.assertEqual(config.news_graph_size, 26)
        self.assertEqual(Path(config.test_root), self.prepared / "dataset" / "test")
        self.assertFalse(config.historical_R2_settings_verified)

    def test_checkpoint_staging_uses_relative_windows_safe_path(self):
        checkpoint = self.root / "model-weights"
        checkpoint.write_bytes(b"synthetic-checkpoint-not-model-weights")
        output = self.root / "checkpoint-run"
        output.mkdir()
        relative = runner.stage_checkpoint(checkpoint, output, prepare.sha256_file(checkpoint), "MSA-DIGAT")
        self.assertEqual(relative, "input_checkpoint/MSA-DIGAT")
        self.assertFalse(Path(relative).is_absolute())
        self.assertNotIn(":", relative)
        self.assertEqual((output / relative).read_bytes(), checkpoint.read_bytes())

    def test_upstream_source_integrity(self):
        metadata = runner.verify_upstream()
        self.assertEqual(metadata["commit"], runner.COMMIT)
        self.assertIn("graphEncoders.py", metadata["sha256"])
        self.assertIn("LICENSE", metadata["sha256"])

    def test_settings_cannot_override_input_identity_or_labels(self):
        self.stage()
        path = self.root / "settings.json"
        path.write_text(json.dumps({"dataset": "MIND-large"}))
        with self.assertRaisesRegex(ValueError, "only documented"):
            runner.preflight(self.args(settings=path))


if __name__ == "__main__":
    unittest.main()
