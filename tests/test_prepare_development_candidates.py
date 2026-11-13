"""Synthetic CPU tests for matched RandS train/development preparation."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("matched_candidate_preparation", ROOT / "scripts/prepare_development_candidates.py")
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


class MatchedCandidateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "input"
        for split in ("train", "dev", "test"):
            (self.data / split).mkdir(parents=True)
        self.train_row = "T1\tU1\t11/10/2019 12:00:00 PM\tH\tP1-1 P2-1 L-0\n"
        self.dev_row = "D1\tU1\t11/10/2019 01:00:00 PM\tH\tF-1 L-0\n"
        (self.data / "train/behaviors.tsv").write_text(self.train_row, encoding="utf-8")
        (self.data / "dev/behaviors.tsv").write_text(self.dev_row, encoding="utf-8")
        (self.data / "train/behaviors_rs50.tsv").write_text("LEGACY FILE IS NOT INPUT\n", encoding="utf-8")
        (self.data / "test/behaviors.tsv").write_text("TEST MUST NOT BE OPENED\n", encoding="utf-8")
        self.ids = ["H", "P1", "P2", "F", "L", "OLD", "FUT", "LEFT", "RIGHT"] + ["N%d" % n for n in range(65)]
        news = "".join(news_id + "\tcategory\tsubcategory\ttitle\tabstract\turl\t[]\t[]\n" for news_id in self.ids)
        for split in ("train", "dev"):
            (self.data / split / "news.tsv").write_text(news, encoding="utf-8")
        stamps = {news_id: "11/10/2019 10:00:00 AM" for news_id in self.ids}
        stamps.update(OLD="11/08/2019 12:00:00 PM", FUT="11/10/2019 02:00:00 PM",
                      LEFT="11/09/2019 12:00:00 PM", RIGHT="11/10/2019 12:00:00 PM")
        self.proxy = self.data / "news_publish_time.tsv"
        self.proxy.write_text("".join(news_id + "\t" + stamps[news_id] + "\n" for news_id in self.ids), encoding="utf-8")
        self.manifest = {
            "status": "complete", "source": str(self.data),
            "output_files": {split + "/behaviors.tsv": policy.fingerprint(self.data / split / "behaviors.tsv")
                             for split in ("train", "dev")},
            "assets": {split + "/news.tsv": {"output": policy.fingerprint(self.data / split / "news.tsv")}
                       for split in ("train", "dev")},
            "RandS_training_construction": {
                "publication_proxy": policy.fingerprint(self.proxy),
                "exclusion": "Legacy retrospective train/test click union; must not be reused"},
        }
        self.save_manifest()

    def save_manifest(self):
        (self.data / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def prepare(self, name="output", split="both", seed=42):
        output = self.root / name
        result = policy.prepare(SimpleNamespace(data_root=self.data, output=output, split=split, seed=seed))
        return output, result

    def assembled(self):
        output, certificate = self.prepare()
        # Dataset assembly is explicit and separate from candidate generation.
        for split in ("train", "dev"):
            shutil.copy2(self.data / split / "behaviors.tsv", output / split / "behaviors.tsv")
        return output, certificate

    def test_both_splits_are_deterministic_and_pool_size_is_fifty(self):
        first, manifest1 = self.prepare("first")
        second, manifest2 = self.prepare("second")
        for split in ("train", "dev"):
            a = first / split / "behaviors_rs50.tsv"
            b = second / split / "behaviors_rs50.tsv"
            self.assertEqual(a.read_bytes(), b.read_bytes())
            records, _ = policy.read_imp(a)
            for _, parts, values in records:
                self.assertEqual(sum(label == 1 for _, label in values), 1)
                self.assertEqual(sum(label == 0 for _, label in values), 50)
                self.assertEqual(len(values), len({news for news, _ in values}))
                self.assertFalse({"H", "OLD", "FUT"} & {news for news, label in values if not label})
        self.assertEqual(manifest1["statistics"]["train"]["rows"], 2)
        self.assertEqual(manifest2["sampling"], policy.RECIPE)

    def test_request_time_exclusion_does_not_use_later_user_clicks(self):
        index = policy.PublicationIndex(self.proxy)
        parts = self.train_row.rstrip().split("\t")
        pool = index.eligible(parts, ["P1", "P2"], set(self.ids))
        self.assertIn("F", pool)  # F is clicked by this user only in the later dev request.
        self.assertNotIn("P1", pool)
        self.assertNotIn("P2", pool)
        self.assertNotIn("H", pool)
        self.assertNotIn("OLD", pool)
        self.assertNotIn("FUT", pool)
        self.assertIn("LEFT", pool)
        self.assertIn("RIGHT", pool)

    def test_preparation_never_opens_test_or_legacy_rands_files(self):
        original_open = Path.open
        test_root = self.data / "test"
        legacy = self.data / "train/behaviors_rs50.tsv"
        def guarded(path, *args, **kwargs):
            if path == legacy or path == test_root or test_root in path.parents:
                raise AssertionError("Forbidden data file opened: " + str(path))
            return original_open(path, *args, **kwargs)
        with patch.object(Path, "open", guarded):
            output, result = self.prepare()
        self.assertTrue((output / policy.MANIFEST_NAME).is_file())
        self.assertFalse(result["validation"]["test_files_opened"])

    def test_existing_output_is_preserved(self):
        output, _ = self.prepare()
        before = (output / policy.MANIFEST_NAME).read_bytes()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.prepare()
        self.assertEqual((output / policy.MANIFEST_NAME).read_bytes(), before)

    def test_input_hash_mismatch_fails_before_output_creation(self):
        (self.data / "dev/behaviors.tsv").write_text(self.dev_row + self.dev_row, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "differs from"):
            self.prepare()
        self.assertFalse((self.root / "output").exists())

    def test_missing_proxy_evidence_has_no_test_fallback(self):
        del self.manifest["RandS_training_construction"]["publication_proxy"]
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.prepare()
        self.assertFalse((self.root / "output").exists())

    def test_group_overlap_is_rejected(self):
        (self.data / "dev/behaviors.tsv").write_text(self.train_row, encoding="utf-8")
        self.manifest["output_files"]["dev/behaviors.tsv"] = policy.fingerprint(self.data / "dev/behaviors.tsv")
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "groups overlap"):
            self.prepare()

    def test_complete_profile_validates_actual_train_and_dev_inputs(self):
        output, certificate = self.assembled()
        result = policy.validate_matched_candidate_profile(output)
        self.assertEqual(result["profile"], policy.PROFILE)
        self.assertEqual(result["seed"], 42)
        self.assertEqual(result["outputs"], certificate["outputs"])
        self.assertEqual(result["sha256"], policy.fingerprint(output / policy.MANIFEST_NAME)["sha256"])

    def test_mixing_legacy_train_with_new_dev_is_rejected(self):
        output, _ = self.assembled()
        (output / "train/behaviors_rs50.tsv").write_text("LEGACY DIFFERENT CANDIDATES\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "differs from"):
            policy.validate_matched_candidate_profile(output)

    def test_old_profile_cannot_be_relabelled_as_matched(self):
        output, certificate = self.assembled()
        certificate["profile"] = "legacy_closed_period"
        (output / policy.MANIFEST_NAME).write_text(json.dumps(certificate), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "matched RandS50"):
            policy.validate_matched_candidate_profile(output)

    def test_partial_bundle_is_not_a_trainable_profile(self):
        output, certificate = self.prepare(split="dev")
        self.assertEqual(certificate["status"], "partial")
        self.assertFalse((output / "train").exists())
        with self.assertRaisesRegex(ValueError, "complete matched"):
            policy.validate_matched_candidate_profile(output)

    def test_output_inside_historical_source_is_rejected(self):
        historical = self.root / "historical"
        self.manifest["source"] = str(historical)
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "historical source"):
            policy.prepare(SimpleNamespace(data_root=self.data, output=historical / "new", split="both", seed=42))
        self.assertFalse(historical.exists())

    def test_proxy_cannot_be_sourced_from_test_directory(self):
        test_proxy = self.data / "test" / "proxy.tsv"
        test_proxy.write_bytes(self.proxy.read_bytes())
        self.manifest["RandS_training_construction"]["publication_proxy"] = policy.fingerprint(test_proxy)
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "Test-directory"):
            self.prepare()
        self.assertFalse((self.root / "output").exists())

    def test_sampler_retains_all_available_when_fewer_than_fifty(self):
        index = policy.PublicationIndex(self.proxy)
        parts = self.train_row.rstrip().split("\t")
        allowed = {"F", "L", "LEFT", "RIGHT", "P1", "P2", "H"}
        # Use a genuinely smaller proxy rather than hiding metadata.
        small = self.root / "small_proxy.tsv"
        small.write_text("".join(news + "\t11/10/2019 10:00:00 AM\n" for news in allowed), encoding="utf-8")
        records, _ = policy.read_imp(self.data / "train/behaviors.tsv")
        target = self.root / "small.tsv"
        stats = policy.write_split(records, policy.PublicationIndex(small), allowed, target, "train", 42)
        self.assertEqual(stats["negative_max"], 4)
        self.assertEqual(stats["negative_min"], 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)

