"""Known-answer saved-artifact fixtures; no model imports or real data."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import summarize_small_experiment as summary


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def artifact(self, model, protocol, value, training="imp", groups=8):
        prefix = self.root / model / training / ("test_" + protocol)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        records = [{"impression_id": str(i), "user_id": "U" + str(i),
                    "time": "11/15/2019 01:00:00 AM", "history_sha256": "history" + str(i),
                    "positive_news": ["N" + str(i)], "candidate_count": 11}
                   for i in range(groups)]
        values = np.full((groups, 4), value, dtype=np.float64)
        np.save(str(prefix) + ".entry_metrics.npy", values, allow_pickle=False)
        Path(str(prefix) + ".records.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
        metadata = {"metric_order": list(summary.METRICS),
                    "metrics": dict.fromkeys(summary.METRICS, value), "rows": groups,
                    "candidates": groups * 11,
                    "input": {"sha256": hashlib.sha256(protocol.encode()).hexdigest()}}
        Path(str(prefix) + ".metrics.json").write_text(json.dumps(metadata), encoding="utf-8")
        return summary.load_artifact(model, training, protocol, prefix)

    def cohort(self, ties=False):
        artifacts = []
        for model in ("NAML", "NRMS"):
            for protocol in summary.PROTOCOLS:
                value = .8 if model == "NAML" else .6
                if protocol == "imp":
                    value = 1.4 - value
                if ties and protocol == "total5":
                    value = .7
                artifacts.append(self.artifact(model, protocol, value))
        return artifacts

    def test_known_reversal_and_paired_reproducible_samples(self):
        artifacts = self.cohort()
        result = summary.compute_tables(artifacts)
        for row in result["table3"]:
            self.assertAlmostEqual(row["matching_percent"], 0 if row["protocol"] == "imp" else 100)
            self.assertEqual(row["eligible_comparisons"], 10)
        self.assertEqual([row["seed"] for row in result["samples"]], list(range(42, 52)))
        self.assertTrue(all(row["group_count"] == 4 for row in result["samples"]))
        self.assertEqual(result["samples"], summary.compute_tables(artifacts)["samples"])
        by_seed = {}
        for row in result["table3_pairs"]:
            by_seed.setdefault(row["seed"], set()).add(row["protocol"])
        self.assertTrue(all(protocols == set(summary.PROTOCOLS[:-1]) for protocols in by_seed.values()))

    def test_reference_ties_are_not_inflated_matches(self):
        result = summary.compute_tables(self.cohort(ties=True))
        for row in result["table3"]:
            self.assertIsNone(row["matching_percent"])
            self.assertIsNone(row["macro_pair_matching_percent"])
            self.assertEqual(row["reference_ties_excluded"], 10)
            self.assertEqual(row["eligible_comparisons"], 0)

    def test_candidate_tie_counts_as_mismatch(self):
        rows = [{"aggregation": "native_entry_macro", "protocol": "imp", "metric": "AUC",
                 "model_a": "NAML", "model_b": "NRMS", "eligible": 1, "matched": 0,
                 "candidate_sign": 0}]
        row = summary.pair_summary(rows)[0]
        self.assertEqual(row["matching_percent"], 0)
        self.assertEqual(row["candidate_ties_with_nontied_reference"], 1)

    def test_primary_matching_rate_gives_each_model_pair_equal_weight(self):
        common = {"aggregation": "native_entry_macro", "protocol": "imp", "metric": "AUC",
                  "model_a": "NAML", "eligible": 1, "candidate_sign": 1}
        rows = [dict(common, model_b="NRMS", matched=1) for _ in range(2)]
        rows += [dict(common, model_b="NPA", matched=0) for _ in range(8)]
        row = summary.pair_summary(rows)[0]
        self.assertEqual(row["matching_percent"], 50)
        self.assertEqual(row["pooled_matching_percent"], 20)

    def test_request_macro_does_not_overweight_multiple_positive_entries(self):
        grouped = (np.array([[2., 2., 2., 2.], [0., 0., 0., 0.]]), np.array([2, 1]))
        self.assertAlmostEqual(summary.means(grouped, np.arange(2), "native_entry_macro")[0], 2 / 3)
        self.assertAlmostEqual(summary.means(grouped, np.arange(2), "request_macro")[0], .5)

    def test_table5_gain_and_no_missing_zero(self):
        imp = self.artifact("NAML", "total5", .5)
        rs = self.artifact("NAML", "total5", .75, training="rs50")
        result = summary.compute_tables([imp, rs])
        self.assertEqual(len(result["table5"]), 8)
        self.assertTrue(all(row["relative_gain_percent"] == 50 for row in result["table5"]))
        self.assertEqual(summary.compute_tables([imp])["table5"], [])
        imp.values[:] = 0
        result = summary.compute_tables([imp, rs])
        self.assertTrue(all(row["relative_gain_percent"] is None for row in result["table5"]))

    def test_training_cohorts_stay_separate(self):
        artifacts = self.cohort()
        for protocol in summary.PROTOCOLS:
            artifacts.append(self.artifact("DIGAT", protocol, .9, training="rs50"))
        primary = summary.compute_tables(artifacts)
        auxiliary = summary.compute_tables(artifacts, ranking_training="rs50")
        self.assertEqual(primary["table2_3_cohort"], ["NAML", "NRMS"])
        self.assertEqual(auxiliary["table2_3_cohort"], ["DIGAT"])
        self.assertEqual(auxiliary["table3"], [])
        self.assertTrue(all(row["train_type"] == "rs50" for row in auxiliary["table2"]))

    def test_corrupt_metric_artifact_is_rejected(self):
        artifact = self.artifact("NAML", "imp", .5)
        np.save(str(artifact.prefix) + ".entry_metrics.npy", np.full((8, 4), .7))
        with self.assertRaisesRegex(ValueError, "Logged mean"):
            summary.load_artifact("NAML", "imp", "imp", artifact.prefix)

    def test_input_hash_mismatch_is_rejected(self):
        first = self.artifact("NAML", "imp", .5)
        second = self.artifact("NRMS", "imp", .5)
        second.report["input"]["sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "Different evaluated input"):
            summary.compute_tables([first, second])

    def test_gt_alignment_excludes_mismatched_requests_and_reports_loss(self):
        first = self.artifact("NAML", "imp", .5)
        second = self.artifact("NAML", "total5", .5)
        key = next(iter(second.positives))
        second.positives[key] = {"different": 1}
        keys, audit = summary.aligned_groups([first, second])
        self.assertEqual(len(keys), 7)
        self.assertEqual(audit["gt_mismatch_groups"], 1)
        self.assertEqual(audit["excluded_groups"], 1)

    def test_timestamp_normalization_and_group_history_identity(self):
        self.assertEqual(summary.normalize_time("11/15/2019 01:00:00 AM"),
                         summary.normalize_time("2019-11-15T01:00:00"))
        with self.assertRaises(ValueError):
            summary.normalize_time("not a timestamp")
        self.assertNotEqual(summary.group_key({"user_id": "U", "time": "2019-11-15", "history_sha256": "a"}),
                            summary.group_key({"user_id": "U", "time": "2019-11-15", "history_sha256": "b"}))

    def test_pending_end_to_end_and_output_preservation(self):
        artifact = self.artifact("NAML", "imp", .5)
        index = self.root / "index.json"
        index.write_text(json.dumps([{"model": "NAML", "train_type": "imp",
                                     "run_dir": str(artifact.prefix.parent),
                                     "evaluations": {"imp": "test_imp"}}]), encoding="utf-8")
        args = argparse.Namespace(run_index=index, output_dir=self.root / "report",
                                  resample_fraction=.5, repetitions=10, seed=42, tie_tolerance=1e-12)
        report = summary.summarize(args)
        self.assertEqual(len(report["source_artifacts"]), 1)
        self.assertEqual(report["cohort"], [])
        self.assertTrue(any(row["model"] == "DIGAT" and row["status"] == "pending" for row in report["status"]))
        before = (args.output_dir / "summary.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "must be new or empty"):
            summary.summarize(args)
        self.assertEqual(before, (args.output_dir / "summary.json").read_bytes())


if __name__ == "__main__":
    unittest.main()
