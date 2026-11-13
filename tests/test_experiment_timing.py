"""Timer-boundary regression cases; no real model work."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from experiment_timing import PROTOCOLS, normalize_timing


class TimingTests(unittest.TestCase):
    def legacy(self):
        return {"status": "completed", "elapsed_seconds": 162.4, "preprocess_seconds": 14.2,
                "trained_epochs": [{"epoch": 1, "train_seconds": 95.4}],
                "epoch_records": [{"epoch": 1, "train_seconds": 95.4, "dev": {"elapsed_seconds": 1.1}}],
                "evaluations": {p: {"elapsed_seconds": i + 1, "cache_timings_cumulative": {"user_encoding_seconds": 5}}
                                for i, p in enumerate(PROTOCOLS)},
                "embedding_builds": [{"constructor_seconds": 1.5}, {"constructor_seconds": .8}]}

    def test_old_internal_elapsed_never_becomes_process_wall(self):
        row = normalize_timing(self.legacy())
        self.assertIsNone(row["process_wall_seconds"])
        self.assertEqual(row["internal_total_seconds"], 162.4)
        self.assertEqual(row["internal_total_scope"], "M:manifest_internal_elapsed")
        self.assertEqual(row["training_seconds"], 95.4)
        self.assertEqual(row["test_protocol_sum_seconds"], 15)

    def test_supervisor_and_internal_timers_remain_separate(self):
        row = normalize_timing(self.legacy(), {"status": "completed", "process_wall_seconds": 172.9})
        self.assertEqual(row["process_wall_seconds"], 172.9)
        self.assertEqual(row["manifest_elapsed_seconds"], 162.4)

    def test_incomplete_protocol_set_has_no_full_test_sum(self):
        info = self.legacy()
        del info["evaluations"]["total5"]
        info["status"] = "evaluating"
        row = normalize_timing(info)
        self.assertIsNone(row["test_protocol_sum_seconds"])
        self.assertEqual(row["test_completed_protocol_sum_seconds"], 10)
        self.assertEqual(row["test_protocols_completed"], 4)

    def test_cache_snapshots_are_not_added_repeatedly(self):
        row = normalize_timing(self.legacy())
        self.assertAlmostEqual(row["cache_constructor_seconds"], 2.3)
        self.assertEqual(row["dev_seconds"], 1.1)

    def test_DIGAT_training_test_loop_runner_and_process_scopes(self):
        info = {"status": "completed", "elapsed_seconds": 290,
                "timings_seconds": {"training": 100, "dev_evaluation": 5, "preprocessing": 20,
                                    "cache_construction": 7, "tests": dict.fromkeys(PROTOCOLS, 20),
                                    "test_total": 110, "total": 300, "evaluation_configuration_benchmark": 9},
                "cache_construction_records": [{"seconds": 3}, {"seconds": 4}]}
        row = normalize_timing(info, {"status": "completed", "process_wall_seconds": 312})
        self.assertEqual(row["training_seconds"], 100)
        self.assertEqual(row["test_protocol_sum_seconds"], 100)
        self.assertEqual(row["test_loop_wall_seconds"], 110)
        self.assertEqual(row["cache_constructor_seconds"], 7)
        self.assertEqual(row["internal_total_seconds"], 300)
        self.assertEqual(row["manifest_elapsed_seconds"], 290)
        self.assertEqual(row["process_wall_seconds"], 312)

    def test_initialized_zeros_are_not_completed_phase_measurements(self):
        info = {"status": "preparing", "timings_seconds": {"training": 0, "dev_evaluation": 0,
                                                           "test_total": 0, "cache_construction": 0}}
        row = normalize_timing(info)
        for key in ("training_seconds", "dev_seconds", "test_protocol_sum_seconds", "test_loop_wall_seconds", "cache_constructor_seconds"):
            self.assertIsNone(row[key])

    def test_completed_metric_artifact_precedence_is_explicit(self):
        info = self.legacy()
        row = normalize_timing(info, evaluation_metadata={"imp": {"elapsed_seconds": 2}})
        self.assertEqual(row["protocol_seconds"]["imp"], 2)
        self.assertTrue(row["warnings"])

    def test_recovery_total_sums_observed_segments_without_reboot_gap(self):
        info = self.legacy()
        info['execution_segments'] = [
            {'job_name': 'original', 'start_unix': 1000, 'end_unix': 1162.4, 'duration_seconds': 162.4},
            {'job_name': 'recovery', 'start_unix': 100000, 'end_unix': 100020, 'duration_seconds': 20},
        ]
        info['timings_seconds'] = {'total': 182.4}
        row = normalize_timing(info, {'status': 'completed', 'process_wall_seconds': 25})
        self.assertAlmostEqual(row['internal_total_seconds'], 182.4)
        self.assertTrue(row['internal_total_scope'].startswith('S:'))
        self.assertIsNone(row['process_wall_seconds'])
        self.assertIsNone(row['runner_entry_total_seconds'])
        self.assertEqual(row['recovery_process_wall_seconds'], 25)
        self.assertIn('unmeasured interrupted work', row['warnings'][0])

    def test_recovery_rejects_total_that_includes_reboot_gap(self):
        info = {'execution_segments': [{'duration_seconds': 10}, {'duration_seconds': 20}],
                'timings_seconds': {'total': 3600}}
        with self.assertRaisesRegex(ValueError, 'sum of recorded'):
            normalize_timing(info)

    def test_interrupted_supervisor_overrides_stale_evaluating_status(self):
        row = normalize_timing({'status': 'evaluating'}, {'status': 'interrupted_for_reboot'})
        self.assertEqual(row['status'], 'interrupted_for_reboot')
        self.assertIsNone(row['process_wall_seconds'])


if __name__ == "__main__":
    unittest.main()
