"""Synthetic cooldown accounting only; never suspend or resume a process."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from thermal_timing import attach_net_times, merge_intervals, pause_account


def event(kind, when, interval="a", **extra):
    result = {"event": kind, "event_unix": when, "event_monotonic": when,
              "interval_id": interval, "job": "example", "pid": 22, "create_time": 1}
    result.update(extra)
    return result


def log(*events):
    return {"complete_read": True, "warnings": [], "events": [
        {"event": "guard_start", "event_unix": 0, "event_monotonic": 0,
         "no_managed_pauses_before_guard_start": True}, *events]}


def wall_row():
    return {"training_seconds": 30., "dev_seconds": 5., "process_wall_seconds": 100.,
            "internal_total_seconds": 90., "protocol_seconds": {"imp": 10., "rs30": 10., "rs90": 10., "rs150": 10., "total5": 10.}}


class ThermalTimingTests(unittest.TestCase):
    def test_confirmed_zero_requires_no_phase_boundaries(self):
        row = attach_net_times(wall_row(), {}, {}, log(), "example")
        self.assertEqual(row["managed_cooldown_seconds"], 0)
        self.assertEqual(row["training_net_seconds"], 30)
        self.assertEqual(row["test_protocol_sum_net_seconds"], 50)
        self.assertEqual(row["process_net_seconds"], 100)
        self.assertEqual(row["internal_total_net_seconds"], 90)

    def test_no_log_does_not_prove_zero(self):
        row = attach_net_times(wall_row(), {}, {}, {"events": [], "complete_read": False}, "example")
        self.assertIsNone(row["managed_cooldown_seconds"])
        self.assertIsNone(row["process_net_seconds"])
        self.assertEqual(row["process_wall_seconds"], 100)

    def test_positive_pause_without_phase_bounds_leaves_phase_net_unknown(self):
        events = log(event("pause_start", 10), event("pause_end", 21, resume_requested_unix=20))
        row = attach_net_times(wall_row(), {}, {}, events, "example")
        self.assertEqual(row["process_net_seconds"], 90)
        self.assertIsNone(row["training_net_seconds"])
        self.assertIsNone(row["test_protocol_sum_net_seconds"])

    def test_phase_overlap_is_clipped_and_zero_overlap_is_kept(self):
        events = log(event("pause_start", 10), event("pause_end", 21, resume_requested_unix=20))
        phases = [{"phase": "training", "start_unix": 0, "end_unix": 30, "start_monotonic": 0, "end_monotonic": 30}]
        for index, protocol in enumerate(wall_row()["protocol_seconds"]):
            phases.append({"phase": "test", "protocol": protocol, "start_unix": 30 + index * 10, "end_unix": 40 + index * 10})
        row = attach_net_times(wall_row(), {"timing_intervals": phases}, {}, events, "example")
        self.assertEqual(row["training_net_seconds"], 20)
        self.assertEqual(row["test_protocol_sum_net_seconds"], 50)

    def test_watchdog_earliest_resume_prevents_double_counting(self):
        events = log(event("pause_start", 10), event("watchdog_resume", 19, resume_requested_unix=18),
                     event("pause_end", 22, resume_requested_unix=21))
        result = pause_account(events, "example")
        self.assertEqual(result["managed_pause_seconds"], 8)
        self.assertEqual(result["closed_interval_count"], 1)

    def test_overlapping_intervals_are_unioned(self):
        events = log(event("pause_start", 10), event("pause_end", 21, resume_requested_unix=20),
                     event("pause_start", 15, interval="b"), event("pause_end", 26, interval="b", resume_requested_unix=25))
        result = pause_account(events, "example")
        self.assertEqual(result["managed_pause_seconds"], 15)
        self.assertEqual(merge_intervals([(10, 20), (15, 25)]), [(10, 25)])

    def test_open_or_recycled_process_interval_is_not_zero(self):
        self.assertIsNone(pause_account(log(event("pause_start", 10)), "example")["managed_pause_seconds"])
        result = pause_account(log(event("pause_start", 10), event("pause_end", 21, pid=23, resume_requested_unix=20)), "example")
        self.assertIsNone(result["managed_pause_seconds"])

    def test_clock_jump_is_not_subtracted(self):
        events = log(event("pause_start", 10), event("pause_end", 120, event_monotonic=20, resume_requested_unix=119))
        result = pause_account(events, "example")
        self.assertIsNone(result["managed_pause_seconds"])

    def test_resume_before_suspend_is_unconfirmed(self):
        events = log(event("pause_start", 10), event("pause_end", 21, resume_requested_unix=9))
        self.assertIsNone(pause_account(events, "example")["managed_pause_seconds"])

    def test_segment_jobs_union_pauses_without_cross_boot_monotonic_comparison(self):
        events = log(event('pause_start', 10), event('pause_end', 16, resume_requested_unix=15),
                     event('pause_start', 1000, job='recovery', event_monotonic=2),
                     event('pause_end', 1006, job='recovery', event_monotonic=8, resume_requested_unix=1005))
        row = wall_row()
        row['process_wall_seconds'] = None
        row['internal_total_seconds'] = 50
        info = {'execution_segments': [{'job_name': 'example'}, {'job_name': 'recovery'}],
                'timing_intervals': [{'phase': 'total', 'start_unix': 0, 'end_unix': 30,
                                      'start_monotonic': 0, 'end_monotonic': 30},
                                     {'phase': 'total', 'start_unix': 990, 'end_unix': 1010,
                                      'start_monotonic': -8, 'end_monotonic': 12}]}
        row = attach_net_times(row, info, {}, events, 'recovery')
        self.assertEqual(row['managed_cooldown_seconds'], 10)
        self.assertEqual(row['internal_total_net_seconds'], 40)
        self.assertIsNone(row['process_net_seconds'])

    def test_derived_phase_start_does_not_allocate_positive_pause(self):
        events = log(event('pause_start', 10), event('pause_end', 21, resume_requested_unix=20))
        info = {'timing_intervals': [{'phase': 'total', 'start_unix': 0, 'end_unix': 90,
                                     'start_boundary_derived': True}]}
        row = attach_net_times(wall_row(), info, {}, events, 'example')
        self.assertIsNone(row['internal_total_net_seconds'])
        self.assertEqual(row['net_time_status']['internal_total'], 'derived_phase_boundary_cannot_allocate_positive_pause')


if __name__ == "__main__":
    unittest.main()
