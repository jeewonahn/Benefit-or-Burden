"""Synthetic CPU-only contracts for separately audited net timing."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from summarize_converged_experiments import PROTOCOLS, validated_net_timing


class NetTimingTests(unittest.TestCase):
    def fixture(self, directory, rests=True):
        train_rest, dev_rest = (2.0, 3.0) if rests else (0.0, 0.0)
        reports = {p: dict(elapsed_seconds=10.0, cooling_rest_seconds=0.0,
                           net_elapsed_seconds=10.0) for p in PROTOCOLS}
        ledger = Path(directory) / 'cooling_events.jsonl'
        rows = [dict(event='guard_start', session_id='session', method='cooperative_sleep',
                     no_managed_pauses_before_guard_start=True)]
        for key, phase, start, duration in (('a', 'training', 10.0, train_rest), ('b', 'dev', 20.0, dev_rest)):
            if not duration:
                continue
            shared = dict(session_id='session', method='cooperative_sleep', interval_id=key,
                          phase=phase, epoch=1, protocol=None, kind='short')
            rows += [dict(shared, event='pause_start', event_unix=start),
                     dict(shared, event='pause_end', event_unix=start+duration,
                          start_unix=start, end_unix=start+duration, duration_seconds=duration)]
        ledger.write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf-8')
        rest = train_rest+dev_rest
        return dict(epoch_records=[dict(epoch=1, train_seconds=10.0,
            train_cooling_rest_seconds=train_rest, train_net_seconds=10.0-train_rest,
            dev=dict(elapsed_seconds=10.0, cooling_rest_seconds=dev_rest,
                     net_elapsed_seconds=10.0-dev_rest))], evaluations=reports,
            execution_segments=[dict(elapsed_seconds=100.0, cooling_rest_seconds=rest,
                                     net_elapsed_seconds=100.0-rest)],
            elapsed_seconds=100.0, elapsed_net_seconds=100.0-rest,
            timings_seconds_net=dict(training=10.0-train_rest, dev_evaluation=10.0-dev_rest,
                tests={p:10.0 for p in PROTOCOLS}, total=100.0-rest, cooling_rest=rest),
            cooling=dict(enabled=True, ledger=str(ledger)))

    def test_new_explicit_zero_rest_is_verified(self):
        with tempfile.TemporaryDirectory() as folder:
            result=validated_net_timing(self.fixture(folder, False))
            self.assertEqual(result['status'], 'verified_measured_net_time')
            self.assertEqual(result['active_runner_seconds'],100.0)
            self.assertEqual(result['all_recorded_rest_seconds'],0.0)

    def test_measured_rest_is_subtracted_once(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest=self.fixture(folder)
            original=copy.deepcopy(manifest)
            result=validated_net_timing(manifest)
            self.assertEqual(result['training_seconds'],8.0)
            self.assertEqual(result['validation_seconds'],7.0)
            self.assertEqual(result['active_runner_seconds'],95.0)
            self.assertEqual(result['runner_only_rest_seconds'],0.0)
            self.assertEqual(manifest,original)

    def test_double_subtraction_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest=self.fixture(folder)
            manifest['epoch_records'][0]['train_net_seconds']=6.0
            with self.assertRaisesRegex(ValueError,'raw-minus-rest'):
                validated_net_timing(manifest)

    def test_old_manifest_is_unknown_not_zero_rest(self):
        result=validated_net_timing(dict(elapsed_seconds=100.0))
        self.assertEqual(result['status'],'unknown_legacy_net_time')
        self.assertIsNone(result['active_runner_seconds'])
        self.assertIsNone(result['provenance'])

    def test_wrong_phase_cannot_support_training_rest(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest=self.fixture(folder)
            path=Path(manifest['cooling']['ledger'])
            rows=[json.loads(row) for row in path.read_text().splitlines()]
            for row in rows:
                if row.get('interval_id')=='a': row['phase']='development_setup'
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            with self.assertRaisesRegex(ValueError,'matching phase'):
                validated_net_timing(manifest)

    def test_duplicate_interval_end_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest=self.fixture(folder)
            path=Path(manifest['cooling']['ledger'])
            with path.open('a') as stream: stream.write(path.read_text().splitlines()[-1]+'\n')
            with self.assertRaisesRegex(ValueError,'duplicate'):
                validated_net_timing(manifest)

    def test_merged_report_requires_actual_chunk_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest=self.fixture(folder, False)
            manifest['evaluations']['total']['chunk_count']=2
            with self.assertRaisesRegex(ValueError,'Missing chunk'):
                validated_net_timing(manifest)


if __name__ == '__main__':
    unittest.main()
