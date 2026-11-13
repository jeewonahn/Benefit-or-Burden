"""CPU-only fake-clock checks for safe, measured cooperative cooling."""
import ast
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import scheduled_cooling as cooling


class Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []
        self.syncs = []
        self.messages = []
    def clock(self): return self.now
    def wall(self): return 100000.0 + self.now
    def advance(self, seconds): self.now += seconds
    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.advance(seconds)
    def sync(self): self.syncs.append(self.now)
    def announce(self, *args, **kwargs): self.messages.append(args[0])


class CoolingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.clock = Clock()
        self.path = Path(self.temp.name)/'job'/'cooling_events.jsonl'
        self.policy = cooling.CoolingPolicy(self.path, self.clock.sync,
            clock=self.clock.clock, wall=self.clock.wall, sleep=self.clock.sleep,
            announce=self.clock.announce)
        self.old = cooling._ACTIVE
        cooling._ACTIVE = self.policy
        self.addCleanup(setattr, cooling, '_ACTIVE', self.old)
    def events(self): return [json.loads(line) for line in self.path.read_text().splitlines()]

    def test_short_rest_only_after_completed_boundary(self):
        cooling.set_phase('training', epoch=1)
        self.clock.advance(599)
        cooling.training_boundary()
        self.assertEqual(self.clock.sleeps, [])
        self.clock.advance(1)
        cooling.training_boundary()
        self.assertEqual(self.clock.sleeps, [120])
        self.assertEqual(self.clock.syncs, [600])
        self.assertEqual(cooling.marker(), 120)
        self.assertEqual(self.events()[1]['phase'], 'training')

    def test_training_clock_carries_epochs_but_not_dev(self):
        cooling.set_phase('training', epoch=1)
        self.clock.advance(300); cooling.training_boundary()
        cooling.set_phase('dev', epoch=1)
        self.clock.advance(3600); cooling.safe_boundary()
        cooling.set_phase('training', epoch=2)
        self.clock.advance(300); cooling.training_boundary()
        self.assertEqual(self.clock.sleeps, [120])

    def test_long_rest_during_test_after_safe_boundary(self):
        cooling.set_phase('test', protocol='total')
        self.clock.advance(10800)
        self.assertEqual(self.clock.sleeps, [])
        cooling.safe_boundary()
        self.assertEqual(self.clock.sleeps, [900])
        self.assertEqual(self.policy.next_long, 22500)
        ended = self.events()[-1]
        self.assertEqual((ended['phase'], ended['protocol']), ('test','total'))

    def test_short_and_long_are_independent_adjacent_intervals(self):
        self.clock.advance(10200)
        cooling.set_phase('training', epoch=8)
        self.clock.advance(600)
        cooling.training_boundary()
        self.assertEqual(self.clock.sleeps, [120,900])
        intervals = self.policy.summary()['session_intervals']
        self.assertEqual(intervals[0]['end_unix'], intervals[1]['start_unix'])
        self.assertNotEqual(intervals[0]['interval_id'], intervals[1]['interval_id'])
        self.assertEqual(self.policy.next_long, 22620)

    def test_short_rest_counts_toward_long_wall_clock(self):
        self.clock.advance(10100)
        cooling.set_phase('training', epoch=1)
        self.clock.advance(600); cooling.training_boundary()
        self.assertEqual(self.clock.sleeps, [120,900])

    def test_oversized_training_entry_gets_one_short_break(self):
        cooling.set_phase('training', epoch=1)
        self.clock.advance(1900); cooling.training_boundary()
        self.assertEqual(self.clock.sleeps, [120])
        cooling.training_boundary()
        self.assertEqual(self.clock.sleeps, [120])

    def test_sync_failure_is_not_hidden_or_logged_as_a_rest(self):
        def broken(): raise RuntimeError('CUDA context failed')
        self.policy.synchronize = broken
        cooling.set_phase('training', epoch=1)
        self.clock.advance(600)
        with self.assertRaisesRegex(RuntimeError,'CUDA context failed'):
            cooling.training_boundary()
        self.assertEqual(self.clock.sleeps, [])
        self.assertEqual(len(self.events()), 1)

    def test_interrupted_sleep_records_only_measured_duration(self):
        def interrupted(seconds):
            self.clock.advance(17)
            raise KeyboardInterrupt()
        self.policy.sleep = interrupted
        cooling.set_phase('training',epoch=1)
        self.clock.advance(600)
        with self.assertRaises(KeyboardInterrupt): cooling.training_boundary()
        self.assertEqual(cooling.marker(),17)
        self.assertEqual(self.events()[-1]['duration_seconds'],17)

    def test_duration_net_is_separate_and_does_not_double_subtract(self):
        cooling.set_phase('training',epoch=1)
        mark = cooling.marker()
        self.clock.advance(600); cooling.training_boundary()
        result = cooling.duration_fields(720,mark)
        self.assertEqual(result, {'cooling_rest_seconds':120,'net_elapsed_seconds':600})
        self.assertEqual(cooling.duration_fields(720,mark), result)
        with self.assertRaises(ValueError): cooling.duration_fields(10,mark)

    def test_phase_summary_preserves_raw_and_overall_timer(self):
        cooling.set_phase('training',epoch=1)
        self.clock.advance(600); cooling.training_boundary()
        segment={'elapsed_seconds':720}
        raw={'epoch_records':[{'train_seconds':720,'train_net_seconds':600,
              'dev':{'elapsed_seconds':100,'net_elapsed_seconds':100}}],
             'evaluations':{'total':{'elapsed_seconds':2000,'net_elapsed_seconds':1100}},
             'execution_segments':[{'elapsed_seconds':1000,'net_elapsed_seconds':900},segment]}
        cooling.record_manifest(raw,segment)
        cooling.finalize_timing(raw)
        self.assertEqual(raw['epoch_records'][0]['train_seconds'],720)
        self.assertEqual(raw['timings_seconds_net']['training'],600)
        self.assertEqual(raw['timings_seconds_net']['total'],1500)
        segment['elapsed_seconds']=730
        cooling.record_manifest(raw,segment)
        self.assertEqual(raw['timings_seconds_net']['total'],1510)
        self.assertEqual(raw['elapsed_net_seconds'],1510)

    def test_ledger_closed_intervals_work_with_existing_passive_timing_audit(self):
        from thermal_timing import read_thermal_log, pause_account
        cooling.set_phase('training',epoch=1)
        self.clock.advance(600); cooling.training_boundary()
        pause=pause_account(read_thermal_log(self.path),'job')
        self.assertEqual(pause['status'],'known')
        self.assertEqual(pause['managed_pause_seconds'],120)

    def test_policy_does_not_touch_random_state(self):
        previous=random.getstate()
        cooling.set_phase('training',epoch=1)
        self.clock.advance(600); cooling.training_boundary()
        self.assertEqual(random.getstate(),previous)

    def test_disabled_evaluator_hooks_are_noops(self):
        cooling._ACTIVE=None
        cooling.safe_boundary(); cooling.training_boundary()
        self.assertEqual(cooling.marker(),0)
        self.assertEqual(cooling.duration_fields(10,0)['net_elapsed_seconds'],10)

    def test_runner_boundaries_follow_optimizer_updates(self):
        for filename in ('run_converged_nnr.py','run_converged_digat.py'):
            text=(ROOT/'scripts'/filename).read_text(encoding='utf-8-sig')
            tree=ast.parse(text)
            loops=[node for node in ast.walk(tree) if isinstance(node,ast.For)
                   and any(isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute)
                           and isinstance(call.func.value,ast.Name) and call.func.value.id=='optimizer'
                           and call.func.attr=='step' for call in ast.walk(node))]
            self.assertEqual(len(loops),1)
            calls=[node for node in ast.walk(loops[0]) if isinstance(node,ast.Call)]
            step=next(c for c in calls if isinstance(c.func,ast.Attribute) and c.func.attr=='step')
            boundary=next(c for c in calls if isinstance(c.func,ast.Attribute) and c.func.attr=='training_boundary')
            self.assertGreater(boundary.lineno,step.lineno)
            self.assertIn("--cooling",text)

    def test_evaluator_hook_follows_cpu_score_copy(self):
        for filename in ('naml_cached_eval.py','nnr_cached_eval.py','digat_cached_eval.py'):
            text=(ROOT/'scripts'/filename).read_text(encoding='utf-8-sig')
            self.assertIn('cooling.safe_boundary()',text)
            self.assertIn('report.update(cooling.duration_fields(',text)
            self.assertIn('.cpu().numpy()\n',text)


if __name__=='__main__': unittest.main()
