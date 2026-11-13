"""Cooperative scheduled rests after completed training or inference work.

Raw timers stay unchanged. Closed sleep intervals are measured with monotonic
clocks and exposed separately for one-time subtraction. No process suspension,
GPU polling, power transition, or asynchronous model mutation is performed.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import uuid

POLICY = {"training_interval_seconds": 600, "training_rest_seconds": 120,
          "long_interval_seconds": 10800, "long_rest_seconds": 900,
          "long_clock": "wall since process start or previous long rest end; short rests included",
          "resume_clock": "restart both clocks in each invocation; preserve previous measured intervals"}
_ACTIVE = None


class CoolingPolicy:
    """Run two independent clocks and sleep only at explicit safe boundaries."""
    def __init__(self, path, synchronize, *, clock=time.monotonic, wall=time.time,
                 sleep=time.sleep, announce=print):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.synchronize, self.clock, self.wall = synchronize, clock, wall
        self.sleep, self.announce = sleep, announce
        self.session = uuid.uuid4().hex
        self.started = clock()
        self.next_long = self.started + POLICY['long_interval_seconds']
        self.training_active = 0.0
        self.previous_training = None
        self.context = {'phase': 'preparation', 'protocol': None, 'epoch': None}
        self.rest_seconds = 0.0
        self.intervals = []
        self._write('guard_start', no_managed_pauses_before_guard_start=True,
                    method='cooperative_sleep', policy=POLICY)

    def _write(self, event, **fields):
        row = {'event': event, 'event_unix': self.wall(), 'event_monotonic': self.clock(),
               'time_utc': datetime.now(timezone.utc).isoformat(), 'session_id': self.session,
               'pid': os.getpid(), 'create_time': None, 'job': self.path.parent.name,
               'method': 'cooperative_sleep', **fields}
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        return row

    def set_phase(self, phase, *, protocol=None, epoch=None):
        self.context = {'phase': phase, 'protocol': protocol, 'epoch': epoch}
        self.previous_training = self.clock() if phase == 'training' else None

    def boundary(self, *, training=False):
        now = self.clock()
        if training:
            if self.context['phase'] != 'training' or self.previous_training is None:
                raise ValueError('A training boundary requires begin_training/phase setup')
            self.training_active += max(0.0, now - self.previous_training)
            if self.training_active >= POLICY['training_interval_seconds']:
                self._pause('short', POLICY['training_rest_seconds'])
                self.training_active = 0.0
        if self.clock() >= self.next_long:
            self._pause('long', POLICY['long_rest_seconds'])
            self.next_long = self.clock() + POLICY['long_interval_seconds']
        if training:
            self.previous_training = self.clock()

    def _pause(self, kind, seconds):
        # Synchronize only when a break is due. An invalid CUDA context must fail
        # normally rather than being hidden; committed checkpoints stay intact.
        self.synchronize()
        try:
            from experiment_progress import clear_console_progress
            clear_console_progress()
        except ImportError:
            pass
        interval = uuid.uuid4().hex
        start = self._write('pause_start', interval_id=interval, kind=kind,
                            configured_seconds=seconds, **self.context)
        self.announce('[Cooling] %s rest: %d seconds; phase=%s%s' %
                      (kind, seconds, self.context['phase'],
                       ' / ' + self.context['protocol'] if self.context['protocol'] else ''), flush=True)
        try:
            self.sleep(seconds)
        finally:
            ended = self.clock()
            duration = max(0.0, ended - start['event_monotonic'])
            end_unix = self.wall()
            self._write('pause_end', interval_id=interval, kind=kind,
                        configured_seconds=seconds, resume_requested_unix=end_unix,
                        start_unix=start['event_unix'], end_unix=end_unix,
                        duration_seconds=duration, **self.context)
            self.rest_seconds += duration
            self.intervals.append({'interval_id': interval, 'kind': kind,
                'start_unix': start['event_unix'], 'end_unix': end_unix,
                'seconds': duration, **self.context})
            self.announce('[Cooling] resumed; measured rest %.1f seconds' % duration, flush=True)

    def summary(self):
        return {'enabled': True, 'policy': dict(POLICY), 'ledger': str(self.path),
                'session_id': self.session, 'session_rest_seconds': self.rest_seconds,
                'session_intervals': list(self.intervals),
                'timing_rule': 'Raw seconds preserved; subtract the measured sleep exactly once.'}


def install(path, synchronize):
    """Activate the policy for this runner; evaluators otherwise remain no-ops."""
    global _ACTIVE
    _ACTIVE = CoolingPolicy(path, synchronize)
    return _ACTIVE


def set_phase(phase, *, protocol=None, epoch=None):
    if _ACTIVE is not None:
        _ACTIVE.set_phase(phase, protocol=protocol, epoch=epoch)


def safe_boundary():
    """Long rest only, after a finished inference batch or CPU metric row."""
    if _ACTIVE is not None:
        _ACTIVE.boundary()


def training_boundary():
    """Count completed training work and apply short/long rests if due."""
    if _ACTIVE is not None:
        _ACTIVE.boundary(training=True)


def marker():
    return _ACTIVE.rest_seconds if _ACTIVE is not None else 0.0


def duration_fields(raw_seconds, started_at):
    rest = marker() - started_at
    if rest < -1e-6 or rest > raw_seconds + 0.1:
        raise ValueError('Measured cooling interval is outside its raw timer')
    return {'cooling_rest_seconds': max(0.0, rest),
            'net_elapsed_seconds': max(0.0, raw_seconds - rest)}


def record_manifest(manifest, segment):
    """Keep per-invocation sleep evidence separate from original raw seconds."""
    rest = marker()
    segment['cooling_rest_seconds'] = rest
    segment['net_elapsed_seconds'] = max(0.0, segment['elapsed_seconds'] - rest)
    manifest['elapsed_net_seconds'] = sum(row.get('net_elapsed_seconds', row['elapsed_seconds'])
                                          for row in manifest['execution_segments'])
    if 'timings_seconds_net' in manifest:
        manifest['timings_seconds_net']['total'] = manifest['elapsed_net_seconds']
        manifest['timings_seconds_net']['cooling_rest'] = sum(
            row.get('cooling_rest_seconds', 0.0) for row in manifest['execution_segments'])
    if _ACTIVE is not None:
        manifest['cooling'] = _ACTIVE.summary()


def finalize_timing(manifest):
    """Calculate user-facing net timers without mutating raw timers or results."""
    records = manifest['epoch_records']
    manifest['timings_seconds_net'] = {
        'training': sum(row.get('train_net_seconds', row['train_seconds']) for row in records),
        'dev_evaluation': sum(row['dev'].get('net_elapsed_seconds', row['dev']['elapsed_seconds']) for row in records),
        'tests': {name: row.get('net_elapsed_seconds', row['elapsed_seconds'])
                  for name, row in manifest['evaluations'].items()},
        'total': manifest['elapsed_net_seconds'],
        'cooling_rest': sum(row.get('cooling_rest_seconds', 0.0)
                            for row in manifest['execution_segments'])}
