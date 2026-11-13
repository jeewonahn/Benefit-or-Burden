"""Shared convergence, restart and owned-cache controls for new experiments.

No training library is imported until RNG/checkpoint functions are requested.
Patience is measured only from the fixed development metric, never test scores.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import uuid


def selection_value(metrics, criterion='avg'):
    names = ('AUC', 'MRR', 'nDCG@5', 'nDCG@10')
    if criterion not in ('avg',) + names:
        raise ValueError('Unknown development selection metric')
    values = {name: float(metrics[name]) for name in names}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError('Development metrics must be finite')
    if criterion == 'avg':
        return (values['AUC'] + values['MRR'] + (values['nDCG@5'] + values['nDCG@10']) / 2.0) / 3.0
    return values[criterion]


class EarlyStopping:
    def __init__(self, max_epochs=50, min_epochs=5, patience=5, min_delta=1e-4):
        if any(type(value) is not int or value < 1 for value in (max_epochs, min_epochs, patience)):
            raise ValueError('Epoch limits and patience must be positive integers')
        if min_epochs > max_epochs or not math.isfinite(min_delta) or min_delta < 0:
            raise ValueError('Invalid minimum epoch or improvement threshold')
        self.max_epochs, self.min_epochs = max_epochs, min_epochs
        self.patience, self.min_delta = patience, float(min_delta)
        self.last_epoch = self.best_epoch = self.bad_epochs = 0
        self.best_score = self.significance_anchor = None
        self.stop_reason = None

    def update(self, epoch, score):
        if type(epoch) is not int or epoch != self.last_epoch + 1 or epoch > self.max_epochs:
            raise ValueError('Development updates must cover each successive epoch exactly once')
        if self.stop_reason is not None:
            raise ValueError('Early-stopping decision is already final')
        score = float(score)
        if not math.isfinite(score):
            raise ValueError('Development selection metric must be finite')
        is_best = self.best_score is None or score > self.best_score
        significant = self.significance_anchor is None or score > self.significance_anchor + self.min_delta
        if is_best:
            self.best_score, self.best_epoch = score, epoch
        if significant:
            self.significance_anchor, self.bad_epochs = score, 0
        else:
            self.bad_epochs += 1
        self.last_epoch = epoch
        if epoch >= self.min_epochs and self.bad_epochs >= self.patience:
            self.stop_reason = 'patience'
        elif epoch >= self.max_epochs:
            self.stop_reason = 'max_epochs'
        return {'is_best': is_best, 'significant_improvement': significant,
                'should_stop': self.stop_reason is not None, 'stop_reason': self.stop_reason,
                'best_epoch': self.best_epoch, 'best_score': self.best_score,
                'bad_epochs': self.bad_epochs}

    def state_dict(self):
        return dict(vars(self))

    @classmethod
    def from_state_dict(cls, state):
        expected = {'max_epochs','min_epochs','patience','min_delta','last_epoch',
                    'best_epoch','bad_epochs','best_score','significance_anchor','stop_reason'}
        if set(state) != expected:
            raise ValueError('Unknown or incomplete early-stopping checkpoint state')
        result = cls(**{name: state[name] for name in ('max_epochs','min_epochs','patience','min_delta')})
        if not 0 <= state['best_epoch'] <= state['last_epoch'] <= state['max_epochs']:
            raise ValueError('Invalid checkpoint epoch ordering')
        if type(state['bad_epochs']) is not int or state['bad_epochs'] < 0:
            raise ValueError('Invalid patience counter')
        if state['stop_reason'] not in (None, 'patience', 'max_epochs'):
            raise ValueError('Invalid stop reason')
        if state['last_epoch']:
            if not all(isinstance(state[key], (int, float)) and math.isfinite(state[key])
                       for key in ('best_score','significance_anchor')):
                raise ValueError('Invalid saved development scores')
        result.__dict__.update(state)
        return result


def capture_rng_state():
    import numpy as np
    import torch
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch_cpu': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'torch_version': str(torch.__version__)}


def restore_rng_state(state):
    import numpy as np
    import torch
    if str(torch.__version__) != state['torch_version']:
        raise ValueError('Exact restart requires the same PyTorch version')
    saved_cuda = state['torch_cuda']
    if saved_cuda is not None and (not torch.cuda.is_available() or len(saved_cuda) != torch.cuda.device_count()):
        raise ValueError('CUDA RNG device inventory changed')
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'].cpu())
    if saved_cuda is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in saved_cuda])


def atomic_torch_save(payload, path):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('xb') as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def save_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare_epoch_cache(run_dir, epoch):
    root = Path(run_dir).resolve()
    if type(epoch) is not int or epoch < 0:
        raise ValueError('Nonnegative epoch required')
    parent = root / 'epoch_caches'
    parent.mkdir(exist_ok=True)
    path = parent / ('epoch_%04d_%s' % (epoch, uuid.uuid4().hex))
    path.mkdir(exist_ok=False)
    save_json_atomic(path / '.convergence_cache_owned.json',
                     {'run_dir': str(root), 'cache_path': str(path.resolve())})
    return path


def cleanup_epoch_cache(run_dir, cache_path):
    """Remove only a marker-owned generated cache below this run's epoch_caches."""
    root, path = Path(run_dir).resolve(), Path(cache_path).resolve()
    allowed = (root / 'epoch_caches').resolve()
    if allowed == path or allowed not in path.parents or root not in allowed.parents:
        raise ValueError('Cache deletion target is outside this run epoch cache area')
    marker = path / '.convergence_cache_owned.json'
    owner = json.loads(marker.read_text(encoding='utf-8'))
    if owner != {'run_dir': str(root), 'cache_path': str(path)}:
        raise ValueError('Cache ownership marker differs from resolved target')
    if any(item.is_symlink() for item in path.rglob('*')):
        raise ValueError('Refuse linked content in owned cache cleanup')
    shutil.rmtree(path)
    return {'removed_cache': str(path), 'ownership_checked': True}
