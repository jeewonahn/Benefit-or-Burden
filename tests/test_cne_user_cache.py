"""CPU-only transactional storage and actual CNE score equivalence checks."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

STAGING = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get('TORS_CANONICAL_ROOT', STAGING))
sys.path[:0] = [str(STAGING / 'scripts'), str(ROOT / 'scripts'), str(ROOT / 'NNR-main')]
import cne_user_cache as store
from nnr_cached_eval import NNRCachedEvaluator
from nnr_batch_alignment import apply_batch_alignment_fixes
from model_fixtures import tiny_config
from model import Model
from naml_cached_eval import _file_hash


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='cne-generation-cpu-')
        self.directory = Path(self.temp.name)
        self.maps = []
        self.meta = {'identity': 'fixture', 'provenance': {'source': 'fixture'}, 'files': {}}

    def tearDown(self):
        for value in self.maps:
            store.close_mapping(value)
        self.temp.cleanup()

    @staticmethod
    def encode(keys):
        return np.asarray([np.full((3, 4), int(key), dtype=np.float32) for key in keys])

    def append(self, old, keys, pending, **kwargs):
        result, selected = store.append_generation(self.directory, old, keys, pending,
            kwargs.pop('encoder', self.encode), self.meta, batch_rows=2, copy_chunk_bytes=48,
            **kwargs)
        self.maps.append(result)
        return result, selected

    def metadata(self):
        return json.loads((self.directory / 'manifest.json').read_text(encoding='utf-8'))

    def test_append_twice_reload_keeps_indices_and_exact_bytes(self):
        empty = np.empty((0, 3, 4), dtype=np.float32)
        first, keys = self.append(empty, [], ['1', '2', '3'])
        original = self.metadata()
        old_path = self.directory / original['user_data_file']
        old_hash = store.file_hash(old_path)
        second, keys = self.append(first, keys, ['4', '5'])
        self.assertEqual(keys, ['1', '2', '3', '4', '5'])
        self.assertIsInstance(second, np.memmap)
        self.assertFalse(second.flags.writeable)
        np.testing.assert_array_equal(second, self.encode(keys))
        self.assertEqual(store.file_hash(old_path), old_hash)
        reloaded, replay_keys = store.load_generation(self.directory, self.metadata())
        self.maps.append(reloaded)
        self.assertEqual(replay_keys, keys)
        np.testing.assert_array_equal(reloaded, second)

    def test_encoder_interruption_preserves_committed_manifest_and_mapping(self):
        first, keys = self.append(np.empty((0, 3, 4), np.float32), [], ['1'])
        prior = (self.directory / 'manifest.json').read_bytes()
        calls = []
        def fail_later(batch):
            calls.append(batch)
            if len(calls) == 2:
                raise RuntimeError('interrupted encoding')
            return self.encode(batch)
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            self.append(first, keys, ['2', '3', '4'], encoder=fail_later)
        self.assertEqual((self.directory / 'manifest.json').read_bytes(), prior)
        self.assertEqual(keys, ['1'])
        np.testing.assert_array_equal(first, self.encode(['1']))
        replay, replay_keys = store.load_generation(self.directory, self.metadata())
        self.maps.append(replay)
        self.assertEqual(replay_keys, ['1'])
        partials = list(self.directory.glob('*.incomplete'))
        self.assertTrue(partials)
        # A failed writer has closed its mmap; Windows permits renaming the file.
        partials[0].rename(Path(str(partials[0]) + '.retained'))

    def test_failure_at_commit_preserves_previous_generation(self):
        first, keys = self.append(np.empty((0, 3, 4), np.float32), [], ['1'])
        prior = (self.directory / 'manifest.json').read_bytes()
        replace = store.os.replace
        def reject_manifest(source, target):
            if Path(target).name == 'manifest.json':
                raise OSError('interrupted manifest commit')
            return replace(source, target)
        with patch.object(store.os, 'replace', side_effect=reject_manifest):
            with self.assertRaisesRegex(OSError, 'manifest commit'):
                self.append(first, keys, ['2'])
        self.assertEqual((self.directory / 'manifest.json').read_bytes(), prior)
        replay, replay_keys = store.load_generation(self.directory, self.metadata())
        self.maps.append(replay)
        self.assertEqual(replay_keys, ['1'])
        self.assertGreater(len(list(self.directory.glob('users-*.npy'))), 1)

    def test_nonfinite_batch_rejected_without_changing_old_cache(self):
        first, keys = self.append(np.empty((0, 3, 4), np.float32), [], ['1'])
        prior = (self.directory / 'manifest.json').read_bytes()
        def bad(batch):
            values = self.encode(batch); values[0, 0, 0] = np.nan; return values
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            self.append(first, keys, ['2'], encoder=bad)
        self.assertEqual((self.directory / 'manifest.json').read_bytes(), prior)

    def test_tampered_array_and_key_hashes_are_rejected(self):
        first, _ = self.append(np.empty((0, 3, 4), np.float32), [], ['1'])
        store.close_mapping(first)
        meta = self.metadata()
        for name in (meta['user_data_file'], meta['user_keys_file']):
            path = self.directory / name
            before = path.read_bytes()
            path.write_bytes(before + b'tampered')
            with self.assertRaisesRegex(ValueError, 'integrity mismatch'):
                store.load_generation(self.directory, meta)
            path.write_bytes(before)

    def test_finite_checks_and_copies_are_chunk_bounded(self):
        old = np.ones((7, 3, 4), np.float32)
        real = store.np.isfinite
        checked = []
        def bounded(values):
            checked.append(values.shape[0])
            self.assertLessEqual(values.shape[0], 2)
            return real(values)
        with patch.object(store.np, 'isfinite', side_effect=bounded), \
             patch.object(store.np, 'concatenate', side_effect=AssertionError('full concatenate forbidden')):
            result, _ = self.append(old, [str(x) for x in range(7)], ['7', '8', '9'])
        self.assertTrue(checked)
        self.assertEqual(result.shape, (10, 3, 4))


class ActualCNEModelTests(unittest.TestCase):
    def test_cpu_cne_append_reload_scores_and_provenance(self):
        # Import the existing fixture without its CUDA decorators probing hardware.
        fixture_spec = importlib.util.spec_from_file_location('fixture_naml_cpu', ROOT / 'tests/test_naml_cached_eval.py')
        fixture_module = importlib.util.module_from_spec(fixture_spec)
        with patch('torch.cuda.is_available', return_value=False):
            fixture_spec.loader.exec_module(fixture_module)
        fixture = fixture_module.CachedNamlTest(); fixture.setUp()
        evaluators = []
        try:
            cfg = tiny_config('CNE-SUE'); cfg.dropout_rate = 0.0
            with patch('torch.cuda.is_available', return_value=False):
                model = Model(cfg).cpu(); model.initialize(); model.eval()
            for module in model.modules():
                if hasattr(module, 'device'):
                    module.device = torch.device('cpu')
            apply_batch_alignment_fixes(model)
            # A staging audit can additionally compare the actual frozen pre-patch
            # evaluator. Repository tests still check native forward, indices and
            # reload without depending on a local archived source path.
            old = None
            if os.environ.get('TORS_CNE_BEFORE_SOURCE'):
                before = Path(os.environ['TORS_CNE_BEFORE_SOURCE'])
                spec = importlib.util.spec_from_file_location('cne_before_memorymap', before)
                legacy_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(legacy_module)
                old = legacy_module.NNRCachedEvaluator(model, fixture.corpus, cfg, fixture.root / 'old', user_batch_size=2)
                evaluators.append(old)
            new = NNRCachedEvaluator(model, fixture.corpus, cfg, fixture.root / 'new', user_batch_size=2)
            evaluators.append(new)
            self.assertEqual(new.provenance['user_cache_implementation_sha256'], _file_hash(store.__file__))
            self.assertEqual(new.provenance['user_cache_storage'], store.STORAGE_VERSION)
            rows = fixture.behaviors.read_text(encoding='utf-8').splitlines()
            keys = []
            for line in rows:
                parts = line.split('\t'); keys.append(new._user_key(parts[1], new._history(parts[3])))
            np.testing.assert_array_equal(new._ensure_users(keys[:1]), [0])
            np.testing.assert_array_equal(new._ensure_users(keys[1:]), [1, 2, 0])
            if old is not None:
                old._ensure_users(keys[:1]); old._ensure_users(keys[1:])
                np.testing.assert_array_equal(new.users, old.users)
            before_hash = _file_hash(new.cache_manifest)
            new._save_cache()
            self.assertEqual(_file_hash(new.cache_manifest), before_hash)
            current = new.evaluate_file(fixture.behaviors, fixture.root / 'new_result')
            if old is not None:
                prior = old.evaluate_file(fixture.behaviors, fixture.root / 'old_result')
                np.testing.assert_array_equal(np.load(current['files']['scores.npy']), np.load(prior['files']['scores.npy']))
            proof = new.verify_against_forward(fixture.behaviors, max_entries=4, max_candidates=3)
            self.assertLess(proof['max_absolute_error'], 1e-6)
            replay = NNRCachedEvaluator(model, fixture.corpus, cfg, fixture.root / 'new', user_batch_size=2)
            evaluators.append(replay)
            self.assertTrue(replay.cache_loaded)
            self.assertIsInstance(replay.users, np.memmap)
            replay_result = replay.evaluate_file(fixture.behaviors, fixture.root / 'replay')
            np.testing.assert_array_equal(np.load(current['files']['scores.npy']), np.load(replay_result['files']['scores.npy']))
            meta = json.loads(new.cache_manifest.read_text(encoding='utf-8'))
            meta['provenance']['user_cache_implementation_sha256'] = 'tampered-helper-hash'
            new.cache_manifest.write_text(json.dumps(meta), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'identity/provenance mismatch'):
                NNRCachedEvaluator(model, fixture.corpus, cfg, fixture.root / 'new')
        finally:
            for evaluator in evaluators:
                store.close_mapping(evaluator.users)
            fixture.doCleanups()


if __name__ == '__main__':
    # Never initialize CUDA or run CUDA kernels in this staged suite.
    with patch('torch.cuda._lazy_init', side_effect=AssertionError('GPU initialization forbidden')):
        unittest.main(verbosity=2)
