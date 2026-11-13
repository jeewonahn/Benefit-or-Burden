"""Actual-model equivalence for NRMS and LSTUR cached inference."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'NNR-main'))
from nnr_cached_eval import NNRCachedEvaluator, strict_fp32_evaluation
from model_fixtures import tiny_config
from model import Model
import test_naml_cached_eval as fixtures
from nnr_batch_alignment import apply_batch_alignment_fixes
from naml_cached_eval import FEATURES
from model_fixtures import synthetic_inputs


class GeneralCachedTest(unittest.TestCase):
    def check_model(self, name, device):
        fixture = fixtures.CachedNamlTest()
        fixture.setUp()
        try:
            fixture.cfg = tiny_config(name)
            fixture.cfg.dropout_rate = 0.2
            fixture.model = Model(fixture.cfg).to(device)
            fixture.model.initialize()
            fixture.model.eval()
            with strict_fp32_evaluation(name == 'LSTUR'):
                direct = fixture.original_scores(device)
            evaluator = NNRCachedEvaluator(fixture.model, fixture.corpus, fixture.cfg,
                                           fixture.root / 'cache', news_batch_size=3,
                                           user_batch_size=4, score_batch_size=5)
            proof = evaluator.verify_against_forward(fixture.behaviors, max_entries=4, max_candidates=3)
            self.assertEqual(proof['candidate_scores'], 12)
            self.assertLess(proof['max_absolute_error'], 1e-5)
            result = evaluator.evaluate_file(fixture.behaviors, fixture.root / 'result')
            cached = np.load(result['files']['scores.npy'])
            np.testing.assert_allclose(cached, direct, atol=1e-6, rtol=1e-5)
            self.assertEqual(result['cached_user_history_count'], 3)
            again = NNRCachedEvaluator(fixture.model, fixture.corpus, fixture.cfg, fixture.root / 'cache')
            self.assertTrue(again.cache_loaded)
            fixture.corpus.user_ID_dict['U1'] = 0
            with self.assertRaisesRegex(ValueError, 'User ID mapping changed'):
                again.evaluate_file(fixture.behaviors, fixture.root / 'bad_mapping')
        finally:
            fixture.doCleanups()

    def test_nrms_cpu(self):
        self.check_model('NRMS', torch.device('cpu'))

    def test_lstur_cpu(self):
        self.check_model('LSTUR', torch.device('cpu'))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_nrms_cuda(self):
        self.check_model('NRMS', torch.device('cuda'))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_lstur_cuda(self):
        self.check_model('LSTUR', torch.device('cuda'))

    def test_lstur_precision_state_restoration_on_success_and_exception(self):
        fixture = fixtures.CachedNamlTest()
        fixture.setUp()
        original = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        try:
            cfg = tiny_config('LSTUR'); model = Model(cfg).cpu(); model.initialize(); model.eval()
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            def assert_restored():
                self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
                self.assertTrue(torch.backends.cudnn.allow_tf32)
            evaluator = NNRCachedEvaluator(model, fixture.corpus, cfg, fixture.root / 'cache')
            assert_restored()
            precision = evaluator.provenance['evaluation_precision']
            self.assertEqual(precision['policy'], 'strict_fp32_eval_v1')
            self.assertFalse(precision['cuda_matmul_allow_tf32'])
            self.assertFalse(precision['cudnn_allow_tf32'])
            evaluator.verify_against_forward(fixture.behaviors)
            assert_restored()
            evaluator.evaluate_file(fixture.behaviors, fixture.root / 'result')
            assert_restored()
            with self.assertRaisesRegex(ValueError, 'already has artifacts'):
                evaluator.evaluate_file(fixture.behaviors, fixture.root / 'result')
            assert_restored()
            with strict_fp32_evaluation():
                replay = NNRCachedEvaluator(model, fixture.corpus, cfg, fixture.root / 'cache')
                self.assertTrue(replay.cache_loaded)
                self.assertEqual(replay.identity, evaluator.identity)
            assert_restored()
            def fail_encoding(self):
                if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
                    raise AssertionError('Strict policy not active inside constructor')
                raise RuntimeError('deliberate constructor failure')
            with patch.object(NNRCachedEvaluator, '_encode_news', fail_encoding):
                with self.assertRaisesRegex(RuntimeError, 'deliberate constructor failure'):
                    NNRCachedEvaluator(model, fixture.corpus, cfg, fixture.root / 'constructor_failure')
            assert_restored()
            with torch.no_grad():
                next(model.parameters()).add_(0.1)
            with self.assertRaisesRegex(ValueError, 'Model changed'):
                evaluator.verify_against_forward(fixture.behaviors)
            assert_restored()
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = original
            fixture.doCleanups()

    def check_fixed_model(self, name, device):
        fixture = fixtures.CachedNamlTest()
        fixture.setUp()
        try:
            cfg = tiny_config(name)
            cfg.dropout_rate = 0.0
            model = Model(cfg).to(device)
            model.initialize()
            model.eval()
            for module in model.modules():
                if hasattr(module, 'device'):
                    module.device = device
            corpus = fixture.corpus
            corpus.news_category[1:4] = [0, 1, 0]
            # Opposite title/content lengths expose CNE's cross-sort misalignment.
            corpus.news_title_mask[1, 1:] = False
            corpus.news_abstract_mask[2, 1:] = False
            corpus.news_title_mask[3, 2:] = False
            corpus.news_abstract_mask[4, 2:] = False
            if name == 'NPA':
                with torch.no_grad():
                    model.user_embedding.weight[1].fill_(1.0)
                    model.user_embedding.weight[2].fill_(-1.0)
                    model.news_encoder.dense.weight.fill_(0.5)
                    model.news_encoder.dense.bias.fill_(0.2)
            def encode(indices, user_ids):
                inputs = [torch.as_tensor(getattr(corpus, key)[indices], device=device) for key in FEATURES]
                user = model.user_embedding(torch.tensor(user_ids, device=device)) if model.use_user_embedding else None
                return model.news_encoder(*inputs, user)
            indices = np.array([[1, 2], [3, 4]])
            with torch.no_grad():
                legacy = encode(indices, [1, 2])
                singles = torch.stack([encode(np.array([[int(index)]]), [1 if i < 2 else 2])[0, 0]
                                       for i, index in enumerate(indices.flatten())]).reshape(2, 2, -1)
                legacy_error = float((legacy - singles).abs().max().item())
            self.assertGreater(legacy_error, 1e-7, 'Fixture did not expose the legacy batch defect')
            self.assertTrue(apply_batch_alignment_fixes(model))
            with torch.no_grad():
                fixed = encode(indices, [1, 2])
                fixed_singles = torch.stack([encode(np.array([[int(index)]]), [1 if i < 2 else 2])[0, 0]
                                             for i, index in enumerate(indices.flatten())]).reshape(2, 2, -1)
                permutation = encode(indices[::-1].copy(), [2, 1]).flip(0)
            # cuDNN packed LSTM kernels can differ by ~1e-5 across batch shapes
            # on this GPU. CPU checks retain the tighter numerical threshold.
            tolerance = 1e-4 if name == 'CNE-SUE' and device.type == 'cuda' else 1e-6
            torch.testing.assert_close(fixed, fixed_singles, atol=tolerance, rtol=1e-5)
            torch.testing.assert_close(fixed, permutation, atol=tolerance, rtol=1e-5)
            model.train()
            inputs = [value.to(device) for value in synthetic_inputs(torch, cfg)]
            loss = torch.nn.functional.cross_entropy(model(*inputs), torch.zeros(2, device=device, dtype=torch.long))
            loss.backward()
            gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
            self.assertTrue(gradients and all(torch.isfinite(value).all() for value in gradients))
            self.assertTrue(any((value != 0).any() for value in gradients))
            model.eval()
            evaluator = NNRCachedEvaluator(model, corpus, cfg, fixture.root / 'cache',
                                           news_batch_size=3, user_batch_size=2, score_batch_size=5)
            if name == 'CNE-SUE':
                graph, mask, categories = evaluator._history_graph([1, 2, 0])
                adjacency = np.array([[1, 0, 0, 1, 0], [0, 1, 0, 0, 1], [0, 0, 1, 0, 0],
                                      [1, 0, 0, 1, 1], [0, 1, 0, 1, 1]], dtype=np.float32)
                inverse = np.diag(np.sqrt(1 / adjacency.sum(axis=1)))
                np.testing.assert_allclose(graph, inverse @ adjacency @ inverse)
                np.testing.assert_array_equal(categories, [0, 1, 2])
            proof = evaluator.verify_against_forward(fixture.behaviors, max_entries=4, max_candidates=3)
            self.assertLess(proof['max_absolute_error'], tolerance)
            result = evaluator.evaluate_file(fixture.behaviors, fixture.root / 'result')
            self.assertEqual(result['cached_user_history_count'], 3)
            again = NNRCachedEvaluator(model, corpus, cfg, fixture.root / 'cache')
            replay = again.evaluate_file(fixture.behaviors, fixture.root / 'replay')
            np.testing.assert_array_equal(np.load(result['files']['scores.npy']), np.load(replay['files']['scores.npy']))
            print(name, str(device), 'legacy_batch_error=', legacy_error,
                  'fixed_cache_error=', proof['max_absolute_error'], flush=True)
        finally:
            fixture.doCleanups()

    def test_npa_alignment_cpu(self):
        self.check_fixed_model('NPA', torch.device('cpu'))

    def test_npa_personalized_batch_and_device_gather_cpu(self):
        fixture = fixtures.CachedNamlTest(); fixture.setUp()
        try:
            cfg = tiny_config('NPA'); model = Model(cfg).cpu(); model.initialize(); model.eval()
            apply_batch_alignment_fixes(model)
            first = NNRCachedEvaluator(model, fixture.corpus, cfg, fixture.root / 'cache',
                                       personalized_batch_size=2, npa_features_on_gpu=True)
            self.assertFalse(first.feature_residency['actual_on_gpu'])
            self.assertEqual(first.feature_residency['reason'], 'requires_NPA_on_CUDA')
            small = first.evaluate_file(fixture.behaviors, fixture.root / 'small')
            large = NNRCachedEvaluator(model, fixture.corpus, cfg, fixture.root / 'cache',
                                       personalized_batch_size=2048)
            self.assertTrue(large.cache_loaded)
            # Exercise the device-index-select feature branch on CPU without
            # claiming that this substitutes for a real CUDA memory benchmark.
            large.npa_features_device = torch.tensor(np.asarray(large.news))
            big = large.evaluate_file(fixture.behaviors, fixture.root / 'large')
            np.testing.assert_allclose(np.load(small['files']['scores.npy']),
                                       np.load(big['files']['scores.npy']), atol=1e-6, rtol=1e-5)
            proof = large.verify_against_forward(fixture.behaviors)
            self.assertLess(proof['max_absolute_error'], 1e-6)
            self.assertEqual(big['runtime_memory']['personalized_batch_size'], 2048)
        finally:
            fixture.doCleanups()

    def test_npa_memory_budget_rejects_allocation_without_gpu_execution(self):
        fixture = fixtures.CachedNamlTest(); fixture.setUp()
        try:
            cfg = tiny_config('NPA'); model = Model(cfg).cpu(); model.initialize(); model.eval()
            apply_batch_alignment_fixes(model)
            evaluator = NNRCachedEvaluator(model, fixture.corpus, cfg, fixture.root / 'cache')
            evaluator.device = torch.device('cuda')
            with patch('torch.cuda.empty_cache'), \
                 patch('torch.cuda.mem_get_info', return_value=(1024, 8 * 1024 ** 3)), \
                 patch('torch.cuda.memory_allocated', return_value=0), \
                 patch('torch.tensor') as allocate:
                evaluator._place_npa_features_on_gpu()
                allocate.assert_not_called()
            self.assertFalse(evaluator.feature_residency['actual_on_gpu'])
            self.assertEqual(evaluator.feature_residency['reason'], 'insufficient_budget_with_attention_workspace')
        finally:
            fixture.doCleanups()

    def test_cne_sue_alignment_cpu(self):
        self.check_fixed_model('CNE-SUE', torch.device('cpu'))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_npa_alignment_cuda(self):
        self.check_fixed_model('NPA', torch.device('cuda'))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cne_sue_alignment_cuda(self):
        self.check_fixed_model('CNE-SUE', torch.device('cuda'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
