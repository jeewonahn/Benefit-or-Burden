"""Equivalence and cache-isolation checks using the actual NAML implementation."""
from io import StringIO
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'NNR-main'))
from naml_cached_eval import NamlCachedEvaluator, metrics_from_ranks, FEATURES
from model_fixtures import tiny_config
from model import Model
from evaluate import scoring


class CachedNamlTest(unittest.TestCase):
    def setUp(self):
        self.old_cwd = Path.cwd()
        self.tmp = tempfile.TemporaryDirectory(prefix='tors-naml-cache-test-')
        self.root = Path(self.tmp.name)
        os.chdir(self.root)
        self.addCleanup(self.cleanup)
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.cfg = tiny_config('NAML')
        self.cfg.dropout_rate = 0.2
        cfg = self.cfg
        embedding_name = f'word_embedding-{cfg.word_threshold}-{cfg.word_embedding_dim}-{cfg.tokenizer}-{cfg.max_title_length}-{cfg.max_abstract_length}-{cfg.dataset}.pkl'
        with open(embedding_name, 'wb') as stream:
            pickle.dump(torch.randn(cfg.vocabulary_size, cfg.word_embedding_dim) * 0.1, stream)
        self.model = Model(cfg).cpu()
        self.model.initialize()
        self.model.eval()
        rng = np.random.RandomState(42)
        count = 8
        values = {'max_history_num': cfg.max_history_num,
                  'news_ID_dict': {'<PAD>': 0, **{'N' + str(i): i for i in range(1, count)}},
                  'user_ID_dict': {'U1': 1, 'U2': 2}}
        for part, length in [('title', cfg.max_title_length), ('abstract', cfg.max_abstract_length)]:
            values['news_' + part + '_text'] = rng.randint(1, cfg.vocabulary_size, (count, length)).astype(np.int32)
            values['news_' + part + '_text'][0] = 0
            values['news_' + part + '_mask'] = np.ones((count, length), dtype=bool)
            values['news_' + part + '_mask'][0] = False
            values['news_' + part + '_mask'][0, 0] = True
            values['news_' + part + '_entity'] = np.zeros((count, length), dtype=np.int32)
        values['news_category'] = rng.randint(0, cfg.category_num, count).astype(np.int32)
        values['news_subCategory'] = rng.randint(0, cfg.subCategory_num, count).astype(np.int32)
        values['news_category'][0] = values['news_subCategory'][0] = 0
        self.corpus = SimpleNamespace(**values)
        self.behaviors = self.root / 'behaviors.tsv'
        self.behaviors.write_text(
            '1\tU1\t11/15/2019 1:00:00 PM\tN1 N2\tN3-1 N4-0 N5-0\n'
            '2\tU1\t11/15/2019 2:00:00 PM\tN2 N3\tN1-0 N4-1 N6-0\n'
            '3\tU2\t11/15/2019 3:00:00 PM\t\tN2-0 N5-1 N6-0\n'
            '4\tU1\t11/15/2019 4:00:00 PM\tN1 N2\tN6-0 N3-1 N7-0\n', encoding='utf-8')

    def cleanup(self):
        os.chdir(self.old_cwd)
        self.tmp.cleanup()

    def original_scores(self, device):
        result = []
        corpus, cfg = self.corpus, self.cfg
        def features(indices):
            return [torch.as_tensor(getattr(corpus, name)[indices], device=device).unsqueeze(0)
                    for name in ('news_category', 'news_subCategory', 'news_title_text', 'news_title_mask',
                                 'news_title_entity', 'news_abstract_text', 'news_abstract_mask', 'news_abstract_entity')]
        with torch.no_grad():
            for line in self.behaviors.read_text(encoding='utf-8').splitlines():
                impression, user, _, raw_history, candidates = line.split('\t')
                h = [corpus.news_ID_dict[x] for x in raw_history.split()][-cfg.max_history_num:]
                mask = [True] * len(h) + [False] * (cfg.max_history_num - len(h))
                h += [0] * (cfg.max_history_num - len(h))
                news = [corpus.news_ID_dict[x.rsplit('-', 1)[0]] for x in candidates.split()]
                inputs = [torch.tensor([corpus.user_ID_dict[user]], device=device), *features(h),
                          torch.tensor([mask], device=device), torch.eye(cfg.max_history_num, device=device)[None],
                          torch.ones((1, cfg.category_num + 1), device=device, dtype=torch.bool),
                          torch.zeros((1, cfg.max_history_num), device=device, dtype=torch.long), *features(news)]
                result.extend(self.model(*inputs).detach().cpu().numpy().ravel().tolist())
        return np.asarray(result)

    def check_device(self, device):
        self.model.to(device).eval()
        baseline = self.original_scores(device)
        evaluator = NamlCachedEvaluator(self.model, self.corpus, self.root / 'cache',
                                        news_batch_size=3, user_batch_size=2, score_batch_size=5)
        verification = evaluator.verify_against_forward(self.behaviors, max_entries=4, max_candidates=3)
        self.assertEqual(verification['status'], 'passed')
        self.assertEqual(verification['candidate_scores'], 12)
        self.assertLess(verification['max_absolute_error'], 1e-5)
        report = evaluator.evaluate_file(self.behaviors, self.root / 'result')
        cached = np.load(report['files']['scores.npy'])
        np.testing.assert_allclose(cached, baseline, rtol=1e-5, atol=1e-6)
        self.assertEqual(report['cached_news_count'], 8)
        self.assertEqual(report['cached_user_history_count'], 3)
        with open(report['files']['truth.txt']) as truth, open(report['files']['predictions.txt']) as prediction:
            expected = scoring(truth, prediction)
        np.testing.assert_allclose(list(report['metrics'].values()), expected, rtol=0, atol=1e-12)
        again = NamlCachedEvaluator(self.model, self.corpus, self.root / 'cache', news_batch_size=2)
        second = again.evaluate_file(self.behaviors, self.root / 'result2')
        self.assertTrue(second['cache_loaded_from_disk'])
        self.assertEqual(second['cached_user_history_count'], 3)
        np.testing.assert_array_equal(np.load(second['files']['scores.npy']), cached)
        with self.assertRaisesRegex(ValueError, 'already has artifacts'):
            again.evaluate_file(self.behaviors, self.root / 'result2')

    def test_cpu_matches_actual_forward_with_empty_history_and_multiple_contexts(self):
        self.check_device(torch.device('cpu'))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_matches_actual_forward(self):
        self.check_device(torch.device('cuda'))

    def test_state_change_invalidates_instance_and_creates_new_cache(self):
        first = NamlCachedEvaluator(self.model, self.corpus, self.root / 'cache')
        with torch.no_grad():
            self.model.news_encoder.category_affine.bias.add_(0.2)
        with self.assertRaisesRegex(ValueError, 'Model changed'):
            first.evaluate_file(self.behaviors, self.root / 'changed')
        second = NamlCachedEvaluator(self.model, self.corpus, self.root / 'cache')
        self.assertNotEqual(first.identity, second.identity)

    def test_test_progress_uses_protocol_rows_across_a_resumed_chunk(self):
        from unittest.mock import patch
        from evaluation_progress import evaluate_test_file
        ev = NamlCachedEvaluator(self.model, self.corpus, self.root / 'cache', score_batch_size=4)
        expected = ev.evaluate_file(self.behaviors, self.root / 'plain')
        with patch('evaluation_progress.test_progress') as display:
            actual = evaluate_test_file(ev, self.behaviors, self.root / 'with_progress',
                                        'total', start_row=6, total_rows=10)
        self.assertEqual(expected['metrics'], actual['metrics'])
        np.testing.assert_array_equal(np.load(expected['files']['scores.npy']),
                                      np.load(actual['files']['scores.npy']))
        calls = display.call_args_list
        ranked = [call.args[1] for call in calls]
        self.assertEqual(set(ranked), set(range(6, 11)))
        self.assertEqual(ranked, sorted(ranked))
        self.assertTrue(all(call.args[0] == 'total' and call.args[2] == 10 for call in calls))
        scored = [(call.kwargs['scored'], call.kwargs['score_total']) for call in calls
                  if call.kwargs.get('scored') is not None]
        self.assertEqual(scored, [(4, 12), (8, 12), (12, 12)])
        self.assertEqual(ranked[-1], 10)

    def test_frozen_evaluator_keeps_its_two_argument_signature(self):
        from unittest.mock import patch
        from evaluation_progress import evaluate_test_file
        class OldEvaluator:
            def evaluate_file(self, path, prefix):
                return {'rows': 4}
        with patch('evaluation_progress.test_progress') as display:
            evaluate_test_file(OldEvaluator(), self.behaviors, self.root / 'old',
                               'rs30', start_row=4, total_rows=8)
        self.assertEqual([call.args for call in display.call_args_list], [('rs30', 4, 8), ('rs30', 8, 8)])

    def test_mapping_change_invalidates_cache(self):
        first = NamlCachedEvaluator(self.model, self.corpus, self.root / 'cache')
        self.corpus.news_ID_dict['N1'], self.corpus.news_ID_dict['N2'] = 2, 1
        with self.assertRaisesRegex(ValueError, 'Corpus changed'):
            first.evaluate_file(self.behaviors, self.root / 'changed_mapping')
        second = NamlCachedEvaluator(self.model, self.corpus, self.root / 'cache')
        self.assertNotEqual(first.identity, second.identity)

    def test_corrupt_cache_rejected(self):
        first = NamlCachedEvaluator(self.model, self.corpus, self.root / 'cache')
        (first.cache_dir / 'news_embeddings.npy').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'integrity mismatch'):
            NamlCachedEvaluator(self.model, self.corpus, self.root / 'cache')

    def test_unknown_news_rejected(self):
        evaluator = NamlCachedEvaluator(self.model, self.corpus, self.root / 'cache')
        bad = self.root / 'bad.tsv'
        bad.write_text('1\tU1\tdate\tN1\tN99-1 N2-0\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Candidate news missing'):
            evaluator.evaluate_file(bad, self.root / 'unknown')

    def test_candidate_conditioned_or_non_dot_model_rejected(self):
        self.model.click_predictor = 'mlp'
        with self.assertRaisesRegex(ValueError, 'only NAML'):
            NamlCachedEvaluator(self.model, self.corpus, self.root / 'cache')

    def test_rank_metrics_match_official_multi_positive_and_ties(self):
        for labels, scores in [([1, 0, 1, 0, 0], [0.5, 0.5, 0.7, 0.3, 0.9]),
                               ([0, 1, 0], [0.2, 0.2, 0.2])]:
            ranks = np.empty(len(scores), dtype=int)
            ranks[np.argsort(-np.asarray(scores), kind='stable')] = np.arange(1, len(scores) + 1)
            expected = scoring(StringIO('1 ' + json.dumps(labels, separators=(',', ':')) + '\n'),
                               StringIO('1 ' + json.dumps(ranks.tolist(), separators=(',', ':')) + '\n'))
            np.testing.assert_allclose(metrics_from_ranks(labels, ranks), expected, rtol=0, atol=1e-12)


if __name__ == '__main__':
    unittest.main(verbosity=2)
