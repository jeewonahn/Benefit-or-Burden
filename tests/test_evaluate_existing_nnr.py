"""Protect full-Total/checkpoint input boundaries without model execution."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('nnr_existing_guard',
    Path(__file__).resolve().parents[1] / 'scripts/evaluate_existing_nnr.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ExistingEvaluationGuards(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.dataset = self.root / 'dataset'
        self.dataset.mkdir()
        self.dataset_manifest = self.dataset / 'manifest.json'
        self.dataset_manifest.write_text('{}', encoding='utf-8')
        self.behaviors = self.root / 'behaviors_total.tsv'
        self.behaviors.write_text('1\tu\tt\t\tN1-1 N2-0 N3-0 N4-0\n2\tu\tt\t\tN5-1 N2-0 N3-0 N4-0\n', encoding='utf-8')
        self.spec = {'status': 'complete', 'protocol': 'total', 'dataset_root': str(self.dataset),
            'dataset_manifest': MODULE.fingerprint(self.dataset_manifest),
            'output': MODULE.fingerprint(self.behaviors),
            'statistics': {'rows': 2, 'candidates': 8, 'positive_labels': 2},
            'validation': {key: True for key in MODULE.REQUIRED_CHECKS},
            'source_existing_evaluations': {'rs30': {'sha256': 'r'}, 'total5': {'sha256': 't'}}}
        self.old = {'data_root': str(self.dataset), 'evaluations': {
            'imp': {'rows': 1}, 'rs30': {'rows': 2, 'input': {'sha256': 'r'}},
            'total5': {'rows': 2, 'candidates': 4, 'input': {'sha256': 't'}}}}
        self.manifest = self.root / 'candidate.json'

    def validate(self, spec=None):
        self.manifest.write_text(json.dumps(self.spec if spec is None else spec), encoding='utf-8')
        return MODULE.validate_candidates(self.behaviors, self.manifest, self.old)

    def test_positive_expanded_requests_need_not_equal_imp_rows(self):
        self.assertEqual(self.validate()['statistics']['rows'], 2)

    def test_refuses_total5_label_or_incomplete_alignment(self):
        bad = copy.deepcopy(self.spec)
        bad['protocol'] = 'total5'
        with self.assertRaisesRegex(ValueError, 'full Total'):
            self.validate(bad)
        bad = copy.deepcopy(self.spec)
        bad['validation']['rands30_subset_of_total'] = False
        with self.assertRaisesRegex(ValueError, 'checks'):
            self.validate(bad)

    def test_refuses_mutated_candidate_bytes(self):
        self.behaviors.write_text('changed', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'hash/size'):
            self.validate()

    def test_refuses_wrong_previous_candidate_identity(self):
        self.old['evaluations']['rs30']['input']['sha256'] = 'different'
        with self.assertRaisesRegex(ValueError, 'original rs30'):
            self.validate()

    def test_refuses_wrong_dataset_or_unexpanded_candidates(self):
        bad = copy.deepcopy(self.spec)
        bad['dataset_root'] = str(self.root / 'different')
        with self.assertRaisesRegex(ValueError, 'different dataset'):
            self.validate(bad)
        bad = copy.deepcopy(self.spec)
        bad['statistics']['candidates'] = 4
        with self.assertRaisesRegex(ValueError, 'did not expand'):
            self.validate(bad)

    def test_never_prepares_over_original_or_existing_output(self):
        with self.assertRaisesRegex(ValueError, 'new separate'):
            MODULE.prepare_run(self.dataset, self.dataset, {})
        with self.assertRaisesRegex(ValueError, 'new separate'):
            MODULE.prepare_run(self.dataset, self.root, {})


if __name__ == '__main__':
    unittest.main()
