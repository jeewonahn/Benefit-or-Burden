"""CPU-only recovery checks: provenance, immutable outputs and timing bounds."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import resume_small_evaluation as recovery


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.original = self.root / 'original'
        self.original.mkdir()
        self.data = self.root / 'data'
        self.data.mkdir()
        source_hashes = {}
        for relative in recovery.REQUIRED_RUNTIME:
            path = self.original / 'source_snapshot' / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# frozen source ' + relative, encoding='utf-8')
            source_hashes[relative] = recovery.fingerprint(path)['sha256']
        checkpoint = self.original / 'best_model.pt'
        checkpoint.write_bytes(b'checkpoint')
        trained = self.original / 'trained_epoch1.pt'
        trained.write_bytes(checkpoint.read_bytes())
        prep = self.original / 'vocabulary.json'
        prep.write_bytes(b'{}')
        outputs = {}
        for filename in recovery.PROTOCOL_FILES.values():
            path = self.data / 'test' / filename
            path.parent.mkdir(exist_ok=True)
            path.write_text(filename, encoding='utf-8')
            outputs['test/' + filename] = recovery.fingerprint(path)
        write_json(self.data / 'manifest.json', {'status': 'complete', 'output_files': outputs})
        config = dict.fromkeys(('dataset', 'tokenizer', 'word_threshold', 'word_embedding_dim',
            'max_title_length', 'max_abstract_length', 'max_history_num',
            'no_self_connection', 'no_adjacent_normalization', 'gcn_normalization_type'), 1)
        inputs = {'files': [outputs['test/behaviors.tsv']], 'glove': {'sha256': 'glove'}}
        prep_spec = {'input_hashes': [x['sha256'] for x in inputs['files']], 'glove': 'glove',
                     'seed': 42, 'parameters': config,
                     'corpus_source_sha256': source_hashes['NNR-main/MIND_corpus.py']}
        signature = hashlib.sha256(json.dumps(prep_spec, sort_keys=True).encode()).hexdigest()
        evaluations = {}
        for kind in ('imp', 'rs30', 'rs90', 'rs150'):
            prefix = self.original / 'evaluation' / ('test_' + kind)
            prefix.parent.mkdir(exist_ok=True)
            files = {suffix: str(prefix) + '.' + suffix for suffix in recovery.ARTIFACT_SUFFIXES}
            report = {'input': outputs['test/' + recovery.PROTOCOL_FILES[kind]], 'files': files,
                      'elapsed_seconds': 2.0, 'metrics': {'AUC': .5}}
            for suffix, path in files.items():
                Path(path).write_text('artifact ' + suffix, encoding='utf-8')
            write_json(Path(files['metrics.json']), report)
            evaluations[kind] = dict(report, prefix=str(prefix))
        self.manifest = {'status': 'evaluating', 'model': 'CNE-SUE', 'train_type': 'imp', 'seed': 42,
            'source_sha256': source_hashes, 'checkpoint': recovery.fingerprint(checkpoint),
            'trained_epochs': [{'epoch': 1, 'checkpoint': recovery.fingerprint(trained), 'train_seconds': 50}],
            'epoch_records': [{'epoch': 1, 'train_seconds': 50, 'dev': {'elapsed_seconds': 3}}],
            'preprocess_files': {'vocabulary.json': recovery.fingerprint(prep)},
            'effective_config': config, 'preprocess_signature': signature, 'inputs': inputs,
            'data_root': str(self.data), 'evaluations': evaluations, 'best_epoch': 1,
            'process_started_unix': 900.0, 'elapsed_seconds': 100.0,
            'timing_intervals': [{'phase': 'training', 'start_unix': 902., 'end_unix': 952.,
                                  'start_monotonic': 2000., 'end_monotonic': 2050.},
                                 {'phase': 'test', 'protocol': 'rs150', 'start_unix': 1000., 'end_unix': 1002.,
                                  'start_monotonic': 2098., 'end_monotonic': 2100.}]}
        write_json(self.original / 'cache_equivalence_epoch1.json', {'status': 'passed'})
        self.save_manifest()

    def save_manifest(self):
        write_json(self.original / 'experiment.json', self.manifest)
        write_json(self.original / 'INTERRUPTED_FOR_REBOOT.json', {
            'original_manifest_sha256': recovery.fingerprint(self.original / 'experiment.json')['sha256'],
            'status_before_pause': 'evaluating', 'completed_protocols': list(self.manifest['evaluations'])})

    def validate(self):
        return recovery.validate_original(self.original)

    def test_validated_copy_keeps_original_and_evaluation_origins(self):
        old, evidence = self.validate()
        before = {str(p.relative_to(self.original)): recovery.fingerprint(p) for p in self.original.rglob('*') if p.is_file()}
        output = self.root / 'recovered'
        copies = recovery.prepare_new_run(self.original, output, old, evidence)
        result = recovery.combined_manifest(self.original, output, old, evidence, copies, 99999)
        after = {str(p.relative_to(self.original)): recovery.fingerprint(p) for p in self.original.rglob('*') if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(old['evaluations'], result['evaluations'])
        self.assertEqual(old['source_sha256'], result['source_sha256'])
        self.assertFalse(result['recovery']['training_performed'])
        self.assertEqual(result['timing_intervals'][0]['job_name'], 'original')
        self.assertEqual(old['epoch_records'], result['epoch_records'])
        self.assertEqual(copies['checkpoint']['sha256'], evidence['checkpoint']['sha256'])

    def test_edited_manifest_is_rejected_before_runtime_import(self):
        self.manifest['seed'] = 99
        write_json(self.original / 'experiment.json', self.manifest)
        with self.assertRaisesRegex(ValueError, 'hash/size mismatch'):
            self.validate()

    def test_frozen_source_tampering_is_rejected(self):
        (self.original / 'source_snapshot' / recovery.REQUIRED_RUNTIME[0]).write_text('changed')
        with self.assertRaisesRegex(ValueError, 'hash/size mismatch'):
            self.validate()

    def test_preprocessing_and_dataset_tampering_is_rejected(self):
        for path in (self.original / 'vocabulary.json', self.data / 'test/behaviors_total5.tsv'):
            original_bytes = path.read_bytes()
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'hash/size mismatch'):
                self.validate()
            path.write_bytes(original_bytes)

    def test_completed_metric_mismatch_or_missing_scores_is_rejected(self):
        report_path = Path(self.manifest['evaluations']['imp']['files']['metrics.json'])
        payload = report_path.read_bytes()
        report = json.loads(payload)
        report['metrics']['AUC'] = .7
        write_json(report_path, report)
        with self.assertRaisesRegex(ValueError, 'differs from original manifest'):
            self.validate()
        report_path.write_bytes(payload)
        Path(self.manifest['evaluations']['imp']['files']['scores.npy']).unlink()
        with self.assertRaises(FileNotFoundError):
            self.validate()

    def test_only_missing_protocol_can_be_requested(self):
        with self.assertRaisesRegex(ValueError, 'exactly the missing'):
            recovery.validate_original(self.original, ('imp',))
        with self.assertRaisesRegex(ValueError, 'Unique supported'):
            recovery.validate_original(self.original, ('total5', 'total5'))

    def test_scope_prevents_original_nested_and_existing_output_writes(self):
        old, evidence = self.validate()
        for output in (self.original, self.original / 'nested', self.root):
            with self.assertRaisesRegex(ValueError, 'new, separate'):
                recovery.prepare_new_run(self.original, output, old, evidence)
        with self.assertRaisesRegex(ValueError, 'outside'):
            recovery.within(self.original, '../escape')

    def test_timing_excludes_reboot_gap_and_keeps_original_clock_domain(self):
        segment = recovery.original_segment(self.original, self.manifest)
        self.assertEqual(segment['duration_seconds'], 100.)
        self.assertEqual(segment['end_unix'], 1002.)
        self.assertEqual(segment['start_unix'], 902.)
        self.assertTrue(segment['start_boundary_derived'])
        self.assertNotIn('start_monotonic', segment)
        # A new process at 99999 must not inflate the earlier measured segment.
        old, evidence = self.validate()
        copies = {'checkpoint': evidence['checkpoint'], 'preprocess_files': evidence['preprocess_files']}
        result = recovery.combined_manifest(self.original, self.root / 'new', old, evidence, copies, 99999.)
        self.assertEqual(result['execution_segments'][0]['duration_seconds'], 100.)
        self.assertEqual(result['timing_intervals'][0]['start_monotonic'], 2000.)
        self.assertFalse(result['recovery']['reboot_gap_included'])

    def test_preloaded_runtime_module_is_rejected(self):
        with patch.dict(sys.modules, {'config': object()}):
            with self.assertRaisesRegex(ValueError, 'already imported'):
                recovery.activate_frozen_runtime(self.original / 'source_snapshot', self.manifest['source_sha256'])


if __name__ == '__main__':
    unittest.main()
