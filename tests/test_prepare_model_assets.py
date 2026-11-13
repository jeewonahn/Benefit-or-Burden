"""Synthetic asset registration tests: no network, torch or model execution."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import prepare_model_assets as assets


def semantic_fixture(root):
    root.mkdir(parents=True)
    values = {
        'config.json': {'model_type': 'mpnet', 'hidden_size': 768},
        'config_sentence_transformers.json': {'__version__': {'sentence_transformers': '2.2.2'}},
        'sentence_bert_config.json': {'max_seq_length': 384, 'do_lower_case': False},
        'modules.json': [
            {'idx': 0, 'name': '0', 'path': '', 'type': 'sentence_transformers.models.Transformer'},
            {'idx': 1, 'name': '1', 'path': '1_Pooling', 'type': 'sentence_transformers.models.Pooling'},
        ],
        '1_Pooling/config.json': {'word_embedding_dimension': 768, 'pooling_mode_mean_tokens': True},
        'tokenizer.json': {}, 'tokenizer_config.json': {}, 'special_tokens_map.json': {},
    }
    for name, value in values.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')
    (root / 'vocab.txt').write_text('<pad>\nhello\n', encoding='utf-8')
    (root / 'pytorch_model.bin').write_bytes(b'synthetic weights; never deserialized')
    return root


class AssetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.archive = self.root / 'glove.zip'
        with zipfile.ZipFile(self.archive, 'w') as z:
            z.writestr('glove.6B.300d.txt', 'one 1 2 3\ntwo 4 5 6\n')
            z.writestr('../../unwanted.txt', 'must not be extracted')
            z.writestr('glove.6B.50d.txt', 'unused dimension')

    def tearDown(self):
        self.temp.cleanup()

    def glove(self, output=None, **kwargs):
        return assets.prepare_glove(output or self.root / 'vectors', dimensions=3,
                                    expected_rows=2, **kwargs)

    def test_extracts_only_selected_member_and_preserves_source(self):
        original = assets.fingerprint(self.archive)
        result = self.glove(archive=self.archive)
        self.assertEqual(result['vectors']['rows'], 2)
        self.assertEqual(set(result['files']), {'glove.6B.300d.txt'})
        self.assertEqual(original, assets.fingerprint(self.archive))
        self.assertFalse((self.root / 'unwanted.txt').exists())
        self.assertFalse((self.root / 'vectors' / 'glove.6B.50d.txt').exists())

    def test_existing_complete_is_verified_no_download(self):
        first = self.glove(archive=self.archive)
        with mock.patch.object(assets, '_download', side_effect=AssertionError('network')):
            self.assertEqual(first, self.glove(download=True))
        (self.root / 'vectors' / 'glove.6B.300d.txt').write_text('changed', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'differs'):
            self.glove(download=True)

    def test_expected_archive_checksum_and_wrong_checksum(self):
        sha = assets.fingerprint(self.archive)['sha256']
        result = self.glove(archive=self.archive, expected_sha256=sha)
        self.assertEqual(result['input_expected_sha256'], sha)
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            self.glove(self.root / 'wrong', archive=self.archive, expected_sha256='0' * 64)

    def test_explicit_download_is_injectable_and_checked(self):
        calls = []
        def download(url, output):
            calls.append(url)
            shutil.copyfile(self.archive, output)
        self.glove(download=True, download_fn=download)
        self.assertEqual(calls, [assets.GLOVE_URL])

    def test_missing_source_is_rejected_before_creating_output(self):
        output = self.root / 'missing'
        with self.assertRaises(ValueError):
            self.glove(output)
        self.assertFalse(output.exists())

    def test_local_text_registration(self):
        text = self.root / 'local.txt'
        text.write_text('one 1 2 3\ntwo 4 5 6\n', encoding='utf-8')
        result = self.glove(text=text)
        self.assertEqual(result['input_kind'], 'text')
        self.assertEqual(result['files']['glove.6B.300d.txt'], assets.fingerprint(text))

    def test_invalid_shapes_nonfinite_and_row_count(self):
        for index, value in enumerate(('x 1 2\ny 3 4\n', 'x nan 2 3\ny 4 5 6\n', 'x 1 2 3\n')):
            path = self.root / ('invalid' + str(index))
            path.write_text(value, encoding='utf-8')
            with self.assertRaises(ValueError):
                assets.validate_glove_text(path, dimensions=3, expected_rows=2)

    def test_duplicate_member_is_rejected(self):
        path = self.root / 'duplicate.zip'
        with zipfile.ZipFile(path, 'w') as z:
            z.writestr('glove.6B.300d.txt', 'one 1 2 3\ntwo 4 5 6\n')
            with self.assertWarns(UserWarning):
                z.writestr('glove.6B.300d.txt', 'duplicate')
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            self.glove(archive=path)

    def test_unregistered_output_is_not_overwritten(self):
        output = self.root / 'vectors'
        output.mkdir()
        sentinel = output / 'keep.txt'
        sentinel.write_text('user file', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Preserve'):
            self.glove(archive=self.archive)
        self.assertEqual(sentinel.read_text(), 'user file')

    def test_local_semantic_copy_records_declared_revision_and_hashes(self):
        source = semantic_fixture(self.root / 'model_source')
        result = assets.prepare_semantic(self.root / 'model', source=source)
        self.assertEqual(result['revision'], assets.MODEL_REVISION)
        self.assertIn('user-declared', result['revision_evidence'])
        self.assertEqual(result, assets.verify_semantic_snapshot(self.root / 'model'))
        self.assertTrue((source / 'pytorch_model.bin').exists())

    def test_semantic_download_uses_pinned_revision_only(self):
        source = semantic_fixture(self.root / 'model_source')
        calls = []
        def fetch(output, revision):
            calls.append(revision)
            shutil.copytree(source, output, dirs_exist_ok=True)
        result = assets.prepare_semantic(self.root / 'model', download=True, snapshot_fetcher=fetch)
        self.assertEqual(calls, [assets.MODEL_REVISION])
        self.assertEqual(result['revision_evidence'], 'pinned Hub download')
        with self.assertRaisesRegex(ValueError, 'full model commit'):
            assets.prepare_semantic(self.root / 'mutable', download=True, revision='main', snapshot_fetcher=fetch)
        self.assertEqual(len(calls), 1)

    def test_mutated_semantic_file_is_rejected(self):
        source = semantic_fixture(self.root / 'model_source')
        assets.prepare_semantic(self.root / 'model', source=source)
        (self.root / 'model' / 'vocab.txt').write_text('tampered', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'differs'):
            assets.verify_semantic_snapshot(self.root / 'model')

    def test_missing_weight_lfs_pointer_wrong_architecture_custom_module(self):
        source = semantic_fixture(self.root / 'model_source')
        weight = source / 'pytorch_model.bin'
        weight.write_bytes(b'version https://git-lfs.github.com/spec/v1\noid sha256:fake\n')
        with self.assertRaisesRegex(ValueError, 'LFS pointer'):
            assets.validate_semantic_directory(source)
        weight.unlink()
        with self.assertRaisesRegex(ValueError, 'Missing local'):
            assets.validate_semantic_directory(source)
        weight.write_bytes(b'fake')
        (source / 'config.json').write_text('{"model_type":"bert","hidden_size":768}')
        with self.assertRaisesRegex(ValueError, 'MPNet'):
            assets.validate_semantic_directory(source)
        (source / 'config.json').write_text('{"model_type":"mpnet","hidden_size":768}')
        (source / 'modules.json').write_text('[{"type":"custom.remote.Code","path":""}]')
        with self.assertRaisesRegex(ValueError, 'custom'):
            assets.validate_semantic_directory(source)

    def test_snapshot_module_path_cannot_escape(self):
        source = semantic_fixture(self.root / 'model_source')
        (source / 'modules.json').write_text('[{"type":"sentence_transformers.models.Transformer","path":"../outside"}]')
        with self.assertRaisesRegex(ValueError, 'escaped'):
            assets.validate_semantic_directory(source)

    def test_cli_check_only_needs_no_source_download_or_model_import(self):
        self.glove(archive=self.archive)
        source = semantic_fixture(self.root / 'model_source')
        assets.prepare_semantic(self.root / 'model', source=source)
        for kind, output in (('glove', self.root / 'vectors'), ('semantic', self.root / 'model')):
            with mock.patch.object(sys, 'argv', ['prepare_model_assets.py', kind, '--output', str(output), '--check-only']), \
                 mock.patch.dict(sys.modules, {'torch': None, 'transformers': None}), \
                 mock.patch.object(assets, '_download', side_effect=AssertionError('network')), \
                 mock.patch('builtins.print') as printed:
                assets.main()
                self.assertFalse(json.loads(printed.call_args.args[0])['model_executed'])
    def test_verification_does_not_import_torch(self):
        source = semantic_fixture(self.root / 'model_source')
        with mock.patch.dict(sys.modules, {'torch': None, 'transformers': None, 'sentence_transformers': None}):
            assets.prepare_semantic(self.root / 'model', source=source)
            assets.verify_semantic_snapshot(self.root / 'model')


if __name__ == '__main__':
    unittest.main()

