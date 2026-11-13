"""CPU-only shared-DIGAT preparation contract tests with a synthetic backend."""
from argparse import Namespace
import hashlib
import json
from pathlib import Path
import pickle
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import prepare_digat_assets as builder


def identity(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class FakeRow(list):
    def __iadd__(self, other):
        for row in range(len(self)):
            for col in range(len(self[row])):
                self[row][col] = self[row][col] or other[row][col]
        return self


class FakeGraph(list):
    @property
    def shape(self):
        return (len(self), len(self[0]), len(self[0][0]))


class FakeNumpy:
    @staticmethod
    def identity(size, dtype):
        return [[row == col for col in range(size)] for row in range(size)]


class DigatBuilderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.args = Namespace(data_root=self.root / 'data', embedding_root=self.root / 'glove',
                              sag_model=self.root / 'semantic', output=self.root / 'shared',
                              seed=42, semantic_batch_size=32, check_only=False, resume=False)
        self.spec = {'seed': 42, 'fixture': 'synthetic-only'}
        self.helpers = mock.patch.object(builder, '_legacy_helpers', return_value=(None, identity, None))
        self.helpers.start()
        self.addCleanup(self.helpers.stop)

    def tearDown(self):
        self.temp.cleanup()

    def backend(self, args, manifest, record):
        for name in builder.REQUIRED_FILES:
            (args.output / name).write_bytes(('synthetic ' + name).encode())
        manifest['news_count'] = 2
        record()

    def prepare(self, **kwargs):
        return builder.prepare(self.args, spec_builder=lambda args: self.spec,
                               backend=kwargs.get('backend', self.backend))

    def test_complete_schema_matches_runner_six_shared_file_contract(self):
        result = self.prepare()
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(set(result['files']), set(builder.REQUIRED_FILES))
        self.assertEqual(result['identity'], identity(result['spec']))
        self.assertEqual(result['spec']['seed'], 42)
        self.assertNotIn('user_history_graph-MIND-small.pkl', result['files'])
        self.assertFalse(result['historical_R2_reproduction'])

    def test_check_only_never_calls_backend_imports_model_or_writes(self):
        self.args.check_only = True
        with mock.patch.dict(sys.modules, {'torch': None, 'numpy': None, 'sentence_transformers': None}):
            result = self.prepare(backend=mock.Mock(side_effect=AssertionError('executed')))
        self.assertEqual(result['status'], 'inputs_verified')
        self.assertFalse(result['execution_requested'])
        self.assertFalse(self.args.output.exists())

    def test_complete_output_idempotence_and_mutation_guard(self):
        first = self.prepare()
        self.assertEqual(first, self.prepare(backend=mock.Mock(side_effect=AssertionError('executed'))))
        (self.args.output / builder.REQUIRED_FILES[-1]).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'differs'):
            self.prepare()

    def test_source_or_input_identity_change_rejected(self):
        self.prepare()
        self.spec['seed'] = 43
        with self.assertRaisesRegex(ValueError, 'different inputs'):
            self.prepare()

    def test_unregistered_folder_preserved(self):
        self.args.output.mkdir()
        keep = self.args.output / 'keep'
        keep.write_bytes(b'unchanged')
        with self.assertRaisesRegex(ValueError, 'preserved'):
            self.prepare()
        self.assertEqual(keep.read_bytes(), b'unchanged')

    def test_incomplete_output_requires_explicit_resume(self):
        def fail(args, manifest, record):
            raise RuntimeError('synthetic interrupted SAG')
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            self.prepare(backend=fail)
        state = builder.read(self.args.output / 'asset_prep.json')
        self.assertEqual(state['status'], 'failed')
        with self.assertRaisesRegex(ValueError, '--resume'):
            self.prepare()
        self.args.resume = True
        result = self.prepare()
        self.assertEqual(result['resumed_from_status'], 'failed')
        self.assertEqual(result['status'], 'complete')

    def test_changed_metadata_prevents_resume(self):
        def fail(args, manifest, record):
            path = args.output / builder.REQUIRED_FILES[0]
            path.write_bytes(b'original metadata')
            manifest['metadata_files'] = {path.name: builder.fingerprint(path)}
            record()
            raise RuntimeError('synthetic interruption')
        with self.assertRaises(RuntimeError):
            self.prepare(backend=fail)
        (self.args.output / builder.REQUIRED_FILES[0]).write_bytes(b'changed metadata')
        self.args.resume = True
        with self.assertRaisesRegex(ValueError, 'Partial metadata changed'):
            self.prepare()

    def test_missing_backend_file_does_not_mark_complete(self):
        def incomplete(args, manifest, record):
            (args.output / builder.REQUIRED_FILES[0]).write_bytes(b'only one')
        with self.assertRaises(FileNotFoundError):
            self.prepare(backend=incomplete)
        self.assertEqual(builder.read(self.args.output / 'asset_prep.json')['status'], 'failed')

    def test_output_cannot_overlap_input(self):
        self.args.output = self.args.data_root / 'shared'
        with self.assertRaisesRegex(ValueError, 'separate'):
            self.prepare()

    def test_seed_and_batch_are_validated_before_execution(self):
        for seed, batch in ((-1, 32), (2 ** 32, 32), (42, 0)):
            self.args.seed, self.args.semantic_batch_size = seed, batch
            with self.assertRaises(ValueError):
                self.prepare()

    def test_backend_calls_vendor_sag_without_run_local_user_graph(self):
        calls = []
        self.args.embedding_root.mkdir()
        (self.args.embedding_root / 'glove.6B.300d.txt').write_text('synthetic-vector-text', encoding='utf-8')
        fake_np = SimpleNamespace(identity=FakeNumpy.identity,
                                  random=SimpleNamespace(seed=lambda seed: calls.append(('numpy_seed', seed))))
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True, manual_seed_all=lambda seed: None),
            manual_seed=lambda seed: None, set_num_threads=lambda count: None)
        fake_sag = SimpleNamespace(pretrain_model_name='original-model')
        def encoder(path, device):
            calls.append(('encoder', path, device))
            return SimpleNamespace(encode=lambda text, **kwargs: ['synthetic-embedding'])
        fake_sag.SentenceTransformer = encoder
        fake_mind = SimpleNamespace(GloVe=lambda **kwargs: None)
        def sag(*positional):
            calls.append(('vendor_sag', positional[0]))
            model = fake_sag.SentenceTransformer(fake_sag.pretrain_model_name)
            model.encode(['same exact text'])
            model.encode(['same exact text'])
            return ([0, 1], FakeGraph([FakeRow([[False, False], [False, False]]),
                                       FakeRow([[False, True], [True, False]])]), [True, True])
        fake_mind.construct_SAG = sag
        def metadata(config):
            calls.append(('metadata', config.dataset))
            for name in builder.METADATA_FILES:
                (self.args.output / name).write_text('{}', encoding='utf-8')
            (self.args.output / builder.REQUIRED_FILES[0]).write_text('{"PAD":0,"N1":1}', encoding='utf-8')
            fake_mind.construct_SAG()
            calls.append(('unexpected_run_local_user_graph',))
        fake_mind.MIND_Corpus = SimpleNamespace(preprocess=metadata)
        config = SimpleNamespace(dataset='MIND-small', train_root='train', dev_root='dev',
                                 test_root='test', SAG_neighbors=5, SAG_hops=2, news_graph_size=2)
        original_environment = {name: builder.os.environ.get(name) for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE')}
        original_directory = Path.cwd()
        with mock.patch.dict(sys.modules, {'numpy': fake_np, 'torch': fake_torch,
                                          'MIND_corpus': fake_mind, 'construct_SAG': fake_sag}), \
             mock.patch.object(builder, '_legacy_helpers', return_value=(lambda root, seed: config, identity, None)):
            result = self.prepare(backend=builder.execute_backend)
        self.assertEqual(result['exact_text_cache_hits'], 1)
        self.assertEqual(len([call for call in calls if call[0] == 'encoder']), 1)
        self.assertEqual(len([call for call in calls if call[0] == 'vendor_sag']), 1)
        self.assertFalse(any(call[0] == 'unexpected_run_local_user_graph' for call in calls))
        self.assertIs(fake_mind.construct_SAG, sag)
        self.assertIs(fake_sag.SentenceTransformer, encoder)
        self.assertEqual(fake_sag.pretrain_model_name, 'original-model')
        self.assertEqual(Path.cwd(), original_directory)
        for name, value in original_environment.items():
            self.assertEqual(builder.os.environ.get(name), value)

    def test_backend_requires_cuda_instead_of_silently_changing_algorithm(self):
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with mock.patch.dict(sys.modules, {'numpy': FakeNumpy, 'torch': fake_torch,
                                          'MIND_corpus': SimpleNamespace(), 'construct_SAG': SimpleNamespace()}):
            with self.assertRaisesRegex(RuntimeError, 'CUDA GPU'):
                self.prepare(backend=builder.execute_backend)
        self.assertEqual(builder.read(self.args.output / 'asset_prep.json')['status'], 'failed')
    def test_vendor_self_loop_and_pickle_structure_preserved(self):
        graph = FakeGraph([FakeRow([[False, True], [True, False]])])
        output = self.root / 'graph.pkl'
        builder.add_self_loops_and_save(([11, 12], graph, [True, True]), output, 2,
                                       numpy_module=FakeNumpy)
        with output.open('rb') as stream:
            result = pickle.load(stream)
        self.assertEqual(set(result), {'news_node_ID', 'news_graph', 'news_graph_mask'})
        self.assertEqual(result['news_graph'][0], [[True, True], [True, True]])
        self.assertEqual(output.read_bytes()[:2], b'\x80\x04')


if __name__ == '__main__':
    unittest.main()



