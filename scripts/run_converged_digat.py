"""Fresh multi-epoch DIGAT training with dev-only selection and restart state.

Training-matched held-out development weighted AvgMetric selects checkpoints. No test score is read
during training. Once training stops, the best checkpoint is evaluated on IMP,
RandS30/90/150 and full Total, using bounded resumable chunks.
"""
import argparse
import builtins
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
TEST_FILES = {'imp': 'behaviors.tsv', 'rs30': 'behaviors_rs30.tsv',
              'rs90': 'behaviors_rs90.tsv', 'rs150': 'behaviors_rs150.tsv'}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def fingerprint(path):
    path = Path(path).resolve()
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            sha.update(block)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha.hexdigest()}


def profile(args):
    return {name: str(value.resolve()) if isinstance(value, Path) else value
            for name, value in vars(args).items()
            if name not in ('resume', 'execute_snapshot', 'check_only')}


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_assets(shared, data_root, embedding_root, seed):
    # Standard-library checks also support safe --check-only before CUDA import.
    info = read_json(shared / 'asset_prep.json')
    if info.get('status') != 'complete' or signature(info['spec']) != info['identity']:
        raise ValueError('Complete, provenance-verified DIGAT assets are required')
    if info['spec']['seed'] != seed:
        raise ValueError('DIGAT metadata, OOV embeddings and sampled SAG must match this training seed')
    for split in ('train', 'dev', 'test'):
        if fingerprint(data_root / split / 'news.tsv')['sha256'] != info['spec']['news'][split]['sha256']:
            raise ValueError('Shared semantic graph belongs to a different news corpus')
    if fingerprint(embedding_root / 'glove.6B.300d.txt')['sha256'] != info['spec']['glove']['sha256']:
        raise ValueError('Shared GloVe source differs')
    required = ('news_ID-MIND-small.json', 'category-MIND-small.json', 'subCategory-MIND-small.json',
                'vocabulary-3-32-MIND-small.json', 'word_embedding-3-300-32-MIND-small.pkl',
                'news_graph-2-5-MIND-small.pkl')
    for name in required:
        if fingerprint(shared / name)['sha256'] != info['files'][name]['sha256']:
            raise ValueError('Shared DIGAT asset changed: ' + name)
    return {'manifest': fingerprint(shared / 'asset_prep.json'), 'identity': info['identity'],
            'seed': seed, 'files': {name: info['files'][name] for name in required}}


def bind_test_inputs(args, candidate_profile, candidate):
    """Use the new certificate's eligible IMP/RS/Total paths; preserve legacy routing."""
    consistent = candidate_profile is not None and candidate_profile.get('profile') == 'complete_proxy_strict_request_consistent_v1'
    if consistent:
        from consistent_candidate_profile import evaluation_paths
        paths = evaluation_paths(args.data_root, candidate_profile)
        if args.full_total != paths['total'].resolve():
            raise ValueError('New profile requires its certified full Total output')
        test_inputs = {}
        for kind, path in paths.items():
            actual = fingerprint(path)
            expected = candidate_profile['outputs']['test_' + kind]
            if any(actual[key] != expected[key] for key in ('sha256', 'bytes')):
                raise ValueError('Consistent test candidate hash differs: ' + kind)
            test_inputs[kind] = actual
        dataset = read_json(args.data_root / 'manifest.json')
        total_statistics = dataset['statistics']['test/behaviors_total.tsv']
        if any(candidate['statistics'][key] != total_statistics[key] for key in ('rows', 'candidates', 'positive_labels')):
            raise ValueError('New Total descriptor disagrees with dataset statistics')
    else:
        test_inputs = {kind: fingerprint(args.data_root / 'test' / name) for kind, name in TEST_FILES.items()}
        test_inputs['total'] = fingerprint(args.full_total)
    return test_inputs


def paired_imp_reference(args, assets, candidate_profile):
    """Allow fresh IMP, and bind RandS initialization to its completed paired IMP."""
    consistent = candidate_profile is not None and candidate_profile.get('profile') == 'complete_proxy_strict_request_consistent_v1'
    reference = getattr(args, 'paired_preprocessing_from', None)
    if reference is None:
        if consistent and args.train_type == 'rs50':
            raise ValueError('RandS training requires its completed paired DIGAT IMP reference')
        return None
    reference = Path(reference).resolve()
    old = read_json(reference / 'experiment.json')
    if (old.get('status') != 'completed' or old.get('model') != 'DIGAT'
            or old.get('train_type') != 'imp' or old.get('seed') != args.seed):
        raise ValueError('Require a completed matching-seed DIGAT IMP reference')
    if old['input_binding']['assets']['identity'] != assets['identity']:
        raise ValueError('Paired IMP and fresh DIGAT semantic metadata/SAG differ')
    initial = old.get('initial_model_sha256')
    if not isinstance(initial, str) or len(initial) != 64:
        raise ValueError('Preserved IMP initialization fingerprint is missing')
    names = ('news_ID-MIND-small.json', 'user_ID-MIND-small.json',
             'category-MIND-small.json', 'subCategory-MIND-small.json',
             'vocabulary-3-32-MIND-small.json')
    dictionaries = {}
    for name in names:
        actual = fingerprint(reference / name)
        expected = old['preprocess_files'][name]
        if any(actual[key] != expected[key] for key in ('bytes', 'sha256')):
            raise ValueError('Original paired DIGAT dictionary changed: ' + name)
        dictionaries[name] = actual
    return {'manifest': fingerprint(reference / 'experiment.json'),
            'initial_model_sha256': initial, 'dictionaries': dictionaries,
            'semantic_asset_identity': assets['identity']}


def configure_development(args):
    """Resolve dev candidates from training, before inputs or a model are loaded."""
    from training_validation import resolve_dev_path, resolve_dev_protocol
    args.dev_protocol = resolve_dev_protocol(args.train_type, args.dev_protocol)
    expected = resolve_dev_path(args.data_root, args.train_type, args.dev_protocol)
    supplied = getattr(args, 'dev_behaviors', None)
    if supplied is not None and Path(supplied).resolve() != expected:
        raise ValueError('Development must use the training-matched held-out split file')
    args.dev_behaviors = expected
    return expected


def corpus_behavior_overrides(data_root, train_type, dev_protocol='auto'):
    """Redirect only selected train/dev behavior files; leave news/test untouched."""
    from training_validation import resolve_dev_path
    root = Path(data_root).resolve()
    training = 'behaviors.tsv' if train_type == 'imp' else 'behaviors_rs50.tsv'
    return {root / 'train/behaviors.tsv': root / 'train' / training,
            root / 'dev/behaviors.tsv': resolve_dev_path(root, train_type, dev_protocol)}


def verify_resume_development(args):
    """Never dispatch a changed dev objective to an older frozen runner."""
    if not args.resume:
        return
    manifest = read_json(args.run_dir / 'experiment.json')
    stored = manifest.get('selection_protocol', {})
    protocol = stored.get('candidates', manifest.get('profile', {}).get('dev_protocol'))
    if protocol != args.dev_protocol:
        raise ValueError('Resume development protocol differs; use a new run directory')
    current = fingerprint(args.dev_behaviors)
    original = stored.get('file')
    if original is None:
        files = manifest.get('input_binding', {}).get('inputs', {}).get('files', [])
        original = next((item for item in files
                         if Path(item['path']).resolve() == args.dev_behaviors), None)
    if original is None or original['sha256'] != current['sha256']:
        raise ValueError('Resume development input changed; use a new run directory')


def freeze_or_execute(args):
    from digat_source_snapshot import SCRIPTS
    root, run = ROOT.resolve(), args.run_dir.resolve()
    snapshot = run / 'source_snapshot'
    manifest_path = snapshot / 'SOURCE_MANIFEST.json'
    if args.execute_snapshot:
        if root != snapshot:
            raise ValueError('Expected execution from this run source snapshot')
        manifest = read_json(manifest_path)
        for name, expected in manifest['files'].items():
            file = (snapshot / name).resolve()
            if snapshot not in file.parents or fingerprint(file)['sha256'] != expected['sha256']:
                raise ValueError('Frozen runtime source changed: ' + name)
        return manifest
    if args.resume:
        manifest = read_json(manifest_path)
        if Path(manifest['snapshot_root']).resolve() != snapshot:
            raise ValueError('Resume snapshot root changed')
    else:
        if run.exists():
            raise ValueError('Use a new run directory; --resume is explicit')
        names = list(SCRIPTS) + ['run_converged_digat.py', 'convergence_control.py', 'evaluate_existing_digat.py']
        paths = [root / 'scripts' / name for name in names]
        paths += sorted(path for path in (root / 'third_party/DIGAT').iterdir() if path.is_file())
        blobs = {path.relative_to(root).as_posix(): path.read_bytes() for path in paths}
        if any((root / name).read_bytes() != data for name, data in blobs.items()):
            raise ValueError('Source changed while freezing')
        snapshot.mkdir(parents=True, exist_ok=False)
        files = {}
        for name, data in blobs.items():
            destination = snapshot / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            files[name] = fingerprint(destination)
        manifest = {'snapshot_root': str(snapshot), 'runtime_executes_snapshot': True,
                    'files': files, 'created_unix': time.time()}
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    for name, expected in manifest['files'].items():
        if fingerprint(snapshot / name)['sha256'] != expected['sha256']:
            raise ValueError('Snapshot source validation failed')
    sys.dont_write_bytecode = True
    raise SystemExit(subprocess.call([sys.executable, '-B', str(snapshot / 'scripts/run_converged_digat.py'),
                                     *sys.argv[1:], '--execute-snapshot']))


def remove_owned_checkpoint(run_dir, path):
    """Only prune generated old best checkpoints in this new run, never originals."""
    root, target = Path(run_dir).resolve(), Path(path).resolve()
    if target.parent != root or not target.name.startswith('best_epoch_') or target.suffix != '.pt' or target.is_symlink():
        raise ValueError('Refuse checkpoint removal outside the generated run scope')
    target.unlink()


def learning_rate_epoch(max_epochs):
    # Exact upstream trainer formula: decay once at the start of the final 10%.
    return max_epochs - ((max_epochs - 1) // 10 + 1) + 1


def offload_optimizer_state(optimizer):
    """Offload Adam moments without recreating the optimizer or resetting steps."""
    import torch
    locations = []
    for index, state in enumerate(optimizer.state.values()):
        for key, value in state.items():
            if torch.is_tensor(value):
                locations.append((index, key, str(value.device)))
                state[key] = value.to('cpu')
            elif not isinstance(value, (int, float, bool)):
                raise ValueError('Unexpected nested Adam state; explicit offload implementation required')
    return locations


def restore_optimizer_devices(optimizer, locations):
    """Restore each Adam tensor to its original device, including CPU step counters."""
    import torch
    states = list(optimizer.state.values())
    expected = {(index, key) for index, state in enumerate(states)
                for key, value in state.items() if torch.is_tensor(value)}
    if expected != {(index, key) for index, key, device in locations}:
        raise ValueError('Adam state layout changed across memory offload')
    for index, key, device in locations:
        states[index][key] = states[index][key].to(device)


def evaluate_protocol(evaluator, behavior, protocol, run_dir, state, record, chunk_rows, expected=None):
    from evaluate_existing_digat import shard_input, validated_chunk, merge_chunks, ARTIFACTS
    from evaluation_progress import evaluate_test_file
    from experiment_progress import test_progress, finish_progress
    if expected is None:
        with Path(behavior).open('rb') as source:
            total_rows = sum(1 for line in source)
    else:
        total_rows = expected['rows']
    progress = state.setdefault('test_chunks', {}).setdefault(protocol, [])
    completed = {item['index']: item for item in progress}
    for shard in shard_input(behavior, run_dir / 'test_input_chunks' / protocol, chunk_rows):
        index = shard['index']
        if index in completed:
            if any(completed[index][key] != shard[key] for key in ('input', 'rows', 'start_row')):
                raise ValueError('Existing test chunk identity changed')
            validated_chunk(completed[index])
            test_progress(protocol, shard['start_row'] + shard['rows'], total_rows)
            continue
        destination = run_dir / 'test_chunks' / protocol / ('chunk_%05d_attempt_%d' % (index, time.time_ns()))
        destination.mkdir(parents=True, exist_ok=False)
        report = copy.deepcopy(evaluate_test_file(evaluator, Path(shard['input']['path']), destination / 'test',
                                                 protocol, start_row=shard['start_row'], total_rows=total_rows))
        progress.append({**shard, 'report': fingerprint(report['files']['metrics.json']),
                         'artifacts': {name: fingerprint(report['files'][name]) for name in ARTIFACTS}})
        state['current_test'] = {'protocol': protocol, 'completed_rows': sum(item['rows'] for item in progress)}
        record()
    ordered = sorted(progress, key=lambda item: item['index'])
    reports = [validated_chunk(item) for item in ordered]
    actual = {'rows': sum(item['rows'] for item in reports), 'candidates': sum(item['candidates'] for item in reports)}
    if expected and any(actual[name] != expected[name] for name in actual):
        raise ValueError('Full Total test coverage differs from certified input')
    spec = {'statistics': actual, 'output': fingerprint(behavior)}
    prefix = run_dir / 'evaluation' / (protocol + '_' + str(time.time_ns())) / ('test_' + protocol)
    result = merge_chunks(ordered, behavior, prefix, spec, evaluator.identity)
    result['protocol'] = protocol
    result['scope'] = 'Best development-selected multi-epoch checkpoint; no test-driven selection'
    from convergence_control import save_json_atomic
    save_json_atomic(result['files']['metrics.json'], result)
    finish_progress()
    return {'prefix': str(prefix), **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data-root', 'embedding-root', 'shared-preprocessing', 'run-dir', 'full-total', 'candidate-manifest'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--train-type', required=True, choices=('imp', 'rs50'))
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--max-epochs', type=int, default=50)
    parser.add_argument('--min-epochs', type=int, default=5)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--min-delta', type=float, default=1e-4)
    parser.add_argument('--dev-protocol', choices=('auto', 'imp', 'rs50'), default='auto')
    parser.add_argument('--dev-behaviors', type=Path)
    parser.add_argument('--selection-metric', choices=('avg',), default='avg')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Explicit RTX3070 experiment profile; upstream default64 is not silently assumed')
    parser.add_argument('--eval-batch-size', type=int, default=256)
    parser.add_argument('--chunk-rows', type=int, default=5000)
    parser.add_argument('--sparse-rejection-policy', choices=('abort', 'dense256'), default='dense256')
    parser.add_argument('--paired-preprocessing-from', type=Path, help='Preserved paired IMP run for exact dictionaries and initial weights')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--cooling', action='store_true', help='Rest 2 minutes per 10 training minutes and 15 minutes per 3 wall-clock hours')
    parser.add_argument('--execute-snapshot', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    configure_development(args)
    verify_resume_development(args)
    if not 0 <= args.seed < 2**32 or min(args.batch_size, args.eval_batch_size, args.chunk_rows) < 1:
        raise ValueError('Invalid seed or batch/chunk size')
    from convergence_control import EarlyStopping
    control = EarlyStopping(args.max_epochs, args.min_epochs, args.patience, args.min_delta)
    from nnr_artifacts import check_inputs
    from digat_prepare import matched_candidate_profile
    candidate_profile = matched_candidate_profile(args.data_root, args.train_type)
    inputs = check_inputs(args.data_root, args.train_type, args.dev_protocol, 'imp')
    assets = validate_assets(args.shared_preprocessing, args.data_root, args.embedding_root, args.seed)
    paired = paired_imp_reference(args, assets, candidate_profile)
    candidate = read_json(args.candidate_manifest)
    if candidate.get('status') != 'complete' or candidate.get('protocol') != 'total':
        raise ValueError('Certified complete full Total input required')
    if Path(candidate['dataset_root']).resolve() != args.data_root or Path(candidate['output']['path']).resolve() != args.full_total:
        raise ValueError('Full Total belongs to a different dataset or file')
    if fingerprint(args.full_total)['sha256'] != candidate['output']['sha256']:
        raise ValueError('Full Total input changed')
    test_inputs = bind_test_inputs(args, candidate_profile, candidate)
    consistent = candidate_profile is not None and candidate_profile.get('profile') == 'complete_proxy_strict_request_consistent_v1'
    binding = {'inputs': inputs, 'assets': assets, 'candidate_manifest': fingerprint(args.candidate_manifest),
               'test_inputs': test_inputs, 'matched_candidate_profile': candidate_profile}
    if paired is not None:
        binding['paired_imp_training'] = paired
    if args.check_only:
        print(json.dumps({'status': 'validated', 'model': 'DIGAT', 'seed': args.seed,
                          'train_type': args.train_type, 'profile': profile(args), 'training_started': False}))
        return
    snapshot = freeze_or_execute(args)
    from convergence_control import (atomic_torch_save, capture_rng_state, restore_rng_state,
        save_json_atomic, prepare_epoch_cache, cleanup_epoch_cache, selection_value)
    from run_small_digat import copy_shared_assets, verified_evaluator
    from digat_small_assets import config_for
    start, start_unix = time.monotonic(), time.time()
    target = args.run_dir / 'experiment.json'
    latest_path = args.run_dir / 'latest_training_state.pt'
    if args.resume and target.exists():
        manifest = read_json(target)
        if manifest['profile'] != profile(args) or manifest['input_binding'] != binding:
            raise ValueError('Resume profile or input identity changed')
        if manifest['status'] == 'completed':
            print(json.dumps({'status': 'already_completed', 'manifest': str(target)}))
            return
    else:
        manifest = {'schema': 1, 'status': 'preparing', 'model': 'DIGAT', 'train_type': args.train_type,
                    'seed': args.seed, 'profile': profile(args), 'input_binding': binding,
                    'source_snapshot': snapshot, 'historical_R2_reproduction': False,
                    'training_reused': False, 'epoch_records': [], 'evaluations': {},
                    'test_chunks': {}, 'execution_segments': [], 'cleanup_records': [],
                    'selection': {'protocol': args.dev_protocol, 'file': fingerprint(args.dev_behaviors),
                        'matched_candidate_profile': candidate_profile, 'metric': args.selection_metric,
                        'policy': 'training_matched', 'test_used_for_selection': False},
                    'selection_protocol': {'split': 'dev', 'candidates': args.dev_protocol, 'metric': 'avg',
                        'file': fingerprint(args.dev_behaviors), 'policy': 'training_matched',
                        'test_used_for_selection': False,
                        'matched_candidate_profile': candidate_profile,
                        'formula': '(AUC + MRR + (nDCG@5 + nDCG@10)/2)/3'},
                    'scheduler': {'policy': 'upstream_final10percent_decay', 'factor': 0.1,
                                  'decay_epoch': learning_rate_epoch(args.max_epochs)},
                    'batch_policy': {'value': args.batch_size, 'upstream_default': 64,
                                     'explicit_hardware_profile': 'RTX3070', 'adaptive_changes': False}}
    segment = {'start_unix': start_unix, 'elapsed_seconds': 0.0, 'status': 'running'}
    manifest['execution_segments'].append(segment)
    import scheduled_cooling as cooling
    def record():
        segment['elapsed_seconds'] = time.monotonic() - start
        manifest['elapsed_seconds'] = sum(item['elapsed_seconds'] for item in manifest['execution_segments'])
        cooling.record_manifest(manifest, segment)
        save_json_atomic(target, manifest)
    record()
    previous_cwd = Path.cwd()
    try:
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(ROOT / 'third_party/DIGAT'))
        import numpy as np
        import torch
        from torch.utils.data import DataLoader
        import MIND_corpus
        from MIND_dataset import MIND_Train_Dataset
        from model import Model
        from digat_cached_eval import DigatCachedEvaluator, news_categories
        from naml_cached_eval import model_fingerprint
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required')
        torch.set_num_threads(4)
        torch.cuda.set_device(0)
        if args.cooling:
            cooling.install(args.run_dir/'cooling_events.jsonl', torch.cuda.synchronize)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        os.chdir(args.run_dir)
        cfg = config_for(args.data_root, args.seed, args.batch_size, args.max_epochs)
        if not latest_path.exists():
            copy_shared_assets(args.shared_preprocessing, args.data_root, args.embedding_root, args.run_dir, args.seed)
        selected_train = args.data_root / 'train' / ('behaviors.tsv' if args.train_type == 'imp' else 'behaviors_rs50.tsv')
        behavior_overrides = corpus_behavior_overrides(args.data_root, args.train_type, args.dev_protocol)
        user_ids = {'<UNK>': 0}
        # Keep the first-seen IMP user dictionary even when zero-negative
        # requests are excluded from derived RandS training. This is a model
        # vocabulary reference only, never a training-example substitution.
        user_reference = args.data_root / 'train/behaviors.tsv' if consistent else selected_train
        manifest['user_ID_reference'] = {'policy': 'IMP_master_first_seen_order' if consistent else 'selected_training_first_seen_order',
                                        'input': fingerprint(user_reference)}
        with user_reference.open(encoding='utf-8') as stream:
            for line in stream:
                user = line.split('\t')[1]
                if user not in user_ids:
                    user_ids[user] = len(user_ids)
        if not latest_path.exists():
            save_json_atomic(args.run_dir / 'user_ID-MIND-small.json', user_ids)
        elif read_json(args.run_dir / 'user_ID-MIND-small.json') != user_ids:
            raise ValueError('Saved training user mapping changed')
        def selected_open(file, *positional, **kwargs):
            if isinstance(file, (str, os.PathLike)):
                file = behavior_overrides.get(Path(file).resolve(), file)
            return builtins.open(file, *positional, **kwargs)
        MIND_corpus.open = selected_open
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
        MIND_corpus.MIND_Corpus.preprocess(cfg)
        corpus = MIND_corpus.MIND_Corpus(cfg)
        categories = news_categories(corpus, args.data_root)
        cfg.attribute_dict = {name: value for name, value in vars(cfg).items() if name != 'attribute_dict'}
        manifest['effective_config'] = cfg.attribute_dict
        preprocess_names = ('news_ID-MIND-small.json', 'category-MIND-small.json',
            'subCategory-MIND-small.json', 'vocabulary-3-32-MIND-small.json',
            'word_embedding-3-300-32-MIND-small.pkl', 'news_graph-2-5-MIND-small.pkl',
            'user_ID-MIND-small.json', 'user_history_graph-50-MIND-small.pkl')
        preprocess = {name: fingerprint(args.run_dir / name) for name in preprocess_names}
        if paired is not None:
            for name, expected in paired['dictionaries'].items():
                if any(preprocess[name][key] != expected[key] for key in ('bytes', 'sha256')):
                    raise ValueError('Fresh DIGAT dictionary differs from preserved IMP: ' + name)
            manifest['paired_initialization_reference'] = paired
        if 'preprocess_files' in manifest and manifest['preprocess_files'] != preprocess:
            raise ValueError('Saved preprocessing files changed')
        manifest['preprocess_files'] = preprocess
        environment = {'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                       'gpu': torch.cuda.get_device_name(0), 'cuda_device_count': torch.cuda.device_count()}
        if 'environment' in manifest and manifest['environment'] != environment:
            raise ValueError('Exact restart requires the original torch/CUDA/GPU environment')
        manifest['environment'] = environment
        # Pair IMP/RandS runs by an explicit seed reset after all preprocessing.
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
        model = Model(cfg).cuda(); model.initialize()
        initial_hash = model_fingerprint(model)
        if paired is not None and initial_hash != paired['initial_model_sha256']:
            raise ValueError('Fresh DIGAT initialization differs from the preserved IMP model')
        if 'initial_model_sha256' in manifest and manifest['initial_model_sha256'] != initial_hash:
            raise ValueError('Restart reconstructed a different initial model')
        manifest['initial_model_sha256'] = initial_hash
        manifest['initialization_rng_reset_after_preprocessing'] = True
        no_decay = ('.bias', 'embed', 'graph_encoder.')
        groups = [
            {'params': [q for n, q in model.named_parameters() if q.requires_grad and not any(x in n.lower() for x in no_decay)], 'weight_decay': cfg.weight_decay},
            {'params': [q for n, q in model.named_parameters() if q.requires_grad and any(x in n.lower() for x in no_decay)], 'weight_decay': 0.0}]
        optimizer = torch.optim.Adam(groups, lr=cfg.lr)
        dataset = MIND_Train_Dataset(corpus)
        state = {'phase': 'ready', 'epoch': 0, 'controller': control.state_dict(), 'epoch_records': [],
                 'best_path': None, 'best_cache': None, 'best_evaluation': None,
                 'scheduler_state': {'decay_applied': False}, 'profile_signature': signature(profile(args)),
                 'input_signature': signature(binding), 'initial_model_sha256': initial_hash}
        if latest_path.exists():
            saved = torch.load(latest_path, map_location='cpu')
            if saved['state']['profile_signature'] != signature(profile(args)) or saved['state']['input_signature'] != signature(binding):
                raise ValueError('Training checkpoint profile/input identity changed')
            if saved['config'] != cfg.attribute_dict:
                raise ValueError('Training checkpoint effective configuration changed')
            state = saved['state']
            model.load_state_dict(saved['model'], strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            control = EarlyStopping.from_state_dict(state['controller'])
            restore_rng_state(saved['rng'])
            del saved
        def checkpoint(phase):
            state['phase'] = phase
            state['controller'] = control.state_dict()
            atomic_torch_save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                               'rng': capture_rng_state(), 'config': cfg.attribute_dict,
                               'state': copy.deepcopy(state)}, latest_path)
            manifest['latest_checkpoint'] = fingerprint(latest_path)
            manifest['epoch_records'] = copy.deepcopy(state['epoch_records'])
            manifest['early_stopping'] = control.state_dict()
            manifest['epochs_completed'] = control.last_epoch
            manifest['best_epoch'] = control.best_epoch
            record()
        if not latest_path.exists():
            checkpoint('ready')
        from experiment_progress import train_progress, finish_progress, clear_console_progress
        while control.stop_reason is None:
            epoch = state['epoch'] if state['phase'] == 'awaiting_dev' else control.last_epoch + 1
            if state['phase'] != 'awaiting_dev':
                if 'optimizer_devices' in state:
                    restore_optimizer_devices(optimizer, state['optimizer_devices'])
                if epoch == learning_rate_epoch(args.max_epochs) and not state['scheduler_state']['decay_applied']:
                    for group in optimizer.param_groups:
                        group['lr'] /= 10
                    state['scheduler_state']['decay_applied'] = True
                manifest['status'] = 'training'; manifest['current_epoch'] = epoch; record()
                cooling.set_phase('training', epoch=epoch)
                train_rest_start = cooling.marker()
                train_start, train_unix, last_print = time.monotonic(), time.time(), time.monotonic()
                dataset.negative_sampling(verbose=False)
                loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
                model.train(); loss_sum, samples = 0.0, 0
                steps = len(loader)
                train_progress(epoch, args.max_epochs, 0, steps)
                for step, batch in enumerate(loader, 1):
                    batch = tuple(value.cuda(non_blocking=True) for value in batch)
                    logits = model(*batch[:3], *batch[4:])
                    loss = -torch.log_softmax(logits, dim=1).select(1, 0).mean()
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite training loss')
                    optimizer.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_norm)
                    optimizer.step()
                    count = batch[0].size(0); samples += count; loss_sum += float(loss) * count
                    train_progress(epoch, args.max_epochs, step, steps)
                    cooling.training_boundary()
                    if time.monotonic() - last_print >= 300:
                        clear_console_progress()
                        print(json.dumps({'stage': 'training', 'epoch': epoch, 'step': step,
                                          'steps': len(loader), 'mean_loss': loss_sum / samples}), flush=True)
                        record(); last_print = time.monotonic()
                if not samples:
                    raise ValueError('No training examples')
                torch.cuda.synchronize()
                finish_progress()
                state['epoch'] = epoch
                train_seconds = time.monotonic() - train_start
                train_cooling = cooling.duration_fields(train_seconds, train_rest_start)
                state['pending_train'] = {'epoch': epoch, 'mean_loss': loss_sum / samples,
                                          'train_cooling_rest_seconds': train_cooling['cooling_rest_seconds'],
                                          'train_net_seconds': train_cooling['net_elapsed_seconds'],
                                          'train_seconds': train_seconds,
                                          'start_unix': train_unix, 'end_unix': time.time(),
                                          'learning_rates': [group['lr'] for group in optimizer.param_groups]}
                optimizer.zero_grad(set_to_none=True)
                state['optimizer_devices'] = offload_optimizer_state(optimizer)
                del loader, batch, logits, loss
                gc.collect(); torch.cuda.empty_cache()
                # Preserve trained weights/optimizer/RNG before any dev failure.
                state['pending_cache'] = str(prepare_epoch_cache(args.run_dir, epoch))
                checkpoint('awaiting_dev')
            else:
                # torch.load/optimizer.load_state_dict may have restored moments
                # to CUDA; pending validation must release them again.
                restored_locations = offload_optimizer_state(optimizer)
                if 'optimizer_devices' not in state:
                    state['optimizer_devices'] = restored_locations
                gc.collect(); torch.cuda.empty_cache()
            cooling.set_phase('development_setup', epoch=epoch)
            cache_root = Path(state['pending_cache'])
            manifest['status'] = 'validating'; record()
            def factory(**kwargs):
                return DigatCachedEvaluator(model, corpus, categories, cache_root, news_batch_size=256, **kwargs)
            fallback_path = args.run_dir / ('sparse_fallback_epoch_%04d.json' % epoch)
            pending = state.get('pending_evaluation')
            if pending is None and fallback_path.exists():
                prior_fallback = read_json(fallback_path)
                rejected = prior_fallback['failed_gate_evidence']
                if prior_fallback['model_state_sha256'] != model_fingerprint(model) or fingerprint(rejected['path'])['sha256'] != rejected['sha256']:
                    raise ValueError('Pending sparse rejection belongs to different weights/evidence')
                if prior_fallback['status'] not in ('dense_verification_pending', 'dense_verified'):
                    raise ValueError('Unsupported pending fallback status')
                pending = {'graph_backend': 'dense', 'projection_cache': False, 'batch_size': 256}
            pending = pending or {'graph_backend': 'sparse_csr', 'projection_cache': True,
                                  'batch_size': args.eval_batch_size}
            evaluator, gate, fallback = verified_evaluator(factory, args.dev_behaviors,
                sparse_eval=pending['graph_backend'] == 'sparse_csr', projection_cache=pending['projection_cache'], score_batch_size=pending['batch_size'],
                rejection_policy=args.sparse_rejection_policy if pending['graph_backend'] == 'sparse_csr' else 'abort',
                fallback_path=fallback_path,
                release=torch.cuda.empty_cache)
            if gate.get('status') != 'passed' or gate.get('rows_with_rank_disagreement') != 0:
                raise ValueError('Development cache/native score and rank equivalence must pass')
            save_json_atomic(args.run_dir / ('cache_equivalence_epoch_%04d.json' % epoch), gate)
            selected = {'projection_cache': evaluator.use_projection_cache, 'batch_size': evaluator.score_batch_size,
                        'graph_backend': evaluator.provenance['graph_backend']}
            state['pending_evaluation'] = selected
            checkpoint('awaiting_dev')
            dev_dir = args.run_dir / 'development' / ('epoch_%04d' % epoch)
            dev_report_path = dev_dir / 'dev.metrics.json'
            if dev_report_path.exists():
                dev = read_json(dev_report_path)
                if dev['cache_identity'] != evaluator.identity or dev['input']['sha256'] != fingerprint(args.dev_behaviors)['sha256']:
                    raise ValueError('Pending development report belongs to a different checkpoint/input')
                for file in dev['files'].values():
                    if not Path(file).exists():
                        raise ValueError('Pending development artifacts are incomplete')
            else:
                if dev_dir.exists():
                    dev_dir = args.run_dir / 'development' / ('epoch_%04d_attempt_%d' % (epoch, time.time_ns()))
                cooling.set_phase('dev', epoch=epoch)
                dev = copy.deepcopy(evaluator.evaluate_file(args.dev_behaviors, dev_dir / 'dev'))
            score = selection_value(dev['metrics'], 'avg')
            decision = control.update(epoch, score)
            prior_best_path, prior_best_cache = state['best_path'], state['best_cache']
            if decision['is_best']:
                best_path = args.run_dir / ('best_epoch_%04d.pt' % epoch)
                atomic_torch_save({model.model_name: model.state_dict(), 'epoch': epoch,
                                  'config': cfg.attribute_dict, 'preprocessing_identity': assets['identity'],
                                  'selection_score': score}, best_path)
                state['best_path'], state['best_cache'] = str(best_path), str(cache_root)
                state['best_evaluation'] = selected
            state['epoch_records'].append({**state['pending_train'], 'dev': dev,
                'selection_score': score, 'decision': decision, 'selected_evaluation': selected,
                'sparse_fallback': fallback})
            state.pop('pending_train', None); state.pop('pending_cache', None); state.pop('pending_evaluation', None)
            del evaluator; gc.collect(); torch.cuda.empty_cache()
            # This snapshot includes RNG after all dev/cache operations.
            checkpoint('training_complete' if decision['should_stop'] else 'ready')
            if prior_best_path and prior_best_path != state['best_path']:
                remove_owned_checkpoint(args.run_dir, prior_best_path)
            for obsolete in (prior_best_cache, str(cache_root)):
                if obsolete and obsolete != state['best_cache'] and Path(obsolete).exists():
                    manifest['cleanup_records'].append(cleanup_epoch_cache(args.run_dir, obsolete))
            record()
            print(json.dumps({'stage': 'epoch_completed', 'epoch': epoch, 'dev_selection': score,
                              'best_epoch': control.best_epoch, 'stop_reason': control.stop_reason}), flush=True)
        manifest.update(status='training_complete', epochs_completed=control.last_epoch,
                        best_epoch=control.best_epoch, stop_reason=control.stop_reason,
                        converged=control.stop_reason == 'patience', early_stopping=control.state_dict())
        manifest['checkpoint'] = fingerprint(state['best_path'])
        record()
        # Release optimizer before final evaluation; never test every epoch.
        del optimizer, groups, dataset
        gc.collect(); torch.cuda.empty_cache()
        best = torch.load(state['best_path'], map_location='cpu')
        model.load_state_dict(best[model.model_name], strict=True); del best
        selected = state['best_evaluation']
        cooling.set_phase('test_setup')
        evaluator = DigatCachedEvaluator(model, corpus, categories, Path(state['best_cache']),
            news_batch_size=256, score_batch_size=selected['batch_size'],
            sparse_eval=selected['graph_backend'] == 'sparse_csr', projection_cache=selected['projection_cache'])
        gate = evaluator.verify_against_forward(args.dev_behaviors)
        if gate.get('status') != 'passed' or gate.get('rows_with_rank_disagreement') != 0:
            raise ValueError('Best-checkpoint cache/native score and rank equivalence must pass')
        save_json_atomic(args.run_dir / 'cache_equivalence_best.json', gate)
        manifest['selected_evaluation'] = selected
        manifest['status'] = 'evaluating'; record()
        for protocol, item in test_inputs.items():
            if protocol in manifest['evaluations']:
                done = manifest['evaluations'][protocol]
                stored = read_json(done['files']['metrics.json'])
                if stored['input']['sha256'] != item['sha256'] or stored['cache_identity'] != evaluator.identity:
                    raise ValueError('Completed test result identity changed')
                continue
            cooling.set_phase('test', protocol=protocol)
            expected = candidate['statistics'] if protocol == 'total' else None
            manifest['evaluations'][protocol] = evaluate_protocol(evaluator, Path(item['path']), protocol,
                args.run_dir, manifest, record, args.chunk_rows, expected)
            record()
        manifest['timings_seconds'] = {'training': sum(item['train_seconds'] for item in state['epoch_records']),
            'dev_evaluation': sum(item['dev']['elapsed_seconds'] for item in state['epoch_records']),
            'tests': {kind: item['elapsed_seconds'] for kind, item in manifest['evaluations'].items()},
            'test_total': sum(item['elapsed_seconds'] for item in manifest['evaluations'].values()),
            'total': sum(item['elapsed_seconds'] for item in manifest['execution_segments'])}
        cooling.finalize_timing(manifest)
        manifest['timing_scope'] = 'Raw completed epoch/dev/test timers preserve cooling. timings_seconds_net subtracts measured rests once; process segments include setup/retries and exclude inter-invocation gaps.'
        manifest['status'] = 'completed'; segment['status'] = 'completed'; record()
        manifest['timings_seconds']['total'] = manifest['elapsed_seconds']; record()
        print(json.dumps({'stage': 'completed', 'manifest': str(target), 'best_epoch': control.best_epoch,
                          'epochs_completed': control.last_epoch, 'stop_reason': control.stop_reason}), flush=True)
    except BaseException:
        manifest['status'] = 'failed'; manifest['error'] = traceback.format_exc()
        segment['status'] = 'failed'; record()
        raise
    finally:
        os.chdir(previous_cwd)


if __name__ == '__main__':
    main()
