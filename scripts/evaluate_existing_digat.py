"""Evaluate full Total using a completed DIGAT checkpoint, without retraining.

The original run is read-only. Frozen runtime sources and preprocessing are
verified; its score cache is copied to a new output directory. Chunking bounds
memory and preserves completed chunks across explicit --resume invocations.
"""
import argparse
import builtins
import copy
import gc
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback
from types import SimpleNamespace

METRICS = ('AUC', 'MRR', 'nDCG@5', 'nDCG@10')
ARTIFACTS = ('scores.npy', 'labels.npy', 'offsets.npy', 'news_indices.npy',
             'user_indices.npy', 'entry_metrics.npy', 'records.jsonl',
             'predictions.txt', 'truth.txt')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def fingerprint(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def verify_file(path, expected):
    result = fingerprint(path)
    digest = expected if isinstance(expected, str) else expected['sha256']
    if result['sha256'] != digest or (isinstance(expected, dict) and
            'bytes' in expected and result['bytes'] != expected['bytes']):
        raise ValueError('Artifact hash/size mismatch: ' + str(path))
    return result


def inside(root, relative):
    root = Path(root).resolve()
    target = (root / relative).resolve()
    if root not in target.parents:
        raise ValueError('Artifact path escapes required root: ' + str(relative))
    return target


def validate_candidate_manifest(candidate_manifest, behaviors, original):
    candidate = read_json(candidate_manifest)
    if candidate.get('status') != 'complete' or candidate.get('protocol') != 'total':
        raise ValueError('A completed FULL Total candidate manifest is required')
    if Path(candidate['dataset_root']).resolve() != Path(original['data_root']).resolve():
        raise ValueError('Full Total belongs to a different dataset root')
    if Path(candidate['output']['path']).resolve() != Path(behaviors).resolve():
        raise ValueError('Selected candidate path differs from manifest')
    required = ('request_order_matches_existing_evaluation_mapping',
                'positive_labels_match_existing_evaluation_mapping',
                'total5_subset_of_total', 'rands30_subset_of_total',
                'all_news_in_model_vocabularies')
    if any(candidate.get('validation', {}).get(key) is not True for key in required):
        raise ValueError('Candidate alignment/subset/vocabulary checks are incomplete')
    verify_file(behaviors, candidate['output'])
    verify_file(candidate['dataset_manifest']['path'], candidate['dataset_manifest'])
    for protocol in ('rs30', 'rs90', 'rs150', 'total5'):
        if candidate['source_existing_evaluations'][protocol]['sha256'] != original['evaluations'][protocol]['input']['sha256']:
            raise ValueError('Full Total alignment source differs from completed ' + protocol + ' evaluation')
    if candidate['source_imp']['sha256'] != original['evaluations']['imp']['input']['sha256']:
        raise ValueError('Full Total source IMP identity differs from the completed run')
    prior = original['evaluations']['total5']
    if candidate['statistics']['rows'] != prior['rows']:
        raise ValueError('Full Total must preserve the positive-expanded Total5 request rows')
    if candidate['statistics']['candidates'] <= prior['candidates']:
        raise ValueError('Expected expanded full Total, not the existing Total5 input')
    return candidate


def validate_original(original_run, behaviors, candidate_manifest):
    original_run = Path(original_run).resolve()
    original = read_json(original_run / 'experiment.json')
    if original.get('status') != 'completed' or original.get('model') != 'DIGAT':
        raise ValueError('Require a completed DIGAT training/evaluation run')
    if original.get('smoke_only') or original.get('train_type') not in ('imp', 'rs50'):
        raise ValueError('Require full completed IMP or RandS50 training')
    if 'total' in original.get('evaluations', {}):
        raise ValueError('Full Total already exists; do not repeat it')
    evidence = {'original_manifest': fingerprint(original_run / 'experiment.json')}
    snapshot = original_run / 'source_snapshot'
    source_manifest = read_json(snapshot / 'SOURCE_MANIFEST.json')
    if source_manifest != original['source_snapshot']:
        raise ValueError('Original runtime source manifest disagrees with experiment')
    evidence['source_snapshot'] = fingerprint(snapshot / 'SOURCE_MANIFEST.json')
    evidence['source_files'] = {name: verify_file(inside(snapshot, name), expected)
                                for name, expected in source_manifest['files'].items()}
    evidence['checkpoint'] = verify_file(original_run / 'best_model.pt', original['checkpoint'])
    evidence['preprocess_files'] = {name: verify_file(inside(original_run, name), expected)
                                    for name, expected in original['preprocess_files'].items()}
    evidence['inputs'] = [verify_file(item['path'], item) for item in original['inputs']['files']]
    gate = read_json(original_run / 'cache_equivalence_best.json')
    if gate.get('status') != 'passed' or gate.get('rows_with_rank_disagreement') != 0:
        raise ValueError('Original best-checkpoint score/rank validation did not pass')
    evidence['original_gate'] = fingerprint(original_run / 'cache_equivalence_best.json')
    selected = original['selected_evaluation']
    if selected['graph_backend'] not in ('dense', 'sparse_csr') or selected['batch_size'] <= 0:
        raise ValueError('Unsupported original evaluation profile')
    cache_name = original['evaluations']['total5']['cache_identity']
    cache_dir = inside(original_run / 'embedding_cache', cache_name)
    cache = read_json(cache_dir / 'manifest.json')
    if cache['identity'] != cache_name or gate['cache_identity'] != cache_name:
        raise ValueError('Best checkpoint and completed evaluation cache differ')
    provenance = cache['provenance']
    if provenance['model_state_sha256'] != gate['model_state_sha256']:
        raise ValueError('Checkpoint state fingerprint differs from cache')
    if provenance['graph_backend'] != selected['graph_backend'] or provenance['projection_cache'] != selected['projection_cache']:
        raise ValueError('Selected cache backend/projection profile differs')
    evidence['cache_manifest'] = fingerprint(cache_dir / 'manifest.json')
    evidence['cache_files'] = {name: verify_file(inside(cache_dir, name), digest)
                               for name, digest in cache['files'].items()}
    if selected['graph_backend'] == 'sparse_csr':
        gate_path = cache_dir / 'sparse_checkpoint_gate' / 'verification.json'
        sparse_gate = read_json(gate_path)
        if sparse_gate.get('status') != 'passed' or sparse_gate['provenance']['sparse_batch_size'] != selected['batch_size']:
            raise ValueError('Original sparse checkpoint gate does not support selected batch')
        evidence['sparse_gate'] = verify_file(gate_path, gate['sparse_dense_gate']['report_sha256'])
        evidence['sparse_gate_files'] = {name: verify_file(inside(gate_path.parent, name), digest)
                                        for name, digest in sparse_gate['files'].items()}
    candidate = validate_candidate_manifest(candidate_manifest, behaviors, original)
    evidence['candidate_manifest'] = fingerprint(candidate_manifest)
    evidence['candidate_file'] = dict(candidate['output'])
    return original, candidate, evidence, cache_dir


def shard_input(behavior_path, directory, chunk_rows):
    """Create deterministic whole-row chunks; retain and verify existing chunks."""
    if chunk_rows < 1:
        raise ValueError('Positive chunk rows required')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with Path(behavior_path).open(encoding='utf-8') as stream:
        index, start = 0, 0
        while True:
            lines = list(itertools.islice(stream, chunk_rows))
            if not lines:
                break
            content = ''.join(line.rstrip('\r\n') + '\n' for line in lines).encode('utf-8')
            target = directory / ('chunk_%05d.tsv' % index)
            digest = hashlib.sha256(content).hexdigest()
            if target.exists():
                verify_file(target, {'sha256': digest, 'bytes': len(content)})
            else:
                target.write_bytes(content)
            yield {'index': index, 'start_row': start, 'rows': len(lines),
                   'input': {'path': str(target.resolve()), 'bytes': len(content), 'sha256': digest}}
            start += len(lines)
            index += 1


def validated_chunk(record):
    """Check a completed chunk before reuse; a partial attempt is never completed."""
    verify_file(record['input']['path'], record['input'])
    report = read_json(record['report']['path'])
    verify_file(record['report']['path'], record['report'])
    if report['input']['sha256'] != record['input']['sha256'] or report['rows'] != record['rows']:
        raise ValueError('Completed chunk input/order/count mismatch')
    for name in ARTIFACTS:
        verify_file(report['files'][name], record['artifacts'][name])
    return report


def merge_chunks(records, behaviors, output_prefix, candidate, cache_identity):
    """Merge raw per-entry results without averaging shard averages equally."""
    import numpy as np
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    reports = [validated_chunk(record) for record in records]
    rows = sum(report['rows'] for report in reports)
    count = sum(report['candidates'] for report in reports)
    if rows != candidate['statistics']['rows'] or count != candidate['statistics']['candidates']:
        raise ValueError('Completed chunks do not cover full candidate file')
    if any(report['cache_identity'] != cache_identity for report in reports):
        raise ValueError('Chunk caches differ; cannot combine results')
    paths = {name: Path(str(prefix) + '.' + name) for name in ARTIFACTS + ('metrics.json',)}
    if any(path.exists() for path in paths.values()):
        raise ValueError('Merged output exists; preserve it and choose a fresh merge prefix')
    specs = {'scores': (np.float32, (count,)), 'labels': (np.int8, (count,)),
             'news_indices': (np.int32, (count,)), 'offsets': (np.int64, (rows + 1,)),
             'user_indices': (np.int64, (rows,)), 'entry_metrics': (np.float64, (rows, 4))}
    arrays = {name: np.lib.format.open_memmap(paths[name + '.npy'], mode='w+', dtype=dtype, shape=shape)
              for name, (dtype, shape) in specs.items()}
    arrays['offsets'][0] = 0
    row_at = candidate_at = 0
    sums = np.zeros(4, dtype=np.float64)
    with paths['predictions.txt'].open('x', encoding='utf-8', newline='\n') as predictions, \
            paths['truth.txt'].open('x', encoding='utf-8', newline='\n') as truth, \
            paths['records.jsonl'].open('x', encoding='utf-8', newline='\n') as metadata:
        for record, report in zip(records, reports):
            n, k = report['rows'], report['candidates']
            for name in ('scores', 'labels', 'news_indices'):
                source = np.load(report['files'][name + '.npy'], mmap_mode='r', allow_pickle=False)
                if source.shape != (k,):
                    raise ValueError('Chunk candidate-array shape mismatch')
                arrays[name][candidate_at:candidate_at + k] = source
                del source
            for name in ('user_indices', 'entry_metrics'):
                source = np.load(report['files'][name + '.npy'], mmap_mode='r', allow_pickle=False)
                expected = (n, 4) if name == 'entry_metrics' else (n,)
                if source.shape != expected or (name == 'entry_metrics' and not np.isfinite(source).all()):
                    raise ValueError('Chunk row-array shape or finite-value mismatch')
                arrays[name][row_at:row_at + n] = source
                if name == 'entry_metrics':
                    sums += source.sum(axis=0, dtype=np.float64)
                del source
            offsets = np.load(report['files']['offsets.npy'], mmap_mode='r', allow_pickle=False)
            if len(offsets) != n + 1 or offsets[0] != 0 or offsets[-1] != k or (np.diff(offsets) <= 0).any():
                raise ValueError('Invalid chunk offsets')
            arrays['offsets'][row_at + 1:row_at + n + 1] = offsets[1:] + candidate_at
            del offsets
            for filename, stream in (('predictions.txt', predictions), ('truth.txt', truth)):
                lines = 0
                with Path(report['files'][filename]).open(encoding='utf-8') as source:
                    for lines, line in enumerate(source, 1):
                        local_id, remainder = line.rstrip('\r\n').split(' ', 1)
                        if int(local_id) != lines:
                            raise ValueError('Chunk prediction/truth ordering mismatch')
                        stream.write(str(row_at + lines) + ' ' + remainder + '\n')
                if lines != n:
                    raise ValueError('Chunk prediction/truth count mismatch')
            with Path(report['files']['records.jsonl']).open(encoding='utf-8') as source:
                lines = 0
                for lines, line in enumerate(source, 1):
                    metadata.write(line)
                if lines != n:
                    raise ValueError('Chunk record count mismatch')
            row_at += n
            candidate_at += k
    for value in arrays.values():
        value.flush()
    del arrays
    report = {'protocol': 'total', 'metrics': dict(zip(METRICS, (sums / rows).tolist())),
              'metric_order': list(METRICS), 'rows': rows, 'candidates': count,
              'input': dict(candidate['output']), 'cache_identity': cache_identity,
              'elapsed_seconds': sum(item['elapsed_seconds'] for item in reports),
              'chunk_count': len(reports), 'aggregation': 'entry-weighted raw per-entry metrics',
              'tie_policy': 'stable_original_candidate_order',
              'files': {name: str(path.resolve()) for name, path in paths.items()},
              'scope': 'Full Total evaluation of the selected checkpoint on the declared input'}
    # Aggregate completed chunk timers; never subtract their rests twice.
    if all('cooling_rest_seconds' in item and 'net_elapsed_seconds' in item for item in reports):
        report['cooling_rest_seconds'] = sum(item['cooling_rest_seconds'] for item in reports)
        report['net_elapsed_seconds'] = sum(item['net_elapsed_seconds'] for item in reports)
    save_json(paths['metrics.json'], report)
    return report


def run(args):
    beginning, begin_unix = time.monotonic(), time.time()
    original_run, output = args.original_run.resolve(), args.output.resolve()
    if output == original_run or original_run in output.parents or output in original_run.parents:
        raise ValueError('Output must be separate from the original run')
    original, candidate, evidence, original_cache = validate_original(original_run, args.behaviors, args.candidate_manifest)
    if args.check_only:
        print(json.dumps({'status': 'validated', 'model': 'DIGAT', 'train_type': original['train_type'],
                          'protocol': 'total', 'rows': candidate['statistics']['rows'],
                          'candidates': candidate['statistics']['candidates'], 'training_required': False}))
        return
    target = output / 'experiment.json'
    if args.resume:
        manifest = read_json(target)
        if manifest.get('status') == 'completed':
            # An explicit resume of completed work is a no-op, not another evaluation.
            verify_file(manifest['merged_report']['path'], manifest['merged_report'])
            print(json.dumps({'status': 'already_completed', 'manifest': str(target)}))
            return
        if manifest['original_run'] != str(original_run) or manifest['chunk_rows'] != args.chunk_rows:
            raise ValueError('Resume original run or chunk size changed')
        if manifest['evidence'] != evidence or manifest['runner_sha256'] != fingerprint(__file__)['sha256']:
            raise ValueError('Resume source/input/checkpoint evidence changed')
    else:
        output.mkdir(parents=True, exist_ok=False)
        manifest = {'schema': 1, 'status': 'preparing', 'model': 'DIGAT',
                    'train_type': original['train_type'], 'seed': original['seed'],
                    'protocol': 'total', 'original_run': str(original_run), 'data_root': original['data_root'],
                    'candidate_manifest': str(args.candidate_manifest.resolve()), 'evidence': evidence,
                    'runner_sha256': fingerprint(__file__)['sha256'], 'chunk_rows': args.chunk_rows,
                    'training_reused': True, 'training_seconds_new': 0.0, 'evaluations': {},
                    'chunks': [], 'execution_segments': [], 'timing_intervals': [],
                    'selected_evaluation': copy.deepcopy(original['selected_evaluation'])}
        save_json(target, manifest)
        shutil.copytree(original_cache, output / 'embedding_cache' / original_cache.name)
        manifest['cache_copy'] = {'original': str(original_cache),
                                 'new_root': str(output / 'embedding_cache'),
                                 'file_bytes': sum(path.stat().st_size for path in original_cache.rglob('*') if path.is_file())}
        save_json(target, manifest)
    segment = {'start_unix': begin_unix, 'status': 'running'}
    manifest['execution_segments'].append(segment)
    def record():
        segment['elapsed_seconds'] = time.monotonic() - beginning
        manifest['elapsed_seconds'] = sum(item['elapsed_seconds'] for item in manifest['execution_segments'])
        save_json(target, manifest)
    previous_cwd = Path.cwd()
    try:
        # Import only the verified frozen computation; never mutate its .pyc files.
        sys.dont_write_bytecode = True
        snapshot = original_run / 'source_snapshot'
        sys.path[:0] = [str(snapshot / 'scripts'), str(snapshot / 'third_party/DIGAT')]
        import numpy as np
        import torch
        import MIND_corpus
        from model import Model
        from digat_cached_eval import DigatCachedEvaluator, news_categories
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required')
        if torch.__version__ != original['environment']['torch']:
            raise ValueError('Original torch version required for score-cache identity')
        torch.set_num_threads(4)
        torch.cuda.set_device(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        random.seed(original['seed']); np.random.seed(original['seed'])
        torch.manual_seed(original['seed']); torch.cuda.manual_seed_all(original['seed'])
        cfg = SimpleNamespace(**copy.deepcopy(original['effective_config']))
        cfg.attribute_dict = copy.deepcopy(original['effective_config'])
        selected_train = Path(cfg.train_root) / ('behaviors.tsv' if original['train_type'] == 'imp' else 'behaviors_rs50.tsv')
        default_train = Path(cfg.train_root) / 'behaviors.tsv'
        def selected_open(file, *positional, **keywords):
            if isinstance(file, (str, os.PathLike)) and Path(file).resolve() == default_train.resolve():
                file = selected_train
            return builtins.open(file, *positional, **keywords)
        MIND_corpus.open = selected_open
        # __init__ reads existing exact preprocessing; preprocess() is deliberately not called.
        os.chdir(original_run)
        corpus = MIND_corpus.MIND_Corpus(cfg)
        model = Model(cfg).cuda()
        checkpoint = torch.load(original_run / 'best_model.pt', map_location='cpu')
        if checkpoint['config'] != original['effective_config'] or checkpoint['epoch'] != original['best_epoch']:
            raise ValueError('Checkpoint config/epoch differs from original completed run')
        if checkpoint['preprocessing_identity'] != original['shared_preprocessing']['identity']:
            raise ValueError('Checkpoint preprocessing identity differs')
        model.load_state_dict(checkpoint[model.model_name], strict=True)
        del checkpoint
        categories = news_categories(corpus, Path(original['data_root']))
        selected = original['selected_evaluation']
        evaluator = DigatCachedEvaluator(model, corpus, categories, output / 'embedding_cache',
                       news_batch_size=256, score_batch_size=selected['batch_size'],
                       projection_cache=selected['projection_cache'], sparse_eval=selected['graph_backend'] == 'sparse_csr')
        if evaluator.identity != original_cache.name or not evaluator.cache_loaded:
            raise ValueError('Expected exact original checkpoint-bound copied cache')
        manifest['cache_identity'] = evaluator.identity
        manifest['status'] = 'verifying'; record()
        gate = evaluator.verify_against_forward(Path(original['data_root']) / 'dev/behaviors.tsv')
        if gate.get('status') != 'passed' or gate.get('rows_with_rank_disagreement') != 0:
            raise ValueError('Reloaded checkpoint native rank gate failed')
        save_json(output / 'cache_equivalence.json', gate)
        manifest['verification'] = fingerprint(output / 'cache_equivalence.json')
        # Corpus training/test lists are unused by the cache after the native dev gate.
        for name in tuple(vars(corpus)):
            if name.startswith(('train_', 'test_', 'dev_')):
                delattr(corpus, name)
        gc.collect()
        manifest['status'] = 'evaluating'; record()
        from evaluation_progress import evaluate_test_file
        from experiment_progress import test_progress, finish_progress
        total_rows = candidate['statistics']['rows']
        done = {item['index']: item for item in manifest['chunks']}
        for shard in shard_input(args.behaviors, output / 'input_chunks', args.chunk_rows):
            index = shard['index']
            if index in done:
                if any(done[index][key] != shard[key] for key in ('start_row', 'rows', 'input')):
                    raise ValueError('Resume shard identity differs')
                validated_chunk(done[index])
                test_progress('total', shard['start_row'] + shard['rows'], total_rows)
                continue
            attempt = output / 'chunk_results' / ('chunk_%05d_attempt_%d' % (index, time.time_ns()))
            attempt.mkdir(parents=True, exist_ok=False)
            start_unix, start = time.time(), time.monotonic()
            result = copy.deepcopy(evaluate_test_file(evaluator, Path(shard['input']['path']), attempt / 'test_total',
                                                     'total', start_row=shard['start_row'], total_rows=total_rows))
            result['protocol'] = 'total_chunk'
            save_json(result['files']['metrics.json'], result)
            record_chunk = {**shard, 'report': fingerprint(result['files']['metrics.json']),
                            'artifacts': {name: fingerprint(result['files'][name]) for name in ARTIFACTS}}
            manifest['chunks'].append(record_chunk)
            manifest['timing_intervals'].append({'phase': 'test', 'protocol': 'total', 'chunk': index,
                'start_unix': start_unix, 'end_unix': time.time(), 'seconds': time.monotonic() - start})
            record()
            print(json.dumps({'stage': 'total_chunk_completed', 'chunk': index,
                              'rows_completed': shard['start_row'] + shard['rows'],
                              'rows_total': candidate['statistics']['rows'],
                              'seconds': result['elapsed_seconds']}), flush=True)
        manifest['chunks'].sort(key=lambda item: item['index'])
        manifest['status'] = 'merging'; record()
        merge_dir = output / ('merged_' + str(time.time_ns()))
        result = merge_chunks(manifest['chunks'], args.behaviors, merge_dir / 'test_total', candidate, evaluator.identity)
        finish_progress()
        manifest['evaluations']['total'] = {'prefix': str(merge_dir / 'test_total'), **result}
        manifest['merged_report'] = fingerprint(result['files']['metrics.json'])
        manifest['timings_seconds'] = {'training_new': 0.0,
                                      'test_total': sum(item['seconds'] for item in manifest['timing_intervals'])}
        manifest['status'] = 'completed'; segment['status'] = 'completed'; segment['end_unix'] = time.time(); record()
        print(json.dumps({'stage': 'completed', 'manifest': str(target), 'metrics': result['metrics']}), flush=True)
    except BaseException:
        manifest['status'] = 'failed'; manifest['error'] = traceback.format_exc()
        segment['status'] = 'failed'; segment['end_unix'] = time.time(); record()
        raise
    finally:
        os.chdir(previous_cwd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-run', required=True, type=Path)
    parser.add_argument('--behaviors', required=True, type=Path)
    parser.add_argument('--candidate-manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--chunk-rows', type=int, default=5000)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.chunk_rows < 1:
        parser.error('--chunk-rows must be positive')
    run(args)


if __name__ == '__main__':
    main()
