"""Evaluate full Total with an existing completed NNR checkpoint; never train.

Original runs, completed scores and caches stay immutable. This wrapper verifies
and copies the trained preprocessing, checkpoint and current embedding cache,
then imports the original run's frozen inference implementation. Its outputs are
new measurements, not the paper's reported reference values.
"""
import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
MODELS = ('NAML', 'NRMS', 'LSTUR', 'CNE-SUE', 'NPA')
REQUIRED_CHECKS = (
    'request_order_matches_existing_evaluation_mapping',
    'positive_labels_match_existing_evaluation_mapping',
    'total5_subset_of_total', 'rands30_subset_of_total',
    'all_news_in_model_vocabularies',
)


def load_common():
    # An alias avoids colliding with the old frozen module inventory.
    spec = importlib.util.spec_from_file_location(
        '_tors_existing_eval_common', ROOT / 'scripts' / 'resume_small_evaluation.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COMMON = load_common()
fingerprint = COMMON.fingerprint
read_json = COMMON.read_json
save_json = COMMON.save_json
within = COMMON.within
verify_file = COMMON.verify_file
copy_verified = COMMON.copy_verified


def validate_candidates(path, manifest_path, old):
    """Bind the new file to the separately exhaustively validated preparation."""
    path, manifest_path = Path(path).resolve(), Path(manifest_path).resolve()
    spec = read_json(manifest_path)
    if spec.get('status') != 'complete' or spec.get('protocol') != 'total':
        raise ValueError('A completed full Total candidate manifest is required')
    if Path(spec['dataset_root']).resolve() != Path(old['data_root']).resolve():
        raise ValueError('Total candidates belong to a different dataset')
    if any(spec.get('validation', {}).get(name) is not True for name in REQUIRED_CHECKS):
        raise ValueError('Full Total identity/subset/vocabulary checks did not all pass')
    if Path(spec['output']['path']).resolve() != path:
        raise ValueError('Candidate file is not the file certified by its manifest')
    actual = verify_file(path, spec['output'])
    dataset_path = Path(old['data_root']) / 'manifest.json'
    dataset = verify_file(dataset_path, spec['dataset_manifest'])
    stats = spec['statistics']
    if stats['rows'] != old['evaluations']['rs30']['rows']:
        raise ValueError('Full Total must match the positive-expanded RandS request count')
    if stats['candidates'] <= old['evaluations']['total5']['candidates']:
        raise ValueError('Full Total did not expand the prior Total5 candidate set')
    if stats['positive_labels'] != stats['rows']:
        raise ValueError('Expected one positive per expanded Total request')
    prior_inputs = spec.get('source_existing_evaluations', {})
    for name in ('rs30', 'total5'):
        if prior_inputs.get(name, {}).get('sha256') != old['evaluations'][name]['input']['sha256']:
            raise ValueError('New candidates were not aligned against the original ' + name + ' input')
    return {'manifest': fingerprint(manifest_path), 'file': actual,
            'dataset_manifest': dataset, 'statistics': stats,
            'validation': copy.deepcopy(spec['validation'])}


def validate_original(original):
    """CPU-only artifact identity checks; no torch import or model execution."""
    original = Path(original).resolve()
    old = read_json(original / 'experiment.json')
    if old.get('status') != 'completed' or old.get('model') not in MODELS:
        raise ValueError('A completed supported NNR run is required')
    if old.get('train_type') not in ('imp', 'rs50') or old['effective_config']['dataset'] != 'small':
        raise ValueError('Expected a completed MIND-Small IMP/RandS50 run')
    if 'total' in old.get('evaluations', {}):
        raise ValueError('Full Total is already recorded; do not repeat it')
    if not {'imp', 'rs30', 'total5'}.issubset(old.get('evaluations', {})):
        raise ValueError('Reusable IMP/RandS/Total5 evidence is incomplete')
    source = old['source_sha256']
    if any(name not in source for name in COMMON.REQUIRED_RUNTIME):
        raise ValueError('Frozen runtime source inventory is incomplete')
    evidence = {'original_manifest': fingerprint(original / 'experiment.json')}
    evidence['source_files'] = {
        name: verify_file(within(original / 'source_snapshot', name), {'sha256': digest})
        for name, digest in source.items()}
    evidence['checkpoint'] = verify_file(original / 'best_model.pt', old['checkpoint'])
    evidence['preprocess_files'] = {
        name: verify_file(within(original, name), expected)
        for name, expected in old['preprocess_files'].items()}
    config = old['effective_config']
    prep_spec = {'input_hashes': [row['sha256'] for row in old['inputs']['files']],
        'glove': old['inputs']['glove']['sha256'], 'seed': old['seed'],
        'parameters': {key: config[key] for key in (
            'dataset', 'tokenizer', 'word_threshold', 'word_embedding_dim',
            'max_title_length', 'max_abstract_length', 'max_history_num',
            'no_self_connection', 'no_adjacent_normalization', 'gcn_normalization_type')},
        'corpus_source_sha256': source['NNR-main/MIND_corpus.py']}
    if hashlib.sha256(json.dumps(prep_spec, sort_keys=True).encode()).hexdigest() != old['preprocess_signature']:
        raise ValueError('Preprocessing signature differs from trained configuration')
    evidence['input_files'] = [verify_file(row['path'], row) for row in old['inputs']['files']]
    equivalence_path = original / ('cache_equivalence_epoch%d.json' % old['best_epoch'])
    equivalence = read_json(equivalence_path)
    if equivalence.get('status') != 'passed':
        raise ValueError('Original trained-checkpoint cache equivalence did not pass')
    evidence['original_equivalence'] = fingerprint(equivalence_path)
    evidence['original_equivalence_result'] = equivalence
    evidence['evaluation_reports'] = {}
    for protocol, result in old['evaluations'].items():
        prefix = Path(result['prefix']).resolve()
        if original not in prefix.parents:
            raise ValueError('Existing evaluation prefix escaped original run')
        report_path = Path(str(prefix) + '.metrics.json')
        report = read_json(report_path)
        if {k: v for k, v in result.items() if k != 'prefix'} != report:
            raise ValueError('Original metric report differs from manifest: ' + protocol)
        evidence['evaluation_reports'][protocol] = fingerprint(report_path)
    cache = Path(old['evaluations']['total5']['cache_dir']).resolve()
    if original not in cache.parents:
        raise ValueError('The completed cache must be owned by the original run')
    metadata = read_json(cache / 'manifest.json')
    if metadata['identity'] != equivalence['cache_identity'] or metadata['identity'] != old['evaluations']['total5']['cache_identity']:
        raise ValueError('Checkpoint/cache identity mismatch')
    if metadata['provenance']['model_state_sha256'] != equivalence['model_state_sha256']:
        raise ValueError('Cache model state differs from verified checkpoint state')
    evidence['cache_manifest'] = fingerprint(cache / 'manifest.json')
    evidence['cache_metadata'] = metadata
    evidence['cache_files'] = {
        name: verify_file(within(cache, name), {'sha256': digest})
        for name, digest in metadata['files'].items()}
    return old, evidence


def prepare_run(original, output, evidence):
    original, output = Path(original).resolve(), Path(output).resolve()
    if output.exists() or output == original or original in output.parents or output in original.parents:
        raise ValueError('Use a new separate output directory; preserve existing work')
    output.mkdir(parents=True)
    copies = {'source_files': {}, 'preprocess_files': {}, 'cache_files': {}}
    for name, expected in evidence['source_files'].items():
        copies['source_files'][name] = copy_verified(expected['path'], within(output / 'source_snapshot', name), expected)
    for name, expected in evidence['preprocess_files'].items():
        copies['preprocess_files'][name] = copy_verified(expected['path'], within(output, name), expected)
    copies['checkpoint'] = copy_verified(evidence['checkpoint']['path'], output / 'best_model.pt', evidence['checkpoint'])
    copy_verified(original / 'experiment.json', output / 'original_experiment.json', evidence['original_manifest'])
    cache_root = output / 'embedding_cache' / evidence['cache_metadata']['identity']
    for name, expected in evidence['cache_files'].items():
        copies['cache_files'][name] = copy_verified(expected['path'], within(cache_root, name), expected)
    copies['cache_manifest'] = copy_verified(evidence['cache_manifest']['path'], cache_root / 'manifest.json', evidence['cache_manifest'])
    return copies


def evaluate_chunks(evaluator, args, output, manifest, record):
    # This shared merger is standard-library-only until it actually merges.
    from evaluate_existing_digat import shard_input, validated_chunk, merge_chunks, ARTIFACTS
    from evaluation_progress import evaluate_test_file
    from experiment_progress import test_progress, finish_progress
    total_rows = manifest['candidate_validation']['statistics']['rows']
    completed = {item['index']: item for item in manifest['chunks']}
    last_print = time.monotonic()
    for shard in shard_input(args.behaviors, output / 'candidate_chunks', args.chunk_rows):
        index = shard['index']
        if index in completed:
            previous = completed[index]
            if any(previous[key] != shard[key] for key in ('index', 'start_row', 'rows', 'input')):
                raise ValueError('Completed chunk no longer matches full Total request order')
            validated_chunk(previous)
            test_progress('total', shard['start_row'] + shard['rows'], total_rows)
            continue
        directory = output / 'chunk_results'
        directory.mkdir(exist_ok=True)
        attempt = 1
        while True:
            prefix = directory / ('chunk_%05d_attempt_%03d' % (index, attempt))
            if not list(directory.glob(prefix.name + '.*')):
                break
            attempt += 1
        report = evaluate_test_file(evaluator, shard['input']['path'], prefix, 'total',
                                    start_row=shard['start_row'], total_rows=total_rows)
        chunk = dict(shard, report=fingerprint(str(prefix) + '.metrics.json'),
                     artifacts={name: fingerprint(report['files'][name]) for name in ARTIFACTS})
        validated_chunk(chunk)
        manifest['chunks'].append(chunk)
        manifest['current_chunk'] = index
        record()
        if time.monotonic() - last_print >= 300:
            print(json.dumps({'stage': 'total_evaluation', 'model': manifest['model'],
                'training': manifest['train_type'], 'completed_rows': sum(x['rows'] for x in manifest['chunks']),
                'total_rows': manifest['candidate_validation']['statistics']['rows']}), flush=True)
            last_print = time.monotonic()
        gc.collect()
    candidate_spec = read_json(args.candidate_manifest)
    prefix = output / 'evaluation' / 'test_total'
    attempt = 1
    while list(prefix.parent.glob(prefix.name + '.*')):
        attempt += 1
        prefix = output / 'evaluation' / ('test_total_attempt_%03d' % attempt)
    ordered = sorted(manifest['chunks'], key=lambda row: row['index'])
    if [row['index'] for row in ordered] != list(range(len(ordered))):
        raise ValueError('Missing or duplicate completed chunk index')
    merged = merge_chunks(ordered, args.behaviors, prefix, candidate_spec, evaluator.identity)
    finish_progress()
    return merged, prefix


def main():
    started, started_unix = time.monotonic(), time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-run-dir', type=Path, required=True)
    parser.add_argument('--behaviors', type=Path, required=True)
    parser.add_argument('--candidate-manifest', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--validation-output', type=Path)
    parser.add_argument('--chunk-rows', type=int, default=5000)
    parser.add_argument('--resume', action='store_true', help='Verify and reuse completed full-Total chunks')
    args = parser.parse_args()
    if args.chunk_rows <= 0:
        raise ValueError('Chunk rows must be positive')
    original = args.original_run_dir.resolve()
    old, evidence = validate_original(original)
    candidate = validate_candidates(args.behaviors, args.candidate_manifest, old)
    evidence['candidate_validation'] = candidate
    if args.check_only:
        if args.validation_output:
            destination = args.validation_output.resolve()
            if destination.exists() or original in destination.parents:
                raise ValueError('Use a new validation output outside the original run')
            save_json(destination, evidence)
        print(json.dumps({'status': 'validated', 'model': old['model'],
            'train_type': old['train_type'], 'protocol': 'total',
            'rows': candidate['statistics']['rows'], 'candidates': candidate['statistics']['candidates'],
            'training_performed': False}), flush=True)
        return
    if not args.run_dir:
        raise ValueError('--run-dir is required unless --check-only')
    output = args.run_dir.resolve()
    if output == original or original in output.parents or output in original.parents:
        raise ValueError('Output must be separate from original run')
    manifest_path = output / 'experiment.json'
    if args.resume:
        manifest = read_json(manifest_path)
        if manifest['original_run'] != str(original) or manifest['candidate_validation'] != candidate:
            raise ValueError('Resume original run or candidate identity changed')
        if manifest['chunk_rows'] != args.chunk_rows or manifest['runner']['sha256'] != fingerprint(__file__)['sha256']:
            raise ValueError('Resume runner or chunking settings changed')
        if manifest['merger']['sha256'] != fingerprint(ROOT / 'scripts/evaluate_existing_digat.py')['sha256']:
            raise ValueError('Resume shared merger source changed')
        if manifest['common_guard']['sha256'] != fingerprint(ROOT / 'scripts/resume_small_evaluation.py')['sha256']:
            raise ValueError('Resume source validation helper changed')
        prior_validation = read_json(output / 'input_validation.json')
        copies = prior_validation.pop('verified_copies')
        if prior_validation != evidence:
            raise ValueError('Resume original evidence changed')
        if manifest['status'] == 'completed':
            verify_file(manifest['merged_report']['path'], manifest['merged_report'])
            print(json.dumps({'status': 'already_completed', 'manifest': str(manifest_path)}), flush=True)
            return
        for group in ('source_files', 'preprocess_files'):
            for expected in copies[group].values():
                verify_file(expected['path'], expected)
        verify_file(copies['checkpoint']['path'], copies['checkpoint'])
    else:
        copies = prepare_run(original, output, evidence)
        save_json(output / 'input_validation.json', dict(evidence, verified_copies=copies))
        manifest = {'schema': 1, 'status': 'loading', 'model': old['model'],
        'train_type': old['train_type'], 'seed': old['seed'], 'best_epoch': old['best_epoch'],
        'original_run': str(original), 'training_reused': True, 'training_performed': False,
        'original_manifest': evidence['original_manifest'], 'checkpoint': copies['checkpoint'],
        'data_root': old['data_root'], 'candidate_validation': candidate,
        'source_sha256': old['source_sha256'], 'effective_config': old['effective_config'],
        'evaluation_origins': {kind: {'prefix': row['prefix'], 'reused_without_scoring': True}
                               for kind, row in old['evaluations'].items()},
        'evaluations': copy.deepcopy(old['evaluations']), 'timing_intervals': [],
        'process_started_unix': started_unix, 'source_snapshot': str(output / 'source_snapshot'),
        'embedding_cache_reused': True, 'runner': fingerprint(__file__),
        'common_guard': fingerprint(ROOT / 'scripts' / 'resume_small_evaluation.py'),
        'merger': fingerprint(ROOT / 'scripts' / 'evaluate_existing_digat.py'),
        'chunk_rows': args.chunk_rows, 'chunks': [], 'execution_segments': []}
    segment = {'start_unix': started_unix, 'status': 'running', 'elapsed_seconds': 0.0}
    manifest['execution_segments'].append(segment)
    def record():
        segment['elapsed_seconds'] = time.monotonic() - started
        manifest['elapsed_seconds'] = sum(row['elapsed_seconds'] for row in manifest['execution_segments'])
        save_json(manifest_path, manifest)
    record()
    try:
        os.chdir(output)
        modules, imported = COMMON.activate_frozen_runtime(output / 'source_snapshot', old['source_sha256'])
        manifest['loaded_runtime_sources'] = imported
        import numpy as np
        import torch
        torch.set_num_threads(4)
        if str(torch.__version__) != old['environment']['torch']:
            raise ValueError('PyTorch version differs from original run')
        config = SimpleNamespace(**copy.deepcopy(old['effective_config']))
        torch.cuda.set_device(config.device_id)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        precision = evidence['cache_metadata']['provenance'].get('evaluation_precision')
        if precision is None:
            # Original NAML cache schema omitted precision; recorded PyTorch 1.12
            # defaults plus a new bounded native-forward equivalence are retained.
            precision = {'cuda_matmul_allow_tf32': False, 'cudnn_allow_tf32': True,
                         'basis': 'PyTorch 1.12 defaults; original NAML cache did not record flags'}
        torch.backends.cuda.matmul.allow_tf32 = precision['cuda_matmul_allow_tf32']
        torch.backends.cudnn.allow_tf32 = precision['cudnn_allow_tf32']
        manifest['evaluation_precision'] = precision
        random.seed(config.seed); np.random.seed(config.seed); torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        corpus_start = time.monotonic()
        corpus = modules['MIND_corpus'].MIND_Corpus(config)
        manifest['corpus_load_seconds'] = time.monotonic() - corpus_start
        if vars(config) != old['effective_config']:
            raise ValueError('Loading corpus changed trained effective configuration')
        for name, expected in copies['preprocess_files'].items():
            verify_file(output / name, expected)
        checkpoint = torch.load(output / 'best_model.pt', map_location='cpu')
        if checkpoint['epoch'] != old['best_epoch'] or checkpoint['config'] != config.attribute_dict:
            raise ValueError('Checkpoint epoch/configuration mismatch')
        model = modules['model'].Model(config).cuda()
        adjustments = modules['nnr_batch_alignment'].apply_batch_alignment_fixes(model) if old['model'] in ('CNE-SUE', 'NPA') else []
        if adjustments != old.get('implementation_adjustments', []):
            raise ValueError('Inference implementation adjustments changed')
        model.load_state_dict(checkpoint[model.model_name], strict=True)
        del checkpoint
        resources = old.get('evaluation_resources_requested', {})
        cache_start = time.monotonic()
        if old['model'] == 'NAML':
            evaluator = sys.modules['naml_cached_eval'].NamlCachedEvaluator(
                model, corpus, output / 'embedding_cache', news_batch_size=128, user_batch_size=256)
        else:
            evaluator = modules['nnr_cached_eval'].NNRCachedEvaluator(
                model, corpus, config, output / 'embedding_cache',
                personalized_batch_size=resources.get('personalized_batch_size', 512),
                npa_features_on_gpu=resources.get('npa_features_on_gpu', False))
        cache_seconds = time.monotonic() - cache_start
        if evaluator.identity != evidence['cache_metadata']['identity'] or not evaluator.cache_loaded:
            raise ValueError('Existing checkpoint-bound embeddings were not reused')
        manifest['cache_identity'] = evaluator.identity
        manifest['status'] = 'verifying'; record()
        verification = evaluator.verify_against_forward(args.behaviors, max_entries=8, max_candidates=32)
        if verification['status'] != 'passed':
            raise ValueError('Native inference and copied cache are not equivalent')
        save_json(output / 'cache_equivalence_total.json', verification)
        manifest['native_verification'] = verification
        print(json.dumps({'stage': 'total_evaluation_ready', 'model': old['model'],
            'training': old['train_type'], 'cache_reused': True}), flush=True)
        manifest['status'] = 'evaluating'; record()
        begin, begin_unix = time.monotonic(), time.time()
        result, prefix = evaluate_chunks(evaluator, args, output, manifest, record)
        evaluation_seconds = time.monotonic() - begin
        if result['rows'] != candidate['statistics']['rows'] or result['candidates'] != candidate['statistics']['candidates']:
            raise ValueError('Evaluated Total size differs from certified candidates')
        if result['input']['sha256'] != candidate['file']['sha256']:
            raise ValueError('Total candidates changed during evaluation')
        manifest['evaluations']['total'] = dict(result, prefix=str(prefix))
        manifest['merged_report'] = fingerprint(str(prefix) + '.metrics.json')
        manifest['evaluation_origins']['total'] = {'prefix': str(prefix), 'reused_without_scoring': False}
        manifest['timing_intervals'].append({'phase': 'test', 'protocol': 'total',
            'start_unix': begin_unix, 'end_unix': time.time(), 'duration_seconds': evaluation_seconds})
        for expected in (evidence['original_manifest'], evidence['checkpoint'], evidence['cache_manifest']):
            verify_file(expected['path'], expected)
        for expected in evidence['evaluation_reports'].values():
            verify_file(expected['path'], expected)
        for expected in evidence['cache_files'].values():
            verify_file(expected['path'], expected)
        verify_file(candidate['manifest']['path'], candidate['manifest'])
        manifest['original_artifacts_unchanged'] = True
        segment['status'] = 'completed'
        segment['evaluation_seconds'] = evaluation_seconds
        segment['cache_setup_seconds'] = cache_seconds
        segment['setup_seconds'] = begin - started
        segment['elapsed_seconds'] = time.monotonic() - started
        manifest['timings_seconds'] = {'training': 0.0,
            'training_reused': sum(row['train_seconds'] for row in old['epoch_records']),
            'evaluation': result['elapsed_seconds'],
            'evaluation_orchestration': sum(row.get('evaluation_seconds', 0) for row in manifest['execution_segments']),
            'cache_setup': sum(row.get('cache_setup_seconds', 0) for row in manifest['execution_segments']),
            'setup': sum(row.get('setup_seconds', 0) for row in manifest['execution_segments']),
            'total': sum(row['elapsed_seconds'] for row in manifest['execution_segments'])}
        manifest['timing_scope'] = {
            'total': 'Sum of recorded new runner execution segments; excludes inter-run/resume gaps. A hard-killed final segment may be only partially measured.',
            'evaluation': 'Sum of completed shard evaluate_file timers, including shards reused after an interruption.',
            'evaluation_orchestration': 'Recorded new evaluation loops plus merge/hash work; includes measured failed attempts.',
            'training': 'Zero new training; the existing checkpoint is reused.',
            'training_reused': 'Raw original epoch timers only. Use the original pause-adjusted timing audit for user-facing historical training time.'}
        manifest['status'] = 'completed'; record()
        save_json(output / 'run_index_entry.json', {'model': old['model'], 'train_type': old['train_type'],
            'run_dir': str(output), 'evaluations': {kind: row['prefix'] for kind, row in manifest['evaluations'].items()}})
        print(json.dumps({'stage': 'completed', 'model': old['model'], 'training': old['train_type'],
            'protocol': 'total', 'metrics': result['metrics'], 'evaluation_seconds': evaluation_seconds,
            'manifest': str(manifest_path), 'training_performed': False}), flush=True)
    except BaseException:
        segment['status'] = 'failed'
        if 'begin' in locals():
            segment['evaluation_seconds'] = time.monotonic() - begin
        if 'cache_seconds' in locals():
            segment['cache_setup_seconds'] = cache_seconds
        if 'begin' in locals():
            segment['setup_seconds'] = begin - started
        manifest['status'] = 'failed'; manifest['error'] = traceback.format_exc(); record()
        raise


if __name__ == '__main__':
    main()
