"""Verified artifact copying and frozen-runtime activation for NNR evaluation."""
import argparse
import copy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback
from types import SimpleNamespace


PROTOCOL_FILES = {'imp': 'behaviors.tsv', 'rs30': 'behaviors_rs30.tsv',
                  'rs90': 'behaviors_rs90.tsv', 'rs150': 'behaviors_rs150.tsv',
                  'total5': 'behaviors_total5.tsv'}
REQUIRED_RUNTIME = ('NNR-main/config.py', 'NNR-main/MIND_corpus.py',
    'NNR-main/model.py', 'NNR-main/newsEncoders.py', 'NNR-main/userEncoders.py',
    'NNR-main/variantEncoders.py', 'NNR-main/layers.py',
    'scripts/naml_cached_eval.py', 'scripts/nnr_cached_eval.py',
    'scripts/nnr_batch_alignment.py')
ARTIFACT_SUFFIXES = ('metrics.json', 'scores.npy', 'labels.npy', 'offsets.npy',
    'news_indices.npy', 'user_indices.npy', 'entry_metrics.npy', 'records.jsonl',
    'predictions.txt', 'truth.txt')


def fingerprint(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def within(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if path == root or root not in path.parents:
        raise ValueError('Path is outside the required root: ' + str(relative))
    return path


def verify_file(path, expected):
    actual = fingerprint(path)
    if actual['sha256'] != expected['sha256'] or (
            'bytes' in expected and actual['bytes'] != expected['bytes']):
        raise ValueError('Artifact hash/size mismatch: ' + str(path))
    return actual


def original_segment(original, manifest):
    intervals = manifest.get('timing_intervals', [])
    complete = [x for x in intervals if x.get('phase') == 'test'
                and x.get('protocol') in manifest['evaluations']]
    elapsed = manifest.get('elapsed_seconds')
    if not complete or not isinstance(elapsed, (float, int)) or not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError('Original completed-work timing evidence is missing')
    end = max(float(x['end_unix']) for x in complete)
    return {'segment_id': 'original_completed_work', 'job_name': Path(original).name,
        'start_unix': end - elapsed, 'end_unix': end, 'duration_seconds': elapsed,
        'start_boundary_derived': True,
        'timing_basis': 'Original measured manifest elapsed; endpoint is last completed protocol boundary',
        'coverage': 'Partial original process; excludes upfront source/input work and unmeasured interrupted evaluation effort',
        'original_runner_entry_unix': manifest.get('process_started_unix'),
        'source_manifest': str(Path(original) / 'experiment.json')}


def validate_original(original, protocols=('total5',)):
    """Standard-library-only, read-only validation; never imports torch/CUDA."""
    original = Path(original).resolve()
    if not protocols or len(set(protocols)) != len(protocols) or any(x not in PROTOCOL_FILES for x in protocols):
        raise ValueError('Unique supported missing protocols are required')
    manifest_path = original / 'experiment.json'
    manifest = read_json(manifest_path)
    marker_path = original / 'INTERRUPTED_FOR_REBOOT.json'
    marker = read_json(marker_path)
    evidence = {'original_manifest': verify_file(manifest_path, {'sha256': marker['original_manifest_sha256']}),
                'interruption_marker': fingerprint(marker_path)}
    if manifest.get('status') != 'evaluating' or marker.get('status_before_pause') != 'evaluating':
        raise ValueError('Only an explicitly interrupted evaluation run can be recovered')
    if manifest.get('model') not in ('NAML', 'NRMS', 'LSTUR', 'CNE-SUE', 'NPA'):
        raise ValueError('Not a supported NNR run')
    completed = manifest.get('evaluations', {})
    if set(completed) != set(marker['completed_protocols']):
        raise ValueError('Completed protocols disagree with interruption marker')
    missing = set(PROTOCOL_FILES) - set(completed)
    if set(protocols) != missing:
        raise ValueError('Requested protocols must be exactly the missing set: ' + repr(sorted(missing)))
    source = manifest['source_sha256']
    if any(name not in source for name in REQUIRED_RUNTIME):
        raise ValueError('Frozen runtime source inventory is incomplete')
    snapshot = original / 'source_snapshot'
    evidence['source_files'] = {name: verify_file(within(snapshot, name), {'sha256': expected})
                               for name, expected in source.items()}
    evidence['checkpoint'] = verify_file(original / 'best_model.pt', manifest['checkpoint'])
    evidence['trained_epochs'] = [verify_file(within(original, Path(row['checkpoint']['path']).name), row['checkpoint'])
                                 for row in manifest['trained_epochs']]
    evidence['preprocess_files'] = {name: verify_file(within(original, name), expected)
                                    for name, expected in manifest['preprocess_files'].items()}
    config = manifest['effective_config']
    prep_spec = {'input_hashes': [x['sha256'] for x in manifest['inputs']['files']],
        'glove': manifest['inputs']['glove']['sha256'], 'seed': manifest['seed'],
        'parameters': {key: config[key] for key in ('dataset', 'tokenizer', 'word_threshold',
            'word_embedding_dim', 'max_title_length', 'max_abstract_length', 'max_history_num',
            'no_self_connection', 'no_adjacent_normalization', 'gcn_normalization_type')},
        'corpus_source_sha256': source['NNR-main/MIND_corpus.py']}
    signature = hashlib.sha256(json.dumps(prep_spec, sort_keys=True).encode()).hexdigest()
    if signature != manifest['preprocess_signature']:
        raise ValueError('Saved config/input/source preprocessing signature mismatch')
    evidence['input_files'] = [verify_file(row['path'], row) for row in manifest['inputs']['files']]
    data_root = Path(manifest['data_root']).resolve()
    dataset_manifest = read_json(data_root / 'manifest.json')
    evidence['dataset_manifest'] = fingerprint(data_root / 'manifest.json')
    if dataset_manifest.get('status') not in ('complete', 'completed'):
        raise ValueError('Dataset preparation is incomplete')
    outputs = dataset_manifest['output_files']
    evidence['evaluation_inputs'] = {}
    for protocol, filename in PROTOCOL_FILES.items():
        relative = 'test/' + filename
        expected = outputs[relative]
        evidence['evaluation_inputs'][protocol] = verify_file(within(data_root, relative), expected)
    evidence['completed_evaluations'] = {}
    for protocol, result in completed.items():
        prefix = original / 'evaluation' / ('test_' + protocol)
        if Path(result['prefix']).resolve() != prefix.resolve():
            raise ValueError('Original evaluation prefix is not inside the original run')
        report = read_json(str(prefix) + '.metrics.json')
        if {key: value for key, value in result.items() if key != 'prefix'} != report:
            raise ValueError('Saved evaluation report differs from original manifest: ' + protocol)
        if report['input']['sha256'] != evidence['evaluation_inputs'][protocol]['sha256']:
            raise ValueError('Completed evaluation input identity mismatch: ' + protocol)
        files = {}
        for suffix in ARTIFACT_SUFFIXES:
            path = Path(str(prefix) + '.' + suffix)
            if Path(report['files'][suffix]).resolve() != path.resolve():
                raise ValueError('Completed artifact path is not the expected original prefix')
            files[suffix] = fingerprint(path)
        evidence['completed_evaluations'][protocol] = files
    equivalence = read_json(original / ('cache_equivalence_epoch%d.json' % manifest['best_epoch']))
    if equivalence.get('status') != 'passed':
        raise ValueError('Original trained-checkpoint equivalence did not pass')
    evidence['original_equivalence'] = fingerprint(original / ('cache_equivalence_epoch%d.json' % manifest['best_epoch']))
    evidence['original_equivalence_result'] = equivalence
    evidence['original_segment'] = original_segment(original, manifest)
    return manifest, evidence


def copy_verified(source, target, expected):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise ValueError('Preserve existing output: ' + str(target))
    shutil.copyfile(source, target)
    return verify_file(target, expected)


def prepare_new_run(original, output, manifest, evidence):
    original, output = Path(original).resolve(), Path(output).resolve()
    if output == original or original in output.parents or output in original.parents or output.exists():
        raise ValueError('A new, separate run directory is required')
    output.mkdir(parents=True, exist_ok=False)
    copies = {'source_files': {}, 'preprocess_files': {}}
    for name, expected in evidence['source_files'].items():
        copies['source_files'][name] = copy_verified(expected['path'], within(output / 'source_snapshot', name), expected)
    for name, expected in evidence['preprocess_files'].items():
        copies['preprocess_files'][name] = copy_verified(expected['path'], within(output, name), expected)
    copies['checkpoint'] = copy_verified(evidence['checkpoint']['path'], output / 'best_model.pt', evidence['checkpoint'])
    copy_verified(original / 'experiment.json', output / 'original_experiment.json', evidence['original_manifest'])
    copy_verified(original / 'INTERRUPTED_FOR_REBOOT.json', output / 'original_interruption.json', evidence['interruption_marker'])
    return copies


def activate_frozen_runtime(snapshot, source_hashes):
    """Refuse preloaded colliding modules and verify every local runtime import."""
    snapshot = Path(snapshot).resolve()
    local_modules = {Path(name).stem for name in source_hashes if name.endswith('.py')}
    collisions = [name for name in local_modules if name in sys.modules]
    if collisions:
        raise ValueError('Runtime source modules already imported: ' + repr(collisions))
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(snapshot / 'scripts'), str(snapshot / 'NNR-main')]
    modules = {name: importlib.import_module(name) for name in
               ('MIND_corpus', 'model', 'nnr_batch_alignment', 'nnr_cached_eval')}
    imported = {}
    for name in local_modules:
        module = sys.modules.get(name)
        if module is None:
            continue
        path = Path(module.__file__).resolve()
        if snapshot not in path.parents:
            raise ValueError('Runtime module escaped frozen snapshot: ' + name)
        relative = path.relative_to(snapshot).as_posix()
        imported[name] = verify_file(path, {'sha256': source_hashes[relative]})
    return modules, imported


def combined_manifest(original, output, old, evidence, copies, started_unix):
    result = copy.deepcopy(old)
    for key in ('timings_seconds', 'timing_scope', 'elapsed_seconds', 'error'):
        result.pop(key, None)
    result.update(schema=2, status='preparing', process_started_unix=started_unix,
                  source_snapshot=str(Path(output) / 'source_snapshot'),
                  checkpoint=copies['checkpoint'], preprocess_files=copies['preprocess_files'])
    result['execution_segments'] = [evidence['original_segment']]
    for interval in result.get('timing_intervals', []):
        interval.update(segment_id='original_completed_work', job_name=Path(original).name)
    result['recovery'] = {'mode': 'evaluation_only', 'training_performed': False,
        'original_run_dir': str(original), 'original_status': old['status'],
        'original_run_remains_interrupted': True,
        'completed_protocols_preserved': list(old['evaluations']),
        'requested_protocols': [x for x in PROTOCOL_FILES if x not in old['evaluations']],
        'artifact_policy': 'Absolute originating prefixes for completed evaluations; new prefix for recovered evaluations',
        'original_evaluation_effort_after_last_completed_record': 'unmeasured; excluded',
        'reboot_gap_included': False,
        'source_validation': str(Path(output) / 'recovery_validation.json'),
        'runner': fingerprint(__file__)}
    result['evaluation_origins'] = {kind: {'run_dir': str(original), 'prefix': value['prefix'],
        'source_manifest_sha256': evidence['original_manifest']['sha256'], 'reused_without_scoring': True}
        for kind, value in old['evaluations'].items()}
    return result




