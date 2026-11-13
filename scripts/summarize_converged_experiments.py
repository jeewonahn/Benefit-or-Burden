"""Summarize new converged experiments; never execute a model or import old times.

Requires six models x two training protocols x explicitly declared seeds, all evaluated
on fixed IMP, RandS30/90/150 and full Total inputs. Tables retain per-seed
results and use sample SD across training seeds, not across test resamples.
"""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import random

import numpy as np

from convergence_control import EarlyStopping, selection_value
from training_validation import resolve_dev_protocol, resolve_dev_path, validate_candidate_profile
from candidate_policy import PROFILE as CONSISTENT_PROFILE
from consistent_candidate_profile import evaluation_paths
from summarize_small_experiment import (load_artifact, aligned_groups, means,
    direction, average_ranks, group_hash, file_hash)
from summarize_full_total_matrix import validate_metrics


MODELS = ('NAML', 'NRMS', 'NPA', 'LSTUR', 'CNE-SUE', 'DIGAT')
TRAINING = ('imp', 'rs50')
LEGACY_FIXED_CANDIDATE_SEED_SETS = ((42,), (42, 43, 44))
PROTOCOLS = ('imp', 'rs30', 'rs90', 'rs150', 'total')
METRICS = ('AUC', 'MRR', 'nDCG@5', 'nDCG@10')
AGGREGATIONS = ('native_entry_macro', 'request_macro')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def write_csv(path, rows):
    if not rows:
        raise ValueError('Refuse empty expected result: ' + str(path))
    with Path(path).open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sample_stats(values):
    valid = np.asarray([float(v) for v in values if v is not None], dtype=float)
    if not np.isfinite(valid).all():
        raise ValueError('Nonfinite value cannot enter seed aggregation')
    return {'mean': float(valid.mean()) if len(valid) else None,
            'sample_sd': float(valid.std(ddof=1)) if len(valid) > 1 else None,
            'n_seeds': len(valid)}


def declared_seeds(plan):
    values = plan.get('seeds')
    if not isinstance(values, list) or not values or any(type(v) is not int or not 0 <= v < 2**32 for v in values):
        raise ValueError('Plan must explicitly declare unsigned 32-bit integer training seeds')
    seeds = tuple(sorted(values))
    if len(set(seeds)) != len(seeds):
        raise ValueError('Plan training seeds must be distinct')
    return seeds


def declared_candidate_seed(plan):
    """New plans pin the preprocessing RNG; known historical plans retain seed 42."""
    seed = plan.get('candidate_seed')
    if seed is None and declared_seeds(plan) in LEGACY_FIXED_CANDIDATE_SEED_SETS:
        seed = 42
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Plan must explicitly declare an unsigned 32-bit candidate_seed')
    return seed


def declared_resample_seed(plan, override=None):
    """Keep request resampling separate from training repetitions and resumable."""
    recorded = plan.get('analysis_seeds', {}).get('table3_request_resampling', {}).get('seed')
    if recorded is not None and (type(recorded) is not int or not 0 <= recorded < 2**32):
        raise ValueError('Recorded Table 3 resample seed must be unsigned 32-bit')
    seed = recorded if override is None else override
    if seed is None:
        seed = 42  # Preserve standalone historical analysis plans.
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Table 3 resample seed must be unsigned 32-bit')
    if recorded is not None and seed != recorded:
        raise ValueError('Table 3 resample seed differs from the sealed plan')
    return seed


def summarize_rows(rows, keys, fields, seeds):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    result = []
    for key, items in sorted(groups.items()):
        if sorted(x['training_seed'] for x in items) != list(seeds):
            raise ValueError('Each summary row must have exactly the declared training seeds')
        row = dict(zip(keys, key))
        for field in fields:
            for label, value in sample_stats([x[field] for x in items]).items():
                row[field + '_' + label] = value
        result.append(row)
    return result


def shared_groups(reference, artifact):
    keys, audit = aligned_groups([reference, artifact])
    if not keys or audit['excluded_groups'] or audit['gt_mismatch_groups']:
        raise ValueError('Request support or positive multiset differs; no silent intersection')
    return keys, audit


def verify_convergence(manifest):
    state = manifest.get('convergence', manifest.get('early_stopping'))
    if not state:
        raise ValueError('Convergence controller state is missing')
    policy = {k: state[k] for k in ('max_epochs','min_epochs','patience','min_delta')}
    if policy != dict(max_epochs=50, min_epochs=5, patience=5, min_delta=1e-4):
        raise ValueError('Unexpected convergence policy')
    controller = EarlyStopping(**policy)
    records = manifest['epoch_records']
    for row in records:
        score = selection_value(row['dev']['metrics'])
        controller.update(row['epoch'], score)
        logged = row.get('selection_value', row.get('selection_score'))
        if logged is not None and not math.isclose(float(logged), score, abs_tol=1e-12, rel_tol=0):
            raise ValueError('Validation selection score differs from exact avg formula')
    if controller.stop_reason not in ('patience','max_epochs'):
        raise ValueError('Run has no completed stopping decision')
    if manifest['best_epoch'] != controller.best_epoch or manifest['epochs_completed'] != controller.last_epoch:
        raise ValueError('Selected or completed epoch differs from replayed validation curve')
    saved = controller.state_dict()
    for key in ('last_epoch','best_epoch','bad_epochs','stop_reason'):
        if state[key] != saved[key]:
            raise ValueError('Saved convergence state differs: ' + key)
    for key in ('best_score','significance_anchor'):
        if not math.isclose(float(state[key]), saved[key], abs_tol=1e-12, rel_tol=0):
            raise ValueError('Saved convergence score differs: ' + key)
    return {'best_epoch': controller.best_epoch, 'epochs_completed': controller.last_epoch,
            'stop_reason': controller.stop_reason, 'patience_satisfied': controller.stop_reason == 'patience'}


def checked_seconds(value, name):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError('Invalid measured seconds: ' + name)
    return value


def initialization_hash(manifest):
    values = [manifest[k] for k in ('initial_model_sha256','initial_model_state_sha256') if manifest.get(k)]
    if not values or len(set(values)) != 1:
        raise ValueError('Missing or inconsistent paired initialization identity')
    return values[0]


def development_expectation(data_root, dataset, training):
    """Pin the actual held-out file for each training objective."""
    protocol = resolve_dev_protocol(training)
    path = resolve_dev_path(data_root, training)
    profile = validate_candidate_profile(data_root, training)
    if protocol == 'imp':
        metadata = dataset['output_files']['dev/behaviors.tsv']
        stats = dataset['statistics']['dev/behaviors.tsv']
        expected = metadata['sha256']
    else:
        expected = file_hash(path)
        stats = {'rows': 0, 'candidates': 0}
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                fields = line.rstrip('\r\n').split('\t')
                if len(fields) != 5:
                    raise ValueError('Malformed matched validation input')
                stats['rows'] += 1
                stats['candidates'] += len(fields[4].split())
    return {'protocol': protocol, 'sha256': expected, 'rows': stats['rows'],
            'candidates': stats['candidates'], 'candidate_profile': profile}


def verify_development_records(manifest, expected):
    """Reject relabelled old checkpoints and incomplete validation passes."""
    selection = manifest.get('selection') or {}
    if (not selection and expected['protocol'] == 'imp'
            and manifest.get('model') == 'DIGAT' and manifest.get('train_type') == 'imp'
            and manifest.get('profile', {}).get('dev_protocol') == 'imp'):
        # Original IMP runs remain reusable without rewriting their manifests.
        files = manifest.get('input_binding', {}).get('inputs', {}).get('files', [])
        dev_files = [row for row in files if str(row.get('path', '')).replace(chr(92), '/').endswith('/dev/behaviors.tsv')]
        if len(dev_files) == 1:
            selection = {'protocol': 'imp', 'file': dev_files[0]}
    if (selection.get('protocol') != expected['protocol']
            or selection.get('file', {}).get('sha256') != expected['sha256']):
        raise ValueError('Checkpoint selection validation input/protocol differs from training')
    if expected['protocol'] == 'rs50' and selection.get('matched_candidate_profile') != expected['candidate_profile']:
        raise ValueError('Checkpoint selection has a different matched RandS candidate profile')
    for epoch in manifest['epoch_records']:
        dev = epoch['dev']
        if dev.get('input', {}).get('sha256') != expected['sha256']:
            raise ValueError('Checkpoint selection validation input differs from training')
        if any(dev[key] != expected[key] for key in ('rows', 'candidates')):
            raise ValueError('Incomplete validation input during checkpoint selection')


def evaluation_expectations(data_root, dataset, candidate):
    """Use eligible IMP support for the consistent profile; preserve legacy paths."""
    profile = validate_candidate_profile(Path(data_root), 'imp')
    if (profile or {}).get('profile') == CONSISTENT_PROFILE:
        paths = evaluation_paths(Path(data_root).resolve(), profile)
        inputs = {p: dataset['output_files'][paths[p].relative_to(Path(data_root).resolve()).as_posix()]['sha256']
                  for p in PROTOCOLS if p != 'total'}
        sizes = {p: dataset['statistics'][paths[p].relative_to(Path(data_root).resolve()).as_posix()]
                 for p in PROTOCOLS if p != 'total'}
    else:
        filenames = {'imp': 'behaviors.tsv', 'rs30': 'behaviors_rs30.tsv',
                     'rs90': 'behaviors_rs90.tsv', 'rs150': 'behaviors_rs150.tsv'}
        inputs = {p: dataset['output_files']['test/' + name]['sha256'] for p, name in filenames.items()}
        sizes = {p: dataset['statistics']['test/' + name] for p, name in filenames.items()}
    inputs['total'] = candidate['output']['sha256']
    sizes['total'] = candidate['statistics']
    return inputs, sizes


def collect(plan):
    seeds = declared_seeds(plan)
    jobs = [j for j in plan['jobs'] if j.get('model') is not None]
    expected = {(seed,t,m) for seed in seeds for t in TRAINING for m in MODELS}
    identities = [(j['seed'],j['train_type'],j['model']) for j in jobs]
    if len(identities) != len(expected) or set(identities) != expected:
        raise ValueError(f'Expected exactly {len(expected)} distinct training runs for the declared seeds')
    dataset_path = Path(plan['data_root']) / 'manifest.json'
    dataset, candidate = read(dataset_path), read(plan['candidate_manifest'])
    candidate_seed = declared_candidate_seed(plan)
    result_configuration = plan.get('result_configuration')
    if result_configuration is not None:
        if file_hash(result_configuration['path']) != result_configuration['sha256']:
            raise ValueError('Result configuration fingerprint differs from the sealed table plan')
        config = read(result_configuration['path'])
        if config.get('training_seeds') != list(seeds) or config.get('candidate_seed') != candidate_seed:
            raise ValueError('Result configuration seeds differ from the table plan')
    if dataset['status'] != 'complete' or dataset['fraction'] != 1.0 or type(dataset['seed']) is not int or dataset['seed'] != candidate_seed:
        raise ValueError('Full-request dataset does not match the declared candidate/split seed')
    if 'seed' in candidate and (type(candidate['seed']) is not int or candidate['seed'] != candidate_seed):
        raise ValueError('Candidate manifest seed differs from the declared candidate/split seed')
    if candidate['status'] != 'complete' or candidate['protocol'] != 'total':
        raise ValueError('Full Total candidate manifest is not complete')
    if candidate['dataset_manifest']['sha256'] != file_hash(dataset_path):
        raise ValueError('Full Total belongs to a different dataset manifest')
    expected_inputs, expected_sizes = evaluation_expectations(plan['data_root'], dataset, candidate)
    grouped, audits, timing, init_hashes = {}, [], [], {}
    reference, common = None, None
    for job in sorted(jobs, key=lambda j:(j['seed'],j['train_type'],j['model'])):
        seed, training, model = job['seed'], job['train_type'], job['model']
        manifest_path = Path(job['success_file'])
        manifest = read(manifest_path)
        if manifest.get('status') != 'completed' or manifest.get('training_reused') is True:
            raise ValueError('Expected a newly trained completed run: ' + str(manifest_path))
        if (manifest['seed'],manifest['train_type'],manifest['model']) != (seed,training,model):
            raise ValueError('Run identity differs from plan')
        expected_dev = development_expectation(plan['data_root'], dataset, training)
        verify_development_records(manifest, expected_dev)
        stop = verify_convergence(manifest)
        init_hash = initialization_hash(manifest)
        key = (seed,model)
        if key in init_hashes and init_hashes[key] != init_hash:
            raise ValueError('IMP/RandS paired initialization differs')
        init_hashes[key] = init_hash
        seconds = {}
        for protocol in PROTOCOLS:
            entry = manifest['evaluations'][protocol]
            artifact = load_artifact(model, training, protocol, entry['prefix'])
            if artifact is None or artifact.report['input']['sha256'] != expected_inputs[protocol]:
                raise ValueError('Missing or wrong evaluation input: ' + protocol)
            if artifact.report['rows'] != expected_sizes[protocol]['rows'] or artifact.report['candidates'] != expected_sizes[protocol]['candidates']:
                raise ValueError('Incomplete evaluation rows/candidates: ' + protocol)
            for metric in METRICS:
                if abs(artifact.report['metrics'][metric] - entry['metrics'][metric]) > 1e-12:
                    raise ValueError('Manifest and artifact metric differ')
            _, raw_audit = validate_metrics(entry['prefix'], expected_inputs[protocol])
            if reference is None:
                reference = artifact
            keys, alignment = shared_groups(reference, artifact)
            if common is None:
                common = keys
            elif keys != common:
                raise ValueError('Shared request ordering differs across seeds or protocols')
            grouped[(seed,training,model,protocol)] = artifact.grouped(common)
            seconds[protocol] = checked_seconds(artifact.report['elapsed_seconds'], protocol)
            audits.append({'training_seed':seed,'train_type':training,'model':model,'protocol':protocol,
                           'manifest':str(manifest_path),'manifest_sha256':file_hash(manifest_path),
                           'prefix':str(artifact.prefix),'input_sha256':expected_inputs[protocol],
                           'source_hashes':artifact.source_hashes,'alignment':alignment,**raw_audit})
            if artifact is not reference:
                del artifact
        timers = manifest['timings_seconds']
        training_seconds = sum(checked_seconds(x['train_seconds'],'epoch training') for x in manifest['epoch_records'])
        dev_seconds = sum(checked_seconds(x['dev']['elapsed_seconds'],'epoch dev') for x in manifest['epoch_records'])
        if abs(training_seconds-checked_seconds(timers['training'],'training total')) > 1e-5:
            raise ValueError('Completed epoch training time differs from new-run total')
        if abs(dev_seconds-checked_seconds(timers['dev_evaluation'],'dev total')) > 1e-5:
            raise ValueError('Completed dev time differs from new-run total')
        for protocol in PROTOCOLS:
            if abs(seconds[protocol]-checked_seconds(timers['tests'][protocol],'test total')) > 1e-5:
                raise ValueError('Test time differs from saved evaluation timer')
        segments = manifest['execution_segments']
        active_seconds = sum(checked_seconds(s['elapsed_seconds'],'execution segment') for s in segments)
        if abs(active_seconds-checked_seconds(manifest['elapsed_seconds'],'active runner total')) > 1e-3:
            raise ValueError('New execution-segment sum differs from runner total')
        timing.append({'training_seed':seed,'train_type':training,'model':model,**stop,
                       'training_seconds':training_seconds,'validation_seconds':dev_seconds,
                       **{p+'_test_seconds':seconds[p] for p in PROTOCOLS},
                       'all_test_seconds':sum(seconds.values()),
                       'training_plus_all_tests_seconds':training_seconds+sum(seconds.values()),
                       'active_runner_seconds':active_seconds,'execution_segments':len(segments)})
        print(f'Validated {model}/{training}/seed{seed}: {stop["epochs_completed"]} epochs, 5 evaluations', flush=True)
    return grouped, common, audits, timing, {'candidate_seed':candidate_seed,'result_configuration':result_configuration,'dataset_manifest_sha256':file_hash(dataset_path),
        'candidate_manifest_sha256':file_hash(plan['candidate_manifest']), 'input_sha256':expected_inputs,
        'validation_policy':'training_matched', 'development_protocols':{'imp':'imp','rs50':'rs50'}}



def validated_net_timing(manifest):
    """Audit measured rests separately; missing historical evidence stays unknown."""
    fields = ('training_seconds', 'validation_seconds',
              *(p + '_test_seconds' for p in PROTOCOLS),
              'all_test_seconds', 'training_plus_all_tests_seconds',
              'active_runner_seconds', 'all_recorded_rest_seconds', 'runner_only_rest_seconds')
    unknown = dict(status='unknown_legacy_net_time', provenance=None, **{key: None for key in fields})
    net = manifest.get('timings_seconds_net')
    if net is None:
        return unknown

    def close(left, right, label, tolerance=1e-4):
        left, right = checked_seconds(left, label), checked_seconds(right, label)
        if abs(left - right) > tolerance:
            raise ValueError('Cooling timing mismatch: ' + label)
        return left

    def pair(raw, rest, value, label):
        raw, rest, value = (checked_seconds(x, label) for x in (raw, rest, value))
        if rest > raw + 1e-4:
            raise ValueError('Cooling rest exceeds raw timer: ' + label)
        close(raw - rest, value, label + ' raw-minus-rest')
        return rest, value

    training_rest = training_net = dev_rest = dev_net = 0.0
    committed = defaultdict(float)
    for epoch in manifest['epoch_records']:
        number = epoch['epoch']
        rest, value = pair(epoch['train_seconds'], epoch['train_cooling_rest_seconds'],
                           epoch['train_net_seconds'], 'training epoch ' + str(number))
        training_rest += rest; training_net += value
        committed[('training', number, None)] += rest
        report = epoch['dev']
        rest, value = pair(report['elapsed_seconds'], report['cooling_rest_seconds'],
                           report['net_elapsed_seconds'], 'development epoch ' + str(number))
        dev_rest += rest; dev_net += value
        committed[('dev', number, None)] += rest
    close(training_net, net['training'], 'training total')
    close(dev_net, net['dev_evaluation'], 'development total')
    tests, test_rest = {}, 0.0
    chunk_groups = dict(manifest.get('test_chunks', {}))
    if manifest.get('chunks'):
        chunk_groups['total'] = manifest['chunks']
    for protocol in PROTOCOLS:
        report = manifest['evaluations'][protocol]
        rest, value = pair(report['elapsed_seconds'], report['cooling_rest_seconds'],
                           report['net_elapsed_seconds'], protocol + ' evaluation')
        tests[protocol] = close(value, net['tests'][protocol], protocol + ' net total')
        test_rest += rest
        committed[('test', None, protocol)] += rest
        chunks = chunk_groups.get(protocol, [])
        if report.get('chunk_count', 0) and not chunks:
            raise ValueError('Missing chunk timer evidence: ' + protocol)
        if chunks:
            totals = dict(raw=0.0, rest=0.0, net=0.0)
            for chunk in chunks:
                evidence = chunk['report']
                if file_hash(evidence['path']) != evidence['sha256']:
                    raise ValueError('Changed chunk timer evidence')
                item = read(evidence['path'])
                chunk_rest, chunk_net = pair(item['elapsed_seconds'], item['cooling_rest_seconds'],
                                             item['net_elapsed_seconds'], 'chunk ' + protocol)
                totals['raw'] += item['elapsed_seconds']; totals['rest'] += chunk_rest; totals['net'] += chunk_net
            close(totals['raw'], report['elapsed_seconds'], protocol + ' chunk raw sum')
            close(totals['rest'], rest, protocol + ' chunk rest sum')
            close(totals['net'], value, protocol + ' chunk net sum')
    segment_rest = segment_net = 0.0
    for segment in manifest['execution_segments']:
        rest, value = pair(segment['elapsed_seconds'], segment['cooling_rest_seconds'],
                           segment['net_elapsed_seconds'], 'execution segment')
        segment_rest += rest; segment_net += value
    close(segment_rest, net['cooling_rest'], 'runner rest total')
    close(segment_net, net['total'], 'runner net total')
    close(segment_net, manifest['elapsed_net_seconds'], 'saved runner net total')
    close(segment_net + segment_rest, manifest['elapsed_seconds'], 'saved runner raw total')

    cooling = manifest.get('cooling', {})
    ledger_proof = None
    by_phase = defaultdict(float)
    if cooling.get('enabled'):
        path = Path(cooling['ledger'])
        starts, ended, guards, intervals = {}, set(), set(), []
        with path.open(encoding='utf-8-sig') as stream:
            for line in stream:
                row = json.loads(line)
                if row.get('method') != 'cooperative_sleep':
                    raise ValueError('Unsupported cooling ledger method')
                session = row['session_id']
                if row['event'] == 'guard_start':
                    if session in guards or row.get('no_managed_pauses_before_guard_start') is not True:
                        raise ValueError('Invalid cooling session proof')
                    guards.add(session)
                elif row['event'] == 'pause_start':
                    key = row['interval_id']
                    if session not in guards or key in starts:
                        raise ValueError('Duplicate or unguarded cooling interval')
                    starts[key] = row
                elif row['event'] == 'pause_end':
                    key = row['interval_id']
                    if key not in starts or key in ended:
                        raise ValueError('Unpaired or duplicate cooling interval end')
                    begin = starts[key]
                    for name in ('session_id', 'phase', 'epoch', 'protocol', 'kind'):
                        if row.get(name) != begin.get(name):
                            raise ValueError('Cooling interval context changed')
                    # Wall and monotonic stamps are sampled separately; allow clock-read jitter.
                    seconds = close(row['duration_seconds'], row['end_unix'] - row['start_unix'], 'ledger duration', tolerance=0.1)
                    close(row['start_unix'], begin['event_unix'], 'ledger start')
                    if row['end_unix'] < row['start_unix']:
                        raise ValueError('Reversed cooling interval')
                    phase, epoch, protocol = row.get('phase'), row.get('epoch'), row.get('protocol')
                    if phase not in ('training', 'dev', 'test', 'development_setup', 'test_setup', 'preparation'):
                        raise ValueError('Unknown cooling phase')
                    if phase in ('training', 'dev') and (type(epoch) is not int or epoch < 1):
                        raise ValueError('Cooling training/development epoch is missing')
                    if phase == 'test' and protocol not in PROTOCOLS:
                        raise ValueError('Cooling evaluation protocol is missing')
                    by_phase[(phase, epoch if phase in ('training', 'dev') else None,
                              protocol if phase == 'test' else None)] += seconds
                    intervals.append((row['start_unix'], row['end_unix']))
                    ended.add(key)
        if set(starts) != ended:
            raise ValueError('An interrupted cooling interval needs explicit recovery evidence')
        intervals.sort()
        if any(left[1] > right[0] + 1e-4 for left, right in zip(intervals, intervals[1:])):
            raise ValueError('Overlapping cooling intervals would double-subtract time')
        close(sum(by_phase.values()), segment_rest, 'ledger versus segment rest')
        for key, rest in committed.items():
            if rest > by_phase[key] + 1e-4:
                raise ValueError('Committed timer rest has no matching phase/epoch/protocol evidence')
        ledger_proof = json.dumps({'path': str(path), 'sha256': file_hash(path),
            'closed_intervals': len(intervals), 'sessions': len(guards)}, sort_keys=True)
    elif segment_rest > 1e-4:
        raise ValueError('Measured rests require a cooperative cooling ledger')
    committed_rest = training_rest + dev_rest + test_rest
    if committed_rest > segment_rest + 1e-4:
        raise ValueError('Phase rests exceed total rest; probable double subtraction')
    return dict(status='verified_measured_net_time', provenance=ledger_proof or 'explicit_zero_rest_fields',
        training_seconds=training_net, validation_seconds=dev_net,
        **{p+'_test_seconds':tests[p] for p in PROTOCOLS}, all_test_seconds=sum(tests.values()),
        training_plus_all_tests_seconds=training_net+sum(tests.values()), active_runner_seconds=segment_net,
        all_recorded_rest_seconds=segment_rest, runner_only_rest_seconds=max(0.0, segment_rest-committed_rest))


def net_timing_rows(plan):
    """Return separate net rows without altering raw or accuracy result tables."""
    rows = []
    for job in sorted((j for j in plan['jobs'] if j.get('model') is not None),
                      key=lambda j:(j['seed'], j['train_type'], j['model'])):
        result = validated_net_timing(read(job['success_file']))
        rows.append(dict(training_seed=job['seed'], train_type=job['train_type'], model=job['model'], **result))
    return rows


def compute_tables(grouped, common, seeds, fraction=0.5, repetitions=10, resample_seed=42, tolerance=1e-12):
    seeds = declared_seeds({'seeds': list(seeds)})
    if not 0 < fraction <= 1 or repetitions != 10 or not math.isfinite(tolerance) or tolerance < 0 or not common:
        raise ValueError('Require shared groups, ten repetitions and a valid fraction/tolerance')
    expected = {(s,t,m,p) for s in seeds for t in TRAINING for m in MODELS for p in PROTOCOLS}
    if set(grouped) != expected:
        raise ValueError('Incomplete seed/training/model/protocol matrix')
    all_indices = np.arange(len(common))
    size = max(1, math.floor(len(common)*fraction))
    choices = [np.asarray(sorted(random.Random(resample_seed+r).sample(range(len(common)),size)),dtype=np.int64)
               for r in range(repetitions)]
    samples = [{'repetition':r+1,'resampling_seed':resample_seed+r,'group_count':size,
                'group_indices':chosen.tolist(),
                'selected_group_hash_sha256':hashlib.sha256('\n'.join(group_hash(common[i]) for i in chosen).encode()).hexdigest()}
               for r,chosen in enumerate(choices)]
    table2, pairs, table3, table5 = [], [], [], []
    for seed in seeds:
        for training in TRAINING:
            for aggregation in AGGREGATIONS:
                full_means = {(m,p):means(grouped[seed,training,m,p],all_indices,aggregation)
                              for m in MODELS for p in PROTOCOLS}
                for protocol in PROTOCOLS:
                    for j,metric in enumerate(METRICS):
                        vals = [full_means[m,protocol][j] for m in MODELS]
                        ranks = average_ranks(vals,tolerance)
                        for model,value,rank in zip(MODELS,vals,ranks):
                            table2.append({'training_seed':seed,'train_type':training,'aggregation':aggregation,
                                           'protocol':protocol,'metric':metric,'model':model,
                                           'value':float(value),'rank':rank})
                local = []
                for repetition,chosen in enumerate(choices,1):
                    sampled = {(m,p):means(grouped[seed,training,m,p],chosen,aggregation)
                               for m in MODELS for p in PROTOCOLS}
                    for a,b in itertools.combinations(MODELS,2):
                        for j,metric in enumerate(METRICS):
                            ref_delta = float(sampled[a,'total'][j]-sampled[b,'total'][j])
                            ref_sign = direction(ref_delta,tolerance)
                            for protocol in PROTOCOLS[:-1]:
                                delta = float(sampled[a,protocol][j]-sampled[b,protocol][j])
                                sign = direction(delta,tolerance)
                                local.append({'training_seed':seed,'train_type':training,'aggregation':aggregation,
                                              'repetition':repetition,'protocol':protocol,'reference':'total',
                                              'metric':metric,'model_a':a,'model_b':b,
                                              'reference_difference':ref_delta,'candidate_difference':delta,
                                              'reference_sign':ref_sign,'candidate_sign':sign,
                                              'eligible':int(ref_sign!=0),'matched':int(ref_sign!=0 and sign==ref_sign)})
                pairs.extend(local)
                for protocol in PROTOCOLS[:-1]:
                    for metric in METRICS:
                        rows = [r for r in local if r['protocol']==protocol and r['metric']==metric]
                        eligible = [r for r in rows if r['eligible']]
                        per_pair = defaultdict(list)
                        for row in eligible:
                            per_pair[row['model_a'],row['model_b']].append(row['matched'])
                        matched = sum(r['matched'] for r in eligible)
                        macro = 100*float(np.mean([np.mean(v) for v in per_pair.values()])) if per_pair else None
                        table3.append({'training_seed':seed,'train_type':training,'aggregation':aggregation,
                                       'protocol':protocol,'reference':'total','metric':metric,
                                       'matching_percent':macro,
                                       'pooled_matching_percent':100*matched/len(eligible) if eligible else None,
                                       'comparisons':len(rows),'eligible_comparisons':len(eligible),
                                       'reference_ties_excluded':len(rows)-len(eligible),'eligible_pairs':len(per_pair)})
        for aggregation in AGGREGATIONS:
            for model in MODELS:
                imp = means(grouped[seed,'imp',model,'total'],all_indices,aggregation)
                rs = means(grouped[seed,'rs50',model,'total'],all_indices,aggregation)
                for j,metric in enumerate(METRICS):
                    difference = float(rs[j]-imp[j])
                    table5.append({'training_seed':seed,'model':model,'aggregation':aggregation,'protocol':'total',
                                   'metric':metric,'imp':float(imp[j]),'rs50':float(rs[j]),
                                   'paired_difference':difference,
                                   'relative_gain_percent':100*difference/float(imp[j]) if imp[j]!=0 else None})
    table2_summary = summarize_rows(table2,('train_type','aggregation','protocol','metric','model'),('value','rank'),seeds)
    rank_groups = defaultdict(list)
    for row in table2_summary:
        rank_groups[tuple(row[k] for k in ('train_type','aggregation','protocol','metric'))].append(row)
    for rows in rank_groups.values():
        for row,rank in zip(rows,average_ranks([r['value_mean'] for r in rows],tolerance)):
            row['rank_of_mean_score'] = rank
    table3_summary = summarize_rows(table3,('train_type','aggregation','protocol','reference','metric'),
                                    ('matching_percent','pooled_matching_percent'),seeds)
    table5_summary = summarize_rows(table5,('model','aggregation','protocol','metric'),
                                    ('imp','rs50','paired_difference','relative_gain_percent'),seeds)
    for row in table5_summary:
        row['relative_gain_of_means_percent'] = (100*(row['rs50_mean']-row['imp_mean'])/row['imp_mean']
                                                 if row['imp_mean']!=0 else None)
    return {'table2_per_seed':table2,'table2_summary':table2_summary,'table3_pairs':pairs,
            'table3_per_seed':table3,'table3_summary':table3_summary,
            'table5_per_seed':table5,'table5_summary':table5_summary,'samples':samples}


def format_stat(row,field,scale=1):
    mean,sd = row[field+'_mean'],row[field+'_sample_sd']
    return 'undefined' if mean is None else (f'{mean/scale:.4f} ± {sd/scale:.4f}' if sd is not None else f'{mean/scale:.4f} (SD: NA; n={row[field+"_n_seeds"]})')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--resample-fraction',type=float,default=0.5)
    parser.add_argument('--repetitions',type=int,default=10)
    parser.add_argument('--resample-seed',type=int,help='Use the sealed plan analysis seed; legacy standalone plans default to 42')
    parser.add_argument('--tie-tolerance',type=float,default=1e-12)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve existing summary; choose a new output directory')
    plan = read(args.plan)
    seeds = declared_seeds(plan)
    resample_seed = declared_resample_seed(plan, args.resample_seed)
    expected_runs = len(seeds)*len(MODELS)*len(TRAINING)
    expected_evaluations = expected_runs*len(PROTOCOLS)
    grouped,common,audits,timing,inputs = collect(plan)
    tables = compute_tables(grouped,common,seeds,args.resample_fraction,args.repetitions,resample_seed,args.tie_tolerance)
    timing_fields = ('training_seconds','validation_seconds',*tuple(p+'_test_seconds' for p in PROTOCOLS),
                     'all_test_seconds','training_plus_all_tests_seconds','active_runner_seconds')
    timing_summary = summarize_rows(timing,('model','train_type'),timing_fields,seeds)
    args.output.mkdir(parents=True,exist_ok=False)
    for name,rows in tables.items():
        if name!='samples':
            write_csv(args.output/(name+'.csv'),rows)
    write_csv(args.output/'timings_per_seed.csv',timing)
    write_csv(args.output/'timings_summary.csv',timing_summary)
    measured_net = net_timing_rows(plan)
    write_csv(args.output/'timings_net_per_seed.csv', [{key: ('NA' if value is None else value) for key,value in row.items()} for row in measured_net])
    write_json(args.output/'request_groups.json',[{'index':i,'group_sha256':group_hash(k)} for i,k in enumerate(common)])
    write_json(args.output/'resampling_manifest.json',{'shared_across':'all models, training methods and training seeds',
        'request_group_count':len(common),'fraction':args.resample_fraction,'repetitions':args.repetitions,
        'resample_seed':resample_seed,'analysis_seed_provenance':plan.get('analysis_seeds',{}).get('table3_request_resampling'),
        'request_groups_sha256':file_hash(args.output/'request_groups.json'),'samples':tables['samples']})
    lines=['# Convergence-run Tables 2, 3 and 5','',
           'New measured results, not original-paper reference values or exact R2 reproduction.',
           f'All supplied MIND-Small master requests before recorded zero-negative eligibility exclusions; candidate/split seed {declared_candidate_seed(plan)}; paired training seeds: {", ".join(map(str,seeds))}.',
           (f'Values below are mean ± sample SD across {len(seeds)} training seeds.' if len(seeds)>1
            else 'Values below are single-seed results; training-seed sample SD is unavailable (NA/null).'),
           'Table3 test resamples are not training replicates.',
           'Training-matched development candidates and the recorded weighted avg criterion select each best checkpoint; full Total is the reference.',
           f'Completed plan: {len(timing)}/{expected_runs} training runs and {len(audits)}/{expected_evaluations} evaluations.',
           f'Patience stopping: {sum(r["stop_reason"]=="patience" for r in timing)}/{expected_runs} runs; epoch cap reached: {sum(r["stop_reason"]=="max_epochs" for r in timing)}/{expected_runs}.',
           'Primary display uses native-entry macro metrics. Request-macro sensitivity and all per-seed ranks are in the CSV files.',
           '', '## Table 2: recommendation accuracy and ranks','',
           'Rank of the mean score and each seed rank are separate CSV fields. The following tables report all four metrics.', '']
    for training in TRAINING:
        for protocol in PROTOCOLS:
            lines += [f'### Training {training} / evaluation {protocol}','',
                      '| Model | AUC | MRR | nDCG@5 | nDCG@10 |','| --- | --- | --- | --- | --- |']
            for model in MODELS:
                values = [next(r for r in tables['table2_summary'] if r['train_type']==training and r['protocol']==protocol
                               and r['aggregation']=='native_entry_macro' and r['model']==model and r['metric']==metric) for metric in METRICS]
                lines.append('| '+' | '.join([model]+[format_stat(r,'value') for r in values])+' |')
            lines.append('')
    lines += ['## Table 3: model-pair matching with full Total (%)','',
              '| Training | Evaluation | AUC | MRR | nDCG@5 | nDCG@10 |','| --- | --- | --- | --- | --- | --- |']
    for training in TRAINING:
        for protocol in PROTOCOLS[:-1]:
            values=[next(r for r in tables['table3_summary'] if r['train_type']==training and r['protocol']==protocol
                         and r['aggregation']=='native_entry_macro' and r['metric']==metric) for metric in METRICS]
            lines.append('| '+' | '.join([training,protocol]+[format_stat(r,'matching_percent') for r in values])+' |')
    lines += ['', '## Table 5: training comparison on full Total','',
              '| Model | Metric | IMP | RandS50 | Paired difference (RandS50 − IMP) |',
              '| --- | --- | --- | --- | --- |']
    for model in MODELS:
        for metric in METRICS:
            row=next(r for r in tables['table5_summary'] if r['model']==model and r['metric']==metric and r['aggregation']=='native_entry_macro')
            lines.append('| '+' | '.join([model,metric]+[format_stat(row,f) for f in ('imp','rs50','paired_difference')])+' |')
    lines += ['', '## Raw recorded time (minutes; includes measured rests)','',
              '| Model | Training | Completed-epoch training | Five tests | Active runner |','| --- | --- | --- | --- | --- |']
    for row in timing_summary:
        lines.append('| '+' | '.join([row['model'],row['train_type']]+[format_stat(row,f,60) for f in ('training_seconds','all_test_seconds','active_runner_seconds')])+' |')
    lines += ['', 'Training time sums completed epochs; failed incomplete-epoch effort remains only in active runner segments.',
              'Raw CSVs retain original timers, including measured rests. timings_net_per_seed.csv separately audits measured rests; historical missing net evidence is NA.',
              'Active runner time includes setup/validation/checks and recorded retries, excludes gaps between process segments, and is not training-plus-test time.',
              'No old one-epoch timings are imported. Shared asset preparation is outside model-run timing and should be reported separately.',
              'The per-seed table retains best epochs and stopping reasons. Hitting max_epochs is not proof of convergence.',
              'Table3 excludes ties in the Total reference and counts them in per-seed CSVs; candidate ties against an ordered reference count as mismatches.',
              ('Training-seed SD is descriptive.' if len(seeds)>1 else 'One training seed cannot establish variability across training seeds.'),
              'No significance claim or causal explanation is inferred from direction alone.', '']
    (args.output/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    manifest={'status':'completed','created_utc':datetime.now(timezone.utc).isoformat(),'training_runs':len(timing),'evaluations':len(audits),
              'expected_training_runs':expected_runs,'expected_evaluations':expected_evaluations,
              'plan_sha256':file_hash(args.plan),'inputs':inputs,'seeds':list(seeds),'reference':'total',
              'request_groups':len(common),'table3_resample_seed':resample_seed,'resampling_scope':'ten fixed shared request subsets; not ten training seeds',
              'matching_definition':'mean per-model-pair matched/eligible repetitions; exclude reference ties',
              'raw_score_validation_scope':'all per-entry metric/mean/range checks plus 32 deterministic raw-score rows per evaluation',
              'runs_at_epoch_cap':[r for r in timing if r['stop_reason']=='max_epochs'],
              'source_artifacts':audits,'timing_scope':'raw recorded segments including rests; separate audited timings_net_per_seed.csv; legacy net evidence is unknown',
              'net_timing_status':{str(row['model'])+'/'+row['train_type']+'/seed'+str(row['training_seed']):row['status'] for row in measured_net},
              'output_sha256':{p.name:file_hash(p) for p in args.output.iterdir() if p.is_file()}}
    write_json(args.output/'validation.json',manifest)
    print(json.dumps({'status':'completed','training_runs':len(timing),'evaluations':len(audits),'seeds':list(seeds),'output':str(args.output)}))


if __name__=='__main__':
    main()
