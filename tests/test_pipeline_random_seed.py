"""CPU-only regressions for fresh random study seeds and sealed resume behavior."""
import argparse
from collections import Counter
import contextlib
import csv
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_mind_pipeline as pipeline
import summarize_converged_experiments as summary
from convergence_control import EarlyStopping, selection_value
from summarize_small_experiment import Artifact


REQUIRED = ('prepare_mind_masters.py', 'prepare_consistent_candidates.py',
            'prepare_model_assets.py', 'prepare_nnr_embedding_cache.py',
            'context_embeddings.py', 'context_feature_policy.py', 'candidate_policy.py',
            'consistent_candidate_profile.py', 'run_converged_nnr.py',
            'summarize_converged_experiments.py', 'run_converged_digat.py',
            'prepare_digat_assets.py')


def arguments(work, *extra):
    return pipeline.parser().parse_args(['--work-dir', str(work), '--dry-run', *extra])


def source_fixture(root):
    source = root / 'source'
    (source / 'scripts').mkdir(parents=True)
    for name in REQUIRED:
        (source / 'scripts' / name).write_text('# synthetic source\n', encoding='utf-8')
    return source


class RandomStudySeedTests(unittest.TestCase):
    def test_previews_leave_random_unresolved_and_write_nothing(self):
        with tempfile.TemporaryDirectory() as temporary:
            for mode in ('--dry-run', '--check-only'):
                work = Path(temporary) / mode[2:]
                output = io.StringIO()
                with mock.patch.object(sys, 'argv', ['pipeline', '--work-dir', str(work), mode]), \
                     mock.patch.object(pipeline.secrets, 'randbits') as draw, \
                     mock.patch.object(pipeline, 'environment_report', return_value={'GPU_test_performed': False}), \
                     contextlib.redirect_stdout(output):
                    pipeline.main()
                preview = json.loads(output.getvalue())
                if mode == '--check-only':
                    preview = preview['plan_preview']
                self.assertEqual(preview['options']['seed'], 'random')
                self.assertFalse(preview['files_written'])
                self.assertFalse(work.exists())
                draw.assert_not_called()

    def test_fresh_seal_draws_once_persists_and_shares_all_paired_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = source_fixture(root)
            work = root / 'study'
            chosen = 3901234567
            with mock.patch.object(pipeline, 'REPO', source), \
                 mock.patch.object(pipeline.secrets, 'randbits', return_value=chosen) as draw:
                options = pipeline.normalize(arguments(work, '--models', 'all6'))
                plan = pipeline.create_plan(options)
                draw.assert_called_once_with(32)
            self.assertEqual(options['seed'], 'random')
            sealed = pipeline.read(work / 'control/pipeline_plan.json')
            self.assertEqual(sealed, plan)
            self.assertEqual(plan['options']['seed'], chosen)
            self.assertEqual(plan['seed_selection']['policy'], 'random_new_study')
            jobs = pipeline.job_commands(plan)
            self.assertEqual(len(jobs), 12)
            self.assertEqual({j['seed'] for j in jobs}, {chosen})
            for job in jobs:
                command = job['command']
                self.assertEqual(command[command.index('--seed') + 1], str(chosen))
                self.assertIn('seed' + str(chosen), job['run_dir'])
            phases = pipeline.phase_commands(plan)
            for name in ('raw', 'candidates'):
                command = phases[name]
                self.assertEqual(command[command.index('--seed') + 1], str(chosen))
            table_plan = pipeline.table_summary_plan(plan, jobs)
            self.assertEqual((table_plan['seeds'], table_plan['candidate_seed']), ([chosen], chosen))
            analysis = table_plan['analysis_seeds']['table3_request_resampling']
            self.assertEqual(analysis['parent_study_seed'], chosen)
            self.assertNotEqual(analysis['seed'], chosen)
            self.assertEqual(analysis, pipeline.derived_analysis_seed(chosen))
            self.assertEqual(summary.declared_resample_seed(table_plan), analysis['seed'])

    def test_resume_reuses_recorded_seed_and_does_not_create_a_second_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = source_fixture(root)
            work = root / 'study'
            with mock.patch.object(pipeline, 'REPO', source), \
                 mock.patch.object(pipeline.secrets, 'randbits', return_value=1789012345):
                plan = pipeline.create_plan(pipeline.normalize(arguments(work)))
            planpath = work / 'control/pipeline_plan.json'
            before = planpath.read_bytes()
            for extra in ([], ['--seed', 'random'], ['--seed', '1789012345']):
                with mock.patch.object(pipeline.secrets, 'randbits') as draw:
                    current = pipeline.normalize(arguments(work, *extra), plan['options'])
                self.assertEqual(current, plan['options'])
                draw.assert_not_called()
            with self.assertRaisesRegex(ValueError, 'seed'):
                pipeline.normalize(arguments(work, '--seed', '43'), plan['options'])
            with mock.patch.object(pipeline.secrets, 'randbits') as draw:
                with self.assertRaises(FileExistsError):
                    pipeline.create_plan(plan['options'])
                draw.assert_not_called()
            self.assertEqual(before, planpath.read_bytes())

    def test_main_resume_uses_saved_plan_and_does_not_reset_rng(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = source_fixture(root)
            work = root / 'study'
            with mock.patch.object(pipeline, 'REPO', source):
                plan = pipeline.create_plan(pipeline.normalize(arguments(work, '--seed', '123')))
            state = copy.deepcopy(plan)
            with mock.patch.object(sys, 'argv', ['pipeline', '--work-dir', str(work), '--execute', '--resume', '--seed', 'random']), \
                 mock.patch.object(pipeline.secrets, 'randbits') as draw, \
                 mock.patch.object(pipeline, 'create_plan') as create, \
                 mock.patch.object(pipeline, 'Pipeline') as runner:
                pipeline.main()
            draw.assert_not_called()
            create.assert_not_called()
            runner.assert_called_once_with(state, resume=True)
            runner.return_value.execute.assert_called_once_with(prepare_only=False)

    def test_independent_studies_draw_independently_and_explicit_replay_does_not_draw(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = source_fixture(root)
            with mock.patch.object(pipeline, 'REPO', source), \
                 mock.patch.object(pipeline.secrets, 'randbits', side_effect=[101, 202]) as draw:
                plans = [pipeline.create_plan(pipeline.normalize(arguments(root / str(i)))) for i in range(2)]
                replay = pipeline.create_plan(pipeline.normalize(arguments(root / 'replay', '--seed', '101')))
            self.assertEqual(draw.call_count, 2)
            self.assertEqual([p['options']['seed'] for p in plans], [101, 202])
            self.assertEqual(replay['options']['seed'], 101)
            self.assertEqual(replay['seed_selection']['policy'], 'explicit_reproduction')
            self.assertEqual(replay['analysis_seeds'], plans[0]['analysis_seeds'])

    def test_failed_source_preflight_never_consumes_a_seed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = source_fixture(root)
            (source / 'scripts/candidate_policy.py').unlink()
            with mock.patch.object(pipeline, 'REPO', source), \
                 mock.patch.object(pipeline.secrets, 'randbits') as draw:
                with self.assertRaisesRegex(ValueError, 'Incomplete source'):
                    pipeline.create_plan(pipeline.normalize(arguments(root / 'study')))
            draw.assert_not_called()
            self.assertFalse((root / 'study/control/pipeline_plan.json').exists())

    def test_seed_boundary_validation(self):
        for value in (0, 2**32 - 1, '0', '4294967295'):
            self.assertEqual(pipeline.parse_seed(value), int(value))
        self.assertEqual(pipeline.parse_seed('RaNdOm'), 'random')
        for value in (-1, 2**32, True, False, 1.5, None, 'bad', '1.0'):
            with self.assertRaises(argparse.ArgumentTypeError):
                pipeline.parse_seed(value)

    def test_summary_requires_candidate_seed_for_new_arbitrary_seed_plans(self):
        self.assertEqual(summary.declared_candidate_seed({'seeds': [42]}), 42)
        self.assertEqual(summary.declared_candidate_seed({'seeds': [42, 43, 44]}), 42)
        with self.assertRaisesRegex(ValueError, 'candidate_seed'):
            summary.declared_candidate_seed({'seeds': [31337]})
        self.assertEqual(summary.declared_candidate_seed({'seeds': [31337], 'candidate_seed': 31337}), 31337)
        plan = {'analysis_seeds': {'table3_request_resampling': pipeline.derived_analysis_seed(31337)}}
        expected = plan['analysis_seeds']['table3_request_resampling']['seed']
        self.assertEqual(summary.declared_resample_seed(plan), expected)
        self.assertEqual(summary.declared_resample_seed(plan, expected), expected)
        with self.assertRaisesRegex(ValueError, 'sealed plan'):
            summary.declared_resample_seed(plan, (expected + 1) % 2**32)

    def test_result_configuration_matches_all_runs_and_is_immutable_on_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = source_fixture(root)
            work = root / 'study'
            with mock.patch.object(pipeline, 'REPO', source):
                plan = pipeline.create_plan(pipeline.normalize(arguments(work, '--models', 'all6', '--seed', '31337')))
            jobs = pipeline.job_commands(plan)
            proofs = {}
            for job in jobs:
                manifest = {'model': job['model'], 'train_type': job['train_type'], 'seed': job['seed'],
                            'effective_config': {'seed': job['seed'], 'model': job['model'], 'batch_size': 16},
                            'evaluations': {p: {'metrics': dict.fromkeys(summary.METRICS, .4)} for p in pipeline.PROTOCOLS},
                            'best_epoch': 2, 'epochs_completed': 7}
                pipeline.save(job['manifest'], manifest)
                proofs[job['name']] = {'timings_seconds_net': {'training': 60, 'dev_evaluation': 10, 'total': 100,
                                       'cooling_rest': 120, 'tests': dict.fromkeys(pipeline.PROTOCOLS, 2)}}
            output = work / 'results'
            pipeline.write_selected_results(output, jobs, proofs, plan=plan)
            configpath = output / 'config.json'
            before = configpath.read_bytes()
            config = pipeline.read(configpath)
            self.assertEqual(config['training_seeds'], [31337])
            self.assertEqual(config['candidate_seed'], 31337)
            self.assertEqual(config['resolved_options'], plan['options'])
            self.assertEqual(config['pipeline_plan'], pipeline.fp(work / 'control/pipeline_plan.json'))
            self.assertEqual(config['analysis_seeds'], plan['analysis_seeds'])
            self.assertEqual(len(config['runs']), 12)
            by_id = {run['configuration_id']: run for run in config['runs']}
            for filename, expected_rows in (('metrics.csv', 60), ('timings_net.csv', 12)):
                with (output / filename).open(encoding='utf-8', newline='') as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), expected_rows)
                for row in rows:
                    run = by_id[row['configuration_id']]
                    self.assertEqual(row['configuration_file'], 'config.json')
                    self.assertEqual((run['model'], run['training'], run['seed']), (row['model'], row['training'], int(row['seed'])))
                    self.assertEqual(run['effective_config']['seed'], int(row['seed']))
                    self.assertEqual(run['effective_config_status'], 'recorded_native')
                    pipeline.verify(run['experiment_manifest'])
            table_plan = pipeline.table_summary_plan(plan, jobs)
            self.assertEqual(table_plan['result_configuration'], pipeline.fp(configpath))
            pipeline.write_selected_results(output, jobs, proofs, plan=plan)
            self.assertEqual(configpath.read_bytes(), before)
            manifestpath = Path(jobs[0]['manifest'])
            changed = pipeline.read(manifestpath)
            changed['effective_config']['batch_size'] = 32
            pipeline.save(manifestpath, changed)
            with self.assertRaisesRegex(ValueError, 'configuration changed'):
                pipeline.write_selected_results(output, jobs, proofs, plan=plan)
            self.assertEqual(configpath.read_bytes(), before)
            changed['effective_config']['seed'] = 42
            pipeline.save(manifestpath, changed)
            with self.assertRaisesRegex(ValueError, 'configuration seed differs'):
                pipeline.write_selected_results(output, jobs, proofs, plan=plan)

    def test_missing_native_config_is_explicitly_legacy_but_new_pipeline_rejects_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job = {'model': 'NAML', 'train_type': 'imp', 'seed': 42,
                   'manifest': str(root / 'experiment.json'), 'name': 'NAML_imp'}
            manifest = {'model': 'NAML', 'train_type': 'imp', 'seed': 42, 'best_epoch': 1, 'epochs_completed': 6,
                        'evaluations': {p: {'metrics': dict.fromkeys(summary.METRICS, .4)} for p in pipeline.PROTOCOLS}}
            pipeline.save(job['manifest'], manifest)
            proofs = {'NAML_imp': {'timings_seconds_net': {'training': 60, 'dev_evaluation': 10, 'total': 100,
                                    'cooling_rest': 120, 'tests': dict.fromkeys(pipeline.PROTOCOLS, 2)}}}
            pipeline.write_selected_results(root / 'legacy-results', [job], proofs)
            config = pipeline.read(root / 'legacy-results/config.json')
            self.assertIsNone(config['runs'][0]['effective_config'])
            self.assertEqual(config['runs'][0]['effective_config_status'], 'unavailable_legacy')
            with self.assertRaisesRegex(ValueError, 'Native effective configuration is absent'):
                pipeline.write_selected_results(root / 'new-results', [job], proofs,
                                                plan={'options': {'work_dir': str(root), 'seed': 42}})

    def test_arbitrary_seed_all_six_model_collect_and_table_outputs(self):
        seed = 3901234567
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = {'status': 'complete', 'fraction': 1.0, 'seed': seed,
                       'output_files': {}, 'statistics': {}}
            for protocol in ('imp', 'rs30', 'rs90', 'rs150'):
                name = 'behaviors.tsv' if protocol == 'imp' else 'behaviors_' + protocol + '.tsv'
                dataset['output_files']['test/' + name] = {'sha256': 'hash-' + protocol}
                dataset['statistics']['test/' + name] = {'rows': 2, 'candidates': 4}
            (root / 'manifest.json').write_text(json.dumps(dataset), encoding='utf-8')
            candidate = {'status': 'complete', 'protocol': 'total', 'seed': seed,
                         'dataset_manifest': {'sha256': summary.file_hash(root / 'manifest.json')},
                         'output': {'sha256': 'hash-total'}, 'statistics': {'rows': 2, 'candidates': 4}}
            (root / 'candidate.json').write_text(json.dumps(candidate), encoding='utf-8')
            stop = EarlyStopping()
            epochs = []
            for epoch, score in enumerate((.8, .7, .6, .5, .4, .3), 1):
                metrics = dict.fromkeys(summary.METRICS, score)
                stop.update(epoch, selection_value(metrics))
                epochs.append({'epoch': epoch, 'train_seconds': 2.,
                               'dev': {'metrics': metrics, 'elapsed_seconds': 1.,
                                       'input': {'sha256': 'placeholder'}, 'rows': 2, 'candidates': 4}})
            plan = {'seeds': [seed], 'candidate_seed': seed,
                    'analysis_seeds': {'table3_request_resampling': pipeline.derived_analysis_seed(seed)},
                    'data_root': str(root), 'candidate_manifest': str(root / 'candidate.json'), 'jobs': []}
            for training in summary.TRAINING:
                for model in summary.MODELS:
                    path = root / (model + '-' + training + '.json')
                    manifest = {'status': 'completed', 'seed': seed, 'train_type': training, 'model': model,
                                'initial_model_state_sha256': str(seed) + '-' + model,
                                'selection': {'protocol': training, 'file': {'sha256': 'hash-dev-' + training},
                                              'matched_candidate_profile': {'sha256': 'candidate-proof'} if training == 'rs50' else None},
                                'convergence': stop.state_dict(), 'epoch_records': copy.deepcopy(epochs),
                                'best_epoch': 1, 'epochs_completed': 6,
                                'evaluations': {p: {'prefix': model + '-' + training + '-' + p,
                                                   'metrics': dict.fromkeys(summary.METRICS, .4)} for p in summary.PROTOCOLS},
                                'timings_seconds': {'training': 12., 'dev_evaluation': 6., 'tests': dict.fromkeys(summary.PROTOCOLS, 3.)},
                                'execution_segments': [{'elapsed_seconds': 55.}], 'elapsed_seconds': 55.}
                    for record in manifest['epoch_records']:
                        record['dev']['input']['sha256'] = 'hash-dev-' + training
                    path.write_text(json.dumps(manifest), encoding='utf-8')
                    plan['jobs'].append({'seed': seed, 'train_type': training, 'model': model, 'success_file': str(path)})
            common = [('u1', 't', 'h1'), ('u2', 't', 'h2')]
            def artifact(model, training, protocol, prefix):
                report = {'input': {'sha256': 'hash-' + protocol}, 'metrics': dict.fromkeys(summary.METRICS, .4),
                          'rows': 2, 'candidates': 4, 'elapsed_seconds': 3.}
                return Artifact(model, training, protocol, Path(prefix), report, np.full((2, 4), .4),
                                {key: [i] for i, key in enumerate(common)},
                                {key: Counter({'N' + str(i): 1}) for i, key in enumerate(common)}, np.array([2, 2]), {})
            def development(data_root, supplied_dataset, training):
                return {'protocol': training, 'sha256': 'hash-dev-' + training, 'rows': 2, 'candidates': 4,
                        'candidate_profile': {'sha256': 'candidate-proof'} if training == 'rs50' else None}
            configpath = root / 'results/config.json'
            pipeline.save(configpath, {'training_seeds': [seed], 'candidate_seed': seed})
            plan['result_configuration'] = pipeline.fp(configpath)
            planpath = root / 'plan.json'
            planpath.write_text(json.dumps(plan), encoding='utf-8')
            output = root / 'tables'
            with mock.patch.object(summary, 'load_artifact', side_effect=artifact), \
                 mock.patch.object(summary, 'validate_metrics', return_value=({}, {})), \
                 mock.patch.object(summary, 'development_expectation', side_effect=development), \
                 mock.patch.object(sys, 'argv', ['summary', '--plan', str(planpath), '--output', str(output)]), \
                 contextlib.redirect_stdout(io.StringIO()):
                summary.main()
                wrong = dict(plan, candidate_seed=seed - 1)
                with self.assertRaisesRegex(ValueError, 'configuration seeds'):
                    summary.collect(wrong)
                pipeline.save(configpath, {'training_seeds': [42], 'candidate_seed': 42})
                with self.assertRaisesRegex(ValueError, 'configuration fingerprint'):
                    summary.collect(plan)
            audit = summary.read(output / 'validation.json')
            self.assertEqual((audit['seeds'], audit['training_runs'], audit['evaluations']), ([seed], 12, 60))
            self.assertEqual(audit['inputs']['candidate_seed'], seed)
            self.assertEqual(audit['table3_resample_seed'], plan['analysis_seeds']['table3_request_resampling']['seed'])
            resampling = summary.read(output / 'resampling_manifest.json')
            self.assertEqual(resampling['resample_seed'], audit['table3_resample_seed'])
            self.assertEqual(resampling['analysis_seed_provenance'], plan['analysis_seeds']['table3_request_resampling'])
            readme = (output / 'README.md').read_text(encoding='utf-8')
            self.assertIn('candidate/split seed ' + str(seed), readme)
            self.assertIn('paired training seeds: ' + str(seed), readme)
            self.assertIn('SD: NA; n=1', readme)
            self.assertNotIn('fixed candidate/split seed42', readme)


if __name__ == '__main__':
    unittest.main()
