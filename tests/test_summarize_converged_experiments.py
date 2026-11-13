from pathlib import Path
from types import SimpleNamespace
from collections import Counter
import json
import copy
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import summarize_converged_experiments as summary
from convergence_control import EarlyStopping, selection_value
from summarize_small_experiment import Artifact


class ConvergedSummaryTests(unittest.TestCase):
    def test_training_seed_sd_is_sample_sd(self):
        self.assertEqual(summary.sample_stats([1,2,3]),{'mean':2.0,'sample_sd':1.0,'n_seeds':3})
        self.assertEqual(summary.sample_stats([1]),{'mean':1.0,'sample_sd':None,'n_seeds':1})
        self.assertEqual(summary.sample_stats([None,None,None]),{'mean':None,'sample_sd':None,'n_seeds':0})
        with self.assertRaises(ValueError):
            summary.summarize_rows([{'key':'x','training_seed':42,'value':1.0}],('key',),('value',),(42,43,44))
        row=summary.summarize_rows([{'key':'x','training_seed':42,'value':1.0}],('key',),('value',),(42,))[0]
        self.assertIsNone(row['value_sample_sd'])
        self.assertIn('SD: NA; n=1',summary.format_stat(row,'value'))

    def test_seed_scope_must_be_explicit_valid_and_distinct(self):
        self.assertEqual(summary.declared_seeds({'seeds':[42]}),(42,))
        self.assertEqual(summary.declared_seeds({'seeds':[44,42,43]}),(42,43,44))
        self.assertEqual(summary.declared_seeds({'seeds':[3901234567]}),(3901234567,))
        self.assertEqual(summary.declared_seeds({'seeds':[42,43]}),(42,43))
        for values in (None,[],[42,42],['42'],[True],[-1],[2**32]):
            with self.assertRaises(ValueError):summary.declared_seeds({'seeds':values})

    def test_best_epoch_and_stopping_are_replayed(self):
        control=EarlyStopping()
        rows=[]
        for epoch,score in enumerate([.8,.7,.6,.5,.4,.3],1):
            metrics=dict.fromkeys(summary.METRICS,score)
            value=selection_value(metrics)
            control.update(epoch,value)
            rows.append({'epoch':epoch,'dev':{'metrics':metrics},'selection_value':value})
        manifest={'convergence':control.state_dict(),'epoch_records':rows,'best_epoch':1,'epochs_completed':6}
        self.assertEqual(summary.verify_convergence(manifest)['stop_reason'],'patience')
        with self.assertRaises(ValueError):
            summary.verify_convergence(dict(manifest,best_epoch=6))

    def test_alignment_must_not_silently_drop_groups_or_positives(self):
        key=('u','t','h')
        reference=SimpleNamespace(groups={key:[0]},positives={key:Counter({'N1':1})})
        wrong=SimpleNamespace(groups={key:[0]},positives={key:Counter({'N2':1})})
        with self.assertRaises(ValueError):
            summary.shared_groups(reference,wrong)
        self.assertEqual(summary.shared_groups(reference,reference)[0],[key])

    def test_initialization_aliases_are_strict(self):
        self.assertEqual(summary.initialization_hash({'initial_model_state_sha256':'a'}),'a')
        with self.assertRaises(ValueError):
            summary.initialization_hash({'initial_model_state_sha256':'a','initial_model_sha256':'b'})

    def test_complete_manifest_schema_and_wrong_validation_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            dataset={'status':'complete','fraction':1.0,'seed':42,'output_files':{},'statistics':{}}
            names={'imp':'behaviors.tsv','rs30':'behaviors_rs30.tsv','rs90':'behaviors_rs90.tsv','rs150':'behaviors_rs150.tsv'}
            for protocol,name in names.items():
                dataset['output_files']['test/'+name]={'sha256':'hash-'+protocol}
                dataset['statistics']['test/'+name]={'rows':2,'candidates':4}
            dataset['output_files']['dev/behaviors.tsv']={'sha256':'hash-dev'}
            dataset['statistics']['dev/behaviors.tsv']={'rows':2,'candidates':4}
            (root/'manifest.json').write_text(json.dumps(dataset),encoding='utf-8')
            candidate={'status':'complete','protocol':'total','dataset_manifest':{'sha256':summary.file_hash(root/'manifest.json')},
                       'output':{'sha256':'hash-total'},'statistics':{'rows':2,'candidates':4}}
            (root/'candidate.json').write_text(json.dumps(candidate),encoding='utf-8')
            control=EarlyStopping();epochs=[]
            for epoch,score in enumerate([.8,.7,.6,.5,.4,.3],1):
                metrics=dict.fromkeys(summary.METRICS,score)
                control.update(epoch,selection_value(metrics))
                epochs.append({'epoch':epoch,'train_seconds':2.0,'dev':{'metrics':metrics,'elapsed_seconds':1.0,
                               'input':{'sha256':'hash-dev'},'rows':2,'candidates':4}})
            plan={'seeds':[42,43,44],'data_root':str(root),'candidate_manifest':str(root/'candidate.json'),'jobs':[]}
            for seed in (42,43,44):
                for training in summary.TRAINING:
                    for model in summary.MODELS:
                        path=root/f'{seed}-{training}-{model}.json'
                        manifest={'status':'completed','seed':seed,'train_type':training,'model':model,
                                  'initial_model_state_sha256':f'{seed}-{model}',
                                  'selection':{'protocol':training,'file':{'sha256':'hash-dev-'+training},
                                               'matched_candidate_profile':({'sha256':'candidate-proof'} if training=='rs50' else None)},
                                  'convergence':control.state_dict(),'epoch_records':copy.deepcopy(epochs),'best_epoch':1,'epochs_completed':6,
                                  'evaluations':{p:{'prefix':f'{seed}-{training}-{model}-{p}','metrics':dict.fromkeys(summary.METRICS,.4)} for p in summary.PROTOCOLS},
                                  'timings_seconds':{'training':12.,'dev_evaluation':6.,'tests':dict.fromkeys(summary.PROTOCOLS,3.)},
                                  'execution_segments':[{'elapsed_seconds':55.}],'elapsed_seconds':55.}
                        for record in manifest['epoch_records']:
                            record['dev']['input']['sha256']='hash-dev-'+training
                        path.write_text(json.dumps(manifest),encoding='utf-8')
                        plan['jobs'].append({'seed':seed,'train_type':training,'model':model,'success_file':str(path)})
            keys=[('u1','t','h1'),('u2','t','h2')]
            def artifact(model,training,protocol,prefix):
                report={'input':{'sha256':'hash-'+protocol},'metrics':dict.fromkeys(summary.METRICS,.4),
                        'rows':2,'candidates':4,'elapsed_seconds':3.}
                return Artifact(model,training,protocol,Path(prefix),report,np.full((2,4),.4),
                                {key:[i] for i,key in enumerate(keys)},
                                {key:Counter({'N'+str(i):1}) for i,key in enumerate(keys)},np.array([2,2]),{})
            def development(data_root,dataset,training):
                return {'protocol':training,'sha256':'hash-dev-'+training,'rows':2,'candidates':4,
                        'candidate_profile':({'sha256':'candidate-proof'} if training=='rs50' else None)}
            with patch.object(summary,'load_artifact',side_effect=artifact),patch.object(summary,'validate_metrics',return_value=({},{})),patch.object(summary,'development_expectation',side_effect=development):
                grouped,common,audits,times,_=summary.collect(plan)
                self.assertEqual((len(grouped),len(audits),len(times)),(180,180,36))
                self.assertEqual(common,keys)
                self.assertTrue(all(row['training_seconds']==12 and row['all_test_seconds']==15 for row in times))
                single=dict(plan,seeds=[42],jobs=[j for j in plan['jobs'] if j['seed']==42])
                grouped,common,audits,times,_=summary.collect(single)
                self.assertEqual((len(grouped),len(audits),len(times)),(60,60,12))
                with self.assertRaises(ValueError):summary.collect(dict(single,jobs=plan['jobs']))
                path=Path(plan['jobs'][0]['success_file'])
                changed=json.loads(path.read_text(encoding='utf-8'))
                changed['epoch_records'][0]['dev']['input']['sha256']='wrong-validation'
                path.write_text(json.dumps(changed),encoding='utf-8')
                with self.assertRaises(ValueError):
                    summary.collect(plan)

    def test_full_total_reference_and_paired_seed_statistics(self):
        grouped={}
        common=[('u1','t','h1'),('u2','t','h2')]
        profiles={42:[.9,.8,.3,.25,.2,.1],43:[.5,.7,.3,.25,.2,.1],44:[.5,.4,.3,.25,.2,.1]}
        for seed in (42,43,44):
            for train in summary.TRAINING:
                for index,model in enumerate(summary.MODELS):
                    for protocol in summary.PROTOCOLS:
                        value=profiles[seed][index]+(.01 if train=='rs50' else 0)
                        if protocol=='imp': value=1-value
                        grouped[seed,train,model,protocol]=(np.full((2,4),value),np.ones(2,dtype=np.int64))
        tables=summary.compute_tables(grouped,common,(42,43,44))
        self.assertEqual(len(tables['samples']),10)
        for row in tables['table3_per_seed']:
            self.assertEqual(row['reference'],'total')
            self.assertEqual(row['comparisons'],150)
            self.assertEqual(row['matching_percent'],0 if row['protocol']=='imp' else 100)
        for row in tables['table5_summary']:
            self.assertAlmostEqual(row['paired_difference_mean'],.01)
            self.assertEqual(row['paired_difference_n_seeds'],3)
        row=next(r for r in tables['table2_summary'] if r['train_type']=='imp' and r['aggregation']=='native_entry_macro'
                 and r['protocol']=='total' and r['metric']=='AUC' and r['model']=='NAML')
        self.assertEqual(row['rank_of_mean_score'],1.5)
        self.assertNotEqual(row['rank_mean'],row['rank_of_mean_score'])
        single={key:value for key,value in grouped.items() if key[0]==42}
        single_tables=summary.compute_tables(single,common,(42,))
        for name in ('table2_summary','table3_summary','table5_summary'):
            for item in single_tables[name]:
                for key,value in item.items():
                    if key.endswith('_sample_sd'):self.assertIsNone(value)
                    if key.endswith('_n_seeds'):self.assertEqual(value,1)
        with self.assertRaises(ValueError):summary.compute_tables(single,common,(42,43,44))
        del grouped[42,'imp','NAML','total']
        with self.assertRaises(ValueError):
            summary.compute_tables(grouped,common,(42,43,44))

    def test_single_seed_main_reports_actual_scope_and_na_sd(self):
        common=[('u1','t','h1'),('u2','t','h2')]
        grouped={(42,t,m,p):(np.full((2,4),.1+i*.1),np.ones(2,dtype=np.int64))
                 for t in summary.TRAINING for i,m in enumerate(summary.MODELS) for p in summary.PROTOCOLS}
        timing=[]
        for training in summary.TRAINING:
            for model in summary.MODELS:
                timing.append({'training_seed':42,'train_type':training,'model':model,'stop_reason':'patience',
                               'training_seconds':12.,'validation_seconds':6.,
                               **{p+'_test_seconds':3. for p in summary.PROTOCOLS},'all_test_seconds':15.,
                               'training_plus_all_tests_seconds':27.,'active_runner_seconds':40.})
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);plan=root/'plan.json';output=root/'summary'
            plan.write_text(json.dumps({'seeds':[42]}),encoding='utf-8')
            with patch.object(sys,'argv',['summary','--plan',str(plan),'--output',str(output)]),\
                 patch.object(summary,'collect',return_value=(grouped,common,[{}]*60,timing,{})),\
                 patch.object(summary,'net_timing_rows',return_value=[dict(training_seed=row['training_seed'], train_type=row['train_type'], model=row['model'], **summary.validated_net_timing({})) for row in timing]):
                summary.main()
            audit=json.loads((output/'validation.json').read_text(encoding='utf-8'))
            self.assertEqual((audit['seeds'],audit['training_runs'],audit['evaluations']),([42],12,60))
            readme=(output/'README.md').read_text(encoding='utf-8')
            self.assertIn('12/12 training runs and 60/60 evaluations',readme)
            self.assertIn('single-seed results',readme)
            self.assertNotIn('three training seeds',readme)
            self.assertIn('SD: NA; n=1',readme)
            self.assertTrue((output/'timings_net_per_seed.csv').is_file())
            net_text=(output/'timings_net_per_seed.csv').read_text(encoding='utf-8')
            self.assertIn('unknown_legacy_net_time',net_text)
            self.assertIn('NA',net_text)


if __name__=='__main__':
    unittest.main()
