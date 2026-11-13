"""Cold-start chronology/lazy-SAG guards using synthetic files and mocked workers."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest import mock
import unittest

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
import run_mind_pipeline as pipeline
import run_converged_digat as digat
import prepare_digat_assets as builder


def args(work,*extra):
    return pipeline.parser().parse_args(['--work-dir',str(work),'--dry-run',*extra])


def prepared_controller(root,models=('all6',)):
    work=root/'work'
    plan={'options':pipeline.normalize(args(work,'--models',*models)),
          'runtime':str(work/'runtime'),'controller_sha256':pipeline.sha(pipeline.__file__),'source_files':[]}
    (work/'control').mkdir(parents=True)
    pipeline.save(work/'control/pipeline_plan.json',plan)
    return pipeline.Pipeline(plan),plan


class ChronologicalPipelineTests(unittest.TestCase):
    def test_default_is_chronological_and_no_unneeded_graph(self):
        with tempfile.TemporaryDirectory()as tmp:
            o=pipeline.normalize(args(Path(tmp)/'work'))
            self.assertEqual(o['split_policy'],'chronological')
            self.assertEqual(o['context_feature_mode'],'context_not_used_by_paper_models')
            self.assertIsNone(o['wikidata_graph'])
            raw=pipeline.phase_commands({'options':o,'runtime':str(Path(tmp)/'runtime')})['raw']
            self.assertEqual(raw[raw.index('--split-policy')+1],'chronological')
            self.assertEqual(raw[raw.index('--context-feature-mode')+1],o['context_feature_mode'])
            self.assertNotIn('--wikidata-graph',raw)
            self.assertEqual(raw[raw.index('--selected-models')+1:raw.index('--selected-models')+2],['NAML'])
    def test_supplied_real_graph_selects_required_context_route(self):
        with tempfile.TemporaryDirectory()as tmp:
            o=pipeline.normalize(args(Path(tmp)/'work','--wikidata-graph',str(Path(tmp)/'graph.zip')))
            self.assertEqual(o['context_feature_mode'],'required')
            raw=pipeline.phase_commands({'options':o,'runtime':str(Path(tmp)/'runtime')})['raw']
            self.assertIn('--wikidata-graph',raw)
    def test_explicit_unused_context_refuses_graph_or_checksum(self):
        with tempfile.TemporaryDirectory()as tmp:
            work=Path(tmp)/'work'
            for extra in (['--wikidata-graph',str(Path(tmp)/'graph.zip')],['--wikidata-sha256','a'*64]):
                with self.assertRaisesRegex(ValueError,'takes no graph'):
                    pipeline.normalize(args(work,'--context-feature-mode','context_not_used_by_paper_models',*extra))
    def test_split_policy_is_part_of_resume_identity(self):
        with tempfile.TemporaryDirectory()as tmp:
            work=Path(tmp)/'work';old=pipeline.normalize(args(work))
            self.assertEqual(pipeline.normalize(args(work),old),old)
            with self.assertRaisesRegex(ValueError,'split_policy'):
                pipeline.normalize(args(work,'--split-policy','random_groups'),old)
    def test_chronological_fraction_cannot_tune_holdout(self):
        with tempfile.TemporaryDirectory()as tmp:
            with self.assertRaisesRegex(ValueError,'last official training day'):
                pipeline.normalize(args(Path(tmp)/'work','--dev-fraction','.1'))
    def test_mirror_catalog_is_the_maintained_recommenders_source(self):
        self.assertIn('recommenders-team/recommenders',pipeline.MIND_DOWNLOAD_CATALOG)
        for name in ('train','dev'):
            self.assertIn('/datasets/Recommenders/MIND/',pipeline.DOWNLOADS[name][1])


class LazySAGTests(unittest.TestCase):
    def test_all_ten_nnr_conditions_run_before_first_sag_and_completed_jobs_skip(self):
        with tempfile.TemporaryDirectory()as tmp:
            instance,plan=prepared_controller(Path(tmp));events=[];built=[]
            jobs=pipeline.job_commands(plan)
            def ensure(experiments):
                if not built:events.append('SAG');built.append(True)
            def command(command,name,**kwargs):
                events.append(name)
                run=Path(command[command.index('--run-dir')+1]);run.mkdir(parents=True)
                pipeline.save(run/'experiment.json',{'status':'completed','initial_model_state_sha256':'a'*64})
            instance.ensure_digat_assets=ensure;instance.command=command
            proof={'initial_model_sha256':'a'*64}
            with mock.patch.object(pipeline,'complete_job',return_value=proof):
                instance.run_jobs({'jobs':jobs})
                self.assertEqual(events[:10],[j['name']for j in jobs[:10]])
                self.assertEqual(events[10],'SAG')
                self.assertEqual(events[11:],[j['name']for j in jobs[10:]])
                self.assertEqual(len(built),1)
                instance.ensure_digat_assets=lambda x:self.fail('Completed DIGAT must not rebuild SAG')
                instance.command=lambda *a,**k:self.fail('Completed models must not execute again')
                instance.run_jobs({'jobs':jobs})
            self.assertEqual(len(instance.state['jobs']),12)
    def test_main_seal_needs_pretrained_weights_but_not_unbuilt_sag(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);instance,plan=prepared_controller(root);work=instance.work;o=plan['options']
            data=work/'dataset';paths={};stats={}
            for kind in pipeline.PROTOCOLS:
                name='test/behaviors_'+kind+'.tsv';path=data/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('synthetic')
                paths[kind]=path;stats[name]={'rows':1,'candidates':2,'positive_labels':1}
            for name in ('dev/behaviors.tsv','dev/behaviors_rs50.tsv'):
                path=data/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('synthetic')
            pipeline.save(data/'manifest.json',{'statistics':stats})
            glove=Path(o['embedding_root']);glove.mkdir(parents=True);(glove/'glove.6B.300d.txt').write_text('vectors')
            semantic=Path(o['sag_model']);semantic.mkdir(parents=True);weights=semantic/'weights.bin';weights.write_bytes(b'weights')
            cache=Path(o['nnr_cache_dir']);cache.mkdir();(cache/'vectors.pt').write_bytes(b'CPU cache')
            fake=SimpleNamespace(validate_consistent_candidate_profile=lambda d:{'profile':'synthetic'},evaluation_paths=lambda d,p:paths)
            with mock.patch.dict(sys.modules,{'consistent_candidate_profile':fake}):
                seal=instance.seal_experiments()
            self.assertFalse(Path(o['digat_assets']).exists())
            self.assertTrue(any(row['path']==str(weights)for row in seal['input_files']))
            self.assertFalse(any(Path(o['digat_assets'])in Path(row['path']).parents for row in seal['input_files']))
            self.assertEqual(seal['derived_asset_policy'],'lazy_DIGAT_separate_immutable_seal_v1')
    def make_derived_fixture(self,root):
        instance,plan=prepared_controller(root);work=instance.work;semantic=Path(plan['options']['sag_model']);semantic.mkdir(parents=True)
        weight=semantic/'weights.bin';weight.write_bytes(b'pinned pretrained weights')
        experiments={'input_files':[pipeline.fp(weight)]};pipeline.save(work/'control/experiment_plan.json',experiments)
        graph=Path(plan['options']['digat_assets']);calls=[]
        def command(cmd,name,**kw):
            calls.append((cmd,name));self.assertEqual(name,'digat_graph_assets');self.assertIn('--cooling',cmd)
            graph.mkdir(parents=True);(graph/'news.json').write_text('{}');(graph/'graph.pkl').write_bytes(b'synthetic graph')
            pipeline.save(graph/'asset_prep.json',{'status':'complete','identity':'derived-id'})
        instance.command=command
        validated={'identity':'derived-id','files':{'news.json':{},'graph.pkl':{}}}
        return instance,experiments,calls,weight,graph,validated
    def test_separate_immutable_derived_seal_and_no_duplicate_sag(self):
        with tempfile.TemporaryDirectory()as tmp:
            instance,experiments,calls,weight,graph,validated=self.make_derived_fixture(Path(tmp))
            original=(instance.control/'experiment_plan.json').read_bytes()
            with mock.patch.object(digat,'validate_assets',return_value=validated):
                first=instance.ensure_digat_assets(experiments)
                second=instance.ensure_digat_assets(experiments)
            self.assertEqual(first,second);self.assertEqual(len(calls),1)
            self.assertEqual((instance.control/'experiment_plan.json').read_bytes(),original)
            self.assertTrue((instance.control/'digat_asset_seal.json').is_file())
            (graph/'graph.pkl').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Pinned file'):
                instance.ensure_digat_assets(experiments)
            self.assertEqual(len(calls),1)
    def test_changed_model_weights_refuse_sag_before_backend(self):
        with tempfile.TemporaryDirectory()as tmp:
            instance,experiments,calls,weight,graph,validated=self.make_derived_fixture(Path(tmp))
            weight.write_bytes(b'changed pretrained weights')
            with self.assertRaisesRegex(ValueError,'Pinned file'):
                instance.ensure_digat_assets(experiments)
            self.assertFalse(calls);self.assertFalse(graph.exists())


class NativeAndRestTests(unittest.TestCase):
    def test_fresh_consistent_imp_is_allowed_but_unpaired_rands_is_rejected(self):
        args=SimpleNamespace(seed=42,train_type='imp',paired_preprocessing_from=None)
        profile={'profile':'complete_proxy_strict_request_consistent_v1'}
        self.assertIsNone(digat.paired_imp_reference(args,{},profile))
        args.train_type='rs50'
        with self.assertRaisesRegex(ValueError,'completed paired DIGAT IMP'):
            digat.paired_imp_reference(args,{},profile)
    def test_shared_asset_boundary_preserves_result_and_runs_after_success(self):
        events=[];expected=object()
        function=builder.after_completed_work(lambda *a,**k:(events.append('work'),expected)[1],lambda:events.append('rest_boundary'))
        self.assertIs(function('unchanged argument',option=True),expected)
        self.assertEqual(events,['work','rest_boundary'])
    def test_failed_asset_work_never_takes_a_rest_before_error(self):
        events=[]
        def fail():events.append('work');raise ValueError('failed')
        function=builder.after_completed_work(fail,lambda:events.append('rest_boundary'))
        with self.assertRaises(ValueError):function()
        self.assertEqual(events,['work'])


if __name__=='__main__':unittest.main(verbosity=2)
