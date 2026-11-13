"""CPU-only tests for the public pipeline; synthetic data and mocked workers only."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from unittest import mock
import unittest
import urllib.error
import zipfile

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
import run_mind_pipeline as pipeline
import prepare_nnr_embedding_cache as cache


def arguments(work,*extra):
    return pipeline.parser().parse_args(['--work-dir',str(work),'--dry-run',*extra])


def plan_for(work,models=None):
    options=pipeline.normalize(arguments(work,'--seed','42',*(['--models',*models]if models else[])))
    return {'options':options,'runtime':str(work/'runtime'),'controller_sha256':pipeline.sha(pipeline.__file__),'source_files':[]}


class Response(io.BytesIO):
    def geturl(self):return 'https://cdn.example/archive.zip?signature=secret'


def zip_bytes():
    data=io.BytesIO()
    with zipfile.ZipFile(data,'w')as archive:archive.writestr('x.txt','tiny synthetic archive')
    return data.getvalue()


class PipelineTests(unittest.TestCase):
    def test_declared_seed_is_shared_by_two_fresh_naml_conditions(self):
        with tempfile.TemporaryDirectory()as temporary:
            plan=plan_for(Path(temporary)/'work');jobs=pipeline.job_commands(plan)
            self.assertEqual([(j['model'],j['train_type'])for j in jobs],[('NAML','imp'),('NAML','rs50')])
            self.assertNotIn('--paired-preprocessing-from',jobs[0]['command'])
            self.assertIn('--paired-preprocessing-from',jobs[1]['command'])
            for job in jobs:
                command=job['command'];self.assertIn('--cooling',command)
                self.assertEqual(command[command.index('--dev-protocol')+1],job['train_type'])
                self.assertEqual(command[command.index('--embedding-root')+1],plan['options']['nnr_cache_dir'])
                self.assertEqual(command[command.index('--max-epochs')+1],'50')
                self.assertEqual(command[command.index('--min-epochs')+1],'5')
                self.assertEqual(command[command.index('--patience')+1],'5')
                self.assertNotIn('--resume',command);self.assertNotIn('--batch-size',command)
                self.assertIn('--full-total-behaviors',command)
    def test_all6_has12_conditions_digat_uses_native_assets_and_full_total(self):
        with tempfile.TemporaryDirectory()as temporary:
            plan=plan_for(Path(temporary)/'work',['all6']);jobs=pipeline.job_commands(plan)
            self.assertEqual(len(jobs),12);self.assertEqual(len({j['name']for j in jobs}),12)
            for index,job in enumerate(jobs):
                self.assertEqual(job['train_type'],('imp','rs50')[index%2])
                if job['model']=='DIGAT':
                    cmd=job['command'];self.assertIn('--shared-preprocessing',cmd);self.assertIn('--full-total',cmd)
                    self.assertEqual(cmd[cmd.index('--embedding-root')+1],plan['options']['embedding_root'])
    def test_all5_and_custom_models_are_distinct_and_ordered(self):
        self.assertEqual(pipeline.resolve_models(['all5']),list(pipeline.NNR_MODELS))
        self.assertEqual(pipeline.resolve_models(['DIGAT','NAML']),['NAML','DIGAT'])
        for values in ([],['NAML','NAML'],['all6','NAML'],['invalid']):
            with self.assertRaises(ValueError):pipeline.resolve_models(values)
    def test_seed_and_license_and_pinned_semantic_defaults(self):
        with tempfile.TemporaryDirectory()as temporary:
            work=Path(temporary)/'work'
            self.assertEqual(pipeline.normalize(arguments(work))['semantic_revision'],pipeline.SEMANTIC_REVISION)
            self.assertEqual(pipeline.normalize(arguments(work))['seed'],'random')
            self.assertEqual(pipeline.normalize(arguments(work,'--seed','43'))['seed'],43)
            with self.assertRaisesRegex(ValueError,'accept-mind-license'):pipeline.normalize(arguments(work,'--download'))
            self.assertTrue(pipeline.normalize(arguments(work,'--download','--accept-mind-license'))['download'])
    def test_work_cannot_be_inside_or_ancestor_of_repository(self):
        for path in (pipeline.REPO,pipeline.REPO/'work',pipeline.REPO.parent):
            with self.assertRaises(ValueError):pipeline.normalize(arguments(path))
    def test_resume_reloads_options_but_rejects_protocol_changes(self):
        with tempfile.TemporaryDirectory()as temporary:
            work=Path(temporary)/'work';old=pipeline.normalize(arguments(work,'--models','all5','--glove-text',str(Path(temporary)/'glove.6B.300d.txt')))
            self.assertEqual(pipeline.normalize(arguments(work),old),old)
            with self.assertRaisesRegex(ValueError,'models'):pipeline.normalize(arguments(work,'--models','NAML'),old)
    def test_raw_commands_receive_observed_input_sha_and_local_archives(self):
        with tempfile.TemporaryDirectory()as temporary:
            plan=plan_for(Path(temporary)/'work');plan['options'].update(context_feature_mode='required',wikidata_graph=str(Path(temporary)/'graph.zip'));pins={'train_zip':{'sha256':'a'*64},'dev_zip':{'sha256':'b'*64}}
            commands=pipeline.phase_commands(plan,pins);raw=commands['raw']
            self.assertEqual(raw[raw.index('--expected-train-sha256')+1],'a'*64)
            self.assertEqual(raw[raw.index('--expected-dev-sha256')+1],'b'*64)
            self.assertIn('--wikidata-graph',raw);self.assertIn('--source-data-root',commands['candidates'])
    def test_source_freeze_checks_source_changes_and_preserves_originals(self):
        with tempfile.TemporaryDirectory()as temporary:
            root=Path(temporary);repo=root/'repo';scripts=repo/'scripts';scripts.mkdir(parents=True)
            required=['prepare_mind_masters.py','prepare_consistent_candidates.py','prepare_model_assets.py','prepare_nnr_embedding_cache.py',
                'context_embeddings.py','context_feature_policy.py','candidate_policy.py','consistent_candidate_profile.py','run_converged_nnr.py','summarize_converged_experiments.py']
            for name in required:(scripts/name).write_text('# synthetic source\n')
            work=root/'work'
            with mock.patch.object(pipeline,'REPO',repo):
                plan=pipeline.create_plan(pipeline.normalize(arguments(work)))
                pipeline.verify_source(plan)
                original=(scripts/'candidate_policy.py').read_bytes()
                target=work/'runtime/scripts/candidate_policy.py';target.write_text('# modified')
                with self.assertRaisesRegex(ValueError,'Pinned file'):pipeline.verify_source(plan)
                self.assertEqual((scripts/'candidate_policy.py').read_bytes(),original)
    def test_freeze_rejects_data_in_source_tree(self):
        with tempfile.TemporaryDirectory()as temporary:
            root=Path(temporary);(root/'scripts').mkdir();(root/'scripts/weights.pt').write_bytes(b'no')
            with self.assertRaisesRegex(ValueError,'clean source'):pipeline.source_inventory(root)
    def test_download_receipt_observed_sha_and_signed_query_redaction(self):
        with tempfile.TemporaryDirectory()as temporary:
            data=zip_bytes();out=Path(temporary)/'train.zip'
            receipt=pipeline.download_official(pipeline.DOWNLOADS['train'][1],out,opener=lambda*args,**kw:Response(data))
            self.assertEqual(out.read_bytes(),data);self.assertFalse(receipt['expected_checksum_verified'])
            self.assertEqual(receipt['file']['sha256'],hashlib.sha256(data).hexdigest())
            self.assertNotIn('secret',json.dumps(receipt));self.assertEqual(receipt['source_kind'],'maintained Microsoft Recommenders mirror')
            with self.assertRaises(FileExistsError):pipeline.download_official(pipeline.DOWNLOADS['train'][1],out,opener=lambda*a,**k:self.fail())
    def test_download_checksum_failure_preserves_partial_not_destination(self):
        with tempfile.TemporaryDirectory()as temporary:
            out=Path(temporary)/'train.zip'
            with self.assertRaisesRegex(ValueError,'checksum'):pipeline.download_official(pipeline.DOWNLOADS['train'][1],out,expected_sha256='0'*64,opener=lambda*a,**k:Response(zip_bytes()))
            self.assertFalse(out.exists());self.assertTrue(list(out.parent.glob('*.partial-*')))
    def test_download_fallback_works_without_hardlinks_and_never_replaces(self):
        with tempfile.TemporaryDirectory()as temporary:
            root=Path(temporary);part=root/'part';dest=root/'archive';part.write_bytes(b'bytes')
            with mock.patch.object(pipeline.os,'link',side_effect=OSError('unsupported')):
                mode=pipeline.publish_exclusive(part,dest)
            self.assertEqual(dest.read_bytes(),b'bytes');self.assertFalse(part.exists())
            self.assertIn(mode,('atomic_windows_nonreplace_rename','exclusive_copy_nonatomic_preserve_incomplete_on_failure'))
            part.write_bytes(b'other')
            with mock.patch.object(pipeline.os,'link',side_effect=OSError('unsupported')):
                with self.assertRaises(FileExistsError):pipeline.publish_exclusive(part,dest)
            self.assertEqual(dest.read_bytes(),b'bytes');self.assertEqual(part.read_bytes(),b'other')
    def test_download_http401_reports_local_inputs_without_credentials(self):
        def fail(*a,**k):raise urllib.error.HTTPError('https://x?token=secret',401,'gated',{},None)
        with tempfile.TemporaryDirectory()as temporary:
            with self.assertRaisesRegex(RuntimeError,'--train-zip')as caught:
                pipeline.download_official(pipeline.DOWNLOADS['dev'][1],Path(temporary)/'dev.zip',opener=fail)
            self.assertNotIn('secret',str(caught.exception))
    def test_hf_token_is_unredirected_and_not_in_receipts(self):
        captured=[]
        def opener(request,**kw):captured.append(request);return Response(zip_bytes())
        with tempfile.TemporaryDirectory()as temporary,mock.patch.dict(pipeline.os.environ,{'HF_TOKEN':'private-secret'}):
            receipt=pipeline.download_official(pipeline.DOWNLOADS['dev'][1],Path(temporary)/'dev.zip',opener=opener)
            self.assertNotIn('Authorization',captured[0].headers)
            self.assertEqual(captured[0].unredirected_hdrs['Authorization'],'Bearer private-secret')
            self.assertNotIn('private-secret',json.dumps(receipt))
    def test_unapproved_urls_are_rejected_before_network(self):
        with tempfile.TemporaryDirectory()as temporary:
            with self.assertRaises(ValueError):pipeline.download_official('https://example.com/private',Path(temporary)/'file',opener=lambda*a,**k:self.fail())
    def test_asset_commands_registered_glove_and_semantic_are_revalidated(self):
        with tempfile.TemporaryDirectory()as temporary:
            work=Path(temporary)/'work';plan=plan_for(work,['all6']);(work/'control').mkdir(parents=True)
            pipeline.save(work/'control/pipeline_plan.json',plan);glove=Path(plan['options']['embedding_root']);glove.mkdir(parents=True)
            (glove/'glove.6B.300d.txt').write_text('synthetic');(glove/'GLOVE_ASSET_MANIFEST.json').write_text('{}')
            semantic=Path(plan['options']['sag_model']);semantic.mkdir(parents=True);(semantic/'TORS_ASSET_MANIFEST.json').write_text('{}')
            instance=pipeline.Pipeline(plan);calls=[];instance.command=lambda cmd,name,**kw:calls.append((cmd,name))
            instance.assets(semantic=True)
            self.assertEqual(len(calls),2);self.assertTrue(all('--check-only'in cmd for cmd,name in calls))
    def test_generic_result_summary_exports_net_times_and_marks_single_seed(self):
        with tempfile.TemporaryDirectory()as temporary:
            root=Path(temporary);manifest=root/'experiment.json';protocols={p:{'metrics':{'AUC':.6,'MRR':.3,'nDCG@5':.4,'nDCG@10':.5}}for p in pipeline.PROTOCOLS}
            pipeline.save(manifest,{'model':'NAML','train_type':'rs50','seed':42,'effective_config':{'seed':42},'evaluations':protocols,'best_epoch':1,'epochs_completed':6})
            job={'model':'NAML','train_type':'rs50','seed':42,'manifest':str(manifest),'name':'NAML_rs50'}
            net={'training':60,'dev_evaluation':10,'total':100,'cooling_rest':120,'tests':{p:2 for p in pipeline.PROTOCOLS}}
            proof={'NAML_rs50':{'timings_seconds_net':net}}
            result=pipeline.write_selected_results(root/'results',[job],proof)
            self.assertEqual(set(result),{'README.md','config.json','metrics.csv','timings_net.csv'})
            self.assertEqual(len((root/'results/metrics.csv').read_text().splitlines()),6)
            self.assertIn('Table 3 requires all six',(root/'results/README.md').read_text())
            self.assertIn('NA',(root/'results/timings_net.csv').read_text())
    def test_prepare_only_command_modes_are_explicit(self):
        with mock.patch.object(sys,'stderr',io.StringIO()),self.assertRaises(SystemExit):pipeline.parser().parse_args(['--work-dir','work'])
        with tempfile.TemporaryDirectory()as temporary:
            work=Path(temporary)/'not-created'
            with mock.patch.object(sys,'argv',['run_mind_pipeline.py','--work-dir',str(work),'--dry-run']),mock.patch('builtins.print'):
                pipeline.main()
            self.assertFalse(work.exists())


class PrivateCacheTests(unittest.TestCase):
    def test_clean_cache_build_is_pinned_and_reused_without_loading(self):
        with tempfile.TemporaryDirectory()as temporary:
            root=Path(temporary);asset=root/'source';asset.mkdir();source=asset/cache.NAME;source.write_text('synthetic text');output=root/'cache'
            calls=[]
            def loader(directory):calls.append(directory);(directory/(cache.CACHE_NAME)).write_bytes(b'synthetic CPU cache')
            original=source.read_bytes();receipt=cache.build(source,output,loader=loader)
            self.assertEqual(len(calls),1);self.assertFalse(receipt['model_loaded_or_trained']);self.assertEqual(source.read_bytes(),original)
            cache.build(source,output,loader=lambda path:self.fail('Completed cache must not be rebuilt'))
            (output/(cache.CACHE_NAME)).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'fingerprint'):cache.build(source,output,check_only=True)
    def test_existing_unregistered_cache_is_preserved(self):
        with tempfile.TemporaryDirectory()as temporary:
            root=Path(temporary);asset=root/'source';asset.mkdir();source=asset/cache.NAME;source.write_text('x');output=root/'cache';output.mkdir();(output/'marker').write_text('keep')
            with self.assertRaisesRegex(ValueError,'Preserve incomplete'):cache.build(source,output,loader=lambda p:self.fail())
            self.assertEqual((output/'marker').read_text(),'keep')
    def test_missing_cache_readonly_check_creates_nothing(self):
        with tempfile.TemporaryDirectory()as temporary:
            root=Path(temporary);asset=root/'source';asset.mkdir();source=asset/cache.NAME;source.write_text('x');output=root/'cache'
            with self.assertRaises(FileNotFoundError):cache.build(source,output,check_only=True)
            self.assertFalse(output.exists())
    def test_source_mutation_during_construction_is_rejected(self):
        with tempfile.TemporaryDirectory()as temporary:
            root=Path(temporary);asset=root/'source';asset.mkdir();source=asset/cache.NAME;source.write_text('x');output=root/'cache'
            def bad(directory):source.write_text('changed');(directory/(cache.CACHE_NAME)).write_bytes(b'cache')
            with self.assertRaisesRegex(ValueError,'source changed'):cache.build(source,output,loader=bad)
            self.assertFalse((output/cache.RECEIPT).exists())


if __name__=='__main__':unittest.main()
