"""CPU lifecycle tests for a controller executed from its real sealed snapshot."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import uuid

SOURCE=Path(__file__).resolve().parents[1]/'scripts/run_mind_pipeline.py'
REQUIRED=('prepare_mind_masters.py','prepare_consistent_candidates.py','prepare_model_assets.py',
 'prepare_nnr_embedding_cache.py','context_embeddings.py','context_feature_policy.py','candidate_policy.py',
 'consistent_candidate_profile.py','run_converged_nnr.py','summarize_converged_experiments.py')


def load(path):
    spec=importlib.util.spec_from_file_location('pipeline_fixture_'+uuid.uuid4().hex,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def snapshot(root):
    return {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*')if p.is_file()}


class FrozenResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.repo=self.root/'checkout';scripts=self.repo/'scripts'
        scripts.mkdir(parents=True)
        for name in REQUIRED:(scripts/name).write_text('# Synthetic helper fixture.\n',encoding='utf-8')
        shutil.copy2(SOURCE,scripts/'run_mind_pipeline.py')
        self.pipeline=load(scripts/'run_mind_pipeline.py');self.work=self.root/'study'
        args=self.pipeline.parser().parse_args(['--work-dir',str(self.work),'--dry-run'])
        options=self.pipeline.normalize(args)
        with mock.patch.object(self.pipeline.secrets,'randbits',return_value=1273813)as entropy:
            self.plan=self.pipeline.create_plan(options)
        entropy.assert_called_once_with(32)
        self.controller=self.work/'runtime/scripts/run_mind_pipeline.py'
        self.frozen=load(self.controller)
    def args(self,*extra,work=None):
        return self.frozen.parser().parse_args(['--work-dir',str(work or self.work),'--dry-run','--resume',*extra])
    def persist(self,plan):
        self.frozen.save(self.work/'control/pipeline_plan.json',plan)
    def normalize(self,plan=None,*extra):
        plan=self.plan if plan is None else plan
        return self.frozen.normalize(self.args(*extra),plan['options'],sealed_plan=plan)
    def test_fresh_source_overlap_is_still_rejected(self):
        for work in (self.repo,self.repo/'data',self.repo.parent):
            args=self.pipeline.parser().parse_args(['--work-dir',str(work),'--dry-run'])
            with self.assertRaisesRegex(ValueError,'outside'):
                self.pipeline.normalize(args)
    def test_real_frozen_cli_resume_preview_preserves_every_file_and_seed(self):
        before=snapshot(self.root)
        result=subprocess.run([sys.executable,'-B',str(self.controller),'--work-dir',str(self.work),
            '--dry-run','--resume','--seed','random'],capture_output=True,text=True,encoding='utf-8',
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8'),cwd=str(self.root))
        self.assertEqual(result.returncode,0,result.stderr)
        preview=json.loads(result.stdout)
        self.assertEqual(preview['options'],self.plan['options'])
        self.assertFalse(preview['files_written']);self.assertFalse(preview['GPU_or_model_test_performed'])
        self.assertEqual(snapshot(self.root),before)
    def test_copied_main_execute_resume_uses_existing_plan_and_original_workers(self):
        before=snapshot(self.root);fake=mock.Mock()
        with mock.patch.object(self.frozen.sys,'argv',[str(self.controller),'--work-dir',str(self.work),'--execute','--resume']),\
             mock.patch.object(self.frozen,'Pipeline',return_value=fake)as runner,\
             mock.patch.object(self.frozen,'create_plan',side_effect=AssertionError('Do not reseal')),\
             mock.patch.object(self.frozen.secrets,'randbits',side_effect=AssertionError('Do not redraw')):
            self.frozen.main()
        runner.assert_called_once_with(self.plan,resume=True)
        fake.execute.assert_called_once_with(prepare_only=False)
        jobs=self.frozen.job_commands(self.plan)
        self.assertTrue(all(str(self.work/'runtime/scripts')in j['command'][3]for j in jobs))
        self.assertEqual(snapshot(self.root),before)
    def test_check_only_from_frozen_source_stays_read_only(self):
        before=snapshot(self.root)
        with mock.patch.object(self.frozen.sys,'argv',[str(self.controller),'--work-dir',str(self.work),'--check-only']),\
             mock.patch.object(self.frozen,'environment_report',return_value={'GPU_test_performed':False}),\
             mock.patch.object(self.frozen.secrets,'randbits',side_effect=AssertionError('Do not redraw')),\
             contextlib.redirect_stdout(io.StringIO())as output:
            self.frozen.main()
        report=json.loads(output.getvalue())
        self.assertTrue(report['frozen_source_verified']);self.assertFalse(report['prepared_dataset_present'])
        self.assertEqual(snapshot(self.root),before)
    def test_no_plan_exception_for_arbitrary_nested_source(self):
        nested=self.work/'arbitrary/scripts';nested.mkdir(parents=True)
        shutil.copy2(SOURCE,nested/'run_mind_pipeline.py');module=load(nested/'run_mind_pipeline.py')
        with self.assertRaisesRegex(ValueError,'original sealed runtime'):
            module.normalize(self.args(),self.plan['options'],sealed_plan=self.plan)
        with self.assertRaisesRegex(ValueError,'original sealed runtime'):
            self.frozen.normalize(self.args(),self.plan['options'])
    def test_wrong_runtime_or_work_declaration_is_rejected(self):
        for field,value in [('runtime',str(self.work/'another_runtime')),('work_dir',str(self.root/'other_study'))]:
            plan=copy.deepcopy(self.plan)
            if field=='runtime':plan[field]=value
            else:plan['options'][field]=value
            self.persist(plan)
            with self.assertRaises(ValueError):self.normalize(plan)
    def test_on_disk_plan_mismatch_is_rejected(self):
        plan=copy.deepcopy(self.plan);plan['scope']='Different supplied plan'
        with self.assertRaisesRegex(ValueError,'on-disk plan'):self.normalize(plan)
    def test_changed_frozen_helper_is_rejected(self):
        (self.work/'runtime/scripts/candidate_policy.py').write_text('# Changed after sealing.\n',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'Pinned file'):self.normalize()
    def test_changed_frozen_controller_is_rejected(self):
        with self.controller.open('a',encoding='utf-8')as out:out.write('\n# Changed after sealing.\n')
        self.frozen=load(self.controller)
        with self.assertRaisesRegex(ValueError,'Pipeline control source changed'):self.normalize()
    def test_inventory_escape_and_absent_controller_are_rejected(self):
        for change in ('traversal','empty','no_controller'):
            plan=copy.deepcopy(self.plan)
            if change=='traversal':plan['source_files'][0]['relative']='../escape.py'
            elif change=='empty':plan['source_files']=[]
            else:plan['source_files']=[p for p in plan['source_files']if p['relative']!='scripts/run_mind_pipeline.py']
            self.persist(plan)
            with self.assertRaises(ValueError):self.normalize(plan)
    def test_state_fingerprint_blocks_changed_sealed_options(self):
        self.frozen.save(self.work/'control/pipeline_state.json',{'plan_sha256':self.frozen.sha(self.work/'control/pipeline_plan.json')})
        plan=copy.deepcopy(self.plan);plan['options']['seed']=99421;self.persist(plan)
        with self.assertRaisesRegex(ValueError,'different plan'):self.normalize(plan)
    def test_matching_saved_state_and_options_resume_without_entropy(self):
        self.frozen.save(self.work/'control/pipeline_state.json',{'plan_sha256':self.frozen.sha(self.work/'control/pipeline_plan.json')})
        with mock.patch.object(self.frozen.secrets,'randbits',side_effect=AssertionError('Do not redraw')):
            self.assertEqual(self.normalize(None,'--seed','random'),self.plan['options'])
    def test_source_alias_guard_is_portable_without_link_creation_privilege(self):
        target=self.work/'runtime/scripts/candidate_policy.py'
        original=Path.is_symlink
        def source_alias(path):return path==target or original(path)
        with mock.patch.object(Path,'is_symlink',autospec=True,side_effect=source_alias):
            with self.assertRaisesRegex(ValueError,'symbolic links'):self.normalize()
    def test_symbolic_source_alias_is_rejected_even_when_bytes_match(self):
        target=self.work/'runtime/scripts/candidate_policy.py';outside=self.root/'same_policy.py'
        shutil.copy2(target,outside);target.unlink()
        try:target.symlink_to(outside)
        except (OSError,NotImplementedError)as error:self.skipTest('Symlinks unavailable: '+str(error))
        with self.assertRaisesRegex(ValueError,'symbolic links'):self.normalize()


if __name__=='__main__':unittest.main()