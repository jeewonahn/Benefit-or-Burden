from pathlib import Path
from types import SimpleNamespace
import json
import sys
import tempfile
import unittest
sys.dont_write_bytecode=True
STAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(STAGE/'scripts'))
from candidate_policy import PROFILE
from consistent_candidate_profile import fingerprint
from nnr_consistent_preprocessing import configure_reference,paired_dictionary_gate,verify_paired_initialization
import run_converged_nnr as runner

class PairedDictionaryTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
  self.ref=self.root/'old_IMP';self.new=self.root/'new_RS';self.ref.mkdir();self.new.mkdir()
  self.cfg=SimpleNamespace(dataset='MIND-small',word_threshold=3,tokenizer='MIND',max_title_length=32,max_abstract_length=400)
  names=[n+'-MIND-small.json'for n in ['user_ID','news_ID','category','subCategory','entity']]+['vocabulary-3-MIND-32-400-MIND-small.json']
  self.files={}
  for name in names:
   for directory in [self.ref,self.new]:(directory/name).write_text(json.dumps({'PAD':0,'one':1,'two':2}),encoding='utf-8')
   self.files[name]=fingerprint(self.ref/name)
  self.manifest={'initial_model_state_sha256':'a'*64,'status':'completed','model':'NAML','seed':42,'train_type':'imp','preprocess_files':self.files}
  self.save()
 def save(self):(self.ref/'experiment.json').write_text(json.dumps(self.manifest),encoding='utf-8')
 def test_exact_maps_are_checked_before_model_tensor_access(self):
  proof=paired_dictionary_gate(self.new,self.ref,self.cfg,model='NAML',seed=42)
  self.assertEqual(proof['status'],'passed');self.assertEqual(len(proof['dictionaries']),6)
 def test_model_fingerprint_is_checked_before_any_optimizer_step(self):
  verify_paired_initialization('a'*64,{'status':'passed','initial_model_sha256':'a'*64})
  with self.assertRaises(ValueError):verify_paired_initialization('b'*64,{'status':'passed','initial_model_sha256':'a'*64})
 def test_missing_original_initialization_fingerprint_is_rejected(self):
  del self.manifest['initial_model_state_sha256'];self.save()
  with self.assertRaises(ValueError):paired_dictionary_gate(self.new,self.ref,self.cfg,model='NAML',seed=42)
 def test_user_drop_cannot_silently_change_embedding_identity(self):
  (self.new/'user_ID-MIND-small.json').write_text(json.dumps({'PAD':0,'two':1}),encoding='utf-8')
  with self.assertRaisesRegex(ValueError,'dictionary hashes differ'):paired_dictionary_gate(self.new,self.ref,self.cfg,model='NAML',seed=42)
 def test_stale_original_mapping_is_rejected(self):
  (self.ref/'news_ID-MIND-small.json').write_text('{}',encoding='utf-8')
  with self.assertRaisesRegex(ValueError,'differs from its manifest'):paired_dictionary_gate(self.new,self.ref,self.cfg,model='NAML',seed=42)
 def test_wrong_training_family_seed_or_model_is_rejected(self):
  self.manifest['train_type']='rs50';self.save()
  with self.assertRaises(ValueError):paired_dictionary_gate(self.new,self.ref,self.cfg,model='NAML',seed=42)
 def test_new_profile_records_master_reference_legacy_unchanged(self):
  data=self.root/'data';(data/'train').mkdir(parents=True);(data/'train/behaviors.tsv').write_text('master',encoding='utf-8')
  old=SimpleNamespace()
  self.assertIsNone(configure_reference(old,data,None));self.assertFalse(hasattr(old,'user_id_reference_file'))
  actual=SimpleNamespace();fp=configure_reference(actual,data,{'profile':PROFILE})
  self.assertEqual(Path(actual.user_id_reference_file),data/'train/behaviors.tsv')
  self.assertEqual(fp['sha256'],fingerprint(data/'train/behaviors.tsv')['sha256'])

class NewTotalNativeGateTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
  (self.root/'test').mkdir();(self.root/'manifest.json').write_text('{"fraction":1.0}',encoding='utf-8')
  files={}
  for name in ['behaviors_imp_eligible.tsv','behaviors_rs30.tsv','behaviors_rs90.tsv','behaviors_rs150.tsv','behaviors_total.tsv']:
   p=self.root/'test'/name;p.write_text('fixed-'+name,encoding='utf-8');files[name]=fingerprint(p)
  self.spec={'status':'complete','protocol':'total','dataset_root':str(self.root),'candidate_profile':PROFILE,
   'output':files['behaviors_total.tsv'],'dataset_manifest':fingerprint(self.root/'manifest.json'),
   'validation':{k:True for k in ['same_GT_request_support_all_protocols','history_and_current_positives_excluded','strict_time_window','complete_proxy_coverage','all_news_in_model_vocabularies']},
   'source_evaluations':{p:files[n]for p,n in {'imp':'behaviors_imp_eligible.tsv','rs30':'behaviors_rs30.tsv','rs90':'behaviors_rs90.tsv','rs150':'behaviors_rs150.tsv'}.items()}}
  self.path=self.root/'total.json';self.save()
  self.args=SimpleNamespace(candidate_manifest=self.path,data_root=self.root,full_total_behaviors=self.root/'test/behaviors_total.tsv',matched_candidate_profile={'profile':PROFILE})
 def save(self):self.path.write_text(json.dumps(self.spec),encoding='utf-8')
 def test_new_total_does_not_require_fake_legacy_total5_flag(self):
  self.assertNotIn('total5_subset_of_total',self.spec['validation'])
  self.assertEqual(runner.validate_total(self.args)['candidate_profile'],PROFILE)
 def test_new_profile_requires_its_real_alignment_and_exclusion_guards(self):
  self.spec['validation']['history_and_current_positives_excluded']=False;self.save()
  with self.assertRaises(ValueError):runner.validate_total(self.args)
 def test_changed_eligible_IMP_or_total_input_is_rejected(self):
  (self.root/'test/behaviors_imp_eligible.tsv').write_text('wrong support',encoding='utf-8')
  with self.assertRaises(ValueError):runner.validate_total(self.args)

if __name__=='__main__':unittest.main(verbosity=2)