"""CPU-only synthetic checks for fresh DIGAT's certified test routing."""
import importlib.util,json
from pathlib import Path
import sys,tempfile,unittest
from types import SimpleNamespace
BASE=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(BASE))
spec=importlib.util.spec_from_file_location('fresh_digat',BASE/'run_converged_digat.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
from consistent_candidate_profile import OUTPUT_FILES

class RoutingTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);outputs={};statistics={}
  for key,name in OUTPUT_FILES.items():
   p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((key+'\n').encode());fp=m.fingerprint(p);outputs[key]={'path':name,'bytes':fp['bytes'],'sha256':fp['sha256']};statistics[name]={'rows':1,'candidates':51,'positive_labels':1}
  (self.root/'manifest.json').write_text(json.dumps({'statistics':statistics}))
  self.profile={'profile':'complete_proxy_strict_request_consistent_v1','outputs':outputs,'seed':42}
  self.args=SimpleNamespace(data_root=self.root,full_total=self.root/'test/behaviors_total.tsv')
  self.candidate={'statistics':statistics['test/behaviors_total.tsv']}
 def test_all_five_paths_are_certificate_outputs(self):
  result=m.bind_test_inputs(self.args,self.profile,self.candidate)
  self.assertEqual(set(result),{'imp','rs30','rs90','rs150','total'})
  self.assertTrue(result['imp']['path'].endswith('behaviors_imp_eligible.tsv'))
  self.assertEqual(result['total']['sha256'],self.profile['outputs']['test_total']['sha256'])
 def test_mutated_eligible_imp_rejected(self):
  (self.root/'test/behaviors_imp_eligible.tsv').write_bytes(b'changed')
  with self.assertRaises(ValueError):m.bind_test_inputs(self.args,self.profile,self.candidate)
 def test_total_count_mismatch_rejected(self):
  self.candidate={'statistics':{'rows':2,'candidates':51,'positive_labels':2}}
  with self.assertRaises(ValueError):m.bind_test_inputs(self.args,self.profile,self.candidate)
 def test_old_total_path_cannot_be_silently_used(self):
  self.args.full_total=self.root/'old_total.tsv'
  with self.assertRaises(ValueError):m.bind_test_inputs(self.args,self.profile,self.candidate)
 def test_new_policy_helpers_are_frozen(self):
  s=(BASE/'digat_source_snapshot.py').read_text()
  self.assertIn('"candidate_policy.py"',s);self.assertIn('"consistent_candidate_profile.py"',s)
 def test_imp_master_defines_user_order_only_for_new_profile(self):
  s=(BASE/'run_converged_digat.py').read_text()
  self.assertIn("args.data_root / 'train/behaviors.tsv' if consistent else selected_train",s)
  self.assertIn("'IMP_master_first_seen_order' if consistent",s)

class PairedInitializationTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
  self.args=SimpleNamespace(seed=42,train_type='rs50',paired_preprocessing_from=None)
  self.assets={'identity':'semantic-assets'}
  self.profile={'profile':'complete_proxy_strict_request_consistent_v1'}
 def reference(self):
  names=('news_ID-MIND-small.json','user_ID-MIND-small.json','category-MIND-small.json','subCategory-MIND-small.json','vocabulary-3-32-MIND-small.json')
  pins={}
  for name in names:
   p=self.root/name;p.write_text('{}');pins[name]=m.fingerprint(p)
  old={'status':'completed','model':'DIGAT','train_type':'imp','seed':42,'input_binding':{'assets':self.assets},'initial_model_sha256':'a'*64,'preprocess_files':pins}
  (self.root/'experiment.json').write_text(json.dumps(old));self.args.paired_preprocessing_from=self.root
  return old
 def test_new_profile_requires_original_imp(self):
  with self.assertRaises(ValueError):m.paired_imp_reference(self.args,self.assets,self.profile)
 def test_fresh_imp_under_consistent_profile_needs_no_trained_reference(self):
  self.args.train_type='imp'
  self.assertIsNone(m.paired_imp_reference(self.args,self.assets,self.profile))
 def test_legacy_profile_has_no_new_requirement(self):
  self.assertIsNone(m.paired_imp_reference(self.args,self.assets,None))
 def test_matching_exact_dictionary_reference(self):
  self.reference();result=m.paired_imp_reference(self.args,self.assets,self.profile)
  self.assertEqual(result['initial_model_sha256'],'a'*64);self.assertEqual(len(result['dictionaries']),5)
 def test_changed_original_dictionary_rejected(self):
  self.reference();(self.root/'user_ID-MIND-small.json').write_text('changed')
  with self.assertRaises(ValueError):m.paired_imp_reference(self.args,self.assets,self.profile)
 def test_wrong_semantic_assets_rejected(self):
  self.reference()
  with self.assertRaises(ValueError):m.paired_imp_reference(self.args,{'identity':'other'},self.profile)
 def test_initial_gate_precedes_optimizer_creation(self):
  source=(BASE/'run_converged_digat.py').read_text()
  self.assertLess(source.index("if paired is not None and initial_hash !="),source.index('optimizer = torch.optim.Adam'))


if __name__=='__main__':unittest.main(verbosity=2)
