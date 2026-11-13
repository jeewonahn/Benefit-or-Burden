from copy import deepcopy
from pathlib import Path
import json
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from candidate_policy import PROFILE,RECIPE
from consistent_candidate_profile import (validate_consistent_candidate_profile,evaluation_paths,
 OUTPUT_FILES,MASTER_FILES,MANIFEST_NAME,REQUIRED_VALIDATION,fingerprint)

class ConsistentProfileTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
  for relative in set(MASTER_FILES)|set(OUTPUT_FILES.values()):
   p=self.root/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('1\tU\t11/15/2019 12:00:00 PM\t\tGT-1 N-0\n',encoding='utf-8')
  p=self.root/'provenance/observed_first_impression_min.json';p.parent.mkdir()
  p.write_text(json.dumps({'all_impressions_min':{'N':'2019-11-15T11:00:00','GT':'2019-11-15T12:00:00'}}),encoding='utf-8')
  (self.root/'news_publish_time.tsv').write_text('N\t11/15/2019 11:00:00 AM\nGT\t11/15/2019 12:00:00 PM\n',encoding='utf-8')
  (self.root/'exclusions.jsonl').write_text('',encoding='utf-8')
  def fp(relative):return {**fingerprint(self.root/relative),'path':relative}
  self.cert={'schema':1,'status':'complete','profile':PROFILE,'seed':42,'sampling':deepcopy(RECIPE),
   'publication_proxy':fp('news_publish_time.tsv'),'source_minimum_map':fp('provenance/observed_first_impression_min.json'),
   'exclusions':fp('exclusions.jsonl'),'imp_inputs':{r:fp(r)for r in MASTER_FILES},
   'outputs':{k:fp(r)for k,r in OUTPUT_FILES.items()},'coverage':{'expected_observed_ids':2,'proxy_ids':2,'missing_ids':0,'unexpected_ids':0,'timestamp_mismatches':0},
   'validation':{k:True for k in REQUIRED_VALIDATION}}
  self.save()
 def save(self):
  (self.root/MANIFEST_NAME).write_text(json.dumps(self.cert),encoding='utf-8')
 def assert_rejected(self):
  with self.assertRaises(ValueError):validate_consistent_candidate_profile(self.root)
 def test_complete_certificate_is_accepted(self):
  p=validate_consistent_candidate_profile(self.root)
  self.assertEqual(p['profile'],PROFILE);self.assertEqual(set(p['outputs']),set(OUTPUT_FILES))
 def test_partial_proxy_cannot_be_relabelled_complete(self):
  p=self.root/'news_publish_time.tsv';p.write_text('N\t11/15/2019 11:00:00 AM\n',encoding='utf-8')
  self.cert['publication_proxy']={**fingerprint(p),'path':'news_publish_time.tsv'}
  self.cert['coverage']={'expected_observed_ids':2,'proxy_ids':1,'missing_ids':1,'unexpected_ids':0,'timestamp_mismatches':0}
  self.save();self.assert_rejected()
 def test_changed_timestamp_is_rejected_even_with_updated_file_hash(self):
  p=self.root/'news_publish_time.tsv';p.write_text('N\t11/15/2019 10:00:00 AM\nGT\t11/15/2019 12:00:00 PM\n',encoding='utf-8')
  self.cert['publication_proxy']={**fingerprint(p),'path':'news_publish_time.tsv'};self.save();self.assert_rejected()
 def test_same_legacy_profile_or_inclusive_recipe_is_rejected(self):
  self.cert['profile']='request_time_history_and_current_positives_v1';self.save();self.assert_rejected()
  self.cert['profile']=PROFILE;self.cert['sampling']['window_endpoints']='inclusive';self.save();self.assert_rejected()
 def test_old_test_output_content_is_rejected(self):
  (self.root/OUTPUT_FILES['test_rs30']).write_text('old data',encoding='utf-8');self.assert_rejected()
 def test_missing_one_output_is_not_a_complete_bundle(self):
  del self.cert['outputs']['test_total'];self.save();self.assert_rejected()
 def test_output_path_escape_is_rejected(self):
  self.cert['outputs']['test_rs30']['path']='../outside.tsv';self.save();self.assert_rejected()
 def test_changed_IMP_master_or_exclusions_is_rejected(self):
  (self.root/'exclusions.jsonl').write_text('{}\n',encoding='utf-8');self.assert_rejected()
 def test_missing_support_certification_is_rejected(self):
  self.cert['validation']['GT_multiset_consistent_all_test_protocols']=False;self.save();self.assert_rejected()
 def test_new_profile_routes_IMP_to_eligible_support(self):
  profile=validate_consistent_candidate_profile(self.root)
  paths=evaluation_paths(self.root,profile)
  self.assertEqual(paths['imp'],self.root/'test/behaviors_imp_eligible.tsv')
  self.assertEqual(paths['total'],self.root/'test/behaviors_total.tsv')
 def test_legacy_routing_is_preserved_without_certificate(self):
  (self.root/MANIFEST_NAME).unlink()
  self.assertEqual(evaluation_paths(self.root)['imp'],self.root/'test/behaviors.tsv')
  self.assertNotIn('total',evaluation_paths(self.root))
 def test_optional_test_hash_deferral_is_explicit(self):
  (self.root/OUTPUT_FILES['test_total']).write_text('will fail native exact hash gate',encoding='utf-8')
  self.assertEqual(validate_consistent_candidate_profile(self.root,verify_test_files=False)['profile'],PROFILE)
  self.assert_rejected()

if __name__=='__main__':unittest.main(verbosity=2)