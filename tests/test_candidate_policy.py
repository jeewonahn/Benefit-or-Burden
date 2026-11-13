import json
from datetime import datetime,timedelta,timezone
from pathlib import Path
import tempfile
import unittest
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from candidate_policy import (PublicationIndex,parse_timestamp,sample_negative_prefixes,
                              training_negative_pool,candidate_pairs,PROFILE,RECIPE)

class StrictCandidatePolicyTests(unittest.TestCase):
 def setUp(self):self.t=datetime(2019,11,15,12,0,0)
 def test_both_time_endpoints_are_excluded(self):
  index=PublicationIndex({'at_lower':self.t-timedelta(hours=24),'old':self.t-timedelta(hours=24,seconds=1),
     'inside':self.t-timedelta(seconds=1),'at_now':self.t,'future':self.t+timedelta(seconds=1)})
  self.assertEqual(index.eligible(self.t,(),['GT']),('inside',))
 def test_history_and_every_current_positive_are_excluded(self):
  index=PublicationIndex({n:self.t-timedelta(hours=1)for n in ['history','GT1','GT2','available']})
  self.assertEqual(index.eligible(self.t,['history'],['GT1','GT2']),('available',))
 def test_current_positive_union_shared_by_multiple_source_rows(self):
  index=PublicationIndex({n:self.t-timedelta(hours=1)for n in ['GT1','GT2','negative']})
  first=index.eligible(self.t,(),{'GT1','GT2'})
  second=index.eligible(self.t,(),{'GT2','GT1'})
  self.assertEqual(first,second)
  self.assertEqual(candidate_pairs('GT1',first),(('GT1',1),('negative',0)))
  self.assertEqual(candidate_pairs('GT2',second),(('GT2',1),('negative',0)))
 def test_no_global_future_click_exclusion_is_accepted(self):
  index=PublicationIndex({'later_clicked':self.t-timedelta(minutes=1)})
  self.assertEqual(index.eligible(self.t,(),['GT']),('later_clicked',))
 def test_positive_lifetime_is_never_used_to_drop_GT(self):
  index=PublicationIndex({'old_GT':self.t-timedelta(days=10),'negative':self.t-timedelta(hours=1)})
  pool=index.eligible(self.t,(),['old_GT'])
  self.assertEqual(candidate_pairs('old_GT',pool),(('old_GT',1),('negative',0)))
  self.assertEqual(candidate_pairs('GT_without_proxy',pool)[0],('GT_without_proxy',1))
 def test_earliest_strict_request_can_be_explicitly_ineligible(self):
  index=PublicationIndex({'GT':self.t,'other':self.t})
  self.assertEqual(index.eligible(self.t,(),['GT']),())
  self.assertEqual(sample_negative_prefixes(()),{30:(),90:(),150:()})
  with self.assertRaises(ValueError):candidate_pairs('GT',())
 def test_missing_eligible_metadata_fails_instead_of_shrinking_universe(self):
  index=PublicationIndex({'unmapped':self.t-timedelta(hours=1)})
  with self.assertRaises(ValueError):index.eligible(self.t,(),['GT'],{'GT'})
 def test_index_order_does_not_depend_on_dictionary_insertion(self):
  values={'z':self.t-timedelta(hours=1),'a':self.t-timedelta(hours=2),'b':self.t-timedelta(hours=1)}
  x=PublicationIndex(values).eligible(self.t,(),['GT'])
  y=PublicationIndex(dict(reversed(list(values.items())))).eligible(self.t,(),['GT'])
  self.assertEqual(x,('a','b','z'));self.assertEqual(x,y)
 def test_one_test_draw_produces_nested_unique_prefixes(self):
  pool=tuple('N'+str(i)for i in range(300))
  drawn=sample_negative_prefixes(pool,seed=42,split='test',source_row=8,positive_index=1)
  self.assertEqual(drawn[30],drawn[150][:30]);self.assertEqual(drawn[90],drawn[150][:90])
  self.assertEqual(len(set(drawn[150])),150);self.assertTrue(set(drawn[150])<=set(pool))
 def test_small_pool_uses_min_k_N_without_replacement(self):
  drawn=sample_negative_prefixes(('A','B'),seed=42)
  self.assertEqual(drawn[30],drawn[90]);self.assertEqual(drawn[90],drawn[150])
  self.assertEqual(set(drawn[150]),{'A','B'});self.assertEqual(len(drawn[150]),2)
 def test_seed_and_row_are_reproducible_and_distinct(self):
  pool=tuple('N'+str(i)for i in range(300))
  a=sample_negative_prefixes(pool,seed=42,source_row=1)
  self.assertEqual(a,sample_negative_prefixes(pool,seed=42,source_row=1))
  self.assertNotEqual(a,sample_negative_prefixes(pool,seed=43,source_row=1))
  self.assertNotEqual(a,sample_negative_prefixes(pool,seed=42,source_row=2))
 def test_train_and_dev_use_same_pool_sampler(self):
  pool=tuple('N'+str(i)for i in range(200))
  for split in ['train','dev']:
   actual=training_negative_pool(pool,seed=42,split=split,source_row=3)
   self.assertEqual(actual,sample_negative_prefixes(pool,counts=(50,),seed=42,split=split,source_row=3)[50])
   self.assertEqual(len(actual),50)
 def test_duplicate_pool_or_GT_conflict_is_rejected(self):
  with self.assertRaises(ValueError):sample_negative_prefixes(('A','A'))
  with self.assertRaises(ValueError):candidate_pairs('GT',('GT','A'))
  with self.assertRaises(ValueError):candidate_pairs('GT',('A','A'))
 def test_invalid_random_stream_identity_is_rejected(self):
  for kwargs in [{'seed':True},{'seed':-1},{'source_row':0},{'positive_index':-1},{'split':'later'}]:
   with self.assertRaises(ValueError):sample_negative_prefixes(('A','B'),**kwargs)
 def test_naive_time_parsing_and_aware_time_rejection(self):
  self.assertEqual(parse_timestamp('11/15/2019 12:00:00 PM'),self.t)
  self.assertEqual(parse_timestamp('2019-11-15T12:00:00'),self.t)
  with self.assertRaises(ValueError):parse_timestamp(self.t.replace(tzinfo=timezone.utc))
 def test_headerless_proxy_rejects_duplicate_or_bad_rows(self):
  with tempfile.TemporaryDirectory()as folder:
   p=Path(folder)/'proxy.tsv'
   p.write_text('A\t11/15/2019 11:00:00 AM\n',encoding='utf-8')
   self.assertEqual(PublicationIndex.from_proxy(p).eligible(self.t,(),['GT']),('A',))
   p.write_text('A\t2019-11-15T11:00:00\nA\t2019-11-15T11:00:00\n',encoding='utf-8')
   with self.assertRaises(ValueError):PublicationIndex.from_proxy(p)
 def test_empty_or_invalid_requests_do_not_silently_pass(self):
  with self.assertRaises(ValueError):PublicationIndex({})
  with self.assertRaises(ValueError):PublicationIndex({'bad id':self.t})
  with self.assertRaises(ValueError):PublicationIndex({'A':self.t}).eligible(self.t,(),[])
 def test_recipe_preserves_master_and_excludes_future_user_clicks(self):
  self.assertEqual(PROFILE,'complete_proxy_strict_request_consistent_v1')
  self.assertEqual(RECIPE['window_endpoints'],'strict')
  self.assertFalse(RECIPE['future_user_clicks_used_for_exclusion'])
  self.assertIn('preserve_IMP_master',RECIPE['zero_negative_policy'])

if __name__=='__main__':unittest.main(verbosity=2)