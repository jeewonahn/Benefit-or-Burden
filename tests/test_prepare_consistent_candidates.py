"""End-to-end synthetic CPU checks for the portable candidate CLI."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from unittest import mock

REPO=Path(__file__).resolve().parents[1]
SCRIPTS=REPO/'scripts'
sys.path.insert(0,str(SCRIPTS))
spec=importlib.util.spec_from_file_location('portable_prepare',SCRIPTS/'prepare_consistent_candidates.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
from consistent_candidate_profile import validate_consistent_candidate_profile,OUTPUT_FILES

def make_source(root):
 source=root/'source';source.mkdir()
 news=''.join('N%d\tcat\tsub\ttitle\tabstract\turl\t[]\t[]\n'%i for i in range(6))
 rows={'train':['a\tuser_a\t11/09/2019 12:00:00 AM\t\tN1-1 N0-0\n',
                 'b\tuser_b\t11/09/2019 01:00:00 AM\t\tN2-1 N0-0\n'],
       'dev':['c\tuser_c\t11/09/2019 02:00:00 AM\t\tN3-1 N0-0\n'],
       'test':['d\tuser_d\t11/09/2019 03:00:00 AM\t\tN4-1 N0-0\n']}
 for split in ('train','dev','test'):
  folder=source/split;folder.mkdir();(folder/'behaviors.tsv').write_text(''.join(rows[split]),encoding='utf-8')
  (folder/'news.tsv').write_text(news,encoding='utf-8')
  for name in ('entity_embedding.vec','context_embedding.vec'):(folder/name).write_bytes(b'synthetic\t0\n')
 return source

def args(source,output,minimum=None,control=None):return SimpleNamespace(source_data_root=source,output=output,seed=42,source_minimum_map=minimum,control_output=control)


def make_large_source(root):
 source=make_source(root)
 # The earlier N3 and the article N5 are observations outside the sampled cohort.
 raw_pins={role:{'bytes':4,'sha256':hashlib.sha256(role.encode()).hexdigest()}
           for role in ('official_train','official_dev')}
 observed={'N'+str(i):'2019-11-08T23:00:00' for i in range(6)}
 path=source/mod.LARGE_MASTER_MAP;path.parent.mkdir()
 path.write_text(json.dumps({'input_files':raw_pins,'all_impressions_min':observed}))
 cohort=source/mod.SAMPLED_USER_IDS;cohort.write_text(json.dumps(['user_a','user_b','user_c','user_d']))
 master={'status':'complete','dataset':'MIND-200K','user_sample':{'count':4,'user_ids':mod.fingerprint(cohort)},
         'full_source_minimum_map':mod.fingerprint(path),
         'source_archives':{role:{'extracted_members':{'behaviors.tsv':pin}} for role,pin in raw_pins.items()}}
 (source/'manifest.json').write_text(json.dumps(master))
 return source,observed

class PortableTests(unittest.TestCase):
 def test_large_uses_earlier_and_unsampled_impressions_across_splits_and_after_move(self):
  with tempfile.TemporaryDirectory()as tmp:
   base=Path(tmp);source,observed=make_large_source(base);out=base/'complete'
   mod.prepare(args(source,out))
   for relative in ('train/behaviors_rs50.tsv','dev/behaviors_rs50.tsv','test/behaviors_total.tsv'):
    for row in (out/relative).read_text().splitlines():
     self.assertIn('N5-0',row.split('\t')[4].split())
   moved=base/'moved';out.rename(moved);source.rename(base/'old_source')
   checked=validate_consistent_candidate_profile(moved)
   self.assertEqual(checked['publication_scope'],mod.LARGE_PROXY_SCOPE)
   self.assertEqual(checked['coverage']['expected_observed_ids'],len(observed))
   self.assertEqual(checked['dataset'],'MIND-200K')
   # A changed map and proxy cannot pass by updating their candidate-side fingerprints alone.
   p=moved/mod.MINIMUM_MAP;data=json.loads(p.read_text());data['all_impressions_min']['N5']='2019-11-08T22:00:00'
   p.write_text(json.dumps(data));proxy=moved/'news_publish_time.tsv'
   proxy.write_text(proxy.read_text().replace('N5\t11/08/2019 11:00:00 PM','N5\t11/08/2019 10:00:00 PM'))
   c=moved/'consistent_candidate_manifest.json';cert=json.loads(c.read_text())
   for key,path in [('source_minimum_map',p),('publication_proxy',proxy)]:
    cert[key].update({k:mod.fingerprint(path)[k] for k in ('bytes','sha256')})
   c.write_text(json.dumps(cert))
   with self.assertRaisesRegex(ValueError,'differs from its source master'):validate_consistent_candidate_profile(moved)

 def test_large_rejects_changed_raw_source_or_proxy_later_than_sampled_observation(self):
  for problem in ('raw','timestamp','missing'):
   with self.subTest(problem=problem),tempfile.TemporaryDirectory()as tmp:
    base=Path(tmp);source,_=make_large_source(base);p=source/mod.LARGE_MASTER_MAP;data=json.loads(p.read_text())
    if problem=='raw':data['input_files']['official_dev']['sha256']='0'*64
    elif problem=='missing':del data['all_impressions_min']['N3']
    else:data['all_impressions_min']['N3']='2019-11-10T00:00:00'
    p.write_text(json.dumps(data));m=source/'manifest.json';master=json.loads(m.read_text())
    master['full_source_minimum_map']=mod.fingerprint(p);m.write_text(json.dumps(master))
    with self.assertRaisesRegex(ValueError,'different raw behaviors|sampled minima'):mod.prepare(args(source,base/'dataset'))

 def test_large_refuses_a_different_cohort_even_if_the_user_list_is_repinned(self):
  with tempfile.TemporaryDirectory()as tmp:
   base=Path(tmp);source,_=make_large_source(base)
   cohort=source/mod.SAMPLED_USER_IDS;cohort.write_text(json.dumps(['user_a','user_b','user_c','other']))
   m=source/'manifest.json';master=json.loads(m.read_text())
   master['user_sample']['user_ids']=mod.fingerprint(cohort);m.write_text(json.dumps(master))
   with self.assertRaisesRegex(ValueError,'Sampled user list differs'):mod.prepare(args(source,base/'dataset'))

 def test_large_space_bound_uses_only_the_request_availability_window(self):
  from candidate_policy import PublicationIndex
  index=PublicationIndex({'N'+str(i):'2019-11-01T00:00:00' for i in range(10000)})
  rows={'test':[{'head':['1','U','11/12/2019 01:00:00 AM','N0'],'positives':['N1']}]}
  estimate=mod.large_output_bound(index,rows,{'test':{'bytes':200}})
  self.assertGreater(estimate,200)
  self.assertLess(estimate,10000)

 def test_random_access_window_matches_materialized_window_and_seeded_sampling(self):
  from candidate_policy import PublicationIndex, sample_negative_prefixes
  from datetime import datetime,timedelta
  now=datetime(2019,11,15,12)
  times={'N'+str(i):now+timedelta(minutes=i-1500) for i in range(1600)}
  idx=PublicationIndex(times)
  positives=['N100'];history=['N99','N101','N102','N550','unknown']
  expected=tuple(n for n in idx.ids if now-timedelta(days=1)<times[n]<now and n not in positives+history)
  view=idx.eligible_window(now,history,positives)
  self.assertEqual(tuple(view),expected)
  self.assertEqual(tuple(view[i] for i in range(len(view))),expected)
  self.assertEqual(view[-1],expected[-1])
  self.assertEqual(sample_negative_prefixes(view,seed=25),sample_negative_prefixes(expected,seed=25))
  # A 50-negative draw on a large window accesses only the selected ranks.
  with mock.patch.object(type(view),'__iter__',side_effect=AssertionError('full population iteration')):
   self.assertEqual(len(sample_negative_prefixes(view,counts=(50,),seed=25)[50]),50)
 def test_end_to_end_zero_ledger_complete_proxy_native_master_and_certificate(self):
  with tempfile.TemporaryDirectory()as temporary:
   base=Path(temporary);source=make_source(base);before={str(p):p.read_bytes()for p in source.rglob('*')if p.is_file()}
   had_torch=('torch' in sys.modules)
   output=base/'complete';control=base/'control';receipt=mod.prepare(args(source,output,control=control))
   self.assertEqual(receipt['split_evidence']['train']['excluded_rows'],1)
   self.assertEqual(receipt['split_evidence']['train']['excluded_positives'],1)
   self.assertEqual(receipt['split_evidence']['test']['excluded_rows'],0)
   checked=validate_consistent_candidate_profile(output)
   self.assertEqual(checked['coverage']['proxy_ids'],5)
   minimum=json.loads((output/'provenance/observed_first_impression_min.json').read_text(encoding='utf-8'))
   for split,proof in minimum['input_files'].items():
    self.assertEqual(Path(proof['path']),output/split/'behaviors.tsv')
    self.assertTrue(Path(proof['path']).is_file())
    self.assertNotIn('.partial-',proof['path'])
   self.assertEqual((output/'test/behaviors_imp_eligible.tsv').read_bytes(),(source/'test/behaviors.tsv').read_bytes())
   self.assertEqual((output/'train/behaviors.tsv').read_bytes(),(source/'train/behaviors.tsv').read_bytes())
   for filename,raw in before.items():self.assertEqual(Path(filename).read_bytes(),raw)
   self.assertTrue((control/'full_total/manifest.json').is_file())
   self.assertEqual(len((output/'exclusions.jsonl').read_text().splitlines()),1)
   for protocol in ('rs30','rs90','rs150','total'):
    fields=(output/'test'/('behaviors_'+protocol+'.tsv')).read_text().strip('\n').split('\t')
    self.assertEqual(len(fields[4].split()),5)
   self.assertEqual('torch' in sys.modules,had_torch)
 def test_test_protocols_are_nested_subsets_of_the_exact_strict_base(self):
  with tempfile.TemporaryDirectory()as temporary:
   base=Path(temporary);source=make_source(base)
   news=''.join('N%d\tcat\tsub\ttitle\tabstract\turl\t[]\t[]\n'%i for i in range(170))
   for split in ('train','dev','test'):(source/split/'news.tsv').write_text(news,encoding='utf-8')
   early='a\tuser_a\t11/09/2019 12:00:00 AM\t\tN1-1 '+' '.join('N%d-0'%i for i in range(170)if i!=1)+'\n'
   later='b\tuser_b\t11/09/2019 01:00:00 AM\t\tN2-1 N0-0\n'
   (source/'train/behaviors.tsv').write_text(early+later,encoding='utf-8')
   output=base/'complete';mod.prepare(args(source,output))
   negatives={}
   for protocol in ('rs30','rs90','rs150','total'):
    raw=(output/'test'/('behaviors_'+protocol+'.tsv')).read_text(encoding='utf-8').strip()
    values=[token.rsplit('-',1)for token in raw.split('\t')[4].split()]
    self.assertEqual([n for n,y in values if y=='1'],['N4'])
    negatives[protocol]=[n for n,y in values if y=='0']
   self.assertEqual(negatives['rs30'],negatives['rs150'][:30])
   self.assertEqual(negatives['rs90'],negatives['rs150'][:90])
   self.assertEqual(len(negatives['rs150']),150)
   self.assertEqual(set(negatives['total']),{'N%d'%i for i in range(170)}-{'N4'})
   self.assertEqual(len(negatives['total']),169)
   self.assertTrue(set(negatives['rs150'])<=set(negatives['total']))
 def test_seed_reproducibility_and_independent_minimum_map(self):
  with tempfile.TemporaryDirectory()as temporary:
   base=Path(temporary);source=make_source(base);a=base/'a';b=base/'b'
   mod.prepare(args(source,a));minimum=a/'provenance/observed_first_impression_min.json'
   mod.prepare(args(source,b,minimum))
   for relative in OUTPUT_FILES.values():self.assertEqual((a/relative).read_bytes(),(b/relative).read_bytes())
 def test_source_subdirectory_output_is_rejected_without_writes(self):
  with tempfile.TemporaryDirectory()as temporary:
   source=make_source(Path(temporary))
   with self.assertRaisesRegex(ValueError,'separate'):mod.prepare(args(source,source/'bad'))
   self.assertFalse((source/'bad').exists())
 def test_existing_output_is_never_overwritten(self):
  with tempfile.TemporaryDirectory()as temporary:
   base=Path(temporary);source=make_source(base);out=base/'existing';out.mkdir();marker=out/'marker';marker.write_bytes(b'preserve')
   with self.assertRaises(ValueError):mod.prepare(args(source,out))
   self.assertEqual(marker.read_bytes(),b'preserve')
 def test_partial_minimum_map_is_rejected(self):
  with tempfile.TemporaryDirectory()as temporary:
   base=Path(temporary);source=make_source(base);minimum=base/'partial.json'
   minimum.write_text(json.dumps({'all_impressions_min':{'N0':'2019-11-09T00:00:00'}}))
   with self.assertRaisesRegex(ValueError,'complete exact minimum'):mod.prepare(args(source,base/'out',minimum))
 def test_control_inside_source_is_rejected_without_source_writes(self):
  with tempfile.TemporaryDirectory()as temporary:
   base=Path(temporary);source=make_source(base);before={str(p):p.read_bytes()for p in source.rglob('*')if p.is_file()}
   with self.assertRaisesRegex(ValueError,'separate'):mod.prepare(args(source,base/'out',control=source/'receipts'))
   self.assertFalse((source/'receipts').exists())
   self.assertEqual({str(p):p.read_bytes()for p in source.rglob('*')if p.is_file()},before)
 def test_ancestor_control_cannot_write_its_descriptor_inside_source(self):
  with tempfile.TemporaryDirectory()as temporary:
   base=Path(temporary);initial=make_source(base);source=base/'full_total';initial.rename(source)
   before={str(p):p.read_bytes()for p in source.rglob('*')if p.is_file()}
   with self.assertRaisesRegex(ValueError,'separate'):mod.prepare(args(source,base/'out',control=base))
   self.assertFalse((source/'manifest.json').exists())
   self.assertEqual({str(p):p.read_bytes()for p in source.rglob('*')if p.is_file()},before)
 def test_same_request_positive_union_is_shared_across_source_rows(self):
  with tempfile.TemporaryDirectory()as temporary:
   base=Path(temporary);source=make_source(base)
   with (source/'dev/behaviors.tsv').open('a',encoding='utf-8')as out:
    out.write('c2\tuser_c\t11/09/2019 02:00:00 AM\t\tN2-1 N0-0\n')
   output=base/'complete';receipt=mod.prepare(args(source,output))
   self.assertEqual(receipt['split_evidence']['dev']['eligible_positives'],2)
   rows=(output/'dev/behaviors_rs50.tsv').read_text(encoding='utf-8').splitlines()
   self.assertEqual(len(rows),2)
   observed=[]
   for raw in rows:
    values=[token.rsplit('-',1)for token in raw.split('\t')[4].split()]
    observed.extend(news for news,label in values if label=='1')
    self.assertFalse({'N2','N3'}&{news for news,label in values if label=='0'})
   self.assertEqual(observed,['N3','N2'])
 def test_public_cli_has_no_internal_absolute_paths_or_run_labels(self):
  text=(SCRIPTS/'prepare_consistent_candidates.py').read_text()
  self.assertNotIn('G:\\',text);self.assertNotIn('D:\\',text)
  for value in ('22771','343110861','3060009764','CodexRuns','ELIGIBILITY_PREFLIGHT'):self.assertNotIn(value,text)

if __name__=='__main__':unittest.main(verbosity=2)

