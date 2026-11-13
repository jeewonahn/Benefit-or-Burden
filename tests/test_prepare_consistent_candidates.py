"""End-to-end synthetic CPU checks for the portable candidate CLI."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest

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

class PortableTests(unittest.TestCase):
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

