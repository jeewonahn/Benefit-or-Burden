"""Chronological official-archive roles, boundaries and source preservation."""
import json
from pathlib import Path
import sys
import tempfile
import subprocess
from unittest import mock
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import prepare_mind_masters as master
from test_prepare_mind_masters import fixture,arguments,replace_member


def chronological_fixture(root,train_days=range(9,15),test_day=15):
    train,dev,graph,_,_,_=fixture(root,graph_zip=True)
    rows=[]
    # Deliberately unsorted source order: chronology selects by parsed date,
    # preserves source row bytes/order within each output, and keeps duplicates.
    for day in reversed(list(train_days)):
        rows.append(f'day{day}\tuser{day}\t11/{day}/2019 02:00:00 PM\tN2\tN1-1 N2-0\r\n')
        if day==14:
            rows.append('devduplicate\tuser14\t11/14/2019 02:00:00 PM\tN2\tN2-1 N1-0\r\n')
    raw=''.join(rows).encode()
    final=f'final\ttestuser\t11/{test_day}/2019 01:00:00 AM\tN2\tN3-1 N1-0\r\n'.encode()
    replace_member(train,'behaviors.tsv',raw);replace_member(dev,'behaviors.tsv',final)
    return train,dev,graph,raw,final


class ChronologicalMasterTests(unittest.TestCase):
    def test_explicit_cli_policy_and_legacy_default(self):
        base=['--train-zip','train.zip','--dev-zip','dev.zip','--wikidata-graph','graph.zip','--output','out']
        self.assertEqual(master.parser().parse_args(base).split_policy,'random_groups')
        self.assertEqual(master.parser().parse_args(base+['--split-policy','chronological']).split_policy,'chronological')

    def test_exact_five_one_one_day_roles_preserve_every_source_row_and_duplicate_group(self):
        with tempfile.TemporaryDirectory()as temporary:
            root=Path(temporary);train,dev,graph,raw,final=chronological_fixture(root)
            originals={p:p.read_bytes()for p in(train,dev,graph)}
            out=root/'masters';master.prepare(arguments(train,dev,graph,out,split_policy='chronological'))
            m=json.loads((out/'manifest.json').read_text())
            self.assertEqual(m['profile'],master.CHRONOLOGICAL_PROFILE)
            self.assertEqual(m['split_policy'],'chronological')
            self.assertEqual(set(m['statistics']['train/behaviors.tsv']['dates']),{f'2019-11-{day:02d}'for day in range(9,14)})
            self.assertEqual(set(m['statistics']['dev/behaviors.tsv']['dates']),{'2019-11-14'})
            self.assertEqual(set(m['statistics']['test/behaviors.tsv']['dates']),{'2019-11-15'})
            self.assertEqual(m['statistics']['train/behaviors.tsv']['rows'],5)
            self.assertEqual(m['statistics']['dev/behaviors.tsv']['rows'],2)
            self.assertEqual(m['statistics']['dev/behaviors.tsv']['groups'],1)
            self.assertEqual((out/'test/behaviors.tsv').read_bytes(),final)
            self.assertEqual((out/'dev/behaviors.tsv').read_bytes(),b''.join(raw.splitlines(keepends=True)[:2]))
            self.assertTrue(m['validation']['train_max_before_dev_min'])
            self.assertTrue(m['validation']['dev_max_before_test_min'])
            self.assertTrue(m['date_audit']['exact_expected_chronological_source_dates'])
            self.assertFalse(m['date_audit']['development_is_random_request_group_holdout_not_chronological'])
            self.assertIsNone(m['development_split']['fraction_requested'])
            self.assertIsNone(m['development_split']['random_stream'])
            self.assertEqual(m['development_split']['group_count'],1)
            self.assertFalse(m['historical_R2_reproduction'])
            reconstructed={};streams={s:(out/s/'behaviors.tsv').read_bytes().splitlines(keepends=True)for s in('train','dev','test')}
            for record in map(json.loads,(out/'request_mapping.jsonl').read_text().splitlines()):
                if record['source_role']=='official_train':reconstructed[record['source_row']]=streams[record['split']][record['output_row']-1]
            self.assertEqual(b''.join(reconstructed[i]for i in sorted(reconstructed)),raw)
            for p,b in originals.items():self.assertEqual(p.read_bytes(),b)

    def test_seed_and_random_fraction_do_not_change_chronological_membership(self):
        with tempfile.TemporaryDirectory()as temporary:
            root=Path(temporary);train,dev,graph,*_=chronological_fixture(root)
            a=root/'a';b=root/'b'
            master.prepare(arguments(train,dev,graph,a,split_policy='chronological',seed=42))
            master.prepare(arguments(train,dev,graph,b,split_policy='chronological',seed=43,dev_fraction=.4))
            for split in('train','dev','test'):
                self.assertEqual((a/split/'behaviors.tsv').read_bytes(),(b/split/'behaviors.tsv').read_bytes())

    def test_missing_training_calendar_day_is_rejected(self):
        with tempfile.TemporaryDirectory()as temporary:
            root=Path(temporary);train,dev,graph,*_=chronological_fixture(root,train_days=range(10,15))
            with self.assertRaisesRegex(ValueError,'official train dates'):
                master.prepare(arguments(train,dev,graph,root/'out',split_policy='chronological'))
            self.assertFalse((root/'out').exists())

    def test_missing_dev_day_is_not_silently_replaced(self):
        with tempfile.TemporaryDirectory()as temporary:
            root=Path(temporary);train,dev,graph,*_=chronological_fixture(root,train_days=range(9,14))
            with self.assertRaisesRegex(ValueError,'official train dates'):
                master.prepare(arguments(train,dev,graph,root/'out',split_policy='chronological'))

    def test_official_dev_before_or_after_expected_final_day_is_rejected(self):
        for day in(14,16):
            with self.subTest(day=day),tempfile.TemporaryDirectory()as temporary:
                root=Path(temporary);train,dev,graph,*_=chronological_fixture(root,test_day=day)
                with self.assertRaisesRegex(ValueError,'official dev dates'):
                    master.prepare(arguments(train,dev,graph,root/'out',split_policy='chronological'))

    def test_unrecognized_split_policy_is_rejected(self):
        with tempfile.TemporaryDirectory()as temporary:
            root=Path(temporary);train,dev,graph,*_=chronological_fixture(root)
            with self.assertRaisesRegex(ValueError,'Unknown split policy'):
                master.prepare(arguments(train,dev,graph,root/'out',split_policy='future_tuned'))


    def test_explicit_graphless_mode_has_no_fake_context_assets(self):
        with tempfile.TemporaryDirectory()as temporary:
            root=Path(temporary);train,dev,graph,raw,final=chronological_fixture(root)
            out=root/'masters'
            with mock.patch('context_embeddings.prepare_context_embeddings',side_effect=AssertionError('Unused context helper must not run')):
                master.prepare(arguments(train,dev,None,out,split_policy='chronological',
                    context_feature_mode=master.CONTEXT_NOT_USED_BY_PAPER_MODELS,selected_models=list(master.PAPER_MODELS)))
            m=json.loads((out/'manifest.json').read_text())
            self.assertEqual(m['context_feature_mode'],master.CONTEXT_NOT_USED_BY_PAPER_MODELS)
            self.assertEqual(m['selected_models'],list(master.PAPER_MODELS))
            self.assertFalse(m['context_features']['required_by_selected_models'])
            self.assertEqual(m['context_features']['selected_models'],m['selected_models'])
            self.assertEqual(m['context_features']['allowed_models'],list(master.PAPER_MODELS))
            self.assertEqual(m['context_features']['context_outputs'],[])
            self.assertIsNone(m['context']['source_graph']);self.assertIsNone(m['context_helper'])
            self.assertFalse(m['context']['outputs_created']);self.assertFalse(m['context']['dummy_vectors_created'])
            self.assertFalse((out/'provenance/context_embeddings.json').exists())
            for split in('train','dev','test'):
                self.assertFalse((out/split/'context_embedding.vec').exists())
                self.assertNotIn(split+'/context_embedding.vec',m['assets'])
                self.assertTrue((out/split/'entity_embedding.vec').is_file())
                self.assertTrue((out/split/'relation_embedding.vec').is_file())
            self.assertEqual((out/'test/behaviors.tsv').read_bytes(),final)

    def test_graphless_missing_unknown_and_duplicate_selected_models_fail(self):
        for selected in(None,[],['KCNN'],['NAML','NAML']):
            with self.subTest(selected=selected),tempfile.TemporaryDirectory()as temporary:
                root=Path(temporary);train,dev,graph,*_=chronological_fixture(root)
                with self.assertRaisesRegex(ValueError,'selected paper models'):
                    master.prepare(arguments(train,dev,None,root/'out',split_policy='chronological',
                        context_feature_mode=master.CONTEXT_NOT_USED_BY_PAPER_MODELS,selected_models=selected))
                self.assertFalse((root/'out').exists())

    def test_graphless_mode_does_not_silently_read_supplied_graph(self):
        with tempfile.TemporaryDirectory()as temporary:
            root=Path(temporary);train,dev,graph,*_=chronological_fixture(root)
            with self.assertRaisesRegex(ValueError,'Do not supply a Wikidata graph'):
                master.prepare(arguments(train,dev,graph,root/'out',split_policy='chronological',
                    context_feature_mode=master.CONTEXT_NOT_USED_BY_PAPER_MODELS,selected_models=['NAML']))

    def test_default_context_mode_still_requires_graph(self):
        with tempfile.TemporaryDirectory()as temporary:
            root=Path(temporary);train,dev,graph,*_=chronological_fixture(root)
            with self.assertRaisesRegex(FileNotFoundError,'Wikidata graph is required'):
                master.prepare(arguments(train,dev,None,root/'out',split_policy='chronological'))
            self.assertFalse((root/'out').exists())

    def test_graphless_chronological_actual_cli_needs_only_official_shape_zips(self):
        with tempfile.TemporaryDirectory()as temporary:
            root=Path(temporary);train,dev,graph,raw,final=chronological_fixture(root);out=root/'masters'
            command=[sys.executable,'-B',str(Path(master.__file__)),
                '--train-zip',str(train),'--dev-zip',str(dev),'--output',str(out),'--split-policy','chronological',
                '--context-feature-mode',master.CONTEXT_NOT_USED_BY_PAPER_MODELS,'--selected-models','NAML']
            result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            self.assertEqual(result.returncode,0,result.stdout.decode())
            m=json.loads((out/'manifest.json').read_text())
            self.assertTrue(m['validation']['train_max_before_dev_min'])
            self.assertEqual(m['context_features']['context_outputs'],[])
            self.assertEqual(m['selected_models'],['NAML'])
            self.assertEqual((out/'test/behaviors.tsv').read_bytes(),final)



if __name__=='__main__':unittest.main()
