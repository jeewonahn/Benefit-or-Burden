"""Guarded graphless paper-model mode: CPU synthetic correctness only."""
import argparse
import ast
import contextlib
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import random
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from context_feature_policy import (REQUIRED,UNUSED,PAPER_MODELS,NNR_ENCODERS,
    context_feature_spec,validate_declaration)
from nnr_artifacts import check_inputs


def declaration(models=None):
    selected=list(models or PAPER_MODELS)
    return {'context_feature_mode':UNUSED,'selected_models':selected,
        'context_features':{'mode':UNUSED,'required_by_selected_models':False,
            'allowed_models':list(PAPER_MODELS),'selected_models':selected,'context_outputs':[]}}


def input_bundle(root,graphless=True):
    root.mkdir(parents=True)
    meta=json.dumps([{'WikidataId':'Q1','OccurrenceOffsets':[0],'SurfaceForms':['Example']}])
    news='N1\tcat\tsub\tExample one\tAbstract one\turl\t'+meta+'\t[]\n'
    news+='N2\tcat\tsub\tExample two\tAbstract two\turl\t'+meta+'\t[]\n'
    for i,split in enumerate(('train','dev','test')):
        folder=root/split;folder.mkdir()
        (folder/'news.tsv').write_text(news,encoding='utf-8')
        (folder/'entity_embedding.vec').write_text('Q1\t'+'\t'.join(['1']*100)+'\n',encoding='utf-8')
        (folder/'relation_embedding.vec').write_text('P1\t'+'\t'.join(['2']*100)+'\n',encoding='utf-8')
        (folder/'behaviors.tsv').write_text(f'{i}\tU1\t11/{9+i}/2019 01:00:00 PM\tN1\tN2-1 N1-0\n',encoding='utf-8')
        if not graphless:(folder/'context_embedding.vec').write_text('Q1\t'+'\t'.join(['7']*100)+'\n',encoding='utf-8')
    (root/'manifest.json').write_text(json.dumps(declaration()if graphless else{}),encoding='utf-8')


class ContextPolicyTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/'dataset';input_bundle(self.root)

    def test_explicit_mode_records_actual_models_and_manifest(self):
        value=context_feature_spec(self.root,model='NAML',encoder='NAML',user_encoder='ATT')
        self.assertFalse(value['required']);self.assertEqual(value['manifest']['sha256'],hashlib.sha256((self.root/'manifest.json').read_bytes()).hexdigest())

    def test_all_five_native_pairs_are_whitelisted(self):
        for model,(news,user)in NNR_ENCODERS.items():
            with self.subTest(model=model):self.assertFalse(context_feature_spec(self.root,model=model,encoder=news,user_encoder=user)['required'])

    def test_entity_encoder_cannot_silently_omit_context(self):
        with self.assertRaises(ValueError):context_feature_spec(self.root,encoder='KCNN',user_encoder='CATT')

    def test_undeclared_model_rejected(self):
        (self.root/'manifest.json').write_text(json.dumps(declaration(['NAML'])),encoding='utf-8')
        with self.assertRaises(ValueError):context_feature_spec(self.root,model='NRMS')

    def test_missing_empty_or_unknown_selection_rejected(self):
        for models in([],['KCNN'],['NAML','NAML']):
            doc=declaration();doc['selected_models']=models;doc['context_features']['selected_models']=models
            with self.subTest(models=models),self.assertRaises(ValueError):validate_declaration(doc)

    def test_inconsistent_false_or_missing_declaration_rejected(self):
        for field,value in [('required_by_selected_models',True),('context_outputs',['fake.vec']),('mode','required'),('selected_models',['DIGAT']),('allowed_models',['NAML'])]:
            doc=declaration();doc['context_features'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):validate_declaration(doc)
        doc=declaration();del doc['context_features']
        with self.assertRaises(ValueError):validate_declaration(doc)

    def test_unused_mode_rejects_context_file_injection(self):
        (self.root/'train/context_embedding.vec').write_text('fake',encoding='utf-8')
        with self.assertRaises(ValueError):context_feature_spec(self.root)

    def test_artifact_check_retains_real_entity_and_binds_manifest(self):
        checked=check_inputs(self.root,'imp','imp','imp')
        self.assertFalse(checked['context_feature_policy']['required'])
        self.assertEqual(len(checked['files']),10)
        self.assertTrue(any(f['path'].endswith('entity_embedding.vec')for f in checked['files']))
        self.assertTrue(any(f['path'].endswith('manifest.json')for f in checked['files']))
        self.assertFalse(any(f['path'].endswith('context_embedding.vec')for f in checked['files']))

    def test_legacy_missing_context_still_fails(self):
        (self.root/'manifest.json').write_text('{}',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'Missing explicit'):check_inputs(self.root,'imp','imp','imp')

    def test_real_entity_missing_still_fails(self):
        (self.root/'dev/entity_embedding.vec').unlink()
        with self.assertRaisesRegex(ValueError,'Missing explicit'):check_inputs(self.root,'imp','imp','imp')


    def test_real_graphless_candidate_chain_preserves_declaration_and_entities(self):
        import subprocess
        # Real candidate CLI; no model and no copied historical behavior data.
        news=(self.root/'train/news.tsv').read_text(encoding='utf-8')
        news+='N3\tcat\tsub\tExample three\tAbstract three\turl\t[]\t[]\n'
        for split in('train','dev','test'):(self.root/split/'news.tsv').write_text(news,encoding='utf-8')
        (self.root/'train/behaviors.tsv').write_text('t0\tU1\t11/09/2019 01:00:00 PM\t\tN1-1 N2-0\nt1\tU1\t11/09/2019 02:00:00 PM\t\tN2-1 N1-0\n',encoding='utf-8')
        (self.root/'dev/behaviors.tsv').write_text('d\tU2\t11/10/2019 12:00:00 PM\tN1\tN3-1 N2-0\n',encoding='utf-8')
        (self.root/'test/behaviors.tsv').write_text('e\tU3\t11/11/2019 11:00:00 AM\tN1\tN1-1 N3-0\n',encoding='utf-8')
        before={str(p.relative_to(self.root)):p.read_bytes()for p in self.root.rglob('*')if p.is_file()}
        output=self.root.parent/'candidate'
        result=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/prepare_consistent_candidates.py'),
            '--source-data-root',str(self.root),'--output',str(output),'--seed','42'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        self.assertEqual(result.returncode,0,result.stdout.decode('utf-8',errors='replace'))
        from consistent_candidate_profile import validate_consistent_candidate_profile
        profile=validate_consistent_candidate_profile(output)
        self.assertEqual(profile['context_feature_mode'],UNUSED)
        for split in('train','dev','test'):
            self.assertFalse((output/split/'context_embedding.vec').exists())
            self.assertEqual((output/split/'entity_embedding.vec').read_bytes(),(self.root/split/'entity_embedding.vec').read_bytes())
            self.assertEqual((output/split/'relation_embedding.vec').read_bytes(),(self.root/split/'relation_embedding.vec').read_bytes())
        self.assertEqual(before,{str(p.relative_to(self.root)):p.read_bytes()for p in self.root.rglob('*')if p.is_file()})
        certificate=output/'consistent_candidate_manifest.json'
        doc=json.loads(certificate.read_text(encoding='utf-8'));doc['context_feature_mode']='required'
        certificate.write_text(json.dumps(doc),encoding='utf-8')
        with self.assertRaises(ValueError):validate_consistent_candidate_profile(output)

class NativeContextEquivalenceTests(unittest.TestCase):
    def test_preprocessing_rng_mappings_and_five_model_states_scores_identical(self):
        import numpy as np
        import torch
        from torch.utils.data import DataLoader
        # Import only CPU libraries; forbid accidental CUDA initialization.
        torch.set_num_threads(1)
        sys.path.insert(0,str(ROOT/'NNR-main'))
        import MIND_corpus as corpus_module
        from config import Config
        from model import Model
        from MIND_dataset import MIND_Train_Dataset
        from nnr_batch_alignment import apply_batch_alignment_fixes
        tree=ast.parse((ROOT/'NNR-main/config.py').read_text(encoding='utf-8'))
        klass=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='Config')
        parse=next(n for n in klass.body if isinstance(n,ast.FunctionDef)and n.name=='parse_argument')
        namespace={'argparse':argparse,'sys':sys,'json':json,'os':os,'time':time}
        exec(compile(ast.Module(body=[parse],type_ignores=[]),'<actual Config.parse_argument>','exec'),namespace)
        class Glove:
            def __init__(self,*a,**kw):
                self.stoi={word:i for i,word in enumerate(('example','one','two','abstract'))}
                self.vectors=torch.arange(200,dtype=torch.float32).reshape(4,50)/200
        def seed():random.seed(42);np.random.seed(42);torch.manual_seed(42)
        with tempfile.TemporaryDirectory()as temp:
            base=Path(temp);datasets=[base/'required',base/'unused'];runs=[base/'required_run',base/'unused_run'];models=[];configs=[];rngs=[];corpora=[]
            embedding=base/'emb';embedding.mkdir();(embedding/'glove.6B.50d.txt').write_text('synthetic fixture; loader mocked',encoding='utf-8')
            cwd=Path.cwd()
            try:
                for data,run,omitted in zip(datasets,runs,(False,True)):
                    input_bundle(data,graphless=omitted);run.mkdir();os.chdir(run)
                    cfg=SimpleNamespace()
                    args=['test','--dataset','small','--news_encoder','NAML','--user_encoder','ATT','--data_root',str(data),'--embedding_root',str(embedding),
                        '--word_embedding_dim','50','--word_threshold','1','--max_title_length','5','--max_abstract_length','6','--max_history_num','2',
                        '--hidden_dim','4','--attention_dim','4','--cnn_kernel_num','4','--head_num','2','--head_dim','2',
                        '--user_embedding_dim','4','--personalized_embedding_dim','4','--category_embedding_dim','3','--subCategory_embedding_dim','3']
                    with patch.object(sys,'argv',args):namespace['parse_argument'](cfg)
                    Config.preliminary_setup(cfg)
                    self.assertEqual(cfg.context_feature_mode,UNUSED if omitted else REQUIRED)
                    self.assertEqual(cfg.context_features_required,not omitted)
                    seed()
                    with patch.object(corpus_module,'GloVe',Glove),patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('GPU forbidden')):
                        corpus=corpus_module.MIND_Corpus(cfg)
                    rngs.append((random.getstate(),np.random.get_state(),torch.get_rng_state().clone()))
                    configs.append(cfg);corpora.append(corpus)
                    self.assertEqual((run/('context_embedding-'+cfg.dataset+'.pkl')).exists(),not omitted)
                self.assertEqual(rngs[0][0],rngs[1][0]);np.testing.assert_array_equal(rngs[0][1][1],rngs[1][1][1]);self.assertTrue(torch.equal(rngs[0][2],rngs[1][2]))
                for path in runs[0].glob('*.json'):self.assertEqual(path.read_bytes(),(runs[1]/path.name).read_bytes())
                for field in('news_title_text','news_abstract_text','news_title_entity','news_abstract_entity','news_category','news_subCategory','train_user_history_graph'):
                    np.testing.assert_array_equal(getattr(corpora[0],field),getattr(corpora[1],field))
                for name,(news,user)in NNR_ENCODERS.items():
                    values=[];states=[]
                    for cfg,corpus,run in zip(configs,corpora,runs):
                        os.chdir(run);cfg.news_encoder=news;cfg.user_encoder=user
                        seed()
                        with patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('GPU forbidden')):
                            model=Model(cfg);model.initialize();apply_batch_alignment_fixes(model);model.user_encoder.device=torch.device('cpu');model.eval()
                            ds=MIND_Train_Dataset(corpus);ds.negative_sampling();batch=next(iter(DataLoader(ds,batch_size=1,shuffle=False)))
                            with torch.no_grad():score=model(*batch).clone()
                        values.append(score);states.append({key:value.clone()for key,value in model.state_dict().items()})
                    with self.subTest(model=name):
                        self.assertEqual(states[0].keys(),states[1].keys())
                        self.assertTrue(all(torch.equal(states[0][key],states[1][key])for key in states[0]))
                        self.assertTrue(torch.equal(values[0],values[1]))
            finally:os.chdir(cwd)


if __name__=='__main__':unittest.main(verbosity=2)
