"""Synthetic archive tests for raw MIND master preparation; no GPU or network."""
import hashlib
import json
from pathlib import Path
import stat
import sys
import tempfile
from types import SimpleNamespace
import unittest
import zipfile

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
import prepare_mind_masters as master


def fixture(root,prefix=True,graph_zip=False,newline='\r\n'):
    news_train=''.join(n+'\tcategory\tsub\ttitle\tabstract\turl\t[]\t[]\n'for n in('N1','N2'))
    news_test=news_train+'N3\tcategory\tsub\ttitle\tabstract\turl\t[]\t[]\n'
    entity='Q1\t'+'\t'.join(['1']*100)+'\nQ2\t'+'\t'.join(['3']*100)+'\n'
    relation='P1\t'+'\t'.join(['99']*100)+'\n'
    rows=[]
    for i in range(20):
        row=f'i{i}\tuser{i}\t11/09/2019 {i%12+1:02d}:00:00 AM\t\tN1-1 N2-0'+newline
        rows.append(row)
        if i==0:rows.append('duplicate\tuser0\t11/09/2019 01:00:00 AM\t\tN2-1 N1-0'+newline)
    final='test\ttest_user\t11/10/2019 01:00:00 AM\t\tN3-1 N1-0'+newline
    archives={}
    for role,behavior,news in(('train',''.join(rows),news_train),('dev',final,news_test)):
        path=root/('MINDsmall_'+role+'.zip');archives[role]=path
        with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED)as archive:
            parent=('MINDsmall_'+role+'/')if prefix else''
            for name,text in(('behaviors.tsv',behavior),('news.tsv',news),('entity_embedding.vec',entity),('relation_embedding.vec',relation)):
                archive.writestr(parent+name,text.encode())
    graph_text='Q1\tP1\tQ2\nQ1\tP1\tQ2\nQ1\tP1\tQ9\n'
    graph=root/('graph.zip'if graph_zip else'graph.tsv')
    if graph_zip:
        with zipfile.ZipFile(graph,'w',compression=zipfile.ZIP_DEFLATED)as archive:archive.writestr('data/wikidata-graph.tsv',graph_text)
    else:graph.write_text(graph_text,encoding='utf-8')
    return archives['train'],archives['dev'],graph,''.join(rows).encode(),final.encode(),relation.encode()


def arguments(train,dev,graph,output,**changes):
    values=dict(dataset='MIND-small',train_zip=train,dev_zip=dev,wikidata_graph=graph,output=output,seed=42,dev_fraction=.05,
        expected_train_sha256=None,expected_dev_sha256=None,max_member_bytes=master.DEFAULT_MAX_MEMBER_BYTES,
        max_total_bytes=master.DEFAULT_MAX_TOTAL_BYTES,max_compression_ratio=master.DEFAULT_MAX_RATIO)
    values.update(changes);return SimpleNamespace(**values)


def replace_member(path,basename,value):
    with zipfile.ZipFile(path)as archive:items={i.filename:archive.read(i)for i in archive.infolist()}
    keys=[key for key in items if key.rsplit('/',1)[-1]==basename]
    if value is None:
        for key in keys:items.pop(key)
    else:
        for key in keys:items[key]=value
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED)as archive:
        for key,raw in items.items():archive.writestr(key,raw)


class RawMasterTests(unittest.TestCase):
    def test_archive_to_masters_preserves_bytes_groups_assets_dates_and_real_context(self):
        with tempfile.TemporaryDirectory()as temporary:
            root=Path(temporary);train,dev,graph,train_raw,test_raw,relation=fixture(root)
            old={str(path):path.read_bytes()for path in(train,dev,graph)};had_torch='torch'in sys.modules
            output=root/'masters';result=master.prepare(arguments(train,dev,graph,output))
            manifest=json.loads((output/'manifest.json').read_text())
            self.assertEqual(result['status'],'complete');self.assertEqual(manifest['development_split']['group_count'],1)
            self.assertEqual(manifest['development_split']['source_group_count'],20)
            self.assertEqual(manifest['statistics']['train/behaviors.tsv']['rows']+manifest['statistics']['dev/behaviors.tsv']['rows'],21)
            self.assertEqual((output/'test/behaviors.tsv').read_bytes(),test_raw)
            reconstructed={};where=[]
            split_rows={s:(output/s/'behaviors.tsv').read_bytes().splitlines(keepends=True)for s in('train','dev','test')}
            for item in map(json.loads,(output/'request_mapping.jsonl').read_text().splitlines()):
                if item['source_role']=='official_train':
                    reconstructed[item['source_row']]=split_rows[item['split']][item['output_row']-1]
                    if item['source_row']in(1,2):where.append(item['split'])
            self.assertEqual(b''.join(reconstructed[k]for k in sorted(reconstructed)),train_raw)
            self.assertEqual(len(set(where)),1)
            for split in('train','dev','test'):
                self.assertEqual((output/split/'relation_embedding.vec').read_bytes(),relation)
                values=(output/split/'context_embedding.vec').read_text().splitlines()[0].split()
                self.assertEqual(len(values),101);self.assertTrue(all(float(x)==2.0 for x in values[1:]))
                self.assertNotEqual((output/split/'context_embedding.vec').read_bytes(),relation)
            self.assertTrue(manifest['date_audit']['source_train_max_before_final_test_min'])
            self.assertEqual(manifest['proxy']['status'],'not_generated_here')
            self.assertFalse(manifest['context']['relation_embeddings_renamed'])
            self.assertNotIn('.partial-',(output/'manifest.json').read_text())
            self.assertNotIn('.partial-',(output/'provenance/context_embeddings.json').read_text())
            for path,raw in old.items():self.assertEqual(Path(path).read_bytes(),raw)
            self.assertEqual('torch'in sys.modules,had_torch)
    def test_zip_graph_and_seed_are_reproducible(self):
        with tempfile.TemporaryDirectory()as temporary:
            root=Path(temporary);train,dev,graph,*_=fixture(root,prefix=False,graph_zip=True,newline='\n')
            a=root/'a';b=root/'b';master.prepare(arguments(train,dev,graph,a));master.prepare(arguments(train,dev,graph,b))
            for split in('train','dev','test'):
                for name in('behaviors.tsv','news.tsv','entity_embedding.vec','relation_embedding.vec','context_embedding.vec'):
                    self.assertEqual((a/split/name).read_bytes(),(b/split/name).read_bytes())
    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory()as temporary:
            root=Path(temporary);train,dev,graph,*_=fixture(root);out=root/'out';out.mkdir();(out/'marker').write_bytes(b'keep')
            with self.assertRaises(FileExistsError):master.prepare(arguments(train,dev,graph,out))
            self.assertEqual((out/'marker').read_bytes(),b'keep')
    def test_wrong_expected_archive_sha_is_rejected(self):
        with tempfile.TemporaryDirectory()as temporary:
            root=Path(temporary);train,dev,graph,*_=fixture(root)
            with self.assertRaisesRegex(ValueError,'SHA-256'):master.prepare(arguments(train,dev,graph,root/'out',expected_train_sha256='0'*64))
            self.assertFalse((root/'out').exists())
    def test_official_dev_must_be_labeled(self):
        with tempfile.TemporaryDirectory()as temporary:
            root=Path(temporary);train,dev,graph,*_=fixture(root)
            replace_member(dev,'behaviors.tsv',b'bad\tu\t11/10/2019 01:00:00 AM\t\tN1 N2\n')
            with self.assertRaisesRegex(ValueError,'official_dev behaviors row'):master.prepare(arguments(train,dev,graph,root/'out'))
            self.assertFalse((root/'out').exists())
    def test_same_request_in_train_and_final_test_is_rejected(self):
        with tempfile.TemporaryDirectory()as temporary:
            root=Path(temporary);train,dev,graph,raw,*_=fixture(root)
            replace_member(dev,'behaviors.tsv',raw.splitlines(keepends=True)[0])
            with self.assertRaisesRegex(ValueError,'share complete request groups'):master.prepare(arguments(train,dev,graph,root/'out'))
    def test_missing_official_relation_asset_is_not_fabricated(self):
        with tempfile.TemporaryDirectory()as temporary:
            root=Path(temporary);train,dev,graph,*_=fixture(root);replace_member(train,'relation_embedding.vec',None)
            with self.assertRaisesRegex(ValueError,'must provide'):master.prepare(arguments(train,dev,graph,root/'out'))
    def test_missing_graph_is_explicit(self):
        with tempfile.TemporaryDirectory()as temporary:
            root=Path(temporary);train,dev,graph,*_=fixture(root);graph.unlink()
            with self.assertRaises(FileNotFoundError):master.prepare(arguments(train,dev,graph,root/'out'))
    def test_small_is_not_200k_substitute(self):
        with tempfile.TemporaryDirectory()as temporary:
            root=Path(temporary);train,dev,graph,*_=fixture(root)
            with self.assertRaisesRegex(ValueError,'200,000 unique users'):
                master.prepare(arguments(train,dev,graph,root/'out',dataset='MIND-200K'))
            self.assertFalse((root/'out').exists())
    def test_zip_path_traversal_absolute_ads_and_reserved_names_are_rejected(self):
        for name in('../escape.txt','/absolute.txt','C:/drive.txt','file:stream','CON','a/../news.tsv'):
            with self.subTest(name=name),tempfile.TemporaryDirectory()as temporary:
                path=Path(temporary)/'bad.zip'
                with zipfile.ZipFile(path,'w')as archive:archive.writestr(name,b'payload')
                with self.assertRaises(ValueError):master.inspect_archive(path,max_member_bytes=1024,max_total_bytes=4096,max_compression_ratio=1000)
    def test_zip_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory()as temporary:
            path=Path(temporary)/'bad.zip';info=zipfile.ZipInfo('news.tsv');info.create_system=3;info.external_attr=(stat.S_IFLNK|0o777)<<16
            with zipfile.ZipFile(path,'w')as archive:archive.writestr(info,'../../outside')
            with self.assertRaisesRegex(ValueError,'symlink'):master.inspect_archive(path,max_member_bytes=1024,max_total_bytes=4096,max_compression_ratio=1000)
    def test_duplicate_case_collision_and_ambiguous_basenames_are_rejected(self):
        for names in(('news.tsv','NEWS.TSV'),('one/news.tsv','two/news.tsv')):
            with self.subTest(names=names),tempfile.TemporaryDirectory()as temporary:
                path=Path(temporary)/'bad.zip'
                with zipfile.ZipFile(path,'w')as archive:
                    for name in names:archive.writestr(name,'x')
                with self.assertRaises(ValueError):master.inspect_archive(path,max_member_bytes=1024,max_total_bytes=4096,max_compression_ratio=1000)
    def test_zip_uncompressed_and_ratio_limits_are_enforced(self):
        with tempfile.TemporaryDirectory()as temporary:
            root=Path(temporary);train,dev,graph,*_=fixture(root)
            with self.assertRaisesRegex(ValueError,'size exceeds'):master.prepare(arguments(train,dev,graph,root/'a',max_member_bytes=1))
            with self.assertRaisesRegex(ValueError,'ratio'):master.prepare(arguments(train,dev,graph,root/'b',max_compression_ratio=1.0))
    def test_no_local_paths_or_historical_counts_are_hardcoded(self):
        text=(SCRIPTS/'prepare_mind_masters.py').read_text(encoding='utf-8-sig')
        for marker in('D:\\','G:\\','CodexRuns','224445','22771','73152'):self.assertNotIn(marker,text)


if __name__=='__main__':unittest.main(verbosity=2)
