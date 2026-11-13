"""Real CPU child-CLI chain using invented ZIPs; never execute a model.

TORS_CANDIDATE_CODE_DIR optionally points a staging test to the release's
candidate scripts. Each test copies only known source files to an isolated
throwaway package; it never mutates the source release or a running study.
"""
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixtures(root):
    news=''.join(f'N{i}\tcategory\tsub\ttitle\tabstract\turl\t[]\t[]\n'for i in range(9))
    entity='Q1\t'+'\t'.join(['1']*100)+'\nQ2\t'+'\t'.join(['3']*100)+'\n'
    relation='P1\t'+'\t'.join(['99']*100)+'\n'
    rows=[]
    for hour in range(20):
        suffix=(' N2-0'if hour==1 else' N3-0'if hour==2 else' N5-0 N6-0'if hour==3 else'')
        clock=f'{hour%12 or 12:02d}:00:00 '+('AM'if hour<12 else'PM')
        header=f'train{hour}\ttrain_user{hour:02d}\t11/09/2019 {clock}\t\t'
        rows.append(header+'N1-1 N0-0'+suffix+'\n')
        if hour==3:
            rows.append('duplicate\ttrain_user03\t11/09/2019 03:00:00 AM\t\tN6-1 N0-0\n')
    raw_train=''.join(rows).encode()
    # At t=24h: N0/N1 are exactly on the excluded lower boundary; N7 is
    # exactly on the excluded upper boundary. N2 is history and N3 is another
    # current GT, leaving only N5/N6 as eligible negatives for the first row.
    # The second request reads every live negative, so only derivatives exclude it.
    raw_dev=('test1\ttest_user1\t11/10/2019 12:00:00 AM\tN2\tN4-1 N3-1 N1-1 N7-0 N5-0\n'
             'test2\ttest_user2\t11/10/2019 12:00:00 AM\tN2 N3 N5 N6\tN8-1 N0-0\n').encode()
    paths={}
    for role,raw in(('train',raw_train),('dev',raw_dev)):
        archive=root/('MINDsmall_'+role+'.zip');paths[role]=archive
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED)as stream:
            for name,value in(('behaviors.tsv',raw),('news.tsv',news.encode()),('entity_embedding.vec',entity.encode()),('relation_embedding.vec',relation.encode())):
                stream.writestr('official_'+role+'/'+name,value)
    graph=root/'wikidata-graph.zip'
    with zipfile.ZipFile(graph,'w',compression=zipfile.ZIP_DEFLATED)as stream:
        stream.writestr('wikidata-graph.tsv','Q1\tP1\tQ2\nQ1\tP1\tQ2\n')
    return paths['train'],paths['dev'],graph,raw_train,raw_dev


def positive_support(path):
    result=Counter();rows=[]
    with Path(path).open(encoding='utf-8-sig')as stream:
        for raw in stream:
            fields=raw.rstrip('\r\n').split('\t');key=(fields[1],fields[2],' '.join(fields[3].split()))
            values=[token.rsplit('-',1)for token in fields[4].split()]
            for news,label in values:
                if label=='1':result[key+(news,)]+=1
            rows.append((fields,values))
    return result,rows


class RawCandidateChainTests(unittest.TestCase):
    def test_actual_raw_master_context_candidate_and_certificate_chain(self):
        candidate_scripts=Path(os.environ.get('TORS_CANDIDATE_CODE_DIR',str(SCRIPTS))).resolve()
        sources={name:SCRIPTS/name for name in('prepare_mind_masters.py','context_embeddings.py')}
        sources.update({name:candidate_scripts/name for name in
                        ('prepare_consistent_candidates.py','candidate_policy.py','consistent_candidate_profile.py','context_feature_policy.py')})
        for source in sources.values():self.assertTrue(source.is_file(),'Missing release dependency: '+str(source))
        before_code={name:digest(path)for name,path in sources.items()}
        with tempfile.TemporaryDirectory()as temporary:
            root=Path(temporary);package=root/'isolated_package';scripts=package/'scripts';scripts.mkdir(parents=True)
            for name,source in sources.items():shutil.copyfile(source,scripts/name)
            train,dev,graph,raw_train,raw_dev=fixtures(root)
            original={str(path):path.read_bytes()for path in(train,dev,graph)}
            env=dict(os.environ,PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='-1')
            env.pop('PYTHONPATH',None)
            commands=[]
            def command(arguments):
                commands.append(arguments)
                result=subprocess.run([sys.executable,'-B','-u']+arguments,cwd=package,env=env,
                    capture_output=True,text=True,encoding='utf-8',timeout=120)
                self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
                return result
            masters=root/'masters';dataset=root/'consistent'
            command([str(scripts/'prepare_mind_masters.py'),'--train-zip',str(train),'--dev-zip',str(dev),
                     '--wikidata-graph',str(graph),'--output',str(masters),'--seed','42','--dev-fraction','0.05'])
            self.assertEqual((masters/'test/behaviors.tsv').read_bytes(),raw_dev)
            master_manifest=json.loads((masters/'manifest.json').read_text())
            self.assertEqual(master_manifest['development_split']['source_group_count'],20)
            self.assertEqual(master_manifest['development_split']['group_count'],1)
            self.assertEqual(master_manifest['statistics']['train/behaviors.tsv']['rows']+
                             master_manifest['statistics']['dev/behaviors.tsv']['rows'],21)
            self.assertNotIn('.partial-',(masters/'provenance/context_embeddings.json').read_text())
            command([str(scripts/'prepare_consistent_candidates.py'),'--source-data-root',str(masters),
                     '--output',str(dataset),'--seed','42'])
            verify_script=scripts/'verify_certificate.py'
            verify_script.write_text('import json,sys\nfrom consistent_candidate_profile import validate_consistent_candidate_profile\n'
                'p=validate_consistent_candidate_profile(sys.argv[1])\nprint(json.dumps(p))\n',encoding='utf-8')
            check=command([str(verify_script),str(dataset)])
            checked=json.loads(check.stdout.strip().splitlines()[-1])
            self.assertEqual(checked['coverage'],dict(expected_observed_ids=9,proxy_ids=9,
                missing_ids=0,unexpected_ids=0,timestamp_mismatches=0))
            minimum=json.loads((dataset/'provenance/observed_first_impression_min.json').read_text())['all_impressions_min']
            self.assertEqual(set(minimum),{'N'+str(i)for i in range(9)})
            self.assertEqual(minimum['N3'],'2019-11-09T02:00:00')
            self.assertEqual(minimum['N7'],'2019-11-10T00:00:00')
            self.assertEqual((dataset/'test/behaviors.tsv').read_bytes(),raw_dev)
            self.assertEqual((dataset/'train/behaviors.tsv').read_bytes(),(masters/'train/behaviors.tsv').read_bytes())
            self.assertEqual((dataset/'dev/behaviors.tsv').read_bytes(),(masters/'dev/behaviors.tsv').read_bytes())
            exclusions=[json.loads(line)for line in(dataset/'exclusions.jsonl').read_text().splitlines()]
            excluded_test=[row for row in exclusions if row['split']=='test']
            self.assertEqual(len(excluded_test),1);self.assertEqual(excluded_test[0]['source_row'],2)
            self.assertEqual(excluded_test[0]['positive_news'],['N8'])
            profiles=json.loads((dataset/'consistent_candidate_manifest.json').read_text())
            self.assertEqual(profiles['split_evidence']['test']['source_positives'],4)
            self.assertEqual(profiles['split_evidence']['test']['eligible_positives'],3)
            supports={};records={}
            for protocol in('imp','rs30','rs90','rs150','total'):
                path=dataset/checked['outputs']['test_'+protocol]['path']
                supports[protocol],records[protocol]=positive_support(path)
            for support in supports.values():self.assertEqual(support,supports['imp'])
            self.assertEqual(sum(supports['imp'].values()),3)
            raw_support,_=positive_support(dataset/'test/behaviors.tsv')
            self.assertEqual(sum(raw_support.values()),4);self.assertNotEqual(raw_support,supports['imp'])
            self.assertEqual(len(records['imp']),1);self.assertEqual(len(records['total']),3)
            for fields,values in records['total']:
                positives=[n for n,y in values if y=='1'];negatives=[n for n,y in values if y=='0']
                self.assertEqual(len(positives),1);self.assertEqual(set(negatives),{'N5','N6'})
                self.assertNotIn('N0',negatives);self.assertNotIn('N7',negatives)
                self.assertNotIn('N2',negatives);self.assertNotIn('N3',negatives)
            self.assertTrue(any(values[0]==['N1','1']for _,values in records['total']))
            for r30,r90,r150 in zip(records['rs30'],records['rs90'],records['rs150']):
                self.assertEqual(r30[1],r90[1]);self.assertEqual(r90[1],r150[1]);self.assertEqual(len(r30[1]),3)
            # The duplicate train request has N1 and N6 as current positives.
            # Its N6-derived row must never use N1 as a negative.
            for split in('train','dev'):
                _,derived=positive_support(dataset/split/'behaviors_rs50.tsv')
                for _,values in derived:
                    if values[0]==['N6','1']:self.assertNotIn(['N1','0'],values)
            for path,raw in original.items():self.assertEqual(Path(path).read_bytes(),raw)
            for name,source in sources.items():self.assertEqual(digest(source),before_code[name])
            self.assertEqual(len(commands),3)


if __name__=='__main__':unittest.main(verbosity=2)
