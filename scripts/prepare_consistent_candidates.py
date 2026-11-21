"""Prepare a complete first-impression proxy and consistent strict candidates.

No model is loaded. Inputs remain read-only, IMP masters are copied unchanged,
and zero-negative requests are documented rather than silently removed.
"""
import argparse
from pathlib import Path
from context_feature_policy import context_feature_spec,declaration_fields
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib,json,os,shutil,sys,time,uuid
from candidate_policy import PROFILE,RECIPE,PublicationIndex,sample_negative_prefixes,parse_timestamp
from consistent_candidate_profile import (validate_consistent_candidate_profile, OUTPUT_FILES,
    LARGE_MASTER_MAP, LARGE_PROXY_SCOPE, MASTER_MANIFEST_COPY, MINIMUM_MAP, SAMPLED_USER_IDS,
    verify_large_minimum_source)
SOURCE=TARGET=STAGE=None
START=0.0
TIMEFMT='%m/%d/%Y %I:%M:%S %p'
copied={}
FIRST_MIN={}
CLICK_MIN={}
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def status(message):print('[Candidate preparation] '+message,flush=True)

def now():return datetime.now(timezone.utc).isoformat()

def group_sha(key):return hashlib.sha256('\t'.join(key).encode()).hexdigest()

def fingerprint(path):
    path=Path(path);before=path.stat();digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('File changed while hashing: '+str(path))
    return dict(path=str(path),bytes=before.st_size,sha256=digest.hexdigest(),mtime_ns=before.st_mtime_ns)

def relative_fp(path):
    result=fingerprint(path);result['path']=str(Path(path).relative_to(STAGE)).replace('\\','/');return result

def expected_same(actual,expected):
    if any(actual[key]!=expected[key] for key in ('bytes','sha256')):raise ValueError('Fingerprint differs: '+str(actual['path']))
    return actual

def write_json(path,value):
    path=Path(path)
    if path.exists():raise FileExistsError(str(path))
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())

def copy_verified(source,relative,expected=None):
    source=Path(source);before=source.stat();target=STAGE/relative;target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():raise FileExistsError(str(target))
    digest=hashlib.sha256();written=0
    with source.open('rb') as src,target.open('xb') as dst:
        for block in iter(lambda:src.read(8*1024*1024),b''):
            dst.write(block);digest.update(block);written+=len(block)
        dst.flush();os.fsync(dst.fileno())
    after=source.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Copy source changed')
    actual=dict(path=str(source),bytes=written,sha256=digest.hexdigest())
    if expected is not None:expected_same(actual,expected)
    expected_same(fingerprint(target),actual)
    copied[relative]=dict(source=dict(path=str(source),bytes=written,sha256=digest.hexdigest(),
        mtime_before_ns=before.st_mtime_ns,mtime_after_ns=after.st_mtime_ns),output=relative_fp(target))
    return copied[relative]['output']

class Writer:
    def __init__(self,relative):
        self.path=STAGE/relative;self.path.parent.mkdir(parents=True,exist_ok=True)
        self.stream=self.path.open('xb',buffering=1024*1024);self.digest=hashlib.sha256();self.bytes=0
        self.stats=dict(rows=0,candidates=0,positive_labels=0,candidate_min=None,candidate_max=0)
        self.gt=hashlib.sha256();self.groups=set()
    def raw(self,raw):self.stream.write(raw);self.digest.update(raw);self.bytes+=len(raw)
    def json(self,value):self.raw((json.dumps(value,separators=(',',':'),ensure_ascii=False)+'\n').encode())
    def row(self,head,positive,negatives,gsha):
        values=(positive+'-1'+' '+ ' '.join(news+'-0' for news in negatives)) if negatives else positive+'-1'
        raw=('\t'.join(head+[values])+'\n').encode();self.raw(raw)
        self.add_stats(len(negatives)+1,1);self.gt.update((gsha+'\t'+positive+'\n').encode());self.groups.add(gsha)
    def native(self,raw,count,positives,gsha):
        self.raw(raw if raw.endswith(b'\n') else raw+b'\n');self.add_stats(count,len(positives));self.groups.add(gsha)
        for positive in positives:self.gt.update((gsha+'\t'+positive+'\n').encode())
    def add_stats(self,candidates,positives):
        item=self.stats;item['rows']+=1;item['candidates']+=candidates;item['positive_labels']+=positives
        item['candidate_min']=candidates if item['candidate_min'] is None else min(item['candidate_min'],candidates)
        item['candidate_max']=max(item['candidate_max'],candidates)
    def close(self):
        self.stream.flush();os.fsync(self.stream.fileno());self.stream.close()
        if self.path.stat().st_size!=self.bytes:raise ValueError('Output size changed')
        self.stats['candidate_mean']=self.stats['candidates']/self.stats['rows'] if self.stats['rows'] else None
        self.stats['groups']=len(self.groups)
        return dict(path=str(self.path.relative_to(STAGE)).replace('\\','/'),bytes=self.bytes,sha256=self.digest.hexdigest())

def load_records(split,expected):
    path=STAGE/split/'behaviors.tsv';before=path.stat();digest=hashlib.sha256();records=[];groups=defaultdict(set)
    with path.open('rb') as stream:
        for index,raw in enumerate(stream,1):
            digest.update(raw);parts=raw.decode('utf-8-sig' if index==1 else 'utf-8').rstrip('\r\n').split('\t')
            if len(parts)!=5:raise ValueError('Bad IMP columns')
            normalized=' '.join(parts[3].split())
            current=datetime.strptime(parts[2],TIMEFMT)
            positives=[];seen=set();count=0
            for token in parts[4].split():
                news,label=token.rsplit('-',1)
                if label not in ('0','1') or news in seen:raise ValueError('Bad IMP label or duplicate')
                seen.add(news);count+=1
                if news not in FIRST_MIN or current<FIRST_MIN[news]:FIRST_MIN[news]=current
                if label=='1' and (news not in CLICK_MIN or current<CLICK_MIN[news]):CLICK_MIN[news]=current
                if label=='1':positives.append(news)
            if not positives:raise ValueError('No positive in IMP request')
            key=(parts[1],parts[2],normalized);groups[key].update(positives)
            records.append(dict(source_row=index,head=parts[:4],positives=tuple(positives),key=key,
                candidate_count=count,raw=raw if split=='test' else None))
    after=path.stat();proof=dict(path=str(path),bytes=before.st_size,sha256=digest.hexdigest(),
        mtime_before_ns=before.st_mtime_ns,mtime_after_ns=after.st_mtime_ns)
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Master changed during read')
    expected_same(proof,expected)
    return records,groups,proof

def _source_expected(manifest,relative,asset=False):
    if not manifest:return None
    item=manifest.get('assets',{}).get(relative,{}) if asset else manifest.get('output_files',{}).get(relative)
    return item.get('output') if asset else item


def large_output_bound(index, records_by_split, source_proofs):
    """Budget for files still to be written, using each request's actual availability window."""
    token_bytes = max(len(news.encode('utf-8')) + 3 for news in index.ids)
    total = source_proofs['test']['bytes']
    for split, records in records_by_split.items():
        for record in records:
            left, right = index.window_bounds(datetime.strptime(record['head'][2], TIMEFMT))
            window = right - left
            # All history/positive exclusions can only shrink this count.
            if split == 'test':
                sizes = (min(30,window), min(90,window), min(150,window), window)
            else:
                sizes = (min(50,window),)
            header = sum(len(value.encode('utf-8'))+1 for value in record['head'][1:]) + 64
            # Allow for one GT, request/positive mapping records and exclusion evidence.
            n_positive = len(record['positives'])
            total += n_positive * (len(sizes)*(header+token_bytes) + sum(sizes)*token_bytes
                                    + 2048 + 8*token_bytes)
    return total

def prepare(args):
    global SOURCE,TARGET,STAGE,START,copied,FIRST_MIN,CLICK_MIN
    START=time.perf_counter();copied={};FIRST_MIN={};CLICK_MIN={}
    SOURCE=args.source_data_root.resolve();TARGET=args.output.resolve()
    if not SOURCE.is_dir():raise ValueError('Source dataset root is missing')
    if TARGET.exists() or TARGET==SOURCE or SOURCE in TARGET.parents or TARGET in SOURCE.parents:
        raise ValueError('Output must be a new directory separate from the read-only source')
    control=args.control_output.resolve() if args.control_output else None
    if control and (control==SOURCE or SOURCE in control.parents or control in SOURCE.parents
                    or control==TARGET or TARGET in control.parents or control in TARGET.parents):
        raise ValueError('Control output must be separate from source and atomic dataset output')
    if control and ((control/'PREPARATION_VERIFICATION.json').exists() or (control/'full_total/manifest.json').exists()):
        raise FileExistsError('Control evidence already exists')
    if type(args.seed)is not int or not 0<=args.seed<2**32:raise ValueError('Seed must be unsigned 32-bit')
    context=context_feature_spec(SOURCE)
    asset_names=('news.tsv','entity_embedding.vec')+(('context_embedding.vec',)if context['required']else())
    required=[SOURCE/split/name for split in ('train','dev','test')for name in('behaviors.tsv',)+asset_names]
    if any(not path.is_file() for path in required):raise FileNotFoundError('Explicit labeled IMP masters and all three split assets are required')
    TARGET.parent.mkdir(parents=True,exist_ok=True)
    STAGE=TARGET.with_name(TARGET.name+'.partial-'+uuid.uuid4().hex)
    STAGE.mkdir();(STAGE/'provenance').mkdir()
    source_manifest=read(SOURCE/'manifest.json') if (SOURCE/'manifest.json').is_file() else None
    source_manifest_proof=fingerprint(SOURCE/'manifest.json') if source_manifest else None
    large = (source_manifest or {}).get('dataset') == 'MIND-200K'
    status('Copying unchanged masters and assets; source files are never rewritten')
    assets={}
    for split in ('train','dev','test'):
        relative=split+'/behaviors.tsv';copy_verified(SOURCE/relative,relative,_source_expected(source_manifest,relative))
        names=asset_names+(('relation_embedding.vec',)if(SOURCE/split/'relation_embedding.vec').is_file()else())
        for name in names:
            relative=split+'/'+name
            output=copy_verified(SOURCE/relative,relative,_source_expected(source_manifest,relative,True))
            assets[relative]=dict(source=copied[relative]['source'],output=output)
    known=set();news_order={'<PAD>':0}
    for split in ('train','dev','test'):
        with (STAGE/split/'news.tsv').open(encoding='utf-8-sig')as stream:
            for raw in stream:
                fields=raw.rstrip('\r\n').split('\t')
                if len(fields)!=8 or not fields[0]:raise ValueError('Expected eight-column news metadata')
                news=fields[0];known.add(news)
                if news not in news_order:news_order[news]=len(news_order)
    records_by_split={};groups_by_split={};source_proofs={};master_stats={};seen_groups=set()
    for split in ('train','dev','test'):
        relative=split+'/behaviors.tsv'
        records,groups,proof=load_records(split,copied[relative]['output'])
        hashes={group_sha(key)for key in groups}
        if seen_groups&hashes:raise ValueError('Source train/dev/test request groups overlap; establish split provenance first')
        seen_groups.update(hashes);records_by_split[split]=records;groups_by_split[split]=groups;source_proofs[split]=proof
        sizes=[record['candidate_count']for record in records]
        if not records:raise ValueError('Every supplied split must have labeled IMP requests')
        master_stats[relative]=dict(rows=len(records),candidates=sum(sizes),positive_labels=sum(len(r['positives'])for r in records),
            candidate_min=min(sizes),candidate_max=max(sizes),groups=len(groups))
    if not FIRST_MIN or not set(FIRST_MIN)<=known:raise ValueError('Observed article has no metadata; no implicit universe shrink is allowed')
    publication=FIRST_MIN
    for records in records_by_split.values():
        if any(not set(record['head'][3].split())<=known for record in records):raise ValueError('History article has no metadata')
    minimum_relative=MINIMUM_MAP
    if large:
        copy_verified(SOURCE/'manifest.json',MASTER_MANIFEST_COPY,source_manifest_proof)
        sample = source_manifest.get('user_sample',{})
        if not sample.get('user_ids'):
            raise ValueError('MIND-200K requires a pinned sample of user IDs')
        cohort_pin = copy_verified(SOURCE/SAMPLED_USER_IDS,SAMPLED_USER_IDS,sample['user_ids'])
        observed_users = {key[0] for groups in groups_by_split.values() for key in groups}
        selected = read(STAGE/SAMPLED_USER_IDS)
        if (not isinstance(selected,list) or len(observed_users)!=sample.get('count')
                or len(selected)!=len(observed_users) or set(selected)!=observed_users):
            raise ValueError('Sampled user list differs from users in the three MIND-200K masters')
        pin = source_manifest.get('full_source_minimum_map')
        if not pin:
            raise ValueError('MIND-200K requires the pinned first-impression map from full MIND-Large')
        map_copy = copy_verified(SOURCE/LARGE_MASTER_MAP,minimum_relative,pin)
        provided = read(STAGE/minimum_relative)
        verify_large_minimum_source(source_manifest,provided,map_copy)
        full_map = provided.get('all_impressions_min')
        if not isinstance(full_map,dict) or not full_map:
            raise ValueError('Empty full MIND-Large first-impression map')
        publication = {news:parse_timestamp(stamp) for news,stamp in full_map.items()}
        if publication.keys() - known:
            raise ValueError('Full MIND-Large first-impression map contains an article without metadata')
        if any(news not in publication or publication[news]>stamp for news,stamp in FIRST_MIN.items()):
            raise ValueError('Full MIND-Large first impressions must cover and precede or equal sampled minima')
        if args.source_minimum_map and read(args.source_minimum_map).get('all_impressions_min')!=full_map:
            raise ValueError('Supplied minimum map differs from the pinned full MIND-Large map')
    elif args.source_minimum_map:
        provided=read(args.source_minimum_map)
        expected={news:parse_timestamp(when)for news,when in provided.get('all_impressions_min',{}).items()}
        if expected!=publication:raise ValueError('Supplied minimum map is not the complete exact minimum of the supplied IMP masters')
        copy_verified(args.source_minimum_map.resolve(),minimum_relative,fingerprint(args.source_minimum_map))
    else:
        write_json(STAGE/minimum_relative,dict(schema=1,time_format='%Y-%m-%dT%H:%M:%S',timezone='unspecified_naive',
            definition='Minimum impression-candidate timestamp over supplied train/dev/test; history appearance and click time are not publication time.',
            input_files={split:dict(proof,path=str(TARGET/split/'behaviors.tsv'))for split,proof in source_proofs.items()},
            all_impressions_min={n:t.isoformat()for n,t in sorted(publication.items())},
            clicked_only_min={n:t.isoformat()for n,t in sorted(CLICK_MIN.items())}))
    with(STAGE/'news_publish_time.tsv').open('x',encoding='utf-8',newline='\n')as stream:
        for news,when in sorted(publication.items()):stream.write(news+'\t'+when.strftime(TIMEFMT)+'\n')
    index=PublicationIndex(publication)
    # Conservative size preflight; candidate generation below never truncates to fit disk.
    test_positives=master_stats['test/behaviors.tsv']['positive_labels']
    largest_token=max(len(n.encode())+3 for n in publication)
    upper_bytes=(large_output_bound(index,records_by_split,source_proofs) if large else
                 test_positives*(len(publication)*largest_token+1024)+sum(p['bytes']for p in source_proofs.values()))
    if shutil.disk_usage(TARGET.parent).free<upper_bytes*1.1:
        raise ValueError('Insufficient disk headroom for conservative full-Total output bound')
    exclusions=Writer('exclusions.jsonl');mapping=Writer('candidate_mapping.jsonl');evaluation_map=Writer('evaluation_mapping.jsonl')
    outputs={};stats={};support={}
    try:
        for split in ('train','dev','test'):
            status('Generating '+split+' with one strict request-time base')
            records=records_by_split[split];groups=groups_by_split[split]
            names=(split+'_rs50',)if split!='test'else('test_rs30','test_rs90','test_rs150','test_total')
            writers={name:Writer(OUTPUT_FILES[name])for name in names}
            native=Writer(OUTPUT_FILES['test_imp'])if split=='test'else None
            counts=Counter({name:0 for name in ('source_rows','source_positives','excluded_rows','excluded_positives','eligible_rows','eligible_positives')})
            excluded_groups=set();gt=hashlib.sha256()
            for record in records:
                row,head,positives,key=(record[k]for k in('source_row','head','positives','key'))
                gsha=group_sha(key)
                pool=index.eligible_window(datetime.strptime(head[2],TIMEFMT),head[3].split(),groups[key])
                counts['source_rows']+=1;counts['source_positives']+=len(positives)
                if not pool:
                    counts['excluded_rows']+=1;counts['excluded_positives']+=len(positives);excluded_groups.add(gsha)
                    exclusions.json(dict(split=split,source_row=row,group_sha256=gsha,positive_news=list(positives),
                        positive_occurrences=len(positives),reason='zero_eligible_negative_pool',master_preserved=True))
                    continue
                counts['eligible_rows']+=1;counts['eligible_positives']+=len(positives)
                if native is not None:native.native(record['raw'],record['candidate_count'],positives,gsha)
                for positive_index,positive in enumerate(positives):
                    identifier=split+'_'+str(row)+'_p'+str(positive_index);header=[identifier]+head[1:]
                    gt.update((gsha+'\t'+positive+'\n').encode())
                    selected=sample_negative_prefixes(pool,counts=(50,)if split!='test'else(30,90,150),
                        seed=args.seed,split=split,source_row=row,positive_index=positive_index)
                    if split!='test':writers[split+'_rs50'].row(header,positive,selected[50],gsha)
                    else:
                        for k in(30,90,150):writers['test_rs'+str(k)].row(header,positive,selected[k],gsha)
                        writers['test_total'].row(header,positive,pool,gsha)
                    item=dict(split=split,source_row=row,positive_index=positive_index,output_id=identifier,
                        group_sha256=gsha,positive_news=positive,eligible_negative_count=len(pool),
                        candidate_counts={str(k):len(value)+1 for k,value in selected.items()})
                    mapping.json(item)
                    if split=='test':
                        item.update(eval_row=writers['test_total'].stats['rows'],source_eval_id=identifier,full_total_candidates=len(pool)+1)
                        evaluation_map.json(item)
            if not counts['eligible_positives']:raise ValueError('A derived split has no eligible positives; preserve exclusion evidence and choose a valid source scope')
            for name,writer in writers.items():
                outputs[name]=writer.close();stats[OUTPUT_FILES[name]]=writer.stats
                if writer.gt.hexdigest()!=gt.hexdigest():raise ValueError('Derived positive occurrence mapping differs')
            if native is not None:
                outputs['test_imp']=native.close();stats[OUTPUT_FILES['test_imp']]=native.stats
                if native.gt.hexdigest()!=gt.hexdigest():raise ValueError('Eligible IMP support differs')
                if not counts['excluded_rows']:expected_same(outputs['test_imp'],copied['test/behaviors.tsv']['output'])
            support[split]=dict(counts,source_request_groups=len(groups),excluded_request_groups=len(excluded_groups),
                eligible_request_groups=len(groups)-len(excluded_groups),eligible_GT_multiset_sha256=gt.hexdigest())
    finally:
        exclusions_fp=exclusions.close();mapping_fp=mapping.close();evaluation_fp=evaluation_map.close()
    coverage=dict(expected_observed_ids=len(publication),proxy_ids=len(publication),missing_ids=0,unexpected_ids=0,timestamp_mismatches=0)
    producer=fingerprint(Path(__file__))
    cert=dict(schema=1,status='complete',profile=PROFILE,seed=args.seed,sampling=RECIPE,created_utc=now(),
        publication_proxy=relative_fp(STAGE/'news_publish_time.tsv'),source_minimum_map=relative_fp(STAGE/minimum_relative),
        imp_inputs={split+'/behaviors.tsv':relative_fp(STAGE/split/'behaviors.tsv')for split in('train','dev','test')},
        outputs=outputs,exclusions=exclusions_fp,coverage=coverage,statistics=stats,split_evidence=support,
        validation=dict(same_base_policy_all_splits=True,master_IMP_preserved=True,GT_multiset_consistent_all_test_protocols=True,exclusions_recorded=True),
        current_positive_scope='Union within the same (user,timestamp,normalized full history) request group; not later user clicks.',
        metadata_universe=dict(news_ids=len(known),all_proxy_IDs_present=True),producer=producer)
    cert.update(declaration_fields(context))
    if large:
        cert.update(dataset='MIND-200K',publication_scope=LARGE_PROXY_SCOPE,
                    source_master_copy=relative_fp(STAGE/MASTER_MANIFEST_COPY),user_ids=cohort_pin)
    write_json(STAGE/'consistent_candidate_manifest.json',cert)
    manifest=dict(schema=2,status='complete',created_utc=now(),seed=args.seed,fraction=1.0,
        fraction_scope='All supplied source master rows; this does not establish official dataset completeness or historical submission provenance.',
        source=str(SOURCE),output=str(TARGET),historical_R2_reproduction=False,candidate_profile=PROFILE,
        assets=assets,output_files={},statistics=dict(master_stats,**stats),consistent_candidate_certificate=relative_fp(STAGE/'consistent_candidate_manifest.json'),
        source_dataset_manifest=source_manifest_proof,script_sha256=producer['sha256'],split_evidence=support,
        validation=dict(IMP_master_files_byte_identical=True,train_dev_group_overlap=0,train_test_group_overlap=0,dev_test_group_overlap=0),
        notes=['First-impression proxies are not actual publication times. Histories do not define first impression.',
            'The negative rule is shared by RandS train/dev/test and Total. Native IMP candidate sets are preserved separately.',
            'Own GT is always retained for eligible requests; zero-negative requests are logged and excluded only from derivatives.',
            'The RandS50 pool is separate from the optimizer negative-sampling count.'])
    manifest.update(declaration_fields(context))
    if large:
        manifest.update(dataset='MIND-200K',publication_scope=LARGE_PROXY_SCOPE,
                        user_sample=dict(sample,user_ids=cohort_pin))
    for relative,item in copied.items():
        if relative.endswith('/behaviors.tsv'):manifest['output_files'][relative]=dict(item['output'],path=str(TARGET/relative))
    for name,item in outputs.items():manifest['output_files'][item['path']]=dict(item,path=str(TARGET/item['path']))
    for relative,item in(('exclusions.jsonl',exclusions_fp),('candidate_mapping.jsonl',mapping_fp),('evaluation_mapping.jsonl',evaluation_fp),('news_publish_time.tsv',cert['publication_proxy'])):
        manifest['output_files'][relative]=dict(item,path=str(TARGET/relative))
    write_json(STAGE/'manifest.json',manifest)
    status('Hash-reading output files and checking complete proxy/certificate coverage')
    checked=validate_consistent_candidate_profile(STAGE)
    for item in copied.values():
        src=Path(item['source']['path']);stat=src.stat()
        if stat.st_size!=item['source']['bytes']or stat.st_mtime_ns!=item['source']['mtime_after_ns']:
            raise ValueError('An immutable source changed after copying')
    os.rename(STAGE,TARGET)
    total=dict(schema=2,status='complete',protocol='total',candidate_profile=PROFILE,dataset_root=str(TARGET),
        dataset_manifest=fingerprint(TARGET/'manifest.json'),consistent_candidate_certificate=fingerprint(TARGET/'consistent_candidate_manifest.json'),
        output=dict(outputs['test_total'],path=str(TARGET/OUTPUT_FILES['test_total'])),statistics=stats[OUTPUT_FILES['test_total']],
        source_evaluations={kind:dict(outputs['test_'+kind],path=str(TARGET/OUTPUT_FILES['test_'+kind]))for kind in('imp','rs30','rs90','rs150')},
        evaluation_mapping=dict(evaluation_fp,path=str(TARGET/'evaluation_mapping.jsonl')),
        validation=dict(same_GT_request_support_all_protocols=True,history_and_current_positives_excluded=True,
            strict_time_window=True,complete_proxy_coverage=True,all_news_in_model_vocabularies=True),
        news_vocabulary_scope='News IDs are the ordered union of supplied news.tsv files; every generated proxy/article ID is checked against that union.',
        source_legacy_Total_reused=False,new_candidate_generation_performed=True)
    descriptor=TARGET/'provenance/full_total_manifest.json';write_json(descriptor,total)
    receipt=dict(status='complete',data_root=str(TARGET),created_utc=now(),elapsed_seconds=time.perf_counter()-START,
        dataset_manifest=fingerprint(TARGET/'manifest.json'),consistent_candidate_manifest=fingerprint(TARGET/'consistent_candidate_manifest.json'),
        full_total_manifest=fingerprint(descriptor),statistics=stats,split_evidence=support,master_IMP_preserved=True,
        gpu_used=False,models_loaded=False,source_data_modified=False,source_files={k:v['source']for k,v in copied.items()})
    write_json(TARGET/'provenance/PREPARATION_VERIFICATION.json',receipt)
    if control:
        control.mkdir(parents=True,exist_ok=True);(control/'full_total').mkdir(exist_ok=True)
        write_json(control/'full_total/manifest.json',total);write_json(control/'PREPARATION_VERIFICATION.json',receipt)
    print(json.dumps(dict(status='complete',data_root=str(TARGET),candidate_manifest=str(descriptor),elapsed_seconds=receipt['elapsed_seconds'])),flush=True)
    return receipt

def parser():
    result=argparse.ArgumentParser(description=__doc__)
    result.add_argument('--source-data-root',type=Path,required=True,help='Disjoint labeled IMP train/dev/test masters plus split news/entity/context assets')
    result.add_argument('--output',type=Path,required=True,help='New dataset directory; inputs are never overwritten')
    result.add_argument('--seed',type=int,default=42)
    result.add_argument('--source-minimum-map',type=Path,help='Optional independent map to compare against the specified dataset scope; MIND-200K always uses its pinned full MIND-Large map')
    result.add_argument('--control-output',type=Path,help='Optional separate receipt/Total descriptor directory')
    return result

def main():
    try:prepare(parser().parse_args())
    except BaseException:
        if STAGE is not None and STAGE.is_dir():
            import traceback
            failure=STAGE/'BUILD_FAILED.json'
            if not failure.exists():write_json(failure,dict(status='failed',error=traceback.format_exc(),source_preserved=True))
        raise

if __name__=='__main__':main()
