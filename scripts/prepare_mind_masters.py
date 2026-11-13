"""Build labeled MIND-small IMP masters from official train/dev ZIP inputs.

Official train is divided by complete request groups. Official dev is the final
labeled test source. Raw row bytes and archives remain unchanged. This command
never loads a model, downloads data, invents publication times, or renames a
relation embedding as a contextual embedding.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import random
import re
import shutil
import stat
import time
import uuid
import zipfile
import zlib

PROFILE='official_mind_small_request_group_holdout_v1'
CHRONOLOGICAL_PROFILE='official_mind_small_chronological_5d_1d_1d_v1'
CHRONOLOGICAL_TRAIN_DATES=tuple(f'2019-11-{day:02d}'for day in range(9,14))
CHRONOLOGICAL_DEV_DATE='2019-11-14'
CHRONOLOGICAL_TEST_DATE='2019-11-15'
CONTEXT_NOT_USED_BY_PAPER_MODELS='context_not_used_by_paper_models'
PAPER_MODELS=('NAML','NRMS','LSTUR','NPA','CNE-SUE','DIGAT')
MEMBERS=('behaviors.tsv','news.tsv','entity_embedding.vec','relation_embedding.vec')
TIME_FORMAT='%m/%d/%Y %I:%M:%S %p'
DEFAULT_MAX_MEMBER_BYTES=512*1024*1024
DEFAULT_MAX_TOTAL_BYTES=1024*1024*1024
DEFAULT_MAX_RATIO=1000.0


def fingerprint(path):
    path=Path(path).resolve();before=path.stat();digest=hashlib.sha256()
    with path.open('rb')as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
        raise ValueError('Source changed while hashing: '+str(path))
    return dict(path=str(path),bytes=before.st_size,sha256=digest.hexdigest(),mtime_ns=before.st_mtime_ns)


def write_json(path,value):
    path=Path(path)
    if path.exists():raise FileExistsError('Preserve existing evidence: '+str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n')as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def _safe_member(info):
    original=getattr(info,'orig_filename',info.filename)
    if '\x00'in original:raise ValueError('ZIP member contains NUL')
    name=original.replace('\\','/')
    if name.startswith('/')or re.match(r'^[A-Za-z]:',name):raise ValueError('Absolute ZIP member is forbidden')
    parts=name.rstrip('/').split('/')
    reserved={'CON','PRN','AUX','NUL',*(f'COM{i}'for i in range(1,10)),*(f'LPT{i}'for i in range(1,10))}
    if any(not p or p in ('.','..')or p.rstrip(' .')!=p or any(c in '<>:"|?*'for c in p)
           or p.split('.',1)[0].upper()in reserved for p in parts):
        raise ValueError('Unsafe ZIP member path')
    mode=info.external_attr>>16;kind=stat.S_IFMT(mode)
    if kind not in (0,stat.S_IFREG,stat.S_IFDIR)or kind==stat.S_IFLNK:
        raise ValueError('ZIP symlink or special file is forbidden')
    if kind==stat.S_IFDIR and not info.is_dir():raise ValueError('ZIP directory metadata is inconsistent')
    if info.flag_bits&1:raise ValueError('Encrypted ZIP members are unsupported')
    return '/'.join(parts)


def inspect_archive(path,*,max_member_bytes,max_total_bytes,max_compression_ratio):
    selected={};inventory=[];seen=set();total=0
    with zipfile.ZipFile(path)as archive:
        infos=archive.infolist()
        if len(infos)>4096:raise ValueError('Too many ZIP members for a small-data archive')
        for info in infos:
            name=_safe_member(info);folded=name.casefold()
            if folded in seen:raise ValueError('Duplicate/case-colliding ZIP member')
            seen.add(folded);total+=info.file_size
            if info.file_size>max_member_bytes or total>max_total_bytes:
                raise ValueError('ZIP uncompressed size exceeds explicit limit')
            if info.file_size and (info.compress_size==0 or info.file_size/info.compress_size>max_compression_ratio):
                raise ValueError('ZIP compression ratio exceeds explicit limit')
            basename=PurePosixPath(name).name.casefold()
            used=not info.is_dir()and basename in MEMBERS
            if used:
                if basename in selected:raise ValueError('Ambiguous repeated official asset basename')
                selected[basename]=info.filename
            inventory.append(dict(member=info.filename,bytes=info.file_size,compressed_bytes=info.compress_size,
                crc32=f'{info.CRC:08x}',selected=used,is_directory=info.is_dir()))
    if set(selected)!=set(MEMBERS):raise ValueError('Official train/dev archive must provide behaviors, news, entity and relation files')
    return dict(selected=selected,members=inventory,declared_uncompressed_bytes=total)


def extract_selected(path,inspection,destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False);result={}
    with zipfile.ZipFile(path)as archive:
        for basename,member in inspection['selected'].items():
            info=archive.getinfo(member);target=destination/basename;digest=hashlib.sha256();crc=0;count=0
            with archive.open(info)as source,target.open('xb')as output:
                for block in iter(lambda:source.read(4*1024*1024),b''):
                    count+=len(block)
                    if count>info.file_size:raise ValueError('ZIP expanded beyond its declared size')
                    output.write(block);digest.update(block);crc=zlib.crc32(block,crc)
            if count!=info.file_size or crc&0xffffffff!=info.CRC:raise ValueError('ZIP size/CRC verification failed')
            actual=fingerprint(target)
            if actual['bytes']!=count or actual['sha256']!=digest.hexdigest():raise ValueError('Extracted member readback differs')
            result[basename]=dict(actual,archive_member=member,crc32=f'{crc&0xffffffff:08x}',crc_verified=True)
    return result


def group_digest(key):return hashlib.sha256('\t'.join(key).encode('utf-8')).hexdigest()


def read_behaviors(path,source_role):
    records=[];groups=defaultdict(list);dates=Counter();digest=hashlib.sha256();start=end=None
    with Path(path).open('rb')as stream:
        for source_row,raw in enumerate(stream,1):
            digest.update(raw)
            try:
                parts=raw.decode('utf-8-sig'if source_row==1 else'utf-8').rstrip('\r\n').split('\t')
                if len(parts)!=5 or not all(parts[i]for i in(0,1,2)):raise ValueError('five fields and request identifiers are required')
                current=datetime.strptime(parts[2],TIME_FORMAT)
                seen=set();positive=[];candidate_ids=[];negative=0
                for token in parts[4].split():
                    news,label=token.rsplit('-',1)
                    if not news or label not in ('0','1')or news in seen:raise ValueError('labeled unique candidates are required')
                    seen.add(news);candidate_ids.append(news)
                    if label=='1':positive.append(news)
                    else:negative+=1
                if not positive or not negative:raise ValueError('both positive and negative labels are required')
            except(UnicodeDecodeError,ValueError)as error:
                raise ValueError(f'{source_role} behaviors row {source_row}: {error}')from error
            key=(parts[1],parts[2],' '.join(parts[3].split()))
            record=dict(source_row=source_row,raw=raw,key=key,positive=tuple(positive),candidates=tuple(candidate_ids),
                history=tuple(parts[3].split()),time=current,source_role=source_role)
            records.append(record);groups[key].append(record);dates[current.date().isoformat()]+=1
            start=current if start is None else min(start,current);end=current if end is None else max(end,current)
    if not records:raise ValueError(source_role+' behavior source is empty')
    stats=dict(rows=len(records),groups=len(groups),duplicate_group_rows=len(records)-len(groups),
        candidates=sum(len(r['candidates'])for r in records),positive_labels=sum(len(r['positive'])for r in records),
        dates=dict(sorted(dates.items())),timestamp_min=start.isoformat(),timestamp_max=end.isoformat(),
        timezone='unspecified_naive_MIND_timestamp',sha256=digest.hexdigest())
    return records,groups,stats


def news_ids(path):
    result=set()
    with Path(path).open(encoding='utf-8-sig')as stream:
        for number,raw in enumerate(stream,1):
            fields=raw.rstrip('\r\n').split('\t')
            if len(fields)!=8 or not fields[0]or fields[0]in result:
                raise ValueError('News metadata must contain unique IDs and eight columns; row '+str(number))
            result.add(fields[0])
    return result


def split_stats(rows):
    sizes=[len(row['candidates'])for row in rows];dates=Counter(row['time'].date().isoformat()for row in rows)
    return dict(rows=len(rows),groups=len({r['key']for r in rows}),candidates=sum(sizes),
        positive_labels=sum(len(r['positive'])for r in rows),candidate_min=min(sizes),candidate_max=max(sizes),
        dates=dict(sorted(dates.items())),timestamp_min=min(r['time']for r in rows).isoformat(),
        timestamp_max=max(r['time']for r in rows).isoformat(),timezone='unspecified_naive_MIND_timestamp')


def copy_exact(source,destination,expected):
    if Path(destination).exists():raise FileExistsError(str(destination))
    shutil.copyfile(source,destination);actual=fingerprint(destination)
    if any(actual[k]!=expected[k]for k in('bytes','sha256')):raise ValueError('Asset copy differs from verified ZIP bytes')
    return actual


def rebase(value,staging,output):
    if isinstance(value,dict):return {key:rebase(item,staging,output)for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [rebase(item,staging,output)for item in value]
    if isinstance(value,Path):value=str(value)
    if isinstance(value,str):
        path=Path(value)
        if path.is_absolute()and(path==staging or staging in path.parents):return str(output/path.relative_to(staging))
    return value


def partition_request_groups(train,train_groups,test,args):
    """Partition complete raw request groups without consulting labels or tests.

    The explicit chronological profile is fixed to the documented small-data
    seven-day roles. Unexpected dates fail rather than silently redefining a
    role. The random profile remains available for earlier software plans.
    """
    policy=getattr(args,'split_policy','random_groups')
    if policy not in('random_groups','chronological'):
        raise ValueError('Unknown split policy')
    if policy=='chronological':
        train_dates={row['time'].date().isoformat()for row in train}
        test_dates={row['time'].date().isoformat()for row in test}
        expected=set(CHRONOLOGICAL_TRAIN_DATES)|{CHRONOLOGICAL_DEV_DATE}
        if train_dates!=expected:
            raise ValueError('Chronological MIND-small requires official train dates 2019-11-09 through 2019-11-14 exactly; observed '+str(sorted(train_dates)))
        if test_dates!={CHRONOLOGICAL_TEST_DATE}:
            raise ValueError('Chronological MIND-small requires official dev dates 2019-11-15 only; observed '+str(sorted(test_dates)))
        dev_keys={row['key']for row in train if row['time'].date().isoformat()==CHRONOLOGICAL_DEV_DATE}
        if any(len({r['time'].date()for r in rows})!=1 for rows in train_groups.values()):
            raise ValueError('A raw request group crosses calendar dates')
        development=dict(policy=policy,fraction_requested=None,group_count=len(dev_keys),source_group_count=len(train_groups),
            fraction_actual=len(dev_keys)/len(train_groups),rounding=None,unit='complete request groups selected by request date',
            random_stream=None,group_sort=None,all_rows_of_a_group_stay_together=True,
            training_dates=list(CHRONOLOGICAL_TRAIN_DATES),development_dates=[CHRONOLOGICAL_DEV_DATE],
            final_test_dates=[CHRONOLOGICAL_TEST_DATE],legacy_dev_fraction_ignored=args.dev_fraction)
        roles=dict(train='official_train dates 2019-11-09 through 2019-11-13',
            dev='official_train date 2019-11-14',test='all official_dev labeled rows dated 2019-11-15')
        profile=CHRONOLOGICAL_PROFILE
    else:
        count=math.floor(len(train_groups)*args.dev_fraction)
        if not 1<=count<len(train_groups):raise ValueError('The explicit group fraction must select at least one dev and one train group')
        dev_keys=set(random.Random(f'{args.seed}:development').sample(sorted(train_groups),count))
        development=dict(policy=policy,fraction_requested=args.dev_fraction,group_count=count,source_group_count=len(train_groups),
            fraction_actual=count/len(train_groups),rounding='floor',unit='(user, raw timestamp, normalized full history)',
            random_stream=f'{args.seed}:development',group_sort='lexicographic tuple order',all_rows_of_a_group_stay_together=True)
        roles=dict(train='official_train minus held-out request groups',dev='held-out official_train groups',test='all official_dev labeled rows')
        profile=PROFILE
    splits={'train':[row for row in train if row['key']not in dev_keys],
        'dev':[row for row in train if row['key']in dev_keys],'test':test}
    if any(not rows for rows in splits.values()):raise ValueError('Every split must contain at least one complete request group')
    chronological_train_dev=max(r['time']for r in splits['train'])<min(r['time']for r in splits['dev'])
    chronological_dev_test=max(r['time']for r in splits['dev'])<min(r['time']for r in splits['test'])
    if policy=='chronological'and not(chronological_train_dev and chronological_dev_test):
        raise ValueError('Chronological train/dev/test time boundaries must be strictly ordered')
    development['selected_group_sha256']=hashlib.sha256('\n'.join(sorted(group_digest(k)for k in dev_keys)).encode()).hexdigest()
    return splits,development,roles,profile,dict(
        train_max_before_dev_min=chronological_train_dev,dev_max_before_test_min=chronological_dev_test,
        exact_expected_chronological_source_dates=policy=='chronological')


def context_policy(args):
    mode=getattr(args,'context_feature_mode','required')
    selected=list(getattr(args,'selected_models',None)or[])
    if mode not in('required',CONTEXT_NOT_USED_BY_PAPER_MODELS):
        raise ValueError('Unknown context feature mode')
    if mode==CONTEXT_NOT_USED_BY_PAPER_MODELS:
        if not selected or len(set(selected))!=len(selected)or any(model not in PAPER_MODELS for model in selected):
            raise ValueError('Explicit distinct selected paper models are required for unused-context mode')
        if getattr(args,'wikidata_graph',None)is not None:
            raise ValueError('Do not supply a Wikidata graph in explicit unused-context mode')
    return mode,selected


def prepare(args):
    started=time.perf_counter()
    if args.dataset!='MIND-small':raise NotImplementedError('MIND-200K requires an explicit large-data user sampler and full-source proxy scope; small is not a substitute')
    source_paths={role:Path(getattr(args,name)).resolve()for role,name in(('official_train','train_zip'),('official_dev','dev_zip'))}
    context_mode,selected_models=context_policy(args)
    graph_arg=getattr(args,'wikidata_graph',None)
    graph=Path(graph_arg).resolve()if graph_arg is not None else None
    output=Path(args.output).resolve()
    if context_mode=='required'and graph is None:raise FileNotFoundError('Explicit local Wikidata graph is required unless unused-context mode is selected for supported paper models')
    immutable_inputs=tuple(source_paths.values())+((graph,)if graph is not None else())
    if output.exists():raise FileExistsError('Output must be a new directory')
    if any(not path.is_file()for path in immutable_inputs):raise FileNotFoundError('Explicit local train/dev ZIPs and any required Wikidata graph must exist; this command never downloads')
    if source_paths['official_train']==source_paths['official_dev']:raise ValueError('Train and dev must be separate source archives')
    if any(path==output or path in output.parents for path in immutable_inputs):
        raise ValueError('Output may not overlap an input file')
    if type(args.seed)is not int or not 0<=args.seed<2**32:raise ValueError('Seed must be unsigned 32-bit')
    if not math.isfinite(args.dev_fraction)or not 0<args.dev_fraction<1:raise ValueError('Development fraction must be between zero and one')
    if min(args.max_member_bytes,args.max_total_bytes)<1 or not math.isfinite(args.max_compression_ratio)or args.max_compression_ratio<1:
        raise ValueError('ZIP limits must be positive and finite')
    archives={};inspections={}
    for role,path in source_paths.items():
        archives[role]=fingerprint(path)
        declared=getattr(args,'expected_train_sha256'if role=='official_train'else'expected_dev_sha256',None)
        if declared and(not re.fullmatch(r'[0-9a-fA-F]{64}',declared)or archives[role]['sha256']!=declared.lower()):
            raise ValueError('Declared source archive SHA-256 differs: '+role)
        inspections[role]=inspect_archive(path,max_member_bytes=args.max_member_bytes,max_total_bytes=args.max_total_bytes,
            max_compression_ratio=args.max_compression_ratio)
    graph_proof=fingerprint(graph)if graph is not None else None
    output.parent.mkdir(parents=True,exist_ok=True)
    needed=sum(item['declared_uncompressed_bytes']for item in inspections.values())*4
    if shutil.disk_usage(output.parent).free<needed:raise ValueError('Insufficient staging and asset headroom')
    staging=output.with_name(output.name+'.partial-'+uuid.uuid4().hex);staging.mkdir()
    try:
        raw={role:extract_selected(path,inspections[role],staging/'provenance/raw'/role)for role,path in source_paths.items()}
        train,train_groups,train_source_stats=read_behaviors(raw['official_train']['behaviors.tsv']['path'],'official_train')
        test,test_groups,test_source_stats=read_behaviors(raw['official_dev']['behaviors.tsv']['path'],'official_dev')
        if set(train_groups)&set(test_groups):raise ValueError('Official train/dev share complete request groups; input provenance is inconsistent')
        known=news_ids(raw['official_train']['news.tsv']['path'])|news_ids(raw['official_dev']['news.tsv']['path'])
        if any(not(set(row['history'])|set(row['candidates']))<=known for row in train+test):
            raise ValueError('A behavior/history article is absent from the complete metadata union')
        splits,development,source_roles,profile,temporal_validation=partition_request_groups(train,train_groups,test,args)
        stats={};files={};assets={};partition=hashlib.sha256()
        with(staging/'request_mapping.jsonl').open('x',encoding='utf-8',newline='\n')as mapping:
            for split,rows in splits.items():
                folder=staging/split;folder.mkdir()
                with(folder/'behaviors.tsv').open('xb')as stream:
                    for output_row,row in enumerate(rows,1):
                        stream.write(row['raw'])
                        mapping.write(json.dumps(dict(split=split,output_row=output_row,source_role=row['source_role'],
                            source_row=row['source_row'],group_sha256=group_digest(row['key']),positive_news=list(row['positive'])),separators=(',',':'))+'\n')
                stats[split+'/behaviors.tsv']=split_stats(rows);files[split+'/behaviors.tsv']=fingerprint(folder/'behaviors.tsv')
                role='official_train'if split in('train','dev')else'official_dev'
                for name in('news.tsv','entity_embedding.vec','relation_embedding.vec'):
                    source=raw[role][name];copied=copy_exact(source['path'],folder/name,source)
                    assets[split+'/'+name]=dict(source_archive_role=role,source=source,output=copied)
        for row in train:partition.update(row['raw'])
        if partition.hexdigest()!=raw['official_train']['behaviors.tsv']['sha256']:
            raise ValueError('Source row bytes were not preserved during group partitioning')
        if files['train/behaviors.tsv']['bytes']+files['dev/behaviors.tsv']['bytes']!=raw['official_train']['behaviors.tsv']['bytes']:
            raise ValueError('Source train rows were lost or duplicated')
        if files['test/behaviors.tsv']['sha256']!=raw['official_dev']['behaviors.tsv']['sha256']:
            raise ValueError('Official dev bytes were changed in the final test master')
        if stats['train/behaviors.tsv']['positive_labels']+stats['dev/behaviors.tsv']['positive_labels']!=train_source_stats['positive_labels']:
            raise ValueError('A source positive occurrence was lost')
        context_source=None
        if context_mode=='required':
            from context_embeddings import prepare_context_embeddings
            context_receipt=prepare_context_embeddings(
                embedding_files={split:staging/split/'entity_embedding.vec'for split in splits},graph_path=graph,
                output_paths={split:staging/split/'context_embedding.vec'for split in splits},dimension=100)
            if not isinstance(context_receipt,dict):raise TypeError('Context helper must return a JSON receipt')
            for split in splits:
                path=staging/split/'context_embedding.vec'
                if not path.is_file()or not path.stat().st_size:raise ValueError('Context generation produced no usable file')
                assets[split+'/context_embedding.vec']=dict(kind='generated_context',source_entity=assets[split+'/entity_embedding.vec']['output'],
                    source_graph=graph_proof,receipt='provenance/context_embeddings.json',output=fingerprint(path))
            write_json(staging/'provenance/context_embeddings.json',rebase(context_receipt,staging,output))
            context_source=Path(__import__('context_embeddings').__file__).resolve()
        context_features=dict(mode=context_mode,required_by_selected_models=context_mode=='required',
            allowed_models=list(PAPER_MODELS)if context_mode==CONTEXT_NOT_USED_BY_PAPER_MODELS else None,
            selected_models=selected_models,context_outputs=[split+'/context_embedding.vec'for split in splits]if context_mode=='required'else[])
        context_details=dict(mode=context_mode,
            algorithm='See context helper receipt: genuine entity-plus-Wikidata contextual embeddings'if context_mode=='required'else'Explicitly omitted: supported selected paper encoders do not consume contextual KG embeddings',
            dimension=100 if context_mode=='required'else None,relation_embeddings_renamed=False,source_graph=graph_proof,
            receipt='provenance/context_embeddings.json'if context_mode=='required'else None,
            outputs_created=context_mode=='required',dummy_vectors_created=False)
        files['request_mapping.jsonl']=fingerprint(staging/'request_mapping.jsonl')
        overlap={name:0 for name in('train_dev_group_overlap','train_test_group_overlap','dev_test_group_overlap')}
        manifest=dict(schema=1,status='complete',profile=profile,dataset='MIND-small',created_utc=datetime.now(timezone.utc).isoformat(),
            output=str(output),seed=args.seed,fraction=1.0,fraction_scope='All labeled rows in the supplied archive roles; upstream authenticity depends on recorded input provenance.',
            historical_R2_reproduction=False,split_policy=development['policy'],source_roles=source_roles,
            context_feature_mode=context_mode,selected_models=selected_models,context_features=context_features,
            source_archives={role:dict(archives[role],inspection=inspections[role],extracted_members=raw[role])for role in raw},
            source_statistics=dict(official_train=train_source_stats,official_dev=test_source_stats),output_files=files,statistics=stats,assets=assets,
            development_split=development,
            date_audit=dict(timezone='unspecified_naive_MIND_timestamp',timezone_conversion_performed=False,
                source_train_max_before_final_test_min=train_source_stats['timestamp_max']<test_source_stats['timestamp_min'],
                development_is_random_request_group_holdout_not_chronological=development['policy']=='random_groups',
                **temporal_validation),
            validation=dict(overlap,**temporal_validation,source_train_rows_partitioned_once=True,all_source_positive_occurrences_preserved=True,
                official_dev_is_byte_identical_final_test=True,raw_row_bytes_preserved=True,all_news_references_have_metadata=True),
            proxy=dict(status='not_generated_here',next_step='prepare_consistent_candidates.py computes complete first-impression minima across these three masters; histories are not appearance timestamps'),
            context=context_details,
            producer=fingerprint(Path(__file__)),context_helper=fingerprint(context_source)if context_source is not None else None,elapsed_seconds=time.perf_counter()-started,
            notes=['MIND-200K is not constructed by this command and must not be substituted with MIND-small.',
                'Only the four documented official ZIP assets are extracted; unsupported members are inventoried without execution.',
                'The original ZIPs and graph are read-only. IMP masters preserve source row bytes, labels and order within each split.',
                'Publication proxies and candidate files are separate downstream steps; no historical submission reproduction is asserted.'])
        proofs={**{role:archives[role]for role in archives},**({'wikidata_graph':graph_proof}if graph_proof is not None else{})}
        for name,proof in proofs.items():
            actual=Path(proof['path']).stat()
            if actual.st_size!=proof['bytes']or actual.st_mtime_ns!=proof['mtime_ns']:raise ValueError('An immutable source changed during preparation: '+name)
        write_json(staging/'manifest.json',rebase(manifest,staging,output))
        # Every persisted stage-local receipt path was normalized before publishing.
        staging.rename(output)
        result=dict(status='complete',data_root=str(output),manifest=fingerprint(output/'manifest.json'),
            elapsed_seconds=time.perf_counter()-started,statistics=stats,gpu_used=False,source_data_modified=False)
        print(json.dumps(result,ensure_ascii=False),flush=True)
        return result
    except BaseException:
        import traceback
        write_json(staging/'BUILD_FAILED.json',dict(status='failed',error=traceback.format_exc(),source_preserved=True))
        raise


def parser():
    result=argparse.ArgumentParser(description=__doc__)
    result.add_argument('--dataset',choices=('MIND-small',),default='MIND-small')
    result.add_argument('--train-zip',type=Path,required=True)
    result.add_argument('--dev-zip',type=Path,required=True,help='Official labeled dev archive; used as final test source')
    result.add_argument('--wikidata-graph',type=Path,help='Explicit local graph TSV or ZIP, required in context feature mode required; no automatic download')
    result.add_argument('--context-feature-mode',choices=('required',CONTEXT_NOT_USED_BY_PAPER_MODELS),default='required',
        help='Only explicit supported paper-model mode omits unused contextual KG files; no dummy vectors are generated')
    result.add_argument('--selected-models',nargs='+',help='Explicit paper models when contextual KG features are unused')
    result.add_argument('--output',type=Path,required=True,help='New IMP master directory')
    result.add_argument('--seed',type=int,default=42)
    result.add_argument('--split-policy',choices=('random_groups','chronological'),default='random_groups',
        help='chronological: official train Nov9-13/train, Nov14/dev; official dev Nov15/test; random_groups preserves the earlier grouped holdout')
    result.add_argument('--dev-fraction',type=float,default=0.05)
    result.add_argument('--expected-train-sha256')
    result.add_argument('--expected-dev-sha256')
    result.add_argument('--max-member-bytes',type=int,default=DEFAULT_MAX_MEMBER_BYTES)
    result.add_argument('--max-total-bytes',type=int,default=DEFAULT_MAX_TOTAL_BYTES)
    result.add_argument('--max-compression-ratio',type=float,default=DEFAULT_MAX_RATIO)
    return result


def main():prepare(parser().parse_args())
if __name__=='__main__':main()
