"""Portable raw-MIND -> strict candidates -> paired IMP/RandS experiments.

Preparation and execution are explicit. No environment is installed or changed,
no repository/Git operation is performed, and data/results stay in the external
work directory. This is a controlled protocol, not certified R2 numeric recovery.
"""
import argparse
import csv
import importlib.metadata
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import math
import os
import secrets
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request
import urllib.error
from urllib.parse import urlsplit,urlunsplit
import uuid
import zipfile

REPO=Path(__file__).resolve().parents[1]
MODELS=('NAML','NRMS','LSTUR','NPA','CNE-SUE','DIGAT')
NNR_MODELS=MODELS[:-1]
PROTOCOLS=('imp','rs30','rs90','rs150','total')
MIND_PAGE='https://msnews.github.io/'
MIND_DOWNLOAD_CATALOG='https://github.com/recommenders-team/recommenders/blob/main/recommenders/datasets/mind.py'
MIND_LICENSE='https://github.com/msnews/MIND/blob/master/MSR%20License_Data.pdf'
SEMANTIC_REVISION='e8c3b32edf5434bc2275fc9bab85f82640a19130'
DOWNLOADS={
 'train':('MINDsmall_train.zip','https://huggingface.co/datasets/Recommenders/MIND/resolve/main/MINDsmall_train.zip'),
 'dev':('MINDsmall_dev.zip','https://huggingface.co/datasets/Recommenders/MIND/resolve/main/MINDsmall_dev.zip'),
 'graph':('wikidata-graph.zip','https://mind201910.blob.core.windows.net/knowledge-graph/wikidata-graph.zip'),
}


def utc():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def fp(path):
 path=Path(path).resolve();return {'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)}
def verify(pin):
 path=Path(pin['path'])
 if not path.is_file()or path.stat().st_size!=pin['bytes']or sha(path)!=pin['sha256']:
  raise ValueError('Pinned file changed: '+str(path))
def save(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 temporary=path.with_name(path.name+'.tmp.'+uuid.uuid4().hex)
 with temporary.open('x',encoding='utf-8',newline='\n')as stream:
  json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
 os.replace(temporary,path)
def initial_hash(manifest):
 values=[manifest[k]for k in ('initial_model_state_sha256','initial_model_sha256')if manifest.get(k)]
 if not values or len(set(values))!=1:raise ValueError('Initial model fingerprint is missing or inconsistent')
 return values[0]


def resolve_models(values):
 values=['NAML']if values is None else list(values)
 if values==['all5']:return list(NNR_MODELS)
 if values==['all6']:return list(MODELS)
 if not values or any(v not in MODELS for v in values)or len(set(values))!=len(values):
  raise ValueError('Choose distinct model names, all5, or all6')
 return [m for m in MODELS if m in values]


def publish_exclusive(temporary,destination):
 """Never replace an existing target; tolerate drives without hard links.

 Windows rename refuses replacement atomically. The final exclusive-copy fallback
 is non-atomic: a crash may leave an incomplete target, which is preserved and
 rejected on retry rather than silently adopted.
 """
 try:
  os.link(temporary,destination);temporary.unlink();return 'atomic_exclusive_hardlink'
 except FileExistsError:raise
 except OSError:pass
 if os.name=='nt':
  try:
   os.rename(temporary,destination);return 'atomic_windows_nonreplace_rename'
  except FileExistsError:raise
  except OSError:pass
 with destination.open('xb')as out,temporary.open('rb')as inp:
  shutil.copyfileobj(inp,out,8*1024*1024);out.flush();os.fsync(out.fileno())
 temporary.unlink();return 'exclusive_copy_nonatomic_preserve_incomplete_on_failure'


def public_url(url):
 """Do not persist signed redirect query strings or authentication details."""
 parsed=urlsplit(url)
 return urlunsplit((parsed.scheme,parsed.hostname or '',parsed.path,'',''))


def download_official(url,destination,*,expected_sha256=None,opener=None):
 """Fetch a site-linked mirror or upstream graph URL, without overwriting.

 Availability and access are external: a local licensed archive is the supported
 fallback. An observed SHA alone is not an authoritative expected checksum.
 """
 if not any(url==value[1]for value in DOWNLOADS.values()):raise ValueError('Only declared dataset/graph source URLs are supported')
 if expected_sha256 is not None and (len(expected_sha256)!=64 or any(c not in '0123456789abcdef'for c in expected_sha256)):
  raise ValueError('Expected SHA-256 must be 64 lowercase hexadecimal characters')
 destination=Path(destination).resolve()
 if destination.exists():raise FileExistsError('Preserve existing download: '+str(destination))
 destination.parent.mkdir(parents=True,exist_ok=True)
 temporary=destination.with_name(destination.name+'.partial-'+uuid.uuid4().hex)
 h=hashlib.sha256();size=0;started=utc();open_url=opener or urllib.request.urlopen
 request=urllib.request.Request(url,headers={'User-Agent':'TORS-reproducibility-pipeline/1'})
 # A local environment token is never stored, printed, or forwarded to redirects.
 token=os.environ.get('HF_TOKEN')
 if token and urlsplit(url).hostname=='huggingface.co':request.add_unredirected_header('Authorization','Bearer '+token)
 try:
  with open_url(request,timeout=120)as response,temporary.open('xb')as out:
   final_url=response.geturl()
   if not final_url.startswith('https://'):raise ValueError('Unexpected non-HTTPS archive redirect')
   for block in iter(lambda:response.read(8*1024*1024),b''):
    out.write(block);h.update(block);size+=len(block)
   out.flush();os.fsync(out.fileno())
 except urllib.error.HTTPError as error:
  raise RuntimeError('Dataset source returned HTTP '+str(error.code)+'. Obtain licensed local files and use --train-zip, --dev-zip and --wikidata-graph; no access restrictions are bypassed.')from None
 except urllib.error.URLError:
  raise RuntimeError('Dataset source is unavailable. Supply licensed local --train-zip, --dev-zip and --wikidata-graph files; no destination was committed.')from None
 checksum=h.hexdigest()
 if expected_sha256 is not None and checksum!=expected_sha256:raise ValueError('Archive checksum differs; partial file preserved, destination not committed')
 if not zipfile.is_zipfile(temporary):raise ValueError('Downloaded content is not a ZIP archive; partial preserved')
 mode=publish_exclusive(temporary,destination)
 receipt={'status':'complete','url':url,'resolved_url':public_url(final_url),'started_utc':started,'completed_utc':utc(),
  'file':{'path':str(destination),'bytes':size,'sha256':checksum},'expected_sha256':expected_sha256,
  'expected_checksum_verified':expected_sha256 is not None,'source_page':MIND_PAGE,'license':MIND_LICENSE,
  'source_kind':'maintained Microsoft Recommenders mirror'if 'huggingface.co/'in url else 'upstream-referenced graph host',
  'download_catalog_source':MIND_DOWNLOAD_CATALOG if 'huggingface.co/'in url else None,
  'publication_mode':mode,'archive_member_validation':'Raw-MIND preparation validates members; ZIP framing only checked here'}
 save(destination.with_name(destination.name+'.download.json'),receipt)
 return receipt


def environment_report(models):
 required=['psutil','numpy','torch','torchtext','torch_scatter','nltk','sklearn']
 if 'DIGAT'in models:required+=['sentence_transformers','transformers']
 modules={n:importlib.util.find_spec(n)is not None for n in required}
 versions={}
 for name in required:
  try:versions[name]=importlib.metadata.version({'sklearn':'scikit-learn','torch_scatter':'torch-scatter'}.get(name,name))
  except importlib.metadata.PackageNotFoundError:versions[name]=None
 resources={}
 if modules['nltk']:
  import nltk
  for name in ('tokenizers/punkt','tokenizers/punkt_tab'):
   try:nltk.data.find(name);resources[name]=True
   except LookupError:resources[name]=False
 return {'python':sys.version,'executable':sys.executable,'module_available':modules,'package_versions':versions,
  'nltk_resources':resources,'default_NNR_tokenizer':'MIND regex; Punkt data is optional for this default',
  'GPU_test_performed':False,'packages_installed_or_changed':False}


def process_identity(pid=None):
 try:import psutil
 except ImportError as error:raise RuntimeError('Install psutil in the chosen environment; the pipeline never installs packages')from error
 p=psutil.Process(pid or os.getpid());return {'pid':p.pid,'create_time':p.create_time(),'command':p.cmdline()}
def live(identity):
 if not identity:return False
 try:return process_identity(identity['pid'])==identity
 except Exception:return False


SEED_LIMIT=2**32
TABLE3_SEED_TAG='tors/table3-request-resampling/v1'


def parse_seed(value):
 """Accept an unsigned 32-bit replay seed or defer fresh entropy until sealing."""
 if isinstance(value,str)and value.lower()=='random':return 'random'
 if type(value)is int:seed=value
 elif isinstance(value,str):
  try:seed=int(value)
  except ValueError as error:raise argparse.ArgumentTypeError('Seed must be random or an unsigned 32-bit integer')from error
 else:raise argparse.ArgumentTypeError('Seed must be random or an unsigned 32-bit integer')
 if not 0<=seed<SEED_LIMIT:raise argparse.ArgumentTypeError('Seed must be random or an unsigned 32-bit integer')
 return seed


def derived_analysis_seed(study_seed):
 """Use a distinct, reproducible substream for Table 3 request resampling."""
 if type(study_seed)is not int or not 0<=study_seed<SEED_LIMIT:raise ValueError('Study seed must be unsigned 32-bit')
 digest=hashlib.sha256((TABLE3_SEED_TAG+':'+str(study_seed)).encode('ascii')).digest()
 seed=int.from_bytes(digest[:4],'big')
 if seed==study_seed:seed^=0xffffffff
 return {'seed':seed,'parent_study_seed':study_seed,'derivation_tag':TABLE3_SEED_TAG,
  'derivation':'SHA256(tag + colon + decimal study seed), first four bytes big-endian; XOR ffffffff if equal to parent',
  'scope':'Table 3 shared request resampling only; not a training replicate'}


def verify_frozen_work_location(work,stored,plan):
 """Allow only the original sealed runtime to live below its own work root.

 A new source checkout must remain disjoint from data/results. Its later frozen
 snapshot is intentionally WORK/runtime, so a same-plan resume needs this narrow
 location check in addition to the unchanged source and input guards.
 """
 expected=work/'runtime';controller=expected/'scripts/run_mind_pipeline.py'
 message='Keep work/data/results outside the source repository; nested execution requires its original sealed runtime'
 if stored is None or not isinstance(plan,dict)or REPO!=expected:raise ValueError(message)
 if Path(plan.get('runtime','')).absolute()!=expected or Path(__file__).absolute()!=controller:raise ValueError(message)
 if plan.get('options')!=stored or Path(stored.get('work_dir','')).resolve()!=work:raise ValueError('Frozen runtime does not belong to the declared work directory')
 planpath=work/'control/pipeline_plan.json'
 if planpath.is_symlink()or planpath.parent.is_symlink()or not planpath.is_file()or read(planpath)!=plan:raise ValueError('Frozen runtime requires the original on-disk plan')
 pins=plan.get('source_files')
 if not isinstance(pins,list)or not pins:raise ValueError('Frozen runtime source inventory is absent')
 seen=set();controller_pin=None
 for item in pins:
  relative=Path(item['relative'])
  if relative.is_absolute()or '..'in relative.parts or str(relative)in seen:raise ValueError('Invalid frozen source path')
  seen.add(str(relative));source=expected/relative
  for component in (source,*source.parents):
   if component.is_symlink():raise ValueError('Frozen source paths must not be symbolic links')
   if component==expected:break
  if relative.as_posix()=='scripts/run_mind_pipeline.py':controller_pin=item
 if controller_pin is None or controller_pin.get('sha256')!=plan.get('controller_sha256'):raise ValueError('Frozen controller fingerprint is absent or inconsistent')
 statepath=work/'control/pipeline_state.json'
 if statepath.is_symlink():raise ValueError('Frozen runtime state must not be a symbolic link')
 if statepath.is_file()and read(statepath).get('plan_sha256')!=sha(planpath):raise ValueError('Frozen runtime state belongs to a different plan')
 verify_source(plan)


def normalize(args,stored=None,*,sealed_plan=None):
 work=args.work_dir.resolve()
 if work==REPO or REPO in work.parents:raise ValueError('Keep work/data/results outside the source repository')
 if work in REPO.parents:verify_frozen_work_location(work,stored,sealed_plan)
 defaults={'models':['NAML'],'seed':'random','dev_fraction':.05,'download':False,'accept_mind_license':False,
  'split_policy':'chronological','context_feature_mode':('required' if args.wikidata_graph is not None else 'context_not_used_by_paper_models'),'semantic_revision':SEMANTIC_REVISION,'download_nltk':False,'npa_features_on_gpu':False,'personalized_batch_size':512}
 names=list(defaults)+['train_zip','dev_zip','wikidata_graph','embedding_root','glove_archive','glove_text','sag_model','sag_model_source',
 'digat_assets','nnr_cache_dir','mind_train_sha256','mind_dev_sha256','wikidata_sha256']
 options={'work_dir':str(work)}
 for name in names:
  given=getattr(args,name,None)
  if name=='seed'and given is not None:given=parse_seed(given)
  if stored is not None:
   if name=='seed'and given=='random':given=None
   previous=stored.get(name)
   value=previous if given is None else given
   comparable=str(given.resolve())if isinstance(given,Path)else resolve_models(given)if name=='models'and given is not None else given
   if given is not None and comparable!=previous:raise ValueError('Resume cannot change declared option: '+name)
  else:value=defaults.get(name)if given is None else given
  options[name]=str(value.resolve())if isinstance(value,Path)else value
 options['models']=resolve_models(options['models'])
 for name,default in [('train_zip',work/'downloads'/DOWNLOADS['train'][0]),('dev_zip',work/'downloads'/DOWNLOADS['dev'][0]),
  ('wikidata_graph',work/'downloads'/DOWNLOADS['graph'][0]),('embedding_root',work/'assets/glove6b'),
  ('nnr_cache_dir',work/'assets/nnr_cache'),('sag_model',work/'assets/all-mpnet-base-v2'),('digat_assets',work/'assets/digat')]:
  if options[name]is None:options[name]=str(default)
 if options['download']and not options['accept_mind_license']:
  raise ValueError('Review the official dataset terms at '+MIND_LICENSE+' and pass --accept-mind-license with --download')
 if options['seed']!='random'and(type(options['seed'])is not int or not 0<=options['seed']<SEED_LIMIT):
  raise ValueError('Recorded seed must be unsigned 32-bit')
 if not 0<options['dev_fraction']<1:raise ValueError('Development fraction must be between zero and one')
 if options['split_policy']=='chronological' and options['dev_fraction']!=.05:
  raise ValueError('Chronological development uses the last official training day; keep --dev-fraction at its default')
 if options['context_feature_mode']=='context_not_used_by_paper_models':
  if args.wikidata_graph is not None or options['wikidata_sha256'] is not None:
   raise ValueError('The paper-model unused-context mode takes no graph or graph checksum')
  options['wikidata_graph']=None
 if options['glove_archive'] and options['glove_text']:raise ValueError('Choose --glove-archive or --glove-text, not both')
 return options


def source_inventory(root):
 result=[]
 for folder in ('scripts','NNR-main','third_party'):
  for path in sorted((root/folder).rglob('*'))if(root/folder).is_dir()else[]:
   if not path.is_file()or any(x in path.parts for x in ('__pycache__','.git','.venv')):continue
   if path.is_symlink()or path.suffix.lower()in('.pt','.pth','.pkl','.npy','.npz','.vec','.zip','.tsv')or path.stat().st_size>16*1024*1024:
    raise ValueError('Use a clean source release; data/model artifacts are not frozen as source: '+str(path))
   result.append((path.relative_to(root).as_posix(),fp(path)))
 return result


def create_plan(options):
 work=Path(options['work_dir']);control=work/'control';runtime=work/'runtime';planpath=control/'pipeline_plan.json'
 if planpath.exists():raise FileExistsError('A pipeline plan already exists; use --resume or --check-only')
 if runtime.exists():raise FileExistsError('Preserve an existing runtime; use another work directory')
 needed=['prepare_mind_masters.py','prepare_consistent_candidates.py','prepare_model_assets.py','prepare_nnr_embedding_cache.py','context_embeddings.py','context_feature_policy.py','candidate_policy.py','consistent_candidate_profile.py','run_converged_nnr.py','summarize_converged_experiments.py']
 if 'DIGAT'in options['models']:needed+=['run_converged_digat.py','prepare_digat_assets.py']
 for name in needed:
  if not(REPO/'scripts'/name).is_file():raise ValueError('Incomplete source package: scripts/'+name)
 inventory=source_inventory(REPO);staging=work/('runtime.partial-'+uuid.uuid4().hex);staging.mkdir(parents=True)
 source=[]
 for relative,pin in inventory:
  destination=staging/relative;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(pin['path'],destination)
  if sha(destination)!=pin['sha256']:raise ValueError('Source changed during snapshot')
  source.append({'relative':relative,'bytes':pin['bytes'],'sha256':pin['sha256']})
 if any(sha(REPO/relative)!=pin['sha256']for relative,pin in inventory):raise ValueError('Source changed while freezing')
 staging.rename(runtime)
 # A fresh study draws once, only when its source snapshot is ready to seal.
 options=dict(options);requested=options['seed']
 options['seed']=secrets.randbits(32)if requested=='random'else requested
 seed_selection={'policy':'random_new_study'if requested=='random'else'explicit_reproduction',
  'generator':'secrets.randbits(32)'if requested=='random'else'explicit CLI integer',
  'seed':options['seed'],'scope':'One shared seed for paired models, training conditions, splits and candidates; resumes reuse it'}
 plan={'schema':1,'created_utc':utc(),'options':options,'runtime':str(runtime),'source_files':source,
  'seed_selection':seed_selection,'analysis_seeds':{'table3_request_resampling':derived_analysis_seed(options['seed'])},
  'controller_sha256':sha(Path(__file__)),'protocol':'complete_proxy_strict_request_consistent_v1',
  'training_settings':{'min_epochs':5,'max_epochs':50,'patience':5,'min_delta':.0001,'selection_metric':'avg'},
  'cooling':{'training_interval_seconds':600,'training_rest_seconds':120,'long_interval_seconds':10800,'long_rest_seconds':900},
  'scope':'Fresh paired IMP/RandS training and all five evaluations; not exact historical R2 numbers'}
 save(planpath,plan);return plan


def verify_source(plan):
 if sha(Path(__file__))!=plan['controller_sha256']:
  raise ValueError('Pipeline control source changed. Resume with the frozen runtime/scripts/run_mind_pipeline.py')
 root=Path(plan['runtime'])
 for item in plan['source_files']:verify({'path':str(root/item['relative']),'bytes':item['bytes'],'sha256':item['sha256']})


def job_commands(plan):
 o=plan['options'];work=Path(o['work_dir']);runtime=Path(plan['runtime']);data=work/'dataset'
 total=data/'provenance/full_total_manifest.json';jobs=[];first={}
 for model in o['models']:
  for training in ('imp','rs50'):
   run=work/'runs'/(model+'_'+training+'_seed'+str(o['seed']));script='run_converged_digat.py'if model=='DIGAT'else'run_converged_nnr.py'
   cmd=[sys.executable,'-B','-u',str(runtime/'scripts'/script),'--data-root',str(data),
    '--embedding-root',o['embedding_root']if model=='DIGAT'else o['nnr_cache_dir'],'--run-dir',str(run),'--train-type',training,'--seed',str(o['seed']),
    '--max-epochs','50','--min-epochs','5','--patience','5','--min-delta','0.0001','--dev-protocol',training,
    '--selection-metric','avg','--candidate-manifest',str(total),'--chunk-rows','5000','--cooling']
   if model=='DIGAT':cmd+=['--shared-preprocessing',o['digat_assets'],'--full-total',str(data/'test/behaviors_total.tsv')]
   else:
    cmd+=['--model',model,'--full-total-behaviors',str(data/'test/behaviors_total.tsv')]
    if training in first:cmd+=['--preprocessing-from',first[training]]
    else:first[training]=str(run)
    if model=='NPA':
     cmd+=['--personalized-batch-size',str(o['personalized_batch_size'])]
     if o['npa_features_on_gpu']:cmd.append('--npa-features-on-gpu')
   if training=='rs50':cmd+=['--paired-preprocessing-from',str(work/'runs'/(model+'_imp_seed'+str(o['seed'])))]
   jobs.append({'name':model+'_'+training,'model':model,'train_type':training,'seed':o['seed'],'run_dir':str(run),
    'manifest':str(run/'experiment.json'),'command':cmd,'cwd':str(runtime)})
 return jobs


def phase_commands(plan,raw_inputs=None):
 o=plan['options'];work=Path(o['work_dir']);scripts=Path(plan['runtime'])/'scripts';py=[sys.executable,'-B','-u']
 raw=py+[str(scripts/'prepare_mind_masters.py'),'--train-zip',o['train_zip'],'--dev-zip',o['dev_zip'],
  '--output',str(work/'masters'),'--seed',str(o['seed']),'--dev-fraction',str(o['dev_fraction']),
  '--split-policy',o['split_policy'],'--context-feature-mode',o['context_feature_mode'],'--selected-models',*o['models']]
 if o['context_feature_mode']=='required':raw+=['--wikidata-graph',o['wikidata_graph']]
 result={'raw':raw,'candidates':py+[str(scripts/'prepare_consistent_candidates.py'),'--source-data-root',str(work/'masters'),
  '--output',str(work/'dataset'),'--seed',str(o['seed'])]}
 if raw_inputs:
  result['raw'] += ['--expected-train-sha256',raw_inputs['train_zip']['sha256'],'--expected-dev-sha256',raw_inputs['dev_zip']['sha256']]
 return result


def complete_job(job,experiment_plan):
 m=read(job['manifest'])
 if m.get('status')!='completed'or(m.get('model'),m.get('train_type'),m.get('seed'))!=(job['model'],job['train_type'],job['seed']):
  raise ValueError('Model run is not complete with its declared identity')
 if m.get('training_reused'):raise ValueError('This portable pipeline requires fresh training')
 selected=m.get('selection',{})
 expected_dev=experiment_plan['dev_inputs'][job['train_type']]
 if selected.get('protocol')!=job['train_type']or selected.get('file',{}).get('sha256')!=expected_dev['sha256']:
  raise ValueError('Checkpoint selection used another development protocol')
 if selected.get('matched_candidate_profile')!=experiment_plan['candidate_profile']:raise ValueError('Candidate certificate differs')
 stop=m.get('convergence',m.get('early_stopping',{}))
 if stop.get('stop_reason')not in('patience','max_epochs')or m['best_epoch']!=stop['best_epoch']:
  raise ValueError('Development-best or completed stopping decision is absent')
 reports={}
 for protocol in PROTOCOLS:
  entry=m['evaluations'][protocol];path=Path(entry.get('files',{}).get('metrics.json')or(str(entry['prefix'])+'.metrics.json'));report=read(path)
  for artifact in m.get('evaluation_artifacts',{}).get(protocol,{}).values():verify(artifact)
  spec=experiment_plan['test_inputs'][protocol]
  if report['input']['sha256']!=spec['file']['sha256']or any(report[k]!=spec['statistics'][k]for k in('rows','candidates')):
   raise ValueError('Incomplete/wrong '+protocol+' test coverage')
  for key in ('AUC','MRR','nDCG@5','nDCG@10'):
   a,b=float(entry['metrics'][key]),float(report['metrics'][key])
   if not math.isfinite(a)or not math.isfinite(b)or abs(a-b)>1e-12:raise ValueError('Invalid or inconsistent logged metrics')
  reports[protocol]=fp(path)
 if 'timings_seconds_net'not in m:raise ValueError('Measured rest-adjusted timing is absent')
 if not m.get('checkpoint'):raise ValueError('Selected checkpoint fingerprint is missing')
 verify(m['checkpoint'])
 return {'manifest':fp(job['manifest']),'reports':reports,'initial_model_sha256':initial_hash(m),
  'best_epoch':m['best_epoch'],'epochs_completed':m['epochs_completed'],'timings_seconds_net':m['timings_seconds_net']}


def write_selected_results(output,jobs,proofs,plan=None):
 """Export native metrics and audited measured-net time; no fabricated Table 3."""
 output=Path(output);output.mkdir(parents=True,exist_ok=True);metrics=[];timings=[];configurations=[]
 for job in jobs:
  m=read(job['manifest']);net=proofs[job['name']]['timings_seconds_net']
  if (m.get('model'),m.get('train_type'),m.get('seed'))!=(job['model'],job['train_type'],job['seed']):
   raise ValueError('Result identity differs from its resolved run configuration')
  effective=m.get('effective_config')
  if plan is not None and(not isinstance(effective,dict)or not effective):raise ValueError('Native effective configuration is absent')
  if isinstance(effective,dict)and 'seed'in effective and effective['seed']!=job['seed']:raise ValueError('Native effective configuration seed differs from result seed')
  configuration_id=job['name']+'_seed'+str(job['seed'])
  configurations.append({'configuration_id':configuration_id,'model':job['model'],'training':job['train_type'],
   'development':job['train_type'],'seed':job['seed'],'effective_config':effective,
   'effective_config_status':'recorded_native'if isinstance(effective,dict)and effective else'unavailable_legacy',
   'effective_config_sha256':m.get('effective_config_sha256'),
   'experiment_manifest':fp(job['manifest']),'selected_checkpoint':m.get('checkpoint'),
   'best_epoch':m['best_epoch'],'epochs_completed':m['epochs_completed'],'command':job.get('command')})
  for protocol in PROTOCOLS:
   metrics.append(dict(configuration_id=configuration_id,configuration_file='config.json',model=job['model'],training=job['train_type'],development=job['train_type'],evaluation=protocol,seed=job['seed'],
    best_epoch=m['best_epoch'],epochs_completed=m['epochs_completed'],**m['evaluations'][protocol]['metrics']))
  times={key:float(net[key])for key in ('training','dev_evaluation','total','cooling_rest')}
  tests={p:float(net['tests'][p])for p in PROTOCOLS}
  if any(not math.isfinite(v)or v<0 for v in list(times.values())+list(tests.values())):raise ValueError('Invalid measured-net timer')
  timings.append(dict(configuration_id=configuration_id,configuration_file='config.json',model=job['model'],training=job['train_type'],seed=job['seed'],training_seconds=times['training'],
   validation_seconds=times['dev_evaluation'],**{p+'_test_seconds':v for p,v in tests.items()},all_test_seconds=sum(tests.values()),
   active_runner_seconds=times['total'],measured_rest_seconds=times['cooling_rest'],training_seed_SD='NA'))
 seeds=sorted({j['seed']for j in jobs})
 if plan is not None and seeds!=[plan['options']['seed']]:raise ValueError('Result seeds differ from the sealed pipeline configuration')
 planpath=Path(plan['options']['work_dir'])/'control/pipeline_plan.json'if plan is not None else None
 if planpath is not None and not planpath.is_file():raise ValueError('Sealed pipeline plan is absent for result configuration')
 result_config={'schema':1,'training_seeds':seeds,'candidate_seed':plan['options']['seed']if plan is not None else None,
  'seed_selection':plan.get('seed_selection')if plan is not None else None,
  'analysis_seeds':plan.get('analysis_seeds')if plan is not None else None,
  'resolved_options':plan['options']if plan is not None else None,
  'training_settings':plan.get('training_settings')if plan is not None else None,'cooling':plan.get('cooling')if plan is not None else None,
  'pipeline_plan':fp(planpath)if planpath is not None else None,'runs':configurations}
 configpath=output/'config.json'
 if configpath.is_file():
  if read(configpath)!=result_config:raise ValueError('Preserve existing results: resolved configuration changed')
 else:save(configpath,result_config)
 for name,rows in [('metrics.csv',metrics),('timings_net.csv',timings)]:
  temporary=output/(name+'.tmp-'+uuid.uuid4().hex)
  with temporary.open('x',encoding='utf-8',newline='')as out:
   writer=csv.DictWriter(out,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
  os.replace(temporary,output/name)
 lines=['# New measured results','','Native runner metrics, not paper-reported values or exact historical reproduction.',
  'Declared training seed(s): '+', '.join(map(str,sorted({j['seed']for j in jobs})))+'. Training-seed standard deviation is NA for a single seed.',
  'Training IMP uses IMP development; training RandS50 uses RandS50 development.',
  'config.json saves the resolved settings and actual seed for every result run, linked by configuration_id in both CSV files.',
  'metrics.csv records five evaluations per selected training condition.',
  'timings_net.csv excludes measured rests and inter-process gaps; failed recomputation remains active work.',
  'Training and validation sum committed epochs; incomplete-epoch work is represented only in active_runner_seconds.',
  'Shared input/context/candidate/asset preparation is outside model timings (control/pipeline_state.json attempts).',
  'Table 3 requires all six models; it is not computed for a NAML-only or all5 run.','']
 (output/'README.md').write_text('\n'.join(lines),encoding='utf-8')
 return {p.name:fp(p)for p in sorted(output.iterdir())if p.is_file()}



def table_summary_plan(plan,jobs):
 """Bind table postprocessing to the sealed study and a separate analysis RNG."""
 o=plan['options'];work=Path(o['work_dir'])
 seed=o['seed']
 if type(seed)is not int or not 0<=seed<SEED_LIMIT:raise ValueError('Seal the study seed before constructing table jobs')
 analysis=plan.get('analysis_seeds',{'table3_request_resampling':{'seed':42,'derivation':'legacy fixed analysis seed'}})
 return {'schema':1,'seeds':[seed],'candidate_seed':seed,'analysis_seeds':analysis,
  'data_root':str(work/'dataset'),'candidate_manifest':str(work/'dataset/provenance/full_total_manifest.json'),
  'jobs':[dict(model=j['model'],train_type=j['train_type'],seed=j['seed'],success_file=j['manifest'])for j in jobs],
  'result_configuration':fp(work/'results/config.json')if(work/'results/config.json').is_file()else None}


class Pipeline:
 def __init__(self,plan,resume=False):
  self.plan=plan;self.options=plan['options'];self.work=Path(self.options['work_dir']);self.control=self.work/'control'
  self.statepath=self.control/'pipeline_state.json';self.lock=self.control/'pipeline.lock.json';self.resume=resume
  self.state=read(self.statepath)if self.statepath.is_file()else{'schema':1,'phases':{},'jobs':{},'created_utc':utc()}
  plan_sha=sha(self.control/'pipeline_plan.json')
  if self.state.get('plan_sha256',plan_sha)!=plan_sha:raise ValueError('Pipeline state belongs to another plan')
  self.state['plan_sha256']=plan_sha
 def record(self):self.state['updated_utc']=utc();save(self.statepath,self.state)
 def command(self,cmd,name,progress=False):
  logs=self.work/'logs';logs.mkdir(exist_ok=True);path=logs/(name+'_'+uuid.uuid4().hex+'.log')
  env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1')
  started=time.monotonic();attempt={'command':cmd,'started_utc':utc(),'log':str(path)};self.state['current']=name;self.state['current_attempt']=attempt;self.record()
  print('Running '+name+'; detailed log: '+str(path),flush=True)
  from contextlib import nullcontext
  if progress:
   sys.path.insert(0,str(Path(self.plan['runtime'])/'scripts'))
   from experiment_progress import batch_progress
   context=batch_progress(env,name)
  else:context=nullcontext()
  with path.open('x',encoding='utf-8',newline='\n')as stream,context:
   child=subprocess.Popen(cmd,cwd=str(self.plan['runtime']),env=env,stdout=stream,stderr=subprocess.STDOUT)
   attempt['worker']=process_identity(child.pid);attempt['progress_file']=env.get('TORS_EXPERIMENT_PROGRESS_FILE');self.record()
   code=child.wait();attempt['returncode']=code;attempt['ended_utc']=utc();attempt['elapsed_seconds']=time.monotonic()-started
   self.state.setdefault('attempts',[]).append(dict(attempt));self.record()
  if code:raise RuntimeError(name+' exited '+str(code)+'; preserve artifacts and inspect '+str(path))
  return attempt
 def phase(self,name,cmd,manifest):
  prior=self.state['phases'].get(name)
  if prior:
   verify(prior['manifest']);return
  if Path(manifest).exists():raise ValueError('Unrecorded prepared output exists; preserve it and use a new work directory or review its receipts')
  self.command(cmd,name)
  value=read(manifest)
  if value.get('status')!='complete':raise ValueError('Preparation phase did not commit a complete output')
  self.state['phases'][name]={'manifest':fp(manifest)};self.record()
 def inputs(self):
  o=self.options
  declared=[('train_zip','train','mind_train_sha256'),('dev_zip','dev','mind_dev_sha256')]
  if o['context_feature_mode']=='required':declared.append(('wikidata_graph','graph','wikidata_sha256'))
  for name,key,expected in declared:
   path=Path(o[name])
   if not path.is_file():
    if not o['download']:raise FileNotFoundError('Supply local archives or explicitly --download --accept-mind-license: '+str(path))
    default=self.work/'downloads'/DOWNLOADS[key][0]
    if path!=default:raise ValueError('Downloads never write a user-supplied input location; omit its archive flag to use work/downloads')
    download_official(DOWNLOADS[key][1],path,expected_sha256=o[expected])
   actual=fp(path)
   if o[expected]is not None and actual['sha256']!=o[expected]:raise ValueError('Local input checksum differs: '+name)
   previous=self.state.setdefault('raw_inputs',{}).get(name)
   if previous:verify(previous)
   else:self.state['raw_inputs'][name]=actual;self.record()
 def assets(self,semantic=False):
  o=self.options;scripts=Path(self.plan['runtime'])/'scripts';py=[sys.executable,'-B','-u'];glove=Path(o['embedding_root'])/'glove.6B.300d.txt'
  cmd=py+[str(scripts/'prepare_model_assets.py'),'glove','--output',o['embedding_root']]
  registered=(Path(o['embedding_root'])/'GLOVE_ASSET_MANIFEST.json').is_file()
  if registered:cmd.append('--check-only')
  elif o['glove_archive']:cmd+=['--archive',o['glove_archive']]
  elif o['glove_text']:cmd+=['--text',o['glove_text']]
  elif o['download']:cmd.append('--download')
  else:raise FileNotFoundError('Use registered --embedding-root, --glove-text LOCALTXT or --glove-archive ZIP. Unregistered/partial outputs are preserved; choose a new asset output directory.')
  self.command(cmd,'glove_assets')
  pin=fp(glove);previous=self.state.setdefault('asset_inputs',{}).get('glove')
  if previous:verify(previous)
  else:self.state['asset_inputs']['glove']=pin;self.record()
  if semantic:
   cmd=py+[str(scripts/'prepare_model_assets.py'),'semantic','--output',o['sag_model'],'--revision',o['semantic_revision']]
   if (Path(o['sag_model'])/'TORS_ASSET_MANIFEST.json').is_file():cmd.append('--check-only')
   elif o['sag_model_source']:cmd+=['--source',o['sag_model_source']]
   elif o['download']:cmd.append('--download')
   else:raise FileNotFoundError('DIGAT needs a registered local semantic model, --sag-model-source or explicit download. Preserve partial outputs and choose a new output directory.')
   self.command(cmd,'semantic_model_assets')
  if o['download_nltk']:
   self.command(py+['-m','nltk.downloader','punkt','punkt_tab'],'explicit_nltk_resources')
 def seal_experiments(self):
  path=self.control/'experiment_plan.json'
  if path.is_file():
   plan=read(path)
   if plan.get('pipeline_plan_sha256')!=sha(self.control/'pipeline_plan.json'):raise ValueError('Experiment plan belongs to another pipeline')
   if plan['jobs']!=job_commands(self.plan):raise ValueError('Experiment commands differ from the frozen plan')
   for pin in plan['input_files']:verify(pin)
   return plan
  sys.path.insert(0,str(Path(self.plan['runtime'])/'scripts'))
  from consistent_candidate_profile import validate_consistent_candidate_profile,evaluation_paths
  data=self.work/'dataset';profile=validate_consistent_candidate_profile(data);manifest=read(data/'manifest.json');paths=evaluation_paths(data,profile)
  plan={'schema':1,'pipeline_plan_sha256':sha(self.control/'pipeline_plan.json'),'candidate_profile':profile,
   'derived_asset_policy':'lazy_DIGAT_separate_immutable_seal_v1',
   'jobs':job_commands(self.plan),'dev_inputs':{t:fp(data/'dev'/('behaviors.tsv'if t=='imp'else'behaviors_rs50.tsv'))for t in('imp','rs50')},
   'test_inputs':{p:{'file':fp(paths[p]),'statistics':manifest['statistics'][str(paths[p].relative_to(data)).replace('\\','/')]}for p in PROTOCOLS},
   'input_files':[fp(p)for p in sorted(data.rglob('*'))if p.is_file()]+[fp(Path(self.options['embedding_root'])/'glove.6B.300d.txt')]}
  if 'DIGAT'in self.options['models']:
   for folder in [Path(self.options['sag_model'])]:
    plan['input_files'] += [fp(p)for p in sorted(folder.rglob('*'))if p.is_file()]
  if any(m in NNR_MODELS for m in self.options['models']):
   plan['input_files'] += [fp(p)for p in sorted(Path(self.options['nnr_cache_dir']).iterdir())if p.is_file()]
  save(path,plan);return plan
 def ensure_digat_assets(self,experiments):
  """Build SAG lazily, then seal derived files separately from the used main plan."""
  seal_path=self.control/'digat_asset_seal.json'
  expected={'schema':1,'pipeline_plan_sha256':sha(self.control/'pipeline_plan.json'),
   'experiment_plan_sha256':sha(self.control/'experiment_plan.json'),
   'policy':'lazy_DIGAT_separate_immutable_seal_v1'}
  if seal_path.is_file():
   sealed=read(seal_path)
   if any(sealed.get(key)!=value for key,value in expected.items()):raise ValueError('DIGAT derived-asset seal belongs to another immutable plan')
   verify(sealed['manifest'])
   for pin in sealed['files'].values():verify(pin)
   return sealed
  # The shared graph cannot silently adopt changed pretrained weights after the
  # earlier NNR jobs. The main experiment plan already pins these inputs.
  semantic=Path(self.options['sag_model']).resolve()
  for pin in experiments['input_files']:
   path=Path(pin['path']).resolve()
   if path==semantic or semantic in path.parents:verify(pin)
  scripts=Path(self.plan['runtime'])/'scripts'
  cmd=[sys.executable,'-B','-u',str(scripts/'prepare_digat_assets.py'),
   '--data-root',str(self.work/'dataset'),'--embedding-root',self.options['embedding_root'],
   '--sag-model',self.options['sag_model'],'--output',self.options['digat_assets'],
   '--seed',str(self.options['seed']),'--cooling']
  manifest=Path(self.options['digat_assets'])/'asset_prep.json'
  if manifest.is_file():cmd.append('--resume')
  self.command(cmd,'digat_graph_assets',progress=True)
  sys.path.insert(0,str(scripts))
  from run_converged_digat import validate_assets
  validated=validate_assets(Path(self.options['digat_assets']),self.work/'dataset',Path(self.options['embedding_root']),self.options['seed'])
  sealed={**expected,'manifest':fp(manifest),'asset_identity':validated['identity'],
   'files':{name:fp(Path(self.options['digat_assets'])/name)for name in validated['files']},
   'cooling':'Long scheduled rests at completed semantic/category boundaries; not optimizer training'}
  # This file is a new immutable derived-artifact seal. Never rewrite the main
  # experiment plan after its first NNR job has used it.
  if seal_path.exists():raise FileExistsError('Preserve an existing DIGAT asset seal')
  save(seal_path,sealed)
  return sealed
 def run_jobs(self,experiments):
  """Run NNR first; only the first unfinished DIGAT job prepares shared SAG."""
  for job in experiments['jobs']:
   manifest=Path(job['manifest'])
   if manifest.is_file()and read(manifest).get('status')=='completed':
    proof=complete_job(job,experiments);self.state['jobs'][job['name']]={'status':'completed','proof':proof};self.record();continue
   if job['model']=='DIGAT':self.ensure_digat_assets(experiments)
   cmd=list(job['command'])
   if Path(job['run_dir']).exists():
    if not self.resume or not manifest.is_file():raise ValueError('A partial model run needs same-plan --resume and its manifest')
    checkpoint=Path(job['run_dir'])/('latest_training_state.pt'if job['model']=='DIGAT'else'last_checkpoint.pt')
    if not checkpoint.is_file():raise ValueError('No committed optimizer/RNG/early-stop checkpoint; preserve this attempt')
    cmd.append('--resume')
   self.command(cmd,job['name'],progress=True)
   proof=complete_job(job,experiments)
   if job['train_type']=='rs50':
    paired=read(self.work/'runs'/(job['model']+'_imp_seed'+str(job['seed']))/'experiment.json')
    if initial_hash(paired)!=proof['initial_model_sha256']:raise ValueError('IMP/RandS paired initialization differs')
   self.state['jobs'][job['name']]={'status':'completed','proof':proof};self.record()
 def execute(self,prepare_only=False):
  verify_source(self.plan)
  if self.lock.exists():
   if live(read(self.lock).get('controller')):raise ValueError('A pipeline controller is already running')
   self.lock.rename(self.lock.with_name('pipeline.stale.'+uuid.uuid4().hex+'.json'))
  import psutil
  for p in psutil.process_iter(['cmdline']):
   cmd=p.info.get('cmdline')or[]
   try:
    if '--run-dir'in cmd and self.work/'runs'in Path(cmd[cmd.index('--run-dir')+1]).resolve().parents:
     raise ValueError('A worker from this work directory is still alive; do not duplicate it')
   except (IndexError,psutil.Error):continue
  with self.lock.open('x',encoding='utf-8')as out:json.dump({'controller':process_identity()},out)
  self.state['status']='preparing';self.record()
  try:
   self.inputs();self.assets(semantic='DIGAT'in self.options['models'])
   for pin in self.state['raw_inputs'].values():verify(pin)
   phases=phase_commands(self.plan,self.state['raw_inputs']);self.phase('raw_masters',phases['raw'],self.work/'masters/manifest.json')
   raw=read(self.work/'masters/manifest.json')
   raw_bindings=[('train_zip',raw['source_archives']['official_train']),('dev_zip',raw['source_archives']['official_dev'])]
   if self.options['context_feature_mode']=='required':raw_bindings.append(('wikidata_graph',raw['context']['source_graph']))
   if raw.get('context_feature_mode','required')!=self.options['context_feature_mode']:raise ValueError('Raw master context mode differs from the frozen plan')
   for option,actual in raw_bindings:
    verify(self.state['raw_inputs'][option])
    if any(actual[k]!=self.state['raw_inputs'][option][k]for k in ('bytes','sha256')):raise ValueError('Raw preparation used a different pinned input: '+option)
   for pin in raw['output_files'].values():verify(pin)
   for entry in raw['assets'].values():verify(entry['output'])
   self.phase('consistent_candidates',phases['candidates'],self.work/'dataset/manifest.json')
   if prepare_only:
    self.state['status']='prepared';self.state['GPU_test_performed']=False;self.record();return
   environment=environment_report(self.options['models'])
   missing=[n for n,available in environment['module_available'].items()if not available]
   if missing:raise RuntimeError('Install required packages manually in the active environment: '+', '.join(missing))
   self.state['environment']=environment;self.record()
   if any(m in NNR_MODELS for m in self.options['models']):
    self.command([sys.executable,'-B','-u',str(Path(self.plan['runtime'])/'scripts/prepare_nnr_embedding_cache.py'),
     '--source',str(Path(self.options['embedding_root'])/'glove.6B.300d.txt'),'--output',self.options['nnr_cache_dir']],'private_nnr_cache')
   experiments=self.seal_experiments();self.state['status']='running';self.record()
   self.run_jobs(experiments)
   result_proofs={name:value['proof']for name,value in self.state['jobs'].items()}
   self.state['results']=write_selected_results(self.work/'results',experiments['jobs'],result_proofs,plan=self.plan)
   if self.options['models']==list(MODELS):
    summary_plan=table_summary_plan(self.plan,experiments['jobs'])
    summary_path=self.control/'table_plan.json'
    if summary_path.is_file()and read(summary_path)!=summary_plan:raise ValueError('Table plan differs')
    if not summary_path.is_file():save(summary_path,summary_plan)
    tables=self.work/'results/tables';validation=tables/'validation.json'
    if validation.is_file():
     proof=read(validation)
     if proof.get('plan_sha256')!=sha(summary_path)or proof.get('training_runs')!=12 or proof.get('evaluations')!=60:raise ValueError('Existing table summary does not bind this plan')
     for name,checksum in proof['output_sha256'].items():
      if sha(tables/name)!=checksum:raise ValueError('Table summary output changed')
    else:
     if tables.exists():
      # Preserve a partial summary before retrying CPU-only postprocessing.
      if tables.resolve()!=self.work/'results/tables':raise ValueError('Unexpected table output path')
      tables.rename(tables.with_name('tables_incomplete_'+uuid.uuid4().hex))
     self.command([sys.executable,'-B','-u',str(Path(self.plan['runtime'])/'scripts/summarize_converged_experiments.py'),'--plan',str(summary_path),'--output',str(tables),
      '--resample-seed',str(summary_plan['analysis_seeds']['table3_request_resampling']['seed'])],'tables_2_3_5')
    self.state['tables']=fp(validation)
   self.state.update(status='completed',current=None,logical_training_jobs=len(experiments['jobs']),logical_test_evaluations=5*len(experiments['jobs']))
   self.record();save(self.control/'completion.json',{'status':'complete','experiment_plan':fp(self.control/'experiment_plan.json'),'jobs':self.state['jobs'],
    'claim':'Completed native runner artifacts; not a claim of recovered historical R2 numbers'})
  except BaseException as error:self.state.update(status='failed',error=repr(error));self.record();raise
  finally:
   if self.lock.is_file()and read(self.lock).get('controller')==process_identity():self.lock.unlink()


def parser():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--work-dir',type=Path,required=True)
 mode=p.add_mutually_exclusive_group(required=True)
 for name in ('execute','prepare-only','dry-run','check-only','status'):mode.add_argument('--'+name,action='store_true')
 p.add_argument('--resume',action='store_true');p.add_argument('--models',nargs='+')
 p.add_argument('--seed',type=parse_seed,help='random (default): draw and record once for a new study; an integer replays a declared seed; resume always reuses its recorded seed');p.add_argument('--dev-fraction',type=float)
 p.add_argument('--split-policy',choices=('random_groups','chronological'))
 p.add_argument('--context-feature-mode',choices=('required','context_not_used_by_paper_models'))
 for name in ('train-zip','dev-zip','wikidata-graph','embedding-root','glove-archive','sag-model','sag-model-source','digat-assets','glove-text','nnr-cache-dir'):p.add_argument('--'+name,type=Path)
 p.add_argument('--download-nltk',action='store_true',default=None);p.add_argument('--semantic-revision');p.add_argument('--download',action='store_true',default=None);p.add_argument('--accept-mind-license',action='store_true',default=None)
 for name in ('mind-train-sha256','mind-dev-sha256','wikidata-sha256'):p.add_argument('--'+name)
 p.add_argument('--npa-features-on-gpu',action='store_true',default=None);p.add_argument('--personalized-batch-size',type=int)
 return p


def main():
 args=parser().parse_args();planpath=args.work_dir.resolve()/'control/pipeline_plan.json'
 if args.status:
  state=read(args.work_dir.resolve()/'control/pipeline_state.json');print(json.dumps(state,indent=2));return
 old=read(planpath)if planpath.is_file()else None
 options=normalize(args,old['options']if old else None,sealed_plan=old)
 preview={'options':options,'planned_jobs':[(m,t)for m in options['models']for t in('imp','rs50')],
  'stages':['explicit downloads/local files','model asset files','official raw masters/context','complete proxy and strict candidates','optional DIGAT SAG','fresh IMP then paired RandS with all five tests'],
  'GPU_or_model_test_performed':False,'files_written':False}
 if args.dry_run:print(json.dumps(preview,indent=2));return
 if args.check_only:
  result={'environment':environment_report(options['models']),'plan_preview':preview}
  if old:
   verify_source(old)
   result['frozen_source_verified']=True
   data=args.work_dir.resolve()/'dataset'
   result['prepared_dataset_present']=(data/'consistent_candidate_manifest.json').is_file()
   if result['prepared_dataset_present']:
    sys.path.insert(0,str(Path(old['runtime'])/'scripts'))
    from consistent_candidate_profile import validate_consistent_candidate_profile
    result['prepared_dataset_verified']=validate_consistent_candidate_profile(data)
   statepath=args.work_dir.resolve()/'control/pipeline_state.json'
   if statepath.is_file():
    state=read(statepath)
    for group in ('raw_inputs','asset_inputs'):
     for pin in state.get(group,{}).values():verify(pin)
    result['recorded_input_pins_verified']=True
   result['native_model_check']='Not executed by this read-only preflight; no CUDA/model import' 
  print(json.dumps(result,indent=2));return
 if old and not args.resume:raise ValueError('Existing work requires --resume; completed work is never automatically repeated')
 plan=old or create_plan(options)
 Pipeline(plan,resume=args.resume).execute(prepare_only=args.prepare_only)

if __name__=='__main__':main()