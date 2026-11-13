"""New NNR experiment with training-matched dev selection and restartable training.

Uses the preserved loss, Adam, per-epoch negative sampling and model defaults.
Test protocols are evaluated only after development selects the best checkpoint.
This experiment does not claim the unknown historical R2 settings were recovered.
"""
import argparse
import contextlib
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
from nnr_artifacts import MODELS, check_inputs, fingerprint
from context_feature_policy import context_feature_spec
from model_batch_defaults import resolve_training_batch
from training_validation import resolve_dev_protocol, resolve_dev_path, validate_candidate_profile
from candidate_policy import PROFILE as CONSISTENT_PROFILE
from consistent_candidate_profile import evaluation_paths
from nnr_consistent_preprocessing import configure_reference, paired_dictionary_gate, verify_paired_initialization
from convergence_control import (EarlyStopping, selection_value, capture_rng_state,
    restore_rng_state, atomic_torch_save, save_json_atomic, file_sha256,
    prepare_epoch_cache, cleanup_epoch_cache)

PROTOCOL_FILES = {'imp':'behaviors.tsv','rs30':'behaviors_rs30.tsv',
                  'rs90':'behaviors_rs90.tsv','rs150':'behaviors_rs150.tsv'}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_file(path, expected):
    actual = fingerprint(Path(path))
    if actual['sha256'] != expected['sha256'] or ('bytes' in expected and actual['bytes'] != expected['bytes']):
        raise ValueError('Artifact changed: ' + str(path))
    return actual


def validate_total(args):
    spec = read_json(args.candidate_manifest)
    consistent = (getattr(args, 'matched_candidate_profile', None) or {}).get('profile') == CONSISTENT_PROFILE
    required = ('request_order_matches_existing_evaluation_mapping',
        'positive_labels_match_existing_evaluation_mapping','total5_subset_of_total',
        'rands30_subset_of_total','all_news_in_model_vocabularies')
    if consistent:
        if spec.get('candidate_profile') != CONSISTENT_PROFILE:
            raise ValueError('Full Total does not bind the new strict complete-proxy profile')
        required = ('same_GT_request_support_all_protocols','history_and_current_positives_excluded',
                    'strict_time_window','complete_proxy_coverage','all_news_in_model_vocabularies')
    if spec.get('status') != 'complete' or spec.get('protocol') != 'total':
        raise ValueError('Completed full Total manifest required')
    if Path(spec['dataset_root']).resolve() != args.data_root:
        raise ValueError('Full Total dataset root differs')
    if any(spec.get('validation',{}).get(key) is not True for key in required):
        raise ValueError('Full Total alignment/vocabulary validation is incomplete')
    if Path(spec['output']['path']).resolve() != args.full_total_behaviors:
        raise ValueError('Full Total path differs from manifest')
    verify_file(args.full_total_behaviors, spec['output'])
    verify_file(args.data_root / 'manifest.json', spec['dataset_manifest'])
    data = read_json(args.data_root / 'manifest.json')
    if data.get('fraction') != 1.0:
        raise ValueError('This experiment requires the fixed full MIND-Small request split')
    if consistent:
        paths = evaluation_paths(args.data_root, args.matched_candidate_profile)
        if paths['total'] != args.full_total_behaviors:
            raise ValueError('Full Total path differs from consistent candidate certificate')
        for protocol in ('imp','rs30','rs90','rs150'):
            verify_file(paths[protocol], spec['source_evaluations'][protocol])
    else:
        for protocol in ('rs30','rs90','rs150'):
            verify_file(args.data_root / 'test' / PROTOCOL_FILES[protocol], spec['source_existing_evaluations'][protocol])
    return spec


def source_inventory():
    return {path.relative_to(ROOT).as_posix(): file_sha256(path)
            for directory in ('NNR-main','scripts') for path in (ROOT / directory).glob('*.py')}


def snapshot_source(output, sources):
    snapshot = output / 'source_snapshot'
    for relative, digest in sources.items():
        source, target = ROOT / relative, snapshot / relative
        content = source.read_bytes()
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError('Source changed while freezing experiment: ' + relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return snapshot


def check_snapshot(output, sources):
    snapshot = output / 'source_snapshot'
    for relative, digest in sources.items():
        if file_sha256(snapshot / relative) != digest:
            raise ValueError('Frozen source changed: ' + relative)
    for relative in ('scripts/run_converged_nnr.py','scripts/convergence_control.py',
                     'scripts/nnr_artifacts.py','scripts/model_batch_defaults.py',
                     'scripts/training_validation.py','scripts/prepare_development_candidates.py',
                     'scripts/scheduled_cooling.py','scripts/candidate_policy.py',
                     'scripts/consistent_candidate_profile.py','scripts/nnr_consistent_preprocessing.py','scripts/context_feature_policy.py'):
        if file_sha256(ROOT / relative) != sources[relative]:
            raise ValueError('Restart control source changed: ' + relative)
    return snapshot


def config_identity(args, inputs, total, batch):
    identity = {'model':args.model,'train_type':args.train_type,'seed':args.seed,
        'data_root':str(args.data_root),'embedding_root':str(args.embedding_root),
        'max_epochs':args.max_epochs,'min_epochs':args.min_epochs,'patience':args.patience,
        'min_delta':args.min_delta,'batch_size':batch['value'],
        'dev_protocol':args.dev_protocol,'dev_behaviors':str(args.dev_behaviors),
        'selection_metric':args.selection_metric,'inputs':inputs,
        'full_total':total['output'],'candidate_manifest':fingerprint(args.candidate_manifest),
        'chunk_rows':args.chunk_rows,'personalized_batch_size':args.personalized_batch_size,
        'npa_features_on_gpu':args.npa_features_on_gpu}



    if getattr(args, 'cooling', False):
        from scheduled_cooling import POLICY
        identity['cooling_policy'] = dict(POLICY)
    if getattr(args, 'matched_candidate_profile', None) is not None:
        identity['matched_candidate_profile'] = args.matched_candidate_profile
    if (getattr(args, 'matched_candidate_profile', None) or {}).get('profile') == CONSISTENT_PROFILE:
        identity['user_id_reference'] = fingerprint(args.data_root/'train/behaviors.tsv')
    if getattr(args, 'paired_preprocessing_from', None) is not None:
        identity['paired_preprocessing_reference'] = fingerprint(args.paired_preprocessing_from/'experiment.json')
    return identity


def configure_development(args):
    """Bind validation to the training family before reading any run artifacts."""
    args.dev_protocol = resolve_dev_protocol(args.train_type, args.dev_protocol)
    expected = resolve_dev_path(args.data_root, args.train_type, args.dev_protocol)
    chosen = (args.dev_behaviors or expected).resolve()
    if chosen != expected:
        raise ValueError("Development file must be the matching held-out split, never test candidates")
    args.dev_behaviors = chosen
    return args


def nnr_runtime_arguments(args, batch):
    """Forward the actual matching dev protocol into the legacy corpus loader."""
    ne, ue = MODELS[args.model]
    return ['converged_nnr','--mode','train','--dataset','small','--news_encoder',ne,
        '--user_encoder',ue,'--train_type',args.train_type,'--dev_type',args.dev_protocol,'--test_type','imp',
        '--data_root',str(args.data_root),'--embedding_root',str(args.embedding_root),
        '--seed',str(args.seed),'--epoch',str(args.max_epochs),'--batch_size',str(batch['value'])]


def restore_committed_training(manifest, checkpoint, identity):
    """Discard only uncommitted JSON bookkeeping, not files or completed epochs."""
    if checkpoint.get('schema') != 'converged_training_v1' or checkpoint['identity'] != identity:
        raise ValueError('Last checkpoint is not an exact restart for this experiment')
    control = EarlyStopping.from_state_dict(checkpoint['early_stopping'])
    if checkpoint['epoch'] != control.last_epoch or len(checkpoint['epoch_records']) != control.last_epoch:
        raise ValueError('Committed checkpoint epoch/history mismatch')
    manifest['epoch_records'] = copy.deepcopy(checkpoint['epoch_records'])
    manifest['best_epoch'] = control.best_epoch
    manifest['epochs_completed'] = control.last_epoch
    manifest['convergence'] = control.state_dict()
    return control


def main():
    start, start_unix = time.monotonic(), time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--train-type', choices=('imp','rs50'), required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--embedding-root', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--max-epochs', type=int, default=50)
    parser.add_argument('--min-epochs', type=int, default=5)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--min-delta', type=float, default=1e-4)
    parser.add_argument('--dev-protocol', choices=('auto','imp','rs50'), default='auto',
                        help='Match training: IMP uses IMP dev; RandS50 uses RandS50 dev')
    parser.add_argument('--dev-behaviors', type=Path)
    parser.add_argument('--selection-metric', choices=('avg','AUC','MRR','nDCG@5','nDCG@10'), default='avg')
    parser.add_argument('--full-total-behaviors', type=Path, required=True)
    parser.add_argument('--candidate-manifest', type=Path, required=True)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--preprocessing-from', type=Path)
    parser.add_argument('--paired-preprocessing-from', type=Path, help='Completed matching IMP run whose dictionary hashes must agree before model initialization')
    parser.add_argument('--personalized-batch-size', type=int, default=512)
    parser.add_argument('--npa-features-on-gpu', action='store_true')
    parser.add_argument('--chunk-rows', type=int, default=5000)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--train-only', action='store_true')
    parser.add_argument('--cooling', action='store_true', help='Rest 2 minutes per 10 training minutes and 15 minutes per 3 wall-clock hours')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    for name in ('data_root','embedding_root','run_dir','full_total_behaviors','candidate_manifest'):
        setattr(args,name,getattr(args,name).resolve())
    configure_development(args)
    args.matched_candidate_profile = validate_candidate_profile(args.data_root, args.train_type)
    if not 0 <= args.seed < 2**32 or args.chunk_rows < 1 or not 1 <= args.personalized_batch_size <= 8192:
        raise ValueError('Invalid seed or chunk size')
    control = EarlyStopping(args.max_epochs,args.min_epochs,args.patience,args.min_delta)
    batch = resolve_training_batch(args.model,args.batch_size,ROOT)
    context_feature_spec(args.data_root,model=args.model)
    inputs = check_inputs(args.data_root,args.train_type,args.dev_protocol,'imp')
    inputs['glove'] = fingerprint(args.embedding_root/'glove.6B.300d.txt')
    # Preserve the old preprocessing-signature format; optional cache metadata is
    # recorded separately because a local GloVe cache appearing is not data drift.
    total = validate_total(args)
    identity = config_identity(args,inputs,total,batch)
    if args.check_only:
        print(json.dumps({'status':'validated','model':args.model,'training':args.train_type,
            'seed':args.seed,'selection':'training-matched development only',
            'dev_protocol':args.dev_protocol,'dev_file':fingerprint(args.dev_behaviors),
            'matched_candidate_profile':args.matched_candidate_profile,'max_epochs':args.max_epochs,
            'min_epochs':args.min_epochs,'patience':args.patience,'new_training_started':False}),flush=True)
        return
    output = args.run_dir
    manifest_path, last_path, best_path = output/'experiment.json',output/'last_checkpoint.pt',output/'best_model.pt'
    if args.resume:
        manifest = read_json(manifest_path)
        if manifest['identity'] != identity:
            raise ValueError('Resume data/settings/seed identity differs')
        snapshot = check_snapshot(output,manifest['source_sha256'])
        if manifest['status'] == 'completed':
            print(json.dumps({'status':'already_completed','manifest':str(manifest_path)}),flush=True)
            return
        if not last_path.exists():
            raise ValueError('No committed optimizer/RNG checkpoint; preserve this attempt and start a new run')
        for name, expected in manifest['preprocess_files'].items():
            verify_file(output/name,expected)
    else:
        output.mkdir(parents=True,exist_ok=False)
        sources = source_inventory()
        snapshot = snapshot_source(output,sources)
        manifest = {'schema':3,'status':'preparing','historical_R2_reproduction':False,
            'model':args.model,'train_type':args.train_type,'seed':args.seed,'identity':identity,
            'data_root':str(args.data_root),'inputs':inputs,'source_sha256':sources,
            'source_snapshot':str(snapshot),'training_batch_resolution':batch,'batch_size':batch['value'],
            'epochs_requested':args.max_epochs,'epochs_completed':0,'epoch_records':[],
            'convergence':control.state_dict(),'evaluations':{},'evaluation_artifacts':{},
            'timing_intervals':[],'execution_segments':[],'embedding_builds':[],
            'candidate_validation':{'manifest':fingerprint(args.candidate_manifest),'file':total['output'],
                'statistics':total['statistics'],'validation':total['validation']},'chunks':[],
            'selection':{'protocol':args.dev_protocol,'metric':args.selection_metric,'file':fingerprint(args.dev_behaviors),
                'matched_candidate_profile':args.matched_candidate_profile,
                'formula':'(AUC+MRR+(nDCG@5+nDCG@10)/2)/3' if args.selection_metric=='avg' else args.selection_metric,
                'test_used_for_selection':False,
                'limit':'Development candidates match training; test protocols never choose the checkpoint.'},
            'evaluation_resources_requested':{'personalized_batch_size':args.personalized_batch_size,
                'npa_features_on_gpu':args.npa_features_on_gpu}}
    segment = {'start_unix':start_unix,'status':'running','elapsed_seconds':0.0}
    manifest['execution_segments'].append(segment)
    import scheduled_cooling as cooling
    def record():
        segment['elapsed_seconds'] = time.monotonic()-start
        manifest['elapsed_seconds'] = sum(row['elapsed_seconds'] for row in manifest['execution_segments'])
        cooling.record_manifest(manifest, segment)
        save_json_atomic(manifest_path,manifest)
    record()
    try:
        os.chdir(output)
        sys.dont_write_bytecode = True
        sys.path[:0] = [str(snapshot/'scripts'),str(snapshot/'NNR-main')]
        import numpy as np
        import torch
        from torch.utils.data import DataLoader
        from config import Config
        from MIND_corpus import MIND_Corpus
        from MIND_dataset import MIND_Train_Dataset
        from model import Model
        from naml_cached_eval import NamlCachedEvaluator, model_fingerprint
        from nnr_cached_eval import NNRCachedEvaluator
        from nnr_batch_alignment import apply_batch_alignment_fixes
        from evaluate_existing_nnr import evaluate_chunks
        from evaluation_progress import evaluate_test_file
        from experiment_progress import train_progress, finish_progress, clear_console_progress
        torch.set_num_threads(4)
        if args.cooling:
            cooling.install(output/'cooling_events.jsonl', torch.cuda.synchronize)
        sys.argv = nnr_runtime_arguments(args, batch)
        with (output/'runtime_initialization.log').open('a',encoding='utf-8') as initlog, contextlib.redirect_stdout(initlog):
            config = Config()
            reference_fp = configure_reference(config, args.data_root, args.matched_candidate_profile)
            prep_spec = {'input_hashes':[row['sha256'] for row in inputs['files']],
                'glove':inputs['glove']['sha256'],'seed':args.seed,
                'parameters':{key:getattr(config,key) for key in ('dataset','tokenizer','word_threshold',
                    'word_embedding_dim','max_title_length','max_abstract_length','max_history_num',
                    'no_self_connection','no_adjacent_normalization','gcn_normalization_type')},
                'corpus_source_sha256':manifest['source_sha256']['NNR-main/MIND_corpus.py']}
            if reference_fp is not None:
                prep_spec['user_id_reference'] = reference_fp
            signature = hashlib.sha256(json.dumps(prep_spec,sort_keys=True).encode()).hexdigest()
            if args.resume:
                if signature != manifest['preprocess_signature']:
                    raise ValueError('Resume preprocessing identity mismatch')
            elif args.preprocessing_from:
                origin = args.preprocessing_from.resolve()
                previous = read_json(origin/'experiment.json')
                if signature != previous['preprocess_signature']:
                    raise ValueError('Preprocessing reuse requires identical seed/data/source/parameters')
                for name, expected in previous['preprocess_files'].items():
                    verify_file(origin/name,expected)
                    target = output/name
                    if target.exists():
                        raise ValueError('Preserve existing preprocessing target')
                    shutil.copyfile(origin/name,target)
                manifest['preprocessing_reused_from'] = str(origin)
            prep_started = time.monotonic()
            corpus = MIND_Corpus(config)
        segment['corpus_load_seconds'] = time.monotonic()-prep_started
        manifest['preprocess_signature'] = signature
        if not args.resume:
            manifest['preprocess_seconds'] = segment['corpus_load_seconds']
            manifest['preprocess_files'] = {path.name:fingerprint(path) for path in output.iterdir()
                if path.is_file() and path.suffix in ('.pkl','.json') and path.name!='experiment.json'}
            manifest['effective_config'] = {key:value for key,value in vars(config).items()
                if isinstance(value,(str,bool,int,float,list,dict,type(None)))}
        elif vars(config) != manifest['effective_config']:
            raise ValueError('Resume effective model configuration changed')
        if config.negative_sample_num != 4:
            raise ValueError('The agreed experiment samples four negatives per positive each epoch')
        environment = {'python':sys.version,'torch':str(torch.__version__),
            'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(config.device_id),
            'cuda_matmul_allow_tf32':bool(torch.backends.cuda.matmul.allow_tf32),
            'cudnn_allow_tf32':bool(torch.backends.cudnn.allow_tf32),
            'cudnn_deterministic':bool(torch.backends.cudnn.deterministic),
            'cudnn_benchmark':bool(torch.backends.cudnn.benchmark)}
        if args.resume and environment != manifest['environment']:
            raise ValueError('Exact training restart requires the recorded environment/precision')
        manifest['environment'] = environment
        if args.paired_preprocessing_from is not None:
            manifest['paired_dictionary_gate'] = paired_dictionary_gate(output, args.paired_preprocessing_from, config, model=args.model, seed=args.seed)
            record()
        # Pair initialization across IMP/RandS and isolate it from preprocessing RNG.
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
        model = Model(config).cuda()
        adjustments = apply_batch_alignment_fixes(model) if args.model in ('NPA','CNE-SUE') else []
        manifest['implementation_adjustments'] = adjustments
        model.initialize()
        initialized_hash = model_fingerprint(model)
        if args.paired_preprocessing_from is not None:
            verify_paired_initialization(initialized_hash, manifest['paired_dictionary_gate'])
        if args.resume and initialized_hash != manifest['initial_model_state_sha256']:
            raise ValueError('Reconstructed paired initialization differs')
        manifest['initial_model_state_sha256'] = initialized_hash
        manifest['effective_config_sha256'] = hashlib.sha256(json.dumps(config.attribute_dict,sort_keys=True).encode()).hexdigest()
        dataset = MIND_Train_Dataset(corpus)
        optimizer = torch.optim.Adam(filter(lambda value:value.requires_grad,model.parameters()),
                                     lr=config.lr,weight_decay=config.weight_decay)
        best_state = None
        if args.resume:
            last = torch.load(last_path,map_location='cpu')
            control = restore_committed_training(manifest,last,identity)
            model.load_state_dict(last['model_state'],strict=True)
            optimizer.load_state_dict(last['optimizer_state'])
            best_state = last['best_model_state']
            restore_rng_state(last['rng_state'])
            # The last checkpoint is the atomic transaction authority; restore the
            # derived best file if interruption occurred between the two writes.
            atomic_torch_save({model.model_name:best_state,'epoch':control.best_epoch,
                'config':config.attribute_dict},best_path)
            del last
            stale_parent=output/'epoch_caches'
            if stale_parent.exists():
                for stale_cache in stale_parent.iterdir():
                    if stale_cache.is_dir():
                        manifest.setdefault('cache_cleanup',[]).append(
                            dict(cleanup_epoch_cache(output,stale_cache),reason='restart_after_committed_epoch'))
        manifest['training_samples'] = len(dataset)
        record()
        def make_evaluator(cache_root,epoch,phase):
            cache_start = time.monotonic()
            if args.model=='NAML':
                result = NamlCachedEvaluator(model,corpus,cache_root,news_batch_size=128,user_batch_size=256)
            else:
                result = NNRCachedEvaluator(model,corpus,config,cache_root,
                    personalized_batch_size=args.personalized_batch_size,npa_features_on_gpu=args.npa_features_on_gpu)
            manifest['embedding_builds'].append({'epoch':epoch,'phase':phase,
                'constructor_seconds':time.monotonic()-cache_start,'cache_identity':result.identity})
            return result
        while control.stop_reason is None:
            epoch = control.last_epoch+1
            manifest['status']='training'; manifest['current_epoch']=epoch; record()
            cooling.set_phase('training', epoch=epoch)
            train_rest_start = cooling.marker()
            epoch_begin,epoch_unix = time.monotonic(),time.time()
            with (output/'negative_sampling.log').open('a',encoding='utf-8') as sampling_log, contextlib.redirect_stdout(sampling_log):
                dataset.negative_sampling()
            loader = DataLoader(dataset,batch_size=batch['value'],shuffle=True,num_workers=0,pin_memory=True)
            model.train(); total_loss=0.0; samples=0; last_print=time.monotonic()
            steps = len(loader)
            train_progress(epoch, args.max_epochs, 0, steps)
            for step,values in enumerate(loader,1):
                values = tuple(value.cuda(non_blocking=True) for value in values)
                logits = model(*values)
                loss = -torch.log_softmax(logits,dim=1)[:,0].mean()
                for encoder in (model.news_encoder,model.user_encoder):
                    if encoder.auxiliary_loss is not None:
                        loss = loss+encoder.auxiliary_loss.mean()
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite training loss')
                optimizer.zero_grad(); loss.backward()
                if config.gradient_clip_norm>0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),config.gradient_clip_norm)
                optimizer.step()
                size=values[0].size(0); total_loss+=loss.item()*size; samples+=size
                train_progress(epoch, args.max_epochs, step, steps)
                cooling.training_boundary()
                if time.monotonic()-last_print>=3600:
                    clear_console_progress()
                    print(json.dumps({'stage':'training','model':args.model,'training':args.train_type,
                        'seed':args.seed,'epoch':epoch,'step':step,'steps':len(loader),
                        'mean_loss':total_loss/samples}),flush=True)
                    last_print=time.monotonic(); record()
            train_seconds=time.monotonic()-epoch_begin
            train_cooling = cooling.duration_fields(train_seconds, train_rest_start)
            cooling.set_phase('development_setup', epoch=epoch)
            finish_progress()
            manifest['timing_intervals'].append({'phase':'training','epoch':epoch,'start_unix':epoch_unix,
                'end_unix':time.time(),'duration_seconds':train_seconds})
            manifest['status']='validating'; record()
            epoch_cache=prepare_epoch_cache(output,epoch)
            evaluator=make_evaluator(epoch_cache,epoch,'development')
            gate=evaluator.verify_against_forward(args.dev_behaviors,max_entries=8)
            if gate['status']!='passed':
                raise RuntimeError('Development native/cache equivalence failed')
            attempt=len(list((output/'evaluation').glob('dev_epoch%03d_attempt_*.metrics.json'%epoch)))+1 if (output/'evaluation').exists() else 1
            dev_prefix=output/'evaluation'/('dev_epoch%03d_attempt_%03d'%(epoch,attempt))
            while list(dev_prefix.parent.glob(dev_prefix.name+'.*')):
                attempt+=1; dev_prefix=output/'evaluation'/('dev_epoch%03d_attempt_%03d'%(epoch,attempt))
            cooling.set_phase('dev', epoch=epoch)
            dev_start,dev_unix=time.monotonic(),time.time()
            result=evaluator.evaluate_file(args.dev_behaviors,dev_prefix)
            value=selection_value(result['metrics'],args.selection_metric)
            decision=control.update(epoch,value)
            gate_path=output/'development_gates'/('epoch%03d_attempt_%03d.json'%(epoch,attempt))
            save_json_atomic(gate_path,gate)
            row={'epoch':epoch,'mean_loss':total_loss/samples,'train_seconds':train_seconds,
                'train_cooling_rest_seconds':train_cooling['cooling_rest_seconds'],
                'train_net_seconds':train_cooling['net_elapsed_seconds'],
                'dev':dict(result,prefix=str(dev_prefix)),'selection_value':value,'decision':decision,
                'native_equivalence':fingerprint(gate_path)}
            manifest['epoch_records'].append(row)
            manifest['timing_intervals'].append({'phase':'dev','epoch':epoch,'start_unix':dev_unix,
                'end_unix':time.time(),'duration_seconds':time.monotonic()-dev_start})
            if decision['is_best']:
                best_state={name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
            manifest['best_epoch']=control.best_epoch; manifest['epochs_completed']=epoch
            manifest['convergence']=control.state_dict()
            # Save AFTER all validation/cache effects. Last is the atomic commit;
            # best_model.pt and manifest JSON can be reconstructed from it.
            atomic_torch_save({'schema':'converged_training_v1','epoch':epoch,'identity':identity,
                'config':config.attribute_dict,'model_state':model.state_dict(),
                'optimizer_state':optimizer.state_dict(),'rng_state':capture_rng_state(),
                'early_stopping':control.state_dict(),'epoch_records':manifest['epoch_records'],
                'best_model_state':best_state},last_path)
            if decision['is_best']:
                atomic_torch_save({model.model_name:best_state,'epoch':control.best_epoch,
                    'config':config.attribute_dict},best_path)
            manifest['last_checkpoint']=fingerprint(last_path)
            record()
            del evaluator; gc.collect(); torch.cuda.empty_cache()
            manifest.setdefault('cache_cleanup',[]).append(cleanup_epoch_cache(output,epoch_cache))
            record()
            print(json.dumps({'stage':'epoch_completed','model':args.model,'training':args.train_type,
                'seed':args.seed,'epoch':epoch,'dev_selection':value,'best_epoch':control.best_epoch,
                'bad_epochs':control.bad_epochs,'stop_reason':control.stop_reason}),flush=True)
        manifest['checkpoint']=fingerprint(best_path)
        manifest['status']='trained'; record()
        if args.train_only:
            segment['status']='completed'; record()
            print(json.dumps({'status':'trained','best_epoch':control.best_epoch,'stop_reason':control.stop_reason,
                'test_evaluations':0,'manifest':str(manifest_path)}),flush=True)
            return
        model.load_state_dict(best_state,strict=True)
        manifest['status']='evaluating'; record()
        cooling.set_phase('test_setup')
        evaluator=make_evaluator(output/'embedding_cache',control.best_epoch,'test')
        gate=evaluator.verify_against_forward(args.dev_behaviors,max_entries=8)
        if gate['status']!='passed':
            raise RuntimeError('Selected best checkpoint cache gate failed')
        save_json_atomic(output/'cache_equivalence_best.json',gate)
        protocol_paths = evaluation_paths(args.data_root, args.matched_candidate_profile)
        for protocol in PROTOCOL_FILES:
            behavior_path = protocol_paths[protocol]
            if protocol in manifest['evaluations']:
                for expected in manifest['evaluation_artifacts'][protocol].values():
                    verify_file(expected['path'],expected)
                continue
            attempt=1; prefix=output/'evaluation'/('test_'+protocol)
            while list(prefix.parent.glob(prefix.name+'.*')):
                attempt+=1; prefix=output/'evaluation'/('test_%s_attempt_%03d'%(protocol,attempt))
            cooling.set_phase('test', protocol=protocol)
            begin,begin_unix=time.monotonic(),time.time()
            result=evaluate_test_file(evaluator, behavior_path, prefix, protocol)
            finish_progress()
            manifest['evaluations'][protocol]=dict(result,prefix=str(prefix))
            manifest['evaluation_artifacts'][protocol]={name:fingerprint(path) for name,path in result['files'].items()}
            manifest['timing_intervals'].append({'phase':'test','protocol':protocol,'start_unix':begin_unix,
                'end_unix':time.time(),'duration_seconds':time.monotonic()-begin})
            record()
        if 'total' not in manifest['evaluations']:
            full_args=argparse.Namespace(behaviors=args.full_total_behaviors,
                candidate_manifest=args.candidate_manifest,chunk_rows=args.chunk_rows)
            cooling.set_phase('test', protocol='total')
            begin,begin_unix=time.monotonic(),time.time()
            result,prefix=evaluate_chunks(evaluator,full_args,output,manifest,record)
            manifest['evaluations']['total']=dict(result,prefix=str(prefix))
            manifest['evaluation_artifacts']['total']={name:fingerprint(path) for name,path in result['files'].items()}
            manifest['timing_intervals'].append({'phase':'test','protocol':'total','start_unix':begin_unix,
                'end_unix':time.time(),'duration_seconds':time.monotonic()-begin})
            record()
        else:
            for expected in manifest['evaluation_artifacts']['total'].values():
                verify_file(expected['path'],expected)
        segment['status']='completed'; record()
        manifest['timings_seconds']={'training':sum(row['train_seconds'] for row in manifest['epoch_records']),
            'dev_evaluation':sum(row['dev']['elapsed_seconds'] for row in manifest['epoch_records']),
            'tests':{key:value['elapsed_seconds'] for key,value in manifest['evaluations'].items()},
            'cache_construction':sum(row['constructor_seconds'] for row in manifest['embedding_builds']),
            'preprocessing':manifest['preprocess_seconds'],'total':manifest['elapsed_seconds']}
        manifest['timing_scope']={'total':'Sum of recorded active segments; excludes restart gaps; abrupt-kill tail may be unmeasured.',
            'training':'Committed epochs only; incomplete reattempt work is represented only by measured active segments.',
            'tests':'Each completed evaluate_file timer, or completed full-Total shard timers; no prior one-epoch timing reused.'}
        cooling.finalize_timing(manifest)
        manifest['status']='completed'; record()
        print(json.dumps({'status':'completed','model':args.model,'training':args.train_type,'seed':args.seed,
            'epochs_completed':control.last_epoch,'best_epoch':control.best_epoch,'stop_reason':control.stop_reason,
            'manifest':str(manifest_path)}),flush=True)
    except BaseException:
        segment['status']='failed'; manifest['status']='failed'; manifest['error']=traceback.format_exc(); record()
        raise


if __name__=='__main__':
    main()
