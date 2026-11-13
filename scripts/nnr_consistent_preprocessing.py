"""Keep new-profile user dictionaries paired with preserved IMP masters."""
import json
from pathlib import Path
from context_feature_policy import context_feature_spec
from candidate_policy import PROFILE
from consistent_candidate_profile import fingerprint


def configure_reference(config,data_root,profile):
 context=context_feature_spec(data_root,encoder=getattr(config,'news_encoder',None),user_encoder=getattr(config,'user_encoder',None))
 if not context['required']:
  config.context_feature_mode=context['mode'];config.context_features_required=False
 if profile is not None and profile.get('profile')==PROFILE:
  path=(Path(data_root).resolve()/'train/behaviors.tsv').resolve()
  config.user_id_reference_file=str(path)
  return fingerprint(path)
 return None


def paired_dictionary_gate(run_dir,reference_dir,config,*,model,seed):
 """Check exact mapping bytes against a completed IMP run before model init.

 Only dictionary JSONs are compared. No behavior-dependent graph/cache or model
 tensor is copied from the old run or used to weaken preprocessing identities.
 """
 run_dir=Path(run_dir).resolve();reference_dir=Path(reference_dir).resolve()
 manifest_path=reference_dir/'experiment.json'
 original=json.loads(manifest_path.read_text(encoding='utf-8-sig'))
 if (original.get('status')!='completed' or original.get('model')!=model
     or original.get('seed')!=seed or original.get('train_type')!='imp'):
  raise ValueError('Paired dictionary reference must be the matching completed IMP run')
 initial_hash=original.get('initial_model_state_sha256')
 if not isinstance(initial_hash,str)or len(initial_hash)!=64:
  raise ValueError('Original IMP initial model fingerprint is absent')
 names=[name+'-'+config.dataset+'.json'for name in ('user_ID','news_ID','category','subCategory','entity')]
 names.append('vocabulary-%s-%s-%s-%s-%s.json'%(config.word_threshold,config.tokenizer,
                 config.max_title_length,config.max_abstract_length,config.dataset))
 proof={}
 for name in names:
  expected=original.get('preprocess_files',{}).get(name)
  if not expected:raise ValueError('Original IMP dictionary provenance is absent: '+name)
  reference=fingerprint(reference_dir/name);actual=fingerprint(run_dir/name)
  if any(reference[k]!=expected[k]for k in ('bytes','sha256')):
   raise ValueError('Original IMP dictionary differs from its manifest: '+name)
  if any(actual[k]!=reference[k]for k in ('bytes','sha256')):
   raise ValueError('New/IMP dictionary hashes differ before model initialization: '+name)
  if json.loads((run_dir/name).read_text(encoding='utf-8-sig'))!=json.loads((reference_dir/name).read_text(encoding='utf-8-sig')):
   raise ValueError('New/IMP dictionary values differ: '+name)
  proof[name]={'new':actual,'original':reference}
 return {'status':'passed','initial_model_sha256':initial_hash,'reference_run':str(reference_dir),'reference_manifest':fingerprint(manifest_path),'dictionaries':proof}

def verify_paired_initialization(initialized_hash,proof):
 if proof.get('status')!='passed' or initialized_hash!=proof.get('initial_model_sha256'):
  raise ValueError('Paired IMP/RandS initial model fingerprint differs before training')
