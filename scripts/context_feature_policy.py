"""Explicit context-feature omission for the six paper models only.

No context vector or graph is fabricated. Other encoders retain the legacy
requirement. The declaration is part of dataset/candidate/run provenance.
"""
import hashlib
import json
from pathlib import Path

REQUIRED='required'
UNUSED='context_not_used_by_paper_models'
PAPER_MODELS=('NAML','NRMS','LSTUR','NPA','CNE-SUE','DIGAT')
NNR_ENCODERS={'NAML':('NAML','ATT'),'NRMS':('MHSA','MHSA'),'LSTUR':('CNN','LSTUR'),
              'NPA':('PNE','PUE'),'CNE-SUE':('CNE','SUE')}


def validate_declaration(document,*,model=None,encoder=None,user_encoder=None):
    mode=document.get('context_feature_mode',REQUIRED)
    if mode==REQUIRED:
        declared=document.get('context_features')
        if declared and declared.get('mode',REQUIRED)!=REQUIRED:
            raise ValueError('Contradictory context-feature declaration')
        return {'mode':REQUIRED,'required':True}
    if mode!=UNUSED:
        raise ValueError('Unknown context-feature mode')
    selected=document.get('selected_models')
    features=document.get('context_features')
    if (not isinstance(selected,list)or not selected or len(set(selected))!=len(selected)
        or any(value not in PAPER_MODELS for value in selected)):
        raise ValueError('Graphless context mode requires distinct declared paper models')
    if (not isinstance(features,dict)or features.get('mode')!=UNUSED
        or features.get('required_by_selected_models')is not False
        or features.get('selected_models')!=selected
        or not isinstance(features.get('allowed_models'),list)
        or len(features['allowed_models'])!=len(PAPER_MODELS)
        or set(features['allowed_models'])!=set(PAPER_MODELS)
        or features.get('context_outputs')!=[]):
        raise ValueError('Complete explicit unused-context declaration is required')
    if model is not None and (model not in PAPER_MODELS or model not in selected):
        raise ValueError('Selected model is not certified for unused context features')
    if encoder is not None or user_encoder is not None:
        models=[name for name,pair in NNR_ENCODERS.items()if pair==(encoder,user_encoder)]
        if len(models)!=1 or models[0]not in selected:
            raise ValueError('This news/user encoder pair requires real context features or a matching declaration')
    return {'mode':UNUSED,'required':False,'selected_models':selected,'declaration':features}


def context_feature_spec(data_root,*,model=None,encoder=None,user_encoder=None):
    if not data_root:
        return {'mode':REQUIRED,'required':True}
    root=Path(data_root).resolve();manifest=root/'manifest.json'
    if not manifest.is_file():
        return {'mode':REQUIRED,'required':True}
    raw=manifest.read_bytes()
    document=json.loads(raw.decode('utf-8-sig'))
    result=validate_declaration(document,model=model,encoder=encoder,user_encoder=user_encoder)
    if not result['required']:
        for split in ('train','dev','test'):
            if(root/split/'context_embedding.vec').exists():
                raise ValueError('Unused-context mode must not contain fabricated or unbound context vectors')
        result['manifest']={'path':str(manifest),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    return result


def declaration_fields(spec):
    if spec['required']:
        return {}
    return {'context_feature_mode':UNUSED,'selected_models':spec['selected_models'],
            'context_features':spec['declaration']}
