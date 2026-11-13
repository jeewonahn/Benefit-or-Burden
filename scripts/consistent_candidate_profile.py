"""Verify the new complete-proxy, strict, split-consistent candidate certificate.

This validator reads files and hashes only. It never generates candidates,
repairs a legacy dataset, loads a model, or weakens a source/checkpoint gate.
"""
import hashlib
import json
from pathlib import Path
from context_feature_policy import context_feature_spec,declaration_fields,validate_declaration
from candidate_policy import PROFILE,RECIPE,MANIFEST_NAME,PublicationIndex,parse_timestamp

OUTPUT_FILES={
 'train_rs50':'train/behaviors_rs50.tsv',
 'dev_rs50':'dev/behaviors_rs50.tsv',
 'test_imp':'test/behaviors_imp_eligible.tsv',
 'test_rs30':'test/behaviors_rs30.tsv',
 'test_rs90':'test/behaviors_rs90.tsv',
 'test_rs150':'test/behaviors_rs150.tsv',
 'test_total':'test/behaviors_total.tsv',
}
MASTER_FILES=('train/behaviors.tsv','dev/behaviors.tsv','test/behaviors.tsv')
REQUIRED_VALIDATION=('same_base_policy_all_splits','master_IMP_preserved',
                     'GT_multiset_consistent_all_test_protocols','exclusions_recorded')


def fingerprint(path):
 path=Path(path).resolve();h=hashlib.sha256()
 with path.open('rb')as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
 return {'path':str(path),'bytes':path.stat().st_size,'sha256':h.hexdigest()}


def _read(path):
 try:return json.loads(Path(path).read_text(encoding='utf-8-sig'))
 except (OSError,ValueError)as error:raise ValueError('Missing or invalid consistent-candidate certificate/reference')from error


def _expected(value,name):
 if not isinstance(value,dict)or type(value.get('bytes'))is not int or value['bytes']<0:
  raise ValueError('Missing size evidence for '+name)
 digest=value.get('sha256')
 if not isinstance(digest,str)or len(digest)!=64 or any(c not in '0123456789abcdef'for c in digest):
  raise ValueError('Missing SHA-256 evidence for '+name)
 return value


def _path(root,relative):
 if not isinstance(relative,str)or not relative or Path(relative).is_absolute():
  raise ValueError('Certificate output/reference paths must be relative')
 path=(root/relative).resolve()
 if path==root or root not in path.parents:raise ValueError('Certificate path escapes dataset root')
 return path


def _verify(root,expected,relative):
 expected=_expected(expected,relative)
 path=_path(root,relative)
 stated=expected.get('path')
 if stated is not None:
  p=Path(stated)
  declared=p.resolve()if p.is_absolute()else _path(root,stated)
  if declared!=path:raise ValueError('Certificate points to a different file: '+relative)
 try:actual=fingerprint(path)
 except OSError as error:raise ValueError('Missing consistent candidate file: '+relative)from error
 if any(actual[k]!=expected[k]for k in ('bytes','sha256')):
  raise ValueError('File differs from candidate certificate: '+relative)
 return actual


def validate_consistent_candidate_profile(data_root,*,verify_test_files=True):
 """Return the checked certificate identity and prescribed output bindings.

 Training validates all master/held-out inputs and complete proxy coverage.
 Test file hashes may be deferred only when a separate native evaluation gate
 will verify their exact recorded identities. The default verifies every file.
 """
 root=Path(data_root).resolve();path=root/MANIFEST_NAME;cert=_read(path)
 context=context_feature_spec(root)
 declared=validate_declaration(cert)
 if context['mode']!=declared['mode'] or declaration_fields(context)!=declaration_fields(declared):
  raise ValueError('Dataset/candidate context-feature declarations differ')
 if (type(cert.get('schema'))is not int or cert['schema']!=1 or cert.get('status')!='complete'
     or cert.get('profile')!=PROFILE or cert.get('sampling')!=RECIPE):
  raise ValueError('Complete strict split-consistent candidate profile is required')
 if type(cert.get('seed'))is not int or not 0<=cert['seed']<2**32:
  raise ValueError('Invalid candidate seed')
 if any(cert.get('validation',{}).get(k)is not True for k in REQUIRED_VALIDATION):
  raise ValueError('Candidate support/exclusion/master preservation was not certified')
 inputs=cert.get('imp_inputs',{});outputs=cert.get('outputs',{})
 if set(inputs)!=set(MASTER_FILES)or set(outputs)!=set(OUTPUT_FILES):
  raise ValueError('All three IMP masters and all seven candidate outputs are required')
 for relative in MASTER_FILES:_verify(root,inputs[relative],relative)
 for key,relative in OUTPUT_FILES.items():
  item=_expected(outputs[key],relative)
  if item.get('path')!=relative:raise ValueError('Unexpected candidate output path: '+key)
  _path(root,relative)
  if verify_test_files or key in ('train_rs50','dev_rs50'):_verify(root,item,relative)
 _verify(root,cert.get('publication_proxy'),'news_publish_time.tsv')
 _verify(root,cert.get('source_minimum_map'),'provenance/observed_first_impression_min.json')
 _verify(root,cert.get('exclusions'),'exclusions.jsonl')
 expected_map=_read(root/'provenance/observed_first_impression_min.json').get('all_impressions_min')
 if not isinstance(expected_map,dict)or not expected_map:raise ValueError('Independent observed minimum map is empty or absent')
 expected={news:parse_timestamp(stamp)for news,stamp in expected_map.items()}
 proxy=PublicationIndex.from_proxy(root/'news_publish_time.tsv').publication
 coverage={'expected_observed_ids':len(expected),'proxy_ids':len(proxy),
    'missing_ids':len(set(expected)-set(proxy)),'unexpected_ids':len(set(proxy)-set(expected)),
    'timestamp_mismatches':sum(proxy[n]!=expected[n]for n in set(proxy)&set(expected))}
 if cert.get('coverage')!=coverage or any(coverage[k]for k in ('missing_ids','unexpected_ids','timestamp_mismatches')):
  raise ValueError('Publication proxy is not a complete exact observed-minimum map')
 result=fingerprint(path)
 result.update(profile=PROFILE,seed=cert['seed'],outputs=outputs,coverage=coverage)
 if not context['required']:result.update(declaration_fields(context))
 return result


def evaluation_paths(data_root,profile=None):
 """Route the new profile's tests through its eligible-support files.

 Legacy datasets retain their original routing. IMP masters remain untouched.
 """
 root=Path(data_root).resolve()
 if profile is None and (root/MANIFEST_NAME).is_file():
  profile=validate_consistent_candidate_profile(root)
 if profile is not None and profile.get('profile')==PROFILE:
  return {p:root/OUTPUT_FILES['test_'+p]for p in ('imp','rs30','rs90','rs150','total')}
 return {p:root/'test'/name for p,name in {'imp':'behaviors.tsv','rs30':'behaviors_rs30.tsv',
  'rs90':'behaviors_rs90.tsv','rs150':'behaviors_rs150.tsv'}.items()}