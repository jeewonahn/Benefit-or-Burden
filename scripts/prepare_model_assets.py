"""Prepare public GloVe/MPNet inputs without importing torch or running a model.

Downloads occur only with --download. Local archives/model directories are read
only; new output folders contain manifests with actual relative-file hashes.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import urllib.request
import zipfile

GLOVE_URL='https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip'
GLOVE_PAGE='https://nlp.stanford.edu/projects/glove/'
GLOVE_LICENSE='https://opendatacommons.org/licenses/pddl/1-0/'
MODEL_ID='sentence-transformers/all-mpnet-base-v2'
MODEL_REVISION='e8c3b32edf5434bc2275fc9bab85f82640a19130'
MODEL_PATTERNS=('config.json','config_sentence_transformers.json','modules.json','sentence_bert_config.json',
                'tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.txt',
                '1_Pooling/config.json','pytorch_model.bin','README.md','LICENSE')
SEMANTIC_MANIFEST='TORS_ASSET_MANIFEST.json'
GLOVE_MANIFEST='GLOVE_ASSET_MANIFEST.json'


def now():return datetime.now(timezone.utc).isoformat()
def fingerprint(path):
    path=Path(path);h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return {'bytes':path.stat().st_size,'sha256':h.hexdigest()}
def save(path,value):
    p=Path(path);tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()))
    with tmp.open('w',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def manifest_files(root,paths):
    root=Path(root).resolve();out={}
    for item in paths:
        p=Path(item).resolve()
        if root not in p.parents:raise ValueError('Asset file escaped its output root')
        out[p.relative_to(root).as_posix()]=fingerprint(p)
    return out

def verify_asset_manifest(root,name,kind):
    root=Path(root).resolve();m=read(root/name)
    if m.get('schema')!=1 or m.get('status')!='complete' or m.get('kind')!=kind or not m.get('files'):
        raise ValueError('Complete asset manifest is required: '+str(root/name))
    for relative,pin in m['files'].items():
        p=(root/relative).resolve()
        if root not in p.parents or fingerprint(p)!=pin:raise ValueError('Asset file differs from manifest: '+relative)
    return m


def validate_glove_text(path, *, dimensions=300, expected_rows=400000):
    """Validate vocabulary rows, allowing the official final <unk> vector.

    Stanford's 6B 300d distribution contains 400,000 ordinary tokens followed
    by one <unk> row. Some compatible original distributions omit that row.
    No extra ordinary rows, duplicate <unk>, headers or malformed vectors pass.
    """
    count = ordinary = unknown = 0
    with Path(path).open(encoding='utf-8') as stream:
        for count, line in enumerate(stream, 1):
            parts = line.rstrip('\r\n').split()
            if len(parts) != dimensions + 1 or not all(math.isfinite(float(v)) for v in parts[1:]):
                raise ValueError('Malformed/nonfinite GloVe vector at row ' + str(count))
            if parts[0] == '<unk>':
                unknown += 1
                if unknown > 1:
                    raise ValueError('Duplicate GloVe <unk> row')
            else:
                if unknown:
                    raise ValueError('The optional GloVe <unk> vector must be last')
                ordinary += 1
    if ordinary != expected_rows or count != ordinary + unknown:
        raise ValueError('Expected ' + str(expected_rows) + ' ordinary GloVe rows, got ' + str(ordinary))
    return {'rows': count, 'ordinary_rows': ordinary, 'unknown_rows': unknown,
            'dimensions': dimensions, 'finite_values_checked': 'all rows, streamed'}


def _download(url,target):
    if not url.startswith('https://'):raise ValueError('HTTPS asset URL required')
    req=urllib.request.Request(url,headers={'User-Agent':'TORS-reproducibility-assets/1'})
    with urllib.request.urlopen(req,timeout=60) as response,Path(target).open('wb') as output:
        shutil.copyfileobj(response,output,length=8*1024*1024)


def prepare_glove(output, *, archive=None, text=None, download=False,
                  expected_sha256=None, download_fn=None,
                  dimensions=300, expected_rows=400000):
    """Register a verified local vector file, or extract one selected ZIP member."""
    output = Path(output).resolve()
    target = output / 'glove.6B.300d.txt'
    if expected_sha256 and not re.fullmatch('[0-9a-fA-F]{64}', expected_sha256):
        raise ValueError('Expected SHA-256 must have 64 hexadecimal characters')
    expected_sha256 = expected_sha256.lower() if expected_sha256 else None
    if (output / GLOVE_MANIFEST).is_file():
        result = verify_asset_manifest(output, GLOVE_MANIFEST, 'glove.6B.300d')
        if expected_sha256 and result['input']['sha256'] != expected_sha256:
            raise ValueError('Existing GloVe input identity differs')
        return result
    if output.exists():
        raise ValueError('Preserve existing unregistered output; choose a new GloVe folder')
    if sum((archive is not None, text is not None, bool(download))) != 1:
        raise ValueError('Choose one local --archive, local --text, or explicit --download')
    source = Path(archive or text).resolve() if archive or text else output / 'glove.6B.zip'
    if archive or text:
        if not source.is_file():
            raise ValueError('Local GloVe input does not exist: ' + str(source))
        if output in source.parents:
            raise ValueError('Local input must be outside the new output')
    output.mkdir(parents=True)
    if download:
        temporary = output / 'glove.6B.zip.part'
        (download_fn or _download)(GLOVE_URL, temporary)
        os.replace(temporary, source)
    input_fp = fingerprint(source)
    if expected_sha256 and input_fp['sha256'] != expected_sha256:
        raise ValueError('GloVe input SHA-256 mismatch')
    temporary = output / 'glove.6B.300d.txt.part'
    if text:
        shutil.copy2(source, temporary)
        integrity = 'observed text SHA-256; all vector shapes and finite values'
    else:
        with zipfile.ZipFile(source) as archive_file:
            members = [i for i in archive_file.infolist() if i.filename == 'glove.6B.300d.txt']
            if len(members) != 1:
                raise ValueError('Archive needs exactly one glove.6B.300d.txt')
            info = members[0]
            mode = (info.external_attr >> 16) & 0o170000
            if info.is_dir() or mode == 0o120000 or not 0 < info.file_size <= 4 * 1024 ** 3:
                raise ValueError('Invalid GloVe archive member')
            with archive_file.open(info) as inp, temporary.open('wb') as out:
                shutil.copyfileobj(inp, out, length=8 * 1024 * 1024)
        # ZipExtFile checks the selected member's CRC at EOF. Other dimensions
        # are not extracted or claimed as CRC-verified.
        integrity = 'observed archive SHA-256; selected 300d member ZIP CRC; all vector shapes and finite values'
    summary = validate_glove_text(temporary, dimensions=dimensions, expected_rows=expected_rows)
    os.replace(temporary, target)
    result = {
        'schema': 1, 'status': 'complete', 'kind': 'glove.6B.300d', 'created_utc': now(),
        'source_url': GLOVE_URL if download else None,
        'source_page': GLOVE_PAGE, 'license_url': GLOVE_LICENSE,
        'input_kind': 'text' if text else 'zip', 'input': input_fp,
        'input_expected_sha256': expected_sha256, 'integrity_scope': integrity,
        'vectors': summary, 'files': manifest_files(output, [target]),
    }
    save(output / GLOVE_MANIFEST, result)
    return result

def validate_semantic_directory(directory):
    directory=Path(directory).resolve()
    needed=('config.json','config_sentence_transformers.json','sentence_bert_config.json',
            'modules.json','1_Pooling/config.json','vocab.txt','tokenizer.json',
            'tokenizer_config.json','special_tokens_map.json')
    if any(not (directory/name).is_file() for name in needed):raise ValueError('Incomplete local all-mpnet-base-v2 snapshot')
    config=read(directory/'config.json')
    if config.get('model_type')!='mpnet' or config.get('hidden_size')!=768:raise ValueError('Expected the DIGAT MPNet-base 768-dimensional architecture')
    modules=read(directory/'modules.json')
    allowed={'sentence_transformers.models.Transformer','sentence_transformers.models.Pooling','sentence_transformers.models.Normalize'}
    if not isinstance(modules,list) or not modules or any(m.get('type') not in allowed for m in modules):
        raise ValueError('Unexpected custom SentenceTransformer module')
    for item in modules:
        path=Path(item.get('path',''))
        resolved=(directory/path).resolve()
        if path.is_absolute() or (resolved!=directory and directory not in resolved.parents):raise ValueError('SentenceTransformer module path escaped snapshot')
    if not any((directory/name).is_file() for name in ('pytorch_model.bin','model.safetensors')):
        raise ValueError('Missing local MPNet model weights')
    for name in ('pytorch_model.bin','model.safetensors'):
        path=directory/name
        if path.is_file():
            if path.stat().st_size == 0:raise ValueError('Empty model weight file')
            with path.open('rb') as stream:header=stream.read(100)
            if header.startswith(b'version https://git-lfs.github.com/spec/'):
                raise ValueError('Git LFS pointer is not a downloaded model weight file')
    return modules


def _snapshot_fetch(directory,revision):
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=MODEL_ID,revision=revision,local_dir=str(directory),
                      local_dir_use_symlinks=False,allow_patterns=list(MODEL_PATTERNS),resume_download=True)


def prepare_semantic(output,*,source=None,download=False,revision=MODEL_REVISION,snapshot_fetcher=None):
    output=Path(output).resolve()
    if not re.fullmatch('[0-9a-f]{40}',revision):raise ValueError('Use an explicit full model commit SHA, not a mutable branch')
    if (output/SEMANTIC_MANIFEST).is_file():
        result=verify_asset_manifest(output,SEMANTIC_MANIFEST,'sentence-transformer')
        if result['repo_id']!=MODEL_ID or result['revision']!=revision:raise ValueError('Existing model revision differs')
        validate_semantic_directory(output);return result
    if output.exists():raise ValueError('Preserve existing unregistered model output; choose a new folder')
    if bool(source)==bool(download):raise ValueError('Choose one local --source or explicit --download')
    if source:
        source=Path(source).resolve();validate_semantic_directory(source)
        if source==output or source in output.parents or output in source.parents:raise ValueError('Use a separate snapshot output')
        output.mkdir(parents=True)
        for name in MODEL_PATTERNS+('model.safetensors',):
            path=source/name
            if path.is_file():
                dest=output/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    else:
        output.mkdir(parents=True);(snapshot_fetcher or _snapshot_fetch)(output,revision)
    modules=validate_semantic_directory(output)
    paths=[p for p in output.rglob('*') if p.is_file() and '.cache' not in p.relative_to(output).parts]
    result={'schema':1,'status':'complete','kind':'sentence-transformer','created_utc':now(),
            'repo_id':MODEL_ID,'revision':revision,'revision_evidence':'pinned Hub download' if download else 'user-declared local-source revision; content hashes independently recorded',
            'source_url':'https://huggingface.co/'+MODEL_ID+'/tree/'+revision,'license':'apache-2.0',
            'modules':modules,'files':manifest_files(output,paths)}
    save(output/SEMANTIC_MANIFEST,result);return result


def verify_semantic_snapshot(directory):
    result=verify_asset_manifest(directory,SEMANTIC_MANIFEST,'sentence-transformer')
    if result.get('repo_id')!=MODEL_ID or not re.fullmatch('[0-9a-f]{40}', result.get('revision','')):
        raise ValueError('Different semantic model repository or unpinned revision')
    validate_semantic_directory(directory);return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='asset', required=True)
    glove = sub.add_parser('glove')
    glove.add_argument('--output', type=Path, required=True)
    group = glove.add_mutually_exclusive_group()
    group.add_argument('--archive', type=Path)
    group.add_argument('--text', type=Path)
    group.add_argument('--download', action='store_true')
    glove.add_argument('--expected-sha256')
    glove.add_argument('--check-only', action='store_true')
    model = sub.add_parser('semantic')
    model.add_argument('--output', type=Path, required=True)
    group = model.add_mutually_exclusive_group()
    group.add_argument('--source', type=Path)
    group.add_argument('--download', action='store_true')
    model.add_argument('--revision', default=MODEL_REVISION)
    model.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.check_only:
        if args.download or (args.asset == 'glove' and (args.archive or args.text)) or (args.asset == 'semantic' and args.source):
            parser.error('--check-only validates an existing registered output; do not combine it with preparation inputs')
        result = (verify_asset_manifest(args.output, GLOVE_MANIFEST, 'glove.6B.300d')
                  if args.asset == 'glove' else verify_semantic_snapshot(args.output))
        if args.asset == 'glove' and args.expected_sha256 and result['input']['sha256'] != args.expected_sha256.lower():
            raise ValueError('Requested GloVe input checksum differs from the registered asset')
        if args.asset == 'semantic' and result['revision'] != args.revision:
            raise ValueError('Requested semantic revision differs from the existing snapshot')
    elif args.asset == 'glove':
        result = prepare_glove(args.output, archive=args.archive, text=args.text,
                               download=args.download, expected_sha256=args.expected_sha256)
    else:
        result = prepare_semantic(args.output, source=args.source, download=args.download, revision=args.revision)
    print(json.dumps({'status': result['status'], 'asset': result['kind'],
                      'output': str(args.output.resolve()), 'model_executed': False}), flush=True)


if __name__ == '__main__':
    main()



