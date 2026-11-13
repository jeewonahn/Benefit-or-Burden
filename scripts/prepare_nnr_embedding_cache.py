"""Create a private, fingerprinted NNR torchtext cache from verified GloVe text.

This CPU-only preparation is explicit and never adopts an existing unregistered
cache. No model is trained or loaded. Incomplete output is preserved on failure.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import uuid

NAME='glove.6B.300d.txt'
RECEIPT='NNR_CACHE_MANIFEST.json'
MAX_VECTORS=10000000000
CACHE_NAME=NAME+'_'+str(MAX_VECTORS)+'.pt'


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()


def fingerprint(path):
    path=Path(path).resolve();return {'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)}


def verify(pin):
    path=Path(pin['path'])
    if not path.is_file()or path.stat().st_size!=pin['bytes']or sha(path)!=pin['sha256']:
        raise ValueError('Private embedding-cache fingerprint differs: '+str(path))


def validate(output,source):
    receipt=json.loads((output/RECEIPT).read_text(encoding='utf-8'))
    if receipt.get('status')!='complete'or receipt.get('schema')!=1 or receipt.get('source_text')!=fingerprint(source):
        raise ValueError('Private embedding cache source/receipt differs')
    expected={item['name']for item in receipt['files']}
    if {p.name for p in output.iterdir()if p.is_file()}!=expected|{RECEIPT}:
        raise ValueError('Unrecorded file in private embedding cache')
    for item in receipt['files']:verify({'path':str(output/item['name']),'bytes':item['bytes'],'sha256':item['sha256']})
    return receipt


def build(source,output,check_only=False,loader=None):
    source=Path(source).resolve();output=Path(output).resolve()
    if not source.is_file()or source.name!=NAME:raise ValueError('A registered GloVe 6B 300d text file is required')
    if output.exists():
        if not(output/RECEIPT).is_file():raise ValueError('Preserve incomplete/unregistered cache; select a new --nnr-cache-dir')
        return validate(output,source)
    if check_only:raise FileNotFoundError('Private NNR cache is not prepared; --execute prepares it on CPU')
    if output==source.parent or source.parent in output.parents or output in source.parents:
        raise ValueError('Private cache must not overlap its source asset directory')
    initial=fingerprint(source);output.mkdir(parents=True,exist_ok=False)
    copied=output/NAME;shutil.copyfile(source,copied)
    if sha(copied)!=initial['sha256']:raise ValueError('Source changed during private cache copy')
    if loader is None:
        from torchtext.vocab import GloVe
        loader=lambda root:GloVe(name='6B',dim=300,max_vectors=MAX_VECTORS,cache=str(root))
    vectors=loader(output)
    cache=output/CACHE_NAME
    if not cache.is_file():raise ValueError('torchtext did not commit the expected CPU cache')
    if fingerprint(source)!=initial or sha(copied)!=initial['sha256']:raise ValueError('GloVe source changed during cache preparation')
    files=[dict(name=p.name,bytes=p.stat().st_size,sha256=sha(p))for p in sorted(output.iterdir())if p.is_file()]
    if {v['name']for v in files}!={NAME,CACHE_NAME}:raise ValueError('Unexpected files from private cache preparation')
    receipt={'schema':1,'status':'complete','source_text':initial,'files':files,
        'construction':'torchtext.vocab.GloVe(name=6B, dim=300, max_vectors=10000000000) in a new isolated cache',
        'CPU_only':True,'GPU_test_performed':False,'model_loaded_or_trained':False}
    temp=output/(RECEIPT+'.tmp-'+uuid.uuid4().hex)
    with temp.open('x',encoding='utf-8',newline='\n')as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
    os.replace(temp,output/RECEIPT)
    return validate(output,source)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    result=build(args.source,args.output,args.check_only)
    print(json.dumps({'status':result['status'],'output':str(args.output),'CPU_only':True}))


if __name__=='__main__':main()
