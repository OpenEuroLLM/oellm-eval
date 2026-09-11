"""Preserve release_v2 while making its historical default cache name explicit."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

EXPECTED='990ee2b048cbedee7575d882f439bc9ec9dcd00cca1bbcbfeb272a86e79169fd'
REL='eval/chat_benchmarks/LiveCodeBench/eval_instruct.py'

def prepare(source,out):
    original=source/REL
    digest=hashlib.sha256(original.read_bytes()).hexdigest()
    if digest!=EXPECTED:
        raise ValueError('Unreviewed LiveCodeBench source')
    text=original.read_text()
    before='"livecodebench/code_generation_lite",\n            version_tag="release_v2",'
    if text.count(before)!=1:
        raise ValueError('Dataset anchor differs')
    text=text.replace(before,'"livecodebench/code_generation_lite",\n            name="release_latest",  # Original dataset script DEFAULT_CONFIG_NAME\n            version_tag="release_v2",',1)
    shutil.copytree(source,out,ignore=shutil.ignore_patterns('.git','__pycache__'))
    (out/REL).write_text(text)
    (out/'lcb_cache_manifest.json').write_text(json.dumps({'source':str(original),'before_sha256':digest,'after_sha256':hashlib.sha256(text.encode()).hexdigest(),'release':'release_v2','original_script_default_config':'release_latest'},indent=2)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','out'):parser.add_argument('--'+name,type=Path,required=True)
    prepare(**vars(parser.parse_args()))
