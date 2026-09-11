"""Prepare the validated HumanEval/LiveCodeBench source without replacing an image."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from prepare_humaneval_fix import prepare as human
from prepare_lcb_cache_fix import prepare as cache
from prepare_lcb_grading_fix import prepare as grading


def prepare(source,out):
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='code-grading-',dir=out.parent) as tmp:
        tmp=Path(tmp)
        human(source,tmp/'human')
        cache(tmp/'human',tmp/'cache')
        grading(tmp/'cache',out)
    files={}
    for manifest in ['humaneval_fix_manifest.json','lcb_grading_manifest.json']:
        files.update(json.loads((out/manifest).read_text())['after'])
    files={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in files}
    recipe={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    (out/'code_grading_manifest.json').write_text(json.dumps(dict(source=str(source),files=files,recipe=recipe),indent=2)+'\n')
    print(out/'code_grading_manifest.json')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','out']:p.add_argument('--'+name,type=Path,required=True)
    prepare(**vars(p.parse_args()))
