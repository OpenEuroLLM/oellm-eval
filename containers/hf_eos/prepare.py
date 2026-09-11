"""Prepare an isolated HF EOS fallback without changing configured EOS lists."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

FALLBACK='''        # Some checkpoint exports omit EOS from generation_config.json even
        # though the tokenizer defines it. Text stopping is batch-wide and
        # decoding removes special tokens, so it cannot enforce per-row EOS.
        if (
            "eos_token_id" not in generation_kwargs
            and self.model.generation_config.eos_token_id is None
            and self.eot_token_id is not None
        ):
            generation_kwargs["eos_token_id"] = self.eot_token_id

'''


def prepare(source, out):
    text=source.read_text();marker='        # build stopping criteria\n'
    if text.count(marker)!=1:raise ValueError('Unexpected HF generation implementation')
    if 'Some checkpoint exports omit EOS' in text:raise ValueError('Already patched')
    updated=text.replace(marker,FALLBACK+marker,1)
    ast.parse(updated)
    with out.open('x') as stream:stream.write(updated)
    out.with_suffix('.manifest.json').write_text(json.dumps(dict(source=str(source),before=hashlib.sha256(text.encode()).hexdigest(),after=hashlib.sha256(updated.encode()).hexdigest()),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','out']:p.add_argument('--'+name,type=Path,required=True)
    prepare(**vars(p.parse_args()))
