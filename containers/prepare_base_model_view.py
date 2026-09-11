"""Create a self-contained explicit plain-completion view of a base HF export.

This reproduces the established identity-template prompt view. Metadata is
copied; immutable weights are hard-linked on the same filesystem, so container
mounts never follow external symlinks. Never use this to replace an instruct
model's chat protocol. The source export is not modified.
"""
import argparse
import json
import os
from pathlib import Path
import shutil

IDENTITY="{% for message in messages %}{{ message['content'] }}{% endfor %}"


def prepare(source, out, template):
    if template != 'identity':raise ValueError('Explicit identity template selection is required')
    source=source.resolve();out=out.absolute()
    cfg=json.loads((source/'tokenizer_config.json').read_text())
    if cfg.get('chat_template') not in (None,IDENTITY) or (source/'chat_template.jinja').exists():
        raise ValueError('Refusing to replace an existing chat protocol')
    if not (source/'config.json').is_file():raise ValueError('Missing model configuration')
    if out.exists():raise FileExistsError(out)
    weights=[p for p in source.iterdir() if p.name.endswith(('.safetensors','.bin')) and p.is_file()]
    if not weights:raise ValueError('No model weights found')
    out.parent.mkdir(parents=True,exist_ok=True)
    if any(p.resolve().stat().st_dev != out.parent.stat().st_dev for p in weights):
        raise ValueError('Choose an output on the weights filesystem; automatic large copies are disabled')
    out.mkdir()
    files={}
    for path in source.iterdir():
        if not path.is_file() or path.name=='tokenizer_config.json':continue
        if path in weights:
            os.link(path.resolve(),out/path.name)
            stat=path.resolve().stat()
            files[path.name]=dict(source=str(path.resolve()),bytes=stat.st_size,device=stat.st_dev,inode=stat.st_ino,mode='hardlink')
        else:
            shutil.copyfile(path,out/path.name)
            files[path.name]=dict(source=str(path.resolve()),mode='copy')
    cfg['chat_template']=IDENTITY
    (out/'tokenizer_config.json').write_text(json.dumps(cfg,indent=2,ensure_ascii=False)+'\n')
    (out/'prompt-view-manifest.json').write_text(json.dumps(dict(source=str(source),template='identity',files=files),indent=2)+'\n')
    assert not any(p.is_symlink() for p in out.iterdir())
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--template',choices=['identity'],required=True)
    print(prepare(**vars(p.parse_args())))
