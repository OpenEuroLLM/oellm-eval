import importlib.util
import json
from pathlib import Path
import sys

import pytest

CONTAINERS = Path(__file__).resolve().parents[1] / 'containers'
sys.path.insert(0, str(CONTAINERS))
import prepare_complete_runtime as complete
import oellm_ifeval as policy
from build_vllm_image import sha


def fixture(tmp_path):
    base, source = tmp_path/'base.sif', tmp_path/'source.json'
    base.write_bytes(b'base'); source.write_bytes(b'{}')
    root=tmp_path/'complete'; root.mkdir()
    files={}
    for i, target in enumerate(complete.TARGETS.values()):
        name=f'file{i}.py'; (root/name).write_text('pass\n')
        files[name]=dict(target=target,sha256=sha(root/name))
    manifest=dict(base_sha256=sha(base),source_manifest_sha256=sha(source),files=files)
    (root/'complete-manifest.json').write_text(json.dumps(manifest))
    return root,base,source,manifest


def test_accepts_exact_inputs_and_rejects_post_prepare_tampering(tmp_path):
    root,base,source,data=fixture(tmp_path)
    assert complete.verify(root,base,source)==data
    (root/'file0.py').write_text('changed')
    with pytest.raises(ValueError,match='mismatch'): complete.verify(root,base,source)


def test_rejects_different_base(tmp_path):
    root,base,source,data=fixture(tmp_path); base.write_bytes(b'other image')
    with pytest.raises(ValueError,match='inputs differ'): complete.verify(root,base,source)


@pytest.mark.parametrize('kind',['target','missing','traversal'])
def test_rejects_partial_or_escaping_manifest(tmp_path,kind):
    root,base,source,data=fixture(tmp_path)
    if kind=='target': data['files']['file0.py']['target']='/etc/passwd'
    elif kind=='missing': del data['files']['file0.py']
    else:
        (tmp_path/'escape.py').write_text('pass\n')
        data['files']['../escape.py']=data['files'].pop('file0.py')
    (root/'complete-manifest.json').write_text(json.dumps(data))
    with pytest.raises(ValueError): complete.verify(root,base,source)


def test_explicit_policy_requires_full_task_and_fixed_decoding():
    args=['--model','hf','--tasks','ifeval','--output_path','/run/out.json']
    assert policy.validate_args(args)[0]=='hf'
    for extra in [['--limit','5'],['--gen_kwargs','max_gen_toks=10'],['--samples=x'],['--predict_only']]:
        with pytest.raises(ValueError): policy.validate_args(args+extra)
    with pytest.raises(ValueError): policy.validate_args(['--model','vllm','--tasks','gsm8k','--output_path','x'])
    kwargs,stops=policy.hf_kwargs({'do_sample':False},['EOS'],'EOS')
    assert kwargs['eos_token_id'] is None and kwargs['forced_eos_token_id'] is None and stops==[]
    kwargs,stops,limit=policy.vllm_kwargs({'temperature':0},['EOS'],'EOS',1280)
    assert kwargs['ignore_eos'] and kwargs['skip_special_tokens'] and stops==[] and limit==1280
    with pytest.raises(ValueError): policy.vllm_kwargs({'temperature':1},['EOS'],'EOS',1280)
    with pytest.raises(ValueError): policy.vllm_kwargs({'temperature':0},['EOS'],'EOS',100)
    with pytest.raises(ValueError): policy.hf_kwargs({},['user stop'],'EOS')
