import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'containers'))
from prepare_base_model_view import prepare, IDENTITY


def source(tmp_path,template=None):
    root=tmp_path/'source';root.mkdir()
    (root/'config.json').write_text('{}')
    (root/'tokenizer_config.json').write_text(json.dumps(dict(chat_template=template)))
    (root/'weights.safetensors').write_bytes(b'immutable test weights')
    return root


def test_self_contained_view_preserves_source_and_weight_identity(tmp_path):
    root=source(tmp_path);before=(root/'tokenizer_config.json').read_bytes()
    out=prepare(root,tmp_path/'view','identity')
    assert (root/'tokenizer_config.json').read_bytes()==before
    assert json.loads((out/'tokenizer_config.json').read_text())['chat_template']==IDENTITY
    assert (out/'weights.safetensors').stat().st_ino==(root/'weights.safetensors').stat().st_ino
    assert not any(p.is_symlink() for p in out.iterdir())
    with pytest.raises(FileExistsError):prepare(root,out,'identity')


def test_never_replaces_an_instruct_template(tmp_path):
    root=source(tmp_path,'existing instruct template')
    with pytest.raises(ValueError,match='chat protocol'):prepare(root,tmp_path/'view','identity')
    assert not (tmp_path/'view').exists()
