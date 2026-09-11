import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'containers'))
import bootstrap_jupiter_base as base
import check_reference_results as results


def test_base_plan_pins_public_inputs_without_downloading(tmp_path):
    cmd = [sys.executable, str(ROOT/'containers/bootstrap_jupiter_base.py'),
           '--work', str(tmp_path/'new'), '--tmp-dir', '/tmp/oellm-bootstrap-test']
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    plan = json.loads(result.stdout)
    assert base.OCI in plan['definition']
    assert base.WHEEL_SHA in plan['definition']
    assert 'ray.get(ping.remote())' in plan['definition']
    assert not (tmp_path/'new').exists()


def test_base_rejects_shared_build_tmp_before_download(tmp_path):
    cmd = [sys.executable, str(ROOT/'containers/bootstrap_jupiter_base.py'),
           '--work', str(tmp_path/'new'), '--tmp-dir', '/e/fscratch/build']
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode != 0 and 'local /tmp' in result.stderr
    assert not (tmp_path/'new').exists()


def test_new_base_profile_is_used_without_changing_shared_defaults(tmp_path):
    profile = json.loads((ROOT/'containers/vllm_profiles.json').read_text())
    profile['jupiter']['base_sha256'] = 'a' * 64
    custom = tmp_path/'profiles.json'
    custom.write_text(json.dumps(profile))
    cmd = [sys.executable, str(ROOT/'containers/build_vllm_image.py'), '--machine', 'jupiter',
           '--profiles', str(custom), '--tmp-dir', '/tmp/oellm-bootstrap-test']
    for name in ['base-image','runtime','evalchemy','humaneval','source-manifest','output']:
        cmd += ['--'+name, str(tmp_path/name)]
    plan = json.loads(subprocess.check_output(cmd, text=True))
    assert plan['base']['base_sha256'] == 'a' * 64
    assert ('oellm.base.sha256 ' + 'a' * 64) in plan['definition']
    assert json.loads((ROOT/'containers/vllm_profiles.json').read_text())['jupiter']['base_sha256'] != 'a' * 64


def test_reference_plan_needs_no_hf_packages_or_downloads(tmp_path):
    cmd = [sys.executable, str(ROOT/'containers/prepare_jupiter_reference.py'),
           '--model-dir', str(tmp_path/'model'), '--hf-home', str(tmp_path/'cache'),
           '--manifest', str(tmp_path/'manifest.json')]
    plan = json.loads(subprocess.check_output(cmd, text=True))
    assert len(plan['model_revision']) == 40
    assert plan['tasks'] == ['piqa','gsm8k','MATH500']
    assert not list(tmp_path.iterdir())


def test_conda_lock_contains_only_explicit_public_hashed_packages():
    lines = (ROOT/'containers/locks/jupiter-control-linux-aarch64.explicit').read_text().splitlines()
    assert '@EXPLICIT' in lines
    packages = [s for s in lines if s and not s.startswith(('#','@'))]
    assert len(packages) >= 20
    for s in packages:
        assert s.startswith('https://conda.anaconda.org/conda-forge/')
        assert len(s.rsplit('#',1)[1]) == 32


def test_result_check_rejects_duplicate_ids_despite_correct_effective_count(tmp_path):
    tmp_path = tmp_path/'results'
    tmp_path.mkdir()
    data = {'config':{'limit':None}, 'results':{'piqa':{'acc_norm,none':0.5}},
            'n-samples':{'piqa':{'effective':1838}}}
    (tmp_path/'result.json').write_text(json.dumps(data))
    with (tmp_path/'samples_piqa_test.jsonl').open('w') as f:
        for i in range(1838):
            f.write(json.dumps({'doc_id':i if i < 1837 else 0})+'\n')
    with pytest.raises(ValueError, match='sample IDs'):
        results.check(tmp_path, 'piqa')


def test_result_check_requires_every_gpqa_repeat_and_ignores_its_own_summary(tmp_path):
    tmp_path = tmp_path/'results'
    tmp_path.mkdir()
    examples = [{'Question':str(i), 'model_outputs':['A']*3} for i in range(198)]
    data = {'config':{'limit':None,'max_tokens':'default'},
            'results':{'GPQADiamond':{'examples':examples,'num_repeat':3,'accuracy_avg':0.25}}}
    path = tmp_path/'result.json'
    path.write_text(json.dumps(data))
    summary = results.check(tmp_path, 'GPQADiamond')
    (tmp_path/'checked.json').write_text(json.dumps([summary]))
    assert results.check(tmp_path, 'GPQADiamond')['examples'] == 198
    examples[-1]['model_outputs'].pop()
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='repeats'):
        results.check(tmp_path, 'GPQADiamond')


def test_humaneval_checker_requires_both_languages_and_reproduces_verdicts(tmp_path):
    root = tmp_path/'results'; root.mkdir()
    data = {'config':{'limit':None,'max_tokens':'default'},
            'results':{'HumanEval':{'python_pass@1':1.0,'sh_pass@1':1.0}}}
    (root/'result.json').write_text(json.dumps(data))
    for language, count in [('python',164),('sh',158)]:
        rows=[{'task_id':f'{language}/{i}','passed':True} for i in range(count)]
        (root/f'generated_{language}.jsonl.graded.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    assert results.check(root,'HumanEval')['examples']==322
    artifact=root/'generated_sh.jsonl.graded.jsonl'
    valid=artifact.read_text();artifact.unlink()
    with pytest.raises(ValueError,match='missing'):results.check(root,'HumanEval')
    artifact.write_text(valid.replace('"passed": true','"passed": "true"',1))
    with pytest.raises(ValueError,match='nonboolean'):results.check(root,'HumanEval')
    artifact.write_text(valid.replace('"passed": true','"passed": false',1))
    with pytest.raises(ValueError,match='mismatch'):results.check(root,'HumanEval')
