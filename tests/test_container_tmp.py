"""Run the real rendered container launcher and inspect owned tmp cleanup and failure handling."""
import json
from pathlib import Path
import subprocess

import pytest
from test_vllm_schedule import render


@pytest.mark.parametrize('failure', [0, 42])
def test_private_tmp_bind_and_cleanup(tmp_path, monkeypatch, failure):
    root = tmp_path / 'temporary'
    root.mkdir()
    marker = root / 'unrelated'
    marker.write_text('preserve')
    monkeypatch.setenv('OELLM_EVAL_TMP_ROOT', str(root))
    script = render(tmp_path, monkeypatch, suite='evalchemy', container=True,
                    model_backend='vllm', data_parallel_size=4,
                    slurm_template_var=json.dumps({'SINGULARITY_ARGS': '--contain'}))
    monkeypatch.setenv('FAIL', str(failure))
    result = subprocess.run(['bash', str(script)], capture_output=True, text=True)
    assert result.returncode == failure, result.stderr
    argv = (tmp_path / 'argv').read_text().splitlines()
    binds = argv[argv.index('--bind') + 1].split(',')
    temporary = next(bind.split(':')[0] for bind in binds if bind.endswith(':/tmp'))
    assert temporary.startswith(str(root / 'oellm-eval-123.'))
    assert temporary + ':/var/tmp' in binds
    assert 'TMPDIR=/tmp' in argv
    assert not Path(temporary).exists()
    assert marker.read_text() == 'preserve'


def test_failed_evaluation_does_not_skip_the_rest(tmp_path, monkeypatch):
    script = render(tmp_path, monkeypatch, container=True, model_backend='vllm',
                    tasks=('piqa', 'mbpp', 'gsm8k'), queue_limit=1,
                    slurm_template_var=json.dumps({'SINGULARITY_ARGS': '--contain'}))
    assert 'NUM_JOBS=1' in script.read_text()
    monkeypatch.setenv('FAIL_TASK', 'mbpp')
    monkeypatch.setenv('READ_STDIN', '1')  # an evaluation reading stdin must not consume the CSV rows
    result = subprocess.run(['bash', str(script)], capture_output=True, text=True)
    assert result.returncode == 7, result.stderr
    argv = (tmp_path / 'argv').read_text().splitlines()
    assert [argv[i + 1] for i, a in enumerate(argv) if a == '--tasks'] == ['piqa', 'mbpp', 'gsm8k']
    # evaluate's metrics cache (MBPP code_eval) is private to this array task
    assert f"HF_METRICS_CACHE={tmp_path / 'cache'}/metrics/job_123" in argv
