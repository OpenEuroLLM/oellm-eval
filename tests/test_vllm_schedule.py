"""Execute rendered jobs with recording launchers; no GPU or scheduler needed."""

import json
import os
import subprocess
from unittest.mock import patch

import pytest

from oellm.main import schedule_evals
from oellm.utils import _ensure_singularity_image


def render(tmp_path, monkeypatch, *, suite="lm_eval", container=False, internal_evalchemy=False, **options):
    for key in ("WORLD_SIZE", "LMEVAL_DP", "NODES", "GPUS_PER_NODE"):
        monkeypatch.delenv(key, raising=False)
    env = {
        "EVAL_OUTPUT_DIR": str(tmp_path / "output"),
        "EVAL_BASE_DIR": str(tmp_path),
        "EVALCHEMY_DIR": "/opt/packed-evalchemy" if internal_evalchemy else str(tmp_path),
        "EVAL_CONTAINER_IMAGE": "/shared/images/evaluation.sif",
        "HF_HOME": str(tmp_path / "cache"),
        "GPUS_PER_NODE": "1",
        "QUEUE_LIMIT": "10",
        "SLURM_ARRAY_TASK_ID": "0",
        "SLURM_JOB_ID": "123",
        "CUDA_VISIBLE_DEVICES": "GPU-uuid-0,GPU-uuid-1,GPU-uuid-2,GPU-uuid-3",
        "RECORD": str(tmp_path / "argv"),
        "SINGULARITY_ARGS": "",
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    recorder = '#!/bin/bash\nprintf "%s\\n" "$CUDA_VISIBLE_DEVICES" "$@" >> "$RECORD"\nexit "${FAIL:-0}"\n'
    for name in ("python", "accelerate", "singularity"):
        p = bindir / name
        p.write_text(recorder)
        p.chmod(0o755)
    (bindir / "activate").write_text(f'export PATH="{bindir}:$PATH"\n')
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    csv = tmp_path / "input.csv"
    csv.write_text(f"model_path,task_path,n_shot,eval_suite\n/model with spaces,piqa,10,{suite}\n")
    with patch("oellm.main._load_cluster_env"), patch("oellm.main._num_jobs_in_queue", return_value=0):
        schedule_evals(
            eval_csv_path=str(csv), skip_checks=True, dry_run=True,
            venv_path=None if container else str(tmp_path), **options,
        )
    return next(tmp_path.glob("output/**/submit_evals.sbatch"))


@pytest.mark.parametrize("suite", ["lm_eval", "evalchemy"])
@pytest.mark.parametrize("container", [False, True])
@pytest.mark.parametrize("dp,tp", [(1, 1), (4, 1), (2, 2)])
def test_native_launch(tmp_path, monkeypatch, suite, container, dp, tp):
    script = render(tmp_path, monkeypatch, suite=suite, container=container,
                    model_backend="vllm", data_parallel_size=dp, tensor_parallel_size=tp,
                    model_args="dtype=bfloat16", log_samples=True)
    assert f"#SBATCH --gres=gpu:{dp * tp}" in script.read_text()
    assert "#SBATCH --nodes=1" in script.read_text()
    subprocess.run(["bash", "-n", str(script)], check=True)
    subprocess.run(["bash", str(script)], check=True, capture_output=True)
    argv = (tmp_path / "argv").read_text().splitlines()
    assert argv.count("--model") == 1
    assert argv[argv.index("--model") + 1] == "vllm"
    assert argv[argv.index("--model_args") + 1] == (
        f"pretrained=/model with spaces,trust_remote_code=True,dtype=bfloat16,tensor_parallel_size={tp},data_parallel_size={dp}"
    )
    assert "launch" not in argv and "torch.distributed.run" not in argv
    assert argv[0] == os.environ["CUDA_VISIBLE_DEVICES"]
    assert "--limit" not in argv
    if suite == "lm_eval":
        assert argv[argv.index("--num_fewshot") + 1] == "10"
        assert "--log_samples" in argv


@pytest.mark.parametrize("suite", ["lm_eval", "evalchemy"])
def test_backend_failure_is_job_failure(tmp_path, monkeypatch, suite):
    script = render(tmp_path, monkeypatch, suite=suite, model_backend="vllm")
    monkeypatch.setenv("FAIL", "42")
    assert subprocess.run(["bash", str(script)], capture_output=True).returncode == 42


@pytest.mark.parametrize("variable", ["WORLD_SIZE", "LMEVAL_DP"])
def test_nested_launcher_refused(tmp_path, monkeypatch, variable):
    script = render(tmp_path, monkeypatch, model_backend="vllm", data_parallel_size=4)
    monkeypatch.setenv(variable, "4")
    assert subprocess.run(["bash", str(script)], capture_output=True).returncode != 0
    assert not (tmp_path / "argv").exists()


@pytest.mark.parametrize("options", [
    {"model_backend": "invalid"},
    {"data_parallel_size": 4},
    {"model_backend": "vllm", "data_parallel_size": 0},
    {"model_backend": "vllm", "tensor_parallel_size": -1},
    {"model_args": "data_parallel_size=4"},
    {"model_args": "pretrained=other"},
    {"model_backend": "vllm", "slurm_template_var": json.dumps({"NODES": 2})},
    {"model_backend": "vllm", "data_parallel_size": 4,
     "slurm_template_var": json.dumps({"GPUS_PER_NODE": 1})},
])
def test_invalid_configuration(tmp_path, monkeypatch, options):
    with pytest.raises(ValueError):
        render(tmp_path, monkeypatch, **options)


@pytest.mark.parametrize("suite", ["lm_eval", "evalchemy"])
def test_hf_default_preserved(tmp_path, monkeypatch, suite):
    script = render(tmp_path, monkeypatch, suite=suite)
    subprocess.run(["bash", str(script)], check=True, capture_output=True)
    argv = (tmp_path / "argv").read_text().splitlines()
    assert argv[argv.index("--model") + 1] == "hf"
    assert ("launch" in argv) == (suite == "evalchemy")


def test_image_internal_evalchemy_and_absolute_image(tmp_path, monkeypatch):
    script = render(tmp_path, monkeypatch, suite="evalchemy", container=True,
                    internal_evalchemy=True, model_backend="vllm", data_parallel_size=4)
    subprocess.run(["bash", str(script)], check=True, capture_output=True)
    argv = (tmp_path / "argv").read_text().splitlines()
    assert argv[argv.index("--pwd") + 1] == "/opt/packed-evalchemy"
    assert "/shared/images/evaluation.sif" in argv
    assert not any("/opt/packed-evalchemy:" in arg for arg in argv)


def test_explicit_local_image_never_downloads(tmp_path, monkeypatch):
    monkeypatch.setenv("EVAL_BASE_DIR", str(tmp_path))
    image = tmp_path / "pinned.sif"
    image.write_bytes(b"fixture")
    with patch("huggingface_hub.hf_hub_download") as download:
        _ensure_singularity_image(str(image))
        with pytest.raises(RuntimeError, match="does not exist"):
            _ensure_singularity_image(str(tmp_path / "missing.sif"))
        download.assert_not_called()
