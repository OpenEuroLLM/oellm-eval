"""The generated job script must not report a failed evaluation as finished.

Every eval is a subprocess whose exit status the script used to ignore, so a
container that fails to start, a missing venv or a crashed harness all ended
with "Evaluation finished" and a zero exit code. The array element was then
COMPLETED with no result file, and only a later `collect --check` noticed.

These tests render a real script through `schedule_evals` and run it against a
stub harness, so they exercise the template rather than a copy of its logic.
In ELLIOT the script is built by ``oellm.scheduler``, so that is what is patched.
"""

import os
import stat
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from oellm.main import schedule_evals

# Stub `python` implementations placed on PATH by the fake venv below.
HARNESS_OK = """#!/bin/bash
# lm_eval creates the results directory itself and appends a timestamp to the
# path it is given; mimic both so the script's result check sees a real file.
for arg in "$@"; do
    if [ "$prev" = "--output_path" ]; then
        mkdir -p "$(dirname "$arg")"
        touch "${arg%.json}_2026.json"
    fi
    prev="$arg"
done
exit 0
"""
HARNESS_CRASHES = """#!/bin/bash
echo "boom" >&2
exit 1
"""
HARNESS_SILENT = """#!/bin/bash
exit 0
"""


def _fake_venv(tmp_path: Path, harness: str) -> Path:
    """A venv whose `activate` puts a stub `python` ahead of the real one."""
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "activate").write_text(f'export PATH="{venv}/bin:$PATH"\n')
    python = venv / "bin" / "python"
    python.write_text(harness)
    python.chmod(python.stat().st_mode | stat.S_IEXEC)
    return venv


def _render(tmp_path: Path, venv: Path) -> Path:
    with (
        patch("oellm.scheduler._load_cluster_env"),
        patch("oellm.scheduler._num_jobs_in_queue", return_value=0),
        patch.dict(os.environ, {"EVAL_OUTPUT_DIR": str(tmp_path), "GPUS_PER_NODE": "1"}),
    ):
        schedule_evals(
            models="EleutherAI/pythia-70m",
            tasks="hellaswag",
            n_shot=0,
            skip_checks=True,
            venv_path=str(venv),
            dry_run=True,
        )
    scripts = list(tmp_path.glob("**/submit_evals.sbatch"))
    assert len(scripts) == 1
    return scripts[0]


def _run(script: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(script)],
        env={
            **os.environ,
            "SLURM_ARRAY_TASK_ID": "0",
            "SLURM_ARRAY_JOB_ID": "1",
            "SLURM_JOB_ID": "1",
        },
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "harness, expected_error",
    [
        pytest.param(HARNESS_CRASHES, "exited with status 1", id="harness-crashes"),
        pytest.param(HARNESS_SILENT, "wrote no result", id="harness-writes-nothing"),
    ],
)
def test_failed_evaluation_fails_the_job(tmp_path, harness, expected_error):
    """A crash and a silent no-op are both failures, not finished evaluations."""
    script = _render(tmp_path, _fake_venv(tmp_path, harness))
    result = _run(script)

    assert result.returncode == 1
    assert expected_error in result.stdout
    assert "1 failed evaluation" in result.stdout
    assert "Evaluation finished" not in result.stdout


def test_successful_evaluation_passes(tmp_path):
    """The happy path still exits 0 and reports the evaluation as finished."""
    script = _render(tmp_path, _fake_venv(tmp_path, HARNESS_OK))
    result = _run(script)

    assert result.returncode == 0
    assert "Evaluation finished" in result.stdout
    assert "[error]" not in result.stdout


def test_failure_count_survives_the_read_loop(tmp_path):
    """Both rows of a two-eval job are counted.

    The loop reads the CSV by redirection rather than a pipe; a pipeline would
    run the body in a subshell and lose the counter at `done`.
    """
    venv = _fake_venv(tmp_path, HARNESS_CRASHES)
    # ELLIOT sizes the array from QUEUE_LIMIT, so this puts both rows in one job.
    with (
        patch("oellm.scheduler._load_cluster_env"),
        patch("oellm.scheduler._num_jobs_in_queue", return_value=0),
        patch.dict(
            os.environ,
            {"EVAL_OUTPUT_DIR": str(tmp_path), "GPUS_PER_NODE": "1", "QUEUE_LIMIT": "1"},
        ),
    ):
        schedule_evals(
            models="EleutherAI/pythia-70m",
            tasks="hellaswag,arc_easy",
            n_shot=0,
            skip_checks=True,
            venv_path=str(venv),
            dry_run=True,
        )
    script = next(iter(tmp_path.glob("**/submit_evals.sbatch")))
    result = _run(script)

    assert result.returncode == 1
    assert "2 failed evaluation" in result.stdout


def test_sbatch_failure_exits_non_zero(tmp_path):
    """A rejected submission must not look like a successful schedule."""
    error = subprocess.CalledProcessError(1, ["sbatch"], stderr="AssocMaxSubmitJobLimit")
    with (
        patch("oellm.scheduler._load_cluster_env"),
        patch("oellm.scheduler._num_jobs_in_queue", return_value=0),
        patch("oellm.scheduler.subprocess.run", side_effect=error),
        patch.dict(os.environ, {"EVAL_OUTPUT_DIR": str(tmp_path)}),
        pytest.raises(SystemExit) as excinfo,
    ):
        schedule_evals(
            models="EleutherAI/pythia-70m",
            tasks="hellaswag",
            n_shot=0,
            skip_checks=True,
            venv_path=str(Path(sys.prefix)),
        )
    assert excinfo.value.code == 1
