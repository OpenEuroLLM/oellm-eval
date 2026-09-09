"""Tests for batching lm-eval-harness tasks into one `lm_eval` invocation.

The sbatch template used to run one `lm_eval` per row of ``jobs.csv``, so a
model evaluated on N tasks was loaded from disk and onto the GPU N times. The
template now collapses adjacent rows that share a model, shot count and
lm_eval suite into a single invocation with a comma-separated ``--tasks``
list, and ``schedule_evals`` orders the CSV so those rows sit together.

Two properties matter and are checked here: the ordering has to keep a group
adjacent for the collapse to fire at all, and the collapse must not merge rows
whose ``--num_fewshot`` or suite differ, since both are per-invocation
arguments rather than per-task ones.
"""

import os
import re
import shutil
import subprocess
import sys
from importlib.resources import files
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from oellm.main import schedule_evals

MODEL_A = "/scratch/checkpoints/iter_0002000"
MODEL_B = "/scratch/checkpoints/iter_0004000"

SAMPLE_ROWS = [
    f"{MODEL_A},sib200_eus_Latn,0,lm-eval-harness",
    f"{MODEL_A},include_base_44_basque,0,lm-eval-harness",
    f"{MODEL_A},xnli_spa_Latn,0,lm-eval-harness",
    f"{MODEL_A},belebele_eus_Latn_cf,0,lighteval",
    f"{MODEL_A},mmlu,5,lm-eval-harness",
    f"{MODEL_B},sib200_eus_Latn,0,lm-eval-harness",
]

TEMPLATE_FIELDS = {
    "csv_path": "/tmp/jobs.csv",
    "max_array_len": 128,
    "array_limit": 0,
    "num_jobs": 1,
    "total_evals": len(SAMPLE_ROWS),
    "log_dir": "/tmp/logs",
    "evals_dir": "/tmp/results",
    "time_limit": "03:59:00",
    "slurm_mem": "96G",
    "limit": "",
    "venv_path": "",
    "lm_eval_include_path": "/tmp/tasks",
    "hf_hub_offline": 1,
    "additional_model_args": "batch_size=32",
    "evalchemy_dir": "/opt/evalchemy",
    "tasks_per_job": 8,
}

requires_awk = pytest.mark.skipif(shutil.which("awk") is None, reason="awk not available")


def _render(**overrides: object) -> str:
    template = (files("oellm.resources") / "template.sbatch").read_text()
    return template.format(**{**TEMPLATE_FIELDS, **overrides}).replace("\r\n", "\n")


def _awk_program() -> str:
    """The batching program, taken from the rendered script rather than duplicated."""
    match = re.search(r"^awk -F, .*?\n(.*?)^' \| \\$", _render(), re.S | re.M)
    assert match, "batching awk block not found in the rendered template"
    return match.group(1)


def _batches(rows: list[str], tasks_per_job: int = 8) -> list[dict[str, str]]:
    """Run the template's batching step over CSV rows, as the job would."""
    result = subprocess.run(
        [
            "awk",
            "-F,",
            "-v",
            f"max_tasks={tasks_per_job}",
            "-v",
            "OFS=\t",
            _awk_program(),
        ],
        input="\n".join(rows) + "\n",
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    batches = []
    for line in result.stdout.strip().split("\n"):
        model, tasks, n_shot, suite = line.split("\t")
        batches.append({"model": model, "tasks": tasks, "n_shot": n_shot, "suite": suite})
    return batches


@requires_awk
def test_tasks_sharing_a_model_become_one_invocation() -> None:
    batches = _batches(SAMPLE_ROWS)

    lm_eval_zero_shot = [
        b
        for b in batches
        if b["model"] == MODEL_A
        and b["suite"] == "lm-eval-harness"
        and b["n_shot"] == "0"
    ]
    assert len(lm_eval_zero_shot) == 1
    assert (
        lm_eval_zero_shot[0]["tasks"]
        == "sib200_eus_Latn,include_base_44_basque,xnli_spa_Latn"
    )


@requires_awk
def test_no_evaluation_is_lost_or_duplicated() -> None:
    batches = _batches(SAMPLE_ROWS)

    scheduled = [
        (b["model"], task, b["n_shot"]) for b in batches for task in b["tasks"].split(",")
    ]
    expected = [
        (row.split(",")[0], row.split(",")[1], row.split(",")[2]) for row in SAMPLE_ROWS
    ]

    assert sorted(scheduled) == sorted(expected)


@requires_awk
def test_shot_count_suite_and_model_are_batch_boundaries() -> None:
    """--num_fewshot and the suite are per-invocation, so they cannot be merged."""
    batches = _batches(SAMPLE_ROWS)

    by_key = {(b["model"], b["n_shot"], b["suite"]): b["tasks"] for b in batches}
    assert by_key[(MODEL_A, "5", "lm-eval-harness")] == "mmlu"
    assert by_key[(MODEL_A, "0", "lighteval")] == "belebele_eus_Latn_cf"
    assert by_key[(MODEL_B, "0", "lm-eval-harness")] == "sib200_eus_Latn"


@requires_awk
def test_cap_bounds_the_batch_size() -> None:
    for tasks_per_job in (1, 2, 3):
        batches = _batches(SAMPLE_ROWS, tasks_per_job=tasks_per_job)
        assert all(len(b["tasks"].split(",")) <= tasks_per_job for b in batches)


@requires_awk
def test_cap_of_one_keeps_an_invocation_per_task() -> None:
    """The escape hatch: --tasks_per_job 1 is the pre-batching behaviour."""
    batches = _batches(SAMPLE_ROWS, tasks_per_job=1)

    assert len(batches) == len(SAMPLE_ROWS)


def test_scheduled_csv_keeps_a_batch_adjacent(tmp_path: Path) -> None:
    """The collapse only fires on adjacent rows, so ordering has to preserve groups."""
    with (
        patch("oellm.main._load_cluster_env"),
        patch("oellm.main._num_jobs_in_queue", return_value=0),
        patch.dict(os.environ, {"EVAL_OUTPUT_DIR": str(tmp_path)}),
    ):
        schedule_evals(
            models="EleutherAI/pythia-70m,EleutherAI/pythia-160m",
            task_groups="sib200-eu",
            n_shot=0,
            skip_checks=True,
            venv_path=str(Path(sys.prefix)),
            dry_run=True,
        )

    jobs_csv = next(iter(tmp_path.glob("**/jobs.csv")))
    df = pd.read_csv(jobs_csv)
    assert df["model_path"].nunique() > 1

    keys = list(zip(df["model_path"], df["n_shot"], df["eval_suite"], strict=True))
    runs = [key for index, key in enumerate(keys) if index == 0 or key != keys[index - 1]]
    assert len(runs) == len(set(runs)), "rows sharing a batch key are not adjacent"
