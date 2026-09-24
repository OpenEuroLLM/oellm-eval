"""Scheduling inputs: CLI flags, task groups and the login-shell environment."""

import csv
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from oellm.main import schedule_evals
from oellm.task_groups import _collect_dataset_specs, _expand_task_groups


def _schedule(tmp_path, env=None, **kw):
    """Dry-run schedule; return the rendered job script and jobs.csv rows."""
    out = tmp_path / "out"
    with (
        patch("oellm.scheduler._load_cluster_env"),
        patch("oellm.scheduler._num_jobs_in_queue", return_value=0),
        patch.dict(os.environ, {"EVAL_OUTPUT_DIR": str(out), **(env or {})}),
    ):
        schedule_evals(
            models="EleutherAI/pythia-70m", dry_run=True, skip_checks=True, **kw
        )
    run = next(out.iterdir())
    with open(run / "jobs.csv") as f:
        rows = list(csv.DictReader(f))
    return (run / "submit_evals.sbatch").read_text(), rows


def test_exported_job_variables_do_not_override_flags(tmp_path):
    env = {"VENV_PATH": "/stray/venv", "LIMIT": "100", "MODEL_DIR": "/stray/models"}
    sbatch, _ = _schedule(
        tmp_path, env=env, tasks="copa", n_shot=[0], venv_path="/my/venv", limit=5
    )
    assert "/stray" not in sbatch
    assert 'VENV_PATH="/my/venv"' in sbatch
    assert 'source "$VENV_PATH/bin/activate"' in sbatch
    assert 'export LIMIT="5"' in sbatch
    assert "${LIMIT:+--limit $LIMIT}" in sbatch


def test_tasks_and_groups_are_both_scheduled(tmp_path):
    _, rows = _schedule(
        tmp_path, tasks="gsm8k,hellaswag", n_shot=[10], task_groups="open-sci-0.01"
    )
    tasks = [(r["task_path"], r["n_shot"]) for r in rows]
    assert ("gsm8k", "10") in tasks
    assert tasks.count(("hellaswag", "10")) == 1  # already in the group
    assert len(rows) == len(_expand_task_groups(["open-sci-0.01"])) + 1


def test_n_shot_with_groups_only_is_refused(tmp_path):
    with pytest.raises(ValueError, match="n_shot applies to tasks only"):
        _schedule(tmp_path, n_shot=[0], task_groups="open-sci-0.01")


def test_shared_tasks_keep_every_groups_shots():
    # hellaswag: 10-shot in open-sci-0.01, 0- and 10-shot in dclm-core-22.
    both = _expand_task_groups(["open-sci-0.01", "dclm-core-22"])
    assert {r.n_shot for r in both if r.task == "hellaswag"} == {0, 10}
    alone = _expand_task_groups(["open-sci-0.01"])
    assert {r.n_shot for r in alone if r.task == "hellaswag"} == {10}


def test_tasks_next_to_groups_get_their_data_staged(tmp_path, monkeypatch):
    staged = []
    monkeypatch.setattr(
        "oellm.envcheck.check_scheduled_environment", lambda *a, **k: None
    )
    with (
        patch("oellm.scheduler._load_cluster_env"),
        patch("oellm.scheduler._num_jobs_in_queue", return_value=0),
        patch("oellm.scheduler._ensure_runtime_environment"),
        patch("oellm.scheduler._process_model_paths", return_value={}),
        patch("oellm.scheduler._probe_engine_versions", return_value={}),
        patch(
            "oellm.scheduler._pre_download_datasets_from_specs",
            side_effect=lambda specs, **_: staged.extend(specs),
        ),
        patch.dict(os.environ, {"EVAL_OUTPUT_DIR": str(tmp_path / "out")}),
    ):
        schedule_evals(
            models="EleutherAI/pythia-70m",
            tasks="xcopa",
            n_shot=[0],
            task_groups="open-sci-0.01",
            dry_run=True,
            venv_path=str(tmp_path / "venv"),
        )
    group_repos = {s.repo_id for s in _collect_dataset_specs(["open-sci-0.01"])}
    assert {s.repo_id for s in staged} == group_repos | {"xcopa"}


def test_the_job_uses_the_cache_the_pre_download_filled(tmp_path):
    sbatch, _ = _schedule(tmp_path, env={"HF_HOME": "/hf"}, tasks="copa", n_shot=[0])
    line = next(
        ln for ln in sbatch.splitlines() if ln.startswith("export HF_DATASETS_CACHE=")
    )
    for exported, expected in (("/scratch/ds", "/scratch/ds"), (None, "/hf/datasets")):
        env = {
            "PATH": os.environ["PATH"],
            **({"HF_DATASETS_CACHE": exported} if exported else {}),
        }
        out = subprocess.run(
            ["bash", "-c", f'{line}; echo "$HF_DATASETS_CACHE"'],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert out.stdout.strip() == expected


def test_local_run_defaults_hf_home(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    with patch.dict(os.environ, {"EVAL_OUTPUT_DIR": str(tmp_path)}):
        schedule_evals(
            models="EleutherAI/pythia-70m",
            tasks="copa",
            n_shot=[0],
            local=True,
            venv_path=str(tmp_path / "venv"),
            dry_run=True,
            skip_checks=True,
        )
        assert os.environ["HF_HOME"] == str(Path.home() / ".cache" / "huggingface")
