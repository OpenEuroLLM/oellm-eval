"""Tests for the eval_command() CLI subcommand."""

from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from oellm.main import eval_command, schedule_evals


def _write_yaml(path: Path, data: dict) -> str:
    path.write_text(yaml.dump(data))
    return str(path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_eval(tmp_path, **kwargs):
    """Run eval_command with standard dry-run patches."""
    with (
        patch("oellm.scheduler._load_cluster_env"),
        patch("oellm.scheduler._num_jobs_in_queue", return_value=0),
        patch("oellm.runner.detect_lmms_model_type", return_value="llava"),
        patch.dict(os.environ, {"EVAL_OUTPUT_DIR": str(tmp_path)}),
    ):
        eval_command(**kwargs)


# ---------------------------------------------------------------------------
# Basic invocation
# ---------------------------------------------------------------------------


def test_eval_without_config(tmp_path: Path) -> None:
    _run_eval(
        tmp_path,
        models="EleutherAI/pythia-70m",
        task_groups="open-sci-0.01",
        skip_checks=True,
        venv_path=str(Path(sys.prefix)),
        dry_run=True,
    )


def test_eval_with_config_yaml(tmp_path: Path) -> None:
    cfg = _write_yaml(
        tmp_path / "eval.yaml",
        {
            "models": ["EleutherAI/pythia-70m"],
            "task_groups": ["open-sci-0.01"],
        },
    )
    _run_eval(
        tmp_path,
        config=cfg,
        skip_checks=True,
        venv_path=str(Path(sys.prefix)),
        dry_run=True,
    )


# ---------------------------------------------------------------------------
# CLI overrides YAML
# ---------------------------------------------------------------------------


def test_eval_cli_overrides_yaml_model(tmp_path: Path) -> None:
    cfg = _write_yaml(
        tmp_path / "eval.yaml",
        {
            "models": ["yaml-model"],
            "task_groups": ["open-sci-0.01"],
        },
    )
    with patch("oellm.scheduler.schedule_evals") as mock_sched:
        with (
            patch("oellm.scheduler._load_cluster_env"),
            patch.dict(os.environ, {"EVAL_OUTPUT_DIR": str(tmp_path)}),
        ):
            eval_command(
                config=cfg,
                models="cli-model",
                skip_checks=True,
                venv_path=str(Path(sys.prefix)),
                dry_run=True,
            )
    assert mock_sched.called
    called_models = mock_sched.call_args.kwargs.get("models") or mock_sched.call_args[
        1
    ].get("models")
    assert called_models == "cli-model"


# ---------------------------------------------------------------------------
# ModelConfig in YAML
# ---------------------------------------------------------------------------


def test_eval_model_config_path_extracted(tmp_path: Path) -> None:
    cfg = _write_yaml(
        tmp_path / "eval.yaml",
        {
            "models": [{"path": "/hpc/models/llava", "name": "LLaVA"}],
            "task_groups": ["open-sci-0.01"],
        },
    )
    with patch("oellm.scheduler.schedule_evals") as mock_sched:
        with (
            patch("oellm.scheduler._load_cluster_env"),
            patch.dict(os.environ, {"EVAL_OUTPUT_DIR": str(tmp_path)}),
        ):
            eval_command(
                config=cfg,
                skip_checks=True,
                venv_path=str(Path(sys.prefix)),
                dry_run=True,
            )
    called_models = mock_sched.call_args.kwargs.get("models") or mock_sched.call_args[
        1
    ].get("models")
    assert called_models == "/hpc/models/llava"


def test_eval_mixed_model_list(tmp_path: Path) -> None:
    cfg = _write_yaml(
        tmp_path / "eval.yaml",
        {
            "models": [
                {"path": "/hpc/models/model-a", "name": "ModelA"},
                "EleutherAI/pythia-70m",
            ],
            "task_groups": ["open-sci-0.01"],
        },
    )
    with patch("oellm.scheduler.schedule_evals") as mock_sched:
        with (
            patch("oellm.scheduler._load_cluster_env"),
            patch.dict(os.environ, {"EVAL_OUTPUT_DIR": str(tmp_path)}),
        ):
            eval_command(
                config=cfg,
                skip_checks=True,
                venv_path=str(Path(sys.prefix)),
                dry_run=True,
            )
    called_models = mock_sched.call_args.kwargs.get("models") or mock_sched.call_args[
        1
    ].get("models")
    assert "/hpc/models/model-a" in called_models
    assert "EleutherAI/pythia-70m" in called_models


# ---------------------------------------------------------------------------
# Validation / error cases
# ---------------------------------------------------------------------------


def test_eval_invalid_config_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        eval_command(config=str(tmp_path / "nonexistent.yaml"))


def test_eval_no_models_raises(tmp_path: Path) -> None:
    cfg = _write_yaml(
        tmp_path / "eval.yaml",
        {
            "task_groups": ["open-sci-0.01"],
        },
    )
    with pytest.raises(ValueError, match="model"):
        eval_command(config=cfg)


def test_eval_no_tasks_raises(tmp_path: Path) -> None:
    cfg = _write_yaml(
        tmp_path / "eval.yaml",
        {
            "models": ["EleutherAI/pythia-70m"],
        },
    )
    with pytest.raises(ValueError, match="task"):
        eval_command(config=cfg)


# ---------------------------------------------------------------------------
# schedule_evals signature unchanged
# ---------------------------------------------------------------------------


def test_schedule_evals_signature_unchanged() -> None:
    expected_params = {
        "models",
        "tasks",
        "task_groups",
        "n_shot",
        "eval_csv_path",
        "max_array_len",
        "limit",
        "verbose",
        "download_only",
        "dry_run",
        "skip_checks",
        "trust_remote_code",
        "venv_path",
        "slurm_template_var",
    }
    actual_params = set(inspect.signature(schedule_evals).parameters.keys())
    assert expected_params.issubset(actual_params)
