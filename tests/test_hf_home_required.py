"""HF_HOME must be set before a cluster run, and must not block --local."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from oellm.main import schedule_evals


def _schedule(tmp_path, **kwargs):
    return schedule_evals(
        models="EleutherAI/pythia-70m",
        tasks="hellaswag",
        n_shot=0,
        skip_checks=True,
        venv_path=str(tmp_path / "venv"),
        dry_run=True,
        **kwargs,
    )


def _cluster(monkeypatch, tmp_path, **settings):
    """Run the real _load_cluster_env (where ELLIOT checks HF_HOME) on a synthetic cluster."""
    cluster = {
        "hostname_pattern": "*",
        "PARTITION": "gpu",
        "EVAL_BASE_DIR": str(tmp_path),
        "EVAL_OUTPUT_DIR": str(tmp_path),
        "GPUS_PER_NODE": 1,
        **settings,
    }
    monkeypatch.setattr("oellm.utils.yaml.safe_load", lambda *_a, **_k: {"test": cluster})
    monkeypatch.delenv("HF_HOME", raising=False)


def test_missing_hf_home_is_rejected(tmp_path, monkeypatch):
    """Unset, HF_HOME renders an empty bind path and every job dies in singularity."""
    _cluster(monkeypatch, tmp_path)
    with (
        patch("oellm.scheduler._num_jobs_in_queue", return_value=0),
        patch.dict(os.environ, {"EVAL_OUTPUT_DIR": str(tmp_path)}),
        pytest.raises(RuntimeError, match="HF_HOME"),
    ):
        _schedule(tmp_path)


def test_hf_home_from_clusters_yaml_is_accepted(tmp_path, monkeypatch):
    """Clusters that declare HF_HOME (e.g. jupiter) set it in _load_cluster_env."""
    _cluster(monkeypatch, tmp_path, HF_HOME=str(tmp_path / "cache"))
    with (
        patch("oellm.scheduler._num_jobs_in_queue", return_value=0),
        patch.dict(os.environ, {"EVAL_OUTPUT_DIR": str(tmp_path)}),
    ):
        _schedule(tmp_path)

    assert list(tmp_path.glob("**/submit_evals.sbatch"))


def test_local_runs_still_default_hf_home(tmp_path, monkeypatch):
    """--local falls back to ~/.cache/huggingface and must not hit the guard."""
    monkeypatch.delenv("HF_HOME", raising=False)
    # --local uses setdefault for these, so a value inherited from the caller's
    # shell would win over the local default.
    monkeypatch.delenv("EVAL_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("EVAL_BASE_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    with (
        patch("oellm.scheduler._num_jobs_in_queue", return_value=0),
        patch.dict(os.environ),
    ):
        _schedule(tmp_path, local=True)
        assert os.environ["HF_HOME"] == str(Path.home() / ".cache" / "huggingface")
