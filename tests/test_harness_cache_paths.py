"""Both harnesses must cache under HF_HOME, not in $HOME or the install dir."""

import os
from unittest.mock import patch

import pytest

from oellm.main import schedule_evals

HF_HOME = "/scratch/project/cache"


@pytest.fixture
def script(tmp_path) -> str:
    with (
        patch("oellm.main._load_cluster_env"),
        patch("oellm.main._num_jobs_in_queue", return_value=0),
        patch.dict(os.environ, {"EVAL_OUTPUT_DIR": str(tmp_path), "HF_HOME": HF_HOME}),
    ):
        schedule_evals(
            models="EleutherAI/pythia-70m",
            tasks="hellaswag",
            n_shot=0,
            skip_checks=True,
            venv_path=str(tmp_path / "venv"),
            dry_run=True,
        )
    return next(iter(tmp_path.glob("**/submit_evals.sbatch"))).read_text()


def test_cache_dirs_live_under_hf_home(script):
    assert f'export LIGHTEVAL_CACHE_DIR="{HF_HOME}/lighteval"' in script
    assert f'export LM_HARNESS_CACHE_PATH="{HF_HOME}/lm_eval"' in script
    assert 'mkdir -p "$LIGHTEVAL_CACHE_DIR" "$LM_HARNESS_CACHE_PATH"' in script


def test_lighteval_is_told_where_to_cache(script):
    """lighteval ignores HF_HOME, so it only honours cache_dir as a model arg."""
    assert script.count("cache_dir=$LIGHTEVAL_CACHE_DIR") == 2  # venv and container


def test_lm_harness_cache_reaches_contained_containers(script):
    """--contain clusters forward only the variables named here."""
    env_args = next(
        line
        for line in script.splitlines()
        if "SINGULARITY_ENV_ARGS=" in line and "--env" in line
    )
    assert "--env LM_HARNESS_CACHE_PATH=$LM_HARNESS_CACHE_PATH" in env_args


def test_lm_eval_request_cache_is_opt_in(script):
    """--cache_requests makes lm-eval ignore --limit while building contexts, so
    it stays off unless CACHE_REQUESTS is set."""
    assert "${CACHE_REQUESTS:+--cache_requests $CACHE_REQUESTS}" in script
    assert "--cache_requests true" not in script
