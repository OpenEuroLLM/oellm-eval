import os
import sys
from importlib.resources import files
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from oellm.main import schedule_evals

_config = yaml.safe_load((files("oellm.resources") / "task-groups.yaml").read_text())
ALL_TASK_GROUPS = list(_config["task_groups"].keys())


@pytest.mark.parametrize("n_shot", [None, 0])
@pytest.mark.parametrize("task_groups", ALL_TASK_GROUPS)
def test_schedule_evals(tmp_path, n_shot, task_groups):
    with (
        patch("oellm.scheduler._load_cluster_env"),
        patch("oellm.scheduler._num_jobs_in_queue", return_value=0),
        patch("oellm.runner.detect_lmms_model_type", return_value="llava"),
        patch.dict(os.environ, {"EVAL_OUTPUT_DIR": str(tmp_path)}),
    ):
        # ``allow_missing_judge`` so this parametrized smoke test can exercise
        # judge-required task groups (audio-alpaca-audio, video-activitynet-qa,
        # …) without setting ``OPENAI_API_KEY`` in CI.
        schedule_evals(
            models="EleutherAI/pythia-70m",
            task_groups=task_groups,
            n_shot=n_shot,
            skip_checks=True,
            venv_path=str(Path(sys.prefix)),
            dry_run=True,
            allow_missing_judge=True,
        )


def test_schedule_evals_slurm_template_var_overrides(tmp_path):
    """Verify --slurm_template_var JSON overrides appear in the generated sbatch."""
    with (
        patch("oellm.scheduler._load_cluster_env"),
        patch("oellm.scheduler._num_jobs_in_queue", return_value=0),
        patch.dict(
            os.environ,
            {
                "EVAL_OUTPUT_DIR": str(tmp_path),
                "PARTITION": "default_partition",
                "ACCOUNT": "test_account",
            },
        ),
    ):
        schedule_evals(
            models="EleutherAI/pythia-70m",
            tasks="hellaswag",
            n_shot=0,
            skip_checks=True,
            venv_path=str(Path(sys.prefix)),
            dry_run=True,
            slurm_template_var='{"PARTITION":"dev-g","ACCOUNT":"myproject","TIME":"02:15:00","GPUS_PER_NODE":2}',
        )

    sbatch_files = list(tmp_path.glob("**/submit_evals.sbatch"))
    assert len(sbatch_files) == 1
    sbatch_content = sbatch_files[0].read_text()
    assert "#SBATCH --partition=dev-g" in sbatch_content
    assert "#SBATCH --account=myproject" in sbatch_content
    assert "#SBATCH --time=02:15:00" in sbatch_content
    assert "#SBATCH --gres=gpu:2" in sbatch_content


def test_schedule_evals_slurm_template_var_invalid_json(tmp_path):
    """Verify invalid slurm_template_var raises ValueError."""
    with (
        patch("oellm.scheduler._load_cluster_env"),
        patch("oellm.scheduler._num_jobs_in_queue", return_value=0),
        patch.dict(os.environ, {"EVAL_OUTPUT_DIR": str(tmp_path)}),
    ):
        with pytest.raises(ValueError, match="valid JSON object"):
            schedule_evals(
                models="EleutherAI/pythia-70m",
                tasks="hellaswag",
                n_shot=0,
                skip_checks=True,
                venv_path=str(Path(sys.prefix)),
                dry_run=True,
                slurm_template_var="not valid json",
            )
        with pytest.raises(ValueError, match="must be a JSON object"):
            schedule_evals(
                models="EleutherAI/pythia-70m",
                tasks="hellaswag",
                n_shot=0,
                skip_checks=True,
                venv_path=str(Path(sys.prefix)),
                dry_run=True,
                slurm_template_var='["partition", "dev-g"]',
            )
