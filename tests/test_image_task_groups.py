import os
import sys
from importlib.resources import files
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from oellm.task_groups import (
    _collect_dataset_specs,
    _expand_task_groups,
    get_all_task_group_names,
)

IMAGE_TASK_GROUP = "image-vqa"

EXPECTED_TASKS = {
    "vqav2_val",
    "mmbench_en_dev",
    "mmmu_val",
    "chartqa",
    "docvqa_val",
    "textvqa_val",
    "ocrbench",
    "mathvista_testmini_cot",
    "mathvista_testmini_format",
    "mathvista_testmini_solution",
}

EXPECTED_DATASETS = {
    "lmms-lab/VQAv2",
    "lmms-lab/MMBench",
    "MMMU/MMMU",
    "lmms-lab/ChartQA",
    "lmms-lab/DocVQA",
    "lmms-lab/textvqa",
    "echo840/OCRBench",
    "AI4Math/MathVista",
}


class TestImageTaskGroupInRegistry:
    def test_image_vqa_present_in_yaml(self):
        all_groups = get_all_task_group_names()
        assert IMAGE_TASK_GROUP in all_groups

    def test_image_vqa_suite_is_lmms_eval(self):
        data = yaml.safe_load((files("oellm.resources") / "task-groups.yaml").read_text())
        suite = data["task_groups"][IMAGE_TASK_GROUP]["suite"]
        assert suite == "lmms_eval"

    def test_image_vqa_has_ten_tasks(self):
        data = yaml.safe_load((files("oellm.resources") / "task-groups.yaml").read_text())
        tasks = data["task_groups"][IMAGE_TASK_GROUP]["tasks"]
        # 7 single-metric leaves + 3 mathvista_testmini_* leaves
        assert len(tasks) == 10


class TestImageTaskGroupExpansion:
    def test_expands_to_correct_task_names(self):
        results = _expand_task_groups([IMAGE_TASK_GROUP])
        task_names = {r.task for r in results}
        assert task_names == EXPECTED_TASKS

    def test_all_tasks_have_zero_shot(self):
        results = _expand_task_groups([IMAGE_TASK_GROUP])
        for r in results:
            assert r.n_shot == 0, f"{r.task} has n_shot={r.n_shot}, expected 0"

    def test_all_tasks_route_to_lmms_eval(self):
        results = _expand_task_groups([IMAGE_TASK_GROUP])
        for r in results:
            assert r.suite == "lmms_eval", (
                f"{r.task} has suite='{r.suite}', expected 'lmms_eval'"
            )

    def test_expand_unknown_group_raises(self):
        with pytest.raises(ValueError, match="Unknown task group"):
            _expand_task_groups(["nonexistent-group"])


class TestImageTaskGroupDatasetSpecs:
    def test_all_expected_datasets_present(self):
        specs = _collect_dataset_specs([IMAGE_TASK_GROUP])
        repo_ids = {s.repo_id for s in specs}
        assert repo_ids == EXPECTED_DATASETS

    def test_no_duplicate_dataset_specs(self):
        specs = _collect_dataset_specs([IMAGE_TASK_GROUP])
        keys = [(s.repo_id, s.subset) for s in specs]
        assert len(keys) == len(set(keys)), "Duplicate dataset specs found"

    def test_vqav2_dataset_included(self):
        specs = _collect_dataset_specs([IMAGE_TASK_GROUP])
        repo_ids = {s.repo_id for s in specs}
        assert "lmms-lab/VQAv2" in repo_ids

    def test_mmmu_dataset_included(self):
        specs = _collect_dataset_specs([IMAGE_TASK_GROUP])
        repo_ids = {s.repo_id for s in specs}
        assert "MMMU/MMMU" in repo_ids


class TestImageTaskGroupScheduleEvals:
    """Verify image-vqa integrates with the schedule_evals dry-run path."""

    def test_schedule_evals_dry_run_image_vqa(self, tmp_path):
        from oellm.main import schedule_evals

        with (
            patch("oellm.scheduler._load_cluster_env"),
            patch("oellm.scheduler._num_jobs_in_queue", return_value=0),
            patch("oellm.runner.detect_lmms_model_type", return_value="llava"),
            patch.dict(os.environ, {"EVAL_OUTPUT_DIR": str(tmp_path)}),
        ):
            schedule_evals(
                models="EleutherAI/pythia-70m",
                task_groups=IMAGE_TASK_GROUP,
                skip_checks=True,
                venv_path=str(Path(sys.prefix)),
                dry_run=True,
            )

        sbatch_files = list(tmp_path.glob("**/submit_evals.sbatch"))
        assert len(sbatch_files) == 1
        sbatch_content = sbatch_files[0].read_text()
        # The generated sbatch should contain the lmms_eval suite value
        assert "lmms_eval" in sbatch_content

    def test_schedule_evals_jobs_csv_has_lmms_eval_suite(self, tmp_path):
        import pandas as pd

        from oellm.main import schedule_evals

        with (
            patch("oellm.scheduler._load_cluster_env"),
            patch("oellm.scheduler._num_jobs_in_queue", return_value=0),
            patch("oellm.runner.detect_lmms_model_type", return_value="llava"),
            patch.dict(os.environ, {"EVAL_OUTPUT_DIR": str(tmp_path)}),
        ):
            schedule_evals(
                models="EleutherAI/pythia-70m",
                task_groups=IMAGE_TASK_GROUP,
                skip_checks=True,
                venv_path=str(Path(sys.prefix)),
                dry_run=True,
            )

        csv_files = list(tmp_path.glob("**/jobs.csv"))
        assert len(csv_files) == 1
        df = pd.read_csv(csv_files[0])
        assert all(s.startswith("lmms_eval") for s in df["eval_suite"].unique())
        assert set(df["task_path"].unique()) == EXPECTED_TASKS
