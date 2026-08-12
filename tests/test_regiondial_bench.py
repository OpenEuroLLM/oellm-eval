"""Tests for the RegionDial-Bench contrib benchmark integration."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from oellm.core.base_metric import BaseMetric
from oellm.core.base_task import BaseTask
from oellm.task_groups import (
    _collect_dataset_specs,
    _collect_hf_dataset_files,
    _expand_task_groups,
    get_all_task_group_names,
)

RD_TASK_GROUP = "regiondial-bench"
RD_TASK_REFCOCOG = "regiondial_refcocog"
RD_TASK_REFCOCOPLUS = "regiondial_refcocoplus"
RD_TEST_DATA_REPO = "lmsdss/regionreasoner_test_data"


# ---------------------------------------------------------------------------
# Task group integration (end-to-end: registry → task_groups merge → YAML)
# ---------------------------------------------------------------------------


class TestRegionDialBenchTaskGroup:
    def test_task_group_present_in_all_names(self):
        assert RD_TASK_GROUP in get_all_task_group_names()

    def test_task_group_expands_to_both_splits(self):
        results = _expand_task_groups([RD_TASK_GROUP])
        task_names = {r.task for r in results}
        assert RD_TASK_REFCOCOG in task_names
        assert RD_TASK_REFCOCOPLUS in task_names

    def test_task_group_suite_is_regiondial_bench(self):
        results = _expand_task_groups([RD_TASK_GROUP])
        for r in results:
            assert r.suite == "regiondial_bench", (
                f"Expected suite 'regiondial_bench', got '{r.suite}'"
            )

    def test_task_group_n_shot_is_zero(self):
        results = _expand_task_groups([RD_TASK_GROUP])
        for r in results:
            assert r.n_shot == 0

    def test_no_dataset_pre_download(self):
        specs = _collect_dataset_specs([RD_TASK_GROUP])
        assert specs == []

    def test_hf_dataset_files_declared(self):
        import oellm.contrib.regiondial_bench.suite as s

        tasks = s.TASK_GROUPS["task_groups"][RD_TASK_GROUP]["tasks"]
        repo_ids = [
            spec["repo_id"] for task in tasks for spec in task.get("hf_dataset_files", [])
        ]
        assert RD_TEST_DATA_REPO in repo_ids

    def test_collect_hf_dataset_files_returns_correct_repo(self):
        # _collect_hf_dataset_files deduplicates by repo_id, so both splits
        # (same repo) produce a single spec entry.
        specs = _collect_hf_dataset_files([RD_TASK_GROUP])
        assert len(specs) >= 1
        assert specs[0]["repo_id"] == RD_TEST_DATA_REPO

    def test_task_groups_contains_both_task_metrics(self):
        import oellm.contrib.regiondial_bench.suite as s

        assert RD_TASK_REFCOCOG in s.TASK_GROUPS["task_metrics"]
        assert RD_TASK_REFCOCOPLUS in s.TASK_GROUPS["task_metrics"]


# ---------------------------------------------------------------------------
# BaseTask subclasses
# ---------------------------------------------------------------------------


class TestRegionDialRefCOCOgTask:
    @pytest.fixture
    def task(self):
        from oellm.contrib.regiondial_bench.task import RegionDialRefCOCOgTask

        return RegionDialRefCOCOgTask()

    def test_is_base_task_instance(self, task):
        assert isinstance(task, BaseTask)

    def test_name(self, task):
        assert task.name == RD_TASK_REFCOCOG

    def test_suite(self, task):
        assert task.suite == "regiondial_bench"

    def test_n_shots(self, task):
        assert task.n_shots == [0]

    def test_dataset_specs_empty(self, task):
        assert task.dataset_specs == []

    def test_hf_dataset_files(self, task):
        repo_ids = [f["repo_id"] for f in task.hf_dataset_files]
        assert RD_TEST_DATA_REPO in repo_ids
        patterns = task.hf_dataset_files[0]["patterns"]
        assert "raw/refcocog_multi_turn.json" in patterns

    def test_hf_models(self, task):
        assert "Ricky06662/TaskRouter-1.5B" in task.hf_models
        assert "facebook/sam2-hiera-large" in task.hf_models

    def test_primary_metric(self, task):
        assert task.primary_metric == "gIoU"

    def test_task_group_name(self, task):
        assert task.task_group_name == RD_TASK_GROUP

    def test_engine_task_name_defaults_to_name(self, task):
        assert task.engine_task_name == task.name

    def test_to_task_groups_dict_structure(self, task):
        d = task.to_task_groups_dict()
        assert "task_metrics" in d
        assert d["task_metrics"][RD_TASK_REFCOCOG] == "gIoU"
        assert RD_TASK_GROUP in d["task_groups"]
        tasks = d["task_groups"][RD_TASK_GROUP]["tasks"]
        assert any(t["task"] == RD_TASK_REFCOCOG for t in tasks)


class TestRegionDialRefCOCOplusTask:
    @pytest.fixture
    def task(self):
        from oellm.contrib.regiondial_bench.task import RegionDialRefCOCOplusTask

        return RegionDialRefCOCOplusTask()

    def test_is_base_task_instance(self, task):
        assert isinstance(task, BaseTask)

    def test_name(self, task):
        assert task.name == RD_TASK_REFCOCOPLUS

    def test_suite(self, task):
        assert task.suite == "regiondial_bench"

    def test_n_shots(self, task):
        assert task.n_shots == [0]

    def test_dataset_specs_empty(self, task):
        assert task.dataset_specs == []

    def test_hf_dataset_files(self, task):
        repo_ids = [f["repo_id"] for f in task.hf_dataset_files]
        assert RD_TEST_DATA_REPO in repo_ids
        patterns = task.hf_dataset_files[0]["patterns"]
        assert "raw/refcocoplus_multi_turn.json" in patterns

    def test_hf_models(self, task):
        assert "Ricky06662/TaskRouter-1.5B" in task.hf_models
        assert "facebook/sam2-hiera-large" in task.hf_models

    def test_primary_metric(self, task):
        assert task.primary_metric == "gIoU"

    def test_task_group_name(self, task):
        assert task.task_group_name == RD_TASK_GROUP

    def test_to_task_groups_dict_structure(self, task):
        d = task.to_task_groups_dict()
        assert d["task_metrics"][RD_TASK_REFCOCOPLUS] == "gIoU"
        tasks = d["task_groups"][RD_TASK_GROUP]["tasks"]
        assert any(t["task"] == RD_TASK_REFCOCOPLUS for t in tasks)


# ---------------------------------------------------------------------------
# BaseMetric subclasses
# ---------------------------------------------------------------------------


def _sample(
    intersection: int, union: int, bbox_iou: float = 0.0, round: int | None = None
) -> str:
    """Helper: JSON-serialise a sample dict for metric inputs."""
    d = {
        "intersection": intersection,
        "union": union,
        "bbox_iou": bbox_iou,
    }
    if round is not None:
        d["round"] = round
    return d


class TestGIoU:
    @pytest.fixture
    def metric(self):
        from oellm.contrib.regiondial_bench.metrics import GIoU

        return GIoU()

    def test_is_base_metric(self, metric):
        assert isinstance(metric, BaseMetric)

    def test_name(self, metric):
        assert metric.name == "gIoU"

    def test_perfect_overlap(self, metric):
        s = _sample(100, 100)
        assert metric.compute([s]) == pytest.approx(1.0)

    def test_zero_overlap(self, metric):
        s = _sample(0, 200)
        assert metric.compute([s]) == pytest.approx(0.0)

    def test_partial_overlap(self, metric):
        s = _sample(25, 175)
        assert metric.compute([s]) == pytest.approx(25 / 175, abs=1e-4)

    def test_mean_over_multiple_samples(self, metric):
        perfect = _sample(100, 100)
        zero = _sample(0, 200)
        score = metric.compute([perfect, zero])
        assert score == pytest.approx(0.5)

    def test_empty_input(self, metric):
        assert metric.compute([]) == pytest.approx(0.0)

    def test_null_sample(self, metric):
        score = metric.compute([None])
        assert score == pytest.approx(0.0)


class TestCIoU:
    @pytest.fixture
    def metric(self):
        from oellm.contrib.regiondial_bench.metrics import CIoU

        return CIoU()

    def test_is_base_metric(self, metric):
        assert isinstance(metric, BaseMetric)

    def test_name(self, metric):
        assert metric.name == "cIoU"

    def test_perfect_overlap(self, metric):
        s = _sample(100, 100)
        assert metric.compute([s]) == pytest.approx(1.0)

    def test_zero_overlap(self, metric):
        s = _sample(0, 200)
        assert metric.compute([s]) == pytest.approx(0.0)

    def test_cumulative_formula_differs_from_giou(self, metric):
        from oellm.contrib.regiondial_bench.metrics import GIoU

        giou = GIoU()
        s1 = _sample(100, 100)
        s2 = _sample(50, 200)
        preds = [s1, s2]
        ciou_val = metric.compute(preds)  # (100+50)/(100+200) = 0.5
        giou_val = giou.compute(preds)  # (1.0+0.25)/2 = 0.625
        assert ciou_val == pytest.approx(0.5)
        assert giou_val == pytest.approx(0.625)
        assert ciou_val != pytest.approx(giou_val)

    def test_empty_input(self, metric):
        assert metric.compute([]) == pytest.approx(0.0)


class TestBboxAP:
    @pytest.fixture
    def metric(self):
        from oellm.contrib.regiondial_bench.metrics import BboxAP

        return BboxAP()

    def test_is_base_metric(self, metric):
        assert isinstance(metric, BaseMetric)

    def test_name(self, metric):
        assert metric.name == "bbox_AP"

    def test_all_correct(self, metric):
        s = _sample(100, 100, bbox_iou=0.9)
        assert metric.compute([s, s]) == pytest.approx(1.0)

    def test_none_correct(self, metric):
        s = _sample(10, 200, bbox_iou=0.3)
        assert metric.compute([s]) == pytest.approx(0.0)

    def test_threshold_at_half(self, metric):
        above = _sample(80, 100, bbox_iou=0.6)
        below = _sample(10, 100, bbox_iou=0.4)
        assert metric.compute([above, below]) == pytest.approx(0.5)

    def test_empty_input(self, metric):
        assert metric.compute([]) == pytest.approx(0.0)


class TestPassRate:
    @pytest.fixture(params=[0.3, 0.5, 0.7, 0.9])
    def threshold(self, request):
        return request.param

    def test_name_includes_threshold(self, threshold):
        from oellm.contrib.regiondial_bench.metrics import PassRate

        pr = PassRate(threshold)
        assert pr.name == f"pass_rate_{threshold}"

    def test_is_base_metric(self):
        from oellm.contrib.regiondial_bench.metrics import PassRate

        assert isinstance(PassRate(0.5), BaseMetric)

    def test_all_pass(self):
        from oellm.contrib.regiondial_bench.metrics import PassRate

        s = _sample(100, 100)
        pr = PassRate(0.5)
        assert pr.compute([s, s]) == pytest.approx(1.0)

    def test_none_pass(self):
        from oellm.contrib.regiondial_bench.metrics import PassRate

        s = _sample(0, 100)
        pr = PassRate(0.3)
        assert pr.compute([s]) == pytest.approx(0.0)

    def test_half_pass(self):
        from oellm.contrib.regiondial_bench.metrics import PassRate

        perfect = _sample(100, 100)
        zero = _sample(0, 100)
        pr = PassRate(0.5)
        score = pr.compute([perfect, zero])
        assert score == pytest.approx(0.5)

    def test_invalid_threshold_raises(self):
        from oellm.contrib.regiondial_bench.metrics import PassRate

        with pytest.raises(ValueError):
            PassRate(0.0)
        with pytest.raises(ValueError):
            PassRate(1.0)
        with pytest.raises(ValueError):
            PassRate(-0.1)

    def test_empty_input(self):
        from oellm.contrib.regiondial_bench.metrics import PassRate

        assert PassRate(0.5).compute([]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Suite plugin protocol
# ---------------------------------------------------------------------------


class TestSuiteProtocol:
    @pytest.fixture
    def suite(self):
        import oellm.contrib.regiondial_bench.suite as s

        return s

    def test_suite_name(self, suite):
        assert suite.SUITE_NAME == "regiondial_bench"

    def test_cluster_env_vars_declared(self, suite):
        assert "REGION_REASONER_DIR" in suite.CLUSTER_ENV_VARS

    def test_task_groups_structure(self, suite):
        tg = suite.TASK_GROUPS
        assert "task_metrics" in tg
        assert "task_groups" in tg
        assert RD_TASK_REFCOCOG in tg["task_metrics"]
        assert RD_TASK_REFCOCOPLUS in tg["task_metrics"]
        assert RD_TASK_GROUP in tg["task_groups"]

    def test_task_groups_has_both_tasks(self, suite):
        tasks = suite.TASK_GROUPS["task_groups"][RD_TASK_GROUP]["tasks"]
        task_names = {t["task"] for t in tasks}
        assert RD_TASK_REFCOCOG in task_names
        assert RD_TASK_REFCOCOPLUS in task_names

    def test_detect_model_flags_region_reasoner_model(self, suite):
        assert suite.detect_model_flags("lmsdss/RegionReasoner-7B") == "vision_reasoner"

    def test_detect_model_flags_qwen2_model(self, suite):
        assert suite.detect_model_flags("Qwen/Qwen2.5-VL-7B-Instruct") == "qwen2.5"

    def test_detect_model_flags_qwen1_model(self, suite):
        assert suite.detect_model_flags("Qwen/Qwen-VL-Chat") == "qwen"

    def test_detect_model_flags_unknown_defaults_to_vision_reasoner(self, suite):
        assert suite.detect_model_flags("some/unknown-model") == "vision_reasoner"

    def test_parse_results_refcocog_json(self, suite):
        data = {
            "model_name_or_path": "/path/to/model",
            "results": {
                RD_TASK_REFCOCOG: {
                    "gIoU": 0.42,
                    "cIoU": 0.45,
                    "bbox_AP": 0.38,
                }
            },
            "configs": {RD_TASK_REFCOCOG: {"num_fewshot": 0}},
        }
        result = suite.parse_results(data)
        assert result is not None
        model_id, task_name, n_shot, metrics = result
        assert model_id == "/path/to/model"
        assert task_name == RD_TASK_REFCOCOG
        assert n_shot == 0
        assert metrics["gIoU"] == pytest.approx(0.42)

    def test_parse_results_refcocoplus_json(self, suite):
        data = {
            "model_name_or_path": "/path/to/model",
            "results": {
                RD_TASK_REFCOCOPLUS: {
                    "gIoU": 0.55,
                    "cIoU": 0.50,
                    "bbox_AP": 0.48,
                }
            },
            "configs": {RD_TASK_REFCOCOPLUS: {"num_fewshot": 0}},
        }
        result = suite.parse_results(data)
        assert result is not None
        _, task_name, _, metrics = result
        assert task_name == RD_TASK_REFCOCOPLUS
        assert metrics["gIoU"] == pytest.approx(0.55)

    def test_parse_results_non_matching_json_returns_none(self, suite):
        data = {
            "model_name": "some_model",
            "results": {"mmlu": {"acc,none": 0.55}},
            "n-shot": {"mmlu": 5},
        }
        assert suite.parse_results(data) is None

    def test_parse_results_empty_results_returns_none(self, suite):
        assert suite.parse_results({}) is None


# ---------------------------------------------------------------------------
# ModelAdapter
# ---------------------------------------------------------------------------


class TestRegionDialModelAdapter:
    @pytest.fixture
    def adapter_cls(self):
        from oellm.contrib.regiondial_bench.adapter import RegionDialModelAdapter
        from oellm.core.base_model_adapter import BaseModelAdapter

        return RegionDialModelAdapter, BaseModelAdapter

    def test_is_base_model_adapter(self, adapter_cls):
        cls, base = adapter_cls
        assert issubclass(cls, base)

    def test_contrib_flags_region_reasoner(self, adapter_cls):
        cls, _ = adapter_cls
        assert cls("lmsdss/RegionReasoner-7B").to_contrib_flags() == "vision_reasoner"

    def test_contrib_flags_qwen2(self, adapter_cls):
        cls, _ = adapter_cls
        assert cls("Qwen/Qwen2.5-VL-7B").to_contrib_flags() == "qwen2.5"

    def test_contrib_flags_qwen(self, adapter_cls):
        cls, _ = adapter_cls
        assert cls("Qwen/Qwen-VL-Chat").to_contrib_flags() == "qwen"

    def test_contrib_flags_unknown_defaults_to_vision_reasoner(self, adapter_cls):
        cls, _ = adapter_cls
        assert cls("some/unknown-model").to_contrib_flags() == "vision_reasoner"

    def test_detect_model_flags_delegates_to_adapter(self):
        import oellm.contrib.regiondial_bench.suite as s

        assert s.detect_model_flags("lmsdss/RegionReasoner-7B") == "vision_reasoner"
        assert s.detect_model_flags("Qwen/Qwen2.5-VL-7B") == "qwen2.5"


# ---------------------------------------------------------------------------
# Schedule evals dry-run integration
# ---------------------------------------------------------------------------


class TestRegionDialSchedule:
    def test_schedule_evals_dry_run(self, tmp_path):
        from oellm.main import schedule_evals

        with (
            patch("oellm.scheduler._load_cluster_env"),
            patch("oellm.scheduler._num_jobs_in_queue", return_value=0),
            patch.dict(os.environ, {"EVAL_OUTPUT_DIR": str(tmp_path)}),
        ):
            schedule_evals(
                models="lmsdss/RegionReasoner-7B",
                task_groups=RD_TASK_GROUP,
                skip_checks=True,
                venv_path=str(Path(sys.prefix)),
                dry_run=True,
            )

        sbatch_files = list(tmp_path.glob("**/submit_evals.sbatch"))
        assert len(sbatch_files) == 1
        sbatch_content = sbatch_files[0].read_text()
        assert "oellm.contrib.dispatch" in sbatch_content

    def test_jobs_csv_has_regiondial_bench_suite(self, tmp_path):
        import pandas as pd

        from oellm.main import schedule_evals

        with (
            patch("oellm.scheduler._load_cluster_env"),
            patch("oellm.scheduler._num_jobs_in_queue", return_value=0),
            patch.dict(os.environ, {"EVAL_OUTPUT_DIR": str(tmp_path)}),
        ):
            schedule_evals(
                models="lmsdss/RegionReasoner-7B",
                task_groups=RD_TASK_GROUP,
                skip_checks=True,
                venv_path=str(Path(sys.prefix)),
                dry_run=True,
            )

        csv_files = list(tmp_path.glob("**/jobs.csv"))
        assert len(csv_files) == 1
        df = pd.read_csv(csv_files[0])
        assert all(s.startswith("regiondial_bench") for s in df["eval_suite"])
        assert set(df["task_path"]) == {RD_TASK_REFCOCOG, RD_TASK_REFCOCOPLUS}


# ---------------------------------------------------------------------------
# _aggregate_shards (actual metric computation path)
# ---------------------------------------------------------------------------


class TestAggregateShards:
    """Verify _aggregate_shards routes through metrics.py BaseMetric classes."""

    def _write_shard(self, shard_dir, idx, samples):
        path = shard_dir / f"output_{idx}.json"
        path.write_text(json.dumps(samples))

    def test_perfect_overlap(self, tmp_path):
        from oellm.contrib.regiondial_bench.suite import _aggregate_shards

        self._write_shard(
            tmp_path,
            0,
            [{"intersection": 100, "union": 100, "bbox_iou": 1.0}],
        )
        m = _aggregate_shards(str(tmp_path))
        assert m["gIoU"] == pytest.approx(1.0)
        assert m["cIoU"] == pytest.approx(1.0)
        assert m["bbox_AP"] == pytest.approx(1.0)
        assert m["pass_rate_0.3"] == pytest.approx(1.0)
        assert m["pass_rate_0.9"] == pytest.approx(1.0)

    def test_zero_overlap(self, tmp_path):
        from oellm.contrib.regiondial_bench.suite import _aggregate_shards

        self._write_shard(
            tmp_path,
            0,
            [{"intersection": 0, "union": 200, "bbox_iou": 0.0}],
        )
        m = _aggregate_shards(str(tmp_path))
        assert m["gIoU"] == pytest.approx(0.0)
        assert m["cIoU"] == pytest.approx(0.0)
        assert m["bbox_AP"] == pytest.approx(0.0)
        assert m["pass_rate_0.3"] == pytest.approx(0.0)

    def test_pass_rates_differ_across_thresholds(self, tmp_path):
        from oellm.contrib.regiondial_bench.suite import _aggregate_shards

        self._write_shard(
            tmp_path,
            0,
            [
                {"intersection": 100, "union": 100, "bbox_iou": 1.0},
                {"intersection": 10, "union": 200, "bbox_iou": 0.05},
            ],
        )
        m = _aggregate_shards(str(tmp_path))
        assert m["pass_rate_0.3"] == pytest.approx(0.5)
        assert m["pass_rate_0.5"] == pytest.approx(0.5)
        assert m["pass_rate_0.7"] == pytest.approx(0.5)
        assert m["pass_rate_0.9"] == pytest.approx(0.5)

    def test_pass_rates_actually_differ(self, tmp_path):
        from oellm.contrib.regiondial_bench.suite import _aggregate_shards

        self._write_shard(
            tmp_path,
            0,
            [
                {"intersection": 100, "union": 100, "bbox_iou": 1.0},
                {"intersection": 64, "union": 100, "bbox_iou": 0.64},
            ],
        )
        m = _aggregate_shards(str(tmp_path))
        assert m["pass_rate_0.3"] == pytest.approx(1.0)
        assert m["pass_rate_0.5"] == pytest.approx(1.0)
        assert m["pass_rate_0.7"] == pytest.approx(0.5)
        assert m["pass_rate_0.9"] == pytest.approx(0.5)

    def test_multiple_shards_aggregated(self, tmp_path):
        from oellm.contrib.regiondial_bench.suite import _aggregate_shards

        self._write_shard(
            tmp_path,
            0,
            [{"intersection": 100, "union": 100, "bbox_iou": 1.0}],
        )
        self._write_shard(
            tmp_path,
            1,
            [{"intersection": 0, "union": 100, "bbox_iou": 0.0}],
        )
        m = _aggregate_shards(str(tmp_path))
        assert m["gIoU"] == pytest.approx(0.5)
        assert m["cIoU"] == pytest.approx(0.5)

    def test_no_shard_files_raises(self, tmp_path):
        from oellm.contrib.regiondial_bench.suite import _aggregate_shards

        with pytest.raises(RuntimeError, match="No shard output files"):
            _aggregate_shards(str(tmp_path))

    def test_empty_shard_raises(self, tmp_path):
        from oellm.contrib.regiondial_bench.suite import _aggregate_shards

        self._write_shard(tmp_path, 0, [])
        with pytest.raises(RuntimeError, match="No samples found"):
            _aggregate_shards(str(tmp_path))

    def test_per_round_metrics_present(self, tmp_path):
        """Two images with two turns each produce per-round gIoU and bbox_AP keys."""
        from oellm.contrib.regiondial_bench.suite import _aggregate_shards

        # Turns are consecutive per image (img1 T1, img1 T2, img2 T1, img2 T2).
        # The turn counter assigns: first occurrence of each image_id → R1,
        # second occurrence → R2.
        self._write_shard(
            tmp_path,
            0,
            [
                {"image_id": "img1", "intersection": 100, "union": 100, "bbox_iou": 1.0},
                {"image_id": "img1", "intersection": 0, "union": 100, "bbox_iou": 0.0},
                {"image_id": "img2", "intersection": 50, "union": 100, "bbox_iou": 0.6},
                {"image_id": "img2", "intersection": 80, "union": 100, "bbox_iou": 0.8},
            ],
        )
        m = _aggregate_shards(str(tmp_path))
        # Per-round keys must exist
        assert "gIoU_R1" in m
        assert "gIoU_R2" in m
        assert "bbox_AP_R1" in m
        assert "bbox_AP_R2" in m
        # R1: gIoU = mean(1.0, 0.5) = 0.75
        assert m["gIoU_R1"] == pytest.approx(0.75)
        # R2: gIoU = mean(0.0, 0.8) = 0.4
        assert m["gIoU_R2"] == pytest.approx(0.4)
        # R1 bbox_AP: both > 0.5 → 1.0
        assert m["bbox_AP_R1"] == pytest.approx(1.0)
        # R2 bbox_AP: one >0.5 (0.8), one =0.0 → 0.5
        assert m["bbox_AP_R2"] == pytest.approx(0.5)

    def test_per_round_metrics_always_present(self, tmp_path):
        """Per-round keys are always produced — turns are inferred from image_id order."""
        from oellm.contrib.regiondial_bench.suite import _aggregate_shards

        self._write_shard(
            tmp_path,
            0,
            [{"image_id": "img1", "intersection": 100, "union": 100, "bbox_iou": 1.0}],
        )
        m = _aggregate_shards(str(tmp_path))
        assert "gIoU_R1" in m
        assert "bbox_AP_R1" in m

    def test_per_round_metrics_seven_rounds(self, tmp_path):
        """All 7 rounds produce per-round metrics when present."""
        from oellm.contrib.regiondial_bench.suite import _aggregate_shards

        samples = []
        for rnd in range(1, 8):
            samples.append(
                {
                    "intersection": 100 - rnd * 10,
                    "union": 100,
                    "bbox_iou": (100 - rnd * 10) / 100,
                    "round": rnd,
                }
            )
        self._write_shard(tmp_path, 0, samples)
        m = _aggregate_shards(str(tmp_path))
        for rnd in range(1, 8):
            assert f"gIoU_R{rnd}" in m
            assert f"bbox_AP_R{rnd}" in m


# ---------------------------------------------------------------------------
# collect_results compatibility
# ---------------------------------------------------------------------------


class TestCollectResultsCompatibility:
    """Verify collect_results() parses RegionDial-Bench output without modification."""

    def test_collect_results_parses_refcocog_json(self, tmp_path):
        import pandas as pd

        from oellm.main import collect_results

        results_dir = tmp_path / "results"
        results_dir.mkdir()

        mock_output = {
            "model_name_or_path": "/cluster/models/RegionReasoner-7B",
            "results": {
                RD_TASK_REFCOCOG: {
                    "gIoU": 0.42,
                    "cIoU": 0.45,
                    "bbox_AP": 0.38,
                    "pass_rate_0.3": 0.71,
                    "pass_rate_0.5": 0.55,
                }
            },
            "configs": {RD_TASK_REFCOCOG: {"num_fewshot": 0}},
        }
        (results_dir / "abc123.json").write_text(json.dumps(mock_output))

        output_csv = str(tmp_path / "results.csv")
        collect_results(str(tmp_path), output_csv=output_csv)

        assert Path(output_csv).exists()
        df = pd.read_csv(output_csv)
        assert len(df) == 1
        row = df.iloc[0]
        assert row["task"] == RD_TASK_REFCOCOG
        assert row["metric_name"] in ("gIoU", "gIoU,none")
        assert float(row["performance"]) == pytest.approx(0.42)
        assert row["model_name"] == "/cluster/models/RegionReasoner-7B"

    def test_collect_results_parses_refcocoplus_json(self, tmp_path):
        import pandas as pd

        from oellm.main import collect_results

        results_dir = tmp_path / "results"
        results_dir.mkdir()

        mock_output = {
            "model_name_or_path": "/cluster/models/RegionReasoner-7B",
            "results": {
                RD_TASK_REFCOCOPLUS: {
                    "gIoU": 0.55,
                    "cIoU": 0.50,
                    "bbox_AP": 0.48,
                }
            },
            "configs": {RD_TASK_REFCOCOPLUS: {"num_fewshot": 0}},
        }
        (results_dir / "def456.json").write_text(json.dumps(mock_output))

        output_csv = str(tmp_path / "results.csv")
        collect_results(str(tmp_path), output_csv=output_csv)

        assert Path(output_csv).exists()
        df = pd.read_csv(output_csv)
        assert len(df) == 1
        row = df.iloc[0]
        assert row["task"] == RD_TASK_REFCOCOPLUS
        assert float(row["performance"]) == pytest.approx(0.55)


# ---------------------------------------------------------------------------
# Multi-GPU sharding (regression: shards 1..G-1 must not evaluate empty slices)
# ---------------------------------------------------------------------------


def _conversation(image_id: str, ious: list[tuple[int, int, float]]) -> dict:
    """Build a test-JSON item with one conversational turn per (i, u, bbox) tuple."""
    return {
        "image_id": image_id,
        "image_path": f"{image_id}.jpg",
        "conversational_turns": [
            {"question": f"q{k}", "bboxes": [[0, 0, 1, 1]], "ann_ids": [k]}
            for k, _ in enumerate(ious)
        ],
        # carried separately for the fake inference below
        "_expected_ious": ious,
    }


class _FakeInferencePopen:
    """Stands in for the RegionReasoner inference subprocess.

    Reads the shard file given via --test_data_path and writes one per-turn
    record per conversational turn into --output_path/output_0.json — i.e.
    behaves like the real script invoked correctly. Records every cmd for
    later assertions.
    """

    calls: list[list[str]] = []
    returncode_for_dir: dict[str, int] = {}
    write_empty_for_dir: set[str] = set()

    def __init__(self, cmd, env=None, cwd=None):
        type(self).calls.append(list(cmd))
        out_dir = Path(cmd[cmd.index("--output_path") + 1])
        shard_file = Path(cmd[cmd.index("--test_data_path") + 1])
        self._rc = type(self).returncode_for_dir.get(out_dir.name, 0)

        records = []
        if out_dir.name not in type(self).write_empty_for_dir:
            items = json.loads(shard_file.read_text())
            for item in items:
                for inter, union, bbox_iou in item["_expected_ious"]:
                    records.append(
                        {
                            "image_id": str(item["image_id"]),
                            "intersection": inter,
                            "union": union,
                            "bbox_iou": bbox_iou,
                        }
                    )
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "output_0.json").write_text(json.dumps(records))

    def wait(self):
        return self._rc


class TestMultiGpuSharding:
    @pytest.fixture
    def rr_env(self, tmp_path):
        """A fake REGION_REASONER_DIR plus a 2-conversation test JSON."""
        rr_dir = tmp_path / "RegionReasoner"
        script = rr_dir / "test" / "evaluation" / "evaluation_multi_segmentation.py"
        script.parent.mkdir(parents=True)
        script.write_text("# fake inference script\n")

        test_json = tmp_path / "refcocog_multi_turn.json"
        test_json.write_text(
            json.dumps(
                [
                    _conversation("img1", [(100, 100, 1.0), (0, 100, 0.0)]),
                    _conversation("img2", [(50, 100, 0.6), (80, 100, 0.8)]),
                ]
            )
        )
        return {
            "REGION_REASONER_DIR": str(rr_dir),
            "GPUS_PER_NODE": "2",
            "REGION_REASONER_TEST_JSON_REFCOCOG": str(test_json),
        }

    @pytest.fixture(autouse=True)
    def _reset_fake(self):
        _FakeInferencePopen.calls = []
        _FakeInferencePopen.returncode_for_dir = {}
        _FakeInferencePopen.write_empty_for_dir = set()
        yield

    def _run(self, rr_env, tmp_path):
        from oellm.contrib.regiondial_bench import suite

        output_path = tmp_path / "result.json"
        with patch.object(suite.subprocess, "Popen", _FakeInferencePopen):
            suite.run(
                model_path="/models/RegionReasoner-7B",
                task=RD_TASK_REFCOCOG,
                n_shot=0,
                output_path=output_path,
                model_flags="vision_reasoner",
                env=rr_env,
            )
        return output_path

    def test_stream_preshard_counts_turns(self, tmp_path):
        from oellm.contrib.regiondial_bench.suite import _stream_preshard

        src = tmp_path / "src.json"
        src.write_text(
            json.dumps(
                [
                    _conversation("a", [(1, 1, 1.0), (1, 1, 1.0)]),
                    _conversation("b", [(1, 1, 1.0)]),
                    _conversation("c", [(1, 1, 1.0)] * 3),
                ]
            )
        )
        out = tmp_path / "shards"
        out.mkdir()
        paths, total_turns = _stream_preshard(str(src), str(out), 2)
        assert len(paths) == 2
        assert total_turns == 6
        # conversations stay whole: 2 items in shard 0, 1 in shard 1
        assert len(json.loads(Path(paths[0]).read_text())) == 2
        assert len(json.loads(Path(paths[1]).read_text())) == 1

    def test_every_shard_runs_with_idx_zero_and_own_output_dir(self, rr_env, tmp_path):
        self._run(rr_env, tmp_path)

        assert len(_FakeInferencePopen.calls) == 2
        out_dirs = set()
        for cmd in _FakeInferencePopen.calls:
            assert cmd[cmd.index("--idx") + 1] == "0"
            assert cmd[cmd.index("--num_parts") + 1] == "1"
            out_dirs.add(cmd[cmd.index("--output_path") + 1])
        assert len(out_dirs) == 2  # no output filename collision possible

    def test_metrics_cover_all_shards(self, rr_env, tmp_path):
        output_path = self._run(rr_env, tmp_path)

        result = json.loads(output_path.read_text())
        metrics = result["results"][RD_TASK_REFCOCOG]
        # gIoU over all four turns from both shards: (1.0+0.0+0.5+0.8)/4
        assert metrics["gIoU"] == pytest.approx(0.575)
        # per-round: R1 = mean(1.0, 0.5), R2 = mean(0.0, 0.8)
        assert metrics["gIoU_R1"] == pytest.approx(0.75)
        assert metrics["gIoU_R2"] == pytest.approx(0.4)

    def test_empty_shard_output_is_refused(self, rr_env, tmp_path):
        """Regression for the --idx/--num_parts bug: a shard that silently
        evaluates nothing must fail the run, not shrink the benchmark."""
        _FakeInferencePopen.write_empty_for_dir = {"shard_1"}
        with pytest.raises(RuntimeError, match="subset|partial"):
            self._run(rr_env, tmp_path)

    def test_failed_shard_reported_after_all_finish(self, rr_env, tmp_path):
        _FakeInferencePopen.returncode_for_dir = {"shard_1": 3}
        with pytest.raises(RuntimeError, match="shard 1 exited with code 3"):
            self._run(rr_env, tmp_path)
