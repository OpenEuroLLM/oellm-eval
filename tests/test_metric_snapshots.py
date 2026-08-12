"""Tier 1 metric-resolution snapshot test.

For every image and video benchmark wired in ``task-groups.yaml``'s
``task_metrics`` mapping, this test asserts that ``_resolve_metric`` returns a
non-null float when handed a realistic lmms-eval result_dict.

The fixtures here are not real evaluation runs — they are minimal snippets that
mirror the JSON shape lmms-eval writes (``"<task>/<metric>,none": <value>``).
The values are placeholders; what we are pinning is the **metric-key contract**
between this repo's YAML and lmms-eval's task definitions. If lmms-eval renames
a key upstream, this test fails immediately rather than letting a production
run silently emit ``null``.

When adding a new benchmark to ``task_metrics``:
  1. Add the corresponding entry in ``SNAPSHOTS`` below.
  2. The metric-key in the fixture must match the value in ``task_metrics``.
"""

from importlib.resources import files

import pytest
import yaml

from oellm.results import _resolve_metric

# Realistic lmms-eval result_dict snippets (one per benchmark).
# Format mirrors what lmms-eval writes: ``"<task>/<metric>,none": <value>``.
# Values are illustrative — the test only checks that the configured metric key
# resolves to a non-null float, not that any specific number is correct.
SNAPSHOTS: dict[str, dict] = {
    # ── Image benchmarks ──
    "vqav2_val": {
        "vqav2_val/exact_match,none": 0.755,
        "vqav2_val/exact_match_stderr,none": 0.004,
    },
    "mmbench_en_dev": {
        "mmbench_en_dev/gpt_eval_score,none": 72.4,
    },
    "mmmu_val": {
        "mmmu_val/mmmu_acc,none": 0.412,
    },
    "chartqa": {
        "chartqa/relaxed_overall,none": 0.681,
        "chartqa/relaxed_human_split,none": 0.40,
        "chartqa/relaxed_augmented_split,none": 0.96,
    },
    "docvqa_val": {
        "docvqa_val/anls,none": 0.832,
    },
    "textvqa_val": {
        "textvqa_val/exact_match,none": 0.604,
    },
    # OCRBench: lmms-eval normalizes raw /1000 score into 0–1.
    "ocrbench": {
        "ocrbench/ocrbench_accuracy,none": 0.612,
    },
    "mathvista_testmini_cot": {
        "mathvista_testmini_cot/llm_as_judge_eval,none": 47.3,
    },
    "mathvista_testmini_format": {
        "mathvista_testmini_format/llm_as_judge_eval,none": 47.5,
    },
    "mathvista_testmini_solution": {
        "mathvista_testmini_solution/llm_as_judge_eval,none": 47.0,
    },
    # ── Video benchmarks ──
    "video_mmmu_perception": {
        "video_mmmu_perception/mmmu_acc,none": 0.589,
    },
    "video_mmmu_comprehension": {
        "video_mmmu_comprehension/mmmu_acc,none": 0.512,
    },
    "video_mmmu_adaptation": {
        "video_mmmu_adaptation/mmmu_acc,none": 0.443,
    },
    # MVBench: 0–100 from utils.mvbench_aggregate_results.
    "mvbench": {
        "mvbench/mvbench_accuracy,none": 56.2,
    },
    # EgoSchema subset: 0–1, generation-style scoring on the 500-q val split.
    "egoschema_subset": {
        "egoschema_subset/score,none": 0.658,
    },
    # VideoMME: 0–100. The "perception_score" name is upstream's misnomer for
    # the overall accuracy without subtitles.
    "videomme": {
        "videomme/videomme_perception_score,none": 65.4,
    },
    "activitynetqa": {
        "activitynetqa/gpt_eval_accuracy,none": 51.7,
        "activitynetqa/gpt_eval_score,none": 3.4,
    },
    "longvideobench_val_v": {
        "longvideobench_val_v/lvb_acc,none": 0.473,
    },
    # MME emits raw point sums (cognition /800 preferred per task_metrics).
    "mme": {
        "mme/mme_cognition_score,none": 512.3,
        "mme/mme_perception_score,none": 1373.2,
    },
    # ── Newer image benchmarks (added 41eabb8) ──
    "ocrbench_v2": {
        "ocrbench_v2/ocrbench_v2_accuracy,none": 0.412,
    },
    "realworldqa": {
        "realworldqa/exact_match,none": 0.583,
    },
    # NOTE: emitted scale not yet confirmed against a real run — the fixture
    # pins map↔scale consistency (0–1 assumed).
    "mmerealworld": {
        "mmerealworld/mme_realworld_score,none": 0.437,
    },
    "mmstar": {
        "mmstar/average,none": 0.451,
    },
    "ai2d": {
        "ai2d/exact_match,none": 0.702,
    },
    # NOTE: emitted scale not yet confirmed against a real run (0–100 assumed).
    "mathvision_test": {
        "mathvision_test/mathvision_standard_eval,none": 19.2,
    },
    "seedbench": {
        "seedbench/seed_image,none": 0.612,
    },
    # ── Per-task scale overrides (TASK_METRIC_SCALE_OVERRIDES) ──
    # lm-eval squadv2 wraps the official squad_v2 metric → 0–100 percentages.
    "squadv2": {
        "f1,none": 55.3,
        "exact,none": 50.1,
    },
    # lmms-eval voicebench judge returns the raw 1–5 average.
    "voicebench_commoneval": {
        "voicebench_commoneval/llm_as_judge_eval,none": 3.4,
    },
}


@pytest.fixture(scope="module")
def task_metrics() -> dict:
    data = yaml.safe_load((files("oellm.resources") / "task-groups.yaml").read_text())
    return data["task_metrics"]


@pytest.mark.parametrize("task_name", sorted(SNAPSHOTS.keys()))
def test_configured_metric_key_resolves_against_snapshot(
    task_name: str, task_metrics: dict
) -> None:
    """``_resolve_metric`` must return a non-null float for every wired benchmark.

    Failure mode this catches: lmms-eval renames a metric key (e.g.
    ``mmmu_acc`` → ``accuracy``) and our YAML mapping silently produces ``null``
    in production results.
    """
    expected_key = task_metrics.get(task_name)
    assert expected_key is not None, (
        f"{task_name} has a snapshot but no entry in task-groups.yaml::task_metrics"
    )

    result_dict = SNAPSHOTS[task_name]
    value, resolved_key = _resolve_metric(task_name, result_dict, task_metrics)

    assert value is not None, (
        f"_resolve_metric returned None for {task_name}. The fixture contains "
        f"'{expected_key},none' but the configured mapping "
        f"'{task_name}: {expected_key}' did not resolve. Either lmms-eval "
        f"renamed the key upstream or the fixture is stale."
    )
    assert isinstance(value, float)
    assert resolved_key is not None
    assert expected_key in resolved_key, (
        f"_resolve_metric for {task_name} resolved to '{resolved_key}', "
        f"which does not contain configured key '{expected_key}'."
    )


def test_every_snapshot_has_a_task_metrics_entry(task_metrics: dict) -> None:
    """Snapshots and task_metrics must be kept in lockstep — adding a new
    benchmark requires both."""
    missing = [t for t in SNAPSHOTS if t not in task_metrics]
    assert not missing, (
        f"Tasks have snapshots but no task_metrics entry: {missing}. "
        f"Add them to task-groups.yaml::task_metrics or remove the snapshots."
    )


# Every task of every image-* / video-* task group MUST have a snapshot —
# derived from task-groups.yaml so the set cannot drift when benchmarks are
# added.
def _image_video_tasks_from_yaml() -> set[str]:
    data = yaml.safe_load((files("oellm.resources") / "task-groups.yaml").read_text())
    tasks: set[str] = set()
    for gname, g in data.get("task_groups", {}).items():
        if gname.startswith(("image-", "video-")):
            for t in g.get("tasks", []):
                if t.get("task"):
                    tasks.add(t["task"])
    return tasks


REQUIRED_TASKS_WITH_SNAPSHOT: set[str] = _image_video_tasks_from_yaml()


def test_all_required_image_video_tasks_have_snapshot() -> None:
    """When a new image/video benchmark lands in REQUIRED_TASKS_WITH_SNAPSHOT,
    its snapshot must land at the same time."""
    missing = REQUIRED_TASKS_WITH_SNAPSHOT - set(SNAPSHOTS.keys())
    assert not missing, (
        f"Required image/video tasks missing a snapshot fixture: {missing}. "
        f"Add an entry in SNAPSHOTS for each before merging."
    )


# ── Normalization (post-resolve) ─────────────────────────────────────────────
#
# For every image+video benchmark, after _resolve_metric returns the raw
# value, the normalization helper must produce a 0–100 number. Catches the
# case where someone wires a new metric in task_metrics but forgets to
# register its native scale in METRIC_NATIVE_SCALE.


@pytest.mark.parametrize("task_name", sorted(SNAPSHOTS.keys()))
def test_normalized_value_is_in_zero_to_hundred_range(
    task_name: str, task_metrics: dict
) -> None:
    from oellm.results import _normalize_to_100

    value, resolved_key = _resolve_metric(task_name, SNAPSHOTS[task_name], task_metrics)
    normalized = _normalize_to_100(value, resolved_key, task_name)
    assert normalized is not None, (
        f"_normalize_to_100 returned None for {task_name} "
        f"(value={value}, key={resolved_key}). The metric's native scale is "
        f"not registered in METRIC_NATIVE_SCALE in oellm/results.py."
    )
    assert 0.0 <= normalized <= 100.0, (
        f"Normalized {task_name} = {normalized} is outside [0, 100]. "
        f"Native scale entry in METRIC_NATIVE_SCALE is likely wrong."
    )
