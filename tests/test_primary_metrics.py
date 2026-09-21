"""Primary-metric resolution for `oellm-eval collect`.

The metric a task reports is declared alongside the task itself in
task-groups.yaml -- a group sets `metric:`, an individual task entry may
override it. These tests pin that contract, and the `fetch_all_metrics`
escape hatch that bypasses it.
"""

import json

import pandas as pd
import pytest

from oellm.main import collect_results
from oellm.task_groups import _load_task_groups_data, primary_metric_map


def test_group_metric_applies_to_every_templated_task():
    """A group's `metric:` reaches every task its {lang} template expands to,
    so adding a language needs no second edit."""
    metrics = primary_metric_map()
    groups = _load_task_groups_data()["task_groups"]
    for name, group in groups.items():
        declared = group.get("metric")
        if not declared:
            continue
        for task in group.get("tasks") or []:
            expected = task.get("metric", declared)
            assert metrics[task["task"]] == expected, (
                f"{name}: {task['task']} resolved to {metrics[task['task']]}, "
                f"expected {expected}"
            )


def test_task_entry_overrides_its_group():
    """dclm-core-22 defaults to acc but scores its generative tasks with
    exact_match and its span tasks with f1."""
    metrics = primary_metric_map()
    assert metrics["arc_easy"] == "acc_norm"
    assert metrics["coqa"] == "f1"
    assert metrics["squadv2"] == "f1"
    assert metrics["jeopardy"] == "exact_match"
    assert metrics["bigbench_operators_generate_until"] == "exact_match"
    assert metrics["boolq"] == "acc"  # falls through to the group default


def test_groups_without_a_metric_declare_nothing():
    """Groups that declare no metric leave their tasks out of the map, which is
    what makes collect fall back to keeping every metric."""
    metrics = primary_metric_map()
    groups = _load_task_groups_data()["task_groups"]
    for name, group in groups.items():
        if group.get("metric"):
            continue
        for task in group.get("tasks") or []:
            if "metric" in task:
                continue
            assert task["task"] not in metrics or metrics[task["task"]], name


def _write_result(tmp_path, task, metrics):
    run = tmp_path / "run"
    run.mkdir()
    (run / "jobs.csv").write_text(
        f"model_path,task_path,n_shot,eval_suite\nm,{task},0,lm-eval-harness\n"
    )
    (run / "r.json").write_text(
        json.dumps(
            {
                "results": {task: metrics},
                "configs": {task: {}},
                "n-shot": {task: 0},
                "model_name": "m",
            }
        )
    )
    return run


@pytest.mark.parametrize(
    "fetch_all, expected",
    [(False, {"acc_norm"}), (True, {"acc", "acc_norm", "acc_norm_stderr"})],
)
def test_fetch_all_metrics_flag(tmp_path, fetch_all, expected):
    """Default keeps only the declared primary metric; the flag keeps all."""
    run = _write_result(
        tmp_path,
        "arc_easy",
        {"acc,none": 0.5, "acc_norm,none": 0.6, "acc_norm_stderr,none": 0.01},
    )
    out = tmp_path / "out.csv"
    collect_results(str(run), str(out), fetch_all_metrics=fetch_all)
    df = pd.read_csv(out)
    assert set(df["metric_name"]) == expected


def test_unknown_metric_does_not_drop_the_task(tmp_path):
    """If the declared metric is missing from a result, keep what is there
    rather than emitting nothing for that task."""
    run = _write_result(tmp_path, "arc_easy", {"acc,none": 0.5})
    out = tmp_path / "out.csv"
    collect_results(str(run), str(out), fetch_all_metrics=False)
    df = pd.read_csv(out)
    assert set(df["metric_name"]) == {"acc"}
