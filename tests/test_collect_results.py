"""Tests for ``collect_results`` aggregating lm-eval group results.

An lm-eval result JSON carries a ``groups`` map holding one aggregated entry
per group *and* per intermediate subgroup. Running ``lm_eval --tasks mmlu``
produces five: ``mmlu`` plus ``mmlu_humanities``, ``mmlu_other``,
``mmlu_social_sciences`` and ``mmlu_stem``. Subgroups are aggregated before
their parent, so the parent is not the first entry in the map.

``collect_results`` used to take ``next(iter(groups_map.items()))`` and then
``continue`` out of the per-task loop. That recorded whichever group happened
to come first, publishing a subcategory score under the wrong task name,
and discarded every other group and task in the same file.

The loss is silent twice over: the aggregate the job asked for never reaches
the CSV, and ``--check`` then reports that evaluation as missing and invites
the user to re-run it, spending cluster time on work that already finished.
"""

import json
from pathlib import Path

import pandas as pd

from oellm.main import collect_results

MODEL = "/scratch/project_465002530/checkpoints/iter_0002000"
MMLU_SUBGROUPS = [
    "mmlu_humanities",
    "mmlu_other",
    "mmlu_social_sciences",
    "mmlu_stem",
]


def _metrics(**values: float) -> dict:
    """A result entry as lm-eval writes it, with its ``,none`` filter suffix."""
    entry: dict = {"alias": "task"}
    for name, value in values.items():
        entry[f"{name},none"] = value
        entry[f"{name}_stderr,none"] = 0.01
    return entry


def _write_case(tmp_path: Path, name: str, payload: dict, jobs: list[dict]) -> Path:
    case_dir = tmp_path / name
    (case_dir / "results").mkdir(parents=True)
    pd.DataFrame(jobs).to_csv(case_dir / "jobs.csv", index=False)
    (case_dir / "results" / "a1b2c3d4e5.json").write_text(json.dumps(payload))
    return case_dir


def _collect(case_dir: Path) -> pd.DataFrame:
    output_csv = case_dir / "eval_results.csv"
    collect_results(str(case_dir), str(output_csv), check=True)
    if not output_csv.exists():
        return pd.DataFrame(columns=["model_name", "task", "n_shot", "performance"])
    return pd.read_csv(output_csv)


def _missing(case_dir: Path) -> pd.DataFrame:
    missing_csv = case_dir / "eval_results_missing.csv"
    if not missing_csv.exists():
        return pd.DataFrame(columns=["model_path", "task_path", "n_shot"])
    return pd.read_csv(missing_csv)


def _mmlu_case(tmp_path: Path) -> Path:
    """`lm_eval --tasks mmlu`: subgroups aggregated before the parent."""
    results = {sub: _metrics(acc=0.30) for sub in MMLU_SUBGROUPS}
    results["mmlu"] = _metrics(acc=0.4123)
    results["mmlu_abstract_algebra"] = _metrics(acc=0.25)

    groups = {sub: results[sub] for sub in MMLU_SUBGROUPS}
    groups["mmlu"] = results["mmlu"]

    payload = {
        "model_name": MODEL,
        "results": results,
        "groups": groups,
        "group_subtasks": {
            **{sub: ["mmlu_abstract_algebra"] for sub in MMLU_SUBGROUPS},
            "mmlu": MMLU_SUBGROUPS,
        },
        "n-shot": {"mmlu_abstract_algebra": 5},
    }
    jobs = [{"model_path": MODEL, "task_path": "mmlu", "n_shot": 5}]
    return _write_case(tmp_path, "mmlu", payload, jobs)


def test_mmlu_reports_the_aggregate_not_a_subcategory(tmp_path: Path) -> None:
    df = _collect(_mmlu_case(tmp_path))

    assert list(df["task"]) == ["mmlu"]
    assert df.loc[0, "performance"] == 0.4123


def test_mmlu_is_not_reported_as_missing(tmp_path: Path) -> None:
    case_dir = _mmlu_case(tmp_path)
    _collect(case_dir)

    assert _missing(case_dir).empty


def test_every_top_level_group_is_collected(tmp_path: Path) -> None:
    """Two independent groups in one file: neither may be dropped."""
    results = {
        "arc_challenge": _metrics(acc=0.50, acc_norm=0.55),
        "hellaswag": _metrics(acc=0.60, acc_norm=0.66),
    }
    payload = {
        "model_name": MODEL,
        "results": results,
        "groups": dict(results),
        "group_subtasks": {"arc_challenge": [], "hellaswag": []},
        "n-shot": {"arc_challenge": 0, "hellaswag": 0},
    }
    jobs = [
        {"model_path": MODEL, "task_path": "arc_challenge", "n_shot": 0},
        {"model_path": MODEL, "task_path": "hellaswag", "n_shot": 0},
    ]
    case_dir = _write_case(tmp_path, "two_groups", payload, jobs)

    df = _collect(case_dir)

    assert dict(zip(df["task"], df["performance"], strict=True)) == {
        "arc_challenge": 0.55,
        "hellaswag": 0.66,
    }
    assert _missing(case_dir).empty


def test_a_plain_task_beside_a_group_survives(tmp_path: Path) -> None:
    """The per-task loop must still run when the file also holds a group."""
    payload = {
        "model_name": MODEL,
        "results": {
            "mmlu": _metrics(acc=0.41),
            "mmlu_humanities": _metrics(acc=0.30),
            "winogrande": _metrics(acc=0.72),
        },
        "groups": {
            "mmlu_humanities": _metrics(acc=0.30),
            "mmlu": _metrics(acc=0.41),
        },
        "group_subtasks": {"mmlu": ["mmlu_humanities"], "mmlu_humanities": []},
        "n-shot": {"mmlu": 5, "mmlu_humanities": 5, "winogrande": 5},
    }
    jobs = [
        {"model_path": MODEL, "task_path": "mmlu", "n_shot": 5},
        {"model_path": MODEL, "task_path": "winogrande", "n_shot": 5},
    ]
    case_dir = _write_case(tmp_path, "group_and_task", payload, jobs)

    df = _collect(case_dir)

    assert dict(zip(df["task"], df["performance"], strict=True)) == {
        "mmlu": 0.41,
        "winogrande": 0.72,
    }
    assert _missing(case_dir).empty


def test_results_without_groups_are_unaffected(tmp_path: Path) -> None:
    """Control: a lighteval-style file carries no `groups` map."""
    payload = {
        "config_general": {"model_name": MODEL},
        "results": {
            "belebele_eus_Latn_cf|0": {"acc_norm": 0.31},
            "all": {"acc_norm": 0.31},
        },
    }
    jobs = [{"model_path": MODEL, "task_path": "belebele_eus_Latn_cf", "n_shot": 0}]
    case_dir = _write_case(tmp_path, "lighteval", payload, jobs)

    df = _collect(case_dir)

    assert list(df["task"]) == ["belebele_eus_Latn_cf"]
    assert df.loc[0, "performance"] == 0.31
    assert _missing(case_dir).empty
