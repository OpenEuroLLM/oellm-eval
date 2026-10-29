"""Task groups that leave out the documents a decontamination run marked.

A list (resources/decontaminated/<name>.json) gives, for each task, its
documents, the ones left out and the answers of the rest; schedule runs the
group's tasks with lm-eval's --samples, and collect recognises the results by
their file name and checks them against the list.
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from oellm import decontaminated
from oellm.main import collect_results, schedule_evals
from oellm.sample_exclusion import item_key
from oellm.task_groups import TaskGroup, _expand_task_groups

TS = "2026-10-29T10-00-00.000000"
ANSWERS = ["0", "1", "2", "3", "0"]
TOY = {
    "name": "toy-0.1",
    "tasks": {
        "toy": {
            "documents": 5,
            "excluded": [1, 3],
            "kept_answers": item_key([ANSWERS[0], ANSWERS[2], ANSWERS[4]]),
        }
    },
}


@pytest.fixture
def toy_list(monkeypatch):
    def load(name):
        if name != "toy-0.1":
            raise ValueError(f"No list of decontaminated samples named {name!r}")
        return TOY

    monkeypatch.setattr(decontaminated, "load", load)
    monkeypatch.setattr(decontaminated, "names", lambda: ["toy-0.1"])


def test_the_samples_run_are_every_document_not_left_out(toy_list):
    assert decontaminated.samples("toy-0.1") == {"toy": [0, 2, 4]}


def _write_run(directory, doc_ids, original=5, name="", log_samples=True, task="toy"):
    """lm-eval's results for *task*, run on *doc_ids* of its *original*
    documents, named as the job template names a decontaminated run's."""
    directory.mkdir(parents=True, exist_ok=True)
    accs = [1.0 if i % 2 == 0 else 0.0 for i in doc_ids]
    data = {
        "results": {task: {"alias": task, "acc,none": sum(accs) / len(accs)}},
        "groups": {},
        "group_subtasks": {task: []},
        "configs": {task: {"output_type": "multiple_choice"}},
        "n-shot": {task: 0},
        "n-samples": {task: {"original": original, "effective": len(doc_ids)}},
        "config": {"limit": None},
        "model_name": "m",
    }
    run = "3fa9c1d2e0" + (f"{decontaminated.MARK}{name}" if name else "")
    results_file = directory / f"{run}_{TS}.json"
    results_file.write_text(json.dumps(data))
    if log_samples:
        (directory / f"samples_{task}_{TS}.jsonl").write_text(
            "".join(
                json.dumps(
                    {"doc_id": i, "doc": {"q": f"q{i}"}, "target": ANSWERS[i], "filter": "none", "acc": a}
                )
                + "\n"
                for i, a in zip(doc_ids, accs, strict=True)
            )
        )  # fmt: skip
    return results_file


# --- recognising a decontaminated run ------------------------------------------


def test_a_run_on_whole_tasks_is_no_decontaminated_one(tmp_path, toy_list):
    run = _write_run(tmp_path, [0, 1, 2, 3, 4])
    assert decontaminated.run_of(run, json.loads(run.read_text())) is None


def test_a_decontaminated_run_is_known_by_its_file_name(tmp_path, toy_list):
    run = _write_run(tmp_path, [0, 2, 4], name="toy-0.1")
    assert decontaminated.run_of(run, json.loads(run.read_text())) == "toy-0.1"


def test_a_run_without_saved_samples_is_checked_by_its_counts(tmp_path, toy_list):
    run = _write_run(tmp_path, [0, 2, 4], name="toy-0.1", log_samples=False)
    assert decontaminated.run_of(run, json.loads(run.read_text())) == "toy-0.1"


@pytest.mark.parametrize(
    "doc_ids, original",
    [
        ([0, 1, 2, 3, 4], 5),  # ran on every document
        ([0, 2, 4], 6),  # a task with another number of documents
    ],
)
def test_a_run_on_other_documents_than_the_list_s_is_refused(
    tmp_path, toy_list, doc_ids, original
):
    run = _write_run(tmp_path, doc_ids, original, name="toy-0.1", log_samples=False)
    with pytest.raises(ValueError, match="the decontaminated list toy-0.1 keeps"):
        decontaminated.run_of(run, json.loads(run.read_text()))


def test_saved_samples_must_be_the_documents_the_list_keeps(tmp_path, toy_list):
    run = _write_run(tmp_path, [0, 1, 4], name="toy-0.1")
    with pytest.raises(ValueError, match="not the ones the decontaminated list"):
        decontaminated.run_of(run, json.loads(run.read_text()))


def test_saved_samples_must_have_the_answers_the_list_keeps(tmp_path, toy_list):
    """A dataset reordered since the list was built would leave out other
    documents than the marked ones."""
    run = _write_run(tmp_path, [0, 2, 4], name="toy-0.1")
    samples = run.with_name(f"samples_toy_{TS}.jsonl")
    lines = [json.loads(x) for x in samples.read_text().splitlines()]
    lines[1]["target"] = "3"
    samples.write_text("".join(json.dumps(x) + "\n" for x in lines))
    with pytest.raises(ValueError, match="not the ones the decontaminated list"):
        decontaminated.run_of(run, json.loads(run.read_text()))


def test_a_task_the_list_does_not_hold_runs_whole(tmp_path, toy_list):
    run = _write_run(tmp_path, [0, 1, 2], original=4, name="toy-0.1", task="other")
    with pytest.raises(ValueError, match="other: .* ran on 3 of 4 documents"):
        decontaminated.run_of(run, json.loads(run.read_text()))


def test_a_run_of_an_unknown_list_is_refused(tmp_path, toy_list):
    run = _write_run(tmp_path, [0, 2, 4], name="toy-9")
    with pytest.raises(
        ValueError, match="No list of decontaminated samples named 'toy-9'"
    ):
        decontaminated.run_of(run, json.loads(run.read_text()))


# --- collect -------------------------------------------------------------------


def _collect(tmp_path, **kwargs):
    out = tmp_path / "out.csv"
    collect_results(str(tmp_path / "runs"), str(out), fetch_all_metrics=True, **kwargs)
    return pd.read_csv(out)


def test_collect_reports_a_decontaminated_run_next_to_the_full_one(tmp_path, toy_list):
    _write_run(tmp_path / "runs/full", [0, 1, 2, 3, 4])
    _write_run(tmp_path / "runs/decon", [0, 2, 4], name="toy-0.1")
    df = _collect(tmp_path).set_index("subset")
    assert df.loc["full", "performance"] == pytest.approx(3 / 5)
    assert df.loc["decontaminated:toy-0.1", "performance"] == 1.0
    assert df.loc["decontaminated:toy-0.1", "n_excluded"] == 2
    assert df.loc["full", "n_excluded"] == 0


def test_collect_skips_a_decontaminated_run_that_does_not_match_its_list(
    tmp_path, toy_list, capfd
):
    _write_run(tmp_path / "runs/full", [0, 1, 2, 3, 4])
    _write_run(tmp_path / "runs/decon", [0, 1, 4], name="toy-0.1")
    df = _collect(tmp_path)
    assert list(df["performance"]) == [pytest.approx(3 / 5)]
    # collect logs through Rich, which wraps lines.
    logged = " ".join(capfd.readouterr().out.split())
    assert "a decontaminated run" in logged


def test_with_a_blocklist_a_decontaminated_run_keeps_its_own_scores(
    tmp_path, toy_list, monkeypatch
):
    """--exclude_samples recomputes full runs only; a decontaminated run has
    already left its marked documents out."""
    _write_run(tmp_path / "runs/full", [0, 1, 2, 3, 4])
    _write_run(tmp_path / "runs/decon", [0, 2, 4], name="toy-0.1")
    blocklist = tmp_path / "blocklist"
    blocklist.mkdir()
    (blocklist / "Toy.jsonl").write_text(json.dumps({"q": "q1"}) + "\n")
    from oellm import sample_exclusion

    monkeypatch.setattr(sample_exclusion, "load_task_map", lambda: {"Toy": ["toy"]})
    df = _collect(tmp_path, exclude_samples=str(blocklist)).set_index("subset")
    assert df.loc["full", "performance"] == pytest.approx(3 / 5)
    assert df.loc["clean", "performance"] == pytest.approx(3 / 4)
    assert df.loc["decontaminated:toy-0.1", "performance"] == 1.0


def test_check_tells_a_decontaminated_job_from_a_full_one(tmp_path, toy_list):
    runs = tmp_path / "runs"
    _write_run(runs / "full", [0, 1, 2, 3, 4])
    pd.DataFrame(
        [
            {"model_path": "m", "task_path": "toy", "n_shot": 0, "eval_suite": "lm_eval", "decontaminated": ""},
            {"model_path": "m", "task_path": "toy", "n_shot": 0, "eval_suite": "lm_eval", "decontaminated": "toy-0.1"},
        ]
    ).to_csv(runs / "jobs.csv", index=False)  # fmt: skip
    _collect(tmp_path, check=True)
    missing = pd.read_csv(tmp_path / "out_missing.csv")
    assert list(missing["decontaminated"]) == ["toy-0.1"]


# --- schedule ------------------------------------------------------------------

GROUP = {
    "description": "toy",
    "suite": "lm-eval-harness",
    "decontaminated": "toy-0.1",
    "n_shots": [0],
    "tasks": [{"task": "toy"}],
}


def test_a_decontaminated_group_s_tasks_carry_its_list():
    group = TaskGroup.from_dict("toy-decontaminated", GROUP)
    assert [t.decontaminated for t in group.tasks] == ["toy-0.1"]


def test_the_all_super_group_leaves_decontaminated_groups_out():
    data = {
        "task_groups": {
            "toy-full": dict(GROUP, decontaminated=None),
            "toy-decontaminated": GROUP,
        }
    }
    with patch("oellm.task_groups._load_task_groups_data", return_value=data):
        expanded = _expand_task_groups(["all"])
    assert [(r.task, r.decontaminated) for r in expanded] == [("toy", None)]


def _schedule(tmp_path, groups, **kwargs):
    data = {"task_groups": groups}
    with (
        patch("oellm.task_groups._load_task_groups_data", return_value=data),
        patch("oellm.main._load_cluster_env"),
        patch("oellm.main._num_jobs_in_queue", return_value=0),
        patch.dict(
            os.environ,
            {"EVAL_OUTPUT_DIR": str(tmp_path), "HF_HOME": str(tmp_path / "hf")},
        ),
    ):
        schedule_evals(
            models="EleutherAI/pythia-70m",
            task_groups=",".join(groups),
            skip_checks=True,
            venv_path=str(Path(sys.prefix)),
            dry_run=True,
            **kwargs,
        )
    return next(iter(tmp_path.glob("**/jobs.csv")))


def test_a_task_in_a_full_and_a_decontaminated_group_is_scheduled_twice():
    data = {
        "task_groups": {
            "toy-full": dict(GROUP, decontaminated=None),
            "toy-decontaminated": GROUP,
        }
    }
    with patch("oellm.task_groups._load_task_groups_data", return_value=data):
        expanded = _expand_task_groups(["toy-full", "toy-decontaminated"])
    assert [(r.task, r.decontaminated) for r in expanded] == [
        ("toy", None),
        ("toy", "toy-0.1"),
    ]


def test_schedule_runs_a_decontaminated_group_with_its_samples(tmp_path, toy_list):
    jobs = _schedule(tmp_path, {"toy-decontaminated": GROUP})
    df = pd.read_csv(jobs, keep_default_na=False)
    assert list(df["decontaminated"]) == ["toy-0.1"]
    samples = jobs.with_name("samples-toy-0.1.json")
    assert json.loads(samples.read_text()) == {"toy": [0, 2, 4]}


def test_schedule_keeps_jobs_csv_as_it_was_without_a_decontaminated_group(tmp_path):
    jobs = _schedule(tmp_path, {"toy-full": dict(GROUP, decontaminated=None)})
    assert list(pd.read_csv(jobs).columns) == [
        "model_path",
        "task_path",
        "n_shot",
        "eval_suite",
    ]


def test_a_decontaminated_group_cannot_run_with_a_limit(tmp_path, toy_list):
    with pytest.raises(ValueError, match="cannot be combined with --limit"):
        _schedule(tmp_path, {"toy-decontaminated": GROUP}, limit=10)


def test_only_lm_eval_tasks_can_run_decontaminated(tmp_path, toy_list):
    group = dict(GROUP, suite="lighteval")
    with pytest.raises(
        ValueError, match="Only lm-eval tasks can run decontaminated: toy"
    ):
        _schedule(tmp_path, {"toy-decontaminated": group})


def test_a_group_naming_an_unknown_list_is_refused(tmp_path, toy_list):
    with pytest.raises(ValueError, match="named 'toy-9'"):
        _schedule(tmp_path, {"toy-decontaminated": dict(GROUP, decontaminated="toy-9")})
