"""Task groups that leave out the documents a decontamination run marked.

A list (resources/decontaminated/<name>.json) gives, for each task, its
documents, the ones left out and the answers of the rest; schedule runs the
group's tasks with lm-eval's --samples, and collect recognises the results by
their file name and checks them against the list.
"""

import json
import logging
import os
import shutil
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


@pytest.fixture(autouse=True)
def _restore_root_logger():
    # collect_results and schedule_evals reconfigure the root logger.
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


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


def test_a_run_without_saved_samples_is_refused(tmp_path, toy_list):
    """Counts alone would pass a dataset reordered since the list was built."""
    run = _write_run(tmp_path, [0, 2, 4], name="toy-0.1", log_samples=False)
    with pytest.raises(ValueError, match="toy: no saved samples"):
        decontaminated.run_of(run, json.loads(run.read_text()))


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


def test_a_run_without_its_numbers_of_documents_is_refused(tmp_path, toy_list):
    run = _write_run(tmp_path, [0, 2, 4], name="toy-0.1")
    data = json.loads(run.read_text())
    del data["n-samples"]
    with pytest.raises(ValueError, match="records no number of documents"):
        decontaminated.run_of(run, data)


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


def test_check_misses_a_full_and_a_decontaminated_job_of_one_task(tmp_path, toy_list):
    runs = tmp_path / "runs"
    _write_run(runs / "other", [0, 1, 2, 3, 4], task="other")
    pd.DataFrame(
        [
            {"model_path": "m", "task_path": "toy", "n_shot": 0, "eval_suite": "lm_eval", "decontaminated": "toy-0.1"},
            {"model_path": "m", "task_path": "toy", "n_shot": 0, "eval_suite": "lm_eval", "decontaminated": ""},
        ]
    ).to_csv(runs / "jobs.csv", index=False)  # fmt: skip
    _collect(tmp_path, check=True)
    missing = pd.read_csv(tmp_path / "out_missing.csv", keep_default_na=False)
    assert sorted(missing["decontaminated"]) == ["", "toy-0.1"]


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
            **{"log_samples": True} | kwargs,
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


def test_a_decontaminated_group_saves_its_samples(tmp_path, toy_list):
    """collect checks a decontaminated run's documents from its samples; other
    groups keep sample logging opt-in."""
    for name, group in [("decon", GROUP), ("full", dict(GROUP, decontaminated=None))]:
        jobs = _schedule(tmp_path / name, {f"toy-{name}": group}, log_samples=False)
        script = jobs.with_name("submit_evals.sbatch").read_text()
        assert f'LOG_SAMPLES="{"true" if name == "decon" else ""}"' in script


def test_only_lm_eval_tasks_can_run_decontaminated(tmp_path, toy_list):
    group = dict(GROUP, suite="lighteval")
    with pytest.raises(
        ValueError, match="Only lm-eval tasks can run decontaminated: toy"
    ):
        _schedule(tmp_path, {"toy-decontaminated": group})


def test_a_decontaminated_group_needs_oellm_s_mmmlu_tasks(tmp_path, toy_list):
    """Without them, lm-eval's MMMLU high_school_mathematics would show marked
    documents as few-shot examples."""
    other = tmp_path / "tasks"
    other.mkdir()
    with pytest.raises(ValueError, match="lacks oellm's MMMLU tasks"):
        _schedule(
            tmp_path, {"toy-decontaminated": GROUP}, lm_eval_include_path=str(other)
        )
    bundled = Path(decontaminated.__file__).parent / "resources/custom_lm_eval_tasks"
    shutil.copytree(bundled / "mmmlu_decontaminated", other / "mmmlu")
    (other / "mmmlu/utils.py").write_text("")  # another version
    with pytest.raises(ValueError, match="lacks oellm's MMMLU tasks"):
        _schedule(
            tmp_path, {"toy-decontaminated": GROUP}, lm_eval_include_path=str(other)
        )
    shutil.copy(bundled / "mmmlu_decontaminated/utils.py", other / "mmmlu/utils.py")
    _schedule(tmp_path, {"toy-decontaminated": GROUP}, lm_eval_include_path=str(other))


def test_a_group_naming_an_unknown_list_is_refused(tmp_path, toy_list):
    with pytest.raises(ValueError, match="named 'toy-9'"):
        _schedule(tmp_path, {"toy-decontaminated": dict(GROUP, decontaminated="toy-9")})


# --- the bundled lists and groups ----------------------------------------------


def _groups():
    from oellm.task_groups import _load_task_groups_data

    return _load_task_groups_data()["task_groups"]


DECONTAMINATED_GROUPS = {
    name: data for name, data in _groups().items() if data.get("decontaminated")
}


@pytest.mark.parametrize("name", decontaminated.names())
def test_a_list_is_well_formed(name):
    tasks = decontaminated.load(name)["tasks"]
    assert tasks
    for task, entry in tasks.items():
        assert set(entry) == {"documents", "excluded", "kept_answers"}, task
        excluded = entry["excluded"]
        assert excluded == sorted(set(excluded)), task
        assert all(0 <= i < entry["documents"] for i in excluded), task
        assert 0 < len(excluded) < entry["documents"], task
        assert len(entry["kept_answers"]) == 16, task


def test_every_list_has_a_group_and_every_group_a_list():
    used = {data["decontaminated"] for data in DECONTAMINATED_GROUPS.values()}
    assert used == set(decontaminated.names())


@pytest.mark.parametrize("group", sorted(DECONTAMINATED_GROUPS))
def test_a_group_holds_exactly_the_tasks_its_list_leaves_documents_out_of(group):
    """Each of the list's tasks is a group task or a subtask of one (MMLU's
    subjects), and each group task loses documents."""
    data = DECONTAMINATED_GROUPS[group]
    listed = decontaminated.load(data["decontaminated"])["tasks"]
    names = [t["task"] for t in data["tasks"]]

    def holds(name, task):
        return task == name or task.startswith(f"{name}_")

    assert all(any(holds(n, task) for n in names) for task in listed)
    assert all(any(holds(n, task) for task in listed) for n in names)


@pytest.mark.parametrize("group", sorted(DECONTAMINATED_GROUPS))
def test_a_group_runs_its_tasks_as_their_other_groups_do(group):
    """The same shots, dataset and subset, so its scores compare with the full
    ones."""
    others = {}
    for data in _groups().values():
        if data.get("decontaminated"):
            continue
        for t in data["tasks"]:
            others.setdefault(t["task"], []).append(
                (
                    t.get("n_shots") or data.get("n_shots"),
                    t.get("dataset") or data.get("dataset"),
                    t.get("subset", data.get("subset")),
                )
            )
    for t in DECONTAMINATED_GROUPS[group]["tasks"]:
        assert t["task"] in others, t["task"]
        shots_and_datasets = {(str(s), d) for s, d, _ in others[t["task"]]}
        assert (str(t["n_shots"]), t["dataset"]) in shots_and_datasets, t["task"]
        subsets = {sub for *_, sub in others[t["task"]]}
        assert t.get("subset") in subsets or None in subsets, t["task"]


# --- MMMLU's few-shot examples -------------------------------------------------


def _mmmlu_utils():
    import importlib.util

    path = (
        Path(decontaminated.__file__).parent
        / "resources/custom_lm_eval_tasks/mmmlu_decontaminated/utils.py"
    )
    spec = importlib.util.spec_from_file_location("mmmlu_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mmmlu_split():
    from datasets import Dataset

    subjects = ["high_school_mathematics", "anatomy"] * 5
    # MMMLU's Swahili config embeds CSV file names in its Subject column.
    subjects[2] = "high_school_mathematics_test.csv_sw-KE.csv"
    return Dataset.from_dict(
        {"Question": [f"q{i}" for i in range(10)], "Subject": subjects}
    )


def test_mmmlu_few_shot_examples_are_lm_eval_s_outside_a_decontaminated_run(
    monkeypatch,
):
    utils = _mmmlu_utils()
    monkeypatch.delenv(utils.SAMPLES, raising=False)
    docs = utils.fewshot_de_de(_mmmlu_split())
    assert docs["Question"] == ["q0", "q2", "q4", "q6", "q8"]
    assert (
        utils.process_high_school_mathematics(_mmmlu_split())["Question"]
        == docs["Question"]
    )


def test_mmmlu_few_shot_examples_are_the_kept_documents_in_a_decontaminated_run(
    tmp_path, monkeypatch
):
    """The marked second question is no example; the next one takes its place."""
    utils = _mmmlu_utils()
    samples = tmp_path / "samples-toy.json"
    samples.write_text(json.dumps({"mmmlu_de_de_high_school_mathematics": [0, 2, 3, 4]}))
    monkeypatch.setenv(utils.SAMPLES, str(samples))
    assert utils.fewshot_de_de(_mmmlu_split())["Question"] == ["q0", "q4", "q6", "q8"]
    # A task the run does not leave documents out of keeps all.
    assert utils.fewshot_fr_fr(_mmmlu_split())["Question"] == [
        "q0",
        "q2",
        "q4",
        "q6",
        "q8",
    ]


def test_mmmlu_overrides_cover_the_tasks_whose_examples_are_marked():
    """In the bundled list, only high_school_mathematics has a marked document
    among MMMLU's first examples (5 shots, 6 for the first documents)."""
    from oellm.task_groups import _load_task_groups_data

    shots = {
        t["task"]: (t.get("n_shots") or data.get("n_shots"))[0]
        for data in _load_task_groups_data()["task_groups"].values()
        for t in data["tasks"]
        if t["task"].startswith("mmmlu_")
    }
    directory = (
        Path(decontaminated.__file__).parent
        / "resources/custom_lm_eval_tasks/mmmlu_decontaminated"
    )
    overridden = {p.stem for p in directory.glob("mmmlu_*.yaml")}
    for name in decontaminated.names():
        for task, entry in decontaminated.load(name)["tasks"].items():
            group = next((g for g in shots if task.startswith(f"{g}_")), None)
            if group and min(entry["excluded"]) <= shots[group]:
                assert task in overridden, task


def test_the_bundled_tasks_leave_lm_eval_s_groups_and_tags_as_they_were():
    """lm-eval replaces a group or tag of its own that the include path also
    defines, members and all: MMMLU's overrides must stay in their subjects'
    categories without defining them. Needs lm-eval, e.g. in the eval
    container."""
    pytest.importorskip("lm_eval")
    import lm_eval.tasks
    from lm_eval.tasks._index import Kind, TaskIndex

    bundled = Path(decontaminated.__file__).parent / "resources/custom_lm_eval_tasks"
    theirs = TaskIndex.build([Path(lm_eval.tasks.__file__).parent])
    ours = TaskIndex.build([Path(lm_eval.tasks.__file__).parent, bundled])
    changed = {
        name
        for name, entry in theirs.items()
        if entry.kind in (Kind.GROUP, Kind.TAG)
        and (ours[name].tags, ours[name].yaml_path) != (entry.tags, entry.yaml_path)
    }
    # oellm's ARC languages (arc_mt/) and bigbench generate_until tasks (#112).
    assert changed == {"arc_challenge_mt", "bigbench_generate_until"}
    for path in (bundled / "mmmlu_decontaminated").glob("mmmlu_*.yaml"):
        lang = path.stem.removeprefix("mmmlu_").removesuffix("_high_school_mathematics")
        assert Path(ours[path.stem].yaml_path).resolve() == path.resolve()
        assert path.stem in ours[f"mmmlu_{lang}_stem_tasks"].tags
