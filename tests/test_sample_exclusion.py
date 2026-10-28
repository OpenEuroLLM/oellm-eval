"""Scores recomputed without blocklisted samples (``collect --exclude_samples``).

The runs here are written in lm-eval's output layout: ``results_<ts>.json``
with ``samples_<task>_<ts>.jsonl`` next to it. Their reported scores are
computed with the ``statistics`` module, independently of the code under test,
so a recomputation that drifts from lm-eval's formulas fails the self-check.
"""

import json
import logging
import math
import os
import statistics
from pathlib import Path

import pandas as pd
import pytest

from oellm import sample_exclusion
from oellm.main import collect_results
from oellm.sample_exclusion import RunNotFilterable, SampleExclusionError, clean_results
from oellm.task_groups import _load_task_groups_data

TS = "2026-10-27T10-00-00.000000"
# A real blocklist, e.g. agenttrove-0.0_filtered/dropped on LUMI.
BLOCKLIST_DIR = Path(os.environ.get("OELLM_BLOCKLIST_DIR", "/nonexistent"))


@pytest.fixture(autouse=True)
def _restore_root_logger():
    # collect_results reconfigures the root logger.
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


def _docs(n, prefix="q"):
    return [{"question": f"{prefix}{i}", "answer": i % 3} for i in range(n)]


def _score(values):
    return statistics.mean(values), statistics.stdev(values) / math.sqrt(len(values))


def _task(docs, scores, metric="acc", output_type="multiple_choice", aggregation="mean"):
    """One task: *scores* maps a filter name to per-document values."""
    result, samples = {"alias": "", "sample_len": len(docs)}, []
    for filter_name, values in scores.items():
        mean, stderr = _score(values)
        result[f"{metric},{filter_name}"] = mean
        result[f"{metric}_stderr,{filter_name}"] = stderr
        samples += [
            {"doc_id": i, "doc": doc, "filter": filter_name, metric: value}
            for i, (doc, value) in enumerate(zip(docs, values, strict=True))
        ]
    config = {
        "output_type": output_type,
        "metric_list": [{"metric": metric, "aggregation": aggregation}],
    }
    return {"result": result, "samples": samples, "config": config, "n": len(docs)}


def _write_run(
    run_dir, tasks, groups=None, group_subtasks=None, limit=None, name="results"
):
    """Write lm-eval's results file and sample files; return the results path.

    lm-eval names the results file results_<date>.json when --output_path is a
    directory and <name>_<date>.json when it is <name>.json, as oellm passes it.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    group_subtasks = group_subtasks or {name: [] for name in tasks}
    data = {
        "results": {name: t["result"] for name, t in tasks.items()} | (groups or {}),
        "groups": groups or {},
        "group_subtasks": group_subtasks,
        "configs": {name: t["config"] for name, t in tasks.items()},
        "n-shot": dict.fromkeys(tasks, 0),
        "n-samples": {
            name: {"original": t["n"], "effective": limit or t["n"]}
            for name, t in tasks.items()
        },
        "config": {"limit": limit, "samples": None},
        "model_name": "m",
    }
    results_file = run_dir / f"{name}_{TS}.json"
    results_file.write_text(json.dumps(data))
    for name, t in tasks.items():
        (run_dir / f"samples_{name}_{TS}.jsonl").write_text(
            "".join(json.dumps(s) + "\n" for s in t["samples"])
        )
    return results_file


def _clean(results_file, blocklist, task_map=None):
    data = json.loads(results_file.read_text())
    return clean_results(results_file, data, blocklist, task_map or {"Toy": ["toy"]})


# --- recomputing a task ------------------------------------------------------


def test_excluding_samples_gives_the_hand_computed_mean_and_stderr(tmp_path):
    docs = _docs(6)
    run = _write_run(tmp_path, {"toy": _task(docs, {"none": [1, 0, 1, 1, 0, 1]})})
    clean, n_excluded = _clean(run, {"Toy": [docs[0], docs[2]]})["toy"]
    # Left: 0, 1, 0, 1 -> mean 0.5, stdev sqrt(1/3), stderr sqrt(1/3) / 2.
    assert n_excluded == 2
    assert clean["acc,none"] == 0.5
    assert clean["acc_stderr,none"] == pytest.approx(math.sqrt(1 / 3) / 2, rel=1e-12)
    assert clean["sample_len"] == 4


@pytest.mark.parametrize("name", ["results", "3fa9c1d2e0", "out"])
def test_samples_are_found_for_either_results_file_name(tmp_path, name):
    """oellm's job template passes --output_path <run id>.json, so lm-eval
    writes <run id>_<date>.json next to samples_<task>_<date>.jsonl."""
    docs = _docs(4)
    run = _write_run(tmp_path, {"toy": _task(docs, {"none": [1, 0, 1, 0]})}, name=name)
    assert run.name == f"{name}_{TS}.json"
    assert _clean(run, {"Toy": [docs[0]]})["toy"][0]["acc,none"] == pytest.approx(1 / 3)


def test_nothing_excluded_reproduces_the_reported_scores(tmp_path):
    task = _task(_docs(7), {"none": [1, 0, 0, 1, 1, 1, 0]})
    clean = sample_exclusion._task_scores(
        "toy", task["result"], task["config"], task["samples"], set()
    )
    for key in ("acc,none", "acc_stderr,none"):
        assert clean[key] == pytest.approx(task["result"][key], rel=1e-12)


@pytest.mark.parametrize(
    "source",
    [
        "def pass_at_1(\n    references, predictions\n) -> float:\n    ...\n",
        "@functools.cache\ndef pass_at_1(references, predictions):\n    ...\n",
    ],
)
def test_metric_function_sources_are_recognised(source):
    config = {"metric_list": [{"metric": source, "aggregation": "mean"}]}
    assert sample_exclusion._is_mean_metric("pass_at_1", config)
    assert not sample_exclusion._is_mean_metric("pass_at", config)


@pytest.mark.parametrize("key", ["acc,none", "acc_stderr,none"])
def test_samples_that_do_not_reproduce_the_reported_score_are_refused(tmp_path, key):
    docs = _docs(5)
    task = _task(docs, {"none": [1, 0, 1, 1, 0]})
    task["result"][key] += 0.01
    run = _write_run(tmp_path, {"toy": task})
    with pytest.raises(SampleExclusionError, match="reported"):
        _clean(run, {"Toy": [docs[0]]})


def test_a_document_is_excluded_from_every_filter(tmp_path):
    """gsm8k logs each document once per filter (strict-match, flexible-extract)."""
    docs = _docs(5)
    task = _task(
        docs,
        {"strict-match": [1, 0, 0, 1, 0], "flexible-extract": [1, 1, 0, 1, 1]},
        metric="exact_match",
        output_type="generate_until",
    )
    run = _write_run(tmp_path, {"toy": task})
    clean, n_excluded = _clean(run, {"Toy": [docs[0]]})["toy"]
    assert n_excluded == 1
    assert clean["exact_match,strict-match"] == 0.25
    assert clean["exact_match,flexible-extract"] == 0.75


def test_metrics_that_are_not_means_are_not_recomputed(tmp_path):
    docs = _docs(5)
    task = _task(docs, {"none": [0.1, 0.4, 0.2, 0.3, 0.5]}, metric="f1", aggregation="f1")
    run = _write_run(tmp_path, {"toy": task})
    clean, _ = _clean(run, {"Toy": [docs[0]]})["toy"]
    assert "f1,none" not in clean


def test_a_metric_given_as_a_function_is_recomputed(tmp_path):
    """lm-eval saves `metric: !function utils.pass_at_1` (mbpp) as source code."""
    docs = _docs(5)
    task = _task(
        docs, {"none": [1, 0, 0, 1, 1]}, metric="pass_at_1", output_type="generate_until"
    )
    task["config"]["metric_list"][0]["metric"] = (
        "def pass_at_1(\n    references, predictions\n) -> float:\n    ...\n"
    )
    run = _write_run(tmp_path, {"toy": task})
    clean, _ = _clean(run, {"Toy": [docs[0]]})["toy"]
    assert clean["pass_at_1,none"] == 0.5


def test_too_few_documents_left_is_refused(tmp_path):
    docs = _docs(3)
    run = _write_run(tmp_path, {"toy": _task(docs, {"none": [1, 0, 1]})})
    with pytest.raises(SampleExclusionError, match="1 documents remain"):
        _clean(run, {"Toy": docs[:2]})


# --- matching rows to documents ---------------------------------------------


def test_fields_the_task_added_to_its_documents_do_not_prevent_a_match(tmp_path):
    docs = [dict(d, query=f"Q: {d['question']}") for d in _docs(4)]
    run = _write_run(tmp_path, {"toy": _task(docs, {"none": [1, 0, 1, 0]})})
    row = {"question": "q1", "answer": 1}
    assert _clean(run, {"Toy": [row]})["toy"][1] == 1


@pytest.mark.parametrize(
    "row",
    [
        {"question": "q1", "answer": 2},  # different value
        {"question": "q1", "answer": "1"},  # different type
        {"question": "q1 ", "answer": 1},  # whitespace
        {"question": "q1", "answer": 1, "source": "x"},  # field the docs lack
    ],
)
def test_a_row_that_matches_no_document_is_an_error(tmp_path, row):
    run = _write_run(tmp_path, {"toy": _task(_docs(4), {"none": [1, 0, 1, 0]})})
    with pytest.raises(SampleExclusionError, match="matches no document"):
        _clean(run, {"Toy": [row]})


def test_annotation_fields_of_the_blocklist_are_ignored(tmp_path):
    (tmp_path / "Toy.jsonl").write_text(
        json.dumps({"question": "q1", "answer": 1, "_matched_ngrams": "['q1']"}) + "\n\n"
    )
    assert sample_exclusion.load_blocklist(tmp_path) == {
        "Toy": [{"question": "q1", "answer": 1}]
    }


def test_a_row_with_only_annotations_is_refused(tmp_path):
    """Without its `_` fields it would match, and exclude, every document."""
    (tmp_path / "Toy.jsonl").write_text(
        json.dumps({"question": "q1"})
        + "\n"
        + json.dumps({"_matched_ngrams": "x"})
        + "\n"
    )
    with pytest.raises(SampleExclusionError, match="only `_` annotation fields"):
        sample_exclusion.load_blocklist(tmp_path)


def test_identical_documents_are_all_excluded(tmp_path):
    docs = _docs(4) + [{"question": "q1", "answer": 1}]
    run = _write_run(tmp_path, {"toy": _task(docs, {"none": [1, 1, 0, 0, 1]})})
    clean, n_excluded = _clean(run, {"Toy": [{"question": "q1", "answer": 1}]})["toy"]
    assert n_excluded == 2
    assert clean["acc,none"] == pytest.approx(1 / 3)


# --- runs that cannot be filtered -------------------------------------------


def test_a_limit_run_is_refused(tmp_path):
    docs = _docs(5)
    run = _write_run(tmp_path, {"toy": _task(docs, {"none": [1, 0, 1, 1, 0]})}, limit=5)
    data = json.loads(run.read_text())
    data["n-samples"]["toy"]["original"] = 100
    run.write_text(json.dumps(data))
    with pytest.raises(RunNotFilterable, match="5 of 100 documents"):
        _clean(run, {"Toy": [docs[0]]})


def test_a_run_on_selected_samples_is_refused(tmp_path):
    docs = _docs(5)
    run = _write_run(tmp_path, {"toy": _task(docs, {"none": [1, 0, 1, 1, 0]})})
    data = json.loads(run.read_text())
    data["config"]["samples"] = {"toy": [0, 1, 2, 3, 4]}
    run.write_text(json.dumps(data))
    with pytest.raises(RunNotFilterable, match="--samples"):
        _clean(run, {"Toy": [docs[0]]})


def test_a_run_without_saved_samples_is_refused(tmp_path):
    docs = _docs(5)
    run = _write_run(tmp_path, {"toy": _task(docs, {"none": [1, 0, 1, 1, 0]})})
    (tmp_path / f"samples_toy_{TS}.jsonl").unlink()
    with pytest.raises(RunNotFilterable, match="--log_samples"):
        _clean(run, {"Toy": [docs[0]]})


def test_a_run_without_document_counts_is_refused(tmp_path):
    docs = _docs(5)
    run = _write_run(tmp_path, {"toy": _task(docs, {"none": [1, 0, 1, 1, 0]})})
    data = json.loads(run.read_text())
    del data["n-samples"]
    run.write_text(json.dumps(data))
    with pytest.raises(RunNotFilterable, match="how many documents"):
        _clean(run, {"Toy": [docs[0]]})


def test_tasks_without_blocklisted_rows_are_left_alone(tmp_path):
    """No samples are read for them, so runs without --log_samples still collect."""
    run = _write_run(tmp_path, {"toy": _task(_docs(4), {"none": [1, 0, 1, 0]})})
    (tmp_path / f"samples_toy_{TS}.jsonl").unlink()
    assert _clean(run, {"Other": [{"question": "q1", "answer": 1}]}) == {}
    assert _clean(run, {"Toy": []}) == {}


# --- groups ------------------------------------------------------------------


def _group_run(tmp_path, weighted=True, sizes=(4, 6)):
    """A two-level group like mmlu -> mmlu_stem -> subjects."""
    a = _task(_docs(sizes[0], "a"), {"none": [1, 0, 1, 1, 0, 1][: sizes[0]]})
    b = _task(_docs(sizes[1], "b"), {"none": [0, 0, 1, 0, 1, 1][: sizes[1]]})
    means = [a["result"]["acc,none"], b["result"]["acc,none"]]
    stderrs = [a["result"]["acc_stderr,none"], b["result"]["acc_stderr,none"]]
    if weighted:
        mean = sum(m * n for m, n in zip(means, sizes, strict=True)) / sum(sizes)
        stderr = _pooled(stderrs, sizes)
    else:
        mean = statistics.mean(means)
        stderr = math.sqrt(sum(se**2 for se in stderrs)) / len(stderrs)
    group = {"acc,none": mean, "acc_stderr,none": stderr, "sample_len": sum(sizes)}
    run = _write_run(
        tmp_path,
        {"toy_a": a, "toy_b": b},
        groups={"toy": group, "toy_sub": dict(group)},
        group_subtasks={"toy": ["toy_sub"], "toy_sub": ["toy_a", "toy_b"]},
    )
    return run, a, b


def _pooled(stderrs, sizes):
    # lm-eval's pooled_sample_stderr, written out.
    var = sum((n - 1) * se**2 * n for n, se in zip(sizes, stderrs, strict=True)) / (
        sum(sizes) - len(sizes)
    )
    return math.sqrt(var / sum(sizes))


def test_a_size_weighted_group_is_the_mean_over_its_remaining_documents(tmp_path):
    run, a, b = _group_run(tmp_path)
    blocked = [a["samples"][0]["doc"], b["samples"][2]["doc"]]  # values 1 and 1
    out = _clean(run, {"Toy": blocked})
    clean, n_excluded = out["toy"]
    left = [0, 1, 1] + [0, 0, 0, 1, 1]
    assert n_excluded == 2
    assert clean["acc,none"] == pytest.approx(statistics.mean(left), rel=1e-12)
    assert clean["acc_stderr,none"] == pytest.approx(
        _pooled([_score([0, 1, 1])[1], _score([0, 0, 0, 1, 1])[1]], [3, 5]), rel=1e-12
    )
    assert out["toy_sub"][0]["acc,none"] == clean["acc,none"]
    assert set(out) == {"toy", "toy_sub", "toy_a", "toy_b"}


def test_an_unweighted_group_is_the_mean_of_its_tasks(tmp_path):
    run, a, _ = _group_run(tmp_path, weighted=False)
    clean, _ = _clean(run, {"Toy": [a["samples"][0]["doc"]]})["toy"]
    b_mean = statistics.mean([0, 0, 1, 0, 1, 1])
    assert clean["acc,none"] == pytest.approx((2 / 3 + b_mean) / 2, rel=1e-12)


def test_a_group_of_equal_scores_is_told_apart_by_its_stderr(tmp_path):
    """With equal subtask scores both weightings give the reported score, but
    for subtasks of different sizes only one gives the reported stderr."""
    a = _task(_docs(4, "a"), {"none": [1, 0, 1, 0]})
    b = _task(_docs(6, "b"), {"none": [1, 0, 1, 0, 1, 0]})
    stderrs = [a["result"]["acc_stderr,none"], b["result"]["acc_stderr,none"]]
    group = {"acc,none": 0.5, "acc_stderr,none": _pooled(stderrs, [4, 6])}
    run = _write_run(
        tmp_path,
        {"toy_a": a, "toy_b": b},
        groups={"toy": group},
        group_subtasks={"toy": ["toy_a", "toy_b"]},
    )
    clean, _ = _clean(run, {"Toy": [a["samples"][0]["doc"]]})["toy"]
    assert clean["acc,none"] == pytest.approx(4 / 9)  # weighted: 1 + 3 of 3 + 6


def test_a_group_whose_weighting_does_not_matter_is_recomputed(tmp_path):
    """A model scoring 0 everywhere: both weightings reproduce the group and
    agree after exclusion."""
    a = _task(_docs(4, "a"), {"none": [0, 0, 0, 0]})
    b = _task(_docs(6, "b"), {"none": [0, 0, 0, 0, 0, 0]})
    group = {"acc,none": 0.0, "acc_stderr,none": 0.0}
    run = _write_run(
        tmp_path,
        {"toy_a": a, "toy_b": b},
        groups={"toy": group},
        group_subtasks={"toy": ["toy_a", "toy_b"]},
    )
    clean, _ = _clean(run, {"Toy": [a["samples"][0]["doc"]]})["toy"]
    assert clean["acc,none"] == 0.0


def test_a_group_whose_weighting_is_unknown_is_refused(tmp_path):
    """With equal-sized tasks both weightings reproduce the score, but differ
    once a document is excluded."""
    run, a, _ = _group_run(tmp_path, sizes=(5, 5))
    with pytest.raises(SampleExclusionError, match="they differ without"):
        _clean(run, {"Toy": [a["samples"][0]["doc"]]})


# --- collect -----------------------------------------------------------------


def _collect(tmp_path, monkeypatch, blocklist_rows, fetch_all=True, **task_kwargs):
    docs = _docs(6)
    task = _task(docs, {"none": [1, 0, 1, 1, 0, 1]}, **task_kwargs)
    _write_run(
        tmp_path / "run", {"toy": task, "other": _task(_docs(3), {"none": [1, 0, 1]})}
    )
    blocklist = tmp_path / "blocklist"
    blocklist.mkdir()
    for name, rows in blocklist_rows(docs).items():
        (blocklist / f"{name}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
    monkeypatch.setattr(sample_exclusion, "load_task_map", lambda: {"Toy": ["toy"]})
    out = tmp_path / "out.csv"
    collect_results(
        str(tmp_path / "run"),
        str(out),
        fetch_all_metrics=fetch_all,
        exclude_samples=str(blocklist),
    )
    return pd.read_csv(out)


def test_collect_reports_full_and_clean_scores(tmp_path, monkeypatch):
    df = _collect(tmp_path, monkeypatch, lambda docs: {"Toy": [docs[0]], "Unused": []})
    toy = df[df.task == "toy"].set_index(["subset", "metric_name"])
    assert toy.loc[("full", "acc"), "performance"] == pytest.approx(4 / 6)
    assert toy.loc[("clean", "acc"), "performance"] == pytest.approx(3 / 5)
    assert toy.loc[("clean", "acc"), "n_excluded"] == 1
    assert toy.loc[("full", "acc"), "n_excluded"] == 0
    assert set(df[df.task == "other"].subset) == {"full"}


def test_collect_takes_the_clean_score_from_the_same_filter(tmp_path, monkeypatch):
    """collect keeps the first filter's value of each metric, in the order of
    the results file; the clean row must use the same filter."""
    docs = _docs(5)
    task = _task(docs, {"A": [1, 0, 0, 1, 0], "B": [1, 1, 1, 0, 0]})
    result = task["result"]
    task["result"] = {k: result[k] for k in ("acc,A", "acc,B", "acc_stderr,B")}
    task["result"]["acc_stderr,A"] = result["acc_stderr,A"]
    _write_run(tmp_path / "run", {"toy": task})
    (tmp_path / "blocklist").mkdir()
    (tmp_path / "blocklist" / "Toy.jsonl").write_text(json.dumps(docs[0]) + "\n")
    monkeypatch.setattr(sample_exclusion, "load_task_map", lambda: {"Toy": ["toy"]})
    out = tmp_path / "out.csv"
    collect_results(
        str(tmp_path / "run"),
        str(out),
        fetch_all_metrics=True,
        exclude_samples=str(tmp_path / "blocklist"),
    )
    df = pd.read_csv(out).set_index(["subset", "metric_name"])["performance"]
    assert df["full", "acc"] == pytest.approx(0.4)  # filter A
    assert df["clean", "acc"] == pytest.approx(0.25)  # filter A without doc 0
    assert df["full", "acc_stderr"] == pytest.approx(_score([1, 1, 1, 0, 0])[1])
    assert df["clean", "acc_stderr"] == pytest.approx(_score([1, 1, 0, 0])[1])


def test_collect_refuses_a_metric_it_cannot_recompute(tmp_path, monkeypatch):
    with pytest.raises(SampleExclusionError, match="cannot be recomputed"):
        _collect(
            tmp_path,
            monkeypatch,
            lambda docs: {"Toy": [docs[0]]},
            metric="bleu",
            aggregation="bleu",
        )


def test_collect_skips_a_run_it_cannot_filter_and_filters_the_rest(
    tmp_path, monkeypatch, caplog
):
    """An older run without saved samples in the same directory gets no clean
    rows and a warning, but does not stop collect."""
    docs = _docs(5)
    for run in ("old", "new"):
        _write_run(tmp_path / run, {"toy": _task(docs, {"none": [1, 0, 1, 1, 0]})})
    (tmp_path / "old" / f"samples_toy_{TS}.jsonl").unlink()
    data = json.loads((tmp_path / "old" / f"results_{TS}.json").read_text())
    data["model_name"] = "old-model"
    (tmp_path / "old" / f"results_{TS}.json").write_text(json.dumps(data))
    (tmp_path / "blocklist").mkdir()
    (tmp_path / "blocklist" / "Toy.jsonl").write_text(json.dumps(docs[0]) + "\n")
    monkeypatch.setattr(sample_exclusion, "load_task_map", lambda: {"Toy": ["toy"]})
    monkeypatch.setattr("oellm.main._setup_logging", lambda verbose=False: None)
    caplog.set_level(logging.INFO)
    out = tmp_path / "out.csv"
    collect_results(str(tmp_path), str(out), exclude_samples=str(tmp_path / "blocklist"))
    df = pd.read_csv(out)
    clean = df[df.subset == "clean"]
    assert set(clean.model_name) == {"m"}
    assert set(df[df.model_name == "old-model"].subset) == {"full"}
    assert "--log_samples" in caplog.text


def test_collect_logs_a_mapped_task_reported_only_through_a_group(
    tmp_path, monkeypatch, caplog
):
    docs = _docs(4)
    toy = _task(docs, {"none": [1, 0, 1, 0]})
    other = _task(_docs(4, "o"), {"none": [1, 1, 0, 0]})
    group = {"acc,none": 0.5, "acc_stderr,none": 0.1, "sample_len": 8}
    _write_run(
        tmp_path / "run",
        {"toy": toy, "other": other},
        groups={"big": group},
        group_subtasks={"big": ["toy", "other"]},
    )
    (tmp_path / "blocklist").mkdir()
    (tmp_path / "blocklist" / "Toy.jsonl").write_text(json.dumps(docs[0]) + "\n")
    monkeypatch.setattr(sample_exclusion, "load_task_map", lambda: {"Toy": ["toy"]})
    monkeypatch.setattr("oellm.main._setup_logging", lambda verbose=False: None)
    caplog.set_level(logging.INFO)
    out = tmp_path / "out.csv"
    collect_results(
        str(tmp_path / "run"), str(out), exclude_samples=str(tmp_path / "blocklist")
    )
    assert set(pd.read_csv(out).subset) == {"full"}
    assert "toy: 1 blocklisted documents" in caplog.text
    assert "only reported as part of a larger group" in caplog.text


# --- the task map ------------------------------------------------------------


def test_every_mapped_task_is_in_the_task_groups():
    groups = _load_task_groups_data()["task_groups"]
    tasks = {t["task"] for g in groups.values() for t in g.get("tasks") or []}
    for benchmark, mapped in sample_exclusion.load_task_map().items():
        assert set(mapped) <= tasks, benchmark


@pytest.mark.skipif(not BLOCKLIST_DIR.is_dir(), reason="set OELLM_BLOCKLIST_DIR")
def test_every_mapped_row_matches_one_lm_eval_document():
    """Each blocklisted row of a mapped benchmark is a document of its task.

    Needs lm-eval and the datasets in the Hugging Face cache, e.g. inside the
    eval container on LUMI.
    """
    pytest.importorskip("lm_eval")
    from lm_eval.tasks import TaskManager

    blocklist = sample_exclusion.load_blocklist(BLOCKLIST_DIR)
    manager = TaskManager()
    for benchmark, mapped in sample_exclusion.load_task_map().items():
        loaded = manager.load(mapped)["tasks"]
        samples, stack = {}, [loaded]
        while stack:
            for name, task in stack.pop().items():
                if isinstance(task, dict):
                    stack.append(task)
                else:
                    samples[name] = [
                        {"doc_id": i, "doc": json.loads(json.dumps(doc))}
                        for i, doc in enumerate(task.eval_docs)
                    ]
        excluded = sample_exclusion._excluded_docs(
            blocklist[benchmark], samples, benchmark
        )
        assert sum(map(len, excluded.values())) == len(blocklist[benchmark]), benchmark
