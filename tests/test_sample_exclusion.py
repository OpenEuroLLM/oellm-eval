"""Scores recomputed without blocklisted samples (``collect --exclude_samples``).

The runs here are written in lm-eval's output layout: ``results_<ts>.json``
with ``samples_<task>_<ts>.jsonl`` next to it. Their reported scores are
computed with the ``statistics`` module, independently of the code under test,
so a recomputation that drifts from lm-eval's formulas fails the self-check.
"""

import hashlib
import importlib.util
import json
import logging
import math
import os
import statistics
import sys
import types
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
# An evalchemy checkout, as oellm's job template finds it, and the questions
# each of its benchmarks loads.
EVALCHEMY_DIR = Path(os.environ.get("EVALCHEMY_DIR", "/opt/evalchemy"))
EVALCHEMY_QUESTIONS = {"MATH500": "eval/chat_benchmarks/MATH500/data/math500.jsonl"}
# agenttrove-0.0's rows that are not among the questions oellm evaluates.
ROWS_OUTSIDE_EVALUATED_COPY = {"Jeopardy": 4}


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
    run_dir,
    tasks,
    groups=None,
    group_subtasks=None,
    limit=None,
    results_name="results",
):
    """Write lm-eval's results file and sample files; return the results path.

    lm-eval names the results file results_<date>.json when --output_path is a
    directory and <results_name>_<date>.json when it is <results_name>.json, as
    oellm passes it.
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
    results_file = run_dir / f"{results_name}_{TS}.json"
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
    run = _write_run(
        tmp_path, {"toy": _task(docs, {"none": [1, 0, 1, 0]})}, results_name=name
    )
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


@pytest.mark.parametrize("metric, is_mean", [("acc", True), ("f1", False)])
def test_a_metric_without_aggregation_takes_lm_eval_s_default(metric, is_mean):
    """boolq lists `metric: acc` alone; lm-eval then averages it."""
    config = {"output_type": "multiple_choice", "metric_list": [{"metric": metric}]}
    assert sample_exclusion._is_mean_metric(metric, config) is is_mean


def test_a_boolq_like_task_is_recomputed(tmp_path):
    docs = _docs(6)
    task = _task(docs, {"none": [1, 0, 1, 1, 0, 1]})
    task["config"]["metric_list"] = [{"metric": "acc"}]
    run = _write_run(tmp_path, {"toy": task})
    clean, _ = _clean(run, {"Toy": [docs[0]]})["toy"]
    assert clean["acc,none"] == pytest.approx(3 / 5)
    assert "acc_stderr,none" in clean


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


def test_a_metric_whose_first_filter_is_not_recomputed_is_left_out(tmp_path):
    """collect reports a metric's first filter. Recomputing only a later one
    would pair the full strict-match score with a clean flexible-extract one."""
    docs = _docs(5)
    task = _task(
        docs,
        {"strict-match": [1, 0, 0, 1, 0], "flexible-extract": [1, 1, 0, 1, 1]},
        metric="exact_match",
        output_type="generate_until",
    )
    task["samples"][4]["exact_match"] = None  # strict-match, last document
    run = _write_run(tmp_path, {"toy": task})
    clean, _ = _clean(run, {"Toy": [docs[0]]})["toy"]
    assert not any(k.startswith("exact_match,") for k in clean)
    assert not any(k.startswith("exact_match_stderr,") for k in clean)


def test_a_filter_reported_without_a_number_is_passed_over(tmp_path):
    """collect skips a filter whose reported value is not a number ("N/A"), and
    so does the order of the recomputed keys."""
    docs = _docs(5)
    task = _task(
        docs,
        {"strict-match": [1, 0, 0, 1, 0], "flexible-extract": [1, 1, 0, 1, 1]},
        metric="exact_match",
        output_type="generate_until",
    )
    task["result"]["exact_match,strict-match"] = "N/A"
    task["result"]["exact_match_stderr,strict-match"] = "N/A"
    run = _write_run(tmp_path, {"toy": task})
    clean, _ = _clean(run, {"Toy": [docs[0]]})["toy"]
    assert clean["exact_match,flexible-extract"] == 0.75
    assert "exact_match_stderr,flexible-extract" in clean


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


def _ifeval_like(docs, prompt, instructions):
    """ifeval: a boolean per prompt, and a list of booleans per prompt (one per
    instruction) that lm-eval flattens before averaging, with no stderr."""
    flat = [x for item in instructions for x in item]
    result = {
        "alias": "",
        "sample_len": len(docs),
        "prompt_level_strict_acc,none": statistics.mean(prompt),
        "prompt_level_strict_acc_stderr,none": _score(prompt)[1],
        "inst_level_strict_acc,none": sum(flat) / len(flat),
        "inst_level_strict_acc_stderr,none": "N/A",
    }
    samples = [
        {
            "doc_id": i,
            "doc": doc,
            "filter": "none",
            "prompt_level_strict_acc": p,
            "inst_level_strict_acc": inst,
        }
        for i, (doc, p, inst) in enumerate(zip(docs, prompt, instructions, strict=True))
    ]
    config = {
        "output_type": "generate_until",
        "metric_list": [
            {"metric": "prompt_level_strict_acc", "aggregation": "mean"},
            {
                "metric": "inst_level_strict_acc",
                "aggregation": "def agg_inst_level_acc(items):\n    flat_items = ...\n",
            },
        ],
    }
    return {"result": result, "samples": samples, "config": config, "n": len(docs)}


def test_ifeval_s_instruction_level_accuracy_is_recomputed(tmp_path):
    docs = _docs(4)
    prompt = [True, False, True, False]
    instructions = [[True, True], [False, True, False], [True], [False]]
    run = _write_run(tmp_path, {"toy": _ifeval_like(docs, prompt, instructions)})
    clean, _ = _clean(run, {"Toy": [docs[1]]})["toy"]
    assert clean["inst_level_strict_acc,none"] == 3 / 4  # [T, T], [T], [F]
    assert clean["prompt_level_strict_acc,none"] == pytest.approx(2 / 3)
    assert clean["prompt_level_strict_acc_stderr,none"] == pytest.approx(
        _score([1, 1, 0])[1]
    )
    assert "inst_level_strict_acc_stderr,none" not in clean


def _perplexity_like(docs, loglikelihoods, accs):
    """lambada: a loglikelihood task without a metric_list gets acc and
    perplexity, exp(-mean(ll)), whose stderr lm-eval bootstraps."""
    result = {
        "alias": "",
        "sample_len": len(docs),
        "perplexity,none": math.exp(-statistics.mean(loglikelihoods)),
        "perplexity_stderr,none": 0.123,
        "acc,none": statistics.mean(accs),
        "acc_stderr,none": _score(accs)[1],
    }
    samples = [
        {"doc_id": i, "doc": doc, "filter": "none", "perplexity": ll, "acc": a}
        for i, (doc, ll, a) in enumerate(zip(docs, loglikelihoods, accs, strict=True))
    ]
    config = {"output_type": "loglikelihood"}
    return {"result": result, "samples": samples, "config": config, "n": len(docs)}


def test_perplexity_is_recomputed_without_its_bootstrapped_stderr(tmp_path):
    docs = _docs(4)
    task = _perplexity_like(docs, [-1.0, -2.0, -0.5, -1.5], [1, 0, 1, 0])
    run = _write_run(tmp_path, {"toy": task})
    clean, _ = _clean(run, {"Toy": [docs[0]]})["toy"]
    assert clean["perplexity,none"] == pytest.approx(math.exp(4 / 3))
    assert "perplexity_stderr,none" not in clean
    assert clean["acc,none"] == pytest.approx(1 / 3)


def test_collect_leaves_out_a_bootstrapped_stderr_and_says_so(
    tmp_path, monkeypatch, capfd
):
    docs = _docs(4)
    task = _perplexity_like(docs, [-1.0, -2.0, -0.5, -1.5], [1, 0, 1, 0])
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
    df = pd.read_csv(out)
    clean = df[df.subset == "clean"].set_index("metric_name")["performance"]
    assert clean["perplexity"] == pytest.approx(math.exp(4 / 3))
    assert "perplexity_stderr" not in clean
    assert "acc_stderr" in clean
    # collect logs through rich, which wraps lines.
    logged = " ".join(capfd.readouterr().out.split())
    assert "perplexity_stderr, which the harness estimates by bootstrapping" in logged


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


def test_fields_the_documents_lack_are_left_out_of_the_match(tmp_path):
    """The evaluated copy of a dataset may keep only some of its fields."""
    run = _write_run(tmp_path, {"toy": _task(_docs(4), {"none": [1, 0, 1, 0]})})
    row = {"question": "q1", "answer": 1, "source": "x"}
    assert _clean(run, {"Toy": [row]})["toy"][1] == 1


def test_a_row_sharing_no_field_with_the_documents_matches_none(tmp_path):
    run = _write_run(tmp_path, {"toy": _task(_docs(4), {"none": [1, 0, 1, 0]})})
    assert _clean(run, {"Toy": [{"source": "x"}]}) == {}


def test_a_row_matching_many_documents_is_refused(tmp_path):
    """Sharing only a label with the documents, a row would exclude every
    document with that label."""
    docs = [{"question": f"q{i}", "answer": i % 2} for i in range(30)]
    run = _write_run(tmp_path, {"toy": _task(docs, {"none": [1, 0] * 15})})
    with pytest.raises(SampleExclusionError, match="Toy: a blocklist row matches 15"):
        _clean(run, {"Toy": [{"label": "x", "answer": 1}]})


@pytest.mark.parametrize(
    "row",
    [
        {"question": "q1", "answer": 2},  # different value
        {"question": "q1", "answer": "1"},  # different type
        {"question": "q1 ", "answer": 1},  # whitespace
    ],
)
def test_a_row_that_matches_no_document_is_not_excluded(tmp_path, row, caplog):
    """A row outside the evaluated copy (oellm's Jeopardy is a subset) leaves
    nothing to exclude, and says so."""
    run = _write_run(tmp_path, {"toy": _task(_docs(4), {"none": [1, 0, 1, 0]})})
    with caplog.at_level(logging.INFO):
        assert _clean(run, {"Toy": [row]}) == {}
    (record,) = [r for r in caplog.records if r.getMessage().startswith("Toy:")]
    assert record.levelno == logging.WARNING
    assert "there are no samples to exclude" in record.getMessage()


def test_the_rows_that_match_are_excluded_and_the_others_counted(tmp_path, caplog):
    docs = _docs(4)
    run = _write_run(tmp_path, {"toy": _task(docs, {"none": [1, 0, 1, 0]})})
    rows = [docs[0], {"question": "q9", "answer": 0}, {"question": "q8", "answer": 2}]
    with caplog.at_level(logging.WARNING):
        clean, n_excluded = _clean(run, {"Toy": rows})["toy"]
    assert n_excluded == 1
    assert clean["acc,none"] == pytest.approx(1 / 3)
    assert "3 blocklisted rows, 2 of them match no document" in caplog.text


SPLIT_MAP = {"Toy": ["toy_low", "toy_medium"]}


def test_a_run_with_part_of_a_split_benchmark_skips_the_other_rows(tmp_path, caplog):
    """PolyMath's rows come from polymath_en_low and _medium, which can be run
    separately; the medium rows match nothing in a run of the low task, which is
    expected, so it is no warning."""
    low, medium = _docs(4, "low"), _docs(4, "medium")
    run = _write_run(tmp_path, {"toy_low": _task(low, {"none": [1, 0, 1, 0]})})
    with caplog.at_level(logging.INFO):
        clean = _clean(run, {"Toy": [low[0], medium[1]]}, SPLIT_MAP)
    assert clean.keys() == {"toy_low"}
    assert clean["toy_low"][1] == 1
    (record,) = [r for r in caplog.records if r.getMessage().startswith("Toy:")]
    assert record.levelno == logging.INFO


def test_a_run_with_all_of_a_split_benchmark_excludes_from_each_task(tmp_path):
    low, medium = _docs(4, "low"), _docs(4, "medium")
    tasks = {
        "toy_low": _task(low, {"none": [1, 0, 1, 0]}),
        "toy_medium": _task(medium, {"none": [0, 1, 1, 0]}),
    }
    run = _write_run(tmp_path, tasks)
    clean = _clean(run, {"Toy": [low[0], medium[1]]}, SPLIT_MAP)
    assert {name: n for name, (_, n) in clean.items()} == {"toy_low": 1, "toy_medium": 1}


def test_a_benchmark_can_be_matched_on_some_fields(tmp_path):
    """oellm's Jeopardy keeps 3 of the original dataset's fields."""
    (tmp_path / "Toy.jsonl").write_text(
        json.dumps({"question": "q1", "answer": 1, "air_date": "2008", "_n": 1}) + "\n"
    )
    match_fields = {"Toy": ["question", "answer"]}
    blocklist = sample_exclusion.load_blocklist(tmp_path, match_fields)
    assert blocklist == {"Toy": [{"question": "q1", "answer": 1}]}
    run = _write_run(tmp_path / "run", {"toy": _task(_docs(4), {"none": [1, 0, 1, 0]})})
    assert _clean(run, blocklist)["toy"][1] == 1


def test_a_row_without_a_field_it_is_matched_on_is_refused(tmp_path):
    (tmp_path / "Toy.jsonl").write_text(json.dumps({"question": "q1"}) + "\n")
    with pytest.raises(SampleExclusionError, match="line 1: the row has no answer"):
        sample_exclusion.load_blocklist(tmp_path, {"Toy": ["question", "answer"]})


def test_documents_without_a_field_their_benchmark_is_matched_on_are_refused(tmp_path):
    """HellaSwag's ind repeats without source_id: matching on ind alone would
    exclude other documents than the blocklisted ones."""
    docs = [{"question": f"q{i}", "ind": i % 2} for i in range(4)]
    run = _write_run(tmp_path, {"toy": _task(docs, {"none": [1, 0, 1, 0]})})
    data = json.loads(run.read_text())
    rows = {"Toy": [{"ind": 1, "source_id": "s1"}]}
    with pytest.raises(RunNotFilterable, match="toy have no source_id"):
        clean_results(run, data, rows, {"Toy": ["toy"]}, match_on={"Toy"})


def test_match_fields_come_from_the_task_map():
    assert sample_exclusion.load_match_fields()["Jeopardy"] == ["context", "continuation"]
    assert sample_exclusion.load_task_map()["Jeopardy"] == ["jeopardy"]
    assert "MMLU" not in sample_exclusion.load_match_fields()


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
    with pytest.raises(
        SampleExclusionError, match="Toy.jsonl, line 2: the row has only `_` annotation"
    ):
        sample_exclusion.load_blocklist(tmp_path)


@pytest.mark.parametrize(
    "line, message",
    [
        ("[1, 2]", "expected a JSON object .* got a list"),
        ('"q1"', "expected a JSON object .* got a str"),
        ("{not json", "not valid JSON"),
    ],
)
def test_a_line_that_is_not_a_row_is_refused(tmp_path, line, message):
    (tmp_path / "Toy.jsonl").write_text(
        json.dumps({"question": "q1"}) + "\n\n" + line + "\n"
    )
    with pytest.raises(SampleExclusionError, match=f"Toy.jsonl, line 3: {message}"):
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
    """lm-eval does not record --samples; a run with it scores fewer documents
    than the task has, and saves them under their own doc_ids."""
    docs = _docs(5)
    task = _task([docs[0], docs[2], docs[4]], {"none": [1, 0, 1]})
    for s, doc_id in zip(task["samples"], [0, 2, 4], strict=True):
        s["doc_id"] = doc_id
    run = _write_run(tmp_path, {"toy": task})
    data = json.loads(run.read_text())
    data["n-samples"]["toy"]["original"] = 5
    run.write_text(json.dumps(data))
    with pytest.raises(
        RunNotFilterable, match=r"3 of 5 documents \(a --limit or --samples"
    ):
        _clean(run, {"Toy": [docs[0]]})


def test_a_run_without_saved_samples_is_refused(tmp_path):
    docs = _docs(5)
    run = _write_run(tmp_path, {"toy": _task(docs, {"none": [1, 0, 1, 1, 0]})})
    (tmp_path / f"samples_toy_{TS}.jsonl").unlink()
    with pytest.raises(RunNotFilterable, match="--log_samples"):
        _clean(run, {"Toy": [docs[0]]})


@pytest.mark.parametrize("cut", ["mid-line", "whole lines"])
def test_a_run_cut_short_while_saving_samples_gets_no_clean_rows(
    tmp_path, monkeypatch, cut
):
    """lm-eval writes the results before the samples, so a job stopped by its
    time limit can leave them incomplete; the other runs keep their clean rows."""
    docs = _docs(5)
    _write_run(tmp_path / "a", {"toy": _task(docs, {"none": [1, 0, 1, 1, 0]})})
    _write_run(tmp_path / "b", {"toy2": _task(docs, {"none": [1, 1, 1, 1, 0]})})
    samples = tmp_path / "b" / f"samples_toy2_{TS}.jsonl"
    text = samples.read_text()
    samples.write_text(text[:-20] if cut == "mid-line" else text.split("\n", 1)[1])
    with pytest.raises(RunNotFilterable, match="toy2"):
        _clean(
            tmp_path / "b" / f"results_{TS}.json", {"Toy": [docs[0]]}, {"Toy": ["toy2"]}
        )
    (tmp_path / "blocklist").mkdir()
    (tmp_path / "blocklist" / "Toy.jsonl").write_text(json.dumps(docs[0]) + "\n")
    monkeypatch.setattr(
        sample_exclusion, "load_task_map", lambda: {"Toy": ["toy", "toy2"]}
    )
    out = tmp_path / "out.csv"
    collect_results(str(tmp_path), str(out), exclude_samples=str(tmp_path / "blocklist"))
    df = pd.read_csv(out).set_index(["task", "subset"])
    assert ("toy", "clean") in df.index
    assert ("toy2", "full") in df.index
    assert ("toy2", "clean") not in df.index


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


def test_a_group_from_lm_eval_before_0_4_12_is_refused(tmp_path):
    """Older lm-eval writes no sample_len, which weighs a group's tasks."""
    run, a, _ = _group_run(tmp_path)
    data = json.loads(run.read_text())
    for name in ("toy_a", "toy_b"):
        del data["results"][name]["sample_len"]
    run.write_text(json.dumps(data))
    with pytest.raises(SampleExclusionError, match="lm-eval 0.4.12 or later"):
        _clean(run, {"Toy": [a["samples"][0]["doc"]]})


def _fewshot_from_scored(tmp_path, sampler, shots=5, n=8):
    """A task whose few-shot examples come from the split it scores, as MMMLU's."""
    docs = _docs(n)
    task = _task(docs, {"none": [1, 0] * (n // 2)})
    task["config"] |= {
        "test_split": "test",
        "fewshot_split": "test",
        "fewshot_config": {"sampler": sampler},
    }
    run = _write_run(tmp_path, {"toy": task})
    data = json.loads(run.read_text())
    data["n-shot"]["toy"] = shots
    run.write_text(json.dumps(data))
    return run, docs


@pytest.mark.parametrize(
    "sampler, excluded, shown",
    [
        ("first_n", 3, True),  # one of the first 5
        ("first_n", 5, True),  # the 6th, for the first 5 documents' prompts
        ("first_n", 6, False),
        ("default", 6, True),  # drawn at random from all
    ],
)
def test_blocklisted_few_shot_examples_are_reported(
    tmp_path, caplog, sampler, excluded, shown
):
    """MMMLU takes its examples from the first test questions of a subject; a
    clean score from a full run still shows a blocklisted one among them."""
    run, docs = _fewshot_from_scored(tmp_path, sampler)
    with caplog.at_level(logging.WARNING):
        _clean(run, {"Toy": [docs[excluded]]})
    assert ("toy: its few-shot examples come from" in caplog.text) == shown


def test_few_shot_examples_from_another_split_are_not_reported(tmp_path, caplog):
    run, docs = _fewshot_from_scored(tmp_path, "first_n")
    data = json.loads(run.read_text())
    data["configs"]["toy"]["fewshot_split"] = "dev"
    run.write_text(json.dumps(data))
    with caplog.at_level(logging.WARNING):
        _clean(run, {"Toy": [docs[0]]})
    assert "few-shot examples" not in caplog.text


# --- translations ------------------------------------------------------------


def test_a_row_excludes_its_item_in_every_language(tmp_path):
    """PAWS-X's id is the same in every language, so a row matched on it excludes
    one document of each task, however many languages there are."""
    langs = [f"toy_{i}" for i in range(12)]
    run = _write_run(
        tmp_path,
        {
            t: _task(
                [{"id": i, "sentence": f"{t} {i}"} for i in range(4)],
                {"none": [1, 0, 1, 0]},
            )
            for t in langs
        },
    )
    clean = _clean(run, {"Toy": [{"id": 1}]}, {"Toy": langs})
    assert {t: n for t, (_, n) in clean.items()} == dict.fromkeys(langs, 1)


def test_text_is_matched_by_its_letters_and_digits_where_normalized(tmp_path):
    """openeurollm/jeopardy drops apostrophes and some spaces of the original."""
    docs = [
        {"context": "HISTORY:   In 1453 the French won its last battle", "continuation": "the Hundred Years War"},
        {"context": "HISTORY: In 1756 soldiers were held in this 15 x 18 room", "continuation": "the Black Hole"},
        {"context": "SCIENCE: Mass number counts these & protons", "continuation": "neutrons"},
    ]  # fmt: skip
    row = {
        "context": "HISTORY: In 1453 the French won its last battle",
        "continuation": "the Hundred Years\\' War",
    }
    run = _write_run(tmp_path, {"toy": _task(docs, {"none": [1, 0, 1]})})
    data = json.loads(run.read_text())
    assert clean_results(run, data, {"Toy": [row]}, {"Toy": ["toy"]}) == {}
    clean = clean_results(run, data, {"Toy": [row]}, {"Toy": ["toy"]}, normalized={"Toy"})
    assert clean["toy"][1] == 1
    assert clean["toy"][0]["acc,none"] == 0.5


def test_benchmarks_matched_by_letters_and_digits_come_from_the_task_map():
    assert sample_exclusion.load_normalized() == {"Jeopardy"}


ENGLISH = [{"premise": f"p{i}", "hypothesis": f"h{i}", "label": i % 3} for i in range(5)]
GERMAN = [
    {"premise": f"de p{i}", "hypothesis": f"de h{i}", "label": i % 3} for i in range(5)
]
TRANSLATED = {"Toy": ["toy_en", "toy_de"]}


def _table(rows, tasks, key=("premise", "hypothesis")):
    return {
        "key": list(key),
        "items": [sample_exclusion.item_key([r[f] for f in key]) for r in rows],
        "tasks": tasks,
    }


def _entry(docs, positions=None, first=0):
    """A task's table entry; the answers are what lm-eval saves (str of the label)."""
    answers = sample_exclusion.item_key([str(d["label"]) for d in docs])
    where = {"first": first} if positions is None else {"positions": positions}
    return {"documents": len(docs), "answers": answers} | where


IDENTITY = _table(ENGLISH, {"toy_en": _entry(ENGLISH), "toy_de": _entry(GERMAN)})


def _translated_task(docs, values):
    """A task whose samples carry the target lm-eval saves (str of the label)."""
    task = _task(docs, {"none": values})
    for s in task["samples"]:
        s["target"] = str(docs[s["doc_id"]]["label"])
    return task


def _translated_run(tmp_path, german=GERMAN, german_values=(0, 0, 1, 1, 1)):
    return _write_run(
        tmp_path,
        {
            "toy_en": _translated_task(ENGLISH, [1, 0, 1, 1, 0]),
            "toy_de": _translated_task(german, list(german_values)),
        },
    )


def _clean_translated(run, rows, table=IDENTITY, task_map=TRANSLATED):
    data = json.loads(run.read_text())
    return clean_results(run, data, {"Toy": rows}, task_map, {"Toy": table})


def _collect_with_settings(tmp_path, monkeypatch, tasks, rows, task_map, **settings):
    """collect over one run, with the task map and the yaml's other settings
    (alignments, match_fields, normalized) as given."""
    _write_run(tmp_path / "run", tasks)
    (tmp_path / "blocklist").mkdir()
    for benchmark, benchmark_rows in rows.items():
        (tmp_path / "blocklist" / f"{benchmark}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in benchmark_rows)
        )
    monkeypatch.setattr(sample_exclusion, "load_task_map", lambda: task_map)
    for name, value in settings.items():
        monkeypatch.setattr(sample_exclusion, f"load_{name}", lambda v=value: v)
    out = tmp_path / "out.csv"
    collect_results(
        str(tmp_path / "run"), str(out), exclude_samples=str(tmp_path / "blocklist")
    )
    return pd.read_csv(out)


def test_collect_matches_by_the_yaml_s_tables_fields_and_normalization(
    tmp_path, monkeypatch
):
    """Each row excludes its document only through its benchmark's setting: a
    table (toy_de shares no field with the row), match_on (the row's answer
    differs) or normalize (the text differs in spaces and apostrophes)."""
    jeopardy = [{"context": f"A  {i}", "continuation": f"x's {i}"} for i in range(3)]
    df = _collect_with_settings(
        tmp_path,
        monkeypatch,
        {
            "toy_en": _translated_task(ENGLISH, [1, 0, 1, 1, 0]),
            "toy_de": _translated_task(GERMAN, [0, 0, 1, 1, 1]),
            "toy_m": _task(_docs(4), {"none": [1, 0, 1, 0]}),
            "toy_j": _task(jeopardy, {"none": [1, 0, 1]}),
        },
        {
            "Tab": [ENGLISH[1]],
            "Match": [{"question": "q1", "answer": 99}],
            "Norm": [{"context": "A 1", "continuation": "xs 1"}],
        },
        {"Tab": ["toy_en", "toy_de"], "Match": ["toy_m"], "Norm": ["toy_j"]},
        alignments={"Tab": IDENTITY},
        match_fields={"Match": ["question"]},
        normalized={"Norm"},
    )
    clean = df[df.subset == "clean"].groupby("task")["n_excluded"].first()
    assert clean.to_dict() == {"toy_de": 1, "toy_en": 1, "toy_j": 1, "toy_m": 1}


def test_collect_skips_a_run_whose_documents_lack_a_match_on_field(tmp_path, monkeypatch):
    docs = [{"question": f"q{i}", "ind": i % 2} for i in range(4)]
    df = _collect_with_settings(
        tmp_path,
        monkeypatch,
        {"toy": _task(docs, {"none": [1, 0, 1, 0]})},
        {"Toy": [{"ind": 1, "source_id": "s1"}]},
        {"Toy": ["toy"]},
        match_fields={"Toy": ["ind", "source_id"]},
    )
    assert set(df.subset) == {"full"}


def test_a_row_excludes_its_translations_by_their_position(tmp_path):
    """XNLI's languages share no field identifying an item, but list the items
    in the same order: the alignment table finds the German document of an
    English row by its position."""
    clean = _clean_translated(_translated_run(tmp_path), [ENGLISH[1]])
    assert {t: n for t, (_, n) in clean.items()} == {"toy_en": 1, "toy_de": 1}
    assert clean["toy_en"][0]["acc,none"] == 0.75  # 1, 1, 1, 0
    assert clean["toy_de"][0]["acc,none"] == 0.75  # 0, 1, 1, 1


def test_a_row_is_found_in_the_table_by_its_key_fields_only(tmp_path):
    row = {"premise": "p3", "hypothesis": "h3", "label": "other", "_n": 1}
    clean = _clean_translated(_translated_run(tmp_path), [row])
    assert {t: n for t, (_, n) in clean.items()} == {"toy_en": 1, "toy_de": 1}


def test_translations_in_another_order_are_found_at_their_positions(tmp_path):
    """Belebele orders each language's questions differently."""
    order = [3, 0, 4, 1, 2]  # German document i is item order[i]
    german = [GERMAN[i] for i in order]
    table = _table(
        ENGLISH,
        {
            "toy_en": _entry(ENGLISH),
            "toy_de": _entry(german, positions=[order.index(i) for i in range(5)]),
        },
    )
    run = _translated_run(tmp_path, german, german_values=[1, 1, 1, 0, 1])
    clean = _clean_translated(run, [ENGLISH[1]], table)
    # Item 1 is German document 3, the only 0.
    assert clean["toy_de"] == (
        {"acc,none": 1.0, "acc_stderr,none": 0.0, "sample_len": 4},
        1,
    )


def test_an_item_a_translation_lacks_is_excluded_where_it_is(tmp_path):
    """X-CSQA translates only some of CommonsenseQA's questions."""
    table = _table(
        ENGLISH,
        {
            "toy_en": _entry(ENGLISH),
            "toy_de": _entry(GERMAN, positions=[0, None, 2, 3, 4]),
        },
    )
    clean = _clean_translated(_translated_run(tmp_path), [ENGLISH[1]], table)
    assert {t: n for t, (_, n) in clean.items()} == {"toy_en": 1}


def test_a_row_outside_the_table_is_not_excluded(tmp_path, caplog):
    row = {"premise": "p9", "hypothesis": "h9"}
    with caplog.at_level(logging.WARNING):
        assert _clean_translated(_translated_run(tmp_path), [row]) == {}
    assert "there are no samples to exclude" in caplog.text


def test_documents_with_other_answers_than_the_table_are_refused(tmp_path):
    """A dataset that changed since the table was built (reordered, say) would
    put the excluded items elsewhere."""
    german = [GERMAN[1], GERMAN[0], *GERMAN[2:]]
    with pytest.raises(RunNotFilterable, match="toy_de: its documents are not the ones"):
        _clean_translated(_translated_run(tmp_path, german), [ENGLISH[2]])


def test_items_after_the_first_are_found_from_the_task_s_first_item(tmp_path):
    """PolyMath's levels are tasks of their own: the medium task holds the
    items after the low ones."""
    low, medium = ENGLISH[:2], ENGLISH[2:]
    run = _write_run(
        tmp_path,
        {
            "toy_low": _translated_task(low, [1, 0]),
            "toy_medium": _translated_task(medium, [1, 0, 1]),
        },
    )
    table = _table(
        ENGLISH, {"toy_low": _entry(low), "toy_medium": _entry(medium, first=2)}
    )
    clean = _clean_translated(
        run, [ENGLISH[3]], table, {"Toy": ["toy_low", "toy_medium"]}
    )
    assert {t: n for t, (_, n) in clean.items()} == {"toy_medium": 1}
    assert clean["toy_medium"][0]["acc,none"] == 1.0  # its 0 was item 3


def test_a_repeated_item_is_excluded_wherever_it_is(tmp_path):
    """MMLU repeats some questions; a row excludes each copy, as it does where
    its documents are matched by content."""
    english = [ENGLISH[0], ENGLISH[1], ENGLISH[0], ENGLISH[3], ENGLISH[4]]
    table = _table(english, {"toy_en": _entry(english), "toy_de": _entry(GERMAN)})
    run = _write_run(
        tmp_path,
        {
            "toy_en": _translated_task(english, [1, 0, 1, 1, 0]),
            "toy_de": _translated_task(GERMAN, [0, 0, 1, 1, 1]),
        },
    )
    clean = _clean_translated(run, [ENGLISH[0]], table)
    assert {t: n for t, (_, n) in clean.items()} == {"toy_en": 2, "toy_de": 2}


def test_a_task_with_more_or_fewer_documents_than_the_table_is_refused(tmp_path):
    run = _write_run(
        tmp_path,
        {
            "toy_en": _translated_task(ENGLISH, [1, 0, 1, 1, 0]),
            "toy_de": _translated_task(GERMAN[:4], [0, 0, 1, 1]),
        },
    )
    with pytest.raises(RunNotFilterable, match="toy_de: 4 documents, but"):
        _clean_translated(run, [ENGLISH[2]])


def test_tasks_outside_the_table_are_matched_by_content(tmp_path):
    """GSM8K's own task is matched on its rows' fields, MGSM's translations of
    some of its questions by the table; a row is matched if either matches."""
    source = [{"premise": f"p{i}", "hypothesis": f"h{i}", "answer": i} for i in range(6)]
    run = _write_run(
        tmp_path,
        {
            "toy_src": _task(source, {"none": [1, 0, 1, 1, 0, 1]}),
            "toy_de": _translated_task(GERMAN, [0, 0, 1, 1, 1]),
        },
    )
    table = _table(ENGLISH, {"toy_de": _entry(GERMAN)})
    rows = [source[1], source[5]]  # source[5] has no translation
    clean = _clean_translated(run, rows, table, {"Toy": ["toy_src", "toy_de"]})
    assert {t: n for t, (_, n) in clean.items()} == {"toy_src": 2, "toy_de": 1}


def test_two_benchmarks_holding_a_task_exclude_both_their_rows(tmp_path):
    """MGSM translates questions of GSM8K and of Global-MGSM."""
    run = _translated_run(tmp_path)
    data = json.loads(run.read_text())
    clean = clean_results(
        run,
        data,
        {"A": [ENGLISH[1]], "B": [ENGLISH[2]]},
        {"A": TRANSLATED["Toy"], "B": TRANSLATED["Toy"]},
        {"A": IDENTITY, "B": IDENTITY},
    )
    assert {t: n for t, (_, n) in clean.items()} == {"toy_en": 2, "toy_de": 2}


def _write_translated_lighteval_run(tmp_path, ids, gold_index=0):
    """A lighteval run whose details list the documents out of their dataset
    order, as lighteval does; a document's id is its position there."""
    docs = [
        {
            "query": f"Q {i}",
            "choices": [" A", " B"],
            "gold_index": gold_index,
            "id": str(i),
        }
        for i in ids
    ]
    accs = [1, 0, 1, 1, 0]  # by details row
    return _write_lighteval_run(tmp_path, accs, docs=docs)


def _lighteval_table(n=5):
    rows = [{"q": f"q{i}"} for i in range(n)]
    # lighteval's details save gold_index, compared as JSON.
    entry = {"documents": n, "answers": sample_exclusion.item_key(["0"] * n), "first": 0}
    return rows, _table(rows, {"toy:it": entry}, key=["q"])


def test_a_lighteval_document_is_found_by_its_position_in_the_dataset(tmp_path):
    run = _write_translated_lighteval_run(tmp_path, ids=[2, 0, 1, 3, 4])
    rows, table = _lighteval_table()
    data = json.loads(run.read_text())
    clean = clean_results(
        run, data, {"Toy": [rows[0]]}, {"Toy": ["toy:it"]}, {"Toy": table}
    )
    # Item 0 is the second details row, the first 0.
    assert clean["toy:it"][1] == 1
    assert clean["toy:it"][0]["acc"] == 0.75


def test_lighteval_documents_with_other_answers_than_the_table_are_refused(tmp_path):
    run = _write_translated_lighteval_run(tmp_path, ids=[2, 0, 1, 3, 4], gold_index=1)
    rows, table = _lighteval_table()
    data = json.loads(run.read_text())
    with pytest.raises(RunNotFilterable, match="toy:it: its documents are not the ones"):
        clean_results(run, data, {"Toy": [rows[0]]}, {"Toy": ["toy:it"]}, {"Toy": table})


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


def _collect_runs(tmp_path, monkeypatch, runs):
    """collect over one run per (directory, scores, with saved samples)."""
    docs = _docs(6)
    for run_dir, values, with_samples in runs:
        _write_run(tmp_path / run_dir, {"toy": _task(docs, {"none": values})})
        if not with_samples:
            (tmp_path / run_dir / f"samples_toy_{TS}.jsonl").unlink()
    (tmp_path / "blocklist").mkdir()
    (tmp_path / "blocklist" / "Toy.jsonl").write_text(json.dumps(docs[0]) + "\n")
    monkeypatch.setattr(sample_exclusion, "load_task_map", lambda: {"Toy": ["toy"]})
    out = tmp_path / "out.csv"
    collect_results(str(tmp_path), str(out), exclude_samples=str(tmp_path / "blocklist"))
    df = pd.read_csv(out)
    assert "_source" not in df.columns
    return df[df.metric_name == "acc"].set_index("subset")


def test_full_and_clean_scores_come_from_the_same_run(tmp_path, monkeypatch, caplog):
    """Run b sorts after run a, so its full score is kept. b saved no samples,
    so a's clean score must not be paired with it."""
    monkeypatch.setattr("oellm.main._setup_logging", lambda verbose=False: None)
    caplog.set_level(logging.WARNING)
    df = _collect_runs(
        tmp_path,
        monkeypatch,
        [("a", [1, 1, 1, 1, 0, 0], True), ("b", [1, 0, 0, 0, 0, 0], False)],
    )
    assert df.loc["full", "performance"] == pytest.approx(1 / 6)
    assert "clean" not in df.index
    assert "m | toy | n_shot=0: the full score kept comes from another run" in caplog.text


def test_the_kept_run_gives_both_scores(tmp_path, monkeypatch):
    df = _collect_runs(
        tmp_path,
        monkeypatch,
        [("a", [1, 0, 0, 0, 0, 0], False), ("b", [1, 1, 1, 1, 0, 0], True)],
    )
    assert df.loc["full", "performance"] == pytest.approx(4 / 6)
    assert df.loc["clean", "performance"] == pytest.approx(3 / 5)


def test_of_two_runs_with_samples_the_kept_one_gives_both_scores(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setattr("oellm.main._setup_logging", lambda verbose=False: None)
    caplog.set_level(logging.WARNING)
    df = _collect_runs(
        tmp_path,
        monkeypatch,
        [("a", [1, 1, 1, 1, 0, 0], True), ("b", [1, 0, 0, 0, 0, 1], True)],
    )
    assert df.loc["full", "performance"] == pytest.approx(2 / 6)
    assert df.loc["clean", "performance"] == pytest.approx(1 / 5)
    assert "No scores without blocklisted samples" not in caplog.text


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


# --- lighteval ---------------------------------------------------------------

PREMISES = [
    f"the {thing} fell over." for thing in ["cup", "lamp", "tree", "chair", "vase"]
]


def _write_lighteval_run(
    output_dir, accs, max_samples=None, details=True, model="org/model", docs=None
):
    """lighteval's output with --save-details: results/<model>/results_<date>.json,
    and details/<model>/<date>/details_<task>|<shot>_<date>.parquet holding each
    sample's prompt (query and choices, built from the dataset row), its
    position in the dataset (id) and its metrics. lighteval strips the leading
    "/" of a local checkpoint's path from the directory names."""
    key = "toy:it|0"
    mean, stderr = _score(accs)
    data = {
        "config_general": {"model_name": model, "max_samples": max_samples},
        "results": {key: {"acc": mean, "acc_stderr": stderr}, "all": {"acc": mean}},
        "config_tasks": {
            key: {
                "name": "toy:it",
                "metrics": [{"metric_name": "acc", "corpus_level_fn": "mean"}],
            }
        },
    }
    results_file = output_dir / "results" / model.strip("/") / f"results_{TS}.json"
    results_file.parent.mkdir(parents=True)
    results_file.write_text(json.dumps(data))
    if details:
        # The template capitalises and re-punctuates the dataset's text.
        docs = docs or [
            {"query": f"Premise: {p.capitalize()[:-1]}!\nQuestion: why", "choices": [" A.", " B."], "gold_index": 0, "id": str(i)}
            for i, p in enumerate(PREMISES[: len(accs)])
        ]  # fmt: skip
        path = (
            output_dir / "details" / model.strip("/") / TS / f"details_{key}_{TS}.parquet"
        )
        path.parent.mkdir(parents=True)
        pd.DataFrame({"doc": docs, "metric": [{"acc": a} for a in accs]}).to_parquet(path)
    return results_file


def _clean_lighteval(results_file, rows):
    data = json.loads(results_file.read_text())
    return clean_results(results_file, data, {"Toy": rows}, {"Toy": ["toy:it"]})


def test_a_lighteval_sample_is_found_by_its_text_in_the_prompt(tmp_path):
    run = _write_lighteval_run(tmp_path, [1, 1, 1, 0, 1])
    row = {"premise": PREMISES[0], "idx": 0, "url": "https://x"}  # not in the prompt
    clean, n_excluded = _clean_lighteval(run, [{"premise": row["premise"]}])["toy:it"]
    assert n_excluded == 1
    assert clean == {"acc": 0.75, "acc_stderr": pytest.approx(_score([1, 1, 0, 1])[1])}


def test_every_text_field_of_a_lighteval_row_must_be_in_the_prompt(tmp_path, caplog):
    run = _write_lighteval_run(tmp_path, [1, 1, 1, 0, 1])
    with caplog.at_level(logging.WARNING):
        assert _clean_lighteval(run, [{"premise": PREMISES[0], "url": "https://x"}]) == {}
    assert "there are no samples to exclude" in caplog.text


@pytest.mark.parametrize("in_choices", [False, True])
def test_a_sentence_part_of_another_document_s_matches_its_own(in_choices):
    """FLORES' 33rd sentence is part of its 32nd; English is in the query
    (eng_Latn-deu_Latn) or the choices (deu_Latn-eng_Latn)."""
    english = ["The cat sat on the mat, and then it slept.", "The cat sat on the mat."]
    english.append(english[1])  # a sentence a benchmark repeats
    samples = {
        "t": [
            {
                "doc_id": i,
                "doc": {"query": f"DE: {i} EN:", "choices": [f" {e}"]}
                if in_choices
                else {"query": f"EN: {e} DE:", "choices": [f" {i}"]},
            }
            for i, e in enumerate(english)
        ]
    }
    rows = [{"sentence": english[1]}, {"sentence": english[0]}]
    excluded, _ = sample_exclusion._contained_docs(rows[:1], samples)
    assert excluded == {"t": {1, 2}}
    excluded, _ = sample_exclusion._contained_docs(rows[1:], samples)
    assert excluded == {"t": {0}}


def test_a_lighteval_run_with_max_samples_is_refused(tmp_path):
    run = _write_lighteval_run(tmp_path, [1, 1, 1, 0, 1], max_samples=5)
    with pytest.raises(RunNotFilterable, match="scored at most 5 samples"):
        _clean_lighteval(run, [{"premise": PREMISES[0]}])


def test_a_lighteval_run_without_details_is_refused(tmp_path):
    run = _write_lighteval_run(tmp_path, [1, 1, 1, 0, 1], details=False)
    with pytest.raises(RunNotFilterable, match="no saved details"):
        _clean_lighteval(run, [{"premise": PREMISES[0]}])


def test_the_details_of_a_local_checkpoint_are_found(tmp_path):
    """oellm passes lighteval the checkpoint's absolute path as model_name."""
    run = _write_lighteval_run(tmp_path, [1, 1, 1, 0, 1], model="/scratch/p/model")
    assert run == tmp_path / "results/scratch/p/model" / f"results_{TS}.json"
    assert _clean_lighteval(run, [{"premise": PREMISES[0]}])["toy:it"][1] == 1


def test_a_lighteval_results_file_moved_out_of_its_directory_is_refused(tmp_path):
    run = _write_lighteval_run(tmp_path / "out", [1, 1, 1, 0, 1])
    moved = tmp_path / run.name
    run.rename(moved)
    with pytest.raises(RunNotFilterable, match="not in lighteval's results/org/model/"):
        _clean_lighteval(moved, [{"premise": PREMISES[0]}])


@pytest.mark.parametrize("key", ["acc", "acc_stderr"])
def test_lighteval_details_that_do_not_reproduce_the_score_are_refused(tmp_path, key):
    run = _write_lighteval_run(tmp_path, [1, 1, 1, 0, 1])
    data = json.loads(run.read_text())
    data["results"]["toy:it|0"][key] = 0.6
    run.write_text(json.dumps(data))
    with pytest.raises(SampleExclusionError, match="lighteval reported 0.6"):
        _clean_lighteval(run, [{"premise": PREMISES[0]}])


def test_a_lighteval_row_in_more_than_10_prompts_is_refused():
    """Its text then does not identify a document."""
    samples = {
        "t": [
            {"doc_id": i, "doc": {"query": f"EN: The cat sat {i}. DE:", "choices": []}}
            for i in range(11)
        ]
    }
    with pytest.raises(SampleExclusionError, match="text is in 11 prompts of t"):
        sample_exclusion._contained_docs([{"sentence": "The cat sat"}], samples, "Toy")


def test_collect_reports_full_and_clean_lighteval_scores(tmp_path, monkeypatch):
    _write_lighteval_run(tmp_path / "run", [1, 1, 1, 0, 1])
    (tmp_path / "blocklist").mkdir()
    (tmp_path / "blocklist" / "Toy.jsonl").write_text(
        json.dumps({"premise": PREMISES[3]}) + "\n"
    )
    monkeypatch.setattr(sample_exclusion, "load_task_map", lambda: {"Toy": ["toy:it"]})
    out = tmp_path / "out.csv"
    collect_results(
        str(tmp_path / "run"),
        str(out),
        fetch_all_metrics=True,
        exclude_samples=str(tmp_path / "blocklist"),
    )
    df = pd.read_csv(out).set_index(["task", "subset", "metric_name"])["performance"]
    assert df["toy:it", "full", "acc"] == pytest.approx(0.8)
    assert df["toy:it", "clean", "acc"] == 1.0
    assert ("toy:it", "clean", "acc_stderr") in df.index


def test_collect_leaves_out_lighteval_s_corpus_chrf_and_says_so(
    tmp_path, monkeypatch, capfd
):
    """flores reports chrF++ next to BLEU; the other metrics keep their clean
    scores."""
    run = _write_lighteval_run(tmp_path / "run", [1, 1, 1, 0, 1])
    data = json.loads(run.read_text())
    data["results"]["toy:it|0"] |= {"chrf++": 50.0, "chrf++_stderr": 1.0}
    data["config_tasks"]["toy:it|0"]["metrics"].append(
        {
            "metric_name": "chrf++",
            "corpus_level_fn": "CorpusLevelTranslationMetric(metric_type=chrf++, lang=)",
        }
    )
    run.write_text(json.dumps(data))
    (tmp_path / "blocklist").mkdir()
    (tmp_path / "blocklist" / "Toy.jsonl").write_text(
        json.dumps({"premise": PREMISES[3]}) + "\n"
    )
    monkeypatch.setattr(sample_exclusion, "load_task_map", lambda: {"Toy": ["toy:it"]})
    out = tmp_path / "out.csv"
    collect_results(
        str(tmp_path / "run"),
        str(out),
        fetch_all_metrics=True,
        exclude_samples=str(tmp_path / "blocklist"),
    )
    df = pd.read_csv(out).set_index(["task", "subset", "metric_name"])["performance"]
    assert df["toy:it", "full", "chrf++"] == 50.0
    assert df["toy:it", "clean", "acc"] == 1.0
    assert ("toy:it", "clean", "chrf++") not in df.index
    logged = " ".join(capfd.readouterr().out.split())
    assert "chrf++, chrf++_stderr, which lighteval computes from its first" in logged


TRANSLATIONS = [
    ({"golds": ["Das Haus ist rot."], "preds": ["Das Haus ist rot."]}),
    ({"golds": ["Der Hund schläft."], "preds": ["Ein Hund schläft gern."]}),
    ({"golds": ["Wir essen Brot."], "preds": ["Wir trinken Wasser."]}),
]


def test_corpus_bleu_is_scored_as_lighteval_scores_it():
    sacrebleu = pytest.importorskip("sacrebleu")
    preds = [t["preds"][0] for t in TRANSLATIONS]
    golds = [t["golds"][0] for t in TRANSLATIONS]
    expected = sacrebleu.BLEU().corpus_score(preds, [golds]).score
    assert sample_exclusion._corpus_bleu("", TRANSLATIONS) == expected


def test_corpus_bleu_needs_sacrebleu(monkeypatch):
    monkeypatch.setitem(sys.modules, "sacrebleu", None)  # import fails
    with pytest.raises(RunNotFilterable, match="needs sacrebleu"):
        sample_exclusion._corpus_bleu("", TRANSLATIONS)


@pytest.mark.parametrize("metric_type", ["chrf++", "chrf", "ter"])
def test_corpus_chrf_and_ter_have_no_score_without_blocklisted_samples(metric_type):
    """lighteval passes each sentence's references as a reference set of their
    own, so sacrebleu scores the first translation alone."""
    config = {
        "metrics": [
            {
                "metric_name": "chrf++",
                "corpus_level_fn": f"CorpusLevelTranslationMetric(metric_type={metric_type}, lang=)",
            }
        ]
    }
    samples = [
        {"doc_id": i, "doc": {}, "metric": {"chrf++": t}}
        for i, t in enumerate(TRANSLATIONS)
    ]
    result = {"chrf++": 50.0, "chrf++_stderr": 1.0}
    clean = sample_exclusion._lighteval_scores("t", result, config, samples, {1})
    assert clean == {"chrf++": None}


# --- evalchemy's MATH500 -----------------------------------------------------

# Correct model answers that only lm-eval's is_equiv accepts, by question.
EQUIVALENT_ANSWERS = {
    0: ("\\frac{1}{2}", "1/2"),
    1: ("7", "x = 7"),
    2: ("90", "90^\\circ"),
}


def _math500_question(i):
    answer = EQUIVALENT_ANSWERS.get(i, (str(i),))[0]
    return {
        "problem": f"What is {i}?",
        "solution": f"It is $\\boxed{{{answer}}}$.",
        "answer": answer,
        "subject": "Algebra",
        "level": i % 5 + 1,
        "unique_id": f"test/algebra/{i}.json",
    }


def _write_math500_run(run_dir, solved, n=500, num_solved=None, task="MATH500"):
    """evalchemy's results file: every question with the model's answer, which
    is right for the questions in *solved*."""
    examples = []
    for i in range(n):
        question = _math500_question(i)
        right = EQUIVALENT_ANSWERS.get(i, (None, question["answer"]))[1]
        answer = right if i in solved else "wrong"
        examples.append(
            question | {"model_output": f"\\boxed{{{answer}}}", "model_answer": answer}
        )
    num_solved = len(solved) if num_solved is None else num_solved
    result = {
        "examples": examples,
        "num_total": n,
        "num_solved": num_solved,
        "accuracy": num_solved / n,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    results_file = run_dir / f"results_{TS}.json"
    results_file.write_text(
        json.dumps(
            {
                "results": {task: result},
                "config": {"model": "hf", "limit": None},
                "model_name": "m",
            }
        )
    )
    return results_file


def test_math500_is_rescored_without_blocklisted_questions(tmp_path):
    run = _write_math500_run(tmp_path, solved=set(range(10)))
    blocklist = {"Math500": [_math500_question(0), _math500_question(100)]}
    clean, n_excluded = _clean(run, blocklist, {"Math500": ["MATH500"]})["MATH500"]
    assert n_excluded == 2
    assert clean["num_solved"] == 9
    assert clean["num_total"] == 498
    assert clean["accuracy"] == 9 / 498


def test_math500_answers_that_do_not_reproduce_num_solved_are_refused(tmp_path):
    run = _write_math500_run(tmp_path, solved=set(range(10)), num_solved=11)
    with pytest.raises(
        SampleExclusionError, match="solves 10 questions, but evalchemy reported 11"
    ):
        _clean(run, {"Math500": [_math500_question(0)]}, {"Math500": ["MATH500"]})


def test_a_math500_run_on_part_of_the_questions_is_refused(tmp_path):
    run = _write_math500_run(tmp_path, solved={0}, n=2)
    with pytest.raises(RunNotFilterable, match="scored 2 of 500 questions"):
        _clean(run, {"Math500": [_math500_question(0)]}, {"Math500": ["MATH500"]})


def test_other_evalchemy_benchmarks_are_refused(tmp_path):
    run = _write_math500_run(tmp_path, solved={0}, task="AIME24")
    with pytest.raises(
        SampleExclusionError, match="AIME24: excluding samples from this evalchemy"
    ):
        _clean(run, {"Aime": [_math500_question(0)]}, {"Aime": ["AIME24"]})


def test_collect_reports_full_and_clean_math500_scores(tmp_path, monkeypatch):
    _write_math500_run(tmp_path / "run", solved=set(range(10)))
    blocklist = tmp_path / "blocklist"
    blocklist.mkdir()
    row = _math500_question(0) | {"_matched_ngrams": ["what is 0"]}
    (blocklist / "Math500.jsonl").write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(
        sample_exclusion, "load_task_map", lambda: {"Math500": ["MATH500"]}
    )
    out = tmp_path / "out.csv"
    collect_results(str(tmp_path / "run"), str(out), exclude_samples=str(blocklist))
    df = pd.read_csv(out).set_index(["subset", "metric_name"])
    assert set(df.task) == {"MATH500"}
    assert df.loc[("full", "accuracy"), "performance"] == 10 / 500
    assert df.loc[("clean", "accuracy"), "performance"] == pytest.approx(9 / 499)
    assert df.loc[("clean", "num_solved"), "performance"] == 9
    assert df.loc[("clean", "num_total"), "performance"] == 499
    assert set(df.loc["clean", "n_excluded"]) == {1}


# --- the task map ------------------------------------------------------------


def test_every_mapped_task_is_in_the_task_groups():
    groups = _load_task_groups_data()["task_groups"]
    tasks = {t["task"] for g in groups.values() for t in g.get("tasks") or []}
    # WSC's items are in xwinograd_en alone, of the group xwinograd.
    tasks.add("xwinograd_en")
    for benchmark, mapped in sample_exclusion.load_task_map().items():
        assert set(mapped) <= tasks, benchmark


def test_every_mapped_evalchemy_task_can_be_rescored():
    groups = _load_task_groups_data()["task_groups"]
    evalchemy = {
        t["task"]
        for g in groups.values()
        for t in g.get("tasks") or []
        if t.get("suite", g.get("suite")) == "evalchemy"
    }
    mapped = {t for ts in sample_exclusion.load_task_map().values() for t in ts}
    assert mapped & evalchemy <= sample_exclusion._EVALCHEMY_BENCHMARKS.keys()
    assert EVALCHEMY_QUESTIONS.keys() == sample_exclusion._EVALCHEMY_BENCHMARKS.keys()


# --- alignment tables --------------------------------------------------------


def test_an_item_key_is_a_hash_of_the_key_fields_values():
    """Tables already built keep their keys: this pins the format."""
    assert (
        sample_exclusion.item_key(["p", 1])
        == hashlib.sha256(b'["p", 1]').hexdigest()[:16]
    )
    assert (
        sample_exclusion.item_key(["ü"])
        == hashlib.sha256('["ü"]'.encode()).hexdigest()[:16]
    )


@pytest.mark.parametrize("benchmark", sorted(sample_exclusion.load_alignments()))
def test_an_alignment_table_is_well_formed(benchmark):
    table = sample_exclusion.load_alignments()[benchmark]
    entry = sample_exclusion._load_benchmarks()[benchmark]
    assert set(table) == {"key", "items", "tasks"}
    # Rows are cut down to match_on when loaded, so keep the table's key.
    assert set(table["key"]) <= set(entry.get("match_on") or table["key"])
    assert table["items"]
    assert all(len(k) == 16 and int(k, 16) >= 0 for k in table["items"])
    # A mapped task, or a subtask of a mapped group (global_mmlu_full_de's).
    for task in table["tasks"]:
        assert any(task == t or task.startswith(f"{t}_") for t in entry["tasks"]), task
    for task, e in table["tasks"].items():
        assert e["documents"] > 0 and len(e["answers"]) == 16, task
        if "first" in e:
            assert set(e) == {"documents", "answers", "first"}, task
            assert 0 <= e["first"] <= len(table["items"]) - e["documents"], task
        else:
            assert set(e) == {"documents", "answers", "positions"}, task
            assert len(e["positions"]) == len(table["items"]), task
            taken = [p for p in e["positions"] if p is not None]
            assert taken, task
            # XWinograd repeats WSC items, so their tables share documents.
            if benchmark not in {"WinogradSchemaChallenge", "XWinograd"}:
                assert len(set(taken)) == len(taken), task
            assert all(isinstance(p, int) and 0 <= p < e["documents"] for p in taken), (
                task
            )


def _builder():
    path = Path(__file__).parents[1] / "scripts/build_sample_exclusion_alignments.py"
    spec = importlib.util.spec_from_file_location("build_alignments", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_translations_in_the_same_order_are_put_together_by_position():
    build = _builder()
    english = [{"label": i % 3} for i in range(4)]
    assert build.by_position("X", english, "x_de", english, lambda d: d["label"]) == [
        0,
        1,
        2,
        3,
    ]
    with pytest.raises(build.AlignmentError, match=r"x_de's documents \[0, 1\]"):
        build.by_position(
            "X", english, "x_de", english[1::-1] + english[2:], lambda d: d["label"]
        )
    with pytest.raises(build.AlignmentError, match="has 3 documents, not 4"):
        build.by_position("X", english, "x_de", english[:3], lambda d: d["label"])


def test_a_few_mislabelled_translations_can_be_tolerated(capsys):
    """MMMLU's German labels 5 of MMLU's 14,042 answers differently; documents
    out of order would differ in most."""
    build = _builder()
    english = [{"label": i % 4} for i in range(100)]
    german = [dict(d) for d in english]
    german[7]["label"] = 0
    assert build.by_position(
        "X", english, "x_de", german, lambda d: d["label"], 0.02
    ) == list(range(100))
    assert "x_de's documents [7] have another answer" in capsys.readouterr().out
    with pytest.raises(build.AlignmentError):
        build.by_position("X", english, "x_de", german, lambda d: d["label"])
    shifted = english[1:] + english[:1]
    with pytest.raises(build.AlignmentError):
        build.by_position("X", english, "x_de", shifted, lambda d: d["label"], 0.02)


def test_translations_in_another_order_are_put_together_by_a_shared_field():
    build = _builder()
    english = [{"id": i, "answer": i % 2} for i in range(4)]
    german = [english[2], english[0], english[3]]
    assert build.by_field("X", english, "id", "x_de", german) == [1, None, 0, 2]
    with pytest.raises(build.AlignmentError, match="another answer"):
        build.by_field(
            "X",
            english,
            "id",
            "x_de",
            [dict(english[0], answer=1)],
            lambda d: d["answer"],
        )
    with pytest.raises(build.AlignmentError, match="repeats id 0"):
        build.by_field("X", english, "id", "x_de", [english[0], english[0]])


def test_a_table_entry_lists_positions_only_where_items_are_not_in_a_row():
    build = _builder()
    answers = sample_exclusion.item_key(["a", "b"])
    assert build.entry([0, 1], ["a", "b"]) == {
        "documents": 2,
        "answers": answers,
        "first": 0,
    }
    assert build.entry([None, 0, 1, None], ["a", "b"])["first"] == 1
    assert build.entry([1, None, 0], ["a", "b"])["positions"] == [1, None, 0]
    # Every document must be an item for a run of them.
    assert build.entry([0, None], ["a", "b"])["positions"] == [0, None]
    for positions in ([1, 1], [2, 0], [None, None]):
        with pytest.raises(build.AlignmentError):
            build.entry(positions, ["a", "b"])
    # XWinograd repeats WSC items, so their tables share documents.
    assert build.entry([1, 1], ["a", "b"], shared=True)["positions"] == [1, 1]


def test_other_copies_of_gsm8k_s_first_questions_are_found_by_their_answers():
    """PolyMath's low level is GSM8K's first 125 test questions, MGSM's first
    250; GSM8K gives its answer after "####", MGSM as answer_number."""
    build = _builder()
    mgsm = [{"answer_number": n} for n in (18, 3, 70000, 540)]
    gsm8k = [{"answer": f"So it is {n:,}.\n#### {n:,}"} for n in (18, 3, 70000, 540, 20)]
    polymath = [{"answer": str(n)} for n in (18, 3)]
    assert build.first_items("X", mgsm, "gsm8k", gsm8k, 4) == [0, 1, 2, 3]
    assert build.first_items("X", mgsm, "p", polymath, 2) == [0, 1, None, None]
    with pytest.raises(build.AlignmentError, match=r"p's documents \[0, 1\]"):
        build.first_items("X", mgsm, "p", polymath[::-1], 2)
    with pytest.raises(build.AlignmentError, match="has 2 documents, not 3"):
        build.first_items("X", mgsm, "p", polymath, 3)


def test_x_csqa_s_test_questions_are_found_in_commonsense_qa():
    """By their question and choices, in any order; the test split has no
    answers."""
    build = _builder()
    csqa = [
        {"question": "Where is a cat?", "choices": {"text": ["home", "zoo"]}},
        {"question": "Where  is a dog?", "choices": {"text": ["park", "home"]}},
    ]
    test = [
        {"id": "x0", "question": {"stem": "Where is a dog?", "choices": {"text": ["home", "park"]}}},
        {"id": "x1", "question": {"stem": "Where is a cat?", "choices": {"text": ["home", "car"]}}},
    ]  # fmt: skip

    class Lm:
        manager = types.SimpleNamespace(
            load_task_or_group=lambda names: {
                "xcsqa_eng_Latn": types.SimpleNamespace(dataset={"test": test})
            }
        )

        def docs(self, name):
            assert name == "commonsense_qa"
            return csqa, ["0", "1"]

    table = build.xcsqa(Lm())
    assert table["items"] == [sample_exclusion.item_key([i]) for i in ("x0", "x1")]
    assert table["tasks"]["commonsense_qa"]["positions"] == [1, None]


def test_wsc_items_are_xwinograd_sentences_with_one_word_blanked():
    build = _builder()
    wsc = [
        {"text": "The cup broke the vase because it was hard."},
        {"text": "The cup broke the vase because it was fragile."},
        {"text": "Since it rained, Ann kept the book dry in her bag."},
    ]
    english = [
        {"sentence": "The cup broke the vase because _ was hard."},
        {"sentence": "The cup broke the vase because _ was hard."},  # swapped options
        {"sentence": "Since _ rained, Ann kept the book dry in her bag."},
        {"sentence": "The vase broke because _ was fragile."},
    ]

    class Docs:
        def docs(self, name):
            return {"wsc273": wsc, "xwinograd_en": english}[name], []

    assert build.winograd_pairs(Docs()) == [(0, 0), (0, 1), (2, 2)]
    # A blanked word that tells two WSC sentences apart would pair both.
    english.append({"sentence": "The cup broke the vase because it was _."})
    with pytest.raises(build.AlignmentError, match="xwinograd_en 4 is several"):
        build.winograd_pairs(Docs())


def test_items_must_be_told_apart_by_the_key():
    build = _builder()
    docs = [{"q": "a", "n": 1}, {"q": "a", "n": 2}]
    assert len(build.items_of(["q", "n"], docs, "X")) == 2
    with pytest.raises(build.AlignmentError, match="does not identify one document each"):
        build.items_of(["q"], docs, "X")
    assert len(set(build.items_of(["q"], docs, "X", repeats=True))) == 1


@pytest.mark.parametrize(
    "argv",
    [
        ["lm-eval", "--languages", "xx", "MMLU"],
        ["lm-eval", "MMLU", "--languages", "xx"],
    ],
)
def test_the_languages_of_a_part_are_checked_wherever_they_are_given(
    argv, monkeypatch, capsys
):
    build = _builder()
    monkeypatch.setattr(sys, "argv", ["build", *argv])
    with pytest.raises(SystemExit):
        build.main()
    assert "no MMLU tasks in ['xx']" in capsys.readouterr().err


def test_only_a_table_built_in_parts_takes_languages(monkeypatch, capsys):
    build = _builder()
    monkeypatch.setattr(sys, "argv", ["build", "lm-eval", "--languages", "de", "XNLI"])
    with pytest.raises(SystemExit):
        build.main()
    assert "--languages builds part of MMLU only" in capsys.readouterr().err


def test_a_table_built_in_parts_merges_their_tasks(tmp_path):
    build = _builder()
    path = tmp_path / "MMLU.json"
    first = {"key": ["q"], "items": ["a", "b"], "tasks": {"x_de": {"documents": 2}}}
    assert build.merged(path, first) == first  # nothing built yet
    path.write_text(json.dumps(first))
    second = dict(first, tasks={"x_fr": {"documents": 2}})
    assert build.merged(path, second)["tasks"].keys() == {"x_de", "x_fr"}
    with pytest.raises(build.AlignmentError, match="built from other items"):
        build.merged(path, dict(second, items=["a", "c"]))


@pytest.mark.parametrize(
    "text, value",
    [
        ("18", "18.0"),
        (18, "18.0"),
        ("1,000", "1000.0"),
        ("1 000", "1000.0"),
        ("-2.5", "-2.5"),
    ],
)
def test_answers_are_compared_as_numbers(text, value):
    assert _builder().number(text) == value


def test_an_answer_that_is_not_a_number_is_refused():
    build = _builder()
    with pytest.raises(build.AlignmentError, match="not a number"):
        build.number("eighteen")


@pytest.mark.skipif(not BLOCKLIST_DIR.is_dir(), reason="set OELLM_BLOCKLIST_DIR")
def test_every_mapped_row_matches_its_lm_eval_documents():
    """Each blocklisted row of a mapped benchmark is a document of its tasks
    (one in each language), the documents of a task in its alignment table are
    the ones the table was built from, and each metric whose aggregation collect can recompute is read as lm-eval runs
    it, from the config lm-eval saves in the results file.

    Needs lm-eval and the datasets in the Hugging Face cache, e.g. inside the
    eval container on LUMI, and HF_ALLOW_CODE_EVAL=1: loading mbpp runs a check
    of HF's code_eval metric. Benchmarks without a file in the blocklist are
    not loaded.
    """
    pytest.importorskip("lm_eval")
    from lm_eval.tasks import TaskManager

    match_fields = sample_exclusion.load_match_fields()
    blocklist = sample_exclusion.load_blocklist(BLOCKLIST_DIR, match_fields)
    alignments = sample_exclusion.load_alignments()
    normalized = sample_exclusion.load_normalized()
    # oellm's own tasks (global_mgsm, polymath), as the job template passes them.
    include_path = (
        Path(sample_exclusion.__file__).parent / "resources/custom_lm_eval_tasks"
    )
    manager = TaskManager(include_path=str(include_path))
    for benchmark, mapped in sample_exclusion.load_task_map().items():
        # lm-eval's tasks only, not lighteval's (belebele_*_cf) or evalchemy's.
        mapped = [t for t in mapped if t in manager.task_index]
        if not mapped or benchmark not in blocklist:
            continue
        loaded = manager.load(mapped)["tasks"]
        samples, stack = {}, [loaded]
        while stack:
            for name, task in stack.pop().items():
                if isinstance(task, dict):
                    stack.append(task)
                else:
                    samples[name] = [
                        {
                            "doc_id": i,
                            "doc": json.loads(json.dumps(doc)),
                            "target": str(task.doc_to_target(doc)),
                        }
                        for i, doc in enumerate(task.eval_docs)
                    ]
                    config = json.loads(json.dumps(task.dump_config(), default=str))
                    for metric, aggregation in task.aggregation().items():
                        ours = sample_exclusion._aggregation(metric, config)
                        if {
                            ours,
                            aggregation.__name__,
                        } & sample_exclusion._AGGREGATIONS.keys():
                            assert ours == aggregation.__name__, (name, metric)
        rows = blocklist[benchmark]
        table = alignments.get(benchmark, {"tasks": {}})
        aligned = {n: s for n, s in samples.items() if n in table["tasks"]}
        others = {n: s for n, s in samples.items() if n not in aligned}
        found = [
            # With match_on, also checks that the documents have its fields.
            sample_exclusion._excluded_docs(
                rows,
                others,
                benchmark,
                benchmark in normalized,
                benchmark in match_fields,
            )
        ]
        if aligned:
            # Also checks each task's documents against the table.
            found.append(
                sample_exclusion._aligned_docs(rows, aligned, table, benchmark, False)
            )
        matched = [any(flags) for flags in zip(*(f for _, f in found), strict=True)]
        outside = ROWS_OUTSIDE_EVALUATED_COPY.get(benchmark, 0)
        assert matched.count(False) == outside, benchmark
        # A row is one item: at most one document of each task, but for
        # duplicated documents (MMLU repeats some questions).
        for excluded, _ in found:
            for name, doc_ids in excluded.items():
                assert len(doc_ids) <= len(rows), (benchmark, name)


@pytest.mark.skipif(
    not (BLOCKLIST_DIR.is_dir() and EVALCHEMY_DIR.is_dir()),
    reason="set OELLM_BLOCKLIST_DIR and EVALCHEMY_DIR",
)
def test_every_mapped_row_matches_one_evalchemy_question():
    """Each blocklisted row of a benchmark mapped to evalchemy is one of the
    questions evalchemy loads, which its results file keeps as they are."""
    blocklist = sample_exclusion.load_blocklist(BLOCKLIST_DIR)
    for benchmark, mapped in sample_exclusion.load_task_map().items():
        for task in set(mapped) & EVALCHEMY_QUESTIONS.keys():
            lines = (EVALCHEMY_DIR / EVALCHEMY_QUESTIONS[task]).read_text().splitlines()
            samples = {
                task: [{"doc_id": i, "doc": json.loads(x)} for i, x in enumerate(lines)]
            }
            excluded, matched = sample_exclusion._excluded_docs(
                blocklist[benchmark], samples
            )
            assert all(matched), benchmark
            assert len(excluded[task]) == len(blocklist[benchmark]), benchmark
