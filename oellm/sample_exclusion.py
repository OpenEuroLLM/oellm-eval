"""Scores recomputed without blocklisted samples, for ``collect --exclude_samples``.

A blocklist is a directory with one ``<benchmark>.jsonl`` file per benchmark,
each line an original row of that benchmark's dataset, for example the rows a
decontamination run found in the training data. Fields starting with ``_`` are
annotations and ignored. ``resources/sample-exclusion.yaml`` maps each
benchmark to the lm-eval tasks or groups, or evalchemy benchmarks, that hold
its documents.

The samples lm-eval saves with ``--log_samples`` are matched against the rows
by content, and every per-sample mean metric is recomputed without them, the
way lm-eval computes it. evalchemy saves no samples, but MATH500 keeps every
question and model answer in its results file, so its questions are re-scored
the way evalchemy scores them. Before anything is excluded the recomputation
has to reproduce the reported scores, and anything that cannot be recomputed
exactly is an error rather than an approximate number.
"""

import json
import math
import re
from importlib.resources import files
from numbers import Real
from pathlib import Path

import yaml

from oellm.hendrycks_math import is_equiv

# Metrics lm-eval aggregates with a mean when a task's metric_list does not
# name them, by output type.
_DEFAULT_MEAN_METRICS = {
    "multiple_choice": {"acc", "acc_norm", "acc_mutual_info"},
    "loglikelihood": {"acc"},
    "generate_until": {"exact_match"},
}
# Metrics registered in lm-eval with a mean as their default aggregation, which
# lm-eval uses when a metric_list entry gives none (boolq's acc).
_REGISTERED_MEAN_METRICS = {
    "acc",
    "acc_norm",
    "acc_mutual_info",
    "acc_bytes",
    "acc_all",
    "exact_match",
    "likelihood",
}

# evalchemy benchmarks whose results keep every question and model answer
# (``examples``): their number of questions, and whether a question is solved,
# as evalchemy decides it.
_EVALCHEMY_BENCHMARKS = {
    "MATH500": (500, lambda ex: is_equiv(str(ex["answer"]), ex["model_answer"])),
}


class SampleExclusionError(ValueError):
    """A score cannot be recomputed exactly without the blocklisted samples."""


class RunNotFilterable(SampleExclusionError):
    """A run that cannot give scores without blocklisted samples at all: it has
    no saved samples, or scored only part of a benchmark. collect skips it."""


def load_blocklist(path: str | Path) -> dict[str, list[dict]]:
    """Read ``<benchmark>.jsonl`` files from *path*, without ``_`` fields."""
    path = Path(path)
    if not path.is_dir():
        raise SampleExclusionError(f"Blocklist directory does not exist: {path}")
    blocklist = {}
    for file in sorted(path.glob("*.jsonl")):
        rows = []
        with open(file, encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                if not line.strip():
                    continue
                where = f"{file}, line {line_number}"
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    raise SampleExclusionError(f"{where}: not valid JSON ({e})") from e
                if not isinstance(row, dict):
                    raise SampleExclusionError(
                        f"{where}: expected a JSON object (one dataset row), "
                        f"got a {type(row).__name__}"
                    )
                row = {k: v for k, v in row.items() if not k.startswith("_")}
                if not row:
                    # A row without fields would match every document.
                    raise SampleExclusionError(
                        f"{where}: the row has only `_` annotation fields"
                    )
                rows.append(row)
        blocklist[file.stem] = rows
    if not blocklist:
        raise SampleExclusionError(f"No .jsonl files in blocklist directory {path}")
    return blocklist


def load_task_map() -> dict[str, list[str]]:
    """Benchmark name -> the tasks or groups holding its documents."""
    text = (files("oellm.resources") / "sample-exclusion.yaml").read_text()
    return yaml.safe_load(text)["benchmarks"]


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _is_number(value) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _close(a: float, b: float) -> bool:
    # lm-eval may add the same values in another order.
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _mean_stderr(values: list[float]) -> float:
    # lm-eval's mean_stderr: sample standard deviation over sqrt(n).
    mu = _mean(values)
    return math.sqrt(sum((x - mu) ** 2 for x in values) / (len(values) - 1)) / math.sqrt(
        len(values)
    )


def _pooled_stderr(stderrs: list[float], sizes: list[int]) -> float:
    # lm-eval's pooled_sample_stderr, for groups weighted by subtask size.
    pooled_var = sum(
        (n - 1) * se**2 * n for n, se in zip(sizes, stderrs, strict=True)
    ) / (sum(sizes) - len(sizes))
    return math.sqrt(pooled_var / sum(sizes))


def _unweighted_stderr(stderrs: list[float], sizes: list[int]) -> float:
    # lm-eval's unweighted_mean_stderr, for groups not weighted by size.
    del sizes  # same signature as _pooled_stderr
    return math.sqrt(sum(se**2 for se in stderrs)) / len(stderrs)


def _is_mean_metric(metric: str, config: dict) -> bool:
    for entry in config.get("metric_list") or []:
        name = str(entry.get("metric"))
        # A metric given as a function (mbpp's utils.pass_at_1) is saved as its
        # source code.
        if name == metric or re.search(rf"^def {re.escape(metric)}\(", name, re.M):
            if "aggregation" not in entry:
                return metric in _REGISTERED_MEAN_METRICS
            return entry["aggregation"] == "mean"
    return metric in _DEFAULT_MEAN_METRICS.get(config.get("output_type"), set())


def _metric_keys(result: dict) -> list[str]:
    """``metric,filter`` keys of a result, without stderrs."""
    return [k for k in result if "," in k and "_stderr," not in k]


def _in_order_of(result: dict, scores: dict) -> dict:
    # collect keeps the first of several filters' numeric values for a metric,
    # so the recomputed keys must come in the order of the reported ones. A
    # metric whose first filter was not recomputed is left out, rather than
    # paired with another filter's value.
    first: dict[str, str] = {}
    for key, value in result.items():
        if _is_number(value):
            first.setdefault(key.split(",")[0], key)
    return {
        k: scores[k]
        for k in result
        if k in scores and first.get(k.split(",")[0]) in scores
    } | {"sample_len": scores["sample_len"]}


def _leaves(name: str, group_subtasks: dict) -> list[str]:
    # A plain task is listed in group_subtasks too, with no subtasks.
    subtasks = group_subtasks.get(name) or []
    if not subtasks:
        return [name]
    return [leaf for sub in subtasks for leaf in _leaves(sub, group_subtasks)]


def _read_samples(results_file: Path, task: str, data: dict) -> list[dict]:
    # lm-eval names the results results_<date>.json when --output_path is a
    # directory and <name>_<date>.json when it is <name>.json, as in oellm's
    # job template. The samples are samples_<task>_<date>.jsonl either way, and
    # the date (an ISO timestamp with "-" for ":") has no underscore.
    timestamp = results_file.stem.rsplit("_", 1)[-1]
    path = results_file.with_name(f"samples_{task}_{timestamp}.jsonl")
    if not path.exists():
        raise RunNotFilterable(
            f"{task}: no saved samples next to {results_file} (expected {path.name}). "
            "Excluding samples needs a run scheduled with --log_samples."
        )
    counts = data.get("n-samples", {}).get(task)
    if not counts:
        raise RunNotFilterable(
            f"{task}: {results_file} does not say how many documents were scored."
        )
    if counts.get("effective") != counts.get("original"):
        raise RunNotFilterable(
            f"{task}: {results_file} scored {counts.get('effective')} of "
            f"{counts.get('original')} documents (a --limit run?). Excluding "
            "samples needs a run over the whole benchmark."
        )
    with open(path, encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]
    if len({s["doc_id"] for s in samples}) != counts.get("effective"):
        raise SampleExclusionError(
            f"{task}: {path.name} holds {len({s['doc_id'] for s in samples})} "
            f"documents, but lm-eval scored {counts.get('effective')}."
        )
    return samples


def _is_evalchemy(result: dict) -> bool:
    return isinstance(result.get("examples"), list)


def _read_examples(results_file: Path, task: str, result: dict) -> list[dict]:
    """evalchemy's questions, in the layout of lm-eval's saved samples."""
    if task not in _EVALCHEMY_BENCHMARKS:
        raise SampleExclusionError(
            f"{task}: excluding samples from this evalchemy benchmark is not supported."
        )
    size, _ = _EVALCHEMY_BENCHMARKS[task]
    examples = result["examples"]
    if len(examples) != size:
        raise RunNotFilterable(
            f"{task}: {results_file} scored {len(examples)} of {size} questions "
            "(a --debug run?). Excluding samples needs a run over the whole benchmark."
        )
    return [{"doc_id": i, "doc": example} for i, example in enumerate(examples)]


def _evalchemy_scores(
    task: str, result: dict, examples: list[dict], excluded: set[int]
) -> dict:
    """*result* recomputed without *excluded*, by re-scoring every question."""
    _, solved = _EVALCHEMY_BENCHMARKS[task]
    correct = [bool(solved(s["doc"])) for s in examples]
    if sum(correct) != result.get("num_solved"):
        raise SampleExclusionError(
            f"{task}: re-scoring the saved answers solves {sum(correct)} questions, "
            f"but evalchemy reported {result.get('num_solved')}; refusing to "
            "recompute it."
        )
    kept = [
        c for c, s in zip(correct, examples, strict=True) if s["doc_id"] not in excluded
    ]
    if len(kept) < 2:
        raise SampleExclusionError(
            f"{task}: {len(kept)} questions remain after exclusion"
        )
    scores = {
        "sample_len": len(kept),
        "num_total": len(kept),
        "num_solved": sum(kept),
        "accuracy": sum(kept) / len(kept),
    }
    return _in_order_of(result, scores)


def _excluded_docs(
    rows: list[dict], samples_by_task: dict[str, list[dict]], benchmark: str
) -> dict[str, set[int]]:
    """Doc ids per task whose document equals a blocklist row on all its fields."""
    index: dict[tuple, dict[str, list[tuple[str, int]]]] = {}
    excluded: dict[str, set[int]] = {}
    for row in rows:
        fields = tuple(sorted(row))
        if fields not in index:
            by_key = index[fields] = {}
            for task, samples in samples_by_task.items():
                seen = set()
                for s in samples:
                    doc = s["doc"]
                    if s["doc_id"] in seen or not all(f in doc for f in fields):
                        continue
                    seen.add(s["doc_id"])
                    key = _canonical({f: doc[f] for f in fields})
                    by_key.setdefault(key, []).append((task, s["doc_id"]))
        hits = index[fields].get(_canonical(row))
        if not hits:
            raise SampleExclusionError(
                f"{benchmark}: a blocklist row matches no document of "
                f"{', '.join(samples_by_task)}: {_canonical(row)[:300]}"
            )
        for task, doc_id in hits:
            excluded.setdefault(task, set()).add(doc_id)
    return excluded


def _task_scores(
    task: str, result: dict, config: dict, samples: list[dict], excluded: set[int]
) -> dict:
    """*result* recomputed without *excluded*, for every per-sample mean metric."""
    remaining = {s["doc_id"] for s in samples} - excluded
    if len(remaining) < 2:
        raise SampleExclusionError(
            f"{task}: {len(remaining)} documents remain after exclusion"
        )
    scores = {"sample_len": len(remaining)}
    for key in _metric_keys(result):
        metric, filter_name = key.split(",", 1)
        if not _is_number(result[key]) or not _is_mean_metric(metric, config):
            continue
        lines = [s for s in samples if s["filter"] == filter_name]
        full = [s.get(metric) for s in lines]
        if not full or not all(_is_number(v) for v in full):
            continue
        if not _close(_mean(full), result[key]):
            raise SampleExclusionError(
                f"{task}: the saved samples give {key} = {_mean(full)}, but lm-eval "
                f"reported {result[key]}; refusing to recompute it."
            )
        kept = [s[metric] for s in lines if s["doc_id"] not in excluded]
        scores[key] = _mean(kept)
        stderr_key = f"{metric}_stderr,{filter_name}"
        if _is_number(result.get(stderr_key)):
            if not _close(_mean_stderr(full), result[stderr_key]):
                raise SampleExclusionError(
                    f"{task}: the saved samples give {stderr_key} = "
                    f"{_mean_stderr(full)}, but lm-eval reported {result[stderr_key]}."
                )
            scores[stderr_key] = _mean_stderr(kept)
    return _in_order_of(result, scores)


def _group_mean(mode: str, values: list[float], sizes: list[int]) -> float:
    # lm-eval's aggregate_subtask_metrics, with or without weight_by_size.
    if mode == "unweighted":
        return _mean(values)
    return sum(v * n for v, n in zip(values, sizes, strict=True)) / sum(sizes)


_GROUP_STDERR = {"weighted": _pooled_stderr, "unweighted": _unweighted_stderr}


def _group_scores(
    group: str, result: dict, leaves: list[str], full: dict, clean: dict
) -> dict:
    """*result* of a group recomputed from its recomputed leaf tasks.

    lm-eval scores a group as the mean of its leaf tasks, weighted by their
    size or not. The results file does not say which, so the weighting is the
    one that reproduces the reported score and stderr. If both do, they must
    also agree once the blocklisted samples are gone.
    """
    if not all("sample_len" in full[leaf] for leaf in leaves):
        raise SampleExclusionError(
            f"{group}: its tasks' results have no sample_len, which lm-eval writes "
            "from 0.4.12 on. Recomputing a group needs a run made with lm-eval "
            "0.4.12 or later."
        )
    scores = {"sample_len": sum(clean[leaf]["sample_len"] for leaf in leaves)}
    sizes = [full[leaf]["sample_len"] for leaf in leaves]
    clean_sizes = [clean[leaf]["sample_len"] for leaf in leaves]
    for key in _metric_keys(result):
        if not _is_number(result[key]) or not all(key in clean[leaf] for leaf in leaves):
            continue
        metric, filter_name = key.split(",", 1)
        stderr_key = f"{metric}_stderr,{filter_name}"
        with_stderr = _is_number(result.get(stderr_key)) and all(
            stderr_key in clean[leaf] for leaf in leaves
        )

        values = [full[leaf][key] for leaf in leaves]
        stderrs = [full[leaf][stderr_key] for leaf in leaves] if with_stderr else []
        modes = [
            mode
            for mode in _GROUP_STDERR
            if _close(_group_mean(mode, values, sizes), result[key])
            and (
                not with_stderr
                or _close(_GROUP_STDERR[mode](stderrs, sizes), result[stderr_key])
            )
        ]
        if not modes:
            raise SampleExclusionError(
                f"{group}: neither a weighted nor an unweighted mean of its tasks "
                f"reproduces the reported {key} and its stderr."
            )

        clean_values = [clean[leaf][key] for leaf in leaves]
        clean_stderrs = (
            [clean[leaf][stderr_key] for leaf in leaves] if with_stderr else []
        )
        outcomes = [
            (
                _group_mean(mode, clean_values, clean_sizes),
                _GROUP_STDERR[mode](clean_stderrs, clean_sizes) if with_stderr else 0.0,
            )
            for mode in modes
        ]
        value, stderr = outcomes[0]
        if any(not (_close(v, value) and _close(se, stderr)) for v, se in outcomes):
            raise SampleExclusionError(
                f"{group}: both a size-weighted and an unweighted mean of its tasks "
                f"reproduce the reported {key}, and they differ without the "
                "blocklisted samples, so it cannot be recomputed."
            )
        scores[key] = value
        if with_stderr:
            scores[stderr_key] = stderr
    return _in_order_of(result, scores)


def clean_results(
    results_file: Path,
    data: dict,
    blocklist: dict[str, list[dict]],
    task_map: dict[str, list[str]],
) -> dict[str, tuple[dict, int]]:
    """Results of the tasks and groups in *data* that lose blocklisted samples.

    Returns ``{name: (result, n_excluded)}``, where ``result`` has the layout of
    the reported one (lm-eval's ``metric,filter`` keys, or evalchemy's
    ``accuracy``, ``num_solved`` and ``num_total``) and holds only the
    recomputed metrics. Tasks without blocklisted documents are left out.
    """
    results = data.get("results", {})
    groups = data.get("groups", {})
    group_subtasks = data.get("group_subtasks", {})

    leaves_by_benchmark: dict[str, list[str]] = {}
    for benchmark, rows in blocklist.items():
        targets = [t for t in task_map.get(benchmark, []) if t in results or t in groups]
        if rows and targets:
            leaves_by_benchmark[benchmark] = [
                leaf
                for t in targets
                for leaf in _leaves(t, group_subtasks)
                if leaf in results
            ]
    if not leaves_by_benchmark:
        return {}
    if data.get("config", {}).get("samples"):
        raise RunNotFilterable(
            f"{results_file} was evaluated on selected samples (--samples); "
            "excluding samples needs a run over the whole benchmark."
        )

    leaves = sorted({leaf for ls in leaves_by_benchmark.values() for leaf in ls})
    samples = {
        leaf: _read_examples(results_file, leaf, results[leaf])
        if _is_evalchemy(results[leaf])
        else _read_samples(results_file, leaf, data)
        for leaf in leaves
    }
    excluded: dict[str, set[int]] = {leaf: set() for leaf in leaves}
    for benchmark, benchmark_leaves in leaves_by_benchmark.items():
        found = _excluded_docs(
            blocklist[benchmark],
            {leaf: samples[leaf] for leaf in benchmark_leaves},
            benchmark,
        )
        for leaf, doc_ids in found.items():
            excluded[leaf] |= doc_ids

    full = {leaf: results[leaf] for leaf in leaves}
    clean = {
        leaf: _evalchemy_scores(leaf, results[leaf], samples[leaf], excluded[leaf])
        if _is_evalchemy(results[leaf])
        else _task_scores(
            leaf,
            results[leaf],
            data.get("configs", {}).get(leaf, {}),
            samples[leaf],
            excluded[leaf],
        )
        for leaf in leaves
    }
    out = {leaf: (clean[leaf], len(excluded[leaf])) for leaf in leaves if excluded[leaf]}
    for group, result in groups.items():
        group_leaves = [
            leaf for leaf in _leaves(group, group_subtasks) if leaf in results
        ]
        if not group_leaves or not set(group_leaves) <= set(leaves):
            continue
        n_excluded = sum(len(excluded[leaf]) for leaf in group_leaves)
        if n_excluded:
            out[group] = (
                _group_scores(group, result, group_leaves, full, clean),
                n_excluded,
            )
    return out
