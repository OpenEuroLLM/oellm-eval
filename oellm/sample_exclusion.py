"""Scores recomputed without blocklisted samples, for ``collect --exclude_samples``.

A blocklist is a directory with one ``<benchmark>.jsonl`` per benchmark, one
original dataset row per line; fields starting with ``_`` are ignored.
``resources/sample-exclusion.yaml`` maps each benchmark to the lm-eval,
lighteval or evalchemy tasks holding its documents, in any language or as
copies in other benchmarks.

Rows are matched to lm-eval's saved samples by content, to lighteval's by their
prompt text, or through an alignment table
(``resources/sample-exclusion-alignments``) giving an item's position in each
task. MATH500 is re-scored from the answers in evalchemy's results file. The
recomputation must first reproduce the reported scores; a score that cannot be
recomputed exactly is an error.
"""

import hashlib
import json
import logging
import math
import re
from importlib.resources import files
from numbers import Real
from pathlib import Path

import yaml

from oellm.hendrycks_math import is_equiv

# lm-eval's registered aggregations, used when a metric_list entry names none
# (boolq's acc) or a task has no metric_list.
_REGISTERED_AGGREGATIONS = {
    "acc": "mean",
    "acc_norm": "mean",
    "acc_mutual_info": "mean",
    "acc_bytes": "mean",
    "acc_all": "mean",
    "exact_match": "mean",
    "likelihood": "mean",
    "perplexity": "perplexity",
}
_DEFAULT_METRICS = {
    "multiple_choice": {"acc", "acc_norm", "acc_mutual_info"},
    "loglikelihood": {"acc", "perplexity"},
    "generate_until": {"exact_match"},
}


def _flat_mean(items: list) -> float:
    # ifeval's agg_inst_level_acc: every instruction of every prompt counts once.
    return _mean([x for item in items for x in item])


def _perplexity(items: list) -> float:
    return math.exp(-_mean(items))


# Aggregations recomputable from lm-eval's per-sample values: (score, stderr,
# whether a value is a list). A None stderr is bootstrapped (perplexity) or
# absent (ifeval), and not recomputed.
_AGGREGATIONS = {
    "mean": (lambda items: _mean(items), lambda items: _mean_stderr(items), False),
    "agg_inst_level_acc": (_flat_mean, None, True),
    "perplexity": (_perplexity, None, False),
}

# More matches in one task mean a row's fields do not identify a document
# (a few are duplicates: MMLU repeats some questions).
_MAX_MATCHES_PER_TASK = 10

# Alignment tables, for translations with no field identifying an item (XNLI)
# and for lighteval, which saves no dataset row.
_ALIGNMENTS = "sample-exclusion-alignments"

# evalchemy benchmarks whose results keep every question and answer
# (``examples``): their size, and whether a question is solved.
_EVALCHEMY_BENCHMARKS = {
    "MATH500": (500, lambda ex: is_equiv(str(ex["answer"]), ex["model_answer"])),
}


class SampleExclusionError(ValueError):
    """A score cannot be recomputed exactly without the blocklisted samples."""


class RunNotFilterable(SampleExclusionError):
    """A run that cannot be filtered at all (no saved samples, or only part of a
    benchmark). collect skips it."""


def load_blocklist(
    path: str | Path, match_fields: dict[str, list[str]] | None = None
) -> dict[str, list[dict]]:
    """Read ``<benchmark>.jsonl`` files from *path*, without ``_`` fields.

    The rows of a benchmark in *match_fields* keep only the fields named there.
    """
    path = Path(path)
    match_fields = match_fields or {}
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
                if file.stem in match_fields:
                    missing = [f for f in match_fields[file.stem] if f not in row]
                    if missing:
                        raise SampleExclusionError(
                            f"{where}: the row has no {', '.join(missing)}, which "
                            f"{file.stem} is matched on"
                        )
                    row = {f: row[f] for f in match_fields[file.stem]}
                rows.append(row)
        blocklist[file.stem] = rows
    if not blocklist:
        raise SampleExclusionError(f"No .jsonl files in blocklist directory {path}")
    return blocklist


def _load_benchmarks() -> dict[str, dict]:
    # An entry is a list of tasks, or {tasks: [...], match_on: [...]}.
    text = (files("oellm.resources") / "sample-exclusion.yaml").read_text()
    return {
        name: entry if isinstance(entry, dict) else {"tasks": entry}
        for name, entry in yaml.safe_load(text)["benchmarks"].items()
    }


def load_task_map() -> dict[str, list[str]]:
    """Benchmark name -> the tasks or groups holding its documents."""
    return {name: entry["tasks"] for name, entry in _load_benchmarks().items()}


def load_match_fields() -> dict[str, list[str]]:
    """Benchmark name -> the fields its rows are matched on, where not all."""
    return {
        name: entry["match_on"]
        for name, entry in _load_benchmarks().items()
        if entry.get("match_on")
    }


def load_normalized() -> set[str]:
    """Benchmarks whose text is matched by its letters and digits only."""
    return {name for name, entry in _load_benchmarks().items() if entry.get("normalize")}


def load_alignments() -> dict[str, dict]:
    """Benchmark name -> its alignment table.

    ``items`` holds :func:`item_key` of each item's ``key`` fields. For each
    task, ``tasks`` says where the items are: item ``first + i`` is document i,
    or item i is document ``positions[i]`` (``null`` if absent). ``documents``
    and ``answers`` (:func:`item_key` of the answers in order) describe the
    documents it was built from, which a run must match. Built by
    ``scripts/build_sample_exclusion_alignments.py``.
    """
    directory = files("oellm.resources") / _ALIGNMENTS
    if not directory.is_dir():
        return {}
    return {
        entry.name.removesuffix(".json"): json.loads(entry.read_text())
        for entry in directory.iterdir()
        if entry.name.endswith(".json")
    }


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def item_key(values: list) -> str:
    """An alignment table's key of an item, from the values of its key fields,
    or of a task's answers, from the list of them."""
    return hashlib.sha256(_canonical(values).encode()).hexdigest()[:16]


def _is_number(value) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _is_value(value, is_list: bool) -> bool:
    # A saved per-sample value: a number or a boolean (ifeval's prompt-level
    # accuracy), or a list of them (ifeval's instructions).
    if is_list:
        return isinstance(value, list) and all(isinstance(v, Real) for v in value)
    return isinstance(value, Real)


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


def _function_name(source: str) -> str | None:
    # lm-eval saves a function in a task config as its source code, or as its
    # repr when the source is not available.
    found = re.search(r"^(?:def |<function )(\w+)", source, re.M)
    return found and found.group(1)


def _aggregation(metric: str, config: dict) -> str | None:
    """How lm-eval aggregates *metric* of a task with *config*: ``mean``, a
    registered aggregation's name, or an aggregation function's name."""
    for entry in config.get("metric_list") or []:
        name = str(entry.get("metric"))
        # A metric given as a function (mbpp's utils.pass_at_1).
        if name == metric or _function_name(name) == metric:
            if "aggregation" not in entry:
                return _REGISTERED_AGGREGATIONS.get(metric)
            aggregation = str(entry["aggregation"])
            return _function_name(aggregation) or aggregation
    if metric in _DEFAULT_METRICS.get(config.get("output_type"), set()):
        return _REGISTERED_AGGREGATIONS.get(metric)
    return None


def _is_mean_metric(metric: str, config: dict) -> bool:
    return _aggregation(metric, config) == "mean"


def _metric_keys(result: dict) -> list[str]:
    """``metric,filter`` keys of a result, without stderrs."""
    return [k for k in result if "," in k and "_stderr," not in k]


def _in_order_of(result: dict, scores: dict) -> dict:
    # collect keeps a metric's first numeric filter, so keys keep the reported
    # order. A metric whose first filter was not recomputed is left out.
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
    # Results are <name>_<date>.json (results_<date>.json for a directory
    # --output_path), samples samples_<task>_<date>.jsonl; the date has no "_".
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
            f"{counts.get('original')} documents (a --limit or --samples run?). Excluding "
            "samples needs a run over the whole benchmark."
        )
    # A run stopped while saving its samples leaves them incomplete.
    with open(path, encoding="utf-8") as f:
        try:
            samples = [json.loads(line) for line in f if line.strip()]
        except json.JSONDecodeError as e:
            raise RunNotFilterable(f"{task}: {path.name} is cut short ({e}).") from e
    if len({s["doc_id"] for s in samples}) != counts.get("effective"):
        raise RunNotFilterable(
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
    rows: list[dict],
    samples_by_task: dict[str, list[dict]],
    benchmark: str = "",
    normalize: bool = False,
    match_on: bool = False,
) -> tuple[dict[str, set[int]], list[bool]]:
    """Doc ids per task whose document equals a blocklist row, and whether each
    row matched.

    Only fields the documents have are compared (an evaluated copy may drop
    some). With *match_on*, the documents must have all of the row's fields.
    With *normalize*, text is compared by letters and digits only
    (openeurollm/jeopardy drops apostrophes and spaces).
    """

    def comparable(values: dict) -> str:
        return _canonical(
            {
                f: _normalized(v) if normalize and isinstance(v, str) else v
                for f, v in values.items()
            }
        )

    shared = {
        task: set.intersection(*(set(s["doc"]) for s in samples)) if samples else set()
        for task, samples in samples_by_task.items()
    }
    if match_on:
        for task in samples_by_task:
            missing = sorted({f for row in rows for f in row} - shared[task])
            if missing:
                raise RunNotFilterable(
                    f"{benchmark}: the documents of {task} have no "
                    f"{', '.join(missing)}, which sample-exclusion.yaml matches its "
                    "rows on, so its blocklisted documents cannot be found."
                )
    index: dict[tuple, dict[str, list[int]]] = {}
    excluded: dict[str, set[int]] = {}
    matched = []
    for row in rows:
        hits = []
        for task, samples in samples_by_task.items():
            fields = tuple(sorted(f for f in row if f in shared[task]))
            if not fields:
                continue
            if (task, fields) not in index:
                by_key = index[task, fields] = {}
                seen = set()
                for s in samples:
                    if s["doc_id"] not in seen:
                        seen.add(s["doc_id"])
                        key = comparable({f: s["doc"][f] for f in fields})
                        by_key.setdefault(key, []).append(s["doc_id"])
            doc_ids = index[task, fields].get(comparable({f: row[f] for f in fields}), [])
            if len(doc_ids) > _MAX_MATCHES_PER_TASK:
                raise SampleExclusionError(
                    f"{benchmark}: a blocklist row matches {len(doc_ids)} documents of "
                    f"{task} on the fields they share ({', '.join(fields)}); name the "
                    "fields that identify a document with match_on in "
                    "sample-exclusion.yaml."
                )
            hits += [(task, doc_id) for doc_id in doc_ids]
        matched.append(bool(hits))
        for task, doc_id in hits:
            excluded.setdefault(task, set()).add(doc_id)
    return excluded, matched


def _task_scores(
    task: str, result: dict, config: dict, samples: list[dict], excluded: set[int]
) -> dict:
    """*result* recomputed without *excluded*, for metrics with a recomputable
    aggregation."""
    remaining = {s["doc_id"] for s in samples} - excluded
    if len(remaining) < 2:
        raise SampleExclusionError(
            f"{task}: {len(remaining)} documents remain after exclusion"
        )
    scores = {"sample_len": len(remaining)}
    for key in _metric_keys(result):
        metric, filter_name = key.split(",", 1)
        aggregation = _aggregation(metric, config)
        if not _is_number(result[key]) or aggregation not in _AGGREGATIONS:
            continue
        score, stderr, is_list = _AGGREGATIONS[aggregation]
        lines = [s for s in samples if s["filter"] == filter_name]
        full = [s.get(metric) for s in lines]
        if not full or not all(_is_value(v, is_list) for v in full):
            continue
        if not _close(score(full), result[key]):
            raise SampleExclusionError(
                f"{task}: the saved samples give {key} = {score(full)}, but lm-eval "
                f"reported {result[key]}; refusing to recompute it."
            )
        kept = [s[metric] for s in lines if s["doc_id"] not in excluded]
        scores[key] = score(kept)
        stderr_key = f"{metric}_stderr,{filter_name}"
        if stderr and _is_number(result.get(stderr_key)):
            if not _close(stderr(full), result[stderr_key]):
                raise SampleExclusionError(
                    f"{task}: the saved samples give {stderr_key} = "
                    f"{stderr(full)}, but lm-eval reported {result[stderr_key]}."
                )
            scores[stderr_key] = stderr(kept)
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

    lm-eval averages the leaves weighted by size or not, and the results do not
    say which: the weighting that reproduces the reported score and stderr is
    used. If both do, they must agree without the blocklisted samples.
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


def _is_lighteval(data: dict) -> bool:
    return "config_general" in data and "config_tasks" in data


def _lighteval_keys(data: dict) -> dict[str, str]:
    """Task name -> lighteval's results key (``<task>|<n_shot>``)."""
    keys = {}
    for key in data.get("results", {}):
        name, _, shots = key.rpartition("|")
        if name and shots.isdigit():
            if name in keys:
                raise SampleExclusionError(
                    f"{name} is in one lighteval results file twice ({keys[name]}, {key})"
                )
            keys[name] = key
    return keys


def _plain(value):
    # The details parquet holds numpy arrays where lighteval had lists.
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    return value


def _read_details(results_file: Path, data: dict, key: str) -> list[dict]:
    # lighteval writes <out>/results/<model>/results_<date>.json and, with
    # --save-details, <out>/details/<model>/<date>/details_<key>_<date>.parquet.
    general = data["config_general"]
    if general.get("max_samples"):
        raise RunNotFilterable(
            f"{key}: {results_file} scored at most {general['max_samples']} samples "
            "per task (a --limit run?). Excluding samples needs a run over the whole "
            "benchmark."
        )
    # lighteval strips the leading "/" of a local checkpoint's path.
    model = general["model_name"].strip("/")
    timestamp = results_file.stem.removeprefix("results_")
    depth = len(Path(model).parts) + 1
    if len(results_file.parents) <= depth or results_file.parent != (
        results_file.parents[depth] / "results" / model
    ):
        raise RunNotFilterable(
            f"{key}: {results_file} is not in lighteval's results/{model}/ "
            "directory, so its saved details cannot be found."
        )
    output_dir = results_file.parents[depth]
    path = (
        output_dir / "details" / model / timestamp / f"details_{key}_{timestamp}.parquet"
    )
    if not path.exists():
        raise RunNotFilterable(
            f"{key}: no saved details for {results_file} (expected {path}). "
            "Excluding samples needs a run scheduled with --log_samples."
        )
    import pandas as pd

    frame = pd.read_parquet(path, columns=["doc", "metric"])
    return [
        {"doc_id": i, "doc": _plain(doc), "metric": _plain(metric)}
        for i, (doc, metric) in enumerate(zip(frame["doc"], frame["metric"], strict=True))
    ]


def _normalized(text: str) -> str:
    # Prompt templates capitalise and re-punctuate the dataset's text (lighteval's
    # fix_ending_punct), so text is compared by its letters and digits only.
    return "".join(c for c in text.casefold() if c.isalnum())


def _contained_docs(
    rows: list[dict], samples_by_task: dict[str, list[dict]], benchmark: str = ""
) -> tuple[dict[str, set[int]], list[bool]]:
    """Doc ids per task whose lighteval prompt (query and choices) holds every
    text field of a blocklist row, and whether each row matched.

    If several documents hold it, only those with the least other text match:
    a sentence can be part of another (FLORES' 33rd is in its 32nd).
    """
    texts = {
        task: [
            (
                s["doc_id"],
                [
                    _normalized(part)
                    for part in [
                        s["doc"].get("query") or "",
                        *map(str, s["doc"].get("choices") or []),
                    ]
                ],
            )
            for s in samples
        ]
        for task, samples in samples_by_task.items()
    }
    excluded: dict[str, set[int]] = {}
    matched = []
    for row in rows:
        needles = [
            n for v in row.values() if isinstance(v, str) and (n := _normalized(v))
        ]
        hits = []
        for task, docs in texts.items():
            # The text around each needle in the shortest part holding it.
            around = {}
            for doc_id, parts in docs:
                lengths = [[len(p) - len(n) for p in parts if n in p] for n in needles]
                if needles and all(lengths):
                    around[doc_id] = sum(min(ls) for ls in lengths)
            if len(around) > _MAX_MATCHES_PER_TASK:
                raise SampleExclusionError(
                    f"{benchmark}: a blocklist row's text is in {len(around)} prompts "
                    f"of {task}; name the fields that identify a document with "
                    "match_on in sample-exclusion.yaml."
                )
            least = min(around.values(), default=None)
            hits += [(task, doc_id) for doc_id, a in around.items() if a == least]
        matched.append(bool(hits))
        for task, doc_id in hits:
            excluded.setdefault(task, set()).add(doc_id)
    return excluded, matched


def held_items(entry: dict) -> dict[int, int]:
    """Item -> position among the task's documents, of an alignment table's
    entry for a task."""
    if entry.get("positions") is None:
        first = entry.get("first", 0)
        return {first + p: p for p in range(entry["documents"])}
    return {i: p for i, p in enumerate(entry["positions"]) if p is not None}


def table_items(table: dict, rows: list[dict]) -> list[list[int]]:
    """The items of alignment *table* each row is: none, one, or each copy of
    a question the benchmark repeats (MMLU)."""
    items: dict[str, list[int]] = {}
    for i, key in enumerate(table["items"]):
        items.setdefault(key, []).append(i)
    return [items.get(item_key([row.get(f) for f in table["key"]]), []) for row in rows]


def _aligned_docs(
    rows: list[dict],
    samples_by_task: dict[str, list[dict]],
    table: dict,
    benchmark: str,
    lighteval: bool,
) -> tuple[dict[str, set[int]], list[bool]]:
    """Doc ids per task of the *table* items the blocklist rows are, and whether
    each row matched.

    Documents are found by position (lm-eval's doc_id, lighteval's id), so a run
    must have all the documents the table was built from, with the same answers.
    """
    doc_ids = {}
    for task, samples in samples_by_task.items():
        entry = table["tasks"][task]
        found = {}
        for s in samples:
            if lighteval and not str(s["doc"].get("id", "")).isdigit():
                raise RunNotFilterable(
                    f"{task}: its saved details give no document's position (id)."
                )
            position = int(s["doc"]["id"]) if lighteval else s["doc_id"]
            answer = (
                _canonical(_plain(s["doc"].get("gold_index")))
                if lighteval
                else str(s.get("target"))
            )
            found.setdefault(position, (s["doc_id"], answer))
        where = f"{_ALIGNMENTS}/{benchmark}.json"
        if sorted(found) != list(range(entry["documents"])):
            raise RunNotFilterable(
                f"{task}: {len(found)} documents, but {where} was built from "
                f"{entry['documents']}. Rebuild it with "
                "scripts/build_sample_exclusion_alignments.py."
            )
        if item_key([found[p][1] for p in sorted(found)]) != entry["answers"]:
            raise RunNotFilterable(
                f"{task}: its documents are not the ones {where} was built from "
                "(other answers, or in another order). Rebuild it with "
                "scripts/build_sample_exclusion_alignments.py."
            )
        doc_ids[task] = {i: found[p][0] for i, p in held_items(entry).items()}

    excluded: dict[str, set[int]] = {}
    matched = []
    for indices in table_items(table, rows):
        hits = [
            (task, ids[i]) for task, ids in doc_ids.items() for i in indices if i in ids
        ]
        matched.append(bool(hits))
        for task, doc_id in hits:
            excluded.setdefault(task, set()).add(doc_id)
    return excluded, matched


def _lighteval_aggregation(metric: str, config: dict) -> str | None:
    # config_tasks lists a task's metrics with their corpus_level_fn; a metric
    # with sample parameters is reported as <name>:<params> (gpqa_pass@k:k=1).
    for entry in config.get("metrics") or []:
        names = entry.get("metric_name")
        names = names if isinstance(names, list) else [names]
        if any(metric == n or metric.startswith(f"{n}:") for n in names):
            return str(entry.get("corpus_level_fn"))
    return None


def _corpus_bleu(lang: str, items: list[dict]) -> float:
    """lighteval's CorpusLevelTranslationMetric for BLEU: sacrebleu over the
    samples' saved golds and preds, called exactly as lighteval calls it."""
    try:
        import sacrebleu
    except ImportError as e:
        raise RunNotFilterable(
            "Recomputing lighteval's corpus BLEU needs sacrebleu, which lighteval "
            "scores it with: pip install sacrebleu"
        ) from e
    golds = [[item["golds"][0] for item in items]]
    preds = [
        item["preds"][0] if isinstance(item["preds"], list) else item["preds"]
        for item in items
    ]
    bleu = sacrebleu.BLEU(trg_lang=lang)
    return float(bleu.corpus_score(hypotheses=preds, references=golds).score)


_TRANSLATION_METRIC = re.compile(
    r"CorpusLevelTranslationMetric\(metric_type=(bleu|chrf\+\+|chrf|ter), lang=(\w*)\)"
)


def _lighteval_scores(
    task: str, result: dict, config: dict, samples: list[dict], excluded: set[int]
) -> dict:
    """*result* of a lighteval task recomputed without *excluded*, for means
    over the samples and corpus BLEU; chrF(++) and TER are None."""
    if len({s["doc_id"] for s in samples} - excluded) < 2:
        raise SampleExclusionError(f"{task}: fewer than 2 samples remain after exclusion")
    scores = {}
    for key, value in result.items():
        if key.endswith("_stderr") or not _is_number(value):
            continue
        aggregation = _lighteval_aggregation(key, config) or ""
        translation = _TRANSLATION_METRIC.fullmatch(aggregation)
        if translation:
            metric_type, lang = translation.groups()
            if metric_type != "bleu":
                # lighteval passes each sample's references as a reference set
                # of their own, so chrF(++) and TER score the first sample alone.
                # collect leaves None out and warns.
                scores[key] = None
                continue
            # Its stderr is bootstrapped, and left out.
            full = [s["metric"].get(key) for s in samples]
            if not all(
                isinstance(v, dict) and {"golds", "preds"} <= v.keys() for v in full
            ):
                continue
            score = _corpus_bleu(lang, full)
            if not _close(score, value):
                raise SampleExclusionError(
                    f"{task}: the saved details give {key} = {score}, but lighteval "
                    f"reported {value}; refusing to recompute it."
                )
            kept = [s["metric"][key] for s in samples if s["doc_id"] not in excluded]
            scores[key] = _corpus_bleu(lang, kept)
            continue
        if aggregation != "mean":
            continue
        full = [s["metric"].get(key) for s in samples]
        if not all(_is_value(v, is_list=False) for v in full):
            continue
        if not _close(_mean(full), value):
            raise SampleExclusionError(
                f"{task}: the saved details give {key} = {_mean(full)}, but lighteval "
                f"reported {value}; refusing to recompute it."
            )
        kept = [s["metric"][key] for s in samples if s["doc_id"] not in excluded]
        scores[key] = _mean(kept)
        stderr_key = f"{key}_stderr"
        if _is_number(result.get(stderr_key)):
            if not _close(_mean_stderr(full), result[stderr_key]):
                raise SampleExclusionError(
                    f"{task}: the saved details give {stderr_key} = {_mean_stderr(full)}, "
                    f"but lighteval reported {result[stderr_key]}."
                )
            scores[stderr_key] = _mean_stderr(kept)
    return {k: scores[k] for k in result if k in scores}


def _fewshot_shows(data: dict, task: str, excluded: set[int]) -> bool:
    """Whether *task*'s few-shot examples can be *excluded* documents: they come
    from the scored split (MMMLU's) and are its first n (n + 1 when the scored
    document is among them) or drawn at random."""
    config = data.get("configs", {}).get(task, {})
    shots = data.get("n-shot", {}).get(task)
    scored = config.get("test_split") or config.get("validation_split")
    if not excluded or not shots or config.get("fewshot_split") != scored:
        return False
    sampler = (config.get("fewshot_config") or {}).get("sampler", "default")
    return sampler != "first_n" or min(excluded) <= shots


def clean_results(
    results_file: Path,
    data: dict,
    blocklist: dict[str, list[dict]],
    task_map: dict[str, list[str]],
    alignments: dict[str, dict] | None = None,
    normalized: set[str] | frozenset[str] = frozenset(),
    match_on: set[str] | frozenset[str] = frozenset(),
) -> dict[str, tuple[dict, int]]:
    """Results of the tasks and groups in *data* that lose blocklisted samples.

    Returns ``{name: (result, n_excluded)}``; ``result`` has the reported
    layout and only the recomputed metrics. Tasks without blocklisted documents
    are left out.

    Documents of a task in *alignments* are found by position, others by
    content: by letters and digits for a benchmark in *normalized*, by its
    ``match_on`` fields for one in *match_on*.
    """
    alignments = alignments or {}
    lighteval = _is_lighteval(data)
    if lighteval:
        keys = _lighteval_keys(data)
        results = {name: data["results"][key] for name, key in keys.items()}
        groups, group_subtasks = {}, {}
    else:
        results = data.get("results", {})
        groups = data.get("groups", {})
        group_subtasks = data.get("group_subtasks", {})

    leaves_by_benchmark: dict[str, list[str]] = {}
    # A benchmark split over separate runs (polymath_en_low and _medium): rows
    # matching nothing here belong to the other runs.
    partial: set[str] = set()
    for benchmark, rows in blocklist.items():
        mapped = task_map.get(benchmark, [])
        targets = [t for t in mapped if t in results or t in groups]
        if len(targets) < len(mapped):
            partial.add(benchmark)
        if rows and targets:
            leaves_by_benchmark[benchmark] = [
                leaf
                for t in targets
                for leaf in _leaves(t, group_subtasks)
                if leaf in results
            ]
    if not leaves_by_benchmark:
        return {}

    leaves = sorted({leaf for ls in leaves_by_benchmark.values() for leaf in ls})
    if lighteval:
        samples = {leaf: _read_details(results_file, data, keys[leaf]) for leaf in leaves}
    else:
        samples = {
            leaf: _read_examples(results_file, leaf, results[leaf])
            if _is_evalchemy(results[leaf])
            else _read_samples(results_file, leaf, data)
            for leaf in leaves
        }
    excluded: dict[str, set[int]] = {leaf: set() for leaf in leaves}
    for benchmark, benchmark_leaves in leaves_by_benchmark.items():
        rows = blocklist[benchmark]
        table = alignments.get(benchmark, {"tasks": {}})
        aligned = {
            leaf: samples[leaf] for leaf in benchmark_leaves if leaf in table["tasks"]
        }
        others = {leaf: samples[leaf] for leaf in benchmark_leaves if leaf not in aligned}
        found = []
        if aligned:
            found.append(_aligned_docs(rows, aligned, table, benchmark, lighteval))
        if others:
            # lighteval saves the prompt it built, not the dataset row.
            found.append(
                _contained_docs(rows, others, benchmark)
                if lighteval
                else _excluded_docs(
                    rows,
                    others,
                    benchmark,
                    benchmark in normalized,
                    benchmark in match_on,
                )
            )
        matched = [False] * len(rows)
        for doc_ids_by_leaf, flags in found:
            for leaf, doc_ids in doc_ids_by_leaf.items():
                excluded[leaf] |= doc_ids
            matched = [a or b for a, b in zip(matched, flags, strict=True)]
        unmatched = matched.count(False)
        if unmatched:
            # Rows outside oellm's copy (Jeopardy is a subset) or another run's.
            tasks = ", ".join(
                t for t in task_map[benchmark] if t in results or t in groups
            )
            what = (
                "none of them is a document, so there are no samples to exclude"
                if unmatched == len(rows)
                else f"{unmatched} of them match no document and are not excluded"
            )
            log = logging.info if benchmark in partial else logging.warning
            log(
                f"{benchmark}: {len(rows)} blocklisted rows, {what} "
                f"({tasks} in {results_file})."
            )

    if not lighteval:
        for leaf in leaves:
            if _fewshot_shows(data, leaf, excluded[leaf]):
                logging.warning(
                    f"{leaf}: its few-shot examples come from the documents it "
                    "scores and include blocklisted ones, so its score without "
                    f"them still shows them as examples ({results_file})."
                )

    full = {leaf: results[leaf] for leaf in leaves}
    clean = {
        leaf: _lighteval_scores(
            leaf,
            results[leaf],
            data["config_tasks"].get(keys[leaf], {}),
            samples[leaf],
            excluded[leaf],
        )
        if lighteval
        else _evalchemy_scores(leaf, results[leaf], samples[leaf], excluded[leaf])
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
