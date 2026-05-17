"""Result collection, metric resolution, and structured output for evaluation outputs."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path

import pandas as pd
import yaml

from oellm.constants import METRIC_FALLBACK_KEYS
from oellm.utils import _setup_logging

# Native scale (max value) of each lmms-eval / lm-eval metric.
# Used to normalize all reported metrics to 0–100 for cross-benchmark
# comparison in the Markdown report and JSON envelope. Whenever a new
# metric is added to ``task_metrics`` in ``task-groups.yaml``, also add
# its native scale here so the normalized column renders correctly.
METRIC_NATIVE_SCALE: dict[str, float] = {
    # ── 0–1 scale ──
    "exact_match": 1.0,
    "acc": 1.0,
    "acc_norm": 1.0,
    "accuracy": 1.0,
    "mmmu_acc": 1.0,
    "relaxed_overall": 1.0,
    "relaxed_human_split": 1.0,
    "relaxed_augmented_split": 1.0,
    "anls": 1.0,
    "ocrbench_accuracy": 1.0,
    "lvb_acc": 1.0,
    "score": 1.0,
    "wer": 1.0,
    "mer": 1.0,
    "f1": 1.0,
    "semantic_match": 1.0,
    # ── 0–100 scale (no scaling needed) ──
    "gpt_eval_score": 100.0,
    "llm_as_judge_eval": 100.0,
    "mvbench_accuracy": 100.0,
    "videomme_perception_score": 100.0,
    "gpt_eval_accuracy": 100.0,
    "submission": 100.0,
    "bleu": 100.0,
    # ── 0–5 Likert scale (GPT-judge style) ──
    "gpt_eval": 5.0,
    # ── Unbounded / non-standard ──
    # MME emits raw point sums: cognition is /800 (4 categories × 200),
    # perception is /2000 (10 categories × 200). See
    # lmms_eval/tasks/mme/utils.py::mme_aggregate_results.
    "mme_cognition_score": 800.0,
    "mme_perception_score": 2000.0,
}


def _normalize_to_100(value: float | None, metric_name: str | None) -> float | None:
    """Normalize a metric value to a 0–100 scale for cross-benchmark display.

    Returns ``None`` when the metric's native scale is unknown — caller
    should fall back to the raw value rather than guess.
    """
    if value is None or metric_name is None:
        return None
    clean = metric_name.split(",")[0]
    scale = METRIC_NATIVE_SCALE.get(clean)
    if scale is None:
        return None
    return value * (100.0 / scale)


def _resolve_metric(
    task_name: str, result_dict: dict, task_metrics: dict
) -> tuple[float | None, str | None]:
    """Return (value, metric_name) for task_name from result_dict."""

    # Normalise lmms-eval task-scoped metric keys so lm-eval and lmms-eval
    # output is handled identically.  lmms-eval writes keys like
    # "vqav2/vqa_score,none"; strip the "task_name/" prefix so the lookup
    # below sees "vqa_score,none" regardless of engine.  Keys without "/"
    # (lm-eval format) are passed through unchanged.
    result_dict = {
        (k.split("/", 1)[1] if "/" in k else k): v for k, v in result_dict.items()
    }

    def _first_numeric(d: dict, *candidates: str) -> tuple[float | None, str | None]:
        for c in candidates:
            if c in d and isinstance(d[c], (int, float)):
                return float(d[c]), c
        return None, None

    def _first_matching_prefix(d: dict, prefix: str) -> tuple[float | None, str | None]:
        for k, v in d.items():
            if (k == prefix or k.startswith(prefix + ",")) and isinstance(
                v, (int, float)
            ):
                return float(v), k
        return None, None

    preferred = task_metrics.get(task_name)
    if preferred is not None:
        val, key = _first_numeric(result_dict, f"{preferred},none", preferred)
        if val is not None:
            return val, key
        val, key = _first_matching_prefix(result_dict, preferred)
        return val, key

    for metric in METRIC_FALLBACK_KEYS:
        val, key = _first_numeric(result_dict, metric)
        if val is not None:
            return val, key
        val, key = _first_matching_prefix(result_dict, metric.split(",")[0])
        if val is not None:
            return val, key

    # Last resort: pick the first numeric non-stderr value (catches lmms-eval
    # benchmarks with non-standard metric names like mme_cognition_score)
    for k, v in result_dict.items():
        if (
            isinstance(v, (int, float))
            and "stderr" not in k
            and k not in ("alias", " ", "")
        ):
            return float(v), k
    return None, None


def _split_task_and_nshot(name: str) -> tuple[str, int | None]:
    """Split ``'task|N'`` task names used by some harnesses.

    Returns ``(task, N)`` when the suffix is numeric, ``(task, None)``
    otherwise.  Non-string inputs pass through unchanged.
    """
    if not isinstance(name, str):
        return name, None
    if "|" in name:
        base, after = name.rsplit("|", 1)
        if after.isdigit():
            return base, int(after)
    return name, None


def _infer_global_n_shot(n_shot_data: dict) -> int | None:
    """Infer a global n_shot if exactly one unique value exists."""
    try:
        candidate_values = []
        for _v in n_shot_data.values():
            if isinstance(_v, (int | float)):
                candidate_values.append(int(_v))
            elif isinstance(_v, str) and _v.isdigit():
                candidate_values.append(int(_v))
        unique_values = set(candidate_values)
        if len(unique_values) == 1:
            return next(iter(unique_values))
    except Exception:
        pass
    return None


def _resolve_n_shot(
    task_name: str,
    n_shot_data: dict,
    group_subtasks_map: dict,
    group_aggregate_names: set,
    global_n_shot: int | None,
) -> int | str:
    """Resolve n_shot for a task, with fallbacks for groups and MMLU."""
    n_shot = n_shot_data.get(task_name, "unknown")

    # If this is a group aggregate and n_shot is missing, derive from any subtask
    if task_name in group_aggregate_names and n_shot == "unknown":
        for subtask_name in group_subtasks_map.get(task_name, []):
            if subtask_name in n_shot_data:
                n_shot = n_shot_data[subtask_name]
                break
    if n_shot == "unknown" and global_n_shot is not None:
        n_shot = global_n_shot

    # Special handling for MMLU aggregate - get n_shot from any MMLU subtask
    if task_name == "mmlu" and n_shot == "unknown":
        for key, value in n_shot_data.items():
            if key.startswith("mmlu_"):
                n_shot = value
                break
        if n_shot == "unknown" and global_n_shot is not None:
            n_shot = global_n_shot

    # Special handling for Global MMLU aggregates - get n_shot from subtasks
    if task_name.startswith("global_mmlu_") and n_shot == "unknown":
        prefix = f"{task_name}_"
        for key, value in n_shot_data.items():
            if key.startswith(prefix):
                n_shot = value
                break
        if n_shot == "unknown" and global_n_shot is not None:
            n_shot = global_n_shot

    return n_shot


def _load_task_metrics() -> dict:
    """Load task_metrics from core YAML and all contrib suites."""
    task_groups_yaml = files("oellm.resources") / "task-groups.yaml"
    with open(str(task_groups_yaml)) as _f:
        _tg_cfg = yaml.safe_load(_f)
    task_metrics = _tg_cfg.get("task_metrics", {})

    from oellm.registry import (
        get_all_task_groups as _contrib_task_groups,
    )

    task_metrics.update(_contrib_task_groups().get("task_metrics", {}))
    return task_metrics


def collect_results(
    results_dir: str,
    output_csv: str = "eval_results.csv",
    *,
    check: bool = False,
    verbose: bool = False,
) -> None:
    """
    Collect evaluation results from JSON files and export to CSV.

    Args:
        results_dir: Path to the directory containing result JSON files
        output_csv: Output CSV filename (default: eval_results.csv)
        check: Check for missing evaluations and create a missing jobs CSV
        verbose: Enable verbose logging
    """
    _setup_logging(verbose)

    task_metrics = _load_task_metrics()

    results_path = Path(results_dir)
    if not results_path.exists():
        raise ValueError(f"Results directory does not exist: {results_dir}")

    # lm-eval writes flat JSON files: results/<hex>.json
    # lmms-eval writes nested dirs:  results/<hex>.json/<model>/<ts>_results.json
    # rglob("*.json") + is_file() finds both without breaking backward compat.
    search_root = (
        (results_path / "results")
        if (results_path / "results").is_dir()
        else results_path
    )
    json_files = [p for p in search_root.rglob("*.json") if p.is_file()]

    if not json_files:
        logging.warning(f"No JSON files found in {results_dir}")
        if not check:
            return

    logging.info(f"Found {len(json_files)} result files")

    # If check mode, also load the jobs.csv to compare
    if check:
        jobs_csv_path = results_path / "jobs.csv"
        if not jobs_csv_path.exists():
            logging.warning(f"No jobs.csv found in {results_dir}, cannot perform check")
            check = False
        else:
            jobs_df = pd.read_csv(jobs_csv_path)
            logging.info(f"Found {len(jobs_df)} scheduled jobs in jobs.csv")

    rows = []
    completed_jobs = set()

    for json_file in json_files:
        with open(json_file) as f:
            data = json.load(f)

        # Model name lives in different keys depending on the harness:
        # - lmms-eval: model_name_or_path is the checkpoint, model_name is the
        #   adapter class (e.g. "llava_hf")
        # - lighteval: config_general.{model_name,model,model_path}
        # - legacy: summary_general.model or top-level model
        model_name = (
            data.get("model_name_or_path")
            or data.get("model_name")
            or data.get("config_general", {}).get("model_name")
            or data.get("config_general", {}).get("model")
            or data.get("config_general", {}).get("model_path")
            or data.get("summary_general", {}).get("model")
            or data.get("model")
            or "unknown"
        )

        results = data.get("results", {})
        n_shot_data = data.get("n-shot", {})

        # lmms-eval has no "n-shot" dict; fall back to per-task config "num_fewshot"
        if not n_shot_data:
            for _task, _cfg in data.get("configs", {}).items():
                if isinstance(_cfg, dict):
                    shot = _cfg.get("num_fewshot")
                    if shot is not None:
                        n_shot_data[_task] = shot

        global_n_shot = _infer_global_n_shot(n_shot_data)

        # Aggregate groups (lm-eval harness)
        groups_map = data.get("groups", {})
        group_subtasks_map = data.get("group_subtasks", {})
        group_aggregate_names = set(groups_map.keys()) | set(group_subtasks_map.keys())
        group_subtask_names: set[str] = set()
        for _agg, _subs in group_subtasks_map.items():
            for _s in _subs:
                group_subtask_names.add(_s)

        # Prefer only the first aggregate metric from groups (simplified)
        if groups_map:
            group_name, group_results = next(iter(groups_map.items()))
            orig_group_name = group_name
            n_shot = n_shot_data.get(orig_group_name, "unknown")
            if n_shot == "unknown":
                for subtask_name in group_subtasks_map.get(orig_group_name, []):
                    if subtask_name in n_shot_data:
                        n_shot = n_shot_data[subtask_name]
                        break
            if n_shot == "unknown" and global_n_shot is not None:
                n_shot = global_n_shot
            # Strip ``'|N'`` n-shot suffix from the group name, falling back
            # to the parsed N when n_shot is still unknown.
            group_name, parsed_n = _split_task_and_nshot(orig_group_name)
            if n_shot == "unknown" and parsed_n is not None:
                n_shot = parsed_n
            performance, metric_name = _resolve_metric(
                group_name, group_results, task_metrics
            )
            if performance is not None:
                if check:
                    completed_jobs.add((model_name, group_name, n_shot))
                rows.append(
                    {
                        "model_name": model_name,
                        "task": group_name,
                        "n_shot": n_shot,
                        "performance": performance,
                        "performance_normalized": _normalize_to_100(
                            performance, metric_name
                        ),
                        "metric_name": metric_name if metric_name is not None else "",
                    }
                )
                # Skip per-task iteration when groups are present
                continue

        for task_name, task_results in results.items():
            # Skip entries already added from groups
            if groups_map and task_name in group_aggregate_names:
                continue
            # Skip any lm-eval group subtasks; keep only aggregates
            if task_name in group_subtask_names:
                continue

            # Skip MMLU subtasks - only keep the aggregate score
            if task_name.startswith("mmlu_") and task_name != "mmlu":
                continue

            # Skip Global MMLU subtasks - keep only aggregates like global_mmlu_full_pt
            if task_name.startswith("global_mmlu_") and task_name.count("_") >= 4:
                continue

            # Strip ``'|N'`` n-shot suffix from the task name; use parsed N
            # as a last-resort fallback when n_shot isn't otherwise resolvable.
            task_name_clean, parsed_n = _split_task_and_nshot(task_name)
            n_shot = _resolve_n_shot(
                task_name_clean,
                n_shot_data,
                group_subtasks_map,
                group_aggregate_names,
                global_n_shot,
            )
            if n_shot == "unknown" and parsed_n is not None:
                n_shot = parsed_n

            # Lmms-eval emits some groups (e.g. mvbench) as empty-placeholder
            # parents whose actual values live on the children in
            # `group_subtasks`. Aggregate (mean) to recover the headline.
            if set(task_results.keys()) <= {"alias", " ", ""}:
                subtasks = group_subtasks_map.get(task_name_clean, [])
                if not subtasks:
                    continue
                child_values: list[float] = []
                child_metric_name: str | None = None
                for subtask_name in subtasks:
                    sub_results = results.get(subtask_name, {})
                    sub_val, sub_metric = _resolve_metric(
                        task_name_clean, sub_results, task_metrics
                    )
                    if sub_val is not None:
                        child_values.append(sub_val)
                        if child_metric_name is None:
                            child_metric_name = sub_metric
                if not child_values:
                    continue
                performance = sum(child_values) / len(child_values)
                metric_name = child_metric_name
            else:
                performance, metric_name = _resolve_metric(
                    task_name_clean, task_results, task_metrics
                )

            if performance is not None:
                if check:
                    completed_jobs.add((model_name, task_name_clean, n_shot))

                rows.append(
                    {
                        "model_name": model_name,
                        "task": task_name_clean,
                        "n_shot": n_shot,
                        "performance": performance,
                        "performance_normalized": _normalize_to_100(
                            performance, metric_name
                        ),
                        "metric_name": metric_name if metric_name is not None else "",
                    }
                )
            else:
                # Log missing metrics — for lmms-eval tasks this often means
                # llm_as_judge_eval is null (no judge LLM configured) or the
                # metric key is not yet listed in task_metrics in task-groups.yaml
                logging.warning(
                    f"No numeric metric for '{task_name}' in {json_file.name} "
                    f"— value may be null (LLM judge not configured?) or metric key missing from task_metrics"
                )

    if not rows and not check:
        logging.warning("No results extracted from JSON files")
        return

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_csv, index=False)
        logging.info(f"Results saved to {output_csv}")

        # Write structured outputs alongside the CSV.
        output_stem = Path(output_csv).with_suffix("")
        json_path = Path(f"{output_stem}.json")
        md_path = Path(f"{output_stem}.md")
        write_results_json(rows, json_path)
        write_results_markdown(rows, md_path)
        logging.info(f"Results JSON: {json_path}")
        logging.info(f"Results Markdown: {md_path}")

        logging.info(f"Extracted {len(df)} evaluation results")

        if verbose:
            logging.info("Summary:")
            logging.info(f"Unique models: {df['model_name'].nunique()}")
            logging.info(f"Unique tasks: {df['task'].nunique()}")
            logging.info(
                f"N-shot values: {sorted(str(x) for x in df['n_shot'].unique())}"
            )

    if check:
        logging.info("=== Evaluation Status Check ===")

        missing_jobs = []

        for _, job in jobs_df.iterrows():
            job_tuple = (job["model_path"], job["task_path"], job["n_shot"])

            is_completed = False

            if job_tuple in completed_jobs:
                is_completed = True
            else:
                for completed_job in completed_jobs:
                    completed_model, completed_task, completed_n_shot = completed_job

                    if (
                        job["n_shot"] == completed_n_shot
                        and job["task_path"] == completed_task
                        and (
                            str(job["model_path"]).endswith(completed_model)
                            or completed_model in str(job["model_path"])
                        )
                    ):
                        is_completed = True
                        break

            if not is_completed:
                missing_jobs.append(job)

        completed_count = len(jobs_df) - len(missing_jobs)

        logging.info(f"Total scheduled jobs: {len(jobs_df)}")
        logging.info(f"Completed jobs: {completed_count}")
        logging.info(f"Missing jobs: {len(missing_jobs)}")

        if len(missing_jobs) > 0:
            missing_df = pd.DataFrame(missing_jobs)
            missing_csv = output_csv.replace(".csv", "_missing.csv")
            missing_df.to_csv(missing_csv, index=False)
            logging.info(f"Missing jobs saved to: {missing_csv}")
            logging.info(
                f"You can run these with: oellm schedule-eval --eval_csv_path {missing_csv}"
            )

            if verbose and len(missing_jobs) > 0:
                logging.info("Example missing jobs:")
                for _i, (_, job) in enumerate(missing_df.head(5).iterrows()):
                    logging.info(
                        f"  - {job['model_path']} | {job['task_path']} | n_shot={job['n_shot']}"
                    )
                if len(missing_jobs) > 5:
                    logging.info(f"  ... and {len(missing_jobs) - 5} more")


# ---------------------------------------------------------------------------
# Structured output: versioned JSON and Markdown report
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "1.1"


def write_results_json(
    rows: list[dict],
    output_path: str | Path,
) -> None:
    """Write versioned JSON: {version, generated_at, results: [...]}.

    Each result row has `performance` (raw engine value) and
    `performance_normalized` (0-100 via METRIC_NATIVE_SCALE, or null).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for row in rows:
        results.append(
            {
                "model": row.get("model_name", ""),
                "task": row.get("task", ""),
                "n_shot": row.get("n_shot", 0),
                "metric": row.get("metric_name", ""),
                "performance": row.get("performance", 0.0),
                "performance_normalized": row.get("performance_normalized"),
            }
        )

    envelope = {
        "version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "results": results,
    }

    output_path.write_text(json.dumps(envelope, indent=2))


def write_results_markdown(
    rows: list[dict],
    output_path: str | Path,
) -> None:
    """Write a Markdown results table on a 0-100 normalized scale."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "| Model | Task | N-shot | Metric | Performance (0–100) |",
        "|-------|------|--------|--------|---------------------|",
    ]
    has_raw_fallback = False
    has_lower_is_better = False
    for row in rows:
        model = row.get("model_name", "")
        task = row.get("task", "")
        n_shot = row.get("n_shot", 0)
        metric = row.get("metric_name", "")
        normalized = row.get("performance_normalized")
        if normalized is not None:
            perf_cell = f"{normalized:.2f}"
        else:
            raw = row.get("performance", 0.0)
            perf_cell = f"{raw:.4f}*"
            has_raw_fallback = True
        if any(k in metric.lower() for k in ("wer", "mer", "cer")):
            has_lower_is_better = True
        lines.append(f"| {model} | {task} | {n_shot} | {metric} | {perf_cell} |")

    # Only emit footnotes that apply to this report.
    footnotes = []
    if has_raw_fallback:
        footnotes.append("> `*` = raw value (metric scale not in `METRIC_NATIVE_SCALE`).")
    if has_lower_is_better:
        footnotes.append("> WER/MER/CER are lower-is-better.")
    if footnotes:
        lines.append("")
        lines.extend(footnotes)

    output_path.write_text("\n".join(lines) + "\n")
