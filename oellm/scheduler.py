import getpass
import json
import logging
import math
import os
import re
import socket
import subprocess
from datetime import datetime
from importlib.resources import files
from pathlib import Path
from string import Template

import pandas as pd

from oellm import __version__
from oellm.constants import EvaluationJob
from oellm.core import DefaultHFAdapter
from oellm.results import _collector_git_commit, _load_task_metrics
from oellm.runner import EvalRunner
from oellm.task_groups import (
    _build_task_suite_map,
    _collect_dataset_specs,
    _collect_hf_dataset_files,
    _collect_hf_model_repos,
    _expand_task_groups,
    _lookup_dataset_specs_for_tasks,
    _lookup_hf_dataset_files_for_tasks,
    _lookup_hf_model_repos_for_tasks,
    split_group_tokens,
)
from oellm.utils import (
    _ensure_runtime_environment,
    _expand_local_model_paths,
    _load_cluster_env,
    _num_jobs_in_queue,
    _pre_download_datasets_from_specs,
    _pre_download_hf_dataset_files,
    _pre_download_hf_model_repos,
    _process_model_paths,
    _setup_logging,
    capture_third_party_output_from_kwarg,
)


def _resolve_hf_hub_offline(local: bool) -> int:
    """Value embedded in the generated eval script as HF_HUB_OFFLINE.

    If ``HF_HUB_OFFLINE`` is set in the environment when ``oellm`` runs, that
    value wins. Otherwise defaults to online Hub access for ``--local``
    (typical laptop dev) and offline for SLURM jobs (air-gapped workers).
    """
    raw = os.environ.get("HF_HUB_OFFLINE")
    if raw is not None and str(raw).strip() != "":
        try:
            return int(str(raw).strip())
        except ValueError:
            logging.warning("Invalid HF_HUB_OFFLINE=%r; using default", raw)
    return 0 if local else 1


def _resolve_slurm_mem() -> str:
    """Return the host-memory request for the generated SLURM job.

    Reads ``SLURM_MEM`` from the environment; falls back to ``96G``.  Can also
    be overridden per-invocation via ``--slurm-template-var``.
    """
    explicit_mem = os.environ.get("SLURM_MEM")
    if explicit_mem is not None and str(explicit_mem).strip() != "":
        return str(explicit_mem).strip()

    logging.warning("SLURM_MEM not set; falling back to default memory request '96G'.")
    return "96G"


def _resolve_additional_model_args(local: bool = False) -> str:
    """Return model args for lighteval, defaulting to an explicit batch size.

    - if ``local`` is True: ``batch_size=1``
    - otherwise: ``batch_size=32``

    Override the entire string via ``MODEL_ARGS`` or just the batch size via
    ``BATCH_SIZE``.  Applied to the lighteval suite only.
    """
    explicit_model_args = os.environ.get("MODEL_ARGS")
    if explicit_model_args is not None and str(explicit_model_args).strip() != "":
        return str(explicit_model_args).strip()

    batch_size = os.environ.get("BATCH_SIZE")
    if batch_size is not None and str(batch_size).strip() != "":
        batch_size_value = str(batch_size).strip()
        try:
            if int(batch_size_value) < 1:
                raise ValueError
        except ValueError:
            fallback = "1" if local else "32"
            logging.warning(
                "Invalid BATCH_SIZE=%r; falling back to batch_size=%s",
                batch_size,
                fallback,
            )
            batch_size_value = fallback
    else:
        batch_size_value = "1" if local else "32"

    return f"batch_size={batch_size_value}"


def _probe_engine_versions(venv_path: str | None) -> dict[str, str]:
    """Best-effort versions of the score-relevant packages in the venv that
    will run the jobs, recorded into provenance.json. Venvs are per-user,
    per-cluster state that nothing else records — this is what makes a
    collected number traceable to the engine that produced it. Container
    mode returns {} (versions live in the image; its name is recorded
    separately)."""
    if not venv_path:
        return {}
    python_bin = Path(venv_path).expanduser() / "bin" / "python"
    if not python_bin.exists():
        return {}
    from oellm.envcheck import probe_import

    versions: dict[str, str] = {}
    for module in ("lm_eval", "lighteval", "lmms_eval", "transformers", "torch"):
        ok, ver = probe_import(python_bin, module)
        if ok:
            versions[module] = ver
    return versions


@capture_third_party_output_from_kwarg("verbose")
def schedule_evals(
    models: str | None = None,
    tasks: str | None = None,
    task_groups: str | None = None,
    n_shot: int | list[int] | None = None,
    eval_csv_path: str | None = None,
    *,
    max_array_len: int = 128,
    limit: int | None = None,
    verbose: bool = False,
    download_only: bool = False,
    dry_run: bool = False,
    skip_checks: bool = False,
    trust_remote_code: bool = True,
    venv_path: str | None = None,
    lm_eval_include_path: str | None = None,
    local: bool = False,
    slurm_template_var: str | None = None,
    allow_missing_judge: bool = False,
    nodelist: str | None = None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
) -> None:
    """
    Schedule evaluation jobs for a given set of models, tasks, and number of shots.

    Args:
        models: A string of comma-separated model paths or Hugging Face model identifiers.
            Warning: model args such as `EleutherAI/pythia-160m,revision=step100000` are
            not supported — commas separate models here, and the SLURM-side CSV reader
            also splits rows on commas, so `eval_csv_path` cannot carry them either.
            For local paths:
            - If a directory contains `.safetensors` files directly, it will be treated as a single model
            - If a directory contains subdirectories with models (e.g., converted_checkpoints/),
              all models in subdirectories will be automatically discovered
            - For each model directory, if it has an `hf/iter_XXXXX` structure, all checkpoints will be expanded
            - This allows passing a single directory containing multiple models to evaluate them all
        tasks: A string of comma-separated task names (lm_eval) or paths.
            Requires `n_shot` to be provided. Tasks here are assumed to be lm_eval unless otherwise handled via CSV.
        task_groups: A string of comma-separated task group names defined in `task-groups.yaml`.
            Each group expands into concrete (task, n_shots, suite) entries; `n_shot` is ignored for groups.
        n_shot: An integer or list of integers specifying the number of shots applied to `tasks`.
        eval_csv_path: A path to a CSV file containing evaluation data.
            Warning: exclusive argument. Cannot specify `models`, `tasks`, `task_groups`, or `n_shot` when `eval_csv_path` is provided.
        max_array_len: The maximum number of jobs to schedule to run concurrently.
            Warning: this is not the number of jobs in the array job. This is determined by the environment variable `QUEUE_LIMIT`.
        limit: If set, limit the number of samples per task (useful for quick testing).
            Passes --limit to lm_eval and --max_samples to lighteval.
        download_only: If True, only download the datasets and models and exit.
        dry_run: If True, generate the SLURM script but don't submit it to the scheduler.
        skip_checks: If True, skip container image, model validation, and dataset pre-download checks for faster execution.
        trust_remote_code: If True, trust remote code when downloading datasets AND
            at eval time for lm_eval / lighteval / evalchemy (threaded through the
            adapter-rendered model args; previously hardcoded True at eval time).
            lmms-eval adapters manage their own loading. Default True. Workflow
            might fail if set to False.
        venv_path: Path to a Python virtual environment. If provided, evaluations run directly using
            this venv instead of inside a Singularity/Apptainer container.
        lm_eval_include_path: Path to a directory containing custom lm_eval task YAML definitions.
            Passed as --include_path to lm_eval. Defaults to the bundled custom_lm_eval_tasks
            directory shipped with the package.
        local: If True, run evaluations directly on the local machine using bash instead of
            submitting to SLURM. Requires --venv-path.
        slurm_template_var: JSON object of template variable overrides. Use exact env var names
            (PARTITION, ACCOUNT, GPUS_PER_NODE, SLURM_MEM). "TIME" overrides the time limit.
            Example: '{"PARTITION":"dev-g","ACCOUNT":"FOO","TIME":"02:00:00","GPUS_PER_NODE":2,"SLURM_MEM":"96G"}'
    """
    _setup_logging(verbose)

    if load_in_4bit and load_in_8bit:
        raise ValueError("load_in_4bit and load_in_8bit are mutually exclusive.")

    if local:
        if not venv_path:
            raise ValueError(
                "--local requires --venv-path. Provide a path to a Python virtual "
                "environment with lm_eval/lighteval installed."
            )
        local_output = str(Path.cwd() / "oellm-output")
        os.environ.setdefault("EVAL_BASE_DIR", local_output)
        os.environ.setdefault("EVAL_OUTPUT_DIR", local_output)
        os.environ.setdefault("QUEUE_LIMIT", "1")
        os.environ.setdefault("GPUS_PER_NODE", "1")
        os.environ.setdefault("PARTITION", "local")
        os.environ.setdefault("ACCOUNT", "local")
        os.environ.setdefault("EVAL_CONTAINER_IMAGE", "")
        os.environ.setdefault("SINGULARITY_ARGS", "")
        os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    else:
        _load_cluster_env()

    use_venv = venv_path is not None

    if not skip_checks:
        _ensure_runtime_environment(
            use_venv=use_venv,
            container_image=os.environ.get("EVAL_CONTAINER_IMAGE"),
            venv_path=venv_path,
        )
    else:
        logging.info("Skipping runtime environment check (--skip-checks enabled)")

    if isinstance(models, str):
        models = [m.strip() for m in models.split(",") if m.strip()]  # type: ignore

    if isinstance(tasks, str):
        tasks = [t.strip() for t in tasks.split(",") if t.strip()]  # type: ignore

    if isinstance(n_shot, int):
        n_shot = [n_shot]

    group_names: list[str] | None = None
    if task_groups:
        group_names = split_group_tokens(task_groups)

    eval_jobs: list[EvaluationJob] = []
    if eval_csv_path:
        if models or tasks or task_groups or n_shot:
            raise ValueError(
                "Cannot specify `models`, `tasks`, `task_groups`, or `n_shot` when `eval_csv_path` is provided."
            )
        df = pd.read_csv(eval_csv_path)
        required_cols = {"model_path", "task_path", "n_shot"}
        if not required_cols.issubset(df.columns):
            raise ValueError(
                f"CSV file must contain the columns: {', '.join(required_cols)}"
            )

        if "eval_suite" not in df.columns:
            df["eval_suite"] = "lm_eval"
        else:
            df["eval_suite"] = df["eval_suite"].fillna("lm_eval")

        eval_jobs.extend(
            [
                EvaluationJob(
                    model_path=row["model_path"],
                    task_path=row["task_path"],
                    n_shot=row["n_shot"],
                    eval_suite=row["eval_suite"],
                )
                for _, row in df.iterrows()
            ]
        )

    elif models:
        if group_names is None:
            # Look up each bare task name in the registered groups so
            # ``--tasks belebele_eng_Latn_cf`` (lighteval) or ``--tasks
            # regiondial_refcocog_all`` (contrib) get routed correctly.
            # Tasks not in any group default to lm_eval.
            task_suite_map = _build_task_suite_map()
            eval_jobs.extend(
                [
                    EvaluationJob(
                        model_path=model,
                        task_path=task,
                        n_shot=shot,
                        eval_suite=task_suite_map.get(task, "lm_eval"),
                    )
                    for model in models
                    for task in tasks
                    for shot in n_shot
                ]
            )
        else:
            expanded = _expand_task_groups(group_names)
            eval_jobs.extend(
                [
                    EvaluationJob(
                        model_path=model,
                        task_path=result.task,
                        n_shot=result.n_shot,
                        eval_suite=result.suite,
                    )
                    for model in models
                    for result in expanded
                ]
            )

    # Refuse judge-required tasks without OPENAI_API_KEY (unless explicitly
    # opted out). Runs before any model download / SLURM work so the user
    # sees the failure immediately, not after a long pre-flight.
    from oellm.utils import check_judge_llm_pre_flight

    check_judge_llm_pre_flight(
        {job.task_path for job in eval_jobs},
        allow_missing=allow_missing_judge,
    )

    expanded_eval_jobs = []
    for job in eval_jobs:
        local_model_paths = _expand_local_model_paths(job.model_path)
        if not local_model_paths:
            expanded_eval_jobs.append(job)
        else:
            for path in local_model_paths:
                expanded_eval_jobs.append(
                    EvaluationJob(
                        model_path=path,
                        task_path=job.task_path,
                        n_shot=job.n_shot,
                        eval_suite=job.eval_suite,
                    )
                )

    # Resolve eval_suite for each job: auto-detect lmms-eval adapter classes
    # and contrib suite model flags so users never set them manually.
    runner = EvalRunner()
    runner.prepare_jobs(expanded_eval_jobs)

    # Warn when a scheduled task has no explicit task_metrics entry —
    # collect_results then relies on METRIC_FALLBACK_KEYS, whose choice is
    # insertion-order-dependent for multi-filter engine outputs.
    _tm = _load_task_metrics()
    _unmapped_tasks = sorted(
        {str(j.task_path) for j in expanded_eval_jobs if str(j.task_path) not in _tm}
    )
    if _unmapped_tasks:
        _shown = ", ".join(_unmapped_tasks[:10])
        if len(_unmapped_tasks) > 10:
            _shown += f" (+{len(_unmapped_tasks) - 10} more)"
        logging.warning(
            f"No task_metrics entry for: {_shown} — collect will use fallback "
            f"metric keys; add entries to task-groups.yaml for a stable metric policy."
        )

    # Quantized loading: applied via --model_args for the HF-style
    # engines only. Never silent for the rest — a table mixing 4-bit and
    # full-precision rows is a comparability trap, so unsupported suites are
    # announced here and the choice is recorded in provenance.json.
    quantization = "4bit" if load_in_4bit else "8bit" if load_in_8bit else ""
    quantization_model_args = f",load_in_{quantization}=True" if quantization else ""
    if quantization:
        _quant_ok = {
            "lm_eval",
            "lm-eval",
            "lm-eval-harness",
            "lmms_eval",
            "lmms-eval",
            "evalchemy",
        }
        _unsupported = sorted(
            {
                str(j.eval_suite).split(":", 1)[0].strip().lower()
                for j in expanded_eval_jobs
            }
            - _quant_ok
        )
        if _unsupported:
            logging.warning(
                f"load_in_{quantization} applies to lm_eval/lmms_eval/evalchemy "
                f"only; rows for suite(s) {', '.join(_unsupported)} run at FULL "
                f"precision (contrib suites may opt in via the "
                f"OELLM_QUANTIZATION env var exported to the job)."
            )

    # Engine --model_args now come from the adapter layer (single source of
    # truth — template.sbatch receives only rendered strings; previously the
    # strings were hardcoded in bash and BaseModelAdapter was dead code).
    # "$model_path" stays a literal bash placeholder substituted per CSV row
    # on the compute node. trust_remote_code now genuinely governs eval time
    # for lm_eval / lighteval / evalchemy — it used to be hardcoded True.
    _adapter = DefaultHFAdapter(
        trust_remote_code=trust_remote_code, extra_args=quantization_model_args
    )
    lm_eval_model_args = _adapter.to_lm_eval_args()
    lmms_eval_model_args = _adapter.to_lmms_eval_args()
    evalchemy_model_args = _adapter.to_evalchemy_args()
    lighteval_trc = "trust_remote_code=True," if trust_remote_code else ""

    if not skip_checks:
        # Verify the runtime can actually execute the scheduled suites before
        # any network work: missing engines otherwise fail row-by-row on the
        # compute node hours later, and a version-pinned group (dclm-core-22)
        # on the wrong engine produces silently wrong scores.
        from oellm.envcheck import check_scheduled_environment

        check_scheduled_environment(
            {job.eval_suite for job in expanded_eval_jobs},
            venv_path=venv_path,
            group_names=group_names,
        )

        hub_models: set[str | Path] = {
            job.model_path
            for job in expanded_eval_jobs
            if not Path(job.model_path).exists()
        }
        _revs = _process_model_paths(hub_models)
        # Defensive: tests/legacy callers may stub this with a non-dict.
        model_revisions = _revs if isinstance(_revs, dict) else {}
    else:
        model_revisions = {}
        logging.info(
            "Skipping model path processing and validation (--skip-checks enabled)"
        )

    df = pd.DataFrame(expanded_eval_jobs)

    if df.empty:
        logging.warning("No evaluation jobs to schedule.")
        return None

    # Lowercase the suite name only, preserve any ``:model_flags`` suffix
    # verbatim — contrib dispatch keys can be case-sensitive (e.g.
    # AudioBench's ``Qwen2-Audio-7B-Instruct`` is matched literally).
    def _lower_suite_only(s: str) -> str:
        if ":" in s:
            head, tail = s.split(":", 1)
            return f"{head.lower()}:{tail}"
        return s.lower()

    df["eval_suite"] = df["eval_suite"].map(_lower_suite_only)

    # The SLURM-side reader slices the CSV with bash `IFS=, read`, which cannot
    # parse quoted fields — a comma/quote/newline in any value would be split
    # into the wrong columns silently at eval time. Refuse early instead.
    for _col in ("model_path", "task_path", "eval_suite"):
        _bad = df[df[_col].astype(str).str.contains(r'[,"\n\r]', regex=True)]
        if not _bad.empty:
            raise ValueError(
                f"{_col} value {_bad.iloc[0][_col]!r} contains a comma, quote, or "
                f"newline, which the SLURM job's CSV reader cannot parse. Model "
                f"args like 'model,revision=...' are not supported."
            )

    # Ensure that all datasets required by the tasks are cached locally to avoid
    # network access on compute nodes.
    if not skip_checks:
        dataset_specs = []
        if group_names:
            dataset_specs = _collect_dataset_specs(group_names)
        else:
            # Look up individual tasks in task groups registry
            all_tasks = df["task_path"].unique().tolist()
            dataset_specs = _lookup_dataset_specs_for_tasks(all_tasks)
            if not dataset_specs:
                logging.info(
                    "No dataset specs found for tasks; skipping dataset pre-download"
                )

        if dataset_specs:
            _pre_download_datasets_from_specs(
                dataset_specs, trust_remote_code=trust_remote_code
            )

        # Auxiliary model repos / dataset files declared by tasks must be
        # staged regardless of how the tasks were scheduled — task groups,
        # bare --tasks, or a --check re-schedule CSV — because compute
        # nodes run air-gapped (HF_HUB_OFFLINE=1).
        if group_names:
            hf_model_repos = _collect_hf_model_repos(group_names)
            hf_dataset_files = _collect_hf_dataset_files(group_names)
        else:
            _all_task_names = df["task_path"].unique().tolist()
            hf_model_repos = _lookup_hf_model_repos_for_tasks(_all_task_names)
            hf_dataset_files = _lookup_hf_dataset_files_for_tasks(_all_task_names)
        if hf_model_repos:
            _pre_download_hf_model_repos(hf_model_repos)
        if hf_dataset_files:
            _pre_download_hf_dataset_files(hf_dataset_files)
    else:
        logging.info("Skipping dataset pre-download (--skip-checks enabled)")

    if download_only:
        return None

    remaining_queue_capacity = (
        1 if local else int(os.environ.get("QUEUE_LIMIT", 250)) - _num_jobs_in_queue()
    )

    if remaining_queue_capacity <= 0 and not dry_run:
        logging.warning("No remaining queue capacity. Not scheduling any jobs.")
        return None

    logging.debug(
        f"Remaining capacity in the queue: {remaining_queue_capacity}. Number of "
        f"evals to schedule: {len(df)}."
    )

    # Build a descriptive directory name: {models}_{task_groups}_{timestamp}
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_names = "+".join(m.split("/")[-1].lower() for m in (models or []))
    group_label = "+".join(g.lower() for g in (group_names or []))
    parts = [p for p in [model_names, group_label, timestamp] if p]
    evals_dir = Path(os.environ["EVAL_OUTPUT_DIR"]) / "_".join(parts)
    evals_dir.mkdir(parents=True, exist_ok=True)

    slurm_logs_dir = evals_dir / "slurm_logs"
    slurm_logs_dir.mkdir(parents=True, exist_ok=True)
    csv_path = evals_dir / "jobs.csv"

    # Shuffle the dataframe to distribute fast/slow evaluations evenly across array jobs
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    logging.info(
        "Shuffled evaluation jobs for even load distribution across array workers"
    )

    df.to_csv(csv_path, index=False)

    sbatch_template = (files("oellm.resources") / "template.sbatch").read_text()

    total_evals = len(df)
    # max(1, …): with --dry-run the zero-capacity early-return above is skipped,
    # and a full queue would otherwise make this 0 (ZeroDivisionError below).
    actual_array_size = max(1, min(remaining_queue_capacity, total_evals))
    evals_per_job = max(1, int(math.ceil(total_evals / actual_array_size)))

    time_limit = os.environ.get("TIME_LIMIT", "12:00:00")

    # On low-QUEUE_LIMIT clusters many rows run serially inside one
    # wall-clock budget; without a per-row bound a single hung engine consumes
    # the rest of the slice invisibly.
    if evals_per_job > 1 and not os.environ.get("ROW_TIMEOUT"):
        logging.warning(
            f"{evals_per_job} evaluations run serially per array task under one "
            f"TIME_LIMIT={time_limit} with no per-row timeout — one hung row "
            f"consumes the rest of its slice. Set ROW_TIMEOUT (GNU timeout "
            f"duration, e.g. '7200' or '2h') to bound each row."
        )

    # Apply slurm_template_var overrides (JSON object)
    if slurm_template_var:
        try:
            opts = json.loads(slurm_template_var)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"slurm_template_var must be a valid JSON object: {e}"
            ) from e
        if not isinstance(opts, dict):
            raise ValueError(
                "slurm_template_var must be a JSON object, e.g. "
                '{"PARTITION":"dev-g","ACCOUNT":"FOO","TIME":"02:00:00"}'
            )
        for key, value in opts.items():
            if key.upper() == "TIME":
                time_limit = str(value)
                logging.info(f"Using time limit override: {time_limit}")
            else:
                os.environ[key] = str(value)
                logging.info(f"Using slurm_template_var override: {key}={value}")

    if nodelist:
        os.environ["NODELIST"] = nodelist
        logging.info(f"Constraining job to nodelist: {nodelist}")

    slurm_mem = _resolve_slurm_mem()
    additional_model_args = _resolve_additional_model_args(local)

    logging.info("Evaluation planning:")
    logging.info(f"   Total evaluations: {total_evals}")
    logging.info(
        f"   Array size: {actual_array_size} (queue capacity: {remaining_queue_capacity})"
    )
    logging.info(f"   Evaluations per job: {evals_per_job}")
    logging.info(f"   Time limit: {time_limit}")
    logging.info(f"   Requested host memory: {slurm_mem}")

    # Run-provenance sidecar: schedule-time config, resolved model revisions,
    # and every knob that changes effective eval conditions. Picked up by
    # collect_results and embedded in the results JSON envelope (v1.2) so a
    # collected number can be traced back to its run.
    provenance = {
        "schema": 1,
        "created_at": timestamp,
        "hostname": socket.gethostname(),
        "submitted_by": getpass.getuser()
        if not os.environ.get("OELLM_SUBMITTED_BY")
        else os.environ["OELLM_SUBMITTED_BY"],
        "oellm_version": __version__,
        "scheduler_git_commit": _collector_git_commit(),
        "eval_suites": sorted({str(j.eval_suite) for j in expanded_eval_jobs}),
        "total_evals": total_evals,
        "array_size": actual_array_size,
        "time_limit": time_limit,
        "slurm_mem": slurm_mem,
        "lighteval_model_args": additional_model_args,
        "max_num_frames": os.environ.get("MAX_NUM_FRAMES"),
        "limit": limit,
        "venv_path": venv_path,
        "hf_hub_offline": _resolve_hf_hub_offline(local),
        "quantization": quantization or None,
        "row_timeout": os.environ.get("ROW_TIMEOUT"),
        "trust_remote_code": trust_remote_code,
        "engine_model_args": {
            "lm_eval": lm_eval_model_args,
            "lmms_eval": lmms_eval_model_args,
            "evalchemy": evalchemy_model_args,
        },
        "model_revisions": model_revisions,
        "engine_versions": {} if skip_checks else _probe_engine_versions(venv_path),
        "container_image": None if venv_path else os.environ.get("EVAL_CONTAINER_IMAGE"),
    }
    (evals_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
    logging.info(f"Run provenance: {evals_dir / 'provenance.json'}")

    sbatch_script = sbatch_template.format(
        csv_path=csv_path,
        max_array_len=max_array_len,
        array_limit=actual_array_size - 1,  # Array is 0-indexed
        num_jobs=actual_array_size,  # This is the number of array jobs, not total evals
        total_evals=len(df),  # Pass the total number of evaluations
        log_dir=evals_dir / "slurm_logs",
        evals_dir=str(evals_dir / "results"),
        time_limit=time_limit,  # Dynamic time limit
        slurm_mem=slurm_mem,
        limit=limit if limit else "",  # Sample limit for quick testing
        venv_path=venv_path or "",
        lm_eval_include_path=lm_eval_include_path
        or str(files("oellm.resources") / "custom_lm_eval_tasks"),
        hf_hub_offline=_resolve_hf_hub_offline(local),
        additional_model_args=additional_model_args,
        evalchemy_dir=os.environ.get("EVALCHEMY_DIR", "/opt/evalchemy"),
        quantization=quantization,
        lm_eval_model_args=lm_eval_model_args,
        lmms_eval_model_args=lmms_eval_model_args,
        evalchemy_model_args=evalchemy_model_args,
        lighteval_trc=lighteval_trc,
        lm_eval_trc="1" if trust_remote_code else "",
    )

    # Drop optional #SBATCH directives whose env var is unset, so safe_substitute
    # doesn't leave a literal $ACCOUNT/$NODES/$NODELIST that SLURM rejects (e.g.
    # clusters like ufal define no ACCOUNT; --nodes/--nodelist are only present when
    # the cluster (clusters.yaml) or --nodelist / a slurm_template_var override sets
    # them).
    if not os.environ.get("ACCOUNT"):
        sbatch_script = sbatch_script.replace("#SBATCH --account=$ACCOUNT\n", "")
    if not os.environ.get("NODES"):
        sbatch_script = sbatch_script.replace("#SBATCH --nodes=$NODES\n", "")
    if not os.environ.get("NODELIST"):
        sbatch_script = sbatch_script.replace("#SBATCH --nodelist=$NODELIST\n", "")

    # Substitute $ENV_VAR occurrences from the environment — EXCLUDING SLURM_*
    # runtime variables: when scheduling from inside an allocation
    # (salloc/srun) those are set at render time and would be baked into the
    # script (e.g. JOB_HOME losing its per-job uniqueness) instead of
    # expanding on the compute node.
    _template_env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    sbatch_script = Template(sbatch_script).safe_substitute(_template_env)

    sbatch_script_path = evals_dir / "submit_evals.sbatch"

    with open(sbatch_script_path, "w") as f:
        f.write(sbatch_script)

    if dry_run:
        logging.info(f"Dry run mode: script generated at {sbatch_script_path}")
        logging.info(
            f"Would run {actual_array_size} array job(s) covering {len(df)} evaluations"
        )
        logging.info(
            f"Each job handles ~{(len(df) + actual_array_size - 1) // actual_array_size} evaluations"
        )
        if local:
            logging.info(
                f"To run locally: SLURM_ARRAY_TASK_ID=0 SLURM_ARRAY_JOB_ID=0 "
                f"SLURM_JOB_ID=0 bash {sbatch_script_path}"
            )
        else:
            logging.info("To submit the job, run: sbatch " + str(sbatch_script_path))
        return

    logging.info(f"Evaluation directory: {evals_dir}")
    logging.info(f"Script: {sbatch_script_path}")
    logging.info(f"Job configuration: {csv_path}")
    logging.info(f"Results will be stored in: {evals_dir / 'results'}")

    if local:
        logging.info("Running evaluations locally with bash...")
        local_env = {
            **os.environ,
            "SLURM_ARRAY_TASK_ID": "0",
            "SLURM_ARRAY_JOB_ID": "0",
            "SLURM_JOB_ID": "0",
        }
        try:
            subprocess.run(["bash", str(sbatch_script_path)], env=local_env, check=True)
            logging.info("Local evaluation completed.")
        except subprocess.CalledProcessError as e:
            logging.error(f"Evaluation failed with exit code {e.returncode}")
            raise SystemExit(e.returncode or 1) from e
        return

    try:
        logging.info("Calling sbatch to launch the evaluations")
        logging.info(f"SLURM logs: {slurm_logs_dir}")

        result = subprocess.run(
            ["sbatch"],
            input=sbatch_script,
            text=True,
            check=True,
            capture_output=True,
            env=os.environ,
        )
        logging.info("Job submitted successfully.")
        logging.info(result.stdout)
        job_id_match = re.search(r"Submitted batch job (\d+)", result.stdout)
        if job_id_match:
            job_id = job_id_match.group(1)
            logging.info(f"Monitor job status: squeue -j {job_id}")
            logging.info(f"View job details: scontrol show job {job_id}")
            logging.info(f"Cancel job if needed: scancel {job_id}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to submit job: {e}")
        logging.error(f"sbatch stderr: {e.stderr}")
        raise SystemExit(1) from e
    except FileNotFoundError as e:
        logging.error(
            "sbatch command not found. Please make sure you are on a system with SLURM installed."
        )
        raise SystemExit(1) from e
