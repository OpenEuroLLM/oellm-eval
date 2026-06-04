import json
import logging
import math
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from importlib.resources import files
from pathlib import Path
from string import Template

import pandas as pd
from jsonargparse import auto_cli

from oellm.task_groups import (
    _build_task_suite_map,
    _collect_dataset_specs,
    _expand_task_groups,
    _lookup_dataset_specs_for_tasks,
    split_group_tokens,
)
from oellm.utils import (
    _ensure_runtime_environment,
    _expand_local_model_paths,
    _filter_warnings,
    _load_cluster_env,
    _num_jobs_in_queue,
    _pre_download_datasets_from_specs,
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
    """Return the host-memory request for the generated SLURM job."""
    explicit_mem = os.environ.get("SLURM_MEM")
    if explicit_mem is not None and str(explicit_mem).strip() != "":
        return str(explicit_mem).strip()

    logging.warning("SLURM_MEM not set; falling back to default memory request '96G'.")
    return "96G"


def _resolve_additional_model_args(local: bool = False) -> str:
    """Return model args for lighteval, defaulting to an explicit batch size.
    - if `local` is True: `batch_size=1`
    - otherwise: `batch_size=32`

    Users may override the entire model args via `MODEL_ARGS` or the
    batch size via `BATCH_SIZE` environment variables.

    For now this is only passed to suite lighteval, not to evalchemy and lm-eval yet.
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


@dataclass
class EvaluationJob:
    model_path: Path | str
    task_path: str
    n_shot: int
    eval_suite: str


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
) -> None:
    """
    Schedule evaluation jobs for a given set of models, tasks, and number of shots.

    Args:
        models: A string of comma-separated model paths or Hugging Face model identifiers.
            Warning: does not allow passing model args such as `EleutherAI/pythia-160m,revision=step100000`
            since we split on commas. If you need to pass model args, use the `eval_csv_path` option.
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
            A group (or super_group) may be scoped to one or more languages with a bracket, e.g.
            `--task_groups "oellm-multilingual[deu_Latn]"` or
            `--task_groups "sib200-eu[fra_Latn|deu_Latn],flores200[deu_Latn]"`. Bracketed codes are
            separated by `,` or `|`; unknown codes raise, and a bracket that matches no task in its
            group raises.
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
        trust_remote_code: If True, trust remote code when downloading datasets. Default is True. Workflow might fail if set to False.
        venv_path: Path to a Python virtual environment. If provided, evaluations run directly using
            this venv instead of inside a Singularity/Apptainer container.
        lm_eval_include_path: Path to a directory containing custom lm_eval task YAML definitions.
            Passed as --include_path to lm_eval. Defaults to the bundled custom_lm_eval_tasks
            directory shipped with the package, which overrides broken upstream tasks
            (e.g. mgsm_native_cot_fr/de/es). Override to point at additional task YAMLs.
        local: If True, run evaluations directly on the local machine using bash instead of
            submitting to SLURM. Requires --venv_path. Skips cluster environment detection and
            runs all evaluations sequentially in a single process.
        slurm_template_var: JSON object of template variable overrides. Use exact env var names
            (PARTITION, ACCOUNT, GPUS_PER_NODE, SLURM_MEM). "TIME" overrides the time limit.
            Example: '{"PARTITION":"dev-g","ACCOUNT":"FOO","TIME":"02:00:00","GPUS_PER_NODE":2,"SLURM_MEM":"96G"}'
    """
    _setup_logging(verbose)

    if local:
        if not venv_path:
            raise ValueError(
                "--local requires --venv_path. Provide a path to a Python virtual "
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

    if isinstance(models, str) and models is not None:
        models = [m.strip() for m in models.split(",") if m.strip()]  # type: ignore

    if isinstance(tasks, str) and tasks is not None:
        tasks = [t.strip() for t in tasks.split(",") if t.strip()]  # type: ignore

    if isinstance(n_shot, int) and n_shot is not None:
        n_shot = [n_shot]

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

        # Always expand local model paths, even with skip_checks
        df["model_path"].unique()
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
        if task_groups is None:
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
            group_list = split_group_tokens(task_groups) if task_groups else []
            expanded = _expand_task_groups(group_list)
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

    if not skip_checks:
        hub_models: set[str | Path] = {
            job.model_path
            for job in expanded_eval_jobs
            if not Path(job.model_path).exists()
        }
        _process_model_paths(hub_models)
    else:
        logging.info(
            "Skipping model path processing and validation (--skip-checks enabled)"
        )

    df = pd.DataFrame(expanded_eval_jobs)

    if df.empty:
        logging.warning("No evaluation jobs to schedule.")
        return None

    df["eval_suite"] = df["eval_suite"].str.lower()

    # Ensure that all datasets required by the tasks are cached locally to avoid
    # network access on compute nodes.
    if not skip_checks:
        dataset_specs = []
        if task_groups:
            group_list = split_group_tokens(task_groups)
            dataset_specs = _collect_dataset_specs(group_list)
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

    evals_dir = (
        Path(os.environ["EVAL_OUTPUT_DIR"])
        / f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    )
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

    # Calculate dynamic array size and time limits
    total_evals = len(df)
    minutes_per_eval = 10  # Budget 10 minutes per eval
    total_minutes = total_evals * minutes_per_eval
    max_minutes_per_job = 18 * 60  # 18 hours
    min_array_size_for_time = max(1, int(math.ceil(total_minutes / max_minutes_per_job)))
    desired_array_size = min(128, total_evals) if total_evals >= 128 else total_evals
    if desired_array_size < min_array_size_for_time:
        desired_array_size = min_array_size_for_time
    actual_array_size = min(remaining_queue_capacity, desired_array_size, total_evals)
    evals_per_job = max(1, int(math.ceil(total_evals / actual_array_size)))
    minutes_per_job = evals_per_job * minutes_per_eval
    minutes_with_margin = int(minutes_per_job * 1.2)
    hours_with_margin = max(1, int(math.ceil(minutes_with_margin / 60)))
    hours_with_margin = max(hours_with_margin, 3)
    hours_with_margin = min(hours_with_margin, 23)
    computed_time = f"{hours_with_margin:02d}:59:00"
    time_limit = computed_time

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

    # Log the calculated values
    slurm_mem = _resolve_slurm_mem()
    logging.info("📊 Evaluation planning:")
    logging.info(f"   Total evaluations: {total_evals}")
    logging.info(f"   Estimated time per eval: {minutes_per_eval} minutes")
    logging.info(
        f"   Total estimated time: {total_minutes} minutes ({total_minutes / 60:.1f} hours)"
    )
    logging.info(f"   Desired array size: {desired_array_size}")
    logging.info(
        f"   Actual array size: {actual_array_size} (limited by queue capacity: {remaining_queue_capacity})"
    )
    logging.info(f"   Evaluations per job: {evals_per_job}")
    logging.info(
        f"   Time per job: {minutes_per_job} minutes ({minutes_per_job / 60:.1f} hours)"
    )
    logging.info(f"   Time limit with safety margin: {time_limit}")
    logging.info(f"   Requested host memory: {slurm_mem}")

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
        additional_model_args=_resolve_additional_model_args(local),  # Batch size
        evalchemy_dir=os.environ.get("EVALCHEMY_DIR", "/opt/evalchemy"),
    )

    # substitute any $ENV_VAR occurrences
    sbatch_script = Template(sbatch_script).safe_substitute(os.environ)

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

    logging.info(f"📁 Evaluation directory: {evals_dir}")
    logging.info(f"📄 Script: {sbatch_script_path}")
    logging.info(f"📋 Job configuration: {csv_path}")
    logging.info(f"📊 Results will be stored in: {evals_dir / 'results'}")

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
        return

    try:
        logging.info("Calling sbatch to launch the evaluations")
        logging.info(f"📜 SLURM logs will be stored in: {slurm_logs_dir}")

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
            logging.info(f"🔍 Monitor job status: squeue -j {job_id}")
            logging.info(f"📈 View job details: scontrol show job {job_id}")
            logging.info(f"❌ Cancel job if needed: scancel {job_id}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to submit job: {e}")
        logging.error(f"sbatch stderr: {e.stderr}")
    except FileNotFoundError:
        logging.error(
            "sbatch command not found. Please make sure you are on a system with SLURM installed."
        )


def collect_results(
    results_dir: str,
    output_csv: str = "eval_results.csv",
    *,
    check: bool = False,
    verbose: bool = False,
) -> None:
    """
    Collect evaluation results from JSON files and export to CSV.

    Recursively searches ``results_dir`` for every sub-directory that contains
    a ``jobs.csv`` file, merges all those CSV files into one (later directories
    override duplicate rows), then recursively finds every ``.json`` result
    file under ``results_dir``.

    With ``--check`` the merged jobs are compared against the extracted results
    and any missing evaluations are written to a ``*_missing.csv`` file.
    Without ``--check`` the results are simply merged and written to
    ``output_csv``.

    Args:
        results_dir: Root directory to search for jobs.csv files and JSON results
        output_csv: Output CSV filename (default: eval_results.csv)
        check: Check for missing evaluations and create a missing jobs CSV
        verbose: Enable verbose logging
    """
    import json

    import yaml

    _setup_logging(verbose)

    task_groups_yaml = files("oellm.resources") / "task-groups.yaml"
    with open(str(task_groups_yaml)) as _f:
        _tg_cfg = yaml.safe_load(_f)
    task_metrics = _tg_cfg.get("task_metrics", {})

    def _resolve_metric(
        task_name: str, result_dict: dict
    ) -> tuple[float | None, str | None]:
        """Return (value, metric_name) for task_name from result_dict."""

        # Skip non-metric keys; lm-eval uses suffixes like ",none" or ",remove_whitespace"
        def _first_numeric(d: dict, *candidates: str) -> tuple[float | None, str | None]:
            for c in candidates:
                if c in d and isinstance(d[c], int | float):
                    return float(d[c]), c
            return None, None

        def _first_matching_prefix(
            d: dict, prefix: str
        ) -> tuple[float | None, str | None]:
            for k, v in d.items():
                if (k == prefix or k.startswith(prefix + ",")) and isinstance(
                    v, int | float
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

        for metric in [
            "acc,none",
            "acc",
            "accuracy",
            "acc_norm",
            "f1",
            "exact_match",
            "chrf++",
            "bleu",
        ]:
            val, key = _first_numeric(result_dict, metric)
            if val is not None:
                return val, key
            val, key = _first_matching_prefix(result_dict, metric.split(",")[0])
            if val is not None:
                return val, key
        return None, None

    def _split_task_and_nshot(name: str) -> tuple[str, int | None]:
        """Split task names of the form 'task|N' returning (task, N) or (task, None)."""
        if not isinstance(name, str):
            return name, None
        if "|" in name:
            base, after = name.rsplit("|", 1)
            if after.isdigit():
                return base, int(after)
        return name, None

    results_path = Path(results_dir)
    if not results_path.exists():
        raise ValueError(f"Results directory does not exist: {results_dir}")

    # ------------------------------------------------------------------
    # 1. Find all directories containing jobs.csv and merge them.
    #    Directories are sorted so that deeper / later-alphabetical paths
    #    are processed last, meaning their rows override earlier duplicates.
    # ------------------------------------------------------------------
    jobs_csv_paths = sorted(results_path.rglob("jobs.csv"))
    if jobs_csv_paths:
        logging.info(
            f"Found {len(jobs_csv_paths)} jobs.csv file(s): {[str(p) for p in jobs_csv_paths]}"
        )
        jobs_frames = [pd.read_csv(p) for p in jobs_csv_paths]
        # Concatenate and let later entries win for duplicate (model_path, task_path, n_shot).
        jobs_df = pd.concat(jobs_frames, ignore_index=True)
        dup_cols = [
            c for c in ("model_path", "task_path", "n_shot") if c in jobs_df.columns
        ]
        if dup_cols:
            jobs_df = jobs_df.drop_duplicates(subset=dup_cols, keep="last")
        logging.info(f"Merged jobs.csv: {len(jobs_df)} unique jobs")
    else:
        jobs_df = None
        if check:
            logging.warning(
                f"No jobs.csv found under {results_dir}, cannot perform check"
            )
            check = False

    # ------------------------------------------------------------------
    # 2. Recursively find all JSON result files.
    # ------------------------------------------------------------------
    json_files = sorted(results_path.rglob("*.json"))

    if not json_files:
        logging.warning(f"No JSON files found under {results_dir}")
        if not check:
            return

    logging.info(f"Found {len(json_files)} result file(s)")

    # Collect results
    rows = []
    completed_jobs = set()  # Track (model, task, n_shot) tuples

    for json_file in json_files:
        with open(json_file) as f:
            data = json.load(f)

        # Extract model name/path from a few common locations used in different
        # versions of the result JSON schema.
        model_name = (
            data.get("model_name")
            or data.get("config_general", {}).get("model_name")
            or data.get("config_general", {}).get("model")
            or data.get("config_general", {}).get("model_path")
            or data.get("summary_general", {}).get("model")
            or data.get("model")
            or "unknown"
        )

        # Extract results for each task
        results = data.get("results", {})
        n_shot_data = data.get("n-shot", {})

        # Infer a global n_shot if exactly one unique value exists in this JSON
        global_n_shot = None
        try:
            candidate_values = []
            for _v in n_shot_data.values():
                if isinstance(_v, (int | float)):
                    candidate_values.append(int(_v))
                elif isinstance(_v, str) and _v.isdigit():
                    candidate_values.append(int(_v))
            unique_values = set(candidate_values)
            if len(unique_values) == 1:
                global_n_shot = next(iter(unique_values))
        except Exception:
            pass

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
            # Prefer original extraction from n_shot_data and subtasks, then
            # global_n_shot; only fall back to parsing the group name.
            orig_group_name = group_name
            n_shot = n_shot_data.get(orig_group_name, "unknown")
            if n_shot == "unknown":
                for subtask_name in group_subtasks_map.get(orig_group_name, []):
                    if subtask_name in n_shot_data:
                        n_shot = n_shot_data[subtask_name]
                        break
            if n_shot == "unknown" and global_n_shot is not None:
                n_shot = global_n_shot
            # Fallback: parse possible '|N' suffix from group name
            group_name, parsed_n = _split_task_and_nshot(orig_group_name)
            if n_shot == "unknown" and parsed_n is not None:
                n_shot = parsed_n
            performance, metric_name = _resolve_metric(group_name, group_results)
            if performance is not None:
                if check:
                    completed_jobs.add((model_name, group_name, n_shot))
                rows.append(
                    {
                        "model_name": model_name,
                        "task": group_name,
                        "n_shot": n_shot,
                        "performance": performance,
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

            # Skip the lighteval 'all' aggregate pseudo-task
            if task_name == "all":
                continue

            # Get n_shot for this task.
            # Prefer original extraction from `n_shot_data` and `global_n_shot`,
            # and fall back to parsing a '|N' suffix in the task name.
            # Use explicit None checks because n_shot=0 is falsy.
            task_name_clean, parsed_n = _split_task_and_nshot(task_name)
            _ns = n_shot_data.get(task_name_clean)
            if _ns is None:
                _ns = global_n_shot
            if _ns is None:
                _ns = parsed_n
            n_shot = "unknown" if _ns is None else _ns

            # If this is a group aggregate and n_shot is missing, derive from any subtask
            if task_name_clean in group_aggregate_names and n_shot == "unknown":
                for subtask_name in group_subtasks_map.get(task_name_clean, []):
                    if subtask_name in n_shot_data:
                        n_shot = n_shot_data[subtask_name]
                        break
            if n_shot == "unknown" and global_n_shot is not None:
                n_shot = global_n_shot

            # Special handling for MMLU aggregate - get n_shot from any MMLU subtask
            if task_name_clean == "mmlu" and n_shot == "unknown":
                for key, value in n_shot_data.items():
                    if key.startswith("mmlu_"):
                        n_shot = value
                        break
                if n_shot == "unknown" and global_n_shot is not None:
                    n_shot = global_n_shot

            # Special handling for Global MMLU aggregates - get n_shot from subtasks
            if task_name_clean.startswith("global_mmlu_") and n_shot == "unknown":
                prefix = f"{task_name_clean}_"
                for key, value in n_shot_data.items():
                    if key.startswith(prefix):
                        n_shot = value
                        break
                if n_shot == "unknown" and global_n_shot is not None:
                    n_shot = global_n_shot

            # Get the primary metric (usually acc, acc_norm)
            performance, metric_name = _resolve_metric(task_name, task_results)

            if performance is not None:
                # Track completed job for check mode
                if check:
                    completed_jobs.add((model_name, task_name_clean, n_shot))

                rows.append(
                    {
                        "model_name": model_name,
                        "task": task_name_clean,
                        "n_shot": n_shot,
                        "performance": performance,
                        "metric_name": metric_name if metric_name is not None else "",
                    }
                )
            else:
                # Debug: log cases where we have a task but no performance metric
                if verbose:
                    logging.debug(
                        f"No performance metric found for {model_name} | {task_name} | n_shot={n_shot} in {json_file.name}"
                    )

    if not rows and not check:
        logging.warning("No results extracted from JSON files")
        return

    # Create DataFrame and save to CSV (if we have results)
    if rows:
        df = pd.DataFrame(rows)
        df = df.drop_duplicates(
            subset=["model_name", "task", "n_shot", "metric_name"], keep="last"
        )
        df.to_csv(output_csv, index=False)
        logging.info(f"Results saved to {output_csv}")
        logging.info(f"Extracted {len(df)} evaluation results")

        # Print summary statistics
        if verbose:
            logging.info("Summary:")
            logging.info(f"Unique models: {df['model_name'].nunique()}")
            logging.info(f"Unique tasks: {df['task'].nunique()}")
            logging.info(
                f"N-shot values: {sorted(str(x) for x in df['n_shot'].unique())}"
            )

    # Perform check analysis if requested
    if check:
        logging.info("=== Evaluation Status Check ===")

        # Find missing jobs
        missing_jobs = []

        for _, job in jobs_df.iterrows():
            job_tuple = (job["model_path"], job["task_path"], job["n_shot"])

            # Check if this job corresponds to one of our completed results
            is_completed = False

            # Try exact matching first
            if job_tuple in completed_jobs:
                is_completed = True
            else:
                # Try fuzzy matching for model names
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
                f"You can run these with: oellm-eval schedule --eval_csv_path {missing_csv}"
            )

            # Show some examples if verbose
            if verbose and len(missing_jobs) > 0:
                logging.info("Example missing jobs:")
                for _i, (_, job) in enumerate(missing_df.head(5).iterrows()):
                    logging.info(
                        f"  - {job['model_path']} | {job['task_path']} | n_shot={job['n_shot']}"
                    )
                if len(missing_jobs) > 5:
                    logging.info(f"  ... and {len(missing_jobs) - 5} more")


def main():
    _filter_warnings()
    auto_cli(
        {
            "schedule": schedule_evals,
            "collect": collect_results,
        },
        as_positional=False,
        description="OELLM: Multi-cluster evaluation tool for language models",
    )
