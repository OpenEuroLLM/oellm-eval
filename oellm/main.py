import logging
from pathlib import Path

import typer
from typer import rich_utils

from oellm.config import EvalConfig
from oellm.results import collect_results
from oellm.utils import _filter_warnings, _setup_logging

# Override Typer's default cyan colour scheme with colours that are readable
# on both light (white) and dark terminal backgrounds.
rich_utils.COLOR_OPTIONS_PANEL_TITLE = "bold blue"
rich_utils.COLOR_ARGUMENTS_PANEL_TITLE = "bold blue"
rich_utils.COLOR_COMMANDS_PANEL_TITLE = "bold blue"
rich_utils.STYLE_OPTION = "bold blue"
rich_utils.STYLE_SWITCH = "bold dark_green"
rich_utils.STYLE_NEGATIVE_OPTION = "bold magenta"
rich_utils.STYLE_NEGATIVE_SWITCH = "bold magenta"
rich_utils.STYLE_METAVAR = "dark_orange3"
rich_utils.STYLE_OPTION_DEFAULT = "dim"

app = typer.Typer(
    name="oellm",
    help="ELLIOT: Multi-cluster evaluation tool for language models",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)


def schedule_evals(
    models: str | None = None,
    tasks: str | None = None,
    task_groups: str | None = None,
    n_shot: list[int] | None = None,
    eval_csv_path: str | None = None,
    *,
    config: str | None = None,
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
) -> None:
    """Schedule evaluation jobs for a given set of models, tasks, and number of shots.

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
        n_shot: An integer or list of integers specifying the number of shots applied to `tasks`.
        eval_csv_path: A path to a CSV file containing evaluation data.
            Warning: exclusive argument. Cannot specify `models`, `tasks`, `task_groups`, or `n_shot` when `eval_csv_path` is provided.
        config: Path to a YAML config file. CLI flags override YAML values.
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
        allow_missing_judge: If True, allow scheduling tasks that need an LLM judge
            (e.g. activitynetqa, AudioBench chat-style tasks) when ``OPENAI_API_KEY``
            is not set. Those tasks will emit null performance values. Default False —
            strict pre-flight refuses to schedule without the key.
    """
    from oellm.scheduler import schedule_evals as _sched

    cli_cfg = EvalConfig.from_cli_kwargs(
        models=models,
        tasks=tasks,
        task_groups=task_groups,
        n_shot=n_shot,
        eval_csv_path=eval_csv_path,
        max_array_len=max_array_len,
        limit=limit,
        verbose=verbose,
        download_only=download_only,
        dry_run=dry_run,
        skip_checks=skip_checks,
        trust_remote_code=trust_remote_code,
        venv_path=venv_path,
        lm_eval_include_path=lm_eval_include_path,
        local=local,
        slurm_template_var=slurm_template_var,
    )

    if config:
        yaml_cfg = EvalConfig.from_yaml(config)
        cfg = yaml_cfg.merge(cli_cfg)
        logging.info(f"Loaded config from {config} (CLI flags override)")
    else:
        cfg = cli_cfg

    cfg.validate()
    _setup_logging(cfg.verbose)

    models_str: str | None = None
    if cfg._model_paths():
        models_str = ",".join(cfg._model_paths())

    _sched(
        models=models_str,
        tasks=",".join(cfg.tasks) if cfg.tasks else None,
        task_groups=",".join(cfg.task_groups) if cfg.task_groups else None,
        n_shot=cfg.n_shot,
        eval_csv_path=cfg.eval_csv_path,
        max_array_len=cfg.slurm.max_array_len,
        limit=cfg.limit,
        verbose=cfg.verbose,
        download_only=cfg.download_only,
        dry_run=cfg.dry_run,
        skip_checks=cfg.skip_checks,
        trust_remote_code=cfg.trust_remote_code,
        venv_path=cfg.venv_path,
        lm_eval_include_path=cfg.lm_eval_include_path,
        local=cfg.local,
        slurm_template_var=cfg.slurm_template_var_json,
        allow_missing_judge=allow_missing_judge,
    )


def list_tasks(*, group: str | None = None) -> None:
    """List available task groups and their tasks.

    Args:
        group: If provided, show tasks within this specific group.
    """
    from rich.table import Table

    from oellm.task_groups import TaskGroup, _parse_task_groups, get_all_task_group_names
    from oellm.utils import get_console

    console = get_console()
    all_names = get_all_task_group_names()

    if group:
        all_names = [group]

    parsed = _parse_task_groups(all_names)

    table = Table(title="Available Task Groups")
    table.add_column("Group", style="bold")
    table.add_column("Suite")
    table.add_column("Tasks", justify="right")
    table.add_column("N-shots")
    table.add_column("Description")

    for name in sorted(parsed.keys()):
        g = parsed[name]
        if isinstance(g, TaskGroup):
            n_shots_set = set()
            for t in g.tasks:
                for s in t.n_shots or []:
                    n_shots_set.add(s)
            n_shots_str = ", ".join(str(s) for s in sorted(n_shots_set))
            table.add_row(
                name,
                g.suite,
                str(len(g.tasks)),
                n_shots_str,
                g.description,
            )
        else:
            # SuperGroup
            total_tasks = sum(len(sg.tasks) for sg in g.task_groups)
            table.add_row(
                name,
                "mixed",
                str(total_tasks),
                "",
                g.description,
            )

    console.print(table)


def compare(
    result_a: str,
    result_b: str,
    *,
    verbose: bool = False,
) -> None:
    """Compare two evaluation result files or directories.

    Args:
        result_a: Path to first results JSON file or directory containing results.json
        result_b: Path to second results JSON file or directory containing results.json
        verbose: Enable verbose logging
    """
    import json

    from rich.table import Table

    from oellm.utils import get_console

    _setup_logging(verbose)

    def _load_results(path_str: str) -> list[dict]:
        p = Path(path_str)
        if p.is_dir():
            p = p / "results.json"
        if not p.exists():
            raise FileNotFoundError(f"Results file not found: {p}")
        data = json.loads(p.read_text())
        return data.get("results", [])

    results_a = _load_results(result_a)
    results_b = _load_results(result_b)

    # Index by (task, n_shot, metric)
    def _index(results: list[dict]) -> dict[tuple, float]:
        idx = {}
        for r in results:
            key = (r.get("task", ""), r.get("n_shot", 0), r.get("metric", ""))
            idx[key] = r.get("performance", 0.0)
        return idx

    idx_a = _index(results_a)
    idx_b = _index(results_b)
    all_keys = sorted(set(idx_a.keys()) | set(idx_b.keys()))

    console = get_console()
    table = Table(title="Comparison")
    table.add_column("Task", style="bold")
    table.add_column("N-shot", justify="right")
    table.add_column("Metric")
    table.add_column("A", justify="right")
    table.add_column("B", justify="right")
    table.add_column("\u0394", justify="right")  # Delta

    for task, n_shot, metric in all_keys:
        val_a = idx_a.get((task, n_shot, metric))
        val_b = idx_b.get((task, n_shot, metric))
        str_a = f"{val_a:.4f}" if val_a is not None else "\u2014"
        str_b = f"{val_b:.4f}" if val_b is not None else "\u2014"
        if val_a is not None and val_b is not None:
            delta = val_b - val_a
            str_delta = f"{delta:+.4f}"
        else:
            str_delta = "\u2014"
        table.add_row(task, str(n_shot), metric, str_a, str_b, str_delta)

    console.print(table)


def eval_command(
    config: str | None = None,
    *,
    models: str | None = None,
    tasks: str | None = None,
    task_groups: str | None = None,
    n_shot: list[int] | None = None,
    eval_csv_path: str | None = None,
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
) -> None:
    """Run evaluations from a YAML config file.

    All CLI flags override values in --config. Delegates to schedule-eval.

    Args:
        config: Path to a YAML config file.
        models: Comma-separated model paths or HF identifiers (overrides config).
        tasks: Comma-separated task names (overrides config).
        task_groups: Comma-separated task group names (overrides config).
        n_shot: Number(s) of shots applied to tasks (overrides config).
        eval_csv_path: Path to a CSV with evaluation jobs (overrides config).
        max_array_len: Maximum concurrent SLURM array jobs.
        limit: Limit samples per task.
        download_only: Only pre-download models and datasets, then exit.
        dry_run: Generate the SLURM script without submitting.
        skip_checks: Skip container/model/dataset validation.
        trust_remote_code: Trust remote code when downloading datasets.
        venv_path: Python venv path. When set, runs in venv instead of Singularity.
        lm_eval_include_path: Path to custom lm_eval task YAML definitions directory.
        local: Run evaluations locally instead of submitting to SLURM.
        slurm_template_var: JSON object of SLURM overrides.
        verbose: Enable verbose logging.
    """
    schedule_evals(
        models=models,
        tasks=tasks,
        task_groups=task_groups,
        n_shot=n_shot,
        eval_csv_path=eval_csv_path,
        config=config,
        max_array_len=max_array_len,
        limit=limit,
        verbose=verbose,
        download_only=download_only,
        dry_run=dry_run,
        skip_checks=skip_checks,
        trust_remote_code=trust_remote_code,
        venv_path=venv_path,
        lm_eval_include_path=lm_eval_include_path,
        local=local,
        slurm_template_var=slurm_template_var,
        allow_missing_judge=allow_missing_judge,
    )


# Register CLI commands
app.command("schedule-eval")(schedule_evals)
app.command("eval")(eval_command)
app.command("collect-results")(collect_results)
app.command("list-tasks")(list_tasks)
app.command("compare")(compare)


def main() -> None:
    _filter_warnings()
    app()
