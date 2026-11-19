import json
import logging
import subprocess
from pathlib import Path

from oellm.bbq import bbq_metrics


def read_result(path: Path) -> dict | None:
    """Read JSON results or normalize a successful native Inspect log."""
    if path.suffix == ".eval":
        return _read_inspect_result(path)
    with path.open() as file:
        return json.load(file)


def _dump_inspect_log(eval_file: Path, *, header_only: bool = True) -> dict:
    args = ["inspect", "log", "dump"]
    if header_only:
        args.append("--header-only")
    args.append(str(eval_file))
    try:
        result = subprocess.run(
            args,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            "Inspect .eval logs were found, but the 'inspect' command is not "
            "available. Activate the Inspect environment or add its bin directory "
            "to PATH before collecting results."
        ) from e
    except subprocess.CalledProcessError as e:
        detail = e.stderr.strip() if e.stderr else str(e)
        raise RuntimeError(f"Failed to read Inspect log {eval_file}: {detail}") from e

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Inspect returned invalid JSON while reading {eval_file}"
        ) from e


def _read_inspect_result(eval_file: Path) -> dict | None:
    """Normalize metrics; BBQ also requires saved sample answers and metadata."""
    data = _dump_inspect_log(eval_file)
    if data.get("status") != "success":
        logging.warning(f"Ignoring non-successful Inspect log: {eval_file}")
        return None

    eval_spec = data.get("eval", {})
    model_name = str(eval_spec.get("model", "unknown"))
    model_args = eval_spec.get("model_args", {}) or {}
    if model_name == "hf/local" and model_args.get("model_path"):
        model_name = str(model_args["model_path"])
    elif model_name.startswith("hf/"):
        model_name = model_name.removeprefix("hf/")

    metrics: dict[str, float | int | None] = {}
    for score in (data.get("results", {}) or {}).get("scores", []):
        for metric_name, metric in (score.get("metrics", {}) or {}).items():
            value = metric.get("value") if isinstance(metric, dict) else None
            if isinstance(value, int | float):
                metrics[metric_name] = float(value)

    is_bbq = (eval_spec.get("dataset") or {}).get("name") == "bbq"
    if is_bbq:
        full_log = _dump_inspect_log(eval_file, header_only=False)
        samples = full_log.get("samples") or []
        completed = (full_log.get("results") or {}).get("completed_samples")
        if full_log.get("status") != "success" or completed != len(samples):
            raise ValueError(f"BBQ log has incomplete saved samples: {eval_file}")
        metrics.update(bbq_metrics(samples))

    task_name = str(eval_spec.get("task", "unknown"))
    # File-based tasks are scheduled as file.py@task; preserve that identity.
    task_file = eval_spec.get("task_file")
    if task_file:
        task_name = f"{task_file}@{task_name}"
    normalized = {
        "model_name": model_name,
        "results": {task_name: metrics},
        "n-shot": {task_name: 0},
    }
    if is_bbq:
        normalized["report_all_metrics"] = True
    return normalized
