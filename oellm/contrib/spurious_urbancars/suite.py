"""UrbanCars contrib suite — plugin protocol implementation.

Split out from ``spurious_robustness`` so that ``URBANCARS_DATA_DIR`` can be
declared in ``CLUSTER_ENV_VARS``. That list is validated for every task in a
suite, so keeping UrbanCars alongside ImageNet and CelebA would have failed
those two on any cluster without an UrbanCars tree. As its own suite the
variable is checked by the login-node pre-flight before submission, rather than
failing inside SLURM once the job is already running.

Scoring is shared with ``spurious_robustness`` so the two cannot drift apart.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SUITE_NAME = "spurious_urbancars"

# UrbanCars is composited locally (Stanford Cars + Places + LVIS via
# Whac-A-Mole) rather than published, so the tree must already exist on the
# cluster. Declaring it here gives the operator a failure at schedule time.
CLUSTER_ENV_VARS = ["URBANCARS_DATA_DIR"]

from oellm.contrib.spurious_urbancars.task import SpuriousUrbanCarsTask  # noqa: E402

_TASK = SpuriousUrbanCarsTask()

TASK_GROUPS: dict = {
    "task_metrics": {_TASK.engine_task_name: _TASK.primary_metric},
    "task_groups": {
        _TASK.task_group_name: {
            "suite": SUITE_NAME,
            "n_shots": _TASK.n_shots,
            "description": _TASK.description,
            "tasks": [{"task": _TASK.engine_task_name}],
        }
    },
}

_EXPECTED_GROUPS = 8

_MISSING_DIR_HINT = (
    "URBANCARS_DATA_DIR must be set for spurious_urbancars. UrbanCars is "
    "composited locally (Stanford Cars + Places + LVIS via Whac-A-Mole) rather "
    "than published, so it cannot be fetched from the Hub. Point it at the "
    "bg-0.5_co_occur_obj-0.5 test tree."
)


def detect_model_flags(model_path: str) -> str | None:
    from oellm.contrib.spurious_robustness.adapter import OpenClipAdapter

    return OpenClipAdapter(model_path).to_contrib_flags()


def run(
    *,
    model_path: str,
    task: str,
    n_shot: int,
    output_path: Path,
    model_flags: str | None,
    env: dict[str, str],
) -> None:
    """Evaluate UrbanCars and write a lmms-eval-compatible JSON to *output_path*."""
    from oellm.contrib.spurious_robustness.datasets import (
        URBANCARS_CLASS_NAMES,
        load_urbancars,
    )
    from oellm.contrib.spurious_robustness.prompts import URBANCARS_PROMPTS
    from oellm.contrib.spurious_robustness.runner import (
        load_model,
        read_limit,
        resolve_device,
        run_grouped,
        warn_on_partial_coverage,
        write_results,
    )

    if task != _TASK.engine_task_name:
        raise ValueError(f"Unknown task {task!r}. Expected {_TASK.engine_task_name!r}.")

    # Fail on configuration before paying for a checkpoint load. The scheduler's
    # pre-flight should already have caught this on the login node; this is the
    # compute-node backstop for a cluster whose value points somewhere stale.
    data_dir = env.get("URBANCARS_DATA_DIR", "")
    if not data_dir:
        raise RuntimeError(_MISSING_DIR_HINT)

    limit = read_limit(env)
    device = resolve_device()
    model, preprocess, tokenizer = load_model(model_path, device)

    metrics = run_grouped(
        model,
        preprocess,
        tokenizer,
        device,
        load_urbancars(data_dir, limit),
        URBANCARS_PROMPTS,
        URBANCARS_CLASS_NAMES,
    )
    warn_on_partial_coverage(task, metrics, _EXPECTED_GROUPS)
    write_results(output_path, model_path, task, n_shot, metrics)


def parse_results(data: dict) -> tuple[str, str, int, dict[str, float]] | None:
    """Claim *data* if it is this suite's output, else return None."""
    from oellm.contrib.spurious_robustness.runner import parse_suite_results

    return parse_suite_results(data, "spurious_urbancars")
