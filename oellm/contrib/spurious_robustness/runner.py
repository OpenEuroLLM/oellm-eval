"""Inference helpers shared by the spurious-robustness suites.

UrbanCars ships as a separate suite because it is the only task that needs a
cluster-local data directory, and ``CLUSTER_ENV_VARS`` is validated for every
task in a suite: declaring ``URBANCARS_DATA_DIR`` alongside ImageNet and CelebA
would fail those two on any cluster without an UrbanCars tree. Splitting the
suites lets the login-node pre-flight check the variable only when UrbanCars is
actually scheduled, instead of surfacing the problem inside SLURM hours later.

Both suites share the scoring code here so the two cannot drift apart.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def resolve_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_path: str, device: str):
    import open_clip

    from oellm.contrib.spurious_robustness.adapter import OpenClipAdapter

    spec = OpenClipAdapter(model_path).to_open_clip_spec()
    logger.info("Loading OpenCLIP model %s on %s", spec, device)
    model, _, preprocess = open_clip.create_model_and_transforms(spec, device=device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(spec)
    return model, preprocess, tokenizer


def read_limit(env: dict[str, str]) -> int | None:
    raw = env.get("LIMIT", "").strip()
    return int(raw) if raw else None


def run_grouped(model, preprocess, tokenizer, device, batches, prompts, class_names):
    """Score a grouped benchmark (max-over-prompts, then worst-group)."""
    from oellm.contrib.spurious_robustness.zeroshot import (
        class_scores,
        encode_images,
        encode_texts,
        group_metrics,
        predict,
    )

    prompt_features = [
        encode_texts(model, tokenizer, prompts[c], device) for c in class_names
    ]

    predictions: list[int] = []
    labels: list[int] = []
    groups: list[str] = []
    for images, batch_labels, batch_groups in batches:
        features = encode_images(model, preprocess, images, device)
        predictions.extend(predict(class_scores(features, prompt_features)).tolist())
        labels.extend(batch_labels)
        groups.extend(batch_groups)

    metrics = group_metrics(predictions, labels, groups)
    per_group = metrics.pop("group_accuracies")
    metrics.pop("group_counts")
    for group, acc in sorted(per_group.items()):
        metrics[f"acc_{group}"] = acc
    return metrics


def warn_on_partial_coverage(task: str, metrics: dict, expected: int | None) -> None:
    """Worst-group is a minimum, so a missing group can only flatter the score."""
    observed = metrics.get("n_groups")
    if expected is not None and observed != expected:
        logger.warning(
            "%s covered %s of %d groups — worst-group accuracy is a minimum over "
            "the groups present, so this number is not comparable with a full run.",
            task,
            observed,
            expected,
        )


def write_results(
    output_path: Path, model_path: str, task: str, n_shot: int, metrics: dict
) -> None:
    result_json = {
        "model_name_or_path": model_path,
        "results": {task: metrics},
        "configs": {task: {"num_fewshot": n_shot}},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result_json, f, indent=2)
    logger.info("Results written to %s", output_path)


def parse_suite_results(
    data: dict, task_prefix: str
) -> tuple[str, str, int, dict[str, float]] | None:
    """Claim *data* if it carries this suite's task and metric shape."""
    results = data.get("results", {})
    for task_name, task_results in results.items():
        if not task_name.startswith(task_prefix) or not isinstance(task_results, dict):
            continue
        if "worst_group_accuracy" not in task_results:
            continue
        model_id = data.get("model_name_or_path") or data.get("model_name", "unknown")
        n_shot = data.get("configs", {}).get(task_name, {}).get("num_fewshot", 0)
        return model_id, task_name, int(n_shot), task_results
    return None
