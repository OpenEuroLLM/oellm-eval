"""Spurious-robustness contrib suite — plugin protocol implementation.

Replicates the evaluation of Sarridis et al., "Scaling Vision-Language Models
Fails to Mitigate Bias" (ACM MM 2026), whose reference implementation is at
https://github.com/gsarridis/vlm-spurious-robustness (MIT).

Why this is a contrib suite rather than an lm-eval task: classification here is
image/text *embedding similarity* on an OpenCLIP dual encoder — the image is
assigned to the class whose text embedding is nearest. Neither lm-eval nor
lmms-eval has a similarity-scoring path, and lm-eval's multimodal backends raise
on loglikelihood requests that carry an image, so likelihood-ranked multiple
choice is not an alternative. The benchmark brings its own inference and its own
metric, which is exactly what this plugin interface is for.

Consequence worth stating plainly: this suite evaluates CLIP-family dual
encoders, not the generative VLMs the rest of the platform runs. Scores are
comparable with the paper, and are not comparable with a generatively prompted
model's answers on the same images.

This suite covers the two benchmarks whose data comes from the Hub. UrbanCars
lives in the sibling ``spurious_urbancars`` suite because it needs a
cluster-local directory: ``CLUSTER_ENV_VARS`` is validated for every task in a
suite, so declaring its path here would fail ImageNet and CelebA on any cluster
without an UrbanCars tree. Schedule both with
``--task-groups spurious-robustness,spurious-urbancars``.

Cluster setup
-------------
``IMAGENET_VAL_DIR``
    Optional. An ILSVRC-2012 validation ImageFolder tree to read instead of the
    gated Hub copy. Not required, so it is deliberately absent from
    ``CLUSTER_ENV_VARS`` — the Hub copy is the default path.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SUITE_NAME = "spurious_robustness"

# Nothing here is mandatory: both datasets are pre-staged from the Hub.
CLUSTER_ENV_VARS: list[str] = []

from oellm.contrib.spurious_robustness.task import (  # noqa: E402
    SpuriousCelebATask,
    SpuriousImageNetTask,
)

_TASKS = (SpuriousImageNetTask(), SpuriousCelebATask())

_group_kwargs = {"suite": SUITE_NAME, "n_shots": [0]}


def _task_entry(task) -> dict:
    entry: dict = {"task": task.engine_task_name}
    if task.dataset_specs:
        entry["dataset"] = task.dataset_specs[0].repo_id
    return entry


TASK_GROUPS: dict = {
    "task_metrics": {t.engine_task_name: t.primary_metric for t in _TASKS},
    "task_groups": {
        "spurious-robustness": {
            **_group_kwargs,
            "description": (
                "Zero-shot robustness to spurious correlations for OpenCLIP "
                "models: ImageNet and CelebA. Add spurious-urbancars for the "
                "two-attribute benchmark."
            ),
            "tasks": [_task_entry(t) for t in _TASKS],
        },
        **{
            t.task_group_name: {
                **_group_kwargs,
                "description": t.description,
                "tasks": [_task_entry(t)],
            }
            for t in _TASKS
        },
    },
}

_EXPECTED_GROUPS = {"spurious_celeba": 4}


def detect_model_flags(model_path: str) -> str | None:
    """Delegate to OpenClipAdapter.to_contrib_flags()."""
    from oellm.contrib.spurious_robustness.adapter import OpenClipAdapter

    return OpenClipAdapter(model_path).to_contrib_flags()


def _run_imagenet(model, preprocess, tokenizer, device, limit, env) -> dict:
    """1000-way zero-shot top-1/top-5, the Radford et al. (2021) protocol.

    The classifier averages all 80 OpenAI templates per class — unlike CelebA and
    UrbanCars, which take the maximum over their prompts.
    """
    import torch
    from open_clip import (
        IMAGENET_CLASSNAMES,
        OPENAI_IMAGENET_TEMPLATES,
        build_zero_shot_classifier,
    )

    from oellm.contrib.spurious_robustness.datasets import load_imagenet
    from oellm.contrib.spurious_robustness.zeroshot import encode_images

    classifier = build_zero_shot_classifier(
        model,
        tokenizer=tokenizer,
        classnames=IMAGENET_CLASSNAMES,
        templates=OPENAI_IMAGENET_TEMPLATES,
        num_classes_per_batch=10,
        device=device,
        use_tqdm=False,
    )

    top1 = top5 = total = 0
    for images, labels, _ in load_imagenet(env.get("IMAGENET_VAL_DIR") or None, limit):
        features = encode_images(model, preprocess, images, device)
        logits = 100.0 * features.to(device) @ classifier
        targets = torch.tensor(labels, device=device)
        _, ranked = logits.topk(5, dim=-1)
        hits = ranked == targets.view(-1, 1)
        top1 += hits[:, :1].sum().item()
        top5 += hits.sum().item()
        total += len(labels)

    if not total:
        raise RuntimeError("ImageNet evaluation produced no samples")

    return {
        "top1_accuracy": top1 / total,
        "top5_accuracy": top5 / total,
        # One group by construction: no spurious attribute to split on.
        "worst_group_accuracy": top1 / total,
        "n_groups": 1,
        "total_images": total,
    }


def run(
    *,
    model_path: str,
    task: str,
    n_shot: int,
    output_path: Path,
    model_flags: str | None,
    env: dict[str, str],
) -> None:
    """Evaluate *task* and write a lmms-eval-compatible JSON to *output_path*."""
    from oellm.contrib.spurious_robustness.datasets import (
        CELEBA_CLASS_NAMES,
        load_celeba,
    )
    from oellm.contrib.spurious_robustness.prompts import CELEBA_PROMPTS
    from oellm.contrib.spurious_robustness.runner import (
        load_model,
        read_limit,
        resolve_device,
        run_grouped,
        warn_on_partial_coverage,
        write_results,
    )

    known = [t.engine_task_name for t in _TASKS]
    if task not in known:
        raise ValueError(f"Unknown task {task!r}. Expected one of: {known}")

    limit = read_limit(env)
    device = resolve_device()
    model, preprocess, tokenizer = load_model(model_path, device)

    if task == "spurious_imagenet":
        metrics = _run_imagenet(model, preprocess, tokenizer, device, limit, env)
    else:
        metrics = run_grouped(
            model,
            preprocess,
            tokenizer,
            device,
            load_celeba(limit),
            CELEBA_PROMPTS,
            CELEBA_CLASS_NAMES,
        )

    warn_on_partial_coverage(task, metrics, _EXPECTED_GROUPS.get(task))
    write_results(output_path, model_path, task, n_shot, metrics)


def parse_results(data: dict) -> tuple[str, str, int, dict[str, float]] | None:
    """Claim *data* if it is this suite's output, else return None.

    UrbanCars output belongs to the sibling suite, so it is explicitly not
    claimed here — a suite that recognises a file owns its format.
    """
    from oellm.contrib.spurious_robustness.runner import parse_suite_results

    for prefix in ("spurious_imagenet", "spurious_celeba"):
        claimed = parse_suite_results(data, prefix)
        if claimed is not None:
            return claimed
    return None
