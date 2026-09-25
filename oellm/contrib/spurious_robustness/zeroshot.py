"""Zero-shot inference primitives."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

# Kept out of module scope: the base install has no torch.
if TYPE_CHECKING:
    import torch


def encode_texts(model, tokenizer, texts: Sequence[str], device: str) -> torch.Tensor:
    """L2-normalised text embeddings, ``(len(texts), D)`` on CPU."""
    import torch
    import torch.nn.functional as F

    tokens = tokenizer(list(texts)).to(device)
    with torch.no_grad(), torch.amp.autocast(device):
        features = model.encode_text(tokens)
    return F.normalize(features.float().cpu(), dim=-1)


def encode_images(
    model,
    preprocess,
    images: Iterable[Any],
    device: str,
    batch_size: int = 32,
) -> torch.Tensor:
    """L2-normalised image embeddings, ``(N, D)`` on CPU. Takes PIL images or paths."""
    import torch
    import torch.nn.functional as F
    from PIL import Image

    batch: list[torch.Tensor] = []
    chunks: list[torch.Tensor] = []

    def flush() -> None:
        if not batch:
            return
        stacked = torch.stack(batch).to(device)
        with torch.no_grad(), torch.amp.autocast(device):
            features = model.encode_image(stacked)
        chunks.append(features.float().cpu())
        batch.clear()

    for item in images:
        img = Image.open(item) if isinstance(item, str) else item
        batch.append(preprocess(img.convert("RGB")))
        if len(batch) == batch_size:
            flush()
    flush()

    if not chunks:
        return torch.empty(0)
    return F.normalize(torch.cat(chunks, dim=0), dim=-1)


def class_scores(
    image_features: torch.Tensor, prompt_features: Sequence[torch.Tensor]
) -> np.ndarray:
    """Per-class similarity scores, ``(N, n_classes)``. Max over a class's prompts."""
    columns = [(image_features @ pf.T).numpy().max(axis=1) for pf in prompt_features]
    return np.stack(columns, axis=-1)


def predict(scores: np.ndarray) -> np.ndarray:
    """Predicted class index per image."""
    return np.argmax(scores, axis=-1)


def group_metrics(
    predictions: Sequence[int],
    labels: Sequence[int],
    group_names: Sequence[str],
) -> dict:
    """Overall, per-group and worst-group accuracy.

    ``group_names[i]`` is the subgroup of sample ``i``. Average accuracy is over
    all samples, not the mean of per-group accuracies. Worst-group is the
    minimum over groups present in the data.
    """
    predictions_arr = np.asarray(predictions)
    labels_arr = np.asarray(labels)
    groups = np.asarray(group_names)

    if labels_arr.size == 0:
        raise ValueError("no samples to score")

    correct = predictions_arr == labels_arr
    per_group: dict[str, float] = {}
    counts: dict[str, int] = {}
    for name in sorted(set(groups.tolist())):
        mask = groups == name
        per_group[name] = float(correct[mask].mean())
        counts[name] = int(mask.sum())

    if not per_group:
        raise ValueError("no groups found")

    worst = min(per_group, key=per_group.__getitem__)
    best = max(per_group, key=per_group.__getitem__)
    avg = float(correct.mean())
    return {
        "avg_accuracy": avg,
        "worst_group_accuracy": per_group[worst],
        "worst_group": worst,
        "best_group_accuracy": per_group[best],
        "best_group": best,
        "accuracy_gap": avg - per_group[worst],
        "group_accuracies": per_group,
        "group_counts": counts,
        "n_groups": len(per_group),
        "total_images": int(labels_arr.size),
    }
