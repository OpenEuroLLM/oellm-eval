"""Region-grounding benchmark metrics (model-agnostic).

:class:`BaseMetric` subclasses that compute region-grounding metrics from
per-sample inference results.  These are used by ``suite._aggregate_shards()``
to score any model's predictions on RegionDial-Bench.

Input format (BaseMetric API v2)
--------------------------------
Each sample is a dict of pre-computed fields from the inference script::

    {
      "intersection": 12345,
      "union": 23456,
      "bbox_iou": 0.73,
      "image_id": "0001",
    }

Entries that are not dicts (``None``, upstream parse failures) are treated
as failed samples: the per-sample metrics (gIoU, bbox_AP, pass rates) score
them 0, while cIoU — a corpus-level ratio — excludes them from both sums.

Metrics
-------
- **GIoU**: mean of per-sample mask IoU (intersection / union).
- **CIoU**: sum of all intersections / sum of all unions.
- **BboxAP**: fraction of samples where bbox IoU > 0.5.
- **PassRate**: fraction of samples where mask IoU > *threshold*.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from oellm.core.base_metric import BaseMetric


def _as_sample(s: Any) -> dict | None:
    """Return *s* if it is a usable sample record, else ``None``."""
    return s if isinstance(s, dict) else None


def _mask_iou(sample: dict) -> float:
    """Compute mask IoU from pre-computed intersection and union."""
    intersection = sample.get("intersection", 0)
    union = sample.get("union", 0)
    if union <= 0:
        return 0.0
    return intersection / union


class GIoU(BaseMetric):
    """Mean per-sample mask IoU (gIoU as reported in RegionDial-Bench).

    Formula: ``mean(intersection_i / union_i)`` over all samples; failed
    samples count as 0.
    """

    @property
    def name(self) -> str:
        return "gIoU"

    def compute(self, samples: Sequence[Any]) -> float:
        if not samples:
            return 0.0
        ious = []
        for s in samples:
            sample = _as_sample(s)
            ious.append(_mask_iou(sample) if sample else 0.0)
        return sum(ious) / len(ious)


class CIoU(BaseMetric):
    """cIoU as reported in RegionDial-Bench.

    Formula: ``sum(all intersections) / sum(all unions)``; failed samples
    are excluded from both sums (they carry no area information).
    """

    @property
    def name(self) -> str:
        return "cIoU"

    def compute(self, samples: Sequence[Any]) -> float:
        total_intersection = 0.0
        total_union = 0.0
        for s in samples:
            sample = _as_sample(s)
            if sample is None:
                continue
            total_intersection += sample.get("intersection", 0)
            total_union += sample.get("union", 0)
        if total_union <= 0.0:
            return 0.0
        return total_intersection / total_union


class BboxAP(BaseMetric):
    """Binarised bounding-box AP at IoU threshold 0.5.

    Uses the pre-computed ``bbox_iou`` field from the inference script.
    Formula: ``mean(bbox_iou_i > 0.5)``; failed samples count as misses.
    """

    @property
    def name(self) -> str:
        return "bbox_AP"

    def compute(self, samples: Sequence[Any]) -> float:
        if not samples:
            return 0.0
        hits = 0
        for s in samples:
            sample = _as_sample(s)
            if sample and sample.get("bbox_iou", 0) > 0.5:
                hits += 1
        return hits / len(samples)


class PassRate(BaseMetric):
    """Fraction of samples where mask IoU exceeds a configurable threshold.

    Args:
        threshold: IoU threshold in (0, 1).  Common values: 0.3, 0.5, 0.7, 0.9.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        if not 0 < threshold < 1:
            raise ValueError(f"threshold must be in (0, 1), got {threshold}")
        self._threshold = threshold

    @property
    def name(self) -> str:
        t = self._threshold
        label = f"{t:.10g}"
        return f"pass_rate_{label}"

    def compute(self, samples: Sequence[Any]) -> float:
        if not samples:
            return 0.0
        hits = 0
        for s in samples:
            sample = _as_sample(s)
            if sample and _mask_iou(sample) > self._threshold:
                hits += 1
        return hits / len(samples)
