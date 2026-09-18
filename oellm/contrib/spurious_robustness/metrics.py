"""Metrics for the spurious-robustness benchmarks.

A *sample record* is a dict::

    {"correct": bool, "group": str}

``group`` is the subgroup the sample belongs to — label crossed with every
spurious attribute. The group is attached at data-loading time, where the
attributes are still available, because by the time predictions exist the
attributes are gone. See ``datasets.py`` for the group strings each benchmark
produces and ``README.md`` for what they mean.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from oellm.core.base_metric import BaseMetric


def _valid(samples: Sequence[Any]) -> list[dict]:
    """Keep only well-formed records.

    Malformed entries are dropped rather than scored as incorrect: they mean the
    inference step failed to produce a record at all, and silently converting
    that into a wrong answer would understate a group's accuracy without any
    signal that data was lost.
    """
    return [
        s
        for s in samples
        if isinstance(s, dict) and "correct" in s and s.get("group") is not None
    ]


class AverageAccuracy(BaseMetric):
    """Accuracy over all samples, ignoring group membership."""

    @property
    def name(self) -> str:
        return "avg_accuracy"

    def compute(self, samples: Sequence[Any]) -> float:
        records = _valid(samples)
        if not records:
            return 0.0
        return sum(bool(r["correct"]) for r in records) / len(records)


class WorstGroupAccuracy(BaseMetric):
    """Minimum per-group accuracy across the groups present in *samples*.

    A group with no samples is absent from the minimum rather than scored 0.0 —
    an unobserved group has no accuracy to be worst. This matters whenever a run
    is subsampled: with ``--limit`` the rare groups may vanish entirely, and
    treating them as zero would report a worst-group of 0.0 that says nothing
    about the model. Use :func:`zeroshot.group_metrics` when the per-group
    counts are needed to check that coverage was complete.

    With a single group this degenerates to plain accuracy, which is the correct
    reading: ImageNet has no spurious attribute and therefore one group.
    """

    @property
    def name(self) -> str:
        return "worst_group_accuracy"

    def compute(self, samples: Sequence[Any]) -> float:
        records = _valid(samples)
        if not records:
            return 0.0

        totals: dict[str, int] = {}
        hits: dict[str, int] = {}
        for r in records:
            g = str(r["group"])
            totals[g] = totals.get(g, 0) + 1
            hits[g] = hits.get(g, 0) + bool(r["correct"])

        return min(hits[g] / totals[g] for g in totals)
