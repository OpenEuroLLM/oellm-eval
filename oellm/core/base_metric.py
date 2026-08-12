from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any


class BaseMetric(ABC):
    """Abstract base class for custom metric implementations.

    Implement this when an evaluation task requires a metric not natively
    supported by lm-eval or lmms-eval (e.g. a custom IoU score for grounding
    benchmarks, or a domain-specific accuracy metric).

    Contract (API v2 — see ``oellm.core.CORE_API_VERSION``): ``compute``
    receives the task's per-sample records and returns one scalar. A *sample*
    is whatever record the task's inference step produces — a dict is
    recommended. This replaces the old ``(predictions: list[str],
    references: list[str])`` signature, which could not express
    multi-reference tasks (VQA accuracy, ANLS), weighted aggregation, or
    non-mean corpus aggregates (e.g. summed-IoU ratios), and forced
    implementations to smuggle structured data through JSON strings.

    Conventions:
    - Return a scalar in [0, 1] unless the metric name makes another range
      unambiguous (e.g. an explicit ``/1000`` score).
    - Entries that are not valid sample records (``None``, parse failures)
      must be handled deliberately — scored as failures or excluded — and
      the choice documented on the metric class.

    Example::

        class ExactMatchMetric(BaseMetric):
            @property
            def name(self) -> str:
                return "exact_match"

            def compute(self, samples: Sequence[Any]) -> float:
                if not samples:
                    return 0.0
                correct = sum(
                    1
                    for s in samples
                    if isinstance(s, dict)
                    and s.get("prediction") == s.get("reference")
                )
                return correct / len(samples)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique metric identifier, e.g. ``"vqa_score"`` or ``"anls"``."""

    @abstractmethod
    def compute(self, samples: Sequence[Any]) -> float:
        """Compute the metric over per-sample records.

        Args:
            samples: One record per evaluated sample (dicts recommended;
                multi-reference tasks put their references inside the record).

        Returns:
            Scalar score. Conventionally in [0, 1]; higher is better unless
            the metric name says otherwise (e.g. WER).
        """
