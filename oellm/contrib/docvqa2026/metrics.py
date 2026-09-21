"""Scoring and aggregation for DocVQA 2026."""

from __future__ import annotations

from collections import defaultdict

MARKER = "FINAL ANSWER:"


def score_prediction(raw_prediction: str, ground_truth: str) -> tuple[bool, str, bool]:
    """Official verdict for one prediction: (correct, extracted, has_marker)."""
    from oellm.contrib.docvqa2026._vendor_eval_utils import (
        evaluate_docvqa_prediction,
    )

    correct, extracted = evaluate_docvqa_prediction(raw_prediction, ground_truth)
    return bool(correct), extracted, MARKER in str(raw_prediction)


def aggregate(records: list[dict]) -> dict[str, float | int]:
    """Aggregate per-question verdicts into the reported metric set."""
    if not records:
        raise RuntimeError("DocVQA 2026 evaluation produced no samples")

    per_category: dict[str, list[bool]] = defaultdict(list)
    for r in records:
        per_category[r["doc_category"]].append(bool(r["correct"]))

    n_correct = sum(bool(r["correct"]) for r in records)
    category_rates = {cat: sum(v) / len(v) for cat, v in sorted(per_category.items())}

    n_marked = sum(bool(r.get("has_marker")) for r in records)

    metrics: dict[str, float | int] = {
        "accuracy": n_correct / len(records),
        "format_compliance": n_marked / len(records),
        "macro_accuracy": sum(category_rates.values()) / len(category_rates),
        "n_questions": len(records),
        "n_correct": n_correct,
        "n_categories": len(category_rates),
    }
    for cat, rate in category_rates.items():
        metrics[f"acc_{cat}"] = rate
    return metrics
