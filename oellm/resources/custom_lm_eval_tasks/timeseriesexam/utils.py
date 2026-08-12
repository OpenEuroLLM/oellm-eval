"""TimeSeriesExam prompt construction (AutonLab/TimeSeriesExam1).

Rows carry either one series (``ts``) or a pair (``ts1``/``ts2``), each
1000–2500 raw floats — far too long to inline. Serialization policy
(fixed and versioned; identical for every model, so scores stay
comparable): uniform subsample to at most ``_MAX_POINTS`` values in
original order, formatted to 4 significant digits, with the original
length stated in the prompt.

MCQ is MMLU-style: options are printed lettered, the model scores
single-letter continuations — immune to option-length bias, and robust
to the dataset's variable option counts (binary and 4-way questions).
"""

_MAX_POINTS = 128


def _fmt_series(vals) -> str:
    vals = list(vals or [])
    n = len(vals)
    if n > _MAX_POINTS:
        step = n / _MAX_POINTS
        vals = [vals[int(i * step)] for i in range(_MAX_POINTS)]
    return ", ".join(f"{v:.4g}" for v in vals)


def _series_block(doc) -> str:
    parts = []
    if doc.get("ts"):
        parts.append(
            f"Time series ({len(doc['ts'])} points, uniformly subsampled to "
            f"{min(len(doc['ts']), _MAX_POINTS)}):\n{_fmt_series(doc['ts'])}"
        )
    if doc.get("ts1"):
        parts.append(
            f"Series A ({len(doc['ts1'])} points, uniformly subsampled to "
            f"{min(len(doc['ts1']), _MAX_POINTS)}):\n{_fmt_series(doc['ts1'])}"
        )
    if doc.get("ts2"):
        parts.append(
            f"Series B ({len(doc['ts2'])} points, uniformly subsampled to "
            f"{min(len(doc['ts2']), _MAX_POINTS)}):\n{_fmt_series(doc['ts2'])}"
        )
    return "\n\n".join(parts)


def doc_to_text(doc) -> str:
    options = "\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(doc["options"]))
    return (
        f"{_series_block(doc)}\n\n"
        f"Question: {doc['question']}\n"
        f"Options:\n{options}\n"
        "Answer:"
    )


def doc_to_choice(doc) -> list[str]:
    return [f" {chr(65 + i)}" for i in range(len(doc["options"]))]


def doc_to_target(doc) -> int:
    """Index of the gold option; the answer field stores the option text."""
    options = [str(o).strip() for o in doc["options"]]
    return options.index(str(doc["answer"]).strip())
