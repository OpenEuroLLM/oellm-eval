"""DocVQA 2026 contrib suite (https://github.com/VLR-CVC/DocVQA2026)."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SUITE_NAME = "docvqa2026"

CLUSTER_ENV_VARS: list[str] = []

from oellm.contrib.docvqa2026.task import DocVQA2026ValTask  # noqa: E402

_TASK = DocVQA2026ValTask()

TASK_GROUPS: dict = DocVQA2026ValTask.to_task_groups_dict()


def detect_model_flags(model_path: str) -> str | None:
    """Delegate to DocVQA2026Adapter.to_contrib_flags()."""
    from oellm.contrib.docvqa2026.adapter import DocVQA2026Adapter

    return DocVQA2026Adapter(model_path).to_contrib_flags()


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
    if task != _TASK.engine_task_name:
        raise ValueError(f"Unknown task {task!r}. Expected {_TASK.engine_task_name!r}")

    import json

    from oellm.contrib.docvqa2026.datasets import (
        decode_pages,
        load_val,
        read_max_pages,
    )
    from oellm.contrib.docvqa2026.metrics import aggregate, score_prediction
    from oellm.contrib.docvqa2026.prompts import MASTER_PROMPT
    from oellm.contrib.docvqa2026.runner import (
        generate_answer,
        load_model,
        read_limit,
        read_max_new_tokens,
        resolve_device,
        write_results,
    )

    limit = read_limit(env)
    max_pages = read_max_pages(env)
    max_new_tokens = read_max_new_tokens(env)

    device = resolve_device()
    model, processor = load_model(model_path, device)

    samples = load_val(limit=limit, max_pages=max_pages)
    logger.info(
        "DocVQA 2026: %d questions over %d documents",
        len(samples),
        len({s.doc_id for s in samples}),
    )

    partial_path = output_path.parent / (output_path.stem + ".partial.jsonl")
    partial_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    truncated_docs: set[str] = set()
    hit_token_limit = 0
    current_doc = None
    images: list = []
    with open(partial_path, "w") as partial:
        for i, sample in enumerate(samples, 1):
            if sample.doc_id != current_doc:
                images = decode_pages(sample)
                current_doc = sample.doc_id
            raw, hit_limit = generate_answer(
                model,
                processor,
                sample.question,
                images,
                MASTER_PROMPT,
                device,
                max_new_tokens=max_new_tokens,
            )
            correct, extracted, has_marker = score_prediction(raw, sample.answer)
            if sample.pages_truncated:
                truncated_docs.add(sample.doc_id)
            hit_token_limit += hit_limit
            record = {
                "question_id": sample.question_id,
                "doc_category": sample.doc_category,
                "correct": correct,
                "extracted": extracted,
                "has_marker": has_marker,
                "hit_token_limit": hit_limit,
                "raw": raw,
            }
            records.append(record)
            partial.write(json.dumps(record, ensure_ascii=False) + "\n")
            partial.flush()
            if i % 10 == 0 or i == len(samples):
                logger.info("scored %d/%d questions", i, len(samples))

    metrics = aggregate(records)
    metrics["max_pages"] = max_pages if max_pages else 0
    metrics["n_truncated_documents"] = len(truncated_docs)
    metrics["max_new_tokens"] = max_new_tokens
    metrics["n_hit_token_limit"] = hit_token_limit
    if hit_token_limit:
        logger.warning(
            "%d/%d answers were cut off at DOCVQA2026_MAX_NEW_TOKENS=%d before "
            "finishing; a cut-off answer has no marker and scores wrong",
            hit_token_limit,
            len(records),
            max_new_tokens,
        )
    if not metrics["format_compliance"]:
        logger.warning(
            "no prediction carried the %r marker, so every answer scores wrong "
            "regardless of content — this measures instruction following, not "
            "document reasoning",
            "FINAL ANSWER:",
        )
    if truncated_docs:
        logger.warning(
            "%d/%d documents were truncated (DOCVQA2026_MAX_PAGES=%s); "
            "scores are not comparable with the competition leaderboard",
            len(truncated_docs),
            len({s.doc_id for s in samples}),
            max_pages,
        )

    write_results(output_path, model_path, task, n_shot, metrics)


def parse_results(data: dict) -> tuple[str, str, int, dict[str, float]] | None:
    """Claim *data* if it is this suite's output, else return None."""
    from oellm.contrib.docvqa2026.runner import parse_suite_results

    return parse_suite_results(data, "docvqa2026")
