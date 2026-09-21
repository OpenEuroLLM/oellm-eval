"""Loading the DocVQA 2026 val split as one sample per question."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

HF_REPO = "VLR-CVC/DocVQA-2026"
SPLIT = "val"


@dataclass
class Sample:
    """One scorable question with the still-encoded pages of its document."""

    question_id: str
    doc_id: str
    doc_category: str
    question: str
    answer: str
    encoded_pages: list = field(repr=False, default_factory=list)
    n_pages_total: int = 0

    @property
    def pages_truncated(self) -> bool:
        return len(self.encoded_pages) < self.n_pages_total


def decode_pages(sample: Sample) -> list:
    """Decode one sample's kept pages to PIL RGB images."""
    return [_decode_page(page) for page in sample.encoded_pages]


def _decode_page(entry: dict):
    import io

    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    if entry.get("bytes") is not None:
        return Image.open(io.BytesIO(entry["bytes"])).convert("RGB")
    return Image.open(entry["path"]).convert("RGB")


def read_max_pages(env: dict[str, str]) -> int | None:
    raw = env.get("DOCVQA2026_MAX_PAGES", "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(
            f"DOCVQA2026_MAX_PAGES must be an integer >= 1, got {raw!r}"
        ) from None
    if value < 1:
        raise ValueError(f"DOCVQA2026_MAX_PAGES must be >= 1, got {value!r}")
    return value


def load_val(limit: int | None = None, max_pages: int | None = None) -> list[Sample]:
    """Return the val split's questions in dataset order, read via pyarrow."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(HF_REPO, f"{SPLIT}.parquet", repo_type="dataset")
    columns = ["doc_id", "doc_category", "questions", "answers", "document"]

    samples: list[Sample] = []
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=1, columns=columns):
        for row in batch.to_pylist():
            encoded = list(row["document"] or [])
            kept = encoded[:max_pages] if max_pages else encoded
            answers = dict(
                zip(
                    row["answers"]["question_id"],
                    row["answers"]["answer"],
                    strict=True,
                )
            )
            for qid, question in zip(
                row["questions"]["question_id"],
                row["questions"]["question"],
                strict=True,
            ):
                if qid not in answers:
                    logger.warning("question %s has no answer; skipping", qid)
                    continue
                samples.append(
                    Sample(
                        question_id=qid,
                        doc_id=row["doc_id"],
                        doc_category=row["doc_category"],
                        question=question,
                        answer=answers[qid],
                        encoded_pages=kept,
                        n_pages_total=len(encoded),
                    )
                )
                if limit and len(samples) >= limit:
                    return samples
    return samples
