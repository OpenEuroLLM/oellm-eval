"""Task definition for the DocVQA 2026 competition benchmark."""

from __future__ import annotations

from oellm.core.base_task import BaseTask

SUITE_NAME = "docvqa2026"


class DocVQA2026ValTask(BaseTask):
    """DocVQA 2026 validation split: 80 questions over 25 multi-page documents."""

    @property
    def name(self) -> str:
        return "docvqa2026_val"

    @property
    def suite(self) -> str:
        return SUITE_NAME

    @property
    def task_group_name(self) -> str:
        return "image-docvqa2026"

    @property
    def n_shots(self) -> list[int]:
        return [0]

    @property
    def primary_metric(self) -> str:
        return "accuracy"

    @property
    def description(self) -> str:
        return (
            "DocVQA 2026 (ICDAR) reasoning over multi-page documents in eight "
            "domains, scored with the competition's own strict number/unit/date "
            "matcher and ANLS fallback."
        )

    @property
    def hf_dataset_files(self) -> list[dict]:
        return [{"repo_id": "VLR-CVC/DocVQA-2026", "patterns": ["val.parquet"]}]
