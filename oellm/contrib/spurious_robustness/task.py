"""Task definitions for the spurious-robustness benchmarks."""

from __future__ import annotations

from oellm.core.base_task import BaseTask
from oellm.task_groups import DatasetSpec

SUITE_NAME = "spurious_robustness"


class SpuriousImageNetTask(BaseTask):
    """ImageNet: overall recognition, the no-spurious-attribute reference point."""

    @property
    def name(self) -> str:
        return "spurious_imagenet"

    @property
    def suite(self) -> str:
        return SUITE_NAME

    @property
    def n_shots(self) -> list[int]:
        return [0]

    @property
    def primary_metric(self) -> str:
        return "top1_accuracy"

    @property
    def description(self) -> str:
        return (
            "ImageNet zero-shot top-1 on the ILSVRC-2012 validation split "
            "(1000-way, 80 OpenAI templates averaged per class). No spurious "
            "attribute — the recognition baseline the worst-group numbers are read against."
        )

    @property
    def dataset_specs(self) -> list[DatasetSpec]:
        return [DatasetSpec(repo_id="ILSVRC/imagenet-1k")]


class SpuriousCelebATask(BaseTask):
    """CelebA: one spurious attribute (gender), four groups."""

    @property
    def name(self) -> str:
        return "spurious_celeba"

    @property
    def suite(self) -> str:
        return SUITE_NAME

    @property
    def n_shots(self) -> list[int]:
        return [0]

    @property
    def primary_metric(self) -> str:
        return "worst_group_accuracy"

    @property
    def description(self) -> str:
        return (
            "CelebA worst-group accuracy over hair colour x gender (4 groups) "
            "on the official test split. Single-attribute robustness."
        )

    @property
    def dataset_specs(self) -> list[DatasetSpec]:
        return [DatasetSpec(repo_id="tpremoli/CelebA-attrs")]


# UrbanCars lives in the sibling ``spurious_urbancars`` suite: it is the only
# task needing a cluster-local directory, and CLUSTER_ENV_VARS applies to every
# task in a suite.
