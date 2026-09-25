"""UrbanCars task definition."""

from __future__ import annotations

from oellm.core.base_task import BaseTask
from oellm.task_groups import DatasetSpec

SUITE_NAME = "spurious_urbancars"


class SpuriousUrbanCarsTask(BaseTask):
    """UrbanCars: two spurious attributes (background, co-occurring object)."""

    @property
    def name(self) -> str:
        return "spurious_urbancars"

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
            "UrbanCars worst-group accuracy over car type x background x "
            "co-occurring object (8 groups). Multi-attribute robustness. Reads a "
            "prebuilt image tree from URBANCARS_DATA_DIR; not available on the Hub."
        )

    @property
    def dataset_specs(self) -> list[DatasetSpec]:
        # UrbanCars is composited locally rather than published, so there is
        # nothing for the login node to pre-stage. URBANCARS_DATA_DIR supplies
        # the image tree and is checked before submission via CLUSTER_ENV_VARS.
        return []
