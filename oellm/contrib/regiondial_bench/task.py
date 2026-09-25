"""RegionDial-Bench task definitions.

Two splits from the RegionDial-Bench benchmark (Sun et al., ICLR 2026):
- RefCOCOg Multi-turn (1,580 images, 4,405 turns)
- RefCOCO+ Multi-turn (715 images, 2,355 turns)
"""

from oellm.core.base_task import BaseTask

_TASK_GROUP_ALL = "regiondial-bench"
_SUITE = "regiondial_bench"
_HF_MODELS = ["Ricky06662/TaskRouter-1.5B", "facebook/sam2-hiera-large"]
_HF_REPO = "lmsdss/regionreasoner_test_data"


class RegionDialRefCOCOgTask(BaseTask):
    """RegionDial-Bench — RefCOCOg Multi-turn split."""

    @property
    def name(self) -> str:
        return "regiondial_refcocog"

    @property
    def suite(self) -> str:
        return _SUITE

    @property
    def n_shots(self) -> list[int]:
        return [0]

    @property
    def task_group_name(self) -> str:
        return _TASK_GROUP_ALL

    @property
    def description(self) -> str:
        return (
            "RegionDial-Bench RefCOCOg Multi-turn split (1,580 images, 4,405 turns). "
            "Requires REGION_REASONER_DIR on cluster."
        )

    @property
    def primary_metric(self) -> str:
        return "gIoU"

    @property
    def hf_models(self) -> list[str]:
        return _HF_MODELS

    @property
    def hf_dataset_files(self) -> list[dict]:
        return [
            {
                "repo_id": _HF_REPO,
                "patterns": [
                    "raw/refcocog_multi_turn.json",
                    "raw/refcocog_test_multi_bbox_images/*",
                ],
            }
        ]


class RegionDialRefCOCOplusTask(BaseTask):
    """RegionDial-Bench — RefCOCO+ Multi-turn split."""

    @property
    def name(self) -> str:
        return "regiondial_refcocoplus"

    @property
    def suite(self) -> str:
        return _SUITE

    @property
    def n_shots(self) -> list[int]:
        return [0]

    @property
    def task_group_name(self) -> str:
        return _TASK_GROUP_ALL

    @property
    def description(self) -> str:
        return (
            "RegionDial-Bench RefCOCO+ Multi-turn split (715 images, 2,355 turns). "
            "Requires REGION_REASONER_DIR on cluster."
        )

    @property
    def primary_metric(self) -> str:
        return "gIoU"

    @property
    def hf_models(self) -> list[str]:
        return _HF_MODELS

    @property
    def hf_dataset_files(self) -> list[dict]:
        return [
            {
                "repo_id": _HF_REPO,
                "patterns": [
                    "raw/refcocoplus_multi_turn.json",
                    "raw/refcocoplus_test_multi_bbox_images/*",
                ],
            }
        ]
