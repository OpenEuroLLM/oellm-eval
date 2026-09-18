import pytest

from oellm.core import BaseMetric, BaseModelAdapter, BaseTask
from oellm.task_groups import DatasetSpec

# ── Concrete implementations for testing ─────────────────────────────────────


class MinimalTask(BaseTask):
    @property
    def name(self) -> str:
        return "my_task"

    @property
    def suite(self) -> str:
        return "lmms_eval"

    @property
    def n_shots(self) -> list[int]:
        return [0]


class TaskWithDataset(MinimalTask):
    @property
    def dataset_specs(self) -> list[DatasetSpec]:
        return [DatasetSpec(repo_id="org/repo", subset="val")]


class TaskWithCustomEngineName(MinimalTask):
    @property
    def engine_task_name(self) -> str:
        return "engine_specific_name"


class ExactMatchMetric(BaseMetric):
    @property
    def name(self) -> str:
        return "exact_match"

    def compute(self, samples) -> float:
        if not samples:
            return 0.0
        return sum(1 for s in samples if s["prediction"] == s["reference"]) / len(samples)


class HFAdapter(BaseModelAdapter):
    @property
    def model_path(self) -> str:
        return "/models/llava"

    def to_lm_eval_args(self) -> str:
        return f"pretrained={self.model_path},trust_remote_code=True"

    def to_lmms_eval_args(self) -> str:
        return f"pretrained={self.model_path}"


# ── BaseTask tests ────────────────────────────────────────────────────────────


class TestBaseTask:
    def test_abstract_class_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            BaseTask()  # type: ignore[abstract]

    def test_concrete_subclass_instantiates(self):
        t = MinimalTask()
        assert t.name == "my_task"
        assert t.suite == "lmms_eval"
        assert t.n_shots == [0]

    def test_engine_task_name_defaults_to_name(self):
        t = MinimalTask()
        assert t.engine_task_name == t.name

    def test_engine_task_name_can_be_overridden(self):
        t = TaskWithCustomEngineName()
        assert t.name == "my_task"
        assert t.engine_task_name == "engine_specific_name"

    def test_dataset_specs_defaults_to_empty_list(self):
        t = MinimalTask()
        assert t.dataset_specs == []

    def test_dataset_specs_can_be_overridden(self):
        t = TaskWithDataset()
        assert len(t.dataset_specs) == 1
        spec = t.dataset_specs[0]
        assert spec.repo_id == "org/repo"
        assert spec.subset == "val"

    def test_missing_abstract_name_raises(self):
        class BadTask(BaseTask):
            @property
            def suite(self) -> str:
                return "lm_eval"

            @property
            def n_shots(self) -> list[int]:
                return [5]

        with pytest.raises(TypeError):
            BadTask()  # type: ignore[abstract]

    def test_missing_abstract_suite_raises(self):
        class BadTask(BaseTask):
            @property
            def name(self) -> str:
                return "task"

            @property
            def n_shots(self) -> list[int]:
                return [5]

        with pytest.raises(TypeError):
            BadTask()  # type: ignore[abstract]

    def test_missing_abstract_n_shots_raises(self):
        class BadTask(BaseTask):
            @property
            def name(self) -> str:
                return "task"

            @property
            def suite(self) -> str:
                return "lm_eval"

        with pytest.raises(TypeError):
            BadTask()  # type: ignore[abstract]


# ── BaseMetric tests ──────────────────────────────────────────────────────────


class TestBaseMetric:
    def test_abstract_class_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            BaseMetric()  # type: ignore[abstract]

    def test_concrete_subclass_instantiates(self):
        m = ExactMatchMetric()
        assert m.name == "exact_match"

    def test_compute_perfect_score(self):
        m = ExactMatchMetric()
        assert (
            m.compute([{"prediction": x, "reference": x} for x in ("a", "b", "c")]) == 1.0
        )

    def test_compute_zero_score(self):
        m = ExactMatchMetric()
        assert (
            m.compute(
                [
                    {"prediction": "a", "reference": "x"},
                    {"prediction": "b", "reference": "y"},
                ]
            )
            == 0.0
        )

    def test_compute_partial_score(self):
        m = ExactMatchMetric()
        assert m.compute(
            [
                {"prediction": "a", "reference": "a"},
                {"prediction": "b", "reference": "x"},
            ]
        ) == pytest.approx(0.5)

    def test_compute_empty_returns_zero(self):
        m = ExactMatchMetric()
        assert m.compute([]) == 0.0

    def test_missing_abstract_name_raises(self):
        class BadMetric(BaseMetric):
            def compute(self, samples):
                return 0.0

        with pytest.raises(TypeError):
            BadMetric()  # type: ignore[abstract]

    def test_missing_abstract_compute_raises(self):
        class BadMetric(BaseMetric):
            @property
            def name(self):
                return "bad"

        with pytest.raises(TypeError):
            BadMetric()  # type: ignore[abstract]


# ── BaseModelAdapter tests ────────────────────────────────────────────────────


class TestBaseModelAdapter:
    def test_abstract_class_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            BaseModelAdapter()  # type: ignore[abstract]

    def test_concrete_subclass_instantiates(self):
        a = HFAdapter()
        assert a.model_path == "/models/llava"

    def test_to_lm_eval_args_contains_path(self):
        a = HFAdapter()
        args = a.to_lm_eval_args()
        assert "pretrained=/models/llava" in args
        assert "trust_remote_code=True" in args

    def test_to_lmms_eval_args_contains_path(self):
        a = HFAdapter()
        args = a.to_lmms_eval_args()
        assert args == "pretrained=/models/llava"

    def test_missing_abstract_model_path_raises(self):
        class BadAdapter(BaseModelAdapter):
            def to_lm_eval_args(self):
                return ""

            def to_lmms_eval_args(self):
                return ""

        with pytest.raises(TypeError):
            BadAdapter()  # type: ignore[abstract]

    def test_missing_abstract_to_lm_eval_args_raises(self):
        class BadAdapter(BaseModelAdapter):
            @property
            def model_path(self):
                return "/path"

            def to_lmms_eval_args(self):
                return ""

        with pytest.raises(TypeError):
            BadAdapter()  # type: ignore[abstract]

    def test_missing_abstract_to_lmms_eval_args_raises(self):
        class BadAdapter(BaseModelAdapter):
            @property
            def model_path(self):
                return "/path"

            def to_lm_eval_args(self):
                return ""

        with pytest.raises(TypeError):
            BadAdapter()  # type: ignore[abstract]
