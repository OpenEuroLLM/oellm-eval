from abc import ABC, abstractmethod

from oellm.task_groups import DatasetSpec


class BaseTask(ABC):
    """Abstract base class for evaluation task plugins.

    Example::

        class MyTask(BaseTask):
            @property
            def name(self) -> str:
                return "my_task"

            @property
            def suite(self) -> str:
                return "my_suite"

            @property
            def n_shots(self) -> list[int]:
                return [0]

            @property
            def primary_metric(self) -> str | None:
                return "my_metric"

            @property
            def hf_models(self) -> list[str]:
                return ["org/aux-model"]

            @property
            def hf_dataset_files(self) -> list[dict]:
                return [{"repo_id": "org/dataset", "patterns": ["data/*.json"]}]
    """

    # ------------------------------------------------------------------
    # Required
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Canonical task name — used as the CSV ``task_path`` column value."""

    @property
    @abstractmethod
    def suite(self) -> str:
        """Evaluation suite identifier.

        One of: ``lm_eval``, ``lighteval``, ``lmms_eval``, or a contrib
        ``SUITE_NAME`` (e.g. ``"regiondial_bench"``).
        """

    @property
    @abstractmethod
    def n_shots(self) -> list[int]:
        """List of n-shot values to evaluate at."""

    # ------------------------------------------------------------------
    # Optional — override as needed
    # ------------------------------------------------------------------

    @property
    def engine_task_name(self) -> str:
        """Task name as passed to the eval engine CLI.

        Defaults to :attr:`name`. Override when the engine's registered task
        name differs from the canonical name used in the CSV / YAML.
        """
        return self.name

    @property
    def description(self) -> str:
        """Human-readable description shown in task group listings."""
        return ""

    @property
    def task_group_name(self) -> str:
        """Key used for this task in the ``task_groups`` dict.

        Defaults to :attr:`name` with underscores replaced by dashes.
        """
        return self.name.replace("_", "-")

    @property
    def primary_metric(self) -> str | None:
        """Primary metric key written to the results CSV.

        Maps to the ``task_metrics`` entry in the generated TASK_GROUPS dict.
        Return ``None`` to omit (collect_results will use its fallback chain).
        """
        return None

    @property
    def dataset_specs(self) -> list[DatasetSpec]:
        """HF datasets for pre-download via ``load_dataset()``.

        Use for small datasets accessed through the ``datasets`` library.
        For large files downloaded via ``snapshot_download()``, use
        :attr:`hf_dataset_files` instead.
        """
        return []

    @property
    def hf_models(self) -> list[str]:
        """Auxiliary HF model repos pre-downloaded via ``snapshot_download()``.

        List repos required at eval time but not the primary model under
        evaluation (e.g. a task router or segmentation model).
        """
        return []

    @property
    def hf_dataset_files(self) -> list[dict]:
        """Specific dataset files pre-downloaded via ``snapshot_download()``.

        Use instead of :attr:`dataset_specs` when only a subset of a large
        dataset repo is needed, or when files are read directly (not via the
        ``datasets`` library).

        Each entry: ``{"repo_id": "org/dataset", "patterns": ["glob/..."]}``
        """
        return []

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def to_task_groups_dict(cls) -> dict:
        """Generate the ``TASK_GROUPS`` dict for use in ``suite.py``::

            TASK_GROUPS: dict = MyTask.to_task_groups_dict()

        Note:
            The subclass must be instantiable with no arguments.
        """
        inst = cls()

        # The engine-facing name goes into the task entry: it is what lands
        # in jobs.csv, what the engine CLI receives, and what results come
        # back under — so collect and --check stay consistent. Overriding
        # engine_task_name is the plan's "one-liner plugin" pathway.
        task_entry: dict = {"task": inst.engine_task_name}
        if inst.dataset_specs:
            task_entry["dataset"] = inst.dataset_specs[0].repo_id
            if inst.dataset_specs[0].subset:
                task_entry["subset"] = inst.dataset_specs[0].subset
        if inst.hf_models:
            task_entry["hf_models"] = inst.hf_models
        if inst.hf_dataset_files:
            task_entry["hf_dataset_files"] = inst.hf_dataset_files

        task_group: dict = {
            "suite": inst.suite,
            "n_shots": inst.n_shots,
            "tasks": [task_entry],
        }
        if inst.description:
            task_group["description"] = inst.description

        result: dict = {"task_groups": {inst.task_group_name: task_group}}
        if inst.primary_metric:
            result["task_metrics"] = {inst.engine_task_name: inst.primary_metric}
        return result
