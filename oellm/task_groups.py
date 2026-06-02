import copy
from collections.abc import Iterable
from dataclasses import dataclass
from importlib.resources import files

import yaml


@dataclass
class DatasetSpec:
    repo_id: str
    subset: str | None = None


@dataclass
class _Task:
    name: str
    n_shots: list[int] | None = None
    dataset: str | None = None
    subset: str | None = None
    suite: str | None = None


@dataclass
class TaskGroup:
    name: str
    tasks: list[_Task]
    suite: str
    description: str
    n_shots: list[int] | None = None
    dataset: str | None = None

    def __post_init__(self):
        for task in self.tasks:
            if task.n_shots is None and self.n_shots is not None:
                task.n_shots = self.n_shots
            elif task.n_shots is None and self.n_shots is None:
                raise ValueError(
                    f"N_shots is not set for task {task.name} and no default n_shots is set for the task group: {self.name}"
                )
            if task.dataset is None and self.dataset is not None:
                task.dataset = self.dataset

    @classmethod
    def from_dict(cls, name: str, data: dict) -> "TaskGroup":
        tasks = []
        for task_data in data["tasks"]:
            task_name = task_data["task"]
            task_n_shots = task_data.get("n_shots")
            task_dataset = task_data.get("dataset")
            task_subset = task_data.get("subset")
            tasks.append(
                _Task(
                    name=task_name,
                    n_shots=task_n_shots,
                    dataset=task_dataset,
                    subset=task_subset,
                    suite=task_data.get("suite"),
                )
            )

        return cls(
            name=name,
            tasks=tasks,
            suite=data["suite"],
            description=data["description"],
            n_shots=data.get("n_shots"),
            dataset=data.get("dataset"),
        )


@dataclass
class TaskSuperGroup:
    name: str
    task_groups: list[TaskGroup]
    description: str

    def __post_init__(self):
        resolved_groups = []
        for group in self.task_groups:
            if isinstance(group, str):
                raise ValueError(
                    f"Task group '{group}' not found in available task groups"
                )
            resolved_groups.append(group)
        self.task_groups = resolved_groups

    @classmethod
    def from_dict(
        cls, name: str, data: dict, available_task_groups: dict[str, TaskGroup]
    ) -> "TaskSuperGroup":
        task_groups = []
        for task_group_data in data["task_groups"]:
            group_name = task_group_data["task"]
            if group_name not in available_task_groups:
                raise ValueError(
                    f"Task group '{group_name}' not found in available task groups"
                )
            task_groups.append(available_task_groups[group_name])

        return cls(
            name=name,
            task_groups=task_groups,
            description=data["description"],
        )


def _expand_lang_templates(data: dict) -> dict:
    """Expand ``{lang}`` placeholders in task-group task entries.

    A task group may declare a top-level ``valid_langs`` list.  Every task
    entry whose ``task`` name or ``subset`` value contains the literal string
    ``{lang}`` is expanded into one entry per language, with ``{lang}``
    substituted by that language code.  Entries without ``{lang}`` are left
    unchanged.  The ``valid_langs`` key is removed after expansion.
    """
    result = copy.deepcopy(data)
    for group_data in result.get("task_groups", {}).values():
        valid_langs = group_data.pop("valid_langs", None)
        if not valid_langs:
            continue
        expanded: list[dict] = []
        for task_data in group_data.get("tasks", []):
            task_name = task_data.get("task", "")
            subset = task_data.get("subset", "")
            if "{lang}" in task_name or (subset and "{lang}" in subset):
                for lang in valid_langs:
                    entry = copy.deepcopy(task_data)
                    entry["task"] = task_name.replace("{lang}", lang)
                    if "subset" in entry:
                        entry["subset"] = entry["subset"].replace("{lang}", lang)
                    expanded.append(entry)
            else:
                expanded.append(task_data)
        group_data["tasks"] = expanded
    return result


def _load_task_groups_data() -> dict:
    """Load and pre-process the task-groups YAML, expanding any ``{lang}`` templates."""
    raw = (
        yaml.safe_load((files("oellm.resources") / "task-groups.yaml").read_text()) or {}
    )
    return _expand_lang_templates(raw)


def _parse_task_groups(
    requested_groups: list[str],
) -> dict[str, TaskSuperGroup | TaskGroup]:
    data = _load_task_groups_data()

    task_groups: dict[str, TaskGroup] = {}

    for task_group_name, task_data in data["task_groups"].items():
        task_groups[task_group_name] = TaskGroup.from_dict(task_group_name, task_data)

    super_groups: dict[str, TaskSuperGroup] = {}
    for super_group_name, super_group_data in data.get("super_groups", {}).items():
        super_groups[super_group_name] = TaskSuperGroup.from_dict(
            super_group_name, super_group_data, task_groups
        )

    result = {**task_groups, **super_groups}
    return {
        group_name: group
        for group_name, group in result.items()
        if group_name in requested_groups
    }


@dataclass
class TaskGroupResult:
    task: str
    n_shot: int
    suite: str


def _expand_task_groups(group_names: Iterable[str]) -> list[TaskGroupResult]:
    parsed = _parse_task_groups([str(n).strip() for n in group_names if str(n).strip()])
    missing = {str(n).strip() for n in group_names if str(n).strip()} - set(parsed.keys())
    if missing:
        raise ValueError(f"Unknown task group(s): {', '.join(sorted(missing))}")

    results: list[TaskGroupResult] = []

    for _, group in parsed.items():
        if isinstance(group, TaskGroup):
            suite = group.suite
            for t in group.tasks:
                shots = [int(s) for s in (t.n_shots or [])]
                task_suite = t.suite or suite
                for shot in shots:
                    results.append(
                        TaskGroupResult(task=t.name, n_shot=shot, suite=task_suite)
                    )
        else:
            for g in group.task_groups:
                suite = g.suite
                for t in g.tasks:
                    shots = [int(s) for s in (t.n_shots or [])]
                    task_suite = t.suite or suite
                    for shot in shots:
                        results.append(
                            TaskGroupResult(task=t.name, n_shot=shot, suite=task_suite)
                        )

    return results


def _extract_flores_subsets(task_name: str) -> list[str]:
    """Extract language subsets from flores-style task names like 'flores200:bul_Cyrl-eng_Latn'.

    Returns both the translation pair (e.g. 'bul_Cyrl-eng_Latn') that lighteval needs,
    and the individual languages for potential fallback.
    """
    if not task_name.startswith("flores200:"):
        return []
    lang_part = task_name.split(":", 1)[1]
    if "-" in lang_part:
        return [lang_part] + lang_part.split("-")
    return []


def _collect_dataset_specs(group_names: Iterable[str]) -> list[DatasetSpec]:
    parsed = _parse_task_groups([str(n).strip() for n in group_names if str(n).strip()])

    specs: list[DatasetSpec] = []
    seen: set[tuple[str, str | None]] = set()

    def add_spec(dataset: str | None, subset: str | None):
        if dataset is None:
            return
        key = (dataset, subset)
        if key not in seen:
            seen.add(key)
            specs.append(DatasetSpec(repo_id=dataset, subset=subset))

    for _, group in parsed.items():
        if isinstance(group, TaskGroup):
            for t in group.tasks:
                if t.dataset == "facebook/flores" and not t.subset:
                    for lang in _extract_flores_subsets(t.name):
                        add_spec(t.dataset, lang)
                else:
                    add_spec(t.dataset, t.subset)
        else:
            for g in group.task_groups:
                for t in g.tasks:
                    if t.dataset == "facebook/flores" and not t.subset:
                        for lang in _extract_flores_subsets(t.name):
                            add_spec(t.dataset, lang)
                    else:
                        add_spec(t.dataset, t.subset)

    return specs


def _build_task_dataset_map() -> dict[str, list[DatasetSpec]]:
    """Build a mapping from task names to their dataset specs from all task groups."""
    data = _load_task_groups_data()

    all_group_names = list(data.get("task_groups", {}).keys())
    parsed = _parse_task_groups(all_group_names)

    task_map: dict[str, list[DatasetSpec]] = {}

    for _, group in parsed.items():
        if isinstance(group, TaskGroup):
            for t in group.tasks:
                if t.dataset and t.name not in task_map:
                    if t.dataset == "facebook/flores" and not t.subset:
                        task_map[t.name] = [
                            DatasetSpec(repo_id=t.dataset, subset=lang)
                            for lang in _extract_flores_subsets(t.name)
                        ]
                    else:
                        task_map[t.name] = [
                            DatasetSpec(repo_id=t.dataset, subset=t.subset)
                        ]

    return task_map


def _lookup_dataset_specs_for_tasks(task_names: Iterable[str]) -> list[DatasetSpec]:
    """Look up dataset specs for individual task names from the task groups registry."""
    task_map = _build_task_dataset_map()

    specs: list[DatasetSpec] = []
    seen: set[tuple[str, str | None]] = set()

    for task_name in task_names:
        task_name = str(task_name).strip()
        if not task_name:
            continue
        task_specs = task_map.get(task_name, [])
        for spec in task_specs:
            key = (spec.repo_id, spec.subset)
            if key not in seen:
                seen.add(key)
                specs.append(spec)

    return specs


def _build_task_suite_map() -> dict[str, str]:
    """Build a mapping from task names to their suite from all task groups."""
    data = _load_task_groups_data()

    task_suite_map: dict[str, str] = {}
    for _, group_data in data.get("task_groups", {}).items():
        group_suite = group_data.get("suite", "lm-eval-harness")
        for task_data in group_data.get("tasks", []):
            task_name = task_data.get("task")
            task_suite = task_data.get("suite", group_suite)
            if task_name and task_name not in task_suite_map:
                task_suite_map[task_name] = task_suite

    return task_suite_map


def get_all_task_group_names() -> list[str]:
    """Return all available task group names (excluding super_groups)."""
    data = _load_task_groups_data()
    return list(data.get("task_groups", {}).keys())
