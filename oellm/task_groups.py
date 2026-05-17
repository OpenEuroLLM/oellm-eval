from collections.abc import Iterable
from dataclasses import dataclass, field
from importlib.resources import files

import yaml


@dataclass
class DatasetSpec:
    repo_id: str
    subset: str | None = None
    needs_snapshot_download: bool = False
    # HF dataset revisions to pre-fetch. Most datasets only need `main`;
    # OpenGVLab/MVBench keeps videos on a separate `video` branch.
    revisions: list[str] = field(default_factory=lambda: ["main"])


@dataclass
class _Task:
    name: str
    n_shots: list[int] | None = None
    dataset: str | None = None
    subset: str | None = None
    hf_models: list[str] | None = None
    hf_dataset_files: list[dict] | None = None
    suite: str | None = None
    revisions: list[str] | None = None


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
            task_hf_models = task_data.get("hf_models")
            task_hf_dataset_files = task_data.get("hf_dataset_files")
            task_revisions = task_data.get("revisions")
            tasks.append(
                _Task(
                    name=task_name,
                    n_shots=task_n_shots,
                    dataset=task_dataset,
                    subset=task_subset,
                    hf_models=task_hf_models,
                    hf_dataset_files=task_hf_dataset_files,
                    suite=task_data.get("suite"),
                    revisions=task_revisions,
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


def _parse_task_groups(
    requested_groups: list[str],
) -> dict[str, TaskSuperGroup | TaskGroup]:
    data = (
        yaml.safe_load((files("oellm.resources") / "task-groups.yaml").read_text()) or {}
    )

    from oellm.registry import (
        get_all_task_groups as _contrib_task_groups,  # noqa: PLC0415
    )

    _contrib = _contrib_task_groups()
    data.setdefault("task_metrics", {}).update(_contrib.get("task_metrics", {}))
    data.setdefault("task_groups", {}).update(_contrib.get("task_groups", {}))
    data.setdefault("super_groups", {}).update(_contrib.get("super_groups", {}))

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


def _iter_all_tasks(
    parsed: dict[str, TaskSuperGroup | TaskGroup],
) -> Iterable[tuple[_Task, str, str]]:
    """Yield ``(task, suite, group_name)`` triples from a parsed group dict, flattening super groups."""
    for group_name, group in parsed.items():
        if isinstance(group, TaskGroup):
            for t in group.tasks:
                yield t, t.suite or group.suite, group_name
        else:
            for g in group.task_groups:
                for t in g.tasks:
                    yield t, t.suite or g.suite, g.name


def _expand_task_groups(group_names: Iterable[str]) -> list[TaskGroupResult]:
    parsed = _parse_task_groups([str(n).strip() for n in group_names if str(n).strip()])
    missing = {str(n).strip() for n in group_names if str(n).strip()} - set(parsed.keys())
    if missing:
        raise ValueError(f"Unknown task group(s): {', '.join(sorted(missing))}")

    results: list[TaskGroupResult] = []
    for t, suite, _gname in _iter_all_tasks(parsed):
        for shot in (int(s) for s in (t.n_shots or [])):
            results.append(TaskGroupResult(task=t.name, n_shot=shot, suite=suite))

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

    # Merge specs sharing (repo_id, subset): union their revisions lists.
    by_key: dict[tuple[str, str | None], DatasetSpec] = {}
    order: list[tuple[str, str | None]] = []

    def add_spec(
        dataset: str | None,
        subset: str | None,
        needs_snapshot_download: bool = False,
        revisions: list[str] | None = None,
    ):
        if dataset is None:
            return
        revs = list(revisions) if revisions else ["main"]
        key = (dataset, subset)
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = DatasetSpec(
                repo_id=dataset,
                subset=subset,
                needs_snapshot_download=needs_snapshot_download,
                revisions=revs,
            )
            order.append(key)
        else:
            for r in revs:
                if r not in existing.revisions:
                    existing.revisions.append(r)
            if needs_snapshot_download and not existing.needs_snapshot_download:
                existing.needs_snapshot_download = True

    for t, _, group_name in _iter_all_tasks(parsed):
        needs_snapshot = group_name.startswith(("audio-", "video-"))

        if t.dataset == "facebook/flores" and not t.subset:
            for lang in _extract_flores_subsets(t.name):
                add_spec(t.dataset, lang, revisions=t.revisions)
        else:
            add_spec(
                t.dataset,
                t.subset,
                needs_snapshot_download=needs_snapshot,
                revisions=t.revisions,
            )

    return [by_key[k] for k in order]


def _collect_hf_model_repos(group_names: Iterable[str]) -> list[str]:
    """Return deduplicated HF model repo IDs declared in task ``hf_models`` fields."""
    parsed = _parse_task_groups([str(n).strip() for n in group_names if str(n).strip()])

    repos: list[str] = []
    seen: set[str] = set()

    for t, _, _gname in _iter_all_tasks(parsed):
        for repo_id in t.hf_models or []:
            if repo_id not in seen:
                seen.add(repo_id)
                repos.append(repo_id)

    return repos


def _collect_hf_dataset_files(group_names: Iterable[str]) -> list[dict]:
    """Return deduplicated HF dataset file specs declared in task ``hf_dataset_files`` fields."""
    parsed = _parse_task_groups([str(n).strip() for n in group_names if str(n).strip()])

    # Merge patterns from all tasks that share the same (repo_id, revision)
    # so that a single snapshot_download fetches everything needed.
    merged: dict[tuple[str, str | None], list[str]] = {}

    for t, _, _gname in _iter_all_tasks(parsed):
        for spec in t.hf_dataset_files or []:
            repo_id = spec.get("repo_id", "")
            if not repo_id:
                continue
            revision = spec.get("revision")
            patterns = spec.get("patterns") or []
            key = (repo_id, revision)
            if key not in merged:
                merged[key] = list(patterns)
            else:
                for p in patterns:
                    if p not in merged[key]:
                        merged[key].append(p)

    result = []
    for (rid, rev), pats in merged.items():
        entry: dict = {"repo_id": rid, "patterns": pats}
        if rev:
            entry["revision"] = rev
        result.append(entry)
    return result


def _build_task_dataset_map() -> dict[str, list[DatasetSpec]]:
    """Build a mapping from task names to their dataset specs from all task groups.

    Includes both core YAML task groups and contrib task groups from the registry.
    """
    all_group_names = get_all_task_group_names()
    parsed = _parse_task_groups(all_group_names)

    task_map: dict[str, list[DatasetSpec]] = {}

    for t, _, _gname in _iter_all_tasks(parsed):
        if t.dataset and t.name not in task_map:
            revs = list(t.revisions) if t.revisions else ["main"]
            if t.dataset == "facebook/flores" and not t.subset:
                task_map[t.name] = [
                    DatasetSpec(repo_id=t.dataset, subset=lang, revisions=list(revs))
                    for lang in _extract_flores_subsets(t.name)
                ]
            else:
                task_map[t.name] = [
                    DatasetSpec(repo_id=t.dataset, subset=t.subset, revisions=revs)
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
    """Return ``{task_name: eval_suite}`` across core YAML and contrib plugins.

    Uses :func:`_parse_task_groups` + :func:`_iter_all_tasks` so contrib
    registries (e.g. ``regiondial_bench``) are included, not just the core
    ``task-groups.yaml``.  Task-level ``suite`` overrides group-level.  First
    occurrence wins when a task name appears in multiple groups.

    Consumers should still ``.get(task, "lm_eval")`` — tasks not registered
    in any group simply aren't in the map.
    """
    parsed = _parse_task_groups(get_all_task_group_names())
    task_suite_map: dict[str, str] = {}
    for t, suite, _group in _iter_all_tasks(parsed):
        task_suite_map.setdefault(t.name, suite)
    return task_suite_map


def get_all_task_group_names() -> list[str]:
    """Return all available task group names (core + all contrib suites)."""
    data = (
        yaml.safe_load((files("oellm.resources") / "task-groups.yaml").read_text()) or {}
    )
    core_names = list(data.get("task_groups", {}).keys())

    from oellm.registry import (
        get_all_task_groups as _contrib_task_groups,  # noqa: PLC0415
    )

    contrib_names = list(_contrib_task_groups().get("task_groups", {}).keys())
    return core_names + [n for n in contrib_names if n not in core_names]
