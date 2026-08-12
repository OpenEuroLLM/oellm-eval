import copy
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from importlib.resources import files

import yaml

# Datasets that store media as external URLs (not embedded bytes). A bare
# snapshot_download only fetches the URLs, not the media — they MUST go through
# load_dataset()+_materialize_external_urls() so the per-row HTTP fetch runs on
# the (online) login node. Excluded from the snapshot-only fast path below.
# Currently empty: every in-tree dataset embeds its media (e.g. lmms-lab/textvqa
# ships image bytes in its parquet). Add a repo id here only if it genuinely
# stores external media URLs.
_URL_BASED_DATASETS: set[str] = set()

# Eval suites whose datasets are large media (image/audio/video). Their specs
# are staged with snapshot_download (raw files; the compute node builds the
# dataset at runtime) to avoid OOM-prone load_dataset() builds on the memory-
# capped login node. ``lmms_eval`` = core image/audio/video; ``audiobench`` =
# the audio contrib plugin (suite != lmms_eval but still large audio data).
_SNAPSHOT_SUITES = {"lmms_eval", "audiobench"}

# --- Language normalisation -------------------------------------------------
# Tasks encode their language in several incompatible ways across benchmarks
# (e.g. German is ``deu_Latn``, ``de``, ``German`` and ``deu_latn``). These
# tables fold every benchmark spelling onto a single canonical ``lang_Scri``
# code so that a ``group[deu_Latn]`` bracket can match tasks across benchmarks.
# Note: this folding applies to the *benchmarks'* internal spellings only — the
# user-facing bracket accepts the canonical ``lang_Scri`` form exclusively.
_LANG_ALIAS = {
    # ISO 639-1 two-letter codes (global-mmlu, mgsm, arc-mt)
    "de": "deu_Latn",
    "fr": "fra_Latn",
    "it": "ita_Latn",
    "es": "spa_Latn",
    "pt": "por_Latn",
    "cs": "ces_Latn",
    "el": "ell_Grek",
    "lt": "lit_Latn",
    "nl": "nld_Latn",
    "pl": "pol_Latn",
    "ro": "ron_Latn",
    "ru": "rus_Cyrl",
    "sr": "srp_Cyrl",
    "sv": "swe_Latn",
    "tr": "tur_Latn",
    "uk": "ukr_Cyrl",
    "he": "heb_Hebr",
    "en": "eng_Latn",
    "bg": "bul_Cyrl",
    "da": "dan_Latn",
    "et": "est_Latn",
    "fi": "fin_Latn",
    "hu": "hun_Latn",
    "lv": "lvs_Latn",
    "sk": "slk_Latn",
    "sl": "slv_Latn",
    "is": "isl_Latn",
    "nb": "nob_Latn",
    "no": "nob_Latn",
    "hr": "hrv_Latn",
    "ca": "cat_Latn",
    "eu": "eus_Latn",
    "gl": "glg_Latn",
    # full English names (include)
    "albanian": "als_Latn",
    "armenian": "hye_Armn",
    "azerbaijani": "aze_Latn",
    "basque": "eus_Latn",
    "belarusian": "bel_Cyrl",
    "bulgarian": "bul_Cyrl",
    "croatian": "hrv_Latn",
    "dutch": "nld_Latn",
    "estonian": "est_Latn",
    "finnish": "fin_Latn",
    "french": "fra_Latn",
    "georgian": "kat_Geor",
    "german": "deu_Latn",
    "greek": "ell_Grek",
    "hungarian": "hun_Latn",
    "italian": "ita_Latn",
    "lithuanian": "lit_Latn",
    "north macedonian": "mkd_Cyrl",
    "polish": "pol_Latn",
    "portuguese": "por_Latn",
    "russian": "rus_Cyrl",
    "serbian": "srp_Cyrl",
    "spanish": "spa_Latn",
    "turkish": "tur_Latn",
    "ukrainian": "ukr_Cyrl",
    # xcsqa: the group's `subset` is the HF config (`X-CSQA-<iso1>`, needed for
    # pre-download), so the language is derived from that config spelling here.
    "x-csqa-de": "deu_Latn",
    "x-csqa-en": "eng_Latn",
    "x-csqa-es": "spa_Latn",
    "x-csqa-fr": "fra_Latn",
    "x-csqa-it": "ita_Latn",
    "x-csqa-nl": "nld_Latn",
    "x-csqa-pl": "pol_Latn",
    "x-csqa-pt": "por_Latn",
}
# Distinct individual-language codes folded into a macrolanguage code.
_LANG_SPECIAL = {"ekk_Latn": "est_Latn"}  # global-piqa uses ekk (Standard Estonian)


def _canonical_language(code: str | None) -> str | None:
    """Normalise any language spelling to a canonical ``lang_Scri`` code."""
    if code is None:
        return None
    code = str(code).strip()
    low = code.lower()
    if low in _LANG_ALIAS:
        return _LANG_ALIAS[low]
    # lowercase ``lang_scri`` or ``lang_scri_region`` (e.g. ``por_latn_port``)
    parts = low.split("_")
    if len(parts) >= 2 and len(parts[0]) == 3 and len(parts[1]) == 4:
        base = f"{parts[0]}_{parts[1].capitalize()}"
        return _LANG_SPECIAL.get(base, base)
    # already canonical ``lang_Scri``
    if re.match(r"^[a-z]{3}_[A-Z][a-z]{3}$", code):
        return code
    return None


def _resolve_task_languages(name: str, subset: str | None) -> list[str]:
    """Return the canonical language code(s) a task belongs to, if any.

    Translation pairs (``flores200:src-tgt`` and
    ``opensubtitles_multi40_src_to_tgt``) resolve to their non-English side;
    every other task resolves via its ``subset``. Tasks with no recognisable
    language (e.g. English-only standard benchmarks) return [].
    """
    if name.startswith("flores200:"):
        pair = name.split(":", 1)[1]
        langs = [
            _canonical_language(part) for part in pair.split("-") if part != "eng_Latn"
        ]
        return [lang for lang in langs if lang]
    if name.startswith("opensubtitles_multi40_"):
        pair = name[len("opensubtitles_multi40_") :]
        langs = [_canonical_language(part) for part in pair.split("_to_") if part != "en"]
        return [lang for lang in langs if lang]
    lang = _canonical_language(subset)
    if lang:
        return [lang]
    # Some explicitly-listed tasks omit `subset` and encode the language as a
    # trailing code in the task name (e.g. ``arc_challenge_mt_is``).
    if subset is None:
        m = re.search(r"_([a-z]{2})$", name)
        if m:
            lang = _canonical_language(m.group(1))
            if lang:
                return [lang]
    return []


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
    suite: str | None = None
    languages: list[str] = field(default_factory=list)
    revisions: list[str] = field(default_factory=lambda: ["main"])
    hf_models: list[str] | None = None
    hf_dataset_files: list[dict] | None = None


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
                    languages=_resolve_task_languages(task_name, task_subset),
                    revisions=task_data.get("revisions") or ["main"],
                    hf_models=task_data.get("hf_models"),
                    hf_dataset_files=task_data.get("hf_dataset_files"),
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
    """Load and pre-process the task-groups YAML, expanding any ``{lang}`` templates.

    Core YAML task groups are merged with contrib-plugin task groups from the
    registry, so both schedule through the same code paths. Merging happens
    before ``{lang}`` expansion so contrib groups may also use templates.
    """
    raw = (
        yaml.safe_load((files("oellm.resources") / "task-groups.yaml").read_text()) or {}
    )

    from oellm.registry import (
        get_all_task_groups as _contrib_task_groups,  # noqa: PLC0415
    )

    _contrib = _contrib_task_groups()
    raw.setdefault("task_metrics", {}).update(_contrib.get("task_metrics", {}))
    raw.setdefault("task_groups", {}).update(_contrib.get("task_groups", {}))
    raw.setdefault("super_groups", {}).update(_contrib.get("super_groups", {}))

    return _expand_lang_templates(raw)


def _language_codes_from_groups(task_groups: dict[str, TaskGroup]) -> set[str]:
    """Collect every canonical language code that at least one task resolves to.

    These are the codes accepted in a ``group[lang]`` bracket; a task resolves
    to a code via its ``{lang}`` template expansion or its ``subset`` (see
    ``_resolve_task_languages``).
    """
    return {
        lang
        for group in task_groups.values()
        for t in group.tasks
        for lang in t.languages
    }


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

    # Reserved super_group spanning every task group, generated from the
    # registry so it never needs hand-maintaining as groups are added. Pair it
    # with a [lang] bracket (e.g. ``all[deu_Latn]``) to evaluate a language
    # across every benchmark. Overlapping tasks are de-duplicated downstream.
    if "all" not in task_groups and "all" not in super_groups:
        super_groups["all"] = TaskSuperGroup(
            name="all",
            task_groups=list(task_groups.values()),
            description="Every task group (auto-generated)",
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


def _iter_group_tasks(
    parsed: dict[str, "TaskSuperGroup | TaskGroup"],
) -> Iterable[tuple[str, _Task]]:
    """Yield ``(resolved_suite, task)`` for every task in the parsed groups.

    Flattens both plain task groups and super groups, resolving each task's
    suite from its explicit ``suite`` or the owning group's default.
    """
    for group in parsed.values():
        if isinstance(group, TaskGroup):
            for t in group.tasks:
                yield (t.suite or group.suite), t
        else:
            for g in group.task_groups:
                for t in g.tasks:
                    yield (t.suite or g.suite), t


def _normalise_language_codes(languages: Iterable[str]) -> list[str]:
    """Validate requested language codes against the canonical code set.

    Only the precise ``lang_Scri`` form (e.g. ``deu_Latn``) is accepted in a
    ``group[lang]`` bracket; looser spellings such as ``de`` or ``german`` are
    rejected so the interface stays unambiguous. When a rejected code is a
    recognised alias, the error points at the canonical code to use instead.
    (The alias table is still used internally to *derive* each task's language
    from the benchmark's own spelling; it just isn't accepted as user input.)
    """
    valid = set(get_all_language_codes())
    requested: list[str] = []
    unknown: list[str] = []
    for code in languages:
        raw = str(code).strip()
        if not raw:
            continue
        if raw in valid:
            if raw not in requested:
                requested.append(raw)
            continue
        canon = _canonical_language(raw)
        if canon and canon in valid and canon != raw:
            unknown.append(f"{raw} (use {canon})")
        else:
            unknown.append(raw)
    if unknown:
        raise ValueError(
            f"Unknown language code(s): {', '.join(unknown)}. "
            f"Use the precise lang_Scri form. Valid codes: {', '.join(sorted(valid))}"
        )
    return requested


_GROUP_SPEC = re.compile(r"^(?P<name>[^\[\]]+?)(?:\[(?P<langs>[^\[\]]*)\])?$")


def split_group_tokens(raw: str) -> list[str]:
    """Split a ``--task_groups`` string on top-level commas only.

    Commas inside a per-group ``[...]`` language bracket are preserved so that
    ``sib200-eu[fra_Latn,deu_Latn],flores200`` splits into two tokens, not
    three. Blank tokens are dropped.
    """
    tokens: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in raw:
        if ch == "[":
            depth += 1
            current.append(ch)
        elif ch == "]":
            depth = max(0, depth - 1)
            current.append(ch)
        elif ch == "," and depth == 0:
            tokens.append("".join(current))
            current = []
        else:
            current.append(ch)
    tokens.append("".join(current))
    return [t.strip() for t in tokens if t.strip()]


def _parse_group_spec(token: str) -> tuple[str, list[str] | None]:
    """Split a group token into ``(name, per_group_languages_or_None)``.

    ``sib200-eu`` -> ``("sib200-eu", None)``; ``sib200-eu[fra_Latn|deu_Latn]``
    -> ``("sib200-eu", ["fra_Latn", "deu_Latn"])``. Languages inside the
    bracket may be separated by ``,`` or ``|``. An empty bracket is rejected.
    """
    match = _GROUP_SPEC.match(token.strip())
    if not match:
        raise ValueError(f"Malformed task group spec: {token!r}")
    name = match.group("name").strip()
    langs_raw = match.group("langs")
    if langs_raw is None:
        return name, None
    langs = [part.strip() for part in re.split(r"[,|]", langs_raw) if part.strip()]
    if not langs:
        raise ValueError(f"Empty language bracket in task group spec: {token!r}")
    return name, langs


def _resolve_group_specs(
    group_names: Iterable[str],
) -> list[tuple[str, list[str] | None]]:
    """Resolve requested group tokens to ``(name, language_filter_or_None)``.

    A per-group ``[...]`` bracket scopes that group (or super_group) to the
    given language code(s); a bare token applies no language filter.
    """
    specs: list[tuple[str, list[str] | None]] = []
    for token in (str(n).strip() for n in group_names if str(n).strip()):
        name, per_langs = _parse_group_spec(token)
        if per_langs is not None:
            specs.append((name, _normalise_language_codes(per_langs)))
        else:
            specs.append((name, None))
    return specs


def _select_tasks(group_names: Iterable[str]) -> list[tuple[str, _Task]]:
    """Resolve requested groups to ``(suite, task)`` pairs.

    A per-group ``[...]`` language bracket keeps only the tasks in that group
    (or super_group) that resolve to one of the bracketed languages, and
    hard-errors if it matches nothing in that group.
    """
    specs = _resolve_group_specs(group_names)

    parsed = _parse_task_groups([name for name, _ in specs])
    missing = {name for name, _ in specs} - set(parsed.keys())
    if missing:
        raise ValueError(f"Unknown task group(s): {', '.join(sorted(missing))}")

    selected: list[tuple[str, _Task]] = []
    seen: set[tuple[str, str]] = set()
    for name, filt in specs:
        group_pairs = list(_iter_group_tasks({name: parsed[name]}))
        if filt is None:
            kept = group_pairs
        else:
            kept = [(s, t) for s, t in group_pairs if set(t.languages) & set(filt)]
            if not kept:
                raise ValueError(
                    f"No tasks in task group '{name}' match language(s) "
                    f"{{{', '.join(filt)}}}."
                )
            matched = {lang for _s, t in kept for lang in t.languages if lang in filt}
            unmatched = [lang for lang in filt if lang not in matched]
            if unmatched:
                logging.warning(
                    "No tasks matched language(s) %s in group '%s'; kept %s.",
                    ", ".join(unmatched),
                    name,
                    ", ".join(lang for lang in filt if lang in matched),
                )
        # De-duplicate tasks shared by several groups (e.g. the `all` super_group
        # spans groups whose benchmarks overlap), so they are scheduled once.
        for suite, t in kept:
            key = (suite, t.name)
            if key not in seen:
                seen.add(key)
                selected.append((suite, t))

    return selected


def _expand_task_groups(group_names: Iterable[str]) -> list[TaskGroupResult]:
    results: list[TaskGroupResult] = []
    for suite, t in _select_tasks(group_names):
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
    # Merge specs sharing (repo_id, subset): union their revisions and OR their
    # snapshot flag. A task whose resolved suite is a media suite (see
    # _SNAPSHOT_SUITES) is staged via snapshot_download rather than an
    # OOM-prone load_dataset() build on the memory-capped login node.
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

    for suite, t in _select_tasks(group_names):
        needs_snapshot = suite in _SNAPSHOT_SUITES
        if t.dataset in _URL_BASED_DATASETS:
            # URL-based dataset: a snapshot would grab only links. Force the
            # load_dataset()+materialize path so the media is actually fetched.
            needs_snapshot = False

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
    repos: list[str] = []
    seen: set[str] = set()
    for _suite, t in _select_tasks(group_names):
        for repo_id in t.hf_models or []:
            if repo_id not in seen:
                seen.add(repo_id)
                repos.append(repo_id)
    return repos


def _collect_hf_dataset_files(group_names: Iterable[str]) -> list[dict]:
    """Return deduplicated HF dataset file specs declared in task ``hf_dataset_files`` fields."""
    # Merge patterns from all tasks that share the same (repo_id, revision)
    # so that a single snapshot_download fetches everything needed.
    merged: dict[tuple[str, str | None], list[str]] = {}
    for _suite, t in _select_tasks(group_names):
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
    """Build a mapping from task names to their dataset specs from all task groups."""
    data = _load_task_groups_data()

    all_group_names = list(data.get("task_groups", {}).keys())
    parsed = _parse_task_groups(all_group_names)

    task_map: dict[str, list[DatasetSpec]] = {}

    for _, group in parsed.items():
        if isinstance(group, TaskGroup):
            needs_snapshot = group.suite in _SNAPSHOT_SUITES
            for t in group.tasks:
                if t.dataset and t.name not in task_map:
                    snap = needs_snapshot and t.dataset not in _URL_BASED_DATASETS
                    if t.dataset == "facebook/flores" and not t.subset:
                        task_map[t.name] = [
                            DatasetSpec(
                                repo_id=t.dataset, subset=lang, revisions=t.revisions
                            )
                            for lang in _extract_flores_subsets(t.name)
                        ]
                    else:
                        task_map[t.name] = [
                            DatasetSpec(
                                repo_id=t.dataset,
                                subset=t.subset,
                                needs_snapshot_download=snap,
                                revisions=t.revisions,
                            )
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


def _build_task_aux_map() -> dict[str, tuple[list[str], list[dict]]]:
    """Map task name → (hf_models, hf_dataset_files) from all task groups."""
    data = _load_task_groups_data()
    parsed = _parse_task_groups(list(data.get("task_groups", {}).keys()))
    aux: dict[str, tuple[list[str], list[dict]]] = {}
    for _, group in parsed.items():
        if isinstance(group, TaskGroup):
            for t in group.tasks:
                if t.name not in aux and (t.hf_models or t.hf_dataset_files):
                    aux[t.name] = (t.hf_models or [], t.hf_dataset_files or [])
    return aux


def _lookup_hf_model_repos_for_tasks(task_names: Iterable[str]) -> list[str]:
    """``hf_models`` pre-download repos for individual task names.

    Mirrors :func:`_collect_hf_model_repos` for tasks scheduled without a
    task group (bare ``--tasks`` or a ``--check`` re-schedule CSV) — those
    paths previously skipped auxiliary-model staging entirely, so contrib
    rows failed on the air-gapped compute node.
    """
    aux = _build_task_aux_map()
    repos: list[str] = []
    seen: set[str] = set()
    for name in task_names:
        models, _files = aux.get(str(name).strip(), ([], []))
        for repo_id in models:
            if repo_id not in seen:
                seen.add(repo_id)
                repos.append(repo_id)
    return repos


def _lookup_hf_dataset_files_for_tasks(task_names: Iterable[str]) -> list[dict]:
    """``hf_dataset_files`` specs for individual task names, merged per
    (repo_id, revision) like :func:`_collect_hf_dataset_files`."""
    aux = _build_task_aux_map()
    merged: dict[tuple[str, str | None], list[str]] = {}
    for name in task_names:
        _models, file_specs = aux.get(str(name).strip(), ([], []))
        for spec in file_specs:
            repo_id = spec.get("repo_id", "")
            if not repo_id:
                continue
            pats = merged.setdefault((repo_id, spec.get("revision")), [])
            for p in spec.get("patterns") or []:
                if p not in pats:
                    pats.append(p)
    result = []
    for (rid, rev), pats in merged.items():
        entry: dict = {"repo_id": rid, "patterns": pats}
        if rev:
            entry["revision"] = rev
        result.append(entry)
    return result


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


def get_all_language_codes() -> list[str]:
    """Return all language codes accepted in a ``group[lang]`` bracket."""
    data = _load_task_groups_data()
    task_groups = {
        name: TaskGroup.from_dict(name, task_data)
        for name, task_data in data.get("task_groups", {}).items()
    }
    return sorted(_language_codes_from_groups(task_groups))
