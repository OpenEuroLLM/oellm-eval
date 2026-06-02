"""Tests for auto-derived per-language task groups (e.g. --task_groups deu_Latn).

Languages are derived in code from each task's ``{lang}`` template expansion or
``subset`` (see oellm/task_groups.py), so these groups require no YAML tagging.
"""

from importlib.resources import files

import yaml

from oellm.task_groups import (
    _collect_dataset_specs,
    _expand_task_groups,
    _load_task_groups_data,
    _resolve_task_languages,
    get_all_language_codes,
)

# Multilingual groups still defined with explicit per-language task lists
# (not {lang} templates) whose tasks must also resolve to a language.
EXPLICIT_MULTILINGUAL_GROUPS = ["mgsm-eu", "include"]


def _raw_yaml() -> dict:
    return (
        yaml.safe_load((files("oellm.resources") / "task-groups.yaml").read_text()) or {}
    )


def test_language_codes_available():
    codes = get_all_language_codes()
    assert len(codes) >= 30
    for expected in ["deu_Latn", "fra_Latn", "ita_Latn", "spa_Latn", "por_Latn"]:
        assert expected in codes


def test_language_group_expands_and_mixes_suites():
    jobs = _expand_task_groups(["deu_Latn"])
    assert len(jobs) >= 10
    # German spans both evaluation suites (belebele-cf + flores are lighteval).
    suites = {j.suite for j in jobs}
    assert "lm-eval-harness" in suites
    assert "lighteval" in suites
    assert all(isinstance(j.n_shot, int) for j in jobs)


def test_mgsm_gap_is_handled():
    """Italian/Portuguese lack mgsm; their groups should still resolve (no crash)
    and simply omit the mgsm task rather than fail."""
    for lang in ["ita_Latn", "por_Latn"]:
        tasks = {j.task for j in _expand_task_groups([lang])}
        assert tasks
        assert not any(t.startswith("mgsm") for t in tasks)


def test_language_group_collects_dataset_specs():
    specs = _collect_dataset_specs(["deu_Latn"])
    assert specs
    assert "facebook/belebele" in {s.repo_id for s in specs}


def test_unknown_language_code_rejected():
    import pytest

    with pytest.raises(ValueError):
        _expand_task_groups(["zzz_Fake"])


def test_templated_tasks_all_resolve_to_a_language():
    """Every task in a group that uses `valid_langs` templating, plus the
    explicit multilingual groups, must resolve to a language code. Guards
    against a new language spelling that the normaliser doesn't recognise."""
    raw = _raw_yaml()["task_groups"]
    templated = [name for name, g in raw.items() if g.get("valid_langs")]
    assert templated, "expected at least one {lang}-templated group"

    expanded = _load_task_groups_data()["task_groups"]
    for name in templated + EXPLICIT_MULTILINGUAL_GROUPS:
        for task in expanded[name]["tasks"]:
            langs = _resolve_task_languages(task["task"], task.get("subset"))
            assert langs, (
                f"{name}: task {task['task']} (subset={task.get('subset')}) "
                "did not resolve to a language"
            )
