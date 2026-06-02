"""Tests for auto-derived per-language task groups (e.g. --task_groups deu_Latn)."""

from importlib.resources import files

import pytest
import yaml

from oellm.task_groups import (
    _collect_dataset_specs,
    _expand_task_groups,
    get_all_language_codes,
)

# Groups whose tasks are language-specific and must therefore all carry a
# `languages:` tag. Keep in sync with scripts/tag_languages.py.
MULTILINGUAL_GROUPS = {
    "sib200-eu",
    "belebele-eu-5-shot",
    "belebele-eu-cf",
    "flores-200-eu-to-eng",
    "flores-200-eng-to-eu",
    "global-mmlu-eu",
    "mgsm-eu",
    "arc-challenge-mt-eu",
    "include",
    "global-piqa-eu-completions",
    "global-piqa-eu-prompted",
}


def _load_yaml() -> dict:
    return (
        yaml.safe_load((files("oellm.resources") / "task-groups.yaml").read_text()) or {}
    )


@pytest.mark.parametrize("group_name", sorted(MULTILINGUAL_GROUPS))
def test_every_multilingual_task_is_tagged(group_name):
    """Every task in a multilingual group must carry a `languages` tag, so that
    coverage of the derived language groups cannot silently regress."""
    group = _load_yaml()["task_groups"][group_name]
    untagged = [t["task"] for t in group["tasks"] if not t.get("languages")]
    assert not untagged, f"{group_name} has untagged tasks: {untagged}"


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
    # n_shot must be resolved to ints for every job.
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
    repos = {s.repo_id for s in specs}
    assert "facebook/belebele" in repos
