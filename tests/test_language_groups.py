"""Tests for per-group language brackets (e.g. ``sib200-eu[deu_Latn]``).

Languages are derived in code from each task's ``{lang}`` template expansion or
``subset`` (see oellm/task_groups.py), so they require no YAML tagging. A
``group[lang]`` (or ``super_group[lang]``) bracket keeps only the tasks in that
group that resolve to one of the bracketed languages; codes inside the bracket
may be separated by ``,`` or ``|``.
"""

from importlib.resources import files

import pytest
import yaml

from oellm.task_groups import (
    _collect_dataset_specs,
    _expand_task_groups,
    _load_task_groups_data,
    _resolve_task_languages,
    get_all_language_codes,
    split_group_tokens,
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


def test_bracket_scopes_language_to_one_group():
    jobs = _expand_task_groups(["sib200-eu[fra_Latn]"])
    assert {j.task for j in jobs} == {"sib200_fra_Latn"}


def test_bracket_allows_per_group_languages():
    jobs = _expand_task_groups(["sib200-eu[fra_Latn]", "flores-200-eu-to-eng[deu_Latn]"])
    assert {j.task for j in jobs} == {
        "sib200_fra_Latn",
        "flores200:deu_Latn-eng_Latn",
    }


def test_bracket_accepts_comma_or_pipe_separator():
    comma = {j.task for j in _expand_task_groups(["sib200-eu[fra_Latn,deu_Latn]"])}
    pipe = {j.task for j in _expand_task_groups(["sib200-eu[fra_Latn|deu_Latn]"])}
    assert comma == pipe == {"sib200_fra_Latn", "sib200_deu_Latn"}


def test_split_group_tokens_is_bracket_aware():
    assert split_group_tokens("a[x,y],b") == ["a[x,y]", "b"]
    assert split_group_tokens("sib200-eu , flores200") == ["sib200-eu", "flores200"]
    assert split_group_tokens("g[x|y]") == ["g[x|y]"]


def test_language_codes_are_not_task_groups():
    """A bare language code must not resolve as a task group (the old union
    footgun); it only has meaning inside a ``group[lang]`` bracket."""
    with pytest.raises(ValueError, match="Unknown task group"):
        _expand_task_groups(["deu_Latn"])


def test_bracket_empty_intersection_hard_errors():
    """flores-eu has no Ukrainian side -> an empty bracket match must raise."""
    with pytest.raises(ValueError, match="No tasks in task group 'flores"):
        _expand_task_groups(["flores-200-eu-to-eng[ukr_Cyrl]"])


def test_empty_bracket_rejected():
    with pytest.raises(ValueError, match="Empty language bracket"):
        _expand_task_groups(["sib200-eu[]"])


def test_unknown_language_code_rejected():
    with pytest.raises(ValueError, match="Unknown language code"):
        _expand_task_groups(["sib200-eu[zzz_Fake]"])


@pytest.mark.parametrize("loose", ["de", "german", "German", "deu_latn"])
def test_loose_spellings_rejected_with_canonical_hint(loose):
    """Only the precise lang_Scri code is accepted; aliases like `de`/`german`
    are rejected, and the error points at the canonical code to use instead."""
    with pytest.raises(ValueError) as excinfo:
        _expand_task_groups([f"sib200-eu[{loose}]"])
    assert "use deu_Latn" in str(excinfo.value)


def test_canonical_code_still_accepted():
    jobs = _expand_task_groups(["sib200-eu[deu_Latn]"])
    assert {j.task for j in jobs} == {"sib200_deu_Latn"}


# --- super_group support -----------------------------------------------------


def test_super_group_bracket_resolves_language_subset():
    """Sampo's request: scoping the oellm-multilingual super_group to a language
    selects the applicable subset across its member benchmarks, and spans both
    evaluation suites."""
    jobs = _expand_task_groups(["oellm-multilingual[deu_Latn]"])
    tasks = {j.task for j in jobs}
    assert tasks == {
        "belebele_deu_Latn",
        "flores200:deu_Latn-eng_Latn",
        "flores200:eng_Latn-deu_Latn",
        "global_mmlu_full_de",
        "include_base_44_german",
        "mgsm_native_cot_de",
    }
    suites = {j.suite for j in jobs}
    assert "lm-eval-harness" in suites
    assert "lighteval" in suites


def test_super_group_bracket_collects_dataset_specs():
    specs = _collect_dataset_specs(["oellm-multilingual[deu_Latn]"])
    assert specs
    repos = {s.repo_id for s in specs}
    assert "facebook/belebele" in repos
    assert "facebook/flores" in repos


def test_all_super_group_is_auto_generated():
    """The `all` super_group is synthesised from the registry (no YAML entry),
    so it spans every task group without hand-maintenance, and it is not itself
    listed as a plain task group."""
    from oellm.task_groups import get_all_task_group_names

    assert "all" not in get_all_task_group_names()
    # Reaching it at all proves the synthetic group exists.
    assert _expand_task_groups(["all"])


def test_all_super_group_deduplicates_overlapping_tasks():
    """Groups inside `all` share some benchmarks (e.g. copa, hellaswag); each
    (suite, task, n_shot) must be scheduled exactly once."""
    jobs = _expand_task_groups(["all"])
    keys = [(j.suite, j.task, j.n_shot) for j in jobs]
    assert len(keys) == len(set(keys))


def test_all_super_group_bracket_spans_whole_registry():
    """The `all` super_group + [lang] reaches every benchmark's tasks for that
    language (e.g. German tasks that live outside oellm-multilingual)."""
    tasks = {j.task for j in _expand_task_groups(["all[deu_Latn]"])}
    # tasks unique to groups not in oellm-multilingual
    for extra in [
        "sib200_deu_Latn",
        "arc_challenge_mt_de",
        "belebele_deu_Latn_cf",
        "global_piqa_completions_deu_latn",
        "global_piqa_prompted_deu_latn",
    ]:
        assert extra in tasks
    # and still a superset of the curated multilingual subset
    multiling = {j.task for j in _expand_task_groups(["oellm-multilingual[deu_Latn]"])}
    assert multiling <= tasks


def test_super_group_without_bracket_keeps_all_languages():
    scoped = _expand_task_groups(["oellm-multilingual[deu_Latn]"])
    full = _expand_task_groups(["oellm-multilingual"])
    assert len(full) > len(scoped)


def test_mgsm_gap_is_handled():
    """Italian/Portuguese lack mgsm; scoping the super_group to them should still
    resolve (no crash) and simply omit the mgsm task rather than fail."""
    for lang in ["ita_Latn", "por_Latn"]:
        tasks = {j.task for j in _expand_task_groups([f"oellm-multilingual[{lang}]"])}
        assert tasks
        assert not any(t.startswith("mgsm") for t in tasks)


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
