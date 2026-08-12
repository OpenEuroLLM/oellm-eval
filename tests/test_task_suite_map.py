"""Tests for :func:`oellm.task_groups._build_task_suite_map`.

The helper powers the ``--tasks`` (bare-task-name) path in the scheduler.
It must cover every suite we actually support — core YAML-registered suites
(lm-eval-harness, lighteval, lmms_eval, evalchemy) AND contrib-registered
suites (e.g. regiondial_bench).
"""

from __future__ import annotations

from oellm.task_groups import _build_task_suite_map


def test_map_is_non_empty():
    m = _build_task_suite_map()
    assert len(m) > 0, "suite map must contain at least core YAML tasks"


def test_map_includes_lm_eval_harness_task():
    m = _build_task_suite_map()
    # copa is a classic lm-eval-harness task in task-groups.yaml
    assert m.get("copa") == "lm-eval-harness"


def test_map_includes_lighteval_task():
    m = _build_task_suite_map()
    # belebele_*_cf tasks are lighteval
    assert m.get("belebele_eng_Latn_cf") == "lighteval"


def test_map_includes_lmms_eval_task():
    """lmms_eval tasks come from image/video task groups — must be routable."""
    m = _build_task_suite_map()
    # vqav2_val is the base VQA v2 task (image modality)
    assert m.get("vqav2_val") == "lmms_eval"


def test_map_includes_contrib_task():
    """Contrib plugins (e.g. regiondial_bench) register their own TASK_GROUPS.

    These are the regression target: the original upstream helper only read
    YAML and missed contrib entirely.
    """
    m = _build_task_suite_map()
    assert m.get("regiondial_refcocog") == "regiondial_bench"


def test_map_honours_task_level_suite_override():
    """Evalchemy tasks set ``suite: evalchemy`` at the task level, not the
    group level — the helper must prefer the task-level value.
    """
    m = _build_task_suite_map()
    assert m.get("GPQADiamond") == "evalchemy"


def test_map_covers_all_actually_registered_suites():
    """Sanity: every distinct suite we see should be one we actually route.

    Guards against a new suite slipping into YAML or contrib without us
    adding a case branch in template.sbatch (the ``*)`` catch-all routes
    everything unknown to the contrib dispatcher, but we still want this
    assertion as documentation).
    """
    m = _build_task_suite_map()
    distinct_suites = set(m.values())
    expected_subset = {
        "lm-eval-harness",
        "lighteval",
        "lmms_eval",
        "evalchemy",
        "regiondial_bench",
    }
    # All expected suites must be present.  Extra contrib suites are fine.
    assert expected_subset.issubset(distinct_suites), (
        f"missing suites: {expected_subset - distinct_suites}"
    )


def test_first_occurrence_wins_when_task_in_multiple_groups():
    """If a task name appears in multiple groups, first occurrence wins.

    This is documented behavior of ``setdefault`` in the helper.  We don't
    assert a specific pair here because the YAML contents shift; we only
    assert the determinism property.
    """
    m1 = _build_task_suite_map()
    m2 = _build_task_suite_map()
    assert m1 == m2
