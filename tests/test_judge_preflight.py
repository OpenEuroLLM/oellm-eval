"""Tests for the OPENAI_API_KEY pre-flight check in ``check_judge_llm_pre_flight``."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from oellm.constants import JUDGE_REQUIRED_TASKS
from oellm.utils import check_judge_llm_pre_flight


def test_passes_when_no_judge_required_task() -> None:
    """Non-judge tasks must not trigger the check regardless of API key state."""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("OPENAI_API_KEY", None)
        check_judge_llm_pre_flight(["vqav2_val", "mmmu_val", "chartqa"])


def test_passes_when_openai_api_key_set() -> None:
    """A set ``OPENAI_API_KEY`` (any non-empty value) satisfies the check."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
        check_judge_llm_pre_flight(["activitynetqa", "alpaca_audio"])


def test_passes_with_allow_missing_judge() -> None:
    """``allow_missing=True`` lets judge-required tasks through with a warning."""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("OPENAI_API_KEY", None)
        check_judge_llm_pre_flight(["activitynetqa", "alpaca_audio"], allow_missing=True)


def test_refuses_when_judge_required_and_no_key() -> None:
    """Default-strict: judge-required task without key must abort."""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("OPENAI_API_KEY", None)
        with pytest.raises(SystemExit) as exc:
            check_judge_llm_pre_flight(["activitynetqa", "vqav2_val"])
        # Error message must name the offending task and OPENAI_API_KEY.
        assert "activitynetqa" in str(exc.value)
        assert "OPENAI_API_KEY" in str(exc.value)
        assert "allow-missing-judge" in str(exc.value)


def test_refuses_only_lists_judge_required_tasks() -> None:
    """The error message must list ONLY the judge-required tasks, not all tasks."""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("OPENAI_API_KEY", None)
        with pytest.raises(SystemExit) as exc:
            check_judge_llm_pre_flight(
                ["vqav2_val", "alpaca_audio", "chartqa", "wavcaps"]
            )
        msg = str(exc.value)
        # Judge-required: present
        assert "alpaca_audio" in msg
        assert "wavcaps" in msg
        # Non-judge: NOT present in the offending list
        assert "vqav2_val" not in msg
        assert "chartqa" not in msg


def test_empty_task_list_passes() -> None:
    """No tasks → nothing to check."""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("OPENAI_API_KEY", None)
        check_judge_llm_pre_flight([])


def test_judge_required_tasks_set_is_non_empty() -> None:
    """Guard against accidentally emptying the set."""
    assert len(JUDGE_REQUIRED_TASKS) > 0
    assert "activitynetqa" in JUDGE_REQUIRED_TASKS
