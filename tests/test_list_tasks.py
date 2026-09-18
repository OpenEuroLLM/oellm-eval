"""Tests for the list_tasks() CLI subcommand."""

from __future__ import annotations

import io
from unittest.mock import patch

from rich.console import Console

from oellm.main import list_tasks


def _capture_list_tasks(**kwargs) -> str:
    """Run list_tasks() and return printed output as a string."""
    buf = io.StringIO()
    console = Console(file=buf, highlight=False, markup=False)
    with patch("oellm.utils._RICH_CONSOLE", console):
        list_tasks(**kwargs)
    return buf.getvalue()


def test_list_tasks_runs_without_error() -> None:
    # Should not raise
    _capture_list_tasks()


def test_list_tasks_includes_core_group() -> None:
    output = _capture_list_tasks()
    assert "open-sci-0.01" in output


def test_list_tasks_includes_image_group() -> None:
    output = _capture_list_tasks()
    assert "image-vqa" in output


def test_list_tasks_includes_contrib_group() -> None:
    output = _capture_list_tasks()
    # regiondial_bench or regiondial-bench must appear
    assert "regiondial" in output.lower()


def test_list_tasks_has_suite_column() -> None:
    output = _capture_list_tasks()
    assert "Suite" in output


def test_list_tasks_has_nshots_column() -> None:
    output = _capture_list_tasks()
    assert "N-shots" in output


def test_list_tasks_has_tasks_column() -> None:
    output = _capture_list_tasks()
    assert "Tasks" in output
