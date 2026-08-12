"""Tests for the compare() CLI subcommand."""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from rich.console import Console

from oellm.main import compare


def _make_results_json(path: Path, results: list[dict]) -> Path:
    payload = {
        "version": "1.0",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "results": results,
    }
    path.write_text(json.dumps(payload))
    return path


def _capture_compare(result_a: str, result_b: str, **kwargs) -> str:
    buf = io.StringIO()
    console = Console(file=buf, highlight=False, markup=False)
    with patch("oellm.utils._RICH_CONSOLE", console):
        compare(result_a, result_b, **kwargs)
    return buf.getvalue()


def test_compare_two_json_files(tmp_path: Path) -> None:
    a = _make_results_json(
        tmp_path / "a.json",
        [
            {
                "model": "m",
                "task": "mmlu",
                "n_shot": 5,
                "metric": "acc",
                "performance": 0.70,
            },
        ],
    )
    b = _make_results_json(
        tmp_path / "b.json",
        [
            {
                "model": "m",
                "task": "mmlu",
                "n_shot": 5,
                "metric": "acc",
                "performance": 0.75,
            },
        ],
    )
    # Should not raise
    _capture_compare(str(a), str(b))


def test_compare_shows_task(tmp_path: Path) -> None:
    a = _make_results_json(
        tmp_path / "a.json",
        [
            {
                "model": "m",
                "task": "mmlu",
                "n_shot": 5,
                "metric": "acc",
                "performance": 0.70,
            },
        ],
    )
    b = _make_results_json(
        tmp_path / "b.json",
        [
            {
                "model": "m",
                "task": "mmlu",
                "n_shot": 5,
                "metric": "acc",
                "performance": 0.75,
            },
        ],
    )
    output = _capture_compare(str(a), str(b))
    assert "mmlu" in output


def test_compare_shows_delta(tmp_path: Path) -> None:
    a = _make_results_json(
        tmp_path / "a.json",
        [
            {
                "model": "m",
                "task": "mmlu",
                "n_shot": 5,
                "metric": "acc",
                "performance": 0.70,
            },
        ],
    )
    b = _make_results_json(
        tmp_path / "b.json",
        [
            {
                "model": "m",
                "task": "mmlu",
                "n_shot": 5,
                "metric": "acc",
                "performance": 0.75,
            },
        ],
    )
    output = _capture_compare(str(a), str(b))
    # Delta = 0.05 → "+0.0500"
    assert "0.0500" in output


def test_compare_accepts_directory(tmp_path: Path) -> None:
    dir_a = tmp_path / "run_a"
    dir_b = tmp_path / "run_b"
    dir_a.mkdir()
    dir_b.mkdir()
    _make_results_json(
        dir_a / "results.json",
        [
            {
                "model": "m",
                "task": "vqav2",
                "n_shot": 0,
                "metric": "vqa_score",
                "performance": 0.80,
            },
        ],
    )
    _make_results_json(
        dir_b / "results.json",
        [
            {
                "model": "m",
                "task": "vqav2",
                "n_shot": 0,
                "metric": "vqa_score",
                "performance": 0.82,
            },
        ],
    )
    output = _capture_compare(str(dir_a), str(dir_b))
    assert "vqav2" in output


def test_compare_missing_file_raises(tmp_path: Path) -> None:
    a = _make_results_json(tmp_path / "a.json", [])
    with pytest.raises(FileNotFoundError):
        compare(str(a), str(tmp_path / "nonexistent.json"))


def test_compare_missing_dir_results_json_raises(tmp_path: Path) -> None:
    dir_a = tmp_path / "run_a"
    dir_a.mkdir()
    # No results.json inside
    a = _make_results_json(tmp_path / "a.json", [])
    with pytest.raises(FileNotFoundError):
        compare(str(a), str(dir_a))


def test_compare_task_only_in_one_file(tmp_path: Path) -> None:
    a = _make_results_json(
        tmp_path / "a.json",
        [
            {
                "model": "m",
                "task": "mmlu",
                "n_shot": 5,
                "metric": "acc",
                "performance": 0.70,
            },
        ],
    )
    b = _make_results_json(
        tmp_path / "b.json",
        [
            {
                "model": "m",
                "task": "vqav2",
                "n_shot": 0,
                "metric": "vqa_score",
                "performance": 0.75,
            },
        ],
    )
    output = _capture_compare(str(a), str(b))
    assert "mmlu" in output
    assert "vqav2" in output
    # Em dash indicates missing value
    assert "\u2014" in output


def test_compare_empty_results(tmp_path: Path) -> None:
    a = _make_results_json(tmp_path / "a.json", [])
    b = _make_results_json(tmp_path / "b.json", [])
    # Should not raise, just print an empty table
    _capture_compare(str(a), str(b))
