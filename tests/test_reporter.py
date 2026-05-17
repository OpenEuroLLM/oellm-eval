"""Tests for oellm/reporter.py."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from oellm.results import SCHEMA_VERSION, write_results_json, write_results_markdown

_SAMPLE_ROWS = [
    {
        "model_name": "/models/llava",
        "task": "vqav2",
        "n_shot": 0,
        "performance": 0.75,
        "metric_name": "vqa_score",
    },
    {
        "model_name": "/models/llava",
        "task": "mmmu",
        "n_shot": 0,
        "performance": 0.55,
        "metric_name": "acc",
    },
]


def test_write_json_schema_version(tmp_path: Path) -> None:
    out = tmp_path / "results.json"
    write_results_json(_SAMPLE_ROWS, out)
    data = json.loads(out.read_text())
    assert data["version"] == SCHEMA_VERSION == "1.1"


def test_write_json_result_fields(tmp_path: Path) -> None:
    out = tmp_path / "results.json"
    write_results_json(_SAMPLE_ROWS, out)
    data = json.loads(out.read_text())
    assert len(data["results"]) == 2
    for r in data["results"]:
        assert set(r.keys()) == {
            "model",
            "task",
            "n_shot",
            "metric",
            "performance",
            "performance_normalized",
        }


def test_write_json_result_values(tmp_path: Path) -> None:
    out = tmp_path / "results.json"
    write_results_json(_SAMPLE_ROWS, out)
    data = json.loads(out.read_text())
    first = data["results"][0]
    assert first["model"] == "/models/llava"
    assert first["task"] == "vqav2"
    assert first["n_shot"] == 0
    assert first["metric"] == "vqa_score"
    assert first["performance"] == pytest.approx(0.75)


def test_write_json_generated_at_is_iso8601(tmp_path: Path) -> None:
    out = tmp_path / "results.json"
    write_results_json(_SAMPLE_ROWS, out)
    data = json.loads(out.read_text())
    # Should parse without error
    dt = datetime.fromisoformat(data["generated_at"])
    assert dt.tzinfo is not None  # timezone-aware


def test_write_json_empty_rows(tmp_path: Path) -> None:
    out = tmp_path / "results.json"
    write_results_json([], out)
    data = json.loads(out.read_text())
    assert data["results"] == []
    assert data["version"] == "1.1"


def test_write_json_creates_parent_dirs(tmp_path: Path) -> None:
    out = tmp_path / "subdir" / "nested" / "results.json"
    assert not out.parent.exists()
    write_results_json(_SAMPLE_ROWS, out)
    assert out.exists()


def test_write_json_missing_metric_name(tmp_path: Path) -> None:
    rows = [{"model_name": "m", "task": "t", "n_shot": 0, "performance": 0.5}]
    out = tmp_path / "results.json"
    write_results_json(rows, out)
    data = json.loads(out.read_text())
    assert data["results"][0]["metric"] == ""


def test_write_markdown_header(tmp_path: Path) -> None:
    out = tmp_path / "results.md"
    write_results_markdown(_SAMPLE_ROWS, out)
    text = out.read_text()
    assert "| Model | Task | N-shot | Metric | Performance (0–100) |" in text
    assert "|-------|------|--------|--------|---------------------|" in text


def test_write_markdown_data_row(tmp_path: Path) -> None:
    """When ``performance_normalized`` is missing on the row dict (legacy
    callers), the Markdown falls back to the raw value with a ``*`` suffix
    so the reader knows the scale is not registered."""
    out = tmp_path / "results.md"
    write_results_markdown(_SAMPLE_ROWS, out)
    text = out.read_text()
    assert "0.7500*" in text
    assert "vqav2" in text


def test_write_markdown_uses_normalized_when_present(tmp_path: Path) -> None:
    out = tmp_path / "results.md"
    rows = [
        {
            "model_name": "/models/llava",
            "task": "vqav2",
            "n_shot": 0,
            "performance": 0.755,
            "performance_normalized": 75.5,
            "metric_name": "exact_match",
        },
        {
            "model_name": "/models/llava",
            "task": "mvbench",
            "n_shot": 0,
            "performance": 56.2,
            "performance_normalized": 56.2,
            "metric_name": "mvbench_accuracy",
        },
    ]
    write_results_markdown(rows, out)
    text = out.read_text()
    assert "| 75.50 |" in text
    assert "| 56.20 |" in text
    # No `*` suffix when normalized value is present.
    assert "75.50*" not in text
    assert "56.20*" not in text


def test_write_markdown_creates_parent_dirs(tmp_path: Path) -> None:
    out = tmp_path / "reports" / "results.md"
    write_results_markdown(_SAMPLE_ROWS, out)
    assert out.exists()


def test_write_markdown_empty_rows(tmp_path: Path) -> None:
    out = tmp_path / "results.md"
    write_results_markdown([], out)
    text = out.read_text()
    assert "| Model |" in text  # header still present
