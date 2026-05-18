"""Tests for collect_results – mirrors the output/ folder layout used in practice.

Typical on-disk structure that these tests reproduce:

    output_root/
        hellaswag_mt1/
            jobs.csv
            results/
                <hash>_<timestamp>.json
        hellaswag_mt2/
            jobs.csv
            results/
                <hash>_<timestamp>.json
        global_mmlu1/
            jobs.csv
            results/
                <hash>_<timestamp>.json
        2026-04-28-no-results/
            jobs.csv            ← jobs.csv present but no results dir yet
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from oellm.main import collect_results

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MODEL_A = "/cache/hub/model-a/snapshots/aaa"
MODEL_B = "/cache/hub/model-b/snapshots/bbb"


def _write_jobs_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _lmeval_result(model_name: str, task: str, score: float, n_shot: int = 10) -> dict:
    """Minimal lm-eval result JSON matching the real schema."""
    return {
        "model_name": model_name,
        "results": {
            task: {
                "alias": task,
                "acc,none": score,
                "acc_stderr,none": 0.005,
            }
        },
        "group_subtasks": {task: []},
        "n-shot": {task: n_shot},
    }


def _write_result_json(results_dir: Path, model_name: str, task: str, score: float, n_shot: int = 10) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    safe = model_name.replace("/", "_")
    (results_dir / f"{safe}_{task}.json").write_text(
        json.dumps(_lmeval_result(model_name, task, score, n_shot))
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def output_root(tmp_path: Path) -> Path:
    """
    Builds a directory tree that mirrors the real output/ folder:

        hellaswag_mt1/  – 2 models × 1 task each, both results present
        hellaswag_mt2/  – 1 model × 1 task, result present
        global_mmlu1/   – 1 model × 1 task, result present
        2026-04-28/     – jobs.csv only, no results directory yet
    """
    root = tmp_path / "output"

    # hellaswag_mt1 --------------------------------------------------------
    mt1 = root / "hellaswag_mt1"
    _write_jobs_csv(mt1 / "jobs.csv", [
        {"model_path": MODEL_A, "task_path": "hellaswag_da", "n_shot": 10, "eval_suite": "lm-eval-harness"},
        {"model_path": MODEL_B, "task_path": "hellaswag_nl", "n_shot": 10, "eval_suite": "lm-eval-harness"},
    ])
    _write_result_json(mt1 / "results", MODEL_A, "hellaswag_da", 0.63)
    _write_result_json(mt1 / "results", MODEL_B, "hellaswag_nl", 0.61)

    # hellaswag_mt2 --------------------------------------------------------
    mt2 = root / "hellaswag_mt2"
    _write_jobs_csv(mt2 / "jobs.csv", [
        {"model_path": MODEL_A, "task_path": "hellaswag_fr", "n_shot": 10, "eval_suite": "lm-eval-harness"},
    ])
    _write_result_json(mt2 / "results", MODEL_A, "hellaswag_fr", 0.59)

    # global_mmlu1 ---------------------------------------------------------
    mmlu1 = root / "global_mmlu1"
    _write_jobs_csv(mmlu1 / "jobs.csv", [
        {"model_path": MODEL_A, "task_path": "global_mmlu_full_de", "n_shot": 5, "eval_suite": "lm-eval-harness"},
        {"model_path": MODEL_B, "task_path": "global_mmlu_full_en", "n_shot": 5, "eval_suite": "lm-eval-harness"},
    ])
    _write_result_json(mmlu1 / "results", MODEL_A, "global_mmlu_full_de", 0.52, n_shot=5)
    _write_result_json(mmlu1 / "results", MODEL_B, "global_mmlu_full_en", 0.55, n_shot=5)

    # 2026-04-28 – has jobs.csv but no results yet -------------------------
    pending = root / "2026-04-28"
    _write_jobs_csv(pending / "jobs.csv", [
        {"model_path": MODEL_A, "task_path": "hellaswag_sr", "n_shot": 10, "eval_suite": "lm-eval-harness"},
    ])

    return root


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCollectResultsMerge:
    """Results are gathered from all sub-directories and merged."""

    def test_output_csv_created(self, output_root, tmp_path):
        out_csv = str(tmp_path / "results.csv")
        collect_results(str(output_root), output_csv=out_csv)
        assert Path(out_csv).exists()

    def test_all_results_present(self, output_root, tmp_path):
        out_csv = str(tmp_path / "results.csv")
        collect_results(str(output_root), output_csv=out_csv)
        df = pd.read_csv(out_csv)
        # 5 result JSONs were written across all subdirs
        assert len(df) == 5

    def test_correct_tasks_extracted(self, output_root, tmp_path):
        out_csv = str(tmp_path / "results.csv")
        collect_results(str(output_root), output_csv=out_csv)
        df = pd.read_csv(out_csv)
        tasks = set(df["task"])
        assert tasks == {"hellaswag_da", "hellaswag_nl", "hellaswag_fr", "global_mmlu_full_de", "global_mmlu_full_en"}

    def test_correct_scores(self, output_root, tmp_path):
        out_csv = str(tmp_path / "results.csv")
        collect_results(str(output_root), output_csv=out_csv)
        df = pd.read_csv(out_csv)
        row = df[df["task"] == "hellaswag_da"].iloc[0]
        assert abs(row["performance"] - 0.63) < 1e-6

    def test_n_shot_preserved(self, output_root, tmp_path):
        out_csv = str(tmp_path / "results.csv")
        collect_results(str(output_root), output_csv=out_csv)
        df = pd.read_csv(out_csv)
        assert set(df[df["task"] == "global_mmlu_full_de"]["n_shot"]) == {5}


class TestCollectResultsDuplicateOverride:
    """When the same (model_path, task_path, n_shot) row appears in multiple
    jobs.csv files the last-sorted entry wins."""

    def test_duplicate_jobs_deduplicated(self, tmp_path):
        root = tmp_path / "output"

        # Two subdirectories both schedule the same job
        for subdir_name in ["2026-05-01", "2026-05-02"]:
            sub = root / subdir_name
            _write_jobs_csv(sub / "jobs.csv", [
                {"model_path": MODEL_A, "task_path": "hellaswag_da", "n_shot": 10, "eval_suite": "lm-eval-harness"},
            ])

        _write_result_json(root / "2026-05-02" / "results", MODEL_A, "hellaswag_da", 0.65)

        out_csv = str(tmp_path / "results.csv")
        collect_results(str(root), output_csv=out_csv, check=True)

        # Even though two jobs.csv files declare the same job, the merged jobs
        # table should contain exactly one row for that combination.
        missing_csv = out_csv.replace(".csv", "_missing.csv")
        assert not Path(missing_csv).exists(), "Job counted as missing despite result being present"

    def test_result_duplicate_rows_not_doubled(self, tmp_path):
        """If the same JSON appears (e.g. symlink / copy), the output still has 1 row per result file."""
        root = tmp_path / "output"
        sub = root / "hellaswag_mt1"
        _write_jobs_csv(sub / "jobs.csv", [
            {"model_path": MODEL_A, "task_path": "hellaswag_da", "n_shot": 10, "eval_suite": "lm-eval-harness"},
        ])
        _write_result_json(sub / "results", MODEL_A, "hellaswag_da", 0.63)

        out_csv = str(tmp_path / "results.csv")
        collect_results(str(root), output_csv=out_csv)
        df = pd.read_csv(out_csv)
        assert len(df) == 1


class TestCollectResultsCheckMode:
    """--check compares merged results against merged jobs and writes _missing.csv."""

    def test_all_complete_no_missing_csv(self, output_root, tmp_path):
        """All jobs in `output_root` except the pending dir have results already."""
        # The fixture has hellaswag_sr scheduled but no result JSON
        # Remove the pending jobs.csv to get a fully-complete run
        (output_root / "2026-04-28" / "jobs.csv").unlink()

        out_csv = str(tmp_path / "results.csv")
        collect_results(str(output_root), output_csv=out_csv, check=True)

        missing_csv = out_csv.replace(".csv", "_missing.csv")
        assert not Path(missing_csv).exists()

    def test_missing_jobs_written_to_csv(self, output_root, tmp_path):
        """hellaswag_sr is in jobs.csv but has no result JSON → appears in missing CSV."""
        out_csv = str(tmp_path / "results.csv")
        collect_results(str(output_root), output_csv=out_csv, check=True)

        missing_csv = out_csv.replace(".csv", "_missing.csv")
        assert Path(missing_csv).exists()
        missing_df = pd.read_csv(missing_csv)
        assert len(missing_df) >= 1
        assert "hellaswag_sr" in missing_df["task_path"].values

    def test_missing_csv_columns(self, output_root, tmp_path):
        out_csv = str(tmp_path / "results.csv")
        collect_results(str(output_root), output_csv=out_csv, check=True)

        missing_csv = out_csv.replace(".csv", "_missing.csv")
        missing_df = pd.read_csv(missing_csv)
        for col in ("model_path", "task_path", "n_shot"):
            assert col in missing_df.columns

    def test_check_without_any_jobs_csv(self, tmp_path):
        """If no jobs.csv exists anywhere, check mode is silently disabled."""
        root = tmp_path / "output"
        sub = root / "hellaswag_mt1"
        _write_result_json(sub / "results", MODEL_A, "hellaswag_da", 0.63)

        out_csv = str(tmp_path / "results.csv")
        # Should not raise even though check=True and no jobs.csv exists
        collect_results(str(root), output_csv=out_csv, check=True)

        missing_csv = out_csv.replace(".csv", "_missing.csv")
        assert not Path(missing_csv).exists()


class TestCollectResultsEdgeCases:
    def test_no_json_files_returns_without_output(self, tmp_path):
        root = tmp_path / "output"
        sub = root / "2026-04-28"
        _write_jobs_csv(sub / "jobs.csv", [
            {"model_path": MODEL_A, "task_path": "hellaswag_da", "n_shot": 10, "eval_suite": "lm-eval-harness"},
        ])
        out_csv = str(tmp_path / "results.csv")
        collect_results(str(root), output_csv=out_csv)
        assert not Path(out_csv).exists()

    def test_nonexistent_results_dir_raises(self, tmp_path):
        with pytest.raises(ValueError, match="does not exist"):
            collect_results(str(tmp_path / "nonexistent"))

    def test_groups_json_extracted(self, tmp_path):
        """JSON files that use the 'groups' key (lm-eval aggregate) are handled."""
        root = tmp_path / "output"
        sub = root / "hellaswag_mt1"
        _write_jobs_csv(sub / "jobs.csv", [
            {"model_path": MODEL_A, "task_path": "hellaswag_da", "n_shot": 10, "eval_suite": "lm-eval-harness"},
        ])

        results_dir = sub / "results"
        results_dir.mkdir(parents=True)
        group_json = {
            "model_name": MODEL_A,
            "results": {},
            "groups": {
                "hellaswag_da": {"acc,none": 0.70, "acc_stderr,none": 0.01}
            },
            "group_subtasks": {"hellaswag_da": []},
            "n-shot": {"hellaswag_da": 10},
        }
        (results_dir / "group_result.json").write_text(json.dumps(group_json))

        out_csv = str(tmp_path / "results.csv")
        collect_results(str(root), output_csv=out_csv)
        df = pd.read_csv(out_csv)
        assert len(df) == 1
        assert abs(df.iloc[0]["performance"] - 0.70) < 1e-6
