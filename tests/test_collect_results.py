"""Tests for the collect_results command: all-metrics output behaviour."""

import json
from pathlib import Path

import pandas as pd
import pytest

from oellm.main import collect_results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data))


def _read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Fixture: lm-eval harness result with multiple metrics per task
# ---------------------------------------------------------------------------

LM_EVAL_RESULT = {
    "model_name": "model-a",
    "results": {
        "hellaswag": {
            "acc,none": 0.45,
            "acc_norm,none": 0.63,
            "acc_stderr,none": 0.005,
            "acc_norm_stderr,none": 0.004,
        },
        "arc_challenge": {
            "acc,none": 0.38,
            "acc_norm,none": 0.41,
            "acc_norm_stderr,none": 0.010,
        },
    },
    "n-shot": {"hellaswag": 10, "arc_challenge": 10},
}

# Lighteval / flores200-style JSON
FLORES_RESULT = {
    "config_general": {"model_name": "model-b"},
    "results": {
        "flores200:eng_Latn-ita_Latn|0": {
            "chrf++": 57.676,
            "chrf++_stderr": 0.422,
            "bleu": 21.160,
            "bleu_stderr": 0.028,
            "bleu_1": 0.512,
            "bleu_4": 0.116,
        }
    },
    "n-shot": {},
}


# ---------------------------------------------------------------------------
# _extract_all_metrics unit-style tests (via collect_results output)
# ---------------------------------------------------------------------------

class TestAllMetricsExtracted:
    """All numeric metrics in a result dict appear as separate rows."""

    def test_multiple_metrics_per_task(self, tmp_path):
        """hellaswag with acc, acc_norm, acc_stderr, acc_norm_stderr → 4 rows."""
        _write_json(tmp_path / "result.json", LM_EVAL_RESULT)
        out = str(tmp_path / "out.csv")
        collect_results(str(tmp_path), out)
        df = _read_csv(out)
        hellaswag_rows = df[df["task"] == "hellaswag"]
        assert set(hellaswag_rows["metric_name"]) == {
            "acc", "acc_norm", "acc_stderr", "acc_norm_stderr"
        }

    def test_correct_values(self, tmp_path):
        """Metric values are preserved exactly."""
        _write_json(tmp_path / "result.json", LM_EVAL_RESULT)
        out = str(tmp_path / "out.csv")
        collect_results(str(tmp_path), out)
        df = _read_csv(out)
        row = df[(df["task"] == "hellaswag") & (df["metric_name"] == "acc_norm")]
        assert len(row) == 1
        assert float(row["performance"].iloc[0]) == pytest.approx(0.63)

    def test_filter_suffix_stripped(self, tmp_path):
        """lm-eval ',none' suffix is stripped: metric_name is 'acc', not 'acc,none'."""
        _write_json(tmp_path / "result.json", LM_EVAL_RESULT)
        out = str(tmp_path / "out.csv")
        collect_results(str(tmp_path), out)
        df = _read_csv(out)
        # No metric_name should contain a comma
        assert not df["metric_name"].str.contains(",").any()

    def test_stderr_as_separate_metric(self, tmp_path):
        """_stderr metrics appear as their own rows, not filtered out."""
        _write_json(tmp_path / "result.json", LM_EVAL_RESULT)
        out = str(tmp_path / "out.csv")
        collect_results(str(tmp_path), out)
        df = _read_csv(out)
        hellaswag_metrics = set(df[df["task"] == "hellaswag"]["metric_name"])
        assert "acc_stderr" in hellaswag_metrics
        assert "acc_norm_stderr" in hellaswag_metrics

    def test_non_numeric_values_excluded(self, tmp_path):
        """String-valued keys are not emitted as metric rows."""
        data = {
            "model_name": "model-x",
            "results": {
                "sometask": {
                    "acc,none": 0.9,
                    "alias": "sometask_alias",  # non-numeric
                }
            },
            "n-shot": {"sometask": 0},
        }
        _write_json(tmp_path / "result.json", data)
        out = str(tmp_path / "out.csv")
        collect_results(str(tmp_path), out)
        df = _read_csv(out)
        rows = df[df["task"] == "sometask"]
        assert set(rows["metric_name"]) == {"acc"}

    def test_duplicate_stripped_names_first_wins(self, tmp_path):
        """If two raw keys strip to the same name, only the first is emitted."""
        data = {
            "model_name": "model-x",
            "results": {
                "sometask": {
                    "acc,none": 0.7,
                    "acc,remove_whitespace": 0.8,  # would also strip to 'acc'
                }
            },
            "n-shot": {"sometask": 0},
        }
        _write_json(tmp_path / "result.json", data)
        out = str(tmp_path / "out.csv")
        collect_results(str(tmp_path), out)
        df = _read_csv(out)
        rows = df[df["task"] == "sometask"]
        acc_rows = rows[rows["metric_name"] == "acc"]
        assert len(acc_rows) == 1
        assert float(acc_rows["performance"].iloc[0]) == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Lighteval / flores200 tests
# ---------------------------------------------------------------------------

class TestLightevalFloresAllMetrics:
    """All pre-computed flores200 metrics are emitted."""

    def test_all_flores_metrics_emitted(self, tmp_path):
        _write_json(tmp_path / "flores.json", FLORES_RESULT)
        out = str(tmp_path / "out.csv")
        collect_results(str(tmp_path), out)
        df = _read_csv(out)
        metrics = set(df["metric_name"])
        expected = {"chrf++", "chrf++_stderr", "bleu", "bleu_stderr", "bleu_1", "bleu_4"}
        assert expected == metrics

    def test_flores_task_name_stripped(self, tmp_path):
        """'|0' n-shot suffix is stripped from the task name."""
        _write_json(tmp_path / "flores.json", FLORES_RESULT)
        out = str(tmp_path / "out.csv")
        collect_results(str(tmp_path), out)
        df = _read_csv(out)
        assert (df["task"] == "flores200:eng_Latn-ita_Latn").all()

    def test_flores_n_shot_zero_preserved(self, tmp_path):
        """n_shot=0 (parsed from |0 suffix) is correctly preserved, not coerced to 'unknown'."""
        _write_json(tmp_path / "flores.json", FLORES_RESULT)
        out = str(tmp_path / "out.csv")
        collect_results(str(tmp_path), out)
        df = _read_csv(out)
        assert (df["n_shot"] == 0).all(), f"Expected n_shot=0, got: {df['n_shot'].unique()}"

    def test_flores_model_name_extracted(self, tmp_path):
        _write_json(tmp_path / "flores.json", FLORES_RESULT)
        out = str(tmp_path / "out.csv")
        collect_results(str(tmp_path), out)
        df = _read_csv(out)
        assert (df["model_name"] == "model-b").all()


# ---------------------------------------------------------------------------
# CSV schema tests
# ---------------------------------------------------------------------------

class TestOutputSchema:
    """The output CSV has the expected columns including metric_name."""

    def test_columns_present(self, tmp_path):
        _write_json(tmp_path / "result.json", LM_EVAL_RESULT)
        out = str(tmp_path / "out.csv")
        collect_results(str(tmp_path), out)
        df = _read_csv(out)
        assert {"model_name", "task", "n_shot", "metric_name", "performance"}.issubset(df.columns)

    def test_no_performance_column_collision(self, tmp_path):
        """Each (model, task, n_shot, metric_name) combination is unique."""
        _write_json(tmp_path / "result.json", LM_EVAL_RESULT)
        out = str(tmp_path / "out.csv")
        collect_results(str(tmp_path), out)
        df = _read_csv(out)
        dupes = df.duplicated(subset=["model_name", "task", "n_shot", "metric_name"])
        assert not dupes.any(), f"Duplicate rows found:\n{df[dupes]}"


# ---------------------------------------------------------------------------
# Multiple tasks / models
# ---------------------------------------------------------------------------

class TestMultipleTasksAndModels:
    """Rows from multiple tasks and multiple JSON files are all present."""

    def test_two_tasks_all_metrics_present(self, tmp_path):
        _write_json(tmp_path / "result.json", LM_EVAL_RESULT)
        out = str(tmp_path / "out.csv")
        collect_results(str(tmp_path), out)
        df = _read_csv(out)
        tasks = set(df["task"])
        assert "hellaswag" in tasks
        assert "arc_challenge" in tasks

    def test_two_json_files_merged(self, tmp_path):
        _write_json(tmp_path / "a.json", LM_EVAL_RESULT)
        _write_json(tmp_path / "b.json", FLORES_RESULT)
        out = str(tmp_path / "out.csv")
        collect_results(str(tmp_path), out)
        df = _read_csv(out)
        assert set(df["model_name"]) == {"model-a", "model-b"}


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_no_json_files_no_output(self, tmp_path):
        out = str(tmp_path / "out.csv")
        collect_results(str(tmp_path), out)
        assert not Path(out).exists()

    def test_nonexistent_results_dir_raises(self, tmp_path):
        with pytest.raises(ValueError, match="does not exist"):
            collect_results(str(tmp_path / "nonexistent"), str(tmp_path / "out.csv"))

    def test_results_dict_without_n_shot_key_gives_unknown(self, tmp_path):
        """Tasks with no n_shot information produce 'unknown' rather than crashing."""
        data = {
            "model_name": "model-x",
            "results": {
                "sometask": {"acc,none": 0.8},
            },
            "n-shot": {},
        }
        _write_json(tmp_path / "result.json", data)
        out = str(tmp_path / "out.csv")
        collect_results(str(tmp_path), out)
        df = _read_csv(out)
        assert df["n_shot"].iloc[0] == "unknown"
