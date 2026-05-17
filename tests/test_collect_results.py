"""Tests for collect_results() covering both lm-eval and lmms-eval output formats."""

import json
from pathlib import Path

import pandas as pd
import pytest

from oellm.main import collect_results

# ── Helpers ───────────────────────────────────────────────────────────────────


def write_result(results_dir: Path, data: dict, filename: str = "result.json") -> None:
    (results_dir / filename).write_text(json.dumps(data))


def run_collect(tmp_path: Path, *json_payloads: dict) -> pd.DataFrame:
    """Write JSON result files and run collect_results; return CSV as DataFrame."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    for i, data in enumerate(json_payloads):
        write_result(results_dir, data, filename=f"result_{i}.json")

    output_csv = str(tmp_path / "out.csv")
    collect_results(str(results_dir), output_csv=output_csv)

    csv_path = Path(output_csv)
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


# ── lm-eval format (baseline — must remain unchanged) ────────────────────────


class TestCollectResultsLmEvalFormat:
    def test_standard_acc_metric(self, tmp_path):
        data = {
            "model_name": "/path/to/model",
            "results": {"mmlu": {"acc,none": 0.75}},
            "n-shot": {"mmlu": 5},
        }
        df = run_collect(tmp_path, data)
        assert len(df) == 1
        row = df.iloc[0]
        assert row["model_name"] == "/path/to/model"
        assert row["task"] == "mmlu"
        assert row["performance"] == pytest.approx(0.75)
        assert row["n_shot"] == 5

    def test_acc_norm_metric(self, tmp_path):
        data = {
            "model_name": "/path/to/model",
            "results": {"arc_challenge": {"acc_norm,none": 0.60}},
            "n-shot": {"arc_challenge": 10},
        }
        df = run_collect(tmp_path, data)
        assert df.iloc[0]["performance"] == pytest.approx(0.60)

    def test_multiple_tasks_in_one_file(self, tmp_path):
        data = {
            "model_name": "/models/llm",
            "results": {
                "hellaswag": {"acc_norm,none": 0.80},
                "arc_easy": {"acc_norm,none": 0.85},
            },
            "n-shot": {"hellaswag": 10, "arc_easy": 10},
        }
        df = run_collect(tmp_path, data)
        assert len(df) == 2
        assert set(df["task"].tolist()) == {"hellaswag", "arc_easy"}

    def test_multiple_json_files_aggregated(self, tmp_path):
        data1 = {
            "model_name": "/model",
            "results": {"mmlu": {"acc,none": 0.75}},
            "n-shot": {"mmlu": 5},
        }
        data2 = {
            "model_name": "/model",
            "results": {"hellaswag": {"acc_norm,none": 0.80}},
            "n-shot": {"hellaswag": 10},
        }
        df = run_collect(tmp_path, data1, data2)
        assert len(df) == 2

    def test_no_results_returns_empty(self, tmp_path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        output_csv = str(tmp_path / "out.csv")
        collect_results(str(results_dir), output_csv=output_csv)
        assert not Path(output_csv).exists()

    def test_model_name_fallback_when_no_path_field(self, tmp_path):
        """lm-eval has no model_name_or_path; must fall back to model_name."""
        data = {
            "model_name": "/path/to/model",
            "results": {"hellaswag": {"acc_norm,none": 0.70}},
            "n-shot": {"hellaswag": 10},
        }
        df = run_collect(tmp_path, data)
        assert df.iloc[0]["model_name"] == "/path/to/model"


# ── lmms-eval format (Phase 2 additions) ─────────────────────────────────────


class TestCollectResultsLmmsEvalFormat:
    def test_model_name_or_path_takes_priority_over_model_name(self, tmp_path):
        """lmms-eval sets model_name to adapter type; real path is in model_name_or_path."""
        data = {
            "model_name": "llava_hf",
            "model_name_or_path": "/checkpoints/llava-1.5-7b",
            "results": {"vqav2_val": {"vqav2_val/exact_match,none": 0.82}},
            "n-shot": {"vqav2_val": 0},
        }
        df = run_collect(tmp_path, data)
        assert len(df) == 1
        assert df.iloc[0]["model_name"] == "/checkpoints/llava-1.5-7b"

    def test_task_scoped_vqa_score_key_resolved(self, tmp_path):
        data = {
            "model_name": "llava_hf",
            "model_name_or_path": "/models/llava",
            "results": {"vqav2_val": {"vqav2_val/exact_match,none": 0.82}},
            "n-shot": {"vqav2_val": 0},
        }
        df = run_collect(tmp_path, data)
        assert df.iloc[0]["performance"] == pytest.approx(0.82)
        assert df.iloc[0]["task"] == "vqav2_val"

    def test_mmbench_gpt_eval_score_resolved(self, tmp_path):
        data = {
            "model_name": "llava_hf",
            "model_name_or_path": "/models/llava",
            "results": {"mmbench_en_dev": {"mmbench_en_dev/gpt_eval_score,none": 0.75}},
            "n-shot": {"mmbench_en_dev": 0},
        }
        df = run_collect(tmp_path, data)
        assert df.iloc[0]["performance"] == pytest.approx(0.75)
        assert df.iloc[0]["task"] == "mmbench_en_dev"

    def test_mmmu_task_scoped_key(self, tmp_path):
        data = {
            "model_name": "llava_hf",
            "model_name_or_path": "/models/llava",
            "results": {"mmmu_val": {"mmmu_val/mmmu_acc,none": 0.55}},
            "n-shot": {"mmmu_val": 0},
        }
        df = run_collect(tmp_path, data)
        assert df.iloc[0]["performance"] == pytest.approx(0.55)
        assert df.iloc[0]["task"] == "mmmu_val"

    def test_chartqa_relaxed_overall(self, tmp_path):
        data = {
            "model_name": "llava_hf",
            "model_name_or_path": "/models/llava",
            "results": {"chartqa": {"chartqa/relaxed_overall,none": 0.68}},
            "n-shot": {"chartqa": 0},
        }
        df = run_collect(tmp_path, data)
        assert df.iloc[0]["performance"] == pytest.approx(0.68)

    def test_docvqa_anls_metric(self, tmp_path):
        data = {
            "model_name": "llava_hf",
            "model_name_or_path": "/models/llava",
            "results": {"docvqa_val": {"docvqa_val/anls,none": 0.91}},
            "n-shot": {"docvqa_val": 0},
        }
        df = run_collect(tmp_path, data)
        assert df.iloc[0]["performance"] == pytest.approx(0.91)

    def test_ocrbench_accuracy_metric(self, tmp_path):
        data = {
            "model_name": "llava_hf",
            "model_name_or_path": "/models/llava",
            "results": {"ocrbench": {"ocrbench/ocrbench_accuracy,none": 512.0}},
            "n-shot": {"ocrbench": 0},
        }
        df = run_collect(tmp_path, data)
        assert df.iloc[0]["performance"] == pytest.approx(512.0)

    def test_mathvista_leaf_llm_judge_metric(self, tmp_path):
        """mathvista_testmini is a group upstream; only the three leaves
        (_cot / _format / _solution) emit the llm_as_judge_eval metric."""
        data = {
            "model_name": "llava_hf",
            "model_name_or_path": "/models/llava",
            "results": {
                "mathvista_testmini_cot": {
                    "mathvista_testmini_cot/llm_as_judge_eval,none": 0.49
                }
            },
            "n-shot": {"mathvista_testmini_cot": 0},
        }
        df = run_collect(tmp_path, data)
        assert df.iloc[0]["performance"] == pytest.approx(0.49)

    def test_mvbench_group_aggregates_from_subtasks(self, tmp_path):
        """lmms-eval emits MVBench as an empty-placeholder parent
        (``{" ": " ", "alias": "mvbench"}``) plus 20 metric-bearing
        children, with the parent listed in ``group_subtasks``. Without an
        aggregation fallback, both parent (empty placeholder) and children
        (group-subtask skip) get dropped from the output. This test pins
        the synthetic-row behavior — average of children's metrics."""
        data = {
            "model_name": "qwen2_vl",
            "model_name_or_path": "/models/Qwen2-VL-2B-Instruct",
            "results": {
                # Empty parent — what lmms-eval actually writes for MVBench.
                "mvbench": {" ": " ", "alias": "mvbench"},
                "mvbench_action_antonym": {
                    "mvbench_action_antonym/mvbench_accuracy,none": 75.0,
                },
                "mvbench_moving_direction": {
                    "mvbench_moving_direction/mvbench_accuracy,none": 100.0,
                },
                "mvbench_action_sequence": {
                    "mvbench_action_sequence/mvbench_accuracy,none": 0.0,
                },
                "mvbench_scene_transition": {
                    "mvbench_scene_transition/mvbench_accuracy,none": 25.0,
                },
            },
            "n-shot": {"mvbench": 0},
            "group_subtasks": {
                "mvbench": [
                    "mvbench_action_antonym",
                    "mvbench_moving_direction",
                    "mvbench_action_sequence",
                    "mvbench_scene_transition",
                ]
            },
        }
        df = run_collect(tmp_path, data)

        # One row for the mvbench group, none for its children.
        assert len(df) == 1
        row = df.iloc[0]
        assert row["task"] == "mvbench"
        # Mean of the 4 children: (75 + 100 + 0 + 25) / 4 = 50.
        assert row["performance"] == pytest.approx(50.0)
        # The metric_name should come from a child (mvbench_accuracy,none).
        assert "mvbench_accuracy" in row["metric_name"]

    def test_multiple_image_tasks_in_one_file(self, tmp_path):
        data = {
            "model_name": "llava_hf",
            "model_name_or_path": "/models/llava",
            "results": {
                "vqav2_val": {"vqav2_val/exact_match,none": 0.82},
                "mmbench_en_dev": {"mmbench_en_dev/gpt_eval_score,none": 0.75},
                "chartqa": {"chartqa/relaxed_overall,none": 0.68},
            },
            "n-shot": {
                "vqav2_val": 0,
                "mmbench_en_dev": 0,
                "chartqa": 0,
            },
        }
        df = run_collect(tmp_path, data)
        assert len(df) == 3
        assert set(df["task"].tolist()) == {"vqav2_val", "mmbench_en_dev", "chartqa"}

    def test_lmeval_and_lmms_eval_results_aggregated(self, tmp_path):
        """Both lm-eval and lmms-eval JSON files can coexist in the same results dir."""
        lm_eval_data = {
            "model_name": "/path/to/model",
            "results": {"mmlu": {"acc,none": 0.75}},
            "n-shot": {"mmlu": 5},
        }
        lmms_eval_data = {
            "model_name": "llava_hf",
            "model_name_or_path": "/path/to/model",
            "results": {"vqav2_val": {"vqav2_val/exact_match,none": 0.82}},
            "n-shot": {"vqav2_val": 0},
        }
        df = run_collect(tmp_path, lm_eval_data, lmms_eval_data)
        assert len(df) == 2
        assert set(df["task"].tolist()) == {"mmlu", "vqav2_val"}

    def test_empty_model_name_or_path_falls_back_to_model_name(self, tmp_path):
        """Empty string for model_name_or_path should fall back to model_name."""
        data = {
            "model_name": "llava_hf",
            "model_name_or_path": "",
            "results": {"vqav2_val": {"vqav2_val/exact_match,none": 0.80}},
            "n-shot": {"vqav2_val": 0},
        }
        df = run_collect(tmp_path, data)
        assert df.iloc[0]["model_name"] == "llava_hf"

    def test_n_shot_zero_preserved(self, tmp_path):
        data = {
            "model_name": "llava_hf",
            "model_name_or_path": "/models/llava",
            "results": {"vqav2_val": {"vqav2_val/exact_match,none": 0.82}},
            "n-shot": {"vqav2_val": 0},
        }
        df = run_collect(tmp_path, data)
        assert df.iloc[0]["n_shot"] == 0


# ── lmms-eval audio tasks (Block A) ──────────────────────────────────────────


class TestCollectResultsLmmsEvalAudioFormat:
    """Verify the lmms-eval output → CSV path for audio tasks. Mirrors the
    image-task tests but pins the four metric families audio benchmarks use
    (WER / CER / BLEU / accuracy). Each case asserts that _resolve_metric
    strips the `task_name/` prefix that lmms-eval prepends to its metric keys
    and looks up the right `task_metrics` entry from task-groups.yaml."""

    def test_librispeech_wer_resolved(self, tmp_path):
        """ASR task with WER metric — covers the curated audio-understanding suite."""
        data = {
            "model_name": "qwen2_audio",
            "model_name_or_path": "/models/qwen2-audio",
            "results": {
                "librispeech_test_clean": {"librispeech_test_clean/wer,none": 0.053}
            },
            "n-shot": {"librispeech_test_clean": 0},
        }
        df = run_collect(tmp_path, data)
        assert len(df) == 1
        row = df.iloc[0]
        assert row["task"] == "librispeech_test_clean"
        assert row["performance"] == pytest.approx(0.053)
        assert row["metric_name"] == "wer,none"
        assert row["model_name"] == "/models/qwen2-audio"

    def test_wenet_speech_mer_resolved(self, tmp_path):
        """Chinese ASR uses MER (Mixed Error Rate) per lmms-eval upstream."""
        data = {
            "model_name": "qwen2_audio",
            "model_name_or_path": "/models/qwen2-audio",
            "results": {
                "wenet_speech_test_meeting": {"wenet_speech_test_meeting/mer,none": 0.117}
            },
            "n-shot": {"wenet_speech_test_meeting": 0},
        }
        df = run_collect(tmp_path, data)
        assert df.iloc[0]["performance"] == pytest.approx(0.117)
        assert df.iloc[0]["metric_name"] == "mer,none"

    def test_covost2_bleu_resolved(self, tmp_path):
        """Speech-translation task uses BLEU."""
        data = {
            "model_name": "qwen2_audio",
            "model_name_or_path": "/models/qwen2-audio",
            "results": {"covost2_en_zh_test": {"covost2_en_zh_test/bleu,none": 24.8}},
            "n-shot": {"covost2_en_zh_test": 0},
        }
        df = run_collect(tmp_path, data)
        assert df.iloc[0]["performance"] == pytest.approx(24.8)
        assert df.iloc[0]["metric_name"] == "bleu,none"

    def test_muchomusic_accuracy_resolved(self, tmp_path):
        """Music-MCQ task uses plain accuracy."""
        data = {
            "model_name": "qwen2_audio",
            "model_name_or_path": "/models/qwen2-audio",
            "results": {"muchomusic": {"muchomusic/accuracy,none": 0.41}},
            "n-shot": {"muchomusic": 0},
        }
        df = run_collect(tmp_path, data)
        assert df.iloc[0]["performance"] == pytest.approx(0.41)
        assert df.iloc[0]["metric_name"] == "accuracy,none"

    def test_full_audio_understanding_suite_output(self, tmp_path):
        """End-to-end: the 8 tasks in audio-understanding all resolve to
        numeric performance values in a single collect_results call. This is
        the regression guard that would have caught the original
        air_bench_chat group-expansion bug (a group task emits subset-level
        metric keys that don't match its task_path row in jobs.csv)."""
        data = {
            "model_name": "qwen2_audio",
            "model_name_or_path": "/models/qwen2-audio",
            "results": {
                "librispeech_test_clean": {"librispeech_test_clean/wer,none": 0.053},
                "fleurs_en": {"fleurs_en/wer,none": 0.061},
                "gigaspeech_test": {"gigaspeech_test/wer,none": 0.094},
                "tedlium_dev_test": {"tedlium_dev_test/wer,none": 0.072},
                "wenet_speech_test_meeting": {
                    "wenet_speech_test_meeting/mer,none": 0.117
                },
                "covost2_en_zh_test": {"covost2_en_zh_test/bleu,none": 24.8},
                "vocalsound_test": {"vocalsound_test/accuracy,none": 0.81},
                "muchomusic": {"muchomusic/accuracy,none": 0.41},
            },
            "n-shot": dict.fromkeys(
                [
                    "librispeech_test_clean",
                    "fleurs_en",
                    "gigaspeech_test",
                    "tedlium_dev_test",
                    "wenet_speech_test_meeting",
                    "covost2_en_zh_test",
                    "vocalsound_test",
                    "muchomusic",
                ],
                0,
            ),
        }
        df = run_collect(tmp_path, data)
        assert len(df) == 8
        assert all(df["performance"].notna())
        assert set(df["task"].tolist()) == {
            "librispeech_test_clean",
            "fleurs_en",
            "gigaspeech_test",
            "tedlium_dev_test",
            "wenet_speech_test_meeting",
            "covost2_en_zh_test",
            "vocalsound_test",
            "muchomusic",
        }


# ── Structured output (JSON + Markdown alongside CSV) ──────────────────────


class TestCollectResultsStructuredOutput:
    def test_json_file_written_alongside_csv(self, tmp_path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        data = {
            "model_name": "/path/to/model",
            "results": {"copa": {"acc,none": 0.80}},
            "n-shot": {"copa": 0},
        }
        write_result(results_dir, data)
        output_csv = str(tmp_path / "out.csv")
        collect_results(str(results_dir), output_csv=output_csv)

        json_path = tmp_path / "out.json"
        assert json_path.exists()
        envelope = json.loads(json_path.read_text())
        assert envelope["version"] == "1.1"
        assert len(envelope["results"]) == 1
        record = envelope["results"][0]
        assert record["task"] == "copa"
        # acc is a 0-1 scale metric in METRIC_NATIVE_SCALE, so the
        # normalized value is the raw value × 100.
        assert record["performance"] == pytest.approx(0.80)
        assert record["performance_normalized"] == pytest.approx(80.0)

    def test_markdown_file_written_alongside_csv(self, tmp_path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        data = {
            "model_name": "/path/to/model",
            "results": {"copa": {"acc,none": 0.80}},
            "n-shot": {"copa": 0},
        }
        write_result(results_dir, data)
        output_csv = str(tmp_path / "out.csv")
        collect_results(str(results_dir), output_csv=output_csv)

        md_path = tmp_path / "out.md"
        assert md_path.exists()
        content = md_path.read_text()
        assert "copa" in content
        assert "Model" in content

    def test_no_structured_output_when_no_results(self, tmp_path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        # Write a JSON with no extractable metrics
        data = {"model_name": "m", "results": {}}
        write_result(results_dir, data)
        output_csv = str(tmp_path / "out.csv")
        collect_results(str(results_dir), output_csv=output_csv)

        assert not (tmp_path / "out.json").exists()
        assert not (tmp_path / "out.md").exists()
