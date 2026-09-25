"""Tests for the DocVQA 2026 contrib suite."""

import pytest


@pytest.fixture(scope="module")
def scorer():
    from oellm.contrib.docvqa2026 import _vendor_eval_utils

    return _vendor_eval_utils


class TestSuiteWiring:
    def test_staging_flows_through_the_scheduler_path(self):
        from oellm.task_groups import _collect_dataset_specs, _collect_hf_dataset_files

        assert _collect_dataset_specs(["image-docvqa2026"]) == []
        assert _collect_hf_dataset_files(["image-docvqa2026"]) == [
            {"repo_id": "VLR-CVC/DocVQA-2026", "patterns": ["val.parquet"]}
        ]

    def test_group_and_metric_registered(self):
        from oellm.results import _load_task_metrics
        from oellm.task_groups import _expand_task_groups

        assert _load_task_metrics()["docvqa2026_val"] == "accuracy"
        assert [
            (r.task, r.n_shot, r.suite) for r in _expand_task_groups(["image-docvqa2026"])
        ] == [("docvqa2026_val", 0, "docvqa2026")]

    def test_group_name_follows_the_core_docvqa_pairing(self):
        """image-docvqa/docvqa_val already exists; 2026 must not collide."""
        from oellm.task_groups import _expand_task_groups

        core = _expand_task_groups(["image-docvqa"])
        ours = _expand_task_groups(["image-docvqa2026"])
        assert [r.task for r in core] == ["docvqa_val"]
        assert [r.task for r in ours] == ["docvqa2026_val"]
        assert {r.suite for r in core} != {r.suite for r in ours}

    def test_unknown_task_is_rejected(self, tmp_path):
        from oellm.contrib.docvqa2026 import suite

        with pytest.raises(ValueError, match="Unknown task"):
            suite.run(
                model_path="m",
                task="docvqa_val",
                n_shot=0,
                output_path=tmp_path / "out.json",
                model_flags=None,
                env={},
            )

    def test_parse_results_claims_own_and_rejects_foreign(self):
        from oellm.contrib.docvqa2026 import suite

        mine = {
            "model_name_or_path": "HuggingFaceTB/SmolVLM-256M-Instruct",
            "results": {"docvqa2026_val": {"accuracy": 0.05, "macro_accuracy": 0.05}},
            "configs": {"docvqa2026_val": {"num_fewshot": 0}},
        }
        assert suite.parse_results(mine) == (
            "HuggingFaceTB/SmolVLM-256M-Instruct",
            "docvqa2026_val",
            0,
            {"accuracy": 0.05, "macro_accuracy": 0.05},
        )
        assert suite.parse_results({"results": {"docvqa_val": {"anls": 0.7}}}) is None


class TestAggregation:
    def test_reports_both_averages_and_per_category(self):
        from oellm.contrib.docvqa2026.metrics import aggregate

        records = [
            {"correct": True, "doc_category": "maps"},
            {"correct": False, "doc_category": "maps"},
            {"correct": False, "doc_category": "maps"},
            {"correct": True, "doc_category": "slide"},
            {"correct": True, "doc_category": "slide"},
        ]
        m = aggregate(records)
        assert m["accuracy"] == pytest.approx(0.6)
        assert m["macro_accuracy"] == pytest.approx(2 / 3)
        assert m["acc_maps"] == pytest.approx(1 / 3)
        assert m["acc_slide"] == 1.0
        assert m["n_questions"] == 5 and m["n_correct"] == 3

    def test_averages_agree_when_categories_are_balanced(self):
        """The val split has 10 questions per category, so both must coincide."""
        from oellm.contrib.docvqa2026.metrics import aggregate

        records = [
            {"correct": i < 4, "doc_category": c}
            for c in ("maps", "slide")
            for i in range(10)
        ]
        m = aggregate(records)
        assert m["accuracy"] == pytest.approx(m["macro_accuracy"])

    def test_empty_run_is_an_error_not_a_zero(self):
        from oellm.contrib.docvqa2026.metrics import aggregate

        with pytest.raises(RuntimeError, match="no samples"):
            aggregate([])


class TestPageCap:
    @pytest.mark.parametrize("raw,expected", [("", None), ("4", 4), ("  8 ", 8)])
    def test_read_max_pages(self, raw, expected):
        from oellm.contrib.docvqa2026.datasets import read_max_pages

        assert read_max_pages({"DOCVQA2026_MAX_PAGES": raw}) == expected

    def test_zero_is_rejected(self):
        from oellm.contrib.docvqa2026.datasets import read_max_pages

        with pytest.raises(ValueError, match=">= 1"):
            read_max_pages({"DOCVQA2026_MAX_PAGES": "0"})

    def test_truncation_is_visible_on_the_sample(self):
        from oellm.contrib.docvqa2026.datasets import Sample

        s = Sample("q1", "d1", "maps", "q?", "a", encoded_pages=[1, 2], n_pages_total=36)
        assert s.pages_truncated is True
        assert Sample("q1", "d1", "maps", "q?", "a", [1], 1).pages_truncated is False


class TestGenerationSettings:
    @pytest.mark.parametrize("raw,expected", [("", 2048), ("512", 512), (" 4096 ", 4096)])
    def test_read_max_new_tokens(self, raw, expected):
        from oellm.contrib.docvqa2026.runner import read_max_new_tokens

        assert read_max_new_tokens({"DOCVQA2026_MAX_NEW_TOKENS": raw}) == expected

    @pytest.mark.parametrize("raw", ["0", "-1", "many"])
    def test_bad_max_new_tokens_names_the_variable(self, raw):
        from oellm.contrib.docvqa2026.runner import read_max_new_tokens

        with pytest.raises(ValueError, match="DOCVQA2026_MAX_NEW_TOKENS"):
            read_max_new_tokens({"DOCVQA2026_MAX_NEW_TOKENS": raw})

    def test_no_pages_means_no_image_slots(self):
        from oellm.contrib.docvqa2026.runner import build_messages

        content = build_messages("q?", 0, "PROMPT")[0]["content"]
        assert [c["type"] for c in content] == ["text"]
        assert [c["type"] for c in build_messages("q?", 3, "P")[0]["content"]] == [
            "image",
            "image",
            "image",
            "text",
        ]


class TestPrompt:
    def test_prompt_demands_the_marker_the_scorer_requires(self):
        from oellm.contrib.docvqa2026.prompts import MASTER_PROMPT

        assert "FINAL ANSWER:" in MASTER_PROMPT


class TestRunEndToEnd:
    """run() orchestration with the dataset and model stubbed out."""

    @pytest.fixture
    def stub_run(self, monkeypatch):
        from oellm.contrib.docvqa2026 import datasets as ds_mod
        from oellm.contrib.docvqa2026 import runner as run_mod
        from oellm.contrib.docvqa2026 import suite

        samples = [
            ds_mod.Sample("q1", "d1", "maps", "how many?", "4", ["enc"], 36),
            ds_mod.Sample("q2", "d1", "maps", "which town?", "Wareham", ["enc"], 36),
            ds_mod.Sample("q3", "d2", "slide", "what colour?", "green", ["enc"], 1),
        ]
        replies = {
            "how many?": "FINAL ANSWER: 4",
            "which town?": "FINAL ANSWER: Boston",
            "what colour?": "the colour is green",
        }
        monkeypatch.setattr(
            ds_mod, "load_val", lambda limit=None, max_pages=None: samples
        )
        monkeypatch.setattr(ds_mod, "decode_pages", lambda sample: ["img"])
        monkeypatch.setattr(run_mod, "load_model", lambda *a, **k: ("model", "proc"))
        monkeypatch.setattr(run_mod, "resolve_device", lambda: "cpu")
        monkeypatch.setattr(
            run_mod,
            "generate_answer",
            lambda model, processor, question, images, prompt, device, max_new_tokens: (
                replies[question],
                False,
            ),
        )
        return suite

    def test_writes_scored_results(self, stub_run, tmp_path):
        out = tmp_path / "results" / "docvqa.json"
        stub_run.run(
            model_path="stub/vlm",
            task="docvqa2026_val",
            n_shot=0,
            output_path=out,
            model_flags=None,
            env={"DOCVQA2026_MAX_PAGES": "1"},
        )
        import json

        data = json.loads(out.read_text())
        metrics = data["results"]["docvqa2026_val"]
        assert metrics["n_questions"] == 3
        assert metrics["n_correct"] == 1
        assert metrics["accuracy"] == pytest.approx(1 / 3)
        assert metrics["macro_accuracy"] == pytest.approx(0.25)
        assert metrics["acc_maps"] == 0.5 and metrics["acc_slide"] == 0.0
        assert metrics["max_pages"] == 1
        assert metrics["format_compliance"] == pytest.approx(2 / 3)
        assert metrics["n_truncated_documents"] == 1
        assert metrics["max_new_tokens"] == 2048
        assert metrics["n_hit_token_limit"] == 0

    def test_results_round_trip_through_the_collector(self, stub_run, tmp_path):
        import csv

        from oellm.results import collect_results

        run_dir = tmp_path / "run"
        stub_run.run(
            model_path="stub/vlm",
            task="docvqa2026_val",
            n_shot=0,
            output_path=run_dir / "results" / "docvqa.json",
            model_flags=None,
            env={},
        )
        out_csv = run_dir / "eval.csv"
        collect_results(str(run_dir), str(out_csv))
        rows = list(csv.DictReader(open(out_csv)))
        assert [(r["task"], r["metric_name"]) for r in rows] == [
            ("docvqa2026_val", "accuracy")
        ]
        assert float(rows[0]["performance"]) == pytest.approx(1 / 3)


def _cached_val_parquet():
    """Path to the val parquet if already in the HF cache, else None."""
    try:
        from huggingface_hub import hf_hub_download

        from oellm.contrib.docvqa2026.datasets import HF_REPO, SPLIT

        return hf_hub_download(
            HF_REPO, f"{SPLIT}.parquet", repo_type="dataset", local_files_only=True
        )
    except Exception:
        return None


@pytest.mark.skipif(_cached_val_parquet() is None, reason="val.parquet not cached")
class TestOracleOnRealData:
    """Feeding the real answers back as predictions must score perfectly."""

    @staticmethod
    def _real_answers():
        import pyarrow.parquet as pq

        table = pq.ParquetFile(_cached_val_parquet()).read(
            columns=["doc_category", "questions", "answers"]
        )
        for row in table.to_pylist():
            answers = dict(
                zip(
                    row["answers"]["question_id"],
                    row["answers"]["answer"],
                    strict=True,
                )
            )
            for qid in row["questions"]["question_id"]:
                yield row["doc_category"], answers[qid]

    def test_ground_truth_scores_perfectly(self):
        import ast

        from oellm.contrib.docvqa2026.metrics import aggregate, score_prediction

        records = []
        for category, gt in self._real_answers():
            try:
                parsed = ast.literal_eval(gt)
                spoken = str(parsed[0]) if isinstance(parsed, list) else gt
            except (ValueError, SyntaxError):
                spoken = gt
            correct, _, has_marker = score_prediction(f"FINAL ANSWER: {spoken}", gt)
            records.append(
                {"correct": correct, "doc_category": category, "has_marker": has_marker}
            )

        metrics = aggregate(records)
        assert metrics["n_questions"] == 80
        assert metrics["n_categories"] == 8
        assert metrics["accuracy"] == 1.0
        assert metrics["macro_accuracy"] == 1.0

    def test_dropping_the_marker_zeroes_the_same_answers(self):
        """The 0.0 a non-compliant model earns is the scorer's first rule."""
        from oellm.contrib.docvqa2026.metrics import aggregate, score_prediction

        records = [
            {
                "correct": score_prediction(gt, gt)[0],
                "doc_category": category,
                "has_marker": False,
            }
            for category, gt in self._real_answers()
        ]
        metrics = aggregate(records)
        assert metrics["accuracy"] == 0.0
        assert metrics["format_compliance"] == 0.0


class TestDocumentedQuirksHold:
    """Behaviour the competition scorer depends on, stated explicitly."""

    def test_missing_marker_is_always_wrong(self, scorer):
        correct, extracted = scorer.evaluate_docvqa_prediction("4", "4")
        assert correct is False
        assert extracted == "4"

    def test_numeric_ground_truth_skips_the_anls_fallback(self, scorer):
        assert scorer.evaluate_docvqa_prediction("FINAL ANSWER: four", "4")[0] is False

    def test_value_equal_unit_different_is_wrong(self, scorer):
        assert (
            scorer.evaluate_docvqa_prediction("FINAL ANSWER: 50 g", "50 kg")[0] is False
        )

    def test_unit_and_value_equal_is_right(self, scorer):
        assert (
            scorer.evaluate_docvqa_prediction("FINAL ANSWER: 50 kg", "50 kg")[0] is True
        )

    def test_textual_date_matches_iso_ground_truth(self, scorer):
        assert (
            scorer.evaluate_docvqa_prediction("FINAL ANSWER: Jan 1st 24", "2024-01-01")[0]
            is True
        )

    def test_version_strings_compare_exactly(self, scorer):
        assert (
            scorer.evaluate_docvqa_prediction("FINAL ANSWER: 12.0.0", "12.0.0")[0] is True
        )
        assert (
            scorer.evaluate_docvqa_prediction("FINAL ANSWER: 12.0.1", "12.0.0")[0]
            is False
        )

    def test_list_ground_truth_accepts_any_candidate(self, scorer):
        gt = "['olive green', 'green', 'dark green']"
        assert scorer.evaluate_docvqa_prediction("FINAL ANSWER: green", gt)[0] is True
        assert scorer.evaluate_docvqa_prediction("FINAL ANSWER: blue", gt)[0] is False

    def test_articles_and_punctuation_are_normalised(self, scorer):
        assert (
            scorer.evaluate_docvqa_prediction("FINAL ANSWER: a wrench!", "wrench")[0]
            is True
        )

    def test_anls_threshold_sits_at_0_90(self, scorer):
        assert (
            scorer.evaluate_docvqa_prediction("FINAL ANSWER: adjustablee", "adjustable")[
                0
            ]
            is True
        )
        assert (
            scorer.evaluate_docvqa_prediction("FINAL ANSWER: wrenchh", "wrench")[0]
            is False
        )

    def test_last_marker_wins(self, scorer):
        correct, extracted = scorer.evaluate_docvqa_prediction(
            "FINAL ANSWER: wrong\nFINAL ANSWER: 4", "4"
        )
        assert extracted == "4"
        assert correct is True

    def test_prompt_demands_the_marker(self, scorer):
        assert "FINAL ANSWER:" in scorer.get_evaluation_prompt()
