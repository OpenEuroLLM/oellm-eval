"""Tests for the EvalRunner orchestration layer."""

from __future__ import annotations

from unittest.mock import patch

from oellm.constants import EvaluationJob
from oellm.runner import _ALIAS_MAP, EvalRunner


class TestCanonicalName:
    def test_lm_eval_canonical(self):
        assert EvalRunner.canonical_name("lm_eval") == "lm_eval"

    def test_lm_eval_alias(self):
        assert EvalRunner.canonical_name("lm-eval") == "lm_eval"

    def test_lm_eval_harness_alias(self):
        assert EvalRunner.canonical_name("lm-eval-harness") == "lm_eval"

    def test_lighteval_canonical(self):
        assert EvalRunner.canonical_name("lighteval") == "lighteval"

    def test_lighteval_alias(self):
        assert EvalRunner.canonical_name("light-eval") == "lighteval"

    def test_lmms_eval_canonical(self):
        assert EvalRunner.canonical_name("lmms_eval") == "lmms_eval"

    def test_lmms_eval_alias(self):
        assert EvalRunner.canonical_name("lmms-eval") == "lmms_eval"

    def test_unknown_suite_passes_through(self):
        assert EvalRunner.canonical_name("my_custom_suite") == "my_custom_suite"


class TestKnownEngines:
    def test_includes_lm_eval(self):
        assert "lm_eval" in EvalRunner.known_engines()

    def test_includes_lighteval(self):
        assert "lighteval" in EvalRunner.known_engines()

    def test_includes_lmms_eval(self):
        assert "lmms_eval" in EvalRunner.known_engines()

    def test_returns_list_of_strings(self):
        engines = EvalRunner.known_engines()
        assert isinstance(engines, list)
        assert all(isinstance(e, str) for e in engines)


class TestResolveSuiteLmmsEval:
    def test_lmms_eval_detects_adapter(self):
        runner = EvalRunner()
        job = EvaluationJob(
            model_path="llava-hf/llava-1.5-7b-hf",
            task_path="vqav2_val",
            n_shot=0,
            eval_suite="lmms_eval",
        )
        with patch("oellm.runner.detect_lmms_model_type", return_value="llava_hf"):
            result = runner.resolve_suite(job)
        assert result == "lmms_eval:llava_hf"

    def test_lmms_eval_qwen_adapter(self):
        runner = EvalRunner()
        job = EvaluationJob(
            model_path="Qwen/Qwen2.5-VL-7B-Instruct",
            task_path="mmmu_val",
            n_shot=0,
            eval_suite="lmms_eval",
        )
        with patch("oellm.runner.detect_lmms_model_type", return_value="qwen2_5_vl"):
            result = runner.resolve_suite(job)
        assert result == "lmms_eval:qwen2_5_vl"


class TestResolveSuiteContrib:
    def test_contrib_suite_with_model_flags(self):
        runner = EvalRunner()
        job = EvaluationJob(
            model_path="lmsdss/RegionReasoner-7B",
            task_path="regiondial_refcocog",
            n_shot=0,
            eval_suite="regiondial_bench",
        )
        result = runner.resolve_suite(job)
        assert result == "regiondial_bench:vision_reasoner"

    def test_contrib_suite_qwen_flags(self):
        runner = EvalRunner()
        job = EvaluationJob(
            model_path="Qwen/Qwen2.5-VL-7B-Instruct",
            task_path="regiondial_refcocog",
            n_shot=0,
            eval_suite="regiondial_bench",
        )
        result = runner.resolve_suite(job)
        assert result == "regiondial_bench:qwen2.5"


class TestResolveSuitePassthrough:
    def test_lm_eval_passes_through(self):
        runner = EvalRunner()
        job = EvaluationJob(
            model_path="EleutherAI/pythia-70m",
            task_path="copa",
            n_shot=0,
            eval_suite="lm_eval",
        )
        result = runner.resolve_suite(job)
        assert result == "lm_eval"

    def test_lighteval_passes_through(self):
        runner = EvalRunner()
        job = EvaluationJob(
            model_path="EleutherAI/pythia-70m",
            task_path="flores200:eng_Latn-bul_Cyrl",
            n_shot=0,
            eval_suite="lighteval",
        )
        result = runner.resolve_suite(job)
        assert result == "lighteval"

    def test_unknown_suite_passes_through(self):
        runner = EvalRunner()
        job = EvaluationJob(
            model_path="some/model",
            task_path="some_task",
            n_shot=0,
            eval_suite="future_engine",
        )
        result = runner.resolve_suite(job)
        assert result == "future_engine"


class TestPrepareJobs:
    def test_modifies_jobs_in_place(self):
        runner = EvalRunner()
        jobs = [
            EvaluationJob(
                model_path="EleutherAI/pythia-70m",
                task_path="copa",
                n_shot=0,
                eval_suite="lm_eval",
            ),
        ]
        result = runner.prepare_jobs(jobs)
        assert result is jobs
        assert jobs[0].eval_suite == "lm_eval"

    def test_resolves_mixed_suites(self):
        runner = EvalRunner()
        jobs = [
            EvaluationJob(
                model_path="EleutherAI/pythia-70m",
                task_path="copa",
                n_shot=0,
                eval_suite="lm_eval",
            ),
            EvaluationJob(
                model_path="llava-hf/llava-1.5-7b-hf",
                task_path="vqav2_val",
                n_shot=0,
                eval_suite="lmms_eval",
            ),
        ]
        with patch("oellm.runner.detect_lmms_model_type", return_value="llava_hf"):
            runner.prepare_jobs(jobs)

        assert jobs[0].eval_suite == "lm_eval"
        assert jobs[1].eval_suite == "lmms_eval:llava_hf"

    def test_empty_list(self):
        runner = EvalRunner()
        result = runner.prepare_jobs([])
        assert result == []


class TestAliasMap:
    def test_all_engines_have_canonical_entry(self):
        for name in EvalRunner.known_engines():
            assert name in _ALIAS_MAP
            assert _ALIAS_MAP[name] == name
