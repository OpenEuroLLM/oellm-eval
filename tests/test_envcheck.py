"""Tests for the environment pre-flight (oellm/envcheck.py) and doctor."""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from oellm.envcheck import (
    FAIL,
    OK,
    WARN,
    SuiteRequirements,
    canonical_suites,
    check_scheduled_environment,
    collect_problems,
    probe_import,
    run_doctor_checks,
)

TEST_VENV = str(Path(sys.prefix))


# ---------------------------------------------------------------------------
# probe_import
# ---------------------------------------------------------------------------


class TestProbeImport:
    def test_stdlib_module_imports(self):
        ok, detail = probe_import(Path(sys.prefix) / "bin" / "python", "json")
        assert ok
        assert detail  # version string or "unknown version"

    def test_missing_module_fails_with_error_text(self):
        ok, detail = probe_import(
            Path(sys.prefix) / "bin" / "python", "definitely_not_a_module_xyz"
        )
        assert not ok
        assert "definitely_not_a_module_xyz" in detail

    def test_missing_interpreter_fails(self):
        ok, detail = probe_import("/nonexistent/python", "json")
        assert not ok
        assert detail


# ---------------------------------------------------------------------------
# canonical_suites
# ---------------------------------------------------------------------------


class TestCanonicalSuites:
    def test_strips_model_flags_and_normalises_aliases(self):
        suites = {
            "lm-eval-harness",
            "lmms_eval:llava_hf",
            "audiobench:Qwen2-Audio-7B-Instruct",
            "LIGHTEVAL",
        }
        assert canonical_suites(suites) == {
            "lm_eval",
            "lmms_eval",
            "audiobench",
            "lighteval",
        }


# ---------------------------------------------------------------------------
# collect_problems — container mode (static contract)
# ---------------------------------------------------------------------------


class TestContainerMode:
    def test_lm_eval_and_lighteval_are_fine(self):
        assert collect_problems({"lm_eval", "lighteval"}, venv_path=None) == []

    def test_lmms_eval_is_rejected(self):
        problems = collect_problems({"lmms_eval:llava_hf"}, venv_path=None)
        assert len(problems) == 1
        assert "container" in problems[0]

    def test_contrib_suite_is_rejected(self):
        problems = collect_problems(
            {"audiobench:Qwen2-Audio-7B-Instruct"}, venv_path=None
        )
        assert len(problems) == 1
        assert "container" in problems[0]

    def test_unknown_suite_is_rejected(self):
        problems = collect_problems({"no_such_suite"}, venv_path=None)
        assert len(problems) == 1
        assert "unknown" in problems[0]

    def test_pinned_group_is_rejected_in_container_mode(self):
        problems = collect_problems(
            {"lm_eval"}, venv_path=None, group_names=["dclm-core-22"]
        )
        assert any("dclm-core-22" in p and "silently wrong" in p for p in problems)


# ---------------------------------------------------------------------------
# collect_problems — venv mode (live probes against this test venv)
# ---------------------------------------------------------------------------


class TestVenvMode:
    def test_missing_engine_is_reported(self):
        # The dev/test venv has no lm-eval installed — the probe must fail.
        problems = collect_problems({"lm_eval"}, venv_path=TEST_VENV)
        assert len(problems) == 1
        assert "lm_eval" in problems[0]
        assert TEST_VENV in problems[0]

    def test_satisfied_requirement_passes(self):
        fake = {"lm_eval": SuiteRequirements(modules=("json",), container_ok=True)}
        with patch.dict("oellm.envcheck.SUITE_REQUIREMENTS", fake):
            assert collect_problems({"lm_eval"}, venv_path=TEST_VENV) == []

    def test_contrib_env_var_missing(self):
        # oellm itself is importable in the test venv, so only the env var fails.
        env = {k: v for k, v in os.environ.items() if k != "AUDIOBENCH_DIR"}
        problems = collect_problems({"audiobench"}, venv_path=TEST_VENV, env=env)
        assert len(problems) == 1
        assert "AUDIOBENCH_DIR" in problems[0]
        assert "not set" in problems[0]

    def test_contrib_env_var_nonexistent_path(self, tmp_path):
        env = dict(os.environ)
        env["AUDIOBENCH_DIR"] = str(tmp_path / "missing")
        problems = collect_problems({"audiobench"}, venv_path=TEST_VENV, env=env)
        assert len(problems) == 1
        assert "does not exist" in problems[0]

    def test_contrib_fully_satisfied(self, tmp_path):
        env = dict(os.environ)
        env["AUDIOBENCH_DIR"] = str(tmp_path)
        assert collect_problems({"audiobench"}, venv_path=TEST_VENV, env=env) == []

    def test_version_pin_mismatch_reported(self):
        # json imports fine but its "version" is never 9.9.9.
        pins = {"some-group": (("json", "9.9.9"),)}
        with patch.dict("oellm.envcheck.GROUP_VERSION_PINS", pins):
            problems = collect_problems(
                set(), venv_path=TEST_VENV, group_names=["some-group"]
            )
        assert len(problems) == 1
        assert "9.9.9" in problems[0]
        assert "silently wrong" in problems[0]

    def test_missing_executable_reported(self):
        fake = {
            "lighteval": SuiteRequirements(
                executables=("definitely-not-a-binary-xyz",), container_ok=True
            )
        }
        with patch.dict("oellm.envcheck.SUITE_REQUIREMENTS", fake):
            problems = collect_problems({"lighteval"}, venv_path=TEST_VENV)
        assert len(problems) == 1
        assert "definitely-not-a-binary-xyz" in problems[0]


# ---------------------------------------------------------------------------
# check_scheduled_environment + scheduler wiring
# ---------------------------------------------------------------------------


class TestScheduledEnvironmentCheck:
    def test_raises_systemexit_listing_all_problems(self):
        with pytest.raises(SystemExit) as excinfo:
            check_scheduled_environment(
                {"lmms_eval:llava_hf", "no_such_suite"}, venv_path=None
            )
        msg = str(excinfo.value)
        assert "lmms_eval" in msg
        assert "no_such_suite" in msg
        assert "--skip-checks" in msg

    def test_scheduler_runs_the_check_when_not_skipping(self, tmp_path):
        from oellm.main import schedule_evals

        with (
            patch("oellm.scheduler._load_cluster_env"),
            patch("oellm.scheduler._num_jobs_in_queue", return_value=0),
            patch.dict(os.environ, {"EVAL_OUTPUT_DIR": str(tmp_path)}),
        ):
            with pytest.raises(SystemExit, match="lm_eval"):
                schedule_evals(
                    models="EleutherAI/pythia-70m",
                    tasks="hellaswag",
                    n_shot=0,
                    venv_path=TEST_VENV,  # dev venv has no lm-eval
                    dry_run=True,
                )

    def test_scheduler_passes_with_satisfied_requirements(self, tmp_path):
        from oellm.main import schedule_evals

        fake = {"lm_eval": SuiteRequirements(modules=("json",), container_ok=True)}
        with (
            patch("oellm.scheduler._load_cluster_env"),
            patch("oellm.scheduler._num_jobs_in_queue", return_value=0),
            patch("oellm.scheduler._process_model_paths"),
            patch("oellm.scheduler._lookup_dataset_specs_for_tasks", return_value=[]),
            patch.dict("oellm.envcheck.SUITE_REQUIREMENTS", fake),
            patch.dict(os.environ, {"EVAL_OUTPUT_DIR": str(tmp_path)}),
        ):
            schedule_evals(
                models="EleutherAI/pythia-70m",
                tasks="hellaswag",
                n_shot=0,
                venv_path=TEST_VENV,
                dry_run=True,
            )
        assert list(tmp_path.glob("**/submit_evals.sbatch"))

    def test_skip_checks_bypasses_the_check(self, tmp_path):
        from oellm.main import schedule_evals

        with (
            patch("oellm.scheduler._load_cluster_env"),
            patch("oellm.scheduler._num_jobs_in_queue", return_value=0),
            patch.dict(os.environ, {"EVAL_OUTPUT_DIR": str(tmp_path)}),
        ):
            schedule_evals(
                models="EleutherAI/pythia-70m",
                tasks="hellaswag",
                n_shot=0,
                venv_path=TEST_VENV,
                dry_run=True,
                skip_checks=True,
            )
        assert list(tmp_path.glob("**/submit_evals.sbatch"))


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


class TestDoctor:
    def test_returns_structured_results(self):
        results = run_doctor_checks(venv_path=TEST_VENV)
        assert results
        assert all(r.status in (OK, WARN, FAIL) for r in results)
        names = {r.name for r in results}
        assert "venv" in names
        assert any(n.startswith("env: HF_HOME") or n == "HF_HOME path" for n in names)

    def test_venv_engine_probes_present(self):
        results = run_doctor_checks(venv_path=TEST_VENV)
        probe_names = {r.name for r in results if r.name.startswith("venv: import")}
        # All built-in module-based engines must be probed.
        assert any("lm_eval" in n for n in probe_names)
        assert any("lmms_eval" in n for n in probe_names)

    def test_required_group_without_venv_fails_runtime_mode(self):
        results = run_doctor_checks(venv_path=None, task_groups=["image-vqa"])
        runtime = [r for r in results if r.name == "runtime mode"]
        assert runtime and runtime[0].status == FAIL
        assert "lmms_eval" in runtime[0].detail

    def test_missing_engines_fail_when_group_requires_them(self):
        results = run_doctor_checks(venv_path=TEST_VENV, task_groups=["image-vqa"])
        lmms_probes = [r for r in results if r.name.startswith("venv: import lmms_eval")]
        assert lmms_probes and lmms_probes[0].status == FAIL

    def test_missing_engines_warn_when_not_required(self):
        results = run_doctor_checks(venv_path=TEST_VENV)
        lmms_probes = [r for r in results if r.name.startswith("venv: import lmms_eval")]
        assert lmms_probes and lmms_probes[0].status == WARN

    def test_broken_venv_path_fails(self, tmp_path):
        results = run_doctor_checks(venv_path=str(tmp_path / "nope"))
        venv_checks = [r for r in results if r.name == "venv"]
        assert venv_checks and venv_checks[0].status == FAIL
