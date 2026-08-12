"""Regression tests for collection correctness, provenance, and scheduling
recovery: duplicate-result resolution, silent group/prefix omissions, the
lighteval round-trip, per-task scale overrides, model-aware compare,
cluster-env precedence, eval_suite normalisation, task-scope aux staging,
and the v1.2 provenance envelope.
"""

import csv
import json
import os
import time
from pathlib import Path

import pytest

from oellm import __version__
from oellm.results import _normalize_to_100, collect_results


def _write(results_dir: Path, name: str, payload: dict, mtime: float | None = None):
    p = results_dir / name
    p.write_text(json.dumps(payload))
    if mtime is not None:
        os.utime(p, (mtime, mtime))
    return p


def _rows(output_csv: str) -> list[dict]:
    with open(output_csv) as f:
        return list(csv.DictReader(f))


M = {"model_name_or_path": "m"}


# ── duplicate results resolve to the newest file, loudly ────────────────


class TestDuplicateResolution:
    def test_newest_mtime_wins_regardless_of_name_order(self, tmp_path, capsys):
        d = tmp_path / "results"
        d.mkdir()
        now = time.time()
        # "zz" sorts after "aa" by name; mtime must decide, not the name.
        _write(
            d,
            "zz_stale.json",
            {**M, "results": {"boolq": {"acc,none": 0.10}}, "n-shot": {"boolq": 0}},
            mtime=now - 100,
        )
        _write(
            d,
            "aa_fresh.json",
            {**M, "results": {"boolq": {"acc,none": 0.85}}, "n-shot": {"boolq": 0}},
            mtime=now,
        )
        out = str(tmp_path / "out.csv")
        collect_results(str(tmp_path), out)
        rows = _rows(out)
        assert len(rows) == 1
        assert float(rows[0]["performance"]) == pytest.approx(0.85)
        # rich wraps log lines; compare on whitespace-normalized output
        assert "Duplicate results" in " ".join(capsys.readouterr().out.split())


# ── group/prefix omission mechanisms ────────────────────────────────────


class TestGroupCollection:
    def _collect(self, tmp_path, payload, caplog=None):
        d = tmp_path / "results"
        d.mkdir()
        _write(d, "r.json", payload)
        out = str(tmp_path / "out.csv")
        collect_results(str(tmp_path), out)
        return _rows(out) if Path(out).exists() else []

    def test_all_top_level_groups_collected(self, tmp_path):
        rows = self._collect(
            tmp_path,
            {
                **M,
                "results": {
                    "g1": {"acc,none": 0.5},
                    "g2": {"acc,none": 0.6},
                    "s1": {"acc,none": 0.5},
                    "s2": {"acc,none": 0.6},
                },
                "groups": {"g1": {"acc,none": 0.5}, "g2": {"acc,none": 0.6}},
                "group_subtasks": {"g1": ["s1"], "g2": ["s2"]},
                "n-shot": {"g1": 0, "g2": 0},
            },
        )
        assert {r["task"] for r in rows} == {"g1", "g2"}

    def test_metricless_parent_falls_through_to_child_groups(self, tmp_path):
        rows = self._collect(
            tmp_path,
            {
                **M,
                "results": {
                    "parent": {},
                    "childg": {"acc,none": 0.6},
                    "s1": {"acc,none": 0.6},
                },
                "groups": {"parent": {}, "childg": {"acc,none": 0.6}},
                "group_subtasks": {"parent": ["childg"], "childg": ["s1"]},
                "n-shot": {"childg": 0},
            },
        )
        assert [r["task"] for r in rows] == ["childg"]

    def test_standalone_task_in_group_file_is_kept(self, tmp_path):
        rows = self._collect(
            tmp_path,
            {
                **M,
                "results": {
                    "g1": {"acc,none": 0.5},
                    "s1": {"acc,none": 0.5},
                    "lone": {"acc,none": 0.7},
                },
                "groups": {"g1": {"acc,none": 0.5}},
                "group_subtasks": {"g1": ["s1"]},
                "n-shot": {"g1": 0, "lone": 0},
            },
        )
        assert {r["task"] for r in rows} == {"g1", "lone"}

    def test_mmlu_pro_is_not_eaten_by_prefix_rule(self, tmp_path):
        rows = self._collect(
            tmp_path,
            {
                **M,
                "results": {"mmlu_pro": {"acc,none": 0.4}},
                "n-shot": {"mmlu_pro": 5},
            },
        )
        assert [(r["task"], r["n_shot"]) for r in rows] == [("mmlu_pro", "5")]

    def test_mmlu_subtasks_still_skipped_when_aggregate_present(self, tmp_path):
        rows = self._collect(
            tmp_path,
            {
                **M,
                "results": {
                    "mmlu": {"acc,none": 0.5},
                    "mmlu_abstract_algebra": {"acc,none": 0.3},
                },
                "n-shot": {"mmlu": 5},
            },
        )
        assert [r["task"] for r in rows] == ["mmlu"]

    def test_global_mmlu_underscore_language_aggregate_kept(self, tmp_path):
        rows = self._collect(
            tmp_path,
            {
                **M,
                "results": {
                    "global_mmlu_full_zh_hans": {"acc,none": 0.3},
                    "global_mmlu_full_zh_hans_anatomy": {"acc,none": 0.31},
                },
                "n-shot": {"global_mmlu_full_zh_hans": 0},
            },
        )
        assert [r["task"] for r in rows] == ["global_mmlu_full_zh_hans"]

    def test_partial_children_aggregate_warns(self, tmp_path, capsys):
        d = tmp_path / "results"
        d.mkdir()
        _write(
            d,
            "r.json",
            {
                **M,
                "results": {"par": {"alias": " "}, "c1": {"acc,none": 0.5}},
                "group_subtasks": {"par": ["c1", "c2"]},
                "configs": {"c1": {"num_fewshot": 0}},
            },
        )
        out = str(tmp_path / "out.csv")
        collect_results(str(tmp_path), out)
        assert "1/2 subtasks present" in " ".join(capsys.readouterr().out.split())

    def test_zero_row_file_warns(self, tmp_path, capsys):
        d = tmp_path / "results"
        d.mkdir()
        _write(d, "r.json", {**M, "results": {"ghost": {"alias": " "}}})
        collect_results(str(tmp_path), str(tmp_path / "out.csv"))
        assert "contributed zero rows" in " ".join(capsys.readouterr().out.split())


# ── lighteval-shaped round-trip through collect + --check ───────


class TestLightevalRoundTrip:
    PAYLOAD = {
        "config_general": {"model_name": "m"},
        "results": {
            "all": {"acc_norm": 0.61},
            "belebele_fra_Latn_cf|5": {"acc_norm": 0.61, "acc_norm_stderr": 0.01},
        },
    }

    def test_task_and_nshot_recovered(self, tmp_path):
        d = tmp_path / "results"
        d.mkdir()
        _write(d, "r.json", self.PAYLOAD)
        out = str(tmp_path / "out.csv")
        collect_results(str(tmp_path), out)
        rows = _rows(out)
        assert [(r["task"], r["n_shot"]) for r in rows] == [("belebele_fra_Latn_cf", "5")]

    def test_check_marks_lighteval_job_complete(self, tmp_path):
        d = tmp_path / "results"
        d.mkdir()
        _write(d, "r.json", self.PAYLOAD)
        (tmp_path / "jobs.csv").write_text(
            "model_path,task_path,n_shot,eval_suite\nm,belebele_fra_Latn_cf,5,lighteval\n"
        )
        out = tmp_path / "out.csv"
        collect_results(str(tmp_path), str(out), check=True)
        missing = out.with_name("out_missing.csv")
        assert not missing.exists(), "completed lighteval job was re-reported as missing"


# ── per-task scale overrides ────────────────────────────────────────────


class TestScaleOverrides:
    def test_squadv2_f1_is_percentage_scale(self):
        assert _normalize_to_100(55.3, "f1,none", "squadv2") == pytest.approx(55.3)

    def test_coqa_f1_keeps_fraction_scale(self):
        assert _normalize_to_100(0.553, "f1,none", "coqa") == pytest.approx(55.3)

    def test_voicebench_judge_is_likert_scale(self):
        assert _normalize_to_100(
            3.4, "llm_as_judge_eval,none", "voicebench_commoneval"
        ) == pytest.approx(68.0)


# ── Envelope v1.2 + provenance sidecar ───────────────────────────────────────


class TestProvenanceEnvelope:
    def test_sidecar_embedded_and_namespace_reserved(self, tmp_path):
        d = tmp_path / "results"
        d.mkdir()
        _write(
            d,
            "r.json",
            {**M, "results": {"copa": {"acc,none": 0.8}}, "n-shot": {"copa": 0}},
        )
        (tmp_path / "provenance.json").write_text(
            json.dumps({"schema": 1, "model_revisions": {"m": "abc123"}})
        )
        out = tmp_path / "out.csv"
        collect_results(str(tmp_path), str(out))
        envelope = json.loads((tmp_path / "out.json").read_text())
        assert envelope["version"] == "1.2"
        assert envelope["metadata"] == {}
        assert envelope["oellm_version"] == __version__
        assert envelope["runs"][0]["model_revisions"] == {"m": "abc123"}


# ── eval_suite normalisation in the runner ───────────────────────────────


class TestSuiteNormalisation:
    @pytest.fixture(autouse=True)
    def _fake_adapter(self, monkeypatch):
        monkeypatch.setattr("oellm.runner.detect_lmms_model_type", lambda p: "llava_hf")

    @pytest.mark.parametrize(
        "raw", ["LMMS_EVAL", " lmms_eval ", "Lmms-Eval", "lmms_eval"]
    )
    def test_noncanonical_spellings_resolve(self, raw):
        from oellm.constants import EvaluationJob
        from oellm.runner import EvalRunner

        job = EvaluationJob(model_path="x", task_path="t", n_shot=0, eval_suite=raw)
        assert EvalRunner().resolve_suite(job) == "lmms_eval:llava_hf"

    def test_already_suffixed_row_is_idempotent(self):
        from oellm.constants import EvaluationJob
        from oellm.runner import EvalRunner

        job = EvaluationJob(
            model_path="x", task_path="t", n_shot=0, eval_suite="lmms_eval:qwen2_vl"
        )
        assert EvalRunner().resolve_suite(job) == "lmms_eval:qwen2_vl"


# ── user env overrides propagate into templated cluster vars ────────────


class TestClusterEnvPrecedence:
    def test_eval_base_dir_override_reaches_output_dir(self, monkeypatch):
        from oellm import utils

        monkeypatch.setattr(utils.socket, "gethostname", lambda: "login01.leonardo.local")
        for var in (
            "EVAL_OUTPUT_DIR",
            "PARTITION",
            "ACCOUNT",
            "GPUS_PER_NODE",
            "TIME_LIMIT",
        ):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("USER", "testuser")
        monkeypatch.setenv("EVAL_BASE_DIR", "/custom/base")
        # HF_HOME must come from user env: it is a required var on every
        # cluster but the stock leonardo entry does not define it — the test
        # must not depend on local clusters.yaml additions.
        monkeypatch.setenv("HF_HOME", "/custom/hf")
        utils._load_cluster_env()
        assert os.environ["EVAL_OUTPUT_DIR"] == "/custom/base/testuser"


# ── aux staging resolvable from bare task names ─────────────────────────


class TestTaskScopedAuxStaging:
    def test_hf_models_found_without_group(self):
        from oellm.task_groups import _lookup_hf_model_repos_for_tasks

        repos = _lookup_hf_model_repos_for_tasks(["regiondial_refcocog"])
        assert "Ricky06662/TaskRouter-1.5B" in repos
        assert "facebook/sam2-hiera-large" in repos

    def test_hf_dataset_files_found_without_group(self):
        from oellm.task_groups import _lookup_hf_dataset_files_for_tasks

        specs = _lookup_hf_dataset_files_for_tasks(["regiondial_refcocog"])
        assert any(
            s["repo_id"] == "lmsdss/regionreasoner_test_data" and s["patterns"]
            for s in specs
        )

    def test_unknown_tasks_yield_nothing(self):
        from oellm.task_groups import (
            _lookup_hf_dataset_files_for_tasks,
            _lookup_hf_model_repos_for_tasks,
        )

        assert _lookup_hf_model_repos_for_tasks(["hellaswag"]) == []
        assert _lookup_hf_dataset_files_for_tasks(["hellaswag"]) == []


# ── compare is model-aware and reads collect's own output ────────────────


class TestCompare:
    def _envelope(self, rows):
        return json.dumps({"version": "1.2", "results": rows})

    def test_multi_model_rows_both_shown_and_mixed_nshot_sorts(self, tmp_path, capsys):
        from oellm.main import compare

        rows_a = [
            {
                "model": "modelA",
                "task": "mmlu",
                "n_shot": 5,
                "metric": "acc",
                "performance": 0.5,
            },
            {
                "model": "modelB",
                "task": "mmlu",
                "n_shot": 5,
                "metric": "acc",
                "performance": 0.6,
            },
            {
                "model": "modelA",
                "task": "mvbench",
                "n_shot": "unknown",
                "metric": "mvbench_accuracy",
                "performance": 51.0,
            },
        ]
        a = tmp_path / "a.json"
        b = tmp_path / "b.json"
        a.write_text(self._envelope(rows_a))
        b.write_text(self._envelope(rows_a))
        compare(str(a), str(b))  # must not raise (mixed n_shot types)
        out = capsys.readouterr().out
        assert "modelA" in out and "modelB" in out

    def test_directory_mode_reads_eval_results_json(self, tmp_path, capsys):
        from oellm.main import compare

        run = tmp_path / "run"
        run.mkdir()
        (run / "eval_results.json").write_text(
            self._envelope(
                [
                    {
                        "model": "m",
                        "task": "copa",
                        "n_shot": 0,
                        "metric": "acc",
                        "performance": 0.8,
                    }
                ]
            )
        )
        compare(str(run), str(run))
        assert "copa" in capsys.readouterr().out
