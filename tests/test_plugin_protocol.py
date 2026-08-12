"""End-to-end conformance test for the contrib plugin protocol.

A synthetic third-party plugin is materialized on disk and discovered through
the real registry; every documented protocol member is then exercised on its
real consumer path: TASK_GROUPS via BaseTask (including the engine_task_name
override — the plan's "one-liner plugin" pathway), LMMS_MODEL_ADAPTERS
overrides via detect_lmms_model_type, detect_model_flags via BaseModelAdapter
in EvalRunner.resolve_suite, BaseMetric (API v2) inside run(), and
parse_results as collect_results' first-chance parser. If any protocol member
loses its consumer again (the dead-extension-point failure class), this
file fails.
"""

import csv
import importlib.util
import json
import sys

import pytest


def _load_task_utils(task_dir: str):
    """Import a custom task's utils.py under a unique module name — a bare
    ``import utils`` would collide across task dirs in sys.modules."""
    path = f"oellm/resources/custom_lm_eval_tasks/{task_dir}/utils.py"
    spec = importlib.util.spec_from_file_location(f"{task_dir}_task_utils", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SUITE_SRC = """
from pathlib import Path

from oellm.core import BaseMetric, BaseModelAdapter, BaseTask

SUITE_NAME = "toy_suite"
CLUSTER_ENV_VARS: list[str] = []
LMMS_MODEL_ADAPTERS = [(["toy-vlm"], "toy_adapter")]


class ToyTask(BaseTask):
    @property
    def name(self):
        return "toy_task"

    @property
    def engine_task_name(self):
        return "toy_task_engine"

    @property
    def suite(self):
        return SUITE_NAME

    @property
    def n_shots(self):
        return [0]

    @property
    def primary_metric(self):
        return "toy_score"


class ToyAdapter(BaseModelAdapter):
    def __init__(self, path):
        self._p = path

    @property
    def model_path(self):
        return self._p

    def to_lm_eval_args(self):
        return f"pretrained={self._p}"

    def to_lmms_eval_args(self):
        return f"pretrained={self._p}"

    def to_contrib_flags(self):
        return "toy_backend" if "toy" in str(self._p).lower() else None


class ToyScore(BaseMetric):
    @property
    def name(self):
        return "toy_score"

    def compute(self, samples):
        if not samples:
            return 0.0
        return sum(s["ok"] for s in samples) / len(samples)


TASK_GROUPS = ToyTask.to_task_groups_dict()


def detect_model_flags(model_path):
    return ToyAdapter(model_path).to_contrib_flags()


def run(*, model_path, task, n_shot, output_path, model_flags, env):
    import json as _json

    score = ToyScore().compute([{"ok": 1}, {"ok": 0}])
    Path(output_path).write_text(
        _json.dumps(
            {
                "model_name_or_path": str(model_path),
                "results": {task: {"toy_score": score, "backend": model_flags}},
                "configs": {task: {"num_fewshot": n_shot}},
            }
        )
    )


def parse_results(data):
    results = data.get("results", {})
    if not isinstance(results, dict):
        return None
    for tname, tres in results.items():
        if isinstance(tname, str) and tname.startswith("toy_task") and "toy_score" in tres:
            n_shot = data.get("configs", {}).get(tname, {}).get("num_fewshot", 0)
            return (data.get("model_name_or_path", "?"), tname, int(n_shot), dict(tres))
    return None
"""


@pytest.fixture
def toy_plugin(tmp_path, monkeypatch):
    import oellm.contrib as contrib_pkg
    from oellm import registry

    plug = tmp_path / "toyplug"
    plug.mkdir()
    (plug / "__init__.py").write_text("")
    (plug / "suite.py").write_text(SUITE_SRC)
    monkeypatch.setattr(
        contrib_pkg, "__path__", list(contrib_pkg.__path__) + [str(tmp_path)]
    )
    registry._discover.cache_clear()
    yield
    registry._discover.cache_clear()
    sys.modules.pop("oellm.contrib.toyplug.suite", None)
    sys.modules.pop("oellm.contrib.toyplug", None)


class TestPluginProtocolConformance:
    def test_discovery_and_engine_task_name(self, toy_plugin):
        from oellm.registry import get_all_task_groups, get_suite

        assert get_suite("toy_suite").SUITE_NAME == "toy_suite"
        merged = get_all_task_groups()
        group = merged["task_groups"]["toy-task"]
        # engine_task_name override lands in the task entry (jobs.csv name)
        assert group["tasks"][0]["task"] == "toy_task_engine"
        # ... and keys the task_metrics mapping consistently
        assert merged["task_metrics"]["toy_task_engine"] == "toy_score"

    def test_lmms_adapter_override_consulted_first(self, toy_plugin):
        from oellm.constants import detect_lmms_model_type

        assert detect_lmms_model_type("org/Toy-VLM-7B") == "toy_adapter"

    def test_model_flags_via_adapter_in_runner(self, toy_plugin):
        from oellm.constants import EvaluationJob
        from oellm.runner import EvalRunner

        job = EvaluationJob(
            model_path="ToyModel-1B",
            task_path="toy_task_engine",
            n_shot=0,
            eval_suite="toy_suite",
        )
        assert EvalRunner().resolve_suite(job) == "toy_suite:toy_backend"

    def test_run_output_collected_via_parse_results(self, toy_plugin, tmp_path):
        from oellm.registry import get_suite
        from oellm.results import collect_results

        results_dir = tmp_path / "run" / "results"
        results_dir.mkdir(parents=True)
        get_suite("toy_suite").run(
            model_path="ToyModel-1B",
            task="toy_task_engine",
            n_shot=0,
            output_path=results_dir / "out.json",
            model_flags="toy_backend",
            env={},
        )
        out_csv = tmp_path / "run" / "eval.csv"
        collect_results(str(tmp_path / "run"), str(out_csv))
        rows = list(csv.DictReader(open(out_csv)))
        assert [(r["task"], r["metric_name"], float(r["performance"])) for r in rows] == [
            ("toy_task_engine", "toy_score", 0.5)
        ]

    def test_core_api_version_marker(self):
        from oellm.core import CORE_API_VERSION

        assert CORE_API_VERSION == "1.0"


class TestTabFactTask:
    def test_group_wired_and_metric_mapped(self):
        from oellm.results import _load_task_metrics
        from oellm.task_groups import _expand_task_groups

        expanded = _expand_task_groups(["tabular-tabfact"])
        assert [(r.task, r.n_shot, r.suite) for r in expanded] == [
            ("tabfact", 0, "lm-eval-harness")
        ]
        assert _load_task_metrics()["tabfact"] == "acc"

    def test_prompt_serializes_table(self):
        tabfact_utils = _load_task_utils("tabfact")
        doc = {
            "table_caption": "medals",
            "table_text": "nation#gold#silver\nnorway#10#8\nsweden#7#9",
            "statement": "norway won 10 gold medals",
            "label": 1,
        }
        text = tabfact_utils.doc_to_text(doc)
        assert "nation | gold | silver" in text
        assert "norway | 10 | 8" in text
        assert "Statement: norway won 10 gold medals" in text
        assert text.endswith("Answer:")
        assert json.dumps(doc)  # doc stays JSON-serializable


class TestTimeSeriesExamTask:
    def test_group_wired_and_metric_mapped(self):
        from oellm.results import _load_task_metrics
        from oellm.task_groups import _expand_task_groups

        expanded = _expand_task_groups(["timeseries-tsexam"])
        assert [(r.task, r.n_shot, r.suite) for r in expanded] == [
            ("timeseriesexam", 0, "lm-eval-harness")
        ]
        assert _load_task_metrics()["timeseriesexam"] == "acc"

    def test_prompt_subsamples_and_letters(self):
        ts_utils = _load_task_utils("timeseriesexam")
        doc = {
            "question": "Do the two series share a distribution?",
            "options": ["No, they differ", "Yes, they match"],
            "answer": "Yes, they match",
            "ts1": [float(i) for i in range(1000)],
            "ts2": [1.0, 2.0, 3.0],
        }
        text = ts_utils.doc_to_text(doc)
        assert "Series A (1000 points, uniformly subsampled to 128)" in text
        assert "Series B (3 points" in text
        assert "A. No, they differ" in text and "B. Yes, they match" in text
        assert text.endswith("Answer:")
        # subsample really capped the series
        first_block = text.split("Series B")[0]
        assert first_block.count(",") <= 130
        assert ts_utils.doc_to_choice(doc) == [" A", " B"]
        assert ts_utils.doc_to_target(doc) == 1
