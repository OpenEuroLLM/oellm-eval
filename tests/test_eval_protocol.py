import json
from pathlib import Path
from types import SimpleNamespace
import subprocess

import pytest

from oellm.eval_protocol import freeze, resolve_job
from oellm.protocol_runtime import ensure_context, generation_settings, verify_snapshot
from test_vllm_schedule import render


def test_legacy_inventory_and_calibrated_mapping():
    root = Path(__file__).resolve().parents[1]
    tasks = json.loads((root / "docs/releases/v0.01.json").read_text())["standard_suite"]["tasks"]
    for task in tasks:
        job = dict(model_path="model", task_path=task["task"], n_shot=task["n_shot"], eval_suite=task["suite"])
        assert resolve_job(job, "v0.01") == job
        new = resolve_job(job, "v0.02")
        if task["task"] not in ("gsm8k", "squadv2", "GPQADiamond"):
            assert new == job
    assert len(tasks) == 77
    job = dict(task_path="gsm8k", n_shot=4, eval_suite="lm_eval")
    assert resolve_job(job, "v0.02")["task_path"] == "gsm8k_cot_numeric_v1"
    with pytest.raises(ValueError):
        resolve_job(dict(job, n_shot=0), "v0.02")


def test_snapshot_drift_fails_before_evaluator_import(tmp_path):
    root = freeze(tmp_path / "snapshot", "v0.02", [], [])
    assert verify_snapshot(root)["version"] == "v0.02"
    (root / "evaluation_policies.py").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        verify_snapshot(root)


def test_no_context_truncation_and_no_mutation():
    model = SimpleNamespace(tok_encode=lambda text: list(text), max_length=8, max_gen_toks=4)
    request = SimpleNamespace(args=("1234", dict(max_gen_toks=4)))
    ensure_context(model, [request])
    model.max_length = 7
    with pytest.raises(ValueError, match="requires 8 tokens"):
        ensure_context(model, [request])
    assert request.args == ("1234", dict(max_gen_toks=4))
    old = dict(max_new_tokens=1024, temperature=.7, do_sample=True)
    assert generation_settings("HumanEval", old)["max_gen_toks"] == 4096
    assert old["max_new_tokens"] == 1024
    assert generation_settings("MATH500", old, 16384) == dict(
        max_gen_toks=16384, temperature=0., do_sample=False, until=["<|im_end|>"])


@pytest.mark.parametrize("suite", ["lm_eval", "evalchemy"])
@pytest.mark.parametrize("container", [True, False])
def test_frozen_runtime_is_the_actual_scheduled_entry(tmp_path, monkeypatch, suite, container):
    script = render(tmp_path, monkeypatch, suite=suite, container=container,
                    model_backend="vllm", data_parallel_size=4, eval_version="v0.02")
    subprocess.run(["bash", "-n", str(script)], check=True)
    subprocess.run(["bash", str(script)], check=True, capture_output=True)
    argv = (tmp_path / "argv").read_text().splitlines()
    runtime = next(x for x in argv if x.endswith("/protocol_runtime.py"))
    assert Path(runtime).is_file()
    assert argv[argv.index("--module") + 1] == ("lm_eval" if suite == "lm_eval" else "eval.eval")
    assert verify_snapshot(Path(runtime).parent)["version"] == "v0.02"
    if container:
        assert str(Path(runtime).parent) + ":" + str(Path(runtime).parent) + ":ro" in argv[argv.index("--bind") + 1]


@pytest.mark.parametrize("options", [dict(limit=10), dict(model_backend="hf"),
    dict(calibrated_output_tokens=0), dict(lm_eval_include_path="/custom")])
def test_incompatible_release_options_fail(tmp_path, monkeypatch, options):
    kwargs = dict(model_backend="vllm", eval_version="v0.02")
    kwargs.update(options)
    with pytest.raises(ValueError):
        render(tmp_path, monkeypatch, **kwargs)


def test_collector_keeps_versions_and_output_variants_separate(tmp_path):
    from oellm.main import collect_results
    import pandas as pd
    for name, version, budget, score in [("old", "v0.01", None, .1),
                                        ("new", "v0.02", None, .2),
                                        ("new16k", "v0.02", 16384, .3)]:
        folder = tmp_path / name
        folder.mkdir()
        freeze(folder / "protocol", version, [], [], budget)
        (folder / "jobs.csv").write_text("model_path,task_path,n_shot,eval_suite\nm,HumanEval,0,evalchemy\n")
        (folder / "result.json").write_text(json.dumps(dict(model_name="m", results={"HumanEval": {"pass@1": score}}, **{"n-shot": {"HumanEval": 0}})))
    out = tmp_path / "collected.csv"
    collect_results(str(tmp_path), str(out), check=True)
    rows = pd.read_csv(out)
    assert len(rows) == 3
    assert set(zip(rows.eval_version, rows.protocol_variant)) == {
        ("v0.01", "default"), ("v0.02", "default"), ("v0.02", "output-16384")}


def test_collector_selects_numeric_flexible_not_first_filter(tmp_path):
    from oellm.main import collect_results
    import pandas as pd
    freeze(tmp_path / "protocol", "v0.02", [], [])
    (tmp_path / "result.json").write_text(json.dumps(dict(model_name="m", results={
        "gsm8k_cot_numeric_v1": {"exact_match,strict-match": .81,
                                "numeric_match,strict-match": .84,
                                "numeric_match,flexible-extract": .90}})))
    out = tmp_path / "scores.csv"
    collect_results(str(tmp_path), str(out))
    row = pd.read_csv(out).iloc[0]
    assert row.performance == .90 and row.metric_name == "numeric_match,flexible-extract"


def test_real_local_model_paths_are_serializable(tmp_path, monkeypatch):
    model = tmp_path / "model with spaces"
    monkeypatch.setattr("oellm.main._expand_local_model_paths", lambda _: [model])
    script = render(tmp_path, monkeypatch, model_backend="vllm", eval_version="v0.02")
    manifest = json.loads((script.parent / "protocol/manifest.json").read_text())
    assert manifest["resolved_jobs"][0]["model_path"] == str(model)
