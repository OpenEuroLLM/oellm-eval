import json
import subprocess
from unittest.mock import patch

import pytest

from oellm.results import read_result


def _completed_process(payload: dict) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["inspect"],
        returncode=0,
        stdout=json.dumps(payload),
        stderr="",
    )


def _header(*, status: str = "success", model: str = "hf/org/model") -> dict:
    return {
        "status": status,
        "eval": {
            "model": model,
            "model_args": {"model_path": "/models/local"},
            "task": "inspect_evals/bbq",
        },
        "results": {
            "scores": [
                {
                    "metrics": {
                        "accuracy": {"value": 0.75},
                        "stderr": {"value": 0.1},
                        "label": {"value": "ignored"},
                    }
                }
            ]
        },
    }


def test_read_result_normalizes_hub_model_and_metrics(tmp_path):
    eval_file = tmp_path / "result.eval"
    with patch(
        "oellm.results.subprocess.run",
        return_value=_completed_process(_header()),
    ) as run:
        result = read_result(eval_file)

    assert result == {
        "model_name": "org/model",
        "results": {"inspect_evals/bbq": {"accuracy": 0.75, "stderr": 0.1}},
        "n-shot": {"inspect_evals/bbq": 0},
    }
    run.assert_called_once_with(
        ["inspect", "log", "dump", "--header-only", str(eval_file)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_read_result_normalizes_local_model(tmp_path):
    with patch(
        "oellm.results.subprocess.run",
        return_value=_completed_process(_header(model="hf/local")),
    ):
        result = read_result(tmp_path / "result.eval")

    assert result["model_name"] == "/models/local"


def test_read_result_skips_failed_log(tmp_path, caplog):
    with patch(
        "oellm.results.subprocess.run",
        return_value=_completed_process(_header(status="error")),
    ):
        result = read_result(tmp_path / "result.eval")

    assert result is None
    assert "Ignoring non-successful Inspect log" in caplog.text


def test_read_result_reports_missing_cli(tmp_path):
    with (
        patch("oellm.results.subprocess.run", side_effect=FileNotFoundError),
        pytest.raises(RuntimeError, match="inspect.*not available"),
    ):
        read_result(tmp_path / "result.eval")


def test_read_result_reports_cli_failure(tmp_path):
    error = subprocess.CalledProcessError(2, ["inspect"], stderr="unable to read log")
    with (
        patch("oellm.results.subprocess.run", side_effect=error),
        pytest.raises(RuntimeError, match="unable to read log"),
    ):
        read_result(tmp_path / "result.eval")


def test_read_result_reports_invalid_json(tmp_path):
    process = subprocess.CompletedProcess(
        args=["inspect"], returncode=0, stdout="not-json", stderr=""
    )
    with (
        patch("oellm.results.subprocess.run", return_value=process),
        pytest.raises(RuntimeError, match="invalid JSON"),
    ):
        read_result(tmp_path / "result.eval")


def test_read_result_preserves_json_payload_without_inspect(tmp_path):
    payload = {
        "config_general": {"model_name": "org/model"},
        "results": {"task|0": {"acc": 0.5}, "all": {"acc": 0.5}},
        "groups": {"group": {"acc": 0.5}},
        "group_subtasks": {"group": ["task|0"]},
        "n-shot": {"task|0": 0},
    }
    path = tmp_path / "result.json"
    path.write_text(json.dumps(payload))
    with patch("oellm.results.subprocess.run") as run:
        assert read_result(path) == payload
    run.assert_not_called()


def test_collect_mixed_suites_and_group_aggregates(tmp_path):
    import pandas as pd

    from oellm.main import collect_results

    (tmp_path / "inspect.eval").touch()
    (tmp_path / "light.json").write_text(
        json.dumps(
            {
                "config_general": {"model_name": "org/model"},
                "results": {"light_task|0": {"acc": 0.6}, "all": {"acc": 0.6}},
            }
        )
    )
    (tmp_path / "lm.json").write_text(
        json.dumps(
            {
                "model_name": "org/model",
                "results": {"child": {"acc": 0.2}},
                "groups": {"group": {"acc": 0.8}},
                "group_subtasks": {"group": ["child"]},
                "n-shot": {"child": 5},
            }
        )
    )
    pd.DataFrame(
        [
            {"model_path": "org/model", "task_path": task, "n_shot": shots}
            for task, shots in [("inspect_evals/bbq", 0), ("light_task", 0), ("group", 5)]
        ]
    ).to_csv(tmp_path / "jobs.csv", index=False)
    output = tmp_path / "out.csv"
    with patch(
        "oellm.results.subprocess.run", return_value=_completed_process(_header())
    ):
        collect_results(str(tmp_path), str(output), check=True)

    rows = pd.read_csv(output).to_dict("records")
    assert rows == [
        {
            "model_name": "org/model",
            "task": "inspect_evals/bbq",
            "n_shot": 0,
            "performance": 0.75,
            "metric_name": "accuracy",
        },
        {
            "model_name": "org/model",
            "task": "light_task",
            "n_shot": 0,
            "performance": 0.6,
            "metric_name": "acc",
        },
        {
            "model_name": "org/model",
            "task": "group",
            "n_shot": 5,
            "performance": 0.8,
            "metric_name": "acc",
        },
    ]
    assert not (tmp_path / "out_missing.csv").exists()


@pytest.mark.parametrize("metric", ["custom_score", "stderr"])
def test_collect_inspect_without_primary_metric_keeps_numeric_metrics(tmp_path, metric):
    import pandas as pd

    from oellm.main import collect_results

    payload = _header()
    payload["eval"]["task"] = "inspect_evals/custom"
    payload["results"]["scores"][0]["metrics"] = {metric: {"value": 0.4}}
    (tmp_path / "result.eval").touch()
    pd.DataFrame(
        [
            {
                "model_path": "org/model",
                "task_path": "inspect_evals/custom",
                "n_shot": 0,
            }
        ]
    ).to_csv(tmp_path / "jobs.csv", index=False)
    with patch("oellm.results.subprocess.run", return_value=_completed_process(payload)):
        collect_results(str(tmp_path), str(tmp_path / "out.csv"), check=True)

    rows = pd.read_csv(tmp_path / "out.csv")
    assert rows["metric_name"].tolist() == [metric]
    assert rows["performance"].tolist() == [0.4]
    assert not (tmp_path / "out_missing.csv").exists()


def test_collect_inspect_uses_configured_custom_metric(tmp_path):
    import pandas as pd

    from oellm.main import collect_results

    payload = _header()
    payload["results"]["scores"][0]["metrics"] = {
        "accuracy": {"value": 0.75},
        "custom_score": {"value": 0.9},
    }
    (tmp_path / "result.eval").touch()
    with (
        patch("oellm.results.subprocess.run", return_value=_completed_process(payload)),
        patch(
            "yaml.safe_load",
            return_value={
                "task_metrics": {"inspect_evals/bbq": "custom_score"},
            },
        ),
    ):
        collect_results(str(tmp_path), str(tmp_path / "out.csv"))

    rows = pd.read_csv(tmp_path / "out.csv")
    assert rows["metric_name"].tolist() == ["custom_score"]
    assert rows["performance"].tolist() == [0.9]


def test_read_result_preserves_file_based_task_identity(tmp_path):
    header = _header()
    header["eval"]["task"] = "bbq_reference"
    header["eval"]["task_file"] = "/runs/comparison/tasks.py"
    with patch(
        "oellm.results.subprocess.run",
        return_value=_completed_process(header),
    ):
        result = read_result(tmp_path / "result.eval")
    task = "/runs/comparison/tasks.py@bbq_reference"
    assert result["results"][task]["accuracy"] == 0.75
    assert result["n-shot"] == {task: 0}


def test_collect_bbq_preserves_fairness_metrics_and_completion(tmp_path, monkeypatch):
    monkeypatch.setattr("oellm.bbq._target_locations", lambda: {("Age", "1", "0"): 0})
    import pandas as pd
    from test_bbq import sample

    from oellm.main import collect_results

    payload = _header()
    payload["eval"]["dataset"] = {"name": "bbq"}
    payload["results"]["completed_samples"] = 1
    payload["samples"] = [sample("C", "disambig", "A")]
    (tmp_path / "result.eval").touch()
    pd.DataFrame(
        [{"model_path": "org/model", "task_path": "inspect_evals/bbq", "n_shot": 0}]
    ).to_csv(tmp_path / "jobs.csv", index=False)
    with patch("oellm.results.subprocess.run", return_value=_completed_process(payload)):
        collect_results(str(tmp_path), str(tmp_path / "out.csv"), check=True)
    rows = pd.read_csv(tmp_path / "out.csv").set_index("metric_name")
    assert rows.loc["bbq/Age/disambig/unknown_rate", "performance"] == 1
    assert pd.isna(rows.loc["bbq/Age/disambig/bias_score", "performance"])
    assert not (tmp_path / "out_missing.csv").exists()


def test_bbq_rejects_incomplete_saved_samples(tmp_path):
    payload = _header()
    payload["eval"]["dataset"] = {"name": "bbq"}
    payload["results"]["completed_samples"] = 1
    payload["samples"] = []
    with patch("oellm.results.subprocess.run", return_value=_completed_process(payload)):
        with pytest.raises(ValueError, match="incomplete saved samples"):
            read_result(tmp_path / "result.eval")
