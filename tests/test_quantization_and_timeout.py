"""Tests for the bitsandbytes quantization flags and the per-row timeout guard."""

import json
from unittest.mock import patch

import pytest

from oellm.config import EvalConfig


class TestQuantizationConfig:
    def test_yaml_roundtrip(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        p.write_text("models: [m]\ntask_groups: [image-vqa]\nload_in_4bit: true\n")
        cfg = EvalConfig.from_yaml(p)
        assert cfg.load_in_4bit is True
        assert cfg.load_in_8bit is False

    def test_cli_overrides_yaml(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        p.write_text("models: [m]\ntask_groups: [g]\nload_in_4bit: true\n")
        yaml_cfg = EvalConfig.from_yaml(p)
        cli_cfg = EvalConfig.from_cli_kwargs(load_in_4bit=False)
        merged = yaml_cfg.merge(cli_cfg)
        assert merged.load_in_4bit is False

    def test_mutual_exclusion(self):
        cfg = EvalConfig(
            models=["m"],
            tasks=["t"],
            n_shot=[0],
            load_in_4bit=True,
            load_in_8bit=True,
        )
        with pytest.raises(ValueError, match="mutually exclusive"):
            cfg.validate()


def _schedule(tmp_path, monkeypatch, **kw):
    from oellm.scheduler import schedule_evals

    monkeypatch.setenv("EVAL_OUTPUT_DIR", str(tmp_path))
    with (
        patch("oellm.scheduler._load_cluster_env"),
        patch("oellm.scheduler._num_jobs_in_queue", return_value=0),
    ):
        schedule_evals(dry_run=True, skip_checks=True, **kw)
    sbatch = next(tmp_path.glob("**/submit_evals.sbatch")).read_text()
    prov = json.loads(next(tmp_path.glob("**/provenance.json")).read_text())
    return sbatch, prov


class TestQuantizationScheduling:
    def test_4bit_reaches_engines_and_provenance(self, tmp_path, monkeypatch):
        sbatch, prov = _schedule(
            tmp_path,
            monkeypatch,
            models="org/m",
            tasks="hellaswag",
            n_shot=0,
            load_in_4bit=True,
        )
        # lm_eval + lmms_eval + evalchemy model_args each carry the flag
        assert sbatch.count(",load_in_4bit=True") == 3
        assert 'export OELLM_QUANTIZATION="4bit"' in sbatch
        assert prov["quantization"] == "4bit"

    def test_default_is_full_precision(self, tmp_path, monkeypatch):
        sbatch, prov = _schedule(
            tmp_path, monkeypatch, models="org/m", tasks="hellaswag", n_shot=0
        )
        assert "load_in_4bit" not in sbatch
        assert "load_in_8bit" not in sbatch
        assert 'export OELLM_QUANTIZATION=""' in sbatch
        assert prov["quantization"] is None

    def test_both_flags_rejected(self, tmp_path, monkeypatch):
        with pytest.raises(ValueError, match="mutually exclusive"):
            _schedule(
                tmp_path,
                monkeypatch,
                models="org/m",
                tasks="t",
                n_shot=0,
                load_in_4bit=True,
                load_in_8bit=True,
            )


class TestRowTimeout:
    def test_helper_present_and_wraps_engines(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ROW_TIMEOUT", raising=False)
        sbatch, prov = _schedule(
            tmp_path, monkeypatch, models="org/m", tasks="hellaswag", n_shot=0
        )
        assert "_maybe_timeout()" in sbatch  # helper defined
        # wraps: run_python (venv + container), lighteval (venv + container),
        # evalchemy
        assert sbatch.count("_maybe_timeout ") >= 5
        assert prov["row_timeout"] is None

    def test_serial_slice_without_timeout_warns(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("QUEUE_LIMIT", "1")
        monkeypatch.delenv("ROW_TIMEOUT", raising=False)
        _schedule(
            tmp_path,
            monkeypatch,
            models="org/m",
            tasks="hellaswag,winogrande",
            n_shot=0,
        )
        out = " ".join(capsys.readouterr().out.split())
        assert "no per-row timeout" in out

    def test_row_timeout_recorded_in_provenance(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ROW_TIMEOUT", "2h")
        _, prov = _schedule(
            tmp_path, monkeypatch, models="org/m", tasks="hellaswag", n_shot=0
        )
        assert prov["row_timeout"] == "2h"

    def test_submitter_recorded(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OELLM_SUBMITTED_BY", "ci-bot")
        _, prov = _schedule(
            tmp_path, monkeypatch, models="org/m", tasks="hellaswag", n_shot=0
        )
        assert prov["submitted_by"] == "ci-bot"


class TestEngineVersionProvenance:
    def test_versions_probed_from_venv(self, tmp_path, monkeypatch):
        from oellm.scheduler import schedule_evals

        probed = []

        def fake_probe(python_bin, module):
            probed.append(module)
            # lighteval "missing" in this venv — must be omitted, not error
            return (module != "lighteval", "9.9.9")

        monkeypatch.setattr("oellm.envcheck.probe_import", fake_probe)
        venv = tmp_path / "venv"
        (venv / "bin").mkdir(parents=True)
        (venv / "bin" / "python").write_text("")
        monkeypatch.setenv("EVAL_OUTPUT_DIR", str(tmp_path))
        with (
            patch("oellm.scheduler._load_cluster_env"),
            patch("oellm.scheduler._num_jobs_in_queue", return_value=0),
            patch("oellm.scheduler._process_model_paths", return_value={}),
            patch("oellm.scheduler._lookup_dataset_specs_for_tasks", return_value=[]),
            patch("oellm.envcheck.check_scheduled_environment"),
            patch("oellm.scheduler.check_scheduled_environment", create=True),
        ):
            import oellm.envcheck as envcheck_mod

            monkeypatch.setattr(
                envcheck_mod, "check_scheduled_environment", lambda *a, **k: None
            )
            schedule_evals(
                models="org/m",
                tasks="hellaswag",
                n_shot=0,
                dry_run=True,
                venv_path=str(venv),
            )
        prov = json.loads(next(tmp_path.glob("**/provenance.json")).read_text())
        assert prov["engine_versions"] == {
            "lm_eval": "9.9.9",
            "lmms_eval": "9.9.9",
            "transformers": "9.9.9",
            "torch": "9.9.9",
        }
        assert prov["container_image"] is None
        assert set(probed) == {
            "lm_eval",
            "lighteval",
            "lmms_eval",
            "transformers",
            "torch",
        }

    def test_skip_checks_skips_probing(self, tmp_path, monkeypatch):
        sbatch, prov = _schedule(
            tmp_path, monkeypatch, models="org/m", tasks="hellaswag", n_shot=0
        )
        assert prov["engine_versions"] == {}


class TestAdapterRenderedModelArgs:
    """The engine --model_args strings are rendered by DefaultHFAdapter and
    trust_remote_code genuinely governs eval time."""

    def test_trc_true_renders_like_before(self, tmp_path, monkeypatch):
        sbatch, prov = _schedule(
            tmp_path, monkeypatch, models="org/m", tasks="hellaswag", n_shot=0
        )
        assert 'pretrained="$model_path",trust_remote_code=True' in sbatch
        assert "trust_remote_code=True,pretrained=$model_path" in sbatch
        assert "model_name=$model_path,trust_remote_code=True," in sbatch
        assert 'LM_EVAL_TRC="1"' in sbatch
        assert prov["trust_remote_code"] is True
        assert "lm_eval" in prov["engine_model_args"]

    def test_trc_false_disables_eval_time_trust(self, tmp_path, monkeypatch):
        sbatch, prov = _schedule(
            tmp_path,
            monkeypatch,
            models="org/m",
            tasks="hellaswag",
            n_shot=0,
            trust_remote_code=False,
        )
        assert "trust_remote_code=True" not in sbatch
        assert 'pretrained="$model_path",trust_remote_code=False' in sbatch
        assert 'LM_EVAL_TRC=""' in sbatch
        assert "model_name=$model_path,batch_size" in sbatch
        assert prov["trust_remote_code"] is False
