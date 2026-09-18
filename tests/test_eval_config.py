"""Tests for EvalConfig: YAML loading, CLI construction, merge, and validation."""

import textwrap

import pytest

from oellm.config import EvalConfig, SlurmOverrides

# ---------------------------------------------------------------------------
# SlurmOverrides
# ---------------------------------------------------------------------------


class TestSlurmOverrides:
    def test_to_template_var_dict_empty(self):
        s = SlurmOverrides()
        assert s.to_template_var_dict() == {}

    def test_to_template_var_dict_partial(self):
        s = SlurmOverrides(partition="dev-g", time_limit="02:00:00")
        d = s.to_template_var_dict()
        assert d == {"PARTITION": "dev-g", "TIME": "02:00:00"}
        assert "ACCOUNT" not in d
        assert "GPUS_PER_NODE" not in d

    def test_to_template_var_dict_full(self):
        s = SlurmOverrides(
            partition="gpu", account="FOO", gpus_per_node=4, time_limit="06:00:00"
        )
        d = s.to_template_var_dict()
        assert d == {
            "PARTITION": "gpu",
            "ACCOUNT": "FOO",
            "GPUS_PER_NODE": "4",
            "TIME": "06:00:00",
        }


# ---------------------------------------------------------------------------
# from_yaml
# ---------------------------------------------------------------------------


class TestFromYaml:
    def test_full_config(self, tmp_path):
        cfg_file = tmp_path / "eval.yaml"
        cfg_file.write_text(
            textwrap.dedent("""\
            models:
              - "llava-hf/llava-1.5-7b-hf"
              - "Qwen/Qwen2-VL-7B"
            task_groups:
              - "image-vqa"
              - "image-ocrbench"
            n_shot: 0
            trust_remote_code: true
            venv_path: "~/elliot-venv"
            slurm:
              max_array_len: 64
              partition: "gpu"
              time_limit: "06:00:00"
            """)
        )
        cfg = EvalConfig.from_yaml(cfg_file)

        assert cfg.models == ["llava-hf/llava-1.5-7b-hf", "Qwen/Qwen2-VL-7B"]
        assert cfg.task_groups == ["image-vqa", "image-ocrbench"]
        assert cfg.n_shot == [0]
        assert cfg.trust_remote_code is True
        assert cfg.venv_path == "~/elliot-venv"
        assert cfg.slurm.max_array_len == 64
        assert cfg.slurm.partition == "gpu"
        assert cfg.slurm.time_limit == "06:00:00"

    def test_minimal_config(self, tmp_path):
        cfg_file = tmp_path / "minimal.yaml"
        cfg_file.write_text(
            textwrap.dedent("""\
            models:
              - "meta-llama/Llama-2-7b"
            task_groups:
              - "open-sci-0.01"
            """)
        )
        cfg = EvalConfig.from_yaml(cfg_file)

        assert cfg.models == ["meta-llama/Llama-2-7b"]
        assert cfg.task_groups == ["open-sci-0.01"]
        assert cfg.n_shot is None
        assert cfg.venv_path is None
        assert cfg.slurm.max_array_len == 128  # default

    def test_scalar_n_shot_becomes_list(self, tmp_path):
        cfg_file = tmp_path / "scalar.yaml"
        cfg_file.write_text(
            textwrap.dedent("""\
            models:
              - "some-model"
            tasks:
              - "hellaswag"
            n_shot: 5
            """)
        )
        cfg = EvalConfig.from_yaml(cfg_file)
        assert cfg.n_shot == [5]

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            EvalConfig.from_yaml("/nonexistent/path.yaml")

    def test_empty_yaml(self, tmp_path):
        cfg_file = tmp_path / "empty.yaml"
        cfg_file.write_text("")
        cfg = EvalConfig.from_yaml(cfg_file)
        assert cfg.models is None
        assert cfg.task_groups is None


# ---------------------------------------------------------------------------
# from_cli_kwargs
# ---------------------------------------------------------------------------


class TestFromCliKwargs:
    def test_basic_cli(self):
        cfg = EvalConfig.from_cli_kwargs(
            models="model-a,model-b",
            task_groups="image-vqa",
            venv_path="~/venv",
        )
        assert cfg.models == ["model-a", "model-b"]
        assert cfg.task_groups == ["image-vqa"]
        assert cfg.venv_path == "~/venv"

    def test_n_shot_int(self):
        cfg = EvalConfig.from_cli_kwargs(models="m", tasks="t", n_shot=5)
        assert cfg.n_shot == [5]

    def test_n_shot_list(self):
        cfg = EvalConfig.from_cli_kwargs(models="m", tasks="t", n_shot=[0, 5])
        assert cfg.n_shot == [0, 5]

    def test_slurm_template_var_json(self):
        cfg = EvalConfig.from_cli_kwargs(
            models="m",
            task_groups="g",
            slurm_template_var='{"PARTITION":"dev-g","ACCOUNT":"FOO","TIME":"02:00:00","GPUS_PER_NODE":2}',
        )
        assert cfg.slurm.partition == "dev-g"
        assert cfg.slurm.account == "FOO"
        assert cfg.slurm.time_limit == "02:00:00"
        assert cfg.slurm.gpus_per_node == 2

    def test_slurm_template_var_invalid_json(self):
        with pytest.raises(ValueError, match="valid JSON"):
            EvalConfig.from_cli_kwargs(models="m", slurm_template_var="not-json")

    def test_slurm_template_var_not_dict(self):
        with pytest.raises(ValueError, match="JSON object"):
            EvalConfig.from_cli_kwargs(models="m", slurm_template_var="[1, 2, 3]")


# ---------------------------------------------------------------------------
# merge (YAML base + CLI overrides)
# ---------------------------------------------------------------------------


class TestMerge:
    def test_cli_overrides_yaml(self, tmp_path):
        cfg_file = tmp_path / "base.yaml"
        cfg_file.write_text(
            textwrap.dedent("""\
            models:
              - "yaml-model"
            task_groups:
              - "image-vqa"
            venv_path: "~/yaml-venv"
            limit: 100
            slurm:
              partition: "yaml-partition"
              time_limit: "10:00:00"
            """)
        )
        yaml_cfg = EvalConfig.from_yaml(cfg_file)
        cli_cfg = EvalConfig.from_cli_kwargs(
            models="cli-model",
            venv_path="~/cli-venv",
        )

        merged = yaml_cfg.merge(cli_cfg)

        # CLI wins for models and venv_path
        assert merged.models == ["cli-model"]
        assert merged.venv_path == "~/cli-venv"
        # YAML preserved for task_groups, limit, slurm
        assert merged.task_groups == ["image-vqa"]
        assert merged.limit == 100
        assert merged.slurm.partition == "yaml-partition"
        assert merged.slurm.time_limit == "10:00:00"

    def test_cli_slurm_overrides_yaml_slurm(self, tmp_path):
        cfg_file = tmp_path / "base.yaml"
        cfg_file.write_text(
            textwrap.dedent("""\
            models:
              - "m"
            task_groups:
              - "g"
            slurm:
              partition: "yaml-part"
              account: "yaml-acct"
              time_limit: "10:00:00"
            """)
        )
        yaml_cfg = EvalConfig.from_yaml(cfg_file)
        cli_cfg = EvalConfig.from_cli_kwargs(
            slurm_template_var='{"PARTITION":"cli-part"}',
        )

        merged = yaml_cfg.merge(cli_cfg)

        # CLI overrides partition, YAML keeps account and time_limit
        assert merged.slurm.partition == "cli-part"
        assert merged.slurm.account == "yaml-acct"
        assert merged.slurm.time_limit == "10:00:00"

    def test_no_cli_uses_yaml_entirely(self, tmp_path):
        cfg_file = tmp_path / "base.yaml"
        cfg_file.write_text(
            textwrap.dedent("""\
            models:
              - "yaml-model"
            task_groups:
              - "g"
            limit: 50
            """)
        )
        yaml_cfg = EvalConfig.from_yaml(cfg_file)
        cli_cfg = EvalConfig.from_cli_kwargs()  # all defaults

        merged = yaml_cfg.merge(cli_cfg)

        assert merged.models == ["yaml-model"]
        assert merged.task_groups == ["g"]
        assert merged.limit == 50


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


class TestValidate:
    def test_valid_task_groups(self):
        cfg = EvalConfig(
            models=["m"],
            task_groups=["g"],
        )
        cfg.validate()  # should not raise

    def test_valid_tasks_with_n_shot(self):
        cfg = EvalConfig(
            models=["m"],
            tasks=["t"],
            n_shot=[0],
        )
        cfg.validate()  # should not raise

    def test_valid_csv_mode(self, tmp_path):
        csv = tmp_path / "jobs.csv"
        csv.write_text("model_path,task_path,n_shot\nm,t,0\n")
        cfg = EvalConfig(eval_csv_path=str(csv))
        cfg.validate()  # should not raise

    def test_no_models(self):
        cfg = EvalConfig(task_groups=["g"])
        with pytest.raises(ValueError, match="At least one model"):
            cfg.validate()

    def test_no_tasks_or_groups(self):
        cfg = EvalConfig(models=["m"])
        with pytest.raises(ValueError, match="task_groups or tasks"):
            cfg.validate()

    def test_tasks_without_n_shot(self):
        cfg = EvalConfig(models=["m"], tasks=["t"])
        with pytest.raises(ValueError, match="n_shot is required"):
            cfg.validate()

    def test_negative_n_shot(self):
        cfg = EvalConfig(models=["m"], tasks=["t"], n_shot=[-1])
        with pytest.raises(ValueError, match="non-negative"):
            cfg.validate()

    def test_csv_with_models_conflicts(self, tmp_path):
        csv = tmp_path / "jobs.csv"
        csv.write_text("model_path,task_path,n_shot\nm,t,0\n")
        cfg = EvalConfig(eval_csv_path=str(csv), models=["m"])
        with pytest.raises(ValueError, match="Cannot specify"):
            cfg.validate()

    def test_csv_not_found(self):
        cfg = EvalConfig(eval_csv_path="/nonexistent/jobs.csv")
        with pytest.raises(FileNotFoundError, match="eval_csv_path"):
            cfg.validate()


# ---------------------------------------------------------------------------
# slurm_template_var_json
# ---------------------------------------------------------------------------


class TestSlurmTemplateVarJson:
    def test_empty_returns_none(self):
        cfg = EvalConfig(models=["m"], task_groups=["g"])
        assert cfg.slurm_template_var_json is None

    def test_with_overrides(self):
        cfg = EvalConfig(
            models=["m"],
            task_groups=["g"],
            slurm=SlurmOverrides(partition="dev-g", time_limit="02:00:00"),
        )
        import json

        result = json.loads(cfg.slurm_template_var_json)
        assert result == {"PARTITION": "dev-g", "TIME": "02:00:00"}


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------


class TestModelConfig:
    def test_modelconfig_defaults(self):
        from oellm.config import ModelConfig

        m = ModelConfig(path="/some/model")
        assert m.path == "/some/model"
        assert m.name is None

    def test_yaml_plain_strings_backward_compat(self, tmp_path):
        cfg_file = tmp_path / "eval.yaml"
        cfg_file.write_text(
            "models:\n  - 'EleutherAI/pythia-70m'\ntask_groups:\n  - 'open-sci-0.01'\n"
        )
        cfg = EvalConfig.from_yaml(str(cfg_file))
        assert cfg.models == ["EleutherAI/pythia-70m"]

    def test_yaml_named_model_config(self, tmp_path):
        from oellm.config import ModelConfig

        cfg_file = tmp_path / "eval.yaml"
        cfg_file.write_text(
            "models:\n  - path: '/hpc/models/llava'\n    name: 'LLaVA-1.5'\n"
            "task_groups:\n  - 'open-sci-0.01'\n"
        )
        cfg = EvalConfig.from_yaml(str(cfg_file))
        assert len(cfg.models) == 1
        m = cfg.models[0]
        assert isinstance(m, ModelConfig)
        assert m.path == "/hpc/models/llava"
        assert m.name == "LLaVA-1.5"

    def test_yaml_mixed_models_list(self, tmp_path):
        from oellm.config import ModelConfig

        cfg_file = tmp_path / "eval.yaml"
        cfg_file.write_text(
            "models:\n  - path: '/hpc/models/llava'\n    name: 'LLaVA'\n"
            "  - 'EleutherAI/pythia-70m'\n"
            "task_groups:\n  - 'open-sci-0.01'\n"
        )
        cfg = EvalConfig.from_yaml(str(cfg_file))
        assert len(cfg.models) == 2
        assert isinstance(cfg.models[0], ModelConfig)
        assert cfg.models[1] == "EleutherAI/pythia-70m"

    def test_model_paths_helper(self, tmp_path):
        cfg_file = tmp_path / "eval.yaml"
        cfg_file.write_text(
            "models:\n  - path: '/hpc/models/llava'\n  - 'EleutherAI/pythia-70m'\n"
            "task_groups:\n  - 'open-sci-0.01'\n"
        )
        cfg = EvalConfig.from_yaml(str(cfg_file))
        paths = cfg._model_paths()
        assert paths == ["/hpc/models/llava", "EleutherAI/pythia-70m"]

    def test_cli_kwargs_models_still_str(self):
        cfg = EvalConfig.from_cli_kwargs(models="a,b", task_groups="open-sci-0.01")
        assert cfg.models == ["a", "b"]

    def test_ensure_model_list_missing_path_key(self, tmp_path):
        cfg_file = tmp_path / "eval.yaml"
        cfg_file.write_text(
            "models:\n  - name: 'no-path'\ntask_groups:\n  - 'open-sci-0.01'\n"
        )
        with pytest.raises(KeyError):
            EvalConfig.from_yaml(str(cfg_file))


# ---------------------------------------------------------------------------
# merge provenance: explicit CLI values override YAML even at the default
# ---------------------------------------------------------------------------


class TestMergeProvenance:
    def _yaml_cfg(self, **overrides):
        raw = {"models": ["m"], "task_groups": ["g"], **overrides}
        return EvalConfig._from_dict(raw)

    def test_no_dry_run_overrides_yaml_dry_run(self):
        yaml_cfg = self._yaml_cfg(dry_run=True)
        cli_cfg = EvalConfig.from_cli_kwargs(dry_run=False)
        assert yaml_cfg.merge(cli_cfg).dry_run is False

    def test_trust_remote_code_overrides_yaml_false(self):
        yaml_cfg = self._yaml_cfg(trust_remote_code=False)
        cli_cfg = EvalConfig.from_cli_kwargs(trust_remote_code=True)
        assert yaml_cfg.merge(cli_cfg).trust_remote_code is True

    def test_unprovided_bool_keeps_yaml_value(self):
        yaml_cfg = self._yaml_cfg(dry_run=True, trust_remote_code=False)
        cli_cfg = EvalConfig.from_cli_kwargs()
        merged = yaml_cfg.merge(cli_cfg)
        assert merged.dry_run is True
        assert merged.trust_remote_code is False

    def test_default_valued_max_array_len_overrides_yaml(self):
        yaml_cfg = self._yaml_cfg(slurm={"max_array_len": 64})
        cli_cfg = EvalConfig.from_cli_kwargs(max_array_len=128)
        assert yaml_cfg.merge(cli_cfg).slurm.max_array_len == 128

    def test_unprovided_max_array_len_keeps_yaml(self):
        yaml_cfg = self._yaml_cfg(slurm={"max_array_len": 64})
        cli_cfg = EvalConfig.from_cli_kwargs()
        assert yaml_cfg.merge(cli_cfg).slurm.max_array_len == 64


# ---------------------------------------------------------------------------
# slurm_template_var passthrough: unmodeled keys must survive the round-trip
# ---------------------------------------------------------------------------


class TestSlurmTemplateVarPassthrough:
    def test_slurm_mem_round_trips(self):
        cfg = EvalConfig.from_cli_kwargs(
            slurm_template_var='{"SLURM_MEM":"123G","PARTITION":"dev-g"}'
        )
        assert cfg.slurm.extra_template_vars == {"SLURM_MEM": "123G"}
        assert cfg.slurm.partition == "dev-g"

        rendered = cfg.slurm_template_var_json
        assert rendered is not None
        assert '"SLURM_MEM": "123G"' in rendered or '"SLURM_MEM":"123G"' in rendered

    def test_extras_survive_merge(self):
        yaml_cfg = EvalConfig._from_dict({"models": ["m"], "task_groups": ["g"]})
        cli_cfg = EvalConfig.from_cli_kwargs(
            slurm_template_var='{"SLURM_MEM":"96G","CPUS_PER_TASK":"8"}'
        )
        merged = yaml_cfg.merge(cli_cfg)
        assert merged.slurm.extra_template_vars == {
            "SLURM_MEM": "96G",
            "CPUS_PER_TASK": "8",
        }

    def test_known_keys_not_duplicated_into_extras(self):
        cfg = EvalConfig.from_cli_kwargs(
            slurm_template_var='{"PARTITION":"p","ACCOUNT":"a","GPUS_PER_NODE":2,"TIME":"01:00:00"}'
        )
        assert cfg.slurm.extra_template_vars == {}
        d = cfg.slurm.to_template_var_dict()
        assert d == {
            "PARTITION": "p",
            "ACCOUNT": "a",
            "GPUS_PER_NODE": "2",
            "TIME": "01:00:00",
        }


# ---------------------------------------------------------------------------
# unknown YAML keys warn instead of vanishing silently
# ---------------------------------------------------------------------------


class TestUnknownYamlKeys:
    def test_unknown_top_level_key_warns(self, tmp_path, caplog):
        cfg_file = tmp_path / "eval.yaml"
        cfg_file.write_text("models:\n  - 'm'\ntask_groups:\n  - 'g'\nn_shots: 5\n")
        with caplog.at_level("WARNING"):
            EvalConfig.from_yaml(str(cfg_file))
        assert "n_shots" in caplog.text

    def test_unknown_slurm_key_warns(self, tmp_path, caplog):
        cfg_file = tmp_path / "eval.yaml"
        cfg_file.write_text(
            "models:\n  - 'm'\ntask_groups:\n  - 'g'\nslurm:\n  partion: 'x'\n"
        )
        with caplog.at_level("WARNING"):
            EvalConfig.from_yaml(str(cfg_file))
        assert "partion" in caplog.text
