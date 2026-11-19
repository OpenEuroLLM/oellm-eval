import json
import sys
from importlib import metadata

import pytest

from oellm.main import main
from oellm.utils import _expand_local_model_paths, _num_jobs_in_queue, version_info


class TestExpandLocalModelPaths:
    def test_directory_with_safetensors(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "model.safetensors").touch()
        assert _expand_local_model_paths(model_dir) == [model_dir]

    def test_hf_checkpoint_structure(self, tmp_path):
        model_dir = tmp_path / "model"
        iter1 = model_dir / "hf" / "iter_0001000"
        iter2 = model_dir / "hf" / "iter_0002000"
        for d in [iter1, iter2]:
            d.mkdir(parents=True)
            (d / "model.safetensors").touch()

        result = _expand_local_model_paths(model_dir)
        assert set(result) == {iter1, iter2}

    def test_multiple_models_in_subdirs(self, tmp_path):
        base_dir = tmp_path / "models"
        model1 = base_dir / "pythia-70m"
        model2 = base_dir / "pythia-160m"
        for d in [model1, model2]:
            d.mkdir(parents=True)
            (d / "model.safetensors").touch()

        result = _expand_local_model_paths(base_dir)
        assert set(result) == {model1, model2}

    def test_no_safetensors_returns_empty(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").touch()
        assert _expand_local_model_paths(model_dir) == []


class TestNumJobsInQueue:
    def test_counts_jobs(self, monkeypatch):
        class Result:
            returncode = 0
            stdout = "12345\n12346\n12347\n"

        monkeypatch.setattr("oellm.utils.subprocess.run", lambda *a, **kw: Result())
        assert _num_jobs_in_queue() == 3

    def test_returns_zero_on_error(self, monkeypatch):
        class Result:
            returncode = 1
            stdout = ""
            stderr = "error"

        monkeypatch.setattr("oellm.utils.subprocess.run", lambda *a, **kw: Result())
        assert _num_jobs_in_queue() == 0


class TestVersionInfo:
    @staticmethod
    def _install(monkeypatch, direct_url):
        """Pretend oellm-eval was installed with the given PEP 610 record."""

        class Dist:
            version = "0.3.0b1"

            def read_text(self, name):
                return direct_url

        real = metadata.distribution
        monkeypatch.setattr(
            metadata,
            "distribution",
            lambda name: Dist() if name == "oellm-eval" else real(name),
        )

    def test_git_install_names_the_commit(self, monkeypatch):
        commit = "7c24448a1b2c3d4e5f60718293a4b5c6d7e8f901"
        self._install(
            monkeypatch, json.dumps({"vcs_info": {"vcs": "git", "commit_id": commit}})
        )
        assert version_info().startswith("0.3.0b1 (git 7c24448a1b) | python ")

    def test_editable_install_says_so(self, monkeypatch):
        self._install(monkeypatch, json.dumps({"dir_info": {"editable": True}}))
        assert version_info().startswith("0.3.0b1 (editable install) | python ")

    def test_install_without_a_record_shows_the_version(self, monkeypatch):
        self._install(monkeypatch, None)
        assert version_info().startswith("0.3.0b1 | python ")

    def test_cli_prints_it(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["oellm-eval", "--version"])
        with pytest.raises(SystemExit) as exit_info:
            main()
        assert exit_info.value.code == 0
        assert metadata.version("oellm-eval") in capsys.readouterr().out
