import sys
from types import SimpleNamespace

from oellm.task_groups import DatasetSpec
from oellm.utils import _expand_local_model_paths, _num_jobs_in_queue


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


class TestPreDownloadDatasets:
    def test_omits_trust_remote_code_for_datasets_v4(self, monkeypatch):
        calls = []

        def load_dataset(repo_id, **kwargs):
            calls.append((repo_id, kwargs))

        fake_datasets = SimpleNamespace(
            __version__="4.8.5",
            load_dataset=load_dataset,
            get_dataset_config_names=lambda *args, **kwargs: [],
        )
        monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

        from oellm.utils import _pre_download_datasets_from_specs

        _pre_download_datasets_from_specs(
            [DatasetSpec(repo_id="example/data", subset="en")]
        )

        assert calls == [("example/data", {"name": "en"})]
