import os
from dataclasses import dataclass, field
from unittest.mock import patch

import pytest

from oellm.utils import (
    _expand_local_model_paths,
    _load_cluster_env,
    _materialize_external_urls,
    _num_jobs_in_queue,
    _pre_download_datasets_from_specs,
)


@dataclass
class _FakeSpec:
    repo_id: str
    subset: str | None = None
    needs_snapshot_download: bool = False
    revisions: list[str] = field(default_factory=lambda: ["main"])


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


class TestPreDownloadFailsLoudly:
    """Regression tests for the fail-loudly behavior in
    `_pre_download_datasets_from_specs`.

    Compute nodes run with HF_HUB_OFFLINE=1, so a silent download failure on
    the login node translates to a mid-job ConnectionError on the compute
    node. The pre-download step must therefore *raise* on any unrecoverable
    failure, with a clear message naming every failing dataset.
    """

    def test_raises_on_connection_error_from_load_dataset(self, monkeypatch):
        """A ConnectionError from load_dataset must abort the schedule."""

        def boom_load_dataset(*args, **kwargs):
            raise ConnectionError("simulated offline / DNS failure")

        monkeypatch.setattr("oellm.utils.get_console", lambda: _NoopConsole())
        monkeypatch.setattr("datasets.load_dataset", boom_load_dataset)

        specs = [_FakeSpec(repo_id="some-org/some-dataset")]

        with pytest.raises(RuntimeError) as excinfo:
            _pre_download_datasets_from_specs(specs)

        msg = str(excinfo.value)
        # The error message must identify which dataset failed and what kind
        # of error it was — otherwise the user has no way to debug.
        assert "some-org/some-dataset" in msg
        assert "ConnectionError" in msg
        assert "1/1 dataset" in msg or "1 dataset" in msg

    def test_aggregates_multiple_failures(self, monkeypatch):
        """All failures are reported in one RuntimeError, not just the first."""

        def boom_load_dataset(repo_id, **kwargs):
            raise OSError(f"network unavailable for {repo_id}")

        monkeypatch.setattr("oellm.utils.get_console", lambda: _NoopConsole())
        monkeypatch.setattr("datasets.load_dataset", boom_load_dataset)

        specs = [
            _FakeSpec(repo_id="org/dataset-a"),
            _FakeSpec(repo_id="org/dataset-b"),
            _FakeSpec(repo_id="org/dataset-c"),
        ]

        with pytest.raises(RuntimeError) as excinfo:
            _pre_download_datasets_from_specs(specs)

        msg = str(excinfo.value)
        assert "org/dataset-a" in msg
        assert "org/dataset-b" in msg
        assert "org/dataset-c" in msg
        assert "3/3" in msg or "3 dataset" in msg

    def test_passes_silently_when_all_specs_succeed(self, monkeypatch):
        """The happy path remains silent and non-raising."""
        calls = []

        def fake_load_dataset(repo_id, **kwargs):
            calls.append(repo_id)
            return object()

        monkeypatch.setattr("oellm.utils.get_console", lambda: _NoopConsole())
        monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

        specs = [
            _FakeSpec(repo_id="org/dataset-a"),
            _FakeSpec(repo_id="org/dataset-b"),
        ]

        # No exception expected.
        _pre_download_datasets_from_specs(specs)
        assert calls == ["org/dataset-a", "org/dataset-b"]

    def test_snapshot_failure_raises_even_if_load_dataset_works(self, monkeypatch):
        """Media snapshot failures are STRICT: load_dataset alone does not
        stage the separate media files (video/audio assets), so proceeding
        would schedule rows that fail hours later on the air-gapped node."""

        def boom_snapshot(*args, **kwargs):
            raise OSError("simulated snapshot HTTP 429")

        def fake_load_dataset(*args, **kwargs):
            return object()

        monkeypatch.setattr("oellm.utils.get_console", lambda: _NoopConsole())
        monkeypatch.setattr("huggingface_hub.snapshot_download", boom_snapshot)
        monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

        specs = [_FakeSpec(repo_id="org/dataset-a", needs_snapshot_download=True)]

        with pytest.raises(RuntimeError, match="Pre-download failed"):
            _pre_download_datasets_from_specs(specs)


class TestPreDownloadRevisions:
    """Tests for DatasetSpec.revisions handling in pre-download.

    Datasets like OpenGVLab/MVBench split content across branches: parquet
    metadata on `main`, video assets on `video`. For media datasets
    (needs_snapshot_download) snapshot_download runs once per revision so every
    branch's files — including separate video/audio assets that load_dataset()
    does not fetch — are present on the offline compute node.
    """

    def test_snapshot_download_called_once_per_revision(self, monkeypatch):
        snapshot_calls = []

        def fake_snapshot(*, repo_id, repo_type, revision, max_workers):
            snapshot_calls.append((repo_id, revision))

        def fake_load_dataset(*args, **kwargs):
            return object()

        monkeypatch.setattr("oellm.utils.get_console", lambda: _NoopConsole())
        monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot)
        monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

        specs = [
            _FakeSpec(
                repo_id="OpenGVLab/MVBench",
                needs_snapshot_download=True,
                revisions=["main", "video"],
            )
        ]

        _pre_download_datasets_from_specs(specs)

        # Media datasets snapshot EVERY revision so each branch's files (incl.
        # the separate video/audio assets) are fetched; load_dataset() then
        # builds the Arrow cache.
        assert snapshot_calls == [
            ("OpenGVLab/MVBench", "main"),
            ("OpenGVLab/MVBench", "video"),
        ]

    def test_default_revisions_falls_back_to_main(self, monkeypatch):
        """A media spec without an explicit revisions list still snapshots 'main'."""
        snapshot_calls = []

        def fake_snapshot(*, repo_id, repo_type, revision, max_workers):
            snapshot_calls.append((repo_id, revision))

        def fake_load_dataset(*args, **kwargs):
            return object()

        monkeypatch.setattr("oellm.utils.get_console", lambda: _NoopConsole())
        monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot)
        monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

        # Media spec with default revisions=["main"].
        specs = [_FakeSpec(repo_id="some-org/dataset", needs_snapshot_download=True)]

        _pre_download_datasets_from_specs(specs)

        assert snapshot_calls == [("some-org/dataset", "main")]

    def test_non_media_spec_is_not_snapshotted(self, monkeypatch):
        """Text datasets (needs_snapshot_download=False) skip snapshot entirely —
        load_dataset() fetches + builds them in one step."""
        snapshot_calls = []

        def fake_snapshot(*, repo_id, repo_type, revision, max_workers):
            snapshot_calls.append((repo_id, revision))

        def fake_load_dataset(*args, **kwargs):
            return object()

        monkeypatch.setattr("oellm.utils.get_console", lambda: _NoopConsole())
        monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot)
        monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

        specs = [_FakeSpec(repo_id="text-org/dataset", needs_snapshot_download=False)]

        _pre_download_datasets_from_specs(specs)

        assert snapshot_calls == []


class TestMaterializeExternalUrls:
    """`_materialize_external_urls` forces HF datasets' lazy URL fetches by
    accessing EVERY row of EVERY split.

    Some datasets store image URLs as row fields rather
    than embedded image bytes; the actual HTTP fetch is deferred until each
    row is read. We must trigger it here on the login node so the cache is
    complete before SLURM submission. Touching only the first row leaves
    99%+ of URLs un-cached and the compute-node job fails with
    ConnectionError on the second sample.
    """

    def test_reads_every_row_for_each_split_of_dataset_dict(self):
        """Every index of every split is accessed — that's what triggers
        the per-row URL download for URL-typed Image columns."""
        import threading
        from collections import defaultdict

        accessed = defaultdict(set)
        lock = threading.Lock()

        class _FakeSplit:
            def __init__(self, name, n):
                self.name = name
                self.n = n

            def __len__(self):
                return self.n

            def __getitem__(self, idx):
                with lock:
                    accessed[self.name].add(idx)
                return {"image": b"<bytes>", "answer": "x"}

        class _FakeDatasetDict:
            def __init__(self):
                self._splits = {
                    "train": _FakeSplit("train", 25),
                    "test": _FakeSplit("test", 10),
                }

            def keys(self):
                return self._splits.keys()

            def __getitem__(self, split):
                return self._splits[split]

        _materialize_external_urls(_FakeDatasetDict())

        assert accessed["train"] == set(range(25))
        assert accessed["test"] == set(range(10))

    def test_reads_every_row_for_bare_dataset(self):
        """For a bare Dataset (no .keys()), every index is accessed."""
        import threading

        accessed = set()
        lock = threading.Lock()

        class _FakeDataset:
            def __len__(self):
                return 30

            def __getitem__(self, idx):
                with lock:
                    accessed.add(idx)
                return {"image": b"<bytes>"}

        _materialize_external_urls(_FakeDataset())

        assert accessed == set(range(30))

    def test_skips_empty_splits(self):
        """An empty dataset shouldn't cause IndexError."""

        class _Empty:
            def __len__(self):
                return 0

            def __getitem__(self, idx):  # pragma: no cover — should not be called
                raise AssertionError(
                    "Empty dataset must not be indexed by _materialize_external_urls"
                )

        _materialize_external_urls(_Empty())

    def test_none_input_is_a_no_op(self):
        _materialize_external_urls(None)

    def test_exception_during_materialization_propagates(self):
        """Strict mode: any exception during materialization is propagated.

        The outer ``_pre_download_datasets_from_specs`` loop catches it and
        records the failure. Silently swallowing would let an incomplete
        cache proceed to SLURM submission and cause a compute-node
        ConnectionError that's invisible until you read per-job stderr.
        """

        class _Brittle:
            def __len__(self):
                return 10

            def __getitem__(self, idx):
                raise ConnectionError("simulated external URL fetch failure")

        with pytest.raises(ConnectionError):
            _materialize_external_urls(_Brittle(), max_workers=1)

    def test_materialize_failure_aggregated_into_pre_download_failures(self, monkeypatch):
        """End-to-end: when materialization fails, the outer pre-download
        function records the failure and raises a RuntimeError. The schedule
        is aborted before any SLURM job is submitted."""

        class _Brittle:
            def __len__(self):
                return 5

            def __getitem__(self, idx):
                raise ConnectionError("upstream URL unreachable")

        def fake_load_dataset(*args, **kwargs):
            return _Brittle()

        monkeypatch.setattr("oellm.utils.get_console", lambda: _NoopConsole())
        monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

        specs = [_FakeSpec(repo_id="some-org/url-only-dataset")]

        with pytest.raises(RuntimeError) as excinfo:
            _pre_download_datasets_from_specs(specs)

        msg = str(excinfo.value)
        assert "some-org/url-only-dataset" in msg
        assert "ConnectionError" in msg


def _resolve_filtered_print_in_worker(out_queue):
    """Worker target: import `oellm.utils` and look up `filtered_print` as
    a module attribute. Mirrors what HF datasets' multiprocessing workers
    do when they unpickle a reference to the patched `print`."""
    try:
        from oellm import utils as _u  # noqa: PLC0415

        fn = _u.filtered_print
        out_queue.put(("ok", fn.__name__))
    except Exception as e:
        out_queue.put(("err", f"{type(e).__name__}: {e}"))


class TestFilteredFunctionsAreModuleAttributes:
    """Regression guard: the filtered_* functions used by
    `capture_third_party_output` must be true module-level attributes,
    not closures.

    HF datasets' Audio feature decoder spawns multiprocessing workers
    that pickle/unpickle references to the patched `print`. When
    `filtered_print` was a local closure, workers raised
    ``AttributeError: module 'oellm.utils' has no attribute
    'filtered_print'`` — which surfaced as the audio pre-download
    failing during `_materialize_external_urls`.
    """

    def test_module_exports_filtered_functions(self):
        """Each filtered_* function must be reachable via getattr on the
        module — this is what pickle's resolution uses."""
        from oellm import utils

        for name in (
            "filtered_print",
            "filtered_logger_info",
            "filtered_logger_debug",
            "filtered_module_info",
            "filtered_module_debug",
        ):
            fn = getattr(utils, name, None)
            assert callable(fn), f"oellm.utils.{name} must be a module attribute"
            assert fn.__module__ == "oellm.utils", (
                f"{name}.__module__ is {fn.__module__!r}; "
                f"expected 'oellm.utils'. Closures defined inside "
                f"capture_third_party_output won't resolve in multiprocessing workers."
            )

    def test_filtered_print_resolvable_in_multiprocessing_worker(self):
        """End-to-end: a child process can resolve and call filtered_print."""
        import multiprocessing as _mp

        ctx = _mp.get_context("spawn")
        q = ctx.Queue()
        p = ctx.Process(target=_resolve_filtered_print_in_worker, args=(q,))
        p.start()
        p.join(timeout=30)
        assert p.exitcode == 0, f"Worker process crashed; exitcode={p.exitcode}"
        status, payload = q.get(timeout=5)
        assert status == "ok", (
            f"Worker failed to resolve oellm.utils.filtered_print: {payload}. "
            f"This means a closure regression has been re-introduced and "
            f"HF datasets' multiprocessing workers will crash again."
        )


class _NoopConsole:
    """Stub for rich.console.Console that satisfies the `.status()`
    context-manager interface used by _pre_download_datasets_from_specs."""

    def status(self, *args, **kwargs):
        return _NoopStatus()


class _NoopStatus:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def update(self, *args, **kwargs):
        pass


class TestLoadClusterEnv:
    """_load_cluster_env required-var validation, incl. the no-ACCOUNT cluster case.

    Uses a SYNTHETIC cluster config (not the real, user-customizable
    clusters.yaml) so assertions exercise behavior rather than a specific
    deployment's account/partition strings.
    """

    @staticmethod
    def _clusters():
        # "noacct" mirrors ufal: a SLURM cluster that declares no ACCOUNT.
        # "withacct" mirrors a standard cluster that declares one.
        return {
            "shared": {
                "EVAL_OUTPUT_DIR": "{EVAL_BASE_DIR}/{USER}",
                "GPUS_PER_NODE": 1,
            },
            "withacct": {
                "hostname_pattern": "*.withacct.test",
                "EVAL_BASE_DIR": "/data/evals",
                "PARTITION": "gpu",
                "ACCOUNT": "proj-123",
            },
            "noacct": {
                "hostname_pattern": "*.noacct.test",
                "EVAL_BASE_DIR": "/data/evals",
                "PARTITION": "gpu-a,gpu-b",
            },
        }

    def _run(self, monkeypatch, hostname, env):
        # Pin the hostname AND the cluster config so the test is independent of
        # the real clusters.yaml and the ambient shell env (restored afterwards).
        monkeypatch.setattr("oellm.utils.socket.gethostname", lambda: hostname)
        monkeypatch.setattr("oellm.utils.socket.getfqdn", lambda: hostname)
        monkeypatch.setattr(
            "oellm.utils.yaml.safe_load", lambda *_a, **_k: self._clusters()
        )
        with patch.dict(os.environ, env, clear=True):
            _load_cluster_env()
            return dict(os.environ)

    def test_no_account_cluster_does_not_raise(self, monkeypatch, tmp_path):
        # A cluster that declares no ACCOUNT (ufal-style) must NOT raise; the
        # sbatch --account directive is stripped later when ACCOUNT is unset.
        result = self._run(
            monkeypatch, "node.noacct.test", {"USER": "tester", "HF_HOME": str(tmp_path)}
        )
        assert result["PARTITION"] == "gpu-a,gpu-b"
        assert "ACCOUNT" not in result

    def test_account_cluster_still_sets_account(self, monkeypatch, tmp_path):
        # A cluster that declares ACCOUNT must still resolve and keep it.
        result = self._run(
            monkeypatch,
            "node.withacct.test",
            {"USER": "tester", "HF_HOME": str(tmp_path)},
        )
        assert result["ACCOUNT"] == "proj-123"

    def test_other_required_vars_still_enforced(self, monkeypatch):
        # The fix must be surgical: a genuinely missing required var (HF_HOME)
        # still raises, even on a no-ACCOUNT cluster.
        with pytest.raises(RuntimeError, match="HF_HOME"):
            self._run(monkeypatch, "node.noacct.test", {"USER": "tester"})
