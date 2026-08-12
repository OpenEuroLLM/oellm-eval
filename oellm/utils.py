import builtins
import fnmatch
import logging
import os
import socket
import subprocess
import sys
from collections.abc import Iterable
from contextlib import contextmanager
from functools import wraps
from importlib.resources import files
from pathlib import Path

import yaml
from rich.console import Console
from rich.logging import RichHandler

_RICH_CONSOLE: Console | None = None


def get_console() -> Console:
    global _RICH_CONSOLE
    if _RICH_CONSOLE is None:
        _RICH_CONSOLE = Console()
    return _RICH_CONSOLE


def _ensure_runtime_environment(
    use_venv: bool, container_image: str | None, venv_path: str | None
) -> None:
    if use_venv:
        _ensure_venv(venv_path)
    else:
        _ensure_singularity_image(container_image)


def _ensure_venv(venv_path: str) -> None:
    venv = Path(venv_path)
    python_bin = venv / "bin" / "python"

    if not python_bin.exists():
        raise RuntimeError(
            f"No valid Python virtual environment found at {venv_path}. "
            f"Expected to find {python_bin}. "
            f"Create one with: python -m venv {venv_path} && {python_bin} -m pip install lm-eval lighteval"
        )

    logging.info(f"Using Python virtual environment at {venv_path}")


def _ensure_singularity_image(image_name: str | None) -> None:
    from huggingface_hub import hf_hub_download

    if not image_name:
        raise RuntimeError(
            "No container image specified. Set EVAL_CONTAINER_IMAGE environment variable "
            "or use --exec_mode=venv with a virtual environment."
        )

    eval_base_dir = os.getenv("EVAL_BASE_DIR")
    if not eval_base_dir:
        raise RuntimeError(
            "EVAL_BASE_DIR environment variable is not set. "
            "It should be configured in clusters.yaml for this cluster."
        )

    image_path = Path(eval_base_dir) / image_name

    try:
        console = get_console()
        with console.status(
            "Downloading latest Singularity image from HuggingFace", spinner="dots"
        ):
            hf_hub_download(
                repo_id="openeurollm/evaluation_singularity_images",
                filename=image_name,
                repo_type="dataset",
                local_dir=os.getenv("EVAL_BASE_DIR"),
            )
    except Exception as e:
        logging.warning(
            "Failed to fetch latest container image from HuggingFace: %s", str(e)
        )
        if image_path.exists():
            logging.info("Using existing Singularity image at %s", image_path)
        else:
            raise RuntimeError(
                f"No container image found at {image_path} and failed to download from HuggingFace. "
                f"Cannot proceed with evaluation scheduling."
            ) from e


def _setup_logging(verbose: bool = False):
    rich_handler = RichHandler(
        console=get_console(),
        show_time=True,
        log_time_format="%H:%M:%S",
        show_path=False,
        markup=True,
        rich_tracebacks=True,
    )

    class RichFormatter(logging.Formatter):
        def format(self, record):
            return record.getMessage()

    rich_handler.setFormatter(RichFormatter())

    root_logger = logging.getLogger()
    root_logger.handlers = []
    root_logger.addHandler(rich_handler)
    root_logger.setLevel(logging.DEBUG if verbose else logging.INFO)


def _load_cluster_env() -> None:
    """
    Loads the correct cluster environment variables from `clusters.yaml` based on the hostname.
    """
    clusters = yaml.safe_load((files("oellm.resources") / "clusters.yaml").read_text())

    shared_cfg = clusters.get("shared", {}) or {}

    def _match_cluster(hostname: str) -> dict | None:
        for name, cfg in clusters.items():
            if name == "shared":
                continue
            pattern = cfg.get("hostname_pattern")
            if isinstance(pattern, str):
                patterns = [pattern]
            elif isinstance(pattern, list):
                patterns = pattern
            else:
                continue
            if any(fnmatch.fnmatch(hostname, p) for p in patterns):
                return dict(cfg)
        return None

    hostname = socket.gethostname()
    cluster_cfg_raw = _match_cluster(hostname)
    if cluster_cfg_raw is None:
        fqdn = socket.getfqdn()
        if fqdn != hostname:
            cluster_cfg_raw = _match_cluster(fqdn)
            hostname = fqdn
    if cluster_cfg_raw is None:
        raise ValueError(f"No cluster found for hostname: {hostname}")

    cluster_cfg_raw.pop("hostname_pattern", None)

    class _Default(dict):
        def __missing__(self, key):
            return "{" + key + "}"

    # Template-resolution precedence: user environment > cluster values >
    # shared values. The user env must win INSIDE templated values too —
    # ``export EVAL_BASE_DIR=/custom`` has to propagate into
    # ``EVAL_OUTPUT_DIR: "{EVAL_BASE_DIR}/{USER}"``, not just override the
    # variable itself (the export below is setdefault, so exported vars also
    # win at that level).
    cluster_str = {k: str(v) for k, v in cluster_cfg_raw.items()}
    base_ctx = _Default({**cluster_str, **os.environ})

    resolved_shared = {k: str(v).format_map(base_ctx) for k, v in shared_cfg.items()}

    ctx = _Default({**resolved_shared, **cluster_str, **os.environ})

    resolved_cluster = {k: str(v).format_map(ctx) for k, v in cluster_cfg_raw.items()}

    final_env = {**resolved_shared, **resolved_cluster}
    overridden = {
        k: os.environ[k]
        for k, v in final_env.items()
        if k in os.environ and os.environ[k] != v
    }
    if overridden:
        logging.info(
            f"Using custom environment variables: {', '.join(f'{k}={v}' for k, v in overridden.items())}"
        )
    for k, v in final_env.items():
        os.environ.setdefault(k, v)

    # Validate that critical sbatch variables resolved to real values.
    # HF_HOME is included because the job script derives every cache path from
    # it (`HF_DATASETS_CACHE="$HF_HOME/datasets"`); unset, the compute node
    # would resolve caches to "/datasets" and fail far from the real cause.
    _required_vars = [
        "PARTITION",
        "EVAL_BASE_DIR",
        "EVAL_OUTPUT_DIR",
        "GPUS_PER_NODE",
        "HF_HOME",
    ]
    # ACCOUNT is required only for clusters that declare it. Some clusters
    # (e.g. ufal) intentionally omit it and rely on the submitting user's
    # default SLURM account; for those the sbatch's `#SBATCH --account=$ACCOUNT`
    # directive is stripped entirely in schedule_evals(). If a cluster does
    # declare ACCOUNT, it is still validated (including unresolved "{...}").
    if "ACCOUNT" in final_env:
        _required_vars.append("ACCOUNT")
    missing = [
        v for v in _required_vars if not os.environ.get(v) or "{" in os.environ.get(v, "")
    ]
    if missing:
        raise RuntimeError(
            f"Required cluster variables are missing or unresolved: {', '.join(missing)}. "
            f"Check your clusters.yaml entry or set them in your environment."
        )


def _num_jobs_in_queue() -> int:
    user = os.environ.get("USER")
    cmd: list[str] = ["squeue"]
    if user:
        cmd += ["-u", user]
    cmd += ["-h", "-t", "pending,running", "-r", "-o", "%i"]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        if result.stderr:
            logging.warning(f"squeue error: {result.stderr.strip()}")
        return 0

    output = result.stdout.strip()
    if not output:
        return 0
    return sum(1 for line in output.splitlines() if line.strip())


def _expand_local_model_paths(model: str | Path) -> list[Path]:
    """
    Expands a local model path to include all checkpoints if it's a directory.
    Recursively searches for models in subdirectories.

    Args:
        model: Path to a model or directory containing models

    Returns:
        List of paths to model directories containing safetensors files
    """
    model_paths = []
    model_path = Path(model)

    if not model_path.exists() or not model_path.is_dir():
        return model_paths

    if any(model_path.glob("*.safetensors")):
        model_paths.append(model_path)
        return model_paths

    hf_path = model_path / "hf"
    if hf_path.exists() and hf_path.is_dir():
        for subdir in hf_path.glob("*"):
            if subdir.is_dir() and any(subdir.glob("*.safetensors")):
                model_paths.append(subdir)
        if model_paths:
            return model_paths

    subdirs = [d for d in model_path.iterdir() if d.is_dir()]

    for subdir in subdirs:
        if any(subdir.glob("*.safetensors")):
            model_paths.append(subdir)
        else:
            hf_subpath = subdir / "hf"
            if hf_subpath.exists() and hf_subpath.is_dir():
                for checkpoint_dir in hf_subpath.glob("*"):
                    if checkpoint_dir.is_dir() and any(
                        checkpoint_dir.glob("*.safetensors")
                    ):
                        model_paths.append(checkpoint_dir)

    if len(model_paths) > 1:
        logging.info(f"Expanded '{model}' to {len(model_paths)} model checkpoints")

    return model_paths


def _process_model_paths(models: Iterable[str]) -> dict[str, str | None]:
    """Pre-download hub models and return their resolved revisions.

    Each model string can be a local path or a huggingface model identifier.
    Local directory paths that contain multiple checkpoints are expanded.
    Returns ``{model: resolved_commit_or_None}`` for hub models — the commit
    actually present in the shared cache, recorded into the run's
    ``provenance.json``. Models are otherwise unpinned: whatever revision is
    cached is what the offline compute node evaluates, so recording it is the
    only way a collected number stays traceable.
    """
    from huggingface_hub import snapshot_download

    revisions: dict[str, str | None] = {}
    console = get_console()
    models_list = list(models)

    with console.status(
        f"Processing models… 0/{len(models_list)}", spinner="dots"
    ) as status:
        for idx, model in enumerate(models_list, 1):
            status.update(f"Checking model '{model}' ({idx}/{len(models_list)})")
            per_model_paths: list[Path | str] = []

            local_paths = _expand_local_model_paths(model)
            if local_paths:
                per_model_paths.extend(local_paths)
                status.update(f"Using local model '{model}' ({idx}/{len(models_list)})")
            else:
                logging.info(
                    f"Model {model} not found locally, assuming it is a 🤗 hub model"
                )
                logging.debug(
                    f"Downloading model {model} on the login node since the compute nodes may not have access to the internet"
                )

                if "," in model:
                    model_kwargs = dict(
                        [kv.split("=") for kv in model.split(",") if "=" in kv]
                    )

                    repo_id = model.split(",")[0]

                    snapshot_kwargs = {}
                    if "revision" in model_kwargs:
                        snapshot_kwargs["revision"] = model_kwargs["revision"]

                    status.update(f"Downloading '{repo_id}' ({idx}/{len(models_list)})")
                    try:
                        _local = snapshot_download(
                            repo_id=repo_id,
                            cache_dir=Path(os.getenv("HF_HOME")) / "hub"
                            if "HF_HOME" in os.environ
                            else None,
                            **snapshot_kwargs,
                        )
                        revisions[str(model)] = (
                            Path(_local).name
                            if Path(_local).parent.name == "snapshots"
                            else None
                        )
                        per_model_paths.append(model)
                    except Exception as e:
                        logging.warning(
                            f"Failed to download model {model} from Hugging Face Hub "
                            f"({type(e).__name__}: {e}). The job will still be "
                            f"scheduled and will fail on the offline compute node "
                            f"unless the model is already cached."
                        )
                else:
                    cache_dir = (
                        Path(os.getenv("HF_HOME")) / "hub"
                        if "HF_HOME" in os.environ
                        else None
                    )
                    status.update(f"Downloading '{model}' ({idx}/{len(models_list)})")
                    # snapshot_download is idempotent — it skips files that
                    # are already cached and only fetches missing ones.
                    _local = snapshot_download(
                        repo_id=model,
                        cache_dir=cache_dir,
                    )
                    revisions[str(model)] = (
                        Path(_local).name
                        if Path(_local).parent.name == "snapshots"
                        else None
                    )
                    per_model_paths.append(model)

            if not per_model_paths:
                logging.warning(
                    f"Could not find any valid model for '{model}'. It will be skipped."
                )

    return revisions


def _pre_download_hf_model_repos(repo_ids: list[str]) -> None:
    """Download auxiliary HF model repos (e.g. SAM2) required by contrib suites."""
    from huggingface_hub import snapshot_download

    console = get_console()
    with console.status(
        f"Downloading auxiliary models… {len(repo_ids)} repos", spinner="dots"
    ) as status:
        for idx, repo_id in enumerate(repo_ids, 1):
            status.update(f"Downloading '{repo_id}' ({idx}/{len(repo_ids)})")
            try:
                snapshot_download(
                    repo_id=repo_id,
                    cache_dir=Path(os.getenv("HF_HOME")) / "hub"
                    if "HF_HOME" in os.environ
                    else None,
                )
            except Exception as e:
                logging.warning(f"Failed to download auxiliary model '{repo_id}': {e}")


def _pre_download_hf_dataset_files(dataset_files: list[dict]) -> None:
    """Download specific files from HF dataset repos declared in task ``hf_dataset_files`` fields."""
    from huggingface_hub import snapshot_download

    console = get_console()
    with console.status(
        f"Downloading auxiliary dataset files… {len(dataset_files)} repos", spinner="dots"
    ) as status:
        for idx, spec in enumerate(dataset_files, 1):
            repo_id = spec.get("repo_id", "")
            patterns = spec.get("patterns")
            revision = spec.get("revision")
            status.update(f"Downloading '{repo_id}' ({idx}/{len(dataset_files)})")
            try:
                kwargs = {
                    "repo_id": repo_id,
                    "repo_type": "dataset",
                    "allow_patterns": patterns,
                    "cache_dir": Path(os.getenv("HF_HOME")) / "hub"
                    if "HF_HOME" in os.environ
                    else None,
                }
                if revision:
                    kwargs["revision"] = revision
                snapshot_download(**kwargs)
            except Exception as e:
                logging.warning(f"Failed to download dataset files from '{repo_id}': {e}")


def _materialize_external_urls(ds, *, max_workers: int = 16) -> None:
    """Iterate every row to force HF ``dl_manager`` to fetch external URLs.

    Some datasets store media as external URLs (not bytes) in
    parquet rows; only per-row access triggers the HTTP fetch into the
    cache. Strict: exceptions propagate so ``_pre_download_datasets_…``
    aborts the schedule before SLURM submission.
    """
    if ds is None:
        return

    from concurrent.futures import ThreadPoolExecutor

    def _materialize_split(split) -> None:
        n = len(split)
        if n == 0:
            return
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for _ in pool.map(lambda i: split[i], range(n)):
                pass

    if hasattr(ds, "keys"):
        for split_name in list(ds.keys()):
            _materialize_split(ds[split_name])
    elif hasattr(ds, "__len__") and hasattr(ds, "__getitem__"):
        # Skip anything that isn't a recognizable dataset shape (e.g. test stubs).
        _materialize_split(ds)


def _pre_download_datasets_from_specs(
    specs: Iterable, trust_remote_code: bool = True
) -> None:
    """Pre-fetch every dataset spec into the local HF cache.

    Strict: any failure raises ``RuntimeError`` and aborts the schedule
    before SLURM submission — compute nodes run ``HF_HUB_OFFLINE=1`` and
    can't recover from a cache miss. Override with ``--skip-checks``.
    """
    from datasets import get_dataset_config_names, load_dataset
    from huggingface_hub import snapshot_download

    specs_list = list(specs)
    if not specs_list:
        return

    console = get_console()
    failures: list[tuple[str, Exception]] = []

    with console.status(
        f"Downloading datasets… {len(specs_list)} datasets",
        spinner="dots",
    ) as status:
        for idx, spec in enumerate(specs_list, 1):
            label = f"{spec.repo_id}" + (f"/{spec.subset}" if spec.subset else "")
            status.update(f"Downloading '{label}' ({idx}/{len(specs_list)})")

            # Media datasets (audio/video/image → needs_snapshot_download) keep
            # their media as SEPARATE files in the repo (video .mp4s, audio
            # clips) that load_dataset() does NOT fetch — it only builds the
            # parquet/QA. snapshot_download the full repo (every revision, e.g.
            # OpenGVLab/MVBench's `video` branch) so those media files are
            # present on the offline compute node; load_dataset() below then
            # builds the Arrow cache from the same files. BOTH are required:
            # snapshot for the media assets, load_dataset for offline-loadability.
            revisions = getattr(spec, "revisions", None) or ["main"]
            if spec.needs_snapshot_download:
                for rev in revisions:
                    rev_label = f"{label}@{rev}" if rev != "main" else label
                    status.update(f"Downloading '{rev_label}' ({idx}/{len(specs_list)})")
                    try:
                        # max_workers=2 keeps HEAD requests under HF's per-IP
                        # rate limit; higher triggers HTTP 429.
                        snapshot_download(
                            repo_id=spec.repo_id,
                            repo_type="dataset",
                            revision=rev,
                            max_workers=2,
                        )
                    except Exception as e:
                        # Media files are NOT fetched by load_dataset() below,
                        # so a failed snapshot means every row of this task
                        # fails hours later on the air-gapped compute node.
                        # Strict contract: aggregate and abort before SLURM
                        # submission (bypass with --skip-checks if the cache
                        # is already populated out-of-band).
                        failures.append((rev_label, e))

            # Build the Arrow cache — this is what makes the dataset loadable on the
            # OFFLINE compute nodes. load_dataset() needs the BUILT dataset under
            # HF_DATASETS_CACHE; a bare hub snapshot is not loadable offline (it
            # still tries to reach the Hub for the dataset module → ConnectionError).
            # load_dataset reuses any files already fetched above — it does not
            # re-download them. NOTE: the build can OOM the login node for very
            # large media datasets (e.g. 60 GB librispeech), which then need a
            # separate staging strategy.
            try:
                ds = load_dataset(
                    spec.repo_id,
                    name=spec.subset,
                    trust_remote_code=trust_remote_code,
                )
                _materialize_external_urls(ds)
            except ValueError as e:
                if "Config name is missing" in str(e) and spec.subset is None:
                    try:
                        configs = get_dataset_config_names(
                            spec.repo_id, trust_remote_code=trust_remote_code
                        )
                        logging.info(
                            f"Dataset '{spec.repo_id}' requires config. "
                            f"Downloading all {len(configs)} configs."
                        )
                        for cfg in configs:
                            status.update(
                                f"Downloading '{spec.repo_id}/{cfg}' "
                                f"({idx}/{len(specs_list)})"
                            )
                            ds_cfg = load_dataset(
                                spec.repo_id,
                                name=cfg,
                                trust_remote_code=trust_remote_code,
                            )
                            _materialize_external_urls(ds_cfg)
                    except Exception as inner:
                        failures.append((label, inner))
                    continue
                if "Feature type" in str(e) and "not found" in str(e):
                    hf_datasets_cache = os.environ.get(
                        "HF_DATASETS_CACHE",
                        str(Path.home() / ".cache" / "huggingface" / "datasets"),
                    )
                    safe_name = spec.repo_id.replace("/", "___")
                    cache_dir = os.path.join(hf_datasets_cache, safe_name)
                    raise RuntimeError(
                        f"Cached metadata for '{label}' is incompatible with the installed "
                        f"datasets version ('{e}'). Delete the stale cache and re-run:\n\n"
                        f"    rm -rf {cache_dir}\n"
                    ) from None
                failures.append((label, e))
            except Exception as e:
                # Network / hub / OS errors — catch and aggregate, don't swallow.
                failures.append((label, e))
            else:
                logging.debug(f"Finished downloading dataset '{label}'.")
                continue

    if failures:
        details = "\n".join(
            f"  - {label}: {type(e).__name__}: {e}" for label, e in failures
        )
        raise RuntimeError(
            f"Pre-download failed for {len(failures)}/{len(specs_list)} dataset(s); "
            f"aborting before SLURM submission (compute nodes run offline).\n\n"
            f"Failures:\n{details}\n\n"
            f"Common fixes: set HF_TOKEN / accept dataset license on HF / retry "
            f"after rate-limit cools off. Bypass with `--skip-checks` if the cache "
            f"is already populated out-of-band.\n"
        )


_PACKAGE_ROOT = Path(__file__).resolve().parent
# Saved originals when capture is active. Module-level (not a closure) so
# the filtered_* functions are importable as `oellm.utils.filtered_*` —
# required for HF datasets' multiprocessing workers to resolve them when
# the patched print/logger gets pickled across processes.
_capture_originals: dict = {"active": False}


def _is_internal_stack(skip: int = 2, max_depth: int = 20) -> bool:
    f = sys._getframe(skip)
    depth = 0
    while f and depth < max_depth:
        code = f.f_code
        filename = code.co_filename if code else ""
        if filename:
            p = Path(filename).resolve()
            name = code.co_name if code else ""
            # Skip logging internals and our filtering wrappers to find the real caller.
            if "/logging/__init__.py" in filename or name.startswith("filtered_"):
                f = f.f_back
                depth += 1
                continue
            return p.is_relative_to(_PACKAGE_ROOT)
        f = f.f_back
        depth += 1
    return False


def filtered_print(*args, **kwargs):
    orig = _capture_originals.get("print")
    if orig is None or _is_internal_stack():
        return (orig or builtins.print)(*args, **kwargs)
    return None


def filtered_logger_info(self, msg, *args, **kwargs):
    orig = _capture_originals.get("logger_info")
    if orig is None or _is_internal_stack():
        return (orig or logging.Logger.info)(self, msg, *args, **kwargs)
    return None


def filtered_logger_debug(self, msg, *args, **kwargs):
    orig = _capture_originals.get("logger_debug")
    if orig is None or _is_internal_stack():
        return (orig or logging.Logger.debug)(self, msg, *args, **kwargs)
    return None


def filtered_module_info(msg, *args, **kwargs):
    orig = _capture_originals.get("module_info")
    if orig is None or _is_internal_stack():
        return (orig or logging.info)(msg, *args, **kwargs)
    return None


def filtered_module_debug(msg, *args, **kwargs):
    orig = _capture_originals.get("module_debug")
    if orig is None or _is_internal_stack():
        return (orig or logging.debug)(msg, *args, **kwargs)
    return None


@contextmanager
def capture_third_party_output(verbose: bool = False):
    """Suppress print/logging.info/logging.debug from non-project modules
    unless verbose=True. A call is "third-party" if its caller's file path
    is not under the `oellm` package directory."""
    if verbose:
        yield
        return

    _capture_originals.update(
        {
            "active": True,
            "print": builtins.print,
            "logger_info": logging.Logger.info,
            "logger_debug": logging.Logger.debug,
            "module_info": logging.info,
            "module_debug": logging.debug,
        }
    )
    builtins.print = filtered_print  # type: ignore
    logging.Logger.info = filtered_logger_info  # type: ignore[assignment]
    logging.Logger.debug = filtered_logger_debug  # type: ignore[assignment]
    logging.info = filtered_module_info  # type: ignore[assignment]
    logging.debug = filtered_module_debug  # type: ignore[assignment]

    try:
        yield
    finally:
        builtins.print = _capture_originals["print"]
        logging.Logger.info = _capture_originals["logger_info"]  # type: ignore[assignment]
        logging.Logger.debug = _capture_originals["logger_debug"]  # type: ignore[assignment]
        logging.info = _capture_originals["module_info"]  # type: ignore[assignment]
        logging.debug = _capture_originals["module_debug"]  # type: ignore[assignment]
        _capture_originals["active"] = False


def capture_third_party_output_from_kwarg(
    verbose_kwarg: str = "verbose", default: bool = False
):
    """
    Decorator factory that wraps the function execution inside
    capture_third_party_output(verbose=kwargs.get(verbose_kwarg, default)).
    """

    def _decorator(func):
        @wraps(func)
        def _wrapper(*args, **kwargs):
            verbose_value = bool(kwargs.get(verbose_kwarg, default))
            with capture_third_party_output(verbose=verbose_value):
                return func(*args, **kwargs)

        return _wrapper

    return _decorator


def _filter_warnings():
    """
    Filters warnings from the lm_eval and lighteval libraries.
    """
    import warnings

    warnings.filterwarnings("ignore", module="lm_eval")
    warnings.filterwarnings("ignore", module="lighteval")


def check_judge_llm_pre_flight(
    tasks: Iterable[str], *, allow_missing: bool = False
) -> None:
    """Refuse to schedule judge-graded tasks without ``OPENAI_API_KEY``.

    Runs before SLURM submission. Inspects ``tasks`` against
    ``JUDGE_REQUIRED_TASKS`` and raises ``SystemExit`` if any are present
    and ``OPENAI_API_KEY`` is unset, unless ``allow_missing=True`` (the
    user explicitly opted in to letting those tasks emit null scores).
    """
    import os

    from oellm.constants import JUDGE_REQUIRED_TASKS

    needed = sorted({t for t in tasks if t in JUDGE_REQUIRED_TASKS})
    if not needed:
        return
    if os.environ.get("OPENAI_API_KEY"):
        return
    if allow_missing:
        logging.warning(
            "Scheduling %d judge-required task(s) without OPENAI_API_KEY: %s. "
            "These will emit null performance values in collect.",
            len(needed),
            ", ".join(needed),
        )
        return
    raise SystemExit(
        "Refusing to schedule judge-required task(s) without OPENAI_API_KEY:\n"
        f"  {', '.join(needed)}\n\n"
        "These tasks need an LLM judge / extractor to produce a valid metric. "
        "Either:\n"
        "  - export OPENAI_API_KEY=... before re-running, or\n"
        "  - pass --allow-missing-judge to acknowledge that these tasks will "
        "emit null scores, or\n"
        "  - remove them from the task list."
    )
