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

    image_path = Path(os.getenv("EVAL_BASE_DIR")) / image_name

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
            record.msg = f"{record.getMessage()}"
            return record.msg

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
    hostname = socket.gethostname()

    shared_cfg = clusters.get("shared", {}) or {}

    cluster_cfg_raw: dict | None = None
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
            cluster_cfg_raw = dict(cfg)
            break
    if cluster_cfg_raw is None:
        raise ValueError(f"No cluster found for hostname: {hostname}")

    cluster_cfg_raw.pop("hostname_pattern", None)

    class _Default(dict):
        def __missing__(self, key):
            return "{" + key + "}"

    base_ctx = _Default({**os.environ, **{k: str(v) for k, v in cluster_cfg_raw.items()}})

    resolved_shared = {k: str(v).format_map(base_ctx) for k, v in shared_cfg.items()}

    ctx = _Default({**base_ctx, **resolved_shared})

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


def _process_model_paths(models: Iterable[str]):
    """
    Processes model strings into a dict of model paths.

    Each model string can be a local path or a huggingface model identifier.
    This function expands directory paths that contain multiple checkpoints.
    """
    from huggingface_hub import snapshot_download

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
                        snapshot_download(
                            repo_id=repo_id,
                            cache_dir=Path(os.getenv("HF_HOME")) / "hub",
                            **snapshot_kwargs,
                        )
                        per_model_paths.append(model)
                    except Exception as e:
                        logging.debug(
                            f"Failed to download model {model} from Hugging Face Hub. Continuing..."
                        )
                        logging.debug(e)
                else:
                    status.update(f"Downloading '{model}' ({idx}/{len(models_list)})")
                    snapshot_download(
                        repo_id=model,
                        cache_dir=Path(os.getenv("HF_HOME")) / "hub"
                        if "HF_HOME" in os.environ
                        else None,
                    )
                    per_model_paths.append(model)

            if not per_model_paths:
                logging.warning(
                    f"Could not find any valid model for '{model}'. It will be skipped."
                )


def _pre_download_datasets_from_specs(
    specs: Iterable, trust_remote_code: bool = True
) -> None:
    import datasets
    from datasets import get_dataset_config_names, load_dataset

    specs_list = list(specs)
    if not specs_list:
        return

    console = get_console()
    datasets_major_version = int(datasets.__version__.split(".", 1)[0])
    supports_trust_remote_code = datasets_major_version < 4

    def _load_dataset_compat(repo_id: str, name: str | None = None):
        kwargs = {"name": name}
        if trust_remote_code and supports_trust_remote_code:
            kwargs["trust_remote_code"] = trust_remote_code
        try:
            return load_dataset(repo_id, **kwargs)
        except ValueError as e:
            if "trust_remote_code" not in str(e):
                raise
            kwargs.pop("trust_remote_code", None)
            return load_dataset(repo_id, **kwargs)

    def _get_dataset_config_names_compat(repo_id: str):
        kwargs = {}
        if trust_remote_code and supports_trust_remote_code:
            kwargs["trust_remote_code"] = trust_remote_code
        try:
            return get_dataset_config_names(repo_id, **kwargs)
        except ValueError as e:
            if "trust_remote_code" not in str(e):
                raise
            return get_dataset_config_names(repo_id)

    with console.status(
        f"Downloading datasets… {len(specs_list)} datasets",
        spinner="dots",
    ) as status:
        for idx, spec in enumerate(specs_list, 1):
            label = f"{spec.repo_id}" + (f"/{spec.subset}" if spec.subset else "")
            status.update(f"Downloading '{label}' ({idx}/{len(specs_list)})")

            try:
                _load_dataset_compat(spec.repo_id, name=spec.subset)
            except ValueError as e:
                if "Config name is missing" in str(e) and spec.subset is None:
                    configs = _get_dataset_config_names_compat(spec.repo_id)
                    logging.info(
                        f"Dataset '{spec.repo_id}' requires config. "
                        f"Downloading all {len(configs)} configs."
                    )
                    for cfg in configs:
                        status.update(
                            f"Downloading '{spec.repo_id}/{cfg}' ({idx}/{len(specs_list)})"
                        )
                        _load_dataset_compat(spec.repo_id, name=cfg)
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
                raise

            logging.debug(f"Finished downloading dataset '{label}'.")


@contextmanager
def capture_third_party_output(verbose: bool = False):
    """
    Suppresses print/logging.info/logging.debug originating from non-project modules
    unless verbose=True.

    A call is considered "third-party" if its immediate caller's file path is not
    under the repository root (parent of the `oellm` package directory).
    """
    if verbose:
        yield
        return

    package_root = Path(__file__).resolve().parent

    def is_internal_stack(skip: int = 2, max_depth: int = 20) -> bool:
        f = sys._getframe(skip)
        depth = 0
        while f and depth < max_depth:
            code = f.f_code
            filename = code.co_filename if code else ""
            if filename:
                p = Path(filename).resolve()
                name = code.co_name if code else ""
                # Skip logging internals and our filtering wrappers to find the real caller
                if "/logging/__init__.py" in filename or name.startswith("filtered_"):
                    f = f.f_back
                    depth += 1
                    continue
                return p.is_relative_to(package_root)
            f = f.f_back
            depth += 1
        return False

    orig_print = builtins.print
    orig_logger_info = logging.Logger.info
    orig_logger_debug = logging.Logger.debug
    orig_module_info = logging.info
    orig_module_debug = logging.debug

    def filtered_print(*args, **kwargs):
        if is_internal_stack():
            return orig_print(*args, **kwargs)
        # third-party: drop
        return None

    def filtered_logger_info(self, msg, *args, **kwargs):
        if is_internal_stack():
            return orig_logger_info(self, msg, *args, **kwargs)
        return None

    def filtered_logger_debug(self, msg, *args, **kwargs):
        if is_internal_stack():
            return orig_logger_debug(self, msg, *args, **kwargs)
        return None

    def filtered_module_info(msg, *args, **kwargs):
        if is_internal_stack():
            return orig_module_info(msg, *args, **kwargs)
        return None

    def filtered_module_debug(msg, *args, **kwargs):
        if is_internal_stack():
            return orig_module_debug(msg, *args, **kwargs)
        return None

    builtins.print = filtered_print  # type: ignore
    logging.Logger.info = filtered_logger_info  # type: ignore[assignment]
    logging.Logger.debug = filtered_logger_debug  # type: ignore[assignment]
    logging.info = filtered_module_info  # type: ignore[assignment]
    logging.debug = filtered_module_debug  # type: ignore[assignment]

    try:
        yield
    finally:
        builtins.print = orig_print
        logging.Logger.info = orig_logger_info  # type: ignore[assignment]
        logging.Logger.debug = orig_logger_debug  # type: ignore[assignment]
        logging.info = orig_module_info  # type: ignore[assignment]
        logging.debug = orig_module_debug  # type: ignore[assignment]


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
