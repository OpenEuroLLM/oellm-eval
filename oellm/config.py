"""Typed evaluation configuration with YAML loading and CLI override support.

Users can define a reusable, version-controllable YAML config file instead of
passing a dozen CLI flags every time.  Every field is overridable from the CLI
(CLI wins).  When no ``--config`` is given the existing CLI-only workflow is
unchanged — ``schedule_evals`` builds an ``EvalConfig`` internally from its
loose parameters.
"""

from __future__ import annotations

import logging
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SlurmOverrides:
    """SLURM-specific overrides that map to ``slurm_template_var`` JSON."""

    partition: str | None = None
    account: str | None = None
    gpus_per_node: int | None = None
    time_limit: str | None = None
    max_array_len: int = 128
    # Template vars the dataclass doesn't model (SLURM_MEM, CPUS_PER_TASK, …).
    # The scheduler applies any key to the environment, so these must survive
    # the EvalConfig round-trip verbatim instead of being dropped.
    extra_template_vars: dict[str, str] = field(default_factory=dict)

    def to_template_var_dict(self) -> dict[str, str]:
        """Return a dict suitable for ``slurm_template_var`` JSON consumption."""
        d: dict[str, str] = dict(self.extra_template_vars)
        if self.partition is not None:
            d["PARTITION"] = self.partition
        if self.account is not None:
            d["ACCOUNT"] = self.account
        if self.gpus_per_node is not None:
            d["GPUS_PER_NODE"] = str(self.gpus_per_node)
        if self.time_limit is not None:
            d["TIME"] = self.time_limit
        return d


@dataclass
class ModelConfig:
    """Named model entry for the ``models:`` YAML key.

    Plain strings and dict entries (``path`` + optional ``name``) are both
    accepted; see :meth:`EvalConfig.from_yaml` for examples.
    """

    path: str
    name: str | None = None


@dataclass
class EvalConfig:
    """Unified evaluation configuration.

    Can be constructed from:
    * a YAML file via :meth:`from_yaml`
    * raw CLI keyword arguments via :meth:`from_cli_kwargs`
    * merging both (CLI wins) via :meth:`merge`
    """

    # ---- what to evaluate ----
    models: list[str | ModelConfig] | None = None
    tasks: list[str] | None = None
    task_groups: list[str] | None = None
    n_shot: list[int] | None = None
    eval_csv_path: str | None = None

    # ---- execution flags ----
    limit: int | None = None
    verbose: bool = False
    download_only: bool = False
    dry_run: bool = False
    skip_checks: bool = False
    trust_remote_code: bool = True
    venv_path: str | None = None
    lm_eval_include_path: str | None = None
    local: bool = False
    # Quantized model loading (bitsandbytes) for the HF-style engines
    # (lm_eval / lmms_eval / evalchemy). NVIDIA-first: ROCm (LUMI) support
    # depends on bitsandbytes' ROCm build. lighteval rows are unaffected and
    # contrib suites only opt in via the OELLM_QUANTIZATION env var — the
    # scheduler warns when such rows are scheduled with a flag set.
    load_in_4bit: bool = False
    load_in_8bit: bool = False

    # ---- SLURM overrides ----
    slurm: SlurmOverrides = field(default_factory=SlurmOverrides)

    # Set by from_cli_kwargs: names of fields explicitly provided on the CLI
    # ("slurm.max_array_len" for the nested one). Plain class attribute, not a
    # dataclass field, so it stays out of __init__/fields(). merge() uses it
    # to distinguish "--no-dry-run" from "flag not given".
    _cli_provided = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> EvalConfig:
        """Load config from a YAML file.

        Example YAML::

            models:
              - "llava-hf/llava-1.5-7b-hf"
              - "Qwen/Qwen2-VL-7B"
            task_groups:
              - "image-vqa"
            n_shot: 0
            trust_remote_code: true
            venv_path: "~/elliot-venv"
            slurm:
              max_array_len: 64
              partition: "gpu"
              time_limit: "06:00:00"
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}

        return cls._from_dict(raw)

    @classmethod
    def from_cli_kwargs(
        cls,
        *,
        models: str | None = None,
        tasks: str | None = None,
        task_groups: str | None = None,
        n_shot: int | list[int] | None = None,
        eval_csv_path: str | None = None,
        max_array_len: int | None = None,
        limit: int | None = None,
        verbose: bool | None = None,
        download_only: bool | None = None,
        dry_run: bool | None = None,
        skip_checks: bool | None = None,
        trust_remote_code: bool | None = None,
        venv_path: str | None = None,
        lm_eval_include_path: str | None = None,
        local: bool | None = None,
        load_in_4bit: bool | None = None,
        load_in_8bit: bool | None = None,
        slurm_template_var: str | None = None,
    ) -> EvalConfig:
        """Build an ``EvalConfig`` from the loose CLI parameters.

        Every parameter defaults to ``None`` meaning "not provided on the CLI".
        Provided parameters are recorded so :meth:`merge` can let an explicit
        CLI value override YAML even when it equals the class default
        (e.g. ``--no-dry-run`` against ``dry_run: true``).
        """
        import json

        models_list = (
            [m.strip() for m in models.split(",") if m.strip()]
            if isinstance(models, str)
            else None
        )
        tasks_list = (
            [t.strip() for t in tasks.split(",") if t.strip()]
            if isinstance(tasks, str)
            else None
        )
        groups_list = (
            [g.strip() for g in task_groups.split(",") if g.strip()]
            if isinstance(task_groups, str)
            else None
        )
        n_shot_list: list[int] | None = None
        if isinstance(n_shot, int):
            n_shot_list = [n_shot]
        elif isinstance(n_shot, list):
            n_shot_list = n_shot

        provided: set[str] = set()
        for field_name, value in (
            ("models", models_list),
            ("tasks", tasks_list),
            ("task_groups", groups_list),
            ("n_shot", n_shot_list),
            ("eval_csv_path", eval_csv_path),
            ("limit", limit),
            ("verbose", verbose),
            ("download_only", download_only),
            ("dry_run", dry_run),
            ("skip_checks", skip_checks),
            ("trust_remote_code", trust_remote_code),
            ("venv_path", venv_path),
            ("lm_eval_include_path", lm_eval_include_path),
            ("local", local),
            ("load_in_4bit", load_in_4bit),
            ("load_in_8bit", load_in_8bit),
        ):
            if value is not None:
                provided.add(field_name)

        slurm = SlurmOverrides()
        if max_array_len is not None:
            slurm.max_array_len = max_array_len
            provided.add("slurm.max_array_len")
        if slurm_template_var:
            try:
                opts = json.loads(slurm_template_var)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"slurm_template_var must be a valid JSON object: {e}"
                ) from e
            if not isinstance(opts, dict):
                raise ValueError(
                    "slurm_template_var must be a JSON object, e.g. "
                    '{"PARTITION":"dev-g","ACCOUNT":"FOO","TIME":"02:00:00"}'
                )
            for key, value in opts.items():
                upper = str(key).upper()
                if upper == "PARTITION":
                    slurm.partition = value
                elif upper == "ACCOUNT":
                    slurm.account = value
                elif upper == "GPUS_PER_NODE":
                    slurm.gpus_per_node = int(value)
                elif upper == "TIME" or key == "time_limit":
                    slurm.time_limit = str(value)
                else:
                    # Unmodeled keys (SLURM_MEM, …) pass through verbatim — the
                    # scheduler exports any key into the job environment.
                    slurm.extra_template_vars[str(key)] = str(value)

        cfg = cls(
            models=models_list,
            tasks=tasks_list,
            task_groups=groups_list,
            n_shot=n_shot_list,
            eval_csv_path=eval_csv_path,
            limit=limit,
            verbose=bool(verbose) if verbose is not None else False,
            download_only=bool(download_only) if download_only is not None else False,
            dry_run=bool(dry_run) if dry_run is not None else False,
            skip_checks=bool(skip_checks) if skip_checks is not None else False,
            trust_remote_code=bool(trust_remote_code)
            if trust_remote_code is not None
            else True,
            venv_path=venv_path,
            lm_eval_include_path=lm_eval_include_path,
            local=bool(local) if local is not None else False,
            load_in_4bit=bool(load_in_4bit) if load_in_4bit is not None else False,
            load_in_8bit=bool(load_in_8bit) if load_in_8bit is not None else False,
            slurm=slurm,
        )
        cfg._cli_provided = provided
        return cfg

    _KNOWN_KEYS = frozenset(
        {
            "models",
            "tasks",
            "task_groups",
            "n_shot",
            "eval_csv_path",
            "limit",
            "verbose",
            "download_only",
            "dry_run",
            "skip_checks",
            "trust_remote_code",
            "venv_path",
            "lm_eval_include_path",
            "local",
            "load_in_4bit",
            "load_in_8bit",
            "slurm",
        }
    )
    _KNOWN_SLURM_KEYS = frozenset(
        {"partition", "account", "gpus_per_node", "time_limit", "max_array_len"}
    )

    @classmethod
    def _from_dict(cls, raw: dict[str, Any]) -> EvalConfig:
        """Construct from a raw dict (YAML or programmatic)."""
        unknown = set(raw) - cls._KNOWN_KEYS
        if unknown:
            logging.warning(
                f"Ignoring unknown config key(s): {', '.join(sorted(unknown))}. "
                f"Known keys: {', '.join(sorted(cls._KNOWN_KEYS))}."
            )

        slurm_raw = raw.get("slurm", {}) or {}
        unknown_slurm = set(slurm_raw) - cls._KNOWN_SLURM_KEYS
        if unknown_slurm:
            logging.warning(
                f"Ignoring unknown slurm config key(s): "
                f"{', '.join(sorted(unknown_slurm))}. "
                f"Known keys: {', '.join(sorted(cls._KNOWN_SLURM_KEYS))}."
            )
        slurm = SlurmOverrides(
            partition=slurm_raw.get("partition"),
            account=slurm_raw.get("account"),
            gpus_per_node=_optional_int(slurm_raw.get("gpus_per_node")),
            time_limit=slurm_raw.get("time_limit"),
            max_array_len=int(slurm_raw.get("max_array_len", 128)),
        )

        # Normalise scalar → list for models / tasks / task_groups / n_shot
        models = _ensure_model_list(raw.get("models"))
        tasks = _ensure_str_list(raw.get("tasks"))
        task_groups = _ensure_str_list(raw.get("task_groups"))
        n_shot = _ensure_int_list(raw.get("n_shot"))

        return cls(
            models=models,
            tasks=tasks,
            task_groups=task_groups,
            n_shot=n_shot,
            eval_csv_path=raw.get("eval_csv_path"),
            limit=_optional_int(raw.get("limit")),
            verbose=bool(raw.get("verbose", False)),
            download_only=bool(raw.get("download_only", False)),
            dry_run=bool(raw.get("dry_run", False)),
            skip_checks=bool(raw.get("skip_checks", False)),
            trust_remote_code=bool(raw.get("trust_remote_code", True)),
            venv_path=raw.get("venv_path"),
            lm_eval_include_path=raw.get("lm_eval_include_path"),
            local=bool(raw.get("local", False)),
            load_in_4bit=bool(raw.get("load_in_4bit", False)),
            load_in_8bit=bool(raw.get("load_in_8bit", False)),
            slurm=slurm,
        )

    def merge(self, cli: EvalConfig) -> EvalConfig:
        """Return a new config where *cli* values override *self* (the YAML base).

        When *cli* was built by :meth:`from_cli_kwargs`, "set" means the flag
        was actually provided (tracked explicitly), so a CLI value equal to
        the class default still overrides YAML (e.g. ``--no-dry-run`` beats
        ``dry_run: true``). For configs constructed directly (without the
        tracking attribute), falls back to the legacy heuristic of comparing
        against the class default.
        """
        provided: set[str] | None = getattr(cli, "_cli_provided", None)

        merged_kwargs: dict[str, Any] = {}
        for f in fields(self):
            yaml_val = getattr(self, f.name)
            cli_val = getattr(cli, f.name)
            default_val = _field_default(f)

            if f.name == "slurm":
                merged_kwargs["slurm"] = _merge_slurm(yaml_val, cli_val, provided)
            elif provided is not None:
                merged_kwargs[f.name] = cli_val if f.name in provided else yaml_val
            elif cli_val != default_val:
                # Legacy heuristic: CLI considered set when ≠ class default
                merged_kwargs[f.name] = cli_val
            else:
                merged_kwargs[f.name] = yaml_val

        return EvalConfig(**merged_kwargs)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Raise ``ValueError`` on invalid or contradictory configuration."""
        if self.load_in_4bit and self.load_in_8bit:
            raise ValueError("load_in_4bit and load_in_8bit are mutually exclusive.")

        if self.eval_csv_path:
            if self.models or self.tasks or self.task_groups or self.n_shot:
                raise ValueError(
                    "Cannot specify models, tasks, task_groups, or n_shot "
                    "when eval_csv_path is provided."
                )
            if not Path(self.eval_csv_path).exists():
                raise FileNotFoundError(
                    f"eval_csv_path does not exist: {self.eval_csv_path}"
                )
            return  # CSV mode — nothing else to validate

        if not self.models:
            raise ValueError("At least one model must be specified.")

        if self.task_groups is None and self.tasks is None:
            raise ValueError(
                "Either task_groups or tasks must be specified (or use eval_csv_path)."
            )

        if self.tasks and not self.n_shot:
            raise ValueError("n_shot is required when specifying individual tasks.")

        if self.n_shot:
            for s in self.n_shot:
                if not isinstance(s, int) or s < 0:
                    raise ValueError(
                        f"n_shot values must be non-negative integers, got: {s}"
                    )

        if self.venv_path:
            venv = Path(self.venv_path).expanduser()
            if not (venv / "bin" / "python").exists():
                logging.warning(
                    f"venv_path '{self.venv_path}' does not contain bin/python "
                    f"— this may fail on the cluster."
                )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def _model_paths(self) -> list[str]:
        """Return plain path strings, unwrapping any ModelConfig objects."""
        if not self.models:
            return []
        return [m.path if isinstance(m, ModelConfig) else m for m in self.models]

    @property
    def slurm_template_var_json(self) -> str | None:
        """Return the JSON string for ``slurm_template_var``, or None if empty."""
        import json

        d = self.slurm.to_template_var_dict()
        return json.dumps(d) if d else None


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _ensure_model_list(val: Any) -> list[str | ModelConfig] | None:
    """Normalise the ``models:`` YAML value to a list of str or ModelConfig."""
    if val is None:
        return None
    items = val if isinstance(val, list) else [val]
    result: list[str | ModelConfig] = []
    for item in items:
        if isinstance(item, dict):
            result.append(ModelConfig(path=item["path"], name=item.get("name")))
        else:
            s = str(item).strip()
            if s:
                result.append(s)
    return result or None


def _ensure_str_list(val: Any) -> list[str] | None:
    if val is None:
        return None
    if isinstance(val, str):
        return [s.strip() for s in val.split(",") if s.strip()]
    if isinstance(val, list):
        return [str(v).strip() for v in val if str(v).strip()]
    return [str(val)]


def _ensure_int_list(val: Any) -> list[int] | None:
    if val is None:
        return None
    if isinstance(val, int):
        return [val]
    if isinstance(val, list):
        return [int(v) for v in val]
    return [int(val)]


def _optional_int(val: Any) -> int | None:
    if val is None:
        return None
    return int(val)


def _field_default(f: Any) -> Any:
    """Return the default value for a dataclass field."""
    if f.default is not MISSING:
        return f.default
    if f.default_factory is not MISSING:
        return f.default_factory()
    return None


def _merge_slurm(
    yaml_slurm: SlurmOverrides,
    cli_slurm: SlurmOverrides,
    provided: set[str] | None = None,
) -> SlurmOverrides:
    """Merge two SlurmOverrides — CLI wins when set (None means unset)."""
    if provided is not None:
        max_array_len = (
            cli_slurm.max_array_len
            if "slurm.max_array_len" in provided
            else yaml_slurm.max_array_len
        )
    else:
        # Legacy heuristic for configs without provenance tracking
        max_array_len = (
            cli_slurm.max_array_len
            if cli_slurm.max_array_len != SlurmOverrides().max_array_len
            else yaml_slurm.max_array_len
        )
    return SlurmOverrides(
        partition=cli_slurm.partition
        if cli_slurm.partition is not None
        else yaml_slurm.partition,
        account=cli_slurm.account
        if cli_slurm.account is not None
        else yaml_slurm.account,
        gpus_per_node=cli_slurm.gpus_per_node
        if cli_slurm.gpus_per_node is not None
        else yaml_slurm.gpus_per_node,
        time_limit=cli_slurm.time_limit
        if cli_slurm.time_limit is not None
        else yaml_slurm.time_limit,
        max_array_len=max_array_len,
        extra_template_vars={
            **yaml_slurm.extra_template_vars,
            **cli_slurm.extra_template_vars,
        },
    )
