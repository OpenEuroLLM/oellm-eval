"""Environment pre-flight: verify the runtime can actually execute the scheduled suites.

The wrapper schedules work into *one* runtime per submission (a venv via
``--venv-path``, or the cluster's Singularity image), but the engines have
mutually incompatible dependency stacks (see ``docs/VENV.md``). Nothing else
in the pipeline validates that the chosen runtime contains the engines the
scheduled suites need — failures otherwise surface row-by-row on the compute
node hours later, or worse, run silently on the wrong engine version
(``dclm-core-22`` needs ``lm-eval==0.4.9.2``; a newer lm-eval *changes the
scores* instead of failing).

Two entry points:

* :func:`check_scheduled_environment` — called by the scheduler before
  submission (bypass with ``--skip-checks``). Raises ``SystemExit`` listing
  every problem at once.
* :func:`run_doctor_checks` — powers ``oellm-eval doctor``; returns a list of
  :class:`CheckResult` for human-readable reporting.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

# Engines the cluster Singularity images contain (see containers/*.def:
# lm-eval on the system python, lighteval as an isolated uv tool). Everything
# else — lmms-eval, the oellm package itself (contrib dispatch), evalchemy —
# is NOT in the images and requires --venv-path.
_CONTAINER_SUPPORTED_SUITES = frozenset({"lm_eval", "lighteval"})

# Task groups that only produce valid numbers on a pinned engine version.
# Running them on any other version does not fail — it silently changes the
# scores, which is the worst possible failure for a benchmarking platform.
GROUP_VERSION_PINS: dict[str, tuple[tuple[str, str], ...]] = {
    # v0.4.10+ breaks agieval_lsat_ar few-shot — see pyproject [dclm] extra.
    "dclm-core-22": (("lm_eval", "0.4.9.2"),),
}

_PROBE_TIMEOUT_S = 120


@dataclass(frozen=True)
class SuiteRequirements:
    """What a suite needs from the runtime to run at all."""

    modules: tuple[str, ...] = ()  # importable by the venv's python
    executables: tuple[str, ...] = ()  # in venv/bin or on PATH
    env_vars: tuple[str, ...] = ()  # set in the scheduling environment
    container_ok: bool = False
    hint: str = ""


# Built-in engines. Contrib suites are resolved dynamically from the plugin
# registry (modules=("oellm",) + the plugin's declared CLUSTER_ENV_VARS).
SUITE_REQUIREMENTS: dict[str, SuiteRequirements] = {
    "lm_eval": SuiteRequirements(
        modules=("lm_eval",),
        container_ok=True,
        hint="install the [text] extra in the venv — see docs/VENV.md",
    ),
    "lighteval": SuiteRequirements(
        executables=("lighteval",),
        container_ok=True,
        hint="install lighteval as a uv tool with UV_TOOL_BIN_DIR=<venv>/bin "
        "— see docs/VENV.md",
    ),
    "lmms_eval": SuiteRequirements(
        modules=("lmms_eval",),
        container_ok=False,
        hint="lmms-eval is not in the cluster containers; create the general "
        "venv (docs/VENV.md) and pass --venv-path",
    ),
    "evalchemy": SuiteRequirements(
        modules=("accelerate",),
        env_vars=("EVALCHEMY_DIR",),
        container_ok=False,
        hint="evalchemy needs its own venv ([evalchemy] extra) and "
        "EVALCHEMY_DIR pointing at the pinned clone — see docs/VENV.md",
    ),
}


def _requirements_for_suite(canonical: str) -> SuiteRequirements | None:
    """Look up requirements for a canonical suite name, contrib included."""
    if canonical in SUITE_REQUIREMENTS:
        return SUITE_REQUIREMENTS[canonical]

    from oellm import registry

    try:
        mod = registry.get_suite(canonical)
    except KeyError:
        return None
    return SuiteRequirements(
        modules=("oellm",),
        env_vars=tuple(getattr(mod, "CLUSTER_ENV_VARS", ())),
        container_ok=False,
        hint=f"contrib suite — see oellm/contrib/{canonical}/README.md",
    )


def canonical_suites(suites: set[str] | list[str]) -> set[str]:
    """Strip ``:model_flags`` suffixes and normalise engine aliases."""
    from oellm.runner import EvalRunner

    out = set()
    for s in suites:
        head = str(s).split(":", 1)[0].strip().lower()
        if head:
            out.add(EvalRunner.canonical_name(head))
    return out


def probe_import(python_bin: str | Path, module: str) -> tuple[bool, str]:
    """Import *module* with *python_bin*; return (ok, version-or-error)."""
    code = f"import {module} as _m; print(getattr(_m, '__version__', 'unknown version'))"
    try:
        r = subprocess.run(
            [str(python_bin), "-c", code],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, f"{type(e).__name__}: {e}"
    if r.returncode == 0:
        return True, r.stdout.strip()
    err_lines = r.stderr.strip().splitlines()
    return False, err_lines[-1] if err_lines else "import failed"


def _find_executable(name: str, venv_path: str | Path | None) -> str | None:
    """Resolve *name* from the venv's bin dir first, then PATH."""
    if venv_path:
        candidate = Path(venv_path).expanduser() / "bin" / name
        if candidate.exists():
            return str(candidate)
    return shutil.which(name)


def collect_problems(
    suites: set[str] | list[str],
    *,
    venv_path: str | None,
    group_names: list[str] | None = None,
    env: dict | None = None,
) -> list[str]:
    """Return a human-readable problem list for the scheduled suite set.

    ``venv_path=None`` means container mode (the cluster Singularity image),
    which only supports lm-eval and lighteval. In venv mode each requirement
    is probed against the venv's own interpreter, so the check reflects what
    will actually run on the compute node.
    """
    env = dict(os.environ) if env is None else env
    problems: list[str] = []
    canonical = canonical_suites(suites)
    venv_python = Path(venv_path).expanduser() / "bin" / "python" if venv_path else None

    for suite in sorted(canonical):
        req = _requirements_for_suite(suite)
        if req is None:
            problems.append(
                f"suite '{suite}': unknown — not a built-in engine and not a "
                f"registered contrib plugin (the SLURM job would fail at "
                f"dispatch time)"
            )
            continue

        if venv_path is None:
            if not req.container_ok:
                problems.append(
                    f"suite '{suite}': not available in the cluster container "
                    f"image (it only ships lm-eval and lighteval). {req.hint}"
                )
            # Container contents can't be probed cheaply from the login node;
            # the static container_ok flag is the contract.
            continue

        for module in req.modules:
            ok, detail = probe_import(venv_python, module)
            if not ok:
                problems.append(
                    f"suite '{suite}': module '{module}' is not importable in "
                    f"venv {venv_path} ({detail}). {req.hint}"
                )

        for exe in req.executables:
            if _find_executable(exe, venv_path) is None:
                problems.append(
                    f"suite '{suite}': executable '{exe}' not found in "
                    f"{venv_path}/bin or on PATH. {req.hint}"
                )

        for var in req.env_vars:
            value = env.get(var, "")
            if not value:
                problems.append(
                    f"suite '{suite}': required environment variable {var} is "
                    f"not set (add it to clusters.yaml for this cluster). "
                    f"{req.hint}"
                )
            elif not Path(value).exists():
                problems.append(
                    f"suite '{suite}': {var}={value!r} does not exist on this "
                    f"filesystem. {req.hint}"
                )

    for group in group_names or []:
        for module, required_version in GROUP_VERSION_PINS.get(group, ()):
            if venv_path is None:
                problems.append(
                    f"task group '{group}': requires {module}=={required_version}, "
                    f"but the container image ships an unpinned {module} — the "
                    f"run would produce silently wrong scores. Use the "
                    f"dedicated venv (docs/VENV.md)."
                )
                continue
            ok, version = probe_import(venv_python, module)
            if ok and version != required_version:
                problems.append(
                    f"task group '{group}': requires {module}=={required_version} "
                    f"but venv {venv_path} has {module}=={version} — scores "
                    f"would be silently wrong. Use the dedicated venv "
                    f"(docs/VENV.md)."
                )

    return problems


def check_scheduled_environment(
    suites: set[str] | list[str],
    *,
    venv_path: str | None,
    group_names: list[str] | None = None,
    env: dict | None = None,
) -> None:
    """Raise ``SystemExit`` if the runtime cannot execute the scheduled suites."""
    problems = collect_problems(
        suites, venv_path=venv_path, group_names=group_names, env=env
    )
    if problems:
        bullet_list = "\n".join(f"  - {p}" for p in problems)
        raise SystemExit(
            f"Environment pre-flight failed — the configured runtime cannot "
            f"run the scheduled suites:\n{bullet_list}\n\n"
            f"Fix the environment (docs/VENV.md), run `oellm-eval doctor` for "
            f"a full report, or bypass with --skip-checks if you know better."
        )


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------

OK = "ok"
WARN = "warn"
FAIL = "fail"


@dataclass
class CheckResult:
    name: str
    status: str  # OK | WARN | FAIL
    detail: str = ""


@dataclass
class _Doctor:
    """Accumulates check results; every check is crash-isolated."""

    results: list[CheckResult] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str = "") -> None:
        self.results.append(CheckResult(name, status, detail))

    def run(self, name: str, fn) -> None:
        try:
            fn()
        except Exception as e:  # noqa: BLE001 — a broken check must not kill the report
            self.add(name, WARN, f"check crashed: {type(e).__name__}: {e}")


def run_doctor_checks(
    *,
    venv_path: str | None = None,
    task_groups: list[str] | None = None,
) -> list[CheckResult]:
    """Run the full environment diagnostic and return the results.

    When *task_groups* is given, the suites they expand to are treated as
    required: missing engines become FAIL instead of WARN.
    """
    d = _Doctor()

    # ── cluster detection + required vars ────────────────────────────────
    def _cluster() -> None:
        import socket

        from oellm.utils import _load_cluster_env

        hostname = socket.gethostname()
        try:
            _load_cluster_env()
            d.add("cluster detection", OK, f"hostname {hostname} matched")
        except ValueError:
            d.add(
                "cluster detection",
                WARN,
                f"hostname {hostname} matches no clusters.yaml entry "
                f"(fine on a laptop; SLURM submission needs a match or "
                f"explicit env vars)",
            )
        except RuntimeError as e:
            d.add("cluster detection", FAIL, str(e))

    d.run("cluster detection", _cluster)

    def _required_vars() -> None:
        for var in (
            "PARTITION",
            "ACCOUNT",
            "EVAL_BASE_DIR",
            "EVAL_OUTPUT_DIR",
            "GPUS_PER_NODE",
            "HF_HOME",
        ):
            value = os.environ.get(var, "")
            if value and "{" not in value:
                d.add(f"env: {var}", OK, value)
            else:
                d.add(
                    f"env: {var}",
                    WARN,
                    "not set / unresolved (required for SLURM submission)",
                )

    d.run("required env vars", _required_vars)

    # ── HF cache ──────────────────────────────────────────────────────────
    def _hf_home() -> None:
        hf_home = os.environ.get("HF_HOME")
        if not hf_home:
            return  # already reported above
        p = Path(hf_home)
        if not p.exists():
            d.add("HF_HOME path", WARN, f"{hf_home} does not exist yet")
            return
        if not os.access(p, os.W_OK):
            d.add("HF_HOME path", FAIL, f"{hf_home} is not writable")
            return
        free_gb = shutil.disk_usage(p).free / 1e9
        status = OK if free_gb > 30 else WARN
        d.add("HF_HOME path", status, f"{hf_home} (free: {free_gb:.0f} GB)")

    d.run("HF_HOME path", _hf_home)

    # ── SLURM binaries ────────────────────────────────────────────────────
    def _slurm() -> None:
        for binary in ("sbatch", "squeue"):
            path = shutil.which(binary)
            d.add(
                f"slurm: {binary}",
                OK if path else WARN,
                path or "not on PATH (only matters for SLURM submission)",
            )

    d.run("slurm binaries", _slurm)

    # ── container image ───────────────────────────────────────────────────
    def _container() -> None:
        base = os.environ.get("EVAL_BASE_DIR", "")
        image = os.environ.get("EVAL_CONTAINER_IMAGE", "")
        if not (base and image):
            d.add("container image", WARN, "EVAL_BASE_DIR/EVAL_CONTAINER_IMAGE not set")
            return
        path = Path(base) / image
        if path.exists():
            import datetime

            age_days = (
                datetime.datetime.now()
                - datetime.datetime.fromtimestamp(path.stat().st_mtime)
            ).days
            d.add("container image", OK, f"{path} ({age_days} days old)")
        else:
            d.add(
                "container image",
                WARN,
                f"{path} not present (fetched automatically at schedule time)",
            )

    d.run("container image", _container)

    # ── which suites are required? ────────────────────────────────────────
    required_suites: set[str] = set()
    if task_groups:

        def _expand() -> None:
            from oellm.task_groups import _expand_task_groups

            expanded = _expand_task_groups(task_groups)
            required_suites.update(canonical_suites({r.suite for r in expanded}))
            d.add(
                "task groups",
                OK,
                f"{', '.join(task_groups)} → suites: "
                f"{', '.join(sorted(required_suites))}",
            )

        d.run("task groups", _expand)

    # ── venv engine probes ────────────────────────────────────────────────
    def _venv() -> None:
        if not venv_path:
            if required_suites - _CONTAINER_SUPPORTED_SUITES:
                missing = sorted(required_suites - _CONTAINER_SUPPORTED_SUITES)
                d.add(
                    "runtime mode",
                    FAIL,
                    f"no --venv-path, and the container image cannot run: "
                    f"{', '.join(missing)} (it only ships lm-eval + lighteval)",
                )
            else:
                d.add("runtime mode", OK, "container mode (lm-eval + lighteval)")
            return

        venv = Path(venv_path).expanduser()
        python_bin = venv / "bin" / "python"
        if not python_bin.exists():
            d.add("venv", FAIL, f"{python_bin} does not exist")
            return
        d.add("venv", OK, str(venv))

        probe_targets = set(SUITE_REQUIREMENTS) | required_suites
        for suite in sorted(set(probe_targets)):
            req = _requirements_for_suite(suite)
            if req is None:
                continue
            needed = suite in required_suites
            for module in req.modules:
                ok, detail = probe_import(python_bin, module)
                status = OK if ok else (FAIL if needed else WARN)
                d.add(f"venv: import {module} ({suite})", status, detail)
            for exe in req.executables:
                found = _find_executable(exe, venv_path)
                status = OK if found else (FAIL if needed else WARN)
                d.add(f"venv: executable {exe} ({suite})", status, found or "not found")
            for var in req.env_vars:
                value = os.environ.get(var, "")
                if value and Path(value).exists():
                    d.add(f"env: {var} ({suite})", OK, value)
                else:
                    status = FAIL if needed else WARN
                    detail = f"{value!r} does not exist" if value else "not set"
                    d.add(f"env: {var} ({suite})", status, detail)

    d.run("venv", _venv)

    return d.results
