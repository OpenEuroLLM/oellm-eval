"""Contrib plugin registry.

Auto-discovers suite modules under ``oellm/contrib/*/suite.py`` and provides a
unified view of all task groups and metric parsers contributed by them.

Plugin protocol
---------------
A file ``oellm/contrib/<name>/suite.py`` is a plugin if it exposes:

Required
~~~~~~~~
``SUITE_NAME: str``
    Identifier used in the ``eval_suite`` CSV column, e.g. ``"regiondial_bench"``.

``TASK_GROUPS: dict``
    Task-group definitions in ``task-groups.yaml`` format.  Expected keys:
    ``task_metrics``, ``task_groups``, ``super_groups`` (all optional).

``run(*, model_path, task, n_shot, output_path, model_flags, env) -> None``
    Execute the evaluation.  Must write a lmms-eval-compatible JSON file to
    *output_path* so that :func:`oellm.main.collect_results` can parse it
    without changes.

``parse_results(data: dict) -> tuple | None``
    Try to parse a raw JSON dict produced by this suite.  Returns
    ``(model_id, task_name, n_shot, {metric: value})`` or ``None``.
    Called by :func:`oellm.results.collect_results` as a first-chance
    parser — a suite that claims a file owns its format outright.

Optional
~~~~~~~~
``CLUSTER_ENV_VARS: list[str]``
    Names of environment variables that must be set on the cluster.
    Validated by ``oellm.contrib.dispatch`` before calling ``run()``.

``detect_model_flags(model_path: str) -> str | None``
    Return a model-type suffix for the ``eval_suite`` column
    (e.g. ``"vision_reasoner"``), or ``None``.

``LMMS_MODEL_ADAPTERS: list[tuple[list[str], str]]``
    lmms-eval adapter detection patterns (same shape as
    ``oellm.constants.LMMS_MODEL_ADAPTERS``). Consulted BEFORE the built-in
    table by ``detect_lmms_model_type`` so a plugin can route new model
    families without core edits.

Adding a new benchmark
----------------------
Drop files into ``oellm/contrib/<name>/``.  No core file changes required.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
import types
from functools import cache

logger = logging.getLogger(__name__)


@cache
def _discover() -> dict[str, types.ModuleType]:
    """Discover and return all suite modules, keyed by SUITE_NAME.

    Runs once per process (cached).  Import errors in individual contribs are
    logged as warnings and do not propagate.
    """
    import oellm.contrib as _contrib_pkg

    suites: dict[str, types.ModuleType] = {}

    for _importer, modname, _ispkg in pkgutil.walk_packages(
        path=_contrib_pkg.__path__,
        prefix="oellm.contrib.",
        onerror=lambda name: logger.warning("Error walking package %s", name),
    ):
        if not modname.endswith(".suite"):
            continue
        try:
            mod = importlib.import_module(modname)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to import contrib suite %s: %s", modname, exc)
            continue

        if not hasattr(mod, "SUITE_NAME"):
            continue

        suite_name: str = mod.SUITE_NAME
        if suite_name in suites:
            logger.warning(
                "Duplicate SUITE_NAME %r: %s overrides %s",
                suite_name,
                modname,
                suites[suite_name].__name__,
            )
        suites[suite_name] = mod
        logger.debug("Registered contrib suite: %r (%s)", suite_name, modname)

    return suites


def get_suite(name: str) -> types.ModuleType:
    """Return the suite module registered under *name*.

    Raises
    ------
    KeyError
        If no suite with that name is registered.  The error message includes
        all known suite names to help diagnose typos.
    """
    suites = _discover()
    if name not in suites:
        known = ", ".join(sorted(suites)) or "(none)"
        raise KeyError(
            f"Unknown contrib suite {name!r}.  Known suites: {known}.  "
            "Make sure the suite module is under oellm/contrib/<name>/suite.py "
            "and exposes a SUITE_NAME constant."
        )
    return suites[name]


def get_all_suites() -> list[types.ModuleType]:
    """Return all discovered suite modules."""
    return list(_discover().values())


def get_lmms_adapter_overrides() -> list[tuple[list[str], str]]:
    """Contrib-registered lmms-eval adapter detection patterns.

    Aggregates the optional ``LMMS_MODEL_ADAPTERS`` export of every
    discovered suite; entries take precedence over the built-in table in
    ``oellm.constants``.
    """
    overrides: list[tuple[list[str], str]] = []
    for mod in _discover().values():
        overrides.extend(getattr(mod, "LMMS_MODEL_ADAPTERS", None) or [])
    return overrides


def get_all_task_groups() -> dict:
    """Merge TASK_GROUPS from all discovered suites into a single dict.

    Returns a dict with keys ``task_metrics``, ``task_groups``,
    ``super_groups`` suitable for merging into the core YAML data.
    """
    merged: dict = {"task_metrics": {}, "task_groups": {}, "super_groups": {}}
    for mod in _discover().values():
        tg = getattr(mod, "TASK_GROUPS", {})
        for key in ("task_metrics", "task_groups", "super_groups"):
            merged[key].update(tg.get(key, {}))
    return merged
