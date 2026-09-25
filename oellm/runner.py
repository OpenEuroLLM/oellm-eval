"""EvalRunner — orchestration layer for eval engine routing.

Formalises the suite-resolution logic that determines which eval engine
handles each :class:`~oellm.constants.EvaluationJob`.  The design-doc
calls this *Layer 2 — EvalRunner + Engine Routers*.

Supported engines
-----------------
* **lm-eval** (text / multilingual)
* **lighteval** (text / multilingual)
* **lmms-eval** (image / video / audio) — adapter class auto-detected
* **contrib** suites discovered by :mod:`oellm.registry`

The runner does **not** execute jobs — execution happens inside SLURM via
``template.sbatch``.  Its job is to prepare the ``eval_suite`` column
(including adapter suffixes) so the bash-side ``case`` statement can
route correctly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from oellm.constants import EvaluationJob, detect_lmms_model_type


@dataclass(frozen=True)
class EngineInfo:
    """Metadata for a known eval engine."""

    name: str
    aliases: tuple[str, ...]


# Known built-in engines and their normalised aliases.
ENGINES: tuple[EngineInfo, ...] = (
    EngineInfo(name="lm_eval", aliases=("lm-eval", "lm-eval-harness")),
    EngineInfo(name="lighteval", aliases=("light-eval",)),
    EngineInfo(name="lmms_eval", aliases=("lmms-eval",)),
)

# Build a fast lookup: alias → canonical engine name
_ALIAS_MAP: dict[str, str] = {}
for _engine in ENGINES:
    _ALIAS_MAP[_engine.name] = _engine.name
    for _alias in _engine.aliases:
        _ALIAS_MAP[_alias] = _engine.name


class EvalRunner:
    """Resolve eval suites and prepare jobs for SLURM scheduling.

    Usage::

        runner = EvalRunner()
        prepared = runner.prepare_jobs(expanded_eval_jobs)
    """

    def resolve_suite(self, job: EvaluationJob) -> str:
        """Return the final ``eval_suite`` string for *job*.

        For ``lmms_eval`` jobs the adapter class is auto-detected and
        appended as ``lmms_eval:<adapter>``.  For contrib suites the
        registry's ``detect_model_flags()`` provides the same service.
        """
        raw = str(job.eval_suite)
        suite = raw.strip()
        # Normalise the engine head for the alias lookup: CSV-sourced suites
        # may arrive mixed-case or whitespace-padded (``LMMS_EVAL``,
        # `` lmms_eval``). envcheck normalises before validating, so without
        # this the row would pass pre-flight and crash on the compute node
        # after queue wait and model load.
        head, _sep, tail = suite.partition(":")
        canonical = _ALIAS_MAP.get(head.strip().lower(), suite)

        if canonical == "lmms_eval":
            if tail:
                # Already resolved (e.g. a --check re-schedule CSV round
                # trip) — keep the adapter, normalise the head.
                return f"lmms_eval:{tail}"
            adapter = detect_lmms_model_type(str(job.model_path))
            resolved = f"lmms_eval:{adapter}"
            logging.debug("lmms-eval adapter for %s: %s", job.model_path, adapter)
            return resolved

        # Contrib suites — attempt model-flag detection via the registry.
        from oellm import registry as _registry  # noqa: PLC0415

        try:
            mod = _registry.get_suite(suite)
            if hasattr(mod, "detect_model_flags"):
                flags = mod.detect_model_flags(str(job.model_path))
                if flags:
                    logging.debug(
                        "Contrib suite flags for %s (%s): %s",
                        job.model_path,
                        mod.SUITE_NAME,
                        flags,
                    )
                    return f"{suite}:{flags}"
        except KeyError:
            pass  # Not a registered contrib suite — pass through unchanged

        return suite

    def prepare_jobs(self, jobs: list[EvaluationJob]) -> list[EvaluationJob]:
        """Resolve suites for all *jobs* in-place and return them."""
        for job in jobs:
            job.eval_suite = self.resolve_suite(job)
        return jobs

    @staticmethod
    def canonical_name(suite: str) -> str:
        """Return the canonical engine name for *suite* (or *suite* itself)."""
        return _ALIAS_MAP.get(suite, suite)

    @staticmethod
    def known_engines() -> list[str]:
        """Return canonical names of all built-in engines."""
        return [e.name for e in ENGINES]
