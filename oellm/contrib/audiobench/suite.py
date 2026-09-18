"""AudioBench contrib suite — plugin protocol implementation.

AudioBench is not pip-installable (upstream has no build backend and uses
bare imports like ``from dataset import ...``), so :func:`run` invokes its
``src/main_evaluate.py`` entry point as a subprocess with ``cwd`` set to
``$AUDIOBENCH_DIR``.  :func:`run` then re-shapes AudioBench's result JSON
into a lmms-eval-compatible payload that :func:`oellm.main.collect_results`
can parse unchanged.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path

from oellm.contrib.audiobench.task import (
    AUDIOBENCH_TASKS,
    SUITE_NAME,
    AudioBenchTaskSpec,
    get_task_spec,
)

logger = logging.getLogger(__name__)

CLUSTER_ENV_VARS = ["AUDIOBENCH_DIR"]

_FAMILY_GROUPS = {
    "asr": (
        "audio-audiobench-asr",
        "AudioBench ASR tasks (WER).",
    ),
    "st": (
        "audio-audiobench-st",
        "AudioBench speech-translation tasks (BLEU).",
    ),
    "reasoning": (
        "audio-audiobench-reasoning",
        "AudioBench spoken reasoning / captioning (accuracy / string_match / METEOR).",
    ),
}

_TOP_LEVEL_GROUP = "audio-audiobench"
_TOP_LEVEL_DESC = (
    "AudioBench suite — ASR (WER), speech translation (BLEU), spoken "
    "reasoning (accuracy/string_match), and AudioCaps captioning (METEOR)."
)


def _build_task_groups() -> dict:
    """Build ``TASK_GROUPS`` from :data:`AUDIOBENCH_TASKS`.

    Always zero-shot — AudioBench does not support in-context examples.
    """
    task_metrics: dict[str, str] = {t.name: t.metric for t in AUDIOBENCH_TASKS}

    def _task_entry(t: AudioBenchTaskSpec) -> dict:
        # No ``subset`` — for gigaspeech2 / spoken-mqa the upstream split
        # selection is encoded in ``upstream_name`` itself (e.g.
        # ``gigaspeech2_thai``).  The ``audio-*`` group prefix triggers
        # full-repo snapshot_download in :func:`_collect_dataset_specs`.
        return {"task": t.name, "dataset": t.hf_repo}

    groups: dict[str, dict] = {}

    tasks_by_family: dict[str, list[AudioBenchTaskSpec]] = {
        "asr": [],
        "st": [],
        "reasoning": [],
    }
    for t in AUDIOBENCH_TASKS:
        tasks_by_family[t.family].append(t)

    for family, (group_name, desc) in _FAMILY_GROUPS.items():
        entries = tasks_by_family[family]
        if not entries:
            continue
        groups[group_name] = {
            "suite": SUITE_NAME,
            "n_shots": [0],
            "description": desc,
            "tasks": [_task_entry(t) for t in entries],
        }

    groups[_TOP_LEVEL_GROUP] = {
        "suite": SUITE_NAME,
        "n_shots": [0],
        "description": _TOP_LEVEL_DESC,
        "tasks": [_task_entry(t) for t in AUDIOBENCH_TASKS],
    }

    return {"task_metrics": task_metrics, "task_groups": groups}


TASK_GROUPS: dict = _build_task_groups()


def detect_model_flags(model_path: str) -> str | None:
    """Return AudioBench's literal ``--model_name`` dispatch key for *model_path*.

    Returns ``None`` when *model_path* does not match any AudioBench-supported
    model family — :func:`run` then raises a clear error.  AudioBench has no
    generic loader, so silently falling back to a fictitious key would just
    move the error deeper inside the subprocess.
    """
    from oellm.contrib.audiobench.adapter import AudioBenchModelAdapter

    return AudioBenchModelAdapter(model_path).to_contrib_flags()


def run(
    *,
    model_path: str,
    task: str,
    n_shot: int,
    output_path: Path,
    model_flags: str | None,
    env: dict[str, str],
) -> None:
    """Execute one AudioBench task and write a lmms-eval-shaped result JSON.

    Raises ``RuntimeError`` if AudioBench exits non-zero or produces no
    parseable output, and ``KeyError`` if *task* is not registered.
    """
    ab_dir = env.get("AUDIOBENCH_DIR")
    if not ab_dir:
        raise RuntimeError(
            "AUDIOBENCH_DIR must be set.  Add it to clusters.yaml — "
            "it should point at a local clone of "
            "https://github.com/AudioLLMs/AudioBench."
        )

    entrypoint = Path(ab_dir) / "src" / "main_evaluate.py"
    if not entrypoint.exists():
        raise FileNotFoundError(
            f"AudioBench entry point not found: {entrypoint}\n"
            f"Check that AUDIOBENCH_DIR={ab_dir!r} points at a valid "
            "AudioBench clone."
        )

    spec = get_task_spec(task)
    if not model_flags:
        raise RuntimeError(
            f"Could not map model_path={model_path!r} to an AudioBench-supported "
            f"model.  AudioBench dispatches on a fixed list of literal "
            f"model_name strings (Qwen2-Audio-7B-Instruct, SALMONN_7B, "
            f"whisper_large_v3, …) — see oellm/contrib/audiobench/adapter.py.  "
            f"AudioBench cannot evaluate arbitrary HF checkpoints; it loads "
            f"its own hardcoded HF repos per model family."
        )
    model_key = model_flags  # AudioBench's dispatch key, e.g. "Qwen2-Audio-7B-Instruct"

    cmd = [
        "python",
        "src/main_evaluate.py",
        "--dataset_name",
        spec.upstream_name,
        "--model_name",
        model_key,
        "--metrics",
        spec.upstream_metric,
        # Force re-eval — AudioBench skips by default if a stale score file
        # already exists under log_for_all_models/.
        "--overwrite",
        "True",
    ]

    limit = env.get("LIMIT", "").strip()
    if limit:
        cmd.extend(["--number_of_samples", str(limit)])

    logger.info("AudioBench cmd: %s (cwd=%s)", " ".join(cmd), ab_dir)
    completed = subprocess.run(
        cmd,
        cwd=ab_dir,
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"AudioBench exited with code {completed.returncode} for "
            f"task={task!r} model={model_path!r} (dispatch key={model_key!r})"
        )

    metrics = _extract_metrics(
        audiobench_dir=Path(ab_dir), model_key=model_key, spec=spec
    )
    _write_lmms_shaped_json(
        output_path=output_path,
        model_path=model_path,
        task_name=task,
        n_shot=n_shot,
        metrics=metrics,
    )
    logger.info("Results written to %s", output_path)


def _extract_metrics(
    *,
    audiobench_dir: Path,
    model_key: str,
    spec: AudioBenchTaskSpec,
) -> dict[str, float]:
    """Read AudioBench's score file from its hardcoded output path.

    AudioBench writes to ``$cwd/log_for_all_models/<model_name>/<dataset_name>_<metric>_score.json``
    (see ``main_evaluate.py:118``).  Path is fixed — there is no ``--log_dir``.
    """
    score_file = (
        audiobench_dir
        / "log_for_all_models"
        / model_key
        / f"{spec.upstream_name}_{spec.upstream_metric}_score.json"
    )
    if not score_file.exists():
        raise RuntimeError(
            f"AudioBench did not write expected score file at {score_file}.  "
            f"Either AudioBench crashed silently, or the dispatch key "
            f"{model_key!r} / dataset_name {spec.upstream_name!r} / metric "
            f"{spec.upstream_metric!r} is wrong.  Check stdout/stderr."
        )

    try:
        with open(score_file) as f:
            body = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        raise RuntimeError(
            f"Could not read AudioBench score file {score_file}: {e}"
        ) from e

    value = _find_metric(body, spec.upstream_metric)
    if value is None:
        raise RuntimeError(
            f"Could not locate metric {spec.upstream_metric!r} in AudioBench "
            f"score file {score_file}.  Body: {body!r}"
        )
    # Emit under our canonical key so collect_results' metric resolution
    # picks up task_metrics.yaml.
    return {spec.metric: float(value)}


def _find_metric(body: object, key: str) -> float | None:
    """Recursive search for a numeric value keyed by *key*.

    Tolerates both ``{"wer": 0.04}`` and ``{"metrics": {"wer": {"score":
    0.04}}}`` layouts — upstream log shape has drifted across releases.
    """
    if isinstance(body, dict):
        if key in body:
            candidate = body[key]
            if isinstance(candidate, int | float):
                return float(candidate)
            if isinstance(candidate, dict) and "score" in candidate:
                score = candidate["score"]
                if isinstance(score, int | float):
                    return float(score)
        for v in body.values():
            found = _find_metric(v, key)
            if found is not None:
                return found
    elif isinstance(body, list):
        for item in body:
            found = _find_metric(item, key)
            if found is not None:
                return found
    return None


def _write_lmms_shaped_json(
    *,
    output_path: Path,
    model_path: str,
    task_name: str,
    n_shot: int,
    metrics: dict[str, float],
) -> None:
    payload = {
        "model_name_or_path": model_path,
        "results": {task_name: metrics},
        "configs": {task_name: {"num_fewshot": n_shot}},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)


def parse_results(data: dict) -> tuple[str, str, int, dict[str, float]] | None:
    """Recognise a JSON dict produced by :func:`run` and return
    ``(model_id, task_name, n_shot, metrics)``; ``None`` if it's not ours.
    """
    results = data.get("results", {})
    if not isinstance(results, dict):
        return None
    for task_name, task_results in results.items():
        if not isinstance(task_name, str) or not task_name.startswith("audiobench_"):
            continue
        if not isinstance(task_results, dict):
            continue
        model_id = data.get("model_name_or_path") or data.get("model_name") or "unknown"
        n_shot = data.get("configs", {}).get(task_name, {}).get("num_fewshot", 0)
        coerced: dict[str, float] = {}
        for k, v in task_results.items():
            if isinstance(v, int | float):
                coerced[k] = float(v)
        return model_id, task_name, int(n_shot), coerced
    return None


__all__ = [
    "CLUSTER_ENV_VARS",
    "SUITE_NAME",
    "TASK_GROUPS",
    "detect_model_flags",
    "parse_results",
    "run",
]

_ = os  # exported via env dict passed to subprocess.run
