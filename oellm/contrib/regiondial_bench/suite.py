"""RegionDial-Bench contrib suite — plugin protocol implementation.

This module follows the plugin protocol defined in ``oellm/registry.py``.
It is the reference implementation for custom benchmark integration.

RegionDial-Bench (Sun et al., ICLR 2026) is a multi-round benchmark for
reference-grounded region reasoning, built on RefCOCO+ and RefCOCOg.

Cluster setup
-------------
The following environment variables must be set in ``clusters.yaml`` (or the
cluster's module/profile system) before using the ``regiondial-bench`` task group:

``REGION_REASONER_DIR``
    Absolute path to a local clone of the RegionReasoner repository
    (https://github.com/lmsdss/RegionReasoner).  The eval scripts
    ``test/evaluation/evaluation_multi_segmentation.py`` must be present.

The number of GPUs is read from ``GPUS_PER_NODE`` (set in ``clusters.yaml``),
which also controls the SLURM ``--gres=gpu:`` request.

Output format
-------------
``run()`` writes a **lmms-eval-compatible JSON** file so that
``oellm.main.collect_results()`` works without modification::

    {
      "model_name_or_path": "<model_path>",
      "results": {
        "regiondial_refcocog": {
          "gIoU": 0.42, "cIoU": 0.45, "bbox_AP": 0.38,
          "pass_rate_0.3": 0.71, "pass_rate_0.5": 0.55,
          "pass_rate_0.7": 0.31, "pass_rate_0.9": 0.08,
          "gIoU_R1": 0.55, "gIoU_R2": 0.48, ...
        }
      },
      "configs": {
        "regiondial_refcocog": {"num_fewshot": 0}
      }
    }
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

SUITE_NAME = "regiondial_bench"

CLUSTER_ENV_VARS = [
    "REGION_REASONER_DIR",
]

from oellm.contrib.regiondial_bench.task import (  # noqa: E402
    RegionDialRefCOCOgTask,
    RegionDialRefCOCOplusTask,
)

_refcocog = RegionDialRefCOCOgTask.to_task_groups_dict()
_refcocoplus = RegionDialRefCOCOplusTask.to_task_groups_dict()

_all_name = "regiondial-bench"
_all_metrics = {
    **_refcocog.get("task_metrics", {}),
    **_refcocoplus.get("task_metrics", {}),
}

_group_kwargs = {"suite": SUITE_NAME, "n_shots": [0]}

TASK_GROUPS: dict = {
    "task_metrics": _all_metrics,
    "task_groups": {
        _all_name: {
            **_group_kwargs,
            "description": (
                "RegionDial-Bench: both splits — RefCOCOg + RefCOCO+ "
                "(Sun et al., ICLR 2026)."
            ),
            "tasks": (
                _refcocog["task_groups"][_all_name]["tasks"]
                + _refcocoplus["task_groups"][_all_name]["tasks"]
            ),
        },
        "regiondial-refcocog": {
            **_group_kwargs,
            "description": (
                "RegionDial-Bench: RefCOCOg Multi-turn only (1,580 images, 4,405 turns)."
            ),
            "tasks": _refcocog["task_groups"][_all_name]["tasks"],
        },
        "regiondial-refcocoplus": {
            **_group_kwargs,
            "description": (
                "RegionDial-Bench: RefCOCO+ Multi-turn only (715 images, 2,355 turns)."
            ),
            "tasks": _refcocoplus["task_groups"][_all_name]["tasks"],
        },
    },
}

_TASK_JSON_FILES: dict[str, str] = {
    "regiondial_refcocog": "refcocog_multi_turn.json",
    "regiondial_refcocoplus": "refcocoplus_multi_turn.json",
}


def detect_model_flags(model_path: str) -> str | None:
    """Delegate to RegionDialModelAdapter.to_contrib_flags()."""
    from oellm.contrib.regiondial_bench.adapter import RegionDialModelAdapter

    return RegionDialModelAdapter(model_path).to_contrib_flags()


def run(
    *,
    model_path: str,
    task: str,
    n_shot: int,
    output_path: Path,
    model_flags: str | None,
    env: dict[str, str],
) -> None:
    """Execute the RegionDial-Bench evaluation and write results to *output_path*.

    This function:

    1. Resolves the correct test JSON for the requested split (RefCOCOg or
       RefCOCO+) based on the *task* name.
    2. Runs ``evaluation_multi_segmentation.py`` in parallel GPU shards.
    3. Computes aggregate and per-round (R1–R7) metrics using
       :mod:`oellm.contrib.regiondial_bench.metrics`.
    4. Writes a lmms-eval-compatible JSON to *output_path*.

    Args:
        model_path: Path or HF repo ID of the model checkpoint.
        task: Task name (``"regiondial_refcocog"`` or ``"regiondial_refcocoplus"``).
        n_shot: Number of few-shot examples (always 0 for this benchmark).
        output_path: Where to write the results JSON.
        model_flags: Model type string (e.g. ``"vision_reasoner"``).
        env: Environment variables dict (from ``os.environ``).
    """
    rr_dir = env.get("REGION_REASONER_DIR", "")
    num_gpus = int(env.get("GPUS_PER_NODE", "1"))
    model_type = model_flags or "vision_reasoner"

    if not rr_dir:
        raise RuntimeError(
            "REGION_REASONER_DIR must be set. Add it to clusters.yaml for this cluster."
        )

    json_filename = _TASK_JSON_FILES.get(task)
    if not json_filename:
        raise ValueError(
            f"Unknown task {task!r}. Expected one of: {list(_TASK_JSON_FILES)}"
        )

    test_json = _resolve_test_json(task, json_filename, env)

    inference_script = (
        Path(rr_dir) / "test" / "evaluation" / "evaluation_multi_segmentation.py"
    )
    if not inference_script.exists():
        raise FileNotFoundError(
            f"RegionReasoner inference script not found: {inference_script}\n"
            f"Check that REGION_REASONER_DIR={rr_dir!r} points to a valid clone."
        )

    with tempfile.TemporaryDirectory(prefix="rr_shards_") as tmp_dir:
        shard_paths, expected_turns = _stream_preshard(test_json, tmp_dir, num_gpus)

        procs = []
        for idx in range(num_gpus):
            shard_env = dict(env)
            shard_env["CUDA_VISIBLE_DEVICES"] = str(idx)
            # Each shard gets its own output dir and runs with
            # ``--idx 0 --num_parts 1``: the script slices its input with
            # ``dataset[idx*part_size : ...]``, so any idx > 0 combined with
            # num_parts=1 yields an EMPTY slice (the shard silently evaluates
            # nothing). Distinct output dirs replace distinct --idx values as
            # the collision guard for the script's output_{idx}.json naming.
            shard_out = Path(tmp_dir) / f"shard_{idx}"
            shard_out.mkdir()
            cmd = [
                "python",
                str(inference_script),
                "--model_path",
                model_path,
                "--model",
                model_type,
                "--test_data_path",
                shard_paths[idx],
                "--output_path",
                str(shard_out),
                "--vis_output_path",
                str(shard_out / "vis"),
                "--idx",
                "0",
                "--num_parts",
                "1",
                "--batch_size",
                "1",
                "--task_router_model_path",
                "Ricky06662/TaskRouter-1.5B",
                "--binarize_bbox_iou",
            ]
            logger.info("Starting shard %d/%d: %s", idx + 1, num_gpus, " ".join(cmd))
            proc = subprocess.Popen(cmd, env=shard_env, cwd=str(Path(test_json).parent))
            procs.append(proc)

        # Wait for ALL shards before raising: raising on the first failure
        # would leave sibling GPU processes running while the enclosing
        # TemporaryDirectory tears down the files they are writing to.
        exit_codes = [proc.wait() for proc in procs]
        failed = [(idx, ret) for idx, ret in enumerate(exit_codes) if ret != 0]
        if failed:
            raise RuntimeError(
                "RegionDial-Bench inference shard(s) failed: "
                + ", ".join(f"shard {idx} exited with code {ret}" for idx, ret in failed)
            )

        logger.info("All %d shards completed. Computing metrics.", num_gpus)

        metrics = _aggregate_shards(tmp_dir, expected_samples=expected_turns)

    result_json = {
        "model_name_or_path": model_path,
        "results": {task: metrics},
        "configs": {task: {"num_fewshot": n_shot}},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result_json, f, indent=2)
    logger.info("Results written to %s", output_path)


def _stream_preshard(
    json_path: str, out_dir: str, num_shards: int
) -> tuple[list[str], int]:
    """Split a large JSON array into *num_shards* files using streaming.

    Uses ``ijson`` to iterate over the top-level array without loading the
    entire file into memory.  Items (whole conversations) are distributed
    round-robin — turns within an item always stay together, which the
    per-round metric inference in :func:`_aggregate_shards` relies on.

    Returns ``(shard_paths, total_turns)`` where *total_turns* is the number
    of per-turn result records the inference run is expected to emit (one per
    entry in each item's ``conversational_turns``).
    """
    import ijson

    shard_files = []
    shard_counts = [0] * num_shards
    total_turns = 0
    for idx in range(num_shards):
        p = str(Path(out_dir) / f"shard_{idx}.json")
        shard_files.append(open(p, "w"))  # noqa: SIM115
        shard_files[-1].write("[\n")

    logger.info("Streaming pre-shard of %s into %d files", json_path, num_shards)

    with open(json_path, "rb") as f:
        for i, item in enumerate(ijson.items(f, "item")):
            shard_idx = i % num_shards
            if shard_counts[shard_idx] > 0:
                shard_files[shard_idx].write(",\n")
            # ijson yields decimal.Decimal for JSON floats (bbox coordinates);
            # json.dump cannot serialize Decimal without a default.
            json.dump(item, shard_files[shard_idx], default=float)
            shard_counts[shard_idx] += 1
            if isinstance(item, dict):
                total_turns += len(item.get("conversational_turns") or [])

    shard_paths = []
    for idx in range(num_shards):
        shard_files[idx].write("\n]")
        shard_files[idx].close()
        shard_paths.append(str(Path(out_dir) / f"shard_{idx}.json"))
        logger.info("Shard %d: %d samples", idx, shard_counts[idx])

    logger.info(
        "Pre-sharding complete: %d items, %d turns", sum(shard_counts), total_turns
    )
    return shard_paths, total_turns


def _resolve_test_json(task: str, json_filename: str, env: dict[str, str]) -> str:
    """Resolve the path to the test JSON for the given split.

    Uses env-var overrides if present, otherwise auto-downloads from HF Hub.
    """
    split_key = task.replace("regiondial_", "").upper()
    env_var = f"REGION_REASONER_TEST_JSON_{split_key}"
    override = env.get(env_var)
    if override:
        return override

    # Backward compat: single env var that contains the filename
    legacy = env.get("REGION_REASONER_TEST_JSON")
    if legacy and json_filename in legacy:
        return legacy

    from huggingface_hub import snapshot_download

    split_prefix = json_filename.replace("_multi_turn.json", "")
    local_dir = snapshot_download(
        repo_id="lmsdss/regionreasoner_test_data",
        repo_type="dataset",
        allow_patterns=[
            f"raw/{json_filename}",
            f"raw/{split_prefix}_test_multi_bbox_images/*",
        ],
        cache_dir=Path(env["HF_HOME"]) / "hub" if "HF_HOME" in env else None,
    )
    return str(Path(local_dir) / "raw" / json_filename)


def _aggregate_shards(
    shard_dir: str, *, expected_samples: int | None = None
) -> dict[str, float]:
    """Read per-shard output files and compute all metrics.

    Each shard output file contains a list of per-turn dicts with
    pre-computed ``image_id``, ``intersection``, ``union``, and ``bbox_iou``
    fields written by the upstream ``evaluation_multi_segmentation.py``
    script (there is no explicit round/turn field — rounds are inferred from
    ``image_id`` occurrence order below).

    Output files are discovered recursively (``shard_*/output_0.json`` from
    the per-shard run dirs, or flat ``output_*.json`` from single-GPU runs).

    Computes:
    - Aggregate metrics across all rounds: gIoU, cIoU, bbox_AP, pass_rate_*
    - Per-round metrics (R1–R7): gIoU_R1..R7, bbox_AP_R1..R7

    When *expected_samples* is given, raises if the aggregated sample count
    differs — this catches shards that silently evaluated a partial (or
    empty) slice, e.g. through upstream ``--idx``/``--num_parts`` drift.

    Returns a flat dict of ``{metric_name: value}``.
    """
    from oellm.contrib.regiondial_bench.metrics import (
        BboxAP,
        CIoU,
        GIoU,
        PassRate,
    )

    shard_files = sorted(Path(shard_dir).rglob("output_*.json"))
    if not shard_files:
        raise RuntimeError(
            f"No shard output files found in {shard_dir!r}. "
            "The inference script may have failed silently."
        )

    all_samples: list[dict] = []
    for shard_file in shard_files:
        with open(shard_file) as f:
            shard_data = json.load(f)
        all_samples.extend(shard_data)

    if not all_samples:
        raise RuntimeError(
            "No samples found across shard files. "
            "The inference script produced empty output."
        )
    if expected_samples is not None and len(all_samples) != expected_samples:
        raise RuntimeError(
            f"Shard outputs contain {len(all_samples)} samples but the input "
            f"had {expected_samples} turns. One or more shards evaluated a "
            f"partial or empty slice — refusing to report metrics over a "
            f"subset of the benchmark."
        )
    logger.info(
        "Aggregating %d samples from %d shards", len(all_samples), len(shard_files)
    )

    aggregate_metrics = [
        GIoU(),
        CIoU(),
        BboxAP(),
        PassRate(0.3),
        PassRate(0.5),
        PassRate(0.7),
        PassRate(0.9),
    ]

    metrics: dict[str, float] = {}
    for m in aggregate_metrics:
        val = m.compute(all_samples)
        metrics[m.name] = val
        logger.debug("%s = %.4f", m.name, val)

    # Infer per-round membership by counting each image_id's occurrence order in
    # the output (mirrors calculate_iou_with_bbox_by_turns.py).  The inference
    # script emits turns in sequential order per image, so the k-th time an
    # image_id appears corresponds to turn k (1-indexed).
    rounds_map: dict[int, list[dict]] = defaultdict(list)
    image_turn_counter: dict[str, int] = {}
    for sample_dict in all_samples:
        image_id = str(sample_dict.get("image_id", ""))
        image_turn_counter[image_id] = image_turn_counter.get(image_id, 0) + 1
        rnd = image_turn_counter[image_id]
        rounds_map[rnd].append(sample_dict)

    per_round_metrics = [GIoU(), BboxAP()]
    for rnd in sorted(rounds_map):
        rnd_samples = rounds_map[rnd]
        for m in per_round_metrics:
            val = m.compute(rnd_samples)
            metrics[f"{m.name}_R{rnd}"] = val
            logger.debug("%s_R%d = %.4f", m.name, rnd, val)

    return metrics


def parse_results(data: dict) -> tuple[str, str, int, dict[str, float]] | None:
    """Try to parse *data* as a RegionDial-Bench output JSON.

    Returns ``(model_id, task_name, n_shot, {metric: value})`` if the JSON
    matches this suite's format, otherwise ``None``.

    Detection heuristic: the ``results`` dict contains a key that starts with
    ``"regiondial_"`` and the value dict contains ``"gIoU"``.
    """
    results = data.get("results", {})
    for task_name, task_results in results.items():
        if task_name.startswith("regiondial_") and "gIoU" in task_results:
            model_id = data.get("model_name_or_path") or data.get("model_name", "unknown")
            n_shot = data.get("configs", {}).get(task_name, {}).get("num_fewshot", 0)
            return model_id, task_name, int(n_shot), task_results
    return None
