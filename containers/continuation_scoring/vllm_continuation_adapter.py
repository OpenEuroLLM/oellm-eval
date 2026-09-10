"""Continuation-scoring feature checks and balanced context-local MP sharding."""
from importlib import import_module
import os

KEY = "oellm_prompt_logprobs_start"


def available(enabled):
    if isinstance(enabled, str):
        if enabled.lower() not in {"true", "false"}:
            raise ValueError("continuation_loglikelihood must be true or false")
        enabled = enabled.lower() == "true"
    if not enabled:
        return False
    try:
        helper = import_module("vllm.oellm_continuation")
    except ModuleNotFoundError as exc:
        if exc.name != "vllm.oellm_continuation":
            raise
        return False
    for name in ("vllm.v1.engine.logprobs", "vllm.v1.core.kv_cache_manager",
                 "vllm.v1.worker.gpu.sample.prompt_logprob"):
        if getattr(import_module(name), "OELLM_CONTINUATION_API", None) != helper.OELLM_CONTINUATION_API:
            raise RuntimeError(f"Incomplete continuation-scoring engine installation: {name}")
    if os.environ.get("VLLM_USE_V2_MODEL_RUNNER", "1").lower() not in {"1", "true"}:
        raise ValueError("Continuation scoring requires the vLLM V2 model runner")
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    return True


def partition(requests, params, degree, wave_size=32):
    if len(requests) != len(params):
        raise ValueError("Sampling parameter count differs")
    if not any(KEY in (getattr(p, "extra_args", None) or {}) for p in params):
        return [list(range(rank, len(requests), degree)) for rank in range(degree)]
    groups = {}
    for i, (tokens, p) in enumerate(zip(requests, params, strict=True)):
        boundary = (getattr(p, "extra_args", None) or {}).get(KEY)
        key = ("context", tuple(tokens[:boundary])) if boundary is not None else ("request", i)
        groups.setdefault(key, []).append(i)
    shards = [[] for _ in range(min(degree, len(groups)))]
    load = [0] * len(shards)
    for group in sorted(groups.values(), key=lambda g: sum(len(requests[i]) for i in g), reverse=True):
        rank = min(range(len(shards)), key=load.__getitem__)
        shards[rank].append(group)
        load[rank] += sum(len(requests[i]) for i in group)
    # Submit first endings before later alternatives in bounded tiles. Keeping
    # all four alternatives adjacent would often prefill them concurrently,
    # before any sibling's prefix had become reusable.
    ordered = []
    for groups_on_rank in shards:
        indices = []
        for offset in range(0, len(groups_on_rank), wave_size):
            tile = groups_on_rank[offset:offset+wave_size]
            for choice in range(max(map(len,tile))):
                indices.extend(group[choice] for group in tile if choice < len(group))
        ordered.append(indices)
    return ordered


def gather(shards, outputs, count):
    result = [None] * count
    for indices, values in zip(shards, outputs, strict=True):
        for index, value in zip(indices, values, strict=True):
            if result[index] is not None:
                raise ValueError("Duplicate gathered request")
            result[index] = value
    if any(value is None for value in result):
        raise ValueError("Missing gathered request")
    return result
