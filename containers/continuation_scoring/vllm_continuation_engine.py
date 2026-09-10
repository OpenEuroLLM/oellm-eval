"""Opt-in continuation logprobs for the pinned vLLM 0.28 GPU model runner.

Installed as vllm.oellm_continuation. Unscored context positions are None;
prefix-cache lookup stops before the hidden state predicting the first target.
"""
OELLM_CONTINUATION_API = 1
KEY = "oellm_prompt_logprobs_start"


def start_position(params, prompt_length=None):
    extra = getattr(params, "extra_args", None) or {}
    start = extra.get(KEY, 1)
    if KEY in extra:
        if type(start) is not int or start < 1:
            raise ValueError("Continuation start must be a positive token index")
        if getattr(params, "prompt_logprobs", None) is None:
            raise ValueError("Continuation start requires prompt_logprobs")
        if getattr(params, "flat_logprobs", False):
            raise ValueError("Continuation logprobs currently require flat_logprobs=False")
        if prompt_length is not None and start >= prompt_length:
            raise ValueError("Continuation start must precede the end of the prompt")
    return start


def initial_logprobs(params, prompt_ids, factory):
    start = start_position(params, len(prompt_ids) if prompt_ids is not None else None)
    if start > 1:
        if prompt_ids is None:
            raise ValueError("Continuation scoring requires prompt token IDs")
        return [None] * start
    return factory(params.flat_logprobs)


def cache_limit(request, limit):
    params = getattr(request, "sampling_params", None)
    if KEY in (getattr(params, "extra_args", None) or {}):
        return min(limit, start_position(params, request.num_tokens) - 1)
    return limit


def scoring_ranges(query_starts, computed, scheduled, prompt_lens, starts, active):
    """Map target positions to hidden-state rows, including prefill chunk edges."""
    indices, ranges = [], []
    for i in range(len(active)):
        offset = len(indices)
        if active[i]:
            first_target = max(int(computed[i]) + 1, int(starts[i]))
            last_target = min(int(computed[i]) + int(scheduled[i]), int(prompt_lens[i]) - 1)
            first_row = int(query_starts[i]) + first_target - int(computed[i]) - 1
            count = max(0, last_target - first_target + 1)
            indices.extend(range(first_row, first_row + count))
        ranges.append((offset, len(indices)))
    return indices, ranges


def compute_continuation(worker, logits_fn, hidden_states, batch, all_token_ids,
                         num_computed_tokens, prompt_lens):
    import numpy as np
    import torch
    from vllm.v1.outputs import LogprobsTensors
    from vllm.v1.worker.gpu.sample.prompt_logprob import (
        get_prompt_logprobs_token_ids, compute_prompt_logprobs_with_chunking,
    )
    mapping = batch.idx_mapping_np
    lens = prompt_lens[mapping]
    computed = batch.num_computed_prefill_tokens_np
    active = worker.uses_prompt_logprobs[mapping] & (computed < lens) & ~(lens < batch.prefill_len_np)
    if not np.any(active):
        return {}
    starts = worker.oellm_prompt_starts[mapping]
    indices, ranges = scoring_ranges(batch.query_start_loc_np, computed,
                                    batch.num_scheduled_tokens, lens, starts, active)
    requested = worker.num_prompt_logprobs[mapping]
    values = None
    if indices:
        positions = torch.tensor(indices, dtype=torch.long, device=hidden_states.device)
        token_ids = get_prompt_logprobs_token_ids(batch.num_tokens, batch.query_start_loc,
                                                 batch.idx_mapping, num_computed_tokens, all_token_ids)
        maximum = -1 if np.any(requested[active] == -1) else int(requested[active].max())
        values = compute_prompt_logprobs_with_chunking(
            token_ids.index_select(0, positions), hidden_states.index_select(0, positions),
            logits_fn, maximum, worker.logprobs_mode)
    output = {}
    for i, req_id in enumerate(batch.req_ids):
        if not active[i]:
            continue
        begin, end = ranges[i]
        pending = worker.in_progress_prompt_logprobs[req_id]
        if begin < end:
            tokens, scores, ranks = values
            width = scores.shape[1] if requested[i] == -1 else int(requested[i]) + 1
            pending.append(LogprobsTensors(tokens[begin:end, :width], scores[begin:end, :width], ranks[begin:end]))
        if computed[i] + batch.num_scheduled_tokens[i] < lens[i]:
            continue
        if pending:
            output[req_id] = LogprobsTensors.cat(pending) if len(pending) > 1 else pending[0]
            pending.clear()
    return output
