"""Prepare a hash-checked continuation-scoring overlay for pinned vLLM 0.28."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex

BASE_HASHES = {
    "prompt_logprob.py": "4cf22d390e7bf59e44c458cff7180268db63e95d06af2f1e14d720a2d77c9663",
    "logprobs.py": "b4ac11f40a99d6b409db8c6ee0a190e630c9ef01919104e653d2965f23e36e15",
    "kv_cache_manager.py": "2d20c3d98845cfd8d88a2f66b8fc6402ea1fcfbee16879d2364a4a9e45b8fa47",
    "vllm_causallms.py": "589d7986d617d3227809f6066846be40df777bebd2ab41060d8eda9b82e206e2",
}
ENGINE = "/usr/local/lib/python3.12/dist-packages/vllm"
HARNESS = "/opt/oellm-eval/lib/python3.12/site-packages/lm_eval/models"
TARGETS = {
    "prompt_logprob.py": ENGINE+"/v1/worker/gpu/sample/prompt_logprob.py",
    "logprobs.py": ENGINE+"/v1/engine/logprobs.py",
    "kv_cache_manager.py": ENGINE+"/v1/core/kv_cache_manager.py",
    "oellm_continuation.py": ENGINE+"/oellm_continuation.py",
    "vllm_causallms.py": HARNESS+"/vllm_causallms.py",
    "vllm_single_token.py": HARNESS+"/vllm_single_token.py",
    "vllm_continuation.py": HARNESS+"/vllm_continuation.py",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Source anchor changed: {old[:80]!r}")
    return text.replace(old, new, 1)


def prepare(engine_source, harness_source, out, profile_base=None, profile_out=None):
    sources = {n: engine_source/n for n in BASE_HASHES if n != "vllm_causallms.py"}
    sources["vllm_causallms.py"] = harness_source/"lm_eval/models/vllm_causallms.py"
    for name, p in sources.items():
        if sha(p) != BASE_HASHES[name]:
            raise ValueError(f"Unvalidated source version: {p}")
    out.mkdir(parents=True, exist_ok=False)
    marker = "from vllm.oellm_continuation import OELLM_CONTINUATION_API, start_position, compute_continuation\n"
    text = sources["prompt_logprob.py"].read_text()
    text = once(text, "class PromptLogprobsWorker:", marker+"\nclass PromptLogprobsWorker:")
    text = once(text, "        self.logprobs_mode = logprobs_mode", "        self.logprobs_mode = logprobs_mode\n        self.oellm_prompt_starts = np.ones(max_num_reqs, dtype=np.int32)")
    text = once(text, "        uses_prompt_logprobs = sampling_params.prompt_logprobs is not None", "        self.oellm_prompt_starts[req_idx] = start_position(sampling_params)\n        uses_prompt_logprobs = sampling_params.prompt_logprobs is not None")
    text = once(text, "        idx_mapping_np = input_batch.idx_mapping_np", "        if np.any(self.oellm_prompt_starts[input_batch.idx_mapping_np] > 1):\n            return compute_continuation(self, logits_fn, hidden_states, input_batch,\n                                        all_token_ids, num_computed_tokens, prompt_lens)\n        idx_mapping_np = input_batch.idx_mapping_np")
    (out/"prompt_logprob.py").write_text(text)
    text = sources["logprobs.py"].read_text()
    text = once(text, "logger = init_logger(__name__)", "from vllm.oellm_continuation import OELLM_CONTINUATION_API, initial_logprobs\n\nlogger = init_logger(__name__)")
    text = once(text, "else create_prompt_logprobs(sampling_params.flat_logprobs)", "else initial_logprobs(sampling_params, request.prompt_token_ids, create_prompt_logprobs)")
    (out/"logprobs.py").write_text(text)
    text = sources["kv_cache_manager.py"].read_text()
    text = once(text, "        max_cache_hit_length = request.num_tokens - 1", "        max_cache_hit_length = cache_limit(request, request.num_tokens - 1)")
    anchor = "from __future__ import annotations\n"
    if anchor in text:
        text = once(text, anchor, anchor+"\nfrom vllm.oellm_continuation import OELLM_CONTINUATION_API, cache_limit\n")
    else:
        text = "from vllm.oellm_continuation import OELLM_CONTINUATION_API, cache_limit\n"+text
    (out/"kv_cache_manager.py").write_text(text)
    text = sources["vllm_causallms.py"].read_text()
    text = once(text, "eval_logger = logging.getLogger(__name__)", "from lm_eval.models.vllm_continuation import available, partition, gather, KEY\n\neval_logger = logging.getLogger(__name__)")
    text = once(text, "        single_token_loglikelihood: bool = True,", "        continuation_loglikelihood: bool = True,\n        single_token_loglikelihood: bool = True,")
    text = once(text, "        self.data_parallel_backend = data_parallel_backend", "        self.data_parallel_backend = data_parallel_backend\n        self.continuation_loglikelihood = available(continuation_loglikelihood)")
    text = once(text, "            request_shards = [list(x) for x in distribute(dp_size, requests)]\n            parameter_shards = [list(x) for x in distribute(dp_size, sampling_params)]", "            shards = partition(requests, sampling_params, dp_size, int(self.model_args.get(\"max_num_seqs\") or 32))\n            dp_size = len(shards)\n            request_shards = [[requests[i] for i in shard] for shard in shards]\n            parameter_shards = [[sampling_params[i] for i in shard] for shard in shards]")
    text = once(text, "            return undistribute(run_workers(_vllm_mp_worker, work))", "            return gather(shards, run_workers(_vllm_mp_worker, work), len(requests))")
    text = once(text, '            eval_logger.info("Likelihood scoring:', '''            if self.continuation_loglikelihood:
                for plan, param in zip(plans, params, strict=True):
                    if plan.selected is None and plan.members[0][1] > 0:
                        param.extra_args = {KEY: plan.members[0][1]}
                        param.skip_reading_prefix_cache = False
                eval_logger.info("Continuation-only scoring with boundary-limited prefix reuse: %d requests",
                                 sum(p.extra_args is not None for p in params))
            eval_logger.info("Likelihood scoring:''')
    text = once(text, "        continuation_logprobs_dicts = outputs.prompt_logprobs", "        continuation_logprobs_dicts = outputs.prompt_logprobs\n        if continuation_logprobs_dicts is None or len(continuation_logprobs_dicts) != len(tokens):\n            raise ValueError(\"Incomplete or misaligned prompt logprobs\")\n        if any(p is None for p in continuation_logprobs_dicts[ctxlen:]):\n            raise ValueError(\"Required continuation logprobs are missing\")")
    text = once(text, "            answers = restore_answers(plans, outputs, inputs, self._parse_logprobs)", "            if self.continuation_loglikelihood:\n                eval_logger.info(\"Prompt cache reuse: %d / %d input tokens\",\n                                 sum(o.num_cached_tokens or 0 for o in outputs),\n                                 sum(len(o.prompt_token_ids) for o in outputs))\n            answers = restore_answers(plans, outputs, inputs, self._parse_logprobs)")
    (out/"vllm_causallms.py").write_text(text)
    here = Path(__file__).parent
    (out/"oellm_continuation.py").write_bytes((here/"vllm_continuation_engine.py").read_bytes())
    (out/"vllm_continuation.py").write_bytes((here/"vllm_continuation_adapter.py").read_bytes())
    (out/"vllm_single_token.py").write_bytes((harness_source/"lm_eval/models/vllm_single_token.py").read_bytes())
    manifest = {"base_hashes": BASE_HASHES, "files": {str(out/n):sha(out/n) for n in TARGETS}, "bind_targets": TARGETS}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    if profile_base:
        profile = json.loads(profile_base.read_text())
        profile["tasks"] = ["hellaswag"]
        profile["extra_source_files"] = [str(out/n) for n in [*TARGETS, "manifest.json"]]
        profile["singularity_args"] += " --env VLLM_USE_V2_MODEL_RUNNER=1 " + " ".join("--bind "+shlex.quote(f"{out/n}:{target}:ro") for n,target in TARGETS.items())
        with profile_out.open("x") as stream:
            stream.write(json.dumps(profile,indent=2)+"\n")
    print(out/"manifest.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("engine-source", "harness-source", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--profile-base", type=Path)
    parser.add_argument("--profile-out", type=Path)
    args = parser.parse_args()
    if bool(args.profile_base) != bool(args.profile_out):
        parser.error("Both profile options are required together")
    args.out = args.out.resolve()
    prepare(**vars(args))
