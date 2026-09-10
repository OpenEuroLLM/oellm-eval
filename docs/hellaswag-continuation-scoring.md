# Continuation-only HellaSwag likelihood scoring

The optional `containers/continuation_scoring/` overlay reduces unused vocabulary projections and safely reuses context prefixes for multi-token likelihood tasks. It changes the lm-evaluation-harness vLLM adapter and three pinned vLLM 0.28 source files. It preserves complete continuation log-likelihoods, the full vocabulary denominator, greedy flags, question order, truncation and HellaSwag's original character-length normalization. It does not replace endings with answer letters, restrict the vocabulary, or cap the benchmark. Standard HellaSwag remains 10-shot with 10,042 questions and 40,168 endings.

The engine receives the first continuation-token index in `SamplingParams.extra_args["oellm_prompt_logprobs_start"]`. It projects only hidden states that predict required continuation tokens. Unscored context entries are explicitly `None`. Prefix-cache lookup is capped at `context_length - 1`, ensuring the first required prediction and every ending token are recomputed, including when a complete identical request has previously been cached. Chunked prefills, mixed standard requests and output positions are covered by unit and GPU tests. The original full-vocabulary log-softmax routines remain in use.

The adapter co-locates a context's endings on one multiprocessing replica and submits first endings ahead of alternatives in bounded waves. This allows completed context caches to be reused. Native MP DP4/TP1 with max_num_seqs32 is the validated configuration. Ray sharding, other engine releases, other architectures and other hardware require their own validation. This is an overlay for the pinned ARM runtime, not a claim that all vLLM releases implement this API.

## Reproduce the installation

Activate the machine's documented control environment. No new packages are required. Set `OELLM_REPO`, `OELLM_IMAGE` and a new absolute `OELLM_CONTINUATION_WORK` outside the repository. The tested image SHA-256 is `341cc783fea78418f0cf58f15de463ed34902b2ffda591ee7dd816b288e59e60`; vLLM0.28.0 is installed under `/usr/local/lib/python3.12/dist-packages/vllm`, and the evaluation harness under `/opt/oellm-eval/lib/python3.12/site-packages/lm_eval`.

```bash
mkdir -p "$OELLM_CONTINUATION_WORK"
python "$OELLM_REPO/containers/prepare_vllm_sources.py" \
  --output "$OELLM_CONTINUATION_WORK/sources" --execute
singularity exec --contain \
  --bind "$OELLM_REPO:$OELLM_REPO:ro" \
  --bind "$OELLM_CONTINUATION_WORK:$OELLM_CONTINUATION_WORK" \
  "$OELLM_IMAGE" /opt/oellm-eval/bin/python \
  "$OELLM_REPO/containers/continuation_scoring/read_engine_sources.py" \
  --out "$OELLM_CONTINUATION_WORK/engine"
python "$OELLM_REPO/containers/continuation_scoring/prepare_hellaswag_optimization.py" \
  --engine-source "$OELLM_CONTINUATION_WORK/engine" \
  --harness-source "$OELLM_CONTINUATION_WORK/sources/harness" \
  --out "$OELLM_CONTINUATION_WORK/overlay"
export OELLM_CONTINUATION_BINDS
OELLM_CONTINUATION_BINDS=$(python \
  "$OELLM_REPO/containers/continuation_scoring/bind_args.py" \
  --manifest "$OELLM_CONTINUATION_WORK/overlay/manifest.json")
```

The source reconstruction pins upstream harness `6d642546f4688648fced259eb3302efd36ece5af` plus the native-DP and single-token patches, final tree `c4a3f4235cb852bb7fe4ce4570778a13a1328612`. The continuation preparer checks the three engine sources and reconstructed adapter against exact SHA-256 values before writing a fresh directory. It records every output hash and mount destination. Never modify a submitted overlay. `bind_args.py` verifies all files again; include the resulting arguments in the scheduler's `SINGULARITY_ARGS` through `--slurm_template_var`, alongside the existing container/offline arguments. Record the manifest, image, model, task source and generated Slurm script with the launch. Existing images and default builds are unchanged and do not acquire this feature automatically.

The adapter enables `continuation_loglikelihood=True` when the complete overlay is installed and requires the V2 GPU model runner. A partial installation fails closed. To select the prior multi-token route, set `continuation_loglikelihood=False` in model arguments. The existing MMLU single-token optimization is independent and stays available. Do not resume unattended evaluation campaigns simply to install this overlay.

Run focused tests in the activated control environment:

```bash
python -m pytest -o addopts='' -q tests/test_vllm_continuation.py \
  tests/test_vllm_continuation_overlay.py tests/test_vllm_single_token.py \
  tests/test_vllm_image.py \
  tests/test_jupiter_bootstrap.py
```

## Validation

The four-GPU diagnostic `1751232` completed successfully in 205 allocation seconds. It covers 32 real endings from eight production64k HellaSwag questions plus four synthetic 128-token endings, with forced 256-token prefill chunks. All 816 target log-probabilities match the legacy path exactly when cache reads are disabled. Warmed cache and mixed-request tests preserve complete target coverage, exercise actual cache hits and obey the boundary cap; cached BF16 execution is numerically different, so this does not establish bitwise equivalence. Full-task validation and measured runtime are recorded below.

Full production64000 validation `1751409_0` completed `0:0` in 590 seconds. All 10,042 raw inputs and normalized predictions match the previous vLLM run. Normalized accuracy is exactly 7781/10042 (77.484565%) for both; raw accuracy is 58.344951% for both. Historical HF is 77.375025% normalized and 58.215495% raw. The optimized run takes 9m50s versus 24m15s for previous vLLM (2.47× speedup) and 90m50s for HF (9.24×). These are allocation elapsed times including initialization, tokenization and result writing. The longest worker's prompt processing fell from 17m41s to 4m52s. Actual context-cache reuse is 24,885,216/34,708,179 input tokens (71.70%). No task cap or changed protocol is used.

Thirty focused tests pass, including `tests/test_vllm_continuation_overlay.py` for post-review source tampering and unsafe scheduler paths. The workspace report `oellm_32B_loss-increase_debug/reports/hellaswag_vllm_optimization_2026-09-11.md` in `SLAMPAI/oellm-workflows` provides exact source/image hashes, the standalone full comparison command, per-question CSV and raw-data locations. Acceptance is for this pinned installation and full HellaSwag reference; it does not establish bitwise equality under arbitrary cached BF16 batching.
