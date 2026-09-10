# Shared-context vLLM likelihood scoring

The reconstructed lm-evaluation-harness now shares identical contexts across single-token likelihood alternatives. This addresses the repeated-prefill and unused prompt-logprob work of tasks such as MMLU. The implementation belongs to `lm_eval/models/vllm_causallms.py` and `vllm_single_token.py`; oellm-eval packages it through `patches/harness-single-token-loglikelihood.patch`. vLLM already implements the selected-token logprob API.

`containers/prepare_vllm_sources.py` applies the existing native-DP patch followed by this patch and verifies final tree `c4a3f4235cb852bb7fe4ce4570778a13a1328612` against pinned upstream `6d642546f4688648fced259eb3302efd36ece5af`. `build_vllm_image.py` records the explicit upstream-plus-tree identity. Rebuild the runtime and image from this reconstruction for a self-contained installation. Existing images and older pinned guide revisions do not acquire the change automatically.

For an existing compatible vLLM 0.28 image, an explicit read-only overlay of these two reconstructed files onto the installed `lm_eval/models/` directory is also supported and was used for GPU validation. Resolve the actual installed path first; the validated JUPITER image uses `/opt/oellm-eval/lib/python3.12/site-packages/lm_eval/models/`. Record both file hashes and the image identity in the launch manifest. No task definitions or scoring conventions change.

The adapter enables `single_token_loglikelihood=True` by default when `SamplingParams` supports `logprob_token_ids`; set `single_token_loglikelihood=False` in model arguments for the previous route. Identical already-truncated contexts share an unrestricted next-token distribution, gathering the original answer token IDs. It does not constrain decoding or renormalize among the answers. Mixed/multi-token continuations retain prompt-logprob scoring, result ordering and per-request caching. Groups with more than 128 distinct requested IDs are split to respect the pinned API limit.

## Production64k validation on JUPITER

| Full MMLU 5-shot, 14,042 questions / 57 subjects | Accuracy | Allocation elapsed |
|---|---:|---:|
| Historical HF | 62.761715% | 31m 17s |
| Previous vLLM DP4 | 62.669135% | 28m 35s |
| Shared-context vLLM DP4 | 62.662014% | 6m 47s |

Job `1750789_0` completed with exit `0:0`, using production64k BF16, TP1/DP4, `max_num_seqs=32` and the existing standalone ARM image with SHA-256 `341cc783fea78418f0cf58f15de463ed34902b2ffda591ee7dd816b288e59e60`. Requests fell from 56,168 to 14,015; all four GPUs were used. Speedup is 4.21× relative to previous vLLM and 4.61× relative to HF, including initialization and result writing. All documents, prompts, targets, choices, filters and hashes match the previous vLLM execution. Independent raw scoring yields 8,799 versus 8,800 correct; 102 predictions change, including 34 improvements and 35 regressions. This is matching aggregate performance, not bitwise equivalence across BF16 execution layouts.

An additional four-GPU API diagnostic, job `1750955`, tested eight contexts including the four largest full-run probability discrepancies. With prefix caching disabled, selected-token logprobs, complete 262,144-token-vocabulary logprobs and legacy prompt-logprob scoring match exactly on all 32 option values. The full distributions sum to one within `6.54e-7`. This directly verifies probability gathering without answer-only renormalization; it does not prove every full-run numerical difference has the same cause.

Seventeen focused unit/image/bootstrap tests pass, plus actual-adapter ordering/truncation/cache/mixed-request checks and fresh source reconstruction. Run the bounded feature tests with:

```bash
python -m pytest -o addopts='' -q tests/test_vllm_single_token.py tests/test_vllm_image.py tests/test_jupiter_bootstrap.py
```

Activate the machine's documented control environment first. No additional packages are required by the new planner. The workspace companion report is `oellm_32B_loss-increase_debug/reports/mmlu_vllm_optimization_2026-09-10.md` in `SLAMPAI/oellm-workflows`; it includes full provenance, reproduction helpers and raw comparison locations. Production training remains stopped and unattended downstream evaluation remains paused.
