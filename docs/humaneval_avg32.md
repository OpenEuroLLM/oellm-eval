# HumanEval sampled pass@1

`python -m oellm.humaneval_sampling` provides a separate, versioned HumanEval generation and CPU grading workflow. Existing greedy Evalchemy results are unchanged. The module supports one greedy answer or 32 sampled answers per problem. It requires the native multiprocessing adapter `lm_eval.models.vllm_dp` from the complete oellm-eval runtime.

For avg@32, the score is the mean of the 32 correctness indicators for each problem, averaged over every problem. This estimates sampled pass@1, not pass@32 or the probability that any of 32 answers succeeds. Missing or duplicated answers fail validation; they are never silently counted as incorrect or omitted from the denominator.

## Configuration and execution

Use a JSON case with schema 1, `samples: 32`, `temperature: 0.6`, `top_p: 0.95`, `seed: 0`, the model directory and revision, output directory, full problem files, `dp`, `tp`, context and output limits, parser and prompt policy. `input_sha256` maps immutable input paths to their SHA-256 hashes. The output directory must not exist before generation. Run the three stages in order:

```bash
python -m oellm.humaneval_sampling preflight --case case.json
# In a real GPU allocation, with its CUDA_VISIBLE_DEVICES preserved:
python -m oellm.humaneval_sampling generate --case case.json
# In a separate, contained CPU process:
CUDA_VISIBLE_DEVICES= python -m oellm.humaneval_sampling grade --case case.json
```

The preflight loads the local model configuration/tokenizer and checks every full prompt plus the requested output budget. Generation uses independent native vLLM MP replicas, or native MP tensor parallelism when the model spans GPUs. It saves all answers, token IDs, finish/stop reasons, per-worker resolved sampling parameters and package versions. vLLM receives `n=32`; no adapter selects only the first answer. The seed saved with each answer is its parent request seed; vLLM 0.28.0 derives each child seed as parent seed plus completion index. Problem seeds do not depend on DP sharding or request order.

The `evalchemy` prompt/parser pair preserves the installed DeepSeek-Coder-style instruction, code extraction and multilingual tests. Its standard comparison uses 1024 output tokens and reports Python's 164 problems and Bash's 158 separately. A sampled run changes decoding only. `completion` with the `nvidia` parser implements the published container task: original Python completion prompt, 512 output tokens, stop string `</s>`, original tests without Evalchemy's extra imports, and the pinned NVIDIA EvalPlus sanitizer. It is a Python HumanEval reference task, not the entire multilingual Evalchemy benchmark. HumanEval+ extended tests are a different benchmark and are not implied by EvalPlus sanitization.

For NVIDIA parsing, set `nvidia_sanitizer` to the verified `lm_eval/filters/sanitize.py` distributed in NVIDIA's 26.03 harness image and `test_protocol: "openai"`. The sanitizer's installed EvalPlus/tree-sitter dependencies must be available in the evaluation environment. For Evalchemy, set `evalchemy_benchmark` to the bundled HumanEval directory. Its underlying multilingual scorer supports repeated answers; its outer single-response grader is intentionally bypassed. Grading workers, the 3-second test timeout, exact problem coverage and native pass@1 are checked. Keep grading outside GPU allocations and inside the established isolated CPU runtime.

The oellm-workflows companion entry point `helper_scripts/humaneval_sampling_campaign.sh` pins/downloads Nemotron models, stages their required remote configuration sources, prepares immutable cases and Slurm scripts, runs CPU preflights, and controls the reference queue. GPU visibility must be passed through Apptainer's clean environment explicitly. A Slurm test-only submission cannot catch that container boundary error. The reference comparison and any subsequent B1 backfill remain distinct from production's existing greedy evaluation pipeline.

The standalone CPU grader also waits for an actual process exit code or monotonic join deadline. Under concurrent process cleanup, the original strict grader reported a13-second outer timeout after only0.15seconds on a correct trivial answer. Retrying a prematurely returned join until its existing deadline prevents that false infrastructure failure; it does not change the three-second code execution timeout or model verdicts. Saved generations can be regraded in a fresh runtime without GPU regeneration: dataset identity is checked by recorded hashes, allowing paths to move while retaining the original generation receipt unchanged.

Shell tests run as their own process group with the same three-second timeout. Cleanup kills the entire group, including descendants, on timeout and on completion. This fixes recursive Bash grandchildren surviving the old shell wrapper's termination and exhausting the contained grading process. Test code, prompts, parser and correctness tests are unchanged; diagnostics use a temporary file rather than unbounded RAM capture.
