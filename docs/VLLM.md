# vLLM evaluation on one node

`schedule --model_backend vllm` enables the harness's native vLLM adapter for both lm-eval-harness and Evalchemy. One Python process dispatches requests to the configured replicas. Each task retains its registry-defined dataset, few-shot count, prompts and scoring settings.

```bash
oellm-eval schedule \
  --models /path/to/model \
  --task_groups reasoning \
  --model_backend vllm \
  --data_parallel_size 4 \
  --tensor_parallel_size 1 \
  --model_args dtype=bfloat16,gpu_memory_utilization=0.9 \
  --venv_path /path/to/compatible-eval-environment \
  --log_samples true \
  --dry_run true
```

Inspect the generated `submit_evals.sbatch` before submitting it. The GPU request is `data_parallel_size * tensor_parallel_size`, and `NODES=1`. A conflicting explicit `GPUS_PER_NODE` override is rejected. DP4/TP1 requires that the model and inference state fit on each GPU. DP2/TP2 instead shards each of two replicas over two GPUs. Omit `--venv_path` to use the configured Singularity image; Set `EVAL_CONTAINER_IMAGE` to an explicit absolute path to reuse a verified local image without downloading. Set `EVALCHEMY_DIR` to a host checkout (bound into the container) or to an image-internal directory such as `/opt/evalchemy`.

Do not launch this command with Accelerate or torchrun, or keep an external `LMEVAL_DP` launcher enabled. vLLM manages its own worker processes. The generated job respects an existing `CUDA_VISIBLE_DEVICES` assignment. Backend failures return a nonzero job exit code. Lighteval retains its existing execution route and cannot be mixed into a vLLM schedule.

`--model_args` accepts additional backend settings. Pass the model and DP/TP degrees through their dedicated options; their corresponding model-argument keys are reserved. The existing `MODEL_ARGS` environment setting continues to belong to Lighteval. The default backend remains `hf` and retains its existing launch behavior. `--log_samples true` records harness examples; Evalchemy chat results already include their examples and do not receive that flag.

## Runtime compatibility

For collaborator-ready JUPITER and JUWELS images, follow [portable image preparation](../containers/VLLM.md). It reconstructs the exact patched public sources, creates an isolated evaluation runtime inside a pinned engine image, and packages it under `/opt` with provenance. Agent/control environments are never used to run inference.

The environment must contain a mutually compatible vLLM and lm-eval adapter. The older EtashGuha harness pinned by `requirements-venv-evalchemy.txt` predates the current vLLM engine API; installing vLLM into that environment alone is insufficient. The Evalchemy pin also accesses a parent-process engine to find the context limit, which is absent with DP. Use `model.max_length` in its normalization code. Preserve the existing per-example context cap and safety buffer.

The JUPITER validation uses the operator-provided `evalchemy_arm.sif`, its native adapter, the pinned Evalchemy benchmark code and a compatibility patch under `patches/`. That patch preserves the pinned CLI parser with newer harness layouts, imports lazily registered model modules explicitly, replaces removed logger access and reads the public context-limit property. It does not change benchmark definitions or generation/scoring settings. This is a runtime compatibility path; it does not update the default dependency pins automatically.

In an isolated Evalchemy checkout at `54ac97648230c4c3a22c3a2b93068b5a4e573f8d`, apply `git apply /path/to/oellm-evals/patches/evalchemy-modern-vllm.patch`. Run `python tests/test_vllm_dp_context.py` there. The patch has been checked against that revision and reconstructs the tested files exactly. Keep the default production checkout separate until the chosen runtime passes GPU validation.

## Verification

`tests/test_vllm_schedule.py` renders and executes jobs against recording launchers for both suites, both environment modes and DP/TP combinations. It checks model arguments, allocation, task settings, scheduler device visibility, nested-launcher refusal and failure propagation. GPU equivalence and performance require full-task runs in the selected runtime; scheduler tests alone do not establish them.

## Answer extraction

`patches/evalchemy-math-answer-parser.patch` separately transfers the improved MATH500 answer recognition from Ali-Elganzory/evalchemy commit `212fb1ecc8d8bb13f7f8fe9d897c5cfdcac07648` (TieuDaoChanNhan), fixes decimal truncation and first-match selection, and preserves canonical boxed answers. Apply it after the compatibility patch and run `python tests/test_math_answer_parser.py`. Generation and the mathematical equivalence check are unchanged. Record the parser revision and re-score retained reference outputs with that same revision for comparisons; a scoring change alone can change measured accuracy. GPQA-Diamond randomization is already included in the pinned Evalchemy source and must not be applied twice.

Single-node vLLM DP defaults to `--data_parallel_backend mp` and requires the patched harness in the reproducible runtime described in [containers/VLLM.md](../containers/VLLM.md). Select `--data_parallel_backend ray` to use the preserved upstream Ray implementation. Containers include Ray for this comparison and future work. Both modes use one Slurm launcher; JUPITER requests 288 CPUs and JUWELS Booster 48 physical cores for the procedure, independently of replica count. Set `CPUS_PER_TASK` in `--slurm_template_var` to override. Multi-node evaluation remains out of scope for this implementation.

JUWELS Booster has 48 physical cores and 96 hardware threads. The default requests all 48 physical cores with `CPUS_PER_TASK=48` and `THREADS_PER_CORE=1`. To use SMT explicitly, override both to 96 and 2 respectively; JSC requires the threads-per-core flag for that request. JUPITER uses 288 CPUs and one thread per core. Both variables can be overridden through `--slurm_template_var`, independently of DP/TP and Ray. [JSC batch-system documentation](https://apps.fz-juelich.de/jsc/hps/juwels/batchsystem.html).
