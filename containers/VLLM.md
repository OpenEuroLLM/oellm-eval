# Portable vLLM evaluation images

The evaluation runtime is separate from the agent/control environment. Each image combines a SHA-256-pinned machine engine image, official lm-eval-harness v0.4.12, and the pinned Evalchemy source with the reviewed compatibility and answer-parser patches. JUPITER uses ARM64 with Ray 2.48.0; JUWELS Booster uses x86_64 with Ray 2.56.1. Machine identities are in `vllm_profiles.json`. Existing cluster configuration still selects the scheduler account, partition and container flags independently.

Use an activated control environment for the Python builders below. Runtime package installation happens only through `apptainer exec` into a newly created evaluation venv. Never point preparation at an agent environment or an existing production runtime. These commands do not compile CUDA; native task dependencies must have wheels. Keep longer preparation/build commands in your named tmux session.

## Prepare sources and runtime

Choose a scratch directory visible inside Apptainer, a local `/tmp` build directory, and the matching existing vLLM+Ray base SIF. The `base_filename` in the profile identifies the available artifact, while the SHA-256 and byte count define its identity. Collaborators can place the same artifact elsewhere. The base images originate in the model-deployment workspace; these tools do not download multi-gigabyte engine images automatically.

```bash
repo=/path/to/oellm-evals
work=/path/to/scratch/evaluation-build
image=/path/to/verified/vllm-ray.sif
machine=jupiter  # or juwels_booster on an x86_64 host

# Plan first, then reconstruct exact public commits and apply reviewed patches.
python "$repo/containers/prepare_vllm_sources.py" --output "$work/sources"
python "$repo/containers/prepare_vllm_sources.py" --output "$work/sources" --execute

# Check the base image against the selected profile before executing it.
sha256sum "$image"
stat -c '%s' "$image"

apptainer exec --bind "$work:$work,$repo:$repo" "$image" \
  bash "$repo/containers/prepare_vllm_runtime.sh" --execute \
  "$work/runtime" "$work/sources/harness" \
  "$work/sources/evalchemy" "$work/sources/human-eval"
```

Preparation requires outbound access for task dependency wheels and NLTK data. It checks pip ownership before installation, constrains the base engine packages, records the installation report and full package freeze, and runs imports, context normalization, parser tests and both CLI help commands. The sole tolerated `pip check` discrepancy is the exact pre-existing Torch 2.13.0/NCCL 2.30.7 metadata mismatch in the pinned engine images. Other dependency conflicts fail preparation. Task dependency resolution is recorded rather than claimed to be byte-for-byte reproducible across future package indexes.

The source command verifies the patched Evalchemy Git tree against the tested tree and records every tracked file's checksum. Its `revision` field identifies the tested patched snapshot; `upstream_revision` and patch checksums describe how collaborators reconstruct it without fetching a private or local branch. HumanEval imports unchanged official source because its published source distribution and console entry point have packaging defects.

## Build and verify

```bash
python "$repo/containers/build_vllm_image.py" \
  --machine "$machine" --base-image "$image" \
  --runtime "$work/runtime" --evalchemy "$work/sources/evalchemy" \
  --humaneval "$work/sources/human-eval" \
  --source-manifest "$work/sources/source-manifest.json" \
  --output "$work/oellm-eval-vllm.sif" \
  --tmp-dir "/tmp/$USER-oellm-eval-build"
# Review the plan, then repeat with --execute.
```

The builder verifies host architecture, base size/SHA, source file checksums and successful runtime preparation. It refuses an existing output image. A symlink-preserving archive carries the venv into the base image: copying it with Apptainer `%files` directly would dereference the image-only Python interpreter on the host. Only text launchers and the HumanEval source path are relocated. Runtime paths become `/opt/oellm-eval`, `/opt/evalchemy`, `/opt/human-eval` and `/opt/nltk_data`; the host scratch runtime is no longer needed for execution. Local `/tmp` also avoids Ray's Unix-socket path-length failure on long GPFS build paths. Compression uses two worker threads.

The resulting `.provenance.json` records the image SHA-256, size, definition and source manifest identity. The image contains dependency reports and source checksums under `/opt/oellm-provenance`. Import/parser tests run during the build, but the label remains `gpu-validation-required`: promotion requires complete GPU tasks through both frameworks, DP1/DP4 comparison, model/tokenizer checks, full example counts and recorded GPU allocation/results. A successful build alone does not establish GPU compatibility.

## Schedule with the image

```bash
export EVAL_CONTAINER_IMAGE=/path/to/oellm-eval-vllm.sif
export EVALCHEMY_DIR=/opt/evalchemy
oellm-eval schedule --models /path/to/model \
  --task_groups reasoning --model_backend vllm \
  --data_parallel_size 4 --tensor_parallel_size 1 \
  --model_args dtype=bfloat16,gpu_memory_utilization=0.9 \
  --log_samples true --dry_run true
```

Run scheduling from a separately installed control CLI. Omit `--venv_path` to execute evaluation entirely inside the image. Inspect the rendered batch script before submission. Pre-stage all required datasets/tokenizers on systems with offline compute nodes. Use the versioned task settings without local example caps. See [the backend guide](../docs/VLLM.md) for DP/TP allocation and parser-reference comparison requirements.

### JUPITER Ray startup workaround under investigation

The tested ARM64 image uses Ray 2.48.0. Two initial four-GPU jobs lost Ray workers before vLLM initialized; repeating full PIQA with the following environment completed all 1,838 examples at the same score as the JUWELS reference pair:

```bash
# Set before sbatch; the normal container launcher inherits this environment.
export RAY_OVERRIDE_RESOURCES='{"CPU": 4}'
```

This sets Ray's logical CPU scheduling capacity. It does not set process CPU affinity or restrict evaluation to four physical cores. The native harness adapter schedules one Ray task per GPU replica, each requesting one logical CPU; use at least as many logical CPUs as DP replicas. Ray 2.48.0 also uses this count for prestarted Python workers and maximum worker startup concurrency, so it avoids a large startup pool on JUPITER's 288-core allocations. Excessive worker/thread startup is a hypothesis for the observed failure, not an established root cause. An uncapped repeat with persistent worker logs remains pending; this workaround is explicit rather than a global scheduler default. GPU compatibility for other tasks must still be checked.

If using Apptainer `--cleanenv`, pass the variable explicitly with `--env` as well. Preserve Ray logs through a bind to a short container path such as `/tmp/ray_eval` and set `RAY_TMPDIR` there; long GPFS paths can exceed Ray's Unix-socket limit.
