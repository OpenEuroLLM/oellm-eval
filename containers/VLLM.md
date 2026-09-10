# Portable vLLM evaluation images

The evaluation runtime is separate from the agent/control environment. Each image combines a SHA-256-pinned machine engine image, official lm-eval-harness v0.4.12 plus the reviewed native DP patch, and the pinned Evalchemy source with the reviewed compatibility and answer-parser patches. JUPITER uses ARM64 with Ray 2.48.0; JUWELS Booster uses x86_64 with Ray 2.56.1. Machine identities are in `vllm_profiles.json`. Existing cluster configuration still selects the scheduler account, partition and container flags independently.

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

### Multiprocessing default and optional Ray

The bundled harness patch adds `data_parallel_backend=mp` (the default) using independent vLLM engines in spawned processes. vLLM 0.28 rejects its coordinated offline DP interface for dense models, so each replica receives a disjoint group of visible GPU identifiers and uses a regular engine with DP1 and the requested TP degree. Inherited coordinated DP ranks are cleared. Results are restored to request order after collecting all replicas. The original harness Ray implementation remains available with `--data_parallel_backend ray`. Ray stays installed in both machine images; the multiprocessing evaluation path does not initialize it. These modes currently support one node only. Tensor parallelism is independently configured with `--tensor_parallel_size`.

Slurm requests one launcher with the machine's configured `CPUS_PER_TASK`: 288 on JUPITER and 96 logical CPUs on JUWELS Booster. Override this through `--slurm_template_var` if needed. This allocation is independent of the DP replica count and Ray logical resources.

Two earlier JUPITER Ray jobs failed before engine startup. Both a capped repeat and an uncapped repeat subsequently passed full PIQA; the uncapped control advertised all 288 CPUs to Ray. A CPU cap is therefore not recommended as a demonstrated fix. For Ray diagnostics, preserve logs using a bind to a short path such as `/tmp/ray_eval`; long GPFS paths can exceed Unix-socket limits.
