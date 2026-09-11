# JUPITER: install and run oellm-eval from scratch

This guide starts with a JUPITER account and public internet access on a login node. It downloads every software, image, model and dataset input into your own directories. It does not require another collaborator's container, Python environment, model cache, dataset cache, or model-deployment checkout. The repository/package is named **oellm-eval**; older links and discussion also use **oellm-evals**.

The result is a standalone ARM64 Apptainer image with vLLM 0.28.0, the patched lm-eval-harness 0.4.12 adapter, Evalchemy, and Ray 2.48.0 for optional Ray execution. The complete build includes the validated single-token and continuation-only likelihood optimizations, greedy-decoding correction, HumanEval/LiveCodeBench grading repairs, the HF EOS fallback, and task-budget precedence over implicit HF model defaults. No source overlays are needed at evaluation time. HF remains available. Single-node data parallelism defaults to independent vLLM engines managed by Python multiprocessing. The CLI used to schedule work is installed separately on the login node. This guide covers single-node evaluation; other machines and reuse of shared installations will have separate guides.

## 1. Obtain access and check the machine

You need a JUPITER account, an active Booster compute budget, a writable project/scratch location, and Container Runtime Engine access. Request the latter in JuDoor: **Software → Request access to restricted software → Access to other restricted software → Container Runtime Engine**. Log in again after group membership propagates. See [JSC's container-runtime instructions](https://apps.fz-juelich.de/jsc/hps/jupiter/container-runtime.html) and [JUPITER access instructions](https://apps.fz-juelich.de/jsc/hps/jupiter/access.html).

On a JUPITER login node:

```bash
hostname
uname -m                 # must be aarch64
id -nG                  # must include container
command -v git curl tmux apptainer singularity sbatch squeue sacct
apptainer --version
jutil user projects     # identify your project and its Slurm budget account
```

Apptainer and its `singularity` compatibility command are site-provided and do not require a module. The tested version is 1.4.5. Downloads, package preparation, and these image builds run on the login node; model execution runs only in a Slurm GPU allocation. No CUDA compiler is invoked on the login node. You do not need Docker or sudo.

Allow approximately 60 GB of project/scratch space and 60 GB of free local `/tmp` for this reference installation; retain more for additional models and benchmarks. Check both locations with `df -h`. Keep Apptainer and pip caches off your home filesystem. Use local `/tmp` for image extraction/builds: shared-filesystem extraction is slow, and long temporary paths can break Ray's Unix sockets.

JSC's container page contains older text saying image builds are unavailable alongside newer native-build guidance. Native unprivileged Apptainer builds worked for the tested account. If your account cannot use the site's native fakeroot/build mechanism, resolve that with JSC before proceeding; this guide does not require or suggest acquiring root privileges or using the deprecated Container Build System.

## 2. Start a persistent shell and choose your directories

Start a named session, then run the remaining commands **inside it**:

```bash
tmux new -s oellm-eval-jupiter
```

Replace the two example values below with your own writable scratch root and Slurm budget account. Project directory names and Slurm account names can differ. Paths in this guide must be absolute and contain no spaces or commas.

```bash
set -euo pipefail
umask 077
export OELLM_WORK=/e/fscratch/YOUR_PROJECT/$USER/oellm-eval-install
export OELLM_ACCOUNT=YOUR_SLURM_ACCOUNT
export OELLM_REPO="$OELLM_WORK/src/oellm-eval"
export OELLM_ASSETS="$OELLM_WORK/assets"
export OELLM_MODEL="$OELLM_ASSETS/models/Qwen3-1.7B-Base"
export HF_HOME="$OELLM_ASSETS/hf-cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export MAMBA_ROOT_PREFIX="$OELLM_WORK/mamba"
export PIP_CACHE_DIR="$OELLM_WORK/pip-cache"
export PIP_CONFIG_FILE=/dev/null
export PIP_INDEX_URL=https://pypi.org/simple
unset PIP_EXTRA_INDEX_URL
export APPTAINER_CACHEDIR="$OELLM_WORK/apptainer-cache"
mkdir -p "$OELLM_WORK"/{src,tools,logs,images} "$OELLM_ASSETS" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"
df -h "$OELLM_WORK" /tmp
```

Save these non-secret settings in your own setup file if you need to reopen the shell. Do not save Hugging Face tokens in that file. Detach with `Ctrl-b d`; reconnect with `tmux attach -t oellm-eval-jupiter`.

## 3. Fetch the exact CLI and builder sources

```bash
export OELLM_REVISION=b4996091f4a305ef03dffc4dc0f6cab17df0f38d
git clone https://github.com/OpenEuroLLM/oellm-eval.git "$OELLM_REPO"
git -C "$OELLM_REPO" checkout --detach "$OELLM_REVISION"
git -C "$OELLM_REPO" rev-parse HEAD | tee "$OELLM_WORK/logs/oellm-revision.txt"
```

The detached checkout makes the instructions reproducible even if the feature branch moves. To develop changes, create your own branch from this revision. Do not install the unmodified older Evalchemy requirements file alongside a modern vLLM wheel: that combination uses an incompatible harness adapter.

## 4. Bootstrap a dedicated control environment

This downloads the pinned standalone micromamba binary; it does not assume conda, mamba, Python or pip is installed for you. The explicit conda lock fixes Python, pip and their bootstrap dependencies to exact public package URLs and checksums.

```bash
curl --fail --location --retry 3 \
  https://github.com/mamba-org/micromamba-releases/releases/download/2.8.1-0/micromamba-linux-aarch64 \
  --output "$OELLM_WORK/tools/micromamba"
printf '%s  %s\n' \
  e5ba23b5945aa49dfd11022e592a510d2686a8feee810e00140b73c9fdf0ba2a \
  "$OELLM_WORK/tools/micromamba" | sha256sum --check
chmod 700 "$OELLM_WORK/tools/micromamba"

"$OELLM_WORK/tools/micromamba" --no-rc create --yes \
  --prefix "$OELLM_WORK/control" \
  --file "$OELLM_REPO/containers/locks/jupiter-control-linux-aarch64.explicit"
eval "$("$OELLM_WORK/tools/micromamba" shell hook --shell bash)"
micromamba activate "$OELLM_WORK/control"
test "$(command -v python)" = "$OELLM_WORK/control/bin/python"
python -m pip --version

python -m pip install \
  --constraint "$OELLM_REPO/containers/locks/jupiter-control-constraints.txt" \
  uv_build==0.7.22
python -m pip check
python -m pip install --no-build-isolation \
  --constraint "$OELLM_REPO/containers/locks/jupiter-control-constraints.txt" \
  "$OELLM_REPO"
python -m pip check
python -m pip freeze --all > "$OELLM_WORK/logs/control-pip-freeze.txt"
oellm-eval --help
```

Use this environment for builders and scheduling. All evaluation engines and benchmark execution will run inside the image. Do not install evaluation extras into a shared agent environment or use system Python/pip for these steps.

## 5. Build the vLLM + Ray base directly from public inputs

The builder downloads the official vLLM ARM64 OCI image by its immutable platform digest and the exact Ray wheel by URL and SHA-256. It does not need an input SIF. First inspect its plan, then execute it:

```bash
export OELLM_BASE_TMP=$(mktemp -d /tmp/oellm-base.XXXXXX)
python "$OELLM_REPO/containers/bootstrap_jupiter_base.py" \
  --work "$OELLM_WORK/base" --tmp-dir "$OELLM_BASE_TMP" \
  | tee "$OELLM_WORK/logs/base-plan.json"
python "$OELLM_REPO/containers/bootstrap_jupiter_base.py" \
  --work "$OELLM_WORK/base" --tmp-dir "$OELLM_BASE_TMP" --execute \
  2>&1 | tee "$OELLM_WORK/logs/base-build.log"
export OELLM_BASE_IMAGE="$OELLM_WORK/base/vllm-v0.28.0-ray2.48.0-arm64.sif"
cat "$OELLM_WORK/base/profiles.json"
```

The public inputs are:

| Input | Immutable identity |
|---|---|
| `docker.io/vllm/vllm-openai`, ARM64 | `sha256:2a7cde230b59f3ce6cab33dd245ba6bee41aa87b38c9fe84f966ff24016813ce` |
| vLLM source label | `2cf0a6915ce544dc493a0990f2ea38d81601128a` |
| Ray 2.48.0 CPython 3.12 aarch64 wheel | SHA-256 `f1cf33d260316f92f77558185f1c36fc35506d76ee7fdfed9f5b70f9c4bdba7f` |

The builder runs import/version checks, checks the required Ray CLI flags, and executes a small local Ray task using one logical CPU. That CPU count applies only to this build check. The evaluation Slurm job receives the configured node CPUs independently.

One exact upstream dependency metadata discrepancy is tolerated: Torch 2.13.0+cu130 declares NCCL 2.29.7, while this OCI image contains 2.30.7. Every other `pip check` conflict fails the build. Do not broadly ignore dependency-check failures or upgrade Torch/NCCL independently.

A successful build writes `base.def`, `provenance.json` and `profiles.json` beside the SIF. Independently converted SIF files need not be byte-identical because Apptainer adds build metadata; see [Apptainer's OCI conversion documentation](https://apptainer.org/docs/user/latest/docker_and_oci.html#public-containers). The immutable OCI digest and verified wheel identify the inputs. `profiles.json` pins the hash and size of **your** resulting SIF for the next step. Do not substitute a historical collaborator SIF hash or edit the repository's shared profile to make a new build pass.

The builder refuses an existing work directory. If a build fails, preserve its log and choose a new base work directory and temporary directory after resolving the cause; update the corresponding variables below. Registry rate limits require waiting or your own permitted registry authentication, not changing to an unpinned tag.

## 6. Reconstruct the exact evaluation sources

```bash
python "$OELLM_REPO/containers/prepare_vllm_sources.py" \
  --output "$OELLM_WORK/sources" \
  | tee "$OELLM_WORK/logs/sources-plan.json"
python "$OELLM_REPO/containers/prepare_vllm_sources.py" \
  --output "$OELLM_WORK/sources" --execute \
  2>&1 | tee "$OELLM_WORK/logs/sources-prepare.log"
```

This fetches public upstream revisions, applies the bundled patches and checks the resulting trees. It preserves Evalchemy's corrected GPQA randomization and the reviewed MATH500 parser. All source file hashes are in `sources/source-manifest.json`. Expected patched trees are harness `9b7bbc80fbec2a891efd0fa496707cdab6c96c10` and Evalchemy `2d7ddf635cb11e0b0d4ec85e0d9fe56a58d2277b`. A failed tree check is an error, not permission to drop the check.

## 7. Prepare the pinned evaluation runtime inside the base image

```bash
mkdir "$OELLM_WORK/runtime"
apptainer exec --cleanenv --bind "$OELLM_WORK:$OELLM_WORK" \
  --env OELLM_RUNTIME_CONSTRAINTS="$OELLM_REPO/containers/locks/jupiter-runtime-constraints.txt" \
  "$OELLM_BASE_IMAGE" \
  bash "$OELLM_REPO/containers/prepare_vllm_runtime.sh" --execute \
  "$OELLM_WORK/runtime" "$OELLM_WORK/sources/harness" \
  "$OELLM_WORK/sources/evalchemy" "$OELLM_WORK/sources/human-eval" \
  2>&1 | tee "$OELLM_WORK/logs/runtime-prepare.log"
test -f "$OELLM_WORK/runtime/provenance/prepare-complete"
```

The script creates a new evaluation venv using the immutable base engine packages, applies the committed package constraints, and verifies that pip belongs to that venv before installation. Only explicitly allowed pure-Python packages may build from source. NLTK resources are downloaded from a pinned public source commit and verified by SHA-256. No pre-populated NLTK or pip cache is required. The script records resolved versions, installation metadata and dependency checks, then tests the context-limit and answer-parser patches and both framework CLIs.

HumanEval comes from pinned public source because its published package/entry point has packaging defects. Its source is packaged into the final image. Do not replace it with an unversioned `pip install human-eval` workaround.

## 8. Build the standalone evaluation image

```bash
# Assemble the validated updates from the pinned base and reconstructed sources.
# This step installs no packages and refuses source/hash mismatches.
python "$OELLM_REPO/containers/prepare_complete_runtime.py" \
  --base-image "$OELLM_BASE_IMAGE" --sources "$OELLM_WORK/sources" \
  --out "$OELLM_WORK/complete" | tee "$OELLM_WORK/logs/complete-plan.json"
python "$OELLM_REPO/containers/prepare_complete_runtime.py" \
  --base-image "$OELLM_BASE_IMAGE" --sources "$OELLM_WORK/sources" \
  --out "$OELLM_WORK/complete" --execute \
  2>&1 | tee "$OELLM_WORK/logs/complete-prepare.log"
export OELLM_IMAGE="$OELLM_WORK/images/oellm-eval-complete-jupiter.sif"
export OELLM_IMAGE_TMP=$(mktemp -d /tmp/oellm-image.XXXXXX)
image_build_args=(
  --machine jupiter --profiles "$OELLM_WORK/base/profiles.json"
  --base-image "$OELLM_BASE_IMAGE" --runtime "$OELLM_WORK/runtime"
  --evalchemy "$OELLM_WORK/sources/evalchemy"
  --humaneval "$OELLM_WORK/sources/human-eval"
  --source-manifest "$OELLM_WORK/sources/source-manifest.json"
  --complete-runtime "$OELLM_WORK/complete"
  --output "$OELLM_IMAGE" --tmp-dir "$OELLM_IMAGE_TMP"
)
python "$OELLM_REPO/containers/build_vllm_image.py" "${image_build_args[@]}" \
  | tee "$OELLM_WORK/logs/image-plan.json"
python "$OELLM_REPO/containers/build_vllm_image.py" "${image_build_args[@]}" --execute \
  2>&1 | tee "$OELLM_WORK/logs/image-build.log"
sha256sum "$OELLM_IMAGE" | tee "$OELLM_WORK/logs/image.sha256"
```

The final image contains `/opt/oellm-eval`, `/opt/evalchemy`, `/opt/human-eval`, `/opt/nltk_data` and `/opt/oellm-provenance`. The build verifies the base hash, source manifest and installed adapter contents. It relocates the runtime into the image, checks dependency consistency and both CLIs, and writes a `.def` and `.provenance.json` beside the output. The complete update manifest is installed at `/opt/oellm-provenance/complete-runtime.json`. The build verifies each installed update, runs actual-library CPU checks of implicit greedy decoding and HF EOS/continuation generation, and enables the pinned vLLM V2 runner required by continuation scoring. The final complete manifest contains 19 installed updates, including the HF generation adapter. A real tiny-model regression verifies that a saved `generation_config.max_new_tokens` cannot silently shorten a benchmark’s requested budget; an explicit per-request override remains respected. The original source manifest describes the reconstructed baseline; the complete manifest identifies the final installed replacements. Building/importing successfully is not yet GPU validation.

Verify the final SIF without mounting your source checkout, preparation directory, control environment or runtime venv:

```bash
apptainer exec --cleanenv --containall --no-mount bind-paths,hostfs,cwd,home \
  --pwd /opt/evalchemy "$OELLM_IMAGE" \
  /opt/oellm-eval/bin/python /opt/oellm-provenance/verify_complete_image.py
apptainer exec --cleanenv --containall --no-mount bind-paths,hostfs,cwd,home \
  --pwd /opt/evalchemy "$OELLM_IMAGE" \
  /opt/oellm-eval/bin/python /opt/oellm-provenance/check_complete_runtime_cpu.py
```

## 9. Download a reference model and complete public benchmarks

The initial validation uses the public Qwen3-1.7B-Base checkpoint and complete PIQA, GSM8K and MATH500 tasks. It is a set of individual benchmarks, not the full `reasoning` or `open-sci` group. The exact model/dataset revisions and expected counts are embedded in `prepare_jupiter_reference.py`. MATH500's full 500-example data file is already obtained with the pinned Evalchemy source and packaged into the image.

Run the download **inside the final image on the login node**, with only the asset directory and preparation script mounted. These commands do not expose the host control environment, staged source trees or runtime venv to the image:

```bash
reference_args=(
  --model-dir "$OELLM_MODEL" --hf-home "$HF_HOME"
  --manifest "$OELLM_ASSETS/reference-public.json"
  --tasks piqa gsm8k MATH500
)
reference_container=(
  apptainer exec --cleanenv --containall
  --no-mount bind-paths,hostfs,cwd,home
  --bind "$OELLM_ASSETS:$OELLM_ASSETS"
  --bind "$OELLM_REPO/containers/prepare_jupiter_reference.py:/opt/prepare-reference.py:ro"
  --pwd /opt/evalchemy "$OELLM_IMAGE"
  /opt/oellm-eval/bin/python /opt/prepare-reference.py
)
"${reference_container[@]}" "${reference_args[@]}"                 # plan
"${reference_container[@]}" "${reference_args[@]}" --execute \
  2>&1 | tee "$OELLM_WORK/logs/reference-download.log"
"${reference_container[@]}" "${reference_args[@]}" --offline \
  2>&1 | tee "$OELLM_WORK/logs/reference-offline.log"
```

The offline check loads the model config/tokenizer, verifies model file hashes and HF/vLLM prompt equality, and exercises the same unversioned offline dataset lookup used by the benchmark loaders. It compares complete row hashes and counts against the pinned online preparation. PIQA must have 16,113 training examples for few-shot construction and 1,838 validation examples; GSM8K must have 7,473 training and 1,319 test examples; MATH500 must contain all 500 questions. Cache downloads alone do not establish that the offline loaders work.

### Add GPQA-Diamond with your own approved access

[Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa) is gated. Log into Hugging Face in your browser and accept/request the dataset's access terms yourself. Create a read token with access to that dataset. Neither this repository nor a collaborator's cache grants you that access. See [Hugging Face's gated-dataset documentation](https://huggingface.co/docs/hub/datasets-gated).

After access is granted, prepare a second manifest including GPQA. The helper prompts without echo and holds the token only in memory. Do not put the token in a command argument, shell setup file, log, image, or Slurm script. Do not pipe the interactive download through `tee`.

```bash
reference_args=(
  --model-dir "$OELLM_MODEL" --hf-home "$HF_HOME"
  --manifest "$OELLM_ASSETS/reference-with-gpqa.json"
  --tasks piqa gsm8k MATH500 GPQADiamond
)
"${reference_container[@]}" "${reference_args[@]}" --execute --ask-hf-token
"${reference_container[@]}" "${reference_args[@]}" --offline \
  2>&1 | tee "$OELLM_WORK/logs/reference-with-gpqa-offline.log"
```

GPQA-Diamond must load all 198 questions. Evaluation uses the task's three repeats and preserves its randomized answer ordering. The helper populates the exact cache directory used by the pinned Evalchemy GPQA loader (`HF_HUB_CACHE`); setting only a separate `HF_DATASETS_CACHE` is insufficient for this loader. GPU jobs need cached data, not a token.

## 10. Render the complete evaluation jobs

Return to the activated **control** environment. Set every machine-specific path explicitly so the repository's historical shared-site defaults are not used. Use a fresh output directory for every configuration.

```bash
micromamba activate "$OELLM_WORK/control"
export EVAL_BASE_DIR="$OELLM_WORK/runs"
export EVAL_OUTPUT_DIR="$EVAL_BASE_DIR/reference-mp-dp4"
export EVAL_CONTAINER_IMAGE="$OELLM_IMAGE"
export EVALCHEMY_DIR=/opt/evalchemy
export NLTK_DATA=/opt/nltk_data
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export QUEUE_LIMIT=1
mkdir -p "$EVAL_OUTPUT_DIR"
export OELLM_REFERENCE_TASKS='piqa gsm8k MATH500'
# Only after the GPQA offline check passed:
# export OELLM_REFERENCE_TASKS='piqa gsm8k MATH500 GPQADiamond'

python - <<'PY'
import csv, os
from pathlib import Path
from oellm.task_groups import _expand_task_groups
selected=set(os.environ['OELLM_REFERENCE_TASKS'].split())
tasks=[t for t in _expand_task_groups(['open-sci-0.01','reasoning']) if t.task in selected]
assert len(tasks)==len(selected)
with (Path(os.environ['EVAL_OUTPUT_DIR'])/'input.csv').open('w') as f:
    w=csv.writer(f,lineterminator='\n')
    w.writerow(['model_path','task_path','n_shot','eval_suite'])
    for t in tasks:
        w.writerow([os.environ['OELLM_MODEL'],t.task,t.n_shot,t.suite])
PY

slurm_options=$(python - <<'PY'
import json, os
print(json.dumps({
    'ACCOUNT':os.environ['OELLM_ACCOUNT'], 'PARTITION':'booster', 'NODES':1,
    'CPUS_PER_TASK':288, 'THREADS_PER_CORE':1, 'SLURM_MEM':'400G', 'TIME':'01:00:00',
    'SINGULARITY_ARGS':'--nv --cleanenv --containall --no-mount bind-paths,hostfs,cwd,home '
                       '--env HF_HUB_CACHE='+os.environ['HF_HUB_CACHE']+
                       ' --env HF_DATASETS_OFFLINE=1 --env RAY_USAGE_STATS_ENABLED=0'
}))
PY
)
oellm-eval schedule --eval_csv_path "$EVAL_OUTPUT_DIR/input.csv" \
  --model_backend vllm --data_parallel_backend mp \
  --data_parallel_size 4 --tensor_parallel_size 1 \
  --model_args dtype=bfloat16,gpu_memory_utilization=0.9,max_num_seqs=32 \
  --log_samples true --max_array_len 1 \
  --slurm_template_var "$slurm_options" --skip_checks true --dry_run true \
  2>&1 | tee "$EVAL_OUTPUT_DIR/render.log"
```

`--skip_checks true` is intentional here: the pinned model/image/cache checks were performed explicitly above using the evaluation runtime. It prevents the host CLI's generic downloader from fetching floating dataset versions or rebuilding caches with a different `datasets` version. It does not skip checks during benchmark execution. Do not use this shortcut before the explicit offline validation passes.

Inspect the generated script and CSV before submission. There must be one launcher, one node, four GPUs, 288 CPUs, and one thread per core. The model backend must be `vllm`, DP4/TP1, `data_parallel_backend=mp`, and container mode must use your SIF and `/opt/evalchemy`. PIQA is 10-shot, GSM8K 4-shot, MATH500/GPQA 0-shot, as obtained from the pinned registry. There must be no local example limit or replacement generation-token cap. `QUEUE_LIMIT=1` and `--max_array_len 1` keep this first validation to one Slurm array element, processing the selected full tasks sequentially.

```bash
mapfile -t scripts < <(find "$EVAL_OUTPUT_DIR" -name submit_evals.sbatch -type f)
test "${#scripts[@]}" -eq 1
export OELLM_BATCH="${scripts[0]}"
bash -n "$OELLM_BATCH"
cat "$EVAL_OUTPUT_DIR/input.csv"
sed -n '1,55p' "$OELLM_BATCH"
sha256sum "$OELLM_BATCH" "$EVAL_OUTPUT_DIR/input.csv" \
  > "$EVAL_OUTPUT_DIR/launch.sha256"
```

## 11. Submit, inspect and validate the GPU results

```bash
sha256sum --check "$EVAL_OUTPUT_DIR/launch.sha256"
sbatch --test-only "$OELLM_BATCH"
job_id=$(sbatch --parsable "$OELLM_BATCH")
printf '%s\n' "$job_id" | tee "$EVAL_OUTPUT_DIR/job-id.txt"
squeue --jobs "$job_id"
sacct -X --jobs "$job_id" --format=JobID,State,Elapsed,ExitCode,AllocTRES
```

Submit once. Re-running `sbatch` submits another job; retain the recorded ID and inspect it instead. A successful run must end `COMPLETED` with exit code `0:0`, produce one complete result per selected task, and show effective counts 1,838 / 1,319 / 500 / 198 as applicable. For GPQA, inspect all three repeats as well. Inspect `slurm_logs/` and `results/` under the generated timestamp directory. A result file by itself does not prove the allocation completed cleanly.

After Slurm reports successful completion, run the CPU-only result checker from your control environment:

```bash
python "$OELLM_REPO/containers/check_reference_results.py" \
  --run "$EVAL_OUTPUT_DIR" --tasks $OELLM_REFERENCE_TASKS \
  > "$EVAL_OUTPUT_DIR/result-check.json"
cat "$EVAL_OUTPUT_DIR/result-check.json"
```

It rejects missing results, local example caps, Evalchemy generation-token overrides, missing or duplicate example IDs, incomplete GSM8K scoring filters and missing GPQA repeats. Review the saved harness task/generation configuration alongside these coverage checks. Its output records the scores and resolved configurations. Scheduler completion and worker-partition traces are separate checks; this helper does not claim to verify either.

Harness JSON records `config.model_args`, `n-samples`, metrics and, with `--log_samples`, per-example JSONL. GSM8K can emit each example for multiple scoring filters; validate each filter's complete set rather than treating those records as duplicate questions. Evalchemy records full chat examples in the result JSON. Preserve the resolved task configuration and input manifests with the result; report the exact metric/filter used.

The log line **“Manual batching is not compatible with data parallelism”** is informational: this harness sets its batch size to `auto` when DP is enabled, and vLLM batches requests inside each replica. It does not disable DP. With DP4/TP1, four independent engines each own one GPU; engine-local logs can consequently report DP1 or local GPU0. The earlier DP validation used an additional request-tracing wrapper. These normal launcher commands do not enable that wrapper or produce its `.json.trace` files; a configured launch flag alone is not a partition audit.

In the reference PIQA DP4 log, four separate `Rendering prompts` counters each process 919 requests: 1,838 examples × two candidate continuations = 3,676 requests = four batches of 919. Four `EngineCore` process IDs also appear during initialization. These are useful visible checks of replica activity; exact request identities and ordered reassembly require the separate tracing audit.

For an explicit Ray comparison, render into a fresh output directory with `--data_parallel_backend ray`. For combined tensor/data parallelism, use `--data_parallel_size 2 --tensor_parallel_size 2`. Keep the model, tasks, prompt/generation/scoring settings and dataset pins identical. Slurm CPU allocation remains independent of Ray's logical resources. Do not wrap these jobs in Accelerate, torchrun or a second multi-process launcher.

The existing engine/adapter has passed these complete reference tasks on JUPITER; the separate installation rehearsal status below records what was re-tested from empty inputs. Scores can vary across generation batch layouts even with fixed inputs; do not promise bitwise identical generated responses. A parser change also requires re-scoring reference outputs before comparing accuracy.

## Check the available HF backend

Use a separate output directory for a full HumanEval HF/Accelerate check with the same public checkpoint and image. HumanEval’s standard 1,024-token generation budget makes it a bounded installation check. This exercises the four-GPU Evalchemy route; the vLLM DP flags are intentionally omitted. It retains the task’s standard token budget and all 164 Python and 158 shell problems.

```bash
export EVAL_OUTPUT_DIR="$EVAL_BASE_DIR/reference-hf-humaneval"
mkdir "$EVAL_OUTPUT_DIR"
python - <<'PY_HF'
import csv, os
from pathlib import Path
with (Path(os.environ['EVAL_OUTPUT_DIR'])/'input.csv').open('w') as f:
    w=csv.writer(f,lineterminator='\n')
    w.writerow(['model_path','task_path','n_shot','eval_suite'])
    w.writerow([os.environ['OELLM_MODEL'],'HumanEval',0,'evalchemy'])
PY_HF
hf_slurm_options=$(python -c 'import json,sys; d=json.loads(sys.argv[1]); d["GPUS_PER_NODE"]=4; print(json.dumps(d))' "$slurm_options")
oellm-eval schedule --eval_csv_path "$EVAL_OUTPUT_DIR/input.csv" \
  --model_backend hf --model_args dtype=bfloat16 --log_samples true \
  --slurm_template_var "$hf_slurm_options" --skip_checks true --dry_run true \
  2>&1 | tee "$EVAL_OUTPUT_DIR/render.log"
```

Repeat the script inspection, launch-hash recording, `sbatch --test-only`, single submission and scheduler-completion checks from steps 10–11 for this new output directory. Confirm four Accelerate processes in the runtime log. After completion, run the same result checker with `--tasks HumanEval`. Automatic HF batch-size probing can emit caught CUDA out-of-memory warnings while finding a fitting batch; inspect the eventual batch size and successful full-task completion. Such probing time belongs in end-to-end runtime comparisons. An uncaught OOM or incomplete result fails acceptance.

The public-model HF MATH500 task retains its standard 32,768-token budget and can take much longer than this check. Allocate time deliberately if selecting it; do not substitute the model’s saved 2,048-token default or a local sample cap to make an acceptance run cheaper. The HumanEval checker verifies both full language sets and reproduces the scores from retained Boolean verdicts.

## 12. Evaluate other models or benchmark groups

Use a model checkpoint whose exact revision and file hashes you record. Stage its tokenizer/config/weights on the login node and verify loading offline inside the evaluation image. DP4/TP1 requires the model and inference state to fit on each GPU; choose TP deliberately for larger models. Do not silently shorten context or generation to fit.

Select task groups from the pinned `oellm/resources/task-groups.yaml`. The public reference cache above does not include every registered benchmark. Before scheduling another group, prepare **all** of its required datasets, configurations, splits and auxiliary files using this evaluation image, pin their revisions, record complete counts and verify offline task construction. Some Evalchemy tasks obtain assets through their own loader rather than the CLI's generic dataset registry. Additional language/code/API tasks can require extra pinned dependencies, access grants or a separate code-execution service; those tasks are not validated merely because this image imports successfully. Keep such extensions in a newly versioned image/cache and test the whole selected protocol.

The complete image packages the HumanEval and LiveCodeBench corrections described in [code grading](../../code-grading-runtime.md). Their original full production64k acceptance covered 164 Python and 158 shell HumanEval problems, plus 511 LiveCodeBench questions with six repeats. The scheduler also binds private host temporary storage, avoiding the contained 16 MiB `/tmp` failure. Code execution is not enabled by a package import: for harness code tasks such as MBPP, pass `--confirm_run_unsafe_code true` and `--env HF_ALLOW_CODE_EVAL=1` explicitly in your scheduling/container settings. Preserve the standard benchmark data/repeats and retain all grading verdicts. The three public reference tasks alone are not a fresh code-task GPU acceptance.

Use complete groups when reporting their standard aggregate. Do not replace missing subjects/languages with a subset average or add a local `--limit`. For the initial guide tasks, report the individual scores rather than a `reasoning` or `open-sci` aggregate.

## Template-free base checkpoints

The public Qwen reference already has the model-specific prompt handling checked by the reference preparer. A custom base export may omit a chat template, while Evalchemy calls the chat-template interface even for plain-completion benchmarks. If the established reference protocol is plain completion, prepare an explicit identity-template view using the command below. This is the production64k protocol used in the validation; do not apply it to an instruct checkpoint or substitute it for a different benchmark prompt protocol.

```bash
# Set these two paths for your own template-free base checkpoint:
# export OELLM_BASE_EXPORT=/absolute/path/to/base-export
# export OELLM_PROMPT_VIEW=/absolute/path/on-the-same-filesystem/base-prompt-view
# python "$OELLM_REPO/containers/prepare_base_model_view.py" \
#   --source "$OELLM_BASE_EXPORT" --out "$OELLM_PROMPT_VIEW" --template identity
# Then use OELLM_PROMPT_VIEW as the scheduled model path.
```

The view copies metadata and hard-links immutable weights on the same filesystem. It contains no external symlinks, does not duplicate the weights and does not modify the export. The helper refuses an existing output or an existing instruct/chat template. It records the explicit template and weight identities in `prompt-view-manifest.json`. Mount the resulting directory in the container. Before GPU submission, exercise the actual HF and vLLM prompt rendering with the selected tokenizer and compare the rendered text with the reference protocol; dataset coverage checks alone do not catch a missing template.

For an explicitly selected identity view, this CPU check requires both real adapters to produce the original plain prompt:

```bash
apptainer exec --cleanenv --containall --no-mount bind-paths,hostfs,cwd,home \
  --bind "$OELLM_PROMPT_VIEW:/model:ro" --pwd /opt/evalchemy "$OELLM_IMAGE" \
  /opt/oellm-eval/bin/python - <<'PY_PROMPT'
from transformers import AutoTokenizer
from lm_eval.models.huggingface import HFLM
from lm_eval.models.vllm_causallms import VLLM
tokenizer=AutoTokenizer.from_pretrained('/model',local_files_only=True)
hf=HFLM.__new__(HFLM); hf.tokenizer=tokenizer; hf.chat_template_args={}
vllm=VLLM.__new__(VLLM); vllm.tokenizer=tokenizer
vllm.hf_chat_template=tokenizer.chat_template
vllm.enable_thinking=None; vllm.chat_template_args={}
messages=[{'role':'user','content':'Problem: 2 + 2\nAnswer:'}]
assert hf.apply_chat_template(messages)==vllm.apply_chat_template(messages)==messages[0]['content']
print('HF/vLLM identity prompt check passed')
PY_PROMPT
```

## Explicit IFEval stopping policy and the HF backend

`oellm-eval schedule --model_backend hf` remains supported; the default backend is still HF. For native vLLM, explicitly select `--model_backend vllm --data_parallel_size 4 --tensor_parallel_size 1`. The vLLM size arguments do not configure HF parallelism. The HF container route explicitly binds the compute host’s `/etc/hosts` read-only so Accelerate can resolve its local rendezvous address even with `--containall` and site binds disabled. For Evalchemy in either container or external-venv mode, HF uses the existing Accelerate process-per-GPU recipe; set `GPUS_PER_NODE` explicitly through `--slurm_template_var`. For harness HF, record its actual process/GPU use separately. Record the actual GPUs allocated and used in every comparison.

HF generation also clears an implicit saved-model `max_new_tokens` when the harness has calculated the task’s `max_length`. This prevents a checkpoint’s export defaults, such as 2,048 tokens, from overriding the benchmark budget. An explicit generation-call override remains intact, and the model’s saved configuration is not mutated. The contained CPU acceptance checks exercise both cases.

IFEval has two separately labelled policies. `--ifeval_stopping_policy eos` is the default and uses ordinary per-response model EOS stopping. The HF fallback uses tokenizer EOS only when the checkpoint's generation configuration omits EOS; configured EOS lists and explicit overrides remain intact. `--ifeval_stopping_policy continue` selects the packaged `oellm_ifeval` runner only for the exact IFEval task. It applies the validated greedy 1,280-token budget, ignores token/text EOS, strips special tokens consistently in both backends and grades the complete response. It refuses shortened coverage, generation overrides, non-greedy decoding and early termination, and writes an adjacent `.ifeval-policy-rank*.json` record. Other tasks continue through the ordinary harness. This option requires the complete image built above.

For example, add exactly one of these options to an otherwise identical IFEval scheduling command:

```bash
# Ordinary stopping, comparable with the EOS-stopping reference series:
# oellm-eval schedule ... --tasks ifeval --n_shot 0 --ifeval_stopping_policy eos
# Explicit continuation, a separately reported decoding configuration:
# oellm-eval schedule ... --tasks ifeval --n_shot 0 --ifeval_stopping_policy continue
```

Retain the default task prompt, all 541 examples and all 834 instructions. Do not choose a stopping policy per answer using its score. The production64k matched continuation control scored HF 36.3309% and vLLM 36.0911% strict instruction accuracy; matched EOS stopping scored 30.5755% and 30.6954%. These are different response policies, not competing arithmetic definitions of IFEval. A higher continuation score can include useful completion, repetition-based constraint matches and damage to previously valid answers; preserve the official grading unchanged and report the policy.

## Reproducibility records and recovery

Keep the oellm revision, explicit control lock/freeze, base OCI digest, Ray wheel identity, source manifest, runtime installation reports/constraints/NLTK lock, generated definitions, SIF hashes, input manifests, rendered CSV/script hashes, Slurm ID/state and raw results together. Store large images, caches and results outside Git. The repository supplies versioned recipes and small locks; the run directory supplies the identities of your built artifacts.

Compute nodes cannot repair a missing download. For an offline cache error, return to the login node, inspect the failing task's exact cache path and revision, prepare it through the evaluation image, and repeat the offline check before submitting again. For a source/hash/dependency mismatch, preserve the failed directory and log and resolve the mismatch rather than editing a checksum or broadly suppressing checks. For a timeout, report the run as incomplete and retry the full task with an adequate allocation; do not report partial coverage as a standard score.

Building the same immutable inputs reproduces the software/data configuration, not necessarily the SIF bytes or generated answer strings. Record the hash of each new image and retain the raw outputs used for every reported score.

## Earlier installation rehearsal (base recipe)

On 2026-09-10, these instructions were rehearsed on JUPITER with a fresh public OCI download, verified Ray wheel, fresh control environment/conda/pip caches, reconstructed public sources, newly installed runtime, newly built standalone image, and newly downloaded public Qwen/PIQA/GSM8K inputs. Both final-container online preparation and strict offline model/prompt/full-dataset checks passed; MATH500 contains all 500 source-packaged examples. The final image is 8,644,136,960 bytes, SHA-256 `f89a7521d2c6a4ed0ef45971ae071ba21bd9a60be90bb772f8a4a6a668434c02`. Eleven bootstrap/image tests and syntax checks for all 15 guide Bash blocks pass.

The exact three-task launcher was rendered through the fresh control environment, inspected, accepted by `sbatch --test-only`, and completed as JUPITER job `1750315` with exit code `0:0`: one node, DP4/TP1, elapsed 13m55s, 0.9278 GPU-hours. The complete-result checker passed all 1,838 PIQA examples, 1,319 GSM8K examples with both scoring filters, and 500 MATH500 examples. No local example or generation-token truncation was used.

| Full task | Metric | Rehearsal score |
|---|---|---:|
| PIQA, 10-shot | `acc_norm,none` | 76.4962% |
| GSM8K, 4-shot | `exact_match,flexible-extract` | 69.9773% |
| GSM8K, 4-shot | `exact_match,strict-match` | 61.7134% |
| MATH500, 0-shot | `accuracy` | 48.0000% |

PIQA exactly matches the earlier complete reference score. Generation scores need not match another batching layout bit for bit; preserve both GSM8K filter names when comparing results. Fresh gated GPQA download requires the collaborator's own approved access and was not re-tested in this public-input rehearsal.

## Complete-runtime installation acceptance (2026-09-11)

The complete recipe was exercised with fresh source checkouts, a fresh evaluation venv/package/NLTK preparation, a newly built standalone image and clean dedicated CLI installations. The immutable public OCI-derived base SIF and pinned model/dataset assets were reused from the earlier download rehearsal; this is a fresh software installation, not a claim that unchanged large inputs were downloaded twice. GPU execution used strict contained mode with no engine/harness/grading source overlays. All full-task coverage and retained-verdict gates below passed.

| Full production64k task | Coverage | Metric | Score % | Elapsed |
|---|---:|---|---:|---:|
| mmlu | 14042 | `acc,none` | 62.6905 | 6m57s |
| hellaswag | 10042 | `acc_norm,none` | 77.4746 | 10m19s |
| gsm8k | 1319 | `exact_match,strict-match` | 68.1577 | 4m47s |
| ifeval | 541 prompts / 834 instructions | `inst_level_strict_acc,none` | 30.0959 | 5m19s |
| mbpp | 500 | `pass_at_1,none` | 48.0000 | 4m57s |
| MATH500 | 500 | `accuracy` | 35.6000 | 9m19s |
| LiveCodeBench | 511 × 6 repeats | `accuracy_avg` | 6.5232 | 62m54s |
| HumanEval | 164 Python + 158 shell | `python_pass@1` | 67.0732 | 6m20s |

These production64k tests use its explicit identity prompt view. HumanEval also retains the complete shell result, 6.3291%. IFEval is separately tested with explicit continuation: 541 prompts, 834 instructions, 1,280 generated tokens per prompt, strict instruction accuracy 37.2902%, elapsed 6m18s. Both IFEval policies reproduce all four saved-verdict metrics and match their reference documents/request arguments; the fresh full-policy GPU tests use vLLM, while actual-library CPU tests cover both HF and vLLM policy behavior. This is a configuration/coverage/functional check, not a promise of bitwise identical generations.

The initial 18-update image has SHA-256 `5ae0222748346026ae311845664fec45adfea19a715f2b60eb661082973d06c2`. The final 19-update image, SHA-256 `c05e2f86de179d5128adf6fab8da3d929398ebdadd9894d38e0f314e2febf4c1`, adds the HF task-budget correction and updates its two CPU check scripts. All 16 vLLM, grading and policy files are byte-identical. The final image separately passes full production64k GSM8K (68.2335% strict, 1,319 examples, 4m57s) and public Qwen3-1.7B-Base HF HumanEval through four Accelerate ranks (104/164 Python, 6/158 shell, 18m35s). The public-model HF scores are not a comparison with the different production model. Reconstructing from the guide’s pinned source reproduces all 19 installed files and the same launcher/builder; its only later recipe-file difference is the read-only HumanEval result checker.

The source suite passes 152 tests and five subtests. All 20 Bash blocks in this guide parse; the identity prompt check and fresh-control HF scheduling example execute. The earlier failed prompt-view/contained-rendezvous attempts were fixed and retained. A full public-model HF MATH500 attempt was stopped for cost after the model-default token-budget correction exposed its actual standard 32,768-token workload; it is not reported as a completed check. HumanEval is the complete, standard-budget HF acceptance substitute. All 16 allocations, including failed/cancelled attempts, consumed 10.4144 GPU-hours. No merge or default-backend promotion was performed as part of this acceptance.

The fresh LCB point estimate is 6.5232%, compared with 7.0450% in the earlier overlay installation. All 511 inputs match; 2,782 identical extracted answers have zero changed grading verdicts. The difference comes from generated answers. The paired question-level comparison and its uncertainty are retained in the linked report; functional acceptance is not a claim of numerical score equivalence.

See the [complete installation report and reproducible receipts](https://github.com/SLAMPAI/oellm-workflows/blob/main/oellm_32B_loss-increase_debug/reports/eval_complete_installation_2026-09-11.md) for source/image identities, the historical-HF comparison caveats, all-attempt accounting and raw-artifact replay commands.
