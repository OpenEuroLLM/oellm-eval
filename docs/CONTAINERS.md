# Container Workflow

## Overview

Apptainer containers are built automatically via GitHub Actions and stored on HuggingFace Hub at [`openeurollm/evaluation_singularity_images`](https://huggingface.co/datasets/openeurollm/evaluation_singularity_images).

## How It Works

1. Definition files live in `containers/<cluster>.def`
2. Builds are triggered **manually** ("Run workflow" on the [build-and-push-apptainer](../.github/workflows/build-and-push-apptainer.yml) action — manual since PR #46). The workflow provisions [Lambda Labs](https://lambdalabs.com/) GPU instances via [SkyPilot](https://skypilot.readthedocs.io/) and builds all containers in parallel
3. Built `.sif` images are uploaded to HuggingFace Hub
4. Clusters pull the image specified in `oellm/resources/clusters.yaml` via `EVAL_CONTAINER_IMAGE`

Images are compressed with zstd (level 3) via mksquashfs for a good balance of size and build speed.

## What the Images Contain (and What They Don't)

The shipped images provide exactly **two engines**: `lm-eval` (system Python)
and `lighteval` (installed as an isolated uv tool to avoid the `datasets`
version conflict). They do **not** contain `lmms-eval`, the `oellm` package
(needed by contrib suites), or evalchemy — scheduling those suites in
container mode is rejected by the environment pre-flight; pass `--venv-path`
with a suitable venv instead (see [VENV.md](VENV.md)).

## Image / Video / Audio Evaluation (lmms-eval)

lmms-eval benchmarks require a custom venv:

**Option 1 — Custom venv (recommended):**
Follow the general-venv setup in [VENV.md](VENV.md) — note that `lmms-eval`
is **not** provided by any pyproject extra; it must be installed editable
from git (its wheel build drops required template files), alongside the
`[text,image,audio]` extras.

**Option 2 — Build a container with lmms-eval:**
Extend a `.def` file — install lmms-eval **editable from git** (a plain
`pip install lmms-eval` produces a broken install — the wheel drops
`_default_template_yaml` files) and include the `oellm` package if contrib
suites should run in the container:

```singularity
%post
    uv pip install --system --break-system-packages \
        lm-eval torch transformers accelerate "datasets<4.0.0"
    uv pip install --system --break-system-packages \
        -e "git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git#egg=lmms-eval"
```

Then set `EVAL_CONTAINER_IMAGE` in `clusters.yaml` to point to this image.

## Adding a New Cluster

1. Create `containers/<cluster>.def` with the appropriate base image:
   - NVIDIA: `nvcr.io/nvidia/pytorch:25.06-py3` (or newer)
   - AMD/ROCm: `rocm/pytorch:rocm6.4.1_ubuntu24.04_py3.12_pytorch_release_2.7.1` (or newer)

2. Add the cluster to the matrix in `.github/workflows/build-and-push-apptainer.yml`:
   ```yaml
   matrix:
     include:
       - image: <new-cluster>
         arch: arm64  # omit for default x86_64
   ```

3. Add cluster configuration to `oellm/resources/clusters.yaml`:
   ```yaml
   <cluster>:
     hostname_pattern: "<pattern>"
     EVAL_BASE_DIR: "<path>"
     PARTITION: "<partition>"
     ACCOUNT: "<account>"
     QUEUE_LIMIT: <limit>
     EVAL_CONTAINER_IMAGE: "eval_env-<cluster>.sif"
     SINGULARITY_ARGS: "--nv"  # or "--rocm" for AMD
   ```

4. Push to `main` to trigger the build.
