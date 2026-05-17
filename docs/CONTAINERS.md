# Container Workflow

## Overview

Apptainer containers are built automatically via GitHub Actions and stored on HuggingFace Hub at [`openeurollm/evaluation_singularity_images`](https://huggingface.co/datasets/openeurollm/evaluation_singularity_images).

## How It Works

1. Definition files live in `containers/<cluster>.def`
2. On push to `main` (when `.def` files change), GitHub Actions provisions [Lambda Labs](https://lambdalabs.com/) GPU instances via [SkyPilot](https://skypilot.readthedocs.io/) and builds all containers in parallel
3. Built `.sif` images are uploaded to HuggingFace Hub
4. Clusters pull the image specified in `oellm/resources/clusters.yaml` via `EVAL_CONTAINER_IMAGE`

Images are compressed with zstd (level 3) via mksquashfs for a good balance of size and build speed.

## Image Evaluation (lmms-eval)

Image benchmarks (`suite: lmms_eval`) require `lmms-eval` to be available in the execution environment. There are two ways to provide it:

**Option 1 — Custom venv (recommended for development):**
Install `lmms-eval` via the `[image]` (or `[audio]`/`[video]`) pyproject extra and pass `--venv-path` to the CLI:
```bash
uv pip install --python /path/to/.venv/bin/python -e '.[text,image,audio]'
```
See [VENV.md](VENV.md) for full setup instructions.

**Option 2 — Container with lmms-eval:**
Build a container `.def` file that includes `lmms-eval` alongside `lm-eval`:

```singularity
%post
    pip install lm-eval torch transformers accelerate "datasets<4.0.0" "lmms-eval>=0.2.4"
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
