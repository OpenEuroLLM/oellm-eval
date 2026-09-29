# Container Workflow

## Overview

Apptainer containers are built automatically via GitHub Actions and stored on HuggingFace Hub at [`openeurollm/evaluation_singularity_images`](https://huggingface.co/datasets/openeurollm/evaluation_singularity_images).

## How It Works

1. Definition files live in `containers/<cluster>.def`
2. On push to `main` (when `.def` files change), GitHub Actions provisions [Lambda Labs](https://lambdalabs.com/) GPU instances via [SkyPilot](https://skypilot.readthedocs.io/) and builds all containers in parallel
3. Built `.sif` images are uploaded to HuggingFace Hub
4. Clusters pull the image specified in `oellm/resources/clusters.yaml` via `EVAL_CONTAINER_IMAGE`

Images are compressed with zstd (level 3) via mksquashfs for a good balance of size and build speed.

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

## JudgeArena on LUMI

`containers/lumi-judgearena.def` adds JudgeArena to the pinned ROCm vLLM 0.19.1
image. The existing workflow builds it with the other matrix images and uploads
`eval_env-lumi-judgearena.sif` to the Hugging Face dataset repository above.
Set `JUDGEARENA_GIT_REF` in the definition to the reviewed JudgeArena commit SHA
before building.

On LUMI, use the image for a separate JudgeArena invocation:

```bash
export EVAL_CONTAINER_IMAGE=eval_env-lumi-judgearena.sif
export SINGULARITY_ARGS=""

oellm-eval schedule \
  --models <model> \
  --task_groups judgearena \
  --download_only

oellm-eval schedule \
  --models <model> \
  --tasks arena-hard-v0.1 \
  --n_shot 0 \
  --limit 1 \
  --judgearena_kwargs '{"judge.model":"VLLM/<judge>"}'
```

Do not use LUMI's normal `--rocm` argument with this image. The vLLM image ships
its own ROCm libraries. Run other evaluation suites separately with the normal
LUMI image.
