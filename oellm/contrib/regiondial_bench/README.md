# RegionDial-Bench

Multi-round region grounding benchmark on RefCOCOg and RefCOCO+
(Sun et al., ICLR 2026). Evaluates a model's ability to locate and segment
objects described in multi-turn conversations, measuring robustness to error
accumulation across dialogue turns.

**Splits:**
- RefCOCOg Multi-turn — 1,580 images, 4,405 turns
- RefCOCO+ Multi-turn — 715 images, 2,355 turns

**Metrics:** gIoU (primary), cIoU, bbox_AP, pass_rate@0.3/0.5/0.7/0.9,
plus per-round breakdown (R1–R7) for gIoU and bbox_AP.

---

## Prerequisites

The benchmark calls `test/evaluation/evaluation_multi_segmentation.py` and
the `test/vision_reasoner/` model wrapper from the RegionReasoner
repository as a subprocess, so the repo must be present on the cluster
filesystem. A dedicated venv is required for `flash-attn` (specific
pre-built wheel) and HEIF image support (`pi-heif`); see
[`docs/VENV.md`](../../../docs/VENV.md) for the framework venvs.

### 1. Clone RegionReasoner

```bash
git clone https://github.com/lmsdss/RegionReasoner /path/to/RegionReasoner
```

### 2. Configure clusters.yaml

Add the following to your cluster entry in `oellm/resources/clusters.yaml`:

```yaml
my-cluster:
  ...
  HF_HOME: "/path/to/large/filesystem/huggingface"   # must have ~30 GB free
  REGION_REASONER_DIR: "/path/to/RegionReasoner"
  GPUS_PER_NODE: 4                                   # controls SLURM --gres and shard count
```

> `HF_HOME` must point to a filesystem with at least 30 GB free. On
> CINECA Leonardo, use the work filesystem
> (`/leonardo_work/<project>/huggingface`), not the home filesystem
> (50 GB quota).

### 3. Create a venv and install dependencies

```bash
uv venv --python 3.12 regiondial-venv
source regiondial-venv/bin/activate
uv pip install -e .

# PyTorch — match the cluster's CUDA driver (cu121 for driver supporting CUDA 12.2)
uv pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121

# flash-attn pre-built wheel (Python 3.12 / CUDA 12.x / torch 2.5.1)
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
uv pip install flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

# HEIF image support
uv pip install pi-heif
```

> If your Python / CUDA / torch combination differs, find the matching
> flash-attn wheel at
> <https://github.com/Dao-AILab/flash-attention/releases>.

### What gets auto-downloaded

`oellm-eval schedule` pre-downloads the following on the login node so
compute nodes do not need internet access:

| Asset | HF repo | Size |
|---|---|---|
| TaskRouter-1.5B | `Ricky06662/TaskRouter-1.5B` | ~3 GB |
| SAM2 | `facebook/sam2-hiera-large` | ~1 GB |
| RefCOCOg test JSON | `lmsdss/regionreasoner_test_data` `raw/refcocog_multi_turn.json` | ~26 GB |
| RefCOCOg test images | `lmsdss/regionreasoner_test_data` `raw/refcocog_test_multi_bbox_images/*` | ~200 MB (1 580 images) |
| RefCOCO+ test JSON | `lmsdss/regionreasoner_test_data` `raw/refcocoplus_multi_turn.json` | ~13 GB |
| RefCOCO+ test images | `lmsdss/regionreasoner_test_data` `raw/refcocoplus_test_multi_bbox_images/*` | ~93 MB (715 images) |

All assets are cached under `$HF_HOME/hub`.

---

## Running

Three task groups are available:

| Task group | Splits |
|---|---|
| `regiondial-bench` | Both (RefCOCOg + RefCOCO+) |
| `regiondial-refcocog` | RefCOCOg only (1,580 images, 4,405 turns) |
| `regiondial-refcocoplus` | RefCOCO+ only (715 images, 2,355 turns) |

```bash
# Both splits
oellm-eval schedule \
  --models lmsdss/RegionReasoner-7B \
  --task-groups regiondial-bench \
  --venv-path ~/elliot-venv

# Single split
oellm-eval schedule \
  --models lmsdss/RegionReasoner-7B \
  --task-groups regiondial-refcocog \
  --venv-path ~/elliot-venv
```

### Collecting results

```bash
oellm-eval collect /path/to/evals --output-csv results.csv
```

The primary metric in the CSV is **gIoU**. Per-round metrics (`gIoU_R1..R7`,
`bbox_AP_R1..R7`) are always computed — rounds are inferred from the order
in which each image's turns appear in the inference output (the script does
not emit an explicit round field).

---

## Evaluating a different model

The inference script supports multiple model types via the `--model` flag,
which is detected automatically from the model name. To evaluate a different
model, just pass it to `--models`:

```bash
oellm-eval schedule \
  --models Qwen/Qwen2.5-VL-7B-Instruct \
  --task-groups regiondial-bench \
  --venv-path ~/elliot-venv
```

The model type is resolved as follows:

| Model name pattern (checked in order) | `--model` flag |
|---|---|
| `*regionreasoner*` / `*region_reasoner*` | `vision_reasoner` |
| `*qwen2.5*` | `qwen2.5` |
| `*qwen2*` | `qwen2` |
| `*qwen*` | `qwen` |
| anything else | `vision_reasoner` (default) |

To evaluate multiple models in one go:

```bash
oellm-eval schedule \
  --models "lmsdss/RegionReasoner-7B,Qwen/Qwen2.5-VL-7B-Instruct" \
  --task-groups regiondial-bench \
  --venv-path ~/elliot-venv
```

> If your model name does not match any pattern above and requires a specific
> `--model` flag, extend `detect_model_flags()` in `suite.py`.
