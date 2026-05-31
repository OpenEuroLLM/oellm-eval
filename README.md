# ELLIOT Evaluation Platform

A multimodal evaluation framework for scheduling LLM and VLM evaluations across HPC clusters. Built as an orchestration layer over [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness), [lighteval](https://github.com/huggingface/lighteval), and [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval), with a plugin system for contributing custom benchmarks.

## Features

- **Schedule evaluations** on multiple models and tasks: `oellm-eval schedule`
- **Collect results** and check for missing evaluations: `oellm-eval collect`
- **Task groups** for pre-defined evaluation suites with automatic dataset pre-downloading
- **Multi-cluster support** with auto-detection (Leonardo, LUMI, JURECA, Snellius)
- **Image evaluation** via lmms-eval (VQAv2, MMBench, MMMU, ChartQA, DocVQA, TextVQA, OCRBench, MathVista)
- **Video evaluation** via lmms-eval (VideoMMMU, EgoSchema, VideoMME, ActivityNet-QA, LongVideoBench)
- **Audio evaluation** via lmms-eval (LibriSpeech, FLEURS, GigaSpeech, TED-LIUM, WenetSpeech, CoVoST2, VocalSound, MuChoMusic)
- **Plugin system** for contributing custom benchmarks without touching core code
- **Automatic building and deployment of containers**

## Quick Start

**Prerequisites:**
- Install [uv](https://docs.astral.sh/uv/#installation)
- Set the `HF_HOME` environment variable to point to your HuggingFace cache directory (e.g. `export HF_HOME="/path/to/your/hf_home"`, on LUMI use the path `/scratch/project_462000963/cache/huggingface`). This is where models and datasets will be cached. Compute nodes typically have no internet access, so all assets must be pre-downloaded into this directory.

```bash
# Install
uv tool install -p 3.12 git+https://github.com/elliot-project/elliot-cli.git

# Run evaluations using a task group
oellm schedule-eval \
    --models "EleutherAI/pythia-160m" \
    --task-groups "open-sci-0.01"

# Image evaluation (requires venv with lmms-eval)
oellm schedule-eval \
    --models "llava-hf/llava-1.5-7b-hf" \
    --task-groups "image-vqa" \
    --venv-path ~/elliot-venv
```

This will automatically:
- Detect your current HPC cluster (Leonardo, LUMI, JURECA, or Snellius)
- Download and cache the specified models
- Pre-download datasets for known tasks (see warning below)
- Generate and submit a SLURM job array with appropriate cluster-specific resources and using containers built for this cluster

For custom environments instead of containers, pass `--venv-path` (see [docs/VENV.md](docs/VENV.md)).

## Task Groups

Task groups are pre-defined evaluation suites in [`task-groups.yaml`](oellm/resources/task-groups.yaml). Each group specifies tasks, their n-shot settings, and HuggingFace dataset mappings.

### Text & Multilingual

| Group | Description | Engine |
|---|---|---|
| `open-sci-0.01` | COPA, MMLU, HellaSwag, ARC, etc. | lm-eval |
| `belebele-eu-5-shot` | Belebele in 23 European languages | lm-eval |
| `flores-200-eu-to-eng` | EU to English translation | lighteval |
| `flores-200-eng-to-eu` | English to EU translation | lighteval |
| `global-mmlu-eu` | Global MMLU in EU languages | lm-eval |
| `mgsm-eu` | Multilingual GSM8K | lm-eval |
| `generic-multilingual` | XWinograd, XCOPA, XStoryCloze | lm-eval |
| `include` | INCLUDE benchmarks (44 languages) | lm-eval |

Super groups: `oellm-multilingual` (all multilingual benchmarks combined)

### Image

| Group | Benchmark | Engine |
|---|---|---|
| `image-vqa` | All 8 benchmarks combined | lmms-eval |
| `image-vqav2` | VQAv2 | lmms-eval |
| `image-mmbench` | MMBench | lmms-eval |
| `image-mmmu` | MMMU | lmms-eval |
| `image-chartqa` | ChartQA | lmms-eval |
| `image-docvqa` | DocVQA | lmms-eval |
| `image-textvqa` | TextVQA | lmms-eval |
| `image-ocrbench` | OCRBench | lmms-eval |
| `image-mathvista` | MathVista (CoT / format / solution leaves — needs GPT judge) | lmms-eval |

### Video

| Group | Benchmark | Engine |
|---|---|---|
| `video-understanding` | All 5 benchmarks combined | lmms-eval |
| `video-videommmu` | VideoMMMU (perception / comprehension / adaptation leaves) | lmms-eval |
| `video-egoschema` | EgoSchema (long-form egocentric QA) | lmms-eval |
| `video-videomme` | Video-MME (11s-1h clips) | lmms-eval |
| `video-activitynet-qa` | ActivityNet-QA (requires GPT API) | lmms-eval |
| `video-longvideobench` | LongVideoBench (cross-segment reasoning) | lmms-eval |

The lmms-eval adapter class (`llava_hf`, `llava_onevision`, `qwen2_5_vl`, etc.) is auto-detected from the model name. Install with `pip install oellm[video]` (or use a venv with lmms-eval).

### Audio

| Group | Benchmark | Engine |
|---|---|---|
| `audio-understanding` | Curated suite: 8 leaf tasks, no judge-model dependency | lmms-eval |
| `audio-librispeech` | LibriSpeech ASR (WER on test-clean) | lmms-eval |
| `audio-common-voice-15` | Common Voice 15 (en, fr, zh-CN) | lmms-eval |
| `audio-gigaspeech` | GigaSpeech large-scale ASR | lmms-eval |
| `audio-tedlium` | TED-LIUM v3 ASR | lmms-eval |
| `audio-wenet-speech` | WenetSpeech Chinese ASR (MER) | lmms-eval |
| `audio-covost2` | CoVoST2 en→zh speech translation (BLEU) | lmms-eval |
| `audio-fleurs` | FLEURS multilingual speech | lmms-eval |
| `audio-voxpopuli`, `audio-ami`, `audio-people-speech` | Additional ASR corpora | lmms-eval |
| `audio-vocalsound` | Non-speech vocalisation classification | lmms-eval |
| `audio-muchomusic` | Music understanding MCQ | lmms-eval |
| `audio-air-bench-chat` | AIR-Bench chat (requires GPT judge) | lmms-eval |
| `audio-air-bench-foundation` | AIR-Bench foundation MCQ | lmms-eval |
| `audio-alpaca-audio`, `audio-openhermes`, `audio-wavcaps` | Instruction / captioning (GPT judge) | lmms-eval |
| `audio-clotho-aqa`, `audio-cn-college-listen-mcq`, `audio-dream-tts-mcq`, `audio-voicebench`, `audio-step2-paralinguistic` | QA / MCQ / paralinguistic probes | lmms-eval |

Install with `pip install oellm[audio]`. Judge-model groups (AIR-Bench chat, Alpaca-Audio, OpenHermes, WavCaps) need `OPENAI_API_KEY` on the compute node. The HPC Singularity image must include `ffmpeg` for non-WAV decode.

### Custom Benchmarks (contrib)

Community-contributed benchmarks that run outside the standard evaluation engines. See the [contrib registry](oellm/contrib/README.md) for the full list.

```bash
# Run all 8 image benchmarks
oellm schedule-eval \
    --models "llava-hf/llava-1.5-7b-hf" \
    --task-groups "image-vqa" \
    --venv-path ~/elliot-venv

# Run all 5 video benchmarks
oellm schedule-eval \
    --models "lmms-lab/llava-onevision-7b" \
    --task-groups "video-understanding" \
    --venv-path ~/elliot-venv

# Mix image and text benchmarks in one submission
oellm schedule-eval \
    --models "llava-hf/llava-1.5-7b-hf" \
    --task-groups "image-mmbench,open-sci-0.01" \
    --venv-path ~/elliot-venv

# Use multiple task groups or a super group
oellm schedule-eval --models "model-name" --task-groups "belebele-eu-5-shot,global-mmlu-eu"
oellm schedule-eval --models "model-name" --task-groups "oellm-multilingual"
```

## Running Locally (without SLURM)

The `--local` flag lets you run evaluations directly on your machine without a cluster or Singularity container. It generates the same eval script and executes it with bash, injecting fake SLURM environment variables so all tasks run sequentially in a single process. This is useful for testing that tasks and models are correctly configured before submitting to a cluster.

```bash
# 1. Add eval dependencies to the project venv
uv pip install lm-eval torch transformers accelerate "datasets<4.0.0"

# 2. Run evaluations locally — useful for smoke-testing with a small sample
oellm-eval schedule \
    --models "EleutherAI/pythia-160m" \
    --tasks "gsm8k" \
    --n-shot 0 \
    --venv-path .venv \
    --local true \
    --limit 1
```

Results are written to `./oellm-output/<timestamp>/results/`.

**Air-gapped cluster nodes (no internet):** batch jobs set `HF_HUB_OFFLINE=1` and get `HF_HOME` from your cluster env. With `--local`, the CLI defaults `HF_HOME` to `~/.cache/huggingface` if unset and would otherwise allow Hub access—so on a compute node without network, export your real cache and offline flag before running, for example:

```bash
export HF_HOME=/leonardo_work/OELLM_prod2026/users/shaldar0/oellm-evals/hf_data
export HF_HUB_OFFLINE=1
oellm-eval schedule ... --venv_path .venv --local true
```

The `HF_HUB_OFFLINE` value is read when you invoke `oellm-eval` and baked into the generated script.

## SLURM Overrides

Override cluster defaults (partition, account, time limit, memory, etc.) with `--slurm-template-var` (JSON object). Provide `SLURM_MEM` to request an exact host memory amount, otherwise falls back to a default of `96G`.

```bash
# Use a different partition (e.g. dev-g on LUMI when small-g is crowded)
oellm schedule-eval --models "model-name" --task-groups "open-sci-0.01" \
  --slurm-template-var '{"PARTITION":"dev-g"}'

# Multiple overrides: partition, account, time limit, GPUs, exact RAM
oellm schedule-eval --models "model-name" --task-groups "open-sci-0.01" \
  --slurm-template-var '{"PARTITION":"dev-g","ACCOUNT":"myproject","TIME":"02:00:00","GPUS_PER_NODE":2,"SLURM_MEM":"96G"}'
```

Use exact env var names: `PARTITION`, `ACCOUNT`, `GPUS_PER_NODE`, `SLURM_MEM`. `TIME` (HH:MM:SS) overrides the time limit.

## Lighteval Batch Size

For lighteval runs, generated jobs default to `batch_size=1` for local runs and
`batch_size=32` for non-local (SLURM/cluster) runs. This reduces the risk of
out-of-memory failures where lighteval's auto batch-size detection can be
overly optimistic for multiple-choice loglikelihood tasks. You can still
override these defaults:

```bash
# Set an explicit batch size (overrides the local/cluster default)
BATCH_SIZE=8 oellm-eval schedule \
  --models "model-name" \
  --task-groups "belebele-eu-cf" \
  --venv-path .venv
```

If you need full manual control over all model args, set `MODEL_ARGS`,
for example:

```bash
MODEL_ARGS='batch_size=8' oellm schedule-eval \
  --models "model-name" --task-groups "belebele-eu-cf" --venv-path .venv
```

## ⚠️ Dataset Pre-Download Warning

**Datasets are only automatically pre-downloaded for tasks defined in [`task-groups.yaml`](oellm/resources/task-groups.yaml).**

If you use custom tasks via `--tasks` that are not in the task groups registry, the CLI will attempt to look them up but **cannot guarantee the datasets will be cached**. This may cause failures on compute nodes that don't have network access.

**Recommendation:** Use `--task-groups` when possible, or ensure your custom task datasets are already cached in `$HF_HOME` before scheduling.
## Collecting Results

After evaluations complete, collect results into a CSV.  `collect-results` **recursively** searches the given directory for every `jobs.csv` file and every `.json` result file, so you can point it at a top-level output folder that contains many sub-runs:

```
output/
├── hellaswag_mt1/
│   ├── jobs.csv
│   └── results/
├── hellaswag_mt2/
│   ├── jobs.csv
│   └── results/
└── global_mmlu1/
    ├── jobs.csv
    └── results/
```

```bash
# Basic collection
oellm collect-results --results-dir /path/to/eval-output-dir

# Check for missing evaluations and create a CSV for re-running them
oellm collect-results --results-dir /path/to/eval-output-dir --check true --output-csv results.csv
```

All `jobs.csv` files found under `results_dir` are merged into one; if the same `(model_path, task_path, n_shot)` row appears in multiple files the later-sorted entry wins (override duplicates). The merged jobs list is then compared against all `.json` result files found recursively.

The `--check` flag outputs a `results_missing.csv` that can be used to re-schedule failed jobs:

```bash
oellm schedule-eval --eval-csv-path results_missing.csv
```

## CSV-Based Scheduling

For full control, provide a CSV file with columns: `model_path`, `task_path`, `n_shot`, and optionally `eval_suite`:

```bash
oellm schedule-eval --eval-csv-path custom_evals.csv
```

## Installation

```bash
uv tool install -p 3.12 git+https://github.com/elliot-project/elliot-cli.git
```

Update to latest:
```bash
uv tool upgrade oellm-eval
```

### JURECA/JSC Specifics

Due to limited space in `$HOME` on JSC clusters, set these environment variables:

```bash
export UV_CACHE_DIR="/p/project1/<project>/$USER/.cache/uv-cache"
export UV_INSTALL_DIR="/p/project1/<project>/$USER/.local"
export UV_PYTHON_INSTALL_DIR="/p/project1/<project>/$USER/.local/share/uv/python"
export UV_TOOL_DIR="/p/project1/<project>/$USER/.cache/uv-tool-cache"
```

## Supported Clusters:
We support: Leonardo, Lumi, Jureca, Jupiter, and Snellius

## CLI Options

```bash
oellm-eval schedule --help
```

## Development

```bash
git clone https://github.com/elliot-project/elliot-cli.git
cd elliot-cli
uv sync --extra dev

# Run all unit tests
uv run pytest tests/ -v

# Download-only mode for testing
uv run oellm schedule-eval --models "EleutherAI/pythia-160m" --task-groups "open-sci-0.01" --download-only
```

## Documentation

### Cluster Setup

| Cluster | Guide |
|---|---|
| Leonardo (CINECA) | [docs/LEONARDO.md](docs/LEONARDO.md) |
| LUMI, JURECA | Coming soon |
| Snellius | Coming soon |

### Environment & Infrastructure

| Doc | Description |
|---|---|
| [Using a Virtual Environment](docs/VENV.md) | Setting up a custom venv with lm-eval, lmms-eval, and lighteval |
| [Container Workflow](docs/CONTAINERS.md) | How Apptainer containers are built, deployed, and used |

### Extending the Platform

| Doc | Description |
|---|---|
| [Adding Tasks & Task Groups](docs/TASKS.md) | YAML structure for defining new evaluation suites |
| [Contributing Custom Benchmarks](oellm/contrib/CONTRIBUTING.md) | Step-by-step guide for adding a contrib plugin |
| [Contrib Registry](oellm/contrib/README.md) | List of community-contributed benchmarks |

## Contributing Custom Benchmarks

ELLIOT supports two paths for adding benchmarks:

1. **Benchmark already in lm-eval / lighteval / lmms-eval** -- add a YAML entry to [`task-groups.yaml`](oellm/resources/task-groups.yaml)
2. **Fully custom benchmark** -- drop a contrib plugin into [`oellm/contrib/`](oellm/contrib/)

See the [Contributing Guide](oellm/contrib/CONTRIBUTING.md) for step-by-step instructions.

## Deploying Containers

Containers are deployed manually since [PR #46](https://github.com/elliot-project/elliot-cli/pull/46). To build and deploy, select "Run workflow" in [Actions](https://github.com/elliot-project/elliot-cli/actions/workflows/build-and-push-apptainer.yml).
