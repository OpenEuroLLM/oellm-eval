# ELLIOT Evaluation Platform

A multimodal evaluation framework for scheduling LLM and VLM evaluations across HPC clusters. Built as an orchestration layer over [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness), [lighteval](https://github.com/huggingface/lighteval), and [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval), with a plugin system for contributing custom benchmarks.

## Features

- **Schedule evaluations** on multiple models and tasks: `oellm-eval schedule`
- **Collect results** and check for missing evaluations: `oellm-eval collect`
- **Diagnose your environment** (cluster vars, HF cache, venv engines): `oellm-eval doctor`
- **Task groups** for pre-defined evaluation suites with automatic dataset pre-downloading
- **Multi-cluster support** with auto-detection (Leonardo, LUMI, JURECA, Jupiter, Snellius, UFAL)
- **Image evaluation** via lmms-eval (VQAv2, MMBench, MMMU, ChartQA, DocVQA, TextVQA, OCRBench, OCRBench v2, MathVista, MathVision, MMStar, AI2D, RealWorldQA, MME, MME-RealWorld, SEED-Bench)
- **Video evaluation** via lmms-eval (VideoMMMU, EgoSchema, VideoMME, ActivityNet-QA, LongVideoBench)
- **Audio evaluation** via lmms-eval (LibriSpeech, FLEURS, GigaSpeech, TED-LIUM, WenetSpeech, CoVoST2, VocalSound, MuChoMusic)
- **Tabular & time-series evaluation** via custom lm-eval tasks (TabFact, TimeSeriesExam)
- **Plugin system** for contributing custom benchmarks without touching core code
- **Automatic building and deployment of containers**

## Commands at a Glance

| Command | What it does |
|---|---|
| `oellm-eval schedule` | Expand models × tasks, pre-download models/datasets on the login node, generate and submit a SLURM array job (or run locally with `--local`) |
| `oellm-eval eval --config eval.yaml` | Same as `schedule`, driven by a YAML config file; CLI flags override the file |
| `oellm-eval collect <dir>` | Aggregate result JSONs into `eval_results.csv` + `.json` + `.md`; `--check` writes a re-schedulable CSV of missing jobs |
| `oellm-eval list-tasks` | Show every task group, its engine, task count, and n-shot settings |
| `oellm-eval compare <a> <b>` | Diff two collected results (files or run directories) per model × task × n-shot × metric |
| `oellm-eval doctor` | Diagnose the environment: cluster detection, env vars, HF cache, venv engines |

## Quick Start

**Prerequisites:**
- Install [uv](https://docs.astral.sh/uv/#installation)
- Set the `HF_HOME` environment variable to point to your HuggingFace cache directory (e.g. `export HF_HOME="/path/to/your/hf_home"`, on LUMI use the path `/scratch/project_462000963/cache/huggingface`). This is where models and datasets will be cached. Compute nodes typically have no internet access, so all assets must be pre-downloaded into this directory.

```bash
# Install
uv tool install -p 3.12 git+https://github.com/elliot-project/elliot-cli.git

# Run evaluations using a task group
oellm-eval schedule \
    --models "EleutherAI/pythia-160m" \
    --task-groups "open-sci-0.01"

# Image evaluation (requires venv with lmms-eval)
oellm-eval schedule \
    --models "llava-hf/llava-1.5-7b-hf" \
    --task-groups "image-vqa" \
    --venv-path /path/to/.venv
```

This will automatically:
- Detect your current HPC cluster (Leonardo, LUMI, JURECA, Jupiter, Snellius, or UFAL)
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
| `mmlu-prox-eu` | MMLU-ProX EU subset (5-shot CoT) | lm-eval |
| `mgsm-eu` | Multilingual GSM8K | lm-eval |
| `global-mgsm-eu` | Global-MGSM grade-school math (EU languages) | lm-eval |
| `polymath-eu-low` / `-medium` / `-high` / `-top` | PolyMath EU math reasoning, one group per difficulty tier | lm-eval |
| `xcsqa-eu` | X-CSQA commonsense QA (8 EU languages) | lm-eval |
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
| `image-ocrbench-v2` | OCRBench v2 | lmms-eval |
| `image-mathvista` | MathVista (CoT / format / solution leaves — needs GPT judge) | lmms-eval |
| `image-mathvision` | MathVision (test split) | lmms-eval |
| `image-mme` | MME (perception + cognition) | lmms-eval |
| `image-mme-realworld` | MME-RealWorld (EN) | lmms-eval |
| `image-mmstar` | MMStar | lmms-eval |
| `image-ai2d` | AI2D (diagram QA) | lmms-eval |
| `image-realworldqa` | RealWorldQA | lmms-eval |
| `image-seedbench` | SEED-Bench (image split) | lmms-eval |

### Video

| Group | Benchmark | Engine |
|---|---|---|
| `video-understanding` | All 5 benchmarks combined | lmms-eval |
| `video-videommmu` | VideoMMMU (perception / comprehension / adaptation leaves) | lmms-eval |
| `video-egoschema` | EgoSchema (long-form egocentric QA) | lmms-eval |
| `video-videomme` | Video-MME (11s-1h clips) | lmms-eval |
| `video-activitynet-qa` | ActivityNet-QA (requires GPT API) | lmms-eval |
| `video-longvideobench` | LongVideoBench (cross-segment reasoning) | lmms-eval |

The lmms-eval adapter class (`llava_hf`, `llava_onevision`, `qwen2_5_vl`, etc.) is auto-detected from the model name. Video (like image and audio) tasks run through **lmms-eval, which is not included in the cluster containers or any pip extra** — set up the general venv as described in [docs/VENV.md](docs/VENV.md) and pass `--venv-path`.

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

Audio tasks also run through lmms-eval — use the general venv from [docs/VENV.md](docs/VENV.md) (the `[audio]` extra adds the audio decoding helpers, but lmms-eval itself must be installed per that guide). Judge-model groups (AIR-Bench chat, Alpaca-Audio, OpenHermes, WavCaps) need `OPENAI_API_KEY` on the compute node — scheduling refuses without it unless you pass `--allow-missing-judge`. The HPC Singularity image must include `ffmpeg` for non-WAV decode.

### Tabular & Time Series

| Group | Benchmark | Engine |
|---|---|---|
| `tabular-tabfact` | TabFact — is a statement entailed or refuted by a Wikipedia table (few-shot from the train split) | lm-eval (custom task) |
| `timeseries-tsexam` | TimeSeriesExam — MCQ time-series understanding: trend, seasonality, anomalies, causality (strictly 0-shot) | lm-eval (custom task) |

Both are custom lm-eval task definitions shipped in [`custom_lm_eval_tasks/`](oellm/resources/custom_lm_eval_tasks/) — no plugin needed; see path 2 in the [Contributing Guide](oellm/contrib/CONTRIBUTING.md). TabFact's dataset is script-based, so schedule it with `--trust-remote-code`.

### Custom Benchmarks (contrib)

Community-contributed benchmarks that run outside the standard evaluation engines. See the [contrib registry](oellm/contrib/README.md) for the full list.

```bash
# Run all 8 image benchmarks
oellm-eval schedule \
    --models "llava-hf/llava-1.5-7b-hf" \
    --task-groups "image-vqa" \
    --venv-path /path/to/.venv

# Run all 5 video benchmarks
oellm-eval schedule \
    --models "lmms-lab/llava-onevision-7b" \
    --task-groups "video-understanding" \
    --venv-path /path/to/.venv

# Mix image and text benchmarks in one submission
oellm-eval schedule \
    --models "llava-hf/llava-1.5-7b-hf" \
    --task-groups "image-mmbench,open-sci-0.01" \
    --venv-path /path/to/.venv

# Tabular + time-series benchmarks (custom lm-eval tasks)
oellm-eval schedule \
    --models "meta-llama/Llama-3.1-8B-Instruct" \
    --task-groups "tabular-tabfact,timeseries-tsexam" \
    --venv-path /path/to/.venv --trust-remote-code

# Use multiple task groups or a super group
oellm-eval schedule --models "model-name" --task-groups "belebele-eu-5-shot,global-mmlu-eu"
oellm-eval schedule --models "model-name" --task-groups "oellm-multilingual"
```

### Filtering by language

Scope a task group (or super group) to one or more languages by attaching a
`[...]` bracket to its name. Languages use canonical
[`lang_Script`](https://en.wikipedia.org/wiki/IETF_language_tag) codes (e.g.
`deu_Latn`, `fra_Latn`); codes inside a bracket may be separated by `,` or `|`.

```bash
# The applicable subset of the multilingual super group for one language —
# the simplest way to evaluate a monolingual model on its language across
# every multilingual benchmark (FLORES, Belebele, Global-MMLU, INCLUDE, MGSM,
# …), spanning both lm-eval-harness and lighteval.
oellm-eval schedule --models "my-model" --task-groups "oellm-multilingual[deu_Latn]"

# Every benchmark in the registry for one language. `all` is an auto-generated
# super group (always spans every task group, no hand-maintenance) — use it for
# the complete per-language set rather than the curated `oellm-multilingual`.
oellm-eval schedule --models "my-model" --task-groups "all[deu_Latn]"

# A single benchmark, scoped to German.
oellm-eval schedule --models "my-model" --task-groups "sib200-eu[deu_Latn]"

# Multiple languages inside one bracket.
oellm-eval schedule --models "my-model" --task-groups "sib200-eu[fra_Latn|deu_Latn]"

# Different languages per benchmark in one run — French SIB-200 *and* German FLORES.
oellm-eval schedule --models "my-model" \
    --task-groups "sib200-eu[fra_Latn],flores-200-eu-to-eng[deu_Latn]"

# No bracket: the group is unchanged — all of its languages.
oellm-eval schedule --models "my-model" --task-groups "sib200-eu"
```

Each task's language is derived in code from the `{lang}` value of a
`valid_langs` template, or the task's `subset` for explicitly-listed
multilingual groups (see [docs/TASKS.md](docs/TASKS.md)). No per-task tagging is
needed — adding a benchmark with a `valid_langs` template makes its languages
filterable automatically.

Notes:

- Use the precise canonical `lang_Scri` code (e.g. `deu_Latn`). Looser
  spellings such as `de` or `german` are rejected; if you pass one, the error
  names the canonical code to use instead.
- An **unknown** code (typo) errors and lists the valid codes.
- A bracket that matches **no task** in its group (e.g.
  `flores-200-eu-to-eng[ukr_Cyrl]`, since FLORES-EU has no Ukrainian) errors
  rather than silently scheduling nothing.
- When a bracket lists several languages and only **some** are present in the
  group, it keeps the matches and warns about the rest. Some languages simply
  lack certain benchmarks (e.g. Italian/Portuguese have no MGSM), so a super
  group bracket transparently omits the missing ones.

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
    --local \
    --limit 1
```

Results are written to `./oellm-output/<timestamp>/results/`.

**Air-gapped cluster nodes (no internet):** batch jobs set `HF_HUB_OFFLINE=1` and get `HF_HOME` from your cluster env. With `--local`, the CLI defaults `HF_HOME` to `~/.cache/huggingface` if unset and would otherwise allow Hub access—so on a compute node without network, export your real cache and offline flag before running, for example:

```bash
export HF_HOME=/path/to/your/shared/hf_cache   # e.g. $WORK/hf_cache on Leonardo
export HF_HUB_OFFLINE=1
oellm-eval schedule ... --venv-path .venv --local
```

The `HF_HUB_OFFLINE` value is read when you invoke `oellm-eval` and baked into the generated script.

## SLURM Overrides

Override cluster defaults (partition, account, time limit, memory, etc.) with `--slurm-template-var` (JSON object). Provide `SLURM_MEM` to request an exact host memory amount, otherwise falls back to a default of `96G`.

```bash
# Use a different partition (e.g. dev-g on LUMI when small-g is crowded)
oellm-eval schedule --models "model-name" --task-groups "open-sci-0.01" \
  --slurm-template-var '{"PARTITION":"dev-g"}'

# Multiple overrides: partition, account, time limit, GPUs, exact RAM
oellm-eval schedule --models "model-name" --task-groups "open-sci-0.01" \
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
  --venv-path /path/to/.venv
```

If you need full manual control over all model args, set `MODEL_ARGS`,
for example:

```bash
MODEL_ARGS='batch_size=8' oellm-eval schedule \
  --models "model-name" --task-groups "belebele-eu-cf" --venv-path /path/to/.venv
```

## Quantized Evaluation & Per-Row Timeouts

```bash
# bitsandbytes 4-bit / 8-bit loading (lm-eval, lmms-eval, evalchemy; flags are mutually exclusive)
oellm-eval schedule --models "model-name" --task-groups "open-sci-0.01" --load-in-4bit

# Bound each evaluation row's wall clock: a hung engine fails one row (exit 124)
# instead of consuming the whole job slice; `collect --check` re-schedules just that row
ROW_TIMEOUT=30m oellm-eval schedule --models "model-name" --task-groups "open-sci-0.01"
```

Quantization is recorded in the run's provenance. Rows on engines that cannot
honor it run at full precision — the scheduler lists them at submission time.

## ⚠️ Dataset Pre-Download Warning

**Datasets are only automatically pre-downloaded for tasks defined in [`task-groups.yaml`](oellm/resources/task-groups.yaml).**

If you use custom tasks via `--tasks` that are not in the task groups registry, the CLI will attempt to look them up but **cannot guarantee the datasets will be cached**. This may cause failures on compute nodes that don't have network access.

**Recommendation:** Use `--task-groups` when possible, or ensure your custom task datasets are already cached in `$HF_HOME` before scheduling.

## Collecting Results

After evaluations complete, collect results into a CSV.  `collect` **recursively** searches the given directory for every `jobs.csv` file and every `.json` result file, so you can point it at a top-level output folder that contains many sub-runs. Alongside the CSV/Markdown tables it writes `eval_results.json` — a versioned envelope that embeds each run's provenance (engine versions, model revisions, quantization, submitter), which is also the ingestion format for the ELLIOT evaluation dashboard:

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
# Basic collection — writes eval_results.csv, eval_results.json, eval_results.md
oellm-eval collect /path/to/eval-output-dir

# Check for missing evaluations and create a CSV for re-running them
oellm-eval collect /path/to/eval-output-dir --check --output-csv results.csv
```

Three output files are written next to your `--output-csv` path: the CSV (raw metric per row), a versioned JSON envelope, and a Markdown table with metrics normalized to a 0–100 scale.

All `jobs.csv` files found under `results_dir` are merged into one; if the same `(model_path, task_path, n_shot)` row appears in multiple files the later-sorted entry wins (override duplicates). The merged jobs list is then compared against all `.json` result files found recursively.

The `--check` flag outputs a `results_missing.csv` that can be used to re-schedule failed jobs:

```bash
oellm-eval schedule --eval-csv-path results_missing.csv
```

## CSV-Based Scheduling

For full control, provide a CSV file with columns: `model_path`, `task_path`, `n_shot`, and optionally `eval_suite` (one of `lm_eval` — the default, `lighteval`, `lmms_eval`, `evalchemy`, or a contrib suite name):

```bash
oellm-eval schedule --eval-csv-path custom_evals.csv
```

> **Note:** field values must not contain commas, quotes, or newlines — the SLURM-side reader splits rows on commas, and scheduling rejects such rows with an error. Model args like `model,revision=...` are not supported.

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

### UFAL Specifics

Due to limited space in `$HOME` on UFAL cluster, set these environment variables for your personal copies of tools and models:

```bash
basedir="/lnet/troja/tmp/$USER"
export UV_CACHE_DIR="$basedir/cache/uv-cache"
export UV_INSTALL_DIR="$basedir/local"
export UV_PYTHON_INSTALL_DIR="$basedir/local/share/uv/python"
export UV_TOOL_DIR="$basedir/cache/uv-tool-cache"
export HF_HOME="$basedir/cache/huggingface"
```

Then, please install the virtual environment that is compatible with UFAL cluster as follows:
```bash
uv venv --python 3.11
uv pip install -r requirements-venv-ufal.txt --no-deps
```

To run an evaluation, you **MUST** include the `--venv-path <installed_venv_path>`, as the UFAL cluster does not support containers (for now, Jun 23 2026), for example:
```bash
oellm-eval schedule \
    --models "EleutherAI/pythia-160m" \
    --tasks "gsm8k" \
    --n-shot 0 \
    --venv-path /path/to/.venv
```

Later, we will add recommendation for a project-wide setting to share tools and models.

## Supported Clusters

Leonardo, LUMI, JURECA, Jupiter, Snellius, and UFAL — detected automatically from the login node's hostname (see [`oellm/resources/clusters.yaml`](oellm/resources/clusters.yaml)). Any value there can be overridden by exporting the environment variable before scheduling.

## Environment Diagnostics

`schedule` verifies before submission that the chosen runtime (venv or container) can actually run the scheduled suites — missing engines, missing suite env vars (e.g. `AUDIOBENCH_DIR`), and version-pinned groups on the wrong engine (`dclm-core-22` needs `lm-eval==0.4.9.2`) are rejected with an actionable message instead of failing hours later on a compute node. Bypass with `--skip-checks`.

Run the same checks standalone at any time:

```bash
# Full report: cluster detection, env vars, HF cache, SLURM binaries, venv engines
oellm-eval doctor --venv-path /path/to/.venv

# Treat the engines these groups need as required (exit 1 if missing)
oellm-eval doctor --venv-path /path/to/.venv --task-groups "image-vqa,open-sci-0.01"
```

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
uv run oellm-eval schedule --models "EleutherAI/pythia-160m" --task-groups "open-sci-0.01" --download-only
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

ELLIOT supports three paths for adding benchmarks:

1. **Benchmark already in lm-eval / lighteval / lmms-eval** -- add a YAML entry to [`task-groups.yaml`](oellm/resources/task-groups.yaml)
2. **Custom lm-eval task, no plugin** -- a task YAML plus optional prompt helpers in [`custom_lm_eval_tasks/`](oellm/resources/custom_lm_eval_tasks/); TabFact and TimeSeriesExam are the reference implementations
3. **Fully custom benchmark** -- drop a contrib plugin into [`oellm/contrib/`](oellm/contrib/)

See the [Contributing Guide](oellm/contrib/CONTRIBUTING.md) for step-by-step instructions.

## Deploying Containers

Containers are deployed manually since [PR #46](https://github.com/elliot-project/elliot-cli/pull/46). To build and deploy, select "Run workflow" in [Actions](https://github.com/elliot-project/elliot-cli/actions/workflows/build-and-push-apptainer.yml).
