# OpenEuroLLM Eval (oellm-eval)

A lightweight CLI for scheduling LLM evaluations across multiple HPC clusters using SLURM job arrays and Singularity containers.

## Features

- **Schedule evaluations** on multiple models and tasks: `oellm-eval schedule`
- **Collect results** and check for missing evaluations: `oellm-eval collect`
- **Task groups** for pre-defined evaluation suites with automatic dataset pre-downloading
- **Multi-cluster support** with auto-detection (Leonardo, LUMI, JURECA, Snellius)
- **Automatic building and deployment of containers** 

## Quick Start

**Prerequisites:**
- Install [uv](https://docs.astral.sh/uv/#installation)
- Set the `HF_HOME` environment variable to point to your HuggingFace cache directory (e.g. `export HF_HOME="/path/to/your/hf_home"`, on LUMI use the path `/scratch/project_462000963/cache/huggingface`). This is where models and datasets will be cached. Compute nodes typically have no internet access, so all assets must be pre-downloaded into this directory.

```bash
# Install the package
uv tool install -p 3.12 git+https://github.com/OpenEuroLLM/oellm-eval.git

# Run evaluations using a task group (recommended)
oellm-eval schedule \
    --models "microsoft/DialoGPT-medium,EleutherAI/pythia-160m" \
    --task_groups "open-sci-0.01"

# Or specify individual tasks
oellm-eval schedule \
    --models "EleutherAI/pythia-160m" \
    --tasks "hellaswag,mmlu" \
    --n_shot 5
```

This will automatically:
- Detect your current HPC cluster (Leonardo, LUMI, JURECA, or Snellius)
- Download and cache the specified models
- Pre-download datasets for known tasks (see warning below)
- Generate and submit a SLURM job array with appropriate cluster-specific resources and using containers built for this cluster

In case you do not want to rely on the containers provided on a given cluster or try out specific package versions, you can use a custom environment by passing `--venv_path`, see [docs/VENV.md](docs/VENV.md).

## Task Groups

Task groups are pre-defined evaluation suites in [`task-groups.yaml`](oellm/resources/task-groups.yaml). Each group specifies tasks, their n-shot settings, and HuggingFace dataset mappings.

Available task groups:
- `open-sci-0.01` - Standard benchmarks (COPA, MMLU, HellaSwag, ARC, etc.)
- `belebele-eu-5-shot` - Belebele European language tasks
- `flores-200-eu-to-eng` / `flores-200-eng-to-eu` - Translation tasks
- `global-mmlu-eu` - Global MMLU in EU languages
- `mgsm-eu` - Multilingual GSM benchmarks
- `generic-multilingual` - XWinograd, XCOPA, XStoryCloze
- `include` - INCLUDE benchmarks

Super groups combine multiple task groups:
- `oellm-multilingual` - All multilingual benchmarks combined

```bash
# Use a task group
oellm-eval schedule --models "model-name" --task_groups "open-sci-0.01"

# Use multiple task groups
oellm-eval schedule --models "model-name" --task_groups "belebele-eu-5-shot,global-mmlu-eu"

# Use a super group
oellm-eval schedule --models "model-name" --task_groups "oellm-multilingual"
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
oellm-eval schedule --models "my-model" --task_groups "oellm-multilingual[deu_Latn]"

# Every benchmark in the registry for one language. `all` is an auto-generated
# super group (always spans every task group, no hand-maintenance) — use it for
# the complete per-language set rather than the curated `oellm-multilingual`.
oellm-eval schedule --models "my-model" --task_groups "all[deu_Latn]"

# A single benchmark, scoped to German.
oellm-eval schedule --models "my-model" --task_groups "sib200-eu[deu_Latn]"

# Multiple languages inside one bracket.
oellm-eval schedule --models "my-model" --task_groups "sib200-eu[fra_Latn|deu_Latn]"

# Different languages per benchmark in one run — French SIB-200 *and* German FLORES.
oellm-eval schedule --models "my-model" \
    --task_groups "sib200-eu[fra_Latn],flores-200-eu-to-eng[deu_Latn]"

# No bracket: the group is unchanged — all of its languages.
oellm-eval schedule --models "my-model" --task_groups "sib200-eu"
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
    --n_shot 0 \
    --venv_path .venv \
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

Override cluster defaults (partition, account, time limit, memory, etc.) with `--slurm_template_var` (JSON object). Provide `SLURM_MEM` to request an exact host memory amount, otherwise falls back to a default of `96G`.

```bash
# Use a different partition (e.g. dev-g on LUMI when small-g is crowded)
oellm-eval schedule --models "model-name" --task_groups "open-sci-0.01" \
  --slurm_template_var '{"PARTITION":"dev-g"}'

# Multiple overrides: partition, account, time limit, GPUs, exact RAM
oellm-eval schedule --models "model-name" --task_groups "open-sci-0.01" \
  --slurm_template_var '{"PARTITION":"dev-g","ACCOUNT":"myproject","TIME":"02:00:00","GPUS_PER_NODE":2,"SLURM_MEM":"96G"}'
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
  --task_groups "belebele-eu-cf" \
  --venv_path .venv
```

If you need full manual control over all model args, set `MODEL_ARGS`,
for example:

```bash
MODEL_ARGS='batch_size=8' oellm-eval schedule \
  --models "model-name" --task_groups "belebele-eu-cf" --venv_path .venv
```

## ⚠️ Dataset Pre-Download Warning

**Datasets are only automatically pre-downloaded for tasks defined in [`task-groups.yaml`](oellm/resources/task-groups.yaml).**

If you use custom tasks via `--tasks` that are not in the task groups registry, the CLI will attempt to look them up but **cannot guarantee the datasets will be cached**. This may cause failures on compute nodes that don't have network access.

**Recommendation:** Use `--task_groups` when possible, or ensure your custom task datasets are already cached in `$HF_HOME` before scheduling.

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
oellm-eval collect --results_dir /path/to/eval-output-dir

# Check for missing evaluations and create a CSV for re-running them
oellm-eval collect --results_dir /path/to/eval-output-dir --check true --output_csv results.csv
```

All `jobs.csv` files found under `results_dir` are merged into one; if the same `(model_path, task_path, n_shot)` row appears in multiple files the later-sorted entry wins (override duplicates). The merged jobs list is then compared against all `.json` result files found recursively.

The `--check` flag outputs a `results_missing.csv` that can be used to re-schedule failed jobs:

```bash
oellm-eval schedule --eval_csv_path results_missing.csv
```

## CSV-Based Scheduling

For full control, provide a CSV file with columns: `model_path`, `task_path`, `n_shot`, and optionally `eval_suite`:

```bash
oellm-eval schedule --eval_csv_path custom_evals.csv
```

## Installation

### General Installation

```bash
uv tool install -p 3.12 git+https://github.com/OpenEuroLLM/oellm-eval.git
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
export HF_HOME="$basedir//cache/huggingface"
```

Later, we will add recommendation for a project-wide setting to share tools and models.


## Supported Clusters:
We support: Leonardo, Lumi, Jureca, Jupiter, and Snellius

## CLI Options

```bash
oellm-eval schedule --help
```

## Development

```bash
# Clone and install in dev mode
git clone https://github.com/OpenEuroLLM/oellm-eval.git
cd oellm-eval
uv sync --extra dev

# Run dataset validation tests
uv run pytest tests/test_datasets.py -v

# Download-only mode for testing
uv run oellm-eval schedule --models "EleutherAI/pythia-160m" --task_groups "open-sci-0.01" --download_only
```

## Deploying containers

Containers are deployed manually since [PR #46](https://github.com/OpenEuroLLM/oellm-eval/pull/46) to save costs.

To build and deploy them, select run workflow in [Actions](https://github.com/OpenEuroLLM/oellm-eval/actions/workflows/build-and-push-apptainer.yml).


## Troubleshooting

**HuggingFace quota issues**: Ensure you're logged in with `HF_TOKEN` and are part of the [OpenEuroLLM](https://huggingface.co/OpenEuroLLM) organization.

**Dataset download failures on compute nodes**: Use `--task_groups` for automatic dataset caching, or pre-download datasets manually before scheduling.
