# Using Your Own Virtual Environment

## Overview

Instead of using pre-built containers, you can run evaluations with your own Python virtual environment by passing `--venv_path`.

## Setup

1. Create a venv with Python 3.12:
   ```bash
   uv venv --python 3.12 /path/to/.venv
   ```

2. Install lm-eval dependencies:
   ```bash
   uv pip install --python /path/to/.venv/bin/python -r requirements-venv.txt
   ```

3. Install lighteval as isolated tool (avoids datasets version conflict):
   ```bash
   UV_TOOL_DIR=/path/to/.uv-tools UV_TOOL_BIN_DIR=/path/to/.venv/bin \
     uv tool install --python 3.12 \
       --with "langcodes[data]" --with "pillow" \
       "lighteval[multilingual] @ git+https://github.com/huggingface/lighteval.git"
   ```

## Usage

```bash
oellm schedule-eval \
    --models HuggingFaceTB/SmolLM2-135M-Instruct \
    --task_groups multilingual \
    --venv_path /path/to/.venv
```

## Why Two Install Steps?

lm-eval requires `datasets<4.0.0` while lighteval requires `datasets>=4.0.0`. Installing lighteval as an isolated uv tool (like the containers do) avoids this conflict.

## DCLM-core-22

`dclm-core-22` needs `lm-eval==0.4.9.2` (v0.4.10+ breaks `agieval_lsat_ar` in few-shot). Use `requirements-venv-dclm.txt` instead of the default requirements:

```bash
uv venv --python 3.12 dclm-core-venv
uv pip install --python dclm-core-venv/bin/python -r requirements-venv-dclm.txt
```

```bash
oellm schedule-eval \
    --models Qwen/Qwen3-0.6B-Base \
    --task_groups dclm-core-22 \
    --venv_path dclm-core-venv \
    --skip_checks true
```

## Evalchemy (reasoning)

The `reasoning` task group includes 6 benchmarks: GSM8k, IFEval, and MBPP run via lm-eval-harness, while GPQADiamond, MATH500, and LiveCodeBench run via evalchemy.

> **Note:** The evalchemy versions of GPQA and MATH500 differ from lm-eval-harness. Evalchemy uses free-form generation with CoT reasoning instead of log-likelihood scoring.

We use [Ali's fork](https://github.com/Ali-Elganzory/evalchemy) which includes a [fix to randomize GPQA answer ordering](https://github.com/Ali-Elganzory/evalchemy/tree/fix/randomize-answers-gpqa-diamond) to eliminate positional bias, along with context window safety fixes. The PR is yet to be merged upstream.

1. Clone the repo at the pinned commit:
   ```bash
   git clone https://github.com/Ali-Elganzory/evalchemy.git evalchemy
   cd evalchemy && git checkout 54ac97648230c4c3a22c3a2b93068b5a4e573f8d && cd ..
   ```

2. Create a venv and install dependencies:
   ```bash
   uv venv --python 3.12 evalchemy-venv
   uv pip install --python evalchemy-venv/bin/python -r requirements-venv-evalchemy.txt
   ```

3. Run with `EVALCHEMY_DIR` pointing to the cloned repo:
   ```bash
   export HF_ALLOW_CODE_EVAL=1  # required by MBPP
   EVALCHEMY_DIR=$(pwd)/evalchemy oellm schedule-eval \
       --models HuggingFaceTB/SmolLM2-135M \
       --task_groups reasoning \
       --venv_path evalchemy-venv \
       --skip_checks true
   ```

> **Note:** `HF_ALLOW_CODE_EVAL=1` is required because MBPP (run via lm-eval-harness) uses HuggingFace's `code_eval` metric which executes model-generated code. The evalchemy benchmarks (GPQADiamond, MATH500, LiveCodeBench) do not require this variable as they handle code execution safely through internal guards.
