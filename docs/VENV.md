# Using Your Own Virtual Environment

## Overview

Instead of using pre-built containers, you can run evaluations with your own Python virtual environment by passing `--venv-path`.

## Choosing your venv

Most evaluations share **one general venv**. A handful of framework-level
suites have hard dependency conflicts and need their own venv:

| Task group(s) | Engine | Venv | Setup |
|---|---|---|---|
| `open-sci-*`, `belebele_*_cf`, all text/multilingual tasks | `lm-eval-harness`, `lighteval` | **general** | [Setup](#setup-general-venv) |
| `image-*`, `video-*`, `audio-*` (modality-prefixed) | `lmms-eval` | **general** | [Setup](#setup-general-venv) |
| `dclm-core-22` | `lm-eval-harness` (pinned 0.4.9.2) | **dclm** | [DCLM-core-22](#dclm-core-22) |
| `reasoning` (GPQA/MATH500/AIME/MBPP/etc.) | `evalchemy` + forked lm-eval | **evalchemy** | [Evalchemy](#evalchemy-reasoning) |

Custom contrib benchmarks bring their own dependency stacks and are
documented in `oellm/contrib/<name>/README.md`:

| Task group(s) | Contrib | README |
|---|---|---|
| `audio-audiobench*` | `audiobench` | [`oellm/contrib/audiobench/README.md`](../oellm/contrib/audiobench/README.md) |
| `regiondial-*` | `regiondial_bench` | [`oellm/contrib/regiondial_bench/README.md`](../oellm/contrib/regiondial_bench/README.md) |

Use `oellm-eval list-tasks` to see which suite a given task group routes to,
and `oellm-eval doctor --venv-path <venv> --task-groups <groups>` to verify a
venv against the groups you plan to run.

## Setup (general venv)

```bash
# 1. Create venv
uv venv --python 3.12 /path/to/.venv

# 2. Install lmms-eval editable from git
#    (Editable is required — wheel build drops `_default_template_yaml` files.)
#    Pin a known-good commit (`...lmms-eval.git@<commit>#egg=...`) so two venvs
#    created on different days run the same engine — unpinned `main` drifts.
uv pip install --python /path/to/.venv/bin/python \
    -e "git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git@45c766f60b6f8c153e4c72d06ca636e2db0ebcdb#egg=lmms-eval"

# 3. Install oellm-cli with engine extras
uv pip install --python /path/to/.venv/bin/python -e '.[text,image,audio]'

# 4. Install lighteval as an isolated uv tool (datasets version conflict)
UV_TOOL_DIR=/path/to/.uv-tools UV_TOOL_BIN_DIR=/path/to/.venv/bin \
  uv tool install --python 3.12 \
    --with "langcodes[data]" --with "pillow" \
    "lighteval[multilingual] @ git+https://github.com/huggingface/lighteval.git@64f4f5ae173626509fad6e477ca4ee56ebb26129"
```

Verify with:
```bash
oellm-eval doctor --venv-path /path/to/.venv --task-groups "open-sci-0.01,image-vqa"
```
(or manually: `/path/to/.venv/bin/python -c 'from lmms_eval.tasks import TaskManager; TaskManager("INFO"); print("OK")'`)

## Usage

```bash
# Text evaluation
oellm-eval schedule \
    --models HuggingFaceTB/SmolLM2-135M-Instruct \
    --task-groups open-sci-0.01 \
    --venv-path /path/to/.venv

# Image evaluation (lmms-eval)
oellm-eval schedule \
    --models path/to/vlm \
    --task-groups image-vqa \
    --venv-path /path/to/.venv
```

## Why Multiple Install Steps?


lm-eval requires `datasets<4.0.0` while lighteval requires `datasets>=4.0.0`. Installing lighteval as an isolated uv tool (like the containers do) avoids this conflict. `lmms-eval` is compatible with `datasets<4.0.0` and can be installed alongside lm-eval in the same venv.

## Dependency Summary

| Component | Install | Reason |
|---|---|---|
| `lmms-eval` | `uv pip install -e "git+https://…@<commit>#egg=lmms-eval"` | Editable: wheel drops template files |
| `text`, `image`, `audio` extras | `uv pip install '.[text,image,audio]'` | lm-eval + transformers pin + audio helpers |
| `lighteval` | `uv tool install …` (isolated) | Needs `datasets>=4.0.0`; conflicts with lm-eval |
| `evalchemy` | `uv pip install '.[evalchemy]'` (own venv) | Forked lm-eval |
| `dclm` | `uv pip install '.[dclm]'` (own venv) | Pinned `lm-eval==0.4.9.2` |

## DCLM-core-22

`dclm-core-22` needs `lm-eval==0.4.9.2` (v0.4.10+ breaks `agieval_lsat_ar` in few-shot). Install in its own venv using the `[dclm]` extra:

```bash
uv venv --python 3.12 dclm-core-venv
uv pip install --python dclm-core-venv/bin/python -e '.[dclm]'
```

The `jeopardy` task is a custom task loaded via `--include_path`. lm-eval `0.4.9.2` has a bug where `pretty_print_task` assumes every task YAML lives under `lm_eval/tasks/`, so it crashes on any `--include_path` task (fixed upstream in [lm-evaluation-harness#3436](https://github.com/EleutherAI/lm-evaluation-harness/pull/3436), but we can't upgrade since 0.4.10+ breaks `agieval_lsat_ar`). Apply the one-line fix to the venv after installing:

```bash
dclm-core-venv/bin/python - <<'PY'
import pathlib, lm_eval.tasks as t
f = pathlib.Path(t.__file__); s = f.read_text()
old = "        relative_yaml_path = yaml_path.relative_to(lm_eval_tasks_path)\n"
new = ("        try:\n"
       "            relative_yaml_path = yaml_path.relative_to(lm_eval_tasks_path)\n"
       "        except ValueError:\n"
       "            relative_yaml_path = yaml_path\n")
if "except ValueError" in s:
    print("already patched:", f)
elif old in s:
    f.write_text(s.replace(old, new)); print("patched:", f)
else:
    raise SystemExit(f"target line not found in {f} (unexpected lm-eval version?)")
PY
```

```bash
oellm-eval schedule \
    --models Qwen/Qwen3-0.6B-Base \
    --task-groups dclm-core-22 \
    --venv-path dclm-core-venv \
    --skip-checks   # skips dataset pre-download AND the environment pre-flight — make sure datasets are already cached
```

> Without `--skip-checks`, the scheduler verifies the venv actually contains
> `lm-eval==0.4.9.2` — running this group on any other lm-eval version
> silently changes the scores.

## Evalchemy (reasoning)

The `reasoning` task group includes 10 benchmarks: GSM8k, IFEval, and MBPP run via lm-eval-harness, while GPQADiamond, MATH500, LiveCodeBench, HumanEval, AIME24, AIME25, and AMC23 run via evalchemy.

> **Note:** The evalchemy versions of GPQA and MATH500 differ from lm-eval-harness. Evalchemy uses free-form generation with CoT reasoning instead of log-likelihood scoring.

We use [Ali's fork](https://github.com/Ali-Elganzory/evalchemy) which includes a [fix to randomize GPQA answer ordering](https://github.com/Ali-Elganzory/evalchemy/tree/fix/randomize-answers-gpqa-diamond) to eliminate positional bias, along with context window safety fixes. The PR is yet to be merged upstream.

1. Clone the repo at the pinned commit:
   ```bash
   git clone https://github.com/Ali-Elganzory/evalchemy.git evalchemy
   cd evalchemy && git checkout 54ac97648230c4c3a22c3a2b93068b5a4e573f8d && cd ..
   ```

2. Create a venv and install dependencies using the `[evalchemy]` extra:
   ```bash
   uv venv --python 3.12 evalchemy-venv
   uv pip install --python evalchemy-venv/bin/python -e '.[evalchemy]'
   ```

3. Run with `EVALCHEMY_DIR` pointing to the cloned repo:
   ```bash
   export HF_ALLOW_CODE_EVAL=1  # required by MBPP
   EVALCHEMY_DIR=$(pwd)/evalchemy oellm-eval schedule \
       --models HuggingFaceTB/SmolLM2-135M \
       --task-groups reasoning \
       --venv-path evalchemy-venv \
       --skip-checks
   ```

> **Note:** `HF_ALLOW_CODE_EVAL=1` is required because MBPP (run via lm-eval-harness) uses HuggingFace's `code_eval` metric which executes model-generated code. The evalchemy benchmarks (GPQADiamond, MATH500, LiveCodeBench) do not require this variable as they handle code execution safely through internal guards.
