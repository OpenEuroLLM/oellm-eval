# Using Your Own Virtual Environment

## Overview

Instead of using pre-built containers, you can run evaluations with your own Python virtual environment by passing `--venv_path`.

## Setup

1. Create a venv with Python 3.12:
   ```bash
   uv venv --python 3.12 /path/to/.venv
   ```

2. Install oellm-evals + lm-eval-harness via the `[eval]` extras:
   ```bash
   uv pip install --python /path/to/.venv/bin/python -e .[eval]
   ```

   `[eval]` pulls `lm_eval[hf,vllm,api,tasks]>=0.4.12` and
   `datasets>=4.0` — local HF + vLLM backends, OpenAI-style API
   backends, and the full task family aggregate (ifeval, math,
   multilingual, longbench, …). From a container that already ships
   torch/transformers (Leonardo, Lumi), use `.[eval-base]` instead — it
   drops `[hf]` and `[vllm]` so the container's pre-built torch (and on
   Lumi the custom ROCm-vllm) isn't replaced by PyPI wheels.

3. Install lighteval as isolated tool (keeps its heavier dep tree out of
   the lm-eval env):
   ```bash
   UV_TOOL_DIR=/path/to/.uv-tools UV_TOOL_BIN_DIR=/path/to/.venv/bin \
     uv tool install --python 3.12 \
       --with "langcodes[data]" --with "pillow" \
       "lighteval[multilingual] @ git+https://github.com/huggingface/lighteval.git"
   ```

## Usage

```bash
oellm-eval schedule \
    --models HuggingFaceTB/SmolLM2-135M-Instruct \
    --task_groups multilingual \
    --venv_path /path/to/.venv
```

## Why Two Install Steps?

Historically lm-eval required `datasets<4.0.0` while lighteval required
`datasets>=4.0.0`, and the `lighteval-as-uv-tool` split existed to
resolve that conflict. lm-eval-harness >=0.4.12 (what `[eval]` pulls)
now works with `datasets>=4.0`, so the split is no longer about a
version conflict — it's about keeping lighteval's wider dep tree out of
the lm-eval venv.

## DCLM-core-22

`dclm-core-22` needs `lm-eval==0.4.9.2` (v0.4.10+ breaks `agieval_lsat_ar` in few-shot). Use `requirements-venv-dclm.txt` instead of the default requirements:

```bash
uv venv --python 3.12 dclm-core-venv
uv pip install --python dclm-core-venv/bin/python -r requirements-venv-dclm.txt
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
    --task_groups dclm-core-22 \
    --venv_path dclm-core-venv \
    --skip_checks true
```

## Evalchemy (reasoning)

The `reasoning` task group includes 10 benchmarks: GSM8k, IFEval, and MBPP run via lm-eval-harness, while GPQADiamond, MATH500, LiveCodeBench, HumanEval, AIME24, AIME25, and AMC23 run via evalchemy.

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
   EVALCHEMY_DIR=$(pwd)/evalchemy oellm-eval schedule \
       --models HuggingFaceTB/SmolLM2-135M \
       --task_groups reasoning \
       --venv_path evalchemy-venv \
       --skip_checks true
   ```

> **Note:** `HF_ALLOW_CODE_EVAL=1` is required because MBPP (run via lm-eval-harness) uses HuggingFace's `code_eval` metric which executes model-generated code. The evalchemy benchmarks (GPQADiamond, MATH500, LiveCodeBench) do not require this variable as they handle code execution safely through internal guards.

## LMMs-Eval (image understanding)

The `vqa` task group (VQAv2, GQA, TextVQA, ScienceQA-img) and `mmmu` task
group (MMMU, MMMU-Pro standard) run via
[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval), a separate
harness for vision-language models. Validated for parity against public
numbers: LLaVA-1.5-7B on `vqa` (GQA/TextVQA/ScienceQA-img within a few
points of lmms-eval's own reported LLaVA-1.5-7B numbers) and
Qwen2.5-VL-7B-Instruct on `mmmu`.

> **Note:** Unlike lm-eval-harness/evalchemy, lmms-eval has no single
> architecture-agnostic model wrapper — each VLM family needs its own
> registered model class (`llava_hf`, `qwen2_5_vl`, `qwen2_vl`, ...). The
> `lmms_eval` case in `oellm/resources/template.sbatch` resolves the right
> class automatically from the checkpoint's own `config.json`
> (`model_type`). If you add a model family not yet in that mapping,
> extend the `case` there.

1. Create a venv and install dependencies:
   ```bash
   uv venv --python 3.12 lmms-eval-venv
   uv pip install --python lmms-eval-venv/bin/python -r requirements-venv-lmms-eval.txt
   ```

2. `decord` (a video-loading dependency pulled in unconditionally by
   `llava_hf` and the Qwen VL model files, even for image-only tasks) has
   no Linux-aarch64 wheel. On aarch64 (e.g. JUPITER's GH200 nodes),
   install a stub package that raises only if actually used — image-only
   tasks never call it:
   ```bash
   lmms-eval-venv/bin/python - <<'PY'
   import pathlib, sysconfig
   site = pathlib.Path(sysconfig.get_paths()["purelib"]) / "decord"
   site.mkdir(exist_ok=True)
   (site / "__init__.py").write_text('''"""Minimal stub: the real decord package has no linux-aarch64 wheel.
   Raises only if actually called; image-only lmms-eval tasks never hit this."""

   def cpu(*args, **kwargs):
       raise NotImplementedError("decord stub: video decoding is not available on this platform")


   class VideoReader:
       def __init__(self, *args, **kwargs):
           raise NotImplementedError("decord stub: video decoding is not available on this platform")
   ''')
   print("stubbed decord at:", site)
   PY
   ```

3. Some `vqa`/`mmmu` datasets need an `HF_TOKEN` set even though the
   underlying repos are public — a couple of their loader scripts pass
   `token=True` explicitly, which fails client-side before ever reaching
   the network if no token file is present. Any token value (even an
   expired one) satisfies the client-side check:
   ```bash
   export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null)
   ```

4. The `lmms_eval` case in `template.sbatch` always launches with
   `--batch_size 1` (not user-configurable via the CLI, unlike the
   `lighteval`/evalchemy task groups): the `llava_hf` model class has no
   real batching support (`assert self.batch_size_per_gpu == 1`), and
   `--batch_size auto` (used elsewhere) fails against it with
   `ValueError: invalid literal for int() with base 10: 'auto'`.
   ```bash
   oellm-eval schedule \
       --models llava-hf/llava-1.5-7b-hf \
       --task_groups vqa \
       --venv_path lmms-eval-venv \
       --skip_checks true
   ```
