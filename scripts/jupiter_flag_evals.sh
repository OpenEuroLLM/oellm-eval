#!/bin/bash
# FLAG evaluation (task group flag-evals, 436 evals) of HF exports on JUPITER, as two launchers:
# flag-evals-vllm (lm-eval + Evalchemy on vLLM, 363 evals) and flag-evals-lighteval (hf backend, 73).
#
#   scripts/jupiter_flag_evals.sh views    <export dir> ...   # model views, hardlinked
#   scripts/jupiter_flag_evals.sh prefetch                    # login node: every dataset into $FLAG_WORK/hf_home
#   scripts/jupiter_flag_evals.sh render   <export name> ...  # both launchers; prints the sbatch commands
#   oellm-eval collect --results_dir $FLAG_WORK/runs/<export name> --output_csv <name>.csv
#
# Defaults are the e-sta-openeurollm setup; override ACCOUNT, FLAG_WORK (on the exports' filesystem,
# for the hardlinks), VLLM_SIF, LIGHTEVAL_SIF (built from containers/lighteval-jupiter.def), CONCURRENCY, TIME.
# Views: identity chat template for vLLM (Evalchemy applies it, lm-eval does not); none for lighteval,
# which applies any template it finds and caches samples in the model directory.
# Prefetch: compute nodes are offline, so each harness fetches its datasets inside its own image
# (JUPITER binds /e, /tmp and $HOME into containers by default).
# facebook/flores, Helsinki-NLP/OpenSubtitles2024-40-langs-15-movies and Idavidrein/gpqa are gated:
# request access and `hf auth login` first.
set -euo pipefail

: "${ACCOUNT:=e-ext-2025e02-108}" "${FLAG_WORK:=/e/scratch/e-sta-openeurollm/$USER/flag-evals}"
: "${VLLM_SIF:=/e/project1/e-sta-openeurollm/container/oellm-eval-vllm.sif}"
: "${LIGHTEVAL_SIF:=/e/project1/e-sta-openeurollm/container/oellm-eval-lighteval.sif}"
TOKEN="${HF_TOKEN_PATH:-${HF_HOME:-$HOME/.cache/huggingface}/token}"   # where `hf auth login` wrote it
# The launcher template is filled from this environment, so pin what the jobs must see.
export HF_HOME="$FLAG_WORK/hf_home" HF_DATASETS_CACHE="$FLAG_WORK/hf_home/datasets" NLTK_DATA=/opt/nltk_data
unset LM_EVAL_INCLUDE_PATH HF_HUB_OFFLINE
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TOOL_PY="$(dirname "$(command -v oellm-eval)")/python"
VIEWS="$FLAG_WORK/views"
MODE="${1:?usage: $0 views|prefetch|render ...}"; shift
if [ "$MODE" != views ]; then
    for f in "$VLLM_SIF" "$LIGHTEVAL_SIF"; do [ -f "$f" ] || { echo "missing image $f" >&2; exit 1; }; done
fi

expand() {  # <super group> <suite>: task<TAB>n_shot
    "$TOOL_PY" - "$1" "$2" <<'EOF'
import sys
from oellm.task_groups import _expand_task_groups
for r in _expand_task_groups([sys.argv[1]]):
    if r.suite == sys.argv[2]:
        print(f"{r.task}\t{r.n_shot}")
EOF
}

case "$MODE" in
views)
    for src in "$@"; do
        src=$(realpath "$src"); name=$(basename "$src")
        if [ -e "$VIEWS/identity/$name" ]; then
            grep -q "\"source\": \"$src\"" "$VIEWS/identity/$name/prompt-view-manifest.json" \
                || { echo "$VIEWS/identity/$name is a view of another export" >&2; exit 1; }
        else
            "$TOOL_PY" "$REPO/containers/prepare_base_model_view.py" \
                --source "$src" --out "$VIEWS/identity/$name" --template identity
        fi
        dst="$VIEWS/plain/$name"
        [ -e "$dst" ] || { mkdir -p "$VIEWS/plain"; cp -al "$src" "$dst.tmp" && mv "$dst.tmp" "$dst"; }
        echo "$dst"
    done
    ;;
prefetch)
    TASKS_DIR=$("$TOOL_PY" -c "from importlib.resources import files; print(files('oellm.resources') / 'custom_lm_eval_tasks')")
    mkdir -p "$HF_HOME"
    ENVS=(--cleanenv --env HF_HOME="$HF_HOME" --env HF_HUB_CACHE="$HF_HOME/hub" --env HF_ALLOW_CODE_EVAL=1
          --env HF_DATASETS_OFFLINE=0 --env HF_HUB_OFFLINE=0 --env HF_TOKEN_PATH="$TOKEN")
    list=$(mktemp); trap 'rm -f "$list"' EXIT
    expand flag-evals-vllm lm-eval-harness > "$list"
    apptainer exec "${ENVS[@]}" "$VLLM_SIF" \
        python - "$list" "$TASKS_DIR" <<'EOF'
import logging, os, sys
logging.disable(logging.WARNING)
import evaluate
from datasets import load_dataset
from lm_eval.tasks import TaskManager, get_task_dict
tm = TaskManager(include_path=sys.argv[2])
bad = []
for task in [line.split("\t")[0] for line in open(sys.argv[1]) if line.strip()]:
    try:
        get_task_dict([task], tm)                   # downloads exactly the dataset the task reads
    except Exception as e:
        bad.append(task); print(f"FAIL {task}: {type(e).__name__}: {str(e)[:160]}", flush=True)
evaluate.load("squad_v2")                           # squadv2 scores with it, offline in the job
# Evalchemy: the loaders of GPQADiamond, JEEBench and LiveCodeBench, with their own arguments
# (the other Evalchemy benchmarks read data bundled in the image)
hub = os.environ["HF_HUB_CACHE"]
load_dataset("Idavidrein/gpqa", "gpqa_diamond", cache_dir=hub)
load_dataset("daman1209arora/jeebench", split="test", cache_dir=hub)
load_dataset("livecodebench/code_generation_lite", name="release_latest", version_tag="release_v2",
             split="test", trust_remote_code=True, cache_dir=hub)
print(f"lm-eval: {len(bad)} failed" + (f": {bad}" if bad else ""))
sys.exit(1 if bad else 0)
EOF
    expand flag-evals-lighteval lighteval > "$list"
    apptainer exec "${ENVS[@]}" "$LIGHTEVAL_SIF" \
        python - "$list" <<'EOF'
import logging, sys
logging.disable(logging.WARNING)
from lighteval.tasks.lighteval_task import LightevalTask
from lighteval.tasks.registry import Registry
rows = [line.rstrip("\n").split("\t") for line in open(sys.argv[1]) if line.strip()]
tasks = Registry(tasks=",".join(f"{t}|{n}" for t, n in rows), load_multilingual=True).load_tasks()
bad = []
for key, task in tasks.items():
    try:
        LightevalTask.download_dataset_worker(task)  # lighteval's own loader and cache key
    except Exception as e:
        bad.append(key); print(f"FAIL {key}: {type(e).__name__}: {str(e)[:160]}", flush=True)
print(f"lighteval: {len(tasks) - len(bad)}/{len(tasks)} fetched")
sys.exit(1 if bad else 0)
EOF
    ;;
render)
    # --containall with an explicit environment; the launcher binds the model, HF_HOME and the task dir.
    ARGS="--nv --cleanenv --containall --no-mount bind-paths,hostfs,cwd,home
          --env HF_HUB_CACHE=$HF_HOME/hub --env HF_DATASETS_OFFLINE=1 --env HF_ALLOW_CODE_EVAL=1
          --env HF_EVALUATE_OFFLINE=1 --env RAY_USAGE_STATS_ENABLED=0 --env OMP_NUM_THREADS=4"
    SLURM=$(printf '{"ACCOUNT":"%s","PARTITION":"booster","NODES":1,"CPUS_PER_TASK":288,"THREADS_PER_CORE":1,"SLURM_MEM":"400G","TIME":"%s"}' \
        "$ACCOUNT" "${TIME:-04:00:00}")
    for name in "$@"; do
        for half in vllm lighteval; do
            if [ $half = vllm ]; then
                sif=$VLLM_SIF; model=$VIEWS/identity/$name
                opts=(--model_backend vllm --data_parallel_size 4 --data_parallel_backend mp
                      --model_args dtype=bfloat16,gpu_memory_utilization=0.9,max_num_seqs=32)
            else
                sif=$LIGHTEVAL_SIF; model=$VIEWS/plain/$name; opts=()
            fi
            [ -f "$model/config.json" ] || { echo "no view $model; run: $0 views <export dir>" >&2; exit 1; }
            out="$FLAG_WORK/runs/$name/$half"
            # One eval per array task (array size = min(max_array_len, evals)); its %N is set below.
            # GPUS_PER_NODE=4: vLLM overrides it with DP x TP; lighteval splits the model over all four.
            EVAL_BASE_DIR="$FLAG_WORK/runs" EVAL_OUTPUT_DIR="$out" QUEUE_LIMIT=1000 GPUS_PER_NODE=4 \
            EVAL_CONTAINER_IMAGE="$sif" SINGULARITY_ARGS="$(echo $ARGS)" \
                oellm-eval schedule --models "$model" --task_groups "flag-evals-$half" "${opts[@]}" \
                    --log_samples true --confirm_run_unsafe_code true --max_array_len 1000 \
                    --slurm_template_var "$SLURM" --skip_checks true --dry_run true > /dev/null
            script=$(ls -t "$out"/*/submit_evals.sbatch | head -1)
            sed -i -E "s/^(#SBATCH --array=[0-9]+-[0-9]+)%[0-9]+$/\1%${CONCURRENCY:-20}/" "$script"
            grep -q "^#SBATCH --array=.*%${CONCURRENCY:-20}$" "$script" || { echo "throttle failed: $script" >&2; exit 1; }
            echo "sbatch $script   # $(grep -m1 '^#SBATCH --array' "$script")"
        done
    done
    ;;
*) echo "unknown mode: $MODE" >&2; exit 2 ;;
esac
