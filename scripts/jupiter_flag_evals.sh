#!/bin/bash
# FLAG evaluation (task group flag-evals, 436 evals) of HF exports on JUPITER, as two launchers:
# flag-evals-vllm (lm-eval + Evalchemy on vLLM, 363 evals) and flag-evals-lighteval (hf backend, 73).
#
#   scripts/jupiter_flag_evals.sh views    <export dir> ...   # model views, hardlinked
#   scripts/jupiter_flag_evals.sh prefetch                    # login node: every dataset into $FLAG_WORK/hf_home
#   scripts/jupiter_flag_evals.sh render   <export name> ...  # both launchers; prints the sbatch commands
#   scripts/jupiter_flag_evals.sh check                       # which oellm source the tool runs
#   scripts/jupiter_flag_evals.sh tasks    <out.yaml>         # eval rows per half, for oellm-autoexp
#   scripts/jupiter_flag_evals.sh run-one  <name> <half> <row> <task>  # one row, in this allocation
#   oellm-eval collect --results_dir $FLAG_WORK/runs/<export name> --output_csv <name>.csv
#
# Defaults are the e-sta-openeurollm setup; override ACCOUNT, FLAG_WORK (on the exports' filesystem,
# for the hardlinks), VLLM_SIF, LIGHTEVAL_SIF (built from containers/lighteval-jupiter.def), CONCURRENCY, TIME,
# HALVES (render / tasks: "vllm", "lighteval", "cot"; default "vllm lighteval"), HUMANEVAL_PATCH.
# The "cot" half is task group flag-evals-cot (custom_lm_eval_tasks/flag_cot: forced-reasoning _cot
# and continuation _cont variants of the reasoning and code evals), on the vLLM image like "vllm".
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
# The oellm tool (task-group expansion, views, `oellm-eval schedule`) runs from THIS checkout on the
# vLLM image's Python, which ships its dependencies (pandas, jsonargparse, datasets, ...): nothing is
# installed on the cluster, and the suite always matches the checkout. OELLM_TOOL=installed uses an
# `oellm-eval` on PATH instead (e.g. `uv tool install -e .`).
: "${OELLM_TOOL:=image}"
if [ "$OELLM_TOOL" = image ]; then
    # apptainer never passes host SINGULARITY_*/APPTAINER_* variables into a container, but the
    # launcher template needs SINGULARITY_ARGS (without it, it silently renders the cluster
    # default --contain: no --cleanenv, no HumanEval patch bind). Hand it over under another name,
    # via an env file because `--env` splits values at commas.
    tool_py() {
        local envf rc=0; envf=$(mktemp)
        if [ -n "${SINGULARITY_ARGS+set}" ]; then    # single-quoted: apptainer reads env files verbatim
            [[ "$SINGULARITY_ARGS" != *"'"* ]] || { echo "error: SINGULARITY_ARGS must not contain single quotes" >&2; rm -f "$envf"; return 1; }
            printf "OELLM_SINGULARITY_ARGS='%s'\n" "$SINGULARITY_ARGS" > "$envf"
        fi
        apptainer exec --env PYTHONPATH="$REPO" --env PYTHONNOUSERSITE=1 --env-file "$envf" "$VLLM_SIF" python "$@" || rc=$?
        rm -f "$envf"; return $rc
    }
else
    TOOL_PY="$(dirname "$(command -v oellm-eval)")/python"
    tool_py() { "$TOOL_PY" "$@"; }
fi
oellm_eval() {
    tool_py -c 'import os
args = os.environ.pop("OELLM_SINGULARITY_ARGS", None)
if args is not None:
    os.environ["SINGULARITY_ARGS"] = args
from oellm.main import main
main()' "$@"
}
VIEWS="$FLAG_WORK/views"
MODE="${1:?usage: $0 views|prefetch|render|tasks|run-one|check ...}"; shift
if [ "$MODE" != views ] || [ "$OELLM_TOOL" = image ]; then
    for f in "$VLLM_SIF" "$LIGHTEVAL_SIF"; do [ -f "$f" ] || { echo "missing image $f" >&2; exit 1; }; done
fi

expand() {  # <super group> <suite>: task<TAB>n_shot
    tool_py - "$1" "$2" <<'EOF'
import sys
from oellm.task_groups import _expand_task_groups
for r in _expand_task_groups([sys.argv[1]]):
    if r.suite == sys.argv[2]:
        print(f"{r.task}\t{r.n_shot}")
EOF
}

render_setup() {  # sets ARGS, VLLM_EXTRA and SLURM for render_half
    # --containall with an explicit environment; the launcher binds the model, HF_HOME and the task dir.
    ARGS="--nv --cleanenv --containall --no-mount bind-paths,hostfs,cwd,home
          --env HF_HUB_CACHE=$HF_HOME/hub --env HF_DATASETS_OFFLINE=1 --env HF_ALLOW_CODE_EVAL=1
          --env HF_EVALUATE_OFFLINE=1 --env RAY_USAGE_STATS_ENABLED=0 --env OMP_NUM_THREADS=4"
    # Patched HumanEval grader (vLLM image only): an overrunning sample counts as 'timed out'
    # instead of aborting the whole HumanEval task, and the outer deadline is 3 s + 60 s (was
    # + 10 s). Bound only while the image still ships the exact file the patch was made from.
    HE_TARGET=/opt/evalchemy/eval/chat_benchmarks/HumanEval/human_eval/execution.py
    HE_ORIG_MD5=d4dcb1a0e2a1dca44c77ec68bf871366
    : "${HUMANEVAL_PATCH:=$REPO/containers/patches/humaneval_execution.py}"
    VLLM_EXTRA=""
    if [ -f "$HUMANEVAL_PATCH" ]; then
        if [ "$(apptainer exec --cleanenv "$VLLM_SIF" md5sum "$HE_TARGET" 2>/dev/null | cut -c1-32)" = "$HE_ORIG_MD5" ]; then
            VLLM_EXTRA="--bind $HUMANEVAL_PATCH:$HE_TARGET:ro"
        else
            echo "warning: $VLLM_SIF ships a different $HE_TARGET; HumanEval patch NOT applied" >&2
        fi
    else
        echo "warning: $HUMANEVAL_PATCH not found; HumanEval runs with the unpatched grader" >&2
    fi
    SLURM=$(printf '{"ACCOUNT":"%s","PARTITION":"booster","NODES":1,"CPUS_PER_TASK":288,"THREADS_PER_CORE":1,"SLURM_MEM":"400G","TIME":"%s"}' \
        "$ACCOUNT" "${TIME:-04:00:00}")
}

render_half() {  # <half> <model dir> <output dir>: renders one launcher, prints its path
    local half=$1 model=$2 out=$3 sif extra="" opts=()
    if [ "$half" = vllm ] || [ "$half" = cot ]; then
        sif=$VLLM_SIF; extra="$VLLM_EXTRA"
        opts=(--model_backend vllm --data_parallel_size 4 --data_parallel_backend mp
              --model_args dtype=bfloat16,gpu_memory_utilization=0.9,max_num_seqs=32)
    else
        sif=$LIGHTEVAL_SIF
    fi
    # One eval per array task (array size = min(max_array_len, evals)).
    # GPUS_PER_NODE=4: vLLM overrides it with DP x TP; lighteval splits the model over all four.
    EVAL_BASE_DIR="$FLAG_WORK/runs" EVAL_OUTPUT_DIR="$out" QUEUE_LIMIT=1000 GPUS_PER_NODE=4 \
    EVAL_CONTAINER_IMAGE="$sif" SINGULARITY_ARGS="$(echo $ARGS $extra)" \
        oellm_eval schedule --models "$model" --task_groups "flag-evals-$half" "${opts[@]}" \
            --log_samples true --confirm_run_unsafe_code true --max_array_len 1000 \
            --slurm_template_var "$SLURM" --skip_checks true --dry_run true > /dev/null
    ls -t "$out"/*/submit_evals.sbatch | head -1
}

case "$MODE" in
views)
    for src in "$@"; do
        src=$(realpath "$src"); name=$(basename "$src")
        if [ -e "$VIEWS/identity/$name" ]; then
            grep -q "\"source\": \"$src\"" "$VIEWS/identity/$name/prompt-view-manifest.json" \
                || { echo "$VIEWS/identity/$name is a view of another export" >&2; exit 1; }
        else
            tool_py "$REPO/containers/prepare_base_model_view.py" \
                --source "$src" --out "$VIEWS/identity/$name" --template identity
        fi
        dst="$VIEWS/plain/$name"
        [ -e "$dst" ] || { mkdir -p "$VIEWS/plain"; cp -al "$src" "$dst.tmp" && mv "$dst.tmp" "$dst"; }
        echo "$dst"
    done
    ;;
prefetch)
    TASKS_DIR=$(tool_py -c "from importlib.resources import files; print(files('oellm.resources') / 'custom_lm_eval_tasks')")
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
    render_setup
    for name in "$@"; do
        for half in ${HALVES:-vllm lighteval}; do
            if [ $half = lighteval ]; then model=$VIEWS/plain/$name; else model=$VIEWS/identity/$name; fi
            [ -f "$model/config.json" ] || { echo "no view $model; run: $0 views <export dir>" >&2; exit 1; }
            script=$(render_half $half "$model" "$FLAG_WORK/runs/$name/$half")
            sed -i -E "s/^(#SBATCH --array=[0-9]+-[0-9]+)%[0-9]+$/\1%${CONCURRENCY:-20}/" "$script"
            grep -q "^#SBATCH --array=.*%${CONCURRENCY:-20}$" "$script" || { echo "throttle failed: $script" >&2; exit 1; }
            echo "sbatch $script   # $(grep -m1 '^#SBATCH --array' "$script")"
        done
    done
    ;;
tasks)
    # The suite's evals per half, in the launchers' row order, as a Hydra config for
    # oellm-autoexp (one array task per row; `run-one` executes a row). Model-independent:
    # rendered once per oellm-eval revision against a placeholder model.
    out=${1:?usage: $0 tasks <out.yaml>}
    render_setup
    tmp=$(mktemp -d "$FLAG_WORK/tasks.XXXXXX"); trap 'rm -rf "$tmp"' EXIT
    csvs=()
    for half in ${HALVES:-vllm lighteval}; do
        script=$(render_half $half "/oellm-flag-tasks-placeholder/$half" "$tmp/$half")
        csvs+=("$half=$(dirname "$script")/jobs.csv")
    done
    rev=$(git -C "$REPO" rev-parse --short=7 HEAD 2>/dev/null || echo unknown)
    [ -z "$(git -C "$REPO" status --porcelain --untracked-files=no 2>/dev/null)" ] || rev="$rev-dirty"
    python3 - "$rev" "${csvs[@]}" > "$out.tmp" <<'PY'
import csv, json, sys
rev, halves = sys.argv[1], sys.argv[2:]
items = []
for spec in halves:
    half, path = spec.split("=", 1)
    with open(path, newline="") as fh:
        for row, r in enumerate(csv.DictReader(fh)):
            items.append({"name": f"{half}/{r['task_path']}/{int(r['n_shot'])}", "half": half,
                          "row": row, "task": r["task_path"], "n_shot": int(r["n_shot"]),
                          "suite": r["eval_suite"]})
names = [i["name"] for i in items]
dupes = sorted({n for n in names if names.count(n) > 1})
if dupes:
    sys.exit(f"duplicate eval names: {dupes}")
print("# GENERATED by submodules/oellm-eval/scripts/jupiter_flag_evals.sh tasks -- do not edit.")
print("# One item per eval. name = half/task/n_shot (stable: select subsets by it); row = the")
print("# launcher row `run-one <export> <half> <row> <task>` executes (order of this revision).")
print(f"oellm_eval_rev: {json.dumps(rev)}")
print(f"count: {len(items)}")
print("tasks:")
for i in items:
    print("  - {" + ", ".join(f"{k}: {json.dumps(v)}" for k, v in i.items()) + "}")
PY
    mv "$out.tmp" "$out"
    echo "wrote $out (oellm-eval $rev)"
    ;;
run-one)
    # One row of an export's rendered launcher, in the CURRENT allocation (an oellm-autoexp
    # array task). The launcher body is the per-eval logic (container, binds, harness, results
    # dir); its #SBATCH header is inert here. TASK must match the row, so a task list from
    # another oellm-eval revision cannot silently run a different eval. A row that already
    # succeeded is skipped, so resubmitting a whole stage only reruns what is missing.
    name=${1:?usage: $0 run-one <name> <half> <row> <task>} half=${2:?half} row=${3:?row} task=${4:?task}
    script=$(ls -t "$FLAG_WORK/runs/$name/$half"/*/submit_evals.sbatch 2>/dev/null | head -1)
    [ -n "$script" ] || { echo "no rendered $half launcher for $name; run: $0 render $name" >&2; exit 1; }
    # split like the launcher's own `IFS=, read` (header = line 1)
    got=$(awk -F, -v n=$((row + 2)) 'NR == n { print $2 }' "$(dirname "$script")/jobs.csv")
    [ "$got" = "$task" ] || { echo "row $row of $script is '$got', expected '$task'" >&2; exit 1; }
    done_dir="$FLAG_WORK/runs/$name/$half/done"; mkdir -p "$done_dir"
    if [ -f "$done_dir/$row.ok" ]; then echo "row $row ($task) already succeeded: $done_dir/$row.ok"; exit 0; fi
    echo "run-one: $name $half row $row = $task  ($script)"
    # Watchdog for a hang after vLLM's shutdown: the eval then sits idle until the time limit
    # with no result (polymath_it_medium, hellaswag_eu, polymath_fr_medium_cot; 2026-10). A normal
    # run writes its results seconds after the "[shutdown]" lines (LiveCodeBench grading: ~3 min),
    # so RUN_ONE_HANG_MIN minutes without output after them end the eval: no done marker, a
    # nonzero exit, and the array task's FAILED policy retries it. The eval runs in its own process
    # group; its output goes to a node-local file that is mirrored into the job log.
    hang_min=${RUN_ONE_HANG_MIN:-15}
    log=$(mktemp "${TMPDIR:-/tmp}/run-one.XXXXXX.log")
    SLURM_ARRAY_TASK_ID=$row setsid bash "$script" > "$log" 2>&1 < /dev/null &
    pid=$!
    tail -n +1 -f --pid="$pid" "$log" &
    mirror=$!
    hung=0
    tick=0
    while sleep 5 && kill -0 "$pid" 2>/dev/null; do
        tick=$((tick + 1))
        [ $((tick % 6)) = 0 ] || continue  # liveness every 5 s, the hang test every 30 s
        if grep -q '\[shutdown\]' "$log" && [ $(( $(date +%s) - $(stat -c %Y "$log") )) -gt $((hang_min * 60)) ]; then
            hung=1
            echo "run-one watchdog: no output for ${hang_min} min after vLLM's shutdown -- ending the eval" >&2
            kill -TERM -- "-$pid" 2>/dev/null || true
            sleep 30
            kill -KILL -- "-$pid" 2>/dev/null || true
            break
        fi
    done
    rc=0
    wait "$pid" || rc=$?
    wait "$mirror" 2>/dev/null || true
    rm -f "$log"
    [ "$hung" = 0 ] || exit 124
    [ "$rc" = 0 ] || exit "$rc"
    printf '%s %s %s\n' "$(date -Is)" "${SLURM_JOB_ID:-local}" "$task" > "$done_dir/$row.ok"
    ;;
check)
    # Where the oellm package in use comes from (callers verify it is this checkout).
    tool_py -c 'import oellm, os; print(os.path.dirname(os.path.realpath(oellm.__file__)))'
    ;;
*) echo "unknown mode: $MODE" >&2; exit 2 ;;
esac
