#!/bin/bash
#SBATCH --job-name=oellm-eval
#SBATCH --time={time_limit}
#SBATCH --gres=gpu:$GPUS_PER_NODE
#SBATCH --mem={slurm_mem}
#SBATCH --output={log_dir}/%x-%A-%a.out
#SBATCH --partition=$PARTITION
#SBATCH --account=$ACCOUNT
#SBATCH --error={log_dir}/%x-%A-%a.err
#SBATCH --array=0-{array_limit}%{max_array_len}
{launcher_resources}#SBATCH --nodes=$NODES
#SBATCH --nodelist=$NODELIST

set -eo pipefail

CSV_PATH="{csv_path}"
NUM_JOBS={num_jobs}
TOTAL_EVALS={total_evals}
LIMIT="{limit}"
VENV_PATH="{venv_path}"
LM_EVAL_INCLUDE_PATH="{lm_eval_include_path}"
LIGHTEVAL_CUSTOM_TASKS="{lighteval_custom_tasks}"
TASKS_PER_JOB={tasks_per_job}
JUDGEARENA_ARGS=({judgearena_args})
export JUDGEARENA_DATA={judgearena_data}
MODEL_BACKEND="{model_backend}"
IFEVAL_STOPPING_POLICY="{ifeval_stopping_policy}"
EVAL_VERSION="{eval_version}"
PROTOCOL_ROOT={protocol_root}
CALIBRATED_OUTPUT_TOKENS="{calibrated_output_tokens}"
BACKEND_MODEL_ARGS={backend_model_args}
EVALCHEMY_WORK_DIR="{evalchemy_dir}"

if [ "$MODEL_BACKEND" = vllm ]; then
    # The adapter owns its replicas; never nest it in Accelerate/torchrun.
    if [ "${{WORLD_SIZE:-1}}" -gt 1 ] || [ "${{LMEVAL_DP:-1}}" -gt 1 ]; then
        echo "[error] vLLM must run in one launcher process; unset LMEVAL_DP/WORLD_SIZE." >&2
        exit 1
    fi
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
fi

# avoiding crashes due to compute nodes not having access to the internet
export HF_HOME=$HF_HOME
export HF_DATASETS_CACHE="$HF_HOME/datasets"
# evaluate metrics (e.g. MBPP's code_eval) cache to a fixed file name, so array tasks sharing
# $HF_HOME/metrics crash each other (FileNotFoundError). Give every array task its own.
export HF_METRICS_CACHE="$HF_HOME/metrics/job_${{SLURM_JOB_ID:-local}}"
export HF_HUB_OFFLINE={hf_hub_offline}

# Neither harness caches under HF_HOME on its own. lighteval hardcodes
# ~/.cache/huggingface/lighteval and never reads HF_HOME, which fills the user's
# home directory and fails mid-run on clusters with a small home quota
# ("OSError: [Errno 122] Disk quota exceeded"). lm-eval keeps request caches
# inside its own install directory, which is read-only in the container. Point
# both at the shared cache instead.
export LIGHTEVAL_CACHE_DIR="$HF_HOME/lighteval"
export LM_HARNESS_CACHE_PATH="$HF_HOME/lm_eval"
mkdir -p "$LIGHTEVAL_CACHE_DIR" "$LM_HARNESS_CACHE_PATH"

# lighteval caches unconditionally, so it always uses LIGHTEVAL_CACHE_DIR.
# lm-eval only caches with --cache_requests, which stays opt-in: set
# CACHE_REQUESTS=true to enable it, or =refresh to rebuild entries after editing
# a task YAML (the cache key covers task, shots and tokenizer, but not the task
# definition). Leaving it off keeps --limit runs fast -- lm-eval ignores --limit
# while building contexts whenever caching is on.

# Path to the shared Singularity image that contains all runtime deps (container mode)
export EVAL_SIF_PATH="$EVAL_CONTAINER_IMAGE"
if [[ "$EVAL_SIF_PATH" != /* ]]; then
    export EVAL_SIF_PATH="$EVAL_BASE_DIR/$EVAL_SIF_PATH"
fi

export MIOPEN_USER_DB_PATH="/tmp/$USER-miopen-${{SLURM_JOB_ID:-$$}}"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"

echo "Running eval on $CSV_PATH with $NUM_JOBS array jobs distributing $TOTAL_EVALS evaluations"
echo "hostname: "`hostname`
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"

if [ "$TOTAL_EVALS" -lt 1 ]; then
    echo "No evaluations to run. Exiting."
    exit 0
fi

# Calculate how many evaluations each array job should handle
EVALS_PER_JOB=$(((TOTAL_EVALS + NUM_JOBS - 1) / NUM_JOBS))

START_INDEX=$((SLURM_ARRAY_TASK_ID * EVALS_PER_JOB + 1))
END_INDEX=$(((SLURM_ARRAY_TASK_ID + 1) * EVALS_PER_JOB))

if [ "$END_INDEX" -gt "$TOTAL_EVALS" ]; then
    END_INDEX=$TOTAL_EVALS
fi

echo "This job (array task $SLURM_ARRAY_TASK_ID) will process evaluations from index $START_INDEX to $END_INDEX."
echo "Each array job handles approximately $EVALS_PER_JOB evaluations."

if [ "$START_INDEX" -gt "$END_INDEX" ]; then
    echo "No evaluations for this job to run. Exiting."
    exit 0
fi

# Evaluations in this job that produced no result. Becomes the job's exit
# status so a failed eval cannot be mistaken for a completed one.
FAILED_EVALS=0

# Use `tail` and `head` to slice the CSV file for the tasks assigned to this job.
# The +1 on START_INDEX accounts for the header row.
# Collapse adjacent rows that share a model, shot count and lm_eval suite into
# a single row whose task field is a comma-separated list. lm_eval loads the
# model once per invocation, so evaluating N tasks in one call replaces N model
# loads with one. TASKS_PER_JOB caps how many tasks share a call: lm_eval aborts
# the whole invocation if one task raises, so the cap bounds what a single
# failure takes down. Suites other than lm_eval pass through one row at a time.
# A failed evaluation does not skip the remaining ones of this array task.
# The output is tab-separated because a joined task list contains commas.
BATCH_ROWS_AWK='
    function flush() {{ if (tasks != "") print model, tasks, shots, suite }}
    {{
        gsub(/\r/, "")
        if ($1 == "") next
        row_suite = ($4 == "" ? "lm_eval" : $4)
        lowered = tolower(row_suite)
        batchable = (lowered == "lm_eval" || lowered == "lm-eval" || lowered == "lm-eval-harness")
        row_key = batchable ? ($1 SUBSEP $3 SUBSEP lowered) : ("unbatched" SUBSEP NR)
        if (row_key != key || count >= max_tasks) {{
            flush()
            key = row_key; model = $1; shots = $3; suite = row_suite; tasks = $2; count = 1
        }} else {{
            tasks = tasks "," $2; count++
        }}
    }}
    END {{ flush() }}
'

# Redirected rather than piped into the loop: a pipeline runs the loop body in a
# subshell, and FAILED_EVALS would be discarded when it exits.
while IFS=$'\t' read -r model_path task_path n_shot eval_suite
do
    # Remove trailing carriage returns if script is edited on Windows
    model_path=$(echo "$model_path" | tr -d '\r')
    task_path=$(echo "$task_path" | tr -d '\r')
    n_shot=$(echo "$n_shot" | tr -d '\r')
    eval_suite=$(echo "${{eval_suite:-lm_eval}}" | tr -d '\r')

    # Skip empty lines
    if [ -z "$model_path" ]; then
        continue
    fi

    echo "----------------------------------------------------"
    echo "Starting evaluation for:"
    echo "  Model: $model_path"
    echo "  Task: $task_path"
    echo "  N-shot: $n_shot"
    echo "  Suite: $eval_suite"
    echo "----------------------------------------------------"

    # Build bind paths: always mount the shared eval directory, and additionally
    # bind the directory that contains the model checkpoint when the path exists
    BIND_PATHS="$EVAL_BASE_DIR:$EVAL_BASE_DIR,$HF_HOME:$HF_HOME,$HF_DATASETS_CACHE:$HF_DATASETS_CACHE${{LM_EVAL_INCLUDE_PATH:+,$$LM_EVAL_INCLUDE_PATH:$$LM_EVAL_INCLUDE_PATH}}${{LIGHTEVAL_CUSTOM_TASKS:+,$$LIGHTEVAL_CUSTOM_TASKS:$$LIGHTEVAL_CUSTOM_TASKS}}"

    if [[ "$EVAL_VERSION" == v0.02 ]]; then
        BIND_PATHS="$BIND_PATHS,$PROTOCOL_ROOT:$PROTOCOL_ROOT:ro"
    fi
    if [ -e "$model_path" ]; then
        # If the model_path is a file, bind its parent directory; otherwise bind the dir itself
        MODEL_DIR="$model_path"
        if [ ! -d "$MODEL_DIR" ]; then
            MODEL_DIR="$(dirname "$MODEL_DIR")"
        fi
        # Avoid adding duplicate bind if it is already covered by EVAL_BASE_DIR
        if [[ $MODEL_DIR != $EVAL_BASE_DIR* ]]; then
            BIND_PATHS="$BIND_PATHS,$MODEL_DIR:$MODEL_DIR"
        fi
    fi

    # When using --contain, provide a writable home for prediction caches
    SINGULARITY_HOME_ARG=""
    SINGULARITY_ENV_ARGS=""
    JOB_TMP=""
    if [[ "$SINGULARITY_ARGS" == *"--contain"* ]]; then
        # Contained /tmp may otherwise be a tiny session tmpfs (16 MiB on JUPITER).
        # Mount a private host directory at a short container path for multiprocessing sockets.
        JOB_TMP=$(mktemp -d "${{OELLM_EVAL_TMP_ROOT:-/tmp}}/oellm-eval-${{SLURM_JOB_ID:-local}}.XXXXXX")
        trap 'rm -rf -- "$JOB_TMP"' EXIT ERR
        # The home holds only compiler/autotune caches (triton, vLLM torch.compile, flashinfer,
        # numba, tilelang, CUDA ComputeCache), rebuilt by every task anyway: node-local, inside
        # JOB_TMP. On the shared filesystem it left a directory tree per task behind (177 GB,
        # 924k files). Mounted at /tmp/home, which is the same directory whether or not the
        # /tmp bind below shadows it -- a home at its host path under /tmp is hidden by it.
        JOB_HOME="$JOB_TMP/home"
        mkdir -p "$JOB_HOME"
        SINGULARITY_HOME_ARG="--home $JOB_HOME:/tmp/home"
        BIND_PATHS="$BIND_PATHS,$JOB_TMP:/tmp,$JOB_TMP:/var/tmp"
        echo "Evaluation temporary storage: $JOB_TMP -> /tmp and /var/tmp"
        df -Pk "$JOB_TMP"
        SINGULARITY_ENV_ARGS="--env TMPDIR=/tmp --env TMP=/tmp --env TEMP=/tmp --env HF_HOME=$HF_HOME --env HF_DATASETS_CACHE=$HF_DATASETS_CACHE --env HF_HUB_OFFLINE=$HF_HUB_OFFLINE --env TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE --env NLTK_DATA=$NLTK_DATA --env HF_METRICS_CACHE=$${{HF_METRICS_CACHE}} --env LM_HARNESS_CACHE_PATH=$LM_HARNESS_CACHE_PATH"
    fi

    GPU_DEVICES=$(seq -s, 0 $(($GPUS_PER_NODE - 1)))
    # Respect scheduler GPU isolation when already set (including UUIDs).
    export CUDA_VISIBLE_DEVICES="${{CUDA_VISIBLE_DEVICES:-$GPU_DEVICES}}"
    RESOLVED_MODEL_ARGS="pretrained=$model_path,trust_remote_code=True${{BACKEND_MODEL_ARGS:+,$$BACKEND_MODEL_ARGS}}"

    MULTI_GPU_FLAG=""
    if [ "$GPUS_PER_NODE" -gt 1 ]; then
        MULTI_GPU_FLAG="--multi-gpu"
    fi

    suite_normalized=$(echo "$eval_suite" | tr '[:upper:]' '[:lower:]')

    # Helper function to run Python commands in the appropriate environment
    run_python() {{
        if [ -n "$VENV_PATH" ]; then
            source "$VENV_PATH/bin/activate"
            python "$@"
        else
            CONTAINER_CWD_ARGS=()
            if [ -n "${{EVAL_WORKDIR:-}}" ]; then
                CONTAINER_CWD_ARGS=(--pwd "$EVAL_WORKDIR")
            fi
            singularity exec $SINGULARITY_ARGS \
                $SINGULARITY_ENV_ARGS \
                $SINGULARITY_HOME_ARG \
                --bind "$BIND_PATHS" \
                "${{CONTAINER_CWD_ARGS[@]}}" \
                $EVAL_SIF_PATH \
                env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
                python "$@"
        fi
    }}

    # Set by every branch below: the harness exit status, and where to look for
    # what it wrote. lm_eval appends a timestamp to --output_path, so the result
    # is a directory plus a filename pattern rather than an exact path.
    EVAL_STATUS=0
    EVAL_OUT_DIR=""
    EVAL_OUT_PAT=""

    # A failure ends only this evaluation (errexit is off for it), and stdin is
    # closed (`esac < /dev/null`) so nothing it starts can consume the remaining rows.
    set +e
    case "$suite_normalized" in
        lm_eval|lm-eval|lm-eval-harness)
            echo
            echo "----------------------------------------------------"
            echo "lm_eval Execution"
            RUN_ID=$(openssl rand -hex 5)
            EVAL_OUT_DIR="{evals_dir}"
            EVAL_OUT_PAT="$RUN_ID*.json"
            HARNESS_MODULE=lm_eval
            if [[ "$task_path" == ifeval ]]; then
                echo "IFEval stopping policy: $IFEVAL_STOPPING_POLICY"
                if [[ "$IFEVAL_STOPPING_POLICY" == continue ]]; then
                    HARNESS_MODULE=oellm_ifeval
                fi
            fi
            HARNESS_ENTRY=(-m "$HARNESS_MODULE")
            if [[ "$EVAL_VERSION" == v0.02 ]]; then
                HARNESS_ENTRY=("$PROTOCOL_ROOT/protocol_runtime.py" --module "$HARNESS_MODULE")
            fi
            run_python "${{HARNESS_ENTRY[@]}}" --model "$MODEL_BACKEND" \
                --model_args "$RESOLVED_MODEL_ARGS" \
                --tasks "$task_path" \
                --num_fewshot "$n_shot" \
                --output_path "$EVAL_OUT_DIR/$RUN_ID.json" \
                --trust_remote_code \
                ${{LM_EVAL_INCLUDE_PATH:+--include_path $$LM_EVAL_INCLUDE_PATH}} \
                --batch_size auto \
                {log_samples} \
                {confirm_run_unsafe_code} \
                ${{CACHE_REQUESTS:+--cache_requests $CACHE_REQUESTS}} \
                ${{LIMIT:+--limit $LIMIT}}
            EVAL_STATUS=$?
            echo "----------------------------------------------------"
            ;;
        lighteval|light-eval)
            LIGHT_TASK="$task_path"

            if [[ -f "$LIGHT_TASK" ]]; then
                LIGHT_TASK_ARG="$LIGHT_TASK"
            else
                last_segment="${{LIGHT_TASK##*|}}"
                if [[ "$LIGHT_TASK" == *"|"* && "$last_segment" =~ ^[0-9]+$ ]]; then
                    if [[ -n "$n_shot" && "$last_segment" != "$n_shot" ]]; then
                        LIGHT_TASK_ARG="${{LIGHT_TASK%|*}}|$n_shot"
                    else
                        LIGHT_TASK_ARG="$LIGHT_TASK"
                    fi
                else
                    LIGHT_TASK_ARG="${{LIGHT_TASK}}|$n_shot"
                fi
            fi

            RESULTS_SUBDIR="{evals_dir}/$(openssl rand -hex 5)"
            mkdir -p "$RESULTS_SUBDIR"
            EVAL_OUT_DIR="$RESULTS_SUBDIR"
            EVAL_OUT_PAT="*"

            if [ -n "$VENV_PATH" ]; then
                source "$VENV_PATH/bin/activate"
                lighteval accelerate \
                    "model_name=$model_path,trust_remote_code=True,cache_dir=$LIGHTEVAL_CACHE_DIR,{additional_model_args}" \
                    "$LIGHT_TASK_ARG" \
                    --load-tasks-multilingual \
                    --custom-tasks "$$LIGHTEVAL_CUSTOM_TASKS" \
                    --output-dir "$RESULTS_SUBDIR" \
                    ${{LIMIT:+--max-samples $LIMIT}}
                EVAL_STATUS=$?
            else
                singularity exec $SINGULARITY_ARGS \
                    $SINGULARITY_ENV_ARGS \
                    $SINGULARITY_HOME_ARG \
                    --bind "$BIND_PATHS" \
                    $EVAL_SIF_PATH \
                    env CUDA_VISIBLE_DEVICES=$GPU_DEVICES \
                    lighteval accelerate \
                        "model_name=$model_path,cache_dir=$LIGHTEVAL_CACHE_DIR,{additional_model_args}" \
                        "$LIGHT_TASK_ARG" \
                        --load-tasks-multilingual \
                        --custom-tasks "$$LIGHTEVAL_CUSTOM_TASKS" \
                        --output-dir "$RESULTS_SUBDIR" \
                        ${{LIMIT:+--max-samples $LIMIT}}
                EVAL_STATUS=$?
            fi
            ;;
        evalchemy)
            RESULTS_SUBDIR="{evals_dir}/$(openssl rand -hex 5)"
            EVAL_OUT_DIR="$RESULTS_SUBDIR"
            EVAL_OUT_PAT="*"
            if [ "$MODEL_BACKEND" = vllm ]; then
                # A container may carry Evalchemy internally at /opt/evalchemy.
                # A host checkout remains supported via an explicit bind.
                if [ -d "$EVALCHEMY_WORK_DIR" ]; then
                    BIND_PATHS="$BIND_PATHS,$EVALCHEMY_WORK_DIR:$EVALCHEMY_WORK_DIR"
                fi
                (
                    if [ -n "$VENV_PATH" ]; then
                        cd "$EVALCHEMY_WORK_DIR"
                    else
                        EVAL_WORKDIR="$EVALCHEMY_WORK_DIR"
                    fi
                    EVALCHEMY_ENTRY=(-m eval.eval)
                    if [[ "$EVAL_VERSION" == v0.02 ]]; then
                        EVALCHEMY_ENTRY=("$PROTOCOL_ROOT/protocol_runtime.py" --module eval.eval)
                        if [[ -n "$CALIBRATED_OUTPUT_TOKENS" && ( "$task_path" == HumanEval || "$task_path" == MATH500 ) ]]; then
                            EVALCHEMY_ENTRY+=(--output-tokens "$CALIBRATED_OUTPUT_TOKENS")
                        fi
                    fi
                    run_python "${{EVALCHEMY_ENTRY[@]}}" \
                        --model vllm \
                        --tasks "$task_path" \
                        --model_args "$RESOLVED_MODEL_ARGS" \
                        --batch_size auto \
                        --output_path "$RESULTS_SUBDIR" \
                        ${{LIMIT:+--limit $LIMIT}}
                )
                EVAL_STATUS=$?
            elif [ -n "$VENV_PATH" ]; then
                source "$VENV_PATH/bin/activate"
                # subshell: the loop no longer runs in one, so this cd would
                # otherwise leak into the next evaluation in this job
                (
                cd "$EVALCHEMY_WORK_DIR"
                accelerate launch --num-processes "$GPUS_PER_NODE" --num-machines 1 \
                    $MULTI_GPU_FLAG -m eval.eval \
                    --model hf \
                    --tasks "$task_path" \
                    --model_args "$RESOLVED_MODEL_ARGS" \
                    --batch_size auto \
                    --output_path "$RESULTS_SUBDIR" \
                    ${{LIMIT:+--limit $LIMIT}}
                )
                EVAL_STATUS=$?
            else
                # The complete image also carries the HF adapter and Accelerate.
                # Preserve the HF process-per-GPU recipe inside that image.
                # --containall with disabled site binds may omit localhost;
                # Accelerate checks its rendezvous port through that hostname.
                BIND_PATHS="$BIND_PATHS,/etc/hosts:/etc/hosts:ro"
                if [ -d "$EVALCHEMY_WORK_DIR" ]; then
                    BIND_PATHS="$BIND_PATHS,$EVALCHEMY_WORK_DIR:$EVALCHEMY_WORK_DIR"
                fi
                (
                    EVAL_WORKDIR="$EVALCHEMY_WORK_DIR"
                    run_python -m accelerate.commands.launch --num_processes "$GPUS_PER_NODE" --num_machines 1 \
                        $MULTI_GPU_FLAG -m eval.eval \
                        --model hf --tasks "$task_path" \
                        --model_args "$RESOLVED_MODEL_ARGS" --batch_size auto \
                        --output_path "$RESULTS_SUBDIR" \
                        ${{LIMIT:+--limit $LIMIT}}
                )
                EVAL_STATUS=$?
            fi
            ;;
        judgearena)
            mkdir -p "$JUDGEARENA_DATA"
            RESULTS_SUBDIR="{evals_dir}/$(openssl rand -hex 5)"
            mkdir -p "$RESULTS_SUBDIR"
            EVAL_OUT_DIR="$RESULTS_SUBDIR"
            EVAL_OUT_PAT="run-metadata.v1.json"

            JUDGEARENA_BIND_PATHS="$BIND_PATHS,$JUDGEARENA_DATA:$JUDGEARENA_DATA"

            if [ -n "$VENV_PATH" ]; then
                source "$VENV_PATH/bin/activate"
                judgearena \
                    "${{JUDGEARENA_ARGS[@]}}" \
                    --task "$task_path" \
                    --model.name "VLLM/$model_path" \
                    --run.result_folder "$RESULTS_SUBDIR" \
                    ${{LIMIT:+--generation.n_instructions $LIMIT}}
                EVAL_STATUS=$?
            else
                singularity exec $SINGULARITY_ARGS \
                    $SINGULARITY_ENV_ARGS \
                    $SINGULARITY_HOME_ARG \
                    --bind "$JUDGEARENA_BIND_PATHS" \
                    "$EVAL_SIF_PATH" \
                    env CUDA_VISIBLE_DEVICES=$GPU_DEVICES \
                        JUDGEARENA_DATA="$JUDGEARENA_DATA" \
                        HF_HOME="$HF_HOME" \
                        HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
                    judgearena \
                        "${{JUDGEARENA_ARGS[@]}}" \
                        --task "$task_path" \
                        --model.name "VLLM/$model_path" \
                        --run.result_folder "$RESULTS_SUBDIR" \
                        ${{LIMIT:+--generation.n_instructions $LIMIT}}
                EVAL_STATUS=$?
            fi
            ;;
        *)
            echo "[warning] Unknown evaluation suite '$eval_suite'. Skipping."
            EVAL_STATUS=1
            ;;
    esac < /dev/null
    set -e

    # An evaluation counts as done only if the harness exited 0 and wrote
    # something. Both halves matter: a container that fails to start and a
    # missing venv both exit 0 while producing no result.
    # A batched lm_eval call covers every task in its list, so a failure leaves
    # all of them without a result.
    EVAL_TASKS=$(( $(echo "$task_path" | tr -cd ',' | wc -c) + 1 ))
    if [ "$EVAL_STATUS" -ne 0 ]; then
        echo "[error] $task_path ($eval_suite) exited with status $EVAL_STATUS"
        FAILED_EVALS=$((FAILED_EVALS + EVAL_TASKS))
    elif [ -n "$EVAL_OUT_DIR" ] && [ -z "$(find "$EVAL_OUT_DIR" -type f -name "$EVAL_OUT_PAT" 2>/dev/null | head -n 1)" ]; then
        echo "[error] $task_path ($eval_suite) exited 0 but wrote no result to $EVAL_OUT_DIR/$EVAL_OUT_PAT"
        FAILED_EVALS=$((FAILED_EVALS + EVAL_TASKS))
    else
        echo "----------------------------------------------------"
        echo "Evaluation finished for model: $model_path"
    fi

    if [ -n "$JOB_TMP" ]; then
        rm -rf -- "$JOB_TMP"
        trap - EXIT ERR
    fi

done < <(tail -n +$((START_INDEX + 1)) "$CSV_PATH" | head -n $((END_INDEX - START_INDEX + 1)) | \
    awk -F, -v max_tasks="$TASKS_PER_JOB" -v OFS='\t' "$BATCH_ROWS_AWK")

if [ "$FAILED_EVALS" -gt 0 ]; then
    echo "Job $SLURM_ARRAY_TASK_ID finished with $FAILED_EVALS failed evaluation(s)."
    exit 1
fi
echo "Job $SLURM_ARRAY_TASK_ID finished."
