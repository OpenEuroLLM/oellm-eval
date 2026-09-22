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
#SBATCH --nodes=$NODES
#SBATCH --nodelist=$NODELIST


CSV_PATH="{csv_path}"
NUM_JOBS={num_jobs}
TOTAL_EVALS={total_evals}
LIMIT="{limit}"
VENV_PATH="{venv_path}"
LM_EVAL_INCLUDE_PATH="{lm_eval_include_path}"
TASKS_PER_JOB={tasks_per_job}

# avoiding crashes due to compute nodes not having access to the internet
export HF_HOME=$HF_HOME
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE={hf_hub_offline}

# Path to the shared Singularity image that contains all runtime deps (container mode)
export EVAL_SIF_PATH="$EVAL_BASE_DIR/$EVAL_CONTAINER_IMAGE"

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
    BIND_PATHS="$EVAL_BASE_DIR:$EVAL_BASE_DIR,$HF_HOME:$HF_HOME,$HF_DATASETS_CACHE:$HF_DATASETS_CACHE${{LM_EVAL_INCLUDE_PATH:+,$$LM_EVAL_INCLUDE_PATH:$$LM_EVAL_INCLUDE_PATH}}"

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
    if [[ "$SINGULARITY_ARGS" == *"--contain"* ]]; then
        JOB_HOME="$EVAL_BASE_DIR/$USER/container_home/$SLURM_JOB_ID"
        mkdir -p "$JOB_HOME"
        SINGULARITY_HOME_ARG="--home $JOB_HOME"
        SINGULARITY_ENV_ARGS="--env HF_HOME=$HF_HOME --env HF_DATASETS_CACHE=$HF_DATASETS_CACHE --env HF_HUB_OFFLINE=$HF_HUB_OFFLINE --env TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE --env NLTK_DATA=$NLTK_DATA"
    fi

    GPU_DEVICES=$(seq -s, 0 $(($GPUS_PER_NODE - 1)))

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
            singularity exec $SINGULARITY_ARGS \
                $SINGULARITY_ENV_ARGS \
                $SINGULARITY_HOME_ARG \
                --bind $BIND_PATHS \
                $EVAL_SIF_PATH \
                env CUDA_VISIBLE_DEVICES=$GPU_DEVICES \
                python "$@"
        fi
    }}

    # Set by every branch below: the harness exit status, and where to look for
    # what it wrote. lm_eval appends a timestamp to --output_path, so the result
    # is a directory plus a filename pattern rather than an exact path.
    EVAL_STATUS=0
    EVAL_OUT_DIR=""
    EVAL_OUT_PAT=""

    case "$suite_normalized" in
        lm_eval|lm-eval|lm-eval-harness)
            echo
            echo "----------------------------------------------------"
            echo "lm_eval Execution"
            RUN_ID=$(openssl rand -hex 5)
            EVAL_OUT_DIR="{evals_dir}"
            EVAL_OUT_PAT="$RUN_ID*.json"
            run_python -m lm_eval --model hf \
                --model_args pretrained="$model_path",trust_remote_code=True \
                --tasks "$task_path" \
                --num_fewshot "$n_shot" \
                --output_path "$EVAL_OUT_DIR/$RUN_ID.json" \
                --trust_remote_code \
                ${{LM_EVAL_INCLUDE_PATH:+--include_path $$LM_EVAL_INCLUDE_PATH}} \
                --batch_size auto \
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
                    "model_name=$model_path,trust_remote_code=True,{additional_model_args}" \
                    "$LIGHT_TASK_ARG" \
                    --load-tasks-multilingual \
                    --output-dir "$RESULTS_SUBDIR" \
                    ${{LIMIT:+--max-samples $LIMIT}}
                EVAL_STATUS=$?
            else
                singularity exec $SINGULARITY_ARGS \
                    $SINGULARITY_ENV_ARGS \
                    $SINGULARITY_HOME_ARG \
                    --bind $BIND_PATHS \
                    $EVAL_SIF_PATH \
                    env CUDA_VISIBLE_DEVICES=$GPU_DEVICES \
                    lighteval accelerate \
                        "model_name=$model_path,{additional_model_args}" \
                        "$LIGHT_TASK_ARG" \
                        --load-tasks-multilingual \
                        --output-dir "$RESULTS_SUBDIR" \
                        ${{LIMIT:+--max-samples $LIMIT}}
                EVAL_STATUS=$?
            fi
            ;;
        evalchemy)
            EVALCHEMY_WORK_DIR="{evalchemy_dir}"
            RESULTS_SUBDIR="{evals_dir}/$(openssl rand -hex 5)"
            EVAL_OUT_DIR="$RESULTS_SUBDIR"
            EVAL_OUT_PAT="*"
            if [ -n "$VENV_PATH" ]; then
                source "$VENV_PATH/bin/activate"
                # subshell: the loop no longer runs in one, so this cd would
                # otherwise leak into the next evaluation in this job
                (
                cd "$EVALCHEMY_WORK_DIR"
                accelerate launch --num-processes "$GPUS_PER_NODE" --num-machines 1 \
                    $MULTI_GPU_FLAG -m eval.eval \
                    --model hf \
                    --tasks "$task_path" \
                    --model_args "trust_remote_code=True,pretrained=$model_path" \
                    --batch_size auto \
                    --output_path "$RESULTS_SUBDIR" \
                    ${{LIMIT:+--limit $LIMIT}}
                )
                EVAL_STATUS=$?
            else
                echo "[error] evalchemy suite requires --venv_path (not supported in container mode)."
                exit 1
            fi
            ;;
        *)
            echo "[warning] Unknown evaluation suite '$eval_suite'. Skipping."
            EVAL_STATUS=1
            ;;
    esac

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

done < <(tail -n +$((START_INDEX + 1)) "$CSV_PATH" | head -n $((END_INDEX - START_INDEX + 1)) | \
    awk -F, -v max_tasks="$TASKS_PER_JOB" -v OFS='\t' "$BATCH_ROWS_AWK")

if [ "$FAILED_EVALS" -gt 0 ]; then
    echo "Job $SLURM_ARRAY_TASK_ID finished with $FAILED_EVALS failed evaluation(s)."
    exit 1
fi

echo "Job $SLURM_ARRAY_TASK_ID finished."
