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
# Exported so contrib suite plugins (which spawn their own Python subprocesses
# via oellm.contrib.dispatch) can read it from os.environ.  Built-in suites
# below still interpolate $LIMIT directly into their CLI flags.
export LIMIT="{limit}"
# Quantization request ("4bit" / "8bit" / empty). HF-style engines receive it
# via --model_args below; contrib plugins may opt in by reading this variable.
export OELLM_QUANTIZATION="{quantization}"
# Non-empty => pass --trust_remote_code to lm_eval (dataset-side trust).
LM_EVAL_TRC="{lm_eval_trc}"
VENV_PATH="{venv_path}"
LM_EVAL_INCLUDE_PATH="{lm_eval_include_path}"

# Compute nodes are air-gapped — every dataset must be cache-resolved.
export HF_HOME=$HF_HOME
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE={hf_hub_offline}
# HF_DATASETS_OFFLINE covers `dl_manager.download(arbitrary_url)` (external
# image/video hosts) which HF_HUB_OFFLINE does not.
export HF_DATASETS_OFFLINE={hf_hub_offline}
# lmms-eval auto-redirects "remote" (GPFS/Lustre) caches to a node-local
# /tmp by default; set this to force it to use the shared cache instead.
# See lmms_eval/api/task.py::_resolve_hf_datasets_cache_dir.
export LMMS_EVAL_DATASETS_CACHE="$HF_HOME/datasets"

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

# Use `tail` and `head` to slice the CSV file for the tasks assigned to this job.
# The +1 on START_INDEX accounts for the header row.
# Process substitution (not a pipe) keeps the loop in the current shell so
# FAILED_ROWS survives past `done` and the job can exit non-zero on failures.
FAILED_ROWS=0
while IFS=, read -r model_path task_path n_shot eval_suite
do
    rc=0
    # Strip CR (Windows line endings) from every field.
    model_path=${{model_path%$'\r'}}
    task_path=${{task_path%$'\r'}}
    n_shot=${{n_shot%$'\r'}}
    eval_suite=${{eval_suite:-lm_eval}}
    eval_suite=${{eval_suite%$'\r'}}

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
    BIND_PATHS="$EVAL_BASE_DIR:$EVAL_BASE_DIR,$HF_HOME:$HF_HOME,$HF_DATASETS_CACHE:$HF_DATASETS_CACHE"

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
        SINGULARITY_ENV_ARGS="--env HF_HOME=$HF_HOME --env HF_DATASETS_CACHE=$HF_DATASETS_CACHE --env HF_HUB_OFFLINE=$HF_HUB_OFFLINE --env HF_DATASETS_OFFLINE=$HF_DATASETS_OFFLINE --env TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE --env NLTK_DATA=$NLTK_DATA --env LMMS_EVAL_DATASETS_CACHE=$LMMS_EVAL_DATASETS_CACHE"
    fi

    GPU_DEVICES=$(seq -s, 0 $(($GPUS_PER_NODE - 1)))

    MULTI_GPU_FLAG=""
    if [ "$GPUS_PER_NODE" -gt 1 ]; then
        MULTI_GPU_FLAG="--multi-gpu"
    fi

    # Strip optional model-flags suffix ("regiondial_bench:vision_reasoner" → "regiondial_bench")
    suite_normalized=$(echo "${{eval_suite%%:*}}" | tr '[:upper:]' '[:lower:]')

    # Helper function to run Python commands in the appropriate environment
    # Optional per-row wall-clock bound: set ROW_TIMEOUT (GNU timeout
    # DURATION, e.g. "7200" or "2h") so a hung engine fails just its row
    # (exit 124) instead of consuming the slice's whole TIME_LIMIT.
    _maybe_timeout() {{
        if [ -n "${{ROW_TIMEOUT:-}}" ]; then
            timeout --kill-after=60 "$ROW_TIMEOUT" "$@"
        else
            "$@"
        fi
    }}

    # stdin is redirected from /dev/null so no child can consume the CSV rows
    # feeding the enclosing while-read loop.
    run_python() {{
        if [ -n "$VENV_PATH" ]; then
            source "$VENV_PATH/bin/activate"
            _maybe_timeout python "$@" < /dev/null
        else
            _maybe_timeout singularity exec $SINGULARITY_ARGS \
                $SINGULARITY_ENV_ARGS \
                $SINGULARITY_HOME_ARG \
                --bind $BIND_PATHS \
                $EVAL_SIF_PATH \
                env CUDA_VISIBLE_DEVICES=$GPU_DEVICES \
                python "$@" < /dev/null
        fi
    }}

    case "$suite_normalized" in
        lm_eval|lm-eval|lm-eval-harness)
            echo
            echo "----------------------------------------------------"
            echo "lm_eval Execution"
            run_python -m lm_eval --model hf \
                --model_args {lm_eval_model_args} \
                --tasks "$task_path" \
                --num_fewshot "$n_shot" \
                --output_path "{evals_dir}/$(openssl rand -hex 5).json" \
                ${{LM_EVAL_TRC:+--trust_remote_code}} \
                --batch_size auto \
                ${{LM_EVAL_INCLUDE_PATH:+--include_path $LM_EVAL_INCLUDE_PATH}} \
                ${{LIMIT:+--limit $LIMIT}}
            rc=$?
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

            if [ -n "$VENV_PATH" ]; then
                source "$VENV_PATH/bin/activate"
                _maybe_timeout lighteval accelerate \
                    "model_name=$model_path,{lighteval_trc}{additional_model_args}" \
                    "$LIGHT_TASK_ARG" \
                    --load-tasks-multilingual \
                    --output-dir "$RESULTS_SUBDIR" \
                    ${{LIMIT:+--max-samples $LIMIT}} < /dev/null
            else
                _maybe_timeout singularity exec $SINGULARITY_ARGS \
                    $SINGULARITY_ENV_ARGS \
                    $SINGULARITY_HOME_ARG \
                    --bind $BIND_PATHS \
                    $EVAL_SIF_PATH \
                    env CUDA_VISIBLE_DEVICES=$GPU_DEVICES \
                    lighteval accelerate \
                        "model_name=$model_path,{lighteval_trc}{additional_model_args}" \
                        "$LIGHT_TASK_ARG" \
                        --load-tasks-multilingual \
                        --output-dir "$RESULTS_SUBDIR" \
                        ${{LIMIT:+--max-samples $LIMIT}} < /dev/null
            fi
            rc=$?
            ;;
        lmms_eval|lmms-eval)
            # lmms-eval is used for image/video/audio benchmarks.
            # Adapter class is encoded in eval_suite as "lmms_eval:<adapter>" by the oellm CLI.
            _lmms_adapter="${{eval_suite#*:}}"
            OUTPUT_JSON="{evals_dir}/$(openssl rand -hex 5).json"

            # LLaVA adapters need model_name to avoid a missing-import bug in lmms-eval
            _lmms_extra_args=""
            if [[ "$_lmms_adapter" == "llava_onevision" || "$_lmms_adapter" == "llava_vid" || "$_lmms_adapter" == "video_llava" ]]; then
                _lmms_extra_args=",model_name=$(basename "$model_path")"
            fi

            # Qwen2-VL / Qwen2.5-VL: cap frames-per-video to fit on a 64GB
            # A100. lmms-eval defaults to 32 which OOMs on long-form video
            # (EgoSchema, VideoMME). Image tasks ignore this flag.
            # Override via MAX_NUM_FRAMES env var on larger GPUs.
            if [[ "$_lmms_adapter" == "qwen2_vl" || "$_lmms_adapter" == "qwen2_5_vl" ]]; then
                _lmms_extra_args+=",max_num_frames=${{MAX_NUM_FRAMES:-8}}"
            fi

            run_python -m lmms_eval \
                --model "$_lmms_adapter" \
                --model_args "{lmms_eval_model_args}" \
                --tasks "$task_path" \
                --num_fewshot "$n_shot" \
                --output_path "$OUTPUT_JSON" \
                ${{LIMIT:+--limit $LIMIT}}
            rc=$?
            ;;
        evalchemy)
            EVALCHEMY_WORK_DIR="{evalchemy_dir}"
            RESULTS_SUBDIR="{evals_dir}/$(openssl rand -hex 5)"
            if [ -n "$VENV_PATH" ]; then
                # Subshell: contain the cd so later rows keep the original cwd.
                (
                    source "$VENV_PATH/bin/activate"
                    cd "$EVALCHEMY_WORK_DIR" || exit 1
                    _maybe_timeout accelerate launch --num-processes "$GPUS_PER_NODE" --num-machines 1 \
                        $MULTI_GPU_FLAG -m eval.eval \
                        --model hf \
                        --tasks "$task_path" \
                        --model_args "{evalchemy_model_args}" \
                        --batch_size auto \
                        --output_path "$RESULTS_SUBDIR" \
                        ${{LIMIT:+--limit $LIMIT}} < /dev/null
                )
                rc=$?
            else
                echo "[error] evalchemy suite requires --venv-path (not supported in container mode)."
                rc=1
            fi
            ;;
        *)
            # Contrib suite: dispatch to the Python plugin registry.
            _CONTRIB_OUTPUT="{evals_dir}/$(openssl rand -hex 5).json"
            run_python -m oellm.contrib.dispatch \
                --suite      "$eval_suite" \
                --model_path "$model_path" \
                --task       "$task_path" \
                --n_shot     "$n_shot" \
                --output_path "$_CONTRIB_OUTPUT"
            rc=$?
            ;;
    esac

    echo "----------------------------------------------------"
    if [ "$rc" -ne 0 ]; then
        FAILED_ROWS=$((FAILED_ROWS + 1))
        _timeout_note=""
        if [ "$rc" -eq 124 ]; then _timeout_note=" (ROW_TIMEOUT=${{ROW_TIMEOUT:-}} exceeded)"; fi
        echo "[ERROR] Evaluation FAILED (exit=$rc)$_timeout_note for model=$model_path task=$task_path n_shot=$n_shot suite=$eval_suite"
    else
        echo "Evaluation finished for model: $model_path"
    fi

done < <(tail -n +$((START_INDEX + 1)) "$CSV_PATH" | head -n $((END_INDEX - START_INDEX + 1)))

if [ "$FAILED_ROWS" -gt 0 ]; then
    echo "Job $SLURM_ARRAY_TASK_ID finished with $FAILED_ROWS failed evaluation(s)."
    exit 1
fi

echo "Job $SLURM_ARRAY_TASK_ID finished."
