# Adding Tasks and Task Groups

## Overview

Tasks are defined in `oellm/resources/task-groups.yaml`. Only tasks in this file are tested and guaranteed to work. The CLI parses this via `task_groups.py` and expands groups into `(task, n_shot, suite)` tuples for scheduling.

## YAML Structure

```yaml
task_groups:
  my-group:
    description: "Short description"
    suite: lm-eval-harness  # or lighteval
    n_shots: [5]            # default for all tasks in group
    dataset: org/dataset    # default HF dataset for pre-download
    tasks:
      - task: task_name
        n_shots: [0, 5]     # overrides group default
        dataset: org/other  # overrides group default
        subset: subset_name # HF dataset config/subset
```

## Adding a Task Group

1. Add your group to `oellm/resources/task-groups.yaml`:

```yaml
task_groups:
  my-benchmark:
    description: "My custom benchmark"
    suite: lm-eval-harness
    n_shots: [0]
    dataset: huggingface/dataset-name
    tasks:
      - task: task_one
        subset: split_a
      - task: task_two
        subset: split_b
```

2. Use it:

```bash
oellm schedule-eval --models "model-name" --task_groups "my-benchmark"
```

## Field Reference

| Field | Required | Level | Description |
|-------|----------|-------|-------------|
| `description` | Yes | group | Short description of the task group |
| `suite` | Yes | group | Evaluation suite: `lm-eval-harness` or `lighteval` |
| `n_shots` | Yes | group or task | List of shot counts; must be set at group or task level |
| `dataset` | Yes | group or task | HuggingFace dataset repo ID (required for pre-download and testing) |
| `task` | Yes | task | Task name as recognized by the evaluation suite |
| `subset` | No | task | HuggingFace dataset config/subset name |

## Important: Dataset Requirement

**You must provide the `dataset` field** (at group or task level) for:
1. **Automatic pre-download** - Compute nodes often lack network access; datasets are cached beforehand
2. **CI testing** - The test suite validates that all datasets in `task-groups.yaml` are accessible

Tasks without a `dataset` field will not have their data pre-downloaded and are not covered by CI validation.

## How to add custom lm-eval YAMLs

There are two related pieces of configuration to be aware of:

- `oellm/resources/task-groups.yaml` — oellm's scheduler registry. It lists the task names to run, the evaluation suite (`lm-eval-harness`, `lighteval`, or `evalchemy`), few-shot counts, and which dataset should be pre-downloaded.
- The evaluation-suite task registry (lm-eval or lighteval) — defines how a task is executed: which dataset split to load, how prompts are formatted, how answers are extracted, and which metric to compute.

If you add a task name to `task-groups.yaml`, the evaluation suite also needs a task definition for that name. If not, place custom task YAMLs in `oellm/resources/custom_lm_eval_tasks`. Then `main.py` ... <TOADD>
If you add a task name to `task-groups.yaml`, the evaluation suite also needs a task definition for that name. If not, place custom task YAMLs in `oellm/resources/custom_lm_eval_tasks`.

Then `main.py` will include that directory when generating the evaluation script: the `schedule_evals` command sets `lm_eval_include_path` (defaulting to the bundled `oellm/resources/custom_lm_eval_tasks`), and the generated SLURM script passes it to lm_eval as `--include_path`. This makes lm_eval aware of custom task YAMLs such as `polymath_en_low`. See `oellm/main.py` and `oellm/resources/template.sbatch` for the wiring.
