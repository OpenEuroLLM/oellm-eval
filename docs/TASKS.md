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
oellm-eval schedule --models "model-name" --task_groups "my-benchmark"
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
