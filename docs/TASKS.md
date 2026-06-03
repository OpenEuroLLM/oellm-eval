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

## Language filtering (`--languages`)

Tasks can be filtered by language with `--languages`, independently of which
groups are selected:

```bash
# All German tasks across every benchmark
oellm-eval schedule --models "m" --languages "deu_Latn"

# Intersection: only German tasks within these benchmarks
oellm-eval schedule --models "m" --task_groups "sib200-eu" --languages "deu_Latn"
```

Languages are **derived in code** — there is no `languages` field to set in the
YAML. A task resolves to a canonical [`lang_Script`](https://en.wikipedia.org/wiki/IETF_language_tag)
code (e.g. `deu_Latn`) from, in order:

1. **`flores200:src-tgt` task names** → the non-English side(s) of the pair.
2. **The `{lang}` value** substituted into a `valid_langs` template (preferred
   for new multilingual groups — see the template expansion above).
3. **The task's `subset`** (e.g. `de`, `german`, `deu_Latn` all fold to
   `deu_Latn`).
4. A **trailing language code in the task name** (e.g. `arc_challenge_mt_de`),
   used only when no `subset` is given.

The normaliser (`oellm/task_groups.py`) folds the many spellings benchmarks use
(`de` / `deu_Latn` / `German` / `deu_latn`) onto one canonical code. To make a
new benchmark's languages filterable, prefer a `valid_langs` template or a
language-coded `subset`; if you introduce a spelling the normaliser doesn't yet
recognise, add it to `_LANG_ALIAS`. The guard test
`tests/test_language_groups.py::test_templated_tasks_all_resolve_to_a_language`
fails if any task in a templated or multilingual group does not resolve to a
language.

A single language filter transparently spans both `lm-eval-harness` and
`lighteval` tasks, since each task carries its own resolved suite. Unknown codes
error; a fully empty intersection errors; a partial match warns and proceeds.

## Important: Dataset Requirement

**You must provide the `dataset` field** (at group or task level) for:
1. **Automatic pre-download** - Compute nodes often lack network access; datasets are cached beforehand
2. **CI testing** - The test suite validates that all datasets in `task-groups.yaml` are accessible

Tasks without a `dataset` field will not have their data pre-downloaded and are not covered by CI validation.
