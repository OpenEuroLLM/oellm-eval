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

## Language filtering (`group[lang]` brackets)

Scope a task group (or super group) to one or more languages by attaching a
`[...]` bracket to its name. Codes inside the bracket may be separated by `,`
or `|`:

```bash
# The applicable subset of the multilingual super group, for one language
oellm-eval schedule --models "m" --task_groups "oellm-multilingual[deu_Latn]"

# A single benchmark, scoped to German
oellm-eval schedule --models "m" --task_groups "sib200-eu[deu_Latn]"

# Different language per benchmark in one run
oellm-eval schedule --models "m" \
    --task_groups "sib200-eu[fra_Latn],flores-200-eu-to-eng[deu_Latn]"
```

Inside the bracket, use the precise canonical `lang_Scri` code (e.g.
`deu_Latn`). Looser spellings such as `de` or `german` are **not** accepted as
input — they are rejected with an error naming the canonical code. (The folding
described below applies only to the benchmarks' own internal spellings when
deriving a task's language, not to what you type in the bracket.)

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

A language bracket transparently spans both `lm-eval-harness` and `lighteval`
tasks, since each task carries its own resolved suite. Unknown codes error; a
bracket that matches no task in its group errors; when a bracket lists several
languages and only some are present, the matches are kept and the rest warned
about (some languages simply lack certain benchmarks).

## Important: Dataset Requirement

**You must provide the `dataset` field** (at group or task level) for:
1. **Automatic pre-download** - Compute nodes often lack network access; datasets are cached beforehand
2. **CI testing** - The test suite validates that all datasets in `task-groups.yaml` are accessible

Tasks without a `dataset` field will not have their data pre-downloaded and are not covered by CI validation.
