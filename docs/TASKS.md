# Adding Tasks and Task Groups

## Overview

Tasks are defined in `oellm/resources/task-groups.yaml`. Only tasks in this file are tested and guaranteed to work. The CLI parses this via `task_groups.py` and expands groups into `(task, n_shot, suite)` tuples for scheduling.

Supported evaluation suites:

| Suite value | Engine | Use case |
|---|---|---|
| `lm_eval` (alias `lm-eval-harness`) | [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness) | Text benchmarks |
| `lighteval` | [lighteval](https://github.com/huggingface/lighteval) | Translation / multilingual |
| `lmms_eval` | [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) | Image / video / audio benchmarks |
| `evalchemy` | [evalchemy](https://github.com/mlfoundations/evalchemy) (fork) | Free-form reasoning (GPQA, MATH500, LiveCodeBench) — needs its own venv, see [VENV.md](VENV.md) |
| `<contrib name>` | a contrib plugin | Custom benchmarks — see [CONTRIBUTING.md](../oellm/contrib/CONTRIBUTING.md) |

`suite` is set at the group level and can be overridden per task — the
`reasoning` group uses this to mix lm-eval and evalchemy tasks in one group.

## YAML Structure

```yaml
task_groups:
  my-group:
    description: "Short description"
    suite: lm_eval        # or lighteval, lmms_eval
    n_shots: [5]          # default for all tasks in group
    dataset: org/dataset  # default HF dataset for pre-download
    tasks:
      - task: task_name
        n_shots: [0, 5]     # overrides group default
        dataset: org/other  # overrides group default
        subset: subset_name # HF dataset config/subset
```

## Adding a Text Task Group

1. Add your group to `oellm/resources/task-groups.yaml`:

```yaml
task_groups:
  my-benchmark:
    description: "My custom benchmark"
    suite: lm_eval
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
oellm-eval schedule --models "model-name" --task-groups "my-benchmark"
```

## Adding an Image Task Group

Image tasks use `suite: lmms_eval`. Each task must correspond to a task name recognized by lmms-eval.

```yaml
task_groups:
  my-image-benchmark:
    description: "My image benchmark via lmms-eval"
    suite: lmms_eval
    n_shots: [0]
    tasks:
      - task: vqav2_val_all
        dataset: HuggingFaceM4/VQAv2
      - task: mmbench_en_dev
        dataset: HuggingFaceM4/MMBench_00
```

Run with:

```bash
oellm-eval schedule --models "path/to/vlm" --task-groups "my-image-benchmark"
```

The lmms-eval model adapter (e.g. `llava_hf`, `qwen2_vl`) is auto-detected
from the model name. No manual override is needed.

## Field Reference

| Field | Required | Level | Description |
|-------|----------|-------|-------------|
| `description` | Yes | group | Short description of the task group |
| `suite` | Yes (group) | group or task | Evaluation suite; a task-level value overrides the group |
| `n_shots` | Yes | group or task | List of shot counts; must be set at group or task level |
| `dataset` | Recommended | group or task | HuggingFace dataset repo ID — without it, nothing is pre-downloaded for the task |
| `task` | Yes | task | Task name as recognized by the evaluation suite |
| `subset` | No | task | HuggingFace dataset config/subset name |
| `revisions` | No | task | HF dataset revisions/branches to pre-fetch (default `["main"]`; e.g. MVBench keeps videos on a `video` branch) |
| `hf_models` | No | task | Auxiliary HF *model* repos to pre-download (e.g. a task router or SAM checkpoint) |
| `hf_dataset_files` | No | task | Specific files to fetch from a dataset repo: `{repo_id, patterns, revision?}` — use for large repos where only a subset is needed |

## Language filtering (`group[lang]` brackets)

Scope a task group (or super group) to one or more languages by attaching a
`[...]` bracket to its name. Codes inside the bracket may be separated by `,`
or `|`:

```bash
# The applicable subset of the multilingual super group, for one language
oellm-eval schedule --models "m" --task-groups "oellm-multilingual[deu_Latn]"

# A single benchmark, scoped to German
oellm-eval schedule --models "m" --task-groups "sib200-eu[deu_Latn]"

# Different language per benchmark in one run
oellm-eval schedule --models "m" \
    --task-groups "sib200-eu[fra_Latn],flores-200-eu-to-eng[deu_Latn]"
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

## Important: Dataset Pre-Download Behavior

**Provide the `dataset` field** (at group or task level) wherever possible:
compute nodes run with `HF_HUB_OFFLINE=1`, so anything not cached on the
login node before submission fails at eval time. Tasks without a `dataset`
field are simply skipped by the pre-download step.

Two details worth knowing:

1. **The group-name prefix selects the download strategy.** Groups whose name
   starts with `audio-`, `video-`, or `image-` are fetched with
   `snapshot_download` (raw repo files; the compute node builds the dataset at
   runtime — this avoids out-of-memory kills on the login node for large
   media datasets). All other groups go through `load_dataset()`. If you add
   a media group, keep the prefix.
2. **Dataset accessibility is verified by `tests/test_datasets.py`**, which
   needs network access and is therefore excluded from the offline CI run —
   execute it locally when adding datasets:
   `uv run pytest tests/test_datasets.py -k my-group`.

## Custom Benchmarks (contrib plugins)

For benchmarks that have their own inference scripts, custom metrics, or
are not part of lm-eval / lighteval / lmms-eval, use the contrib plugin
system. See [`oellm/contrib/CONTRIBUTING.md`](../oellm/contrib/CONTRIBUTING.md)
for the full guide and `oellm/contrib/regiondial_bench/` as a reference
implementation.
