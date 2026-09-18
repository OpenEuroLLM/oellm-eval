# Integrating Custom Benchmarks

There are three ways to add a benchmark to elliot-cli, in increasing order
of effort. Pick the first one that fits.

---

## Path 1 — Benchmark already in lm-eval / lighteval / lmms-eval

Add a task group entry to `oellm/resources/task-groups.yaml`.

```yaml
task_metrics:
  my_task_name: acc

task_groups:
  my-benchmark:
    description: "My benchmark via lmms-eval"
    suite: lmms_eval         # or lm-eval-harness / lighteval
    n_shots: [0]
    tasks:
      - task: my_task_name
        dataset: org/my-hf-dataset
```

```bash
oellm-eval schedule \
  --models org/MyModel \
  --task-groups my-benchmark \
  --venv-path ~/elliot-venv
```

Supported `suite` values:

| `suite` | Framework |
|---|---|
| `lm-eval-harness` | [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) |
| `lighteval` | [lighteval](https://github.com/huggingface/lighteval) |
| `lmms_eval` | [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) |

---

## Path 2 — Custom lm-eval task (YAML + helpers, no plugin)

Use this when the benchmark is **not** in any engine yet but fits lm-eval's
task model: an HF dataset in, a per-sample prompt and target out. You write
only the task definition; scheduling, staging, collection, and reporting are
the platform's existing lm-eval machinery. No core file changes.

1. Create `oellm/resources/custom_lm_eval_tasks/<task>/<task>.yaml` — a
   standard lm-eval task config (plain `.yaml` directly in
   `custom_lm_eval_tasks/` also works for simple tasks). Prompt logic that
   doesn't fit YAML goes in a sibling `utils.py`, referenced as
   `!function utils.my_fn`. The bundled directory is passed to lm_eval via
   `--include_path` automatically; the `lm_eval_include_path` key in a run
   config (`oellm-eval eval --config …`) can point at an out-of-tree
   directory instead.
2. Wire `oellm/resources/task-groups.yaml`: a `task_metrics:` entry naming
   the headline metric, and a task group with `suite: lm-eval-harness`. The
   YAML `task:` field, the group's `task:` entry, and the `task_metrics:`
   key must all match.
3. Add a conformance test class in `tests/test_plugin_protocol.py`: group
   expansion, metric mapping, and prompt construction on a synthetic doc.

Live references, in increasing order of trickiness:

- `jeopardy.yaml`, `sib200/`, `arc_mt/` — plain YAML tasks.
- `tabfact/` — `utils.py` serializes tables into the prompt with a row cap;
  few-shot drawn from the train split.
- `timeseriesexam/` — the full toolbox: `doc_to_choice`/`doc_to_target` as
  functions (the dataset stores answers as option *text* with variable
  option counts), fixed-policy subsampling of 1000+-point series, and
  0-shot only because the dataset ships just a test split.

Gotchas:

- Few-shot needs a split to draw from (`fewshot_split`); test-only datasets
  must schedule with `n_shots: [0]`.
- Script-based datasets need `dataset_kwargs: {trust_remote_code: true}` in
  the task YAML; parquet-native datasets need nothing.
- Any serialization policy (row caps, series subsampling) must be fixed and
  documented in the task — every model has to see the identical prompt or
  scores stop being comparable.

---

## Path 3 — Custom benchmark (new contrib suite)

Use this when the benchmark has its own inference script, custom metrics, or
requires multi-GPU sharding.

Drop files into `oellm/contrib/my_suite/`. No changes to any core file.

```
oellm/contrib/my_suite/
├── __init__.py
├── suite.py
├── task.py
├── adapter.py
├── metrics.py
└── README.md
```

See `oellm/contrib/regiondial_bench/` as a complete reference.

---

### Step 1 — task.py

```python
from oellm.core.base_task import BaseTask


class MyTask(BaseTask):
    # Required
    @property
    def name(self) -> str:
        return "my_task_name"

    @property
    def suite(self) -> str:
        return "my_suite"

    @property
    def n_shots(self) -> list[int]:
        return [0]

    # Optional
    @property
    def task_group_name(self) -> str:
        return "my-benchmark"  # default: name with _ replaced by -

    @property
    def description(self) -> str:
        return "My benchmark on dataset X."

    @property
    def primary_metric(self) -> str | None:
        return "my_primary_metric"

    @property
    def hf_models(self) -> list[str]:
        return ["org/auxiliary-model"]

    @property
    def hf_dataset_files(self) -> list[dict]:
        return [
            {
                "repo_id": "org/my-dataset",
                "patterns": ["data/test.json", "data/images/*"],
            }
        ]
```

`hf_models` and `hf_dataset_files` are downloaded on the login node before
SLURM submission. Compute nodes run with `HF_HUB_OFFLINE=1`. Use `patterns`
to download only the files you need from large repos.

---

### Step 2 — adapter.py

```python
from pathlib import Path
from oellm.core.base_model_adapter import BaseModelAdapter


class MyModelAdapter(BaseModelAdapter):
    def __init__(self, model_path: str) -> None:
        self._path = model_path

    @property
    def model_path(self) -> str:
        return self._path

    def to_lm_eval_args(self) -> str:
        return f"pretrained={self._path},trust_remote_code=True"

    def to_lmms_eval_args(self) -> str:
        return f"pretrained={self._path}"

    def to_contrib_flags(self) -> str | None:
        """Return a model-type suffix passed to run() as model_flags, or None."""
        name = Path(self._path).name.lower()
        if "mymodel_v2" in name:
            return "backend_b"
        return "backend_a"
```

The string returned by `to_contrib_flags()` is appended to `eval_suite` in
`jobs.csv` as `"my_suite:backend_a"` and passed to `run()` as `model_flags`.
Use it to select between different inference backends for the same benchmark.

---

### Step 3 — suite.py

```python
from pathlib import Path
from oellm.contrib.my_suite.task import MyTask

SUITE_NAME = "my_suite"

TASK_GROUPS: dict = MyTask.to_task_groups_dict()

CLUSTER_ENV_VARS = ["MY_DATA_DIR"]


def detect_model_flags(model_path: str) -> str | None:
    from oellm.contrib.my_suite.adapter import MyModelAdapter

    return MyModelAdapter(model_path).to_contrib_flags()


def run(
    *,
    model_path: str,
    task: str,
    n_shot: int,
    output_path: Path,
    model_flags: str | None,
    env: dict[str, str],
) -> None:
    """Run the evaluation and write results to output_path.

    output_path must be a lmms-eval-compatible JSON:

        {
          "model_name_or_path": "<model_path>",
          "results": {
            "<task>": {"metric_a": 0.42, "metric_b": 0.38}
          },
          "configs": {
            "<task>": {"num_fewshot": <n_shot>}
          }
        }
    """
    data_dir = env["MY_DATA_DIR"]
    # ... run inference, compute metrics, write output_path ...


def parse_results(data: dict) -> tuple[str, str, int, dict[str, float]] | None:
    """Try to parse data as output from this suite.

    Return (model_id, task_name, n_shot, {metric: value}) or None.
    """
    results = data.get("results", {})
    for task_name, task_results in results.items():
        if task_name.startswith("my_task_") and "my_primary_metric" in task_results:
            model_id = data.get("model_name_or_path", "unknown")
            n_shot = data.get("configs", {}).get(task_name, {}).get("num_fewshot", 0)
            return model_id, task_name, int(n_shot), task_results
    return None
```

`CLUSTER_ENV_VARS` are validated twice: by the scheduler's environment
pre-flight on the login node (before SLURM submission) and by `dispatch.py`
on the compute node before `run()` is called.

> **Note on `parse_results`:** `oellm-eval collect` calls every suite's
> `parse_results` **first-chance** on each result JSON before falling back
> to the generic lmms-eval-shaped parsing — a suite that recognizes a file
> owns its format (enforced end-to-end by `tests/test_plugin_protocol.py`).
> Returning `None` for files that are not yours is part of the contract.

---

### Step 4 — metrics.py

```python
from oellm.core.base_metric import BaseMetric


class MyMetric(BaseMetric):
    @property
    def name(self) -> str:
        return "my_metric"

    def compute(self, samples) -> float:
        # A sample is whatever record your inference step produces —
        # dicts recommended; put references inside the record.
        if not samples:
            return 0.0
        correct = sum(
            1
            for s in samples
            if isinstance(s, dict) and s.get("prediction") == s.get("reference")
        )
        return correct / len(samples)
```

`compute()` receives the task's per-sample records directly (BaseMetric API
v2 — see `oellm.core.CORE_API_VERSION`). Multi-reference tasks put their
references inside each record; handle invalid records (`None`, parse
failures) deliberately and document the choice.

---

### Step 5 — clusters.yaml

Add cluster-specific paths to `oellm/resources/clusters.yaml`:

```yaml
my-cluster:
  MY_DATA_DIR: "/path/to/benchmark/data"
  MY_SUITE_NUM_GPUS: "4"
```

---

### Step 6 — tests

Add `tests/test_my_suite.py`. Cover at minimum:

- `SUITE_NAME`, `CLUSTER_ENV_VARS`, `TASK_GROUPS` structure
- `MyTask` properties: `name`, `suite`, `n_shots`, `primary_metric`, `hf_models`, `hf_dataset_files`, `task_group_name`
- `MyTask.to_task_groups_dict()` output structure
- `MyModelAdapter.to_contrib_flags()` for expected model name patterns
- `detect_model_flags()` return values
- `parse_results()` for a matching and a non-matching JSON
- Metric `compute()` correctness
- Task group expansion: correct `(task, n_shot, suite)` tuples
- Dry-run `schedule_evals()` produces SBATCH with `oellm.contrib.dispatch`

See `tests/test_regiondial_bench.py` as a reference.

---

## Notes

- Point `HF_HOME` to a filesystem with sufficient space. Home directories on
  HPC clusters typically have small quotas (50 GB or less).

## Runtime environment variables available to `run()`

The dispatcher passes the full job environment via the ``env`` parameter.
Interim channels (a formal ``eval_args`` parameter is planned for a future
plugin-protocol revision):

- ``LIMIT`` — sample cap requested via ``--limit`` (empty = no cap).
- ``OELLM_QUANTIZATION`` — ``"4bit"`` / ``"8bit"`` / empty. Plugins that can
  honor quantized loading should read this; plugins that cannot should ignore
  it (the scheduler already warns the operator that contrib rows run at full
  precision).
