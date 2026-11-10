"""A decontaminated run must score what excluding the marked rows after a full
run gives.

Runs a small model through lm-eval twice on tasks holding rows agenttrove-0.0
marked: over whole tasks with ``--log_samples``, and as the job template runs
a decontaminated task group, with ``--samples`` from the bundled list and the
results named ``<run>--decontaminated--agenttrove-0.0``. The second run must
be recognised and checked against the list, and score exactly what
``clean_results`` recomputes from the first with the marked rows as blocklist.
The tasks are 0-shot or take their first few-shot examples, so both runs give
the same prompts: direct matching (openbookqa, MMLU's subjects), the PolyMath
table, the GSM8K and GlobalMGSM tables (global_mgsm_de) and the MMLU table
(MMMLU's German). The exception is MMMLU's high_school_mathematics, whose first
test questions, its examples, hold a marked one: oellm's version of the task
leaves it out of a decontaminated run's examples, and must otherwise be
lm-eval's.

Needs lm-eval, the model and the datasets in the Hugging Face cache, and the
marked rows (OELLM_BLOCKLIST_DIR, agenttrove-0.0_filtered/dropped), so it only
runs when OELLM_LM_EVAL_MODEL_TESTS=1, e.g. in the eval container in a job on a
GPU node; set OELLM_LM_EVAL_DEVICE=cpu to run without a GPU. The check that
oellm's MMMLU tasks are lm-eval's outside a decontaminated run needs lm-eval
and the datasets only.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from oellm import decontaminated, sample_exclusion

BLOCKLIST_DIR = Path(os.environ.get("OELLM_BLOCKLIST_DIR", "/nonexistent"))
needs_model = pytest.mark.skipif(
    os.environ.get("OELLM_LM_EVAL_MODEL_TESTS") != "1" or not BLOCKLIST_DIR.is_dir(),
    reason="set OELLM_LM_EVAL_MODEL_TESTS=1 and OELLM_BLOCKLIST_DIR",
)

MODEL = "pretrained=EleutherAI/pythia-70m,dtype=float32"
DEVICE = os.environ.get("OELLM_LM_EVAL_DEVICE", "cuda")
NAME = "agenttrove-0.0"
# By shots, as their task groups run them.
TASKS = {
    0: ["openbookqa", "polymath_en_medium", "global_mgsm_de"],
    5: [
        "mmlu_high_school_mathematics",
        "mmlu_high_school_statistics",
        "mmlu_college_computer_science",
        "mmmlu_de_de_high_school_mathematics",
    ],
}
INCLUDE_PATH = Path(sample_exclusion.__file__).parent / "resources/custom_lm_eval_tasks"


# Its examples differ between the runs.
FEWSHOT_OVERRIDDEN = "mmmlu_de_de_high_school_mathematics"


def _lm_eval(output: Path, run: str, shots: int, *args, env=None) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable, "-m", "lm_eval",
            "--model", "hf", "--model_args", MODEL, "--device", DEVICE,
            "--tasks", ",".join(TASKS[shots]), "--num_fewshot", str(shots),
            "--batch_size", "1", "--gen_kwargs", "max_gen_toks=16",
            "--include_path", str(INCLUDE_PATH),
            # As oellm's job template names it.
            "--output_path", str(output / f"{run}.json"),
            "--log_samples",
            *args,
        ],
        check=True,
        env=env,
    )  # fmt: skip
    (results_file,) = output.glob(f"{run}_*.json")
    return results_file


@needs_model
@pytest.mark.parametrize("shots", sorted(TASKS))
def test_a_decontaminated_run_scores_what_excluding_after_a_full_run_gives(
    tmp_path, shots
):
    pytest.importorskip("lm_eval")
    full_file = _lm_eval(tmp_path / "full", "3fa9c1d2e0", shots)
    samples = tmp_path / f"samples-{NAME}.json"
    samples.write_text(json.dumps(decontaminated.samples(NAME)))
    # As the job template runs a decontaminated row.
    env = os.environ | {"OELLM_DECONTAMINATED_SAMPLES": str(samples)}
    decon_file = _lm_eval(
        tmp_path / "decon",
        f"3fa9c1d2e0{decontaminated.MARK}{NAME}",
        shots,
        "--samples",
        str(samples),
        env=env,
    )

    decon = json.loads(decon_file.read_text())
    assert decontaminated.run_of(decon_file, decon) == NAME

    full = json.loads(full_file.read_text())
    blocklist = sample_exclusion.load_blocklist(
        BLOCKLIST_DIR, sample_exclusion.load_match_fields()
    )
    # The run holds single MMLU subjects, which the map lists by their group.
    task_map = {
        benchmark: [
            *tasks,
            *(t for t in TASKS[shots] if any(t.startswith(f"{g}_") for g in tasks)),
        ]
        for benchmark, tasks in sample_exclusion.load_task_map().items()
    }
    clean = sample_exclusion.clean_results(
        full_file,
        full,
        blocklist,
        task_map,
        sample_exclusion.load_alignments(),
        sample_exclusion.load_normalized(),
        set(sample_exclusion.load_match_fields()),
    )
    listed = decontaminated.load(NAME)["tasks"]
    assert set(clean) == set(TASKS[shots])
    for task in TASKS[shots]:
        result, n_excluded = clean[task]
        assert n_excluded == len(listed[task]["excluded"]), task
        if task == FEWSHOT_OVERRIDDEN:
            _check_examples(full_file, decon_file, listed[task]["excluded"])
            continue
        for key, value in result.items():
            assert decon["results"][task][key] == pytest.approx(
                value, rel=1e-9, abs=1e-12
            ), (task, key)


def _prompts(results_file: Path, task: str) -> dict[int, tuple[dict, str]]:
    (path,) = results_file.parent.glob(f"samples_{task}_*.jsonl")
    out = {}
    for line in path.read_text().splitlines():
        sample = json.loads(line)
        out[sample["doc_id"]] = (
            sample["doc"],
            sample["arguments"]["gen_args_0"]["arg_0"],
        )
    return out


def _check_examples(full_file: Path, decon_file: Path, excluded: list[int]) -> None:
    """The full run's prompts show the first marked question as an example,
    the decontaminated run's none of the marked ones."""
    full = _prompts(full_file, FEWSHOT_OVERRIDDEN)
    decon = _prompts(decon_file, FEWSHOT_OVERRIDDEN)
    marked = [full[i][0]["Question"].strip() for i in excluded]
    assert any(marked[0] in prompt for _, prompt in full.values())
    for doc_id, (_, prompt) in decon.items():
        assert not any(q in prompt for q in marked), doc_id


def test_oellm_s_mmmlu_tasks_are_lm_eval_s_outside_a_decontaminated_run(monkeypatch):
    """oellm overrides MMMLU's high_school_mathematics tasks; without
    OELLM_DECONTAMINATED_SAMPLES they must give lm-eval's documents, prompts
    and few-shot examples, and its groups must load them."""
    pytest.importorskip("lm_eval")
    from lm_eval.tasks import TaskManager

    monkeypatch.delenv("OELLM_DECONTAMINATED_SAMPLES", raising=False)
    ours, theirs = TaskManager(include_path=str(INCLUDE_PATH)), TaskManager()
    for lang in ["de_de", "es_la", "fr_fr", "it_it", "pt_br"]:
        name = f"mmmlu_{lang}_high_school_mathematics"
        a = _leaves(ours.load([f"mmmlu_{lang}"])["tasks"])[name]
        b = _leaves(theirs.load([name])["tasks"])[name]
        fewshot = a.config.fewshot_config  # a dict, or lm-eval's FewshotConfig
        process = (
            fewshot.get("process_docs")
            if isinstance(fewshot, dict)
            else fewshot.process_docs
        )
        assert process is not None, name  # oellm's version, through the group
        docs = list(a.eval_docs)
        assert docs == list(b.eval_docs), name
        for doc in docs:
            assert a.doc_to_text(doc) == b.doc_to_text(doc), name
            assert a.sampler.sample(5, eval_doc=doc) == b.sampler.sample(
                5, eval_doc=doc
            ), name


def _leaves(loaded: dict) -> dict:
    out, stack = {}, [loaded]
    while stack:
        for name, task in stack.pop().items():
            if isinstance(task, dict):
                stack.append(task)
            else:
                out[name] = task
    return out
