"""Excluding samples after a run must match lm-eval scoring only the rest.

Runs a small model through lm-eval twice: once over whole benchmarks with
``--log_samples``, and once with ``--samples`` limited to the documents a
blocklist leaves. The scores ``clean_results`` recomputes from the first run
must equal what lm-eval reports for the second. The tasks cover a
multiple-choice task, one whose metric has no aggregation in its config (boolq),
a generation task with two filters, a metric defined as a function (mbpp's
pass@1, which runs the generated code) and a size-weighted group.

Needs lm-eval, the model and the datasets in the Hugging Face cache, so it
only runs when OELLM_LM_EVAL_MODEL_TESTS=1, e.g. in the eval container in a
job on a GPU node; set OELLM_LM_EVAL_DEVICE=cpu to run without a GPU.
"""

import json
import os
import subprocess
import sys

import pytest

from oellm import sample_exclusion

pytestmark = pytest.mark.skipif(
    os.environ.get("OELLM_LM_EVAL_MODEL_TESTS") != "1",
    reason="set OELLM_LM_EVAL_MODEL_TESTS=1 to run lm-eval with a model",
)

MODEL = "pretrained=EleutherAI/pythia-70m,dtype=float32"
DEVICE = os.environ.get("OELLM_LM_EVAL_DEVICE", "cuda")
# 0-shot: lm-eval draws few-shot examples from one random stream across all
# documents, so with --samples a document would get other examples.
TASKS = ["openbookqa", "boolq", "gsm8k", "mbpp", "mmlu_stem"]


def _lm_eval(output, *args):
    subprocess.run(
        [
            sys.executable, "-m", "lm_eval",
            "--model", "hf", "--model_args", MODEL, "--device", DEVICE,
            "--tasks", ",".join(TASKS), "--num_fewshot", "0", "--batch_size", "1",
            "--gen_kwargs", "max_gen_toks=16",
            # As oellm's job template: results go to run_<date>.json.
            "--output_path", str(output / "run.json"),
            "--confirm_run_unsafe_code",  # mbpp runs the generated code
            *args,
        ],
        check=True,
    )  # fmt: skip
    (results_file,) = output.glob("run_*.json")
    return results_file, json.loads(results_file.read_text())


def test_excluding_after_the_run_equals_lm_eval_scoring_the_rest(tmp_path):
    pytest.importorskip("lm_eval")
    full_file, full = _lm_eval(tmp_path / "full", "--log_samples")

    # Blocklist every 7th document of each leaf task, as dataset rows.
    docs, blocklist = {}, {}
    for task in full["results"]:
        if full["group_subtasks"].get(task):
            continue
        (samples_file,) = full_file.parent.glob(f"samples_{task}_*.jsonl")
        docs[task] = {}
        for line in samples_file.read_text().splitlines():
            sample = json.loads(line)
            docs[task][sample["doc_id"]] = sample["doc"]
        blocklist.setdefault("mmlu" if task.startswith("mmlu") else task, []).extend(
            doc for doc_id, doc in docs[task].items() if doc_id % 7 == 3
        )
    # lm-eval scores every document that is not blocklisted. A document with the
    # same content as a blocklisted one is blocklisted too; MMLU has such
    # duplicates (11 pairs in college_physics alone).
    blocked = {
        json.dumps(row, sort_keys=True) for rows in blocklist.values() for row in rows
    }
    keep = {
        task: [
            i
            for i, doc in by_id.items()
            if json.dumps(doc, sort_keys=True) not in blocked
        ]
        for task, by_id in docs.items()
    }

    task_map = {t: [t] for t in TASKS} | {"mmlu": ["mmlu_stem"]}
    clean = sample_exclusion.clean_results(full_file, full, blocklist, task_map)
    _, rest = _lm_eval(tmp_path / "rest", "--samples", json.dumps(keep))

    compared = 0
    for name, (scores, _) in clean.items():
        expected = rest["groups"].get(name) or rest["results"][name]
        reported = {k for k, v in expected.items() if "," in k and v != "N/A"}
        assert reported <= scores.keys(), (name, reported - scores.keys())
        for key, value in scores.items():
            assert value == pytest.approx(expected[key], rel=1e-9, abs=1e-12), (name, key)
            compared += 1
    assert set(TASKS) <= clean.keys()
    assert clean["mmlu_stem"][1] > len(blocklist["mmlu"])  # duplicates were excluded
    assert compared > 50
