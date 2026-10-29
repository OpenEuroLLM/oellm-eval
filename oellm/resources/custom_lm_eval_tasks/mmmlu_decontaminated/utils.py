"""lm-eval's MMMLU high_school_mathematics tasks (openai-mmmlu), with few-shot
examples that leave out marked documents in a decontaminated run.

MMMLU takes its 5 few-shot examples from the first test questions of a
subject, and in high_school_mathematics one of them is a question
agenttrove-0.0 marked as contaminated. In a decontaminated run (oellm's job
template sets OELLM_DECONTAMINATED_SAMPLES to the run's lm-eval --samples
file), the examples are taken from the documents the run keeps, so the next
question takes its place. Otherwise the tasks are lm-eval's, unchanged.
"""

import json
import os
from functools import partial

# lm-eval's --samples file of a decontaminated run.
SAMPLES = "OELLM_DECONTAMINATED_SAMPLES"


def _normalize_subject_name(name: str) -> str:
    # As lm-eval's openai-mmmlu/default/utils.py: some MMMLU configs embed CSV
    # file names in the Subject column (college_mathematics_test.csv_sw-KE.csv).
    if not isinstance(name, str):
        return name
    for marker in ("_test.csv", "_test-"):
        idx = name.find(marker)
        if idx != -1:
            return name[:idx]
    return name


def process_high_school_mathematics(dataset):
    return dataset.filter(
        lambda row: _normalize_subject_name(row["Subject"]) == "high_school_mathematics"
    )


def _fewshot_docs(dataset, task: str):
    docs = process_high_school_mathematics(dataset)
    path = os.environ.get(SAMPLES)
    if not path:
        return docs
    with open(path, encoding="utf-8") as f:
        kept = json.load(f).get(task)
    return docs if kept is None else docs.select(kept)


fewshot_de_de = partial(_fewshot_docs, task="mmmlu_de_de_high_school_mathematics")
fewshot_es_la = partial(_fewshot_docs, task="mmmlu_es_la_high_school_mathematics")
fewshot_fr_fr = partial(_fewshot_docs, task="mmmlu_fr_fr_high_school_mathematics")
fewshot_it_it = partial(_fewshot_docs, task="mmmlu_it_it_high_school_mathematics")
fewshot_pt_br = partial(_fewshot_docs, task="mmmlu_pt_br_high_school_mathematics")
