"""lm-eval's MMMLU high_school_mathematics tasks (openai-mmmlu), whose few-shot
examples skip marked documents in a decontaminated run.

MMMLU's 5 few-shot examples are a subject's first test questions, and
agenttrove-0.0 marked one of them in high_school_mathematics. When the job
template sets OELLM_DECONTAMINATED_SAMPLES (the run's --samples file), the
examples come from the kept documents. Otherwise the tasks are lm-eval's.
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
