r"""

:authors:     Martin Dočekal, Martin Fajčík
"""

import collections
import re
import string

from sklearn.metrics import f1_score


def aggregate_macro_f1_score(items, **kwargs):
    unzipped_list = list(zip(*items, strict=False))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    fscore = f1_score(golds, preds, average="macro")
    return fscore


# MMLU multi-choice style (A....Z)
ANSWER_LETTERS = [chr(ord("A") + i) for i in range(ord("Z") - ord("A") + 1)]


def mmlu_get_choice(dataset):
    choice = [c for c in ANSWER_LETTERS if c in dataset.keys()]
    if len(choice) == 0:
        raise ValueError("No answer columns found in dataset")
    return choice


def mmlu_get_answer_index(dataset):
    return ANSWER_LETTERS.index(dataset["correct_answer"])


def cermat_get_choice(dataset):
    if len(dataset["choices"]) == 4:
        return ["A", "B", "C", "D"]
    elif len(dataset["choices"]) == 5:
        return ["A", "B", "C", "D", "E"]
    else:
        raise ValueError(f"Invalid number of choices: {len(dataset['choices'])}")


def mmlu_get_question_text_umimeto(dataset):
    """
    Umimeto has really bad questions, which require the topic at minimum to make sense
    """
    dataset_answer_keys = mmlu_get_choice(dataset)
    question_text = dataset["question"].strip()
    choices_text = "\n".join(f"{c}. {dataset[c]}" for c in dataset_answer_keys)
    topic = dataset["topic"]
    return f"{topic}: {question_text}\n{choices_text}\nOdpověď: "


# csqad32 metric. SQuAD EM/F1 copied from transformers.data.metrics.squad_metrics,
# which transformers 5 removed; kept verbatim so scores stay comparable.
def _normalize_answer(s):
    s = "".join(ch for ch in s.lower() if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _get_tokens(s):
    return _normalize_answer(s).split() if s else []


def compute_exact(a_gold, a_pred):
    return int(_normalize_answer(a_gold) == _normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks, pred_toks = _get_tokens(a_gold), _get_tokens(a_pred)
    num_same = sum(
        (collections.Counter(gold_toks) & collections.Counter(pred_toks)).values()
    )
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision, recall = num_same / len(pred_toks), num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)


def process_results(doc, results):
    preds = results[0]
    reference = doc["answers"][0]
    return {
        "f1": compute_f1(reference, preds),
        "exact_match": compute_exact(reference, preds),
    }
