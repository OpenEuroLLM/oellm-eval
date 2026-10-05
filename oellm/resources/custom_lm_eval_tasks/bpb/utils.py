"""Bits-per-byte (BPB) scoring for multiple-choice lm-eval-harness tasks.

Why this file exists
--------------------
lm-eval-harness only wires ``bits_per_byte`` into ``loglikelihood_rolling``
tasks (wikitext-style perplexity). For ``multiple_choice`` tasks the built-in
``ConfigurableTask.process_results`` emits a fixed set of keys
(``acc``/``acc_norm``/``acc_bytes``/``f1``/``mcc``/``exact_match``/``brier_score``),
so adding ``bits_per_byte`` to a task's ``metric_list`` alone does nothing.

The tasks in this directory therefore override ``process_results`` with the
functions below. They compute, per document:

    (loglikelihood_of_gold_continuation, byte_length_of_gold_continuation)

which is exactly the ``(a, b)`` pair shape that lm-eval's already-registered
``bits_per_byte`` *aggregation* consumes:

    bits_per_byte = -sum(a) / sum(b) / ln(2)

i.e. a corpus-level, byte-weighted average, not a mean of per-doc ratios. That
is the standard BPB definition used by DCLM/OLMo/Olmix, and it is what makes
BPB a smooth fitness signal for data-mixture regression at swarm scale, where
accuracy on 30M-parameter models is close to chance and very noisy.

Byte-length convention
----------------------
Bytes are counted on the continuation string exactly as returned by the task's
``doc_to_choice`` (UTF-8), *without* the ``target_delimiter`` (the leading
space lm-eval inserts between context and continuation). This matches lm-eval's
own ``acc_bytes`` convention. Two notes:

* Absolute BPB values shift slightly under a different convention, but the
  *ranking* of data mixtures -- the only thing the Olmix regressor consumes --
  is unaffected, as long as every run in the swarm uses the same convention.
* Do not compare these numbers against BPB from another codebase without first
  checking that codebase's byte convention.

Adding a task
-------------
Write a ``process_results_<task>(doc, results)`` that resolves the document's
choice strings and gold index, then returns ``_score(results, choices, gold)``.
``results`` is the list of ``(loglikelihood, is_greedy)`` pairs lm-eval
produces, one per choice, in ``doc_to_choice`` order.

Single-reference tasks (FLORES translation, MGSM answers) use
``output_type: loglikelihood`` instead and return
``_score_continuation(results, text)``. lm-eval does not insert
``target_delimiter`` for ``loglikelihood`` requests, so their ``doc_to_target``
prepends the space itself and ``text`` is the reference *without* it, keeping
the byte convention above.
"""

import re

# --------------------------------------------------------------------------
# Core scoring
# --------------------------------------------------------------------------


def _loglikelihoods(results):
    """Extract per-choice loglikelihoods from lm-eval's result payload."""
    lls = []
    for r in results:
        # multiple_choice requests come back as (loglikelihood, is_greedy)
        lls.append(float(r[0]) if isinstance(r, (tuple, list)) else float(r))
    return lls


def _byte_len(text):
    """UTF-8 byte length, floored at 1 so an empty choice can't divide by zero."""
    return max(len(str(text).encode("utf-8")), 1)


def _resolve_gold(gold, choices):
    """Normalise a gold label to a valid index into ``choices``, or None."""
    if isinstance(gold, (list, tuple)):
        # Multi-target tasks: score against the first valid gold.
        for g in gold:
            resolved = _resolve_gold(g, choices)
            if resolved is not None:
                return resolved
        return None
    if isinstance(gold, str):
        if gold in choices:
            return choices.index(gold)
        try:
            gold = int(gold)
        except (TypeError, ValueError):
            return None
    if isinstance(gold, int) and 0 <= gold < len(choices):
        return gold
    return None


def _score(results, choices, gold, continuation=None):
    """Return the metric dict for one document.

    Args:
        results: list of (loglikelihood, is_greedy) pairs, one per choice.
        choices: the continuation strings, in ``doc_to_choice`` order.
        gold: index (or label) of the correct choice.
        continuation: for ``multiple_input`` tasks (e.g. winogrande) the choices
            are *contexts* and the scored continuation is shared; pass it here
            so bytes are counted on the continuation rather than on the context.

    Returns:
        dict with ``bits_per_byte`` as an (loglikelihood, n_bytes) pair plus
        ``acc``/``acc_norm`` for free, since we already have every choice's
        loglikelihood.
    """
    lls = _loglikelihoods(results)
    choices = [str(c) for c in choices]
    gold_idx = _resolve_gold(gold, choices)

    if gold_idx is None or len(lls) != len(choices):
        # Malformed doc: contribute nothing rather than poisoning the corpus
        # aggregate with a bogus (ll, bytes) pair.
        return {"acc": 0.0, "acc_norm": 0.0, "bits_per_byte": (0.0, 1)}

    scored_text = continuation if continuation is not None else choices[gold_idx]
    gold_bytes = _byte_len(scored_text)

    char_lens = [max(len(c), 1) for c in choices]
    pred = max(range(len(lls)), key=lambda i: lls[i])
    pred_norm = max(range(len(lls)), key=lambda i: lls[i] / char_lens[i])

    return {
        "acc": 1.0 if pred == gold_idx else 0.0,
        "acc_norm": 1.0 if pred_norm == gold_idx else 0.0,
        # Consumed by lm-eval's registered "bits_per_byte" aggregation.
        "bits_per_byte": (lls[gold_idx], gold_bytes),
    }


def _score_continuation(results, continuation):
    """Return the metric dict for one ``loglikelihood`` (single-reference) doc.

    Args:
        results: one-element list holding the reference's (loglikelihood, is_greedy).
        continuation: the reference text, without the leading space that the
            task's ``doc_to_target`` prepends.
    """
    return {"bits_per_byte": (_loglikelihoods(results)[0], _byte_len(continuation))}


# --------------------------------------------------------------------------
# ARC (easy + challenge) and OpenBookQA share the AI2 label/answerKey schema
# --------------------------------------------------------------------------


def _ai2_choices_and_gold(doc, answer_key="answerKey"):
    choices = list(doc["choices"]["text"])
    labels = list(doc["choices"]["label"])
    key = str(doc[answer_key]).strip()
    gold = labels.index(key) if key in labels else _resolve_gold(key, choices)
    return choices, gold


def process_results_arc(doc, results):
    choices, gold = _ai2_choices_and_gold(doc)
    return _score(results, choices, gold)


def process_results_openbookqa(doc, results):
    choices, gold = _ai2_choices_and_gold(doc)
    return _score(results, choices, gold)


# --------------------------------------------------------------------------
# HellaSwag
# --------------------------------------------------------------------------


def _hellaswag_preprocess(text):
    text = text.strip()
    # Brackets are artifacts of the WikiHow portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def process_docs_hellaswag(dataset):
    def _process_doc(doc):
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        return {
            "query": _hellaswag_preprocess(doc["activity_label"] + ": " + ctx),
            "choices": [_hellaswag_preprocess(ending) for ending in doc["endings"]],
            "gold": int(doc["label"]),
        }

    return dataset.map(_process_doc)


def process_results_hellaswag(doc, results):
    gold = doc["gold"] if "gold" in doc else doc["label"]
    return _score(results, list(doc["choices"]), gold)


# --------------------------------------------------------------------------
# PIQA / BoolQ / Social IQa
# --------------------------------------------------------------------------


def process_results_piqa(doc, results):
    return _score(results, [doc["sol1"], doc["sol2"]], int(doc["label"]))


def process_results_boolq(doc, results):
    return _score(results, ["no", "yes"], int(doc["label"]))


def process_results_social_iqa(doc, results):
    choices = [doc["answerA"], doc["answerB"], doc["answerC"]]
    # social_i_qa labels are 1-indexed strings.
    return _score(results, choices, int(str(doc["label"]).strip()) - 1)


# --------------------------------------------------------------------------
# COPA (reimplemented here so this directory has no cross-task imports)
# --------------------------------------------------------------------------


def _copa_convert_choice(choice):
    return choice[0].lower() + choice[1:]


def doc_to_text_copa(doc):
    connector = {"cause": "because", "effect": "therefore"}[doc["question"]]
    # Drop the trailing period before appending the connector.
    return doc["premise"].strip()[:-1] + f" {connector}"


def doc_to_choice_copa(doc):
    return [
        " " + _copa_convert_choice(doc["choice1"]),
        " " + _copa_convert_choice(doc["choice2"]),
    ]


def doc_to_target_copa(doc):
    correct = doc["choice1"] if doc["label"] == 0 else doc["choice2"]
    return " " + _copa_convert_choice(correct)


def process_results_copa(doc, results):
    return _score(results, doc_to_choice_copa(doc), int(doc["label"]))


# --------------------------------------------------------------------------
# CommonsenseQA -- cloze form
#
# The stock lm-eval task scores the letters "A".."E", so its BPB would measure
# how well the model predicts a single letter, not whether it knows the answer.
# These tasks put the answer *text* in the continuation instead.
# --------------------------------------------------------------------------


def process_results_commonsense_qa(doc, results):
    choices, gold = _ai2_choices_and_gold(doc)
    return _score(results, choices, gold)


# --------------------------------------------------------------------------
# MMLU -- cloze/continuation form, for the same reason as CommonsenseQA
# --------------------------------------------------------------------------


def process_results_mmlu(doc, results):
    return _score(results, list(doc["choices"]), int(doc["answer"]))


# --------------------------------------------------------------------------
# Winogrande -- multiple_input: the choices are contexts and the continuation
# is shared, so bytes are counted on the continuation.
# --------------------------------------------------------------------------


def doc_to_text_winogrande(doc):
    return {"1": 0, "2": 1}[doc["answer"]]


def doc_to_target_winogrande(doc):
    idx = doc["sentence"].index("_") + 1
    return doc["sentence"][idx:].strip()


def doc_to_choice_winogrande(doc):
    idx = doc["sentence"].index("_")
    return [doc["sentence"][:idx] + opt for opt in [doc["option1"], doc["option2"]]]


def process_results_winogrande(doc, results):
    return _score(
        results,
        doc_to_choice_winogrande(doc),
        doc_to_text_winogrande(doc),
        continuation=doc_to_target_winogrande(doc),
    )


# ==========================================================================
# Multilingual tasks (the oellm-multilingual-bpb super group)
#
# Task YAMLs live in per-benchmark subdirectories and include the
# _<benchmark>_bpb_template_yaml files next to this module. XWinograd has the
# Winogrande schema and reuses the Winogrande functions above.
# ==========================================================================


# --------------------------------------------------------------------------
# Belebele -- cloze form: the continuation is the answer text, not a letter
# --------------------------------------------------------------------------


def process_results_belebele(doc, results):
    choices = [doc[f"mc_answer{i}"] for i in range(1, 5)]
    return _score(results, choices, int(doc["correct_answer_num"]) - 1)


# --------------------------------------------------------------------------
# Global-MMLU / INCLUDE -- cloze form, for the same reason as MMLU
# --------------------------------------------------------------------------


def _exam_choices(doc):
    return [doc["option_a"], doc["option_b"], doc["option_c"], doc["option_d"]]


def process_results_global_mmlu(doc, results):
    # Global-MMLU stores the gold answer as a letter.
    gold = {"A": 0, "B": 1, "C": 2, "D": 3}.get(str(doc["answer"]).strip())
    return _score(results, _exam_choices(doc), gold)


def process_results_include(doc, results):
    return _score(results, _exam_choices(doc), int(doc["answer"]))


# --------------------------------------------------------------------------
# XCOPA -- same prompt as the upstream lm-eval xcopa tasks. The per-language
# cause/effect connector is in each task's doc_to_text (xcopa/*.yaml).
# --------------------------------------------------------------------------


def doc_to_choice_xcopa(doc):
    return [_copa_convert_choice(doc["choice1"]), _copa_convert_choice(doc["choice2"])]


def process_results_xcopa(doc, results):
    return _score(results, doc_to_choice_xcopa(doc), int(doc["label"]))


# --------------------------------------------------------------------------
# XStoryCloze
# --------------------------------------------------------------------------


def process_results_xstorycloze(doc, results):
    choices = [doc["sentence_quiz1"], doc["sentence_quiz2"]]
    return _score(results, choices, int(doc["answer_right_ending"]) - 1)


# --------------------------------------------------------------------------
# MGSM -- the test split has no worked solutions, only the final number, so
# BPB is measured on the number that follows the upstream "direct" prompt.
# --------------------------------------------------------------------------


def _mgsm_answer(doc):
    answer = doc["answer_number"]
    if isinstance(answer, float) and answer.is_integer():
        answer = int(answer)
    return str(answer)


def doc_to_target_mgsm(doc):
    return " " + _mgsm_answer(doc)


def process_results_mgsm(doc, results):
    return _score_continuation(results, _mgsm_answer(doc))


# --------------------------------------------------------------------------
# FLORES-200 -- BPB of the reference translation given the source sentence.
# The facebook/flores pair configs (e.g. deu_Latn-eng_Latn) carry exactly two
# sentence_<lang> columns, one of which is English.
# --------------------------------------------------------------------------


def _flores_non_english(doc):
    (key,) = [k for k in doc if k.startswith("sentence_") and k != "sentence_eng_Latn"]
    return doc[key]


def doc_to_target_flores_xx_eng(doc):
    return " " + doc["sentence_eng_Latn"]


def process_results_flores_xx_eng(doc, results):
    return _score_continuation(results, doc["sentence_eng_Latn"])


def doc_to_target_flores_eng_xx(doc):
    return " " + _flores_non_english(doc)


def process_results_flores_eng_xx(doc, results):
    return _score_continuation(results, _flores_non_english(doc))
