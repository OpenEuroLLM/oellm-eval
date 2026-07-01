"""Task glue for the multilingual PolyMath (Qwen/PolyMath) generative math task.

Scoring reproduces the official Qwen/PolyMath harness:
  * answer extraction is ``extract_boxed_content`` copied verbatim from
    https://github.com/QwenLM/PolyMath/blob/main/eval/run_eval.py (first
    ``\\boxed{...}`` span, spaces stripped), and
  * correctness is the upstream ``math_equal`` judge (see ``polymath_eval.py``,
    vendored verbatim from ``eval/scripts.py``).

``process_results`` mirrors ``run_eval.py``'s scoring loop: take the first boxed
span (``None`` if absent) and ``math_equal`` it against the gold ``answer``.
"""

import importlib.util as _ilu
import os as _os
import re

# --- Vendored upstream judge (polymath_eval.py, sibling file) ---------------
# lm-eval loads this module via spec_from_file_location, so __file__ is set but
# the task dir is not on sys.path; load the sibling scorer by absolute path.
_spec = _ilu.spec_from_file_location(
    "polymath_eval",
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "polymath_eval.py"),
)
_polymath_eval = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_polymath_eval)
math_equal = _polymath_eval.math_equal


# --- Prompt -----------------------------------------------------------------
def doc_to_text(doc: dict) -> str:
    return (
        "Solve the following math problem step by step. "
        "Put your final answer inside \\boxed{}.\n\n"
        "Problem:\n" + doc["question"] + "\n\nSolution:"
    )


def doc_to_target(doc: dict) -> str:
    return doc["answer"]


# --- Answer extraction (verbatim from PolyMath eval/run_eval.py) -------------
def extract_boxed_content(text):
    pattern = re.compile(r"boxed{")
    text = text.replace(" ", "")

    matches = pattern.finditer(text)
    results = []
    for match in matches:
        start_pos = match.end()
        brace_count = 1
        i = start_pos
        while i < len(text) and brace_count > 0:
            if text[i] == "{":
                brace_count += 1
            elif text[i] == "}":
                brace_count -= 1
            i += 1
        if brace_count == 0:
            results.append(text[start_pos : i - 1])
    return results


def process_results(doc: dict, results: list[str]) -> dict[str, int]:
    # Mirrors run_eval.py: first boxed span (None if absent), then math_equal
    # against the gold answer.
    extracted = extract_boxed_content(results[0])
    pred = extracted[0] if extracted else None
    gold = str(doc["answer"])
    correct = math_equal(pred, gold)
    return {"exact_match": int(bool(correct))}
