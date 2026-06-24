"""Helpers for the multilingual PolyMath (Qwen/PolyMath) generative math task.

The model is prompted to reason and wrap its final answer in ``\\boxed{...}``.
We extract that boxed span, normalise it (and the gold answer) with the
Minerva/Lewkowycz string normalisation, and score with exact match. A light
sympy numeric fallback catches equivalent-but-differently-written numbers
(e.g. ``18`` vs ``18.0``, ``1/2`` vs ``0.5``).

Note: unlike ``lm_eval``'s ``minerva_math`` we deliberately avoid
``math_verify`` / ``sympy.parsing.latex.parse_latex`` here — ``math_verify`` is
not installed in this project's venv and ``parse_latex`` requires
``antlr4-python3-runtime==4.11`` (the venv ships 4.13), so importing them would
break task loading. Everything below is self-contained.
"""

import re
import signal
from typing import Optional


try:
    from sympy import simplify, sympify

    _HAS_SYMPY = True
except ImportError:  # pragma: no cover - sympy is a transitive lm-eval dep
    _HAS_SYMPY = False


# --- Prompt -----------------------------------------------------------------
def doc_to_text(doc: dict) -> str:
    return (
        "Solve the following math problem step by step. "
        "Put your final answer inside \\boxed{}.\n\n"
        "Problem:\n" + doc["question"] + "\n\nSolution:"
    )


def doc_to_target(doc: dict) -> str:
    return doc["answer"]


# --- \boxed{} extraction (from lm_eval/tasks/minerva_math/utils.py) ---------
def last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return string[idx : right_brace_idx + 1]


def remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"
    assert s[: len(left)] == left
    assert s[-1] == "}"
    return s[len(left) : -1]


def _extract_answer(text: str) -> str:
    """Pull the final answer out of a model generation.

    Prefer the last ``\\boxed{...}`` span; otherwise fall back to the last
    number that appears in the text; otherwise return the stripped text.
    """
    boxed = last_boxed_only_string(text)
    if boxed is not None:
        try:
            return remove_boxed(boxed)
        except AssertionError:
            pass
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if numbers:
        return numbers[-1]
    return text.strip()


# --- Normalisation (from Lewkowycz et al. 2022 appendix D, via minerva) -----
SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]
REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "ft",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """Normalise a final answer to a quantitative reasoning question."""
    final_answer = final_answer.split("=")[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract answer that is in LaTeX math, is bold, is boxed, etc.
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalise shorthand TeX (\fracab -> \frac{a}{b}, \sqrta -> \sqrt{a}).
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalise 100,000 -> 100000
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer.strip()


# --- Equivalence ------------------------------------------------------------
class _timeout:
    def __init__(self, seconds: int = 5):
        self.seconds = seconds

    def handle_timeout(self, signum, frame):
        raise TimeoutError

    def __enter__(self):
        # SIGALRM is only settable from the main thread; degrade to no timeout
        # (sympify/simplify are fast for the numeric cases we rely on) otherwise.
        self._armed = False
        try:
            signal.signal(signal.SIGALRM, self.handle_timeout)
            signal.alarm(self.seconds)
            self._armed = True
        except ValueError:
            pass

    def __exit__(self, exc_type, exc_value, traceback):
        if self._armed:
            signal.alarm(0)


def _sympy_numeric_equiv(pred: str, gold: str) -> bool:
    """Best-effort numeric equivalence without LaTeX parsing.

    Handles plain numbers and simple arithmetic (``1/2`` == ``0.5``). LaTeX
    expressions that ``sympify`` cannot parse simply return ``False`` and the
    caller falls back to normalised string equality.
    """
    if not _HAS_SYMPY:
        return False
    try:
        with _timeout(5):
            a = sympify(pred.replace(",", ""))
            b = sympify(gold.replace(",", ""))
            return bool(simplify(a - b) == 0)
    except Exception:  # noqa: BLE001 - sympify/simplify raise many types incl. TimeoutError
        return False


def process_results(doc: dict, results: list[str]) -> dict[str, int]:
    candidate = results[0]
    pred = normalize_final_answer(_extract_answer(candidate))
    gold = normalize_final_answer(doc["answer"])

    correct = pred == gold or _sympy_numeric_equiv(pred, gold)
    return {"exact_match": int(correct)}
