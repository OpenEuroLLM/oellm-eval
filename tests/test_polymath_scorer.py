"""PolyMath's scorer runs its sympy checks in a worker process, so a check that never
finishes counts as wrong instead of freezing the evaluation."""

import importlib.util
import time
from pathlib import Path

import pytest

pytest.importorskip("latex2sympy2_extended")
pytest.importorskip("word2number")

TASK_DIR = Path(__file__).parents[1] / "oellm/resources/custom_lm_eval_tasks/polymath"


@pytest.fixture(scope="module")
def utils():
    # lm-eval loads a task's utils.py by path, not by import name
    spec = importlib.util.spec_from_file_location("polymath_utils", TASK_DIR / "utils.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "output, answer, expected",
    [
        (" \\boxed{18}\n\nSolve the following", "18", 1),
        (" \\boxed{17}", "18", 0),
        (" \\boxed{\\frac{\\pi}{3}}", "$\\frac{\\pi}{3}$", 1),
        (" \\boxed{[2,3)}", "$[2,3)$", 1),
        (" no answer", "18", 0),
    ],
)
def test_scores(utils, output, answer, expected):
    assert utils.process_results({"answer": answer}, [output]) == {
        "exact_match": expected
    }


def test_a_runaway_check_scores_wrong_and_scoring_goes_on(utils):
    # sympy evaluates 10^(10^10) in one C call that never releases the GIL; run in a
    # thread, it froze the evaluating process for hours.
    start = time.time()
    runaway = utils.process_results(
        {"answer": "$\\frac{-1+\\sqrt{5}}{4}$"}, [" \\boxed{10^{10^{10}}}"]
    )
    assert runaway == {"exact_match": 0}
    assert time.time() - start < 60
    correct = utils.process_results(
        {"answer": "$\\frac{\\pi}{3}$"}, [" \\boxed{\\frac{\\pi}{3}}"]
    )
    assert correct == {"exact_match": 1}
