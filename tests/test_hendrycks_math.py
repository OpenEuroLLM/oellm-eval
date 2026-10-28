"""The vendored MATH answer check must score exactly like lm-eval's."""

import pytest

from oellm.hendrycks_math import is_equiv

CASES = [
    ("\\frac{1}{2}", "\\frac{1}{2}", True),
    ("\\frac{1}{2}", "\\dfrac12", True),
    ("\\frac{1}{2}", "1/2", True),
    ("\\frac{1}{2}", "0.5", True),
    ("5", "x = 5", True),
    ("5", "5\\text{ cm}", True),
    ("90", "90^\\circ", True),
    ("(1,2)", "\\left( 1, 2 \\right)", True),
    ("\\sqrt{3}", "\\sqrt3", True),
    ("0.5", ".5", True),  # both become "\frac{1}{2}"
    ("0.25", ".5", False),
    ("5", "", False),
    ("5", "6", False),
    # fix_a_slash_b raises on non-integers; is_equiv falls back to string equality.
    ("x/2", "x/2", True),
    ("x/2", "x / 2", False),
    ("5", None, False),
]


@pytest.mark.parametrize("answer, model_answer, expected", CASES)
def test_is_equiv(answer, model_answer, expected):
    assert is_equiv(answer, model_answer) is expected


@pytest.mark.parametrize("answer, model_answer, expected", CASES)
def test_matches_lm_eval(answer, model_answer, expected):
    lm_eval_utils = pytest.importorskip("lm_eval.tasks.hendrycks_math.utils")
    assert is_equiv(answer, model_answer) == lm_eval_utils.is_equiv(answer, model_answer)
