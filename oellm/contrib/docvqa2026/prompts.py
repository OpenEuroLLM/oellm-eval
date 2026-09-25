"""The competition's baseline prompt, taken from the official scorer."""

from oellm.contrib.docvqa2026._vendor_eval_utils import get_evaluation_prompt

MASTER_PROMPT = get_evaluation_prompt()

__all__ = ["MASTER_PROMPT", "get_evaluation_prompt"]
