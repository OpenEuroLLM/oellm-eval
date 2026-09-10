import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

patch = (Path(__file__).resolve().parents[1] / "patches/harness-single-token-loglikelihood.patch").read_text()
section = patch.split("diff --git a/lm_eval/models/vllm_single_token.py b/lm_eval/models/vllm_single_token.py\n", 1)[1]
source = "\n".join(line[1:] for line in section.splitlines() if line.startswith("+") and not line.startswith("+++")) + "\n"
from types import ModuleType
module = ModuleType("vllm_single_token")
sys.modules[module.__name__] = module
exec(compile(source, "vllm_single_token.py", "exec"), module.__dict__)


def output(values, greedy):
    return NS(outputs=[NS(logprobs=[{t: NS(logprob=v) for t, v in values.items()}], token_ids=[greedy])])


class SharedScoringTests(unittest.TestCase):
    def test_interleaved_and_duplicate_options(self):
        inputs = [[1, 2, 7], [9, 3], [1, 2, 8], [1, 2, 7]]
        plans = module.plan_requests(inputs, [2, 1, 2, 2])
        self.assertEqual([p.tokens for p in plans], [[1, 2], [9]])
        self.assertEqual(plans[0].selected, [7, 8])
        answers = module.restore_answers(plans, [output({7: -2, 8: -3, 0: -1}, 0), output({3: -1}, 3)], inputs, None)
        self.assertEqual(answers, [(-2, False), (-1, True), (-3, False), (-2, False)])

    def test_mixed_and_empty_context_fallback(self):
        inputs = [[1, 2, 3], [9, 4], [7]]
        plans = module.plan_requests(inputs, [1, 1, 0])
        self.assertEqual([p.selected for p in plans], [None, [4], None])
        calls = []
        def parse(**kwargs):
            calls.append(kwargs)
            return (-5., False)
        answers = module.restore_answers(plans, [None, output({4: -.5}, 4), None], inputs, parse)
        self.assertEqual(answers, [(-5., False), (-.5, True), (-5., False)])
        self.assertEqual([c["ctxlen"] for c in calls], [1, 0])

    def test_preserves_already_truncated_context(self):
        self.assertEqual(module.plan_requests([[2, 3, 4]], [2])[0].tokens, [2, 3])

    def test_feature_disable_preserves_each_request(self):
        plans = module.plan_requests([[1, 7], [1, 8]], [1, 1], False)
        self.assertEqual([p.selected for p in plans], [None, None])

    def test_selected_token_api_limit(self):
        plans = module.plan_requests([[1, i] for i in range(130)], [1] * 130)
        self.assertEqual([len(p.selected) for p in plans], [128, 2])

    def test_missing_logprob_and_count_fail_closed(self):
        inputs = [[1, 2]]
        plans = module.plan_requests(inputs, [1])
        with self.assertRaises(ValueError):
            module.restore_answers(plans, [output({3: -1}, 3)], inputs, None)
        with self.assertRaises(ValueError):
            module.restore_answers(plans, [], inputs, None)


if __name__ == "__main__":
    unittest.main()
