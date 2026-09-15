import json
from pathlib import Path
import tempfile
import unittest

from oellm.humaneval_sampling import build_prompt,load_case,request_seed,sampling_parameters,summarize,validate_samples


class SampledHumanEvalTests(unittest.TestCase):
    def test_average_is_not_best_of_32(self):
        rows=[dict(task_id=t,completion_id=i,passed=i<n) for t,n in [('a',8),('b',24)] for i in range(32)]
        result=summarize(rows,{'a','b'},32)
        self.assertEqual(result['value'],.5)
        self.assertEqual(result['per_problem'],{'a':.25,'b':.75})
        self.assertEqual(result['responses'],64)

    def test_missing_duplicate_unexpected_and_invalid_verdict_fail(self):
        rows=[dict(task_id='a',completion_id=i,passed=False) for i in range(32)]
        for invalid in (rows[:-1],rows+[rows[0]],rows+[dict(task_id='b',completion_id=0,passed=True)]):
            with self.assertRaises(ValueError):validate_samples(invalid,{'a'},32)
        rows[0]['passed']='False'
        with self.assertRaises(ValueError):summarize(rows,{'a'},32)

    def test_sample_ids_not_row_order_determine_coverage(self):
        rows=[dict(task_id='a',completion_id=i,passed=i<4) for i in range(32)]
        self.assertEqual(summarize(rows,{'a'},32),summarize(list(reversed(rows)),{'a'},32))

    def test_seeds_depend_on_problem_not_worker_or_order(self):
        self.assertEqual(request_seed(1234,'Python/0'),request_seed(1234,'Python/0'))
        self.assertNotEqual(request_seed(1234,'Python/0'),request_seed(1234,'Python/1'))
        params=sampling_parameters(dict(samples=32,temperature=.2,top_p=.95,max_new_tokens=1024,seed=1234),'Python/0')
        self.assertEqual(params['n'],32);self.assertEqual(params['temperature'],.2)
        self.assertFalse(params['ignore_eos'])

    def test_prompt_preserves_completion_whitespace(self):
        row=dict(language='python',prompt='def f():\n    """doc"""\n')
        self.assertEqual(build_prompt(row,'completion'),row['prompt'])
        self.assertTrue(build_prompt(row,'evalchemy').endswith('```python\ndef f():\n    """doc"""\n```'))

    def test_reject_32_greedy_copies(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'case.json';path.write_text(json.dumps(dict(schema=1,samples=32,temperature=0)))
            with self.assertRaises(ValueError):load_case(path)


if __name__=='__main__':unittest.main()
