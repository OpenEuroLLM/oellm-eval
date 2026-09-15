from pathlib import Path
import importlib.util
import unittest

spec = importlib.util.spec_from_file_location('policies', Path(__file__).resolve().parents[1]/'oellm/evaluation_policies.py')
p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)


class PolicyTests(unittest.TestCase):
    def test_same_extracted_number_without_tolerance(self):
        doc = {'answer':'reasoning #### 16'}
        self.assertEqual(p.gsm8k_results(doc,['16.00']), {'exact_match':0.0,'numeric_match':1.0})
        for wrong in ('16.01', '16 or 17', '', '[invalid]', 'nan', 'inf', '1/0'):
            self.assertEqual(p.gsm8k_results(doc,[wrong])['numeric_match'],0)
        self.assertEqual(p.number('-1,200.00'),p.number('-1200'))
        self.assertEqual(p.number('1/2'),p.number('.5'))

    def test_abstention_only_for_exact_taught_marker(self):
        self.assertEqual(p.abstention(' Unanswerable\n'),'')
        for text in ('unanswerable question','not unanswerable','unknown','no answer'):
            self.assertEqual(p.abstention(text),text)

    def test_marked_choice_is_not_overwritten_by_prose(self):
        self.assertEqual(p.choice(r'\boxed{C} because it is a valid compound.'),'C')
        self.assertEqual(p.choice('The correct answer is (D).'),'D')
        self.assertEqual(p.choice(r'\boxed{B}. Correction: the final answer is (C).'),'C')
        self.assertEqual(p.choice(r'\boxed{C}. The answer is a molecule.'),'C')
        self.assertEqual(p.choice('The correct answer is (d).'),'D')
        self.assertEqual(p.choice('A then D'),'D')

    def test_code_extraction_never_changes_existing_strict_candidate(self):
        self.assertEqual(p.code_blocks('```python\nprint(1)\n```\n``` python3\nprint(2)\n```'),['print(1)\n'])
        self.assertEqual(p.code_blocks('print(1)'),['print(1)'])
        self.assertEqual(p.code_blocks('```python\nprint(1)\n'),['print(1)\n'])
        self.assertEqual(p.code_blocks('``` python3\nprint(1)\n```'),['print(1)\n'])
        self.assertEqual(p.code_blocks('Here is reasoning.\nprint(1)'),[])
        self.assertEqual(p.code_blocks('print('),[])

    def test_completion_preserves_first_and_last_lines_and_indent(self):
        raw='    x = 1\n    return x'
        self.assertEqual(p.completion_body(raw,'python'),raw)
        self.assertEqual(p.completion_body('Here it is\n```python\n'+raw+'\n```\nExplanation','python'),raw)
        self.assertEqual(p.completion_body('```python\n'+raw,'python'),raw)


if __name__ == '__main__':unittest.main()
