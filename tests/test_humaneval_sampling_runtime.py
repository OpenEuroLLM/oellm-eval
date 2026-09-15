"""CPU integration tests in the complete evaluation image; no GPU/model load."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

from oellm.humaneval_sampling import build_prompt,read_jsonl,summarize,write_jsonl,sampling_parameters

BENCH=Path(os.environ.get('HUMANEVAL_BENCHMARK','/opt/evalchemy/eval/chat_benchmarks/HumanEval'))


@unittest.skipUnless(BENCH.exists(),'requires the pinned evaluation image')
class RuntimeTests(unittest.TestCase):
    def test_resolved_vllm_sampling_preserves_intended_settings(self):
        from vllm import SamplingParams
        for n in (1,32):
            case=dict(samples=n,temperature=0.0 if n==1 else .6,top_p=1.0 if n==1 else .95,
                max_new_tokens=512,seed=0,stop=['</s>'])
            expected=sampling_parameters(case,'HumanEval/0')
            params=SamplingParams(**expected)
            params.update_from_generation_config({'eos_token_id':2},2)
            for key,value in expected.items():self.assertEqual(getattr(params,key),value,key)
            self.assertEqual(params.sampling_type.name,'GREEDY' if n==1 else 'RANDOM_SEED')

    def test_all_current_prompts_match_installed_evalchemy(self):
        tree=ast.parse((BENCH/'eval_instruct.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='build_deepseekcoder_instruction')
        module=ast.Module(body=[method],type_ignores=[]);namespace={}
        exec(compile(module,'<installed prompt>','exec'),namespace)
        count=0
        for language,full in [('python','Python'),('sh','Bash')]:
            for row in read_jsonl(BENCH/f'data/humaneval-{language}.jsonl'):
                self.assertEqual(build_prompt(dict(row,language=language),'evalchemy'),
                    namespace[method.name](None,full,row['prompt']))
                count+=1
        self.assertEqual(count,322)

    def test_native_grader_repeated_answers_and_timeout(self):
        sys.path.insert(0,str(BENCH))
        from human_eval.evaluation import evaluate_functional_correctness
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            problem=dict(task_id='Python/0',prompt='def f():\n',test='assert f() == 7')
            write_jsonl(root/'problems.jsonl',[problem])
            rows=[dict(problem,completion_id=i,generation='def f():\n    '+(
                'return 7' if i<16 else 'return 8' if i<31 else 'while True: pass')) for i in range(32)]
            write_jsonl(root/'samples.jsonl',rows)
            score=evaluate_functional_correctness(input_file=str(root/'samples.jsonl'),
                problem_file=str(root/'problems.jsonl'),tmp_dir=str(root),n_workers=4,timeout=.5,k=[1])
            verdicts=read_jsonl(root/'samples.jsonl.graded.jsonl')
            self.assertEqual(summarize(verdicts,{'Python/0'},32)['value'],.5)
            self.assertEqual(score['pass@1'],.5)
            self.assertIn('timed out',next(r for r in verdicts if r['completion_id']==31)['result'].lower())

    @unittest.skipUnless(os.environ.get('NVIDIA_SANITIZER'),'requires pinned NVIDIA source')
    def test_nvidia_sanitizer_retains_helper_dependency(self):
        spec=importlib.util.spec_from_file_location('nvidia_sanitizer',os.environ['NVIDIA_SANITIZER'])
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        sanitized=module.sanitize('Here is the code:\n```python\ndef helper():\n    return 7\ndef f():\n    return helper()\n```',entrypoint='f')
        namespace={};exec(sanitized,namespace)
        self.assertEqual(namespace['f'](),7)


if __name__=='__main__':unittest.main()
