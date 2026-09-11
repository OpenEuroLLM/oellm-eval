"""CPU integration checks inside the selected evaluation container (benign code only)."""
import concurrent.futures
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock
from typing import Dict, Any
from types import SimpleNamespace

# Widen the real race deterministically: filelock's later before-fork callback
# runs first, exposing overlapping fork ownership before this sleep ends.
os.register_at_fork(before=lambda: time.sleep(.1))
import filelock

BENCH = Path(os.environ['HUMANEVAL_BENCH'])
sys.path.insert(0, str(BENCH))
from human_eval.execution import check_correctness


class HumanEvalRuntimeTest(unittest.TestCase):
    def test_execution_body_preserved(self):
        original = Path(os.environ.get('HUMANEVAL_ORIGINAL', '/opt/evalchemy'))/'eval/chat_benchmarks/HumanEval/human_eval/execution.py'
        tree = ast.parse(original.read_text())
        outer = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'check_correctness')
        before = next(n for n in outer.body if isinstance(n, ast.FunctionDef) and n.name == 'unsafe_execute')
        tree = ast.parse((BENCH/'human_eval/execution.py').read_text())
        after = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'unsafe_execute')
        branch = next(n for n in after.body if isinstance(n, ast.If))
        preload = branch.body.pop(0)
        self.assertIsInstance(preload, ast.Import)
        self.assertEqual(preload.names[0].name, 'numpy')
        self.assertEqual([ast.dump(n) for n in before.body], [ast.dump(n) for n in after.body])

    def test_benchmark_propagates_failure_and_preserves_answers(self):
        tree = ast.parse((BENCH/'eval_instruct.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'evaluate_responses')
        namespace = dict(Dict=Dict, Any=Any, os=os, Path=Path, json=json,
                         subprocess=subprocess, sys=sys, __file__=str(BENCH/'eval_instruct.py'))
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(BENCH/'eval_instruct.py'), 'exec'), namespace)
        with tempfile.TemporaryDirectory() as folder:
            samples = Path(folder)/'generated_python.jsonl'
            samples.write_text('{"task_id":"python/0"}\n')
            instance = SimpleNamespace(languages=['python','sh'], data_dir=folder,
                                       num_workers=8, timeout=3, logger=mock.Mock())
            with mock.patch.object(subprocess, 'run', side_effect=subprocess.CalledProcessError(1, 'grader')):
                with self.assertRaises(subprocess.CalledProcessError):
                    namespace['evaluate_responses'](instance, {'artifact_dir':folder})
            self.assertEqual(samples.read_text(), '{"task_id":"python/0"}\n')

    def test_overlapping_process_startup(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(check_correctness, str(i), {'test_code': 'assert 1+1 == 2'}, 'python', 3)
                       for i in range(16)]
            self.assertTrue(all(f.result()['passed'] for f in futures))

    def test_full_grading_pass_fail_timeout_and_coverage(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for lang, codes in {
                'python': ['assert True', 'assert False', 'while True: pass'],
                'sh': ['exit 0', 'exit 1', 'sleep 2'],
            }.items():
                problems = [{'task_id': f'{lang}/{i}', 'prompt': '', 'test': ''} for i in range(3)]
                answers = [dict(problem, generation=code) for problem, code in zip(problems, codes)]
                data = root/f'{lang}-problems.jsonl'
                samples = root/f'{lang}-answers.jsonl'
                data.write_text(''.join(json.dumps(x)+'\n' for x in problems))
                samples.write_text(''.join(json.dumps(x)+'\n' for x in answers))
                request = dict(input_file=str(samples), problem_file=str(data), language=lang,
                               timeout=.5, n_workers=8, tmp_dir=str(root))
                req, score = root/'request.json', root/f'{lang}-score.json'
                req.write_text(json.dumps(request))
                command = [sys.executable, str(BENCH/'grade_responses.py'), str(req), str(score)]
                result = subprocess.run(command, capture_output=True, text=True, timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                self.assertAlmostEqual(json.loads(score.read_text())['pass@1'], 1/3,
                                       msg=lang+' '+Path(str(samples)+'.graded.jsonl').read_text())
                # A missing/duplicated answer must fail, not become a subset score.
                samples.write_text(json.dumps(answers[0])+'\n')
                score.unlink()
                result = subprocess.run(command, capture_output=True, text=True, timeout=30)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(score.exists())
                samples.write_text(''.join(json.dumps(answers[0])+'\n' for _ in range(3)))
                result = subprocess.run(command, capture_output=True, text=True, timeout=30)
                self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    print('filelock', filelock.__version__, 'benchmark', BENCH, flush=True)
    unittest.main(verbosity=2)
