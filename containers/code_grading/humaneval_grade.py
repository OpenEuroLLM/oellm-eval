"""Standalone HumanEval grading subprocess; installed beside eval_instruct.py.

The benchmark's bundled multilingual scorer, examples and timeouts are preserved.
"""
import json
import math
from pathlib import Path
import sys


def grade(request):
    # This script's directory selects the bundled multilingual package explicitly.
    from human_eval.evaluation import evaluate_functional_correctness
    import human_eval.evaluation as scorer
    expected = Path(__file__).parent/'human_eval/evaluation.py'
    if Path(scorer.__file__).resolve() != expected.resolve():
        raise RuntimeError(f'Wrong HumanEval scorer: {scorer.__file__}')
    def ids(path):
        return [json.loads(line)['task_id'] for line in Path(path).read_text().splitlines() if line.strip()]
    inputs, problems = ids(request['input_file']), ids(request['problem_file'])
    if len(inputs) != len(set(inputs)) or len(problems) != len(set(problems)) or set(inputs) != set(problems):
        raise ValueError('HumanEval requires one generated answer for every problem')
    result = evaluate_functional_correctness(**request)
    if 'pass@1' not in result or not all(math.isfinite(float(v)) and 0 <= float(v) <= 1 for v in result.values()):
        raise RuntimeError(f'Invalid HumanEval metrics: {result}')
    print(json.dumps({'language': request['language'], 'examples': len(inputs), 'metrics': result}), flush=True)
    return result


if __name__ == '__main__':
    request_path, output_path = map(Path, sys.argv[1:])
    output_path.write_text(json.dumps(grade(json.loads(request_path.read_text())), indent=2)+'\n')
