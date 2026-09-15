"""Versioned answer policies shared by reference calibration and offline replay.

No model-specific rules, gold-dependent extraction, tolerance, or code repair.
The optional runtime imports remain lazy so extraction can be tested with stdlib.
"""
import ast
from fractions import Fraction
import re

POLICY_VERSION = 'reference-corrections-v1'
STRICT_CODE = re.compile(r'```(?:[a-zA-Z]*)\n(.*?)```', re.S)
RELAXED_CODE = re.compile(r'(?:```|~~~)[ \t]*(?:python(?:3)?|py)?[ \t]*\r?\n(.*?)(?:```|~~~)', re.S | re.I)


def number(value):
    value = str(value).strip().replace(',', '').replace('$', '').replace('−', '-').rstrip('.')
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None


def gsm8k_results(doc, results):
    """Compare the harness's already-extracted answer, retaining its string score."""
    gold = doc['answer'].rsplit('####', 1)[-1].strip()
    prediction = str(results[0])
    def spelling(text):
        text = re.sub(r'(?s).*#### ', '', text.lower())
        return re.sub(r'\.$', '', text.replace(',', '').replace('$', ''))
    target = number(gold)
    if target is None:
        raise ValueError('GSM8K target is not a number')
    return {'exact_match': float(spelling(prediction) == spelling(gold)),
            'numeric_match': float(number(prediction) == target)}


def abstention(text):
    return '' if text.strip().casefold() == 'unanswerable' else text


def squad_class(base):
    """Adapt the actual installed task without altering prompts or thresholds."""
    class RecoveredSQuAD2(base):
        VERSION = 4

        def doc_to_text(self, doc, *unused):
            return super().doc_to_text(doc)

        def doc_to_target(self, doc, *unused):
            return super().doc_to_target(doc)

        def process_results(self, doc, results):
            continuation, probability = results
            return super().process_results(doc, [abstention(continuation), probability])
    return RecoveredSQuAD2


def choice(text):
    """An explicit final answer takes precedence over incidental letters in prose."""
    boxed = [(m.start(),m[1].upper()) for m in re.finditer(r'\\boxed\{\s*([ABCD])\s*\}', text, re.I)]
    # Lowercase letters require delimiters: prose such as "answer is a molecule"
    # must not overwrite an explicit boxed answer with the article "a".
    pattern = r'(?i:\b(?:(?:final|correct)\s+)?answer\s*(?:is|:)\s*)(?:\(\s*([ABCDabcd])\s*\)|([ABCD])(?=[\s.,;:!?]|$))'
    marked = [(m.start(),(m[1] or m[2]).upper()) for m in re.finditer(pattern,text)]
    matches = sorted(boxed + marked)
    if matches:
        return matches[-1][1]
    # Preserve the old fallback for genuinely unmarked responses.
    letters = re.findall(r'\b(A|B|C|D)\b', text.upper())
    return (letters[-1] if letters else text.strip().strip('.')).rstrip('.').rstrip('/')


def python_syntax(text):
    try:
        tree = ast.parse(text.strip())
    except (SyntaxError, ValueError, RecursionError):
        return False
    return any(isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom,
                              ast.For, ast.While, ast.Assign, ast.Call)) for n in ast.walk(tree))


def code_blocks(text):
    """LCB's existing extraction, plus one missing-only syntax-checked candidate."""
    strict = STRICT_CODE.findall(text)
    if strict:
        return strict
    relaxed = RELAXED_CODE.findall(text)
    if relaxed:
        return [relaxed[-1]] if python_syntax(relaxed[-1]) else []
    blocks = re.split(r'```[ \t]*(?:python(?:3)?|py)[ \t]*\r?\n', text, flags=re.I)
    if len(blocks) > 1 and '```' not in blocks[-1] and python_syntax(blocks[-1]):
        return [blocks[-1]]
    return [text] if python_syntax(text) else []


def completion_body(text, language):
    """Fix HumanEval's fixed-index fence scan, preserving unfenced completions."""
    lines = text.split('\n')
    opening = next((i for i, line in enumerate(lines) if line.strip().startswith('```')), None)
    if opening is None:
        return text
    closing = next((i for i in range(opening + 1, len(lines)) if lines[i].strip().startswith('```')), len(lines))
    return '\n'.join(lines[opening + 1:closing])


def install_evalchemy(task):
    """Explicit opt-in; callers record POLICY_VERSION with their run settings."""
    if task == 'GPQADiamond':
        from eval.chat_benchmarks.GPQADiamond import eval_instruct
        eval_instruct.get_multiple_choice_answer = choice
    elif task == 'LiveCodeBench':
        from eval.chat_benchmarks.LiveCodeBench import eval_instruct
        eval_instruct.has_code = code_blocks
    elif task == 'HumanEval':
        from eval.chat_benchmarks.HumanEval.utils import utils
        utils.get_code_block = completion_body
    else:
        raise ValueError('No reviewed extraction change for task: ' + task)


def main():
    """Container entry point; generation settings remain the caller's task recipe."""
    import argparse
    import hashlib
    import json
    from pathlib import Path
    import runpy
    import sys
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy',choices=[POLICY_VERSION],required=True)
    parser.add_argument('--module',choices=['eval.eval'],required=True)
    args, remaining = parser.parse_known_args()
    task_names = remaining[remaining.index('--tasks')+1].split(',')
    applied = []
    for task in task_names:
        if task in ('GPQADiamond','LiveCodeBench','HumanEval'):
            install_evalchemy(task); applied.append(task)
    out = Path(remaining[remaining.index('--output_path')+1]); out.mkdir(parents=True,exist_ok=True)
    receipt = dict(policy=args.policy, changed_tasks=applied,
                   module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   note='Extraction only; task prompts, decoding, tests and timeouts are unchanged.')
    with (out/'answer-policy.json').open('x') as stream:
        json.dump(receipt,stream,indent=2); stream.write('\n')
    print('Answer policy: '+json.dumps(receipt),flush=True)
    sys.argv = [args.module]+remaining
    runpy.run_module(args.module,run_name='__main__',alter_sys=True)


if __name__ == '__main__': main()
