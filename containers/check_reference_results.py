"""Check complete reference-task result files without loading an evaluation engine.

This checks results/configuration, not scheduler completion or worker traces.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

COUNTS = {'piqa':1838, 'gsm8k':1319, 'MATH500':500, 'GPQADiamond':198, 'HumanEval':322}
METRICS = {'piqa':'acc_norm,none', 'gsm8k':'exact_match,strict-match',
           'MATH500':'accuracy', 'GPQADiamond':'accuracy_avg', 'HumanEval':'python_pass@1'}


def check(root, task):
    matches = []
    for path in root.rglob('*.json'):
        if root.name != 'results' and 'results' not in path.relative_to(root).parts:
            continue
        data = json.loads(path.read_text())
        if isinstance(data, dict) and task in data.get('results', {}):
            matches.append((path, data))
    if len(matches) != 1:
        raise ValueError(f'{task}: expected one result, found {len(matches)}')
    path, data = matches[0]
    metric, config, count = data['results'][task], data['config'], COUNTS[task]
    if config.get('limit') is not None:
        raise ValueError(f'{task}: local example cap')
    language_scores = {}
    if task == 'HumanEval':
        if config.get('max_tokens') != 'default':
            raise ValueError('HumanEval: local generation cap')
        for language, expected in [('python',164), ('sh',158)]:
            artifacts = list(root.rglob(f'generated_{language}.jsonl.graded.jsonl'))
            if len(artifacts) != 1:
                raise ValueError(f'HumanEval {language}: missing/ambiguous grading artifact')
            rows = [json.loads(line) for line in artifacts[0].read_text().splitlines() if line.strip()]
            if len(rows) != expected or len({r['task_id'] for r in rows}) != expected:
                raise ValueError(f'HumanEval {language}: incomplete or duplicate examples')
            if any(type(r['passed']) is not bool for r in rows):
                raise ValueError(f'HumanEval {language}: nonboolean verdict')
            score = sum(r['passed'] for r in rows)/expected
            if not math.isclose(score, metric[language+'_pass@1'], abs_tol=1e-12):
                raise ValueError(f'HumanEval {language}: verdict/score mismatch')
            language_scores[language] = dict(examples=expected, score=score, verdicts=str(artifacts[0]))
    elif task in ('MATH500', 'GPQADiamond'):
        if config.get('max_tokens') != 'default':
            raise ValueError(f'{task}: local generation cap')
        records = metric['examples']
        key = 'unique_id' if task == 'MATH500' else 'Question'
        if len(records) != count or len({r[key] for r in records}) != count:
            raise ValueError(f'{task}: incomplete or duplicate examples')
        if task == 'GPQADiamond' and (metric['num_repeat'] != 3 or any(len(r['model_outputs']) != 3 for r in records)):
            raise ValueError('GPQADiamond: incomplete repeats')
    else:
        if data['n-samples'][task]['effective'] != count:
            raise ValueError(f'{task}: incomplete effective count')
        samples = list(path.parent.glob('samples_'+task+'_*.jsonl'))
        if len(samples) != 1:
            raise ValueError(f'{task}: expected one sample file')
        groups = defaultdict(list)
        with samples[0].open() as stream:
            for line in stream:
                row = json.loads(line)
                groups[row.get('filter','none')].append(row['doc_id'])
        required = {'none'} if task == 'piqa' else {'strict-match','flexible-extract'}
        if not required.issubset(groups):
            raise ValueError(f'{task}: missing scoring filters')
        for rows in groups.values():
            if len(rows) != count or set(rows) != set(range(count)):
                raise ValueError(f'{task}: incomplete or duplicate sample IDs')
    return {'task':task, 'examples':count, 'metric':METRICS[task],
            'score':metric[METRICS[task]], 'result':str(path), 'config':config,
            'language_scores':language_scores,
            'scheduler_completion_checked':False, 'worker_trace_checked':False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--tasks', nargs='+', choices=list(COUNTS), default=['piqa','gsm8k','MATH500'])
    args = parser.parse_args()
    print(json.dumps([check(args.run, task) for task in args.tasks], indent=2))


if __name__ == '__main__':
    main()
