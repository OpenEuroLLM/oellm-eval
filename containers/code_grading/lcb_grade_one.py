"""Lightweight spawned LiveCodeBench grading, retaining the original test body."""
import json
import multiprocessing
from pathlib import Path
import sys


def grade(request):
    from livecodebench_utils import run_tests_for_one_example
    context=multiprocessing.get_context('spawn')
    tests=request['problem']['test']
    if not tests:
        raise ValueError('A LiveCodeBench problem must contain tests')
    with context.Manager() as manager:
        results=manager.list()
        process=context.Process(target=run_tests_for_one_example,args=(tests,request['completion'],results,request['is_extracted']))
        process.start()
        # Preserve the original test budget; fresh interpreter startup gets an
        # outer allowance, as in the validated HumanEval spawn correction.
        process.join((request['timeout']+1)*len(tests)+5+10)
        if process.is_alive():
            process.kill();process.join()
        elif process.exitcode!=0:
            raise RuntimeError(f'LiveCodeBench grading worker exited {process.exitcode}')
        values=list(results)
        for _ in range(len(tests)-len(values)):
            values.append((False,'Time out!.','Error: Time out!',float('inf')))
        return values


if __name__=='__main__':
    request,out=map(Path,sys.argv[1:])
    out.write_text(json.dumps(grade(json.loads(request.read_text())))+'\n')
