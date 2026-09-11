import ast,errno,json,logging,os,sys,tempfile,time
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,os.environ['LCB_ROOT'])
from eval.chat_benchmarks.LiveCodeBench import eval_instruct as bench_module
from eval.chat_benchmarks.LiveCodeBench.livecodebench_utils import lcb_run,LCBInfrastructureError
problem={'test':[{'testtype':'stdin','input':'2\n','output':'4\n'}]}
if __name__=='__main__':
    started=time.monotonic()
    if os.environ.get('LCB_EXPECT_SMALL_TMP'):
        large=dict(test=[dict(testtype='stdin',input='2\n'+' '*32*1024*1024,output='4\n')])
        try:lcb_run(large,'print(int(input())*2)',1,False)
        except LCBInfrastructureError as error:
            assert isinstance(error.__cause__,OSError) and error.__cause__.errno==errno.ENOSPC
            print('PASS: actual 16 MiB container tmp exhaustion propagates as infrastructure failure')
            raise SystemExit(0)
        raise AssertionError('Expected actual ENOSPC')
    large=dict(test=[dict(testtype='stdin',input='2\n'+' '*32*1024*1024,output='4\n')])
    assert all(r[0] for r in lcb_run(large,'print(int(input())*2)',1,False))
    for target in ('tempfile.TemporaryDirectory','pathlib.Path.write_text'):
        with patch(target,side_effect=OSError(errno.ENOSPC,'simulated storage exhaustion')):
            try:lcb_run(problem,'print(4)',1,False)
            except LCBInfrastructureError:pass
            else:raise AssertionError('Storage failure became an incorrect answer')
    assert all(r[0] for r in lcb_run(problem,'print(int(input())*2)',1,False))
    assert not all(r[0] for r in lcb_run(problem,'print(0)',1,False))
    assert not all(r[0] for r in lcb_run(problem,'while True: pass',0,False))
    task=bench_module.LiveCodeBenchBenchmark(logger=logging.getLogger('fixture'))
    example=dict(problem,model_answer=['print(4)'],difficulty='easy',is_stdin=True)
    with patch.object(bench_module,'lcb_run',side_effect=LCBInfrastructureError('simulated worker crash')):
        try:task.evaluate_single_example(example)
        except LCBInfrastructureError:pass
        else:raise AssertionError('Infrastructure failure became an incorrect answer')
    original=Path(os.environ['LCB_ORIGINAL'])/'eval/chat_benchmarks/LiveCodeBench/livecodebench_utils.py'
    current=Path(os.environ['LCB_ROOT'])/'eval/chat_benchmarks/LiveCodeBench/livecodebench_utils.py'
    for name in ('run_tests_for_one_example','run_test_std','run_test_func','reliability_guard'):
        def body(path):return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(body(original))==ast.dump(body(current)),name
    print(json.dumps(dict(passed=['32_MiB_request','temporary_creation_ENOSPC','request_write_ENOSPC','correct','incorrect','timeout','infrastructure_propagation','unchanged_test_body'],seconds=time.monotonic()-started)))
