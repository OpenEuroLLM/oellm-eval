"""Create an isolated spawn-safe LiveCodeBench grading overlay; no GPU launch."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil

WRAPPER='''class LCBInfrastructureError(RuntimeError):
    pass


def lcb_run(problem, completion, timeout, is_extracted):
    import subprocess
    import tempfile
    from pathlib import Path
    try:
        with tempfile.TemporaryDirectory(prefix="lcb-grade-") as directory:
            root = Path(directory)
            request, output = root / "request.json", root / "result.json"
            request.write_text(json.dumps(dict(problem=problem, completion=completion,
                                              timeout=timeout, is_extracted=is_extracted), default=str))
            budget = (timeout + 1) * len(problem["test"]) + 5 + 10 + 30
            subprocess.run([sys.executable, str(Path(__file__).with_name("grade_one.py")),
                            str(request), str(output)], check=True, timeout=budget,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            return json.loads(output.read_text())
    except (subprocess.SubprocessError, OSError, ValueError) as error:
        raise LCBInfrastructureError(f"LiveCodeBench grading infrastructure failed: {error}") from error
'''

def prepare(source,out):
    rel=Path('eval/chat_benchmarks/LiveCodeBench')
    utility=(source/rel/'livecodebench_utils.py').read_text()
    tree=ast.parse(utility)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='lcb_run')
    before={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source/rel/'livecodebench_utils.py',source/rel/'eval_instruct.py',source/'eval/eval.py']}
    shutil.copytree(source,out,ignore=shutil.ignore_patterns('.git','__pycache__'))
    lines=utility.splitlines(keepends=True)
    (out/rel/'livecodebench_utils.py').write_text(''.join(lines[:node.lineno-1])+WRAPPER+'\n'+''.join(lines[node.end_lineno:]))
    shutil.copyfile(Path(__file__).with_name('lcb_grade_one.py'),out/rel/'grade_one.py')
    p=out/rel/'eval_instruct.py';text=p.read_text()
    text=text.replace('from .livecodebench_utils import lcb_run,','from .livecodebench_utils import LCBInfrastructureError, lcb_run,',1)
    for anchor in ('            except Exception as e:\n','        except Exception as outer_e:\n','                    except Exception as e:\n'):
        if text.count('\n'+anchor)!=1:
            raise ValueError('Unexpected exception-handler layout')
        indent=anchor[:len(anchor)-len(anchor.lstrip())]
        text=text.replace('\n'+anchor,'\n'+indent+'except LCBInfrastructureError:\n'+indent+'    raise\n'+anchor,1)
    marker='        # First, organize completions by repeat index\n'
    if text.count(marker)!=1:raise ValueError('Unexpected scoring layout')
    text=text.replace(marker,'''        import json
        from pathlib import Path
        artifact = os.environ.get("EVALCHEMY_LCB_ARTIFACT_DIR")
        if not artifact:
            raise LCBInfrastructureError("A result directory is required to retain all LCB generations")
        artifact = Path(artifact)
        artifact.mkdir(parents=True, exist_ok=True)
        with (artifact / "all_generated_responses.json").open("x") as stream:
            json.dump(responses, stream, default=str)

'''+marker,1)
    marker='            # Calculate metrics for this repeat\n'
    if text.count(marker)!=1:raise ValueError('Unexpected repeat metrics layout')
    text=text.replace(marker,'''            # Retain every verdict without duplicating large private test inputs.
            with (artifact / f"verdicts_repeat_{repeat_idx + 1}.jsonl").open("x") as stream:
                for index, (verdict, example) in enumerate(results):
                    stream.write(json.dumps(dict(index=index, difficulty=example["difficulty"],
                                                 correctness=verdict["correctness"], reason=verdict.get("reason", ""))) + "\\n")

'''+marker,1)
    p.write_text(text)
    p=out/'eval/eval.py';text=p.read_text()
    marker='            if task == "HumanEval" and args.output_path:\n'
    if text.count(marker)!=1:raise ValueError('Unexpected CLI layout')
    text=text.replace(marker,'            if task == "LiveCodeBench" and args.output_path:\n                os.environ["EVALCHEMY_LCB_ARTIFACT_DIR"] = os.path.join(args.output_path, "livecodebench_artifacts")\n'+marker,1)
    p.write_text(text)
    paths=[*before,str(rel/'grade_one.py')]
    after={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in paths}
    for name in paths:ast.parse((out/name).read_text())
    (out/'lcb_grading_manifest.json').write_text(json.dumps(dict(source=str(source),before=before,after=after),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','out'):p.add_argument('--'+name,type=Path,required=True)
    prepare(**vars(p.parse_args()))
