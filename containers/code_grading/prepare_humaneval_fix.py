"""Prepare an isolated, hash-checked Evalchemy HumanEval runtime correction."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import textwrap

EXPECTED = {
    'eval/chat_benchmarks/HumanEval/human_eval/execution.py': 'bdcf1c12ea4ecc5ce69df92b3c6cee8e18f5e30eba5879b225efe534ab1bfb6a',
    'eval/chat_benchmarks/HumanEval/eval_instruct.py': '34ef401cdb6c0e3bc925704d9b299f6521a51be47e1c1073664b6128096d288f',
    'eval/eval.py': '0eaf669947c7c477601abd3b71edcc3a37f938cb8df4fecec22a664414a5dd33',
    'eval/chat_benchmarks/HumanEval/human_eval/evaluation.py': '63f92eb61819681559e082fa66d703ebce177fad651294f09ff1052d47ef4883',
}


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Expected one occurrence: {old[:80]}')
    return text.replace(old, new, 1)


def prepare(source, target):
    if target.exists():
        raise FileExistsError(target)
    for name, digest in EXPECTED.items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Unreviewed source: {name}')
    shutil.copytree(source, target, ignore=shutil.ignore_patterns('.git', '__pycache__'))
    execution, benchmark, cli, scoring = [target/name for name in EXPECTED]
    text = execution.read_text()
    start = text.index('    def unsafe_execute(tmp_dir):')
    end = text.index('    manager = multiprocessing.Manager()', start)
    worker = textwrap.dedent(text[start:end]).replace(
        'def unsafe_execute(tmp_dir):',
        'def unsafe_execute(task_id, sample, language_type, timeout, tmp_dir, result):', 1)
    worker = replace_once(worker, '    if "python" in language_type.lower():',
                          '    if "python" in language_type.lower():\n'
                          '        # Fork inherited this standard grader dependency already imported.\n'
                          '        # Spawn must load it before reliability_guard disables OS helpers.\n'
                          '        import numpy\n')
    text = text[:start] + text[end:]
    text = text.replace('def check_correctness(', worker + '\ndef check_correctness(', 1)
    start = text.index('    manager = multiprocessing.Manager()')
    end = text.index('\n\n# Copyright', start)
    text = text[:start] + '''    # Spawn a fresh interpreter; never fork the multithreaded inference process.
    ctx = multiprocessing.get_context("spawn")
    with ctx.Manager() as manager:
        result = manager.list()
        p = ctx.Process(target=unsafe_execute,
                        args=(task_id, sample, language_type, timeout, tmp_dir, result))
        p.start()
        try:
            # The benchmark's execution timeout remains inside unsafe_execute.
            # This outer deadline also permits interpreter startup and cleanup.
            p.join(timeout=timeout + 10)
            if p.is_alive():
                p.kill()
                p.join()
                raise RuntimeError("HumanEval grading worker exceeded its outer deadline")
            if p.exitcode != 0 or not result:
                raise RuntimeError(f"HumanEval grading worker failed: exit={p.exitcode}")
            verdict = result[0]
        finally:
            if p.is_alive():
                p.kill()
                p.join()
    return {
        "task_id": task_id,
        "completion_id": completion_id,
        "result": verdict,
        "passed": verdict == "passed",
        "finish": -1 if "finish" not in sample else sample["finish"],
        "code": sample["test_code"],
    }
''' + text[end:]
    execution.write_text(text)
    shutil.copyfile(Path(__file__).with_name('humaneval_grade.py'),
                    benchmark.parent/'grade_responses.py')
    text = benchmark.read_text().replace('import tempfile\n', 'import tempfile\nimport subprocess\nimport sys\n')
    text = text.replace('from human_eval.evaluation import evaluate_functional_correctness\n', '')
    text = replace_once(text, '        temp_dir_obj = tempfile.TemporaryDirectory()\n        temp_dir = temp_dir_obj.name',
                        '        artifact_root = os.environ.get("EVALCHEMY_ARTIFACT_DIR")\n'
                        '        if artifact_root:\n            Path(artifact_root).mkdir(parents=True, exist_ok=True)\n'
                        '        temp_dir = tempfile.mkdtemp(prefix="humaneval-", dir=artifact_root)')
    text = text.replace('                    self.logger.warning(f"Dataset file not found: {problem_file}")\n                    continue',
                        '                    raise FileNotFoundError(problem_file)')
    text = text.replace('                generated_examples = []',
                        '                if len(outputs) != len(examples):\n'
                        '                    raise ValueError("Incomplete HumanEval generation")\n'
                        '                generated_examples = []')
    text = text.replace('                self.logger.error(f"Error processing language {lang}: {str(e)}")\n                continue',
                        '                self.logger.exception(f"Error processing language {lang}; artifacts: {temp_dir}")\n                raise')
    text = text.replace('results["temp_dir_obj"] = temp_dir_obj', 'results["artifact_dir"] = temp_dir')
    text = replace_once(text, '        temp_dir_obj = results["temp_dir_obj"]\n        temp_dir = temp_dir_obj.name',
                        '        temp_dir = results["artifact_dir"]')
    text = text.replace('                    self.logger.warning(f"Generated file not found: {temp_file_path}")\n                    continue',
                        '                    raise FileNotFoundError(temp_file_path)')
    start = text.index('                result = evaluate_functional_correctness(')
    end = text.index('\n                for metric, value', start)
    text = text[:start] + '''                request = dict(input_file=temp_file_path, tmp_dir=temp_dir,
                               n_workers=self.num_workers, timeout=self.timeout,
                               problem_file=str(Path(problem_file).resolve()), language=lang)
                request_path = Path(temp_dir) / f"request_{lang}.json"
                score_path = Path(temp_dir) / f"scores_{lang}.json"
                request_path.write_text(json.dumps(request))
                # A lightweight CLI keeps spawned graders from importing eval.eval/vLLM.
                with (Path(temp_dir) / f"grading_{lang}.log").open("w") as log:
                    subprocess.run([sys.executable, str(Path(__file__).with_name("grade_responses.py")),
                                    str(request_path), str(score_path)],
                                   check=True, stdout=log, stderr=subprocess.STDOUT)
                result = json.loads(score_path.read_text())
''' + text[end:]
    text = text.replace('                self.logger.error(f"Error evaluating {lang}: {str(e)}")\n                continue',
                        '                self.logger.exception(f"Error evaluating {lang}; artifacts: {temp_dir}")\n                raise')
    text = text.replace('        temp_dir_obj.cleanup()\n', '')
    text = text.replace('            return {"error": str(e)}', '            raise')
    benchmark.write_text(text)
    text = cli.read_text()
    text = replace_once(text, '            result = method(lm)',
                        '            if task == "HumanEval" and args.output_path:\n'
                        '                os.environ["EVALCHEMY_ARTIFACT_DIR"] = os.path.join(args.output_path, "humaneval_artifacts")\n'
                        '            result = method(lm)')
    text = replace_once(text, '                    results["results"][task] = result',
                        '                    if task == "HumanEval" and (not result or "error" in result):\n'
                        '                        raise RuntimeError("HumanEval did not produce valid scores")\n'
                        '                    results["results"][task] = result')
    cli.write_text(text)
    text = replace_once(scoring.read_text(), '    # Calculate pass@k.',
                        '    with open(input_file + ".graded.jsonl", "w") as stream:\n'
                        '        for task_id in sorted(results):\n'
                        '            for _, verdict in results[task_id]:\n'
                        '                stream.write(json.dumps(verdict) + "\\n")\n\n'
                        '    # Calculate pass@k.')
    scoring.write_text(text)
    files = list(EXPECTED) + ['eval/chat_benchmarks/HumanEval/grade_responses.py']
    manifest = dict(source=str(source), target=str(target), before=EXPECTED,
                    after={name: hashlib.sha256((target/name).read_bytes()).hexdigest() for name in files})
    (target/'humaneval_fix_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source.resolve(), args.out.resolve()), indent=2))
