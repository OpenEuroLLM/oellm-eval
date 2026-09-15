"""CPU-only installed-evaluator acceptance; run inside the evaluation image."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--gsm-samples", type=Path, required=True)
    p.add_argument("--model-view", type=Path, required=True)
    p.add_argument("--gpqa-helper", type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.snapshot))
    from protocol_runtime import verify_snapshot, configure_evalchemy
    import evaluation_policies
    verify_snapshot(a.snapshot)
    from lm_eval.tasks import TaskManager, get_task_dict
    manager = TaskManager(include_path=str(a.snapshot / "tasks"))
    names = ["gsm8k_cot_numeric_v1", "squadv2_abstention_v1", "gpqa_diamond_cot0_v2"]
    tasks = get_task_dict(names, task_manager=manager)
    assert len(tasks["gsm8k_cot_numeric_v1"].test_docs()) == 1319
    assert len(tasks["squadv2_abstention_v1"].validation_docs()) == 11873
    assert len(tasks["gpqa_diamond_cot0_v2"].validation_docs()) == 198
    # Same calibrated GSM extraction and all 1319 numeric verdicts.
    seen = set(); correct = 0
    for line in a.gsm_samples.open():
        row = json.loads(line)
        if row.get("filter") != "flexible-extract":
            continue
        answer = row["filtered_resps"][0]
        score = evaluation_policies.gsm8k_results(row["doc"], [answer])
        assert score["numeric_match"] == row["numeric_match"]
        assert score["exact_match"] == row["exact_match"]
        seen.add(row["doc_id"]); correct += score["numeric_match"]
    assert len(seen) == 1319
    assert correct == 1187
    # The new zero-shot task produces exactly the accepted calibration prompt
    # and option order; demonstrations are deliberately absent at zero shots.
    import importlib.util
    spec = importlib.util.spec_from_file_location("old_gpqa", a.gpqa_helper)
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    task = tasks["gpqa_diamond_cot0_v2"]
    old_docs = old.gpqa_docs(task.dataset["train"])
    for doc, previous in zip(task.validation_docs(), old_docs, strict=True):
        assert doc["choices"] == previous["choices"] and doc["answer"] == previous["answer"]
        assert task.doc_to_text(doc) == old.gpqa_text(previous)
    # Confirm template equality through the installed adapter, without engines.
    from transformers import AutoTokenizer
    from lm_eval.models.vllm_causallms import VLLM
    from eval.task import BaseBenchmark
    from eval.chat_benchmarks.MATH500.eval_instruct import MATH500Benchmark, PROMPT
    tokenizer = AutoTokenizer.from_pretrained(str(a.model_view), local_files_only=True)
    fake = SimpleNamespace(tokenizer=tokenizer, hf_chat_template=tokenizer.chat_template,
                           enable_thinking=None, chat_template_args={})
    benchmark = MATH500Benchmark()
    expected = [VLLM.apply_chat_template(fake, [dict(role="user", content=PROMPT.format(problem=d["problem"]))])
                for d in benchmark.load_questions()]
    configure_evalchemy("MATH500", a.snapshot, None)
    actual = [benchmark._prepare_messages([dict(role="user", content=PROMPT.format(problem=d["problem"]))], fake)
              for d in benchmark.load_questions()]
    assert expected == actual and len(actual) == 500
    from lm_eval.api.instance import Instance
    request = Instance("generate_until", {}, ("test", dict(max_new_tokens=32768, temperature=.7, seed=[0,1,2,3])), 0)
    result = BaseBenchmark._normalize_model_args(benchmark, fake, [request])
    assert result[0].args[1]["max_gen_toks"] == 32768
    assert result[0].args[1]["temperature"] == 0
    print(json.dumps(dict(tasks=names, gsm_questions=len(seen), gsm_numeric_correct=correct,
                          gpqa_identical_prompts=198, math_identical_chat_prompts=500,
                          normalization="full output allowance, explicit greedy", gpu_execution=False)))


if __name__ == "__main__":
    main()
