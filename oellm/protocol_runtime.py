"""v0.02 runtime; also executable from the immutable submission snapshot.

Uses the installed evaluator and grader. Never execute generated code here.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import runpy
import sys


def verify_snapshot(root):
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["version"] != "v0.02":
        raise ValueError("Expected a v0.02 snapshot")
    for rel, expected in manifest["files"].items():
        path = root / rel
        if not path.resolve().is_relative_to(root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Frozen evaluation source changed: " + rel)
    return manifest


def generation_settings(task, settings, output_tokens=None):
    result = copy.deepcopy(settings)
    budgets = {"HumanEval": 4096, "MATH500": 32768}
    if task in budgets:
        result.pop("max_new_tokens", None)
        result.update(max_gen_toks=output_tokens or budgets[task], do_sample=False, temperature=0.0)
    elif output_tokens is not None:
        raise ValueError("Output override is only available for HumanEval/MATH500")
    if task == "MATH500":
        result["until"] = ["<|im_end|>"]
    return result


def ensure_context(model, requests):
    """Refuse silent prompt truncation or a reduced generation allowance."""
    for request in requests:
        prompt, settings = request.args
        tokens = model.tok_encode(prompt)
        budget = settings.get("max_gen_toks", settings.get("max_new_tokens", model.max_gen_toks))
        required = len(tokens) + int(budget)
        if required > model.max_length:
            raise ValueError(f"Complete prompt ({len(tokens)}) plus output ({budget}) requires {required} tokens; model context is {model.max_length}. Configure sufficient validated context/RoPE or select an explicitly labelled common output budget. No truncation performed.")


def configure_evalchemy(task, root, output_tokens):
    import evaluation_policies
    if task in ("HumanEval", "LiveCodeBench", "GPQADiamond"):
        evaluation_policies.install_evalchemy(task)
    from eval.task import BaseBenchmark
    original_normalize = BaseBenchmark._normalize_model_args

    def normalize(self, model, instances):
        # Move requested limits to max_gen_toks before Evalchemy's automatic cap.
        # These are vLLM-only releases; the adapter receives the whole allowance.
        for instance in instances:
            prompt, settings = instance.args
            settings = generation_settings(task, settings, output_tokens)
            if "max_new_tokens" in settings:
                settings["max_gen_toks"] = settings.pop("max_new_tokens")
            instance.arguments = (prompt, settings)
        return original_normalize(self, model, instances)

    BaseBenchmark._normalize_model_args = normalize
    if task == "MATH500":
        template = (root / "qwen3_calibration.jinja").read_text()

        def messages(self, messages, model=None):
            if model is None:
                raise ValueError("MATH500 v0.02 requires the local tokenizer")
            messages = copy.deepcopy(messages)
            if self.system_instruction:
                messages.insert(0, dict(role="system", content=self.system_instruction))
            return model.tokenizer.apply_chat_template(messages, chat_template=template,
                                                       tokenize=False, add_generation_prompt=True)
        BaseBenchmark._prepare_messages = messages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", choices=["lm_eval", "eval.eval", "oellm_ifeval"], required=True)
    parser.add_argument("--output-tokens", type=int)
    args, remaining = parser.parse_known_args()
    if args.output_tokens is not None and args.output_tokens < 1:
        parser.error("--output-tokens must be positive")
    root = Path(__file__).resolve().parent
    manifest = verify_snapshot(root)
    task = remaining[remaining.index("--tasks") + 1]
    if "," in task:
        raise ValueError("One task per versioned invocation is required")
    if "--limit" in remaining:
        raise ValueError("Release evaluations require full benchmark coverage")
    if remaining[remaining.index("--model") + 1] != "vllm":
        raise ValueError("v0.02 runtime requires vLLM")
    output = Path(remaining[remaining.index("--output_path") + 1])
    receipt_path = output.with_name(output.name + ".protocol.json")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt = dict(version="v0.02", task=task, module=args.module,
                   output_tokens_override=args.output_tokens,
                   files=manifest["files"], argv=remaining, status="started")
    with receipt_path.open("x") as stream:
        json.dump(receipt, stream, indent=2); stream.write("\n")
    if args.module == "eval.eval":
        configure_evalchemy(task, root, args.output_tokens)
    from lm_eval.models.vllm_causallms import VLLM
    original = VLLM.generate_until
    request_count = 0
    max_prompt_tokens = 0
    effective_settings = set()

    def generate(self, requests, *a, **kw):
        nonlocal request_count, max_prompt_tokens
        ensure_context(self, requests)
        for request in requests:
            max_prompt_tokens = max(max_prompt_tokens, len(self.tok_encode(request.args[0])))
            effective_settings.add(json.dumps(request.args[1], sort_keys=True))
        request_count += len(requests)
        return original(self, requests, *a, **kw)
    VLLM.generate_until = generate
    sys.argv = [args.module] + remaining
    try:
        try:
            runpy.run_module(args.module, run_name="__main__", alter_sys=True)
        except SystemExit as error:
            if error.code not in (None, 0):
                raise
        receipt["status"] = "completed"
    finally:
        receipt.update(requests=request_count, max_prompt_tokens=max_prompt_tokens,
                       effective_generation_settings=[json.loads(x) for x in sorted(effective_settings)])
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
