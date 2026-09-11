"""Explicit full-response IFEval continuation, for HF and vLLM.

CLI-compatible with `python -m lm_eval`; only the exact full IFEval task is
accepted. The ordinary EOS policy remains `python -m lm_eval`.
"""
import json
import os
from pathlib import Path
import runpy
import sys


def hf_kwargs(kwargs, stops, eos):
    if any(stop != eos for stop in stops):
        raise ValueError("IFEval continuation cannot discard task-specific stops")
    if kwargs.get("do_sample", False) or kwargs.get("temperature", 0) not in (None, 0):
        raise ValueError("IFEval continuation requires greedy decoding")
    return dict(kwargs, eos_token_id=None, forced_eos_token_id=None), []


def vllm_kwargs(kwargs, stops, eos, limit):
    if any(stop != eos for stop in stops) or kwargs.get("temperature") != 0 or limit != 1280:
        raise ValueError("IFEval continuation requires greedy decoding, empty task stops and 1280 tokens")
    return dict(kwargs, ignore_eos=True, skip_special_tokens=True), [], limit


def validate_args(argv):
    def value(key):
        if argv.count(key) != 1 or argv.index(key)+1 == len(argv):
            raise ValueError(f"Exactly one {key} argument is required")
        return argv[argv.index(key)+1]
    if value("--tasks") != "ifeval" or value("--model") not in ("hf", "vllm"):
        raise ValueError("Explicit continuation supports only full IFEval with HF or vLLM")
    if any(x.split("=", 1)[0] in {"--limit", "--gen_kwargs", "--predict_only", "--samples"} for x in argv):
        raise ValueError("Do not override coverage, grading or generation in the fixed IFEval policy")
    return value("--model"), Path(value("--output_path"))


def install(backend):
    if backend == "hf":
        from lm_eval.models.huggingface import HFLM
        original = HFLM._model_generate
        def generate(self, context, max_length, stop, **kwargs):
            if max_length-context.shape[1] != 1280:
                raise ValueError("Unexpected HF IFEval generation budget")
            kwargs, stop = hf_kwargs(kwargs, stop, self.tokenizer.eos_token)
            result = original(self, context, max_length, stop, **kwargs)
            if result.shape[1]-context.shape[1] != 1280:
                raise ValueError("HF stopped before the declared continuation budget")
            return result
        HFLM._model_generate = generate
    else:
        from lm_eval.models.vllm_causallms import VLLM
        original = VLLM.modify_gen_kwargs
        generate = VLLM._model_generate
        def modify(gen_kwargs, eos, default_max_gen_toks):
            kwargs, stops, limit = original(gen_kwargs, eos, default_max_gen_toks)
            return vllm_kwargs(kwargs, stops, eos, limit)
        def checked(self, requests, *args, **kwargs):
            result = generate(self, requests, *args, **kwargs)
            for request in result:
                if len(request.outputs[0].token_ids) != 1280 or request.outputs[0].finish_reason != "length":
                    raise ValueError("vLLM stopped before the declared continuation budget")
            return result
        VLLM.modify_gen_kwargs = staticmethod(modify)
        VLLM._model_generate = checked


def main():
    backend, output = validate_args(sys.argv[1:])
    output.parent.mkdir(parents=True, exist_ok=True)
    # Retain an adjacent explicit policy record even if inference fails.
    policy = output.with_name(output.name+".ifeval-policy-rank"+os.environ.get("RANK", "0")+".json")
    with policy.open("x") as stream:
        json.dump(dict(policy="continue", backend=backend, greedy=True, max_new_tokens=1280,
                       ignore_eos=True, skip_special_tokens=True, full_response=True), stream, indent=2)
        stream.write("\n")
    install(backend)
    runpy.run_module("lm_eval", run_name="__main__")


if __name__ == "__main__":
    main()
