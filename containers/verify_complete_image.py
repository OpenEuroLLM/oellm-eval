"""Verify every integrated update inside the final image; CPU-only."""
import hashlib
import inspect
import json
from pathlib import Path
import sys


def main():
    root = Path("/opt/oellm-provenance")
    manifest = json.loads((root/"complete-runtime.json").read_text())
    for item in manifest["files"].values():
        path = Path(item["target"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], path
    assert Path(sys.prefix) == Path("/opt/oellm-eval")
    from lm_eval.models.huggingface import HFLM
    from lm_eval.models.vllm_causallms import VLLM
    from vllm.oellm_continuation import OELLM_CONTINUATION_API
    import oellm_ifeval
    assert "Some checkpoint exports omit EOS" in inspect.getsource(HFLM._model_generate)
    assert inspect.signature(VLLM).parameters["data_parallel_backend"].default == "mp"
    assert inspect.signature(VLLM).parameters["continuation_loglikelihood"].default is True
    kwargs, stops, limit = VLLM.modify_gen_kwargs({"do_sample":False}, "EOS", 1280)
    assert kwargs["temperature"] == 0
    assert oellm_ifeval.vllm_kwargs(kwargs, stops, "EOS", limit)[0]["ignore_eos"]
    from eval import eval
    print(json.dumps(dict(complete=True, files=len(manifest["files"]), hf=True, vllm=True,
                          continuation_api=OELLM_CONTINUATION_API, ifeval_default="eos")))


if __name__ == "__main__":
    main()
