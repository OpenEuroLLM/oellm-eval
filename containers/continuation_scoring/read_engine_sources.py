"""Extract the three installed engine source files without importing GPU code.

Run inside the selected evaluation container with an explicit writable output
mount. The preparer subsequently checks each source hash before patching.
"""
import argparse
from importlib.util import find_spec
from pathlib import Path


def extract(out):
    spec = find_spec("vllm")
    if spec is None or not spec.origin:
        raise RuntimeError("vLLM is not installed in this interpreter")
    root = Path(spec.origin).parent
    relative = (
        "v1/worker/gpu/sample/prompt_logprob.py",
        "v1/engine/logprobs.py",
        "v1/core/kv_cache_manager.py",
    )
    sources = [(root / name, (root / name).read_bytes()) for name in relative]
    out.mkdir(parents=True, exist_ok=False)
    for source, data in sources:
        (out / source.name).write_bytes(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    extract(parser.parse_args().out)
