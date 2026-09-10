"""Verify all overlay hashes and print arguments for the pinned ARM container."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex


def arguments(manifest):
    data = json.loads(manifest.read_text())
    args = ["--env", "VLLM_USE_V2_MODEL_RUNNER=1"]
    for name, target in data["bind_targets"].items():
        path = manifest.parent / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != data["files"][str(path)]:
            raise ValueError(f"Changed overlay file: {path}")
        # The scheduler interpolates these arguments into shell templates;
        # reject unsafe paths rather than relying on a second shell parse.
        for value in (str(path), target):
            if re.fullmatch(r"/[A-Za-z0-9_./+\-]+", value) is None:
                raise ValueError("Container overlay paths must be shell-safe")
        args.extend(("--bind", f"{path}:{target}:ro"))
    return shlex.join(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    print(arguments(parser.parse_args().manifest.resolve()))
