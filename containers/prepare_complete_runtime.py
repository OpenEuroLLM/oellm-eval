"""Assemble the previously validated updates for an immutable standalone image.

No packages are installed. Input source trees and engine hashes are verified;
existing sources/images are never edited. Run in the activated control env.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from build_vllm_image import sha
from continuation_scoring.prepare_hellaswag_optimization import TARGETS
from hf_generation_limit import patch as patch_hf_limit


def verify(root, base_image, source_manifest):
    data = json.loads((root / "complete-manifest.json").read_text())
    if data["base_sha256"] != sha(base_image) or data["source_manifest_sha256"] != sha(source_manifest):
        raise ValueError("Complete runtime inputs differ from this build")
    allowed = set(TARGETS.values()) | {
        "/opt/oellm-eval/lib/python3.12/site-packages/oellm_ifeval.py",
        "/opt/oellm-provenance/verify_complete_image.py",
        "/opt/oellm-provenance/check_complete_runtime_cpu.py",
        "/opt/oellm-eval/lib/python3.12/site-packages/lm_eval/models/huggingface.py",
    }
    for relative, item in data["files"].items():
        path = root / relative
        if not path.resolve().is_relative_to(root.resolve()) or not path.is_file() or sha(path) != item["sha256"]:
            raise ValueError(f"Complete runtime source mismatch: {relative}")
        target = item["target"]
        if target not in allowed and not (target.startswith("/opt/evalchemy/") and ".." not in Path(target).parts):
            raise ValueError(f"Unexpected installation target: {target}")
    if not set(TARGETS.values()).issubset({v["target"] for v in data["files"].values()}):
        raise ValueError("Incomplete engine/adapter installation")
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("base-image", "sources", "out"):
        p.add_argument("--"+name, required=True, type=Path)
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    a.base_image, a.sources, a.out = (x.resolve() for x in (a.base_image, a.sources, a.out))
    for path in (a.base_image, a.sources, a.out):
        if any(c.isspace() for c in str(path)) or any(c in str(path) for c in ",:\n"):
            p.error("Use absolute paths without whitespace, commas or colons")
    here = Path(__file__).resolve().parent
    manifest = a.sources / "source-manifest.json"
    print(json.dumps(dict(base_image=str(a.base_image), sources=str(a.sources), out=str(a.out),
                         updates=["continuation-scoring", "code-grading", "explicit-ifeval-policy"]), indent=2), flush=True)
    if not a.execute:
        return
    # Prove that preparers are consuming the exact reconstructed source tree.
    for name, record in json.loads(manifest.read_text()).items():
        for rel, digest in record["files"].items():
            path = a.sources/name/rel
            if not path.resolve().is_relative_to((a.sources/name).resolve()) or sha(path) != digest:
                raise ValueError(f"Source mismatch: {name}/{rel}")
    a.out.mkdir(parents=True, exist_ok=False)
    subprocess.run(["apptainer", "exec", "--cleanenv", "--containall", "--no-mount", "bind-paths,hostfs,cwd,home",
                    "--bind", f"{here}/continuation_scoring/read_engine_sources.py:/read-engine.py:ro",
                    "--bind", f"{a.out}:/work", str(a.base_image), "/opt/ray-venv/bin/python",
                    "/read-engine.py", "--out", "/work/engine"], check=True)
    subprocess.run([sys.executable, str(here/"continuation_scoring/prepare_hellaswag_optimization.py"),
                    "--engine-source", str(a.out/"engine"), "--harness-source", str(a.sources/"harness"),
                    "--out", str(a.out/"overlay")], check=True)
    subprocess.run([sys.executable, str(here/"code_grading/prepare.py"), "--source", str(a.sources/"evalchemy"),
                    "--out", str(a.out/"evalchemy")], check=True)
    files = {"overlay/"+name: dict(target=target, sha256=sha(a.out/"overlay"/name)) for name,target in TARGETS.items()}
    # Only changed grading files are copied over the base Evalchemy packaged by
    # the existing builder. Unchanged files remain source-manifest verified.
    grading = json.loads((a.out/"evalchemy/code_grading_manifest.json").read_text())
    for rel, digest in grading["files"].items():
        files["evalchemy/"+rel] = dict(target="/opt/evalchemy/"+rel, sha256=digest)
    for name, target in (("oellm_ifeval.py", "/opt/oellm-eval/lib/python3.12/site-packages/oellm_ifeval.py"),
                         ("verify_complete_image.py", "/opt/oellm-provenance/verify_complete_image.py"),
                         ("check_complete_runtime_cpu.py", "/opt/oellm-provenance/check_complete_runtime_cpu.py")):
        (a.out/name).write_bytes((here/name).read_bytes())
        files[name] = dict(target=target, sha256=sha(a.out/name))
    (a.out/"huggingface.py").write_text(patch_hf_limit((a.sources/"harness/lm_eval/models/huggingface.py").read_text()))
    files["huggingface.py"] = dict(target="/opt/oellm-eval/lib/python3.12/site-packages/lm_eval/models/huggingface.py", sha256=sha(a.out/"huggingface.py"))
    data = dict(schema=1, base_sha256=sha(a.base_image), source_manifest_sha256=sha(manifest), files=files,
                ifeval_default="eos", ifeval_choices=["eos", "continue"],
                recipe={str(f.relative_to(here)): sha(f) for f in here.rglob("*.py") if "__pycache__" not in str(f)})
    (a.out/"complete-manifest.json").write_text(json.dumps(data, indent=2)+"\n")
    verify(a.out, a.base_image, manifest)


if __name__ == "__main__":
    main()
