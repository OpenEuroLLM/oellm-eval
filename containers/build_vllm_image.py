"""Pack a verified evaluation runtime over a SHA-pinned machine engine image.

Defaults to a JSON plan. --execute verifies inputs and builds in local /tmp.
The result still requires real GPU evaluation before promotion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tarfile

REVISIONS = {
    # Reconstructed upstream-plus-patches identity, not an upstream commit SHA.
    "harness": "6d642546f4688648fced259eb3302efd36ece5af+tree.9b7bbc80fbec2a891efd0fa496707cdab6c96c10",
    "evalchemy": "b321416135050aa6919b430dbadbd4cc43cc8c15",
    "human-eval": "6d43fb980f9fee3c892a914eda09951f772ad10d",
}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pack_runtime(runtime, output):
    # %files dereferences host symlinks, but the interpreter exists only in
    # the base image. Preserve links in a tar and extract inside that image.
    with tarfile.open(output, "w", dereference=False) as archive:
        archive.add(runtime / "venv", arcname=".")


def definition(args, profile):
    # Apptainer definition fields are line-oriented. Refuse ambiguous paths.
    for path in (args.base_image, args.runtime, args.evalchemy, args.humaneval, args.source_manifest):
        if any(c.isspace() for c in str(path)):
            raise ValueError("Build input paths must not contain whitespace")
    complete = getattr(args, "complete_runtime", None)
    extra_files = f"    {complete} /opt/oellm-updates\n" if complete else ""
    extra_post = ""
    if complete:
        extra_post = """    /opt/oellm-eval/bin/python - <<'PY'
import hashlib, json, shutil
from pathlib import Path
root=Path('/opt/oellm-updates')
manifest=json.loads((root/'complete-manifest.json').read_text())
for relative, item in manifest['files'].items():
    source=root/relative
    assert hashlib.sha256(source.read_bytes()).hexdigest()==item['sha256'], source
    target=Path(item['target'])
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
shutil.copyfile(root/'complete-manifest.json', '/opt/oellm-provenance/complete-runtime.json')
PY
    rm -rf /opt/oellm-updates
    export VLLM_USE_V2_MODEL_RUNNER=1
    /opt/oellm-eval/bin/python /opt/oellm-provenance/verify_complete_image.py
    /opt/oellm-eval/bin/python /opt/oellm-provenance/check_complete_runtime_cpu.py
"""
    return f'''Bootstrap: localimage
From: {args.base_image}

%labels
    oellm.machine {args.machine}
    oellm.base.sha256 {profile['base_sha256']}
    oellm.harness.revision {REVISIONS['harness']}
    oellm.evalchemy.revision {REVISIONS['evalchemy']}
    oellm.humaneval.revision 6d43fb980f9fee3c892a914eda09951f772ad10d
    oellm.status gpu-validation-required

%files
    {args.tmp_dir}/runtime-venv.tar /opt/runtime-venv.tar
    {args.runtime}/nltk_data /opt/nltk_data
    {args.runtime}/provenance /opt/oellm-provenance
    {args.source_manifest} /opt/oellm-provenance/source-manifest.json
    {args.evalchemy} /opt/evalchemy
    {args.humaneval} /opt/human-eval
{extra_files}
%post
    set -eu
    mkdir -p /opt/oellm-eval
    tar -xf /opt/runtime-venv.tar -C /opt/oellm-eval
    rm /opt/runtime-venv.tar
    export PYTHONNOUSERSITE=1
    unset PYTHONPATH
    /opt/oellm-eval/bin/python - <<'PY'
from pathlib import Path
import sys
root=Path('/opt/oellm-eval')
assert Path(sys.prefix) == root
# Relocate text launchers only; never rewrite installed binaries.
old={str(args.runtime / 'venv')!r}
for path in (root/'bin').iterdir():
    if path.is_symlink() or not path.is_file():
        continue
    try:
        text=path.read_text()
    except UnicodeDecodeError:
        continue
    if old in text:
        path.write_text(text.replace(old, str(root)))
site=root/'lib/python3.12/site-packages'
(site/'human-eval-source.pth').write_text('/opt/human-eval\\n')
PY
    export PATH=/opt/oellm-eval/bin:$PATH
    export VIRTUAL_ENV=/opt/oellm-eval
    export PYTHONPATH=/opt/evalchemy
    export NLTK_DATA=/opt/nltk_data
    python -m pip check > /opt/oellm-provenance/image-pip-check.txt 2>&1 || true
    python - <<'PY'
from pathlib import Path
import importlib.metadata as m
import pip, sys
assert Path(pip.__file__).is_relative_to(sys.prefix)
known='torch 2.13.0+cu130 has requirement nvidia-nccl-cu13==2.29.7; platform_system == "Linux", but you have nvidia-nccl-cu13 2.30.7.'
lines=Path('/opt/oellm-provenance/image-pip-check.txt').read_text().splitlines()
assert not [line for line in lines if line not in (known, 'No broken requirements found.')], lines
assert m.version('vllm') == {profile['vllm']!r}
assert m.version('ray') == {profile['ray']!r}
from lm_eval.models.vllm_causallms import VLLM
import inspect
assert inspect.signature(VLLM).parameters["data_parallel_backend"].default == "mp"
from eval import eval
from human_eval.evaluation import evaluate_functional_correctness
PY
    python /opt/evalchemy/tests/test_vllm_dp_context.py
    python /opt/evalchemy/tests/test_math_answer_parser.py
{extra_post}    python -m eval.eval --help > /opt/oellm-provenance/image-evalchemy-help.txt
    python -m lm_eval --help > /opt/oellm-provenance/image-harness-help.txt

%environment
    {'export VLLM_USE_V2_MODEL_RUNNER=1' if complete else ''}
    export PATH=/opt/oellm-eval/bin:$PATH
    export VIRTUAL_ENV=/opt/oellm-eval
    export PYTHONPATH=/opt/evalchemy
    export PYTHONNOUSERSITE=1
    export NLTK_DATA=/opt/nltk_data

%runscript
    exec "$@"
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", choices=["jupiter", "juwels_booster"], required=True)
    for name in ("base-image", "runtime", "evalchemy", "humaneval", "source-manifest", "output", "tmp-dir"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--complete-runtime", type=Path, help="Directory from prepare_complete_runtime.py; packages all validated updates")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--profiles", type=Path, default=Path(__file__).with_name("vllm_profiles.json"),
                        help="Verified base identities; use bootstrap_jupiter_base.py output for a fresh OCI build")
    args = parser.parse_args()
    for name in ("base_image", "runtime", "evalchemy", "humaneval", "source_manifest", "output", "tmp_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.complete_runtime:
        args.complete_runtime = args.complete_runtime.resolve()
        if any(c.isspace() for c in str(args.complete_runtime)):
            parser.error("Complete runtime path must not contain whitespace")
    profile = json.loads(args.profiles.read_text())[args.machine]
    if not args.tmp_dir.is_relative_to("/tmp") or args.tmp_dir == Path("/tmp"):
        parser.error("--tmp-dir must be a dedicated directory under local /tmp")
    recipe = definition(args, profile)
    command = ["apptainer", "build", "--mksquashfs-args", "-processors 2", str(args.output), str(args.output.with_suffix(".def"))]
    plan = {"machine": args.machine, "base": profile, "command": command,
            "profiles_sha256": sha(args.profiles),
            "runtime": str(args.runtime), "definition": recipe,
            "promotion": "Not promoted: real GPU evaluation is required."}
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return
    if platform.machine() != profile["architecture"]:
        parser.error("Host architecture does not match the selected image")
    if args.output.exists():
        parser.error("Output already exists; choose a new filename")
    if args.base_image.stat().st_size != profile["base_bytes"] or sha(args.base_image) != profile["base_sha256"]:
        parser.error("Base image size or SHA-256 mismatch")
    if not (args.runtime / "provenance/prepare-complete").is_file():
        parser.error("Runtime preparation has not passed")
    sources = json.loads(args.source_manifest.read_text())
    for name, root in (("evalchemy", args.evalchemy), ("human-eval", args.humaneval),
                       ("harness", args.runtime / "venv/lib/python3.12/site-packages")):
        if sources[name]["revision"] != REVISIONS[name]:
            parser.error(f"Unexpected source revision for {name}")
        for relative, expected in sources[name]["files"].items():
            if name == "harness" and not (relative.startswith("lm_eval/") and relative.endswith((".py", ".yaml"))):
                continue
            path = root / relative
            if not path.resolve().is_relative_to(root.resolve()) or not path.is_file() or sha(path) != expected:
                parser.error(f"Source content mismatch: {name}/{relative}")
    if args.complete_runtime:
        from prepare_complete_runtime import verify
        plan["complete_runtime"] = verify(args.complete_runtime, args.base_image, args.source_manifest)
        plan["complete_manifest_sha256"] = sha(args.complete_runtime / "complete-manifest.json")
    plan["source_manifest_sha256"] = sha(args.source_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.tmp_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    pack_runtime(args.runtime, args.tmp_dir / "runtime-venv.tar")
    args.output.with_suffix(".def").write_text(recipe)
    env = dict(os.environ, TMPDIR=str(args.tmp_dir), APPTAINER_TMPDIR=str(args.tmp_dir),
               APPTAINER_CACHEDIR=str(args.runtime.parent / "apptainer-cache"))
    subprocess.run(command, env=env, check=True)
    plan.update({"output_sha256": sha(args.output), "output_bytes": args.output.stat().st_size,
                 "definition_sha256": sha(args.output.with_suffix(".def"))})
    args.output.with_suffix(".provenance.json").write_text(json.dumps(plan, indent=2)+"\n")


if __name__ == "__main__":
    main()
