"""Reconstruct pinned runtime sources, including the reviewed Evalchemy patches.

Dry-run by default. Requires Git and a fresh output directory; no packages are
installed. The patched tree is checked against the tested Evalchemy tree.
"""
import argparse
import json
from pathlib import Path
import subprocess

from build_vllm_image import REVISIONS, sha

SOURCES = {
    "harness": ("https://github.com/EleutherAI/lm-evaluation-harness.git", "6d642546f4688648fced259eb3302efd36ece5af"),
    "evalchemy": ("https://github.com/Ali-Elganzory/evalchemy.git", "54ac97648230c4c3a22c3a2b93068b5a4e573f8d"),
    "human-eval": ("https://github.com/openai/human-eval.git", REVISIONS["human-eval"]),
}
HARNESS_TREE = "760c601bf39c8733901d7249b131399efd2c0403"
EVALCHEMY_TREE = "2d7ddf635cb11e0b0d4ec85e0d9fe56a58d2277b"


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(json.dumps({"sources": SOURCES, "patched_evalchemy_tree": EVALCHEMY_TREE, "patched_harness_tree": HARNESS_TREE,
                      "output": str(args.output)}, indent=2), flush=True)
    if not args.execute:
        return
    args.output.mkdir(parents=True, exist_ok=False)
    patches = Path(__file__).resolve().parent.parent / "patches"
    manifest = {}
    for name, (url, revision) in SOURCES.items():
        root = args.output / name
        root.mkdir()
        git(root, "init", "--quiet")
        git(root, "remote", "add", "origin", url)
        git(root, "fetch", "--depth=1", "origin", revision)
        git(root, "checkout", "--detach", "FETCH_HEAD")
        assert git(root, "rev-parse", "HEAD") == revision
        applied = {}
        if name == "evalchemy":
            for filename in ("evalchemy-modern-vllm.patch", "evalchemy-math-answer-parser.patch"):
                patch = patches / filename
                git(root, "apply", "--index", str(patch))
                applied[filename] = sha(patch)
            if git(root, "write-tree") != EVALCHEMY_TREE:
                raise RuntimeError("Patched Evalchemy tree differs from tested sources")
        if name == "harness":
            patch = patches / "harness-native-vllm-dp.patch"
            git(root, "apply", "--index", str(patch))
            applied[patch.name] = sha(patch)
            if git(root, "write-tree") != HARNESS_TREE:
                raise RuntimeError("Patched harness tree differs from tested sources")
        files = git(root, "ls-files", "-z").split("\0")
        manifest[name] = {"url": url, "upstream_revision": revision,
                          "revision": REVISIONS[name], "tree": git(root, "write-tree"),
                          "patches": applied,
                          "files": {p: sha(root / p) for p in files if p}}
    (args.output / "source-manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
