"""Freeze explicitly selected evaluation procedures beside each submission."""
import hashlib
import json
from pathlib import Path
import shutil

VERSIONS = ("v0.01", "v0.02")
MAPPING = {
    "gsm8k": ("gsm8k_cot_numeric_v1", 4, "lm-eval-harness"),
    "squadv2": ("squadv2_abstention_v1", None, "lm-eval-harness"),
    "GPQADiamond": ("gpqa_diamond_cot0_v2", 0, "lm-eval-harness"),
}


def resolve_job(job, version):
    if version not in VERSIONS:
        raise ValueError("Unknown eval_version: " + version)
    result = dict(job)
    if version == "v0.02" and result["task_path"] in MAPPING:
        task, shots, suite = MAPPING[result["task_path"]]
        if result["task_path"] == "gsm8k" and int(result["n_shot"]) != 4:
            raise ValueError("v0.02 GSM8K uses the calibrated fixed four demonstrations")
        if result["task_path"] == "GPQADiamond" and int(result["n_shot"]) != 0:
            raise ValueError("v0.02 GPQA calibration is zero-shot")
        result.update(task_path=task, eval_suite=suite)
        if shots is not None:
            result["n_shot"] = shots
    return result


def freeze(destination, version, original_jobs, resolved_jobs, output_tokens=None):
    """Copy runtime and custom tasks, never bind a mutable development checkout."""
    destination = Path(destination)
    destination.mkdir(exist_ok=False)
    here = Path(__file__).resolve().parent
    if version == "v0.02":
        for name in ("protocol_runtime.py", "evaluation_policies.py"):
            shutil.copyfile(here / name, destination / name)
        shutil.copytree(here / "resources/custom_lm_eval_tasks", destination / "tasks",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copyfile(here / "resources/qwen3_calibration.jinja", destination / "qwen3_calibration.jinja")
    manifest = dict(version=version, original_jobs=original_jobs, resolved_jobs=resolved_jobs,
                    output_tokens_override=output_tokens,
                    files={str(p.relative_to(destination)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(destination.rglob("*")) if p.is_file()})
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return destination


def result_identity(path):
    """Unlabelled historical inputs stay unlabelled; never infer version by score."""
    for parent in Path(path).resolve().parents:
        manifest = parent / "protocol/manifest.json"
        if manifest.is_file():
            data = json.loads(manifest.read_text())
            budget = data.get("output_tokens_override")
            return data["version"], ("default" if budget is None else f"output-{budget}")
    return "unversioned", "unspecified"
