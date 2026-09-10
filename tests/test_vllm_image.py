import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile

SOURCE = Path(__file__).resolve().parents[1] / "containers/build_vllm_image.py"
spec = importlib.util.spec_from_file_location("vllm_image_builder", SOURCE)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def command(root):
    return [sys.executable, str(SOURCE), "--machine", "jupiter",
            "--base-image", str(root / "base.sif"), "--runtime", str(root / "runtime"),
            "--evalchemy", str(root / "evalchemy"), "--humaneval", str(root / "humaneval"),
            "--source-manifest", str(root / "sources.json"), "--output", str(root / "out.sif"),
            "--tmp-dir", "/tmp/test-oellm-image-build"]


def test_plan_does_not_require_inputs_or_run_apptainer(tmp_path):
    result = subprocess.run(command(tmp_path), capture_output=True, text=True, check=True)
    plan = json.loads(result.stdout)
    assert plan["machine"] == "jupiter"
    assert "gpu-validation-required" in plan["definition"]
    assert not list(tmp_path.iterdir())


def test_shared_filesystem_build_root_is_rejected(tmp_path):
    cmd = command(tmp_path)
    cmd[-1] = "/p/scratch/too-long-for-ray"
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode != 0
    assert "local /tmp" in result.stderr


def test_existing_image_is_never_replaced(tmp_path):
    (tmp_path / "out.sif").write_bytes(b"existing image")
    cmd = command(tmp_path)
    # Use the running host's architecture to reach the existing-output guard.
    if builder.platform.machine() == "x86_64":
        cmd[cmd.index("jupiter")] = "juwels_booster"
    result = subprocess.run(cmd + ["--execute"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "Output already exists" in result.stderr
    assert (tmp_path / "out.sif").read_bytes() == b"existing image"


def test_archive_preserves_interpreter_that_exists_only_inside_image(tmp_path):
    runtime = tmp_path / "runtime"
    binary = runtime / "venv/bin"
    binary.mkdir(parents=True)
    (binary / "python3.12").symlink_to("/image-only/python3.12")
    (binary / "python").symlink_to("python3.12")
    (runtime / "venv/pyvenv.cfg").write_text("include-system-site-packages = true\n")
    archive = tmp_path / "runtime.tar"
    builder.pack_runtime(runtime, archive)
    with tarfile.open(archive) as stream:
        assert stream.getmember("./bin/python3.12").issym()
        assert stream.getmember("./bin/python3.12").linkname == "/image-only/python3.12"
        assert stream.getmember("./bin/python").linkname == "python3.12"
        assert stream.extractfile("./pyvenv.cfg").read().startswith(b"include-system-site")
