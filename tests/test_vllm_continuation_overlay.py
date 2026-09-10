"""Fail before submission when a frozen overlay changes or paths are unsafe."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "continuation_bind_args", ROOT / "containers/continuation_scoring/bind_args.py"
)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class OverlayTests(unittest.TestCase):
    def manifest(self, root, name="helper.py"):
        source = root / name
        source.write_text("# frozen source\n")
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({
            "files": {str(source): hashlib.sha256(source.read_bytes()).hexdigest()},
            "bind_targets": {name: "/opt/engine/helper.py"},
        }))
        return source, manifest

    def test_changed_source_fails_before_arguments_are_returned(self):
        with tempfile.TemporaryDirectory() as directory:
            source, manifest = self.manifest(Path(directory))
            args = helper.arguments(manifest)
            self.assertIn("VLLM_USE_V2_MODEL_RUNNER=1", args)
            self.assertIn(":/opt/engine/helper.py:ro", args)
            source.write_text("# changed after review\n")
            with self.assertRaises(ValueError):
                helper.arguments(manifest)

    def test_paths_unsafe_for_scheduler_interpolation_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            _, manifest = self.manifest(Path(directory), "helper;unsafe.py")
            with self.assertRaises(ValueError):
                helper.arguments(manifest)
