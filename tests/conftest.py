import os

import pytest


def pytest_collection_modifyitems(config, items):
    """Skip test_datasets tests when HF_TOKEN is not set (avoids rate-limiting)."""
    if os.environ.get("HF_TOKEN"):
        return
    skip = pytest.mark.skip(reason="HF_TOKEN not set; skipping HuggingFace API tests")
    for item in items:
        if item.fspath.basename == "test_datasets.py":
            item.add_marker(skip)
