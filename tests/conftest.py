"""Shared pytest fixtures."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent


@pytest.fixture
def tmp_splits(tmp_path):
    """Return a factory for writing split JSON files into a temp directory."""

    def _write(splits: dict[str, list[str]]) -> Path:
        for name, ids in splits.items():
            d = {"seed": 42, "created": "2026-10-21", "split": name, "scene_ids": ids}
            (tmp_path / f"{name}.json").write_text(json.dumps(d))
        return tmp_path

    return _write


def pytest_collection_modifyitems(config, items):
    """Skip tests marked @pytest.mark.torch when the [ml] extra is not installed."""
    import importlib.util

    if all(importlib.util.find_spec(m) is not None for m in ("torch", "transformers")):
        return
    skip = pytest.mark.skip(reason="torch/transformers not installed ([ml] extra)")
    for item in items:
        if "torch" in item.keywords:
            item.add_marker(skip)
