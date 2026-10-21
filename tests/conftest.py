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
