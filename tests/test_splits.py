"""Split leakage tests.

These tests enforce the core evaluation rule: no scene may appear in more
than one split partition (train / val / test). Splits are keyed by scene ID,
never by frame.
"""

import json
from itertools import combinations
from pathlib import Path

import pytest

SPLITS_DIR = Path(__file__).parent.parent / "data" / "splits"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_splits(splits_dir: Path) -> dict[str, list[str]]:
    """Return {split_name: [scene_ids]} for every standard split *.json in splits_dir.

    Files that don't have a top-level 'scene_ids' key (e.g. synthetic_splits.json
    which uses a different LOSO structure) are silently skipped.
    """
    splits: dict[str, list[str]] = {}
    for p in sorted(splits_dir.glob("*.json")):
        with p.open() as f:
            data = json.load(f)
        if "scene_ids" not in data:
            continue
        name = data.get("split", p.stem)
        splits[name] = list(data["scene_ids"])
    return splits


def assert_no_leakage(splits: dict[str, list[str]]) -> None:
    """Raise AssertionError if any scene_id appears in more than one split."""
    for (name_a, ids_a), (name_b, ids_b) in combinations(splits.items(), 2):
        overlap = set(ids_a) & set(ids_b)
        assert not overlap, f"Leakage: {sorted(overlap)} appear in both '{name_a}' and '{name_b}'"


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestSplitSchema:
    def test_required_keys_present(self, tmp_splits):
        d = tmp_splits({"train": ["scene_001", "scene_002"]})
        with (d / "train.json").open() as f:
            data = json.load(f)
        for key in ("seed", "split", "scene_ids"):
            assert key in data

    def test_scene_ids_is_list(self, tmp_splits):
        d = tmp_splits({"train": ["scene_001"]})
        with (d / "train.json").open() as f:
            data = json.load(f)
        assert isinstance(data["scene_ids"], list)

    def test_no_duplicates_within_split(self, tmp_splits):
        # Manually inject a duplicate to confirm detection
        p = tmp_splits({"train": ["scene_001"]}) / "train.json"
        data = json.loads(p.read_text())
        data["scene_ids"] = ["scene_001", "scene_001"]
        p.write_text(json.dumps(data))

        splits = load_splits(p.parent)
        ids = splits["train"]
        assert len(ids) != len(set(ids)), "Fixture setup failed"

        with pytest.raises(AssertionError, match="scene_001"):
            for name, scene_ids in splits.items():
                dupes = [sid for sid in set(scene_ids) if scene_ids.count(sid) > 1]
                assert not dupes, f"Duplicate scene IDs in '{name}': {dupes}"


# ---------------------------------------------------------------------------
# Leakage tests
# ---------------------------------------------------------------------------


class TestNoLeakage:
    def test_clean_splits_pass(self, tmp_splits):
        d = tmp_splits(
            {
                "train": ["scene_001", "scene_002", "scene_003"],
                "val": ["scene_004", "scene_005"],
                "test": ["scene_006"],
            }
        )
        assert_no_leakage(load_splits(d))

    def test_train_val_overlap_detected(self, tmp_splits):
        # scene_003 in both train and val
        d = tmp_splits(
            {
                "train": ["scene_001", "scene_002", "scene_003"],
                "val": ["scene_003", "scene_004"],
                "test": ["scene_005"],
            }
        )
        with pytest.raises(AssertionError, match="scene_003"):
            assert_no_leakage(load_splits(d))

    def test_train_test_overlap_detected(self, tmp_splits):
        d = tmp_splits(
            {
                "train": ["scene_001", "scene_002"],
                "val": ["scene_003"],
                "test": ["scene_002"],
            }
        )
        with pytest.raises(AssertionError, match="scene_002"):
            assert_no_leakage(load_splits(d))

    def test_val_test_overlap_detected(self, tmp_splits):
        d = tmp_splits(
            {
                "train": ["scene_001"],
                "val": ["scene_002", "scene_003"],
                "test": ["scene_003"],
            }
        )
        with pytest.raises(AssertionError, match="scene_003"):
            assert_no_leakage(load_splits(d))

    def test_single_split_always_passes(self, tmp_splits):
        d = tmp_splits({"train": ["scene_001", "scene_002"]})
        assert_no_leakage(load_splits(d))  # nothing to cross-check, must pass

    def test_committed_splits_if_present(self):
        """Run leakage check against committed data/splits/ files once they exist."""
        if not SPLITS_DIR.exists() or not any(SPLITS_DIR.glob("*.json")):
            pytest.skip("No committed split files — run scripts/generate_splits.py first")
        assert_no_leakage(load_splits(SPLITS_DIR))
