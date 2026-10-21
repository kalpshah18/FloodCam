"""Tests for Milestone 2: synthetic renderer, manifest schema, and split integrity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from floodcam.io.manifest import ManifestError, ManifestRow, read_jsonl, write_jsonl
from floodcam.synth.renderer import SceneConfig, SynthRenderer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DEPTHS_CM = [0.0, 2.0, 3.0, 5.0, 10.0, 20.0]
SYNTH_SPLITS_FILE = Path(__file__).parent.parent / "data" / "splits" / "synthetic_splits.json"


@pytest.fixture(scope="module")
def default_renderer() -> SynthRenderer:
    return SynthRenderer(SceneConfig(), seed=42)


@pytest.fixture(scope="module")
def all_normal_frames(default_renderer):
    return {d: default_renderer.render_frame(d, frame_idx=0, variant="normal") for d in DEPTHS_CM}


# ---------------------------------------------------------------------------
# Renderer: determinism
# ---------------------------------------------------------------------------


class TestRendererDeterminism:
    def test_same_seed_image_identical(self):
        cfg = SceneConfig()
        r1 = SynthRenderer(cfg, seed=7)
        r2 = SynthRenderer(cfg, seed=7)
        f1 = r1.render_frame(5.0, frame_idx=0)
        f2 = r2.render_frame(5.0, frame_idx=0)
        np.testing.assert_array_equal(f1.image, f2.image)
        np.testing.assert_array_equal(f1.mask, f2.mask)

    def test_same_seed_mask_identical(self):
        cfg = SceneConfig()
        r1 = SynthRenderer(cfg, seed=99)
        r2 = SynthRenderer(cfg, seed=99)
        f1 = r1.render_frame(10.0, frame_idx=2)
        f2 = r2.render_frame(10.0, frame_idx=2)
        np.testing.assert_array_equal(f1.mask, f2.mask)

    def test_different_seeds_differ(self):
        cfg = SceneConfig()
        f1 = SynthRenderer(cfg, seed=1).render_frame(5.0)
        f2 = SynthRenderer(cfg, seed=2).render_frame(5.0)
        assert not np.array_equal(f1.image, f2.image)

    def test_different_frame_idx_differ(self):
        r = SynthRenderer(SceneConfig(), seed=42)
        f1 = r.render_frame(5.0, frame_idx=0)
        f2 = r.render_frame(5.0, frame_idx=1)
        assert not np.array_equal(f1.image, f2.image)


# ---------------------------------------------------------------------------
# Renderer: geometry correctness
# ---------------------------------------------------------------------------


class TestRendererGeometry:
    def test_waterline_matches_analytic_all_depths(self, default_renderer):
        """Stored waterline_row must equal project_height(depth_cm) within 0.5 px."""
        for d in DEPTHS_CM:
            frame = default_renderer.render_frame(d, frame_idx=0)
            analytic_row, _ = default_renderer.project_height(d)
            assert abs(frame.waterline_row - analytic_row) < 0.5, (
                f"depth={d}: waterline_row={frame.waterline_row:.3f}, "
                f"analytic={analytic_row:.3f}"
            )

    def test_ground_contact_matches_analytic(self, default_renderer):
        for d in DEPTHS_CM:
            frame = default_renderer.render_frame(d, frame_idx=0)
            analytic_g, _ = default_renderer.project_height(0.0)
            assert abs(frame.ground_contact_row - analytic_g) < 0.5

    def test_waterline_above_ground_contact_when_flooded(self, default_renderer):
        for d in [2.0, 3.0, 5.0, 10.0, 20.0]:
            frame = default_renderer.render_frame(d, frame_idx=0)
            # Waterline row < ground_contact_row means higher in image (correct)
            assert (
                frame.waterline_row < frame.ground_contact_row
            ), f"depth={d}: expected v_w < v_g, got {frame.waterline_row:.1f} >= {frame.ground_contact_row:.1f}"

    def test_cm_per_pixel_positive(self, default_renderer):
        for d in DEPTHS_CM:
            assert default_renderer.cm_per_pixel(d) > 0

    def test_cm_per_pixel_matches_analytic(self, default_renderer):
        for d in DEPTHS_CM:
            frame = default_renderer.render_frame(d)
            assert abs(frame.cm_per_pixel - default_renderer.cm_per_pixel(d)) < 1e-9

    def test_calib_points_correct_rows(self, default_renderer):
        calib = default_renderer.get_calib_points()
        for row, h_cm in calib:
            expected_row, _ = default_renderer.project_height(h_cm)
            assert abs(row - expected_row) < 1e-9


# ---------------------------------------------------------------------------
# Renderer: mask properties
# ---------------------------------------------------------------------------


class TestRendererMask:
    def test_mask_zero_at_depth_zero(self, default_renderer):
        frame = default_renderer.render_frame(0.0)
        assert frame.mask.sum() == 0, "Mask should be all-zero at depth 0"

    def test_mask_binary(self, all_normal_frames):
        for d, frame in all_normal_frames.items():
            vals = np.unique(frame.mask)
            assert set(vals).issubset({0, 255}), f"depth={d}: non-binary mask values {vals}"

    def test_mask_area_monotone_with_depth(self, default_renderer):
        """Mask pixel count must be non-decreasing as depth increases."""
        prev_count = 0
        for d in sorted(DEPTHS_CM):
            count = int((default_renderer.render_frame(d).mask > 0).sum())
            assert (
                count >= prev_count
            ), f"Mask area decreased: depth={d} gave {count} < prev {prev_count}"
            prev_count = count

    def test_mask_area_strictly_increases_above_zero(self, default_renderer):
        """Each positive depth adds at least one row of mask pixels."""
        counts = [
            int((default_renderer.render_frame(d).mask > 0).sum()) for d in [2.0, 5.0, 10.0, 20.0]
        ]
        for i in range(len(counts) - 1):
            assert (
                counts[i + 1] > counts[i]
            ), f"Mask area did not increase between depths {DEPTHS_CM[i+1]} and {DEPTHS_CM[i+2]}"

    def test_image_shape(self, default_renderer):
        frame = default_renderer.render_frame(5.0)
        cfg = default_renderer._cfg
        assert frame.image.shape == (cfg.img_height, cfg.img_width, 3)
        assert frame.mask.shape == (cfg.img_height, cfg.img_width)

    def test_image_dtype(self, default_renderer):
        frame = default_renderer.render_frame(5.0)
        assert frame.image.dtype == np.uint8
        assert frame.mask.dtype == np.uint8

    def test_synthetic_flag_always_true(self, default_renderer):
        for d in DEPTHS_CM:
            assert default_renderer.render_frame(d).synthetic is True


# ---------------------------------------------------------------------------
# Renderer: variants
# ---------------------------------------------------------------------------


class TestVariants:
    def test_reflection_mask_zero(self, default_renderer):
        frame = default_renderer.render_frame(0.0, variant="reflection")
        assert frame.mask.sum() == 0

    def test_wet_road_mask_zero(self, default_renderer):
        frame = default_renderer.render_frame(0.0, variant="wet_road")
        assert frame.mask.sum() == 0

    def test_puddle_mask_nonzero(self, default_renderer):
        frame = default_renderer.render_frame(0.0, variant="puddle")
        assert frame.mask.sum() > 0

    def test_puddle_mask_area_small(self, default_renderer):
        frame = default_renderer.render_frame(0.0, variant="puddle")
        cfg = default_renderer._cfg
        total_pixels = cfg.img_height * cfg.img_width
        puddle_frac = (frame.mask > 0).sum() / total_pixels
        assert puddle_frac < 0.05, f"Puddle mask too large: {puddle_frac:.3f}"


# ---------------------------------------------------------------------------
# Manifest schema
# ---------------------------------------------------------------------------


class TestManifestSchema:
    def _good_row(self) -> dict:
        return {
            "frame_id": "f001",
            "scene_id": "s001",
            "session_id": "s001",
            "camera_id": "cam0",
            "image_path": "a/b.png",
            "mask_path": "a/m.png",
            "depth_cm": 5.0,
            "lighting": "indoor",
            "is_negative": False,
            "synthetic": True,
            "split": "",
        }

    def test_valid_row_parses(self):
        row = ManifestRow.from_dict(self._good_row())
        assert row.frame_id == "f001"
        assert row.synthetic is True
        assert row.depth_cm == 5.0

    def test_missing_synthetic_raises(self):
        d = self._good_row()
        del d["synthetic"]
        with pytest.raises(ManifestError, match="synthetic"):
            ManifestRow.from_dict(d)

    def test_missing_other_field_raises(self):
        d = self._good_row()
        del d["frame_id"]
        with pytest.raises(ManifestError):
            ManifestRow.from_dict(d)

    def test_bool_coercion_from_string(self):
        d = self._good_row()
        d["synthetic"] = "true"
        d["is_negative"] = "false"
        row = ManifestRow.from_dict(d)
        assert row.synthetic is True
        assert row.is_negative is False

    def test_float_depth_coercion(self):
        d = self._good_row()
        d["depth_cm"] = "10.5"
        row = ManifestRow.from_dict(d)
        assert row.depth_cm == 10.5

    def test_roundtrip_jsonl(self, tmp_path):
        rows = [
            ManifestRow.from_dict(
                {
                    "frame_id": f"f{i:03d}",
                    "scene_id": "s001",
                    "session_id": "s001",
                    "camera_id": "cam0",
                    "image_path": f"img/{i}.png",
                    "mask_path": f"msk/{i}.png",
                    "depth_cm": float(i),
                    "lighting": "synthetic",
                    "is_negative": False,
                    "synthetic": True,
                    "split": "",
                }
            )
            for i in range(5)
        ]
        p = tmp_path / "manifest.jsonl"
        write_jsonl(rows, p)
        loaded = read_jsonl(p)
        assert len(loaded) == 5
        for orig, loaded_row in zip(rows, loaded):
            assert orig.frame_id == loaded_row.frame_id
            assert orig.synthetic == loaded_row.synthetic


# ---------------------------------------------------------------------------
# Synthetic splits: leakage and correctness
# ---------------------------------------------------------------------------


def _load_synth_splits(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


class TestSyntheticSplits:
    @pytest.fixture
    def synth_split_data(self):
        if not SYNTH_SPLITS_FILE.exists():
            pytest.skip(
                f"Synthetic splits not found at {SYNTH_SPLITS_FILE}. "
                "Run: python scripts/generate_synthetic_splits.py --manifest data/synthetic/manifest.jsonl"
            )
        return _load_synth_splits(SYNTH_SPLITS_FILE)

    def test_locked_test_not_in_any_loso_fold(self, synth_split_data):
        locked = synth_split_data["locked_test_scene"]
        for fold in synth_split_data["loso_folds"]:
            assert fold["val_scene"] != locked, "Locked test scene appeared as val in a fold"
            assert locked not in fold["train_scenes"], "Locked test scene in fold train set"

    def test_no_loso_leakage_train_val(self, synth_split_data):
        for fold in synth_split_data["loso_folds"]:
            overlap = set(fold["train_scenes"]) & {fold["val_scene"]}
            assert not overlap, f"Fold {fold['fold_idx']}: train/val overlap {overlap}"

    def test_each_non_test_scene_appears_as_val_exactly_once(self, synth_split_data):
        val_scenes = [fold["val_scene"] for fold in synth_split_data["loso_folds"]]
        loso_pool = synth_split_data["coarse_train_scenes"] + synth_split_data["coarse_val_scenes"]
        for scene in loso_pool:
            count = val_scenes.count(scene)
            assert count == 1, f"Scene {scene} appeared as val {count} times (expected 1)"

    def test_coarse_split_no_leakage(self, synth_split_data):
        train = set(synth_split_data["coarse_train_scenes"])
        val = set(synth_split_data["coarse_val_scenes"])
        locked = {synth_split_data["locked_test_scene"]}
        assert not (train & val), f"Coarse train/val overlap: {train & val}"
        assert not (train & locked), f"Train contains locked test: {train & locked}"
        assert not (val & locked), f"Val contains locked test: {val & locked}"


# ---------------------------------------------------------------------------
# Cross-check: extend existing split leakage tests to synth splits dir
# ---------------------------------------------------------------------------


class TestSyntheticSplitLeakageCrossCheck:
    def test_synth_splits_file_format(self):
        if not SYNTH_SPLITS_FILE.exists():
            pytest.skip("Synthetic splits not generated yet")
        data = _load_synth_splits(SYNTH_SPLITS_FILE)
        for key in ("seed", "type", "locked_test_scene", "loso_folds"):
            assert key in data, f"Missing key '{key}' in synthetic_splits.json"

    def test_all_scenes_in_loso_are_unique_per_fold(self):
        if not SYNTH_SPLITS_FILE.exists():
            pytest.skip("Synthetic splits not generated yet")
        data = _load_synth_splits(SYNTH_SPLITS_FILE)
        for fold in data["loso_folds"]:
            train = fold["train_scenes"]
            assert len(train) == len(set(train)), f"Fold {fold['fold_idx']}: duplicate train scenes"
