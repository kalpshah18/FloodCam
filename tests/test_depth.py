"""Tests for Milestone 4: HSV mask, waterline detection, depth estimation.

Rules enforced:
- No ground-truth labels inside estimate()
- Waterline within 0.5 px on noiseless synthetic
- End-to-end depth within max(0.5, uncertainty_cm)
- Locked scene synth_scene_01 never accessed for tuning
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from floodcam.config import ROI, ExperimentConfig
from floodcam.depth.estimate import estimate, make_camera_config_for_renderer
from floodcam.depth.waterline import DETECT_WIDTH, detect_waterline
from floodcam.seg.hsv import apply_hsv_mask
from floodcam.synth.renderer import SceneConfig, SynthRenderer

ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def default_renderer():
    return SynthRenderer(SceneConfig(), seed=42)


@pytest.fixture(scope="module")
def default_cam_cfg(default_renderer):
    return make_camera_config_for_renderer(default_renderer)


@pytest.fixture(scope="module")
def default_exp_cfg():
    return ExperimentConfig(hsv_lower=[90, 50, 50], hsv_upper=[130, 255, 255])


# ---------------------------------------------------------------------------
# TestHSVMask
# ---------------------------------------------------------------------------


class TestHSVMask:
    def test_deterministic(self, default_renderer, default_exp_cfg):
        frame = default_renderer.render_frame(5.0, frame_idx=0)
        lo = tuple(default_exp_cfg.hsv_lower)
        hi = tuple(default_exp_cfg.hsv_upper)
        m1 = apply_hsv_mask(frame.image, lo, hi)
        m2 = apply_hsv_mask(frame.image, lo, hi)
        np.testing.assert_array_equal(m1, m2)

    def test_blue_water_detected(self):
        frame = np.full((100, 100, 3), [40, 100, 160], dtype=np.uint8)
        mask = apply_hsv_mask(frame, (90, 50, 50), (130, 255, 255))
        assert mask.sum() > 0, "Blue water colour should be detected"

    def test_non_water_not_detected(self):
        # Pure brown (ground colour): RGB ~(110, 95, 75)
        frame = np.full((100, 100, 3), [110, 95, 75], dtype=np.uint8)
        mask = apply_hsv_mask(frame, (90, 50, 50), (130, 255, 255))
        assert mask.sum() == 0, "Brown ground should not be detected as water"

    def test_roi_restricts_mask(self):
        # Blue frame, ROI = right half only
        frame = np.full((100, 100, 3), [40, 100, 160], dtype=np.uint8)
        roi = ROI(x=50, y=0, w=50, h=100)
        mask = apply_hsv_mask(frame, (90, 50, 50), (130, 255, 255), roi=roi)
        # Left half must be all zero
        assert mask[:, :50].sum() == 0, "Left half outside ROI should be masked out"
        # Right half should have detections
        assert mask[:, 50:].sum() > 0, "Right half inside ROI should have water"

    def test_morph_open_removes_isolated_pixel(self):
        # Create a solid brown frame (no water) with one isolated blue pixel
        frame = np.full((50, 50, 3), [110, 95, 75], dtype=np.uint8)
        frame[25, 25] = [40, 100, 160]  # isolated blue pixel
        # Without morphology it might be detected
        mask_raw = apply_hsv_mask(frame, (90, 50, 50), (130, 255, 255))
        if mask_raw.sum() > 0:
            # With open=3 the single pixel should be removed
            mask_open = apply_hsv_mask(frame, (90, 50, 50), (130, 255, 255), morph_open=3)
            assert mask_open.sum() == 0, "morph_open should remove isolated pixel noise"

    def test_no_filesystem_import(self):
        import floodcam.seg.hsv as hsv_mod
        src = inspect.getsource(hsv_mod)
        # Must not contain file open calls
        assert "open(" not in src, "hsv.py must not contain file I/O"


# ---------------------------------------------------------------------------
# TestWaterlineDetection
# ---------------------------------------------------------------------------


class TestWaterlineDetection:
    def _get_waterline_inputs(self, renderer: SynthRenderer, depth_cm: float):
        """Render a frame and extract the inputs needed for detect_waterline."""
        frame = renderer.render_frame(depth_cm, frame_idx=0, variant="normal")
        cam_cfg = make_camera_config_for_renderer(renderer)
        from floodcam.calib.linear import LinearCalib

        calib = LinearCalib.from_config(cam_cfg.calibration)
        v_g = (0.0 - calib.b) / calib.a
        v_pole_top = (cam_cfg.reference_object.known_height_cm - calib.b) / calib.a
        s = calib.cm_per_pixel_at(v_g)
        pole_x = int(renderer._cfg.img_width * renderer._cfg.pole_x_frac)
        pole_hw = renderer._cfg.pole_width_px // 2
        return frame, v_g, v_pole_top, pole_x, pole_hw, s

    def test_noiseless_within_half_pixel(self, default_renderer):
        for depth in [2.0, 5.0, 10.0, 20.0]:
            frame, v_g, v_pole_top, pole_x, pole_hw, s = self._get_waterline_inputs(
                default_renderer, depth
            )
            result = detect_waterline(frame.mask, v_g, v_pole_top, pole_x, pole_hw, s)
            assert result.valid, f"depth={depth}: expected valid, got reason={result.reason}"
            err = abs(result.waterline_row - frame.waterline_row)
            assert err < 0.5, (
                f"depth={depth}: waterline error {err:.3f} px >= 0.5 px "
                f"(estimated={result.waterline_row:.2f}, true={frame.waterline_row:.2f})"
            )

    def test_dry_frame_returns_no_water(self, default_renderer):
        frame, v_g, v_pole_top, pole_x, pole_hw, s = self._get_waterline_inputs(
            default_renderer, 0.0
        )
        # Use the true (all-zero) mask
        result = detect_waterline(frame.mask, v_g, v_pole_top, pole_x, pole_hw, s)
        assert not result.valid
        assert result.reason == "no_water"

    def test_fully_submerged_returns_invalid(self, default_renderer):
        frame = default_renderer.render_frame(5.0, frame_idx=0)
        H, W = frame.mask.shape
        cam_cfg = make_camera_config_for_renderer(default_renderer)
        from floodcam.calib.linear import LinearCalib

        calib = LinearCalib.from_config(cam_cfg.calibration)
        v_g = (0.0 - calib.b) / calib.a
        pole_x = int(default_renderer._cfg.img_width * default_renderer._cfg.pole_x_frac)
        pole_hw = default_renderer._cfg.pole_width_px // 2
        s = calib.cm_per_pixel_at(v_g)

        # Set v_pole_top well below the waterline (so waterline appears above pole top)
        v_pole_top = v_g + 10.0  # pole_top is below ground — forces fully_submerged

        result = detect_waterline(frame.mask, v_g, v_pole_top, pole_x, pole_hw, s)
        assert not result.valid
        assert result.reason == "fully_submerged"

    def test_broken_mask_returns_invalid(self, default_renderer):
        frame = default_renderer.render_frame(10.0, frame_idx=0)
        cam_cfg = make_camera_config_for_renderer(default_renderer)
        from floodcam.calib.linear import LinearCalib

        calib = LinearCalib.from_config(cam_cfg.calibration)
        v_g = (0.0 - calib.b) / calib.a
        v_pole_top = (cam_cfg.reference_object.known_height_cm - calib.b) / calib.a
        pole_x = int(default_renderer._cfg.img_width * default_renderer._cfg.pole_x_frac)
        pole_hw = default_renderer._cfg.pole_width_px // 2
        s = calib.cm_per_pixel_at(v_g)

        H, W = frame.mask.shape
        r_g_int = int(round(v_g))
        r_w_int = int(frame.waterline_row)

        exclusion_hw = pole_hw + 5
        left_start = max(0, pole_x - exclusion_hw - DETECT_WIDTH)
        left_end = max(0, pole_x - exclusion_hw)
        right_start = min(W, pole_x + exclusion_hw)
        right_end = min(W, pole_x + exclusion_hw + DETECT_WIDTH)

        # Create inconsistent waterlines:
        #   left columns: water only within 3px of ground (shallow sliver)
        #   right columns: water from true waterline (full depth)
        # This makes per-col waterline std >> 10px → continuity < 0.5 → broken_mask
        broken = np.zeros(frame.mask.shape, dtype=np.uint8)
        shallow_start = max(0, r_g_int - 3)
        broken[shallow_start : r_g_int + 1, left_start:left_end] = 255
        broken[r_w_int : r_g_int + 1, right_start:right_end] = 255

        result = detect_waterline(broken, v_g, v_pole_top, pole_x, pole_hw, s)
        assert not result.valid, (
            f"Expected invalid broken mask, got valid=True, reason={result.reason}, "
            f"continuity={result.continuity_frac:.3f}"
        )

    def test_confidence_high_for_clean_synthetic(self, default_renderer):
        frame, v_g, v_pole_top, pole_x, pole_hw, s = self._get_waterline_inputs(
            default_renderer, 10.0
        )
        result = detect_waterline(frame.mask, v_g, v_pole_top, pole_x, pole_hw, s)
        assert result.valid
        assert result.confidence >= 0.8, f"Expected confidence >= 0.8, got {result.confidence:.3f}"

    def test_confidence_degrades_with_coarse_resolution(self, default_renderer):
        frame, v_g, v_pole_top, pole_x, pole_hw, _ = self._get_waterline_inputs(
            default_renderer, 10.0
        )
        r_fine = detect_waterline(frame.mask, v_g, v_pole_top, pole_x, pole_hw, 0.4)
        r_coarse = detect_waterline(frame.mask, v_g, v_pole_top, pole_x, pole_hw, 2.0)
        assert r_coarse.confidence < r_fine.confidence, (
            "Coarse resolution should give lower confidence than fine resolution"
        )


# ---------------------------------------------------------------------------
# TestDepthEstimate
# ---------------------------------------------------------------------------


class TestDepthEstimate:
    def test_estimate_has_no_label_param(self):
        sig = inspect.signature(estimate)
        forbidden = {"label", "depth_cm", "truth", "gt", "ground_truth"}
        for param in sig.parameters:
            assert param.lower() not in forbidden, (
                f"estimate() must not have a ground-truth label parameter; found: {param}"
            )

    def test_dry_frame_returns_invalid(self, default_renderer, default_cam_cfg, default_exp_cfg):
        frame = default_renderer.render_frame(0.0, frame_idx=0)
        result = estimate(frame.image, default_cam_cfg, default_exp_cfg)
        assert not result.valid or result.depth_cm < 0.5, (
            f"Dry frame should be invalid or depth < 0.5 cm, got depth={result.depth_cm:.2f}"
        )

    def test_end_to_end_depths_within_tolerance(
        self, default_renderer, default_cam_cfg, default_exp_cfg
    ):
        for true_depth in [2.0, 5.0, 10.0, 20.0]:
            frame = default_renderer.render_frame(true_depth, frame_idx=0)
            result = estimate(frame.image, default_cam_cfg, default_exp_cfg)
            assert result.valid, f"depth={true_depth}: estimate not valid"
            tol = max(0.5, result.uncertainty_cm)
            err = abs(result.depth_cm - true_depth)
            assert err <= tol, (
                f"depth={true_depth}: error {err:.3f} cm > tolerance {tol:.3f} cm "
                f"(estimated={result.depth_cm:.2f})"
            )

    def test_all_valid_for_flooded_synthetic(
        self, default_renderer, default_cam_cfg, default_exp_cfg
    ):
        for true_depth in [2.0, 5.0, 10.0, 20.0]:
            frame = default_renderer.render_frame(true_depth, frame_idx=0)
            result = estimate(frame.image, default_cam_cfg, default_exp_cfg)
            assert result.valid, f"depth={true_depth}: expected valid estimate"

    def test_negative_variant_reflection_no_flood(
        self, default_renderer, default_cam_cfg, default_exp_cfg
    ):
        frame = default_renderer.render_frame(0.0, frame_idx=0, variant="reflection")
        result = estimate(frame.image, default_cam_cfg, default_exp_cfg)
        # A reflection at depth=0 should not trigger a flood
        assert not result.valid or result.depth_cm < 3.5, (
            f"Reflection variant should not trigger flood, got depth={result.depth_cm:.2f}"
        )

    def test_uncertainty_cm_is_positive_when_valid(
        self, default_renderer, default_cam_cfg, default_exp_cfg
    ):
        frame = default_renderer.render_frame(10.0, frame_idx=0)
        result = estimate(frame.image, default_cam_cfg, default_exp_cfg)
        assert result.valid
        assert result.uncertainty_cm > 0, "Valid estimate must have positive uncertainty_cm"


# ---------------------------------------------------------------------------
# TestLockedSceneExclusion
# ---------------------------------------------------------------------------


class TestLockedSceneExclusion:
    def test_scene_loader_rejects_locked_scene(self):
        # Import after scripts/ is on sys.path
        import sys

        scripts_dir = str(ROOT / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        import importlib

        baseline = importlib.import_module("run_hsv_baseline")
        with pytest.raises((AssertionError, RuntimeError, ValueError)):
            baseline.scene_id_to_renderer("synth_scene_01")

    def test_loso_folds_do_not_contain_locked_scene(self):
        splits_path = ROOT / "data" / "splits" / "synthetic_splits.json"
        assert splits_path.exists(), f"Splits file not found: {splits_path}"
        splits = json.loads(splits_path.read_text())
        locked = splits["locked_test_scene"]
        for fold in splits["loso_folds"]:
            assert fold["val_scene"] != locked, (
                f"Fold {fold['fold_idx']} uses locked scene as val_scene"
            )
            assert locked not in fold["train_scenes"], (
                f"Fold {fold['fold_idx']} has locked scene in train_scenes"
            )
