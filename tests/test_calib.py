"""Milestone 3: calibration module tests.

Coverage:
  LinearCalib      — fit, height_at, cm_per_pixel, degenerate rejection, sigma_s
  HomographyCalib  — normalized DLT, height recovery, degenerate rejection
  depth_uncertainty_cm — formula validation
  Noise / Monte Carlo  — 1000 trials, sigma_v = 0.5 px, seed-fixed
  Serialisation    — round-trip YAML + config_hash stability
  Headless CLI     — annotate_headless() creates a valid camera YAML
  Noiseless depth  — linear < 0.5 cm; homography < 0.01 cm noiseless error
  Linear vs Homography — homography outperforms linear on a 20-deg tilt scene

Data-flow note: all calibration points come from renderer.project_height() or
analytic board coordinates, never from rendered-frame depth labels.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from floodcam.calib.annotate import annotate_headless
from floodcam.calib.homography import HomographyCalib
from floodcam.calib.linear import CalibError, LinearCalib
from floodcam.calib.uncertainty import depth_uncertainty_cm
from floodcam.config import CalibrationConfig, CameraConfig
from floodcam.synth.renderer import SceneConfig, SynthRenderer

# ---------------------------------------------------------------------------
# Shared test depths (declared in test_synth.py too; kept local to avoid dep)
# ---------------------------------------------------------------------------

TEST_DEPTHS_CM = [0.0, 2.0, 3.0, 5.0, 10.0, 20.0]
# Locked test scene must not be used for tolerance choices.
_LOCKED = "synth_scene_01"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def default_renderer() -> SynthRenderer:
    return SynthRenderer(SceneConfig(), seed=42)


def _analytic_board_points(
    renderer: SynthRenderer,
    x_offsets_cm: list[float],
    heights_cm: list[float],
) -> list[tuple[float, float, float, float]]:
    """Return (image_u, image_v, plane_x_cm, height_cm) for a vertical calibration board.

    Uses exact pinhole geometry — no rendering noise. Calibration data only.
    """
    pts = []
    for x_w in x_offsets_cm:
        for h in heights_cm:
            alpha = renderer._alpha
            H_cam = renderer._H
            D = renderer._D
            f = renderer._f
            cx = renderer._cx_pole
            cy = renderer._cy

            Y_c = (H_cam - h) * math.cos(alpha) - D * math.sin(alpha)
            Z_c = (H_cam - h) * math.sin(alpha) + D * math.cos(alpha)
            u = cx + f * x_w / Z_c
            v = cy + f * Y_c / Z_c
            pts.append((u, v, float(x_w), float(h)))
    return pts


# ---------------------------------------------------------------------------
# LinearCalib
# ---------------------------------------------------------------------------


class TestLinearCalib:
    def test_fit_two_exact_points_zero_residual(self):
        """Two points define the line exactly; rms_residual == 0."""
        pts = [(100.0, 20.0), (200.0, 10.0)]
        lc = LinearCalib().fit(pts)
        assert abs(lc.rms_residual) < 1e-10
        assert abs(lc.height_at(100.0) - 20.0) < 1e-9
        assert abs(lc.height_at(200.0) - 10.0) < 1e-9

    def test_slope_negative_for_standard_camera(self, default_renderer):
        """Higher pixel row should correspond to lower height — slope must be negative."""
        calib_pts = default_renderer.get_calib_points()
        lc = LinearCalib().fit(calib_pts)
        assert lc.a < 0, f"Expected negative slope, got {lc.a}"

    def test_cm_per_pixel_positive(self, default_renderer):
        lc = LinearCalib().fit(default_renderer.get_calib_points())
        assert lc.cm_per_pixel() > 0
        assert lc.cm_per_pixel_at(300.0) > 0

    def test_height_at_calib_heights_close_to_ground_truth(self, default_renderer):
        """At the ground-contact row (h=0), estimated height should be close to 0."""
        calib_pts = default_renderer.get_calib_points()
        lc = LinearCalib().fit(calib_pts)
        v_g, _ = default_renderer.project_height(0.0)
        assert abs(lc.height_at(v_g)) < 0.5, (
            f"height_at(v_g) = {lc.height_at(v_g):.3f} cm, expected ≈0"
        )

    def test_sigma_s_zero_for_two_points(self):
        lc = LinearCalib().fit([(100.0, 10.0), (200.0, 20.0)])
        assert lc.sigma_s == 0.0

    def test_sigma_s_positive_for_noisy_points(self):
        rng = np.random.default_rng(1)
        vs = np.linspace(100.0, 400.0, 20)
        hs = -0.4 * vs + 150.0 + rng.normal(0, 0.5, 20)
        lc = LinearCalib().fit(list(zip(vs.tolist(), hs.tolist())))
        assert lc.sigma_s > 0.0

    def test_degenerate_too_few_points(self):
        with pytest.raises(CalibError, match="2"):
            LinearCalib().fit([(100.0, 5.0)])

    def test_degenerate_zero_points(self):
        with pytest.raises(CalibError):
            LinearCalib().fit([])

    def test_degenerate_identical_rows(self):
        with pytest.raises(CalibError, match="identical"):
            LinearCalib().fit([(200.0, 0.0), (200.0, 10.0), (200.0, 20.0)])

    def test_unfitted_raises(self):
        lc = LinearCalib()
        with pytest.raises(CalibError):
            lc.height_at(200.0)

    def test_fit_is_self_returning(self):
        lc = LinearCalib()
        returned = lc.fit([(100.0, 0.0), (200.0, 10.0)])
        assert returned is lc


# ---------------------------------------------------------------------------
# HomographyCalib
# ---------------------------------------------------------------------------


class TestHomographyCalib:
    @pytest.fixture(scope="class")
    def board_calib(self, default_renderer):
        pts = _analytic_board_points(
            default_renderer,
            x_offsets_cm=[-20, -10, 0, 10, 20],
            heights_cm=[0.0, 10.0, 20.0, 30.0, 40.0],
        )
        return HomographyCalib().fit(pts)

    def test_reprojection_rms_small(self, board_calib):
        """Noiseless board points should reproject with sub-pixel error."""
        assert board_calib.reprojection_rms < 0.01, (
            f"Reprojection RMS {board_calib.reprojection_rms:.4f} px (expected < 0.01)"
        )

    def test_height_at_center_heights(self, default_renderer, board_calib):
        """At pole centre (x=0), height recovery should be accurate to 0.01 cm."""
        cx = float(default_renderer._cx_pole)
        for h in [0.0, 10.0, 20.0, 30.0, 40.0]:
            v, _ = default_renderer.project_height(h)
            h_pred = board_calib.height_at(cx, v)
            assert abs(h_pred - h) < 0.01, (
                f"h={h}: predicted {h_pred:.4f} cm (error {abs(h_pred-h):.4f})"
            )

    def test_cm_per_pixel_at_positive(self, default_renderer, board_calib):
        cx = float(default_renderer._cx_pole)
        v, _ = default_renderer.project_height(10.0)
        assert board_calib.cm_per_pixel_at(cx, v) > 0

    def test_degenerate_too_few_points(self, default_renderer):
        pts = _analytic_board_points(default_renderer, [0], [0.0, 10.0, 20.0])
        with pytest.raises(CalibError, match="4"):
            HomographyCalib().fit(pts)

    def test_degenerate_collinear_image(self, default_renderer):
        """All points at x=0 (pole centre) are collinear in image space."""
        pts = _analytic_board_points(default_renderer, [0], [0.0, 10.0, 20.0, 30.0])
        # x_w=0 for all → image points lie on a vertical line → collinear
        with pytest.raises(CalibError, match="collinear"):
            HomographyCalib().fit(pts)

    def test_degenerate_duplicate_world_points(self, default_renderer):
        good = _analytic_board_points(
            default_renderer, [-10, 0, 10, 20], [0.0, 10.0, 20.0, 30.0]
        )
        # replace last point with duplicate of first
        dup = list(good)
        dup[-1] = dup[0]
        with pytest.raises(CalibError, match="[Dd]uplicate"):
            HomographyCalib().fit(dup)

    def test_unfitted_raises(self):
        hc = HomographyCalib()
        with pytest.raises(CalibError):
            hc.height_at(320.0, 300.0)

    def test_fit_is_self_returning(self, default_renderer):
        pts = _analytic_board_points(default_renderer, [-10, 0, 10, 20], [0.0, 20.0])
        hc = HomographyCalib()
        returned = hc.fit(pts)
        assert returned is hc


# ---------------------------------------------------------------------------
# Uncertainty formula
# ---------------------------------------------------------------------------


class TestUncertainty:
    def test_all_zero_errors_give_zero_sigma(self):
        assert depth_uncertainty_cm(0.4, 0.0, 350.0, 300.0, 0.0, 0.0) == 0.0

    def test_increases_with_sigma_v(self):
        s = 0.4
        base = depth_uncertainty_cm(s, 0.5, 350.0, 300.0, 0.0, 0.0)
        higher = depth_uncertainty_cm(s, 1.0, 350.0, 300.0, 0.0, 0.0)
        assert higher > base

    def test_increases_with_depth(self):
        """Larger (v_g - v_w) increases the sigma_s contribution."""
        s, sv, ss = 0.4, 0.5, 0.01
        shallow = depth_uncertainty_cm(s, sv, 350.0, 340.0, ss, 0.0)
        deep = depth_uncertainty_cm(s, sv, 350.0, 300.0, ss, 0.0)
        assert deep > shallow

    def test_adds_in_quadrature(self):
        s, sigma_v, sigma_s, sigma_g = 0.4, 0.5, 0.01, 0.2
        v_g, v_w = 350.0, 310.0
        expected = math.sqrt(
            s**2 * sigma_v**2 + (v_g - v_w)**2 * sigma_s**2 + sigma_g**2
        )
        result = depth_uncertainty_cm(s, sigma_v, v_g, v_w, sigma_s, sigma_g)
        assert abs(result - expected) < 1e-12

    def test_sigma_g_increases_uncertainty(self):
        base = depth_uncertainty_cm(0.4, 0.5, 350.0, 300.0, 0.0, 0.0)
        with_g = depth_uncertainty_cm(0.4, 0.5, 350.0, 300.0, 0.0, 0.5)
        assert with_g > base

    def test_formula_dominant_term_scale_times_sigma_v(self):
        """With sigma_s=0, sigma_g=0: sigma_h == s * sigma_v exactly."""
        s, sigma_v = 0.43, 0.7
        result = depth_uncertainty_cm(s, sigma_v, 350.0, 310.0, 0.0, 0.0)
        assert abs(result - s * sigma_v) < 1e-12


# ---------------------------------------------------------------------------
# Monte Carlo noise validation: 1000 trials, sigma_v = 0.5 px, seed fixed
# ---------------------------------------------------------------------------


class TestNoiseSigmaH:
    SIGMA_V = 0.5
    N_TRIALS = 1000
    DEPTH_CM = 10.0

    def test_sigma_h_formula_matches_empirical(self, default_renderer):
        """With sigma_s=0, sigma_g=0: empirical sigma_h must equal s*sigma_v.
        For a linear model height_at(v) = a*v+b, std(h_pred) = |a|*std(noise)
        exactly. 10% tolerance covers small-sample fluctuation."""
        rng = np.random.default_rng(seed=0)
        calib_pts = default_renderer.get_calib_points()
        lc = LinearCalib().fit(calib_pts)

        v_g, _ = default_renderer.project_height(0.0)
        v_w, _ = default_renderer.project_height(self.DEPTH_CM)

        noise = rng.normal(0.0, self.SIGMA_V, self.N_TRIALS)
        depths_pred = np.array([lc.height_at(v_w + n) for n in noise])
        empirical_sigma = float(np.std(depths_pred, ddof=1))

        formula_sigma = depth_uncertainty_cm(
            s=lc.cm_per_pixel(),
            sigma_v=self.SIGMA_V,
            v_g=v_g,
            v_w=v_w,
            sigma_s=0.0,
            sigma_g=0.0,
        )

        ratio = empirical_sigma / formula_sigma
        assert 0.90 <= ratio <= 1.10, (
            f"empirical={empirical_sigma:.4f} cm, formula={formula_sigma:.4f} cm, "
            f"ratio={ratio:.3f} (expected 0.90–1.10)"
        )

    def test_sigma_s_term_increases_formula_output(self, default_renderer):
        """Adding a non-zero sigma_s must increase the formula output."""
        calib_pts = default_renderer.get_calib_points()
        lc = LinearCalib().fit(calib_pts)
        v_g, _ = default_renderer.project_height(0.0)
        v_w, _ = default_renderer.project_height(self.DEPTH_CM)

        s_base = depth_uncertainty_cm(lc.cm_per_pixel(), self.SIGMA_V, v_g, v_w, 0.0)
        s_with = depth_uncertainty_cm(lc.cm_per_pixel(), self.SIGMA_V, v_g, v_w, 0.01)
        assert s_with > s_base

    def test_fitted_sigma_s_is_small_for_synthetic(self, default_renderer):
        """Noiseless analytic calib points should give a very small sigma_s."""
        lc = LinearCalib().fit(default_renderer.get_calib_points())
        assert lc.sigma_s < 0.01, f"sigma_s={lc.sigma_s:.6f} cm/px (expected < 0.01)"


# ---------------------------------------------------------------------------
# Noiseless depth accuracy
# ---------------------------------------------------------------------------


class TestNoiselessDepth:
    """Convert the true waterline row to depth using fitted calibration.
    Tests use non-locked scenes only for tolerance choices.
    synth_scene_01 (locked) is intentionally excluded.
    """

    DENSE_CALIB_HEIGHTS = [0.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 40.0]

    def test_linear_noiseless_error_below_05cm(self, default_renderer):
        """Linear calibration on dense calib heights: max depth error < 0.5 cm.
        Linear approximates the Möbius projection; systematic error grows with
        tilt and depth range. 0.5 cm is achievable for the default 20-deg scene.
        See DECISIONS.md for discussion of the 0.1 cm homography target.
        """
        calib_pts = [
            (default_renderer.project_height(h)[0], h)
            for h in self.DENSE_CALIB_HEIGHTS
        ]
        lc = LinearCalib().fit(calib_pts)

        max_err = 0.0
        for d in TEST_DEPTHS_CM:
            v_w, _ = default_renderer.project_height(d)
            h_pred = lc.height_at(v_w)
            err = abs(h_pred - d)
            max_err = max(max_err, err)

        assert max_err < 0.5, (
            f"Linear max depth error = {max_err:.3f} cm (threshold 0.5 cm)"
        )

    def test_homography_noiseless_error_below_001cm(self, default_renderer):
        """Homography with 2D board calibration should recover depth to < 0.01 cm.
        This is the sub-0.1 cm noiseless target; homography models perspective exactly.
        """
        board_pts = _analytic_board_points(
            default_renderer,
            x_offsets_cm=[-20, -10, 0, 10, 20],
            heights_cm=[0.0, 10.0, 20.0, 30.0, 40.0],
        )
        hc = HomographyCalib().fit(board_pts)
        cx = float(default_renderer._cx_pole)

        max_err = 0.0
        for d in TEST_DEPTHS_CM:
            v_w, _ = default_renderer.project_height(d)
            h_pred = hc.height_at(cx, v_w)
            err = abs(h_pred - d)
            max_err = max(max_err, err)

        assert max_err < 0.01, (
            f"Homography max depth error = {max_err:.5f} cm (threshold 0.01 cm)"
        )


# ---------------------------------------------------------------------------
# Linear vs Homography comparison on a tilted scene
# ---------------------------------------------------------------------------


class TestLinearVsHomography:
    def test_homography_outperforms_linear_on_tilted_scene(self):
        """On a 20-deg tilt scene, homography should have lower max depth error."""
        renderer = SynthRenderer(SceneConfig(camera_tilt_deg=20.0), seed=7)
        cx = float(renderer._cx_pole)

        calib_heights = [0.0, 10.0, 20.0, 30.0, 40.0]
        linear_pts = [(renderer.project_height(h)[0], h) for h in calib_heights]
        lc = LinearCalib().fit(linear_pts)

        board_pts = _analytic_board_points(
            renderer,
            x_offsets_cm=[-20, -10, 0, 10, 20],
            heights_cm=calib_heights,
        )
        hc = HomographyCalib().fit(board_pts)

        test_depths = [0.0, 5.0, 10.0, 20.0, 30.0, 40.0]
        linear_errs, homo_errs = [], []
        for d in test_depths:
            v_w, _ = renderer.project_height(d)
            linear_errs.append(abs(lc.height_at(v_w) - d))
            homo_errs.append(abs(hc.height_at(cx, v_w) - d))

        assert max(homo_errs) < max(linear_errs), (
            f"Homography max err {max(homo_errs):.4f} cm should be less than "
            f"linear max err {max(linear_errs):.4f} cm"
        )


# ---------------------------------------------------------------------------
# Serialisation round-trip
# ---------------------------------------------------------------------------


class TestCalibSerialisation:
    def test_linear_round_trip_yaml(self, default_renderer, tmp_path):
        """Fit → save YAML → load → from_config: predictions must match."""
        calib_pts = default_renderer.get_calib_points()
        lc_orig = LinearCalib().fit(calib_pts)

        cfg = CameraConfig(camera_id="cam_rt")
        cfg.calibration.mode = "linear"
        cfg.calibration.reference_points = [[v, h] for v, h in calib_pts]
        updates = lc_orig.to_config_updates()
        cfg.calibration.scale_cm_per_px = updates["scale_cm_per_px"]
        cfg.calibration.intercept_cm = updates["intercept_cm"]
        cfg.calibration.sigma_s = updates["sigma_s"]

        yaml_path = tmp_path / "cam_rt.yaml"
        cfg.save(yaml_path)

        cfg2 = CameraConfig.load(yaml_path)
        lc2 = LinearCalib.from_config(cfg2.calibration)

        for v_test in [250.0, 300.0, 350.0]:
            assert abs(lc_orig.height_at(v_test) - lc2.height_at(v_test)) < 1e-9

    def test_homography_round_trip_yaml(self, default_renderer, tmp_path):
        """Fit → save YAML → load → from_config: height predictions must match."""
        board_pts = _analytic_board_points(
            default_renderer, [-10, 0, 10], [0.0, 20.0, 40.0]
        )
        hc_orig = HomographyCalib().fit(board_pts)

        cfg = CameraConfig(camera_id="cam_hom_rt")
        cfg.calibration.mode = "homography"
        updates = hc_orig.to_config_updates()
        cfg.calibration.homography_matrix = updates["homography_matrix"]

        yaml_path = tmp_path / "cam_hom_rt.yaml"
        cfg.save(yaml_path)

        cfg2 = CameraConfig.load(yaml_path)
        hc2 = HomographyCalib.from_config(cfg2.calibration)

        cx = float(default_renderer._cx_pole)
        for h in [0.0, 10.0, 20.0]:
            v, _ = default_renderer.project_height(h)
            assert abs(hc_orig.height_at(cx, v) - hc2.height_at(cx, v)) < 1e-9

    def test_config_hash_stable_after_yaml_round_trip(self, default_renderer, tmp_path):
        """config_hash() must be identical before and after save/load."""
        calib_pts = default_renderer.get_calib_points()
        lc = LinearCalib().fit(calib_pts)

        cfg = CameraConfig(camera_id="cam_hash")
        cfg.calibration.mode = "linear"
        upd = lc.to_config_updates()
        cfg.calibration.scale_cm_per_px = upd["scale_cm_per_px"]
        cfg.calibration.intercept_cm = upd["intercept_cm"]
        cfg.calibration.sigma_s = upd["sigma_s"]
        cfg.calibration.reference_points = [[v, h] for v, h in calib_pts]

        h1 = cfg.config_hash()
        yaml_path = tmp_path / "cam_hash.yaml"
        cfg.save(yaml_path)
        cfg2 = CameraConfig.load(yaml_path)
        h2 = cfg2.config_hash()

        assert h1 == h2, f"Hash changed after round-trip: {h1} → {h2}"

    def test_linear_from_config_without_intercept_raises(self):
        cfg = CalibrationConfig(mode="linear", scale_cm_per_px=0.43)
        with pytest.raises(CalibError):
            LinearCalib.from_config(cfg)

    def test_homography_from_config_without_matrix_raises(self):
        cfg = CalibrationConfig(mode="homography")
        with pytest.raises(CalibError):
            HomographyCalib.from_config(cfg)


# ---------------------------------------------------------------------------
# Headless annotation CLI
# ---------------------------------------------------------------------------


class TestAnnotateHeadless:
    def _linear_json(self, renderer: SynthRenderer) -> dict:
        calib_pts = renderer.get_calib_points()
        return {
            "reference_marks": [[float(v), float(h)] for v, h in calib_pts],
            "roi_polygon": [[50, 100], [590, 100], [590, 450], [50, 450]],
        }

    def test_headless_linear_creates_valid_yaml(self, default_renderer, tmp_path):
        pts_file = tmp_path / "pts.json"
        pts_file.write_text(json.dumps(self._linear_json(default_renderer)))

        output = tmp_path / "cam.yaml"
        cfg = annotate_headless(pts_file, "cam_test", output, mode="linear")

        assert output.exists()
        assert cfg.camera_id == "cam_test"
        assert cfg.calibration.mode == "linear"
        assert cfg.calibration.scale_cm_per_px is not None
        assert cfg.calibration.scale_cm_per_px > 0
        assert cfg.calibration.intercept_cm is not None
        assert cfg.calibration.sigma_s is not None
        assert cfg.roi.w > 0
        assert cfg.roi.h > 0

    def test_headless_yaml_is_loadable(self, default_renderer, tmp_path):
        pts_file = tmp_path / "pts2.json"
        pts_file.write_text(json.dumps(self._linear_json(default_renderer)))
        output = tmp_path / "cam2.yaml"
        annotate_headless(pts_file, "cam2", output, mode="linear")

        cfg_loaded = CameraConfig.load(output)
        assert cfg_loaded.calibration.scale_cm_per_px > 0

    def test_headless_too_few_marks_raises(self, tmp_path):
        pts_file = tmp_path / "bad.json"
        pts_file.write_text(json.dumps({"reference_marks": [[300.0, 0.0]]}))
        with pytest.raises(CalibError):
            annotate_headless(pts_file, "cam_bad", tmp_path / "out.yaml", mode="linear")

    def test_headless_missing_marks_key_raises(self, tmp_path):
        pts_file = tmp_path / "empty.json"
        pts_file.write_text(json.dumps({}))
        with pytest.raises(CalibError):
            annotate_headless(pts_file, "cam_x", tmp_path / "out.yaml")

    def test_headless_auto_detects_linear_from_mark_length(
        self, default_renderer, tmp_path
    ):
        pts_file = tmp_path / "auto.json"
        pts_file.write_text(json.dumps(self._linear_json(default_renderer)))
        output = tmp_path / "auto.yaml"
        cfg = annotate_headless(pts_file, "cam_auto", output, mode=None)
        assert cfg.calibration.mode == "linear"

    def test_headless_homography_creates_valid_yaml(self, default_renderer, tmp_path):
        board = _analytic_board_points(
            default_renderer, [-10, 0, 10, 20], [0.0, 20.0, 40.0]
        )
        payload = {
            "reference_marks": [[u, v, xc, h] for u, v, xc, h in board],
        }
        pts_file = tmp_path / "hom_pts.json"
        pts_file.write_text(json.dumps(payload))
        output = tmp_path / "hom.yaml"
        cfg = annotate_headless(pts_file, "cam_hom", output, mode="homography")
        assert cfg.calibration.mode == "homography"
        assert cfg.calibration.homography_matrix is not None
        assert len(cfg.calibration.homography_matrix) == 3

    def test_headless_updates_existing_yaml(self, default_renderer, tmp_path):
        """If output YAML exists, existing non-calibration fields are preserved."""
        existing = CameraConfig(camera_id="cam_existing", description="My camera")
        yaml_path = tmp_path / "existing.yaml"
        existing.save(yaml_path)

        pts_file = tmp_path / "pts3.json"
        pts_file.write_text(json.dumps(self._linear_json(default_renderer)))
        cfg = annotate_headless(pts_file, "cam_existing", yaml_path, mode="linear")

        # camera_id preserved; calibration updated
        assert cfg.camera_id == "cam_existing"
        assert cfg.calibration.scale_cm_per_px is not None
