"""Calibrated depth estimation from a single RGB frame.

estimate() is the public API: takes a frame + camera config + experiment config,
runs HSV segmentation, detects the waterline, converts to depth via calibration.
No ground-truth labels are accessed inside this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np

from floodcam.calib.homography import HomographyCalib
from floodcam.calib.linear import CalibError, LinearCalib
from floodcam.calib.uncertainty import depth_uncertainty_cm
from floodcam.config import ROI, CalibrationConfig, CameraConfig, ExperimentConfig
from floodcam.depth.waterline import detect_waterline
from floodcam.seg.hsv import apply_hsv_mask

SIGMA_V_DEFAULT = 0.5  # assumed waterline localisation noise (px)

_CalibUnion = Union[LinearCalib, HomographyCalib]


@dataclass
class DepthEstimate:
    """Result of a single-frame depth estimate."""

    depth_cm: float
    waterline_row: float
    confidence: float
    valid: bool
    calibration_mode: str
    cm_per_pixel_at_waterline: float
    uncertainty_cm: float


def _load_calib(cfg: CalibrationConfig) -> _CalibUnion:
    if cfg.mode == "homography":
        return HomographyCalib.from_config(cfg)
    return LinearCalib.from_config(cfg)


def _height_at(calib: _CalibUnion, v: float, u: float) -> float:
    if isinstance(calib, HomographyCalib):
        return calib.height_at(u, v)
    return calib.height_at(v)


def _cm_per_pixel_at(calib: _CalibUnion, v: float, u: float) -> float:
    if isinstance(calib, HomographyCalib):
        return calib.cm_per_pixel_at(u, v)
    return calib.cm_per_pixel_at(v)


def _row_at_height(calib: _CalibUnion, h_cm: float, u: float) -> float:
    """Invert calib to find the image row corresponding to height h_cm."""
    if isinstance(calib, HomographyCalib):
        H = calib._H_inv
        denom = H[1, 1] - h_cm * H[2, 1]
        if abs(denom) < 1e-12:
            raise CalibError("Degenerate homography: cannot invert row for given height.")
        v = (h_cm * (H[2, 0] * u + H[2, 2]) - H[1, 0] * u - H[1, 2]) / denom
        return float(v)
    # Linear: h = a*v + b  =>  v = (h - b) / a
    if abs(calib.a) < 1e-12:
        raise CalibError("Degenerate linear calibration: slope is zero.")
    return float((h_cm - calib.b) / calib.a)


def estimate(
    frame_rgb: np.ndarray,
    camera_cfg: CameraConfig,
    exp_cfg: ExperimentConfig,
) -> DepthEstimate:
    """Estimate water depth from a single RGB frame.

    Args:
        frame_rgb: (H, W, 3) uint8 RGB image.
        camera_cfg: Camera-specific calibration and reference-object config.
        exp_cfg: Experiment config with HSV thresholds and morphology params.

    Returns:
        DepthEstimate with depth_cm, confidence, and uncertainty.
        valid=False if the waterline cannot be reliably located.
    """
    mask = apply_hsv_mask(
        frame_rgb,
        tuple(exp_cfg.hsv_lower),  # type: ignore[arg-type]
        tuple(exp_cfg.hsv_upper),  # type: ignore[arg-type]
        roi=camera_cfg.roi,
        morph_open=exp_cfg.hsv_morph_open,
        morph_close=exp_cfg.hsv_morph_close,
    )
    return estimate_from_mask(mask, camera_cfg)


def estimate_from_mask(mask: np.ndarray, camera_cfg: CameraConfig) -> DepthEstimate:
    """Estimate depth from a precomputed binary water mask (any segmentation backend).

    The caller is responsible for applying the ROI to the mask.
    """
    H_img, W_img = mask.shape[:2]

    calib = _load_calib(camera_cfg.calibration)
    calib_mode = camera_cfg.calibration.mode

    pole_x = int(W_img * camera_cfg.reference_object.pole_x_frac)
    pole_hw = camera_cfg.reference_object.pole_width_px // 2

    v_g_raw = _row_at_height(calib, 0.0, pole_x)
    v_g = float(np.clip(v_g_raw, 0, H_img - 1))

    v_pole_top_raw = _row_at_height(calib, camera_cfg.reference_object.known_height_cm, pole_x)
    v_pole_top = float(np.clip(v_pole_top_raw, 0, H_img - 1))

    s_at_ground = _cm_per_pixel_at(calib, v_g, pole_x)

    wl = detect_waterline(mask, v_g, v_pole_top, pole_x, pole_hw, s_at_ground)

    if not wl.valid:
        return DepthEstimate(
            depth_cm=0.0,
            waterline_row=v_g,
            confidence=0.0,
            valid=False,
            calibration_mode=calib_mode,
            cm_per_pixel_at_waterline=s_at_ground,
            uncertainty_cm=0.0,
        )

    depth_cm = max(0.0, _height_at(calib, wl.waterline_row, pole_x))
    s = _cm_per_pixel_at(calib, wl.waterline_row, pole_x)
    sigma_s = camera_cfg.calibration.sigma_s or 0.0
    unc = depth_uncertainty_cm(s, SIGMA_V_DEFAULT, v_g, wl.waterline_row, sigma_s)

    return DepthEstimate(
        depth_cm=depth_cm,
        waterline_row=wl.waterline_row,
        confidence=wl.confidence,
        valid=True,
        calibration_mode=calib_mode,
        cm_per_pixel_at_waterline=s,
        uncertainty_cm=unc,
    )


def make_camera_config_for_renderer(renderer) -> CameraConfig:  # type: ignore[type-arg]
    """Create a CameraConfig fitted to a SynthRenderer's geometry.

    Fits LinearCalib from renderer.get_calib_points(). Used for testing and
    the synthetic baseline script only.
    """
    calib_pts = renderer.get_calib_points()
    lc = LinearCalib().fit(calib_pts)
    cfg = CameraConfig(camera_id="synth_auto")
    cfg.calibration.mode = "linear"
    upd = lc.to_config_updates()
    cfg.calibration.scale_cm_per_px = upd["scale_cm_per_px"]
    cfg.calibration.intercept_cm = upd["intercept_cm"]
    cfg.calibration.sigma_s = upd["sigma_s"]
    cfg.calibration.reference_points = [[v, h] for v, h in calib_pts]
    cfg.reference_object.pole_x_frac = renderer._cfg.pole_x_frac
    cfg.reference_object.pole_width_px = renderer._cfg.pole_width_px
    cfg.reference_object.known_height_cm = renderer._cfg.pole_height_cm
    cfg.roi = ROI(0, 0, renderer._cfg.img_width, renderer._cfg.img_height)
    return cfg
