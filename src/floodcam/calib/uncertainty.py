"""Depth uncertainty propagation for the geometric depth estimator.

Implements the formula from the project spec (CLAUDE.md):

    sigma_h^2 ≈ s^2 * sigma_v^2  +  (v_g - v_w)^2 * sigma_s^2  +  sigma_g^2

Three independent error sources:
    1. s^2 * sigma_v^2       — waterline localisation error (mask edge quality)
    2. (v_g-v_w)^2 * sigma_s^2  — scale calibration error; grows with depth
                                   because scale uncertainty multiplies the
                                   measured pixel distance
    3. sigma_g^2             — ground-reference uncertainty (in cm^2)

All quantities in consistent units:
    s        — cm/pixel  (from LinearCalib.cm_per_pixel() or HomographyCalib.cm_per_pixel_at())
    sigma_v  — pixels
    v_g, v_w — pixels
    sigma_s  — cm/pixel  (from LinearCalib.sigma_s)
    sigma_g  — cm        (caller converts pixel error via sigma_g_cm = s * sigma_g_px)
"""

from __future__ import annotations

import math


def depth_uncertainty_cm(
    s: float,
    sigma_v: float,
    v_g: float,
    v_w: float,
    sigma_s: float,
    sigma_g: float = 0.0,
) -> float:
    """Return sigma_h in cm from the linear uncertainty propagation model.

    Args:
        s:       cm/pixel scale factor at the waterline.
        sigma_v: Waterline row localisation error in pixels (std dev).
        v_g:     Ground-contact row in pixels (larger = closer to bottom of frame).
        v_w:     Waterline row in pixels (v_w < v_g for non-zero depth).
        sigma_s: Standard error of the scale estimate in cm/pixel.
                 Use LinearCalib.sigma_s or 0.0 if unknown.
        sigma_g: Ground-reference uncertainty in cm (not pixels).
                 Convert pixel error with: sigma_g = s * sigma_g_pixels.

    Returns:
        sigma_h >= 0.0 in cm.
    """
    var_h = s ** 2 * sigma_v ** 2 + (v_g - v_w) ** 2 * sigma_s ** 2 + sigma_g ** 2
    return math.sqrt(max(0.0, var_h))
