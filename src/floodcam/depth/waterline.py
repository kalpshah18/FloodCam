"""Waterline detection from a binary water mask near the reference pole.

Scans columns adjacent to the pole (avoiding the excluded pole strip) to find
the topmost contiguous water boundary, then applies sub-pixel correction.

Continuity metric: 1 - (std of per-column waterline rows) / 20 px, clamped to
[0, 1]. A std > 10 px (continuity_frac < 0.5) signals an unreliable mask
(reflections, occlusion, or large gaps creating inconsistent waterline estimates
across columns). See DECISIONS.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DETECT_WIDTH = 20  # columns on each side of the pole exclusion zone to scan
_CONTINUITY_STD_REF = 20.0  # std (px) that maps to continuity_frac = 0.0


@dataclass
class WaterlineResult:
    """Output of detect_waterline()."""

    waterline_row: float
    valid: bool
    confidence: float
    coverage_frac: float
    continuity_frac: float
    reason: str  # "ok" | "no_water" | "above_pole_top" | "fully_submerged" | "broken_mask"


def detect_waterline(
    mask: np.ndarray,
    v_g: float,
    v_pole_top: float,
    pole_x: int,
    pole_half_width: int,
    cm_per_pixel: float,
) -> WaterlineResult:
    """Find the waterline row from a binary mask along columns flanking the pole.

    Sub-pixel correction: v_w = mean(per_col_topmost_water_rows) - 0.5.

    Args:
        mask: (H, W) uint8 binary mask (0 or 255).
        v_g: Ground-contact row (float); water present at or below this row for flood.
        v_pole_top: Row of pole top from calibration (smaller = higher in image).
        pole_x: Pole centre column index.
        pole_half_width: Half-width of the pole in pixels.
        cm_per_pixel: Scale factor at ground contact row (for resolution_factor).

    Returns:
        WaterlineResult with waterline_row, valid, confidence, coverage_frac,
        continuity_frac, and reason.
    """
    H, W = mask.shape[:2]

    exclusion_hw = pole_half_width + 5
    left_start = max(0, pole_x - exclusion_hw - DETECT_WIDTH)
    left_end = max(0, pole_x - exclusion_hw)
    right_start = min(W, pole_x + exclusion_hw)
    right_end = min(W, pole_x + exclusion_hw + DETECT_WIDTH)

    left_cols = list(range(left_start, left_end))
    right_cols = list(range(right_start, right_end))
    detection_cols = left_cols + right_cols

    if not detection_cols:
        return WaterlineResult(
            waterline_row=float(v_g),
            valid=False,
            confidence=0.0,
            coverage_frac=0.0,
            continuity_frac=0.0,
            reason="no_water",
        )

    r_g = int(min(H - 1, max(0, round(v_g))))

    per_col_waterlines: list[int | None] = []

    for c in detection_cols:
        col = mask[:, c]
        if col[r_g] == 0:
            per_col_waterlines.append(None)
            continue
        r = r_g
        while r > 0 and col[r - 1] > 0:
            r -= 1
        per_col_waterlines.append(r)

    valid_waterlines = [r for r in per_col_waterlines if r is not None]
    n_total = len(detection_cols)
    n_valid = len(valid_waterlines)

    coverage_frac = n_valid / n_total if n_total > 0 else 0.0

    # Continuity: inter-column consistency of waterline row estimates.
    # std=0 → 1.0; std=20px → 0.0 (linear mapping, clamped to [0, 1]).
    if n_valid >= 2:
        std_rows = float(np.std(valid_waterlines))
        continuity_frac = float(max(0.0, 1.0 - std_rows / _CONTINUITY_STD_REF))
    elif n_valid == 1:
        continuity_frac = 1.0
    else:
        continuity_frac = 0.0

    if coverage_frac < 0.2:
        return WaterlineResult(
            waterline_row=float(v_g),
            valid=False,
            confidence=0.0,
            coverage_frac=coverage_frac,
            continuity_frac=continuity_frac,
            reason="no_water",
        )

    if continuity_frac < 0.5:
        v_w = float(np.mean(valid_waterlines)) - 0.5
        return WaterlineResult(
            waterline_row=v_w,
            valid=False,
            confidence=0.0,
            coverage_frac=coverage_frac,
            continuity_frac=continuity_frac,
            reason="broken_mask",
        )

    v_w = float(np.mean(valid_waterlines)) - 0.5

    if v_w <= v_pole_top:
        return WaterlineResult(
            waterline_row=v_w,
            valid=False,
            confidence=0.0,
            coverage_frac=coverage_frac,
            continuity_frac=continuity_frac,
            reason="fully_submerged",
        )

    resolution_factor = min(1.0, 0.5 / max(cm_per_pixel, 1e-6))
    confidence = continuity_frac * coverage_frac * resolution_factor

    return WaterlineResult(
        waterline_row=v_w,
        valid=True,
        confidence=float(confidence),
        coverage_frac=coverage_frac,
        continuity_frac=continuity_frac,
        reason="ok",
    )
