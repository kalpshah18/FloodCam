"""HSV colour-threshold water segmentation.

Pure function — no file I/O, no randomness, no state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from floodcam.config import ROI


def apply_hsv_mask(
    frame_rgb: np.ndarray,
    hsv_lower: tuple[int, int, int],
    hsv_upper: tuple[int, int, int],
    roi: Optional["ROI"] = None,
    morph_open: int = 0,
    morph_close: int = 0,
) -> np.ndarray:
    """Threshold an RGB frame in HSV space to produce a binary water mask.

    Args:
        frame_rgb: (H, W, 3) uint8 RGB image.
        hsv_lower: (H_lo, S_lo, V_lo) lower HSV bound (OpenCV range: H in [0,179]).
        hsv_upper: (H_hi, S_hi, V_hi) upper HSV bound.
        roi: If given, zero the mask outside this rectangle.
        morph_open: Square kernel size for morphological opening (0 = disabled).
        morph_close: Square kernel size for morphological closing (0 = disabled).

    Returns:
        (H, W) uint8 binary mask, values 0 or 255.
    """
    import cv2  # lazy import

    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    lo = np.array(hsv_lower, dtype=np.uint8)
    hi = np.array(hsv_upper, dtype=np.uint8)
    mask = cv2.inRange(hsv, lo, hi)

    if morph_open > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_open, morph_open))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if morph_close > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_close, morph_close))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    if roi is not None:
        roi_mask = np.zeros_like(mask)
        roi_mask[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w] = 255
        mask = cv2.bitwise_and(mask, roi_mask)

    return mask
