"""Change-vs-dry-reference segmentation for fixed cameras.

Water (or the wetted part of a pole/curb/wall) changes appearance relative to a dry
reference of the same scene. This is more robust on real footage than a global colour
threshold, which cannot tell water from sky, asphalt or foliage. Requires a reference
that actually shows dry conditions.

Illumination drift is compensated per frame: the median Lab shift over a region that
cannot be under water (rows above the highest calibration mark) is subtracted before
thresholding.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _lab_blur(rgb: np.ndarray, blur: int) -> np.ndarray:
    import cv2

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    k = blur if blur % 2 == 1 else blur + 1
    return cv2.GaussianBlur(lab, (k, k), 0) if k > 1 else lab


class ChangeBackend:
    """SegmentationBackend: pixels that differ from the dry reference by > threshold (Lab dE)."""

    model_version = "change_vs_dry_v1"

    def __init__(
        self,
        ref_rgb: np.ndarray,
        threshold: float = 20.0,
        blur: int = 7,
        morph_open: int = 3,
        morph_close: int = 7,
        gain_rows: Optional[tuple[int, int]] = None,
    ) -> None:
        self.threshold = threshold
        self.blur = blur
        self.morph_open = morph_open
        self.morph_close = morph_close
        self.gain_rows = gain_rows  # (row_start, row_end) of a never-flooded region
        self._ref = _lab_blur(ref_rgb, blur)
        self.last_dE: Optional[np.ndarray] = None

    def diff_map(self, frame_rgb: np.ndarray) -> np.ndarray:
        cur = _lab_blur(frame_rgb, self.blur)
        d = cur - self._ref
        r0, r1 = self.gain_rows if self.gain_rows else (0, d.shape[0])
        r1 = min(r1, d.shape[0])
        if r1 - r0 >= 8:
            patch = d[r0:r1:3, ::3].reshape(-1, 3)
            d = d - np.median(patch, axis=0)
        return np.sqrt((d**2).sum(axis=2))

    def predict_mask(self, frame_rgb: np.ndarray) -> np.ndarray:
        import cv2

        dE = self.diff_map(frame_rgb)
        self.last_dE = dE
        mask = ((dE > self.threshold) * 255).astype(np.uint8)
        if self.morph_open > 1:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (self.morph_open, self.morph_open))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        if self.morph_close > 1:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (self.morph_close, self.morph_close))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        return mask
