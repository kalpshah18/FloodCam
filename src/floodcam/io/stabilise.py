"""Small-shake stabilisation for nominally fixed cameras (translation only).

Estimates the (dx, dy) shift of each frame against a reference frame with phase
correlation and shifts the frame back, so the manually placed gauge stays on the pole.
Large or low-confidence shifts are ignored (returned unchanged): this is not a
substitute for a tripod.
"""

from __future__ import annotations

import numpy as np


class Stabiliser:
    def __init__(
        self,
        ref_rgb: np.ndarray,
        rows: tuple[int, int] | None = None,
        max_shift_frac: float = 0.08,
        scale: float = 0.5,
    ) -> None:
        # rows: only this band of rows is used to estimate the shift. Use a band that can
        # never be under water (above the highest calibration mark); water changes would
        # otherwise be mistaken for camera motion.
        self._rows = rows
        self._scale = scale
        self._max_frac = max_shift_frac
        self._ref = self._prep(ref_rgb)
        h, w = self._ref.shape
        self._win = self._hann(h, w)
        self.last_shift: tuple[float, float] = (0.0, 0.0)

    @staticmethod
    def _hann(h: int, w: int) -> np.ndarray:
        import cv2

        return cv2.createHanningWindow((w, h), cv2.CV_32F)

    def _prep(self, rgb: np.ndarray) -> np.ndarray:
        import cv2

        if self._rows is not None:
            rgb = rgb[self._rows[0] : self._rows[1]]
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, None, fx=self._scale, fy=self._scale, interpolation=cv2.INTER_AREA)
        g = cv2.GaussianBlur(g, (5, 5), 0).astype(np.float32)
        # Edge magnitude: insensitive to exposure drift and smooth gradients (sky, vignetting)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.magnitude(gx, gy)

    def align(self, frame_rgb: np.ndarray) -> np.ndarray:
        import cv2

        if self._rows is not None and self._rows[1] - self._rows[0] < 24:
            self.last_shift = (0.0, 0.0)
            return frame_rgb
        cur = self._prep(frame_rgb)
        if cur.shape != self._ref.shape:
            self.last_shift = (0.0, 0.0)
            return frame_rgb
        (dx, dy), response = cv2.phaseCorrelate(self._ref, cur, self._win)
        dx, dy = dx / self._scale, dy / self._scale
        H, W = frame_rgb.shape[:2]
        if response < 0.1 or abs(dx) > self._max_frac * W or abs(dy) > self._max_frac * H:
            self.last_shift = (0.0, 0.0)
            return frame_rgb
        # Accept the shift only if it actually aligns the band better (guards against
        # spurious correlation peaks on low-texture or noisy bands).
        Ms = np.array([[1, 0, -dx * self._scale], [0, 1, -dy * self._scale]], dtype=np.float32)
        warped = cv2.warpAffine(cur, Ms, (cur.shape[1], cur.shape[0]), borderMode=cv2.BORDER_REPLICATE)
        def z(a: np.ndarray) -> np.ndarray:
            return (a - a.mean()) / (a.std() + 1e-6)  # robust to exposure drift

        zr = z(self._ref)
        before = float(np.abs(z(cur) - zr).mean())
        after = float(np.abs(z(warped) - zr).mean())
        if after > 0.85 * before:
            self.last_shift = (0.0, 0.0)
            return frame_rgb
        self.last_shift = (float(dx), float(dy))
        if abs(dx) < 0.3 and abs(dy) < 0.3:
            return frame_rgb
        M = np.array([[1, 0, -dx], [0, 1, -dy]], dtype=np.float32)
        return cv2.warpAffine(frame_rgb, M, (W, H), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
