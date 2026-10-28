"""Staff-plate geometry: register a frame to a site reference, rectify the plate to a strip.

The plate is a planar rectangle, so the plate quad (given in reference-frame coordinates) maps
to an axis-aligned strip by a homography. Rows of the strip are linear in height (ft).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]


@dataclass
class SiteSpec:
    name: str
    reference: str            # path relative to repo root
    quad: list[list[float]]   # TL, TR, BR, BL in reference pixel coords (normalised 0-1)
    top_ft: float
    bottom_ft: float
    reg_region: list[float]   # [x0, y0, x1, y1] normalised: where static structure is matched
    ft_to_cm: float = 30.48

    @classmethod
    def load(cls, name: str) -> "SiteSpec":
        d = yaml.safe_load((ROOT / "configs" / "sites" / f"{name}.yaml").read_text())
        return cls(**d)


@dataclass
class Registration:
    H: np.ndarray        # frame -> reference
    inliers: int
    ok: bool


class PlateRectifier:
    """Registers frames to the site reference and extracts the rectified plate strip."""

    def __init__(self, spec: SiteSpec, strip_rows: int = 360, strip_cols: int = 48,
                 scale: float = 1.0) -> None:
        self.spec = spec
        ref = cv2.imread(str(ROOT / spec.reference))
        if ref is None:
            raise FileNotFoundError(spec.reference)
        self.scale = scale
        if scale != 1.0:  # simulate a lower-resolution camera: everything happens at this scale
            ref = cv2.resize(ref, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        self.ref_shape = ref.shape[:2]
        self.strip_rows, self.strip_cols = strip_rows, strip_cols
        self._sift = cv2.SIFT_create(nfeatures=2000)
        self._bf = cv2.BFMatcher(cv2.NORM_L2)
        self._ref_kp, self._ref_desc = self._features(ref)
        h, w = self.ref_shape
        self._quad_px = (np.array(spec.quad, np.float32) * [w, h]).astype(np.float32)

    def _features(self, bgr: np.ndarray):
        h, w = bgr.shape[:2]
        x0, y0, x1, y1 = self.spec.reg_region
        mask = np.zeros((h, w), np.uint8)
        mask[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)] = 255
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        g = cv2.createCLAHE(2.0, (8, 8)).apply(g)  # evens out IR / washed-out frames
        kp, desc = self._sift.detectAndCompute(g, mask)
        return kp, desc

    def _scaled(self, bgr: np.ndarray) -> np.ndarray:
        if self.scale == 1.0:
            return bgr
        return cv2.resize(bgr, (self.ref_shape[1], self.ref_shape[0]), interpolation=cv2.INTER_AREA)

    def register(self, bgr: np.ndarray) -> Registration:
        bgr = self._scaled(bgr)
        kp, desc = self._features(bgr)
        if desc is None or self._ref_desc is None or len(kp) < 12:
            return Registration(np.eye(3), 0, False)
        matches = self._bf.knnMatch(desc, self._ref_desc, k=2)
        good = [m for m, n in (p for p in matches if len(p) == 2) if m.distance < 0.75 * n.distance]
        if len(good) < 12:
            return Registration(np.eye(3), len(good), False)
        src = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([self._ref_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if H is None:
            return Registration(np.eye(3), 0, False)
        n_in = int(inl.sum())
        return Registration(H, n_in, n_in >= 15)

    def strip(self, bgr: np.ndarray, reg: Registration | None = None) -> np.ndarray:
        """(strip_rows, strip_cols, 3) uint8 BGR plate strip, row 0 = top_ft."""
        reg = reg or self.register(bgr)
        bgr = self._scaled(bgr)
        # strip -> reference quad homography, composed with reference -> frame
        dst = np.array([[0, 0], [self.strip_cols, 0], [self.strip_cols, self.strip_rows],
                        [0, self.strip_rows]], np.float32)
        Hq = cv2.getPerspectiveTransform(self._quad_px, dst)  # reference -> strip
        M = Hq @ reg.H                                          # frame -> strip
        return cv2.warpPerspective(bgr, M, (self.strip_cols, self.strip_rows),
                                   flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)

    def row_heights_ft(self) -> np.ndarray:
        n = self.strip_rows
        k = (np.arange(n) + 0.5) / n
        return self.spec.top_ft - k * (self.spec.top_ft - self.spec.bottom_ft)
