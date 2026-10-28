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
    H: np.ndarray        # frame -> main reference
    inliers: int
    ok: bool
    ref_id: int = 0
    borrowed: bool = False   # H copied from the last successfully registered earlier frame


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
        # additional reference views (other epochs of the same camera), each with its homography to
        # the main reference; see scripts/build_refs.py
        self._extra: list[tuple] = []
        refs_json = ROOT / "configs" / "sites" / f"{spec.name}_refs.json"
        if refs_json.exists():
            import json

            S = np.diag([scale, scale, 1.0])
            for r in json.loads(refs_json.read_text()):
                img = cv2.imread(str(ROOT / r["file"]))
                if img is not None:
                    if scale != 1.0:
                        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                    kp, desc = self._features(img)
                    Hr = S @ np.array(r["H"], np.float64) @ np.linalg.inv(S)
                    self._extra.append((kp, desc, Hr))
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

    def _match(self, kp, desc, ref_kp, ref_desc):
        if desc is None or ref_desc is None or len(kp) < 12:
            return None, 0
        matches = self._bf.knnMatch(desc, ref_desc, k=2)
        good = [m for m, n in (p for p in matches if len(p) == 2) if m.distance < 0.75 * n.distance]
        if len(good) < 12:
            return None, len(good)
        src = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([ref_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if H is None:
            return None, 0
        return H, int(inl.sum())

    def register(self, bgr: np.ndarray, use_extra: bool = True) -> Registration:
        bgr = self._scaled(bgr)
        kp, desc = self._features(bgr)
        best_H, best_n, best_id = None, 0, 0
        H, n = self._match(kp, desc, self._ref_kp, self._ref_desc)
        if H is not None:
            best_H, best_n = H, n
        if use_extra and best_n < 60:
            for i, (rk, rd, Hr) in enumerate(self._extra, start=1):
                H, n = self._match(kp, desc, rk, rd)
                if H is not None and n > best_n:
                    best_H, best_n, best_id = Hr @ H, n, i
        if best_H is None:
            return Registration(np.eye(3), 0, False)
        return Registration(best_H, best_n, best_n >= 15, best_id)

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
