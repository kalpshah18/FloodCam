"""Inference with a trained plate-reader bundle (models/<site>/reader.joblib).

    model = PlateModel.load("models/in_overland/reader.joblib")
    out = model.read(bgr_image)                 # single frame
    stream = PlateStream(model); out = stream.update(bgr_image, t_hours)   # sequence

`status` is one of: ok, rejected_quality (blur / dirt / glare by the Laplacian-variance gate),
unregistered (the frame could not be matched to a reference view and no earlier pose exists).
A single frame never borrows a pose; PlateStream reuses the last good pose (causal).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import joblib
import numpy as np

from floodcam.real.events import TRIGGER_CM
from floodcam.real.plate import ROOT, PlateRectifier, Registration, SiteSpec
from floodcam.real.reader import PlateReader, row_features, strip_stats

WIDTH = 1280  # frames are processed at this width, as in data/usgs/frames


def prep(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    if w != WIDTH:  # images smaller than 1280 px (e.g. the 640 px demo copies) are upscaled
        interp = cv2.INTER_AREA if w > WIDTH else cv2.INTER_CUBIC
        bgr = cv2.resize(bgr, (WIDTH, round(h * WIDTH / w)), interpolation=interp)
    return bgr


def frame_quality(bgr: np.ndarray) -> tuple[float, float, bool]:
    g = cv2.cvtColor(cv2.resize(bgr, (640, round(bgr.shape[0] * 640 / bgr.shape[1]))), cv2.COLOR_BGR2GRAY)
    lap = float(cv2.Laplacian(g, cv2.CV_64F).var())
    sat = float(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 1].mean())
    return lap, float(g.mean()), sat < 6.0


@dataclass
class Reading:
    depth_cm: float
    status: str
    margin: float
    lapvar: float
    is_ir: bool
    registered: bool
    inliers: int
    flood: bool            # depth_cm >= 3 cm (only meaningful when status == "ok")
    strip: np.ndarray | None = None
    H: np.ndarray | None = None


class PlateModel:
    def __init__(self, bundle: dict) -> None:
        self.b = bundle
        self.spec = SiteSpec(**bundle["spec"])
        self.reader: PlateReader = bundle["reader"]
        self.rect = PlateRectifier(self.spec, bundle["strip_rows"], bundle["strip_cols"])

    @classmethod
    def load(cls, path: str | Path = "models/in_overland/reader.joblib") -> "PlateModel":
        p = Path(path)
        return cls(joblib.load(p if p.is_absolute() else ROOT / p))

    @property
    def info(self) -> dict:
        return {k: v for k, v in self.b.items() if k not in ("reader", "spec")}

    def read(self, bgr: np.ndarray, H: np.ndarray | None = None, borrowed: bool = False) -> Reading:
        bgr = prep(bgr)
        lap, bright, is_ir = frame_quality(bgr)
        reg = self.rect.register(bgr) if H is None else Registration(H, 0, True)
        if not reg.ok and H is None:
            return Reading(0.0, "unregistered", 0.0, lap, is_ir, False, reg.inliers, False)
        use_H = reg.H if reg.ok else H
        strip = self.rect.strip(bgr, Registration(use_H, reg.inliers, True))
        quality = np.array([[np.log1p(lap), bright / 255.0, float(borrowed or not reg.ok)]], np.float32)
        feats = row_features(strip_stats(strip[None]), self.reader.template, np.array([is_ir]),
                             np.array([reg.ok]), extra=quality)
        cm, margin = self.reader.predict_cm(feats)
        ok = self.b["lap_lo"] <= lap <= self.b["lap_hi"]
        d = float(cm[0])
        return Reading(d, "ok" if ok else "rejected_quality", float(margin[0]), lap, is_ir, reg.ok,
                       reg.inliers, d >= TRIGGER_CM - 0.5, strip, use_H)


class PlateStream:
    """Sequence reader: causal pose fallback and a causal median of accepted readings."""

    def __init__(self, model: PlateModel, window: int = 5) -> None:
        self.model, self.window = model, window
        self._H: np.ndarray | None = None
        self._buf: list[float] = []

    def update(self, bgr: np.ndarray) -> tuple[Reading, float | None]:
        r = self.model.read(bgr)
        if r.status == "unregistered" and self._H is not None:
            r = self.model.read(bgr, H=self._H, borrowed=True)
        elif r.registered:
            self._H = r.H
        if r.status == "ok":
            self._buf = (self._buf + [r.depth_cm])[-self.window:]
        return r, (float(np.median(self._buf)) if self._buf else None)
