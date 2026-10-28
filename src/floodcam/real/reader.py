"""Learned staff-plate reader.

Each frame is registered and rectified to a plate strip (floodcam.real.plate). A per-row
classifier predicts whether that row of the plate is under water; the water level is the
boundary row that best explains the row probabilities (monotone change-point). Row labels
come from the official gauge, so no manual annotation is needed.

All statistics (dry template, calibration, classifier) are fitted on training frames only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

FT_TO_CM = 30.48


def _box_mean(x: np.ndarray, w: int, axis: int) -> np.ndarray:
    """Causal-free moving mean along `axis` (centered window of size w, edge-padded)."""
    k = np.ones(w, np.float32) / w
    pad = [(0, 0)] * x.ndim
    pad[axis] = (w // 2, w - 1 - w // 2)
    xp = np.pad(x, pad, mode="edge")
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), axis, xp)


def strip_stats(strips_bgr: np.ndarray) -> dict[str, np.ndarray]:
    """Per-frame, per-row summary statistics. strips: (N, R, C, 3) uint8 BGR."""
    N, R, C, _ = strips_bgr.shape
    lab = np.empty((N, R, C, 3), np.float32)
    for i in range(N):
        lab[i] = cv2.cvtColor(strips_bgr[i], cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0]
    return {
        "lab": lab.mean(axis=2),                                  # (N, R, 3)
        "std_l": L.std(axis=2),                                   # (N, R)
        "edge_l": np.abs(np.diff(L, axis=1, prepend=L[:, :1])).mean(axis=2),
        "sat": (L > 245).mean(axis=(1, 2)),                       # (N,) saturated fraction
        "mean_l": L.mean(axis=(1, 2)),
    }


@dataclass
class DryTemplate:
    lab: np.ndarray      # (R, 3)
    std_l: np.ndarray    # (R,)
    edge_l: np.ndarray   # (R,)

    @classmethod
    def fit(cls, stats: dict[str, np.ndarray], dry_mask: np.ndarray) -> "DryTemplate":
        return cls(
            lab=np.median(stats["lab"][dry_mask], axis=0),
            std_l=np.median(stats["std_l"][dry_mask], axis=0),
            edge_l=np.median(stats["edge_l"][dry_mask], axis=0),
        )


def row_features(stats: dict[str, np.ndarray], tpl: DryTemplate, is_ir: np.ndarray,
                 reg_ok: np.ndarray, ctx: int = 15) -> np.ndarray:
    """(N, R, F) feature tensor."""
    lab = stats["lab"]
    N, R, _ = lab.shape
    top = slice(0, max(4, R // 10))
    # per-frame exposure normalisation from the top rows of the plate (normally above water)
    shift = np.median((lab - tpl.lab)[:, top], axis=1, keepdims=True)
    d = lab - tpl.lab - shift
    dE = np.sqrt((d**2).sum(axis=2))
    dE_raw = np.sqrt(((lab - tpl.lab) ** 2).sum(axis=2))
    w = max(3, int(ctx * R / 360) | 1)
    below = _box_mean(dE, w, 1)
    # directional context: mean of dE over the w rows below / above each row
    csum = np.cumsum(np.pad(dE, ((0, 0), (1, 0))), axis=1)
    idx = np.arange(R)
    hi = np.minimum(idx + w, R)
    lo = np.maximum(idx - w, 0)
    below_m = (csum[:, hi] - csum[:, idx]) / np.maximum(hi - idx, 1)
    above_m = (csum[:, idx] - csum[:, lo]) / np.maximum(idx - lo, 1)
    l_c = _box_mean(lab[..., 0], w, 1)
    texture_ratio = stats["edge_l"] / (tpl.edge_l + 1.0)
    F = [
        lab[..., 0], lab[..., 1], lab[..., 2],
        d[..., 0], d[..., 1], d[..., 2], dE, dE_raw,
        stats["std_l"], stats["std_l"] - tpl.std_l, stats["edge_l"], texture_ratio,
        below, below_m, above_m, below_m - above_m, l_c,
        np.repeat(is_ir[:, None].astype(np.float32), R, 1),
        np.repeat(stats["sat"][:, None], R, 1),
        np.repeat(stats["mean_l"][:, None], R, 1),
        np.repeat(reg_ok[:, None].astype(np.float32), R, 1),
    ]
    return np.stack(F, axis=-1).astype(np.float32)


def decode_boundary(p_sub: np.ndarray, eps: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """Best monotone split. Rows >= b are submerged (row 0 = top). Returns (b, margin).

    margin = log-likelihood gap between the best boundary and the runner-up (confidence).
    """
    p = np.clip(p_sub, eps, 1 - eps)
    lp, lq = np.log(p), np.log(1 - p)
    N, R = p.shape
    cum_q = np.concatenate([np.zeros((N, 1)), np.cumsum(lq, axis=1)], axis=1)   # rows < b dry
    cum_p = np.concatenate([np.zeros((N, 1)), np.cumsum(lp[:, ::-1], axis=1)], axis=1)[:, ::-1]
    score = cum_q + cum_p                                                        # (N, R+1)
    b = score.argmax(axis=1)
    srt = np.sort(score, axis=1)
    return b.astype(np.float32), (srt[:, -1] - srt[:, -2]).astype(np.float32)


@dataclass
class PlateReader:
    top_ft: float
    bottom_ft: float
    model: HistGradientBoostingClassifier | None = None
    template: DryTemplate | None = None
    cal_a: float = 1.0           # gauge_ft ~ cal_a * nominal_ft + cal_b
    cal_b: float = 0.0
    label_offset_ft: float = 0.0
    params: dict = field(default_factory=lambda: {"max_iter": 120, "max_depth": 6,
                                                  "learning_rate": 0.1})

    def heights(self, R: int) -> np.ndarray:
        k = (np.arange(R) + 0.5) / R
        return self.top_ft - k * (self.top_ft - self.bottom_ft)

    def boundary_ft(self, b: np.ndarray, R: int) -> np.ndarray:
        """Nominal water level (ft) for boundary index b (b=R: at/below plate bottom)."""
        return self.top_ft - (b / R) * (self.top_ft - self.bottom_ft)

    def labels(self, gage_ft: np.ndarray, R: int) -> np.ndarray:
        h = self.heights(R)[None, :]
        return (h < (gage_ft[:, None] + self.label_offset_ft)).astype(np.uint8)

    def fit(self, feats: np.ndarray, gage_ft: np.ndarray, max_rows: int = 250_000,
            seed: int = 0) -> "PlateReader":
        N, R, F = feats.shape
        y = self.labels(gage_ft, R)
        X, yy = feats.reshape(-1, F), y.reshape(-1)
        rng = np.random.default_rng(seed)
        if len(yy) > max_rows:
            sel = rng.choice(len(yy), max_rows, replace=False)
            X, yy = X[sel], yy[sel]
        self.model = HistGradientBoostingClassifier(random_state=seed, **self.params).fit(X, yy)
        # affine map from nominal boundary level to gauge, fitted on training frames
        lvl, _ = self.predict_nominal(feats)
        use = (gage_ft > 0.25) & (gage_ft < self.top_ft - 0.1)
        if use.sum() >= 10:
            A = np.stack([lvl[use], np.ones(use.sum())], axis=1)
            self.cal_a, self.cal_b = np.linalg.lstsq(A, gage_ft[use], rcond=None)[0]
        return self

    def proba(self, feats: np.ndarray) -> np.ndarray:
        N, R, F = feats.shape
        return self.model.predict_proba(feats.reshape(-1, F))[:, 1].reshape(N, R)

    def predict_nominal(self, feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = self.proba(feats)
        b, margin = decode_boundary(p)
        return self.boundary_ft(b, feats.shape[1]), margin

    def predict_cm(self, feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lvl, margin = self.predict_nominal(feats)
        ft = np.clip(self.cal_a * lvl + self.cal_b, 0.0, None)
        # at/below the plate bottom the level is unobservable: report 0
        ft = np.where(lvl <= self.bottom_ft + 1e-6, 0.0, ft)
        return ft * FT_TO_CM, margin


def threshold_baseline_cm(feats: np.ndarray, reader: PlateReader, tau: float,
                          dE_index: int = 6) -> np.ndarray:
    """Non-learned baseline: row is 'wet' if its dry-template colour change exceeds tau."""
    dE = feats[..., dE_index]
    p = np.where(dE > tau, 0.9, 0.1)
    b, _ = decode_boundary(p)
    lvl = reader.boundary_ft(b, feats.shape[1])
    ft = np.clip(reader.cal_a * lvl + reader.cal_b, 0, None)
    return np.where(lvl <= reader.bottom_ft + 1e-6, 0.0, ft) * FT_TO_CM
