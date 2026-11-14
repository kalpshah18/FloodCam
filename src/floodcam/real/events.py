"""Temporal smoothing of a real-footage depth series: causal median of accepted readings."""

from __future__ import annotations

import numpy as np


def causal_median(depth: np.ndarray, valid: np.ndarray, window: int = 5) -> np.ndarray:
    """Median of the last `window` VALID readings (invalid frames are skipped, not zeroed)."""
    out = np.full(len(depth), np.nan)
    buf: list[float] = []
    for i, (d, v) in enumerate(zip(depth, valid)):
        if v and np.isfinite(d):
            buf.append(float(d))
            buf = buf[-window:]
        if buf:
            out[i] = float(np.median(buf))
    return out
