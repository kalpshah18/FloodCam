"""Event-level view of a depth time series: causal median filter + hysteresis flood episodes."""

from __future__ import annotations

import numpy as np

TRIGGER_CM, CLEAR_CM = 3.5, 2.5


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


def episodes(depth: np.ndarray, t_hours: np.ndarray, trigger: float = TRIGGER_CM,
             clear: float = CLEAR_CM) -> list[tuple[float, float]]:
    """Hysteresis episodes [(start_h, end_h)] where depth >= trigger until it drops < clear."""
    out, start, active = [], 0.0, False
    for d, t in zip(depth, t_hours):
        if not np.isfinite(d):
            continue
        if not active and d >= trigger:
            active, start = True, float(t)
        elif active and d < clear:
            out.append((start, float(t)))
            active = False
    if active:
        out.append((start, float(t_hours[-1])))
    return out


def match_episodes(gt: list[tuple[float, float]], pr: list[tuple[float, float]],
                   slack_h: float = 3.0) -> dict:
    """An observed episode is detected if a predicted one overlaps it (+/- slack). Predicted
    episodes overlapping no observed one are false episodes."""
    def overlap(a, b):
        return a[0] <= b[1] + slack_h and b[0] <= a[1] + slack_h

    det, delays = 0, []
    for g in gt:
        m = [p for p in pr if overlap(g, p)]
        if m:
            det += 1
            delays.append(min(p[0] for p in m) - g[0])
    false = sum(1 for p in pr if not any(overlap(g, p) for g in gt))
    return {"gt_episodes": len(gt), "detected": det, "false_episodes": false,
            "onset_delay_h": [float(d) for d in delays]}
