"""Depth and flood-indicator metrics for the real-footage experiments."""

from __future__ import annotations

import numpy as np

BINS_CM = [(0.0, 0.5), (0.5, 3.0), (3.0, 10.0), (10.0, 25.0), (25.0, 1e9)]
THRESH_CM = 3.0


def depth_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    err = pred - gt
    out = {"n": int(len(gt)), "mae_cm": float(np.abs(err).mean()) if len(gt) else float("nan"),
           "rmse_cm": float(np.sqrt((err**2).mean())) if len(gt) else float("nan"),
           "bias_cm": float(err.mean()) if len(gt) else float("nan")}
    bins = {}
    for lo, hi in BINS_CM:
        m = (gt >= lo) & (gt < hi)
        key = f"{lo:g}-{hi:g}" if hi < 1e8 else f">={lo:g}"
        bins[key] = {"n": int(m.sum()),
                     "mae_cm": float(np.abs(err[m]).mean()) if m.any() else None}
    out["mae_by_bin_cm"] = bins
    near = (gt >= 1.5) & (gt <= 4.5)
    out["near_3cm"] = {"n": int(near.sum()),
                       "mae_cm": float(np.abs(err[near]).mean()) if near.any() else None}
    return out


def flood_metrics(gt: np.ndarray, pred: np.ndarray, thr: float = THRESH_CM) -> dict:
    y, p = gt >= thr, pred >= thr
    tp, fp, fn, tn = int((y & p).sum()), int((~y & p).sum()), int((y & ~p).sum()), int((~y & ~p).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if tp else (0.0 if (tp + fp + fn) else float("nan"))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec, "f1": f1,
            "false_alarm_rate": fp / (fp + tn) if fp + tn else float("nan")}


def block_summary(blocks: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> dict:
    """Per-held-out-block MAE and the spread across blocks (the honest uncertainty)."""
    per = {}
    for b in sorted(set(blocks.tolist())):
        m = blocks == b
        per[b] = {"n": int(m.sum()), "mae_cm": float(np.abs(pred[m] - gt[m]).mean()),
                  "max_gt_cm": float(gt[m].max())}
    maes = np.array([v["mae_cm"] for v in per.values()])
    return {"per_block": per, "mae_median": float(np.median(maes)),
            "mae_mean": float(maes.mean()), "mae_iqr": [float(np.percentile(maes, 25)),
                                                       float(np.percentile(maes, 75))],
            "mae_worst": float(maes.max())}


def block_bootstrap_ci(blocks: np.ndarray, gt: np.ndarray, pred: np.ndarray, n: int = 2000,
                       seed: int = 0) -> list[float]:
    """95% CI of pooled MAE, resampling whole blocks (frames within a block are correlated)."""
    rng = np.random.default_rng(seed)
    ub = sorted(set(blocks.tolist()))
    sums = {b: (np.abs(pred[blocks == b] - gt[blocks == b]).sum(), int((blocks == b).sum()))
            for b in ub}
    vals = []
    for _ in range(n):
        pick = rng.choice(len(ub), len(ub))
        s = sum(sums[ub[i]][0] for i in pick)
        c = sum(sums[ub[i]][1] for i in pick)
        vals.append(s / c)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
