#!/usr/bin/env python
"""Depth error and 3 cm flood detection vs camera resolution (cm/pixel) on real USGS footage.

Each factor f downsamples every frame by f before registration and plate rectification
(scripts/extract_strips.py --factor f), then runs leave-one-block-out CV on the DEV blocks.
Fits the error model  RMSE^2 = (sigma_v * s)^2 + sigma_floor^2  (s = cm/px along the plate), the
measured counterpart of  sigma_h^2 ~ s^2 sigma_v^2 + sigma_g^2  in docs/DECISIONS.md.

Usage: python scripts/real_resolution.py --site in_overland --factors 1 2 4 8 16 32
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from floodcam.real.experiment import load_dataset, run_cv, summarize, write_run  # noqa: E402
from floodcam.real.plate import SiteSpec  # noqa: E402


def cm_per_px_native(spec: SiteSpec, width_px: int = 1280, height_px: int = 724) -> float:
    q = np.array(spec.quad) * [width_px, height_px]
    left = np.linalg.norm(q[0] - q[3])
    right = np.linalg.norm(q[1] - q[2])
    return (spec.top_ft - spec.bottom_ft) * spec.ft_to_cm / ((left + right) / 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="in_overland")
    ap.add_argument("--factors", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    spec = SiteSpec.load(a.site)
    s0 = cm_per_px_native(spec)
    rows = []
    for f in a.factors:
        ds = load_dataset(a.site, f)
        dev = ds.split["dev_blocks"]
        res = run_cv(ds, spec, dev, dev, seed=a.seed)
        summ = summarize(ds, res)
        write_run(f"real-res-{a.site}-f{f}", ds, res, summ, {"protocol": "LOBO dev, resolution sweep",
                                                              "seed": a.seed, "cm_per_px": s0 * f})
        for scope in ("all_frames", "gated"):
            L = summ[scope]["learned"]
            B = summ[scope]["threshold_baseline"]
            rows.append({
                "factor": f, "cm_per_px": s0 * f, "scope": scope,
                "learned_mae": L["depth"]["mae_cm"], "learned_rmse": L["depth"]["rmse_cm"],
                "learned_f1": L["flood_3cm"]["f1"], "learned_precision": L["flood_3cm"]["precision"],
                "learned_recall": L["flood_3cm"]["recall"], "learned_far": L["flood_3cm"]["false_alarm_rate"],
                "learned_near3_mae": L["depth"]["near_3cm"]["mae_cm"],
                "baseline_mae": B["depth"]["mae_cm"], "baseline_f1": B["flood_3cm"]["f1"],
                "n": L["depth"]["n"], "block_mae_iqr": L["by_block"]["mae_iqr"],
            })
        print(f"f={f} s={s0 * f:.2f} cm/px  gated MAE {rows[-1]['learned_mae']:.2f}  "
              f"F1 {rows[-1]['learned_f1']:.2f}", flush=True)
    # error model fit on gated RMSE
    g = [r for r in rows if r["scope"] == "gated"]
    s = np.array([r["cm_per_px"] for r in g])
    y = np.array([r["learned_rmse"] ** 2 for r in g])
    A = np.stack([s**2, np.ones_like(s)], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    sigma_v = float(np.sqrt(max(coef[0], 0)))
    floor = float(np.sqrt(max(coef[1], 0)))
    fit = {"sigma_v_px": sigma_v, "sigma_floor_cm": floor,
           "predicted_rmse": [float(np.sqrt(max(coef[0] * x**2 + coef[1], 0))) for x in s],
           "cm_per_px_native": s0}
    out = ROOT / "results" / "real" / f"resolution-{a.site}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({"rows": rows, "error_model_fit": fit}, indent=1))
    print("fit:", fit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
