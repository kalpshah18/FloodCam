#!/usr/bin/env python
"""Cross-camera transfer: classifier trained on camera A (in_overland dev blocks), applied to camera B.

Target camera (default hancock: river staff plate, portrait view, 11-21 ft scale) differs in plate
design, scale, viewpoint and water. No target-camera training of the classifier happens. What is fitted on
target data (development blocks only, never the locked flood block):
  * the dry template (frames whose gauge is below the plate bottom),
  * the gate thresholds (Laplacian-variance quantiles),
  * optionally a one-parameter offset of the decoded level (calibration), using dev frames above the
    plate bottom.
The locked target block (the July 2026 flood, stage up to ~18 ft) is evaluated once per variant.

Variants: zero-shot (nominal plate geometry), offset-calibrated.
Usage: python scripts/real_transfer.py --source in_overland --target hancock
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from floodcam.real.experiment import GATE_HI, GATE_LO, fit_models, load_dataset  # noqa: E402
from floodcam.real.plate import SiteSpec  # noqa: E402
from floodcam.real.reader import DryTemplate, decode_boundary, row_features  # noqa: E402

FT = 30.48


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="in_overland")
    ap.add_argument("--target", default="hancock")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    src, tgt = load_dataset(a.source, 1), load_dataset(a.target, 1)
    sspec, tspec = SiteSpec.load(a.source), SiteSpec.load(a.target)
    # 1) source classifier: dev blocks of the source camera only
    s_train = np.isin(src.blocks, src.split["dev_blocks"])
    _, _, reader, _ = fit_models(src, sspec, s_train, a.seed)

    # 2) target preprocessing statistics from target DEV blocks only
    t_dev = np.isin(tgt.blocks, tgt.split["dev_blocks"])
    t_test = np.isin(tgt.blocks, tgt.split["locked_test_blocks"])
    assert not (t_dev & t_test).any()
    dry = t_dev & (tgt.gage_ft < tspec.bottom_ft - 0.3) & ~tgt.is_ir & tgt.reg_ok
    tpl = DryTemplate.fit(tgt.stats, dry)
    feats = row_features(tgt.stats, tpl, tgt.is_ir, tgt.reg_ok, extra=tgt.quality)
    p = reader.model.predict_proba(feats.reshape(-1, feats.shape[2]))[:, 1].reshape(len(feats), -1)
    b, margin = decode_boundary(p)
    R = feats.shape[1]
    level_ft = tspec.top_ft - (b / R) * (tspec.top_ft - tspec.bottom_ft)   # nominal geometry
    below_plate = b >= R - 1                                                # nothing submerged above the strip bottom

    lap = np.expm1(tgt.quality[:, 0])
    med = float(np.median(lap[t_dev]))
    accepted = (lap >= GATE_LO * med) & (lap <= GATE_HI * med)

    # 3) offset calibration on dev frames above the plate bottom
    cal = t_dev & (tgt.gage_ft > tspec.bottom_ft + 0.15) & ~below_plate & accepted
    offset = float(np.median(tgt.gage_ft[cal] - level_ft[cal])) if cal.sum() >= 5 else 0.0

    def report(name, pred_ft):
        m = t_test & accepted & (tgt.gage_ft > tspec.bottom_ft + 0.15)
        err = (pred_ft[m] - tgt.gage_ft[m]) * FT
        mb = t_test & accepted & (tgt.gage_ft <= tspec.bottom_ft + 0.15)
        return {
            "n_on_plate": int(m.sum()), "mae_cm": float(np.abs(err).mean()), "rmse_cm": float(np.sqrt((err**2).mean())),
            "bias_cm": float(err.mean()),
            "mae_by_stage_ft": {f"{lo}-{hi}": (float(np.abs(err[(tgt.gage_ft[m] >= lo) & (tgt.gage_ft[m] < hi)]).mean())
                                                if ((tgt.gage_ft[m] >= lo) & (tgt.gage_ft[m] < hi)).any() else None)
                                for lo, hi in ((11.0, 13.0), (13.0, 16.0), (16.0, 19.0))},
            "n_below_plate_test": int(mb.sum()),
            "below_plate_correct": float(below_plate[mb].mean()) if mb.any() else None,
        }, m

    # 4) pooled variant: add the target's DEV frames (labels shifted by the calibrated offset) to the
    #    source training rows, retrain, re-estimate the offset on dev frames, evaluate on the locked block.
    from sklearn.ensemble import HistGradientBoostingClassifier

    s_feats = row_features(src.stats, reader.template, src.is_ir, src.reg_ok, extra=src.quality)
    Rs = s_feats.shape[1]
    ys = reader.labels(src.gage_ft, Rs)
    Xs, ysf = s_feats[s_train].reshape(-1, s_feats.shape[2]), ys[s_train].reshape(-1)
    heights_t = tspec.top_ft - (np.arange(R) + 0.5) / R * (tspec.top_ft - tspec.bottom_ft)
    yt = (heights_t[None, :] < (tgt.gage_ft[:, None] - offset)).astype(np.uint8)
    use_t = t_dev & accepted
    Xt, ytf = feats[use_t].reshape(-1, feats.shape[2]), yt[use_t].reshape(-1)
    rng = np.random.default_rng(a.seed)
    sel_s = rng.choice(len(ysf), min(len(ysf), 250_000), replace=False)
    sel_t = rng.choice(len(ytf), min(len(ytf), 60_000), replace=False)
    pooled = HistGradientBoostingClassifier(random_state=a.seed, max_iter=120, max_depth=6, learning_rate=0.1).fit(
        np.vstack([Xs[sel_s], Xt[sel_t]]), np.concatenate([ysf[sel_s], ytf[sel_t]]))
    pp = pooled.predict_proba(feats.reshape(-1, feats.shape[2]))[:, 1].reshape(len(feats), -1)
    bp, _ = decode_boundary(pp)
    level_pool = tspec.top_ft - (bp / R) * (tspec.top_ft - tspec.bottom_ft)
    below_pool = bp >= R - 1
    cal_p = t_dev & (tgt.gage_ft > tspec.bottom_ft + 0.15) & ~below_pool & accepted
    off_p = float(np.median(tgt.gage_ft[cal_p] - level_pool[cal_p])) if cal_p.sum() >= 5 else 0.0

    zs, mask = report("zero_shot", level_ft)
    cl, _ = report("calibrated", level_ft + offset)
    po, _ = report("pooled", level_pool + off_p)
    trivial = float(np.abs((tspec.bottom_ft - tgt.gage_ft[mask]) * FT).mean())   # always "at the plate bottom"
    out = {"source": a.source, "target": a.target, "target_dev_blocks": tgt.split["dev_blocks"],
           "target_locked_blocks": tgt.split["locked_test_blocks"],
           "n_calibration_frames": int(cal.sum()), "offset_ft": offset, "coverage_test": float(accepted[t_test].mean()),
           "zero_shot": zs, "calibrated": cl, "pooled_with_target_dev": po, "pooled_offset_ft": off_p, "baseline_always_plate_bottom_mae_cm": trivial,
           "note": "classifier never trained on the target camera; locked flood block evaluated once per variant"}
    d = ROOT / "results" / "real" / f"transfer-{a.source}-to-{a.target}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(json.dumps(out, indent=1))
    pd.DataFrame({"file": tgt.files, "time": tgt.times, "block": tgt.blocks, "gauge_ft": tgt.gage_ft,
                  "zero_shot_ft": level_ft, "calibrated_ft": level_ft + offset, "pooled_ft": level_pool + off_p, "below_plate": below_plate,
                  "accepted": accepted, "margin": margin, "is_ir": tgt.is_ir, "test": t_test}
                 ).to_csv(d / "predictions.csv", index=False)
    print(json.dumps({k: out[k] for k in ("n_calibration_frames", "offset_ft", "coverage_test", "baseline_always_plate_bottom_mae_cm")}))
    print("zero-shot:", {k: zs[k] for k in ("n_on_plate", "mae_cm", "rmse_cm", "bias_cm")})
    print("calibrated:", {k: cl[k] for k in ("n_on_plate", "mae_cm", "rmse_cm", "bias_cm")})
    print("pooled:", {k: po[k] for k in ("n_on_plate", "mae_cm", "rmse_cm", "bias_cm")}, "offset", off_p)
    print("by stage (zero-shot / calibrated / pooled):", zs["mae_by_stage_ft"], cl["mae_by_stage_ft"], po["mae_by_stage_ft"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
