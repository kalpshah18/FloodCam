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

    zs, mask = report("zero_shot", level_ft)
    cl, _ = report("calibrated", level_ft + offset)
    trivial = float(np.abs((tspec.bottom_ft - tgt.gage_ft[mask]) * FT).mean())   # always "at the plate bottom"
    out = {"source": a.source, "target": a.target, "target_dev_blocks": tgt.split["dev_blocks"],
           "target_locked_blocks": tgt.split["locked_test_blocks"],
           "n_calibration_frames": int(cal.sum()), "offset_ft": offset, "coverage_test": float(accepted[t_test].mean()),
           "zero_shot": zs, "calibrated": cl, "baseline_always_plate_bottom_mae_cm": trivial,
           "note": "classifier never trained on the target camera; locked flood block evaluated once per variant"}
    d = ROOT / "results" / "real" / f"transfer-{a.source}-to-{a.target}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(json.dumps(out, indent=1))
    pd.DataFrame({"file": tgt.files, "time": tgt.times, "block": tgt.blocks, "gauge_ft": tgt.gage_ft,
                  "zero_shot_ft": level_ft, "calibrated_ft": level_ft + offset, "below_plate": below_plate,
                  "accepted": accepted, "margin": margin, "is_ir": tgt.is_ir, "test": t_test}
                 ).to_csv(d / "predictions.csv", index=False)
    print(json.dumps({k: out[k] for k in ("n_calibration_frames", "offset_ft", "coverage_test", "baseline_always_plate_bottom_mae_cm")}))
    print("zero-shot:", {k: zs[k] for k in ("n_on_plate", "mae_cm", "rmse_cm", "bias_cm")})
    print("calibrated:", {k: cl[k] for k in ("n_on_plate", "mae_cm", "rmse_cm", "bias_cm")})
    return 0


if __name__ == "__main__":
    sys.exit(main())
