#!/usr/bin/env python
"""Leave-one-camera-out (LOCO) evaluation of the frozen reader on several USGS staff-plate cameras.

For each target camera the classifier is trained on rows from the OTHER cameras only (in_overland dev
blocks; every frame of the others), so the target camera's frames never enter training. What uses target
data, and nothing else:
  * dry template and Laplacian-variance gate thresholds: from the target's dry daytime frames
    (gauge <= 0.02 ft) of its calibration block;
  * calibration variants: zero-shot (nominal plate geometry) and affine-calibrated on the target's
    EARLIEST block only; all other blocks of the target are evaluated.
The pipeline (features, classifier settings, decode, gate) is the one frozen before these cameras were
downloaded; nothing was tuned on them.

Usage: python scripts/real_multicam.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from floodcam.real.experiment import GATE_HI, GATE_LO, load_dataset  # noqa: E402
from floodcam.real.metrics import block_summary, depth_metrics  # noqa: E402
from floodcam.real.plate import SiteSpec  # noqa: E402
from floodcam.real.reader import DryTemplate, PlateReader, decode_boundary, row_features  # noqa: E402

SITES = ["in_overland", "in_north", "oh_wetland", "wi_mt1"]
FT = 30.48
ROWS_PER_SOURCE = 90_000


def prepare(site: str):
    ds = load_dataset(site, 1)
    spec = SiteSpec.load(site)
    usable = np.isin(ds.blocks, ds.split["dev_blocks"])      # never touch locked blocks
    first = sorted(set(ds.blocks[usable]))[0]
    calib = usable & (ds.blocks == first)
    dry = usable & (ds.gage_ft <= 0.02) & ~ds.is_ir & ds.reg_ok
    tpl = DryTemplate.fit(ds.stats, dry & calib if (dry & calib).sum() >= 10 else dry)
    feats = row_features(ds.stats, tpl, ds.is_ir, ds.reg_ok, extra=ds.quality)
    lap = np.expm1(ds.quality[:, 0])
    med = float(np.median(lap[calib]))
    accepted = (lap >= GATE_LO * med) & (lap <= GATE_HI * med)
    return dict(ds=ds, spec=spec, usable=usable, calib=calib, first=first, feats=feats, accepted=accepted)


def main() -> int:
    P = {s: prepare(s) for s in SITES}
    rng = np.random.default_rng(0)
    out_rows, summary = [], {}
    for tgt in SITES:
        X, y = [], []
        for src in SITES:
            if src == tgt:
                continue
            d = P[src]
            R = d["feats"].shape[1]
            labels = PlateReader(d["spec"].top_ft, d["spec"].bottom_ft).labels(d["ds"].gage_ft, R)
            Xs = d["feats"][d["usable"]].reshape(-1, d["feats"].shape[2])
            ys = labels[d["usable"]].reshape(-1)
            sel = rng.choice(len(ys), min(len(ys), ROWS_PER_SOURCE), replace=False)
            X.append(Xs[sel])
            y.append(ys[sel])
        clf = HistGradientBoostingClassifier(random_state=0, max_iter=120, max_depth=6, learning_rate=0.1).fit(
            np.vstack(X), np.concatenate(y))
        d = P[tgt]
        ds, spec, feats = d["ds"], d["spec"], d["feats"]
        R = feats.shape[1]
        p = clf.predict_proba(feats.reshape(-1, feats.shape[2]))[:, 1].reshape(len(feats), -1)
        b, margin = decode_boundary(p)
        lvl = spec.top_ft - (b / R) * (spec.top_ft - spec.bottom_ft)
        below = lvl <= spec.bottom_ft + 1e-6
        zero_cm = np.where(below, 0.0, np.clip(lvl, 0, None)) * FT
        cal = d["calib"] & d["accepted"] & (ds.gage_ft > 0.25) & (ds.gage_ft < spec.top_ft - 0.1) & ~below
        if cal.sum() >= 10:
            A = np.stack([lvl[cal], np.ones(cal.sum())], axis=1)
            a, c = np.linalg.lstsq(A, ds.gage_ft[cal], rcond=None)[0]
        else:
            a, c = 1.0, 0.0
        cal_cm = np.where(below, 0.0, np.clip(a * lvl + c, 0, None)) * FT
        ev = d["usable"] & ~d["calib"]
        m = ev & d["accepted"]
        res = {}
        for name, pred in (("zero_shot", zero_cm), ("calibrated", cal_cm), ("always_0", np.zeros_like(zero_cm))):
            res[name] = {"gated": depth_metrics(ds.gage_cm[m], pred[m]),
                         "by_block": block_summary(ds.blocks[m], ds.gage_cm[m], pred[m]),
                         "all_eval_frames": depth_metrics(ds.gage_cm[ev], pred[ev])}
        res["n_eval"] = int(ev.sum())
        res["coverage"] = float(d["accepted"][ev].mean())
        res["calibration_block"] = d["first"]
        res["n_calibration_frames"] = int(cal.sum())
        res["cal_a"], res["cal_b"] = float(a), float(c)
        res["sources"] = [s for s in SITES if s != tgt]
        res["gauge_max_cm"] = float(ds.gage_cm[ev].max())
        summary[tgt] = res
        out_rows.append(pd.DataFrame({"site": tgt, "file": ds.files[ev], "time": ds.times[ev], "block": ds.blocks[ev],
                                      "gauge_cm": ds.gage_cm[ev], "zero_shot_cm": zero_cm[ev], "calibrated_cm": cal_cm[ev],
                                      "accepted": d["accepted"][ev], "margin": margin[ev]}))
        print(f"{tgt}: n={res['n_eval']} cov={res['coverage']:.2f} "
              f"zero MAE {res['zero_shot']['gated']['mae_cm']:.2f}  cal MAE {res['calibrated']['gated']['mae_cm']:.2f}  "
              f"always0 {res['always_0']['gated']['mae_cm']:.2f}  (max gauge {res['gauge_max_cm']:.0f} cm)", flush=True)
    d = ROOT / "results" / "real" / "loco"
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(json.dumps(summary, indent=1))
    pd.concat(out_rows).to_csv(d / "predictions.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
