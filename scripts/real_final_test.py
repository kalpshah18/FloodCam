#!/usr/bin/env python
"""One-shot evaluation on the LOCKED test blocks of real USGS footage.

Trains on all dev blocks (never the locked ones), evaluates the locked blocks once, logs every
run to results/real/final_test_log.jsonl. Refuses to run a second time without --force.
Do not tune anything using these numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from floodcam.real.experiment import (  # noqa: E402
    fit_models,
    gate,
    load_dataset,
    predict_baseline,
    summarize,
    write_run,
)
from floodcam.real.plate import SiteSpec  # noqa: E402

import numpy as np  # noqa: E402

LOG = ROOT / "results" / "real" / "final_test_log.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="in_overland")
    ap.add_argument("--factor", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    prior = [json.loads(x) for x in LOG.read_text().splitlines()] if LOG.exists() else []
    if any(p["site"] == a.site and p["factor"] == a.factor for p in prior) and not a.force:
        print("Locked test already evaluated for this site/factor; refusing without --force.")
        return 2
    ds = load_dataset(a.site, a.factor)
    spec = SiteSpec.load(a.site)
    locked = ds.split["locked_test_blocks"]
    train = np.isin(ds.blocks, ds.split["dev_blocks"])
    test = np.isin(ds.blocks, locked)
    assert not (train & test).any()
    _, feats, reader, base = fit_models(ds, spec, train, a.seed)
    cm, mg = reader.predict_cm(feats[test])
    res = {"preds": np.full(len(ds.files), np.nan, np.float32), "margin": np.full(len(ds.files), np.nan, np.float32),
           "baseline": np.full(len(ds.files), np.nan, np.float32), "sel": test, "taus": {"all_dev": base["tau"]},
           "accepted": gate(ds.quality[:, 0], base)}
    res["preds"][test], res["margin"][test] = cm, mg
    res["baseline"][test] = predict_baseline(feats[test], reader, base)
    summ = summarize(ds, res)
    run_id = f"real-final-{a.site}-f{a.factor}" + ("-force" if a.force and prior else "")
    out = write_run(run_id, ds, res, summ, {"protocol": "train on all dev blocks, evaluate locked blocks once",
                                            "locked_blocks": locked, "seed": a.seed})
    with LOG.open("a") as f:
        f.write(json.dumps({"time": time.time(), "site": a.site, "factor": a.factor,
                            "run_id": run_id, "forced": bool(a.force)}) + "\n")
    print(f"coverage after sharpness gate: {summ['coverage']:.1%}")
    for scope in ("all_frames", "gated"):
        for name, sc in summ[scope].items():
            d, fl = sc["depth"], sc["flood_3cm"]
            print(f"{scope:10s} {name:20s} MAE {d['mae_cm']:.2f} cm  P {fl['precision']:.2f} R {fl['recall']:.2f}")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
