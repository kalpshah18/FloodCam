#!/usr/bin/env python
"""Leave-one-block-out cross-validation on the DEV blocks of real USGS footage.

Never touches the locked test blocks (data/splits/usgs_<site>.json).
Usage: python scripts/real_cv.py --site in_overland --factor 1 --run-id real-cv-f1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from floodcam.real.experiment import load_dataset, run_cv, summarize, write_run  # noqa: E402
from floodcam.real.plate import SiteSpec  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="in_overland")
    ap.add_argument("--factor", type=int, default=1)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    ds = load_dataset(a.site, a.factor)
    spec = SiteSpec.load(a.site)
    dev = ds.split["dev_blocks"]
    assert not set(dev) & set(ds.split["locked_test_blocks"])
    res = run_cv(ds, spec, dev, dev, seed=a.seed, verbose=True)
    summ = summarize(ds, res)
    run_id = a.run_id or f"real-cv-{a.site}-f{a.factor}"
    out = write_run(run_id, ds, res, summ, {"protocol": "leave-one-block-out on dev blocks",
                                            "seed": a.seed, "dev_blocks": dev})
    for name, s in summ.items():
        d, f = s["depth"], s["flood_3cm"]
        print(f"{name:20s} MAE {d['mae_cm']:.2f} cm (CI {s['mae_ci95_block_bootstrap'][0]:.2f}-"
              f"{s['mae_ci95_block_bootstrap'][1]:.2f}) near3cm {d['near_3cm']['mae_cm']} "
              f"P {f['precision']:.2f} R {f['recall']:.2f} FAR {f['false_alarm_rate']:.3f}")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
