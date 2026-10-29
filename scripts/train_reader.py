#!/usr/bin/env python
"""Train the production plate reader on ALL labelled frames of a site and save the bundle.

Evaluation numbers in results/real/ come from models that never saw the evaluated blocks. This
production model additionally uses the locked test blocks (standard practice once the protocol
is finished), so do not evaluate it on them.

Usage: python scripts/train_reader.py --site in_overland
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from floodcam.real.experiment import fit_models, load_dataset  # noqa: E402
from floodcam.real.plate import SiteSpec  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="in_overland")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    ds = load_dataset(a.site, 1)
    spec = SiteSpec.load(a.site)
    train = np.ones(len(ds.files), bool)
    tpl, feats, reader, base = fit_models(ds, spec, train, a.seed)
    cm, _ = reader.predict_cm(feats)
    in_sample_mae = float(np.abs(cm - ds.gage_cm).mean())   # sanity only; NOT a performance claim
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    bundle = {
        "site": a.site, "spec": asdict(spec), "reader": reader,
        "strip_rows": ds.R, "strip_cols": int(ds.strips.shape[2]),
        "lap_lo": base["lap_lo"], "lap_hi": base["lap_hi"],
        "trained_on_blocks": sorted(set(ds.blocks.tolist())), "n_frames": int(len(ds.files)),
        "seed": a.seed, "git_commit": git, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "manifest_sha256_16": hashlib.sha256((ROOT / "data/usgs/manifest.jsonl").read_bytes()).hexdigest()[:16],
        "in_sample_mae_cm": in_sample_mae,
        "note": "Trained on all blocks including the locked test blocks; see results/real for "
                "held-out evaluation of the dev-only model.",
    }
    out = ROOT / "models" / a.site / "reader.joblib"
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out, compress=3)
    (out.parent / "reader_info.json").write_text(json.dumps(
        {k: v for k, v in bundle.items() if k not in ("reader", "spec")}, indent=1))
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB), in-sample MAE {in_sample_mae:.2f} cm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
