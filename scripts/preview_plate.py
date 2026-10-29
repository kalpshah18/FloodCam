#!/usr/bin/env python
"""Check a site's plate geometry: reference with the quad drawn + rectified strips of sample frames.

Use it when adding a camera: edit configs/sites/<site>.yaml (quad corners TL,TR,BR,BL as 0-1
fractions of the reference image; top_ft / bottom_ft = gauge values at the top / bottom quad edge),
re-run, and iterate until the strips of frames from different months line up and the tick labels sit
at the same rows. Writes results/real/preview_<site>.jpg.

Usage: python scripts/preview_plate.py --site in_overland [--n 14]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from floodcam.real.plate import PlateRectifier, SiteSpec  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--n", type=int, default=14)
    a = ap.parse_args()
    spec = SiteSpec.load(a.site)
    pr = PlateRectifier(spec)
    ref = cv2.imread(str(ROOT / spec.reference))
    h, w = ref.shape[:2]
    cv2.polylines(ref, [(np.array(spec.quad) * [w, h]).astype(int)], True, (0, 255, 255), 2)
    rows = [json.loads(line) for line in (ROOT / "data/usgs/manifest.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r["site"] == a.site and not r["is_ir"]]
    pick = rows[:: max(1, len(rows) // a.n)][: a.n]
    tiles = []
    for r in pick:
        img = cv2.imread(str(ROOT / r["path"]))
        reg = pr.register(img)
        t = cv2.resize(pr.strip(img, reg), (96, 360))
        cv2.putText(t, f"{reg.inliers}", (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 0) if reg.ok else (0, 0, 255), 1)
        cv2.putText(t, r["time"][2:10], (2, 352), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        tiles.append(t)
    strip = np.hstack(tiles)
    ref_small = cv2.resize(ref, (640, round(640 * h / w)))
    out = ROOT / "results" / "real" / f"preview_{a.site}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), strip)
    cv2.imwrite(str(out.with_name(f"preview_{a.site}_reference.jpg")), ref_small)
    print("wrote", out, "and the reference overlay")
    return 0


if __name__ == "__main__":
    sys.exit(main())
