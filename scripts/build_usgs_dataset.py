#!/usr/bin/env python
"""Download real USGS gauge-camera frames and build data/usgs/manifest.jsonl.

Frames come from the public S3 bucket usgs-nims-images; labels from data/usgs/truth/*.csv
(official gauge height). Frames are resized to --width and re-encoded; originals are not kept.

Usage:
    python scripts/build_usgs_dataset.py [--sites in_overland hancock] [--width 1280] [--workers 8]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BUCKET = "https://usgs-nims-images.s3.amazonaws.com/overlay"
CAMERAS = {
    "in_overland": "IN_Overland_Flow_East_of_Bull_Rapids_Road_near_Harlan_STAFF",
    "hancock": "NY_East_Branch_Delaware_River_at_Hancock_Staff",
}
FT_TO_CM = 30.48


def fetch(url: str, retries: int = 3) -> bytes | None:
    for k in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=40) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None
        except Exception:
            pass
        time.sleep(1.5 * (k + 1))
    return None


def process(args: tuple[str, str, float, str, Path, int]) -> dict | None:
    site, fname, gage_ft, t, out_dir, width = args
    dest = out_dir / fname
    if not dest.exists():
        data = fetch(f"{BUCKET}/{CAMERAS[site]}/{fname}")
        if data is None:
            return None
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        if img.shape[1] > width:
            img = cv2.resize(img, (width, round(img.shape[0] * width / img.shape[1])),
                             interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dest), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    img = cv2.imread(str(dest))
    if img is None:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = float(hsv[..., 1].mean())
    return {
        "site": site, "file": fname, "path": str(dest.relative_to(ROOT)), "time": t,
        "gage_ft": float(gage_ft), "gage_cm": float(gage_ft) * FT_TO_CM,
        "mean_sat": sat, "is_ir": bool(sat < 6.0),  # infrared night frames are greyscale
        "height": int(img.shape[0]), "width": int(img.shape[1]), "synthetic": False,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", nargs="+", default=list(CAMERAS))
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N rows per site")
    a = ap.parse_args()

    records = []
    for site in a.sites:
        truth = pd.read_csv(ROOT / "data" / "usgs" / "truth" / f"{site}_gauge.csv")
        if a.limit:
            truth = truth.head(a.limit)
        out_dir = ROOT / "data" / "usgs" / "frames" / site
        out_dir.mkdir(parents=True, exist_ok=True)
        jobs = [(site, r.file, r.gage_ft, r.time, out_dir, a.width) for r in truth.itertuples()]
        done = 0
        with ThreadPoolExecutor(a.workers) as ex:
            for rec in ex.map(process, jobs):
                done += 1
                if rec:
                    records.append(rec)
                if done % 200 == 0:
                    print(f"{site}: {done}/{len(jobs)} ({len(records)} ok)", flush=True)
        print(f"{site}: {sum(r['site'] == site for r in records)}/{len(jobs)} frames")
    records.sort(key=lambda r: (r["site"], r["time"]))
    man = ROOT / "data" / "usgs" / "manifest.jsonl"
    man.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    print(f"wrote {man} ({len(records)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
