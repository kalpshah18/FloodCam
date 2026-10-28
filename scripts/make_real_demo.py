#!/usr/bin/env python
"""Bundle a small real-footage demo for the dashboard (data/usgs/demo/).

Uses held-out cross-validation predictions of one event (the estimate for each frame comes from a
model that never saw that month) and 640-px copies of the frames (USGS, public domain).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BLOCK, START, END, EVERY = "2026-09", "2026-09-17", "2026-09-22", 3


def main() -> int:
    d = pd.read_csv(ROOT / "results/real/real-cv-in_overland-f1/predictions.csv")
    d = d[(d.block == BLOCK) & (d.time >= START) & (d.time < END)].sort_values("time")
    d = d.iloc[::EVERY]
    man = {}
    for line in (ROOT / "data/usgs/manifest.jsonl").read_text().splitlines():
        r = json.loads(line)
        man[r["file"]] = r
    out = ROOT / "data/usgs/demo"
    (out / "frames").mkdir(parents=True, exist_ok=True)
    keep = []
    for r in d.itertuples():
        img = cv2.imread(str(ROOT / man[r.file]["path"]))
        img = cv2.resize(img, (640, round(img.shape[0] * 640 / img.shape[1])))
        cv2.imwrite(str(out / "frames" / r.file), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        keep.append(r.file)
    d[d.file.isin(keep)][["file", "time", "gauge_cm", "learned_cm", "baseline_cm", "accepted",
                          "is_ir", "borrowed", "lapvar"]].to_csv(out / "demo.csv", index=False)
    (out / "README.md").write_text(
        "Real USGS gauge-camera frames (public domain), site IN_Overland_Flow_..._STAFF, Sep 2026.\n"
        "`learned_cm` is the leave-one-block-out estimate (model trained without this month);\n"
        "`gauge_cm` is the official USGS gage height. Built by scripts/make_real_demo.py.\n")
    print(len(keep), "frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
