#!/usr/bin/env python
"""Pick extra registration reference views (one per month block, DEV blocks only).

For each dev block, the daytime frame that registers best to the main reference becomes an extra
reference, stored with its homography to the main reference in configs/sites/<site>_refs.json.
Only image geometry is used; no gauge labels. Locked test blocks never supply references.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from floodcam.real.plate import PlateRectifier, SiteSpec  # noqa: E402


def main(site: str = "in_overland") -> int:
    split = json.loads((ROOT / "data/splits" / f"usgs_{site}.json").read_text())
    man = {}
    for line in (ROOT / "data/usgs/manifest.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["site"] == site:
            man[r["file"]] = r
    pr = PlateRectifier(SiteSpec.load(site))
    refs = []
    for blk in split["dev_blocks"]:
        files = [f for f in split["blocks"][blk] if not man[f]["is_ir"]]
        files = files[:: max(1, len(files) // 12)][:12]          # a dozen candidates per block
        best = None
        for f in files:
            img = cv2.imread(str(ROOT / man[f]["path"]))
            reg = pr.register(img, use_extra=False)
            if reg.ok and (best is None or reg.inliers > best[1]):
                best = (f, reg.inliers, reg.H)
        if best and best[1] >= 30:
            dst = ROOT / "models" / site / f"ref_{len(refs):02d}_{blk}.jpg"   # committed copy
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / man[best[0]]["path"], dst)
            refs.append({"file": str(dst.relative_to(ROOT)), "H": np.asarray(best[2]).tolist(),
                         "block": blk, "inliers": best[1]})
            print(blk, best[0][-24:], best[1])
    (ROOT / "configs/sites" / f"{site}_refs.json").write_text(json.dumps(refs, indent=1))
    print(len(refs), "extra references")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:2]))
