#!/usr/bin/env python
"""Register every manifest frame to its site reference and cache the rectified plate strip.

--factor f simulates a camera with f-times fewer pixels per side (frame downsampled by f before
registration; strip resolution scaled to match), for the resolution experiment.

Usage: python scripts/extract_strips.py --site in_overland --factor 1 [--workers 4]
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from floodcam.real.plate import PlateRectifier, SiteSpec  # noqa: E402

NATIVE_ROWS, NATIVE_COLS = 360, 48
_PR: PlateRectifier | None = None
_F = 1


def _init(site: str, factor: int) -> None:
    global _PR, _F
    _F = factor
    spec = SiteSpec.load(site)
    _PR = PlateRectifier(spec, max(6, round(NATIVE_ROWS / factor)), max(4, round(NATIVE_COLS / factor)),
                         scale=1.0 / factor)


def _work(path: str):
    img = cv2.imread(str(ROOT / path))
    reg = _PR.register(img)
    return _PR.strip(img, reg), reg.ok, reg.inliers


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--factor", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    rows = [json.loads(line) for line in (ROOT / "data/usgs/manifest.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r["site"] == a.site]
    with Pool(a.workers, initializer=_init, initargs=(a.site, a.factor)) as pool:
        out = pool.map(_work, [r["path"] for r in rows], chunksize=16)
    strips = np.stack([o[0] for o in out])
    ok = np.array([o[1] for o in out])
    inl = np.array([o[2] for o in out])
    dest = ROOT / "data/usgs/strips"
    dest.mkdir(exist_ok=True)
    np.savez_compressed(dest / f"{a.site}_f{a.factor}.npz", strips=strips, reg_ok=ok, inliers=inl,
                        files=np.array([r["file"] for r in rows]))
    print(f"{a.site} f={a.factor}: {strips.shape}, registered {ok.mean():.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
