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


def _register(path: str):
    img = cv2.imread(str(ROOT / path))
    reg = _PR.register(img)
    g = cv2.cvtColor(cv2.resize(img, (640, round(img.shape[0] * 640 / img.shape[1]))), cv2.COLOR_BGR2GRAY)
    return reg.H, reg.ok, reg.inliers, reg.ref_id, float(cv2.Laplacian(g, cv2.CV_64F).var()), float(g.mean())


def _strip(arg):
    path, H = arg
    from floodcam.real.plate import Registration

    img = cv2.imread(str(ROOT / path))
    return _PR.strip(img, Registration(H, 0, True))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--factor", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    rows = [json.loads(line) for line in (ROOT / "data/usgs/manifest.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r["site"] == a.site]
    with Pool(a.workers, initializer=_init, initargs=(a.site, a.factor)) as pool:
        regs = pool.map(_register, [r["path"] for r in rows], chunksize=16)
        Hs = [r[0] for r in regs]
        ok = np.array([r[1] for r in regs])
        # Causal fallback: a frame that fails to register borrows the homography of the most recent
        # earlier frame that did (the camera is fixed, so its pose changes rarely). Uses only past
        # frames, as a deployed system could. Frames before the first success keep ok=False.
        borrowed = np.zeros(len(rows), bool)
        last = None
        for i in range(len(rows)):
            if ok[i]:
                last = Hs[i]
            elif last is not None:
                Hs[i] = last
                borrowed[i] = True
        strips = np.stack(pool.map(_strip, [(r["path"], H) for r, H in zip(rows, Hs)], chunksize=16))
    dest = ROOT / "data/usgs/strips"
    dest.mkdir(exist_ok=True)
    np.savez_compressed(
        dest / f"{a.site}_f{a.factor}.npz", strips=strips, reg_ok=ok, borrowed=borrowed,
        inliers=np.array([r[2] for r in regs]), ref_id=np.array([r[3] for r in regs]),
        lapvar=np.array([r[4] for r in regs]), bright=np.array([r[5] for r in regs]),
        files=np.array([r["file"] for r in rows]),
    )
    print(f"{a.site} f={a.factor}: {strips.shape}, registered {ok.mean():.1%}, "
          f"borrowed {borrowed.mean():.1%}, unresolved {(~ok & ~borrowed).mean():.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
