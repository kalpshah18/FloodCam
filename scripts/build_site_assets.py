#!/usr/bin/env python
"""Build the data and image assets of the project website (site/) from committed results.

Nothing on the site is typed by hand: numbers come from paper/numbers.tex (itself generated from
results/ by scripts/make_paper_assets.py), tables from results/real/*/metrics.json and
results/real/tables/*.csv, and the demo from data/usgs/demo/ (held-out cross-validation estimates).

Outputs (all committed, all small):
  site/_data/numbers.json      every paper macro, name -> string
  site/_data/results.json      CV and locked-test tables, per-block MAE, camera table
  site/_data/resolution.json   resolution sweep
  site/_data/failures.json     failure-audit overlays
  site/assets/img/*.png        paper figures (rendered from paper/figs/*.pdf) and result figures
  site/assets/paper.pdf        the paper
  site/assets/failures/*.jpg   failure-audit overlays
  site/assets/demo/            frames.json, frame_XX.jpg (plate outlined), strip_XX.png, resolution.json

Usage: python scripts/build_site_assets.py [--skip-demo]
Needs pdftoppm (poppler / MiKTeX) for the figure conversion.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
DATA = SITE / "_data"
ASSETS = SITE / "assets"
sys.path.insert(0, str(ROOT / "src"))

CV_RUN = "real-cv-in_overland-f1"
FINAL_RUN = "real-final-in_overland-f1-force"
RESULT_FIGS = ["scatter_cv", "block_mae_cv", "resolution", "risk_coverage_cv", "timeseries_2026-09",
               "scatter_final", "timeseries_final_2025-04"]
CAMERAS = [("in_overland", "cream plate", "A"), ("in_north", "cream plate", "B"),
           ("oh_wetland", "white plate", "C"), ("wi_mt1", "orange plate", "D")]
FT_TO_CM = 30.48


def dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=False), encoding="utf-8")


# --------------------------------------------------------------------------------------- numbers
def build_numbers() -> dict[str, str]:
    text = (ROOT / "paper" / "numbers.tex").read_text(encoding="utf-8")
    nums = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", text))
    dump(nums, DATA / "numbers.json")
    print(f"numbers.json: {len(nums)} macros")
    return nums


# --------------------------------------------------------------------------------------- figures
def build_figures() -> None:
    exe = shutil.which("pdftoppm")
    if exe is None:
        raise SystemExit("pdftoppm not found (install poppler or MiKTeX)")
    out = ASSETS / "img"
    out.mkdir(parents=True, exist_ok=True)
    used = set(re.findall(r"includegraphics\[[^\]]*\]\{figs/(\w+)\.pdf\}",
                          (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")))
    for pdf in sorted((ROOT / "paper" / "figs").glob("*.pdf")):
        if pdf.stem not in used:
            continue
        subprocess.run([exe, "-png", "-r", "170", "-singlefile", str(pdf), str(out / pdf.stem)], check=True)
    for name in RESULT_FIGS:
        shutil.copy(ROOT / "results" / "real" / "figures" / f"{name}.png", out / f"res_{name}.png")
    shutil.copy(ROOT / "paper" / "main.pdf", ASSETS / "paper.pdf")
    print("figures:", len(list(out.glob("*.png"))), "png +", "paper.pdf")


# --------------------------------------------------------------------------------------- results
def _scope(m: dict) -> dict:
    out = {}
    for method, sc in m.items():
        d = sc["depth"]
        lo, hi = sc["mae_ci95_block_bootstrap"]
        bins = d["mae_by_bin_cm"]
        hi_n = bins["10-25"]["n"] + bins[">=25"]["n"]   # pool the two bins above 10 cm by frame count
        gt10 = (bins["10-25"]["n"] * bins["10-25"]["mae_cm"] + bins[">=25"]["n"] * bins[">=25"]["mae_cm"]) / hi_n
        out[method] = {"n": d["n"], "mae": d["mae_cm"], "rmse": d["rmse_cm"], "bias": d["bias_cm"],
                       "ci_lo": lo, "ci_hi": hi, "gt10": gt10,
                       "bins": {k: v["mae_cm"] for k, v in bins.items()}}
    return out


def build_results(nums: dict[str, str]) -> None:
    res = {}
    for key, run in (("cv", CV_RUN), ("locked", FINAL_RUN)):
        m = json.loads((ROOT / "results" / "real" / run / "metrics.json").read_text())
        res[key] = {"coverage": m["coverage"], "gate": m["gate"],
                    "all_frames": _scope(m["all_frames"]), "gated": _scope(m["gated"])}
        pb = m["gated"]["learned"]["by_block"]["per_block"]
        res[key]["blocks"] = [{"block": b, "n": v["n"], "mae": v["mae_cm"], "max_gauge": v["max_gt_cm"]}
                              for b, v in sorted(pb.items())]
    cams = []
    for site, plate, k in CAMERAS:
        cams.append({"camera": site, "plate": plate, "max_gauge": nums[f"cam{k}Max"],
                     "in_site": nums[f"cam{k}InMae"], "in_site_zero": nums[f"cam{k}InZero"],
                     "heldout": nums[f"cam{k}LocoZero"], "heldout_trivial": nums[f"cam{k}LocoTriv"],
                     "heldout_calibrated": nums[f"cam{k}LocoCal"], "coverage": nums[f"cam{k}Cov"]})
    res["cameras"] = cams
    res["hancock"] = {"range_cm": nums["hancRangeCm"], "n": nums["hancN"], "zero_shot": nums["hancMaeZero"],
                      "plate_bottom_guess": nums["hancTrivial"], "calibrated": nums["hancMaeCal"]}
    dump(res, DATA / "results.json")

    rows = list(csv.DictReader((ROOT / "results" / "real" / "tables" / "resolution.csv").open()))
    sweep = []
    for r in rows:
        if r["scope"] != "gated":
            continue
        sweep.append({"factor": int(r["factor"]), "cm_per_px": round(float(r["cm_per_px"]), 3),
                      "mae": round(float(r["learned_mae"]), 2), "rmse": round(float(r["learned_rmse"]), 2),
                      "baseline_mae": round(float(r["baseline_mae"]), 2), "n": int(r["n"])})
    dump(sweep, DATA / "resolution.json")
    dump(sweep, ASSETS / "demo" / "resolution.json")

    fails = list(csv.DictReader((ROOT / "results" / "real" / "failures" / "failures.csv").open()))
    (ASSETS / "failures").mkdir(parents=True, exist_ok=True)
    keep = []
    for f in fails:
        src = ROOT / "results" / "real" / "failures" / f["overlay"]
        if src.exists():
            shutil.copy(src, ASSETS / "failures" / f["overlay"])
            keep.append({"kind": f["kind"], "overlay": f["overlay"], "gauge_cm": round(float(f["gauge_cm"]), 1),
                         "est_cm": round(float(f["est_cm"]), 1), "is_ir": f["is_ir"] == "True",
                         "borrowed": float(f["borrowed"]) > 0})
    dump(keep, DATA / "failures.json")
    print("results.json, resolution.json, failures.json:", len(keep), "overlays")


# ------------------------------------------------------------------------------------------ demo
def build_demo() -> None:
    """Frame list for the in-browser demo.

    Depth values are the leave-one-month-out estimates already stored in data/usgs/demo/demo.csv
    (the model never saw September 2026). The plate outline and strip are re-derived here from the
    640 px demo copies by the project's own registration code, for display only.
    """
    from floodcam.real.infer import prep
    from floodcam.real.plate import PlateRectifier, Registration, SiteSpec

    demo_dir = ROOT / "data" / "usgs" / "demo"
    out = ASSETS / "demo"
    for old in list(out.glob("frame_*.jpg")) + list(out.glob("strip_*.png")):
        old.unlink()
    rows = list(csv.DictReader((demo_dir / "demo.csv").open()))
    rows.sort(key=lambda r: r["time"])
    spec = SiteSpec.load("in_overland")
    pr = PlateRectifier(spec)
    quad_ref = pr._quad_px.reshape(-1, 1, 2).astype(np.float64)
    lap_ok = [float(r["lapvar"]) for r in rows if r["accepted"] == "True"]
    lap_med = float(np.median(lap_ok))
    last_H = None
    frames = []
    for i, r in enumerate(rows):
        img640 = cv2.imread(str(demo_dir / "frames" / r["file"]))
        img = prep(img640)
        reg = pr.register(img)
        reg_ok = bool(reg.ok)
        H = reg.H if reg.ok else last_H
        if reg.ok:
            last_H = reg.H
        quad_png = None
        if H is not None:
            reg_use = Registration(H, reg.inliers, True)
            strip = pr.strip(img, reg_use)
            cv2.imwrite(str(out / f"strip_{i:02d}.png"), strip)
            q = cv2.perspectiveTransform(quad_ref, np.linalg.inv(H)).reshape(-1, 2) * (640 / img.shape[1])
            quad_png = [[round(float(x), 1), round(float(y), 1)] for x, y in q]
            cv2.polylines(img640, [q.astype(np.int32)], True, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(out / f"frame_{i:02d}.jpg"), img640, [cv2.IMWRITE_JPEG_QUALITY, 82])
        acc = r["accepted"] == "True"
        lap = float(r["lapvar"])
        reason = None if acc else ("low sharpness (fog, blur or darkness)" if lap < lap_med
                                   else "very high sharpness (glare or dirt on the lens)")
        frames.append({"i": i, "time": r["time"], "gauge_cm": round(float(r["gauge_cm"]), 2),
                       "reader_cm": round(float(r["learned_cm"]), 2),
                       "baseline_cm": round(float(r["baseline_cm"]), 2), "accepted": acc,
                       "reject_reason": reason, "is_ir": r["is_ir"] == "True",
                       "registered": reg_ok, "has_strip": H is not None, "quad": quad_png})
    ft = lambda ft_: round(ft_ * FT_TO_CM, 2)  # noqa: E731
    meta = {"site": "in_overland", "strip_rows": pr.strip_rows, "strip_cols": pr.strip_cols,
            "plate_top_cm": ft(spec.top_ft), "plate_bottom_cm": ft(spec.bottom_ft),
            "frame_width": 640, "n_frames": len(frames),
            "source": "data/usgs/demo/demo.csv (leave-one-month-out estimates)"}
    dump({"meta": meta, "frames": frames}, out / "frames.json")
    n_ok = sum(f["registered"] for f in frames)
    print(f"demo: {len(frames)} frames, {n_ok} registered on their own, "
          f"{sum(f['has_strip'] for f in frames)} with a strip")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-demo", action="store_true")
    a = ap.parse_args()
    nums = build_numbers()
    build_figures()
    build_results(nums)
    if not a.skip_demo:
        build_demo()
    return 0


if __name__ == "__main__":
    sys.exit(main())
