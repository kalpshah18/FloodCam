#!/usr/bin/env python
"""Evaluate a segmentation backend: mask IoU/F1, optionally waterline error and depth MAE.

  python scripts/eval_seg.py --backend hsv --split val --with-depth
  python scripts/eval_seg.py --backend segformer --ckpt checkpoints/seg-ft-001/best \
      --split val --with-depth

Data: a scene manifest (default data/synthetic/manifest.jsonl, from `make synth-demo`) and a
scene-level splits file. The locked final-test scene is always refused here; the final test
goes through scripts/final_test.py only. Output is tagged synthetic=true when any evaluated
row is synthetic: NO real-data numbers exist yet, and synthetic numbers are not results.

--with-depth reports the chosen backend next to the HSV baseline on the same frames
(synthetic scenes only: calibration is rebuilt from the renderer geometry).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from floodcam.config import ExperimentConfig  # noqa: E402
from floodcam.depth.estimate import (  # noqa: E402
    estimate,
    estimate_from_mask,
    make_camera_config_for_renderer,
)
from floodcam.io.manifest import read_jsonl  # noqa: E402
from floodcam.seg.backend import HsvBackend  # noqa: E402
from floodcam.seg.datasets import binarize_mask, file_sha256  # noqa: E402
from floodcam.seg.metrics import ConfusionCounts  # noqa: E402
from floodcam.seg.train import load_locked_scenes  # noqa: E402

DEPTH_BINS = [(0.0, 1.0), (1.0, 3.0), (3.0, 5.0), (5.0, 10.0), (10.0, 1e9)]


def _bin_name(lo: float, hi: float) -> str:
    return f"{lo:g}-{'inf' if hi > 1e8 else f'{hi:g}'}cm"


def _load_img(path: Path) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def _synthetic_camera(scene_id: str):
    from floodcam.synth.timelapse import _scene_for_id

    renderer, _ = _scene_for_id(scene_id)
    return make_camera_config_for_renderer(renderer), renderer


def _mae_report(errs: list[tuple[float, float]]) -> dict:
    """errs: (true_depth, abs_error)."""
    out: dict = {"n": len(errs)}
    if not errs:
        return out
    out["mae_cm"] = float(np.mean([e for _, e in errs]))
    per_bin = {}
    for lo, hi in DEPTH_BINS:
        v = [e for d, e in errs if lo <= d < hi]
        per_bin[_bin_name(lo, hi)] = {"n": len(v), "mae_cm": float(np.mean(v)) if v else None}
    out["mae_per_depth_bin"] = per_bin
    near = [e for d, e in errs if 2.0 <= d <= 5.0]
    out["mae_near_3cm_2to5"] = {"n": len(near), "mae_cm": float(np.mean(near)) if near else None}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--backend", choices=["hsv", "segformer"], default="hsv")
    ap.add_argument("--ckpt", type=Path)
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "synthetic" / "manifest.jsonl")
    ap.add_argument("--splits-file", type=Path,
                    default=ROOT / "data" / "splits" / "synthetic_splits.json")
    ap.add_argument("--split", choices=["train", "val"], default="val")
    ap.add_argument("--with-depth", action="store_true")
    ap.add_argument("--run-id", default="evalseg-001")
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--hsv-lower", type=int, nargs=3, default=[90, 30, 30])
    ap.add_argument("--hsv-upper", type=int, nargs=3, default=[120, 255, 255])
    ap.add_argument("--prob-threshold", type=float, default=0.5)
    a = ap.parse_args()

    exp = ExperimentConfig(
        hsv_lower=a.hsv_lower, hsv_upper=a.hsv_upper, seg_prob_threshold=a.prob_threshold
    )
    if a.backend == "segformer":
        from floodcam.seg.segformer import SegFormerBackend

        backend = SegFormerBackend(exp, a.ckpt)
    else:
        backend = HsvBackend(exp)

    splits = json.loads(a.splits_file.read_text(encoding="utf-8"))
    scenes = set(splits["coarse_train_scenes" if a.split == "train" else "coarse_val_scenes"])
    locked = load_locked_scenes(a.splits_file.parent)
    if scenes & locked:
        print(f"REFUSED: locked scene(s) {sorted(scenes & locked)} in split", file=sys.stderr)
        return 3
    rows = [r for r in read_jsonl(a.manifest) if r.scene_id in scenes and not r.is_negative]
    if not rows:
        print("no evaluable rows (run `make synth-demo`?)", file=sys.stderr)
        return 2
    synthetic = any(r.synthetic for r in rows)

    cc = ConfusionCounts()
    cams: dict = {}
    seg_err: list = []
    hsv_err: list = []
    wl_seg: list = []
    wl_hsv: list = []
    for r in rows:
        img = _load_img(ROOT / r.image_path)
        gt = binarize_mask(cv2.imread(str(ROOT / r.mask_path), cv2.IMREAD_UNCHANGED), None)
        mask = backend.predict_mask(img)
        cc.update(mask > 0, gt > 0)
        if a.with_depth:
            if not synthetic:
                print("--with-depth is implemented for synthetic scenes only", file=sys.stderr)
                return 2
            if r.scene_id not in cams:
                cams[r.scene_id] = _synthetic_camera(r.scene_id)
            cam, rend = cams[r.scene_id]
            true_wl = rend.project_height(r.depth_cm)[0]
            for fn, errs, wls in (
                (lambda: estimate_from_mask(mask, cam), seg_err, wl_seg),
                (lambda: estimate(img, cam, exp), hsv_err, wl_hsv),
            ):
                d = fn()
                if d.valid:
                    errs.append((r.depth_cm, abs(d.depth_cm - r.depth_cm)))
                    wls.append(abs(d.waterline_row - true_wl))
                else:
                    errs.append((r.depth_cm, float("nan")))
    metrics = {
        "run_id": a.run_id, "synthetic": synthetic,
        "note": "SYNTHETIC - pipeline validation only; not a result." if synthetic else "",
        "backend": a.backend, "model_version": backend.model_version, "split": a.split,
        "scenes": sorted(scenes), "n_frames": len(rows), "mask": cc.to_dict(),
        "mask_note": "synthetic masks are half-plane approximations; IoU not interpretable"
        if synthetic else "",
    }
    if a.with_depth:
        def pack(errs, wls):
            valid = [(d, e) for d, e in errs if not np.isnan(e)]
            rep = _mae_report(valid)
            rep["n_invalid"] = len(errs) - len(valid)
            rep["waterline_abs_err_px_mean"] = float(np.mean(wls)) if wls else None
            return rep
        metrics["depth_backend"] = pack(seg_err, wl_seg)
        metrics["depth_hsv_baseline"] = pack(hsv_err, wl_hsv)

    out = a.results_dir / a.run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, cwd=ROOT).stdout.strip()
    except Exception:
        commit = "unknown"
    meta = {
        "run_id": a.run_id, "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit, "manifest": str(a.manifest), "manifest_sha256": file_sha256(a.manifest),
        "splits_file": str(a.splits_file), "synthetic": synthetic, "backend": a.backend,
        "ckpt": str(a.ckpt) if a.ckpt else None, "args": {k: str(v) for k, v in vars(a).items()},
        "locked_scene_touched": False,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({"mask": metrics["mask"], **{k: metrics[k] for k in metrics
                                                   if k.startswith("depth_")}}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
