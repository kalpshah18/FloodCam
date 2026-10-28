"""Shared experiment code for the real-footage studies (cross-validation, final test, resolution)."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from floodcam.real.metrics import block_bootstrap_ci, block_summary, depth_metrics, flood_metrics
from floodcam.real.plate import SiteSpec
from floodcam.real.reader import (
    FT_TO_CM,
    DryTemplate,
    PlateReader,
    decode_boundary,
    row_features,
    strip_stats,
)

ROOT = Path(__file__).resolve().parents[3]
TAUS = [4, 6, 8, 10, 13, 16, 20, 25, 30, 40]


@dataclass
class Dataset:
    site: str
    factor: int
    strips: np.ndarray
    reg_ok: np.ndarray
    files: np.ndarray
    gage_ft: np.ndarray
    gage_cm: np.ndarray
    is_ir: np.ndarray
    blocks: np.ndarray
    times: np.ndarray
    stats: dict
    split: dict

    @property
    def R(self) -> int:
        return self.strips.shape[1]


def load_dataset(site: str, factor: int = 1) -> Dataset:
    man = [json.loads(line) for line in (ROOT / "data/usgs/manifest.jsonl").read_text().splitlines()]
    man = {r["file"]: r for r in man if r["site"] == site}
    z = np.load(ROOT / "data/usgs/strips" / f"{site}_f{factor}.npz")
    files = z["files"]
    rows = [man[f] for f in files]
    split = json.loads((ROOT / "data/splits" / f"usgs_{site}.json").read_text())
    gage = np.array([r["gage_ft"] for r in rows], np.float32)
    return Dataset(
        site=site, factor=factor, strips=z["strips"], reg_ok=z["reg_ok"], files=files,
        gage_ft=gage, gage_cm=gage * FT_TO_CM, is_ir=np.array([r["is_ir"] for r in rows]),
        blocks=np.array([r["time"][:7] for r in rows]), times=np.array([r["time"] for r in rows]),
        stats=strip_stats(z["strips"]), split=split,
    )


def _sub(stats: dict, idx: np.ndarray) -> dict:
    return {k: v[idx] for k, v in stats.items()}


def fit_affine(level: np.ndarray, gage: np.ndarray, plate_top: float):
    use = (gage > 0.25) & (gage < plate_top - 0.1)
    if use.sum() < 10:
        return 1.0, 0.0
    A = np.stack([level[use], np.ones(use.sum())], axis=1)
    a, b = np.linalg.lstsq(A, gage[use], rcond=None)[0]
    return float(a), float(b)


def fit_models(ds: Dataset, spec: SiteSpec, train: np.ndarray, seed: int = 0):
    """Fit template, learned reader and threshold baseline on `train` frames only."""
    dry = train & (ds.gage_ft <= 0.02) & ~ds.is_ir & ds.reg_ok
    if dry.sum() < 5:
        dry = train & (ds.gage_ft <= 0.02)
    tpl = DryTemplate.fit(ds.stats, dry)
    feats = row_features(ds.stats, tpl, ds.is_ir, ds.reg_ok)
    reader = PlateReader(spec.top_ft, spec.bottom_ft, template=tpl)
    reader.fit(feats[train], ds.gage_ft[train], seed=seed)
    # threshold baseline: tune tau on training frames, calibrate its own affine map
    best = None
    for tau in TAUS:
        p = np.where(feats[train][..., 6] > tau, 0.9, 0.1)
        b, _ = decode_boundary(p)
        lvl = reader.boundary_ft(b, ds.R)
        a, c = fit_affine(lvl, ds.gage_ft[train], spec.top_ft)
        pred = np.where(lvl <= spec.bottom_ft + 1e-6, 0.0, np.clip(a * lvl + c, 0, None)) * FT_TO_CM
        mae = np.abs(pred - ds.gage_cm[train]).mean()
        if best is None or mae < best[0]:
            best = (mae, tau, a, c)
    return tpl, feats, reader, {"tau": best[1], "a": best[2], "b": best[3], "train_mae": best[0]}


def predict_baseline(feats: np.ndarray, reader: PlateReader, base: dict) -> np.ndarray:
    p = np.where(feats[..., 6] > base["tau"], 0.9, 0.1)
    b, _ = decode_boundary(p)
    lvl = reader.boundary_ft(b, feats.shape[1])
    ft = np.clip(base["a"] * lvl + base["b"], 0, None)
    return np.where(lvl <= reader.bottom_ft + 1e-6, 0.0, ft) * FT_TO_CM


def run_cv(ds: Dataset, spec: SiteSpec, hold_blocks: list[str], train_blocks: list[str],
           seed: int = 0, verbose: bool = False) -> dict:
    """Leave-one-block-out over `hold_blocks`; training uses `train_blocks` minus the held-out one."""
    preds = np.full(len(ds.files), np.nan, np.float32)
    margin = np.full(len(ds.files), np.nan, np.float32)
    base_pred = np.full(len(ds.files), np.nan, np.float32)
    taus = {}
    for hb in hold_blocks:
        train = np.isin(ds.blocks, [b for b in train_blocks if b != hb])
        test = ds.blocks == hb
        _, feats, reader, base = fit_models(ds, spec, train, seed)
        cm, mg = reader.predict_cm(feats[test])
        preds[test], margin[test] = cm, mg
        base_pred[test] = predict_baseline(feats[test], reader, base)
        taus[hb] = base["tau"]
        if verbose:
            print(f"  fold {hb}: n={test.sum()} mae={np.abs(cm - ds.gage_cm[test]).mean():.2f} cm")
    sel = np.isin(ds.blocks, hold_blocks)
    return {"preds": preds, "margin": margin, "baseline": base_pred, "sel": sel, "taus": taus}


def summarize(ds: Dataset, res: dict) -> dict:
    sel = res["sel"]
    gt = ds.gage_cm[sel]
    blocks = ds.blocks[sel]
    out = {}
    trivial = np.zeros_like(gt)
    for name, pred in (("learned", res["preds"][sel]), ("threshold_baseline", res["baseline"][sel]),
                       ("always_no_flood", trivial)):
        out[name] = {
            "depth": depth_metrics(gt, pred),
            "flood_3cm": flood_metrics(gt, pred),
            "by_block": block_summary(blocks, gt, pred),
            "mae_ci95_block_bootstrap": block_bootstrap_ci(blocks, gt, pred),
        }
    return out


def write_run(run_id: str, ds: Dataset, res: dict, summary: dict, extra: dict) -> Path:
    out = ROOT / "results" / "real" / run_id
    out.mkdir(parents=True, exist_ok=True)
    sel = res["sel"]
    import pandas as pd

    pd.DataFrame({
        "file": ds.files[sel], "time": ds.times[sel], "block": ds.blocks[sel],
        "gauge_cm": ds.gage_cm[sel], "learned_cm": res["preds"][sel],
        "baseline_cm": res["baseline"][sel], "margin": res["margin"][sel],
        "is_ir": ds.is_ir[sel], "reg_ok": ds.reg_ok[sel],
    }).to_csv(out / "predictions.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(summary, indent=1))
    try:
        git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             cwd=ROOT).stdout.strip()
    except Exception:
        git = "unknown"
    man_hash = hashlib.sha256((ROOT / "data/usgs/manifest.jsonl").read_bytes()).hexdigest()[:16]
    split_hash = hashlib.sha256((ROOT / "data/splits" / f"usgs_{ds.site}.json").read_bytes()
                                ).hexdigest()[:16]
    meta = {"run_id": run_id, "site": ds.site, "factor": ds.factor, "git_commit": git,
            "manifest_sha256_16": man_hash, "split_sha256_16": split_hash, "synthetic": False,
            "baseline_taus": res["taus"], **extra}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return out
