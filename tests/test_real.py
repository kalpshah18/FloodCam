"""Unit tests for the real-footage pipeline (tiny synthetic arrays; never reported as results)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from floodcam.real.events import causal_median, episodes, match_episodes
from floodcam.real.metrics import block_bootstrap_ci, depth_metrics, flood_metrics
from floodcam.real.reader import DryTemplate, PlateReader, decode_boundary, row_features, strip_stats

ROOT = Path(__file__).parent.parent


def test_decode_boundary_recovers_split():
    R = 100
    p = np.full((3, R), 0.05)
    p[0, 60:] = 0.95   # rows >= 60 submerged
    p[1, :] = 0.05     # dry
    p[2, 10:] = 0.95
    b, margin = decode_boundary(p)
    assert b.tolist() == [60, R, 10]
    assert (margin > 1).all()


def test_decode_tolerates_noisy_rows():
    rng = np.random.default_rng(0)
    p = np.where(np.arange(120) >= 40, 0.9, 0.1)[None, :].astype(float)
    flip = rng.random(120) < 0.08
    p[0, flip] = 1 - p[0, flip]
    b, _ = decode_boundary(p)
    assert abs(b[0] - 40) <= 3


def _fake_strips(levels, R=60, C=8):
    """Plate strips: dry = bright with dark tick rows, submerged part = brown."""
    out = np.zeros((len(levels), R, C, 3), np.uint8)
    for i, lvl in enumerate(levels):
        out[i, :] = 220
        out[i, ::6] = 40
        b = int(R * (1 - lvl))  # rows >= b submerged
        out[i, b:] = (60, 90, 130)
    return out


def test_reader_fits_and_orders_levels():
    rng = np.random.default_rng(1)
    lv = rng.uniform(0.05, 0.9, 120)
    strips = _fake_strips(lv)
    gage = (lv * 1.9).astype(np.float32)            # plate 0..1.9 ft
    stats = strip_stats(strips)
    tpl = DryTemplate(lab=np.median(strip_stats(_fake_strips([0.0] * 4))["lab"], axis=0),
                      std_l=np.zeros(60), edge_l=np.zeros(60))
    feats = row_features(stats, tpl, np.zeros(len(lv), bool), np.ones(len(lv), bool))
    reader = PlateReader(top_ft=1.9, bottom_ft=0.0).fit(feats[:90], gage[:90])
    pred_cm, _ = reader.predict_cm(feats[90:])
    err = np.abs(pred_cm - gage[90:] * 30.48)
    assert np.median(err) < 6.0
    assert np.corrcoef(pred_cm, gage[90:])[0, 1] > 0.95


def test_event_episodes_and_matching():
    t = np.arange(0, 30, 1.0)
    d = np.zeros(30)
    d[5:12] = 10.0
    d[20] = 4.0          # single isolated spike
    filt = causal_median(d, np.ones(30, bool), window=5)
    ep_f = episodes(filt, t)
    ep_g = episodes(d, t)
    assert ep_g[0][0] == 5.0
    assert len(ep_f) == 1  # the median removes the isolated spike
    m = match_episodes([(5.0, 12.0)], ep_f)
    assert m["detected"] == 1 and m["false_episodes"] == 0


def test_causal_median_skips_invalid():
    out = causal_median(np.array([1.0, 99.0, 3.0]), np.array([True, False, True]), window=5)
    assert out.tolist() == [1.0, 1.0, 2.0]


def test_metrics_basic():
    gt = np.array([0, 0, 5, 20.0])
    pr = np.array([0, 4, 5, 18.0])
    m = depth_metrics(gt, pr)
    assert m["mae_cm"] == pytest.approx(1.5)
    f = flood_metrics(gt, pr)
    assert (f["tp"], f["fp"], f["fn"], f["tn"]) == (2, 1, 0, 1)


def test_block_bootstrap_deterministic():
    blocks = np.array(list("aabbcc"))
    gt, pr = np.zeros(6), np.array([1, 1, 2, 2, 3, 3.0])
    assert block_bootstrap_ci(blocks, gt, pr) == block_bootstrap_ci(blocks, gt, pr)


def test_usgs_split_has_no_leakage():
    p = ROOT / "data/splits/usgs_in_overland.json"
    s = json.loads(p.read_text())
    locked, dev = set(s["locked_test_blocks"]), set(s["dev_blocks"])
    assert locked and dev and not locked & dev
    files = {b: set(v) for b, v in s["blocks"].items()}
    seen: set[str] = set()
    for b, f in files.items():
        assert not seen & f, "a frame appears in two blocks"
        seen |= f
    assert set(files) == locked | dev
    assert s["synthetic"] is False


def test_cv_script_never_trains_on_locked_blocks():
    src = (ROOT / "scripts/real_cv.py").read_text()
    assert "locked_test_blocks" in src and "dev_blocks" in src
