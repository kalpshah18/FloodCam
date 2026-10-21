#!/usr/bin/env python
"""HSV baseline: grid-search on LOSO folds, report depth and flood metrics.

Usage:
    python scripts/run_hsv_baseline.py [--run-id hsv-001] [--seed 42]

Writes:
    results/<run_id>/metrics.json
    results/<run_id>/meta.json

NEVER touches synth_scene_01 (locked test scene).
Synthetic results are for pipeline validation only — do not cite in a paper.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from floodcam.config import ExperimentConfig  # noqa: E402
from floodcam.depth.estimate import estimate, make_camera_config_for_renderer  # noqa: E402
from floodcam.synth.renderer import SceneConfig, SynthRenderer  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOCKED_SCENE = "synth_scene_01"
SPLITS_FILE = ROOT / "data" / "splits" / "synthetic_splits.json"
DEPTHS_CM = [0.0, 2.0, 3.0, 5.0, 10.0, 20.0]
FLOOD_TRIGGER_CM = 3.5

_NORMAL_SCENE_PARAMS = [
    # (camera_height_cm, tilt_deg, focal_px, pole_dist_cm, lighting_tag)
    (120.0, 20.0, 600.0, 200.0, "synthetic_indoor_diffuse"),        # synth_scene_00
    (130.0, 18.0, 580.0, 190.0, "synthetic_indoor_directional"),    # synth_scene_01 LOCKED
    (110.0, 22.0, 620.0, 210.0, "synthetic_outdoor_overcast"),      # synth_scene_02
    (125.0, 25.0, 650.0, 180.0, "synthetic_outdoor_bright"),        # synth_scene_03
    (115.0, 15.0, 560.0, 220.0, "synthetic_indoor_low"),            # synth_scene_04
    (135.0, 28.0, 700.0, 175.0, "synthetic_outdoor_harsh"),         # synth_scene_05
    (100.0, 17.0, 540.0, 230.0, "synthetic_indoor_side"),           # synth_scene_06
    (140.0, 23.0, 640.0, 195.0, "synthetic_outdoor_cloudy"),        # synth_scene_07
]

_NEGATIVE_SCENE_PARAMS = [
    (120.0, 20.0, 600.0, 200.0, "reflection"),   # synth_neg_reflection_00
    (120.0, 20.0, 600.0, 200.0, "wet_road"),      # synth_neg_wet_road_01
]


# ---------------------------------------------------------------------------
# Scene factory
# ---------------------------------------------------------------------------


def scene_id_to_renderer(scene_id: str, seed: int = 42) -> tuple:
    """Return (renderer, scene_cfg) for a non-locked normal scene ID.

    Raises:
        AssertionError: if scene_id == LOCKED_SCENE.
    """
    assert scene_id != LOCKED_SCENE, (
        f"Attempted to access locked test scene: {scene_id}. "
        "Run scripts/final_test.py for test-scene evaluation."
    )
    idx = int(scene_id.split("_")[-1])
    h, tilt, f, dist, _ = _NORMAL_SCENE_PARAMS[idx]
    scene_cfg = SceneConfig(
        camera_height_cm=h,
        camera_tilt_deg=tilt,
        focal_length_px=f,
        pole_distance_cm=dist,
    )
    scene_seed = seed * 1000 + idx
    return SynthRenderer(scene_cfg, seed=scene_seed), scene_cfg


def _neg_renderer(neg_idx: int, seed: int = 42) -> SynthRenderer:
    h, tilt, f, dist, _ = _NEGATIVE_SCENE_PARAMS[neg_idx]
    scene_cfg = SceneConfig(
        camera_height_cm=h,
        camera_tilt_deg=tilt,
        focal_length_px=f,
        pole_distance_cm=dist,
    )
    scene_seed = seed * 1000 + len(_NORMAL_SCENE_PARAMS) + neg_idx
    return SynthRenderer(scene_cfg, seed=scene_seed)


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------


def _eval_combo(
    scene_ids: list[str],
    hsv_lower: list[int],
    hsv_upper: list[int],
    seed: int,
) -> dict:
    """Evaluate one HSV combo on a list of scenes. Returns raw predictions."""
    exp_cfg = ExperimentConfig(
        hsv_lower=hsv_lower,
        hsv_upper=hsv_upper,
        hsv_morph_open=0,
        hsv_morph_close=0,
    )
    preds: list[float] = []
    truths: list[float] = []

    for scene_id in scene_ids:
        assert scene_id != LOCKED_SCENE
        renderer, _ = scene_id_to_renderer(scene_id, seed=seed)
        cam_cfg = make_camera_config_for_renderer(renderer)
        for d in DEPTHS_CM:
            frame = renderer.render_frame(d, frame_idx=0, variant="normal")
            result = estimate(frame.image, cam_cfg, exp_cfg)
            preds.append(result.depth_cm if result.valid else 0.0)
            truths.append(d)

    return {"preds": preds, "truths": truths}


def _compute_metrics(preds: list[float], truths: list[float]) -> dict:

    errors = [abs(p - t) for p, t in zip(preds, truths)]
    mae = sum(errors) / len(errors) if errors else float("nan")

    near_threshold_errors = [
        abs(p - t) for p, t in zip(preds, truths) if t in (2.0, 3.0)
    ]
    mae_near_3cm = (
        sum(near_threshold_errors) / len(near_threshold_errors)
        if near_threshold_errors
        else float("nan")
    )

    per_depth: dict[str, float] = {}
    depth_bins: dict[float, list[float]] = {}
    for p, t in zip(preds, truths):
        depth_bins.setdefault(t, []).append(abs(p - t))
    for d, errs in sorted(depth_bins.items()):
        per_depth[f"{d:.1f}cm"] = sum(errs) / len(errs)

    tp = sum(1 for p, t in zip(preds, truths) if p >= FLOOD_TRIGGER_CM and t >= FLOOD_TRIGGER_CM)
    fp = sum(1 for p, t in zip(preds, truths) if p >= FLOOD_TRIGGER_CM and t < FLOOD_TRIGGER_CM)
    fn = sum(1 for p, t in zip(preds, truths) if p < FLOOD_TRIGGER_CM and t >= FLOOD_TRIGGER_CM)

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

    return {
        "mae_cm": mae,
        "mae_near_3cm": mae_near_3cm,
        "mae_per_depth_bin": per_depth,
        "flood_precision": precision,
        "flood_recall": recall,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="hsv-001")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    run_id = args.run_id
    seed = args.seed

    # Load splits
    splits = json.loads(SPLITS_FILE.read_text())
    locked = splits["locked_test_scene"]
    assert locked == LOCKED_SCENE, f"Unexpected locked scene: {locked}"
    for fold in splits["loso_folds"]:
        assert fold["val_scene"] != LOCKED_SCENE, "Locked scene leaked into LOSO folds."
        assert LOCKED_SCENE not in fold["train_scenes"], "Locked scene leaked into train."

    loso_folds = splits["loso_folds"]

    # Grid definition
    h_lo_values = [90, 100, 110]
    h_hi_values = [120, 130, 140]
    s_lo_values = [30, 50, 70]
    v_lo_values = [30, 50]

    grid = [
        ([h_lo, s_lo, v_lo], [h_hi, 255, 255])
        for h_lo in h_lo_values
        for h_hi in h_hi_values
        for s_lo in s_lo_values
        for v_lo in v_lo_values
    ]

    print(f"Grid search: {len(grid)} combos × {len(loso_folds)} folds")

    best_mae = float("inf")
    best_combo: tuple[list[int], list[int]] | None = None

    for lo, hi in grid:
        fold_maes: list[float] = []
        for fold in loso_folds:
            val_scene = fold["val_scene"]
            result = _eval_combo([val_scene], lo, hi, seed)
            metrics = _compute_metrics(result["preds"], result["truths"])
            fold_maes.append(metrics["mae_cm"])
        mean_mae = sum(fold_maes) / len(fold_maes)
        if mean_mae < best_mae:
            best_mae = mean_mae
            best_combo = (lo, hi)

    assert best_combo is not None
    best_lo, best_hi = best_combo
    print(f"Best HSV: lower={best_lo}, upper={best_hi}, LOSO mean MAE={best_mae:.3f} cm")

    # Evaluate best combo on all non-locked normal scenes
    all_non_locked = [
        f"synth_scene_{i:02d}"
        for i in range(len(_NORMAL_SCENE_PARAMS))
        if f"synth_scene_{i:02d}" != LOCKED_SCENE
    ]
    final_result = _eval_combo(all_non_locked, best_lo, best_hi, seed)
    final_metrics = _compute_metrics(final_result["preds"], final_result["truths"])

    # Trivial baseline: always predict 0 cm
    trivial_mae = sum(abs(t) for t in final_result["truths"]) / len(final_result["truths"])

    # Negative scenes: false flood rate at depth=0
    neg_flood_count = 0
    neg_total = 0
    neg_variants = ["reflection", "wet_road"]
    for ni, variant in enumerate(neg_variants):
        renderer = _neg_renderer(ni, seed=seed)
        cam_cfg = make_camera_config_for_renderer(renderer)
        exp_cfg = ExperimentConfig(
            hsv_lower=best_lo,
            hsv_upper=best_hi,
            hsv_morph_open=0,
            hsv_morph_close=0,
        )
        for fi in range(5):
            frame = renderer.render_frame(0.0, frame_idx=fi, variant=variant)
            result = estimate(frame.image, cam_cfg, exp_cfg)
            neg_total += 1
            if result.valid and result.depth_cm >= FLOOD_TRIGGER_CM:
                neg_flood_count += 1

    false_flood_rate = neg_flood_count / neg_total if neg_total > 0 else 0.0

    metrics_out = {
        "note": (
            "SYNTHETIC RESULTS — pipeline validation only. "
            "Must not appear in a paper or final evaluation."
        ),
        "synthetic": True,
        "run_id": run_id,
        "best_hsv_lower": best_lo,
        "best_hsv_upper": best_hi,
        "loso_mean_mae_cm": round(best_mae, 4),
        "eval_all_non_locked_scenes": {
            k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in final_metrics.items()
            if k != "mae_per_depth_bin"
        },
        "mae_per_depth_bin_cm": {
            k: round(v, 4) for k, v in final_metrics["mae_per_depth_bin"].items()
        },
        "trivial_baseline_mae_cm": round(trivial_mae, 4),
        "false_flood_rate_negatives": round(false_flood_rate, 4),
        "n_negative_frames": neg_total,
    }

    meta_out = {
        "run_id": run_id,
        "date": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "splits_file": str(SPLITS_FILE.relative_to(ROOT)),
        "manifest_path": str(ROOT / "data" / "synthetic" / "manifest.jsonl"),
        "synthetic": True,
        "seed": seed,
        "note": (
            "Synthetic data used for pipeline integration testing only. "
            "Results must not appear in a paper."
        ),
        "locked_test_scene": LOCKED_SCENE,
        "locked_scene_touched": False,
    }

    results_dir = ROOT / "results" / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "metrics.json").write_text(json.dumps(metrics_out, indent=2))
    (results_dir / "meta.json").write_text(json.dumps(meta_out, indent=2))

    print(f"\nResults written to {results_dir}")
    print(f"  MAE (all non-locked): {final_metrics['mae_cm']:.3f} cm")
    print(f"  MAE near 3 cm:        {final_metrics['mae_near_3cm']:.3f} cm")
    print(f"  Flood precision:      {final_metrics['flood_precision']:.3f}")
    print(f"  Flood recall:         {final_metrics['flood_recall']:.3f}")
    print(f"  Trivial baseline MAE: {trivial_mae:.3f} cm")
    print(f"  False flood rate (negatives): {false_flood_rate:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
