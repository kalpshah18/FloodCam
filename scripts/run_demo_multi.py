#!/usr/bin/env python
"""Multi-camera demo: three profiles on three different camera IDs into one DB.

Usage:
    python scripts/run_demo_multi.py [--run-id demo-multi-001]

Writes to results/demo.db (shared with single-camera demo).
All results are synthetic pipeline validation — do not cite.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from floodcam.config import ExperimentConfig  # noqa: E402
from floodcam.depth.estimate import make_camera_config_for_renderer  # noqa: E402
from floodcam.run import process_timelapse  # noqa: E402
from floodcam.seg.backend import HsvBackend  # noqa: E402
from floodcam.synth.renderer import SceneConfig, SynthRenderer  # noqa: E402
from floodcam.synth.timelapse import (  # noqa: E402
    DEMO_SCENE_ID,
    make_health_fault_timelapse,
    render_timelapse,
)
from floodcam.synth.timeseries import short_puddle, steady_rise  # noqa: E402

LOCKED_SCENE = "synth_scene_01"
assert DEMO_SCENE_ID != LOCKED_SCENE, "Must never use the locked test scene"

_SCENE02_PARAMS = (110.0, 22.0, 620.0, 210.0)  # synth_scene_02


def _make_renderer() -> SynthRenderer:
    h, tilt, f, dist = _SCENE02_PARAMS
    cfg = SceneConfig(
        camera_height_cm=h, camera_tilt_deg=tilt,
        focal_length_px=f, pole_distance_cm=dist,
    )
    return SynthRenderer(cfg, seed=42 * 1000 + 2)


def run_multi_demo(run_id: str = "demo-multi-001") -> None:
    renderer = _make_renderer()
    exp_cfg = ExperimentConfig(
        hsv_lower=[90, 30, 30],
        hsv_upper=[120, 255, 255],
    )

    db_path = ROOT / "results" / "demo.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    profiles = [
        ("cam_rise",  "steady_rise",   steady_rise(0.0, 20.0, 900.0, 1.0, seed=1)),
        ("cam_puddle","short_puddle",  short_puddle(depth_cm=1.5, duration_s=300.0, sample_interval_s=1.0, seed=6)),
    ]

    print(f"\nMulti-camera demo: run_id={run_id}, db={db_path}")

    results = []
    for cam_id, profile_name, samples in profiles:
        cfg = make_camera_config_for_renderer(renderer)
        cfg.camera_id = cam_id
        backend = HsvBackend(exp_cfg, cfg)

        sub_run = f"{run_id}_{cam_id}"
        tl_dir = ROOT / "results" / "runs" / sub_run / "timelapse"
        print(f"  Rendering {profile_name} ({len(samples)} frames) for {cam_id}…")
        render_timelapse(samples, tl_dir, scene_id=DEMO_SCENE_ID)

        print(f"  Running pipeline for {cam_id}…")
        summary = process_timelapse(
            source_dir=tl_dir,
            camera_cfg=cfg,
            exp_cfg=exp_cfg,
            db_path=db_path,
            run_id=sub_run,
            backend=backend,
            synthetic=True,
            snapshot_every_s=60.0,
        )
        results.append((cam_id, profile_name, summary["final_state"],
                        summary["alerts"], summary["health_fault_count"]))

    # Health fault camera
    cfg_hf = make_camera_config_for_renderer(renderer)
    cfg_hf.camera_id = "cam_health"
    backend_hf = HsvBackend(exp_cfg, cfg_hf)
    sub_run_hf = f"{run_id}_cam_health"
    hf_dir = ROOT / "results" / "runs" / sub_run_hf / "timelapse"
    print("  Rendering health_fault for cam_health…")
    make_health_fault_timelapse(hf_dir, seed=42, scene_id=DEMO_SCENE_ID)
    print("  Running pipeline for cam_health…")
    summary_hf = process_timelapse(
        source_dir=hf_dir,
        camera_cfg=cfg_hf,
        exp_cfg=exp_cfg,
        db_path=db_path,
        run_id=sub_run_hf,
        backend=backend_hf,
        synthetic=True,
        snapshot_every_s=30.0,
    )
    results.append(("cam_health", "health_fault", summary_hf["final_state"],
                    summary_hf["alerts"], summary_hf["health_fault_count"]))

    print("\n" + "=" * 72)
    print(f"{'Camera':<14} {'Profile':<18} {'Final State':<14} {'Alerts':<8} {'HealthFaults'}")
    print("-" * 72)
    for row in results:
        print(f"{row[0]:<14} {row[1]:<18} {row[2]:<14} {str(row[3]):<8} {row[4]}")
    print("=" * 72)
    print(f"NOTE: All results are synthetic pipeline validation only.\nDB: {db_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="demo-multi-001")
    args = parser.parse_args()
    run_multi_demo(args.run_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
