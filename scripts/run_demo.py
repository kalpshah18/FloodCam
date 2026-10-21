#!/usr/bin/env python
"""Demo: render 4 synthetic time-lapse profiles and run the full pipeline on each.

Writes to results/runs/<run_id>_<profile>/ and results/demo.db.
All results are synthetic pipeline validation — do not cite in a paper.

Usage:
    python scripts/run_demo.py [--run-id run-001]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from floodcam.config import ExperimentConfig  # noqa: E402
from floodcam.depth.estimate import make_camera_config_for_renderer  # noqa: E402
from floodcam.run import process_timelapse  # noqa: E402
from floodcam.seg.backend import HsvBackend  # noqa: E402
from floodcam.store.db import EventStore  # noqa: E402
from floodcam.synth.renderer import SceneConfig, SynthRenderer  # noqa: E402
from floodcam.synth.timelapse import (  # noqa: E402
    DEMO_SCENE_ID,
    make_health_fault_timelapse,
    render_timelapse,
)
from floodcam.synth.timeseries import rise_plateau_fall, short_puddle, steady_rise  # noqa: E402

LOCKED_SCENE = "synth_scene_01"
assert DEMO_SCENE_ID != LOCKED_SCENE

_DEMO_SCENE_PARAMS = (110.0, 22.0, 620.0, 210.0)  # synth_scene_02


def _make_renderer() -> SynthRenderer:
    h, tilt, f, dist = _DEMO_SCENE_PARAMS
    cfg = SceneConfig(camera_height_cm=h, camera_tilt_deg=tilt,
                      focal_length_px=f, pole_distance_cm=dist)
    return SynthRenderer(cfg, seed=42 * 1000 + 2)


def _flood_entry_ts(run_id: str, db: Path) -> str:
    store = EventStore(db)
    events = store.events_for_run(run_id)
    store.close()
    for e in events:
        if e["state"] == "Flood":
            return f"{e['timestamp']:.0f}"
    return "—"


def run_demo(run_id: str = "run-001") -> None:
    renderer = _make_renderer()
    camera_cfg = make_camera_config_for_renderer(renderer)
    camera_cfg.camera_id = "demo_cam"
    exp_cfg = ExperimentConfig(
        hsv_lower=[90, 30, 30],
        hsv_upper=[120, 255, 255],
    )
    backend = HsvBackend(exp_cfg, camera_cfg)

    data_dir = ROOT / "results" / "runs" / run_id
    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = ROOT / "results" / "demo.db"

    profiles = [
        ("steady_rise",
         steady_rise(0.0, 20.0, 900.0, 1.0, seed=1),
         f"{run_id}_steady_rise"),
        ("rise_plateau_fall",
         rise_plateau_fall(rise_to_cm=15.0, plateau_s=600.0, total_s=1500.0,
                           sample_interval_s=1.0, seed=2),
         f"{run_id}_plateau"),
        ("short_puddle",
         short_puddle(depth_cm=1.5, duration_s=300.0, sample_interval_s=1.0, seed=6),
         f"{run_id}_puddle"),
    ]

    results = []
    for profile_name, samples, sub_run_id in profiles:
        tl_dir = ROOT / "results" / "runs" / sub_run_id / "timelapse"
        print(f"Rendering {profile_name} ({len(samples)} frames)…")
        render_timelapse(samples, tl_dir, scene_id=DEMO_SCENE_ID)

        print(f"Running pipeline for {profile_name}…")
        summary = process_timelapse(
            source_dir=tl_dir,
            camera_cfg=camera_cfg,
            exp_cfg=exp_cfg,
            db_path=db_path,
            run_id=sub_run_id,
            backend=backend,
            synthetic=True,
            snapshot_every_s=60.0,
        )
        flood_ts = _flood_entry_ts(sub_run_id, db_path)
        results.append((profile_name, summary["final_state"], flood_ts,
                        summary["alerts"], summary["health_fault_count"]))

    # Health fault profile
    sub_run_id = f"{run_id}_health_fault"
    hf_dir = ROOT / "results" / "runs" / sub_run_id / "timelapse"
    print("Rendering health_fault timelapse…")
    make_health_fault_timelapse(hf_dir, seed=42, scene_id=DEMO_SCENE_ID)
    print("Running pipeline for health_fault…")
    summary = process_timelapse(
        source_dir=hf_dir,
        camera_cfg=camera_cfg,
        exp_cfg=exp_cfg,
        db_path=db_path,
        run_id=sub_run_id,
        backend=backend,
        synthetic=True,
        snapshot_every_s=30.0,
    )
    flood_ts = _flood_entry_ts(sub_run_id, db_path)
    results.append(("health_fault", summary["final_state"], flood_ts,
                    summary["alerts"], summary["health_fault_count"]))

    # Print summary table
    print("\n" + "=" * 78)
    print(f"{'Profile':<22} {'Final State':<14} {'Flood@(s)':<12} {'Alerts':<8} {'HealthFaults'}")
    print("-" * 78)
    for row in results:
        print(f"{row[0]:<22} {row[1]:<14} {row[2]:<12} {str(row[3]):<8} {row[4]}")
    print("=" * 78)
    print("NOTE: All results are synthetic pipeline validation only.")

    # Write meta.json
    meta = {
        "run_id": run_id,
        "synthetic": True,
        "note": "Synthetic demo results — pipeline validation only. Must not appear in a paper.",
        "profiles": [
            {"name": r[0], "final_state": r[1], "flood_entry_s": r[2],
             "alerts": r[3], "health_fault_count": r[4]}
            for r in results
        ],
    }
    (data_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nResults in {db_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="run-001")
    args = parser.parse_args()
    run_demo(args.run_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
