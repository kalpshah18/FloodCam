"""FloodCam runner: per-frame pipeline writing events to SQLite.

Usage:
    python -m floodcam.run --source <timelapse_dir> --camera <id>
        --config <yaml> --db <path> [--sample-fps 1] [--run-id ID]
        [--synthetic] [--snapshot-every-s 30] [--resume]

Source must be a timelapse directory (timestamps.json + frames/*.jpg).
Video/RTSP support is deferred to a later milestone.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from floodcam.config import CameraConfig, ExperimentConfig
from floodcam.depth.estimate import DepthEstimate, estimate
from floodcam.health import HealthChecker, HealthResult
from floodcam.store.db import EventStore
from floodcam.temporal.filter import FilterState, TemporalFilter
from floodcam.temporal.state_machine import MachineState, StateMachine

if TYPE_CHECKING:
    from floodcam.seg.backend import SegmentationBackend


def _git_commit() -> str:
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _source_hash(source_dir: Path) -> str:
    ts_path = source_dir / "timestamps.json"
    if ts_path.exists():
        data = ts_path.read_bytes()
        return hashlib.sha256(data).hexdigest()[:16]
    return "unknown"


def _draw_annotation(
    frame_rgb: np.ndarray,
    mask: Optional[np.ndarray],
    waterline_row: Optional[float],
    depth_cm: float,
    state: str,
) -> np.ndarray:
    import cv2

    img = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    H, W = img.shape[:2]

    # Mask overlay (blue tint)
    if mask is not None and mask.any():
        overlay = img.copy()
        overlay[mask > 0] = (overlay[mask > 0].astype(np.float32) * 0.5 +
                             np.array([200, 100, 0], dtype=np.float32) * 0.5).astype(np.uint8)
        img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

    # Waterline marker
    if waterline_row is not None:
        r = int(round(waterline_row))
        if 0 <= r < H:
            cv2.line(img, (0, r), (W, r), (0, 255, 255), 2)

    # Text: depth and state
    label = f"depth={depth_cm:.1f}cm  state={state}"
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return img


def process_timelapse(
    source_dir: Path,
    camera_cfg: CameraConfig,
    exp_cfg: ExperimentConfig,
    db_path: Path,
    run_id: str,
    backend: "SegmentationBackend",
    synthetic: bool = True,
    snapshot_every_s: float = 30.0,
    resume: bool = False,
) -> dict:
    """Process a timelapse directory through the full pipeline.

    Returns a summary dict: event_count, final_state, alerts, health_fault_count.
    """
    import cv2

    source_dir = Path(source_dir)
    ts_path = source_dir / "timestamps.json"
    manifest = json.loads(ts_path.read_text())

    store = EventStore(db_path)
    cfg_hash = camera_cfg.config_hash()
    git = _git_commit()
    src_hash = _source_hash(source_dir)

    store.upsert_run(
        run_id=run_id,
        source=str(source_dir),
        camera_id=camera_cfg.camera_id,
        config_hash=cfg_hash,
        git_commit=git,
        started_at=time.time(),
        synthetic=synthetic,
    )

    # Set up pipeline
    temporal_cfg = camera_cfg.temporal
    filt = TemporalFilter(temporal_cfg)
    sm = StateMachine(temporal_cfg)

    if resume:
        saved = store.load_machine_state(camera_cfg.camera_id)
        if saved is not None:
            ms = MachineState(**saved.get("machine", {}))
            fs_data = saved.get("filter", {})
            fs = FilterState(
                accepted=fs_data.get("accepted", []),
                excluded_count=fs_data.get("excluded_count", 0),
                stale=fs_data.get("stale", False),
                last_accepted_timestamp=fs_data.get("last_accepted_timestamp"),
            )
            filt = TemporalFilter.from_state(temporal_cfg, fs)
            sm = StateMachine.from_state(temporal_cfg, ms)

    health_checker = HealthChecker(camera_cfg.health)

    run_dir = Path("results") / "runs" / run_id
    snap_dir = run_dir / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    outbox_path = run_dir / "webhook_outbox.jsonl"

    last_snap_ts: float = -999.0
    last_transition: Optional[str] = None
    event_count = 0
    health_fault_count = 0

    for entry in manifest:
        ts = float(entry["timestamp_s"])
        frame_path = source_dir / "frames" / f"frame_{entry['frame_index']:05d}.jpg"

        img_bgr = cv2.imread(str(frame_path))
        if img_bgr is None:
            continue
        frame_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Health check
        health: HealthResult = health_checker.check(frame_rgb)
        store.insert_health(
            run_id=run_id,
            camera_id=camera_cfg.camera_id,
            timestamp=ts,
            frozen=health.frozen,
            black=health.black,
            obstructed=health.obstructed,
            offline=False,
            detail=health.detail or None,
        )
        if not health.healthy:
            health_fault_count += 1

        # Depth estimation (skip if unhealthy)
        if health.healthy:
            mask = backend.predict_mask(frame_rgb)
            depth_est = estimate(frame_rgb, camera_cfg, exp_cfg)
        else:
            mask = None
            depth_est = DepthEstimate(
                depth_cm=0.0,
                waterline_row=0.0,
                confidence=0.0,
                valid=False,
                calibration_mode=camera_cfg.calibration.mode,
                cm_per_pixel_at_waterline=0.0,
                uncertainty_cm=0.0,
            )

        # Temporal filter + state machine
        fo = filt.update(ts, depth_est)
        so = sm.update(ts, fo)

        # Snapshot
        snap_path: Optional[str] = None
        should_snap = (
            (ts - last_snap_ts) >= snapshot_every_s
            or so.transition is not None
        )
        if should_snap:
            wl = depth_est.waterline_row if depth_est.valid else None
            ann = _draw_annotation(
                frame_rgb, mask, wl,
                float(so.filtered_depth_cm or 0.0),
                so.state.value,
            )
            snap_fname = snap_dir / f"snap_{ts:.1f}.jpg"
            cv2.imwrite(str(snap_fname), ann)
            snap_path = str(snap_fname)
            last_snap_ts = ts

        # Write event
        store.insert_event(
            run_id=run_id,
            camera_id=camera_cfg.camera_id,
            timestamp=ts,
            state=so.state.value,
            depth_cm=float(so.filtered_depth_cm or 0.0),
            severity_class=str(so.severity_class.name),
            confidence=float(depth_est.confidence),
            rate_cm_per_10min=float(so.rate_cm_per_10min) if so.rate_cm_per_10min is not None else None,
            snapshot_path=snap_path,
            model_version=backend.model_version,
            config_hash=cfg_hash,
            synthetic=int(synthetic),
            valid=int(depth_est.valid),
            stale=int(so.stale),
        )
        event_count += 1

        # Alert on transition (deduplicated by transition string)
        if so.transition is not None and so.transition != last_transition:
            store.insert_alert(
                run_id=run_id,
                camera_id=camera_cfg.camera_id,
                timestamp=ts,
                transition=so.transition,
                severity_class=str(so.severity_class.name),
                depth_cm=float(so.filtered_depth_cm or 0.0),
            )
            # Webhook outbox (no network call)
            payload = {
                "run_id": run_id,
                "camera_id": camera_cfg.camera_id,
                "timestamp": ts,
                "transition": so.transition,
                "severity_class": str(so.severity_class.name),
                "depth_cm": float(so.filtered_depth_cm or 0.0),
                "synthetic": synthetic,
            }
            with outbox_path.open("a") as f:
                f.write(json.dumps(payload) + "\n")
            last_transition = so.transition

    # Persist machine state for resume
    store.save_machine_state(
        camera_id=camera_cfg.camera_id,
        state_dict={
            "machine": dataclasses.asdict(sm.state),
            "filter": dataclasses.asdict(filt.state),
        },
        updated_at=time.time(),
    )
    store.finalize_run(run_id, time.time())

    alerts = store.alerts_for_run(run_id)

    # Write run meta.json
    meta = {
        "run_id": run_id,
        "source": str(source_dir),
        "camera_id": camera_cfg.camera_id,
        "config_hash": cfg_hash,
        "git_commit": git,
        "source_hash": src_hash,
        "synthetic": synthetic,
        "event_count": event_count,
        "final_state": sm.state.flood_state,
        "alert_count": len(alerts),
        "note": "Synthetic run — pipeline validation only. Do not cite in a paper.",
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    store.close()
    return {
        "event_count": event_count,
        "final_state": sm.state.flood_state,
        "alerts": len(alerts),
        "health_fault_count": health_fault_count,
        "run_id": run_id,
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="FloodCam runner: timelapse → SQLite events"
    )
    parser.add_argument("--source", required=True, help="Timelapse directory")
    parser.add_argument("--camera", required=True, help="Camera ID")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--db", required=True, help="SQLite DB path")
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--snapshot-every-s", type=float, default=30.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    run_id = args.run_id or f"run_{uuid.uuid4().hex[:8]}"

    camera_cfg = CameraConfig.load(args.config)
    camera_cfg.camera_id = args.camera
    exp_cfg = ExperimentConfig()

    from floodcam.seg.backend import HsvBackend

    backend = HsvBackend(exp_cfg, camera_cfg)

    summary = process_timelapse(
        source_dir=Path(args.source),
        camera_cfg=camera_cfg,
        exp_cfg=exp_cfg,
        db_path=Path(args.db),
        run_id=run_id,
        backend=backend,
        synthetic=args.synthetic,
        snapshot_every_s=args.snapshot_every_s,
        resume=args.resume,
    )
    print(f"Done: {summary}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
