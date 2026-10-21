"""Render a depth-vs-time profile into an image sequence for runner testing.

Deterministic. All outputs tagged synthetic=true.
Uses synth_scene_02 by default. NEVER uses synth_scene_01 (locked test scene).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from floodcam.synth.demo import _NORMAL_SCENE_PARAMS
from floodcam.synth.renderer import SceneConfig, SynthRenderer
from floodcam.synth.timeseries import SynthSample

DEMO_SCENE_ID = "synth_scene_02"
LOCKED_SCENE = "synth_scene_01"


def _scene_for_id(scene_id: str, seed: int = 42) -> tuple[SynthRenderer, SceneConfig]:
    assert scene_id != LOCKED_SCENE, f"Attempted to use locked scene: {scene_id}"
    idx = int(scene_id.split("_")[-1])
    h, tilt, f, dist, _ = _NORMAL_SCENE_PARAMS[idx]
    scene_cfg = SceneConfig(
        camera_height_cm=h,
        camera_tilt_deg=tilt,
        focal_length_px=f,
        pole_distance_cm=dist,
    )
    return SynthRenderer(scene_cfg, seed=seed * 1000 + idx), scene_cfg


def render_timelapse(
    samples: list[SynthSample],
    out_dir: Path | str,
    scene_id: str = DEMO_SCENE_ID,
    seed: int = 42,
) -> Path:
    """Render each sample's depth through the renderer, saving frames + timestamps.json.

    Writes:
      out_dir/frames/frame_{i:05d}.jpg
      out_dir/timestamps.json  (list of {frame_index, timestamp_s, depth_cm, synthetic})
    """
    import cv2

    out_dir = Path(out_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    renderer, _ = _scene_for_id(scene_id, seed)

    manifest = []
    for i, s in enumerate(samples):
        synth_frame = renderer.render_frame(depth_cm=s.depth_cm, frame_idx=i, variant="normal")
        img_bgr = cv2.cvtColor(synth_frame.image, cv2.COLOR_RGB2BGR)
        path = frames_dir / f"frame_{i:05d}.jpg"
        cv2.imwrite(str(path), img_bgr)
        manifest.append({
            "frame_index": i,
            "timestamp_s": s.timestamp_s,
            "depth_cm": s.depth_cm,
            "valid": s.valid,
            "synthetic": True,
        })

    ts_path = out_dir / "timestamps.json"
    ts_path.write_text(json.dumps(manifest, indent=2))
    return out_dir


def make_health_fault_timelapse(
    out_dir: Path | str,
    seed: int = 42,
    scene_id: str = DEMO_SCENE_ID,
) -> Path:
    """Render a timelapse with injected health faults for testing.

    Segments:
      t=0–60s   : normal (depth=5.0 cm)
      t=61–70s  : black frames (all zeros)
      t=71–80s  : frozen frames (exact copy of last normal frame)
      t=81–90s  : blurred/obstructed frames (heavy Gaussian blur)
      t=91–180s : normal again (depth=5.0 cm)

    All entries in timestamps.json are marked synthetic=true.
    The 'fault' key in each entry names the injected fault type.
    """
    import cv2

    out_dir = Path(out_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    renderer, _ = _scene_for_id(scene_id, seed)

    manifest = []
    frame_idx = 0
    last_normal_rgb: np.ndarray | None = None

    def _write(img_rgb: np.ndarray, ts: float, depth: float, fault: str = "normal") -> None:
        nonlocal frame_idx, last_normal_rgb
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        path = frames_dir / f"frame_{frame_idx:05d}.jpg"
        cv2.imwrite(str(path), img_bgr)
        manifest.append({
            "frame_index": frame_idx,
            "timestamp_s": ts,
            "depth_cm": depth,
            "valid": True,
            "synthetic": True,
            "fault": fault,
        })
        frame_idx += 1

    # Normal phase (t=0..60)
    for t in range(61):
        sf = renderer.render_frame(depth_cm=5.0, frame_idx=t, variant="normal")
        _write(sf.image, float(t), 5.0, "normal")
        last_normal_rgb = sf.image.copy()

    assert last_normal_rgb is not None

    # Black frames (t=61..70)
    black = np.zeros_like(last_normal_rgb)
    for t in range(61, 71):
        _write(black, float(t), 5.0, "black")

    # Frozen frames (t=71..80) — exact copy of last normal
    for t in range(71, 81):
        _write(last_normal_rgb, float(t), 5.0, "frozen")

    # Obstructed frames (t=81..90) — heavily blurred
    blurred = cv2.GaussianBlur(last_normal_rgb, (51, 51), 0)
    for t in range(81, 91):
        _write(blurred, float(t), 5.0, "obstructed")

    # Normal again (t=91..180)
    for t in range(91, 181):
        sf = renderer.render_frame(depth_cm=5.0, frame_idx=t, variant="normal")
        _write(sf.image, float(t), 5.0, "normal")

    (out_dir / "timestamps.json").write_text(json.dumps(manifest, indent=2))
    return out_dir
