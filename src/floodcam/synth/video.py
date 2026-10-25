"""Render a synthetic demo video (rise -> plateau -> fall) for the upload dashboard.

SYNTHETIC: for demos and tests only; never report results from it. The returned
LiveSettings carry the renderer's analytic calibration marks so the demo is
one-click in the dashboard.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from floodcam.live import LiveSettings
from floodcam.synth.demo import _NORMAL_SCENE_PARAMS
from floodcam.synth.renderer import SceneConfig, SynthRenderer

DEMO_SCENE_IDX = 2  # never the locked test scene (index 1)


def demo_depth_profile(duration_s: float, peak_cm: float = 12.0) -> "callable":
    """Piecewise depth(t): dry 10%, rise 35%, plateau 20%, fall 25%, dry 10%."""
    d = duration_s
    knots_t = [0, 0.10 * d, 0.45 * d, 0.65 * d, 0.90 * d, d]
    knots_h = [0.0, 0.0, peak_cm, peak_cm, 0.0, 0.0]
    return lambda t: float(np.interp(t, knots_t, knots_h))


def render_demo_video(
    out_path: str | Path,
    duration_s: float = 60.0,
    fps: float = 5.0,
    peak_cm: float = 12.0,
    seed: int = 42,
) -> tuple[Path, LiveSettings, "callable"]:
    """Write an mp4 and return (path, matching settings, true-depth function)."""
    import cv2

    h, tilt, f, dist, _ = _NORMAL_SCENE_PARAMS[DEMO_SCENE_IDX]
    scene = SceneConfig(camera_height_cm=h, camera_tilt_deg=tilt,
                        focal_length_px=f, pole_distance_cm=dist)
    r = SynthRenderer(scene, seed=seed * 1000 + DEMO_SCENE_IDX)
    depth_fn = demo_depth_profile(duration_s, peak_cm)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
        (scene.img_width, scene.img_height),
    )
    if not writer.isOpened():
        raise RuntimeError("cv2.VideoWriter could not open the output file")
    n = int(duration_s * fps)
    for i in range(n):
        frame = r.render_frame(depth_cm=depth_fn(i / fps), frame_idx=i, variant="normal")
        writer.write(cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR))
    writer.release()

    v_g, _ = r.project_height(0.0)
    v_top, _ = r.project_height(scene.pole_height_cm)
    settings = LiveSettings(
        pole_x_px=int(scene.img_width * scene.pole_x_frac),
        ground_row=float(v_g),
        mark_row=float(v_top),
        mark_height_cm=scene.pole_height_cm,
        pole_width_px=scene.pole_width_px,
        hsv_lower=[90, 30, 30],
        hsv_upper=[120, 255, 255],
        morph_open=0,
        morph_close=0,
        sample_fps=1.0,
        persistence_s=8.0,
        trend_window_s=16.0,
    )
    return out_path, settings, depth_fn
