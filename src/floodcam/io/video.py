"""Video file I/O: probing, frame sampling, and fixed-width downscaling.

All pipeline coordinates (pole column, ground row, ROI) live in the *processed*
frame space, i.e. after resize_to_max_width(). The dashboard preview and the
analysis loop both call it with the same value, so the two always agree.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

DEFAULT_MAX_WIDTH = 960


class VideoError(RuntimeError):
    """The file cannot be opened or contains no decodable frames."""


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    n_frames: int  # container estimate; may be 0 for some streams
    duration_s: float


def resize_to_max_width(frame_rgb: np.ndarray, max_width: int = DEFAULT_MAX_WIDTH) -> np.ndarray:
    import cv2

    h, w = frame_rgb.shape[:2]
    if max_width <= 0 or w <= max_width:
        return frame_rgb
    scale = max_width / w
    return cv2.resize(frame_rgb, (max_width, round(h * scale)), interpolation=cv2.INTER_AREA)


def _open(path: str | Path):
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise VideoError(f"Cannot open video: {path}")
    return cap


def probe_video(path: str | Path) -> VideoInfo:
    import cv2

    cap = _open(path)
    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if w <= 0 or h <= 0:
        raise VideoError(f"Video reports invalid size {w}x{h}: {path}")
    if not (fps > 0 and np.isfinite(fps)):
        fps = 30.0
    n = max(n, 0)
    return VideoInfo(w, h, fps, n, n / fps if n else 0.0)


def read_frame(
    path: str | Path, t_s: float = 0.0, max_width: int = DEFAULT_MAX_WIDTH
) -> np.ndarray:
    """Decode the frame at (approximately) time t_s; returns RGB, resized."""
    import cv2

    cap = _open(path)
    try:
        if t_s > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, t_s * 1000.0)
        ok, bgr = cap.read()
        if not ok and t_s > 0:  # seek past the end: fall back to the first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, bgr = cap.read()
    finally:
        cap.release()
    if not ok:
        raise VideoError(f"No decodable frames in {path}")
    return resize_to_max_width(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), max_width)


def iter_sampled_frames(
    path: str | Path,
    sample_fps: float = 1.0,
    max_width: int = DEFAULT_MAX_WIDTH,
    max_frames: int | None = None,
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (timestamp_s, frame_rgb) at ~sample_fps using the video's own clock.

    Timestamps are video time (frame_index / fps), so results do not depend on
    decode speed. Frames are decoded sequentially; skipped frames are grabbed
    without conversion.
    """
    import cv2

    if sample_fps <= 0:
        raise ValueError("sample_fps must be > 0")
    info = probe_video(path)
    cap = _open(path)
    step = 1.0 / sample_fps
    next_t = 0.0
    idx = 0
    emitted = 0
    try:
        while True:
            t = idx / info.fps
            if t + 1e-9 >= next_t:
                ok, bgr = cap.read()
                if not ok:
                    break
                yield t, resize_to_max_width(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), max_width)
                emitted += 1
                next_t += step
                if max_frames is not None and emitted >= max_frames:
                    break
            else:
                if not cap.grab():
                    break
            idx += 1
    finally:
        cap.release()
