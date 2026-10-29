"""Write the annotated analysis video (browser-playable H.264 when ffmpeg is available)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np


class AnnotatedVideoWriter:
    def __init__(self, path: str | Path, fps: float = 6.0) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self._writer = None
        self._raw = self.path.with_suffix(".raw.mp4")
        self.frames = 0

    def write(self, frame_rgb: np.ndarray) -> None:
        import cv2

        h, w = frame_rgb.shape[:2]
        w2, h2 = w - w % 2, h - h % 2  # even size for yuv420p
        if self._writer is None:
            self._writer = cv2.VideoWriter(
                str(self._raw), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w2, h2)
            )
            self._size = (w2, h2)
        self._writer.write(cv2.cvtColor(frame_rgb[:h2, :w2], cv2.COLOR_RGB2BGR))
        self.frames += 1

    def close(self) -> Optional[Path]:
        """Finish the file. Returns the final path, or None if no frame was written."""
        if self._writer is None:
            return None
        self._writer.release()
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            try:
                subprocess.run(
                    [ffmpeg, "-y", "-loglevel", "error", "-i", str(self._raw), "-c:v", "libx264",
                     "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.path)],
                    check=True, timeout=300,
                )
                self._raw.unlink(missing_ok=True)
                return self.path
            except (subprocess.SubprocessError, OSError):
                pass
        self._raw.replace(self.path)  # mp4v fallback: downloadable, may not play in-browser
        return self.path
