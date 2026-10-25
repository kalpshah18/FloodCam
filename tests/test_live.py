"""Upload-video path: video I/O, 2-point calibration, live pipeline, dashboard view.

All inputs are synthetic (floodcam.synth.video); nothing here is reportable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from floodcam.calib.linear import CalibError
from floodcam.io.video import VideoError, iter_sampled_frames, probe_video, read_frame
from floodcam.live import LiveSettings, analyze_video, build_camera_config
from floodcam.store.db import EventStore
from floodcam.synth.video import render_demo_video

_APP = Path(__file__).parent.parent / "app" / "dashboard.py"


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    d = tmp_path_factory.mktemp("vid")
    path, settings, depth_fn = render_demo_video(d / "demo.mp4", duration_s=60.0, fps=5.0)
    return path, settings, depth_fn


def test_probe_and_sampling(demo):
    path, _, _ = demo
    info = probe_video(path)
    assert (info.width, info.height) == (640, 480)
    assert info.fps == pytest.approx(5.0, abs=0.01)
    frames = list(iter_sampled_frames(path, sample_fps=1.0))
    assert 59 <= len(frames) <= 61
    ts = [t for t, _ in frames]
    assert ts[0] == 0.0 and all(b - a == pytest.approx(1.0, abs=0.21) for a, b in zip(ts, ts[1:]))
    assert frames[0][1].shape == (480, 640, 3)


def test_max_width_downscale(demo):
    path, _, _ = demo
    assert read_frame(path, 0.0, max_width=320).shape[1] == 320


def test_bad_file(tmp_path):
    bad = tmp_path / "x.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(VideoError):
        probe_video(bad)


def test_calibration_validation():
    s = LiveSettings(pole_x_px=100, ground_row=300, mark_row=100, mark_height_cm=50)
    cfg = build_camera_config(s, (480, 640, 3))
    assert cfg.calibration.scale_cm_per_px == pytest.approx(50 / 200)
    for kw in ({"mark_row": 400.0}, {"mark_height_cm": 0.0}, {"pole_x_px": 900}):
        bad = LiveSettings(**{**s.__dict__, **kw})
        with pytest.raises(CalibError):
            build_camera_config(bad, (480, 640, 3))


def test_pipeline_tracks_true_depth_and_states(demo):
    path, settings, depth_fn = demo
    out = list(analyze_video(path, settings))
    errs = [abs(f.raw_depth_cm - depth_fn(f.timestamp_s)) for f in out if f.raw_depth_cm is not None]
    assert len(errs) >= 55 and np.mean(errs) < 0.7
    assert max(f.depth_cm for f in out) > 10.0
    states = [f.state for f in out]
    assert states[0] == "Normal" and "Flood" in states and "Recovering" in states
    assert out[-1].depth_cm < 2.5  # dry again: no stale value held
    seq = [f.transition for f in out if f.transition]
    assert seq == ["Normal→Pending", "Pending→Flood", "Flood→Recovering", "Recovering→Normal"]


def test_store_roundtrip(demo, tmp_path):
    path, settings, _ = demo
    store = EventStore(tmp_path / "u.db")
    n = sum(1 for _ in analyze_video(path, settings, store=store, run_id="r1",
                                     snapshot_dir=tmp_path / "snaps", synthetic=True))
    events = store.events_for_run("r1")
    store.close()
    assert len(events) == n and all(e["synthetic"] == 1 for e in events)
    assert any((tmp_path / "snaps").iterdir())


def test_upload_view_apptest_demo():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(_APP), default_timeout=90)
    at.run()
    assert not at.exception
    next(b for b in at.sidebar.button if "demo" in b.label.lower()).click()
    at.run()
    assert not at.exception
    next(b for b in at.main.button if b.label == "Start analysis").click()
    at.run()
    assert not at.exception
    text = " ".join(str(m.value) for m in at.main.metric)
    assert "Peak depth" in " ".join(m.label for m in at.main.metric) and text
