"""Tests for EventStore (SQLite) and HealthChecker — Milestone 6."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest
from floodcam.config import HealthConfig
from floodcam.health import HealthChecker
from floodcam.store.db import EventStore

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path: Path) -> EventStore:
    s = EventStore(tmp_path / "test.db")
    yield s
    s.close()


def _minimal_event(run_id: str = "run1", camera_id: str = "cam1", timestamp: float = 1.0) -> dict:
    return dict(
        run_id=run_id,
        camera_id=camera_id,
        timestamp=timestamp,
        state="Normal",
        depth_cm=0.0,
        severity_class="Normal",
        confidence=1.0,
        rate_cm_per_10min=None,
        snapshot_path=None,
        model_version="test",
        config_hash="abc",
        synthetic=True,
        valid=True,
        stale=False,
    )


# ---------------------------------------------------------------------------
# Store tests
# ---------------------------------------------------------------------------


def test_schema_creation(store: EventStore) -> None:
    tables = {
        row[0]
        for row in store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert {"schema_version", "runs", "events", "alerts", "health", "machine_state"} <= tables


def test_schema_version_check(tmp_path: Path) -> None:
    db_path = tmp_path / "vcheck.db"
    s1 = EventStore(db_path)
    s1.close()
    s2 = EventStore(db_path)  # should not raise
    s2.close()


def test_event_round_trip(store: EventStore) -> None:
    store.upsert_run("run1", "src", "cam1", "cfg", "abc", 0.0, True)
    store.insert_event(**_minimal_event(timestamp=42.5))
    rows = store.events_for_run("run1")
    assert len(rows) == 1
    assert abs(rows[0]["timestamp"] - 42.5) < 1e-9
    assert rows[0]["state"] == "Normal"
    assert rows[0]["synthetic"] == 1


def test_alert_round_trip(store: EventStore) -> None:
    store.insert_alert("run1", "cam1", 10.0, "Normal→Pending", "Advisory", 4.0)
    rows = store.alerts_for_run("run1")
    assert len(rows) == 1
    assert rows[0]["transition"] == "Normal→Pending"
    assert abs(rows[0]["depth_cm"] - 4.0) < 1e-9


def test_machine_state_round_trip(store: EventStore) -> None:
    d = {"machine": {"flood_state": "Flood", "pending_since": None}}
    store.save_machine_state("cam1", d, 0.0)
    loaded = store.load_machine_state("cam1")
    assert loaded == d


def test_parameterized_query_hostile_camera_id(store: EventStore) -> None:
    evil_id = "cam'; DROP TABLE events; --"
    store.upsert_run("run1", "src", evil_id, "cfg", "abc", 0.0, True)
    store.insert_event(**_minimal_event(camera_id=evil_id))
    # If injection worked, events table would be gone → fetchall would raise
    rows = store.events_for_run("run1")
    assert len(rows) == 1


def test_concurrent_reader_during_write(tmp_path: Path) -> None:
    db_path = tmp_path / "concurrent.db"
    writer = EventStore(db_path)
    writer.upsert_run("run1", "src", "cam1", "cfg", "abc", 0.0, True)

    errors: list[Exception] = []

    def _read():
        try:
            reader = EventStore(db_path)
            for _ in range(10):
                reader.events_for_run("run1")
                time.sleep(0.001)
            reader.close()
        except Exception as e:
            errors.append(e)

    t = threading.Thread(target=_read)
    t.start()

    for i in range(100):
        writer.insert_event(**_minimal_event(run_id="run1", timestamp=float(i)))

    t.join(timeout=10)
    writer.close()
    assert not errors, f"Reader errors: {errors}"


def test_latest_event(store: EventStore) -> None:
    store.upsert_run("run1", "src", "cam1", "cfg", "abc", 0.0, True)
    for t in [1.0, 2.0, 3.0]:
        store.insert_event(**_minimal_event(timestamp=t))
    row = store.latest_event("cam1")
    assert abs(row["timestamp"] - 3.0) < 1e-9


def test_list_runs(store: EventStore) -> None:
    for i in [1, 2, 3]:
        store.upsert_run(f"run{i}", "src", "cam1", "cfg", "abc", float(i), True)
    runs = store.list_runs()
    assert len(runs) == 3
    assert runs[0]["run_id"] == "run3"  # DESC order


# ---------------------------------------------------------------------------
# HealthChecker tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synth_frame_02() -> np.ndarray:
    from floodcam.synth.renderer import SceneConfig, SynthRenderer

    h, tilt, f, dist, _ = (110.0, 22.0, 620.0, 210.0, "synthetic_outdoor_overcast")
    cfg = SceneConfig(camera_height_cm=h, camera_tilt_deg=tilt,
                      focal_length_px=f, pole_distance_cm=dist)
    renderer = SynthRenderer(cfg, seed=42 * 1000 + 2)
    return renderer.render_frame(depth_cm=5.0, variant="normal").image


def test_normal_frame_is_healthy(synth_frame_02: np.ndarray) -> None:
    checker = HealthChecker(HealthConfig())
    result = checker.check(synth_frame_02)
    assert result.healthy, f"Normal frame wrongly flagged: {result.detail}"


def test_black_frame_detected() -> None:
    checker = HealthChecker(HealthConfig())
    black = np.zeros((480, 640, 3), dtype=np.uint8)
    result = checker.check(black)
    assert result.black


def test_frozen_frame_detected(synth_frame_02: np.ndarray) -> None:
    cfg = HealthConfig(frozen_k_frames=3, frozen_diff_threshold=5.0)
    checker = HealthChecker(cfg)
    # Feed once (no history yet)
    checker.check(synth_frame_02)
    # Feed same frame again — frozen_count rises
    checker.check(synth_frame_02)
    # Third identical feed — should trip frozen (k_frames-1=2 consecutive matches)
    result = checker.check(synth_frame_02)
    assert result.frozen, "Repeated identical frame should be flagged frozen"


def test_normal_synthetic_no_false_positives() -> None:
    from floodcam.synth.renderer import SceneConfig, SynthRenderer

    h, tilt, f, dist, _ = (110.0, 22.0, 620.0, 210.0, "synthetic_outdoor_overcast")
    cfg = SceneConfig(camera_height_cm=h, camera_tilt_deg=tilt,
                      focal_length_px=f, pole_distance_cm=dist)
    renderer = SynthRenderer(cfg, seed=42 * 1000 + 2)
    checker = HealthChecker(HealthConfig())
    for i in range(50):
        frame = renderer.render_frame(depth_cm=5.0, frame_idx=i, variant="normal").image
        result = checker.check(frame)
        assert result.healthy, f"False positive at frame {i}: {result.detail}"


def test_obstructed_frame_detected(synth_frame_02: np.ndarray) -> None:
    import cv2

    cfg = HealthConfig(
        obstruction_laplacian_baseline_frames=30,
        obstruction_laplacian_ratio=0.3,
    )
    from floodcam.synth.renderer import SceneConfig, SynthRenderer

    h, tilt, f, dist, _ = (110.0, 22.0, 620.0, 210.0, "")
    scene_cfg = SceneConfig(camera_height_cm=h, camera_tilt_deg=tilt,
                            focal_length_px=f, pole_distance_cm=dist)
    renderer = SynthRenderer(scene_cfg, seed=42 * 1000 + 2)
    checker = HealthChecker(cfg)

    # Establish baseline with sharp frames
    for i in range(35):
        frame = renderer.render_frame(depth_cm=5.0, frame_idx=i, variant="normal").image
        checker.check(frame)

    # Feed a heavily blurred frame
    blurred = cv2.GaussianBlur(synth_frame_02, (51, 51), 0)
    result = checker.check(blurred)
    assert result.obstructed, (
        f"Heavily blurred frame should be obstructed; detail={result.detail}"
    )
