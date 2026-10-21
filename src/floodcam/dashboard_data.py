"""Pure-Python data layer for the FloodCam dashboard.

No Streamlit imports. All functions open the DB read-only (URI mode=ro).
The dashboard reads numbers from the DB only; it never re-runs the pipeline.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Internal helper — always opens read-only
# ---------------------------------------------------------------------------

def _open_ro(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _row_to_dict(row: Optional[sqlite3.Row]) -> Optional[dict]:
    return dict(row) if row is not None else None


def _rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict]:
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Run / metadata
# ---------------------------------------------------------------------------

def list_runs(db_path: str | Path) -> list[dict]:
    """Return all runs ordered by start time (most recent first)."""
    conn = _open_ro(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM runs ORDER BY started_at DESC"
        ).fetchall()
        return _rows_to_dicts(rows)
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def get_run(db_path: str | Path, run_id: str) -> Optional[dict]:
    """Return a single run row as a dict, or None."""
    conn = _open_ro(db_path)
    try:
        row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return _row_to_dict(row)
    except sqlite3.OperationalError:
        return None
    finally:
        conn.close()


def cameras_for_run(db_path: str | Path, run_id: str) -> list[str]:
    """Distinct camera_ids that have events in this run."""
    conn = _open_ro(db_path)
    try:
        rows = conn.execute(
            "SELECT DISTINCT camera_id FROM events WHERE run_id = ? ORDER BY camera_id",
            (run_id,),
        ).fetchall()
        return [r["camera_id"] for r in rows]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Status panel (live state per camera as of cursor)
# ---------------------------------------------------------------------------

def camera_status(
    db_path: str | Path,
    run_id: str,
    camera_id: str,
    cursor_ts: float,
) -> Optional[dict]:
    """Return the most recent event for camera_id at or before cursor_ts.

    Returns None if no events exist for that camera/run up to the cursor.
    Fields: timestamp, state, depth_cm, severity_class, confidence,
            rate_cm_per_10min, stale, valid, snapshot_path.
    """
    conn = _open_ro(db_path)
    try:
        row = conn.execute(
            """SELECT * FROM events
               WHERE run_id = ? AND camera_id = ? AND timestamp <= ?
               ORDER BY timestamp DESC LIMIT 1""",
            (run_id, camera_id, cursor_ts),
        ).fetchone()
        return _row_to_dict(row)
    except sqlite3.OperationalError:
        return None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Depth time-series panel
# ---------------------------------------------------------------------------

def depth_series(
    db_path: str | Path,
    run_id: str,
    camera_id: str,
    cursor_ts: float,
) -> list[dict]:
    """Return all events for camera_id at or before cursor_ts.

    Each dict has: timestamp, depth_cm, state, severity_class, confidence,
    rate_cm_per_10min, stale, valid.
    Invalid or low-confidence events are included; callers use valid/confidence
    fields to distinguish them in the chart.
    """
    conn = _open_ro(db_path)
    try:
        rows = conn.execute(
            """SELECT timestamp, depth_cm, state, severity_class, confidence,
                      rate_cm_per_10min, stale, valid
               FROM events
               WHERE run_id = ? AND camera_id = ? AND timestamp <= ?
               ORDER BY timestamp""",
            (run_id, camera_id, cursor_ts),
        ).fetchall()
        return _rows_to_dicts(rows)
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Alerts panel
# ---------------------------------------------------------------------------

def alerts_up_to(
    db_path: str | Path,
    run_id: str,
    camera_id: Optional[str],
    cursor_ts: float,
) -> list[dict]:
    """Return alerts at or before cursor_ts.  Pass camera_id=None for all cameras."""
    conn = _open_ro(db_path)
    try:
        if camera_id is not None:
            rows = conn.execute(
                """SELECT * FROM alerts
                   WHERE run_id = ? AND camera_id = ? AND timestamp <= ?
                   ORDER BY timestamp""",
                (run_id, camera_id, cursor_ts),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT * FROM alerts
                   WHERE run_id = ? AND timestamp <= ?
                   ORDER BY timestamp""",
                (run_id, cursor_ts),
            ).fetchall()
        return _rows_to_dicts(rows)
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Health panel
# ---------------------------------------------------------------------------

def camera_health_status(
    db_path: str | Path,
    run_id: str,
    camera_id: str,
    cursor_ts: float,
) -> Optional[dict]:
    """Return the most recent health row at or before cursor_ts."""
    conn = _open_ro(db_path)
    try:
        row = conn.execute(
            """SELECT * FROM health
               WHERE run_id = ? AND camera_id = ? AND timestamp <= ?
               ORDER BY timestamp DESC LIMIT 1""",
            (run_id, camera_id, cursor_ts),
        ).fetchone()
        return _row_to_dict(row)
    except sqlite3.OperationalError:
        return None
    finally:
        conn.close()


def camera_health_history(
    db_path: str | Path,
    run_id: str,
    camera_id: str,
    cursor_ts: float,
) -> list[dict]:
    """Return all health rows at or before cursor_ts (fault history)."""
    conn = _open_ro(db_path)
    try:
        rows = conn.execute(
            """SELECT * FROM health
               WHERE run_id = ? AND camera_id = ? AND timestamp <= ?
               ORDER BY timestamp""",
            (run_id, camera_id, cursor_ts),
        ).fetchall()
        return _rows_to_dicts(rows)
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def last_healthy_timestamp(
    db_path: str | Path,
    run_id: str,
    camera_id: str,
    cursor_ts: float,
) -> Optional[float]:
    """Timestamp of the most recent healthy (all flags=0) frame at or before cursor_ts."""
    conn = _open_ro(db_path)
    try:
        row = conn.execute(
            """SELECT timestamp FROM health
               WHERE run_id = ? AND camera_id = ? AND timestamp <= ?
                 AND frozen = 0 AND black = 0 AND obstructed = 0 AND offline = 0
               ORDER BY timestamp DESC LIMIT 1""",
            (run_id, camera_id, cursor_ts),
        ).fetchone()
        return float(row["timestamp"]) if row else None
    except sqlite3.OperationalError:
        return None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Snapshot lookup
# ---------------------------------------------------------------------------

def nearest_snapshot(
    db_path: str | Path,
    run_id: str,
    camera_id: str,
    cursor_ts: float,
) -> Optional[str]:
    """Return the snapshot_path of the event nearest to but not after cursor_ts.

    Returns None if no snapshot exists or the file no longer exists on disk.
    """
    conn = _open_ro(db_path)
    try:
        row = conn.execute(
            """SELECT snapshot_path FROM events
               WHERE run_id = ? AND camera_id = ? AND timestamp <= ?
                 AND snapshot_path IS NOT NULL
               ORDER BY timestamp DESC LIMIT 1""",
            (run_id, camera_id, cursor_ts),
        ).fetchone()
        if row is None:
            return None
        path = row["snapshot_path"]
        if path and Path(path).is_file():
            return path
        return None
    except sqlite3.OperationalError:
        return None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Run metadata for banner
# ---------------------------------------------------------------------------

def run_metadata(db_path: str | Path, run_id: str) -> dict:
    """Return run metadata fields needed by the banner.

    Keys: run_id, synthetic, config_hash, git_commit, model_version, started_at.
    model_version is taken from the first event's model_version column.
    """
    conn = _open_ro(db_path)
    try:
        run_row = conn.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if run_row is None:
            return {}
        meta = dict(run_row)
        ev_row = conn.execute(
            "SELECT model_version FROM events WHERE run_id = ? LIMIT 1", (run_id,)
        ).fetchone()
        meta["model_version"] = ev_row["model_version"] if ev_row else None
        return meta
    except sqlite3.OperationalError:
        return {}
    finally:
        conn.close()
