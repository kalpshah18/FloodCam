"""SQLite event store with WAL mode for concurrent read-while-write.

Uses parameterised queries only. Never uses string formatting to build SQL.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

SCHEMA_VERSION = 1

_CREATE_SQL = [
    "PRAGMA journal_mode=WAL",
    "PRAGMA foreign_keys=ON",
    """CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS runs (
        run_id TEXT PRIMARY KEY,
        source TEXT,
        camera_id TEXT,
        config_hash TEXT,
        git_commit TEXT,
        started_at REAL,
        ended_at REAL,
        synthetic INTEGER NOT NULL DEFAULT 0
    )""",
    """CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        camera_id TEXT NOT NULL,
        timestamp REAL NOT NULL,
        state TEXT,
        depth_cm REAL,
        severity_class TEXT,
        confidence REAL,
        rate_cm_per_10min REAL,
        snapshot_path TEXT,
        model_version TEXT,
        config_hash TEXT,
        synthetic INTEGER NOT NULL DEFAULT 0,
        valid INTEGER NOT NULL DEFAULT 1,
        stale INTEGER NOT NULL DEFAULT 0
    )""",
    "CREATE INDEX IF NOT EXISTS idx_events_camera_ts ON events (camera_id, timestamp)",
    """CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        camera_id TEXT NOT NULL,
        timestamp REAL NOT NULL,
        transition TEXT,
        severity_class TEXT,
        depth_cm REAL,
        delivered INTEGER NOT NULL DEFAULT 0
    )""",
    """CREATE TABLE IF NOT EXISTS health (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        camera_id TEXT NOT NULL,
        timestamp REAL NOT NULL,
        frozen INTEGER NOT NULL DEFAULT 0,
        black INTEGER NOT NULL DEFAULT 0,
        obstructed INTEGER NOT NULL DEFAULT 0,
        offline INTEGER NOT NULL DEFAULT 0,
        detail TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS machine_state (
        camera_id TEXT PRIMARY KEY,
        state_json TEXT NOT NULL,
        updated_at REAL NOT NULL
    )""",
]

_EVENT_KEYS = [
    "run_id", "camera_id", "timestamp", "state", "depth_cm",
    "severity_class", "confidence", "rate_cm_per_10min", "snapshot_path",
    "model_version", "config_hash", "synthetic", "valid", "stale",
]


class EventStore:
    """WAL-mode SQLite store; safe for one writer + concurrent readers."""

    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn:
            for stmt in _CREATE_SQL:
                self._conn.execute(stmt)
            row = self._conn.execute(
                "SELECT version FROM schema_version"
            ).fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO schema_version VALUES (?)", (SCHEMA_VERSION,)
                )
            elif row[0] != SCHEMA_VERSION:
                raise RuntimeError(
                    f"DB schema version {row[0]} != expected {SCHEMA_VERSION}"
                )

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #

    def upsert_run(
        self,
        run_id: str,
        source: str,
        camera_id: str,
        config_hash: str,
        git_commit: str,
        started_at: float,
        synthetic: bool,
    ) -> None:
        with self._conn:
            self._conn.execute(
                """INSERT OR REPLACE INTO runs
                   (run_id, source, camera_id, config_hash, git_commit, started_at, synthetic)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (run_id, source, camera_id, config_hash, git_commit,
                 started_at, int(synthetic)),
            )

    def finalize_run(self, run_id: str, ended_at: float) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE runs SET ended_at = ? WHERE run_id = ?",
                (ended_at, run_id),
            )

    def insert_event(self, **kwargs: object) -> None:
        row = {k: kwargs.get(k) for k in _EVENT_KEYS}
        row["synthetic"] = int(bool(row.get("synthetic", False)))
        row["valid"] = int(bool(row.get("valid", True)))
        row["stale"] = int(bool(row.get("stale", False)))
        cols = ", ".join(_EVENT_KEYS)
        placeholders = ", ".join(["?"] * len(_EVENT_KEYS))
        with self._conn:
            self._conn.execute(
                f"INSERT INTO events ({cols}) VALUES ({placeholders})",
                [row[k] for k in _EVENT_KEYS],
            )

    def insert_alert(
        self,
        run_id: str,
        camera_id: str,
        timestamp: float,
        transition: str,
        severity_class: str,
        depth_cm: float,
    ) -> None:
        with self._conn:
            self._conn.execute(
                """INSERT INTO alerts
                   (run_id, camera_id, timestamp, transition, severity_class, depth_cm)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (run_id, camera_id, timestamp, transition, str(severity_class), depth_cm),
            )

    def insert_health(
        self,
        run_id: str,
        camera_id: str,
        timestamp: float,
        frozen: bool,
        black: bool,
        obstructed: bool,
        offline: bool,
        detail: Optional[str] = None,
    ) -> None:
        with self._conn:
            self._conn.execute(
                """INSERT INTO health
                   (run_id, camera_id, timestamp, frozen, black, obstructed, offline, detail)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (run_id, camera_id, timestamp,
                 int(frozen), int(black), int(obstructed), int(offline), detail),
            )

    def save_machine_state(
        self, camera_id: str, state_dict: dict, updated_at: float
    ) -> None:
        with self._conn:
            self._conn.execute(
                """INSERT OR REPLACE INTO machine_state (camera_id, state_json, updated_at)
                   VALUES (?, ?, ?)""",
                (camera_id, json.dumps(state_dict), updated_at),
            )

    # ------------------------------------------------------------------ #
    # Reads (used by dashboard)
    # ------------------------------------------------------------------ #

    def load_machine_state(self, camera_id: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT state_json FROM machine_state WHERE camera_id = ?",
            (camera_id,),
        ).fetchone()
        return json.loads(row["state_json"]) if row else None

    def latest_event(self, camera_id: str) -> Optional[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM events WHERE camera_id = ? ORDER BY timestamp DESC LIMIT 1",
            (camera_id,),
        ).fetchone()

    def events_for_run(self, run_id: str) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM events WHERE run_id = ? ORDER BY timestamp",
            (run_id,),
        ).fetchall()

    def alerts_for_run(self, run_id: str) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM alerts WHERE run_id = ? ORDER BY timestamp",
            (run_id,),
        ).fetchall()

    def latest_health(self, camera_id: str) -> Optional[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM health WHERE camera_id = ? ORDER BY timestamp DESC LIMIT 1",
            (camera_id,),
        ).fetchone()

    def list_runs(self) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM runs ORDER BY started_at DESC"
        ).fetchall()
