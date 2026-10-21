"""Scene manifest: one row per frame.

Schema (all fields required unless marked optional):
    frame_id      str   unique frame identifier
    scene_id      str   scene / recording session
    session_id    str   usually same as scene_id; group for CV splits
    camera_id     str
    image_path    str   relative path to RGB image
    mask_path     str   relative path to binary mask (may be empty string)
    depth_cm      float true water depth; NaN if unknown
    lighting      str   e.g. "indoor_diffuse", "synthetic"
    is_negative   bool  True for hard-negative frames (no real flood)
    synthetic     bool  REQUIRED; raises ManifestError if missing
    split         str   "train"|"val"|"test"|"" (filled after splitting)

Storage format: JSONL (one JSON object per line) or CSV.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterable


class ManifestError(ValueError):
    """Raised when a manifest row fails validation."""


_REQUIRED_FIELDS = {
    "frame_id",
    "scene_id",
    "session_id",
    "camera_id",
    "image_path",
    "mask_path",
    "depth_cm",
    "lighting",
    "is_negative",
    "synthetic",
}

_BOOL_FIELDS = {"is_negative", "synthetic"}
_FLOAT_FIELDS = {"depth_cm"}


# ---------------------------------------------------------------------------
# Row dataclass
# ---------------------------------------------------------------------------


@dataclass
class ManifestRow:
    frame_id: str
    scene_id: str
    session_id: str
    camera_id: str
    image_path: str
    mask_path: str
    depth_cm: float
    lighting: str
    is_negative: bool
    synthetic: bool
    split: str = ""

    # ---- construction helpers ------------------------------------------ #

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ManifestRow:
        """Parse and validate one row dict. Raises ManifestError on failure."""
        _validate(d)
        coerced: dict[str, Any] = {}
        for f in fields(cls):
            if f.name not in d:
                coerced[f.name] = f.default  # only 'split' has a default
                continue
            v = d[f.name]
            if f.name in _BOOL_FIELDS:
                if isinstance(v, str):
                    v = v.lower() in ("true", "1", "yes")
                else:
                    v = bool(v)
            elif f.name in _FLOAT_FIELDS:
                v = float(v) if v not in (None, "", "nan") else float("nan")
            coerced[f.name] = v
        return cls(**coerced)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate(d: dict[str, Any]) -> None:
    missing = _REQUIRED_FIELDS - set(d.keys())
    if "synthetic" in missing:
        raise ManifestError(
            f"Row missing required field 'synthetic' (frame_id={d.get('frame_id', '?')}). "
            "Every row must explicitly declare whether it is synthetic."
        )
    if missing:
        raise ManifestError(
            f"Row missing required fields: {sorted(missing)} "
            f"(frame_id={d.get('frame_id', '?')})"
        )


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def write_jsonl(rows: Iterable[ManifestRow], path: Path) -> int:
    """Write rows to a JSONL file. Returns row count."""
    count = 0
    with path.open("w", newline="", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row.to_dict()) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[ManifestRow]:
    """Read and validate all rows from a JSONL file."""
    rows = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestError(f"Line {lineno}: invalid JSON — {exc}") from exc
            rows.append(ManifestRow.from_dict(d))
    return rows


def write_csv(rows: Iterable[ManifestRow], path: Path) -> int:
    """Write rows to a CSV file. Returns row count."""
    field_names = [f.name for f in fields(ManifestRow)]
    rows_list = list(rows)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=field_names)
        writer.writeheader()
        for row in rows_list:
            writer.writerow(row.to_dict())
    return len(rows_list)


def read_csv(path: Path) -> list[ManifestRow]:
    """Read and validate all rows from a CSV file."""
    rows = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for lineno, d in enumerate(reader, 2):
            rows.append(ManifestRow.from_dict(dict(d)))
    return rows
