#!/usr/bin/env python
"""Write data/splits/usgs_<site>.json: frame IDs grouped by calendar-month block (an event/session).

Blocks, not frames, are the unit of splitting (consecutive frames are near-duplicates and lighting,
camera pose and weather are shared inside a block). The locked test blocks are fixed in LOCKED
below, chosen before any model was trained: they come from two different years and both contain
a real flow event. They are never used for tuning; see scripts/real_final_test.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCKED = {"in_overland": ["2025-04", "2026-04"]}


def main() -> int:
    rows = [json.loads(line) for line in (ROOT / "data/usgs/manifest.jsonl").read_text().splitlines()]
    for site, locked in LOCKED.items():
        blocks: dict[str, list[str]] = {}
        for r in rows:
            if r["site"] == site:
                blocks.setdefault(r["time"][:7], []).append(r["file"])
        spec = {
            "site": site, "unit": "calendar month block", "locked_test_blocks": locked,
            "dev_blocks": [b for b in sorted(blocks) if b not in locked],
            "blocks": {b: sorted(v) for b, v in sorted(blocks.items())},
            "synthetic": False,
        }
        (ROOT / "data/splits" / f"usgs_{site}.json").write_text(json.dumps(spec, indent=1))
        print(site, {b: len(v) for b, v in blocks.items()}, "locked:", locked)
    return 0


if __name__ == "__main__":
    sys.exit(main())
