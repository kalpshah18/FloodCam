#!/usr/bin/env python
"""Survey staff-gauge cameras: NIMS metadata + NWIS gage-height (00065) availability.

Needs api.waterdata.usgs.gov and waterservices.usgs.gov. Writes data/usgs/camera_survey.csv.
"""

from __future__ import annotations

import csv
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NIMS = "https://api.waterdata.usgs.gov/nims/v0"
IV = "https://waterservices.usgs.gov/nwis/iv/"
# Requests older than ~120 days are redirected (301) to nwis.waterservices.usgs.gov, which must be allowed
# separately; the default window below stays inside what waterservices.usgs.gov serves directly.
import datetime as _dt
END = _dt.date.today().isoformat()
START = (_dt.date.today() - _dt.timedelta(days=110)).isoformat()


def get(url: str):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read())


def main() -> int:
    cams = [r["camera"] for r in csv.DictReader((ROOT / "data/usgs/cameras.csv").open()) if r["staff_like"] == "1"]
    out = []
    for c in cams:
        row = {"camera": c, "nwis_id": "", "created": "", "newest": "", "n_images": "", "n_gage": "", "gage_min": "",
               "gage_max": "", "gage_first": "", "gage_last": ""}
        try:
            meta = get(f"{NIMS}/cameras?camId={urllib.parse.quote(c)}")[0]
            row.update(nwis_id=meta.get("nwisId", ""), created=meta.get("createdDate", "")[:10],
                       newest=meta.get("newestImageDT", "")[:10])
            files = get(f"{NIMS}/listFiles?camId={urllib.parse.quote(c)}&limit=50000")
            row["n_images"] = len(files)
            q = urllib.parse.urlencode({"sites": row["nwis_id"], "parameterCd": "00065", "startDT": START,
                                        "endDT": END, "format": "json"})
            ts = get(IV + "?" + q)["value"]["timeSeries"]
            if ts:
                vals = [(v["dateTime"], float(v["value"])) for v in ts[0]["values"][0]["value"] if float(v["value"]) > -999]
                if vals:
                    row.update(n_gage=len(vals), gage_min=min(v for _, v in vals), gage_max=max(v for _, v in vals),
                               gage_first=vals[0][0][:10], gage_last=vals[-1][0][:10])
        except Exception as e:  # keep going; record the failure
            row["created"] = row["created"] or f"ERR {type(e).__name__}"
            print("  error", c, e, file=sys.stderr)
        out.append(row)
        print({k: row[k] for k in ("camera", "nwis_id", "n_images", "n_gage", "gage_min", "gage_max")}, flush=True)
    with (ROOT / "data/usgs/camera_survey.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
