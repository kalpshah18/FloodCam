#!/usr/bin/env python
"""Build data/usgs/truth/<site>_gauge.csv for a staff-gauge camera from NIMS + NWIS (recent window).

Needs api.waterdata.usgs.gov and waterservices.usgs.gov. waterservices serves only the last ~120 days
directly; older ranges are redirected to nwis.waterservices.usgs.gov, which must be allowed separately.
Frames are matched to the nearest gage-height sample within --max-gap-min minutes.

Usage: python scripts/fetch_usgs_labels.py --site in_north --camera IN_Overland_Flow_North_of_Cr_1000_N_nr_Decatur_Staff --days 110 --cap 1400
"""

from __future__ import annotations

import argparse
import bisect
import csv
import datetime as dt
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NIMS = "https://api.waterdata.usgs.gov/nims/v0"
IV = "https://waterservices.usgs.gov/nwis/iv/"


def get(url: str):
    with urllib.request.urlopen(url, timeout=90) as r:
        return json.loads(r.read())


def frame_time(name: str) -> dt.datetime:
    d, t = name.rsplit("___", 1)[1].removesuffix(".jpg").removesuffix("Z").split("T")
    return dt.datetime.fromisoformat(f"{d}T{t.replace('-', ':')}").replace(tzinfo=dt.timezone.utc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--camera", required=True)
    ap.add_argument("--days", type=int, default=110)
    ap.add_argument("--cap", type=int, default=1400, help="max frames (uniform stride)")
    ap.add_argument("--max-gap-min", type=float, default=10.0)
    a = ap.parse_args()
    cam = urllib.parse.quote(a.camera)
    meta = get(f"{NIMS}/cameras?camId={cam}")[0]
    end = dt.date.today()
    start = end - dt.timedelta(days=a.days)
    files = get(f"{NIMS}/listFiles?camId={cam}&limit=50000&after={start.isoformat()}")
    files = sorted(f for f in files if f.endswith(".jpg"))
    q = urllib.parse.urlencode({"sites": meta["nwisId"], "parameterCd": "00065", "startDT": start.isoformat(),
                                "endDT": end.isoformat(), "format": "json"})
    ts = get(IV + "?" + q)["value"]["timeSeries"]
    vals = [(dt.datetime.fromisoformat(v["dateTime"]).astimezone(dt.timezone.utc), float(v["value"]))
            for v in ts[0]["values"][0]["value"] if float(v["value"]) > -999]
    vals.sort()
    times = [t for t, _ in vals]
    rows = []
    for f in files:
        t = frame_time(f)
        i = bisect.bisect_left(times, t)
        cands = [j for j in (i - 1, i) if 0 <= j < len(times)]
        j = min(cands, key=lambda k: abs((times[k] - t).total_seconds()))
        if abs((times[j] - t).total_seconds()) <= a.max_gap_min * 60:
            rows.append((f, t.strftime("%Y-%m-%dT%H:%M:%S"), vals[j][1]))
    if len(rows) > a.cap:
        stride = len(rows) / a.cap
        rows = [rows[int(k * stride)] for k in range(a.cap)]
    out = ROOT / "data" / "usgs" / "truth" / f"{a.site}_gauge.csv"
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "time", "gage_ft"])
        w.writerows(rows)
    print(f"{a.site}: {len(files)} frames listed, {len(rows)} labelled -> {out}; nwis {meta['nwisId']}; "
          f"gauge {min(v for _, v in vals):.2f}..{max(v for _, v in vals):.2f} ft")
    return 0


if __name__ == "__main__":
    sys.exit(main())
