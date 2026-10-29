#!/usr/bin/env python
"""List camera folders in the public USGS NIMS image bucket; write data/usgs/cameras.csv.

The bucket listing is public. `staff_like` flags names that suggest a staff gauge in view
(contains 'staff', 'gauge', 'gage', 'plate'); verify visually before use. Gauge height for a
camera still has to come from NWIS (needs api.waterdata.usgs.gov / waterservices.usgs.gov).
"""

from __future__ import annotations

import csv
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = "https://usgs-nims-images.s3.amazonaws.com/"
NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}


def main() -> int:
    names, token = [], None
    while True:
        q = {"list-type": "2", "prefix": "overlay/", "delimiter": "/", "max-keys": "1000"}
        if token:
            q["continuation-token"] = token
        with urllib.request.urlopen(BASE + "?" + urllib.parse.urlencode(q), timeout=60) as r:
            root = ET.fromstring(r.read())
        names += [p.find("s:Prefix", NS).text.split("/")[1] for p in root.findall("s:CommonPrefixes", NS)]
        if root.findtext("s:IsTruncated", namespaces=NS) != "true":
            break
        token = root.findtext("s:NextContinuationToken", namespaces=NS)
    pat = re.compile(r"staff|gauge|gage|plate", re.I)
    out = ROOT / "data/usgs/cameras.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["camera", "staff_like"])
        for n in sorted(names):
            w.writerow([n, int(bool(pat.search(n)))])
    print(len(names), "cameras;", sum(bool(pat.search(n)) for n in names), "staff-like ->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
