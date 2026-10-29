"""Real USGS gauge-camera footage: replay an event with the camera estimate next to the official
gauge, plus the headline results. Everything shown is read from files written by scripts/
(results/real/**, data/usgs/demo/); nothing is computed or typed here.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent
R = ROOT / "results" / "real"
DEMO = ROOT / "data" / "usgs" / "demo"

SEV = [(3.0, "Normal", "#009E73"), (10.0, "Advisory", "#56B4E9"), (25.0, "Warning", "#E69F00"),
       (50.0, "Severe", "#D55E00"), (1e9, "Critical", "#CC79A7")]


def _sev(cm: float) -> tuple[str, str]:
    for hi, name, col in SEV:
        if cm < hi:
            return name, col
    return SEV[-1][1], SEV[-1][2]


def _final_run() -> str | None:
    for r in ("real-final-in_overland-f1-force", "real-final-in_overland-f1"):
        if (R / r / "metrics.json").exists():
            return r
    return None


def _headline() -> None:
    cv = R / "real-cv-in_overland-f1" / "metrics.json"
    fin = _final_run()
    if not cv.exists():
        st.info("Run `scripts/real_cv.py` to generate the results.")
        return
    rows = []
    for label, path in (("Leave-one-month-out CV (dev)", cv),
                        ("Locked test months", R / fin / "metrics.json" if fin else None)):
        if path is None or not path.exists():
            continue
        m = json.loads(path.read_text())
        for scope in ("gated", "all_frames"):
            for meth in ("learned", "threshold_baseline", "always_no_flood"):
                d = m[scope][meth]
                f = d["flood_3cm"]
                rows.append({"evaluation": label, "frames": "quality-gated" if scope == "gated" else "all",
                             "method": meth, "n": d["depth"]["n"], "MAE (cm)": round(d["depth"]["mae_cm"], 2),
                             "F1 (>=3 cm)": None if f["f1"] != f["f1"] else round(f["f1"], 2),
                             "coverage": round(m["coverage"], 2) if scope == "gated" else 1.0})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("Frames are from one camera; uncertainty is the spread across held-out months "
               "(see results/real/SUMMARY.md), not across frames.")


def render() -> None:
    st.header("Real footage: USGS gauge cameras")
    st.caption("Real fixed-camera frames of a staff plate on a shallow overland-flow site, labelled by "
               "the official USGS gauge. The estimate shown for each frame comes from a model that "
               "never saw that month.")
    tab_replay, tab_results, tab_audit = st.tabs(["Event replay", "Results", "Failure audit"])

    with tab_replay:
        demo_csv = DEMO / "demo.csv"
        if not demo_csv.exists():
            st.warning("Demo bundle missing. Run `python scripts/make_real_demo.py`.")
        else:
            d = pd.read_csv(demo_csv, parse_dates=["time"]).reset_index(drop=True)
            c1, c2 = st.columns([3, 1])
            speed = c2.select_slider("Playback", ["Manual", "Slow", "Fast"], value="Manual")
            idx = c1.slider("Frame", 0, len(d) - 1, st.session_state.get("real_idx", 0), key="real_slider")
            ph_top, ph_main = st.empty(), st.empty()

            def show(i: int) -> None:
                r = d.iloc[i]
                est = float(r.learned_cm)
                name, col = _sev(est)
                with ph_top.container():
                    a, b, c, e = st.columns(4)
                    a.metric("Official gauge", f"{max(r.gauge_cm, 0):.1f} cm")
                    b.metric("Camera estimate", f"{est:.1f} cm", delta=f"{est - max(r.gauge_cm, 0):+.1f} vs gauge",
                             delta_color="off")
                    c.markdown(f"**Severity (camera)**<br><span style='background:{col};padding:2px 10px;"
                               f"border-radius:4px;font-weight:600'>{name}</span>", unsafe_allow_html=True)
                    e.markdown("**Image quality gate**<br>" + ("passed" if r.accepted else "rejected (blur/dirt)")
                               + (" · IR night" if r.is_ir else ""), unsafe_allow_html=True)
                with ph_main.container():
                    left, right = st.columns([1, 1])
                    left.image(str(DEMO / "frames" / r.file), caption=str(r.time) + " UTC",
                               use_container_width=True)
                    past = d.iloc[: i + 1].copy()
                    base = alt.Chart(past).encode(x=alt.X("time:T", title="time (UTC)"))
                    g = base.mark_line(color="black", strokeWidth=2).encode(y=alt.Y("gauge_cm:Q", title="depth (cm)"))
                    pts = base.mark_circle(size=40).encode(
                        y="learned_cm:Q",
                        color=alt.Color("accepted:N", scale=alt.Scale(domain=[True, False], range=["#1f77b4", "#bbbbbb"]),
                                        legend=alt.Legend(title="passed gate")))
                    rule = alt.Chart(pd.DataFrame({"y": [3.0]})).mark_rule(color="red", strokeDash=[4, 3]).encode(y="y:Q")
                    right.altair_chart((g + pts + rule).properties(height=330), use_container_width=True)

            show(idx)
            if speed != "Manual":
                for i in range(idx + 1, len(d)):
                    time.sleep(0.45 if speed == "Slow" else 0.12)
                    show(i)
                st.session_state["real_idx"] = len(d) - 1

    with tab_results:
        _headline()
        for name, cap in (("scatter_final.png", "Locked test months: camera vs gauge"),
                          ("scatter_cv.png", "Leave-one-month-out: camera vs gauge"),
                          ("resolution.png", "Error and 3 cm detection vs camera resolution"),
                          ("block_mae_cv.png", "Error per held-out month"),
                          ("risk_coverage_cv.png", "Error vs coverage (confidence-ranked)")):
            p = R / "figures" / name
            if p.exists():
                st.image(str(p), caption=cap, use_container_width=True)

    with tab_audit:
        f = R / "failures" / "failures.csv"
        if not f.exists():
            st.info("Run `scripts/real_report.py`.")
        else:
            df = pd.read_csv(f)
            st.caption("Worst accepted frames and a sample of rejected frames (left: frame, right: "
                       "rectified plate).")
            for r in df.itertuples():
                st.image(str(R / "failures" / r.overlay), caption=f"{r.kind}: gauge {r.gauge_cm:.1f} cm, "
                         f"estimate {r.est_cm:.1f} cm", use_container_width=True)
