"""FloodCam dashboard — reads SQLite event log; never writes.

Usage:
    streamlit run app/dashboard.py
    streamlit run app/dashboard.py -- --db results/demo.db

All numbers are read from the DB. The pipeline is never re-run here.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from floodcam.dashboard_data import (  # noqa: E402
    alerts_up_to,
    camera_health_history,
    camera_health_status,
    camera_status,
    cameras_for_run,
    depth_series,
    last_healthy_timestamp,
    list_runs,
    nearest_snapshot,
    run_metadata,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="FloodCam Monitor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Colorblind-safe severity palette (Okabe-Ito subset; text label always shown)
# ---------------------------------------------------------------------------

_SEVERITY_COLOR = {
    "Normal": "#009E73",      # green
    "Advisory": "#56B4E9",    # sky-blue
    "Warning": "#E69F00",     # orange
    "Severe": "#D55E00",      # vermilion
    "Critical": "#CC79A7",    # reddish-purple
    "Unknown": "#999999",
}

_STATE_COLOR = {
    "Normal": "#009E73",
    "Pending": "#F0E442",
    "Flood": "#D55E00",
    "Recovering": "#56B4E9",
}

# Threshold lines (cm)
_THRESHOLD_LINES = [
    {"depth": 3.5,  "label": "trigger 3.5 cm",  "color": "#E69F00"},
    {"depth": 2.5,  "label": "clear 2.5 cm",    "color": "#56B4E9"},
    {"depth": 3.0,  "label": "Normal/Advisory",  "color": "#009E73"},
    {"depth": 10.0, "label": "Advisory/Warning", "color": "#F0E442"},
    {"depth": 25.0, "label": "Warning/Severe",   "color": "#D55E00"},
    {"depth": 50.0, "label": "Severe/Critical",  "color": "#CC79A7"},
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_db() -> Path:
    """Default DB path: results/demo.db relative to repo root."""
    return Path(__file__).parent.parent / "results" / "demo.db"


def _fmt_ts(ts: float | None) -> str:
    if ts is None:
        return "—"
    m, s = divmod(int(ts), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _sev_badge(sev: str | None) -> str:
    if not sev:
        return "Unknown"
    return sev


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _sidebar() -> tuple[Path, str | None, str | None, float, bool]:
    """Render sidebar; return (db_path, run_id, camera_id, cursor_ts, live_mode)."""
    st.sidebar.header("FloodCam Controls")

    # DB path
    default_db = str(_default_db())
    db_str = st.sidebar.text_input("DB path", value=default_db)
    db_path = Path(db_str)

    if not db_path.exists():
        st.sidebar.warning("DB file not found.")
        return db_path, None, None, 0.0, False

    # Run selector
    runs = list_runs(db_path)
    if not runs:
        st.sidebar.warning("No runs found in DB.")
        return db_path, None, None, 0.0, False

    run_labels = [r["run_id"] for r in runs]
    selected_run = st.sidebar.selectbox(
        "Run", run_labels, index=0, key="run_selector"
    )

    # Camera selector
    cameras = cameras_for_run(db_path, selected_run)
    cam_options = ["(all)"] + cameras
    selected_cam_label = st.sidebar.selectbox(
        "Camera", cam_options, index=0, key="cam_selector"
    )
    selected_cam: str | None = None if selected_cam_label == "(all)" else selected_cam_label

    # Determine time range for the run
    series_data = []
    cams_for_range = cameras if cameras else []
    for cam in cams_for_range:
        series_data.extend(depth_series(db_path, selected_run, cam, float("inf")))

    if series_data:
        all_ts = [r["timestamp"] for r in series_data if r["timestamp"] is not None]
        t_min = min(all_ts)
        t_max = max(all_ts)
    else:
        t_min, t_max = 0.0, 1.0

    # Live mode vs replay
    live_mode = st.sidebar.checkbox("Live mode (auto-poll)", value=False, key="live_mode")

    if live_mode:
        cursor_ts = float("inf")  # always show everything
    else:
        # Replay controls
        st.sidebar.subheader("Replay")
        st.sidebar.select_slider(
            "Speed", options=["1x", "10x", "60x"], value="1x", key="replay_speed"
        )
        cursor_ts = st.sidebar.slider(
            "Cursor (s)",
            min_value=float(t_min),
            max_value=float(t_max),
            value=float(t_max),
            step=1.0,
            key="cursor_slider",
            format="%.0f s",
        )

    return db_path, selected_run, selected_cam, cursor_ts, live_mode


# ---------------------------------------------------------------------------
# Panel 1: Live status per camera
# ---------------------------------------------------------------------------

def _panel_status(db_path: Path, run_id: str, cameras: list[str], cursor_ts: float) -> None:
    st.subheader("Camera Status")
    if not cameras:
        st.info("No cameras found for this run.")
        return

    cols = st.columns(min(len(cameras), 4))
    for idx, cam_id in enumerate(cameras):
        status = camera_status(db_path, run_id, cam_id, cursor_ts)
        col = cols[idx % len(cols)]
        with col:
            if status is None:
                st.markdown(
                    f"**{cam_id}**\n\n_No data at cursor_"
                )
                continue

            sev = status.get("severity_class") or "Unknown"
            state = status.get("state") or "Unknown"
            depth = status.get("depth_cm")
            conf = status.get("confidence")
            rate = status.get("rate_cm_per_10min")
            ts = status.get("timestamp")
            stale = bool(status.get("stale", 0))

            color = _SEVERITY_COLOR.get(sev, "#999999")

            depth_str = f"{depth:.2f} cm" if depth is not None else "—"
            conf_str = f"{conf:.2f}" if conf is not None else "—"
            rate_str = f"{rate:+.2f} cm/10min" if rate is not None else "—"
            ts_str = _fmt_ts(ts)
            stale_note = " ⚠ stale" if stale else ""

            st.markdown(
                f"""<div style="border-left:6px solid {color};padding:8px 12px;
                    background:#f8f8f8;border-radius:4px;margin-bottom:8px">
                <b>{cam_id}</b><br/>
                State: <b>{state}</b>{stale_note}<br/>
                Severity: <span style="color:{color};font-weight:bold">{sev}</span><br/>
                Depth: {depth_str}<br/>
                Confidence: {conf_str}<br/>
                Rate of rise: {rate_str}<br/>
                Last update: {ts_str}
                </div>""",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Panel 2: Depth time-series (Altair)
# ---------------------------------------------------------------------------

def _panel_depth_chart(
    db_path: Path, run_id: str, camera_id: str, cursor_ts: float
) -> None:
    st.subheader(f"Depth Time-Series — {camera_id}")

    data = depth_series(db_path, run_id, camera_id, cursor_ts)
    if not data:
        st.info("No depth data up to this cursor position.")
        return

    df = pd.DataFrame(data)
    df["timestamp"] = df["timestamp"].astype(float)
    df["depth_cm"] = pd.to_numeric(df["depth_cm"], errors="coerce").fillna(0.0)
    df["valid"] = df["valid"].astype(int)
    df["stale"] = df["stale"].astype(int)
    df["marker_type"] = df.apply(
        lambda r: "invalid" if r["valid"] == 0
        else ("low-conf" if (r.get("confidence") or 1.0) < 0.5
              else "normal"),
        axis=1,
    )

    t_min = float(df["timestamp"].min())
    t_max = float(df["timestamp"].max())

    # State bands
    state_changes = df[["timestamp", "state"]].copy()
    state_changes = state_changes.drop_duplicates().reset_index(drop=True)
    band_records = []
    for i, row in state_changes.iterrows():
        t_start = float(row["timestamp"])
        t_end = float(state_changes.iloc[i + 1]["timestamp"]) if i + 1 < len(state_changes) else t_max
        state_val = row["state"]
        band_records.append({
            "t_start": t_start,
            "t_end": t_end,
            "state": state_val,
            "color": _STATE_COLOR.get(state_val, "#cccccc"),
        })
    band_df = pd.DataFrame(band_records)

    bands = (
        alt.Chart(band_df)
        .mark_rect(opacity=0.12)
        .encode(
            x=alt.X("t_start:Q", scale=alt.Scale(domain=[t_min, t_max])),
            x2="t_end:Q",
            color=alt.Color(
                "state:N",
                scale=alt.Scale(
                    domain=list(_STATE_COLOR.keys()),
                    range=list(_STATE_COLOR.values()),
                ),
                legend=alt.Legend(title="State"),
            ),
            tooltip=["state:N"],
        )
    )

    # Threshold lines
    thresh_df = pd.DataFrame(_THRESHOLD_LINES)
    threshold_rules = (
        alt.Chart(thresh_df)
        .mark_rule(strokeDash=[4, 4], opacity=0.7)
        .encode(
            y=alt.Y("depth:Q"),
            color=alt.Color("color:N", scale=None),
            tooltip=["label:N", "depth:Q"],
        )
    )

    # Main depth line
    line = (
        alt.Chart(df[df["marker_type"] == "normal"])
        .mark_line(color="#0072B2", strokeWidth=2)
        .encode(
            x=alt.X("timestamp:Q", title="Time (s)"),
            y=alt.Y("depth_cm:Q", title="Depth (cm)"),
            tooltip=["timestamp:Q", "depth_cm:Q", "state:N", "severity_class:N"],
        )
    )

    # Markers for invalid samples
    invalid_pts = (
        alt.Chart(df[df["marker_type"] == "invalid"])
        .mark_point(shape="cross", color="red", size=60)
        .encode(
            x="timestamp:Q",
            y="depth_cm:Q",
            tooltip=["timestamp:Q", alt.Tooltip("depth_cm:Q", title="depth (invalid)")],
        )
    )

    # Markers for low-confidence samples
    lowconf_pts = (
        alt.Chart(df[df["marker_type"] == "low-conf"])
        .mark_point(shape="triangle-up", color="#E69F00", size=60)
        .encode(
            x="timestamp:Q",
            y="depth_cm:Q",
            tooltip=["timestamp:Q", "depth_cm:Q", alt.Tooltip("confidence:Q", format=".2f")],
        )
    )

    # Alert markers (only if we have them in df)
    alert_data = alerts_up_to(db_path, run_id, camera_id, cursor_ts)
    if alert_data:
        alert_df = pd.DataFrame(alert_data)
        alert_df["depth_cm"] = pd.to_numeric(alert_df["depth_cm"], errors="coerce").fillna(0.0)
        alert_marks = (
            alt.Chart(alert_df)
            .mark_point(shape="diamond", color="#CC79A7", size=120, filled=True)
            .encode(
                x=alt.X("timestamp:Q"),
                y=alt.Y("depth_cm:Q"),
                tooltip=["timestamp:Q", "transition:N", "severity_class:N"],
            )
        )
        chart = alt.layer(bands, threshold_rules, line, invalid_pts, lowconf_pts, alert_marks)
    else:
        chart = alt.layer(bands, threshold_rules, line, invalid_pts, lowconf_pts)

    chart = chart.properties(height=350).interactive()
    st.altair_chart(chart, use_container_width=True)

    st.caption(
        "Bands = state; dashed lines = thresholds; "
        "× = invalid; △ = low-conf; ◆ = alert"
    )


# ---------------------------------------------------------------------------
# Panel 3: Annotated snapshot
# ---------------------------------------------------------------------------

def _panel_snapshot(
    db_path: Path, run_id: str, camera_id: str, cursor_ts: float
) -> None:
    st.subheader("Latest Annotated Frame")
    snap = nearest_snapshot(db_path, run_id, camera_id, cursor_ts)
    if snap is None:
        st.info("No snapshot available at this cursor position.")
        return
    p = Path(snap)
    if not p.is_file():
        st.warning(f"Snapshot file missing: {snap}")
        return
    st.image(str(p), caption=f"Snapshot @ {cursor_ts:.0f} s — {camera_id}", use_column_width=True)


# ---------------------------------------------------------------------------
# Panel 4: Alert log
# ---------------------------------------------------------------------------

def _panel_alerts(
    db_path: Path, run_id: str, camera_id: str | None, cursor_ts: float
) -> None:
    st.subheader("Alert Log")
    data = alerts_up_to(db_path, run_id, camera_id, cursor_ts)
    if not data:
        st.info("No alerts up to this cursor position.")
        return

    df = pd.DataFrame(data)
    # Replace arrow character for readability in table
    if "transition" in df.columns:
        df["transition"] = df["transition"].str.replace("→", "->", regex=False)
    cols = [c for c in ["timestamp", "camera_id", "transition", "severity_class", "depth_cm"] if c in df.columns]
    st.dataframe(df[cols].sort_values("timestamp", ascending=False), use_container_width=True)


# ---------------------------------------------------------------------------
# Panel 5: Camera health
# ---------------------------------------------------------------------------

def _panel_health(
    db_path: Path, run_id: str, cameras: list[str], cursor_ts: float
) -> None:
    st.subheader("Camera Health")
    if not cameras:
        st.info("No cameras found.")
        return

    for cam_id in cameras:
        current = camera_health_status(db_path, run_id, cam_id, cursor_ts)
        last_ok = last_healthy_timestamp(db_path, run_id, cam_id, cursor_ts)
        history = camera_health_history(db_path, run_id, cam_id, cursor_ts)
        fault_count = sum(
            1 for h in history
            if h.get("frozen") or h.get("black") or h.get("obstructed") or h.get("offline")
        )

        with st.expander(f"Health — {cam_id}", expanded=True):
            if current is None:
                st.info("No health data.")
                continue

            flags = []
            if current.get("frozen"):
                flags.append("🔴 Frozen")
            if current.get("black"):
                flags.append("🔴 Black frame")
            if current.get("obstructed"):
                flags.append("🔴 Obstructed")
            if current.get("offline"):
                flags.append("🔴 Offline")

            if flags:
                st.error("  |  ".join(flags))
            else:
                st.success("✓ Healthy")

            col1, col2 = st.columns(2)
            col1.metric("Last healthy timestamp", _fmt_ts(last_ok))
            col2.metric("Fault frames in run", fault_count)

            if current.get("detail"):
                st.caption(f"Detail: {current['detail']}")


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

def _banner(db_path: Path, run_id: str) -> None:
    meta = run_metadata(db_path, run_id)
    if not meta:
        return

    if meta.get("synthetic"):
        st.warning(
            "**SYNTHETIC DATA — pipeline validation only.** "
            "These results must not be cited in any paper or evaluation."
        )

    cols = st.columns(4)
    cols[0].markdown(f"**Run ID:** {run_id}")
    cols[1].markdown(f"**Config hash:** {(meta.get('config_hash') or '—')[:10]}")
    cols[2].markdown(f"**Git commit:** {meta.get('git_commit') or '—'}")
    cols[3].markdown(f"**Model:** {meta.get('model_version') or '—'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("FloodCam Monitor")

    mode = st.sidebar.radio(
        "Mode", ["Upload video (live)", "Recorded runs (replay)"], key="mode",
    )
    if mode == "Upload video (live)":
        from upload_view import render as render_upload

        render_upload()
        return

    db_path, run_id, selected_cam, cursor_ts, live_mode = _sidebar()

    if run_id is None:
        st.info("Select a run from the sidebar to begin.")
        return

    _banner(db_path, run_id)
    st.divider()

    # Camera list (filtered if a single camera is selected)
    all_cameras = cameras_for_run(db_path, run_id)
    cameras = [selected_cam] if selected_cam else all_cameras

    if not cameras:
        st.warning("No events found for this run.")
        return

    if live_mode:
        # Auto-poll for new rows every 5 seconds
        @st.fragment(run_every=5)
        def _live_panels():
            live_cursor = float("inf")
            _panel_status(db_path, run_id, cameras, live_cursor)
            st.divider()
            for cam in cameras:
                _panel_depth_chart(db_path, run_id, cam, live_cursor)
                _panel_snapshot(db_path, run_id, cam, live_cursor)
            _panel_alerts(db_path, run_id, selected_cam, live_cursor)
            st.divider()
            _panel_health(db_path, run_id, cameras, live_cursor)
        _live_panels()
    else:
        _panel_status(db_path, run_id, cameras, cursor_ts)
        st.divider()
        for cam in cameras:
            _panel_depth_chart(db_path, run_id, cam, cursor_ts)
            _panel_snapshot(db_path, run_id, cam, cursor_ts)
        _panel_alerts(db_path, run_id, selected_cam, cursor_ts)
        st.divider()
        _panel_health(db_path, run_id, cameras, cursor_ts)


if __name__ == "__main__":
    main()
