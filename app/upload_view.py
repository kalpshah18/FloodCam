"""Upload-a-video view: calibrate on a frame, then watch flood levels update live.

Rendered by app/dashboard.py in "Upload video (live)" mode. All numbers shown come
from floodcam.live.analyze_video; nothing is hand-set here.
"""

from __future__ import annotations

import hashlib
import tempfile
import time
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from floodcam.calib.linear import CalibError
from floodcam.depth.estimate import estimate_from_mask
from floodcam.io.video import VideoError, probe_video, read_frame
from floodcam.live import (
    DRY_CONFIDENCE,
    HSV_PRESETS,
    LiveSettings,
    analyze_video,
    annotate_frame,
    build_camera_config,
    build_experiment_config,
)
from floodcam.seg.backend import HsvBackend
from floodcam.store.db import EventStore

ROOT = Path(__file__).parent.parent
UPLOAD_DB = ROOT / "results" / "uploads.db"

SEVERITY_COLOR = {
    "Normal": "#009E73", "Advisory": "#56B4E9", "Warning": "#E69F00",
    "Severe": "#D55E00", "Critical": "#CC79A7",
}
STATE_COLOR = {
    "Normal": "#009E73", "Pending": "#B8A800", "Flood": "#D55E00", "Recovering": "#56B4E9",
}
SPEEDS = {"Real time (1x)": 1.0, "5x": 5.0, "20x": 20.0, "Max speed": 0.0}

_SS = st.session_state


# --------------------------------------------------------------------------- helpers

def _badge(text: str, color: str) -> str:
    return (
        f"<span style='background:{color};color:#000;padding:2px 10px;border-radius:4px;"
        f"font-weight:600'>{text}</span>"
    )


def _fmt_t(ts: float) -> str:
    m, s = divmod(int(ts), 60)
    return f"{m:02d}:{s:02d}"


def _save_upload(uploaded) -> tuple[Path, str]:
    data = uploaded.getvalue()
    vid = hashlib.sha256(data).hexdigest()[:12]
    d = Path(tempfile.gettempdir()) / "floodcam_uploads"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{vid}{Path(uploaded.name).suffix.lower() or '.mp4'}"
    if not path.exists():
        path.write_bytes(data)
    return path, vid


def _segformer_ckpt() -> str | None:
    import os

    d = os.environ.get("FLOODCAM_SEG_CKPT")
    if not d:
        return None
    try:
        import torch  # noqa: F401

        from floodcam.seg.segformer import resolve_ckpt_dir

        return str(resolve_ckpt_dir(d))
    except Exception:
        return None


def _depth_chart(rows: list[dict], trigger: float, clear: float) -> alt.Chart | None:
    if not rows:
        return None
    df = pd.DataFrame(rows)
    base = alt.Chart(df).encode(x=alt.X("t:Q", title="video time (s)"))
    y = alt.Y("depth:Q", title="depth (cm)", scale=alt.Scale(domainMin=0, nice=True))
    filt = base.mark_line(color="#1f77b4", strokeWidth=2.5).encode(y=y)
    raw = base.mark_circle(color="#888888", size=18, opacity=0.6).encode(
        y=alt.Y("raw:Q"), tooltip=["t", "raw", "depth", "state", "severity"]
    )
    rules = pd.DataFrame(
        {"y": [3.0, trigger, clear], "label": ["3 cm flood threshold", f"trigger {trigger} cm", f"clear {clear} cm"]}
    )
    rule = alt.Chart(rules).mark_rule(strokeDash=[5, 4]).encode(
        y="y:Q", color=alt.Color("label:N", legend=alt.Legend(title=None, orient="bottom"))
    )
    return (raw + filt + rule).properties(height=300)


def _draw_readout(ph: dict, rows: list[dict], cfg_trigger: float, cfg_clear: float, img: np.ndarray | None) -> None:
    if not rows:
        return
    r = rows[-1]
    with ph["metrics"].container():
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Water depth", f"{r['depth']:.1f} cm")
        c2.markdown("**Severity**<br>" + _badge(r["severity"], SEVERITY_COLOR[r["severity"]]), unsafe_allow_html=True)
        c3.markdown("**State**<br>" + _badge(r["state"], STATE_COLOR[r["state"]]), unsafe_allow_html=True)
        rate = r["rate"]
        c4.metric("Rate (cm/10 min)", "n/a" if rate is None else f"{rate:+.0f}")
        c5.metric("Confidence", f"{r['conf']:.2f}", help=f"{DRY_CONFIDENCE} = dry reading (no water at the pole).")
    if img is not None:
        ph["frame"].image(img, caption=f"t = {_fmt_t(r['t'])}  (blue band = water mask, blue line = waterline)",
                          use_container_width=True)
    chart = _depth_chart(rows, cfg_trigger, cfg_clear)
    if chart is not None:
        ph["chart"].altair_chart(chart, use_container_width=True)


def _draw_alerts(ph: dict, alerts: list[dict]) -> None:
    with ph["alerts"].container():
        st.subheader("Alert log")
        if alerts:
            st.dataframe(pd.DataFrame(alerts), hide_index=True, use_container_width=True)
        else:
            st.caption("No state transitions yet.")


# --------------------------------------------------------------------------- calibration

def _calibration_panel(video: Path, vid: str, demo_defaults: LiveSettings | None, max_width: int) -> tuple[LiveSettings | None, np.ndarray | None]:
    info = probe_video(video)
    st.subheader("1. Calibrate on a frame")
    st.caption(
        f"{info.width}x{info.height} px, {info.fps:.1f} fps"
        + (f", {info.duration_s:.0f} s" if info.duration_s else "")
        + ". Mark the reference pole, then check the test reading on a frame where you know the depth."
    )
    t_prev = 0.0
    if info.duration_s > 1:
        t_prev = st.slider("Preview frame time (s)", 0.0, float(info.duration_s - 0.5), 35.0 if demo_defaults else 0.0, 0.5, key=f"tprev_{vid}")
    frame = read_frame(video, t_prev, max_width)
    H, W = frame.shape[:2]

    d = demo_defaults
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown("**Reference pole**")
        pole_x = st.slider("Pole column (px)", 0, W - 1, int(d.pole_x_px) if d else W // 2, key=f"px_{vid}")
        ground = st.slider("Ground-contact row (px) - where the pole meets the ground, 0 cm", 0, H - 1,
                           int(round(d.ground_row)) if d else int(H * 0.8), key=f"gr_{vid}")
        mark = st.slider("Second mark row (px) - a known height on the pole", 0, H - 1,
                         int(round(d.mark_row)) if d else int(H * 0.35), key=f"mr_{vid}")
        mark_h = st.number_input("Height of the second mark above ground (cm)", 1.0, 1000.0,
                                 float(d.mark_height_cm) if d else 50.0, 1.0, key=f"mh_{vid}")
        pole_w = st.slider("Pole width (px)", 2, 40, int(d.pole_width_px) if d else 6, key=f"pw_{vid}")
        with st.expander("Region of interest (ignore everything outside)"):
            x0, x1 = st.slider("Columns", 0, W, (0, W), key=f"roix_{vid}")
            y0, y1 = st.slider("Rows", 0, H, (0, H), key=f"roiy_{vid}")
    with col_b:
        st.markdown("**Water segmentation (HSV colour threshold)**")
        preset = st.selectbox("Preset", list(HSV_PRESETS) + ["Custom"], key=f"pre_{vid}",
                              index=len(HSV_PRESETS) if d else 0)
        lo, hi = (list(d.hsv_lower), list(d.hsv_upper)) if (d and preset == "Custom") else (
            (list(HSV_PRESETS[preset][0]), list(HSV_PRESETS[preset][1])) if preset != "Custom" else ([85, 40, 40], [130, 255, 255]))
        hh = st.slider("Hue range", 0, 179, (lo[0], hi[0]), key=f"hue_{vid}_{preset}")
        ss = st.slider("Saturation range", 0, 255, (lo[1], hi[1]), key=f"sat_{vid}_{preset}")
        vv = st.slider("Value range", 0, 255, (lo[2], hi[2]), key=f"val_{vid}_{preset}")
        mo = st.slider("Noise removal (open, px)", 0, 15, d.morph_open if d else 3, key=f"mo_{vid}")
        mc = st.slider("Gap filling (close, px)", 0, 25, d.morph_close if d else 5, key=f"mc_{vid}")

    settings = LiveSettings(
        pole_x_px=pole_x, ground_row=float(ground), mark_row=float(mark), mark_height_cm=float(mark_h),
        pole_width_px=pole_w, roi=(x0, y0, max(1, x1 - x0), max(1, y1 - y0)),
        hsv_lower=[hh[0], ss[0], vv[0]], hsv_upper=[hh[1], ss[1], vv[1]],
        morph_open=mo, morph_close=mc, max_width=max_width,
    )

    try:
        cfg = build_camera_config(settings, frame.shape)
    except CalibError as exc:
        st.error(f"Calibration invalid: {exc}")
        st.image(frame, use_container_width=True)
        return None, frame

    backend = HsvBackend(build_experiment_config(settings), cfg)
    mask = backend.predict_mask(frame)
    est = estimate_from_mask(mask, cfg)
    show_mask = st.toggle("Show water mask on preview", value=True, key=f"sm_{vid}")
    overlay = annotate_frame(frame, cfg, mask, est.waterline_row if est.valid else None,
                             est.depth_cm if est.valid else None, "preview", show_mask)
    # second-mark guide line
    import cv2
    cv2.line(overlay, (0, int(mark)), (W - 1, int(mark)), (255, 0, 255), 1)
    cv2.putText(overlay, f"mark {mark_h:g} cm", (4, max(12, int(mark) - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1, cv2.LINE_AA)
    st.image(overlay, caption="Green = ground (0 cm), magenta = second mark, white box = pole, "
                              "yellow box = ROI, blue = water mask, blue line = detected waterline",
             use_container_width=True)
    s_cm = cfg.calibration.scale_cm_per_px or 0.0
    if est.valid:
        msg = f"Test reading on this frame: **{est.depth_cm:.1f} cm** (confidence {est.confidence:.2f}, {s_cm:.3f} cm/px)."
    else:
        msg = {
            "no_water": "No water detected beside the pole on this frame (reads as dry, 0 cm).",
            "fully_submerged": "Water reaches above the second mark: raise the mark or use a taller reference.",
            "broken_mask": "Mask is fragmented beside the pole: adjust HSV range or gap filling.",
        }.get(est.reason, f"Reading invalid ({est.reason}).")
        msg += f" ({s_cm:.3f} cm/px)"
    st.info(msg)
    if s_cm > 0.5:
        st.warning(f"Scale is {s_cm:.2f} cm/px: coarse for detecting a 3 cm threshold (about 6 px). "
                   "Reduce distance or raise resolution if possible.")
    return settings, frame


# --------------------------------------------------------------------------- main view

def render() -> None:
    st.header("Upload a video and watch the flood level")
    st.caption("Fixed camera, a reference pole of known size in view, and a manual calibration. "
               "Depth comes from the waterline position on the pole; see docs/CALIBRATION.md.")

    with st.sidebar:
        st.header("Upload mode")
        uploaded = st.file_uploader("Video file", type=["mp4", "mov", "avi", "mkv", "webm", "m4v"])
        use_demo = st.button("Load synthetic demo video", help="Rendered pole + water at known depth. Synthetic: demo only.")
        speed_label = st.select_slider("Playback speed", list(SPEEDS), value="5x")
        sample_fps = st.select_slider("Analysis rate (frames/s)", [0.25, 0.5, 1.0, 2.0, 5.0], value=1.0)
        max_width = st.select_slider("Working width (px)", [480, 640, 960, 1280], value=960)
        with st.expander("Temporal logic (seconds)"):
            st.caption("Defaults are scaled for short clips. For real events use minutes "
                       "(persistence 300 s, trend window 600 s).")
            persistence = st.number_input("Persistence before Flood (s)", 1.0, 3600.0, 10.0, 1.0)
            trend = st.number_input("Trend window (s)", 4.0, 3600.0, 20.0, 1.0)
            trigger = st.number_input("Trigger (cm)", 0.5, 100.0, 3.5, 0.1)
            clear = st.number_input("Clear (cm)", 0.1, 100.0, 2.5, 0.1)
            min_conf = st.slider("Min confidence", 0.0, 1.0, 0.3, 0.05)
            rate_esc = st.checkbox("Escalate severity on fast rise (5 cm/10 min)", value=False,
                                   help="Off by default: on short clips depth noise looks like a large rate.")
        save_log = st.checkbox("Save run to event log (results/uploads.db)", value=True)
        seg_ckpt = _segformer_ckpt()
        use_seg = st.checkbox("Use SegFormer checkpoint", value=False, disabled=seg_ckpt is None,
                              help="Set FLOODCAM_SEG_CKPT to a trained checkpoint (needs torch).")

    demo_defaults: LiveSettings | None = None
    if use_demo:
        from floodcam.synth.video import render_demo_video

        d = Path(tempfile.gettempdir()) / "floodcam_uploads"
        path, demo_settings, _fn = render_demo_video(d / "synthetic_demo.mp4")
        _SS["video"] = (str(path), "demo", True, demo_settings)
        _SS.pop("upload_result", None)
    elif uploaded is not None:
        try:
            path, vid = _save_upload(uploaded)
        except OSError as exc:
            st.error(f"Could not store upload: {exc}")
            return
        if _SS.get("video", (None, None))[1] != vid:
            _SS.pop("upload_result", None)
        _SS["video"] = (str(path), vid, False, None)

    if "video" not in _SS:
        st.info("Upload a video in the sidebar, or load the synthetic demo video.")
        with st.expander("What makes a good input video"):
            st.markdown(
                "- Camera fixed (tripod or wall mount); no panning or zoom.\n"
                "- A vertical reference object with a known height (marked pole, ruler, curb) in view, base visible.\n"
                "- Water visible on the ground next to the pole.\n"
                "- Avoid people, vehicles passing the pole, strong reflections, and night footage "
                "(these are known failure modes)."
            )
        return

    video_path, vid, is_demo, demo_defaults = _SS["video"]
    video = Path(video_path)
    if not video.exists():
        _SS.pop("video")
        st.warning("The stored video is gone; upload it again.")
        return
    if is_demo:
        st.warning("Synthetic demo video: results are pipeline validation only, never reportable.")

    try:
        settings, _ = _calibration_panel(video, vid, demo_defaults, max_width)
    except VideoError as exc:
        st.error(str(exc))
        return
    if settings is None:
        return
    settings.sample_fps = float(sample_fps)
    settings.persistence_s = float(persistence)
    settings.trend_window_s = float(trend)
    settings.trigger_cm = float(trigger)
    settings.clear_cm = float(clear)
    settings.min_confidence = float(min_conf)
    settings.rate_escalation = bool(rate_esc)
    if clear >= trigger:
        st.error("Clear threshold must be below the trigger threshold.")
        return

    st.divider()
    st.subheader("2. Live analysis")
    start = st.button("Start analysis", type="primary")

    ph = {k: st.empty() for k in ("metrics", "progress")}
    cols = st.columns([1, 1])
    ph["frame"] = cols[0].empty()
    ph["chart"] = cols[1].empty()
    ph["alerts"] = st.empty()

    if start:
        _run_live(video, settings, ph, SPEEDS[speed_label], save_log, is_demo, seg_ckpt if use_seg else None, trigger, clear)
    elif "upload_result" in _SS:
        res = _SS["upload_result"]
        _draw_readout(ph, res["rows"], trigger, clear, res["last_img"])
        _draw_alerts(ph, res["alerts"])
        _results_footer(res)


def _run_live(video, settings, ph, speed, save_log, synthetic, seg_ckpt, trigger, clear) -> None:
    backend = None
    if seg_ckpt:
        from floodcam.seg.segformer import SegFormerBackend

        backend = SegFormerBackend(build_experiment_config(settings), seg_ckpt)

    info = probe_video(video)
    total = max(1, int(info.duration_s * settings.sample_fps)) if info.duration_s else None
    store = EventStore(UPLOAD_DB) if save_log else None
    run_id = f"upload_{int(time.time())}"
    snap_dir = ROOT / "results" / "runs" / run_id / "snapshots" if save_log else None

    rows: list[dict] = []
    alerts: list[dict] = []
    img = None
    t_wall0 = time.time()
    bar = ph["progress"].progress(0.0, text="Starting...")
    try:
        for f in analyze_video(video, settings, backend=backend, store=store, run_id=run_id,
                               camera_id="upload", synthetic=synthetic, snapshot_dir=snap_dir):
            rows.append({
                "t": f.timestamp_s, "depth": f.depth_cm, "raw": f.raw_depth_cm, "state": f.state,
                "severity": f.severity, "rate": f.rate_cm_per_10min, "conf": f.confidence,
                "valid": f.valid, "reason": f.reason, "health": f.health_detail,
            })
            img = f.annotated_rgb
            if f.transition:
                alerts.append({"time": _fmt_t(f.timestamp_s), "transition": f.transition,
                               "severity": f.severity, "depth_cm": round(f.depth_cm, 1)})
            _draw_readout(ph, rows, trigger, clear, img)
            _draw_alerts(ph, alerts)
            if total:
                bar.progress(min(1.0, (f.index + 1) / total), text=f"Analysing t = {_fmt_t(f.timestamp_s)}")
            if speed > 0:
                lag = f.timestamp_s / speed - (time.time() - t_wall0)
                if lag > 0:
                    time.sleep(min(lag, 5.0))
    except (VideoError, CalibError) as exc:
        st.error(str(exc))
    finally:
        if store is not None:
            store.close()
    bar.progress(1.0, text=f"Done: {len(rows)} frames analysed")
    if not rows:
        st.warning("No frames were decoded from this video.")
        return
    _SS["upload_result"] = {"rows": rows, "alerts": alerts, "last_img": img,
                            "run_id": run_id if save_log else None}
    _results_footer(_SS["upload_result"])


def _results_footer(res: dict) -> None:
    df = pd.DataFrame(res["rows"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Peak depth", f"{df['depth'].max():.1f} cm")
    c2.metric("Time at >= 3 cm", f"{(df['depth'] >= 3.0).sum()} frames")
    low = int((~df["valid"]).sum())
    c3.metric("Excluded frames", f"{low} of {len(df)}", help="Invalid or low-confidence readings are not used by the filter.")
    st.download_button("Download per-frame levels (CSV)", df.to_csv(index=False).encode(),
                       file_name="flood_levels.csv", mime="text/csv")
    if res.get("run_id"):
        st.caption(f"Saved as run `{res['run_id']}` in results/uploads.db. "
                   "Open it from 'Recorded runs (replay)' with that DB path.")
