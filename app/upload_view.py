"""Upload-a-video view: click to mark the reference pole, then watch flood levels live.

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
from floodcam.io.export import AnnotatedVideoWriter
from floodcam.io.stabilise import Stabiliser
from floodcam.io.video import VideoError, probe_video, read_frame, read_reference
from floodcam.live import (
    DRY_CONFIDENCE,
    HSV_PRESETS,
    LiveSettings,
    analyze_video,
    annotate_frame,
    build_backend,
    build_camera_config,
    build_experiment_config,
)
from floodcam.store.db import EventStore

try:  # optional click-to-mark component; sliders always work without it
    from streamlit_image_coordinates import streamlit_image_coordinates
except ImportError:  # pragma: no cover
    streamlit_image_coordinates = None

ROOT = Path(__file__).parent.parent
UPLOAD_DB = ROOT / "results" / "uploads.db"
TMP = Path(tempfile.gettempdir()) / "floodcam_uploads"

SEVERITY_COLOR = {
    "Normal": "#009E73", "Advisory": "#56B4E9", "Warning": "#E69F00",
    "Severe": "#D55E00", "Critical": "#CC79A7",
}
STATE_COLOR = {
    "Normal": "#009E73", "Pending": "#B8A800", "Flood": "#D55E00", "Recovering": "#56B4E9",
}
SPEEDS = {"Real time (1x)": 1.0, "5x": 5.0, "20x": 20.0, "Max speed": 0.0}
METHOD_CHANGE = "Change vs dry reference (recommended for real footage)"
METHOD_COLOUR = "Water colour (HSV)"
DISPLAY_W = 760

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
    TMP.mkdir(parents=True, exist_ok=True)
    path = TMP / f"{vid}{Path(uploaded.name).suffix.lower() or '.mp4'}"
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


@st.cache_data(show_spinner=False, max_entries=8)
def _cached_reference(path: str, t_s: float, max_width: int) -> np.ndarray:
    return read_reference(path, t_s, 5, max_width)


@st.cache_data(show_spinner=False, max_entries=16)
def _cached_frame(path: str, t_s: float, max_width: int) -> np.ndarray:
    return read_frame(path, t_s, max_width)


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
    rules = pd.DataFrame({
        "y": [3.0, trigger, clear],
        "label": ["3 cm flood threshold", f"trigger {trigger:g} cm", f"clear {clear:g} cm"],
    })
    rule = alt.Chart(rules).mark_rule(strokeDash=[5, 4]).encode(
        y="y:Q", color=alt.Color("label:N", legend=alt.Legend(title=None, orient="bottom"))
    )
    return (raw + filt + rule).properties(height=300)


def _draw_readout(ph: dict, rows: list[dict], trigger: float, clear: float,
                  img: np.ndarray | None) -> None:
    if not rows:
        return
    r = rows[-1]
    with ph["metrics"].container():
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Water depth", f"{r['depth']:.1f} cm")
        c2.markdown("**Severity**<br>" + _badge(r["severity"], SEVERITY_COLOR[r["severity"]]),
                    unsafe_allow_html=True)
        c3.markdown("**State**<br>" + _badge(r["state"], STATE_COLOR[r["state"]]),
                    unsafe_allow_html=True)
        rate = r["rate"]
        c4.metric("Rate (cm/10 min)", "n/a" if (rate is None or r["depth"] < 1.0) else f"{rate:+.0f}")
        c5.metric("Confidence", f"{r['conf']:.2f}",
                  help=f"{DRY_CONFIDENCE} = dry reading (no water at the pole).")
    if img is not None:
        ph["frame"].image(img, caption=f"t = {_fmt_t(r['t'])}", use_container_width=True)
    chart = _depth_chart(rows, trigger, clear)
    if chart is not None:
        ph["chart"].altair_chart(chart, use_container_width=True)


def _draw_alerts(ph: dict, alerts: list[dict]) -> None:
    with ph["alerts"].container():
        st.subheader("Alert log")
        if alerts:
            st.dataframe(pd.DataFrame(alerts), hide_index=True, use_container_width=True)
        else:
            st.caption("No state transitions yet.")


# --------------------------------------------------------------------------- step 1+2

def _calibration_panel(video: Path, vid: str, demo: LiveSettings | None,
                       max_width: int) -> LiveSettings | None:
    info = probe_video(video)
    st.caption(
        f"{info.width}x{info.height} px, {info.fps:.1f} fps"
        + (f", {info.duration_s:.0f} s" if info.duration_s else "")
    )

    t_prev = 0.0
    if info.duration_s > 1:
        t_prev = st.slider("Frame to work on (s)", 0.0, float(info.duration_s - 0.5),
                           35.0 if demo else 0.0, 0.5, key=f"tprev_{vid}")
    frame = _cached_frame(str(video), t_prev, max_width)
    H, W = frame.shape[:2]

    cal = _SS.setdefault(f"cal_{vid}", {
        "x": int(demo.pole_x_px) if demo else W // 2,
        "ground": int(round(demo.ground_row)) if demo else int(H * 0.80),
        "mark": int(round(demo.mark_row)) if demo else int(H * 0.45),
        "mark_h": float(demo.mark_height_cm) if demo else 30.0,
        "width": int(demo.pole_width_px) if demo else 8,
        "target": "ground", "ver": 0, "set": demo is not None,
    })
    ver = cal["ver"]

    left, right = st.columns([5, 3])

    # ---- right column: controls (created before the image so the image always reflects them)
    with right:
        st.markdown("**Reference object**")
        st.caption("Click the image: first the point where the pole meets the ground "
                   "(0 cm), then a point at a known height on the pole.")
        labels = {"ground": "Next click: base of pole (0 cm)", "mark": "Next click: known height"}
        tgt = st.radio("Click target", list(labels), format_func=labels.get,
                       index=0 if cal["target"] == "ground" else 1, key=f"tgt_{vid}_{ver}",
                       label_visibility="collapsed")
        cal["target"] = tgt
        cal["mark_h"] = st.number_input("Height of the known point (cm)", 1.0, 1000.0,
                                        float(cal["mark_h"]), 1.0, key=f"mh_{vid}")
        with st.expander("Fine-tune positions"):
            cal["x"] = st.slider("Pole column (px)", 0, W - 1, int(cal["x"]), key=f"px_{vid}_{ver}")
            cal["ground"] = st.slider("Ground row (px)", 0, H - 1, int(cal["ground"]),
                                      key=f"gr_{vid}_{ver}")
            cal["mark"] = st.slider("Known-height row (px)", 0, H - 1, int(cal["mark"]),
                                    key=f"mr_{vid}_{ver}")
            cal["width"] = st.slider("Gauge width (px)", 2, 60, int(cal["width"]),
                                     key=f"pw_{vid}")
            x0, x1 = st.slider("ROI columns", 0, W, (0, W), key=f"roix_{vid}")
            y0, y1 = st.slider("ROI rows", 0, H, (0, H), key=f"roiy_{vid}")

        st.markdown("**Water detection**")
        method = st.radio("Method", [METHOD_CHANGE, METHOD_COLOUR],
                          index=1 if (demo and demo.mask_source == "colour") else 0,
                          key=f"meth_{vid}", label_visibility="collapsed")
        s = LiveSettings(
            pole_x_px=int(cal["x"]), ground_row=float(cal["ground"]),
            mark_row=float(cal["mark"]), mark_height_cm=float(cal["mark_h"]),
            pole_width_px=int(cal["width"]), roi=(x0, y0, max(1, x1 - x0), max(1, y1 - y0)),
            max_width=max_width,
        )
        if method == METHOD_CHANGE:
            s.mask_source, s.estimator = "change", "gauge"
            st.caption("Needs a dry reference: pick a moment before water reaches the pole.")
            s.ref_time_s = st.slider("Dry reference at (s)", 0.0,
                                     float(max(info.duration_s - 1, 1.0)), 0.0, 0.5,
                                     key=f"ref_{vid}")
            s.diff_threshold = float(st.slider(
                "Sensitivity (lower = more sensitive)", 5, 60, 20, key=f"thr_{vid}",
                help="Lab colour change needed to count a pixel as wet/changed. Raise it if "
                     "shadows or noise are flagged on the dry pole; lower it if water is missed."))
            s.stabilise = st.checkbox("Compensate small camera shake", value=True,
                                      key=f"stab_{vid}",
                                      help="Uses image texture above the pole (buildings, trees).")
            s.morph_open, s.morph_close = 3, 7
        else:
            s.mask_source = "colour"
            s.estimator = "flank" if demo else "gauge"
            s.stabilise = False
            preset = st.selectbox("Preset", list(HSV_PRESETS) + ["Custom"], key=f"pre_{vid}",
                                  index=len(HSV_PRESETS) if demo else 0)
            lo, hi = (list(demo.hsv_lower), list(demo.hsv_upper)) if (demo and preset == "Custom") \
                else (list(HSV_PRESETS[preset][0]), list(HSV_PRESETS[preset][1])) \
                if preset != "Custom" else ([85, 40, 40], [130, 255, 255])
            hh = st.slider("Hue", 0, 179, (lo[0], hi[0]), key=f"hue_{vid}_{preset}")
            ss = st.slider("Saturation", 0, 255, (lo[1], hi[1]), key=f"sat_{vid}_{preset}")
            vv = st.slider("Value", 0, 255, (lo[2], hi[2]), key=f"val_{vid}_{preset}")
            s.hsv_lower, s.hsv_upper = [hh[0], ss[0], vv[0]], [hh[1], ss[1], vv[1]]
            s.morph_open, s.morph_close = (0, 0) if demo else (3, 5)
            st.caption("Colour thresholds cannot tell water from sky or asphalt. "
                       "Prefer the change method on real footage.")

    # ---- left column: clickable preview
    with left:
        try:
            cfg = build_camera_config(s, frame.shape)
        except CalibError as exc:
            st.error(f"Marks invalid: {exc}")
            st.image(frame, use_container_width=True)
            return None

        ref = None
        if s.mask_source == "change":
            ref = _cached_reference(str(video), s.ref_time_s, max_width)
        view = frame
        if s.stabilise and ref is not None and s.mark_row >= 24:
            view = Stabiliser(ref, rows=(0, int(s.mark_row))).align(frame)
        backend = build_backend(s, cfg, ref)
        mask = backend.predict_mask(view)
        roi = cfg.roi
        roi_mask = np.zeros_like(mask)
        roi_mask[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w] = \
            mask[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w]
        est = estimate_from_mask(roi_mask, cfg, s.estimator)
        show_mask = st.toggle("Show detected water / change", value=True, key=f"sm_{vid}")
        overlay = annotate_frame(
            view, cfg, roi_mask, est.waterline_row if (est.valid and est.reason == "ok") else None,
            est.depth_cm if est.valid else None, "", show_mask,
        )
        if streamlit_image_coordinates is not None:
            disp_w = min(W, DISPLAY_W)
            click = streamlit_image_coordinates(overlay, width=disp_w, key=f"img_{vid}_{ver}")
            if click:
                k = W / float(click.get("width") or disp_w)
                cx, cy = int(click["x"] * k), int(click["y"] * k)
                if cal["target"] == "ground":
                    cal["x"], cal["ground"], cal["target"] = cx, cy, "mark"
                else:
                    cal["mark"], cal["target"] = cy, "ground"
                cal["set"] = True
                cal["ver"] += 1
                st.rerun()
        else:
            st.image(overlay, use_container_width=True)
        st.caption("White box = gauge strip on the pole, green = ground (0 cm), magenta = known "
                   "height, blue = detected change/water, red line = waterline, right bar = depth.")

    s_cm = cfg.calibration.scale_cm_per_px or 0.0
    if not cal["set"]:
        st.info("Click the base of the pole on the image to start.")
    elif est.valid:
        st.success(f"Test reading on this frame: {est.depth_cm:.1f} cm "
                   f"(confidence {est.confidence:.2f}, {s_cm:.2f} cm/px). If you know the true "
                   "depth here, check it matches.")
    else:
        st.info({
            "no_water": "No water/change at the base of the pole on this frame (reads dry, 0 cm).",
            "fully_submerged": "Water reaches the top of the gauge; set the known-height mark higher.",
            "broken_mask": "Mask is fragmented beside the pole; adjust the detection settings.",
        }.get(est.reason, f"Reading invalid ({est.reason})."))
    if s_cm > 0.6:
        st.warning(f"Scale is {s_cm:.2f} cm/px: coarse for a 3 cm threshold. "
                   "Film closer or at higher resolution if you can.")
    return s


# --------------------------------------------------------------------------- main view

def render() -> None:
    st.header("Flood level from video")
    st.caption("Fixed camera and a reference object of known height (marked pole, ruler, curb) "
               "in view. Depth is read where the water meets that object.")

    with st.sidebar:
        st.header("Upload mode")
        uploaded = st.file_uploader("Video file", type=["mp4", "mov", "avi", "mkv", "webm", "m4v"])
        use_demo = st.button("Load synthetic demo video",
                             help="Rendered pole + water at known depth, with camera shake and "
                                  "exposure drift. Synthetic: demo only.")
        speed_label = st.select_slider("Playback speed", list(SPEEDS), value="5x")
        sample_fps = st.select_slider("Analysis rate (frames/s)", [0.25, 0.5, 1.0, 2.0, 5.0],
                                      value=1.0)
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
                                   help="Off by default: on short clips depth noise looks like "
                                        "a large rate.")
        save_log = st.checkbox("Save run to event log (results/uploads.db)", value=True)
        make_video = st.checkbox("Export annotated video", value=True)
        seg_ckpt = _segformer_ckpt()
        use_seg = st.checkbox("Use SegFormer checkpoint", value=False, disabled=seg_ckpt is None,
                              help="Set FLOODCAM_SEG_CKPT to a trained checkpoint (needs torch).")

    demo_defaults: LiveSettings | None = None
    if use_demo:
        from floodcam.synth.video import render_demo_video

        path, demo_settings, _fn = render_demo_video(TMP / "synthetic_demo.mp4",
                                                     degrade=True, method="gauge")
        _SS["video"] = (str(path), "demo", True, demo_settings)
        _SS.pop("upload_result", None)
        _SS.pop("cal_demo", None)
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
        with st.expander("What makes a good input video", expanded=True):
            st.markdown(
                "- Camera fixed (tripod or wall mount). Small shake is compensated; panning and "
                "zooming are not.\n"
                "- A vertical reference object of known height in view with its base visible: "
                "a marked pole, ruler, curb or wall.\n"
                "- The first seconds show the scene dry (this is the reference the water is "
                "compared against), or you can pick another dry moment.\n"
                "- Avoid people or vehicles crossing the pole, strong reflections and night "
                "footage; these are known failure modes."
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

    st.subheader("1. Mark the reference object and check detection")
    try:
        settings = _calibration_panel(video, vid, demo_defaults, max_width)
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
    if not _SS.get(f"cal_{vid}", {}).get("set"):
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
        _run_live(video, settings, ph, SPEEDS[speed_label], save_log, make_video, is_demo,
                  seg_ckpt if use_seg else None, trigger, clear)
    elif "upload_result" in _SS:
        res = _SS["upload_result"]
        _draw_readout(ph, res["rows"], trigger, clear, res["last_img"])
        _draw_alerts(ph, res["alerts"])
        _results_footer(res)


def _run_live(video, settings, ph, speed, save_log, make_video, synthetic, seg_ckpt,
              trigger, clear) -> None:
    backend = None
    if seg_ckpt:
        from floodcam.seg.segformer import SegFormerBackend

        backend = SegFormerBackend(build_experiment_config(settings), seg_ckpt)
        settings.mask_source = "model"

    info = probe_video(video)
    total = max(1, int(info.duration_s * settings.sample_fps)) if info.duration_s else None
    store = EventStore(UPLOAD_DB) if save_log else None
    run_id = f"upload_{int(time.time())}"
    snap_dir = ROOT / "results" / "runs" / run_id / "snapshots" if save_log else None
    writer = AnnotatedVideoWriter(TMP / f"{run_id}_annotated.mp4", fps=6.0) if make_video else None

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
            if writer is not None:
                writer.write(img)
            if f.transition:
                alerts.append({"time": _fmt_t(f.timestamp_s), "transition": f.transition,
                               "severity": f.severity, "depth_cm": round(f.depth_cm, 1)})
            _draw_readout(ph, rows, trigger, clear, img)
            _draw_alerts(ph, alerts)
            if total:
                bar.progress(min(1.0, (f.index + 1) / total),
                             text=f"Analysing t = {_fmt_t(f.timestamp_s)}")
            if speed > 0:
                lag = f.timestamp_s / speed - (time.time() - t_wall0)
                if lag > 0:
                    time.sleep(min(lag, 5.0))
    except (VideoError, CalibError, ValueError) as exc:
        st.error(str(exc))
    finally:
        if store is not None:
            store.close()
    out_video = writer.close() if writer is not None else None
    bar.progress(1.0, text=f"Done: {len(rows)} frames analysed")
    if not rows:
        st.warning("No frames were decoded from this video.")
        return
    _SS["upload_result"] = {"rows": rows, "alerts": alerts, "last_img": img,
                            "run_id": run_id if save_log else None,
                            "video": str(out_video) if out_video else None,
                            "synthetic": synthetic}
    _results_footer(_SS["upload_result"])


def _results_footer(res: dict) -> None:
    df = pd.DataFrame(res["rows"])
    st.divider()
    st.subheader("Summary")
    peak = df.loc[df["depth"].idxmax()]
    above = df[df["depth"] >= 3.0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Peak depth", f"{peak['depth']:.1f} cm", help=f"at t = {_fmt_t(peak['t'])}")
    c2.metric("First reached 3 cm", _fmt_t(above["t"].iloc[0]) if len(above) else "never")
    dt = float(df["t"].diff().median()) if len(df) > 1 else 1.0
    c3.metric("Time at >= 3 cm", f"{len(above) * dt:.0f} s")
    c4.metric("Excluded readings", f"{int((~df['valid']).sum())} of {len(df)}",
              help="Invalid or low-confidence readings are not used by the temporal filter.")
    st.caption("Depths are estimates from the manual calibration marks; expect errors of a few cm "
               "on real footage. Flood/Recovering state is a demonstration of the logic, not a "
               "validated detector.")
    if res.get("video") and Path(res["video"]).exists():
        st.markdown("**Annotated video**")
        st.video(res["video"])
        st.download_button("Download annotated video", Path(res["video"]).read_bytes(),
                           file_name="flood_annotated.mp4", mime="video/mp4")
    st.download_button("Download per-frame levels (CSV)", df.to_csv(index=False).encode(),
                       file_name="flood_levels.csv", mime="text/csv")
    if res.get("run_id"):
        st.caption(f"Saved as run `{res['run_id']}` in results/uploads.db. "
                   "Open it from 'Recorded runs (replay)' with that DB path.")
