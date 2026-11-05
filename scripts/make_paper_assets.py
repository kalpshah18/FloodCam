#!/usr/bin/env python
"""Generate every figure, table and number used in paper/main.tex from results/real/** and data/.

Writes paper/numbers.tex (LaTeX macros), paper/tab_*.tex and paper/figs/*.pdf. The paper contains no
hand-typed metric: re-run this after any change in results/ (after scripts/real_report.py).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from floodcam.real.plate import SiteSpec  # noqa: E402

R = ROOT / "results" / "real"
P = ROOT / "paper"
SITE = "in_overland"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 6.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "pdf.fonttype": 42})
C = {"learned": "#1f77b4", "base": "#ff7f0e", "gauge": "black"}
NUM: dict[str, str] = {}


def num(name: str, value, fmt: str = "{:.2f}") -> None:
    NUM[name] = fmt.format(value) if not isinstance(value, str) else value


def load_run(run: str):
    return pd.read_csv(R / run / "predictions.csv", parse_dates=["time"]), json.loads((R / run / "metrics.json").read_text())


def final_run() -> str:
    return next(r for r in (f"real-final-{SITE}-f1-force", f"real-final-{SITE}-f1") if (R / r).exists())


# ----------------------------------------------------------------------------- data facts
def facts():
    man = [json.loads(x) for x in (ROOT / "data/usgs/manifest.jsonl").read_text().splitlines()]
    m = pd.DataFrame([r for r in man if r["site"] == SITE])
    z = np.load(ROOT / "data/usgs/strips" / f"{SITE}_f1.npz")
    split = json.loads((ROOT / "data/splits" / f"usgs_{SITE}.json").read_text())
    spec = SiteSpec.load(SITE)
    num("nFrames", len(m), "{:d}")
    num("nBlocks", len(split["blocks"]), "{:d}")
    num("nDevBlocks", len(split["dev_blocks"]), "{:d}")
    num("nLockedBlocks", len(split["locked_test_blocks"]), "{:d}")
    num("lockedBlocks", ", ".join(split["locked_test_blocks"]))
    num("nDevFrames", int(m.time.str[:7].isin(split["dev_blocks"]).sum()), "{:d}")
    num("nLockedFrames", int(m.time.str[:7].isin(split["locked_test_blocks"]).sum()), "{:d}")
    num("pctIR", 100 * m.is_ir.mean(), "{:.0f}")
    num("pctDry", 100 * (m.gage_cm <= 0.5).mean(), "{:.0f}")
    num("pctThreeBelowPlate", 100 * ((m.gage_cm >= 3) & (m.gage_cm < spec.bottom_ft * spec.ft_to_cm)).sum() / (m.gage_cm >= 3).sum(), "{:.0f}")
    num("pctOverThree", 100 * (m.gage_cm >= 3).mean(), "{:.0f}")
    num("gaugeMax", m.gage_cm.max(), "{:.0f}")
    num("yearFirst", m.time.min()[:4], "{}")
    num("yearLast", m.time.max()[:4], "{}")
    num("pctRegistered", 100 * z["reg_ok"].mean(), "{:.0f}")
    num("pctBorrowed", 100 * z["borrowed"].mean(), "{:.0f}")
    num("plateBottomCm", spec.bottom_ft * spec.ft_to_cm, "{:.1f}")
    num("plateTopCm", spec.top_ft * spec.ft_to_cm, "{:.0f}")
    from floodcam.real.experiment import GATE_HI, GATE_LO

    num("gateLo", GATE_LO, "{:.2f}")
    num("gateHi", GATE_HI, "{:.1f}")


# ----------------------------------------------------------------------------- main table
def metric_row(label, scope, meth, m):
    d = m[scope][meth]
    dd, f = d["depth"], d["flood_3cm"]
    lo, hi = d["mae_ci95_block_bootstrap"]
    n3 = dd["near_3cm"]["mae_cm"]
    return (f"{label} & {scope_name[scope]} & {meth_name[meth]} & {dd['n']} & {dd['mae_cm']:.2f} "
            f"[{lo:.2f}, {hi:.2f}] & {'--' if n3 is None else f'{n3:.1f}'} & "
            f"{f['f1']:.2f} & {f['false_alarm_rate']:.2f} \\\\")


scope_name = {"gated": "gated", "all_frames": "all"}
meth_name = {"learned": "learned reader", "threshold_baseline": "threshold baseline", "always_no_flood": "always 0 cm"}


def main_table(cv, fin):
    rows = []
    for label, m in (("CV (dev)", cv), ("Locked", fin)):
        for scope in ("gated", "all_frames"):
            for meth in ("learned", "threshold_baseline", "always_no_flood"):
                if scope == "all_frames" and meth == "always_no_flood":
                    continue
                rows.append(metric_row(label if (scope == "gated" and meth == "learned") else "", scope, meth, m))
        rows.append("\\midrule")
    rows = rows[:-1]
    cov = lambda m: f"{100 * m['coverage']:.0f}"
    num("covDev", 100 * cv["coverage"], "{:.0f}")
    num("covLocked", 100 * fin["coverage"], "{:.0f}")
    tex = ("\\begin{tabular}{@{}llllrlccc@{}}\n\\toprule\n"
           "Eval. & Frames & Method & $n$ & MAE (cm) [95\\% CI] & near-3\\,cm MAE & F1 & FAR \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    tex = tex.replace("llllrlccc", "lllrlccc").replace("Eval. & Frames & Method & $n$", "Eval. & Frames & Method & $n$")
    # remove the extra column introduced above (label column holds eval + frames)
    return tex


def _bin_mae(d, lo_keys):
    """Pooled MAE over several depth bins (weighted by n); None if no frames."""
    tot = sum(d["mae_by_bin_cm"][k]["n"] for k in lo_keys)
    if tot == 0:
        return None
    return sum(d["mae_by_bin_cm"][k]["n"] * (d["mae_by_bin_cm"][k]["mae_cm"] or 0.0) for k in lo_keys) / tot


def main_table2(cv, fin):
    head = ("\\begin{tabular}{@{}lllrcccccc@{}}\n\\toprule\nEval. & Frames & Method & $n$ & MAE [95\\% CI] & "
            "RMSE & bias & dry & 0.5--10 & $>$10 \\\\\n\\midrule\n")
    body = []
    for label, m in (("CV", cv), ("Locked", fin)):
        first = True
        for scope in ("gated", "all_frames"):
            for meth in ("learned", "threshold_baseline", "always_no_flood"):
                if (scope == "all_frames" and meth != "learned") or meth == "always_no_flood":
                    continue
                d = m[scope][meth]
                dd = d["depth"]
                lo, hi = d["mae_ci95_block_bootstrap"]
                mae = f"{dd['mae_cm']:.2f}" if label == "Locked" else f"{dd['mae_cm']:.2f} [{lo:.1f}, {hi:.1f}]"
                b_dry = _bin_mae(dd, ["0-0.5"])
                b_sh = _bin_mae(dd, ["0.5-3", "3-10"])
                b_dp = _bin_mae(dd, ["10-25", ">=25"])
                f = lambda v: "--" if v is None else f"{v:.1f}"
                body.append(f"{label if first else ''} & {scope_name[scope]} & {meth_name[meth]} & {dd['n']} & "
                            f"{mae} & {dd['rmse_cm']:.1f} & {dd['bias_cm']:+.1f} & {f(b_dry)} & {f(b_sh)} & {f(b_dp)} \\\\")
                first = False
        body.append("\\midrule")
    num("covDev", 100 * cv["coverage"], "{:.0f}")
    num("covLocked", 100 * fin["coverage"], "{:.0f}")
    num("dropDev", 100 * (1 - cv["coverage"]), "{:.0f}")
    num("dropLocked", 100 * (1 - fin["coverage"]), "{:.0f}")
    return head + "\n".join(body[:-1]) + "\n\\bottomrule\n\\end{tabular}\n"


def smoothing(cv, fin):
    """Causal median of the last five accepted readings, per contiguous session (gaps > 6 h split)."""
    from floodcam.real.events import causal_median

    for tag, d in (("Dev", cv), ("Locked", fin)):
        raw, sm = [], []
        for _, blk in d.groupby("block"):
            blk = blk.sort_values("time")
            t = (blk.time - blk.time.iloc[0]).dt.total_seconds().values / 3600
            seg = np.concatenate([[0], np.cumsum(np.diff(t) > 6.0)])
            for k in np.unique(seg):
                ss = blk[seg == k]
                f = causal_median(ss.learned_cm.values, ss.accepted.values)
                m = ss.accepted.values & np.isfinite(f)
                raw += list(np.abs(ss.learned_cm.values[m] - ss.gauge_cm.values[m]))
                sm += list(np.abs(f[m] - ss.gauge_cm.values[m]))
        num(f"maeRaw{tag}", float(np.mean(raw)))
        num(f"maeSmooth{tag}", float(np.mean(sm)))


# ----------------------------------------------------------------------------- numbers from runs
def run_numbers(cv, fin, cvd, find):
    for tag, m in (("Dev", cv), ("Locked", fin)):
        g, a = m["gated"], m["all_frames"]
        L, B, T = g["learned"], g["threshold_baseline"], g["always_no_flood"]
        num(f"mae{tag}Gated", L["depth"]["mae_cm"])
        lo, hi = L["mae_ci95_block_bootstrap"]
        num(f"ci{tag}Lo", lo)
        num(f"ci{tag}Hi", hi)
        num(f"mae{tag}All", a["learned"]["depth"]["mae_cm"])
        num(f"maeBase{tag}Gated", B["depth"]["mae_cm"])
        num(f"maeBase{tag}All", a["threshold_baseline"]["depth"]["mae_cm"])
        num(f"maeZero{tag}", T["depth"]["mae_cm"])
        num(f"fone{tag}", L["flood_3cm"]["f1"])
        num(f"foneBase{tag}", B["flood_3cm"]["f1"])
        num(f"prec{tag}", L["flood_3cm"]["precision"])
        num(f"rec{tag}", L["flood_3cm"]["recall"])
        num(f"far{tag}", L["flood_3cm"]["false_alarm_rate"])
        num(f"farBase{tag}", B["flood_3cm"]["false_alarm_rate"])
        num(f"recBase{tag}", B["flood_3cm"]["recall"])
        num(f"nearThree{tag}", L["depth"]["near_3cm"]["mae_cm"] or float("nan"), "{:.1f}")
        num(f"nearThreeZero{tag}", T["depth"]["near_3cm"]["mae_cm"] or float("nan"), "{:.1f}")
        num(f"rmse{tag}Gated", L["depth"]["rmse_cm"], "{:.1f}")
        num(f"bias{tag}Gated", L["depth"]["bias_cm"], "{:+.1f}")
        sh = _bin_mae(L["depth"], ["0.5-3", "3-10"])
        shz = _bin_mae(T["depth"], ["0.5-3", "3-10"])
        shb = _bin_mae(B["depth"], ["0.5-3", "3-10"])
        num(f"shallow{tag}", sh if sh is not None else float("nan"), "{:.1f}")
        num(f"shallowZero{tag}", shz if shz is not None else float("nan"), "{:.1f}")
        num(f"shallowBase{tag}", shb if shb is not None else float("nan"), "{:.1f}")
        num(f"shallowN{tag}", sum(L["depth"]["mae_by_bin_cm"][k]["n"] for k in ("0.5-3", "3-10")), "{:d}")
        pb = L["by_block"]
        num(f"blockMedian{tag}", pb["mae_median"])
        num(f"blockWorst{tag}", pb["mae_worst"])
        bins = L["depth"]["mae_by_bin_cm"]
        for k, v in bins.items():
            key = {"0-0.5": "A", "0.5-3": "B", "3-10": "C", "10-25": "D", ">=25": "E"}[k]
            num(f"bin{key}{tag}", v["mae_cm"] if v["mae_cm"] is not None else float("nan"))
            num(f"binN{key}{tag}", v["n"], "{:d}")
    fv1 = json.loads((R / "final_v1_gate_one_sided_metrics.json").read_text())
    num("maeLockedVOneGated", fv1["gated"]["learned"]["depth"]["mae_cm"])
    num("covLockedVOne", 100 * fv1["coverage"], "{:.0f}")
    # event-level
    ev = pd.read_csv(R / "tables" / "event_level.csv")
    for run, tag in (("cv_dev", "Dev"), ("locked_test", "Locked")):
        e = ev[ev.run == run]
        num(f"gtEp{tag}", e.gt_episodes.sum(), "{:d}")
        num(f"detEp{tag}", e.detected.sum(), "{:d}")
        num(f"falseEp{tag}", e.false_episodes.sum(), "{:d}")
        num(f"delay{tag}", e.median_onset_delay_h.median(), "{:.1f}")
    # resolution
    res = json.loads((R / f"resolution-{SITE}" / "summary.json").read_text())
    g = pd.DataFrame([r for r in res["rows"] if r["scope"] == "gated"])
    num("sNative", g.cm_per_px.iloc[0], "{:.2f}")
    num("sigmaV", res["error_model_fit"]["sigma_v_px"])
    num("sigmaFloor", res["error_model_fit"]["sigma_floor_cm"], "{:.1f}")
    labels = "ABCDEFG"
    for lab, (_, r) in zip(labels, g.iterrows()):
        num(f"resMae{lab}", r.learned_mae)
        num(f"resRmse{lab}", r.learned_rmse)
        num(f"resS{lab}", r.cm_per_px, "{:.2f}" if r.cm_per_px < 1 else "{:.1f}")
        num(f"resBase{lab}", r.baseline_mae)
    dbl = g[g.learned_mae >= 2 * g.learned_mae.iloc[0]]
    num("sDouble", dbl.cm_per_px.min() if len(dbl) else float("nan"), "{:.1f}")
    ok = g[g.learned_f1 >= 0.9]
    num("sMaxOk", ok.cm_per_px.max(), "{:.1f}")
    bad = g[g.learned_f1 < 0.9]
    num("sFirstBad", bad.cm_per_px.min() if len(bad) else float("nan"), "{:.1f}")
    num("fOneAtNative", g.learned_f1.iloc[0])
    num("fOneAtBad", bad.learned_f1.iloc[0] if len(bad) else float("nan"))
    num("maeAtBad", bad.learned_mae.iloc[0] if len(bad) else float("nan"))
    num("maeAtNative", g.learned_mae.iloc[0])
    num("regionsMax", g.cm_per_px.max(), "{:.0f}")
    return res, g


# ----------------------------------------------------------------------------- second camera
TR = R / "transfer-in_overland-to-hancock"


def hancock_numbers():
    m = json.loads((TR / "metrics.json").read_text())
    num("hancN", m["zero_shot"]["n_on_plate"], "{:d}")
    num("hancCalN", m["n_calibration_frames"], "{:d}")
    for key, tag in (("zero_shot", "Zero"), ("calibrated", "Cal"), ("pooled_with_target_dev", "Pool")):
        num(f"hancMae{tag}", m[key]["mae_cm"], "{:.0f}")
        num(f"hancBias{tag}", m[key]["bias_cm"], "{:+.0f}")
        for lab, rng_ in zip("ABC", ("11.0-13.0", "13.0-16.0", "16.0-19.0")):
            v = m[key]["mae_by_stage_ft"][rng_]
            num(f"hancMae{tag}{lab}", v if v is not None else float("nan"), "{:.0f}")
    num("hancTrivial", m["baseline_always_plate_bottom_mae_cm"], "{:.0f}")
    num("hancOffsetCm", abs(m["offset_ft"]) * 30.48, "{:.0f}")
    pr = pd.read_csv(TR / "predictions.csv", parse_dates=["time"])
    t = pr[pr.test & pr.accepted & (pr.gauge_ft > 11.15)]
    num("hancPeakFt", t.gauge_ft.max(), "{:.1f}")
    num("hancLowFt", t.gauge_ft.min(), "{:.1f}")
    num("hancRangeCm", (t.gauge_ft.max() - t.gauge_ft.min()) * 30.48, "{:.0f}")
    num("hancFrames", int(len(pr)), "{:d}")
    num("hancDry", int(pr.gauge_ft.lt(11.0).sum()), "{:d}")
    return m, pr


def fig_hancock_axis(ax, pr):
    t = pr[pr.test].sort_values("time")
    t = t[t.time >= "2026-07-29"]
    cm = 30.48
    ax.plot(t.time, t.gauge_ft * cm, color="black", lw=1.4, label="USGS gauge")
    a = t[t.accepted]
    ax.plot(a.time, a.zero_shot_ft * cm, ".", ms=3, color="#d62728", label="zero-shot")
    ax.plot(a.time, a.calibrated_ft * cm, ".", ms=3, color=C["learned"], label="offset-calibrated")
    ax.plot(a.time, a.pooled_ft * cm, ".", ms=3, color="#2ca02c", label="pooled")
    ax.set(ylabel="stage (cm)", title="(c) Hancock flood")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d"))
    ax.set_xlabel("day of Jul 2026")
    ax.legend(frameon=False, loc="upper right", fontsize=5.5, handletextpad=0.3, labelspacing=0.2)


# ----------------------------------------------------------------------------- multi-camera
CAMS = [("in_overland", "cream plate"), ("in_north", "cream plate"),
        ("oh_wetland", "white plate"), ("wi_mt1", "orange plate")]


def multicam():
    loco = json.loads((R / "loco" / "metrics.json").read_text())
    rows = []
    for site, desc in CAMS:
        run = "real-cv-in_overland-f1" if site == "in_overland" else f"real-cv-{site}-f1"
        ins = json.loads((R / run / "metrics.json").read_text())["gated"]
        l = loco[site]
        n_in = ins["learned"]["depth"]["n"]
        rows.append(f"{site.replace('_', chr(92) + '_')} & {desc} & {l['gauge_max_cm']:.0f} & {n_in} & "
                    f"{ins['learned']['depth']['mae_cm']:.2f} / {ins['always_no_flood']['depth']['mae_cm']:.2f} & "
                    f"{l['zero_shot']['gated']['mae_cm']:.2f} / {l['always_0']['gated']['mae_cm']:.2f} & "
                    f"{l['calibrated']['gated']['mae_cm']:.2f} \\\\")
        tag = {"in_overland": "A", "in_north": "B", "oh_wetland": "C", "wi_mt1": "D"}[site]
        num(f"cam{tag}InMae", ins["learned"]["depth"]["mae_cm"])
        num(f"cam{tag}InZero", ins["always_no_flood"]["depth"]["mae_cm"])
        num(f"cam{tag}LocoZero", l["zero_shot"]["gated"]["mae_cm"])
        num(f"cam{tag}LocoCal", l["calibrated"]["gated"]["mae_cm"])
        num(f"cam{tag}LocoTriv", l["always_0"]["gated"]["mae_cm"])
        num(f"cam{tag}Cov", 100 * l["coverage"], "{:.0f}")
        num(f"cam{tag}NLoco", l["zero_shot"]["gated"]["n"], "{:d}")
        num(f"cam{tag}Max", l["gauge_max_cm"], "{:.0f}")
        _lp = pd.read_csv(R / "loco" / "predictions.csv")
        _g = _lp[(_lp.site == site) & _lp.accepted]
        _w = _g[_g.gauge_cm > 3]
        num(f"cam{tag}WetRecall", float((_w.zero_shot_cm > 3).mean()), "{:.2f}")
        num(f"cam{tag}WetMae", float((_w.zero_shot_cm - _w.gauge_cm).abs().mean()), "{:.1f}")
        num(f"cam{tag}WetZero", float(_w.gauge_cm.mean()), "{:.1f}")
    h = json.loads((TR / "metrics.json").read_text())
    rows.append(f"hancock & river plate & {211 + 0:.0f}$^{{\\dagger}}$ & -- & -- & "
                f"{h['zero_shot']['mae_cm']:.0f} / {h['baseline_always_plate_bottom_mae_cm']:.0f} & {h['calibrated']['mae_cm']:.0f} \\\\")
    tex = ("\\begin{tabular}{@{}llrrccc@{}}\n\\toprule\nCamera & Scene & max & $n$ & in-site / 0 & held-out zero-shot / 0 & held-out calib. \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    (P / "tab_cams.tex").write_text(tex)


# ----------------------------------------------------------------------------- extra figures
def fig_pipeline(cv):
    """One real frame through the pipeline: frame + plate quad, rectified strip, row probabilities, gate."""
    from floodcam.real.experiment import fit_models, load_dataset
    from floodcam.real.plate import PlateRectifier
    from floodcam.real.reader import decode_boundary

    ds = load_dataset(SITE, 1)
    spec = SiteSpec.load(SITE)
    d = cv.copy()
    cand = d[(d.block == "2026-09") & d.accepted & d.gauge_cm.between(14, 22) & ~d.is_ir & (d.borrowed < 0.5)]
    row = cand.iloc[len(cand) // 2]
    i = int(np.where(ds.files == row.file)[0][0])
    train = np.isin(ds.blocks, [b for b in ds.split["dev_blocks"] if b != row.block])
    tpl, feats, reader, base = fit_models(ds, spec, train, 0)
    pr = reader.proba(feats[i : i + 1])[0]
    b, _ = decode_boundary(pr[None])
    R = len(pr)
    heights = reader.heights(R)
    h_cm = np.clip(reader.cal_a * heights + reader.cal_b, 0, None) * 30.48
    depth, _ = reader.predict_cm(feats[i : i + 1])
    man = {}
    for line in (ROOT / "data/usgs/manifest.jsonl").read_text().splitlines():
        r = json.loads(line)
        man[r["file"]] = r
    img = cv2.imread(str(ROOT / man[row.file]["path"]))
    rect = PlateRectifier(spec)
    reg = rect.register(img)
    quad = (np.array(spec.quad, np.float32) * [rect.ref_shape[1], rect.ref_shape[0]]).reshape(-1, 1, 2)
    fq = cv2.perspectiveTransform(quad, np.linalg.inv(reg.H)).reshape(-1, 2)
    fig, ax = plt.subplots(1, 4, figsize=(7.2, 1.75), gridspec_kw={"width_ratios": [1.5, 0.45, 1.0, 1.2]})
    vis = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
    cv2.polylines(vis, [fq.astype(np.int32)], True, (255, 220, 0), 3)
    x0, x1 = int(fq[:, 0].min() - 150), int(fq[:, 0].max() + 150)
    y0, y1 = int(max(0, fq[:, 1].min() - 60)), int(min(vis.shape[0], fq[:, 1].max() + 110))
    ax[0].imshow(vis[y0:y1, max(0, x0):x1])
    ax[0].axis("off")
    ax[0].set_title("(a) frame + plate quad", pad=3)
    strip = ds.strips[i][..., ::-1]
    ax[1].imshow(strip, aspect="auto", extent=[0, 1, h_cm.min(), h_cm.max()])
    ax[1].set(xticks=[], ylabel="height (cm)", title="(b) strip", )
    ax[2].plot(pr, h_cm, color=C["learned"], lw=1.2)
    ax[2].axhline(float(depth[0]), color="#d62728", lw=1.0, label=f"decoded {float(depth[0]):.1f} cm")
    ax[2].axhline(max(row.gauge_cm, 0), color="black", ls="--", lw=1.0, label=f"gauge {max(row.gauge_cm, 0):.1f} cm")
    ax[2].set(xlabel="P(row submerged)", title="(c) row classifier", ylim=(h_cm.min(), h_cm.max()), xlim=(0, 1))
    ax[2].legend(frameon=False, fontsize=5.5, loc="upper right", handletextpad=0.3)
    # (d) error vs Laplacian variance (gate)
    dd = cv.copy()
    dd["ae"] = (dd.learned_cm - dd.gauge_cm).abs()
    med = float(np.median(dd.lapvar))
    dd["rel"] = dd.lapvar / med
    edges = [0, 0.3, 0.55, 0.8, 1.1, 1.4, 1.7, 2.2, 10]
    labs, vals, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (dd.rel >= lo) & (dd.rel < hi)
        if m.sum() > 5:
            labs.append(f"{lo:g}" if hi < 10 else f">{lo:g}")
            vals.append(dd.ae[m].mean())
            ns.append(int(m.sum()))
    xs = np.arange(len(vals))
    ax[3].bar(xs, vals, color=[("#2ca02c" if 0.55 <= float(l.strip(">")) < 1.7 else "#bbbbbb") for l in labs])
    ax[3].set_xticks(xs, labs, fontsize=6)
    ax[3].set(xlabel="Laplacian variance / median", ylabel="MAE (cm)", title="(d) why the gate")
    fig.tight_layout(pad=0.4)
    fig.savefig(P / "figs" / "fig_pipeline.pdf", dpi=200)
    plt.close(fig)
    (P / "figs" / "fig_pipeline_file.json").write_text(json.dumps({"file": row.file, "gauge_cm": float(row.gauge_cm)}))


def fig_cams():
    """Held-out (camera left out of training) zero-shot examples: median-error and worst accepted frame per camera."""
    pr = pd.read_csv(R / "loco" / "predictions.csv")
    man = {}
    for line in (ROOT / "data/usgs/manifest.jsonl").read_text().splitlines():
        r = json.loads(line)
        man[r["file"]] = r
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 2.3))
    meta = []
    for col, site in enumerate(("in_north", "oh_wetland", "wi_mt1")):
        z = np.load(ROOT / "data/usgs/strips" / f"{site}_f1.npz")
        idx = {f: k for k, f in enumerate(z["files"])}
        d = pr[(pr.site == site) & pr.accepted].copy()
        d["ae"] = (d.zero_shot_cm - d.gauge_cm).abs()
        wet = d[d.gauge_cm > 3]
        pool = wet if len(wet) > 20 else d
        picks = [("median error", pool.sort_values("ae").iloc[len(pool) // 2]), ("worst", d.sort_values("ae").iloc[-1])]
        for rr, (kind, row) in enumerate(picks):
            img = cv2.imread(str(ROOT / man[row.file]["path"]))
            img = cv2.resize(img, (360, round(360 * img.shape[0] / img.shape[1])))
            st = cv2.resize(z["strips"][idx[row.file]], (34, img.shape[0]), interpolation=cv2.INTER_AREA)
            tile = cv2.cvtColor(np.hstack([img, np.full((img.shape[0], 3, 3), 255, np.uint8), st]), cv2.COLOR_BGR2RGB)
            ax = axes[rr, col]
            ax.imshow(tile)
            ax.axis("off")
            ax.set_title(f"{site.replace('_', chr(92) + '_') if False else site}: {kind}\ngauge {max(row.gauge_cm, 0):.0f} cm, est. {row.zero_shot_cm:.0f} cm",
                         fontsize=7, pad=2)
            meta.append({"camera": site, "kind": kind, "file": row.file, "gauge_cm": float(row.gauge_cm), "est_cm": float(row.zero_shot_cm)})
    fig.tight_layout(pad=0.3, h_pad=0.5, w_pad=0.3)
    fig.savefig(P / "figs" / "fig_cams.pdf", dpi=200)
    plt.close(fig)
    (P / "figs" / "fig_cams_files.json").write_text(json.dumps(meta, indent=1))


# ----------------------------------------------------------------------------- subgroups
def subgroup_table(cv):
    d = cv.copy()
    d["ae"] = (d.learned_cm - d.gauge_cm).abs()
    d["dayir"] = np.where(d.is_ir, "infrared night", "daylight")
    d["pose"] = np.where(d.borrowed > 0.5, "borrowed pose", "registered")
    d["gate"] = np.where(d.accepted, "passes gate", "fails gate")
    rows, out = [], {}
    for col in ("dayir", "pose", "gate"):
        for k, s in d.groupby(col):
            rows.append(f"{k} & {len(s)} & {100 * len(s) / len(d):.0f}\\% & {s.ae.mean():.2f} & "
                        f"{((s.learned_cm >= 3) == (s.gauge_cm >= 3)).mean():.2f} \\\\")
            out[k] = s.ae.mean()
        rows.append("\\midrule")
    num("maeIR", out["infrared night"])
    num("maeDay", out["daylight"])
    num("maeBorrowed", out["borrowed pose"])
    num("maeRegistered", out["registered"])
    num("maeFailGate", out["fails gate"])
    num("maePassGate", out["passes gate"])
    # dirty-lens high variance frames
    hi = d[d.lapvar > d.lapvar.median() * 1.7]
    num("maeHighLap", hi.ae.mean())
    return ("\\begin{tabular}{@{}lrrcc@{}}\n\\toprule\nSubset (CV, all frames) & $n$ & share & MAE (cm) & 3\\,cm agree \\\\\n"
            "\\midrule\n" + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")


# ----------------------------------------------------------------------------- figures
def fig_timeseries_scatter(cv, fin_df, hpr, res=None, g=None):
    fig, ax = plt.subplots(1, 4, figsize=(7.2, 1.6), gridspec_kw={"width_ratios": [1.5, 1.0, 1.1, 1.1]})
    s = cv[(cv.block == "2026-09") & (cv.time >= "2026-09-16") & (cv.time < "2026-09-23")].sort_values("time")
    ax[0].plot(s.time, s.gauge_cm.clip(lower=0), color=C["gauge"], lw=1.4, label="USGS gauge")
    a = s[s.accepted]
    ax[0].plot(a.time, a.learned_cm, ".", ms=2.8, color=C["learned"], label="learned (gated)")
    ax[0].plot(a.time, a.baseline_cm, ".", ms=2, color=C["base"], alpha=0.55, label="threshold baseline")
    r = s[~s.accepted]
    ax[0].plot(r.time, np.full(len(r), -2.5), "|", color="gray", ms=3, label="rejected")
    ax[0].set(ylabel="depth (cm)", title="(a) held-out month, one event")
    ax[0].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d"))
    ax[0].set_xlabel("day of Sep 2026")
    ax[0].legend(loc="upper left", ncol=1, frameon=False, fontsize=5.5, handletextpad=0.3, labelspacing=0.2)
    ax[0].set_ylim(-5, 52)
    gg = cv[cv.accepted]
    for blk, ss in gg.groupby("block"):
        ax[1].scatter(ss.gauge_cm.clip(lower=0), ss.learned_cm, s=2.5, alpha=0.5, linewidths=0)
    ax[1].plot([0, 62], [0, 62], "k--", lw=0.6)
    ax[1].set(xlim=(-1, 62), ylim=(-1, 62), xlabel="gauge (cm)", ylabel="camera (cm)", title="(b) CV, all months")
    fig_hancock_axis(ax[2], hpr)
    fit = res["error_model_fit"]
    ax[3].plot(g.cm_per_px, g.learned_rmse, "o-", ms=2.5, color=C["learned"], label="RMSE")
    ax[3].plot(g.cm_per_px, fit["predicted_rmse"], "k--", lw=0.8, label="fit")
    ax[3].plot(g.cm_per_px, g.learned_mae, "s-", ms=2.5, color="#2ca02c", label="MAE")
    ax[3].plot(g.cm_per_px, g.baseline_mae, "^-", ms=2.5, color=C["base"], alpha=0.7, label="baseline MAE")
    ax[3].set(xscale="log", xlabel="cm per pixel $s$", ylabel="error (cm)", title="(d) error vs resolution")
    ax[3].legend(frameon=False, loc="upper left", fontsize=5.5, handletextpad=0.3, labelspacing=0.2)
    fig.tight_layout(pad=0.4)
    fig.savefig(P / "figs" / "fig_main.pdf")
    plt.close(fig)


def frame_tile(row, strips, idx, man, caption):
    img = cv2.imread(str(ROOT / man[row.file]["path"]))
    h, w = img.shape[:2]
    crop = img[int(0.10 * h) : h, int(0.15 * w) : int(0.85 * w)]
    crop = cv2.resize(crop, (360, round(360 * crop.shape[0] / crop.shape[1])))
    strip = cv2.resize(strips[idx[row.file]], (44, crop.shape[0]), interpolation=cv2.INTER_AREA)
    tile = np.hstack([crop, np.full((crop.shape[0], 3, 3), 255, np.uint8), strip])
    return cv2.cvtColor(tile, cv2.COLOR_BGR2RGB), caption


def fig_frames(cv):
    z = np.load(ROOT / "data/usgs/strips" / f"{SITE}_f1.npz")
    idx = {f: i for i, f in enumerate(z["files"])}
    strips = z["strips"]
    man = {}
    for line in (ROOT / "data/usgs/manifest.jsonl").read_text().splitlines():
        r = json.loads(line)
        man[r["file"]] = r
    d = cv.copy()
    d["ae"] = (d.learned_cm - d.gauge_cm).abs()
    acc = d[d.accepted]

    def median_example(sub):  # frame whose error equals the median error of the subset
        sub = sub.sort_values("ae")
        return sub.iloc[len(sub) // 2]

    ok = [("shallow, day", median_example(acc[(~acc.is_ir) & acc.gauge_cm.between(4, 7)])),
          ("deep, day", median_example(acc[(~acc.is_ir) & acc.gauge_cm.between(30, 45)])),
          ("infrared night", median_example(acc[(acc.is_ir) & acc.gauge_cm.between(12, 30)]))]
    bad = [("worst, accepted (day)", acc[~acc.is_ir].sort_values("ae").iloc[-1]),
           ("worst, accepted (IR)", acc[acc.is_ir].sort_values("ae").iloc[-1]),
           ("rejected: lens dirt", d[(~d.accepted) & (d.lapvar > d.lapvar.median() * 1.7)].sort_values("lapvar").iloc[-1]),
           ("rejected: overexposed", d[(~d.accepted) & (d.lapvar < d.lapvar.median() * 0.55) & (~d.is_ir)].sort_values("lapvar").iloc[0])]
    fig, axes = plt.subplots(3, 4, figsize=(7.2, 2.95))
    meta = []
    items = []
    for cap, row in ok + bad:
        tile, _ = frame_tile(row, strips, idx, man, cap)
        est = "no reading" if (not row.accepted) else f"est. {row.learned_cm:.1f} cm"
        items.append((cap, tile, f"gauge {max(row.gauge_cm, 0):.1f} cm, {est}", row.file))
    # second camera: median-error accepted flood frame of the locked Hancock block (calibrated variant)
    hz = np.load(ROOT / "data/usgs/strips" / "hancock_f1.npz")
    hidx = {f: i for i, f in enumerate(hz["files"])}
    pr = pd.read_csv(TR / "predictions.csv")
    pr = pr[pr.test & pr.accepted & pr.gauge_ft.between(13.5, 16.5)].copy()
    pr["ae"] = (pr.calibrated_ft - pr.gauge_ft).abs()
    hrow = pr.sort_values("ae").iloc[len(pr) // 2]
    himg = cv2.imread(str(ROOT / man[hrow.file]["path"]))
    hh, ww = himg.shape[:2]
    crop = himg[int(0.05 * hh): int(0.75 * hh), int(0.08 * ww): int(0.92 * ww)]
    crop = cv2.resize(crop, (round(300 * crop.shape[1] / crop.shape[0]), 300))
    hs = cv2.resize(hz["strips"][hidx[hrow.file]], (44, 300), interpolation=cv2.INTER_AREA)
    htile = cv2.cvtColor(np.hstack([crop, np.full((300, 3, 3), 255, np.uint8), hs]), cv2.COLOR_BGR2RGB)
    items.insert(0, ("second camera (Hancock)", htile, f"gauge {hrow.gauge_ft * 30.48:.0f} cm, est. {hrow.calibrated_ft * 30.48:.0f} cm", hrow.file))
    items = items[:8]
    # row 3: held-out cameras (zero-shot): median-error frame of three cameras + the worst wi_mt1 frame
    lp = pd.read_csv(R / "loco" / "predictions.csv")
    for site, kind in (("in_north", "median"), ("oh_wetland", "median"), ("wi_mt1", "median"), ("wi_mt1", "worst")):
        zz = np.load(ROOT / "data/usgs/strips" / f"{site}_f1.npz")
        zi = {f: k for k, f in enumerate(zz["files"])}
        dd = lp[(lp.site == site) & lp.accepted].copy()
        dd["ae"] = (dd.zero_shot_cm - dd.gauge_cm).abs()
        pool = dd[dd.gauge_cm > 3] if (dd.gauge_cm > 3).sum() > 20 else dd
        r_ = pool.sort_values("ae").iloc[len(pool) // 2] if kind == "median" else dd.sort_values("ae").iloc[-1]
        im_ = cv2.imread(str(ROOT / man[r_.file]["path"]))
        im_ = cv2.resize(im_, (300, round(300 * im_.shape[0] / im_.shape[1])))
        st_ = cv2.resize(zz["strips"][zi[r_.file]], (28, im_.shape[0]), interpolation=cv2.INTER_AREA)
        tl = cv2.cvtColor(np.hstack([im_, np.full((im_.shape[0], 3, 3), 255, np.uint8), st_]), cv2.COLOR_BGR2RGB)
        items.append((f"{site} (held out), {kind}", tl, f"gauge {max(r_.gauge_cm, 0):.0f} cm, est. {r_.zero_shot_cm:.0f} cm", r_.file))
    for ax, (cap, tile, sub, fname) in zip(axes.ravel(), items):
        ax.imshow(tile)
        ax.axis("off")
        ax.set_title(f"{cap}\n{sub}", fontsize=7, pad=2)
        meta.append({"panel": cap, "file": fname, "caption": sub})
    fig.tight_layout(pad=0.3, h_pad=0.4, w_pad=0.2)
    fig.savefig(P / "figs" / "fig_frames.pdf", dpi=200)
    plt.close(fig)
    (P / "figs" / "fig_frames_files.json").write_text(json.dumps(meta, indent=1, default=str))


def fig_depth(res, g, cv):
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 1.3))
    fit = res["error_model_fit"]
    ax[0].plot(g.cm_per_px, g.learned_rmse, "o-", ms=3, color=C["learned"], label="measured RMSE")
    ax[0].plot(g.cm_per_px, fit["predicted_rmse"], "k--", lw=0.9, label=r"$\sqrt{(\sigma_v s)^2+\sigma_g^2}$ fit")
    ax[0].plot(g.cm_per_px, g.learned_mae, "s-", ms=3, color="#2ca02c", label="MAE")
    ax[0].plot(g.cm_per_px, g.baseline_mae, "^-", ms=3, color=C["base"], alpha=0.7, label="baseline MAE")
    ax[0].set(xscale="log", xlabel="cm per pixel along plate $s$", ylabel="error (cm)", title="(a) error vs resolution")
    ax[0].legend(frameon=False, loc="upper left", ncol=2, columnspacing=0.8, handletextpad=0.3)
    ax[0].set_ylim(0, 16)
    base0 = g.learned_mae.iloc[0]
    ax[1].plot(g.cm_per_px, g.learned_mae / base0, "o-", ms=3, color=C["learned"], label="learned MAE / native")
    ax[1].axhline(2, color="gray", lw=0.7, ls=":")
    ax[1].text(g.cm_per_px.iloc[0], 2.08, "2x native", fontsize=6.5, color="gray")
    ax[1].set(xscale="log", xlabel="cm per pixel along plate $s$", ylabel="MAE relative to native",
              title="(b) error growth with scale")
    ax[1].legend(frameon=False, loc="upper left")
    d = cv.copy()
    d["ae"] = (d.learned_cm - d.gauge_cm).abs()
    d = d.sort_values("margin", ascending=False)
    cov = np.arange(1, len(d) + 1) / len(d)
    ax[2].plot(cov[20:], d.ae.expanding().mean().values[20:], color=C["learned"], label="rank by decode margin")
    gate_cov = float((cv.accepted).mean())
    k = int(round(gate_cov * len(d)))
    num("marginMaeAtCov", float(d.ae.values[:k].mean()))
    gate_mae = float(((cv.learned_cm - cv.gauge_cm).abs())[cv.accepted].mean())
    ax[2].plot(gate_cov, gate_mae, "*", ms=7, color="#d62728", label="Laplacian gate")
    ax[2].set(xlabel="coverage", ylabel="MAE (cm)", title="(c) risk vs coverage (CV)")
    ax[2].legend(frameon=False, loc="upper left")
    fig.tight_layout(pad=0.4)
    fig.savefig(P / "figs" / "fig_depth.pdf")
    plt.close(fig)


def main() -> int:
    P.mkdir(exist_ok=True)
    (P / "figs").mkdir(exist_ok=True)
    cvd, cvm = load_run(f"real-cv-{SITE}-f1")
    find, fm = load_run(final_run())
    facts()
    res, g = run_numbers(cvm, fm, cvd, find)
    smoothing(cvd, find)
    (P / "tab_main.tex").write_text(main_table2(cvm, fm))
    subgroup_table(cvd)  # sets the subgroup macros used in the text
    hm, hpr = hancock_numbers()
    multicam()
    fig_timeseries_scatter(cvd, find, hpr, res, g)
    fig_frames(cvd)
    fig_pipeline(cvd)
    # marginMaeAtCov macro (gate vs margin ranking)
    _d = cvd.copy(); _d['ae'] = (_d.learned_cm - _d.gauge_cm).abs(); _d = _d.sort_values('margin', ascending=False)
    num('marginMaeAtCov', float(_d.ae.values[: int(round(cvd.accepted.mean() * len(_d)))].mean()))
    (P / "numbers.tex").write_text("% generated by scripts/make_paper_assets.py; do not edit\n" + "\n".join(
        f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(NUM.items())) + "\n")
    print(len(NUM), "macros;", "figures in", P / "figs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
