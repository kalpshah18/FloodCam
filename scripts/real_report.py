#!/usr/bin/env python
"""Regenerate every figure, table and the summary from results/real/** (no hand-typed numbers).

Writes results/real/figures/*.png, results/real/tables/*.csv, results/real/SUMMARY.md and the
failure audit (results/real/failures/). Run after real_cv.py, real_resolution.py and
real_final_test.py.
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
from floodcam.real.events import causal_median, episodes, match_episodes  # noqa: E402

R = ROOT / "results" / "real"
FIG, TAB, FAIL = R / "figures", R / "tables", R / "failures"
SITE = "in_overland"


def load(run: str):
    d = pd.read_csv(R / run / "predictions.csv")
    d["time"] = pd.to_datetime(d["time"])
    return d, json.loads((R / run / "metrics.json").read_text())


def scatter(d, title, path):
    g = d[d.accepted]
    fig, ax = plt.subplots(figsize=(5, 5))
    for b, s in g.groupby("block"):
        ax.scatter(s.gauge_cm, s.learned_cm, s=8, alpha=0.6, label=b)
    m = max(g.gauge_cm.max(), g.learned_cm.max()) * 1.05
    ax.plot([0, m], [0, m], "k--", lw=1)
    ax.axhline(3, color="r", lw=0.8)
    ax.axvline(3, color="r", lw=0.8)
    ax.set(xlabel="USGS gauge (cm)", ylabel="camera estimate (cm)", title=title)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def timeseries(d, block, path):
    s = d[d.block == block].sort_values("time")
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(s.time, s.gauge_cm, "k-", lw=1.5, label="USGS gauge")
    acc = s[s.accepted]
    ax.plot(acc.time, acc.learned_cm, ".", ms=4, color="tab:blue", label="learned reader")
    ax.plot(acc.time, acc.baseline_cm, ".", ms=3, color="tab:orange", alpha=0.5, label="threshold baseline")
    rej = s[~s.accepted]
    ax.plot(rej.time, np.zeros(len(rej)) - 1.5, "|", color="gray", label="rejected (blur/obstruction)")
    ax.axhline(3, color="r", lw=0.8, ls="--")
    ax.set(ylabel="depth (cm)", title=f"held-out block {block}")
    ax.legend(fontsize=7, ncol=4)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def block_bars(summ, path):
    pb = summ["gated"]["learned"]["by_block"]["per_block"]
    pbb = summ["gated"]["threshold_baseline"]["by_block"]["per_block"]
    blocks = sorted(pb)
    x = np.arange(len(blocks))
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(x - 0.2, [pb[b]["mae_cm"] for b in blocks], 0.4, label="learned")
    ax.bar(x + 0.2, [pbb[b]["mae_cm"] for b in blocks], 0.4, label="threshold baseline")
    ax.set_xticks(x, [f"{b}\n(n={pb[b]['n']})" for b in blocks], fontsize=6)
    ax.set(ylabel="MAE (cm)", title="Held-out block MAE (gated frames)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def risk_coverage(d, path):
    d = d.copy()
    d["ae"] = (d.learned_cm - d.gauge_cm).abs()
    d = d.sort_values("margin", ascending=False)
    cov = np.arange(1, len(d) + 1) / len(d)
    risk = d.ae.expanding().mean()
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.plot(cov, risk)
    ax.set(xlabel="coverage (frames kept by model confidence)", ylabel="MAE (cm)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return pd.DataFrame({"coverage": cov, "mae_cm": risk.values})


def resolution(path):
    p = R / f"resolution-{SITE}" / "summary.json"
    if not p.exists():
        return None
    j = json.loads(p.read_text())
    df = pd.DataFrame(j["rows"])
    g = df[df.scope == "gated"]
    fit = j["error_model_fit"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(g.cm_per_px, g.learned_rmse, "o-", label="measured RMSE (learned)")
    ax[0].plot(g.cm_per_px, fit["predicted_rmse"], "k--", label="error model fit")
    ax[0].plot(g.cm_per_px, g.baseline_mae, "s-", alpha=0.5, label="baseline MAE")
    ax[0].set(xscale="log", xlabel="cm per pixel along the plate", ylabel="cm")
    ax[0].legend(fontsize=7)
    ax[1].plot(g.cm_per_px, g.learned_f1, "o-", label="F1 (>=3 cm)")
    ax[1].plot(g.cm_per_px, g.learned_recall, "s-", label="recall")
    ax[1].plot(g.cm_per_px, g.learned_precision, "^-", label="precision")
    ax[1].axvline(3.0, color="gray", lw=0.6)
    ax[1].set(xscale="log", xlabel="cm per pixel along the plate", ylim=(0, 1.02))
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    df.to_csv(TAB / "resolution.csv", index=False)
    return j


def event_level(d, run_name):
    rows = []
    for b, s in d.groupby("block"):
        s = s.sort_values("time")
        t = (s.time - s.time.iloc[0]).dt.total_seconds().values / 3600
        valid = s.accepted.values
        filt = causal_median(s.learned_cm.values, valid)
        gt_ep = episodes(s.gauge_cm.values, t)
        pr_ep = episodes(filt, t)
        m = match_episodes(gt_ep, pr_ep)
        days = max((t[-1] - t[0]) / 24, 1e-6)
        rows.append({"run": run_name, "block": b, "span_days": days, **{k: v for k, v in m.items() if k != "onset_delay_h"},
                     "median_onset_delay_h": float(np.median(m["onset_delay_h"])) if m["onset_delay_h"] else None,
                     "false_per_day": m["false_episodes"] / days})
    return pd.DataFrame(rows)


def failure_audit(d, n_worst=10, n_rej=6):
    FAIL.mkdir(parents=True, exist_ok=True)
    z = np.load(ROOT / "data/usgs/strips" / f"{SITE}_f1.npz")
    idx = {f: i for i, f in enumerate(z["files"])}
    man = {}
    for line in (ROOT / "data/usgs/manifest.jsonl").read_text().splitlines():
        r = json.loads(line)
        man[r["file"]] = r
    d = d.copy()
    d["ae"] = (d.learned_cm - d.gauge_cm).abs()
    worst = d[d.accepted].sort_values("ae", ascending=False).head(n_worst).assign(kind="worst_accepted")
    rej = d[~d.accepted].sample(n=min(n_rej, (~d.accepted).sum()), random_state=0).assign(kind="rejected")
    out = []
    for r in pd.concat([worst, rej]).itertuples():
        img = cv2.imread(str(ROOT / man[r.file]["path"]))
        img = cv2.resize(img, (640, round(img.shape[0] * 640 / img.shape[1])))
        strip = cv2.resize(z["strips"][idx[r.file]], (96, img.shape[0]))
        panel = np.hstack([img, strip])
        txt = f"{r.kind} | gauge {r.gauge_cm:.1f} cm | est {r.learned_cm:.1f} cm | lapvar {r.lapvar:.0f}"
        cv2.putText(panel, txt, (6, panel.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        name = f"{r.kind}_{r.file.split('___')[1][:-5]}.jpg"
        cv2.imwrite(str(FAIL / name), panel, [cv2.IMWRITE_JPEG_QUALITY, 80])
        out.append({"kind": r.kind, "file": r.file, "gauge_cm": r.gauge_cm, "est_cm": r.learned_cm,
                    "lapvar": r.lapvar, "borrowed": r.borrowed, "is_ir": r.is_ir, "overlay": name})
    pd.DataFrame(out).to_csv(FAIL / "failures.csv", index=False)


def main() -> int:
    for p in (FIG, TAB):
        p.mkdir(parents=True, exist_ok=True)
    cv, summ = load(f"real-cv-{SITE}-f1")
    scatter(cv, "Leave-one-block-out (gated)", FIG / "scatter_cv.png")
    block_bars(summ, FIG / "block_mae_cv.png")
    risk_coverage(cv, FIG / "risk_coverage_cv.png").to_csv(TAB / "risk_coverage.csv", index=False)
    for b in ("2026-09", "2024-04", "2023-02"):
        if (cv.block == b).any():
            timeseries(cv, b, FIG / f"timeseries_{b}.png")
    failure_audit(cv)
    res = resolution(FIG / "resolution.png")
    ev = [event_level(cv, "cv_dev")]
    fin = None
    final_run = next((r for r in (f"real-final-{SITE}-f1-force", f"real-final-{SITE}-f1")
                      if (R / r / "predictions.csv").exists()), None)
    if final_run:
        fd, fin = load(final_run)
        scatter(fd, "Locked test blocks (gated)", FIG / "scatter_final.png")
        for b in fd.block.unique():
            timeseries(fd, b, FIG / f"timeseries_final_{b}.png")
        ev.append(event_level(fd, "locked_test"))
    pd.concat(ev).to_csv(TAB / "event_level.csv", index=False)

    L = ["# Real-footage results (generated by scripts/real_report.py)\n",
         "Source: USGS gauge cameras + USGS gauge height (public domain); see data/usgs/README.md.\n"]

    def block(title, s):
        L.append(f"\n## {title}\n")
        L.append(f"Coverage after sharpness gate: {s['coverage']:.1%}\n")
        L.append("| scope | method | n | MAE cm | 95% CI (block bootstrap) | MAE near 3 cm | P(>=3cm) | R | F1 | false-alarm rate |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for scope in ("all_frames", "gated"):
            for name, sc in s[scope].items():
                d_, f_ = sc["depth"], sc["flood_3cm"]
                lo, hi = sc["mae_ci95_block_bootstrap"]
                n3 = d_["near_3cm"]["mae_cm"]
                L.append(f"| {scope} | {name} | {d_['n']} | {d_['mae_cm']:.2f} | {lo:.2f}-{hi:.2f} | "
                         f"{'-' if n3 is None else f'{n3:.2f}'} | {f_['precision']:.2f} | {f_['recall']:.2f} | "
                         f"{f_['f1']:.2f} | {f_['false_alarm_rate']:.3f} |")
        L.append("\nMAE by depth bin (gated, learned):\n")
        L.append("| bin cm | n | MAE cm |\n|---|---|---|")
        for k, v in s["gated"]["learned"]["depth"]["mae_by_bin_cm"].items():
            L.append(f"| {k} | {v['n']} | {'-' if v['mae_cm'] is None else f'{v['mae_cm']:.2f}'} |")
        pb = s["gated"]["learned"]["by_block"]
        L.append(f"\nPer-block MAE (gated): median {pb['mae_median']:.2f}, IQR "
                 f"{pb['mae_iqr'][0]:.2f}-{pb['mae_iqr'][1]:.2f}, worst {pb['mae_worst']:.2f} cm.\n")

    block("Leave-one-block-out CV on dev blocks", summ)
    if fin:
        block("Locked test blocks (final gate, second and last run; see note)", fin)
        L.append("\nNote: the locked blocks were evaluated first with a one-sided sharpness gate (results/real/final_v1_gate_one_sided_metrics.json). The two-sided gate was then chosen from DEV-block error only and the locked run repeated with --force; both runs are in results/real/final_test_log.jsonl.\n")
    if res:
        L.append("\n## Resolution sweep (gated, learned reader, LOBO CV)\n")
        L.append("| factor | cm/px | MAE | RMSE | near-3cm MAE | precision | recall | F1 |\n|---|---|---|---|---|---|---|---|")
        for r in res["rows"]:
            if r["scope"] == "gated":
                n3 = r["learned_near3_mae"]
                L.append(f"| {r['factor']} | {r['cm_per_px']:.2f} | {r['learned_mae']:.2f} | {r['learned_rmse']:.2f} | "
                         f"{'-' if n3 is None else f'{n3:.2f}'} | {r['learned_precision']:.2f} | "
                         f"{r['learned_recall']:.2f} | {r['learned_f1']:.2f} |")
        f = res["error_model_fit"]
        L.append(f"\nError-model fit RMSE^2 = (sigma_v*s)^2 + floor^2: sigma_v = {f['sigma_v_px']:.2f} px, "
                 f"floor = {f['sigma_floor_cm']:.2f} cm.\n")
    L.append("\n## Event level (causal median filter, hysteresis 3.5/2.5 cm)\n")
    evt = pd.concat(ev)
    L.append("```\n" + evt.to_string(index=False, float_format=lambda v: f"{v:.2f}") + "\n```")
    (R / "SUMMARY.md").write_text("\n".join(L) + "\n")
    print("wrote", R / "SUMMARY.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
