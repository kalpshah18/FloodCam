#!/usr/bin/env python
"""Run all 7 synthetic time-series scenarios through the temporal pipeline.

Writes results/<run_id>/:
  <scenario_name>.csv  — timestamp_s, raw_depth_cm, filtered_depth_cm,
                         rate_cm_per_10min, state, severity_class, stale, transition
  meta.json            — synthetic=true, note, run_id, date, config_hash

Usage:
    python scripts/simulate_state_machine.py [--run-id temporal-001]

SYNTHETIC RESULTS ONLY — must not appear in any paper or final evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

# Allow running as a script from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from floodcam.config import TemporalConfig
from floodcam.synth.timeseries import (
    SynthSample,
    dropout_gap,
    oscillation_around_threshold,
    rise_plateau_fall,
    short_puddle,
    single_spike,
    slow_fall_from_deep,
    steady_rise,
)
from floodcam.temporal.filter import TemporalFilter
from floodcam.temporal.state_machine import StateMachine

SCENARIOS: list[tuple[str, list[SynthSample]]] = [
    (
        "steady_rise",
        steady_rise(start_depth_cm=0, end_depth_cm=20, duration_s=900, sample_interval_s=1.0, seed=1),
    ),
    (
        "rise_plateau_fall",
        rise_plateau_fall(rise_to_cm=15, plateau_s=300, total_s=900, sample_interval_s=1.0, seed=2),
    ),
    (
        "oscillation_around_threshold",
        oscillation_around_threshold(duration_s=600, sample_interval_s=1.0, seed=3),
    ),
    (
        "single_spike",
        single_spike(
            baseline_cm=1.0, spike_cm=8.0, spike_duration_s=2.0, duration_s=120, sample_interval_s=1.0, seed=4
        ),
    ),
    (
        "dropout_gap",
        dropout_gap(
            baseline_cm=5.0, gap_start_s=60, gap_duration_s=60, total_s=240, sample_interval_s=1.0, seed=5
        ),
    ),
    (
        "short_puddle",
        short_puddle(depth_cm=1.5, duration_s=300, sample_interval_s=1.0, seed=6),
    ),
    (
        "slow_fall_from_deep",
        slow_fall_from_deep(
            start_depth_cm=20, end_depth_cm=0, duration_s=1200, sample_interval_s=1.0, seed=7
        ),
    ),
]


def run_scenario(scenario_name: str, samples: list[SynthSample], cfg: TemporalConfig) -> list[dict]:
    f = TemporalFilter(cfg)
    sm = StateMachine(cfg)
    rows = []
    for s in samples:
        fo = f.update(s.timestamp_s, s)
        so = sm.update(s.timestamp_s, fo)
        rows.append(
            {
                "timestamp_s": s.timestamp_s,
                "raw_depth_cm": s.depth_cm,
                "filtered_depth_cm": fo.filtered_depth_cm if fo.filtered_depth_cm is not None else "",
                "rate_cm_per_10min": fo.rate_cm_per_10min if fo.rate_cm_per_10min is not None else "",
                "state": so.state.value,
                "severity_class": so.severity_class.name,
                "stale": int(so.stale),
                "transition": so.transition or "",
            }
        )
    return rows


def summarise(scenario_name: str, rows: list[dict]) -> dict:
    states = [r["state"] for r in rows]
    transitions = [r["transition"] for r in rows if r["transition"]]
    flood_ts = next(
        (r["timestamp_s"] for r in rows if r["state"] == "Flood"), None
    )
    return {
        "scenario": scenario_name,
        "n_samples": len(rows),
        "final_state": states[-1] if states else "—",
        "flood_entry_s": flood_ts,
        "n_transitions": len(transitions),
        "key_transitions": transitions[:6],  # first 6
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate temporal pipeline on synthetic scenarios.")
    parser.add_argument("--run-id", default="temporal-001")
    args = parser.parse_args()

    cfg = TemporalConfig()
    out_dir = Path("results") / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for scenario_name, samples in SCENARIOS:
        rows = run_scenario(scenario_name, samples, cfg)
        csv_path = out_dir / f"{scenario_name}.csv"
        if rows:
            fieldnames = list(rows[0].keys())
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        summaries.append(summarise(scenario_name, rows))

    meta = {
        "run_id": args.run_id,
        "date": str(date.today()),
        "synthetic": True,
        "note": "SYNTHETIC RESULTS — temporal pipeline validation only. Must not appear in a paper.",
        "config": {
            "filter_window_samples": cfg.filter_window_samples,
            "persistence_minutes": cfg.persistence_minutes,
            "trend_window_minutes": cfg.trend_window_minutes,
            "trigger_cm": cfg.trigger_cm,
            "clear_cm": cfg.clear_cm,
            "falling_slope_cm_per_10min": cfg.falling_slope_cm_per_10min,
            "max_gap_seconds": cfg.max_gap_seconds,
        },
        "scenarios": summaries,
    }
    with (out_dir / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)

    # Print summary table
    header = f"{'Scenario':<35} {'Final':>12} {'Flood@(s)':>12} {'Transitions':>12}"
    print("\n" + header)
    print("-" * len(header))
    for s in summaries:
        flood_str = f"{s['flood_entry_s']:.0f}" if s["flood_entry_s"] is not None else "—"
        print(
            f"{s['scenario']:<35} {s['final_state']:>12} {flood_str:>12} {s['n_transitions']:>12}"
        )
    print(f"\nResults written to: {out_dir}")


if __name__ == "__main__":
    main()
