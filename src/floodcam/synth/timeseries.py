"""Synthetic depth-vs-time profile generators.

All outputs are tagged synthetic=True and must never appear in reported results.
Profiles are purely numerical — no image rendering.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SynthSample:
    """One timestamped depth sample.  Compatible with TemporalFilter.update()."""

    timestamp_s: float
    depth_cm: float
    confidence: float = 1.0
    valid: bool = True
    synthetic: bool = True


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def steady_rise(
    start_depth_cm: float = 0.0,
    end_depth_cm: float = 20.0,
    duration_s: float = 600.0,
    sample_interval_s: float = 1.0,
    noise_sigma: float = 0.0,
    seed: int = 0,
) -> list[SynthSample]:
    rng = _rng(seed)
    n = int(duration_s / sample_interval_s) + 1
    ts = np.linspace(0.0, duration_s, n)
    depths = np.linspace(start_depth_cm, end_depth_cm, n)
    if noise_sigma > 0:
        depths += rng.normal(0.0, noise_sigma, n)
    return [SynthSample(float(t), float(d)) for t, d in zip(ts, depths)]


def rise_plateau_fall(
    rise_to_cm: float = 15.0,
    plateau_s: float = 300.0,
    total_s: float = 900.0,
    sample_interval_s: float = 1.0,
    noise_sigma: float = 0.0,
    seed: int = 0,
) -> list[SynthSample]:
    rng = _rng(seed)
    rise_s = (total_s - plateau_s) / 2.0
    fall_s = total_s - plateau_s - rise_s

    samples: list[SynthSample] = []
    t = 0.0

    n_rise = int(rise_s / sample_interval_s) + 1
    for i, depth in enumerate(np.linspace(0.0, rise_to_cm, n_rise)):
        noise = rng.normal(0.0, noise_sigma) if noise_sigma > 0 else 0.0
        samples.append(SynthSample(t, float(depth + noise)))
        t += sample_interval_s

    n_plateau = int(plateau_s / sample_interval_s)
    for _ in range(n_plateau):
        noise = rng.normal(0.0, noise_sigma) if noise_sigma > 0 else 0.0
        samples.append(SynthSample(t, float(rise_to_cm + noise)))
        t += sample_interval_s

    n_fall = int(fall_s / sample_interval_s)
    for i, depth in enumerate(np.linspace(rise_to_cm, 0.0, n_fall)):
        noise = rng.normal(0.0, noise_sigma) if noise_sigma > 0 else 0.0
        samples.append(SynthSample(t, float(depth + noise)))
        t += sample_interval_s

    return samples


def oscillation_around_threshold(
    center_cm: float = 3.0,
    amplitude_cm: float = 1.5,
    period_s: float = 60.0,
    duration_s: float = 600.0,
    sample_interval_s: float = 1.0,
    noise_sigma: float = 0.0,
    seed: int = 0,
) -> list[SynthSample]:
    rng = _rng(seed)
    n = int(duration_s / sample_interval_s) + 1
    ts = np.linspace(0.0, duration_s, n)
    depths = center_cm + amplitude_cm * np.sin(2 * np.pi * ts / period_s)
    if noise_sigma > 0:
        depths += rng.normal(0.0, noise_sigma, n)
    return [SynthSample(float(t), float(max(0.0, d))) for t, d in zip(ts, depths)]


def single_spike(
    baseline_cm: float = 1.0,
    spike_cm: float = 8.0,
    spike_duration_s: float = 2.0,
    duration_s: float = 120.0,
    sample_interval_s: float = 1.0,
    seed: int = 0,
) -> list[SynthSample]:
    n = int(duration_s / sample_interval_s) + 1
    spike_start = duration_s / 2.0
    spike_end = spike_start + spike_duration_s
    samples = []
    for i in range(n):
        t = i * sample_interval_s
        depth = spike_cm if spike_start <= t < spike_end else baseline_cm
        samples.append(SynthSample(float(t), float(depth)))
    return samples


def dropout_gap(
    baseline_cm: float = 5.0,
    gap_start_s: float = 60.0,
    gap_duration_s: float = 60.0,
    total_s: float = 240.0,
    sample_interval_s: float = 1.0,
    seed: int = 0,
) -> list[SynthSample]:
    """Samples in [gap_start_s, gap_start_s + gap_duration_s) have valid=False."""
    n = int(total_s / sample_interval_s) + 1
    samples = []
    for i in range(n):
        t = i * sample_interval_s
        in_gap = gap_start_s <= t < gap_start_s + gap_duration_s
        samples.append(SynthSample(float(t), float(baseline_cm), valid=not in_gap))
    return samples


def short_puddle(
    depth_cm: float = 1.5,
    duration_s: float = 300.0,
    sample_interval_s: float = 1.0,
    seed: int = 0,
) -> list[SynthSample]:
    n = int(duration_s / sample_interval_s) + 1
    return [SynthSample(float(i * sample_interval_s), float(depth_cm)) for i in range(n)]


def slow_fall_from_deep(
    start_depth_cm: float = 20.0,
    end_depth_cm: float = 0.0,
    duration_s: float = 1200.0,
    sample_interval_s: float = 1.0,
    noise_sigma: float = 0.0,
    seed: int = 0,
) -> list[SynthSample]:
    rng = _rng(seed)
    n = int(duration_s / sample_interval_s) + 1
    ts = np.linspace(0.0, duration_s, n)
    depths = np.linspace(start_depth_cm, end_depth_cm, n)
    if noise_sigma > 0:
        depths += rng.normal(0.0, noise_sigma, n)
    depths = np.maximum(depths, 0.0)
    return [SynthSample(float(t), float(d)) for t, d in zip(ts, depths)]
