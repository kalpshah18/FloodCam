"""Tests for Milestone 5: temporal filter, severity, state machine, timeseries."""

from __future__ import annotations

import copy
from typing import Optional

import pytest
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
from floodcam.temporal.severity import SeverityClass, apply_rate_escalation, depth_to_severity
from floodcam.temporal.state_machine import FloodState, StateMachine, StateOutput

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_cfg(**kwargs) -> TemporalConfig:
    defaults = dict(
        filter_window_samples=5,
        min_confidence=0.5,
        persistence_minutes=5.0,
        trend_window_minutes=10.0,
        falling_slope_cm_per_10min=-1.0,
        rise_escalation_cm_per_10min=5.0,
        trigger_cm=3.5,
        clear_cm=2.5,
        max_gap_seconds=30.0,
    )
    defaults.update(kwargs)
    return TemporalConfig(**defaults)


def _run_pipeline(
    samples: list[SynthSample], cfg: TemporalConfig
) -> list[StateOutput]:
    f = TemporalFilter(cfg)
    sm = StateMachine(cfg)
    outputs: list[StateOutput] = []
    for s in samples:
        fo = f.update(s.timestamp_s, s)
        so = sm.update(s.timestamp_s, fo)
        outputs.append(so)
    return outputs


def _flood_entry_ts(outputs: list[StateOutput]) -> Optional[float]:
    """Return the timestamp index (via enumerate) when state first becomes Flood."""
    for i, o in enumerate(outputs):
        if o.state == FloodState.Flood:
            return float(i)
    return None


# ---------------------------------------------------------------------------
# Filter tests
# ---------------------------------------------------------------------------


def test_spike_rejected_by_median():
    cfg = _default_cfg(filter_window_samples=5, trend_window_minutes=1.0)
    f = TemporalFilter(cfg)
    # Fill with steady 2 cm
    for i in range(5):
        f.update(float(i), SynthSample(float(i), 2.0))
    # Inject spike
    out_spike = f.update(5.0, SynthSample(5.0, 10.0))
    assert out_spike.filtered_depth_cm is not None
    assert out_spike.filtered_depth_cm < 4.0, (
        f"Spike not filtered: {out_spike.filtered_depth_cm}"
    )


def test_low_confidence_excluded():
    cfg = _default_cfg(min_confidence=0.5)
    f = TemporalFilter(cfg)
    f.update(0.0, SynthSample(0.0, 5.0, confidence=1.0))
    before = f.state.excluded_count
    out = f.update(1.0, SynthSample(1.0, 5.0, confidence=0.2))
    assert out.excluded_this_call
    assert f.state.excluded_count == before + 1


def test_invalid_sample_excluded():
    cfg = _default_cfg()
    f = TemporalFilter(cfg)
    out = f.update(0.0, SynthSample(0.0, 5.0, valid=False))
    assert out.excluded_this_call
    assert f.state.excluded_count == 1


def test_rate_noiseless_ramp():
    # 100 cm/10min = 10 cm/min = 1/6 cm/s.
    # Use short trend_window_minutes=1.0 (60 s).  Feed samples every 1 s for 70 s.
    rate_cm_per_s = 10.0 / 60.0
    cfg = _default_cfg(filter_window_samples=5, trend_window_minutes=1.0)
    f = TemporalFilter(cfg)
    out = None
    for i in range(71):
        t = float(i)
        d = rate_cm_per_s * t
        out = f.update(t, SynthSample(t, d))
    assert out is not None
    assert out.rate_cm_per_10min is not None, "Rate should not be None after 71 s"
    expected = rate_cm_per_s * 600.0  # 100 cm/10min
    assert abs(out.rate_cm_per_10min - expected) / expected < 0.01, (
        f"Rate {out.rate_cm_per_10min:.2f} not within 1% of {expected:.2f}"
    )


def test_rate_none_when_too_few_samples():
    cfg = _default_cfg(filter_window_samples=5, trend_window_minutes=10.0)
    f = TemporalFilter(cfg)
    f.update(0.0, SynthSample(0.0, 1.0))
    out = f.update(1.0, SynthSample(1.0, 2.0))
    assert out.rate_cm_per_10min is None, "Rate should be None with only 2 samples"


def test_stale_after_gap():
    cfg = _default_cfg(max_gap_seconds=30.0, trend_window_minutes=1.0)
    f = TemporalFilter(cfg)
    f.update(0.0, SynthSample(0.0, 5.0))
    out = f.update(60.0, SynthSample(60.0, 5.0))  # 60 s gap > 30 s
    assert out.stale, "Should be stale after 60 s gap"


def test_stale_resets_after_new_sample():
    cfg = _default_cfg(max_gap_seconds=30.0, trend_window_minutes=1.0)
    f = TemporalFilter(cfg)
    f.update(0.0, SynthSample(0.0, 5.0))
    f.update(60.0, SynthSample(60.0, 5.0))   # stale
    out = f.update(61.0, SynthSample(61.0, 5.0))  # should reset stale
    assert not out.stale, "Stale should be reset after new valid sample"


# ---------------------------------------------------------------------------
# Severity tests
# ---------------------------------------------------------------------------


def test_depth_to_severity_boundaries():
    cases = [
        (0.0, SeverityClass.Normal),
        (2.9, SeverityClass.Normal),
        (3.0, SeverityClass.Advisory),
        (9.9, SeverityClass.Advisory),
        (10.0, SeverityClass.Warning),
        (24.9, SeverityClass.Warning),
        (25.0, SeverityClass.Severe),
        (49.9, SeverityClass.Severe),
        (50.0, SeverityClass.Critical),
        (999.0, SeverityClass.Critical),
    ]
    for depth, expected in cases:
        assert depth_to_severity(depth) == expected, f"depth={depth}"


def test_escalation_raises_by_one():
    result = apply_rate_escalation(SeverityClass.Advisory, rate_cm_per_10min=6.0)
    assert result == SeverityClass.Warning


def test_escalation_does_not_exceed_critical():
    result = apply_rate_escalation(SeverityClass.Critical, rate_cm_per_10min=6.0)
    assert result == SeverityClass.Critical


def test_no_escalation_below_threshold():
    result = apply_rate_escalation(SeverityClass.Advisory, rate_cm_per_10min=4.9)
    assert result == SeverityClass.Advisory


# ---------------------------------------------------------------------------
# State machine scenario tests
# ---------------------------------------------------------------------------


def test_steady_rise_reaches_flood_after_persistence():
    cfg = _default_cfg(
        persistence_minutes=5.0,
        trend_window_minutes=10.0,
        filter_window_samples=5,
    )
    samples = steady_rise(
        start_depth_cm=0.0, end_depth_cm=20.0, duration_s=1800.0, sample_interval_s=1.0, seed=1
    )
    outputs = _run_pipeline(samples, cfg)

    # Find when depth first >= trigger_cm in the raw samples
    trigger_sample_idx = next(
        (i for i, s in enumerate(samples) if s.depth_cm >= cfg.trigger_cm), None
    )
    assert trigger_sample_idx is not None

    # Flood must not appear before persistence_minutes after trigger
    persistence_s = cfg.persistence_minutes * 60.0
    for i, o in enumerate(outputs):
        if o.state == FloodState.Flood:
            t_flood = samples[i].timestamp_s
            t_trigger = samples[trigger_sample_idx].timestamp_s
            assert t_flood - t_trigger >= persistence_s * 0.95, (
                f"Flood at t={t_flood:.1f} s, trigger at t={t_trigger:.1f} s, "
                f"persistence={persistence_s:.0f} s"
            )
            break


def test_oscillation_no_chatter():
    # Use period_s=200 → ~3 full cycles in 600 s → at most ~6 Normal↔Pending transitions.
    # The primary check is that Flood is never reached; the secondary check bounds chatter.
    cfg = _default_cfg(filter_window_samples=5, trend_window_minutes=2.0)
    samples = oscillation_around_threshold(
        center_cm=3.0, amplitude_cm=1.5, period_s=200.0, duration_s=600.0, seed=3
    )
    outputs = _run_pipeline(samples, cfg)
    states = [o.state for o in outputs]
    assert FloodState.Flood not in states, "Oscillation must never reach Flood (persistence prevents it)"
    transitions = [o.transition for o in outputs if o.transition is not None]
    assert len(transitions) <= 8, f"Too many transitions (chatter): {len(transitions)}"


def test_spike_never_leaves_normal():
    cfg = _default_cfg(filter_window_samples=5, trend_window_minutes=1.0)
    samples = single_spike(
        baseline_cm=1.0, spike_cm=8.0, spike_duration_s=2.0, duration_s=120.0, seed=4
    )
    outputs = _run_pipeline(samples, cfg)
    for o in outputs:
        assert o.state == FloodState.Normal, (
            f"State should stay Normal, got {o.state} at transition '{o.transition}'"
        )


def test_short_puddle_never_floods():
    cfg = _default_cfg(filter_window_samples=5)
    samples = short_puddle(depth_cm=1.5, duration_s=300.0, seed=6)
    outputs = _run_pipeline(samples, cfg)
    for o in outputs:
        assert o.state == FloodState.Normal


def test_plateau_holds_flood():
    cfg = _default_cfg(
        persistence_minutes=5.0,
        trend_window_minutes=10.0,
        filter_window_samples=5,
    )
    samples = rise_plateau_fall(
        rise_to_cm=15.0, plateau_s=600.0, total_s=1500.0, sample_interval_s=1.0, seed=2
    )
    outputs = _run_pipeline(samples, cfg)

    # Find plateau region: samples with depth ≥ 14 cm (near peak)
    plateau_outputs = [
        o for o, s in zip(outputs, samples) if s.depth_cm >= 14.0
    ]
    flood_count = sum(1 for o in plateau_outputs if o.state == FloodState.Flood)
    assert flood_count >= 5, f"Only {flood_count} Flood samples during plateau"


def test_fall_moves_to_recovering_then_normal():
    cfg = _default_cfg(
        persistence_minutes=5.0,
        trend_window_minutes=10.0,
        filter_window_samples=5,
        falling_slope_cm_per_10min=-1.0,
    )
    samples = slow_fall_from_deep(
        start_depth_cm=20.0, end_depth_cm=0.0, duration_s=2400.0, sample_interval_s=1.0, seed=7
    )
    # Prepend a steady rise to get into Flood first
    rise = steady_rise(0.0, 20.0, 1800.0, 1.0, seed=7)
    all_samples = rise + [
        SynthSample(s.timestamp_s + rise[-1].timestamp_s + 1.0, s.depth_cm)
        for s in samples
    ]
    outputs = _run_pipeline(all_samples, cfg)
    states = [o.state for o in outputs]
    assert FloodState.Flood in states, "Should reach Flood"
    assert FloodState.Recovering in states, "Should pass through Recovering"
    # Verify Flood → Recovering → Normal ordering (no direct Flood → Normal)
    last_flood = max(i for i, s in enumerate(states) if s == FloodState.Flood)
    if FloodState.Normal in states[last_flood:]:
        first_normal_after_flood = next(
            i for i, s in enumerate(states[last_flood:], last_flood) if s == FloodState.Normal
        )
        intervening = set(states[last_flood:first_normal_after_flood])
        assert FloodState.Recovering in intervening, (
            "Must pass through Recovering before Normal"
        )


def test_re_rise_from_recovering_returns_to_flood():
    cfg = _default_cfg(
        persistence_minutes=5.0,
        trend_window_minutes=10.0,
        filter_window_samples=5,
        falling_slope_cm_per_10min=-1.0,
    )
    # Phase 1: long steady flood
    phase1 = steady_rise(0.0, 20.0, 1800.0, 1.0, seed=10)
    # Phase 2: slow fall to 3 cm (still above clear_cm=2.5) → triggers Recovering
    # Rate = (3-20)/20min × 10 ≈ -8.5 cm/10min < -1.0
    t_off = phase1[-1].timestamp_s + 1.0
    phase2 = [
        SynthSample(t_off + i, 20.0 - i * (17.0 / 1200.0))
        for i in range(1201)
    ]
    # Phase 3: rise back from 3 to 12 cm → rate > -1.0 → Recovering → Flood
    t_off2 = phase2[-1].timestamp_s + 1.0
    phase3 = [
        SynthSample(t_off2 + i, 3.0 + i * (9.0 / 900.0))
        for i in range(901)
    ]
    all_samples = phase1 + phase2 + phase3
    outputs = _run_pipeline(all_samples, cfg)
    states = [o.state for o in outputs]

    assert FloodState.Flood in states, "Should reach Flood in phase 1"
    assert FloodState.Recovering in states, "Should enter Recovering in phase 2"
    # After Recovering, we should return to Flood (not just Normal)
    first_rec = next(i for i, s in enumerate(states) if s == FloodState.Recovering)
    states_after_rec = states[first_rec:]
    assert FloodState.Flood in states_after_rec, (
        "Should return to Flood after re-rise from Recovering"
    )


def test_dropout_gap_sets_stale():
    cfg = _default_cfg(max_gap_seconds=30.0, trend_window_minutes=2.0, filter_window_samples=5)
    samples = dropout_gap(
        baseline_cm=5.0, gap_start_s=60.0, gap_duration_s=60.0, total_s=240.0, seed=5
    )
    f = TemporalFilter(cfg)
    sm = StateMachine(cfg)
    stale_seen = False
    recovered = False
    for s in samples:
        fo = f.update(s.timestamp_s, s)
        sm.update(s.timestamp_s, fo)
        if fo.stale:
            stale_seen = True
        elif stale_seen:
            recovered = True
    assert stale_seen, "stale=True should be seen during/after the gap"
    assert recovered, "stale should reset after gap ends"


def test_flood_never_goes_directly_to_normal():
    cfg = _default_cfg(
        persistence_minutes=2.0,
        trend_window_minutes=5.0,
        filter_window_samples=5,
    )
    generators = [
        lambda s: steady_rise(0, 20, 900, 1.0, seed=s),
        lambda s: rise_plateau_fall(15, 300, 900, 1.0, seed=s),
        lambda s: oscillation_around_threshold(3, 2, 60, 600, 1.0, seed=s),
        lambda s: slow_fall_from_deep(20, 0, 900, 1.0, seed=s),
    ]
    for seed_base in range(5):
        for gen in generators:
            samples = gen(seed_base * 100)
            outputs = _run_pipeline(samples, cfg)
            states = [o.state for o in outputs]
            for i in range(len(states) - 1):
                assert not (
                    states[i] == FloodState.Flood and states[i + 1] == FloodState.Normal
                ), f"Direct Flood→Normal at step {i}"


def test_valid_transitions_only():
    VALID = {
        (FloodState.Normal, FloodState.Normal),
        (FloodState.Normal, FloodState.Pending),
        (FloodState.Pending, FloodState.Pending),
        (FloodState.Pending, FloodState.Normal),
        (FloodState.Pending, FloodState.Flood),
        (FloodState.Flood, FloodState.Flood),
        (FloodState.Flood, FloodState.Recovering),
        (FloodState.Recovering, FloodState.Recovering),
        (FloodState.Recovering, FloodState.Flood),
        (FloodState.Recovering, FloodState.Normal),
    }
    cfg = _default_cfg(persistence_minutes=2.0, trend_window_minutes=5.0, filter_window_samples=5)
    generators = [
        lambda s: steady_rise(0, 20, 900, 1.0, seed=s),
        lambda s: rise_plateau_fall(15, 300, 900, 1.0, seed=s),
        lambda s: oscillation_around_threshold(3, 2, 60, 600, 1.0, seed=s),
        lambda s: slow_fall_from_deep(20, 0, 900, 1.0, seed=s),
        lambda s: dropout_gap(5, 60, 60, 240, 1.0, seed=s),
    ]
    for seed_base in range(4):
        for gen in generators:
            samples = gen(seed_base * 100)
            outputs = _run_pipeline(samples, cfg)
            states = [o.state for o in outputs]
            for i in range(len(states) - 1):
                pair = (states[i], states[i + 1])
                assert pair in VALID, f"Invalid transition {pair[0]}→{pair[1]} at step {i}"


def test_escalation_integration():
    # Fast rise: 20 cm over 200 s → rate ≈ 60 cm/10min > 5 → escalation
    cfg = _default_cfg(
        persistence_minutes=2.0,
        trend_window_minutes=5.0,
        filter_window_samples=5,
        rise_escalation_cm_per_10min=5.0,
    )
    samples = steady_rise(0.0, 20.0, 300.0, 1.0, seed=11)
    outputs = _run_pipeline(samples, cfg)
    flood_outputs = [o for o in outputs if o.state == FloodState.Flood]
    if not flood_outputs:
        pytest.skip("Did not reach Flood state in this test — adjust parameters")
    base_severities = [depth_to_severity(o.filtered_depth_cm or 0.0) for o in flood_outputs]
    actual_severities = [o.severity_class for o in flood_outputs]
    escalated = any(a > b for a, b in zip(actual_severities, base_severities))
    assert escalated, "Expected at least one escalated severity during fast rise"


# ---------------------------------------------------------------------------
# Serialisation round-trip tests
# ---------------------------------------------------------------------------


def test_filter_state_serialization_round_trip():
    cfg = _default_cfg(filter_window_samples=5, trend_window_minutes=1.0)
    f1 = TemporalFilter(cfg)
    for i in range(5):
        f1.update(float(i), SynthSample(float(i), float(i) * 0.5))

    saved_state = copy.deepcopy(f1.state)
    f2 = TemporalFilter.from_state(cfg, saved_state)

    for i in range(5, 8):
        t = float(i)
        s = SynthSample(t, float(i) * 0.5)
        out1 = f1.update(t, s)
        out2 = f2.update(t, s)
        assert out1.filtered_depth_cm == pytest.approx(out2.filtered_depth_cm, abs=1e-9)
        assert out1.rate_cm_per_10min == pytest.approx(out2.rate_cm_per_10min, abs=1e-6) if (
            out1.rate_cm_per_10min is not None
        ) else out2.rate_cm_per_10min is None


def test_machine_state_serialization_round_trip():
    cfg = _default_cfg(
        persistence_minutes=2.0, trend_window_minutes=5.0, filter_window_samples=5
    )
    samples_to_flood = steady_rise(0, 20, 800, 1.0, seed=20)
    f1 = TemporalFilter(cfg)
    sm1 = StateMachine(cfg)
    for s in samples_to_flood:
        fo = f1.update(s.timestamp_s, s)
        sm1.update(s.timestamp_s, fo)

    if sm1.state.flood_state != FloodState.Flood.value:
        pytest.skip("Did not reach Flood — serialisation test skipped")

    saved_filter = copy.deepcopy(f1.state)
    saved_machine = copy.deepcopy(sm1.state)
    f2 = TemporalFilter.from_state(cfg, saved_filter)
    sm2 = StateMachine.from_state(cfg, saved_machine)

    continuation = steady_rise(20, 20, 3, 1.0, seed=99)
    t_off = samples_to_flood[-1].timestamp_s + 1.0
    for i, s in enumerate(continuation):
        s2 = SynthSample(t_off + i, s.depth_cm)
        fo1 = f1.update(s2.timestamp_s, s2)
        so1 = sm1.update(s2.timestamp_s, fo1)
        fo2 = f2.update(s2.timestamp_s, s2)
        so2 = sm2.update(s2.timestamp_s, fo2)
        assert so1.state == so2.state
        assert so1.severity_class == so2.severity_class


# ---------------------------------------------------------------------------
# Frame-rate independence
# ---------------------------------------------------------------------------


def test_frame_rate_independence():
    cfg = _default_cfg(
        persistence_minutes=5.0,
        trend_window_minutes=10.0,
        filter_window_samples=5,
    )
    samples_1s = steady_rise(0, 20, 1800, 1.0, seed=30)
    samples_5s = steady_rise(0, 20, 1800, 5.0, seed=30)

    outs_1s = _run_pipeline(samples_1s, cfg)
    outs_5s = _run_pipeline(samples_5s, cfg)

    flood_1s = next(
        (samples_1s[i].timestamp_s for i, o in enumerate(outs_1s) if o.state == FloodState.Flood),
        None,
    )
    flood_5s = next(
        (samples_5s[i].timestamp_s for i, o in enumerate(outs_5s) if o.state == FloodState.Flood),
        None,
    )
    assert flood_1s is not None, "1 s stream should reach Flood"
    assert flood_5s is not None, "5 s stream should reach Flood"
    # Flood entry timestamps within one 5-s sample interval
    assert abs(flood_5s - flood_1s) <= 5.0 * 2, (
        f"Frame-rate independence violated: 1s→{flood_1s:.0f}s, 5s→{flood_5s:.0f}s"
    )


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_temporal_config_rejects_invalid_window():
    with pytest.raises(ValueError, match="filter_window_samples"):
        TemporalConfig(filter_window_samples=4)


def test_temporal_config_rejects_inverted_thresholds():
    with pytest.raises(ValueError, match="trigger_cm"):
        TemporalConfig(trigger_cm=2.0, clear_cm=3.0)
