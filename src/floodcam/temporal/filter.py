"""Rolling-median temporal filter with rate-of-rise estimator.

All logic is timestamp-based (float seconds) so behaviour is independent of
frame rate and replay speed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from floodcam.config import TemporalConfig


@dataclass
class FilterState:
    """Serialisable filter state — pass to TemporalFilter.from_state() to resume."""

    # (timestamp_s, depth_cm) pairs in the trailing trend_window; used for both
    # the rolling median (last filter_window_samples entries) and the rate fit
    # (all entries within trend_window_minutes * 60 s of the latest timestamp).
    accepted: list[tuple[float, float]] = field(default_factory=list)
    excluded_count: int = 0
    stale: bool = False
    last_accepted_timestamp: Optional[float] = None


@dataclass
class FilterOutput:
    filtered_depth_cm: Optional[float]   # None only when accepted is empty
    rate_cm_per_10min: Optional[float]   # None when < 3 samples or < 50% window covered
    stale: bool
    excluded_this_call: bool             # True if the current sample was excluded


class TemporalFilter:
    """Rolling median + rate-of-rise estimator for a single camera stream."""

    def __init__(self, cfg: TemporalConfig) -> None:
        self._cfg = cfg
        self._state = FilterState()

    @property
    def state(self) -> FilterState:
        return self._state

    @classmethod
    def from_state(cls, cfg: TemporalConfig, state: FilterState) -> "TemporalFilter":
        f = cls(cfg)
        f._state = state
        return f

    def update(self, timestamp: float, estimate: Any) -> FilterOutput:
        """Process one sample (any object with .valid, .confidence, .depth_cm).

        Exclusion: estimate.valid is False OR estimate.confidence < min_confidence.
        Gap: gap > max_gap_seconds clears history and sets stale=True.
        """
        valid = getattr(estimate, "valid", True)
        confidence = getattr(estimate, "confidence", 1.0)
        excluded = not valid or confidence < self._cfg.min_confidence

        if excluded:
            self._state.excluded_count += 1
            filtered = self._current_median()
            rate = self._compute_rate(
                self._state.last_accepted_timestamp
                if self._state.last_accepted_timestamp is not None
                else timestamp
            )
            return FilterOutput(filtered, rate, self._state.stale, excluded_this_call=True)

        # Gap detection (only for accepted samples)
        if (
            self._state.last_accepted_timestamp is not None
            and timestamp - self._state.last_accepted_timestamp > self._cfg.max_gap_seconds
        ):
            self._state.accepted = []
            self._state.stale = True
        else:
            self._state.stale = False

        depth = float(getattr(estimate, "depth_cm", 0.0))
        self._state.accepted.append((timestamp, depth))
        self._state.last_accepted_timestamp = timestamp

        # Trim to trend window (keeps samples for rate computation)
        window_s = self._cfg.trend_window_minutes * 60.0
        cutoff = timestamp - window_s
        self._state.accepted = [
            (t, d) for t, d in self._state.accepted if t >= cutoff
        ]

        filtered = self._current_median()
        rate = self._compute_rate(timestamp)

        return FilterOutput(filtered, rate, self._state.stale, excluded_this_call=False)

    def _current_median(self) -> Optional[float]:
        if not self._state.accepted:
            return None
        recent = self._state.accepted[-self._cfg.filter_window_samples :]
        return float(np.median([d for _, d in recent]))

    def _compute_rate(self, latest_ts: float) -> Optional[float]:
        window_s = self._cfg.trend_window_minutes * 60.0
        cutoff = latest_ts - window_s
        window_samples = [(t, d) for t, d in self._state.accepted if t >= cutoff]

        if len(window_samples) < 3:
            return None

        t_start, t_end = window_samples[0][0], window_samples[-1][0]
        span = t_end - t_start
        if span < 0.5 * window_s:
            return None

        ts = np.array([t for t, _ in window_samples])
        depths = np.array([d for _, d in window_samples])
        t_rel = ts - ts[0]
        slope_cm_per_s = float(np.polyfit(t_rel, depths, 1)[0])
        return slope_cm_per_s * 600.0  # convert to cm per 10 min
