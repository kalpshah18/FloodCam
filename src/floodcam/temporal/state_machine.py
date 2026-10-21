"""Flood state machine with four states and timestamp-based hysteresis.

Transition table (implemented exactly):
  Normal    → Pending    : filtered_depth >= trigger_cm
  Pending   → Normal     : filtered_depth < clear_cm
  Pending   → Flood      : stayed >= clear_cm for persistence_minutes AND rate > falling_slope
  Flood     → Recovering : rate <= falling_slope_cm_per_10min  (rate=None → no transition)
  Recovering→ Flood      : rate > falling_slope AND filtered_depth >= clear_cm
  Recovering→ Normal     : filtered_depth < clear_cm
  Flood     → Normal     : NEVER (must go through Recovering)

Note: the Recovering state is a dashboard demonstration only — not a validated claim.
Short clips cannot validate minutes-long draining trends (see CLAUDE.md and DECISIONS.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from floodcam.config import TemporalConfig
from floodcam.temporal.filter import FilterOutput
from floodcam.temporal.severity import SeverityClass, apply_rate_escalation, depth_to_severity


class FloodState(str, Enum):
    Normal = "Normal"
    Pending = "Pending"
    Flood = "Flood"
    Recovering = "Recovering"


@dataclass
class MachineState:
    """Serialisable state — pass to StateMachine.from_state() to resume."""

    flood_state: str = "Normal"
    pending_since: Optional[float] = None  # timestamp when Pending started


@dataclass
class StateOutput:
    state: FloodState
    severity_class: SeverityClass
    filtered_depth_cm: Optional[float]
    rate_cm_per_10min: Optional[float]
    stale: bool
    transition: Optional[str]  # e.g. "Normal→Pending" or None
    reason: str


class StateMachine:
    """Pure, deterministic flood state machine for one camera stream."""

    def __init__(self, cfg: TemporalConfig) -> None:
        self._cfg = cfg
        self._mstate = MachineState()

    @classmethod
    def from_state(cls, cfg: TemporalConfig, state: MachineState) -> "StateMachine":
        sm = cls(cfg)
        sm._mstate = MachineState(
            flood_state=state.flood_state,
            pending_since=state.pending_since,
        )
        return sm

    @property
    def state(self) -> MachineState:
        return self._mstate

    def update(self, timestamp: float, filter_output: FilterOutput) -> StateOutput:
        """Apply one FilterOutput to the state machine.  Deterministic."""
        current = FloodState(self._mstate.flood_state)
        depth = filter_output.filtered_depth_cm
        rate = filter_output.rate_cm_per_10min
        stale = filter_output.stale

        # When stale or no depth data: no transitions
        if stale or depth is None:
            sev = self._severity(depth, rate)
            return StateOutput(
                state=current,
                severity_class=sev,
                filtered_depth_cm=depth,
                rate_cm_per_10min=rate,
                stale=stale,
                transition=None,
                reason="stale" if stale else "insufficient_data",
            )

        cfg = self._cfg
        new_state = current
        transition: Optional[str] = None
        reason = "no_transition"

        if current == FloodState.Normal:
            if depth >= cfg.trigger_cm:
                new_state = FloodState.Pending
                self._mstate.pending_since = timestamp
                transition = "Normal→Pending"
                reason = f"depth {depth:.2f} >= trigger {cfg.trigger_cm}"

        elif current == FloodState.Pending:
            if depth < cfg.clear_cm:
                # Fell below clear: reset to Normal (NOT Recovering — no flood was declared)
                new_state = FloodState.Normal
                self._mstate.pending_since = None
                transition = "Pending→Normal"
                reason = f"depth {depth:.2f} < clear {cfg.clear_cm}"
            elif (
                self._mstate.pending_since is not None
                and (timestamp - self._mstate.pending_since) >= cfg.persistence_minutes * 60.0
                and rate is not None
                and rate > cfg.falling_slope_cm_per_10min
            ):
                elapsed = timestamp - self._mstate.pending_since
                new_state = FloodState.Flood
                self._mstate.pending_since = None
                transition = "Pending→Flood"
                reason = f"persisted {elapsed:.0f}s, rate {rate:.2f} cm/10min"

        elif current == FloodState.Flood:
            # Flood → Normal is FORBIDDEN; must go through Recovering
            if rate is not None and rate <= cfg.falling_slope_cm_per_10min:
                new_state = FloodState.Recovering
                transition = "Flood→Recovering"
                reason = f"rate {rate:.2f} <= falling threshold {cfg.falling_slope_cm_per_10min}"

        elif current == FloodState.Recovering:
            if depth < cfg.clear_cm:
                new_state = FloodState.Normal
                transition = "Recovering→Normal"
                reason = f"depth {depth:.2f} < clear {cfg.clear_cm}"
            elif rate is not None and rate > cfg.falling_slope_cm_per_10min and depth >= cfg.clear_cm:
                new_state = FloodState.Flood
                transition = "Recovering→Flood"
                reason = f"rate {rate:.2f} rising and depth {depth:.2f} >= clear"

        self._mstate.flood_state = new_state.value

        sev = self._severity(depth, rate)
        return StateOutput(
            state=new_state,
            severity_class=sev,
            filtered_depth_cm=depth,
            rate_cm_per_10min=rate,
            stale=stale,
            transition=transition,
            reason=reason,
        )

    def _severity(self, depth: Optional[float], rate: Optional[float]) -> SeverityClass:
        base = depth_to_severity(depth if depth is not None else 0.0)
        return apply_rate_escalation(base, rate, self._cfg.rise_escalation_cm_per_10min)
