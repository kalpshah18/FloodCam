"""Temporal filter, severity classification, and flood state machine."""

from floodcam.temporal.filter import FilterOutput, FilterState, TemporalFilter
from floodcam.temporal.severity import SeverityClass, apply_rate_escalation, depth_to_severity
from floodcam.temporal.state_machine import FloodState, MachineState, StateMachine, StateOutput

__all__ = [
    "TemporalFilter",
    "FilterState",
    "FilterOutput",
    "SeverityClass",
    "depth_to_severity",
    "apply_rate_escalation",
    "FloodState",
    "StateMachine",
    "MachineState",
    "StateOutput",
]
