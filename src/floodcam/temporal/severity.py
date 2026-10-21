"""Severity classification: depth-based class + rate-of-rise escalation."""

from __future__ import annotations

from enum import IntEnum
from typing import Optional

# Depth upper bounds for each class (cm).  depth < _UPPER_CM[i] → class i.
_UPPER_CM: list[float] = [3.0, 10.0, 25.0, 50.0, float("inf")]


class SeverityClass(IntEnum):
    Normal = 0
    Advisory = 1
    Warning = 2
    Severe = 3
    Critical = 4


def depth_to_severity(depth_cm: float) -> SeverityClass:
    """Map a depth value to its base severity class (no rate escalation)."""
    for i, upper in enumerate(_UPPER_CM):
        if depth_cm < upper:
            return SeverityClass(i)
    return SeverityClass.Critical


def apply_rate_escalation(
    base: SeverityClass,
    rate_cm_per_10min: Optional[float],
    threshold: float = 5.0,
) -> SeverityClass:
    """Escalate one class when rate >= threshold. Never exceed Critical."""
    if rate_cm_per_10min is not None and rate_cm_per_10min >= threshold:
        return SeverityClass(min(int(base) + 1, int(SeverityClass.Critical)))
    return base
