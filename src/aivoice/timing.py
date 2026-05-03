from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DurationAction = Literal["accept", "atempo", "atempo_warn", "retry_or_hard_duration", "split_or_merge"]


def duration_tolerance_seconds(duration: float, ratio: float = 0.08, cap_seconds: float = 0.4) -> float:
    return max(min(max(duration, 0.0) * ratio, cap_seconds), 0.0)


@dataclass(frozen=True)
class DurationFit:
    target_duration: float
    measured_duration: float
    error_seconds: float
    error_ratio: float
    tolerance_seconds: float
    action: DurationAction

    @property
    def within_tolerance(self) -> bool:
        return self.action == "accept"


def evaluate_duration_fit(
    target_duration: float,
    measured_duration: float,
    tolerance_ratio: float = 0.08,
    tolerance_cap_seconds: float = 0.4,
) -> DurationFit:
    target = max(target_duration, 0.001)
    measured = max(measured_duration, 0.0)
    error_seconds = abs(measured - target)
    error_ratio = error_seconds / target
    tolerance_seconds = duration_tolerance_seconds(
        target,
        ratio=tolerance_ratio,
        cap_seconds=tolerance_cap_seconds,
    )

    if error_seconds <= tolerance_seconds:
        action: DurationAction = "accept"
    elif error_ratio <= 0.08:
        action = "atempo"
    elif error_ratio <= 0.15:
        action = "atempo_warn"
    elif error_ratio <= 0.25:
        action = "retry_or_hard_duration"
    else:
        action = "split_or_merge"

    return DurationFit(
        target_duration=target,
        measured_duration=measured,
        error_seconds=error_seconds,
        error_ratio=error_ratio,
        tolerance_seconds=tolerance_seconds,
        action=action,
    )
