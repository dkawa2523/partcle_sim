"""Optional detailed-timing primitives shared by solver orchestration."""

from __future__ import annotations

import time


def add_timing(
    timing_accumulator: dict[str, float] | None,
    key: str,
    elapsed_s: float,
) -> None:
    if timing_accumulator is None:
        return
    timing_accumulator[key] = float(timing_accumulator.get(key, 0.0)) + float(
        max(0.0, elapsed_s)
    )


def detailed_timer_start(
    timing_accumulator: dict[str, float] | None,
) -> float:
    return time.perf_counter() if timing_accumulator is not None else 0.0


def add_detailed_timing_since(
    timing_accumulator: dict[str, float] | None,
    key: str,
    started_at: float,
) -> None:
    if timing_accumulator is not None:
        add_timing(
            timing_accumulator,
            key,
            time.perf_counter() - float(started_at),
        )
