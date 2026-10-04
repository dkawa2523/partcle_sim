"""Focused checks for the dimensionally consistent event-tolerance gate."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from tools.vv.comsol.evaluate_m3c2_event_tolerance_cross_band import (
    compare_event_results,
)


def _events(
    *,
    time_s: float,
    position_budget_m: float,
    time_budget_s: float,
    normal: tuple[float, float] = (0.0, -1.0),
    velocity_pre_m_s: tuple[float, float] = (-0.83503917, -9.47729851),
    position_offset_m: float = 0.0,
    first_facet: int = 12,
) -> SimpleNamespace:
    return SimpleNamespace(
        particle_id=np.asarray([150]),
        event_ordinal=np.asarray([1]),
        candidate_facet_id=np.asarray([first_facet]),
        candidate_offset=np.asarray([0, 1]),
        primary_facet_id=np.asarray([12]),
        boundary_id=np.asarray([6]),
        material_id=np.asarray([0]),
        law_id=np.asarray(["stick"]),
        outcome=np.asarray(["stuck"]),
        position_m=np.asarray([[0.15 + position_offset_m, 0.02]]),
        time_s=np.asarray([time_s]),
        position_budget_m=np.asarray([position_budget_m]),
        time_budget_s=np.asarray([time_budget_s]),
        normal=np.asarray([normal]),
        velocity_pre_m_s=np.asarray([velocity_pre_m_s]),
    )


def _result(events: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        read_boundary_events=lambda: events,
        read_failure_events=lambda: SimpleNamespace(particle_id=np.empty(0, dtype=np.int64)),
    )


def test_p150_dimensional_stopping_time_pattern_passes() -> None:
    reference = _events(
        time_s=0.011200061328110897,
        position_budget_m=7.676713860751638e-11,
        time_budget_s=1.5916244433513635e-16,
        velocity_pre_m_s=(-0.83503917, -9.47729851325689),
    )
    candidate = _events(
        time_s=0.011200061336211002,
        position_budget_m=7.68031776264627e-12,
        time_budget_s=1.5916244445024578e-16,
        velocity_pre_m_s=(-0.83503899, -9.477298774141534),
    )

    report = compare_event_results(_result(reference), _result(candidate))

    assert report["status"] == "PASS"
    expected_allowance = (
        1.5916244433513635e-16
        + 1.5916244445024578e-16
        + 7.676713860751638e-11 / (0.5 * 9.47729851325689)
        + 7.68031776264627e-12 / (0.5 * 9.477298774141534)
    )
    expected_ratio = (0.011200061336211002 - 0.011200061328110897) / expected_allowance
    assert report["maximum_time_delta_over_composed_stopping_time_allowance"] == pytest.approx(
        expected_ratio
    )


def test_position_uses_factor_one_sum_of_both_budgets() -> None:
    reference = _events(time_s=0.2, position_budget_m=2.0e-12, time_budget_s=1.0e-15)
    within = _events(
        time_s=0.2,
        position_budget_m=3.0e-12,
        time_budget_s=1.0e-15,
        position_offset_m=4.0e-12,
    )
    outside = _events(
        time_s=0.2,
        position_budget_m=3.0e-12,
        time_budget_s=1.0e-15,
        position_offset_m=6.0e-12,
    )

    within_report = compare_event_results(_result(reference), _result(within))
    outside_report = compare_event_results(_result(reference), _result(outside))

    assert within_report["status"] == "PASS"
    assert outside_report["status"] == "FAIL"
    gates = cast(dict[str, Any], outside_report["gates"])
    assert gates["position_composed_budget"] is False


def test_time_uses_factor_one_composed_stopping_time_allowance() -> None:
    reference = _events(time_s=0.2, position_budget_m=5.0e-11, time_budget_s=1.0e-15)
    candidate = _events(
        time_s=0.2 + 3.0e-11,
        position_budget_m=5.0e-11,
        time_budget_s=1.0e-15,
    )

    report = compare_event_results(_result(reference), _result(candidate))

    gates = cast(dict[str, Any], report["gates"])
    assert report["status"] == "FAIL"
    assert gates["cross_band_time_allowance"] is False
    assert cast(float, report["maximum_time_delta_over_composed_stopping_time_allowance"]) > 1.0


@pytest.mark.parametrize(
    ("candidate_normal", "candidate_velocity"),
    [
        ((1.0, 0.0), (-0.83503899, -9.47729877)),
        ((0.0, -1.0), (-0.83503899, 0.0)),
        ((0.0, -1.0), (-0.83503899, 9.47729877)),
    ],
)
def test_normal_mismatch_and_tangent_event_fail_closed(
    candidate_normal: tuple[float, float],
    candidate_velocity: tuple[float, float],
) -> None:
    reference = _events(time_s=0.2, position_budget_m=1.0e-10, time_budget_s=1.0e-15)
    candidate = _events(
        time_s=0.2,
        position_budget_m=1.0e-11,
        time_budget_s=1.0e-15,
        normal=candidate_normal,
        velocity_pre_m_s=candidate_velocity,
    )

    report = compare_event_results(_result(reference), _result(candidate))

    gates = cast(dict[str, Any], report["gates"])
    assert report["status"] == "FAIL"
    assert gates["cross_band_time_allowance"] is False
    assert report["normal_mismatches"] == ([[150, 1]] if candidate_normal != (0.0, -1.0) else [])


def test_normal_speed_mismatch_fails_closed() -> None:
    reference = _events(time_s=0.2, position_budget_m=1.0e-10, time_budget_s=1.0e-15)
    candidate = _events(
        time_s=0.2,
        position_budget_m=1.0e-11,
        time_budget_s=1.0e-15,
        velocity_pre_m_s=(-0.83503899, -9.0),
    )

    report = compare_event_results(_result(reference), _result(candidate))

    gates = cast(dict[str, Any], report["gates"])
    assert report["status"] == "FAIL"
    assert gates["normal_speed_agreement"] is False
    assert report["normal_speed_mismatches"] == [[150, 1]]


def test_corner_candidate_facet_set_mismatch_fails_closed() -> None:
    reference = _events(time_s=0.2, position_budget_m=1.0e-10, time_budget_s=1.0e-15)
    candidate = _events(
        time_s=0.2,
        position_budget_m=1.0e-11,
        time_budget_s=1.0e-15,
        first_facet=99,
    )

    report = compare_event_results(_result(reference), _result(candidate))

    gates = cast(dict[str, Any], report["gates"])
    assert report["status"] == "FAIL"
    assert gates["event_identity_exact"] is False
