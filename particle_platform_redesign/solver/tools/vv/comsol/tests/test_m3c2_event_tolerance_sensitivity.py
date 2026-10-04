"""Focused checks for the predeclared M3-C2 event-tolerance gate."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from tools.vv.comsol.evaluate_m3c2_event_tolerance_sensitivity import (
    compare_event_results,
)


def _events(*, position_delta_m: float = 0.0, first_facet: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        particle_id=np.asarray([1, 3]),
        event_ordinal=np.asarray([1, 1]),
        candidate_facet_id=np.asarray([first_facet, 3, 4]),
        candidate_offset=np.asarray([0, 2, 3]),
        primary_facet_id=np.asarray([2, 4]),
        boundary_id=np.asarray([6, 7]),
        material_id=np.asarray([0, 0]),
        law_id=np.asarray(["stick", "escape"]),
        outcome=np.asarray(["stuck", "escaped"]),
        position_m=np.asarray([[0.1 + position_delta_m, 0.2], [0.3, 0.4]]),
        time_s=np.asarray([0.25 + 3.0e-12, 0.5]),
        position_budget_m=np.asarray([1.0e-12, 2.0e-12]),
        time_budget_s=np.asarray([1.0e-12, 2.0e-12]),
    )


def _result(events: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        read_boundary_events=lambda: events,
        read_failure_events=lambda: SimpleNamespace(particle_id=np.empty(0, dtype=np.int64)),
    )


def test_predeclared_event_budget_gate_accepts_bounded_differences() -> None:
    baseline = _events()
    baseline.time_s[0] = 0.25
    candidate = _events(position_delta_m=2.0e-12)

    report = compare_event_results(
        _result(baseline), _result(candidate), normalized_ratio_limit=4.0
    )

    assert report["status"] == "PASS"
    assert report["maximum_position_delta_over_max_recorded_budget"] == pytest.approx(2.0, rel=2e-6)
    assert report["maximum_time_delta_over_max_recorded_budget"] == pytest.approx(3.0, rel=5e-6)


def test_event_budget_and_exact_facet_identity_fail_closed() -> None:
    baseline = _events()
    baseline.time_s[0] = 0.25
    outside_budget = compare_event_results(
        _result(baseline),
        _result(_events(position_delta_m=5.0e-12)),
        normalized_ratio_limit=4.0,
    )
    facet_mismatch = compare_event_results(
        _result(baseline),
        _result(_events(first_facet=99)),
        normalized_ratio_limit=4.0,
    )

    outside_gates = cast(dict[str, Any], outside_budget["gates"])
    facet_gates = cast(dict[str, Any], facet_mismatch["gates"])
    assert outside_budget["status"] == "FAIL"
    assert outside_gates["position_budget"] is False
    assert facet_mismatch["status"] == "FAIL"
    assert facet_gates["event_identity_exact"] is False
