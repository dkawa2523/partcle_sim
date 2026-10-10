"""Terminal status alone cannot distinguish axis, inlet, and other causes."""

from __future__ import annotations

import pytest
from tools.vv.comsol.boundary_response_mapping import (
    ActualBoundaryResponse,
    resolve_terminal_boundary,
)


def test_observed_boundary_ids_select_actual_response_not_a_status_dictionary() -> None:
    actual = {
        1: ActualBoundaryResponse("held", "axis"),
        2: ActualBoundaryResponse("held", "inlet"),
        3: ActualBoundaryResponse("escaped", "outlet"),
    }
    result = resolve_terminal_boundary("held", [1], actual, True)
    assert result.classification == "SUPPORTED"
    assert result.semantic_group == "axis"
    assert result.observed_boundary_ids == (1,)
    assert result.identification == "observed_ids"
    assert resolve_terminal_boundary("held", [], actual, True).classification == "AMBIGUOUS"


def test_unique_actual_group_identifies_group_without_inventing_boundary_id() -> None:
    actual = {
        2: ActualBoundaryResponse("held", "inlet"),
        4: ActualBoundaryResponse("held", "inlet"),
        1: ActualBoundaryResponse("active", "axis"),
    }
    result = resolve_terminal_boundary("held", [], actual, True)
    assert result.classification == "SUPPORTED"
    assert result.semantic_group == "inlet"
    assert result.observed_boundary_ids == ()
    assert result.identification == "group_only"


@pytest.mark.parametrize("ids", [[1, 2], [99], [3], []])
def test_corner_unknown_id_response_mismatch_and_missing_response_are_unresolved(
    ids: list[int],
) -> None:
    actual = {
        1: ActualBoundaryResponse("held", "axis"),
        2: ActualBoundaryResponse("held", "inlet"),
        3: ActualBoundaryResponse("escaped", "outlet"),
    }
    result = resolve_terminal_boundary("held", ids, actual, True)
    assert result.classification == "AMBIGUOUS"
    assert result.semantic_group is None
    assert result.identification == "NOT_TESTED"


def test_disappear_with_unresolved_interaction_cap_is_not_physical_escape() -> None:
    actual = {3: ActualBoundaryResponse("escaped", "outlet")}
    result = resolve_terminal_boundary("escaped", [3], actual, False)
    assert result.classification == "AMBIGUOUS"
    assert result.semantic_group is None
