"""Resolve terminal boundary meaning from observed IDs and actual responses."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

type Outcome = Literal["active", "stuck", "held", "escaped"]


@dataclass(frozen=True, slots=True)
class ActualBoundaryResponse:
    """An adapter-normalized actual response, rather than a requested setting."""

    outcome: Outcome
    semantic_group: str


@dataclass(frozen=True, slots=True)
class TerminalBoundaryResolution:
    classification: Literal["SUPPORTED", "AMBIGUOUS"]
    semantic_group: str | None
    observed_boundary_ids: tuple[int, ...]
    identification: Literal["observed_ids", "group_only", "NOT_TESTED"]
    reason: str


def resolve_terminal_boundary(
    outcome: Outcome,
    observed_boundary_ids: Sequence[int],
    actual_response_by_id: Mapping[int, ActualBoundaryResponse],
    other_terminal_causes_excluded: bool,
) -> TerminalBoundaryResolution:
    """Identify one semantic group without inventing a facet or a terminal cause.

    With no observed IDs, the caller must supply the complete actual response
    map. An incomplete map cannot prove that one matching group is unique.
    """

    ids = tuple(sorted(set(observed_boundary_ids)))
    if outcome == "active":
        raise ValueError("terminal boundary resolution requires a terminal outcome")
    if not other_terminal_causes_excluded:
        return _ambiguous(ids, "Other terminal causes have not been excluded.")
    if ids:
        if any(boundary_id not in actual_response_by_id for boundary_id in ids):
            return _ambiguous(ids, "An observed boundary ID has no actual response mapping.")
        responses = [actual_response_by_id[boundary_id] for boundary_id in ids]
        if any(response.outcome != outcome for response in responses):
            return _ambiguous(
                ids, "Observed boundary response disagrees with the terminal outcome."
            )
    else:
        responses = [
            response for response in actual_response_by_id.values() if response.outcome == outcome
        ]
    groups = {response.semantic_group for response in responses}
    if len(groups) != 1 or not all(groups):
        return _ambiguous(ids, "Terminal outcome does not identify one semantic boundary group.")
    return TerminalBoundaryResolution(
        "SUPPORTED",
        groups.pop(),
        ids,
        "observed_ids" if ids else "group_only",
        "Actual response and terminal cause identify one group; unobserved facets remain unknown.",
    )


def _ambiguous(ids: tuple[int, ...], reason: str) -> TerminalBoundaryResolution:
    return TerminalBoundaryResolution("AMBIGUOUS", None, ids, "NOT_TESTED", reason)
