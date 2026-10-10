from __future__ import annotations

import json
from pathlib import Path

import pytest
from tools.vv.comsol.actual_run_receipt import (
    RECEIPT_NAME,
    RECEIPT_REVISION,
    normalize_terminal_event,
    read_actual_run_receipt,
)


def _feature(ids: list[int], law: str, group: str) -> dict[str, object]:
    return {
        "active": True,
        "selection_observation": "OBSERVED",
        "boundary_ids": ids,
        "semantic_group": group,
        "properties": {"WallCondition": {"observation": "OBSERVED", "observed_value": [law]}},
    }


def _write_receipt(root: Path, features: dict[str, object], *, ids: list[int]) -> None:
    raw = {
        "schema_version": 1,
        "tool_revision": RECEIPT_REVISION,
        "source": {},
        "companion": {"boundary_features": features},
        "terminal_observations": [
            {
                "particle_id": 1,
                "event_time_s": 0.001,
                "outcome": "held",
                "boundary_ids": ids,
                "other_terminal_causes_excluded": True,
                "cause_evidence": "explicit terminal boundary export with no competing cause",
            }
        ],
    }
    (root / RECEIPT_NAME).write_text(json.dumps(raw), encoding="utf-8")


def test_missing_actual_receipt_keeps_status_and_never_names_inlet(tmp_path: Path) -> None:
    actual = read_actual_run_receipt(tmp_path)
    event, evidence = normalize_terminal_event(actual, 1, "held", 0.001)
    assert event == (1, 0.001, "terminal_status", "held", "")
    assert actual.artifact["observation"] == "NOT_TESTED"
    assert evidence["classification"] == "AMBIGUOUS"
    assert evidence["identification"] == "NOT_TESTED"


@pytest.mark.parametrize("ids", [[], [1, 37], [999]])
def test_freeze_axis_and_inlet_require_observed_identification(
    tmp_path: Path, ids: list[int]
) -> None:
    _write_receipt(
        tmp_path,
        {
            "axis": _feature([1], "Freeze", "axis"),
            "inlet": _feature([37], "Freeze", "gas_inlet"),
        },
        ids=ids,
    )
    actual = read_actual_run_receipt(tmp_path)
    event, evidence = normalize_terminal_event(actual, 1, "held", 0.001)
    assert event[4] == ""
    assert evidence["classification"] == "AMBIGUOUS"


def test_observed_axis_is_not_renamed_to_inlet(tmp_path: Path) -> None:
    _write_receipt(tmp_path, {"axis": _feature([1], "Freeze", "axis")}, ids=[1])
    event, evidence = normalize_terminal_event(read_actual_run_receipt(tmp_path), 1, "held", 0.001)
    assert event == (1, 0.001, "terminal_boundary", "held", "axis")
    assert evidence["observed_boundary_ids"] == [1]


@pytest.mark.parametrize("unknown_selection", [False, True])
def test_competing_or_unobserved_feature_cannot_supply_effective_law(
    tmp_path: Path, unknown_selection: bool
) -> None:
    competitor = _feature([1, 37], "Freeze", "unknown")
    if unknown_selection:
        competitor["selection_observation"] = "NOT_TESTED"
    _write_receipt(
        tmp_path,
        {"inlet": _feature([37], "Freeze", "gas_inlet"), "competitor": competitor},
        ids=[37],
    )
    event, evidence = normalize_terminal_event(read_actual_run_receipt(tmp_path), 1, "held", 0.001)
    assert event[4] == ""
    assert evidence["classification"] == "AMBIGUOUS"


@pytest.mark.parametrize(
    "payload",
    [
        '{"schema_version":true}',
        '{"schema_version":1,"schema_version":1}',
        '{"schema_version":NaN}',
    ],
)
def test_actual_receipt_rejects_boolean_schema_duplicate_and_nonfinite_json(
    tmp_path: Path, payload: str
) -> None:
    (tmp_path / RECEIPT_NAME).write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        read_actual_run_receipt(tmp_path)


@pytest.mark.parametrize("law", ["UnknownWallLaw", "Freeze"])
def test_partial_response_map_never_promotes_group_only_inlet(tmp_path: Path, law: str) -> None:
    features: dict[str, object] = {
        "inlet": _feature([37], "Freeze", "gas_inlet"),
        "axis": _feature([1], law, "axis"),
    }
    if law == "Freeze":
        features["overlapping_axis"] = _feature([1], "Freeze", "axis")
    _write_receipt(tmp_path, features, ids=[])
    event, evidence = normalize_terminal_event(read_actual_run_receipt(tmp_path), 1, "held", 0.001)
    assert event == (1, 0.001, "terminal_status", "held", "")
    assert evidence["classification"] == "AMBIGUOUS"
    assert evidence["identification"] == "NOT_TESTED"
