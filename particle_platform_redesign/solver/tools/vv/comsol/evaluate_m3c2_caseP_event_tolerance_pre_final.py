"""Evaluate the preregistered Case-P geometry-tolerance qualification.

The two candidate runs share seed, macro step, interval-tree depth, and every
physics input.  Only ``geometry_rtol`` changes.  A nonempty transverse-event
cohort uses the existing dimensionally consistent cross-band gate.  If both
runs contain no boundary event, the narrower operational bridge requires
bitwise-identical scheduled and final candidate state and remains valid only
while the confirmatory candidate cohort also contains zero boundary events.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path
from typing import Any, Final, cast

import numpy as np

from chamber_particles import open_result
from tools.vv.comsol.evaluate_m3c2_event_tolerance_cross_band import (
    compare_event_results,
)

TOOL_REVISION: Final = "m3c2_caseP_event_tolerance_pre_final_evaluator_v2"
CANDIDATE_RUNNER_REVISION: Final = "m3c2_candidate_campaign_runner_v4"
RECIPE_ID: Final = "M3-C2A-caseP-100nm-event-tolerance-pre-final"
SEED: Final = 919008
DT_S: Final = 2.0e-5
TREE_DEPTH: Final = 3
REFERENCE_RTOL: Final = 1.0e-8
CANDIDATE_RTOL: Final = 1.0e-9
OUTPUT_COUNT: Final = 121
PARTICLE_COUNT: Final = 287
REFERENCE_LEVEL: Final = "geometry_rtol_reference"
CANDIDATE_LEVEL: Final = "geometry_rtol_candidate"


def _mapping(value: object, location: str) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, Mapping):
        return dict(value)
    raise ValueError(f"{location} must be a mapping")


def _sequence(value: object, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{location} must be a list")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, location: str) -> dict[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), location)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _locked_path(root: Path, record: object, location: str) -> Path:
    artifact = _mapping(record, location)
    if set(artifact) != {"path", "sha256"}:
        raise ValueError(f"{location} must contain exactly path and sha256")
    path = root / str(artifact["path"])
    if not path.is_file() or _sha256(path) != str(artifact["sha256"]).lower():
        raise ValueError(f"{location} identity differs: {path}")
    return path


def _float_array_equal(left: np.ndarray, right: np.ndarray) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    left_nan = np.isnan(left)
    right_nan = np.isnan(right)
    if not np.array_equal(left_nan, right_nan):
        return False
    left_bits = np.ascontiguousarray(left[~left_nan]).view(np.uint8)
    right_bits = np.ascontiguousarray(right[~right_nan]).view(np.uint8)
    return bool(np.array_equal(left_bits, right_bits))


def _array_equal(left_value: object, right_value: object) -> bool:
    left = np.asarray(left_value)
    right = np.asarray(right_value)
    if np.issubdtype(left.dtype, np.floating) and np.issubdtype(right.dtype, np.floating):
        return _float_array_equal(left, right)
    return bool(
        left.shape == right.shape and left.dtype == right.dtype and np.array_equal(left, right)
    )


def _scalar_float_equal(left: float, right: float) -> bool:
    values = np.asarray([left, right], dtype=np.float64).view(np.uint64)
    return bool(values[0] == values[1])


def _scheduled_state_comparison(reference: Any, candidate: Any) -> dict[str, object]:
    reference_frames = list(reference.iter_frames())
    candidate_frames = list(candidate.iter_frames())
    mismatches: list[str] = []
    if len(reference_frames) != OUTPUT_COUNT or len(candidate_frames) != OUTPUT_COUNT:
        mismatches.append("frame_count")
    for index, (left, right) in enumerate(zip(reference_frames, candidate_frames, strict=False)):
        if not _scalar_float_equal(float(left.time_s), float(right.time_s)):
            mismatches.append(f"frame_{index}.time_s")
        names = ("particle_id", "position_m", "velocity_m_s", "charge_number", "lifecycle")
        mismatches.extend(
            f"frame_{index}.{name}"
            for name in names
            if not _array_equal(getattr(left, name), getattr(right, name))
        )
        if len(np.asarray(left.particle_id)) != PARTICLE_COUNT:
            mismatches.append(f"frame_{index}.particle_count_reference")
        if len(np.asarray(right.particle_id)) != PARTICLE_COUNT:
            mismatches.append(f"frame_{index}.particle_count_candidate")
    return {
        "exact": not mismatches,
        "reference_frame_count": len(reference_frames),
        "candidate_frame_count": len(candidate_frames),
        "mismatches": mismatches[:20],
    }


def _final_state_comparison(reference: Any, candidate: Any) -> dict[str, object]:
    left = reference.read_final()
    right = candidate.read_final()
    mismatches = [
        item.name
        for item in fields(left)
        if not _array_equal(getattr(left, item.name), getattr(right, item.name))
    ]
    if len(np.asarray(left.particle_id)) != PARTICLE_COUNT:
        mismatches.append("particle_count_reference")
    if len(np.asarray(right.particle_id)) != PARTICLE_COUNT:
        mismatches.append("particle_count_candidate")
    return {"exact": not mismatches, "mismatches": mismatches}


def compare_results(reference: Any, candidate: Any) -> dict[str, object]:
    event_comparison = compare_event_results(reference, candidate)
    reference_events = int(cast(Any, event_comparison["reference_event_count"]))
    candidate_events = int(cast(Any, event_comparison["candidate_event_count"]))
    no_event_bridge = reference_events == 0 and candidate_events == 0
    scheduled = (
        _scheduled_state_comparison(reference, candidate)
        if no_event_bridge
        else {"exact": None, "mismatches": [], "status": "NOT_APPLICABLE_EVENTS_PRESENT"}
    )
    final = (
        _final_state_comparison(reference, candidate)
        if no_event_bridge
        else {"exact": None, "mismatches": [], "status": "NOT_APPLICABLE_EVENTS_PRESENT"}
    )
    cross_band_pass = event_comparison["status"] == "PASS"
    state_bridge_pass = bool(scheduled["exact"]) and bool(final["exact"])
    passed = bool(cross_band_pass and (not no_event_bridge or state_bridge_pass))
    if not passed:
        classification = "FAIL"
        validity_condition = "not_qualified"
    elif no_event_bridge:
        classification = "PASS_NO_EVENT_OPERATIONAL_BRIDGE"
        validity_condition = "zero_candidate_boundary_events"
    else:
        classification = "PASS_CASEP_TRANSVERSE_TERMINAL_EVENTS"
        validity_condition = "casep_transverse_terminal_events"
    return {
        "status": "PASS" if passed else "FAIL",
        "classification": classification,
        "validity_condition": validity_condition,
        "reference_event_count": reference_events,
        "candidate_event_count": candidate_events,
        "event_comparison": event_comparison,
        "scheduled_state_bitwise_comparison": scheduled,
        "final_state_bitwise_comparison": final,
    }


def _validate_recipe(recipe: dict[str, Any], root: Path) -> None:
    if (
        recipe.get("schema_version"),
        recipe.get("recipe_id"),
        recipe.get("recipe_revision"),
    ) != (1, RECIPE_ID, 2):
        raise ValueError("unexpected Case-P event-tolerance recipe identity")
    pilot = _mapping(recipe.get("pilot"), "recipe pilot")
    if pilot.get("candidate_seeds") != [SEED]:
        raise ValueError("Case-P event-tolerance recipe seed differs")
    levels = [_mapping(value, "recipe level") for value in _sequence(pilot.get("levels"), "levels")]
    expected = [
        (REFERENCE_LEVEL, DT_S, TREE_DEPTH, REFERENCE_RTOL),
        (CANDIDATE_LEVEL, DT_S, TREE_DEPTH, CANDIDATE_RTOL),
    ]
    actual = [
        (
            level.get("name"),
            float(level.get("dt_s", math.nan)),
            int(level.get("brownian_interval_tree_depth", -1)),
            float(level.get("geometry_rtol", math.nan)),
        )
        for level in levels
    ]
    if actual != expected:
        raise ValueError("Case-P event-tolerance level matrix differs")
    gate = _mapping(recipe.get("acceptance_gate"), "acceptance gate")
    if gate.get("zero_event_validity_condition") != "zero_candidate_boundary_events":
        raise ValueError("Case-P no-event validity condition differs")
    _locked_path(root, recipe.get("evaluator"), "Case-P tolerance evaluator")
    companion = _locked_path(
        root, recipe.get("caseA_transverse_event_companion"), "Case-A companion qualification"
    )
    if _load_json(companion, "Case-A companion qualification").get("status") != "PASS":
        raise ValueError("Case-A transverse-event companion is not passing")


def _receipt(prepared: Path, level: str) -> tuple[Path, dict[str, Any]]:
    path = prepared / "levels" / level / f"seed_{SEED}" / "run_receipt.json"
    receipt = _load_json(path, f"{level} receipt")
    if receipt.get("status") != "COMPLETE":
        raise ValueError(f"{level} result is not complete")
    if (
        receipt.get("seed") != SEED
        or receipt.get("level") != level
        or receipt.get("participant") != "candidate"
        or receipt.get("tool_revision") != CANDIDATE_RUNNER_REVISION
        or receipt.get("failure_rows") != 0
        or float(receipt.get("dt_s", math.nan)) != DT_S
        or int(receipt.get("brownian_interval_tree_depth", -1)) != TREE_DEPTH
    ):
        raise ValueError(f"{level} receipt identity or numerical setting differs")
    expected_rtol = REFERENCE_RTOL if level == REFERENCE_LEVEL else CANDIDATE_RTOL
    if float(receipt.get("geometry_rtol", math.nan)) != expected_rtol:
        raise ValueError(f"{level} geometry_rtol differs")
    return path, receipt


def evaluate(recipe_path: Path, prepared: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"Case-P tolerance output already exists: {output}")
    recipe = _load_json(recipe_path, "Case-P tolerance recipe")
    root = _repository_root()
    _validate_recipe(recipe, root)
    recipe_copy = prepared / "candidate_pilot_recipe.json"
    if _sha256(recipe_copy) != _sha256(recipe_path):
        raise ValueError("prepared recipe differs from the preregistered source recipe")
    reference_receipt_path, reference_receipt = _receipt(prepared, REFERENCE_LEVEL)
    candidate_receipt_path, candidate_receipt = _receipt(prepared, CANDIDATE_LEVEL)
    reference = open_result(prepared / str(reference_receipt["result"]))
    candidate = open_result(prepared / str(candidate_receipt["result"]))
    comparison = compare_results(reference, candidate)
    report: dict[str, object] = {
        **comparison,
        "schema_version": 1,
        "qualification_kind": "m3c2_caseP_geometry_rtol_pre_final",
        "tool_revision": TOOL_REVISION,
        "recipe": str(recipe_path),
        "recipe_sha256": _sha256(recipe_path),
        "seed": SEED,
        "selected_setting": {"dt_s": DT_S, "brownian_interval_tree_depth": TREE_DEPTH},
        "reference": {
            "geometry_rtol": REFERENCE_RTOL,
            "receipt": str(reference_receipt_path),
            "receipt_sha256": _sha256(reference_receipt_path),
        },
        "candidate": {
            "geometry_rtol": CANDIDATE_RTOL,
            "receipt": str(candidate_receipt_path),
            "receipt_sha256": _sha256(candidate_receipt_path),
        },
        "final_fail_closed_rule": (
            "when validity_condition is zero_candidate_boundary_events, reject final evaluation "
            "if any candidate final replica contains a boundary event"
        ),
        "claim": "Case-P candidate geometry-tolerance configuration only; no COMSOL fit claim",
    }
    output.mkdir(parents=True)
    (output / "qualification_decision.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("output", type=Path)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    report = evaluate(
        arguments.recipe.resolve(), arguments.prepared.resolve(), arguments.output.resolve()
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
