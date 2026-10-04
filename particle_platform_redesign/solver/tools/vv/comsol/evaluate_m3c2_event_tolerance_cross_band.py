"""Evaluate the predeclared cross-band event-tolerance qualification.

This external V&V tool converts each result's position uncertainty to a
stopping-time allowance with the wall-normal pre-event speed.  It neither
owns event physics nor changes the production solver tolerance policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np

from chamber_particles import open_result

TOOL_REVISION: Final = "m3c2_event_tolerance_cross_band_evaluator_v2"
RECIPE_ID: Final = "M3-C2A-caseA-100nm-event-tolerance-cross-band-sensitivity"
POLICY_REVISION: Final = "cross_band_stopping_time_allowance_v1"
POSITION_RATIO_LIMIT: Final = 1.0
TIME_RATIO_LIMIT: Final = 1.0
TRANSVERSALITY_RATIO: Final = 0.1
NORMAL_SPEED_AGREEMENT_RTOL: Final = 1.0e-3
CERTIFIED_SPEED_FACTOR: Final = 0.5
PARTICLE_COUNT: Final = 287
SEED: Final = 918165
REFERENCE_LEVEL: Final = "event_tolerance_reference"
CANDIDATE_LEVEL: Final = "event_tolerance_candidate"

type EventKey = tuple[int, int]


@dataclass(slots=True)
class _EventComparisons:
    identity_mismatches: list[EventKey] = field(default_factory=list)
    normal_mismatches: list[EventKey] = field(default_factory=list)
    signed_speed_guard_failures: list[EventKey] = field(default_factory=list)
    normal_speed_mismatches: list[EventKey] = field(default_factory=list)
    position_ratios: list[tuple[float, EventKey]] = field(default_factory=list)
    time_ratios: list[tuple[float, EventKey]] = field(default_factory=list)
    transverse_speeds: list[float] = field(default_factory=list)


def _mapping(value: object, location: str) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, Mapping):
        return dict(value)
    raise ValueError(f"{location} must be a mapping")


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
    item = _mapping(record, location)
    if set(item) != {"path", "sha256"}:
        raise ValueError(f"{location} has unexpected keys")
    path = root / str(item["path"])
    if not path.is_file() or _sha256(path) != item["sha256"]:
        raise ValueError(f"{location} identity differs: {path}")
    return path


def _event_rows(events: Any) -> dict[tuple[int, int], int]:
    rows: dict[tuple[int, int], int] = {}
    for row, particle_id_value in enumerate(events.particle_id):
        key = (int(particle_id_value), int(events.event_ordinal[row]))
        if key in rows:
            raise ValueError(f"duplicate event identity: {key}")
        rows[key] = row
    return rows


def _candidate_facets(events: Any, row: int) -> tuple[int, ...]:
    start = int(events.candidate_offset[row])
    stop = int(events.candidate_offset[row + 1])
    return tuple(sorted(int(value) for value in events.candidate_facet_id[start:stop]))


def _event_fates(events: Any) -> tuple[str, ...]:
    fates = ["active"] * PARTICLE_COUNT
    for row, particle_id_value in enumerate(events.particle_id):
        particle_id = int(particle_id_value)
        if not 1 <= particle_id <= PARTICLE_COUNT:
            raise ValueError(
                f"particle identity is outside the qualification cohort: {particle_id}"
            )
        if fates[particle_id - 1] != "active":
            raise ValueError(f"particle has more than one terminal event: {particle_id}")
        fates[particle_id - 1] = str(events.outcome[row])
    return tuple(fates)


def _exact_event_identity(events: Any, row: int) -> tuple[object, ...]:
    return (
        _candidate_facets(events, row),
        int(events.primary_facet_id[row]),
        int(events.boundary_id[row]),
        int(events.material_id[row]),
        str(events.law_id[row]),
        str(events.outcome[row]),
    )


def _positive_budget(value: Any, label: str, key: EventKey) -> float:
    budget = float(value)
    if not math.isfinite(budget) or budget <= 0.0:
        raise ValueError(f"invalid {label} for event {key}")
    return budget


def _round_up(value: float) -> float:
    return math.nextafter(value, math.inf)


def _sum_up(*values: float) -> float:
    total = 0.0
    for value in values:
        total = _round_up(total + value)
    return total


def _position_ratio(
    reference_events: Any,
    candidate_events: Any,
    reference_row: int,
    candidate_row: int,
    key: EventKey,
) -> float:
    budget = _sum_up(
        _positive_budget(reference_events.position_budget_m[reference_row], "position budget", key),
        _positive_budget(candidate_events.position_budget_m[candidate_row], "position budget", key),
    )
    delta = float(
        np.linalg.norm(
            np.asarray(reference_events.position_m[reference_row], dtype=np.float64)
            - np.asarray(candidate_events.position_m[candidate_row], dtype=np.float64)
        )
    )
    if not math.isfinite(delta):
        raise ValueError(f"non-finite position delta for event {key}")
    return _round_up(delta) / budget


def _certified_transverse_speeds(
    reference_events: Any,
    candidate_events: Any,
    reference_row: int,
    candidate_row: int,
) -> tuple[float, float] | str:
    reference_normal = np.asarray(reference_events.normal[reference_row], dtype=np.float64)
    candidate_normal = np.asarray(candidate_events.normal[candidate_row], dtype=np.float64)
    normals_match = all(
        (
            reference_normal.shape == (2,),
            candidate_normal.shape == (2,),
            np.isfinite(reference_normal).all(),
            np.isfinite(candidate_normal).all(),
            np.array_equal(reference_normal, candidate_normal),
        )
    )
    if not normals_match:
        return "normal_mismatch"
    reference_velocity = np.asarray(
        reference_events.velocity_pre_m_s[reference_row], dtype=np.float64
    )
    candidate_velocity = np.asarray(
        candidate_events.velocity_pre_m_s[candidate_row], dtype=np.float64
    )
    velocities_are_finite_vectors = all(
        (
            reference_velocity.shape == (2,),
            candidate_velocity.shape == (2,),
            np.isfinite(reference_velocity).all(),
            np.isfinite(candidate_velocity).all(),
        )
    )
    if not velocities_are_finite_vectors:
        return "signed_speed_guard"
    reference_speed = float(np.dot(reference_normal, reference_velocity))
    candidate_speed = float(np.dot(candidate_normal, candidate_velocity))
    reference_velocity_norm = float(np.linalg.norm(reference_velocity))
    candidate_velocity_norm = float(np.linalg.norm(candidate_velocity))
    speeds_are_transverse = all(
        (
            math.isfinite(reference_speed),
            math.isfinite(candidate_speed),
            math.isfinite(reference_velocity_norm),
            math.isfinite(candidate_velocity_norm),
            reference_speed > 0.0,
            candidate_speed > 0.0,
            reference_speed >= TRANSVERSALITY_RATIO * reference_velocity_norm,
            candidate_speed >= TRANSVERSALITY_RATIO * candidate_velocity_norm,
        )
    )
    if not speeds_are_transverse:
        return "signed_speed_guard"
    relative_speed_difference = abs(reference_speed - candidate_speed) / max(
        reference_speed, candidate_speed
    )
    if relative_speed_difference > NORMAL_SPEED_AGREEMENT_RTOL:
        return "normal_speed_mismatch"
    return (
        CERTIFIED_SPEED_FACTOR * reference_speed,
        CERTIFIED_SPEED_FACTOR * candidate_speed,
    )


def _cross_band_time_metrics(
    reference_events: Any,
    candidate_events: Any,
    reference_row: int,
    candidate_row: int,
    key: EventKey,
) -> tuple[float, float] | str:
    certified_speeds = _certified_transverse_speeds(
        reference_events,
        candidate_events,
        reference_row,
        candidate_row,
    )
    if isinstance(certified_speeds, str):
        return certified_speeds
    reference_certified_speed, candidate_certified_speed = certified_speeds
    reference_position_budget = _positive_budget(
        reference_events.position_budget_m[reference_row], "position budget", key
    )
    candidate_position_budget = _positive_budget(
        candidate_events.position_budget_m[candidate_row], "position budget", key
    )
    reference_time_budget = _positive_budget(
        reference_events.time_budget_s[reference_row], "time budget", key
    )
    candidate_time_budget = _positive_budget(
        candidate_events.time_budget_s[candidate_row], "time budget", key
    )
    reference_position_time = _round_up(reference_position_budget / reference_certified_speed)
    candidate_position_time = _round_up(candidate_position_budget / candidate_certified_speed)
    allowance = _sum_up(
        reference_time_budget,
        candidate_time_budget,
        reference_position_time,
        candidate_position_time,
    )
    time_delta = abs(
        float(reference_events.time_s[reference_row])
        - float(candidate_events.time_s[candidate_row])
    )
    if not math.isfinite(allowance) or allowance <= 0.0 or not math.isfinite(time_delta):
        raise ValueError(f"invalid cross-band stopping-time metric for event {key}")
    return _round_up(time_delta) / allowance, min(
        reference_certified_speed, candidate_certified_speed
    )


def _compare_matching_events(
    reference_events: Any,
    candidate_events: Any,
    reference_rows: dict[EventKey, int],
    candidate_rows: dict[EventKey, int],
) -> _EventComparisons:
    comparison = _EventComparisons()
    guard_failures = {
        "normal_mismatch": comparison.normal_mismatches,
        "normal_speed_mismatch": comparison.normal_speed_mismatches,
        "signed_speed_guard": comparison.signed_speed_guard_failures,
    }
    for key in sorted(reference_rows):
        reference_row = reference_rows[key]
        candidate_row = candidate_rows[key]
        if _exact_event_identity(reference_events, reference_row) != _exact_event_identity(
            candidate_events, candidate_row
        ):
            comparison.identity_mismatches.append(key)
            continue
        comparison.position_ratios.append(
            (
                _position_ratio(
                    reference_events,
                    candidate_events,
                    reference_row,
                    candidate_row,
                    key,
                ),
                key,
            )
        )
        time_metrics = _cross_band_time_metrics(
            reference_events,
            candidate_events,
            reference_row,
            candidate_row,
            key,
        )
        if isinstance(time_metrics, str):
            guard_failures[time_metrics].append(key)
            continue
        time_ratio, transverse_speed = time_metrics
        comparison.time_ratios.append((time_ratio, key))
        comparison.transverse_speeds.append(transverse_speed)
    return comparison


def compare_event_results(
    reference_result: Any,
    candidate_result: Any,
) -> dict[str, object]:
    reference_failures = reference_result.read_failure_events()
    candidate_failures = candidate_result.read_failure_events()
    reference_events = reference_result.read_boundary_events()
    candidate_events = candidate_result.read_boundary_events()
    reference_rows = _event_rows(reference_events)
    candidate_rows = _event_rows(candidate_events)
    keys_equal = set(reference_rows) == set(candidate_rows)
    comparison = (
        _compare_matching_events(reference_events, candidate_events, reference_rows, candidate_rows)
        if keys_equal
        else _EventComparisons()
    )
    maximum_position = max(comparison.position_ratios, default=(0.0, (-1, -1)))
    maximum_time = max(comparison.time_ratios, default=(0.0, (-1, -1)))
    reference_fates = _event_fates(reference_events)
    candidate_fates = _event_fates(candidate_events)
    all_events_time_comparable = all(
        (
            keys_equal,
            not comparison.identity_mismatches,
            not comparison.normal_mismatches,
            not comparison.signed_speed_guard_failures,
            not comparison.normal_speed_mismatches,
            len(comparison.time_ratios) == len(reference_rows),
        )
    )
    gates = {
        "zero_failures": all(
            (
                int(reference_failures.particle_id.size) == 0,
                int(candidate_failures.particle_id.size) == 0,
            )
        ),
        "event_keys_exact": keys_equal,
        "event_identity_exact": all((keys_equal, not comparison.identity_mismatches)),
        "final_fates_exact": reference_fates == candidate_fates,
        "event_normals_exact": not comparison.normal_mismatches,
        "signed_outward_transversality_guard": not comparison.signed_speed_guard_failures,
        "normal_speed_agreement": not comparison.normal_speed_mismatches,
        "position_composed_budget": maximum_position[0] <= POSITION_RATIO_LIMIT,
        "cross_band_time_allowance": all(
            (all_events_time_comparable, maximum_time[0] <= TIME_RATIO_LIMIT)
        ),
    }
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "reference_event_count": len(reference_rows),
        "candidate_event_count": len(candidate_rows),
        "exact_identity_mismatches": [list(key) for key in comparison.identity_mismatches[:10]],
        "normal_mismatches": [list(key) for key in comparison.normal_mismatches[:10]],
        "signed_speed_guard_failures": [
            list(key) for key in comparison.signed_speed_guard_failures[:10]
        ],
        "normal_speed_mismatches": [list(key) for key in comparison.normal_speed_mismatches[:10]],
        "maximum_position_delta_over_composed_budget": maximum_position[0],
        "maximum_position_ratio_event": (
            None if maximum_position[1] == (-1, -1) else list(maximum_position[1])
        ),
        "maximum_time_delta_over_composed_stopping_time_allowance": maximum_time[0],
        "maximum_time_ratio_event": (
            None if maximum_time[1] == (-1, -1) else list(maximum_time[1])
        ),
        "minimum_certified_boundary_transverse_speed_m_per_s": (
            None if not comparison.transverse_speeds else min(comparison.transverse_speeds)
        ),
    }


def _event_work(result: Any) -> dict[str, object]:
    return _mapping(_mapping(result.manifest, "result manifest").get("event_refinement"), "work")


def _receipt(prepared: Path, level: str) -> tuple[Path, dict[str, Any]]:
    path = prepared / "levels" / level / f"seed_{SEED}" / "run_receipt.json"
    receipt = _load_json(path, f"{level} receipt")
    if receipt.get("status") != "COMPLETE":
        raise ValueError(f"{level} result is not complete")
    return path, receipt


def evaluate(recipe_path: Path, prepared: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"cross-band evaluation output already exists: {output}")
    recipe = _load_json(recipe_path, "cross-band recipe")
    if recipe.get("recipe_id") != RECIPE_ID:
        raise ValueError("unexpected cross-band recipe identity")
    gate = _mapping(recipe.get("acceptance_gate"), "acceptance gate")
    if gate.get("policy_revision") != POLICY_REVISION:
        raise ValueError("unexpected cross-band policy revision")
    fixed_policy = {
        "position_ratio_limit": POSITION_RATIO_LIMIT,
        "time_ratio_limit": TIME_RATIO_LIMIT,
        "transversality_ratio": TRANSVERSALITY_RATIO,
        "normal_speed_agreement_rtol": NORMAL_SPEED_AGREEMENT_RTOL,
        "certified_speed_factor": CERTIFIED_SPEED_FACTOR,
    }
    if any(float(gate.get(name, math.nan)) != value for name, value in fixed_policy.items()):
        raise ValueError("cross-band numerical policy differs from the evaluator revision")
    root = _repository_root()
    v3_evidence = _locked_path(root, recipe.get("v3_invalid_gate_evidence"), "v3 evidence")
    recipe_copy = prepared / "candidate_pilot_recipe.json"
    if _sha256(recipe_copy) != _sha256(recipe_path):
        raise ValueError("prepared recipe differs from the predeclared source recipe")
    reference_receipt_path, reference_receipt = _receipt(prepared, REFERENCE_LEVEL)
    candidate_receipt_path, candidate_receipt = _receipt(prepared, CANDIDATE_LEVEL)
    reference_result = open_result(prepared / str(reference_receipt["result"]))
    candidate_result = open_result(prepared / str(candidate_receipt["result"]))
    comparison = compare_event_results(reference_result, candidate_result)
    report: dict[str, object] = {
        **comparison,
        "tool_revision": TOOL_REVISION,
        "policy_revision": POLICY_REVISION,
        "recipe": str(recipe_path),
        "recipe_sha256": _sha256(recipe_path),
        "v3_invalid_gate_evidence": {
            "path": str(v3_evidence),
            "sha256": _sha256(v3_evidence),
            "retained_status": "FAIL",
        },
        "reference": {
            "geometry_rtol": reference_receipt["geometry_rtol"],
            "receipt": str(reference_receipt_path),
            "receipt_sha256": _sha256(reference_receipt_path),
            "event_work": _event_work(reference_result),
        },
        "candidate": {
            "geometry_rtol": candidate_receipt["geometry_rtol"],
            "receipt": str(candidate_receipt_path),
            "receipt_sha256": _sha256(candidate_receipt_path),
            "event_work": _event_work(candidate_result),
        },
        "claim": (
            "event-tolerance configuration qualification on an unseen seed only; "
            "no COMSOL or physics-fit claim"
        ),
    }
    output.mkdir(parents=True)
    (output / "event_tolerance_cross_band.json").write_text(
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
