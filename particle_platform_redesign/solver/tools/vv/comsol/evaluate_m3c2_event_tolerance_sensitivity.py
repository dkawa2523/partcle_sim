"""Evaluate the predeclared M3-C2 event-tolerance sensitivity gate.

This external V&V tool compares canonical result artifacts.  It neither owns
event physics nor changes the production solver tolerance policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, cast

import numpy as np

from chamber_particles import open_result

TOOL_REVISION: Final = "m3c2_event_tolerance_sensitivity_evaluator_v1"
PARTICLE_COUNT: Final = 287


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


def _locked_path(root: Path, record: object, location: str) -> Path:
    item = _mapping(record, location)
    if set(item) != {"path", "sha256"}:
        raise ValueError(f"{location} has unexpected keys")
    path = root / str(item["path"])
    if not path.is_file() or _sha256(path) != item["sha256"]:
        raise ValueError(f"{location} identity differs: {path}")
    return path


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[5]


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


def _event_ratios(
    baseline_events: Any,
    candidate_events: Any,
    baseline_row: int,
    candidate_row: int,
    key: tuple[int, int],
) -> tuple[float, float]:
    position_budget = max(
        float(baseline_events.position_budget_m[baseline_row]),
        float(candidate_events.position_budget_m[candidate_row]),
    )
    time_budget = max(
        float(baseline_events.time_budget_s[baseline_row]),
        float(candidate_events.time_budget_s[candidate_row]),
    )
    if not math.isfinite(position_budget) or position_budget <= 0.0:
        raise ValueError(f"invalid position budget for event {key}")
    if not math.isfinite(time_budget) or time_budget <= 0.0:
        raise ValueError(f"invalid time budget for event {key}")
    position_delta = float(
        np.linalg.norm(
            np.asarray(baseline_events.position_m[baseline_row], dtype=np.float64)
            - np.asarray(candidate_events.position_m[candidate_row], dtype=np.float64)
        )
    )
    time_delta = abs(
        float(baseline_events.time_s[baseline_row]) - float(candidate_events.time_s[candidate_row])
    )
    return position_delta / position_budget, time_delta / time_budget


def compare_event_results(
    baseline_result: Any,
    candidate_result: Any,
    *,
    normalized_ratio_limit: float,
) -> dict[str, object]:
    baseline_failures = baseline_result.read_failure_events()
    candidate_failures = candidate_result.read_failure_events()
    baseline_events = baseline_result.read_boundary_events()
    candidate_events = candidate_result.read_boundary_events()
    baseline_rows = _event_rows(baseline_events)
    candidate_rows = _event_rows(candidate_events)
    identities_equal = set(baseline_rows) == set(candidate_rows)
    exact_mismatches: list[tuple[int, int]] = []
    maximum_position_ratio = 0.0
    maximum_time_ratio = 0.0
    maximum_position_key: tuple[int, int] | None = None
    maximum_time_key: tuple[int, int] | None = None
    if identities_equal:
        for key in sorted(baseline_rows):
            baseline_row = baseline_rows[key]
            candidate_row = candidate_rows[key]
            if _exact_event_identity(baseline_events, baseline_row) != _exact_event_identity(
                candidate_events, candidate_row
            ):
                exact_mismatches.append(key)
                continue
            position_ratio, time_ratio = _event_ratios(
                baseline_events,
                candidate_events,
                baseline_row,
                candidate_row,
                key,
            )
            if position_ratio > maximum_position_ratio:
                maximum_position_ratio = position_ratio
                maximum_position_key = key
            if time_ratio > maximum_time_ratio:
                maximum_time_ratio = time_ratio
                maximum_time_key = key
    baseline_fates = _event_fates(baseline_events)
    candidate_fates = _event_fates(candidate_events)
    gates = {
        "zero_failures": (
            int(baseline_failures.particle_id.size) == 0
            and int(candidate_failures.particle_id.size) == 0
        ),
        "event_keys_exact": identities_equal,
        "event_identity_exact": identities_equal and not exact_mismatches,
        "final_fates_exact": baseline_fates == candidate_fates,
        "position_budget": maximum_position_ratio <= normalized_ratio_limit,
        "time_budget": maximum_time_ratio <= normalized_ratio_limit,
    }
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "baseline_event_count": len(baseline_rows),
        "candidate_event_count": len(candidate_rows),
        "exact_identity_mismatches": [list(key) for key in exact_mismatches[:10]],
        "maximum_position_delta_over_max_recorded_budget": maximum_position_ratio,
        "maximum_position_ratio_event": (
            None if maximum_position_key is None else list(maximum_position_key)
        ),
        "maximum_time_delta_over_max_recorded_budget": maximum_time_ratio,
        "maximum_time_ratio_event": None if maximum_time_key is None else list(maximum_time_key),
    }


def _event_work(result: Any) -> dict[str, object]:
    manifest = _mapping(result.manifest, "result manifest")
    return _mapping(manifest.get("event_refinement"), "event refinement")


def evaluate(recipe_path: Path, candidate_prepared: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"event-tolerance evaluation output already exists: {output}")
    recipe = _load_json(recipe_path, "event-tolerance recipe")
    if recipe.get("recipe_id") != "M3-C2A-caseA-100nm-event-tolerance-sensitivity":
        raise ValueError("unexpected event-tolerance recipe identity")
    root = _repository_root()
    baseline_records = cast(list[object], recipe.get("baseline_diagnostic"))
    baseline_receipt_path = _locked_path(root, baseline_records[0], "baseline receipt")
    _locked_path(root, baseline_records[1], "baseline manifest")
    baseline_receipt = _load_json(baseline_receipt_path, "baseline receipt")
    baseline_prepared = baseline_receipt_path.parents[3]
    baseline_result_path = baseline_prepared / str(baseline_receipt["result"])
    candidate_recipe_copy = candidate_prepared / "candidate_pilot_recipe.json"
    if _sha256(candidate_recipe_copy) != _sha256(recipe_path):
        raise ValueError("candidate prepared recipe differs from the predeclared source recipe")
    candidate_receipt_path = (
        candidate_prepared
        / "levels"
        / "event_tolerance_sensitivity"
        / "seed_918164"
        / "run_receipt.json"
    )
    candidate_receipt = _load_json(candidate_receipt_path, "candidate receipt")
    if candidate_receipt.get("status") != "COMPLETE":
        raise ValueError("candidate sensitivity result is not complete")
    candidate_result_path = candidate_prepared / str(candidate_receipt["result"])
    baseline_result = open_result(baseline_result_path)
    candidate_result = open_result(candidate_result_path)
    gate = _mapping(recipe.get("acceptance_gate"), "acceptance gate")
    comparison = compare_event_results(
        baseline_result,
        candidate_result,
        normalized_ratio_limit=float(gate["normalized_ratio_limit"]),
    )
    report: dict[str, object] = {
        **comparison,
        "tool_revision": TOOL_REVISION,
        "recipe": str(recipe_path),
        "recipe_sha256": _sha256(recipe_path),
        "baseline": {
            "geometry_rtol": baseline_receipt["geometry_rtol"],
            "receipt": str(baseline_receipt_path),
            "receipt_sha256": _sha256(baseline_receipt_path),
            "event_work": _event_work(baseline_result),
        },
        "candidate": {
            "geometry_rtol": candidate_receipt["geometry_rtol"],
            "receipt": str(candidate_receipt_path),
            "receipt_sha256": _sha256(candidate_receipt_path),
            "event_work": _event_work(candidate_result),
        },
        "claim": "event-tolerance configuration qualification only; no COMSOL or physics-fit claim",
    }
    output.mkdir(parents=True)
    (output / "event_tolerance_sensitivity.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("candidate_prepared", type=Path)
    parser.add_argument("output", type=Path)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    report = evaluate(
        arguments.recipe.resolve(),
        arguments.candidate_prepared.resolve(),
        arguments.output.resolve(),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
