"""Bind this campaign to generic field/source producers and register fresh runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import yaml
from tools.source_builder import prepare_case

from chamber_particles import load_case
from chamber_particles.case import SimulationCase
from chamber_particles.case_format import DataBundle, RealizedTableSource, read, write

HERE = Path(__file__).resolve().parent
SOLVER = HERE.parent.parent
OLD = SOLVER / "evidence/comsol_speed10000_comparison_2026_10_13"
STEPS_S = (5e-7, 2.5e-7, 1.25e-7)


def identity(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def source_configuration(source: RealizedTableSource) -> dict[str, object]:
    return {
        "format_version": 1,
        "table": source.name,
        "columns": {
            "particle_id": "particle_id",
            "release_time_s": "time_s",
            "position_m": ["r_m", "z_m"],
            "charge_number": "charge_number_e",
            "drag_diameter_m": "diameter_m",
        },
        "units": {
            "release_time_s": "s",
            "position_m": "m",
            "charge_number": "1",
            "drag_diameter_m": "m",
        },
        "velocity_mode": "vector",
        "velocity_columns": ["velocity_r_m_per_s", "velocity_z_m_per_s"],
        "velocity_unit": "m/s",
        "components": ["r", "z"],
        "basis_to_case": [[1, 0], [0, 1]],
        "frame": "inertial",
        "filter_equals": {"time_s": 0},
        "metadata": {
            "scope": "Same component-basis saved native realized rows; no laboratory-frame certification.",
            "unresolved": [
                "Historical native laboratory-frame interpretation remains unqualified."
            ],
            "property_authority": "Mass/independent radii/volume/weight/material inherited by exact ID from the already inspected canonical source; native diameter and charge supplied explicitly.",
            "native_inventory": identity(OLD / "native/native_readback.jsonl"),
        },
    }


def check_same_scientific_input(old: DataBundle, new: DataBundle) -> None:
    owners = ((old.geometry, new.geometry), (old.geometry.boundary, new.geometry.boundary))
    for left_record, right_record in owners:
        for field in fields(left_record):
            if field.name != "boundary" and not np.array_equal(
                getattr(left_record, field.name), getattr(right_record, field.name)
            ):
                raise ValueError(f"Changed geometry {field.name}")
    left, right = {f.name: f for f in old.fields}, {f.name: f for f in new.fields}
    if set(left) != set(right):
        raise ValueError("Changed selected field inventory")
    for name in left:
        for field in fields(left[name]):
            if not np.array_equal(
                getattr(left[name], field.name), getattr(right[name], field.name)
            ):
                raise ValueError(f"Changed field payload/meaning {name}/{field.name}")
    for old_source, new_source in zip(old.sources, new.sources, strict=True):
        for field in fields(old_source):
            if not np.array_equal(getattr(old_source, field.name), getattr(new_source, field.name)):
                raise ValueError(f"Changed realized source {field.name}")


def prepare_one(name: str) -> dict[str, object]:
    directory = HERE / name
    original_yaml = OLD / name / "prepared/full_h0p25us.yaml"
    original = load_case(original_yaml)
    adapter_path = directory / "adapter/selected_fields.h5"
    imported = read(adapter_path)
    carrier = replace(imported, sources=original.data.sources)
    info = write(directory / "source_property_carrier.h5", carrier)
    document = yaml.safe_load(original_yaml.read_text(encoding="utf-8"))
    document["case"]["data_path"] = "source_property_carrier.h5"
    document["case"]["expected_content_hash"] = info.content_hash
    template = directory / "source_property_carrier.yaml"
    with template.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(document, stream, sort_keys=False)
    (source,) = original.data.sources
    if not isinstance(source, RealizedTableSource):
        raise ValueError("This inspected campaign uses one internal realized table")
    configuration = directory / "source_configuration.json"
    write_json(configuration, source_configuration(source))
    raw_source = OLD / "native" / f"{name}_native_t0_state_normalized.csv"
    built = prepare_case(template, raw_source, configuration, directory / "prepared")
    case_path = Path(built["case"])
    rebuilt = load_case(case_path)
    check_same_scientific_input(original.data, rebuilt.data)
    inputs = {
        "old_case": identity(original_yaml),
        "old_data": identity(original.data_path),
        "adapter": identity(adapter_path),
        "native_initial_rows": identity(raw_source),
        "source_configuration": identity(configuration),
        "new_case": identity(case_path),
        "new_data": identity(rebuilt.data_path),
    }
    parity = {
        "status": "PASS",
        "scope": "Scientific geometry/field/source payload equality; provenance changes. No physical/frame/accuracy certification.",
        "field_count": len(rebuilt.data.fields),
        "particle_count": source.particle_id.size,
        "bindings": inputs,
    }
    parity["particle_count"] = int(parity["particle_count"])
    write_json(directory / "preparation_payload_receipt.json", parity)
    return register_cases(name, case_path, rebuilt, inputs)


def register_cases(
    name: str, prepared: Path, case: SimulationCase, inputs: dict
) -> dict[str, object]:
    directory = HERE / name
    document = yaml.safe_load(prepared.read_text(encoding="utf-8"))
    runs = []
    for level, step in zip(("coarse", "medium", "fine"), STEPS_S, strict=True):
        configured = json.loads(json.dumps(document))
        configured["case"]["name"] = f"{name}_generic_preparation_rk4_{level}"
        configured["case"]["data_path"] = "prepared/case.h5"
        configured["time"]["dt_s"] = step
        configured["solver"]["integrator"] = "rk4_fixed"
        path = directory / f"rk4_{level}.yaml"
        with path.open("x", encoding="utf-8") as stream:
            yaml.safe_dump(configured, stream, sort_keys=False)
        load_case(path)
        runs.append(
            {
                "level": level,
                "step_s": step,
                "case": str(path),
                "output": str(directory / f"rk4_{level}_run"),
            }
        )
    return {
        "case": name,
        "source_ids": [int(i) for i in case.data.sources[0].particle_id],
        "observation_times_s": document["output"]["trajectories"]["schedule"]["explicit_times_s"],
        "bindings": inputs,
        "runs": runs,
    }


def main() -> None:
    cases = [prepare_one(name) for name in ("caseA", "caseP")]
    write_json(
        HERE / "scientific_run_plan.json",
        {
            "objective": "Independently resolve deterministic finite-speed A/P trajectories after generic source/field preparation and compare to preserved original COMSOL results.",
            "intent": "Original versus analogous model; independently matched numerical-method comparison uses the same canonical scientific payload.",
            "physical_changes_from_previous_candidate": "none; fields/source/walls/charge and force models retained",
            "numerical_change": "Coupled RK4 with three fixed steps .5/.25/.125us, independent of saved observation spacing. Early pilot satisfied explicit stability without gate changes.",
            "empirical_resolution_targets": {
                "position_m": 1e-4,
                "velocity_m_s": 1,
                "charge_number": 0.1,
                "first_hit_time_s": 1e-6,
                "fate_count_difference": 1,
            },
            "budget_meaning": "Unchanged prior provisional computational targets, empirical fine-pair sensitivity only, not certified error bounds or COMSOL matching tolerances.",
            "excluded_physical_claims": [
                "nonlinear Brownian dynamics",
                "high-drift thermophoresis",
                "full3D transport",
                "experimental validation",
                "original model numerical parity",
            ],
            "cases": cases,
        },
    )
    print(json.dumps({"cases_prepared": len(cases), "planned_new_runs": 6}))


if __name__ == "__main__":
    main()
