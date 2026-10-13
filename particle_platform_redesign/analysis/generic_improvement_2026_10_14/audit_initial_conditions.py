"""Read-only source audit of saved native expressions, vectors and canonical rows."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

from chamber_particles import load_case

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parents[1] / "solver/evidence/comsol_speed10000_comparison_2026_10_13"
NATIVE = CAMPAIGN / "native"


def binding(path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(HERE.parents[2]).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def native_source(rows: list[dict], name: str) -> tuple[dict, dict, dict]:
    selected = [row for row in rows if row["case"] == name]
    parameters = {
        row["payload"]["name"]: row["payload"] for row in selected if row["kind"] == "parameter"
    }
    (feature,) = [
        row["payload"]
        for row in selected
        if row["kind"] == "feature" and row["payload"].get("type") == "ReleaseGrid"
    ]
    expressions = {
        f"{row['payload']['path']}/{row['payload']['name']}": row["payload"]["expression"]
        for row in selected
        if row["kind"] == "variable_expression"
        and row["payload"]["name"]
        in {"release_theta", "release_speed_P", "AS_release_speed", "AS_ugr", "AS_ugz"}
    }
    return parameters, feature, expressions


def read_initial_states(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return {
            int(row["particle_id"]): row
            for row in csv.DictReader(stream)
            if row["participant"] == "native10" and float(row["time_s"]) == 0.0
        }


def quantity_ranges(records: list[dict]) -> dict:
    return {
        key: [min(record[key] for record in records), max(record[key] for record in records)]
        for key in records[0]
        if key != "particle_id"
    }


def audit_case(rows: list[dict], name: str) -> dict[str, object]:
    parameters, feature, expressions = native_source(rows, name)
    primitive_path = NATIVE / f"{name}_primitive_probes.json"
    primitives = json.loads(primitive_path.read_text(encoding="utf-8"))
    primitive_columns = dict(
        zip(primitives["column_names"], primitives["values_by_column_point"], strict=True)
    )
    case_path = CAMPAIGN / name / "prepared/full_h0p25us.yaml"
    case = load_case(case_path)
    (source,) = case.data.sources
    source_slots = {int(pid): slot for slot, pid in enumerate(source.particle_id)}
    state_path = (
        CAMPAIGN / "original_comparison/particle_trajectory_review" / f"{name}_all_states.csv"
    )
    native = read_initial_states(state_path)
    p = {
        key: float(value["value"])
        for key, value in parameters.items()
        if key.startswith("release_")
    }
    records = []
    for probe_slot, pid in enumerate(primitives["point_ids"]):
        saved = native[int(pid)]
        slot = source_slots[int(pid)]
        r, z, vr, vz = (float(saved[key]) for key in ("r_m", "z_m", "vr_m_s", "vz_m_s"))
        theta = p["release_half_angle"] * math.sin(
            2
            * math.pi
            * (
                p["release_phase_r"] * (r - p["release_rmin"]) / p["release_dr"]
                + p["release_phase_z"] * (z - p["release_zmin"]) / p["release_dz"]
            )
        )
        ug_r = primitive_columns["gas_velocity_r_m_per_s"][probe_slot]
        ug_z = primitive_columns["gas_velocity_z_m_per_s"][probe_slot]
        speed = 10000 * p["release_speed_multiplier"] * math.hypot(ug_r, ug_z)
        reconstructed = (speed * math.sin(theta), p["release_axial_sign"] * speed * math.cos(theta))
        records.append(
            {
                "particle_id": int(pid),
                "r_m": r,
                "z_m": z,
                "theta_expression_rad": theta,
                "theta_saved_rad": math.atan2(vr, vz),
                "gas_speed_m_s": math.hypot(ug_r, ug_z),
                "speed_saved_m_s": math.hypot(vr, vz),
                "gas_slip_m_s": math.hypot(vr - ug_r, vz - ug_z),
                "expression_vector_residual_max_component_m_s": max(
                    abs(reconstructed[0] - vr), abs(reconstructed[1] - vz)
                ),
                "canonical_velocity_residual_max_component_m_s": max(
                    abs(float(source.velocity_m_s[slot, 0]) - vr),
                    abs(float(source.velocity_m_s[slot, 1]) - vz),
                ),
                "canonical_position_residual_max_component_m": max(
                    abs(float(source.position_m[slot, 0]) - r),
                    abs(float(source.position_m[slot, 1]) - z),
                ),
                "mass_kg": float(source.mass_kg[slot]),
                "charge_number": float(source.charge_number[slot]),
                "release_time_s": float(source.release_time_s[slot]),
            }
        )
    csv_path = HERE / f"{name}_initial_condition_audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    properties = feature["properties"]
    return {
        "bindings": [
            binding(primitive_path),
            binding(case_path),
            binding(case.data_path),
            binding(state_path),
            binding(csv_path),
        ],
        "canonical_logical_content_hash": case.content_hash,
        "source_feature_path": feature["path"],
        "active": feature["active"],
        "selected_settings": {
            key: properties[key]["value"]
            for key in (
                "GridType",
                "VelocitySpecification",
                "InitialVelocity",
                "v0",
                "rt",
                "SubtractMovingFrameVelocity",
                "ReleasedParticleProperties",
                "aux0_auxq",
            )
        },
        "inactive_values_not_used": {
            key: properties[key]["value"] for key in ("Ep0", "T0", "alphac", "M", "mp0", "dp0")
        },
        "native_geometry_unit": primitives["request"]["native_geometry_unit"],
        "native_geometry_unit_to_m": primitives["request"]["native_geometry_unit_to_m"],
        "parameters": {
            key: value for key, value in parameters.items() if key.startswith("release_")
        },
        "expressions": expressions,
        "count": len(records),
        "range_by_quantity": quantity_ranges(records),
        "rotating_frame_feature_count": sum(
            row["case"] == name
            and row["kind"] == "feature"
            and "Rotat" in row["payload"].get("type", "")
            for row in rows
        ),
        "limitations": [
            "287 realized initial rows only; no simulation, RNG reconstruction or population-validity claim.",
            "10000 is the evaluated diameter-conditional multiplier in these bound files, never a new-case default.",
            "Saved state and canonical comparisons share existing extraction lineage; the expression replay is independent arithmetic, not independent native extraction.",
            "No general guarantee about laboratory-frame velocity from SubtractMovingFrameVelocity=0; native frame dynamics must also be inspected.",
        ],
    }


def main() -> None:
    inventory = NATIVE / "native_readback.jsonl"
    rows = [json.loads(line) for line in inventory.read_text(encoding="utf-8").splitlines()]
    result = {
        "scope": "Read-only independent source-expression arithmetic and public load_case canonical readback.",
        "inventory_binding": binding(inventory),
        "arithmetic_program_binding": binding(Path(__file__).resolve()),
        "cases": {name: audit_case(rows, name) for name in ("caseA", "caseP")},
    }
    path = HERE / "initial_condition_audit.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: value["range_by_quantity"] for name, value in result["cases"].items()}))


if __name__ == "__main__":
    main()
