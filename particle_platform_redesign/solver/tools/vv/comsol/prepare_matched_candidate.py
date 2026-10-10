"""Prepare and run the solver side of the first deterministic M3-V match.

This is an external V&V utility. It translates already exported COMSOL
primitives into the canonical case format, then exercises only the public
``load_case`` / ``simulate`` / ``open_result`` API. It does not add a COMSOL
mode to the production solver.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Final

import numpy as np
import yaml

from chamber_particles import load_case, open_result, simulate
from chamber_particles.case_format import FieldData, RealizedTableSource, read, write

TOOL_REVISION: Final = "m3v_matched_candidate_v1"
_OUTPUT_INTERVAL_S: Final = 1.0e-5
_END_TIME_S: Final = 4.0e-4
_DT_ROWS: Final = (("10us", 1.0e-5), ("5us", 5.0e-6), ("2p5us", 2.5e-6))
_GAS_MOLECULAR_MASS_KG: Final = (0.8 * 0.0880043 + 0.2 * 0.0319988) / 6.02214076e23
_EPSTEIN_DELTA: Final = 1.0 + 0.9 * math.pi / 8.0
_LIFECYCLE: Final = ("pending", "active", "stuck", "escaped", "failed")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dictionary(path: Path) -> tuple[list[str], dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(line for line in stream if not line.startswith("%"))
        rows = list(reader)
    if not rows or reader.fieldnames != ["column", "COMSOL_expression", "unit"]:
        raise ValueError(f"invalid COMSOL column dictionary: {path}")
    names = [row["column"] for row in rows]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate COMSOL columns: {path}")
    return names, {row["column"]: row["unit"] for row in rows}


def _numeric_table(path: Path, column_count: int) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(line for line in stream if not line.startswith("%"))
        for line_number, row in enumerate(reader, start=1):
            if not row:
                continue
            if len(row) != column_count:
                raise ValueError(f"{path}: row {line_number} has {len(row)} columns")
            rows.append([float(token) for token in row])
    if not rows:
        raise ValueError(f"empty COMSOL numeric table: {path}")
    return np.asarray(rows, dtype=np.float64)


def _coordinate_match(
    canonical: np.ndarray,
    provider: np.ndarray,
    tolerance_m: float,
) -> tuple[np.ndarray, float]:
    if np.unique(provider, axis=0).shape[0] != provider.shape[0]:
        raise ValueError("provider field coordinates are not unique")
    matched = np.empty(canonical.shape[0], dtype=np.int64)
    distance2 = np.empty(canonical.shape[0], dtype=np.float64)
    tolerance2 = tolerance_m * tolerance_m
    for start in range(0, canonical.shape[0], 256):
        stop = min(start + 256, canonical.shape[0])
        delta = canonical[start:stop, None, :] - provider[None, :, :]
        squared = np.sum(delta * delta, axis=2)
        within = np.count_nonzero(squared <= tolerance2, axis=1)
        if bool((within != 1).any()):
            raise ValueError("canonical field nodes do not have a unique provider match")
        local = np.argmin(squared, axis=1)
        matched[start:stop] = local
        distance2[start:stop] = squared[np.arange(stop - start), local]
    if np.unique(matched).size != matched.size:
        raise ValueError("provider-to-canonical field match is not bijective")
    return matched, float(np.sqrt(np.max(distance2, initial=0.0)))


def _reference_electric_fields(
    package: Path,
    nodes_m: np.ndarray,
    layout_name: str,
) -> tuple[tuple[FieldData, FieldData], dict[str, object]]:
    dictionary_path = package / "config" / "background_field_column_dictionary.csv"
    values_path = package / "input_fields" / "background_fields_mesh_points.csv"
    names, units = _dictionary(dictionary_path)
    index = {name: position for position, name in enumerate(names)}
    required = {
        "r_m": "m",
        "z_m": "m",
        "SASS_potential_V": "V",
        "electric_field_r_V_per_m": "V/m",
        "electric_field_z_V_per_m": "V/m",
    }
    if any(units.get(name) != unit for name, unit in required.items()):
        raise ValueError("reference electric-field columns or units do not match")
    values = _numeric_table(values_path, len(names))
    coordinate_columns = [index["r_m"], index["z_m"]]
    value_columns = [
        index["electric_field_r_V_per_m"],
        index["electric_field_z_V_per_m"],
        index["SASS_potential_V"],
    ]
    finite = np.isfinite(values[:, value_columns]).all(axis=1)
    provider_coordinates = values[finite][:, coordinate_columns]
    provider_values = values[finite][:, value_columns]
    matched, maximum_distance = _coordinate_match(nodes_m, provider_coordinates, 1.0e-14)
    ordered = np.ascontiguousarray(provider_values[matched], dtype="<f8")
    axis = nodes_m[:, 0] == 0.0
    maximum_axis_radial = float(np.max(np.abs(ordered[axis, 0]), initial=0.0))
    corrected_axis_nodes = int(np.count_nonzero(ordered[axis, 0]))
    ordered[axis, 0] = 0.0
    fields = (
        FieldData(
            "electric_field",
            layout_name,
            "node",
            ("r", "z"),
            "axisymmetric_rz",
            np.ascontiguousarray(ordered[:, :2], dtype="<f8"),
            "V/m",
        ),
        FieldData(
            "electric_potential",
            layout_name,
            "node",
            ("value",),
            "scalar",
            np.ascontiguousarray(ordered[:, 2:3], dtype="<f8"),
            "V",
        ),
    )
    evidence: dict[str, object] = {
        "maximum_coordinate_match_distance_m": maximum_distance,
        "axis_radial_electric_projection": {
            "corrected_node_count": corrected_axis_nodes,
            "maximum_correction_V_per_m": maximum_axis_radial,
            "reason": "canonical axisymmetric vector regularity",
        },
        "column_dictionary_sha256": _sha256(dictionary_path),
        "field_values_sha256": _sha256(values_path),
    }
    return fields, evidence


def _release_source(package: Path) -> tuple[RealizedTableSource, dict[str, object]]:
    path = package / "results" / "release_state_t0_tidy.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(line for line in stream if not line.startswith("%")))
    if len(rows) != 287:
        raise ValueError(f"expected 287 release rows, got {len(rows)}")

    def vector(name: str) -> np.ndarray:
        return np.asarray([float(row[name]) for row in rows], dtype="<f8")

    particle_id = np.asarray([int(row["particle_id"]) for row in rows], dtype="<i8")
    diameter = vector("particle_diameter_m")
    radius = vector("particle_radius_m")
    mass = vector("particle_mass_kg")
    volume = math.pi * diameter**3 / 6.0
    source = RealizedTableSource(
        name="particles",
        particle_id=particle_id,
        release_time_s=np.zeros(len(rows), dtype="<f8"),
        position_m=np.column_stack((vector("r_m"), vector("z_m"))).astype("<f8"),
        velocity_m_s=np.column_stack(
            (vector("velocity_r_m_per_s"), vector("velocity_z_m_per_s"))
        ).astype("<f8"),
        charge_number=np.full(len(rows), -1.0, dtype="<f8"),
        mass_kg=mass,
        drag_diameter_m=diameter,
        contact_radius_m=np.zeros(len(rows), dtype="<f8"),
        electrostatic_radius_m=radius,
        displaced_volume_m3=volume.astype("<f8"),
        model_weight=np.ones(len(rows), dtype="<f8"),
        material_id=np.zeros(len(rows), dtype="<i4"),
    )
    evidence: dict[str, object] = {
        "release_table_sha256": _sha256(path),
        "particle_count": len(rows),
        "particle_id_min": int(particle_id.min()),
        "particle_id_max": int(particle_id.max()),
    }
    return source, evidence


def _case_document(data_name: str, content_hash: str, dt_s: float) -> dict[str, object]:
    output_times = [index * _OUTPUT_INTERVAL_S for index in range(41)]
    return {
        "format_version": 3,
        "case": {
            "name": f"m3v_caseA_100nm_common_deterministic_dt_{dt_s:.9g}",
            "data_path": data_name,
            "expected_content_hash": content_hash,
        },
        "motion": {"mode": "axisymmetric_rz_meridional"},
        "time": {"start_s": 0.0, "end_s": _END_TIME_S, "dt_s": dt_s},
        "solver": {
            "integrator": "rk4_fixed",
            "backend": "cpu",
            "seed": 20260930,
            "event": {
                "geometry_rtol": 1.0e-12,
                "roundoff_ulps": 64,
                "max_refinements": 48,
                "max_interactions_per_step": 8,
                "corner_policy": "priority_then_combined_normal_v1",
            },
        },
        "resources": {"memory_limit_mb": 512},
        "physics": {
            "charge": {"model": "fixed"},
            "drag": {
                "model": "epstein_linear",
                "revision": "epstein_linear_v1",
                "gas_velocity_field": "gas_velocity",
                "gas_density_field": "gas_density",
                "gas_temperature_field": "gas_temperature",
                "gas_mean_free_path_field": "gas_mean_free_path",
                "gas_molecular_mass_kg": _GAS_MOLECULAR_MASS_KG,
                "delta": _EPSTEIN_DELTA,
                "applicability": "error",
            },
            "electric": {
                "model": "coulomb",
                "revision": "electric_coulomb_v1",
                "electric_field": "electric_field",
            },
            "gravity_buoyancy": {
                "model": "standard",
                "revision": "gravity_buoyancy_standard_v1",
                "gas_density_field": "gas_density",
                "gravity_m_s2": [0.0, -9.80665],
            },
        },
        "sources": [{"name": "release", "type": "table", "table": "particles"}],
        "boundaries": [
            {"boundary_group": name, "priority": 10, "law": law}
            for name, law in (
                ("wafer", "stick"),
                ("grounded_wall", "stick"),
                ("focus_transition", "stick"),
                ("outer_dielectric", "stick"),
                ("pump_outlet", "escape"),
                ("gas_inlet", "escape"),
            )
        ],
        "output": {
            "trajectories": {
                "selection": "all",
                "schedule": {"explicit_times_s": output_times},
            },
            "probes": None,
        },
    }


def prepare(package: Path, base_data: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.mkdir(parents=True)
    base = read(base_data)
    if base.coordinate_system != "axisymmetric_rz" or len(base.layouts) != 1:
        raise ValueError("base canonical data must contain one axisymmetric RZ layout")
    layout = base.layouts[0]
    nodes_m = getattr(layout, "nodes_m", None)
    if nodes_m is None:
        raise ValueError("base canonical data must use a node-based P1/Q1 layout")
    electric_fields, field_evidence = _reference_electric_fields(package, nodes_m, layout.name)
    source, source_evidence = _release_source(package)
    provenance = json.loads(base.provenance_json)
    provenance["matched_candidate"] = {
        "tool_revision": TOOL_REVISION,
        "scope": "caseA_100nm_common_deterministic_pre_first_material_event",
        "base_data_sha256": _sha256(base_data),
        **field_evidence,
        **source_evidence,
    }
    data_path = output / "candidate_input.h5"
    data = replace(
        base,
        provenance_json=json.dumps(
            provenance, sort_keys=True, separators=(",", ":"), allow_nan=False
        ),
        fields=tuple(sorted((*base.fields, *electric_fields), key=lambda field: field.name)),
        sources=(source,),
    )
    info = write(data_path, data)
    cases: dict[str, str] = {}
    for label, dt_s in _DT_ROWS:
        case_path = output / f"candidate_{label}.yaml"
        case_path.write_text(
            yaml.safe_dump(
                _case_document(data_path.name, info.content_hash, dt_s), sort_keys=False
            ),
            encoding="utf-8",
        )
        cases[label] = case_path.name
    report: dict[str, object] = {
        "status": "prepared",
        "tool_revision": TOOL_REVISION,
        "scope": "caseA_100nm_common_deterministic_pre_first_material_event",
        "base_data": str(base_data.resolve()),
        "base_data_sha256": _sha256(base_data),
        "input_data": data_path.name,
        "input_content_hash": info.content_hash,
        "input_file_sha256": _sha256(data_path),
        "cases": cases,
        "field_evidence": field_evidence,
        "source_evidence": source_evidence,
        "settings": {
            "coordinate_system": "axisymmetric_rz_no_swirl",
            "integrator": "classical_rk4_fixed",
            "dt_s": [row[1] for row in _DT_ROWS],
            "end_time_s": _END_TIME_S,
            "output_interval_s": _OUTPUT_INTERVAL_S,
            "charge": "fixed_charge_number_minus_one",
            "brownian": "disabled",
            "forces": [
                "electric_coulomb_v1",
                "epstein_linear_v1_delta_1_plus_0p9_pi_over_8",
                "gravity_buoyancy_standard_v1",
            ],
        },
    }
    (output / "prepare_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _write_trajectory(path: Path, result: object) -> int:
    row_count = 0
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "particle_id",
                "time_s",
                "r_m",
                "z_m",
                "velocity_r_m_per_s",
                "velocity_z_m_per_s",
                "charge_number_e",
                "lifecycle",
            )
        )
        for frame in result.iter_frames():  # type: ignore[attr-defined]
            for index, particle_id in enumerate(frame.particle_id):
                code = int(frame.lifecycle[index])
                writer.writerow(
                    (
                        int(particle_id),
                        frame.time_s,
                        frame.position_m[index, 0],
                        frame.position_m[index, 1],
                        frame.velocity_m_s[index, 0],
                        frame.velocity_m_s[index, 1],
                        frame.charge_number[index],
                        _LIFECYCLE[code],
                    )
                )
                row_count += 1
    return row_count


def _write_events(path: Path, result: object) -> int:
    events = result.read_boundary_events()  # type: ignore[attr-defined]
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "particle_id",
                "event_ordinal",
                "event_time_s",
                "hit_r_m",
                "hit_z_m",
                "normal_r",
                "normal_z",
                "pre_velocity_r_m_per_s",
                "pre_velocity_z_m_per_s",
                "post_velocity_r_m_per_s",
                "post_velocity_z_m_per_s",
                "event_type",
                "boundary_semantic",
                "outcome",
            )
        )
        for index, particle_id in enumerate(events.particle_id):
            writer.writerow(
                (
                    int(particle_id),
                    int(events.event_ordinal[index]),
                    events.time_s[index],
                    events.position_m[index, 0],
                    events.position_m[index, 1],
                    events.normal[index, 0],
                    events.normal[index, 1],
                    events.velocity_pre_m_s[index, 0],
                    events.velocity_pre_m_s[index, 1],
                    events.velocity_post_m_s[index, 0],
                    events.velocity_post_m_s[index, 1],
                    "material_boundary",
                    int(events.boundary_id[index]),
                    events.outcome[index],
                )
            )
    return int(events.particle_id.size)


def run(prepared: Path) -> dict[str, object]:
    report_path = prepared / "prepare_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    runs: dict[str, object] = {}
    for label, _dt_s in _DT_ROWS:
        case_path = prepared / str(report["cases"][label])  # type: ignore[index]
        result_path = prepared / f"result_{label}"
        summary = simulate(load_case(case_path), result_path)
        result = open_result(result_path)
        trajectory_path = prepared / f"candidate_trajectory_{label}.csv"
        events_path = prepared / f"candidate_events_{label}.csv"
        trajectory_rows = _write_trajectory(trajectory_path, result)
        event_rows = _write_events(events_path, result)
        runs[label] = {
            "case": case_path.name,
            "result": result_path.name,
            "status": result.manifest["status"],
            "engine_algorithm_revision": result.manifest["engine_algorithm_revision"],
            "trajectory": trajectory_path.name,
            "trajectory_sha256": _sha256(trajectory_path),
            "trajectory_rows": trajectory_rows,
            "events": events_path.name,
            "events_sha256": _sha256(events_path),
            "event_rows": event_rows,
            "failure_event_count": summary.failure_event_count,
        }
    run_report: dict[str, object] = {
        "status": "complete",
        "tool_revision": TOOL_REVISION,
        "public_api_path": ["load_case", "simulate", "open_result"],
        "runs": runs,
    }
    (prepared / "candidate_run_report.json").write_text(
        json.dumps(run_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return run_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("package", type=Path)
    prepare_parser.add_argument("base_data", type=Path)
    prepare_parser.add_argument("output", type=Path)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("prepared", type=Path)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    if arguments.command == "prepare":
        report = prepare(
            arguments.package.resolve(), arguments.base_data.resolve(), arguments.output
        )
    else:
        report = run(arguments.prepared.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
