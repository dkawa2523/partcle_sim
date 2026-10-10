"""Map COMSOL cache DOFs to current common-P1 nodes and add four primitives."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from chamber_particles.case_format import DataBundle, FieldData, P1TriLayout, read_with_info, write

EXPECTED_MPH_SHA256 = "3bbf08e3469758313eac5de473a7a0dd4cc9a6f72c9722393229f0b856e9b524"
GEOMETRY_CM_TO_M = 0.01
COORDINATE_TOLERANCE_M = 1.0e-14
K_B_J_PER_K = 1.380649e-23
ELEMENTARY_CHARGE_C = 1.602176634e-19
AVOGADRO_PER_MOL = 6.02214076e23


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_provider(path: Path) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    with path.open(newline="", encoding="utf-8") as stream:
        lines = (
            line[2:] if line.startswith("% r_geom_cm,") else line
            for line in stream
            if not line.startswith("% ") or line.startswith("% r_geom_cm,")
        )
        rows = list(csv.DictReader(lines))
    if not rows:
        raise ValueError(f"provider CSV is empty: {path}")
    coordinates_m = (
        np.asarray(
            [[float(row["r_geom_cm"]), float(row["z_geom_cm"])] for row in rows],
            dtype=np.float64,
        )
        * GEOMETRY_CM_TO_M
    )
    values = np.asarray(
        [
            [
                float(row["negative_ion_density_per_m3"]),
                float(row["negative_ion_number_flux_r_per_m2_s"]),
                float(row["negative_ion_number_flux_z_per_m2_s"]),
                float(row["negative_ion_mass_density_kg_per_m3"]),
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(coordinates_m)) or not np.all(np.isfinite(values)):
        raise ValueError(f"provider CSV contains nonfinite data: {path}")
    return coordinates_m, values


def _map_nodes(
    nodes_m: NDArray[np.float64],
    boundary_mask: NDArray[np.bool_],
    domain: tuple[NDArray[np.float64], NDArray[np.float64]],
    boundary: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> tuple[NDArray[np.float64], dict[str, float | int]]:
    mapped = np.empty((nodes_m.shape[0], 4), dtype=np.float64)
    maximum_delta = 0.0
    duplicate_matches = 0
    for index, coordinate in enumerate(nodes_m):
        provider_coordinates, provider_values = boundary if boundary_mask[index] else domain
        deltas = np.max(np.abs(provider_coordinates - coordinate), axis=1)
        matches = np.flatnonzero(deltas <= COORDINATE_TOLERANCE_M)
        if matches.size == 0:
            owner = "boundary" if boundary_mask[index] else "domain"
            raise ValueError(f"canonical node {index} has no {owner} cache DOF")
        reference = provider_values[matches[0]]
        if matches.size > 1:
            duplicate_matches += int(matches.size - 1)
            if not np.allclose(provider_values[matches], reference, rtol=1.0e-10, atol=0.0):
                raise ValueError(f"canonical node {index} has inconsistent duplicate cache DOFs")
        mapped[index] = reference
        maximum_delta = max(maximum_delta, float(np.max(deltas[matches])))
    return mapped, {
        "maximum_coordinate_match_delta_m": maximum_delta,
        "duplicate_provider_rows_verified_consistent": duplicate_matches,
    }


def _checked_gas_temperature(bundle: DataBundle) -> FieldData:
    gas = next((field for field in bundle.fields if field.name == "gas_temperature"), None)
    if gas is None:
        raise ValueError("canonical gas_temperature is required")
    layout = next(layout for layout in bundle.layouts if layout.name == gas.layout)
    if (
        not isinstance(layout, P1TriLayout)
        or gas.association != "node"
        or gas.time_s is not None
        or not np.array_equal(layout.nodes_m, bundle.geometry.nodes_m)
    ):
        raise ValueError("negative-ion projection requires a static common-node P1 temperature")
    return gas


def _validate_negative_primitives(
    density: NDArray[np.float64],
    velocity: NDArray[np.float64],
    effective_mass: NDArray[np.float64],
    thermal_voltage: NDArray[np.float64],
) -> None:
    if np.any(density <= 0.0):
        raise ValueError("negative-ion density must be positive at every canonical node")
    arrays = (density, velocity, effective_mass, thermal_voltage)
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("canonical negative-ion primitives contain nonfinite values")
    lower_mass = 0.016 / AVOGADRO_PER_MOL
    upper_mass = 0.019 / AVOGADRO_PER_MOL
    if np.min(effective_mass) < lower_mass * (1.0 - 1.0e-8) or np.max(
        effective_mass
    ) > upper_mass * (1.0 + 1.0e-8):
        raise ValueError("effective negative-ion mass is outside the F-/O- mixture interval")
    if np.any(thermal_voltage <= 0.0):
        raise ValueError("negative-ion thermal voltage must be positive")


def normalize(args: argparse.Namespace) -> dict[str, object]:
    canonical_input = args.canonical_input.resolve()
    source_mph = args.source_mph.resolve()
    output = args.output.resolve()
    receipt = args.receipt.resolve()
    if output.exists() or receipt.exists():
        raise FileExistsError("M3-C3 output and receipt are no-clobber artifacts")
    input_hash = _sha256(canonical_input)
    if input_hash != args.expected_input_sha256:
        raise ValueError("locked common-P1 input hash mismatch")
    source_hash_before = _sha256(source_mph)
    if source_hash_before != EXPECTED_MPH_SHA256:
        raise ValueError("locked source MPH hash mismatch")

    domain = _read_provider(args.domain_csv.resolve())
    boundary = _read_provider(args.boundary_csv.resolve())
    bundle, input_info = read_with_info(canonical_input)
    nodes_m = bundle.geometry.nodes_m
    lines = bundle.geometry.boundary.line2
    gas = _checked_gas_temperature(bundle)
    gas_temperature_K = gas.values[:, 0]
    boundary_mask = np.zeros(nodes_m.shape[0], dtype=np.bool_)
    boundary_mask[np.unique(lines)] = True
    cache, mapping = _map_nodes(nodes_m, boundary_mask, domain, boundary)

    density = cache[:, 0].copy()
    velocity = cache[:, 1:3] / density[:, None]
    effective_mass = cache[:, 3] / density
    thermal_voltage = K_B_J_PER_K * gas_temperature_K / ELEMENTARY_CHARGE_C
    _validate_negative_primitives(density, velocity, effective_mass, thermal_voltage)

    axis = np.abs(nodes_m[:, 0]) <= COORDINATE_TOLERANCE_M
    source_axis_radial_max = float(np.max(np.abs(velocity[axis, 0]), initial=0.0))
    velocity[axis, 0] = 0.0
    speed_max = float(np.max(np.linalg.norm(velocity, axis=1)))

    output.parent.mkdir(parents=True, exist_ok=True)
    added = (
        FieldData(
            "negative_ion_number_density",
            gas.layout,
            "node",
            ("value",),
            "scalar",
            density[:, None],
            "1/m^3",
        ),
        FieldData(
            "negative_ion_velocity",
            gas.layout,
            "node",
            ("r", "z"),
            "axisymmetric_rz",
            velocity,
            "m/s",
        ),
        FieldData(
            "effective_negative_ion_mass",
            gas.layout,
            "node",
            ("value",),
            "scalar",
            effective_mass[:, None],
            "kg",
        ),
        FieldData(
            "negative_ion_thermal_voltage",
            gas.layout,
            "node",
            ("value",),
            "scalar",
            thermal_voltage[:, None],
            "V",
        ),
    )
    existing = {field.name for field in bundle.fields}
    if any(field.name in existing for field in added):
        raise ValueError("negative-ion primitive already exists")
    output_info = write(output, replace(bundle, fields=bundle.fields + added))

    source_hash_after = _sha256(source_mph)
    if source_hash_after != source_hash_before:
        raise RuntimeError("locked source MPH changed during normalization")
    record: dict[str, object] = {
        "schema_version": 1,
        "evidence_id": "M3-C3-caseP-negative-ion-primitives-v2",
        "tool_revision": "m3c3_negative_ion_primitives_v2",
        "status": "PASS",
        "source_mph_sha256_before": source_hash_before,
        "source_mph_sha256_after": source_hash_after,
        "source_unchanged": True,
        "canonical_input_sha256": input_hash,
        "canonical_input_content_hash": input_info.content_hash,
        "canonical_output_sha256": _sha256(output),
        "canonical_output_content_hash": output_info.content_hash,
        "canonical_schema_version": output_info.schema_version,
        "canonical_node_count": int(nodes_m.shape[0]),
        "boundary_node_count": int(np.count_nonzero(boundary_mask)),
        "domain_provider_point_count": int(domain[0].shape[0]),
        "boundary_provider_point_count": int(boundary[0].shape[0]),
        "coordinate_provenance": {
            "comsol_geometry_unit": "cm",
            "canonical_unit": "m",
            "conversion_factor_cm_to_m": GEOMETRY_CM_TO_M,
            "match_tolerance_m": COORDINATE_TOLERANCE_M,
            **mapping,
        },
        "primitive_definition": {
            "density": "n_F_minus + n_O_minus",
            "velocity": "sum_s(n_s*(gas_velocity+species_diffusion_velocity))/sum_s(n_s)",
            "effective_mass": "sum_s(m_s*n_s)/sum_s(n_s)",
            "thermal_voltage": "k_B*T_g/e",
            "temperature_authority": "canonical gas_temperature field",
        },
        "ranges": {
            "negative_ion_number_density_per_m3": [float(np.min(density)), float(np.max(density))],
            "negative_ion_velocity_r_m_per_s": [
                float(np.min(velocity[:, 0])),
                float(np.max(velocity[:, 0])),
            ],
            "negative_ion_velocity_z_m_per_s": [
                float(np.min(velocity[:, 1])),
                float(np.max(velocity[:, 1])),
            ],
            "negative_ion_speed_max_m_per_s": speed_max,
            "effective_negative_ion_mass_kg": [
                float(np.min(effective_mass)),
                float(np.max(effective_mass)),
            ],
            "negative_ion_thermal_voltage_V": [
                float(np.min(thermal_voltage)),
                float(np.max(thermal_voltage)),
            ],
            "source_axis_radial_velocity_abs_max_m_per_s": source_axis_radial_max,
            "axis_radial_velocity_abs_max_after_projection_m_per_s": float(
                np.max(np.abs(velocity[axis, 0]), initial=0.0)
            ),
        },
        "axis_regularity_projection": {
            "rule": "negative_ion_velocity_r(r=0)=+0.0",
            "reason": "canonical axisymmetric RZ vector regularity",
            "projected_node_count": int(np.count_nonzero(axis)),
            "source_abs_max_m_per_s": source_axis_radial_max,
            "global_speed_max_after_projection_m_per_s": speed_max,
        },
        "artifacts": {
            "domain_csv_sha256": _sha256(args.domain_csv.resolve()),
            "boundary_csv_sha256": _sha256(args.boundary_csv.resolve()),
            "java_source_sha256": _sha256(args.java_source.resolve()),
            "runner_script_sha256": _sha256(args.runner_script.resolve()),
            "normalizer_sha256": _sha256(Path(__file__).resolve()),
        },
        "model_save": False,
        "coordinate_nudge": False,
        "missing_value_imputation": False,
        "different_domain_fallback": False,
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-input", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--domain-csv", type=Path, required=True)
    parser.add_argument("--boundary-csv", type=Path, required=True)
    parser.add_argument("--source-mph", type=Path, required=True)
    parser.add_argument("--java-source", type=Path, required=True)
    parser.add_argument("--runner-script", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser


if __name__ == "__main__":
    normalize(_parser().parse_args())
