"""Map COMSOL cache DOFs to the locked common-P1 nodes and add five primitives."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

import h5py
import numpy as np
from numpy.typing import NDArray

EXPECTED_MPH_SHA256 = "3bbf08e3469758313eac5de473a7a0dd4cc9a6f72c9722393229f0b856e9b524"
EXPECTED_INPUT_SHA256 = "c54b4b658213230e82307ca89018538c0240bfcc83e793206d5093f4f08908e9"
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


def _write_field(
    fields: h5py.Group,
    name: str,
    values: NDArray[np.float64],
    components: tuple[str, ...],
    unit: str,
    stored_basis: str,
) -> None:
    if name in fields:
        raise ValueError(f"field already exists: {name}")
    group = fields.create_group(name)
    text = h5py.string_dtype(encoding="utf-8")
    group.create_dataset("association", data="node", dtype=text)
    group.create_dataset("components", data=np.asarray(components, dtype=text))
    group.create_dataset("layout", data="plasma", dtype=text)
    group.create_dataset("stored_basis", data=stored_basis, dtype=text)
    group.create_dataset("unit", data=unit, dtype=text)
    group.create_dataset("values", data=np.asarray(values, dtype=np.float64))


def normalize(args: argparse.Namespace) -> dict[str, object]:
    canonical_input = args.canonical_input.resolve()
    source_mph = args.source_mph.resolve()
    output = args.output.resolve()
    receipt = args.receipt.resolve()
    if output.exists() or receipt.exists():
        raise FileExistsError("M3-C3 output and receipt are no-clobber artifacts")
    if _sha256(canonical_input) != EXPECTED_INPUT_SHA256:
        raise ValueError("locked common-P1 input hash mismatch")
    source_hash_before = _sha256(source_mph)
    if source_hash_before != EXPECTED_MPH_SHA256:
        raise ValueError("locked source MPH hash mismatch")

    domain = _read_provider(args.domain_csv.resolve())
    boundary = _read_provider(args.boundary_csv.resolve())
    with h5py.File(canonical_input, "r") as source:
        nodes_m = np.asarray(source["geometry/nodes_m"], dtype=np.float64)
        lines = np.asarray(source["geometry/boundary/line2"], dtype=np.int64)
        gas_temperature_K = np.asarray(source["fields/gas_temperature/values"], dtype=np.float64)[
            :, 0
        ]
    boundary_mask = np.zeros(nodes_m.shape[0], dtype=np.bool_)
    boundary_mask[np.unique(lines)] = True
    cache, mapping = _map_nodes(nodes_m, boundary_mask, domain, boundary)

    density = cache[:, 0]
    if np.any(density <= 0.0):
        raise ValueError("negative-ion density must be positive at every canonical node")
    velocity = cache[:, 1:3] / density[:, None]
    effective_mass = cache[:, 3] / density
    thermal_voltage = K_B_J_PER_K * gas_temperature_K / ELEMENTARY_CHARGE_C
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

    axis = np.abs(nodes_m[:, 0]) <= COORDINATE_TOLERANCE_M
    source_axis_radial_max = float(np.max(np.abs(velocity[axis, 0]), initial=0.0))
    velocity[axis, 0] = 0.0
    speed_max = float(np.max(np.linalg.norm(velocity, axis=1)))

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary output already exists: {temporary}")
    try:
        shutil.copy2(canonical_input, temporary)
        with h5py.File(temporary, "r+") as target:
            fields = target["fields"]
            _write_field(
                fields,
                "negative_ion_number_density",
                density[:, None],
                ("value",),
                "1/m^3",
                "scalar",
            )
            _write_field(
                fields,
                "negative_ion_velocity",
                velocity,
                ("r", "z"),
                "m/s",
                "axisymmetric_rz",
            )
            _write_field(
                fields,
                "effective_negative_ion_mass",
                effective_mass[:, None],
                ("value",),
                "kg",
                "scalar",
            )
            _write_field(
                fields,
                "negative_ion_thermal_voltage",
                thermal_voltage[:, None],
                ("value",),
                "V",
                "scalar",
            )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()

    source_hash_after = _sha256(source_mph)
    if source_hash_after != source_hash_before:
        raise RuntimeError("locked source MPH changed during normalization")
    record: dict[str, object] = {
        "schema_version": 1,
        "evidence_id": "M3-C3-caseP-negative-ion-primitives-v1",
        "status": "PASS",
        "source_mph_sha256_before": source_hash_before,
        "source_mph_sha256_after": source_hash_after,
        "source_unchanged": True,
        "canonical_input_sha256": EXPECTED_INPUT_SHA256,
        "canonical_output_sha256": _sha256(output),
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
