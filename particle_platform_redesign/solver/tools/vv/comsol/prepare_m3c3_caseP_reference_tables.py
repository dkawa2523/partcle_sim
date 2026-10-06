"""Materialize the locked M3-C3 Case-P common-P1 COMSOL tables.

This external V&V adapter does not derive physics.  It converts the canonical
P1 fields and the candidate-owned three-current release state into COMSOL
interpolation files and records their hashes.  In particular it never solves
for the initial charge; the supplied release CSV and canonical source must
already contain the same three-current Z0 values.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import h5py
import numpy as np
from numpy.typing import NDArray

TOOL_REVISION: Final = "m3c3_caseP_three_current_reference_tables_v1"
EXPECTED_PARTICLES: Final = 287


@dataclass(frozen=True, slots=True)
class Export:
    field: str
    component: str
    function: str
    unit: str


EXPORTS: Final = (
    Export("gas_density", "value", "m3c1_rhog", "kg/m^3"),
    Export("gas_dynamic_viscosity", "value", "m3c1_mug", "Pa*s"),
    Export("gas_temperature", "value", "m3c1_Tg", "K"),
    Export("gas_mean_free_path", "value", "m3c1_lambdag", "m"),
    Export("electron_number_density", "value", "m3c1_ne", "1/m^3"),
    Export("positive_ion_number_density", "value", "m3c1_ni", "1/m^3"),
    Export("electron_thermal_voltage", "value", "m3c1_Te", "V"),
    Export("positive_ion_thermal_voltage", "value", "m3c1_TiV", "V"),
    Export("effective_positive_ion_mass", "value", "m3c1_mi", "kg"),
    Export("screening_length", "value", "m3c1_lambdaD", "m"),
    Export("ion_neutral_mean_free_path", "value", "m3c1_lambdaIn", "m"),
    Export("azimuthal_gas_vorticity", "value", "m3c1_omegaPhi", "1/s"),
    Export("gas_velocity", "r", "m3c1_ugr", "m/s"),
    Export("gas_velocity", "z", "m3c1_ugz", "m/s"),
    Export("electric_field", "r", "m3c1_Er", "V/m"),
    Export("electric_field", "z", "m3c1_Ez", "V/m"),
    Export("positive_ion_velocity", "r", "m3c1_uir", "m/s"),
    Export("positive_ion_velocity", "z", "m3c1_uiz", "m/s"),
    Export("gradient_mean_e_squared", "r", "m3c1_gradE2r", "V^2/m^3"),
    Export("gradient_mean_e_squared", "z", "m3c1_gradE2z", "V^2/m^3"),
    Export("gas_translational_heat_flux", "r", "m3c1_qr", "W/m^2"),
    Export("gas_translational_heat_flux", "z", "m3c1_qz", "W/m^2"),
    Export("negative_ion_number_density", "value", "m3c3_nn", "1/m^3"),
    Export("negative_ion_velocity", "r", "m3c3_unr", "m/s"),
    Export("negative_ion_velocity", "z", "m3c3_unz", "m/s"),
    Export("negative_ion_thermal_voltage", "value", "m3c3_TnV", "V"),
    Export("effective_negative_ion_mass", "value", "m3c3_mn", "kg"),
)
RELEASE_HEADER: Final = (
    "particle_id",
    "release_time_s",
    "r_m",
    "z_m",
    "velocity_r_m_per_s",
    "velocity_z_m_per_s",
    "charge_number_e",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text(dataset: h5py.Dataset) -> str:
    value = dataset[()]
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _components(dataset: h5py.Dataset) -> tuple[str, ...]:
    return tuple(
        item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in dataset[()]
    )


def _expected_fields() -> set[str]:
    return {export.field for export in EXPORTS}


def _validate_field(
    fields: h5py.Group,
    export: Export,
    node_count: int,
) -> NDArray[np.float64]:
    group = fields[export.field]
    components = _components(group["components"])
    expected_basis = "scalar" if components == ("value",) else "axisymmetric_rz"
    values = np.asarray(group["values"], dtype=np.float64)
    if (
        _text(group["association"]) != "node"
        or _text(group["layout"]) != "plasma"
        or _text(group["stored_basis"]) != expected_basis
        or _text(group["unit"]) != export.unit
        or export.component not in components
        or values.shape != (node_count, len(components))
        or not np.all(np.isfinite(values))
    ):
        raise ValueError(f"canonical field metadata or values differ: {export.field}")
    return values[:, components.index(export.component)]


def _release_rows(path: Path) -> NDArray[np.float64]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != RELEASE_HEADER:
            raise ValueError("three-current release CSV has an unexpected header")
        rows = [[float(row[name]) for name in RELEASE_HEADER] for row in reader]
    result = np.asarray(rows, dtype=np.float64)
    if result.shape != (EXPECTED_PARTICLES, len(RELEASE_HEADER)) or not np.all(np.isfinite(result)):
        raise ValueError("three-current release CSV must contain 287 finite rows")
    if not np.array_equal(result[:, 0], np.arange(1, EXPECTED_PARTICLES + 1)):
        raise ValueError("three-current release particle IDs must be 1..287")
    if not np.all(result[:, 1] == 0.0):
        raise ValueError("three-current release times must all be zero")
    if np.unique(result[:, 2:4], axis=0).shape[0] != EXPECTED_PARTICLES:
        raise ValueError("three-current release positions must be unique")
    return result


def _write_sectionwise(
    path: Path,
    nodes_m: NDArray[np.float64],
    connectivity: NDArray[np.int64],
    function: str,
    values: NDArray[np.float64],
) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("%Coordinates\n")
        np.savetxt(stream, nodes_m, fmt="%.17g", delimiter=" ")
        stream.write("%Elements\n")
        np.savetxt(stream, connectivity + 1, fmt="%d", delimiter=" ")
        stream.write(f"%Data ({function})\n")
        np.savetxt(stream, values, fmt="%.17g")


def _write_release(path: Path, release: NDArray[np.float64], column: int) -> None:
    np.savetxt(path, release[:, [2, 3, column]], fmt="%.17g", delimiter=" ")


def prepare(
    candidate_h5: Path,
    release_csv: Path,
    output: Path,
    expected_content_hash: str | None = None,
) -> dict[str, object]:
    candidate_h5 = candidate_h5.resolve()
    release_csv = release_csv.resolve()
    if output.exists():
        raise FileExistsError(f"reference output already exists: {output}")
    if expected_content_hash is not None and not expected_content_hash.startswith("sha256:"):
        raise ValueError("expected canonical content hash must use the sha256: prefix")
    release = _release_rows(release_csv)

    output.mkdir(parents=True)
    artifacts: list[Path] = []
    try:
        with h5py.File(candidate_h5, "r") as source:
            fields = source["fields"]
            if set(fields.keys()) != _expected_fields():
                raise ValueError(
                    "canonical three-current field set differs; "
                    f"missing={sorted(_expected_fields() - set(fields.keys()))}, "
                    f"extra={sorted(set(fields.keys()) - _expected_fields())}"
                )
            nodes_m = np.asarray(source["layouts/plasma/unstructured/nodes_m"], dtype=np.float64)
            connectivity = np.asarray(
                source["layouts/plasma/unstructured/connectivity"], dtype=np.int64
            )
            support = np.asarray(source["layouts/plasma/unstructured/cell_support"], dtype=np.uint8)
            layout_shape = (
                nodes_m.ndim,
                nodes_m.shape[1:],
                connectivity.ndim,
                connectivity.shape[1:],
                support.shape,
            )
            expected_shape = (2, (2,), 2, (3,), (connectivity.shape[0],))
            if layout_shape != expected_shape or not np.all(support == 1):
                raise ValueError("canonical exact-P1 layout differs")
            for export in EXPORTS:
                values = _validate_field(fields, export, nodes_m.shape[0])
                path = output / f"{export.function}_sectionwise.txt"
                _write_sectionwise(path, nodes_m, connectivity, export.function, values)
                artifacts.append(path)

            source_table = source["sources/particles"]
            source_values = np.column_stack(
                (
                    source_table["particle_id"],
                    source_table["release_time_s"],
                    source_table["position_m"],
                    source_table["velocity_m_s"],
                    source_table["charge_number"],
                )
            ).astype(np.float64, copy=False)
            if not np.array_equal(source_values, release):
                raise ValueError(
                    "candidate-owned release CSV differs from canonical three-current source"
                )
        for name, column in (("m3c1_vr0", 4), ("m3c1_vz0", 5), ("m3c1_Z0", 6)):
            path = output / f"{name}.txt"
            _write_release(path, release, column)
            artifacts.append(path)
        staged_release = output / "three_current_release_state.csv"
        shutil.copyfile(release_csv, staged_release)
        artifacts.append(staged_release)

        receipt: dict[str, object] = {
            "schema_version": 1,
            "tool_revision": TOOL_REVISION,
            "status": "PASS",
            "candidate": {
                "path": str(candidate_h5),
                "file_sha256": _sha256(candidate_h5),
                "content_hash": expected_content_hash,
            },
            "release": {
                "authority": "candidate_owned_three_current_release_state",
                "input_path": str(release_csv),
                "input_sha256": _sha256(release_csv),
                "particle_count": EXPECTED_PARTICLES,
                "initial_charge_recomputed_by_comsol": False,
            },
            "layout": {
                "node_count": int(nodes_m.shape[0]),
                "cell_count": int(connectivity.shape[0]),
                "interpolation": "exact_first_order_triangular_sectionwise",
            },
            "component_count": len(EXPORTS),
            "artifacts": {
                path.name: {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
                for path in sorted(artifacts)
            },
        }
        (output / "m3c3_table_receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return receipt
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_h5", type=Path)
    parser.add_argument("release_csv", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-content-hash", required=True)
    arguments = parser.parse_args()
    prepare(
        arguments.candidate_h5,
        arguments.release_csv,
        arguments.output,
        arguments.expected_content_hash,
    )


if __name__ == "__main__":
    main()
