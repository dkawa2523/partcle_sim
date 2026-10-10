"""Export the full M3-C1 canonical P1 state for a COMSOL common-field run.

This is an external V&V adapter.  It materializes every primitive field and
the realized release state already stored in one canonical ``candidate_input``
file; it neither derives replacement physics nor changes the production
solver.  One explicit table below owns the field/component/function mapping.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from chamber_particles.case_format import (
    CaseFileInfo,
    DataBundle,
    FieldData,
    P1TriLayout,
    RealizedTableSource,
    read_with_info,
)
from chamber_particles.fields import FieldBatch, RequiredFieldMetadata, prepare_required_fields
from tools.vv.comsol.actual_run_receipt import write_boundary_meaning

TOOL_REVISION: Final = "m3c1_full_physics_common_p1_tables_v2"
EXPECTED_RELEASE_COUNT: Final = 287


@dataclass(frozen=True, slots=True)
class ComponentExport:
    """One canonical component and its COMSOL interpolation function."""

    field: str
    component: str
    function: str
    unit: str
    probe_column: str


@dataclass(frozen=True, slots=True)
class _CommonInput:
    info: CaseFileInfo
    layout: P1TriLayout
    fields: dict[str, FieldData]
    source: RealizedTableSource
    sampled: FieldBatch
    release_grid: dict[str, int]


# This is the sole owner of the 17-field/22-component export contract.
COMPONENT_EXPORTS: Final = (
    ComponentExport("gas_density", "value", "m3c1_rhog", "kg/m^3", "gas_density_kg_per_m3"),
    ComponentExport(
        "gas_dynamic_viscosity",
        "value",
        "m3c1_mug",
        "Pa*s",
        "gas_dynamic_viscosity_Pa_s",
    ),
    ComponentExport("gas_temperature", "value", "m3c1_Tg", "K", "gas_temperature_K"),
    ComponentExport("gas_mean_free_path", "value", "m3c1_lambdag", "m", "gas_mean_free_path_m"),
    ComponentExport(
        "electron_number_density", "value", "m3c1_ne", "1/m^3", "electron_density_per_m3"
    ),
    ComponentExport(
        "positive_ion_number_density",
        "value",
        "m3c1_ni",
        "1/m^3",
        "positive_ion_density_per_m3",
    ),
    ComponentExport(
        "electron_thermal_voltage", "value", "m3c1_Te", "V", "electron_thermal_voltage_V"
    ),
    ComponentExport(
        "positive_ion_thermal_voltage",
        "value",
        "m3c1_TiV",
        "V",
        "positive_ion_thermal_voltage_V",
    ),
    ComponentExport(
        "effective_positive_ion_mass",
        "value",
        "m3c1_mi",
        "kg",
        "effective_positive_ion_mass_kg",
    ),
    ComponentExport("screening_length", "value", "m3c1_lambdaD", "m", "screening_length_m"),
    ComponentExport(
        "ion_neutral_mean_free_path",
        "value",
        "m3c1_lambdaIn",
        "m",
        "ion_neutral_mean_free_path_m",
    ),
    ComponentExport(
        "azimuthal_gas_vorticity",
        "value",
        "m3c1_omegaPhi",
        "1/s",
        "azimuthal_gas_vorticity_per_s",
    ),
    ComponentExport("gas_velocity", "r", "m3c1_ugr", "m/s", "gas_velocity_r_m_per_s"),
    ComponentExport("gas_velocity", "z", "m3c1_ugz", "m/s", "gas_velocity_z_m_per_s"),
    ComponentExport("electric_field", "r", "m3c1_Er", "V/m", "electric_field_r_V_per_m"),
    ComponentExport("electric_field", "z", "m3c1_Ez", "V/m", "electric_field_z_V_per_m"),
    ComponentExport(
        "positive_ion_velocity",
        "r",
        "m3c1_uir",
        "m/s",
        "positive_ion_velocity_r_m_per_s",
    ),
    ComponentExport(
        "positive_ion_velocity",
        "z",
        "m3c1_uiz",
        "m/s",
        "positive_ion_velocity_z_m_per_s",
    ),
    ComponentExport(
        "gradient_mean_e_squared",
        "r",
        "m3c1_gradE2r",
        "V^2/m^3",
        "gradient_mean_e_squared_r_V2_per_m3",
    ),
    ComponentExport(
        "gradient_mean_e_squared",
        "z",
        "m3c1_gradE2z",
        "V^2/m^3",
        "gradient_mean_e_squared_z_V2_per_m3",
    ),
    ComponentExport(
        "gas_translational_heat_flux",
        "r",
        "m3c1_qr",
        "W/m^2",
        "gas_translational_heat_flux_r_W_per_m2",
    ),
    ComponentExport(
        "gas_translational_heat_flux",
        "z",
        "m3c1_qz",
        "W/m^2",
        "gas_translational_heat_flux_z_W_per_m2",
    ),
)

_RELEASE_FUNCTIONS: Final = (
    ("m3c1_vr0", "m/s", "velocity_r_m_per_s", 0),
    ("m3c1_vz0", "m/s", "velocity_z_m_per_s", 1),
    ("m3c1_Z0", "1", "charge_number", None),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_sectionwise_p1(
    path: Path,
    nodes: np.ndarray,
    connectivity: np.ndarray,
    function: str,
    values: np.ndarray,
) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("%Coordinates\n")
        np.savetxt(stream, nodes, fmt="%.17g", delimiter=" ")
        stream.write("%Elements\n")
        np.savetxt(stream, connectivity + 1, fmt="%d", delimiter=" ")
        stream.write(f"%Data ({function})\n")
        np.savetxt(stream, values, fmt="%.17g")


def _field_contract() -> dict[str, tuple[tuple[str, ...], str]]:
    components: dict[str, list[str]] = defaultdict(list)
    units: dict[str, str] = {}
    for export in COMPONENT_EXPORTS:
        components[export.field].append(export.component)
        previous = units.setdefault(export.field, export.unit)
        if previous != export.unit:
            raise RuntimeError(f"inconsistent unit mapping for {export.field}")
    return {name: (tuple(items), units[name]) for name, items in components.items()}


def _validate_fields(data: DataBundle, layout: P1TriLayout) -> dict[str, FieldData]:
    fields = {field.name: field for field in data.fields}
    contract = _field_contract()
    if set(fields) != set(contract):
        missing = sorted(set(contract) - set(fields))
        extra = sorted(set(fields) - set(contract))
        raise ValueError(f"candidate field set mismatch; missing={missing}, extra={extra}")
    for name, (components, unit) in contract.items():
        field = fields[name]
        expected_basis = "scalar" if components == ("value",) else "axisymmetric_rz"
        if (
            field.layout != layout.name
            or field.association != "node"
            or field.components != components
            or field.stored_basis != expected_basis
            or field.unit != unit
            or field.values.shape != (layout.nodes_m.shape[0], len(components))
            or not bool(np.isfinite(field.values).all())
        ):
            raise ValueError(f"candidate field metadata or values do not match: {name}")
    return fields


def _release_grid(source: RealizedTableSource) -> tuple[np.ndarray, dict[str, int]]:
    positions = np.asarray(source.position_m, dtype=np.float64)
    if positions.shape != (EXPECTED_RELEASE_COUNT, 2):
        raise ValueError(f"expected {EXPECTED_RELEASE_COUNT} release positions")
    unique_r = np.unique(positions[:, 0])
    unique_z = np.unique(positions[:, 1])
    if (
        np.unique(positions, axis=0).shape[0] != EXPECTED_RELEASE_COUNT
        or unique_r.size * unique_z.size != EXPECTED_RELEASE_COUNT
    ):
        raise ValueError("release positions must form one unique rectangular RZ grid")
    expected = np.asarray([(r, z) for r in unique_r for z in unique_z], dtype=np.float64)
    order = np.lexsort((positions[:, 1], positions[:, 0]))
    if not np.array_equal(positions[order], expected):
        raise ValueError("release positions do not cover the rectangular RZ grid exactly")
    return order, {"r_count": int(unique_r.size), "z_count": int(unique_z.size)}


def _format_row(values: tuple[object, ...]) -> tuple[object, ...]:
    return tuple(format(value, ".17g") if isinstance(value, float) else value for value in values)


def _validate_candidate_source(data: DataBundle) -> tuple[RealizedTableSource, dict[str, int]]:
    """Return the one ordered internal release grid required by this V&V fixture."""

    if len(data.sources) != 1:
        raise ValueError("candidate must contain one realized table source")
    source = data.sources[0]
    if not isinstance(source, RealizedTableSource):
        raise ValueError("candidate release probes must be an internal table")
    if source.particle_id.size != EXPECTED_RELEASE_COUNT:
        raise ValueError(f"expected {EXPECTED_RELEASE_COUNT} release probes")
    expected_ids = np.arange(1, EXPECTED_RELEASE_COUNT + 1, dtype=np.int64)
    if not np.array_equal(source.particle_id, expected_ids):
        raise ValueError("release particle IDs must be the ordered contiguous range 1..287")
    if not bool((source.release_time_s == 0.0).all()):
        raise ValueError("the M3-C1 common-field diagnostic requires release at t=0")
    release_values = np.column_stack(
        (source.position_m, source.velocity_m_s, source.charge_number)
    ).astype(np.float64, copy=False)
    if not bool(np.isfinite(release_values).all()):
        raise ValueError("release position, velocity, and charge must be finite")
    release_order, release_grid = _release_grid(source)
    if not np.array_equal(release_order, np.arange(EXPECTED_RELEASE_COUNT)):
        raise ValueError("release source must be ordered lexicographically by r then z")
    return source, release_grid


def _load_candidate(candidate_h5: Path) -> _CommonInput:
    data, info = read_with_info(candidate_h5)
    if data.coordinate_system != "axisymmetric_rz" or len(data.layouts) != 1:
        raise ValueError("candidate must have one axisymmetric RZ layout")
    layout = data.layouts[0]
    if not isinstance(layout, P1TriLayout) or layout.connectivity.shape[1] != 3:
        raise ValueError("candidate must use one triangular P1 layout")
    if not bool((layout.cell_support == 1).all()):
        raise ValueError("common-field export requires every P1 cell to be supported")
    fields = _validate_fields(data, layout)
    source, release_grid = _validate_candidate_source(data)

    requirements = {
        name: RequiredFieldMetadata(
            unit=unit,
            components=components,
            stored_basis=("scalar" if components == ("value",) else "axisymmetric_rz"),
            positive=False,
        )
        for name, (components, unit) in _field_contract().items()
    }
    sampled = prepare_required_fields(data, requirements).sample(source.position_m)
    if not bool(sampled.support_inside.all()):
        raise ValueError("a release probe lies outside canonical P1 support")
    return _CommonInput(info, layout, fields, source, sampled, release_grid)


def _write_field_tables(
    output: Path,
    common: _CommonInput,
) -> tuple[list[Path], list[dict[str, object]]]:
    nodes = np.asarray(common.layout.nodes_m, dtype=np.float64)
    connectivity = np.asarray(common.layout.connectivity, dtype=np.int64)
    artifacts: list[Path] = []
    records: list[dict[str, object]] = []
    for export in COMPONENT_EXPORTS:
        field = common.fields[export.field]
        component_index = field.components.index(export.component)
        path = output / f"{export.function}_sectionwise.txt"
        _write_sectionwise_p1(
            path,
            nodes,
            connectivity,
            export.function,
            np.asarray(field.values[:, component_index], dtype=np.float64),
        )
        artifacts.append(path)
        records.append(
            {
                "field": export.field,
                "component": export.component,
                "function": export.function,
                "unit": export.unit,
                "probe_column": export.probe_column,
                "file": path.name,
            }
        )
    return artifacts, records


def _write_release_tables(output: Path, source: RealizedTableSource) -> list[Path]:
    artifacts: list[Path] = []
    for function, _unit, _column, velocity_component in _RELEASE_FUNCTIONS:
        values = (
            source.charge_number
            if velocity_component is None
            else source.velocity_m_s[:, velocity_component]
        )
        path = output / f"{function}.txt"
        np.savetxt(
            path,
            np.column_stack((source.position_m, values)),
            fmt="%.17g",
            delimiter=" ",
        )
        artifacts.append(path)
    return artifacts


def _write_probe_table(output: Path, common: _CommonInput) -> Path:
    source = common.source
    path = output / "common_p1_release_probes.csv"
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "particle_id",
                "r_m",
                "z_m",
                "velocity_r_m_per_s",
                "velocity_z_m_per_s",
                "charge_number",
                *(export.probe_column for export in COMPONENT_EXPORTS),
            )
        )
        for row, particle_id in enumerate(source.particle_id):
            primitives = tuple(
                float(
                    common.sampled.values[export.field][
                        row, common.fields[export.field].components.index(export.component)
                    ]
                )
                for export in COMPONENT_EXPORTS
            )
            writer.writerow(
                _format_row(
                    (
                        int(particle_id),
                        float(source.position_m[row, 0]),
                        float(source.position_m[row, 1]),
                        float(source.velocity_m_s[row, 0]),
                        float(source.velocity_m_s[row, 1]),
                        float(source.charge_number[row]),
                        *primitives,
                    )
                )
            )
    return path


def _field_records() -> list[dict[str, object]]:
    return [
        {
            "field": name,
            "components": list(components),
            "unit": unit,
            "association": "node",
            "stored_basis": "scalar" if components == ("value",) else "axisymmetric_rz",
        }
        for name, (components, unit) in _field_contract().items()
    ]


def _receipt(
    candidate_h5: Path,
    common: _CommonInput,
    component_records: list[dict[str, object]],
    probe_path: Path,
    artifact_paths: list[Path],
) -> dict[str, object]:
    layout = common.layout
    source = common.source
    artifacts = {
        path.name: {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
        for path in sorted(artifact_paths)
    }
    fields = _field_records()
    return {
        "schema_version": 1,
        "tool_revision": TOOL_REVISION,
        "classification": "external_vv_full_physics_exact_p1_common_field_input",
        "candidate": {
            "path": str(candidate_h5.resolve()),
            "file_sha256": _sha256(candidate_h5),
            "content_hash": common.info.content_hash,
        },
        "layout": {
            "name": layout.name,
            "type": type(layout).__name__,
            "node_count": int(layout.nodes_m.shape[0]),
            "cell_count": int(layout.connectivity.shape[0]),
            "connectivity_width": 3,
            "connectivity_indexing_in_candidate": "zero_based",
            "connectivity_indexing_in_sectionwise_files": "one_based",
            "interpolation_contract": "first_order_triangular_sectionwise",
        },
        "fields": fields,
        "field_count": len(fields),
        "components": component_records,
        "component_count": len(component_records),
        "release": {
            "source_name": source.name,
            "probe_count": int(source.particle_id.size),
            "grid": common.release_grid,
            "ordering": "particle_id_1_to_287_and_lexicographic_r_then_z",
            "initial_state_columns": [
                "particle_id",
                "r_m",
                "z_m",
                "velocity_r_m_per_s",
                "velocity_z_m_per_s",
                "charge_number",
            ],
            "functions": [
                {
                    "function": function,
                    "unit": unit,
                    "source_column": column,
                    "file": f"{function}.txt",
                    "representation": "rectangular_grid_point_table_r_z_value",
                }
                for function, unit, column, _index in _RELEASE_FUNCTIONS
            ],
            "probe_file": probe_path.name,
            "probe_field_sampling": "canonical_candidate_P1_sampler_at_realized_release_positions",
        },
        "artifacts": artifacts,
    }


def prepare(candidate_h5: Path, output: Path) -> None:
    """Create a no-clobber full-field COMSOL input package and receipt."""

    common = _load_candidate(candidate_h5)

    output.mkdir(parents=True, exist_ok=False)
    artifact_paths, component_records = _write_field_tables(output, common)
    artifact_paths.append(write_boundary_meaning(candidate_h5, output))
    artifact_paths.extend(_write_release_tables(output, common.source))
    probe_path = _write_probe_table(output, common)
    artifact_paths.append(probe_path)
    receipt = _receipt(candidate_h5, common, component_records, probe_path, artifact_paths)
    (output / "common_p1_table_receipt.json").write_text(
        json.dumps(receipt, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_h5", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.candidate_h5, args.output)


if __name__ == "__main__":
    main()
