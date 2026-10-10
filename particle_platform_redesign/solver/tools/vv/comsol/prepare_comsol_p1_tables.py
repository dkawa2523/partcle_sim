"""Materialize canonical P1 fields and deterministic probe points for COMSOL.

This utility is external V&V only.  It does not change the candidate input or
production solver.  The generated table lets an isolated COMSOL model-copy
replace its native finite-element field evaluation with the exact node values
used by the canonical P1 candidate.  A companion probe table records the
candidate P1 interpolant at the same release points before any trajectory run.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from chamber_particles.case_format import DataBundle, P1TriLayout, RealizedTableSource, read
from chamber_particles.fields import (
    PreparedFieldSet,
    RequiredFieldMetadata,
    prepare_required_fields,
)

FIELD_NAMES = (
    "gas_velocity",
    "gas_temperature",
    "gas_density",
    "gas_dynamic_viscosity",
    "electric_field",
)


def _prepared_fields(data: DataBundle) -> PreparedFieldSet:
    fields = {field.name: field for field in data.fields}
    requirements = {
        name: RequiredFieldMetadata(
            unit=fields[name].unit,
            components=fields[name].components,
            stored_basis=fields[name].stored_basis,
            positive=False,
        )
        for name in FIELD_NAMES
    }
    return prepare_required_fields(data, requirements)


def write_sectionwise_p1(
    path: Path,
    nodes: np.ndarray,
    connectivity: np.ndarray,
    function_name: str,
    values: np.ndarray,
) -> None:
    """Write one first-order 2-D function with explicit triangle ownership."""

    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("%Coordinates\n")
        np.savetxt(stream, nodes, fmt="%.17g", delimiter=" ")
        stream.write("%Elements\n")
        np.savetxt(stream, connectivity + 1, fmt="%d", delimiter=" ")
        stream.write(f"%Data ({function_name})\n")
        np.savetxt(stream, values, fmt="%.17g")


def prepare(candidate_h5: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    data = read(candidate_h5)
    if len(data.layouts) != 1 or data.coordinate_system != "axisymmetric_rz":
        raise ValueError("matched candidate must have one axisymmetric RZ layout")
    layout = data.layouts[0]
    if not isinstance(layout, P1TriLayout) or layout.connectivity.shape[1] != 3:
        raise ValueError("matched candidate must use a triangular P1 layout")
    connectivity = layout.connectivity
    fields = {field.name: field for field in data.fields}
    nodes = np.asarray(layout.nodes_m, dtype=np.float64)

    table_path = output / "canonical_p1_fields.csv"
    with table_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "r_m",
                "z_m",
                "gas_velocity_r_m_per_s",
                "gas_velocity_z_m_per_s",
                "gas_temperature_K",
                "gas_density_kg_per_m3",
                "gas_dynamic_viscosity_Pa_s",
                "electric_field_r_V_per_m",
                "electric_field_z_V_per_m",
            )
        )
        values = (
            fields["gas_velocity"].values,
            fields["gas_temperature"].values,
            fields["gas_density"].values,
            fields["gas_dynamic_viscosity"].values,
            fields["electric_field"].values,
        )
        for row in range(nodes.shape[0]):
            writer.writerow(
                (
                    nodes[row, 0],
                    nodes[row, 1],
                    values[0][row, 0],
                    values[0][row, 1],
                    values[1][row, 0],
                    values[2][row, 0],
                    values[3][row, 0],
                    values[4][row, 0],
                    values[4][row, 1],
                )
            )

    function_columns = {
        "m3v_ugr": values[0][:, 0],
        "m3v_ugz": values[0][:, 1],
        "m3v_Tg": values[1][:, 0],
        "m3v_rhog": values[2][:, 0],
        "m3v_mug": values[3][:, 0],
        "m3v_Er": values[4][:, 0],
        "m3v_Ez": values[4][:, 1],
    }
    for name, column in function_columns.items():
        np.savetxt(
            output / f"{name}.txt",
            np.column_stack((nodes, column)),
            fmt="%.17g",
            delimiter=" ",
        )
        write_sectionwise_p1(
            output / f"{name}_sectionwise.txt",
            nodes,
            connectivity,
            name,
            column,
        )

    if len(data.sources) != 1:
        raise ValueError("matched candidate must contain one realized table source")
    source = data.sources[0]
    if not isinstance(source, RealizedTableSource):
        raise ValueError("matched candidate release probes must be an internal table")
    if source.particle_id.size != 287:
        raise ValueError(f"expected 287 release probes, got {source.particle_id.size}")
    positions = np.asarray(source.position_m, dtype=np.float64)
    batch = _prepared_fields(data).sample(positions)
    if not bool(batch.support_inside.all()):
        raise ValueError("a release probe lies outside canonical P1 support")

    probe_path = output / "canonical_p1_release_probes.csv"
    with probe_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "probe_id",
                "r_m",
                "z_m",
                "gas_velocity_r_m_per_s",
                "gas_velocity_z_m_per_s",
                "gas_temperature_K",
                "gas_density_kg_per_m3",
                "gas_dynamic_viscosity_Pa_s",
                "electric_field_r_V_per_m",
                "electric_field_z_V_per_m",
            )
        )
        for row, particle_id in enumerate(source.particle_id):
            writer.writerow(
                (
                    f"p{int(particle_id)}_t0",
                    positions[row, 0],
                    positions[row, 1],
                    batch.values["gas_velocity"][row, 0],
                    batch.values["gas_velocity"][row, 1],
                    batch.values["gas_temperature"][row, 0],
                    batch.values["gas_density"][row, 0],
                    batch.values["gas_dynamic_viscosity"][row, 0],
                    batch.values["electric_field"][row, 0],
                    batch.values["electric_field"][row, 1],
                )
            )

    (output / "table_receipt.json").write_text(
        json.dumps(
            {
                "classification": "external_vv_canonical_p1_table",
                "candidate_input": str(candidate_h5.resolve()),
                "layout": type(layout).__name__,
                "node_count": int(nodes.shape[0]),
                "cell_count": int(connectivity.shape[0]),
                "release_probe_count": int(source.particle_id.size),
                "connectivity_indexing_in_file": "one_based",
                "interpolation_contract": "first_order_triangular_sectionwise",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
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
