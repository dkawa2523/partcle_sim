"""Compare canonical reduced fields with one COMSOL node export.

This is descriptive external V&V.  It never changes the producer or trajectory
solver and does not promote the reference export to a physics definition.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
from pathlib import Path
from typing import Final

import numpy as np

from chamber_particles.case_format import DataBundle, FieldData, P1TriLayout, read_with_info

COMPARISON_REVISION: Final = "reduced_field_comparison_v1"
_FINITE_NUMBER_TOKEN: Final = re.compile(r"[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?")
_REFERENCE_COLUMNS: Final = {
    "electric_potential": ("SASS_potential_V",),
    "electric_field": ("electric_field_r_V_per_m", "electric_field_z_V_per_m"),
    "electron_number_density": ("electron_density_per_m3",),
    "positive_ion_number_density": ("total_positive_ion_density_per_m3",),
    "space_charge_density": ("space_charge_density_C_per_m3",),
    "positive_ion_speed": ("ion_speed_m_per_s",),
    "positive_ion_velocity": ("ion_velocity_r_m_per_s", "ion_velocity_z_m_per_s"),
}
_EXPECTED_FIELD_METADATA: Final = {
    "electric_potential": ("V", ("value",), "scalar"),
    "electric_field": ("V/m", ("r", "z"), "axisymmetric_rz"),
    "electron_number_density": ("1/m^3", ("value",), "scalar"),
    "positive_ion_number_density": ("1/m^3", ("value",), "scalar"),
    "space_charge_density": ("C/m^3", ("value",), "scalar"),
    "positive_ion_speed": ("m/s", ("value",), "scalar"),
    "positive_ion_velocity": ("m/s", ("r", "z"), "axisymmetric_rz"),
}
_EXPECTED_REFERENCE_UNITS: Final = {
    "r_m": "m",
    "z_m": "m",
    "SASS_potential_V": "V",
    "electric_field_r_V_per_m": "V/m",
    "electric_field_z_V_per_m": "V/m",
    "electron_density_per_m3": "1/m^3",
    "total_positive_ion_density_per_m3": "1/m^3",
    "space_charge_density_C_per_m3": "C/m^3",
    "ion_speed_m_per_s": "m/s",
    "ion_velocity_r_m_per_s": "m/s",
    "ion_velocity_z_m_per_s": "m/s",
}


def compare_reduced_fields(
    generated_case_path: str | Path,
    reference_values_path: str | Path,
    reference_columns_path: str | Path,
    output_path: str | Path,
    *,
    coordinate_tolerance_m: float,
) -> dict[str, object]:
    """Write one no-clobber field-comparison report and return its content."""

    if not math.isfinite(coordinate_tolerance_m) or coordinate_tolerance_m <= 0.0:
        raise ValueError("coordinate_tolerance_m must be finite and positive")
    generated_path = Path(generated_case_path).expanduser().resolve()
    values_path = Path(reference_values_path).expanduser().resolve()
    columns_path = Path(reference_columns_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    values_content = values_path.read_bytes()
    columns_content = columns_path.read_bytes()
    data, case_info = read_with_info(generated_path)
    layout_name = _require_supported_geometry(data)
    reference, maximum_match_distance = _reference_at_nodes(
        data,
        values_content,
        columns_content,
        coordinate_tolerance_m,
    )
    weights = _axisymmetric_lumped_weights(data)
    fields = {field.name: field for field in data.fields}
    required_fields = {
        name: _required_field(fields, name, layout_name) for name in _REFERENCE_COLUMNS
    }
    metrics = {
        name: _field_metric(required_fields[name], reference[name], weights)
        for name in _REFERENCE_COLUMNS
    }
    report: dict[str, object] = {
        "status": "complete",
        "classification": "descriptive_external_vv_not_solver_acceptance",
        "comparison_revision": COMPARISON_REVISION,
        "generated_case_path": str(generated_path),
        "generated_case_content_hash": case_info.content_hash,
        "reference_values_sha256": _content_hash(values_content),
        "reference_columns_sha256": _content_hash(columns_content),
        "node_count": int(data.geometry.nodes_m.shape[0]),
        "coordinate_tolerance_m": coordinate_tolerance_m,
        "maximum_coordinate_match_distance_m": maximum_match_distance,
        "weighted_field_metrics": metrics,
        "boundary_potential_trace": _boundary_trace(data, required_fields, reference),
        "generated_charge_balance": _charge_balance(data.provenance_json),
        "coverage": {
            "same_exported_nodes": "TESTED",
            "independent_mesh_convergence": "NOT_TESTED_SINGLE_REFERENCE_MESH",
            "trajectory_parity": "NOT_APPLICABLE_TO_FIELD_COMPARISON",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", errors="strict") as stream:
        stream.write(json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n")
    return report


def _require_supported_geometry(data: DataBundle) -> str:
    geometry = data.geometry
    if data.coordinate_system != "axisymmetric_rz":
        raise ValueError("comparison requires coordinate_system=axisymmetric_rz")
    if geometry.tri3 is None or geometry.quad4 is not None:
        raise ValueError("comparison requires triangle-only canonical geometry")
    matching = [
        layout
        for layout in data.layouts
        if isinstance(layout, P1TriLayout)
        and np.array_equal(layout.nodes_m, geometry.nodes_m)
        and np.array_equal(layout.connectivity, geometry.tri3)
        and bool((layout.cell_support == 1).all())
    ]
    if len(matching) != 1:
        raise ValueError("comparison requires one fully supported geometry-identical P1 layout")
    return matching[0].name


def _reference_at_nodes(
    data: DataBundle,
    values_content: bytes,
    columns_content: bytes,
    tolerance_m: float,
) -> tuple[dict[str, np.ndarray], float]:
    names = _column_names(columns_content)
    index = {name: position for position, name in enumerate(names)}
    required = {"r_m", "z_m", *(name for names_ in _REFERENCE_COLUMNS.values() for name in names_)}
    missing = required.difference(index)
    if missing:
        raise ValueError(f"reference column dictionary is missing columns: {sorted(missing)}")
    try:
        values_text = values_content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("reference values are not valid UTF-8") from error
    values = _numeric_csv_matrix(
        values_text,
        expected_columns=len(names),
        label="reference values",
    )
    value_ids = [index[name] for columns_ in _REFERENCE_COLUMNS.values() for name in columns_]
    finite = np.isfinite(values[:, value_ids]).all(axis=1)
    coordinates = values[finite][:, [index["r_m"], index["z_m"]]]
    selected = values[finite]
    if coordinates.shape[0] != data.geometry.nodes_m.shape[0]:
        raise ValueError("finite reference field rows do not match canonical node count")
    match, maximum = _match_coordinates(data.geometry.nodes_m, coordinates, tolerance_m)
    result = {
        field_name: np.ascontiguousarray(
            selected[match][:, [index[name] for name in column_names]], dtype=np.float64
        )
        for field_name, column_names in _REFERENCE_COLUMNS.items()
    }
    return result, maximum


def _numeric_csv_matrix(text: str, *, expected_columns: int, label: str) -> np.ndarray:
    rows: list[list[float]] = []
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    try:
        numbered_rows = [(reader.line_num, row) for row in reader]
    except csv.Error as error:
        raise ValueError(f"{label}: malformed CSV near line {reader.line_num}") from error
    for line_number, row in numbered_rows:
        if not row or all(not cell.strip() for cell in row):
            continue
        if row[0].lstrip().startswith("%"):
            continue
        if len(row) != expected_columns:
            raise ValueError(
                f"{label}: row {line_number} has {len(row)} columns; expected {expected_columns}"
            )
        parsed: list[float] = []
        for column, token in enumerate(row, start=1):
            value = token.strip()
            if not value:
                raise ValueError(f"{label}: row {line_number} column {column} is empty")
            if value.casefold() in {"nan", "+nan", "-nan"}:
                parsed.append(math.nan)
                continue
            if _FINITE_NUMBER_TOKEN.fullmatch(value) is None:
                raise ValueError(f"{label}: row {line_number} column {column} is not numeric")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"{label}: row {line_number} column {column} is not finite")
            parsed.append(number)
        rows.append(parsed)
    if not rows:
        raise ValueError(f"{label}: numeric table has no data rows")
    return np.ascontiguousarray(np.asarray(rows, dtype=np.float64))


def _column_names(content: bytes) -> list[str]:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("reference column dictionary is not valid UTF-8") from error
    rows = _column_dictionary_rows(text)
    names = [row["column"] for row in rows]
    if not names or any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("reference column names must be nonempty and unique")
    _validate_reference_units(rows)
    return names


def _column_dictionary_rows(text: str) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
    expected = ("column", "COMSOL_expression", "unit")
    if reader.fieldnames is None or tuple(reader.fieldnames) != expected:
        raise ValueError("reference column dictionary has unexpected columns")
    try:
        rows = list(reader)
    except csv.Error as error:
        raise ValueError(
            f"reference column dictionary is malformed near line {reader.line_num}"
        ) from error
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("reference column dictionary contains a malformed row")
    return rows


def _validate_reference_units(rows: list[dict[str, str]]) -> None:
    units = {row["column"]: row["unit"] for row in rows}
    wrong_units = {
        name: (units.get(name), unit)
        for name, unit in _EXPECTED_REFERENCE_UNITS.items()
        if units.get(name) != unit
    }
    if wrong_units:
        raise ValueError(
            f"reference column units do not match the comparison contract: {wrong_units}"
        )


def _match_coordinates(
    canonical: np.ndarray, reference: np.ndarray, tolerance_m: float
) -> tuple[np.ndarray, float]:
    if not bool(np.isfinite(reference).all()) or np.unique(reference, axis=0).shape[0] != len(
        reference
    ):
        raise ValueError("reference coordinates must be finite and unique")
    match = np.empty(canonical.shape[0], dtype=np.int64)
    distance2 = np.empty(canonical.shape[0], dtype=np.float64)
    tolerance2 = tolerance_m * tolerance_m
    for begin in range(0, canonical.shape[0], 256):
        end = min(begin + 256, canonical.shape[0])
        delta = canonical[begin:end, None, :] - reference[None, :, :]
        squared = np.sum(delta * delta, axis=2)
        if bool((np.count_nonzero(squared <= tolerance2, axis=1) != 1).any()):
            raise ValueError(
                "each canonical node must have exactly one reference coordinate within tolerance"
            )
        nearest = np.argmin(squared, axis=1)
        match[begin:end] = nearest
        distance2[begin:end] = squared[np.arange(end - begin), nearest]
    maximum = float(np.sqrt(np.max(distance2, initial=0.0)))
    if maximum > tolerance_m or np.unique(match).size != match.size:
        raise ValueError("reference coordinates do not form a unique match within tolerance")
    return match, maximum


def _axisymmetric_lumped_weights(data: DataBundle) -> np.ndarray:
    triangles = data.geometry.tri3
    if triangles is None:
        raise ValueError("triangle geometry is required")
    local = data.geometry.nodes_m[triangles]
    edge1 = local[:, 1] - local[:, 0]
    edge2 = local[:, 2] - local[:, 0]
    area = 0.5 * (edge1[:, 0] * edge2[:, 1] - edge1[:, 1] * edge2[:, 0])
    if bool((~np.isfinite(area) | (area <= 0.0)).any()):
        raise ValueError("canonical geometry contains an invalid triangle")
    radius_sum = np.sum(local[:, :, 0], axis=1)
    local_weight = 2.0 * math.pi * area[:, None] * (radius_sum[:, None] + local[:, :, 0]) / 12.0
    weights = np.bincount(
        triangles.reshape(-1),
        weights=local_weight.reshape(-1),
        minlength=data.geometry.nodes_m.shape[0],
    )
    if not bool(np.isfinite(weights).all()) or bool((weights <= 0.0).any()):
        raise ValueError("axisymmetric nodal weights must be finite and positive")
    return weights


def _required_field(fields: dict[str, FieldData], name: str, layout_name: str) -> FieldData:
    try:
        field = fields[name]
    except KeyError as error:
        raise ValueError(f"generated case is missing field {name!r}") from error
    if field.association != "node":
        raise ValueError(f"generated field {name!r} must be node-associated")
    expected_unit, expected_components, expected_basis = _EXPECTED_FIELD_METADATA[name]
    if (
        field.layout != layout_name
        or field.unit != expected_unit
        or field.components != expected_components
        or field.stored_basis != expected_basis
    ):
        raise ValueError(
            f"generated field {name!r} metadata does not match the comparison contract"
        )
    return field


def _field_metric(
    generated: FieldData, reference: np.ndarray, weights: np.ndarray
) -> dict[str, object]:
    if generated.values.shape != reference.shape:
        raise ValueError(f"field shape differs for {generated.name!r}")
    difference = generated.values - reference
    difference_norm = _weighted_norm(difference, weights)
    reference_norm = _weighted_norm(reference, weights)
    generated_norm = _weighted_norm(generated.values, weights)
    scale = max(reference_norm, np.finfo(np.float64).tiny)
    return {
        "unit": generated.unit,
        "components": list(generated.components),
        "absolute_linf": float(np.max(np.abs(difference))),
        "difference_weighted_l2": difference_norm,
        "reference_weighted_l2": reference_norm,
        "generated_weighted_l2": generated_norm,
        "relative_weighted_l2": difference_norm / scale,
    }


def _weighted_norm(values: np.ndarray, weights: np.ndarray) -> float:
    squared = np.sum(values * values, axis=1)
    return math.sqrt(float(np.dot(weights, squared)))


def _boundary_trace(
    data: DataBundle,
    fields: dict[str, FieldData],
    reference: dict[str, np.ndarray],
) -> dict[str, object]:
    potential = fields["electric_potential"].values[:, 0]
    expected = reference["electric_potential"][:, 0]
    result: dict[str, object] = {}
    node_groups: dict[int, set[int]] = {}
    for facet, line in enumerate(data.geometry.boundary.line2):
        group = int(data.geometry.boundary.group_id[facet])
        for node in line:
            node_groups.setdefault(int(node), set()).add(group)
    for group_id, name in enumerate(data.geometry.group_names):
        nodes = np.asarray(
            sorted(node for node, groups in node_groups.items() if group_id in groups),
            dtype=np.int64,
        )
        difference = potential[nodes] - expected[nodes]
        exclusive = np.asarray(
            sorted(node for node, groups in node_groups.items() if groups == {group_id}),
            dtype=np.int64,
        )
        exclusive_difference = potential[exclusive] - expected[exclusive]
        result[name] = {
            "node_count": int(nodes.size),
            "absolute_linf_V": float(np.max(np.abs(difference))),
            "rms_V": math.sqrt(float(np.mean(difference * difference))),
            "exclusive_node_count": int(exclusive.size),
            "exclusive_absolute_linf_V": (
                float(np.max(np.abs(exclusive_difference))) if exclusive.size else None
            ),
            "exclusive_rms_V": (
                math.sqrt(float(np.mean(exclusive_difference * exclusive_difference)))
                if exclusive.size
                else None
            ),
        }
    result["shared_group_corner_node_count"] = sum(
        1 for groups in node_groups.values() if len(groups) > 1
    )
    return result


def _charge_balance(provenance_json: str) -> dict[str, object]:
    provenance = json.loads(provenance_json)
    metadata = provenance.get("producer_metadata")
    if not isinstance(metadata, dict):
        return {"status": "unavailable"}
    convergence = metadata.get("convergence")
    if not isinstance(convergence, dict):
        return {"status": "unavailable"}
    return {
        "status": "reported_by_generated_builder",
        "total_space_charge_C": convergence.get("total_space_charge_C"),
        "boundary_reaction_charge_C": convergence.get("boundary_reaction_charge_C"),
        "charge_balance_error_C": convergence.get("charge_balance_error_C"),
    }


def _content_hash(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("generated_case", type=Path)
    parser.add_argument("reference_values", type=Path)
    parser.add_argument("reference_columns", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--coordinate-tolerance-m", type=float, required=True)
    arguments = parser.parse_args()
    report = compare_reduced_fields(
        arguments.generated_case,
        arguments.reference_values,
        arguments.reference_columns,
        arguments.output,
        coordinate_tolerance_m=arguments.coordinate_tolerance_m,
    )
    print(json.dumps(report, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
