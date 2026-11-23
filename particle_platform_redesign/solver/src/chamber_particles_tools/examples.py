"""Small analytic examples shipped as code, editable settings and realized CSVs."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

import numpy as np
import yaml

from chamber_particles.case import SimulationCase
from chamber_particles.case_format import (
    BoundaryData,
    DataBundle,
    FieldData,
    GeometryData,
    P1TriLayout,
    RealizedSource,
    RealizedSurfaceSource,
    RealizedTableSource,
    read,
    write,
)

from .inputs import create_case

EXAMPLE_REVISION = "analytic_editable_examples_v3"


@dataclass(frozen=True, slots=True)
class _Example:
    rz: bool = False
    electric: Literal["none", "constant", "linear_time", "reduced"] = "none"
    brownian: bool = False


_EXAMPLES = {
    "xy-ballistic": _Example(),
    "rz-charged": _Example(rz=True, electric="constant"),
    "xy-time-field": _Example(electric="linear_time"),
    "brownian-wall": _Example(brownian=True),
    "reduced-electrostatic": _Example(rz=True, electric="reduced"),
}
EXAMPLE_NAMES = tuple(_EXAMPLES)


def initialize(name: str, output_directory: str | Path) -> SimulationCase:
    """Create a checked case and its editable source CSV/recipe."""
    if name not in EXAMPLE_NAMES:
        raise ValueError(f"unknown example {name!r}; choose one of {', '.join(EXAMPLE_NAMES)}")
    definition = _EXAMPLES[name]
    bundle = _bundle(name, definition)
    thermal = bundle
    settings = _settings(name, definition)
    if definition.electric == "reduced":
        bundle = _build_electrostatic(bundle)
    supporting_files = _editable_sources(bundle, settings)
    if name == "rz-charged":
        supporting_files.update(_editable_electric_fields(bundle, supporting_files["recipe.yaml"]))
    if definition.electric == "reduced":
        with TemporaryDirectory(prefix="chamber-thermal-input-") as temporary:
            path = Path(temporary) / "thermal.h5"
            write(path, thermal)
            supporting_files["thermal.h5"] = path.read_bytes()
        supporting_files["builder.yaml"] = yaml.safe_dump(
            _electrostatic_configuration(), sort_keys=False
        ).encode("utf-8")
    return create_case(bundle, settings, output_directory, supporting_files=supporting_files)


def _geometry(rz: bool) -> GeometryData:
    low = 0.01 if rz else 0.0
    nodes = np.asarray([[low, 0], [low + 0.01, 0], [low + 0.01, 0.01], [low, 0.01]], dtype="<f8")
    boundary = BoundaryData(
        line2=np.asarray([[0, 1], [1, 2], [2, 3], [3, 0]], dtype="<i8"),
        boundary_id=np.arange(1, 5, dtype="<i4"),
        group_id=np.arange(4, dtype="<i4"),
        material_id=np.zeros(4, dtype="<i4"),
        owner_cell_type=np.ones(4, dtype="u1"),
        owner_cell_local_index=np.asarray([0, 0, 1, 1], dtype="<i8"),
        orientation=np.ones(4, dtype="i1"),
    )
    return GeometryData(
        nodes,
        boundary,
        ("bottom", "right", "top", "left"),
        tri3=np.asarray([[0, 1, 2], [0, 2, 3]], dtype="<i8"),
        tri3_domain_id=np.ones(2, dtype="<i4"),
    )


def _source(rz: bool, brownian: bool) -> RealizedTableSource:
    return RealizedTableSource(
        name="particles",
        particle_id=np.asarray([1], dtype="<i8"),
        release_time_s=np.asarray([0.0]),
        position_m=np.asarray([[0.015 if rz else 0.0025, 0.005]]),
        velocity_m_s=np.asarray([[0.1 if brownian else 0.5, 0.0]]),
        charge_number=np.asarray([3.0 if rz else 0.0]),
        mass_kg=np.asarray([1e-18 if brownian else 1e-15]),
        drag_diameter_m=np.asarray([1e-7]),
        contact_radius_m=np.asarray([0.0]),
        electrostatic_radius_m=np.asarray([5e-8]),
        displaced_volume_m3=np.asarray([5.235987755982989e-22]),
        model_weight=np.asarray([1.0]),
        material_id=np.zeros(1, dtype="<i4"),
    )


def _bundle(name: str, definition: _Example) -> DataBundle:
    rz = definition.rz
    geometry = _geometry(rz)
    assert geometry.tri3 is not None
    layout = P1TriLayout("background", geometry.nodes_m, geometry.tri3, np.ones(2, dtype="u1"))
    coordinates = "axisymmetric_rz" if rz else "cartesian_xy"
    fields: tuple[FieldData, ...] = ()
    if definition.electric == "constant":
        fields = (_field("electric_field", [0.0, 1000.0], "V/m", coordinates),)
    elif definition.electric == "linear_time":
        values = np.zeros((3, 4, 2))
        values[:, :, 0] = np.asarray([0.0, 5000.0, 10000.0])[:, None]
        fields = (
            FieldData(
                "electric_field",
                "background",
                "node",
                ("x", "y"),
                coordinates,
                values,
                "V/m",
                np.asarray([0.0, 0.01, 0.02]),
            ),
        )
    elif definition.brownian or definition.electric == "reduced":
        fields = (
            _field("gas_velocity", [0.0, 0.0], "m/s", coordinates),
            _field("gas_temperature", [300.0], "K"),
        )
        if definition.brownian:
            fields += (
                _field("gas_density", [1e-3], "kg/m^3"),
                _field("gas_mean_free_path", [1e-3], "m"),
            )
    source = _source(rz, definition.brownian)
    if definition.electric == "linear_time":
        source = replace(source, charge_number=np.asarray([3.0]))
    generator_input = {
        "example": name,
        "generator_revision": EXAMPLE_REVISION,
        "rz": definition.rz,
        "electric": definition.electric,
        "brownian": definition.brownian,
    }
    encoded = json.dumps(generator_input, sort_keys=True, separators=(",", ":")).encode()
    provenance = json.dumps(
        {
            "producer": "chamber_particles_tools.examples",
            "producer_version": EXAMPLE_REVISION,
            "source_sha256": "sha256:" + hashlib.sha256(encoded).hexdigest(),
            "field_semantics_revision": EXAMPLE_REVISION,
            "producer_metadata": {
                **generator_input,
                "purpose": "educational analytic input",
                "source_identity": "generator input and revision; not a source-code file hash",
            },
        }
    )
    return DataBundle(
        coordinates, provenance, geometry, (layout,) if fields else (), fields, (source,)
    )


def _field(name: str, values: list[float], unit: str, basis: str = "scalar") -> FieldData:
    components = (
        ("value",)
        if basis == "scalar"
        else (("r", "z") if basis == "axisymmetric_rz" else ("x", "y"))
    )
    return FieldData(
        name,
        "background",
        "node",
        components,
        basis,
        np.broadcast_to(values, (4, len(values))).copy(),
        unit,
    )


def _settings(name: str, definition: _Example) -> dict[str, Any]:
    rz = definition.rz
    brownian = definition.brownian
    physics: dict[str, Any] = {"charge": {"model": "fixed"}}
    if definition.electric != "none":
        physics["electric"] = {
            "model": "coulomb",
            "revision": "electric_coulomb_v1",
            "electric_field": "electric_field",
        }
    if brownian:
        physics.update(
            drag={
                "model": "epstein_linear",
                "revision": "epstein_linear_v1",
                "gas_velocity_field": "gas_velocity",
                "gas_density_field": "gas_density",
                "gas_temperature_field": "gas_temperature",
                "gas_mean_free_path_field": "gas_mean_free_path",
                "gas_molecular_mass_kg": 4.65e-26,
                "delta": 1.2,
                "applicability": "error",
            },
            noise={
                "model": "inertial_langevin_fdt",
                "revision": "inertial_langevin_fdt_epstein_linear_midpoint_2d_v2",
                "interval_tree_depth": 2,
                "adaptive_max_depth": 4,
            },
        )
    return {
        "format_version": 3,
        "case": {"name": name.replace("-", "_")},
        "motion": {"mode": "axisymmetric_rz_meridional" if rz else "cartesian_xy"},
        "time": {"start_s": 0.0, "end_s": 0.02, "dt_s": 0.001},
        "solver": {
            "integrator": "ou_langevin" if brownian else "rk4_fixed",
            "backend": "cpu",
            "seed": 0,
            "event": {
                "geometry_rtol": 1e-12,
                "roundoff_ulps": 64,
                "max_refinements": 48,
                "max_interactions_per_step": 8,
                "corner_policy": "priority_then_combined_normal_v1",
            },
        },
        "resources": {"memory_limit_mb": 128},
        "physics": physics,
        "sources": [{"name": "release", "type": "table", "table": "particles"}],
        "boundaries": [
            {"boundary_group": group, "priority": index, "law": "specular" if brownian else "stick"}
            for index, group in enumerate(("bottom", "right", "top", "left"), 1)
        ],
        "output": {
            "trajectories": {
                "selection": "all",
                "schedule": {"explicit_times_s": [0.0, 0.005, 0.01, 0.015, 0.02]},
            }
        },
    }


def _build_electrostatic(bundle: DataBundle) -> DataBundle:
    from .electrostatic_builder.workflow import build_from_configuration

    with TemporaryDirectory(prefix="chamber-electrostatic-example-") as temporary:
        directory = Path(temporary)
        write(directory / "thermal.h5", bundle)
        configuration = _electrostatic_configuration()
        config_path = directory / "builder.yaml"
        config_path.write_text(yaml.safe_dump(configuration), encoding="utf-8")
        build_from_configuration(config_path, directory / "electric")
        return read(directory / "electric" / "bundle.h5")


def _electrostatic_configuration() -> dict[str, Any]:
    return {
        "format_version": 2,
        "input": {
            "data_path": "thermal.h5",
            "layout": "background",
            "gas_velocity_field": "gas_velocity",
            "gas_temperature_field": "gas_temperature",
        },
        "model": {
            "revision": "boltzmann_bohm_sheath_c2_v1",
            "bulk_number_density_m3": 1e12,
            "electron_temperature_V": 3.0,
            "positive_ion_mass_kg": 6.6335209e-26,
            "bulk_potential_V": 0.0,
            "ion_mobility_m2_V_s": 1.0,
            "ion_flux_regularization_speed_m_s": 1.0,
            "sheath_smoothing_V": 0.05,
            "density_floor_m3": 1.0,
            "ion_speed_floor_m_s": 0.001,
        },
        "boundaries": [
            {
                "group": group,
                "priority": 3 if group == "top" else 1,
                "potential": {"kind": "constant", "value_V": 1.0 if group == "top" else 0.0},
            }
            for group in ("bottom", "right", "top", "left")
        ],
        "solver": {
            "continuation_ramps": [0.0, 1.0],
            "max_newton_iterations": 20,
            "relative_residual_tolerance": 1e-9,
            "absolute_residual_tolerance_C": 1e-22,
            "max_linear_iterations": 100,
            "linear_krylov_dimension": 20,
            "linear_relative_tolerance": 1e-11,
            "minimum_line_search_factor": 0.0009765625,
        },
    }


def _editable_sources(bundle: DataBundle, settings: dict[str, Any]) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    sources = []
    for index, source in enumerate(bundle.sources):
        name = "particles.csv" if index == 0 else f"particles_{index + 1}.csv"
        content, configuration = _editable_source(source, bundle.coordinate_system)
        files[name] = content
        sources.append({"csv": name, "configuration": configuration})
    recipe = {
        "format_version": 2,
        "input_bundle": "case.h5",
        "sources": sources,
        "settings": settings,
    }
    files["recipe.yaml"] = yaml.safe_dump(recipe, sort_keys=False).encode("utf-8")
    return files


def _csv_bytes(header: list[str], rows: list[list[object]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _editable_electric_fields(bundle: DataBundle, recipe_raw: bytes) -> dict[str, bytes]:
    """Expose the existing charged example's exact mesh and field as editable CSVs."""
    geometry = bundle.geometry
    assert geometry.tri3 is not None
    field = next(item for item in bundle.fields if item.name == "electric_field")
    boundary = geometry.boundary
    files = {
        "field-input/vertices.csv": _csv_bytes(
            ["vertex_id", "r_m", "z_m"],
            [[i + 1, *row] for i, row in enumerate(geometry.nodes_m.tolist())],
        ),
        "field-input/triangles.csv": _csv_bytes(
            ["triangle_id", "node1_id", "node2_id", "node3_id", "domain_id"],
            [[i + 1, *(np.asarray(row) + 1).tolist(), 1] for i, row in enumerate(geometry.tri3)],
        ),
        "field-input/quads.csv": _csv_bytes(
            ["quadrilateral_id", "node1_id", "node2_id", "node3_id", "node4_id", "domain_id"],
            [],
        ),
        "field-input/boundaries.csv": _csv_bytes(
            ["boundary_edge_id", "node1_id", "node2_id", "boundary_id"],
            [
                [i + 1, int(pair[0]) + 1, int(pair[1]) + 1, int(boundary.boundary_id[i])]
                for i, pair in enumerate(boundary.line2)
            ],
        ),
        "field-input/columns.csv": _csv_bytes(
            ["column", "expression", "unit"],
            [
                ["vertex_id", "mesh node identity", "1"],
                ["r", "r", "m"],
                ["z", "z", "m"],
                ["inside", "selected domain", "1"],
                ["Er", "example radial E", "V/m"],
                ["Ez", "example axial E", "V/m"],
            ],
        ),
    }
    values = io.StringIO(newline="")
    csv.writer(values).writerows(
        [i + 1, *point, 1, *electric]
        for i, (point, electric) in enumerate(
            zip(geometry.nodes_m.tolist(), field.values.tolist(), strict=True)
        )
    )
    files["field-input/values.csv"] = values.getvalue().encode("utf-8")
    mapping = _electric_field_mapping(geometry)
    files["field-mapping.yaml"] = yaml.safe_dump(mapping, sort_keys=False).encode("utf-8")
    recipe = yaml.safe_load(recipe_raw)
    recipe["input_bundle"] = "background/bundle.h5"
    files["field-recipe.yaml"] = yaml.safe_dump(recipe, sort_keys=False).encode("utf-8")
    files["FIELD_INPUT.md"] = (
        b"Edit field-input/ CSVs, then run from this directory:\n\n"
        b"python -m chamber_particles inputs import-csv field-mapping.yaml background\n"
        b"python -m chamber_particles create field-recipe.yaml -o ../charged-rebuilt\n\n"
        b"The CSVs derive from this example's mesh and electric field. Units, source and settings remain explicit. "
        b"Use fresh output directories; change field-recipe.yaml if the imported bundle is elsewhere.\n"
    )
    return files


def _electric_field_mapping(geometry: GeometryData) -> dict[str, Any]:
    boundary = geometry.boundary
    return {
        "format_version": 4,
        "provider": "csv",
        "source": {
            name + "_path": "field-input/" + filename
            for name, filename in (
                ("vertices", "vertices.csv"),
                ("triangles", "triangles.csv"),
                ("quadrilaterals", "quads.csv"),
                ("boundary_edges", "boundaries.csv"),
                ("field_values", "values.csv"),
                ("field_columns", "columns.csv"),
            )
        },
        "domain": {
            "coordinate_system": "axisymmetric_rz",
            "external_id": 1,
            "layout_name": "background",
            "quadrilateral_order": "q1_reference",
            "excluded_boundary_ids": [],
        },
        "boundary_groups": [
            {
                "name": name,
                "external_ids": sorted(set(boundary.boundary_id[boundary.group_id == i].tolist())),
                "material_id": int(boundary.material_id[boundary.group_id == i][0]),
            }
            for i, name in enumerate(geometry.group_names)
        ],
        "fields": {
            "coordinate_columns": ["r", "z"],
            "inside_column": "inside",
            "node_id_column": "vertex_id",
            "domain_scope": None,
            "coordinate_match_tolerance_m": 1e-14,
            "mappings": [
                {
                    "name": "electric_field",
                    "columns": ["Er", "Ez"],
                    "components": ["r", "z"],
                    "stored_basis": "axisymmetric_rz",
                    "association": "node",
                    "unit": "V/m",
                    "input_unit": "V/m",
                    "scale": 1.0,
                    "offset": 0.0,
                    "axis_transform": None,
                    "producer_record": {
                        "producer": "editable rz-charged analytic example",
                        "time": {
                            "representation": "static",
                            "selection": "constant analytic field",
                        },
                        "domain_ids": [1],
                        "side": "domain",
                        "recovery": "none",
                        "continuity": "continuous",
                        "authority": "supplied analytic primitive",
                        "topology": "fixed",
                        "support": "full_selected_domain",
                        "evidence": "CSV values derive from the same example geometry and electric field",
                    },
                }
            ],
        },
    }


def _editable_source(source: RealizedSource, coordinates: str) -> tuple[bytes, dict[str, Any]]:
    columns: dict[str, Any] = {
        "particle_id": "particle_id",
        "release_time_s": "release_time_s",
    }
    surface = isinstance(source, RealizedSurfaceSource)
    if surface:
        columns.update(facet_id="facet_id", facet_parameter="facet_parameter")
    else:
        columns["position_m"] = ["x0_m", "x1_m"]
    properties = (
        "charge_number",
        "mass_kg",
        "drag_diameter_m",
        "contact_radius_m",
        "electrostatic_radius_m",
        "displaced_volume_m3",
        "model_weight",
        "material_id",
    )
    columns.update({name: name for name in properties})
    units = {
        "release_time_s": "s",
        "charge_number": "1",
        "mass_kg": "kg",
        "drag_diameter_m": "m",
        "contact_radius_m": "m",
        "electrostatic_radius_m": "m",
        "displaced_volume_m3": "m3",
        "model_weight": "1",
        "material_id": "1",
    }
    if not surface:
        units["position_m"] = "m"
    configuration = {
        "format_version": 2,
        "kind": "surface" if surface else "table",
        "table": source.name,
        "columns": columns,
        "units": units,
        "velocity_mode": "vector",
        "velocity_columns": ["v0_m_s", "v1_m_s"],
        "velocity_unit": "m/s",
        "components": ["r", "z"] if coordinates == "axisymmetric_rz" else ["x", "y"],
        "basis_to_case": [[1, 0], [0, 1]],
        "frame": "inertial",
        "metadata": {"scope": "editable analytic example", "unresolved": []},
    }
    stream = io.StringIO(newline="")
    writer = None
    for index in range(source.particle_id.size):
        row = {
            "particle_id": int(source.particle_id[index]),
            "release_time_s": float(source.release_time_s[index]),
            "v0_m_s": float(source.velocity_m_s[index, 0]),
            "v1_m_s": float(source.velocity_m_s[index, 1]),
        }
        if isinstance(source, RealizedSurfaceSource):
            row.update(
                facet_id=int(source.facet_id[index]),
                facet_parameter=float(source.facet_parameter[index]),
            )
        else:
            row.update(
                x0_m=float(source.position_m[index, 0]), x1_m=float(source.position_m[index, 1])
            )
        row.update({name: getattr(source, name)[index].item() for name in properties})
        if writer is None:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
        writer.writerow(row)
    return stream.getvalue().encode("utf-8"), configuration
