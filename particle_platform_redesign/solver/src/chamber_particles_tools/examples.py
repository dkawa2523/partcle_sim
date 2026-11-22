"""Small analytic examples shipped as code, editable settings and realized CSVs."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import yaml

from chamber_particles.case import SimulationCase
from chamber_particles.case_format import (
    BoundaryData,
    DataBundle,
    FieldData,
    GeometryData,
    P1TriLayout,
    RealizedTableSource,
    read,
    write,
)

from .inputs import create_case

EXAMPLE_NAMES = (
    "xy-ballistic",
    "rz-charged",
    "xy-time-field",
    "brownian-wall",
    "reduced-electrostatic",
)


def initialize(name: str, output_directory: str | Path) -> SimulationCase:
    """Create a checked case and its editable source CSV/recipe."""
    if name not in EXAMPLE_NAMES:
        raise ValueError(f"unknown example {name!r}; choose one of {', '.join(EXAMPLE_NAMES)}")
    bundle = _bundle(name)
    thermal = bundle
    settings = _settings(name)
    if name == "reduced-electrostatic":
        bundle = _build_electrostatic(bundle)
    case = create_case(bundle, settings, bundle.sources, output_directory)
    _editable_sources(case, settings)
    if name == "reduced-electrostatic":
        write(case.case_path.parent / "thermal.h5", thermal)
        (case.case_path.parent / "builder.yaml").write_text(
            yaml.safe_dump(_electrostatic_configuration(), sort_keys=False), encoding="utf-8"
        )
    return case


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


def _bundle(name: str) -> DataBundle:
    rz = name in ("rz-charged", "reduced-electrostatic")
    geometry = _geometry(rz)
    assert geometry.tri3 is not None
    layout = P1TriLayout("background", geometry.nodes_m, geometry.tri3, np.ones(2, dtype="u1"))
    coordinates = "axisymmetric_rz" if rz else "cartesian_xy"
    fields: tuple[FieldData, ...] = ()
    if name == "rz-charged":
        fields = (_field("electric_field", [0.0, 1000.0], "V/m", coordinates),)
    elif name == "xy-time-field":
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
    elif name in ("brownian-wall", "reduced-electrostatic"):
        fields = (
            _field("gas_velocity", [0.0, 0.0], "m/s", coordinates),
            _field("gas_temperature", [300.0], "K"),
        )
        if name == "brownian-wall":
            fields += (
                _field("gas_density", [1e-3], "kg/m^3"),
                _field("gas_mean_free_path", [1e-3], "m"),
            )
    source = _source(rz, name == "brownian-wall")
    if name == "xy-time-field":
        source = replace(source, charge_number=np.asarray([3.0]))
    provenance = json.dumps(
        {
            "producer": "chamber_particles_tools.examples",
            "producer_version": "1",
            "source_sha256": "sha256:" + hashlib.sha256(name.encode()).hexdigest(),
            "field_semantics_revision": "analytic_editable_examples_v1",
            "producer_metadata": {"example": name, "purpose": "educational analytic input"},
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


def _settings(name: str) -> dict[str, Any]:
    rz = name in ("rz-charged", "reduced-electrostatic")
    brownian = name == "brownian-wall"
    physics: dict[str, Any] = {"charge": {"model": "fixed"}}
    if name in ("rz-charged", "xy-time-field", "reduced-electrostatic"):
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
        build_from_configuration(config_path, directory / "electric.h5")
        return read(directory / "electric.h5")


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


def _editable_sources(case: SimulationCase, settings: dict[str, Any]) -> None:
    source = case.data.sources[0]
    assert isinstance(source, RealizedTableSource)
    columns = {
        "particle_id": "particle_id",
        "release_time_s": "release_time_s",
        "position_m": ["x0_m", "x1_m"],
    }
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
        "position_m": "m",
        "charge_number": "1",
        "mass_kg": "kg",
        "drag_diameter_m": "m",
        "contact_radius_m": "m",
        "electrostatic_radius_m": "m",
        "displaced_volume_m3": "m3",
        "model_weight": "1",
        "material_id": "1",
    }
    configuration = {
        "format_version": 2,
        "kind": "table",
        "table": source.name,
        "columns": columns,
        "units": units,
        "velocity_mode": "vector",
        "velocity_columns": ["v0_m_s", "v1_m_s"],
        "velocity_unit": "m/s",
        "components": ["r", "z"]
        if case.data.coordinate_system == "axisymmetric_rz"
        else ["x", "y"],
        "basis_to_case": [[1, 0], [0, 1]],
        "frame": "inertial",
        "metadata": {"scope": "editable analytic example", "unresolved": []},
    }
    row = {
        "particle_id": int(source.particle_id[0]),
        "release_time_s": float(source.release_time_s[0]),
        "x0_m": float(source.position_m[0, 0]),
        "x1_m": float(source.position_m[0, 1]),
        "v0_m_s": float(source.velocity_m_s[0, 0]),
        "v1_m_s": float(source.velocity_m_s[0, 1]),
    }
    row.update({name: getattr(source, name)[0].item() for name in properties})
    with (case.case_path.parent / "particles.csv").open(
        "x", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    recipe = {
        "format_version": 1,
        "input_bundle": "case.h5",
        "sources": [{"csv": "particles.csv", "configuration": configuration}],
        "settings": settings,
    }
    (case.case_path.parent / "recipe.yaml").write_text(
        yaml.safe_dump(recipe, sort_keys=False), encoding="utf-8"
    )
