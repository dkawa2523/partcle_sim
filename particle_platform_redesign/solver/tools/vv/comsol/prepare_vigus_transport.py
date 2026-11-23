"""Prepare the explicit Vigus transport equations from inspected static primitives.

The input is a canonical case with realized particles, geometry, wall laws and
an independently chosen numerical clock. Native extraction and equation
correspondence remain external evidence. No MPH reader or trajectory replay is
part of this producer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np

from chamber_particles import load_case
from chamber_particles.case_format import DataBundle, FieldData, content_hash
from chamber_particles.physics.charge import ELECTRON_MASS_KG
from chamber_particles.physics.forces import (
    BOLTZMANN_J_K,
    ELEMENTARY_CHARGE_C,
    VACUUM_PERMITTIVITY_F_M,
)
from chamber_particles.yaml_input import parse_document
from chamber_particles_tools.inputs import create_case

TOOL_REVISION = "vigus_static_coefficient_transport_v2"
_PROPERTY_NAMES = ("mass_kg", "drag_diameter_m", "electrostatic_radius_m", "displaced_volume_m3")
_SCALARS = {
    "gas_density": "kg/m^3",
    "gas_temperature": "K",
    "gas_viscosity": "Pa*s",
    "gas_pressure": "Pa",
    "electron_density": "1/m^3",
    "ion_density": "1/m^3",
    "electron_temperature_V": "V",
    "ion_temperature": "K",
    "ion_mass": "kg",
    "ion_speed": "m/s",
    "rotation_rate": "1/s",
}


def _parameters(raw: bytes) -> dict[str, Any]:
    value = parse_document(raw)
    keys = {
        "format_version",
        "cmd_slip_coefficients",
        "native_particle_density_kg_m3",
        "native_capacitance_F",
        "maximum_abs_charge_number",
        "native_binding",
        "brownian",
    }
    if (
        not isinstance(value, dict)
        or set(value) != keys
        or type(value["format_version"]) is not int
        or value["format_version"] != 2
    ):
        raise ValueError("transport parameters require the documented format_version 2 keys")
    for name in (
        "native_particle_density_kg_m3",
        "native_capacitance_F",
        "maximum_abs_charge_number",
    ):
        _positive_parameter(value[name], name)
    slip = value["cmd_slip_coefficients"]
    if not isinstance(slip, list) or len(slip) != 3:
        raise ValueError("cmd_slip_coefficients must contain native C1, C2 and C3")
    for item in slip:
        _positive_parameter(item, "CMD coefficient")
    if not isinstance(value["native_binding"], dict) or not value["native_binding"]:
        raise ValueError("native_binding must identify the inspected model and exports")
    _brownian_mapping(value["brownian"])
    return value


def _positive_parameter(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a positive finite number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")


def _brownian_mapping(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("brownian must explicitly select a force interpretation")
    if value == {"interpretation": "continuous_white_force"}:
        return {"interpretation": "continuous_white_force", "spectral_density_multiplier": 1.0}
    keys = {"interpretation", "step_s", "force_normalization_step_s", "evidence"}
    if set(value) != keys or value["interpretation"] != "fixed_rk4_white_limit":
        raise ValueError(
            "brownian interpretation must be continuous_white_force or fixed_rk4_white_limit"
        )
    _positive_parameter(value["step_s"], "native RK4 step_s")
    _positive_parameter(value["force_normalization_step_s"], "native force_normalization_step_s")
    if value["step_s"] != value["force_normalization_step_s"]:
        raise ValueError(
            "fixed_rk4_white_limit requires native step equal to force normalization step"
        )
    if not isinstance(value["evidence"], str) or not value["evidence"].strip():
        raise ValueError(
            "fixed_rk4_white_limit requires bound method/random-value correspondence evidence"
        )
    return {
        **value,
        "spectral_density_multiplier": 5.0 / 9.0,
        "derivation": "RK4 time-key weights (1/6,2/3,1/6), shared adjacent endpoint; asymptotic variance rate sum(weights^2)+2*(1/6)^2",
        "scope": "weak white-force limit of fixed-step classical RK4 with identical random values at repeated physical times; not finite-step x/v covariance, event-path, variable-step or arbitrary state-dependent-noise equivalence",
    }


def _particle_properties(data: DataBundle) -> dict[str, float]:
    if not data.sources:
        raise ValueError("transport preparation requires realized sources")
    result = {}
    for name in _PROPERTY_NAMES:
        values = np.concatenate([getattr(source, name) for source in data.sources])
        if not values.size or not np.all(values == values[0]):
            raise ValueError(f"reference coefficient preparation requires homogeneous {name}")
        result[name] = float(values[0])
    return result


def _primitives(data: DataBundle) -> tuple[str, dict[str, np.ndarray]]:
    inventory = {field.name: field for field in data.fields}
    result = {}
    layout_names = set()
    for name, unit in (_SCALARS | {"gas_velocity": "m/s", "electric_field": "V/m"}).items():
        if name not in inventory:
            raise ValueError(f"missing inspected primitive: {name}")
        field = inventory[name]
        layout_names.add(field.layout)
        result[name] = _primitive_values(field, unit, data.coordinate_system)
    if len(layout_names) != 1:
        raise ValueError("transport primitives must share one layout")
    return layout_names.pop(), result


def _primitive_values(field: FieldData, unit: str, coordinate: str) -> np.ndarray:
    name = field.name
    vector = name in {"gas_velocity", "electric_field"}
    components = (
        (("r", "z") if coordinate == "axisymmetric_rz" else ("x", "y")) if vector else ("value",)
    )
    basis = coordinate if vector else "scalar"
    if (field.unit, field.components, field.stored_basis, field.association) != (
        unit,
        components,
        basis,
        "node",
    ) or field.time_s is not None:
        raise ValueError(f"{name} must be a static SI nodal primitive in the canonical basis")
    if not np.all(np.isfinite(field.values)):
        raise ValueError(f"{name} contains nonfinite values")
    if not vector and name not in {"ion_speed", "rotation_rate"} and np.any(field.values <= 0):
        raise ValueError(f"{name} must be positive")
    if name == "ion_speed" and np.any(field.values < 0):
        raise ValueError("ion_speed must be nonnegative")
    return field.values if vector else field.values[:, 0]


def _full_force_fields(
    data: DataBundle,
    parameters: dict[str, Any],
    properties: dict[str, float],
) -> tuple[tuple[FieldData, ...], float]:
    layout, p = _primitives(data)
    mass, diameter = properties["mass_kg"], properties["drag_diameter_m"]
    radius = properties["electrostatic_radius_m"]
    if radius <= 0:
        raise ValueError("Vigus collection and ion force require a positive electrical radius")
    c1, c2, c3 = parameters["cmd_slip_coefficients"]
    k, e, eps = BOLTZMANN_J_K, ELEMENTARY_CHARGE_C, VACUUM_PERMITTIVITY_F_M
    with np.errstate(over="raise", divide="raise", invalid="raise", under="ignore"):
        mean_path = p["gas_viscosity"] * np.sqrt(np.pi / (2 * p["gas_density"] * p["gas_pressure"]))
        kn = mean_path / diameter
        slip = 1 + kn * (c1 + c2 * np.exp(-c3 / kn))
        friction = (
            mass
            * 18
            * p["gas_viscosity"]
            / (parameters["native_particle_density_kg_m3"] * diameter**2 * slip)
        )
        diffusion = 6 * np.pi * p["gas_viscosity"] * diameter * k * p["gas_temperature"]
        diffusion *= _brownian_mapping(parameters["brownian"])["spectral_density_multiplier"]
        capacitance = parameters["native_capacitance_F"]
        alpha = e / (capacitance * p["electron_temperature_V"])
        beta_z = e**2 / (capacitance * k * p["ion_temperature"])
        log_electron = (
            math.log(np.pi * radius**2)
            + np.log(p["electron_density"])
            + 0.5 * np.log(8 * e * p["electron_temperature_V"] / (np.pi * ELECTRON_MASS_KG))
        )
        log_reference = float(np.max(log_electron))
        electron_reference = math.exp(log_reference)
        offset = log_electron - log_reference
        ion_rate = (
            np.pi
            * radius**2
            * p["ion_density"]
            * np.sqrt(8 * k * p["ion_temperature"] / (np.pi * p["ion_mass"]))
        )
        speed = np.sqrt(
            p["ion_speed"] ** 2 + 8 * k * p["ion_temperature"] / (np.pi * p["ion_mass"])
        )
        electric_norm = np.hypot(p["electric_field"][:, 0], p["electric_field"][:, 1])
        if np.any(electric_norm == 0):
            raise ValueError("literal field-directed ion force is undefined at zero electric field")
        direction = p["electric_field"] / electric_norm[:, None]
        prefactor = -p["ion_mass"] * p["ion_density"] * speed * p["ion_speed"]
        screening_ratio = (
            np.sqrt(eps * p["electron_temperature_V"] / (p["ion_density"] * e)) / radius
        )
        orbital = (
            np.pi
            * (e**2 / (2 * np.pi * eps * p["ion_mass"] * speed**2)) ** 2
            * np.log(screening_ratio)
        )
        f0 = (prefactor * np.pi * radius**2)[:, None] * direction
        f1 = -beta_z[:, None] * f0
        f2 = (prefactor * orbital)[:, None] * direction
        omega = p["rotation_rate"]
        rotation = (
            6.46
            * (diameter / 2) ** 2
            * np.sqrt(p["gas_viscosity"] * p["gas_density"] * np.abs(omega))
            * np.sign(omega)
        )
    scalars = {
        "reference_friction": (friction, "kg/s"),
        "reference_force_diffusion": (diffusion, "N^2*s"),
        "reference_electron_offset": (offset, "1"),
        "reference_alpha": (alpha, "1"),
        "reference_ion_rate": (ion_rate, "1/s"),
        "reference_beta_z": (beta_z, "1"),
        "reference_rotation": (rotation, "kg/s"),
    }
    fields = [
        FieldData(
            name, layout, "node", ("value",), "scalar", np.ascontiguousarray(values[:, None]), unit
        )
        for name, (values, unit) in scalars.items()
    ]
    components = ("r", "z") if data.coordinate_system == "axisymmetric_rz" else ("x", "y")
    fields.extend(
        FieldData(
            name,
            layout,
            "node",
            components,
            data.coordinate_system,
            np.ascontiguousarray(values),
            "N",
        )
        for name, values in (
            ("reference_ion_f0", f0),
            ("reference_ion_f1", f1),
            ("reference_ion_f2", f2),
        )
    )
    if any(not np.all(np.isfinite(field.values)) for field in fields):
        raise ValueError("derived transport coefficients are nonfinite")
    return tuple(fields), electron_reference


def _reference_models(properties: dict[str, float], electron_reference: float) -> dict[str, Any]:
    return {
        "drag": {
            "model": "prescribed_linear_friction",
            "revision": "prescribed_linear_friction_v1",
            "gas_velocity_field": "gas_velocity",
            "friction_coefficient_field": "reference_friction",
            "particle_properties": properties,
        },
        "noise": {
            "model": "prescribed_force_diffusion",
            "revision": "prescribed_force_diffusion_midpoint_2d_v1",
            "force_spectral_density_field": "reference_force_diffusion",
            "particle_properties": properties,
        },
        "charge": {
            "model": "prescribed_exponential_linear",
            "revision": "prescribed_exponential_linear_collection_v1",
            "electron_rate_reference_number_s": electron_reference,
            "electron_exponent_offset_field": "reference_electron_offset",
            "electron_charge_slope_field": "reference_alpha",
            "ion_rate_field": "reference_ion_rate",
            "ion_charge_slope_field": "reference_beta_z",
            "particle_properties": properties,
        },
        "ion_drag": {
            "model": "prescribed_charge_polynomial",
            "revision": "prescribed_charge_polynomial_v1",
            "constant_force_field": "reference_ion_f0",
            "linear_charge_force_field": "reference_ion_f1",
            "quadratic_charge_force_field": "reference_ion_f2",
            "particle_properties": properties,
        },
        "lift": {
            "model": "prescribed_rotational_lift",
            "revision": "prescribed_rotational_lift_v1",
            "gas_velocity_field": "gas_velocity",
            "rotation_coefficient_field": "reference_rotation",
            "particle_properties": properties,
        },
    }


def prepare_transport(
    initial_case: Path,
    parameters_path: Path,
    output: Path,
    *,
    interval_tree_depth: int,
    adaptive_max_depth: int | None = None,
) -> dict[str, Any]:
    """Preserve realized source/walls/clock and write one reference coefficient case."""
    case_bytes, parameter_bytes = initial_case.read_bytes(), parameters_path.read_bytes()
    parameters = _parameters(parameter_bytes)
    case = load_case(initial_case)
    if case_bytes != initial_case.read_bytes():
        raise ValueError("initial YAML changed while loading the case")
    document = cast(dict[str, Any], parse_document(case_bytes))
    properties = _particle_properties(case.data)
    fields, electron_reference = _full_force_fields(case.data, parameters, properties)
    if {field.name for field in fields}.intersection(field.name for field in case.data.fields):
        raise ValueError(
            "reference coefficient fields already exist; regenerate from the primitive case"
        )
    models = _reference_models(properties, electron_reference)
    models["noise"]["interval_tree_depth"] = interval_tree_depth
    if adaptive_max_depth is not None:
        models["noise"]["adaptive_max_depth"] = adaptive_max_depth
    models["electric"] = {
        "model": "coulomb",
        "revision": "electric_coulomb_charge_cap_v1",
        "electric_field": "electric_field",
        "maximum_abs_charge_number": parameters["maximum_abs_charge_number"],
    }
    document["physics"].update(models)
    document["solver"]["integrator"] = "ou_langevin"
    metadata = {
        "initial_content_hash": case.content_hash,
        "initial_yaml_sha256": hashlib.sha256(case_bytes).hexdigest(),
        "parameters_sha256": hashlib.sha256(parameter_bytes).hexdigest(),
        "parameters": parameters,
        "particle_properties": properties,
        "coefficient_fields": [field.name for field in fields],
        "primitive_provenance": json.loads(case.data.provenance_json),
        "rotation_meaning": "d(ug_z)/dr-d(ug_r)/dz; near-zero native Saffman regularization requires separate correspondence evidence",
        "brownian_mapping": _brownian_mapping(parameters["brownian"]),
    }
    provenance = json.dumps(
        {
            "producer": "tools.vv.comsol.prepare_vigus_transport",
            "producer_version": TOOL_REVISION,
            "source_sha256": "sha256:" + hashlib.sha256(parameter_bytes + case_bytes).hexdigest(),
            "field_semantics_revision": TOOL_REVISION,
            "producer_metadata": metadata,
        },
        allow_nan=False,
    )
    bundle = replace(case.data, fields=(*case.data.fields, *fields), provenance_json=provenance)
    receipt = {"tool_revision": TOOL_REVISION, "content_hash": content_hash(bundle), **metadata}
    create_case(
        bundle,
        document,
        output,
        supporting_files={
            "build_manifest.json": json.dumps(receipt, indent=2, allow_nan=False).encode("utf-8")
        },
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("initial_case", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--parameters", required=True, type=Path)
    parser.add_argument("--interval-tree-depth", required=True, type=int)
    parser.add_argument("--adaptive-max-depth", type=int)
    args = parser.parse_args()
    prepare_transport(
        args.initial_case,
        args.parameters,
        args.output,
        interval_tree_depth=args.interval_tree_depth,
        adaptive_max_depth=args.adaptive_max_depth,
    )


if __name__ == "__main__":
    main()
