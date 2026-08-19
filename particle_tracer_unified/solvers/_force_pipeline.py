from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from particle_tracer_unified.domain import StageFields

from .force_validation import (
    require_batch_quantity as _require_batch_quantity,
)
from .force_validation import (
    require_force_parameter as _require_force_parameter,
)
from .forces.runtime import ForceRuntimeParameters


@dataclass(frozen=True)
class ForceBatchStatic:
    """Particle properties, with diameter meaning physical sphere geometry."""

    particle_diameter: np.ndarray
    particle_density: np.ndarray
    particle_mass: np.ndarray
    dep_particle_rel_permittivity: np.ndarray
    thermophoretic_coeff: np.ndarray


@dataclass(frozen=True)
class ForceBatchState:
    velocity: np.ndarray
    charge_over_mass: np.ndarray | None = None


@dataclass(frozen=True)
class ForcePipeline:
    evaluator_names: tuple[str, ...]
    params: ForceRuntimeParameters
    gas_molecular_mass_kg: float = float("nan")


_ForceRequirement = tuple[np.ndarray, tuple[int, ...], str, set[str]]
_ForceRequirements = dict[str, _ForceRequirement]


def _add_force_requirement(
    requirements: _ForceRequirements,
    quantity: str,
    raw: np.ndarray,
    shape: tuple[int, ...],
    rule: str,
    force: str,
) -> None:
    existing = requirements.get(quantity)
    if existing is None:
        requirements[quantity] = (raw, shape, rule, {force})
        return
    _, existing_shape, existing_rule, consumers = existing
    if existing_shape != shape or existing_rule != rule:
        raise AssertionError(
            f"conflicting requirements for force quantity {quantity!r}"
        )
    consumers.add(force)


def _add_force_field_requirement(
    requirements: _ForceRequirements,
    fields: StageFields,
    quantity: str,
    shape: tuple[int, ...],
    rule: str,
    force: str,
) -> None:
    if quantity not in fields.values:
        raise ValueError(f"{force} requires field input quantity {quantity!r}")
    _add_force_requirement(
        requirements,
        quantity,
        fields.values[quantity],
        shape,
        rule,
        force,
    )


def _validated_force_output_shape(
    out_accel: np.ndarray,
    fields: StageFields,
) -> tuple[np.ndarray, int, int]:
    out = np.asarray(out_accel)
    if out.ndim != 2 or out.shape[1] not in (2, 3):
        raise ValueError(
            "force output quantity 'acceleration' must have shape (particle, 2|3)"
        )
    count, dim = (int(out.shape[0]), int(out.shape[1]))
    if np.asarray(fields.points_m).shape != (count, dim):
        raise ValueError(
            "force field sample points must match output shape "
            f"({count}, {dim}); received {np.asarray(fields.points_m).shape}"
        )
    return out, count, dim


def _add_electric_input_requirements(
    requirements: _ForceRequirements,
    names: set[str],
    state: ForceBatchState,
    fields: StageFields,
    *,
    count: int,
    dim: int,
) -> None:
    if "electric" not in names:
        return
    if state.charge_over_mass is None:
        raise ValueError("electric requires particle input quantity 'charge_over_mass'")
    _add_force_requirement(
        requirements,
        "charge_over_mass",
        state.charge_over_mass,
        (count,),
        "finite",
        "electric",
    )
    _add_force_field_requirement(
        requirements,
        fields,
        "electric_field",
        (count, dim),
        "finite",
        "electric",
    )


def _add_displaced_fluid_input_requirements(
    requirements: _ForceRequirements,
    names: set[str],
    static: ForceBatchStatic,
    state: ForceBatchState,
    fields: StageFields,
    plan: ForcePipeline,
    *,
    count: int,
    dim: int,
) -> None:
    density_forces = names.intersection({"pressure_gradient", "virtual_mass"})
    for force in density_forces:
        _add_force_requirement(
            requirements,
            "particle_density",
            static.particle_density,
            (count,),
            "positive",
            force,
        )
        _add_force_field_requirement(
            requirements,
            fields,
            "gas_density",
            (count,),
            "positive",
            force,
        )
    if "pressure_gradient" in names:
        _add_force_field_requirement(
            requirements,
            fields,
            "fluid_acceleration",
            (count, dim),
            "finite",
            "pressure_gradient",
        )
    if "virtual_mass" not in names:
        return
    _add_force_requirement(
        requirements,
        "velocity",
        state.velocity,
        (count, dim),
        "finite",
        "virtual_mass",
    )
    _add_force_field_requirement(
        requirements,
        fields,
        "flow_time_derivative",
        (count, dim),
        "finite",
        "virtual_mass",
    )
    _add_force_field_requirement(
        requirements,
        fields,
        "flow_velocity_gradient",
        (count, dim, dim),
        "finite",
        "virtual_mass",
    )
    _require_force_parameter(
        "virtual_mass",
        "coefficient",
        plan.params.virtual_mass_coefficient,
        rule="positive",
    )


def _add_particle_transport_requirements(
    requirements: _ForceRequirements,
    names: set[str],
    static: ForceBatchStatic,
    fields: StageFields,
    *,
    count: int,
) -> None:
    size_mass_forces = names.intersection(
        {"thermophoresis", "dielectrophoresis", "lift"}
    )
    for force in size_mass_forces:
        _add_force_requirement(
            requirements,
            "particle_diameter",
            static.particle_diameter,
            (count,),
            "positive",
            force,
        )
        _add_force_requirement(
            requirements,
            "particle_mass",
            static.particle_mass,
            (count,),
            "positive",
            force,
        )
    gas_transport_forces = names.intersection({"thermophoresis", "lift"})
    for force in gas_transport_forces:
        _add_force_field_requirement(
            requirements,
            fields,
            "gas_density",
            (count,),
            "positive",
            force,
        )
        _add_force_field_requirement(
            requirements,
            fields,
            "dynamic_viscosity",
            (count,),
            "positive",
            force,
        )


def _add_thermophoresis_input_requirements(
    requirements: _ForceRequirements,
    names: set[str],
    static: ForceBatchStatic,
    fields: StageFields,
    plan: ForcePipeline,
    *,
    count: int,
    dim: int,
) -> None:
    if "thermophoresis" not in names:
        return
    _add_force_requirement(
        requirements,
        "thermophoretic_coeff",
        static.thermophoretic_coeff,
        (count,),
        "optional_positive",
        "thermophoresis",
    )
    _add_force_field_requirement(
        requirements,
        fields,
        "temperature",
        (count,),
        "positive",
        "thermophoresis",
    )
    _add_force_field_requirement(
        requirements,
        fields,
        "temperature_gradient",
        (count, dim),
        "finite",
        "thermophoresis",
    )
    for quantity, value in (
        ("gas_thermal_conductivity_W_mK", plan.params.gas_thermal_conductivity_W_mK),
        (
            "particle_thermal_conductivity_W_mK",
            plan.params.particle_thermal_conductivity_W_mK,
        ),
        ("Cs", plan.params.thermophoresis_Cs),
        ("Cm", plan.params.thermophoresis_Cm),
        ("Ct", plan.params.thermophoresis_Ct),
    ):
        _require_force_parameter("thermophoresis", quantity, value, rule="positive")
    if str(plan.params.thermophoresis_model).lower() != "continuum":
        _require_force_parameter(
            "thermophoresis",
            "gas_molecular_mass_kg",
            plan.gas_molecular_mass_kg,
            rule="positive",
        )


def _add_dielectrophoresis_input_requirements(
    requirements: _ForceRequirements,
    names: set[str],
    static: ForceBatchStatic,
    fields: StageFields,
    plan: ForcePipeline,
    *,
    count: int,
    dim: int,
) -> None:
    if "dielectrophoresis" not in names:
        return
    _add_force_requirement(
        requirements,
        "particle_relative_permittivity",
        static.dep_particle_rel_permittivity,
        (count,),
        "optional_positive",
        "dielectrophoresis",
    )
    _add_force_field_requirement(
        requirements,
        fields,
        "electric_magnitude_squared_gradient",
        (count, dim),
        "finite",
        "dielectrophoresis",
    )
    _require_force_parameter(
        "dielectrophoresis",
        "medium_relative_permittivity",
        plan.params.dep_medium_rel_permittivity,
        rule="positive",
    )
    _require_force_parameter(
        "dielectrophoresis",
        "particle_relative_permittivity",
        plan.params.dep_particle_rel_permittivity,
        rule="optional_positive",
    )
    for quantity, value in (
        ("medium_conductivity_Sm", plan.params.dep_medium_conductivity_Sm),
        ("particle_conductivity_Sm", plan.params.dep_particle_conductivity_Sm),
        ("frequency_Hz", plan.params.dep_frequency_Hz),
    ):
        _require_force_parameter(
            "dielectrophoresis",
            quantity,
            value,
            rule="nonnegative",
        )
    amplitude = str(plan.params.dep_electric_field_amplitude)
    if amplitude not in {"rms", "peak"}:
        raise ValueError(
            "dielectrophoresis electric_field_amplitude must be 'rms' or 'peak'"
        )


def _add_lift_input_requirements(
    requirements: _ForceRequirements,
    names: set[str],
    state: ForceBatchState,
    fields: StageFields,
    plan: ForcePipeline,
    *,
    count: int,
    dim: int,
) -> None:
    if "lift" not in names:
        return
    _add_force_requirement(
        requirements,
        "velocity",
        state.velocity,
        (count, dim),
        "finite",
        "lift",
    )
    _add_force_field_requirement(
        requirements,
        fields,
        "flow_velocity",
        (count, dim),
        "finite",
        "lift",
    )
    _add_force_field_requirement(
        requirements,
        fields,
        "vorticity",
        (count, 3),
        "finite",
        "lift",
    )
    _require_force_parameter(
        "lift",
        "coefficient",
        plan.params.lift_coefficient,
        rule="positive",
    )


def _validate_force_requirements(
    requirements: _ForceRequirements,
) -> dict[str, np.ndarray]:
    validated: dict[str, np.ndarray] = {}
    for quantity, (raw, shape, rule, consumers) in requirements.items():
        validated[quantity] = _require_batch_quantity(
            quantity,
            raw,
            shape,
            rule=rule,
            forces=consumers,
        )
    return validated


def _validate_dep_particle_permittivity(
    names: set[str],
    validated: Mapping[str, np.ndarray],
    params: ForceRuntimeParameters,
) -> None:
    if "dielectrophoresis" not in names:
        return
    particle_eps = validated["particle_relative_permittivity"]
    has_config_fallback = bool(
        np.isfinite(float(params.dep_particle_rel_permittivity))
        and float(params.dep_particle_rel_permittivity) > 0.0
    )
    if np.any(np.isnan(particle_eps)) and not has_config_fallback:
        rows = np.flatnonzero(np.isnan(particle_eps)).tolist()
        raise ValueError(
            "dielectrophoresis requires explicit positive particle "
            "relative permittivity "
            f"per particle or in force configuration; invalid particle rows: {rows}"
        )


def _validate_force_inputs(
    out_accel: np.ndarray,
    static: ForceBatchStatic,
    state: ForceBatchState,
    fields: StageFields,
    plan: ForcePipeline,
) -> None:
    """Validate the complete enabled-force batch once before evaluating equations."""

    out, count, dim = _validated_force_output_shape(out_accel, fields)
    names = set(plan.evaluator_names)
    requirements: _ForceRequirements = {}
    _add_force_requirement(
        requirements,
        "acceleration",
        out,
        (count, dim),
        "finite",
        "force_pipeline",
    )
    _add_force_requirement(
        requirements,
        "points_m",
        fields.points_m,
        (count, dim),
        "finite",
        "force_pipeline",
    )
    _add_electric_input_requirements(
        requirements,
        names,
        state,
        fields,
        count=count,
        dim=dim,
    )
    _add_displaced_fluid_input_requirements(
        requirements,
        names,
        static,
        state,
        fields,
        plan,
        count=count,
        dim=dim,
    )
    _add_particle_transport_requirements(
        requirements,
        names,
        static,
        fields,
        count=count,
    )
    _add_thermophoresis_input_requirements(
        requirements,
        names,
        static,
        fields,
        plan,
        count=count,
        dim=dim,
    )
    _add_dielectrophoresis_input_requirements(
        requirements,
        names,
        static,
        fields,
        plan,
        count=count,
        dim=dim,
    )
    _add_lift_input_requirements(
        requirements,
        names,
        state,
        fields,
        plan,
        count=count,
        dim=dim,
    )
    validated = _validate_force_requirements(requirements)
    _validate_dep_particle_permittivity(names, validated, plan.params)


def _force_pipeline_from_names(
    params: ForceRuntimeParameters,
    names: Sequence[str],
    *,
    gas_molecular_mass_kg: float = float("nan"),
) -> ForcePipeline:
    seen: set[str] = set()
    ordered_names: list[str] = []
    for name in names:
        name_text = str(name)
        if not name_text or name_text in seen:
            continue
        seen.add(name_text)
        ordered_names.append(name_text)
    evaluator_names = tuple(ordered_names)
    return ForcePipeline(
        evaluator_names=evaluator_names,
        params=params,
        gas_molecular_mass_kg=float(gas_molecular_mass_kg),
    )


def build_force_pipeline(
    params: ForceRuntimeParameters | None,
    *,
    include_electric: bool = False,
    gas_molecular_mass_kg: float = float("nan"),
) -> ForcePipeline:
    p = params or ForceRuntimeParameters()
    names = (
        ("electric", *p.enabled_evaluator_names())
        if bool(include_electric)
        else p.enabled_evaluator_names()
    )
    return _force_pipeline_from_names(
        p,
        names,
        gas_molecular_mass_kg=float(gas_molecular_mass_kg),
    )
