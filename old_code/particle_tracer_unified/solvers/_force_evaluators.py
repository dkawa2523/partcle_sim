from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np

from particle_tracer_unified.domain import StageFields

from ._force_pipeline import (
    ForceBatchState,
    ForceBatchStatic,
    ForcePipeline,
    _validate_force_inputs,
)
from .force_validation import invalid_particle_rows as _invalid_particle_rows

_EPS0_F_M = 8.8541878128e-12
_K_BOLTZMANN = 1.380649e-23


def _physical_diameter(static: ForceBatchStatic) -> np.ndarray:
    return np.asarray(static.particle_diameter, dtype=np.float64).reshape(-1)


def _cm_factor_real(
    particle_rel_permittivity: float,
    medium_rel_permittivity: float,
    particle_conductivity_Sm: float,
    medium_conductivity_Sm: float,
    frequency_Hz: float,
) -> float:
    eps_p = float(particle_rel_permittivity)
    eps_m = float(medium_rel_permittivity)
    freq = float(frequency_Hz)
    if freq == 0.0:
        return float((eps_p - eps_m) / (eps_p + 2.0 * eps_m))
    omega = 2.0 * np.pi * freq
    rel_p = complex(eps_p, -float(particle_conductivity_Sm) / (omega * _EPS0_F_M))
    rel_m = complex(eps_m, -float(medium_conductivity_Sm) / (omega * _EPS0_F_M))
    value = (rel_p - rel_m) / (rel_p + 2.0 * rel_m)
    return float(value.real)


def add_electric_acceleration(
    out_accel: np.ndarray,
    static: ForceBatchStatic,
    state: ForceBatchState,
    fields: StageFields,
    plan: ForcePipeline,
) -> None:
    del static, plan
    if state.charge_over_mass is None:
        raise ValueError("electric force requires particle charge_over_mass")
    electric = fields.require("electric_field")
    qom = np.asarray(state.charge_over_mass, dtype=np.float64).reshape(-1)
    out_accel[:, : electric.shape[1]] += qom[:, None] * electric


def add_pressure_gradient_acceleration(
    out_accel: np.ndarray,
    static: ForceBatchStatic,
    state: ForceBatchState,
    fields: StageFields,
    plan: ForcePipeline,
) -> None:
    del state, plan
    rho_p = np.asarray(static.particle_density, dtype=np.float64).reshape(-1)
    rho_g = np.asarray(fields.require("gas_density"), dtype=np.float64).reshape(-1)
    fluid = np.asarray(fields.require("fluid_acceleration"), dtype=np.float64)
    scale = rho_g / rho_p
    out_accel[:, : fluid.shape[1]] += scale[:, None] * fluid


def add_virtual_mass_acceleration(
    out_accel: np.ndarray,
    static: ForceBatchStatic,
    state: ForceBatchState,
    fields: StageFields,
    plan: ForcePipeline,
) -> None:
    params = plan.params
    rho_p = np.asarray(static.particle_density, dtype=np.float64).reshape(-1)
    rho_g = np.asarray(fields.require("gas_density"), dtype=np.float64).reshape(-1)
    dudt = np.asarray(fields.require("flow_time_derivative"), dtype=np.float64)
    grad_u = np.asarray(fields.require("flow_velocity_gradient"), dtype=np.float64)
    vel = np.asarray(state.velocity, dtype=np.float64)
    particle_accel = dudt + np.einsum("nij,nj->ni", grad_u, vel[:, : dudt.shape[1]])
    scale = float(params.virtual_mass_coefficient) * rho_g / rho_p
    out_accel[:, : particle_accel.shape[1]] += scale[:, None] * particle_accel


def add_thermophoresis_acceleration(
    out_accel: np.ndarray,
    static: ForceBatchStatic,
    state: ForceBatchState,
    fields: StageFields,
    plan: ForcePipeline,
) -> None:
    del state
    params = plan.params
    physical_diameter = _physical_diameter(static)
    mass = np.asarray(static.particle_mass, dtype=np.float64).reshape(-1)
    radius = 0.5 * physical_diameter
    rho_g = np.asarray(fields.require("gas_density"), dtype=np.float64).reshape(-1)
    mu_g = np.asarray(fields.require("dynamic_viscosity"), dtype=np.float64).reshape(-1)
    temp_g = np.asarray(fields.require("temperature"), dtype=np.float64).reshape(-1)
    grad_T = np.asarray(fields.require("temperature_gradient"), dtype=np.float64)
    if str(params.thermophoresis_model).lower() == "continuum":
        kn = np.zeros_like(rho_g)
    else:
        mol_mass = float(plan.gas_molecular_mass_kg)
        mean_free_path = (mu_g / rho_g) * np.sqrt(
            np.pi * mol_mass / (2.0 * _K_BOLTZMANN * temp_g)
        )
        kn = mean_free_path / radius
    ratio = float(params.gas_thermal_conductivity_W_mK) / float(
        params.particle_thermal_conductivity_W_mK
    )
    factor = (
        2.0
        * float(params.thermophoresis_Cs)
        * (ratio + float(params.thermophoresis_Ct) * kn)
        / (
            (1.0 + 3.0 * float(params.thermophoresis_Cm) * kn)
            * (1.0 + 2.0 * ratio + 2.0 * float(params.thermophoresis_Ct) * kn)
        )
    )
    multiplier = np.where(
        np.isnan(static.thermophoretic_coeff),
        1.0,
        np.asarray(static.thermophoretic_coeff, dtype=np.float64),
    )
    tau_stokes = mass / (3.0 * np.pi * mu_g * physical_diameter)
    scale = -multiplier * factor * mu_g / (rho_g * temp_g * tau_stokes)
    out_accel[:, : grad_T.shape[1]] += scale[:, None] * grad_T


def add_dielectrophoresis_acceleration(
    out_accel: np.ndarray,
    static: ForceBatchStatic,
    state: ForceBatchState,
    fields: StageFields,
    plan: ForcePipeline,
) -> None:
    del state
    params = plan.params
    physical_diameter = _physical_diameter(static)
    mass = np.asarray(static.particle_mass, dtype=np.float64).reshape(-1)
    radius = 0.5 * physical_diameter
    epsp_arr = np.asarray(
        static.dep_particle_rel_permittivity, dtype=np.float64
    ).reshape(-1)
    grad_E2 = np.asarray(
        fields.require("electric_magnitude_squared_gradient"), dtype=np.float64
    )
    epsp = np.where(
        np.isnan(epsp_arr),
        float(params.dep_particle_rel_permittivity),
        epsp_arr,
    )
    epsm = float(params.dep_medium_rel_permittivity)
    if float(params.dep_frequency_Hz) <= 0.0:
        cm_real = (epsp - epsm) / (epsp + 2.0 * epsm)
    else:
        cm_real = np.asarray(
            [
                _cm_factor_real(
                    float(value),
                    epsm,
                    float(params.dep_particle_conductivity_Sm),
                    float(params.dep_medium_conductivity_Sm),
                    float(params.dep_frequency_Hz),
                )
                for value in epsp
            ],
            dtype=np.float64,
        )
    coeff = 2.0 * np.pi * _EPS0_F_M * epsm * radius**3 * cm_real / mass
    if (
        float(params.dep_frequency_Hz) > 0.0
        and str(params.dep_electric_field_amplitude) == "peak"
    ):
        coeff *= 0.5
    out_accel[:, : grad_E2.shape[1]] += coeff[:, None] * grad_E2


def add_lift_acceleration(
    out_accel: np.ndarray,
    static: ForceBatchStatic,
    state: ForceBatchState,
    fields: StageFields,
    plan: ForcePipeline,
) -> None:
    params = plan.params
    physical_diameter = _physical_diameter(static)
    mass = np.asarray(static.particle_mass, dtype=np.float64).reshape(-1)
    radius = 0.5 * physical_diameter
    rho_g = np.asarray(fields.require("gas_density"), dtype=np.float64).reshape(-1)
    mu_g = np.asarray(fields.require("dynamic_viscosity"), dtype=np.float64).reshape(-1)
    flow = np.asarray(fields.require("flow_velocity"), dtype=np.float64)
    vel = np.asarray(state.velocity, dtype=np.float64)
    vorticity = np.asarray(fields.require("vorticity"), dtype=np.float64)
    dim = int(out_accel.shape[1])
    slip = flow[:, :dim] - vel[:, :dim]
    nu = mu_g / rho_g
    if dim == 2:
        omega = vorticity[:, -1]
        omega_abs = np.abs(omega)
        cross = np.column_stack((slip[:, 1] * omega, -slip[:, 0] * omega))
    elif dim == 3:
        if vorticity.ndim != 2 or vorticity.shape[1] != 3:
            raise ValueError("3D lift requires vorticity with shape (particle, 3)")
        omega_abs = np.linalg.norm(vorticity, axis=1)
        cross = np.cross(slip, vorticity)
    else:
        raise ValueError("lift is implemented only for Cartesian 2D or 3D")
    contribution = np.zeros_like(cross)
    nonzero_vorticity = omega_abs > 0.0
    scale = (
        float(params.lift_coefficient)
        * mu_g[nonzero_vorticity]
        * radius[nonzero_vorticity]
        * radius[nonzero_vorticity]
        / np.sqrt(nu[nonzero_vorticity] * omega_abs[nonzero_vorticity])
        / mass[nonzero_vorticity]
    )
    contribution[nonzero_vorticity] = scale[:, None] * cross[nonzero_vorticity]
    out_accel[:, :dim] += contribution


ForceEvaluator = Callable[
    [np.ndarray, ForceBatchStatic, ForceBatchState, StageFields, ForcePipeline],
    None,
]

FORCE_EVALUATORS: Mapping[str, ForceEvaluator] = {
    "electric": add_electric_acceleration,
    "pressure_gradient": add_pressure_gradient_acceleration,
    "virtual_mass": add_virtual_mass_acceleration,
    "thermophoresis": add_thermophoresis_acceleration,
    "dielectrophoresis": add_dielectrophoresis_acceleration,
    "lift": add_lift_acceleration,
}


def evaluate_force_pipeline(
    out_accel: np.ndarray,
    static: ForceBatchStatic,
    state: ForceBatchState,
    fields: StageFields,
    plan: ForcePipeline,
) -> np.ndarray:
    unknown = sorted(
        {str(name) for name in plan.evaluator_names}.difference(FORCE_EVALUATORS)
    )
    if unknown:
        raise ValueError(f"unknown force evaluator(s): {', '.join(unknown)}")
    _validate_force_inputs(out_accel, static, state, fields, plan)
    for name in plan.evaluator_names:
        evaluator = FORCE_EVALUATORS[str(name)]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            evaluator(out_accel, static, state, fields, plan)
        invalid = ~np.isfinite(np.asarray(out_accel))
        if np.any(invalid):
            rows = _invalid_particle_rows(invalid)
            raise ValueError(
                f"{name} produced a non-finite acceleration; "
                f"invalid particle rows: {rows}"
            )
    return out_accel
