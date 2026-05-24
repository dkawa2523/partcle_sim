from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

from .forces.runtime import ForceRuntimeParameters


_EPS0_F_M = 8.8541878128e-12
_K_BOLTZMANN = 1.380649e-23


@dataclass(frozen=True)
class ForceBatchStatic:
    particle_diameter: np.ndarray
    particle_density: np.ndarray
    particle_mass: np.ndarray
    dep_particle_rel_permittivity: np.ndarray
    thermophoretic_coeff: np.ndarray


@dataclass(frozen=True)
class ForceBatchState:
    velocity: np.ndarray


@dataclass(frozen=True)
class ForceBatchSamples:
    electric_field: np.ndarray | None = None
    flow_velocity: np.ndarray | None = None
    gas_density: np.ndarray | None = None
    gas_mu: np.ndarray | None = None
    gas_temperature: np.ndarray | None = None
    grad_T: np.ndarray | None = None
    grad_E2: np.ndarray | None = None
    vorticity_z: np.ndarray | None = None
    fluid_acceleration: np.ndarray | None = None
    flow_time_derivative: np.ndarray | None = None
    flow_velocity_gradient: np.ndarray | None = None
    electric_q_over_m: np.ndarray | None = None
    gas_molecular_mass_kg: float = 60.0 * 1.66053906660e-27


@dataclass(frozen=True)
class ForcePipeline:
    evaluator_names: tuple[str, ...]
    params: ForceRuntimeParameters
    need_gas_properties: bool


def _cm_factor_real(
    particle_rel_permittivity: float,
    medium_rel_permittivity: float,
    particle_conductivity_Sm: float,
    medium_conductivity_Sm: float,
    frequency_Hz: float,
) -> float:
    eps_p = float(particle_rel_permittivity)
    eps_m = float(medium_rel_permittivity)
    if not np.isfinite(eps_p) or eps_p <= 0.0:
        eps_p = 2.0
    if not np.isfinite(eps_m) or eps_m <= 0.0:
        eps_m = 1.0006
    freq = max(float(frequency_Hz), 0.0)
    if freq <= 0.0:
        return float((eps_p - eps_m) / (eps_p + 2.0 * eps_m))
    omega = 2.0 * np.pi * freq
    rel_p = complex(eps_p, -float(particle_conductivity_Sm) / max(omega * _EPS0_F_M, 1.0e-300))
    rel_m = complex(eps_m, -float(medium_conductivity_Sm) / max(omega * _EPS0_F_M, 1.0e-300))
    value = (rel_p - rel_m) / (rel_p + 2.0 * rel_m)
    return float(value.real)


def _force_pipeline_from_names(
    params: ForceRuntimeParameters,
    names: Sequence[str],
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
        need_gas_properties=any(
            name in evaluator_names
            for name in ('pressure_gradient', 'virtual_mass', 'thermophoresis', 'dielectrophoresis', 'lift')
        ),
    )


def build_force_pipeline(
    params: ForceRuntimeParameters | None,
    *,
    include_electric: bool = False,
) -> ForcePipeline:
    p = params or ForceRuntimeParameters()
    names: list[str] = []
    if bool(include_electric):
        names.append('electric')
    if bool(p.pressure_gradient_enabled):
        names.append('pressure_gradient')
    if bool(p.virtual_mass_enabled):
        names.append('virtual_mass')
    if bool(p.thermophoresis_enabled):
        names.append('thermophoresis')
    if bool(p.dielectrophoresis_enabled):
        names.append('dielectrophoresis')
    if bool(p.lift_enabled):
        names.append('lift')
    return _force_pipeline_from_names(p, names)


def add_electric_acceleration(
    out_accel: np.ndarray,
    static: ForceBatchStatic,
    state: ForceBatchState,
    active_idx: np.ndarray | None,
    samples: ForceBatchSamples,
    plan: ForcePipeline,
    t: float,
) -> None:
    del static, state, active_idx, plan, t
    if samples.electric_field is None or samples.electric_q_over_m is None:
        return
    qom = np.asarray(samples.electric_q_over_m, dtype=np.float64).reshape(-1)
    electric = np.asarray(samples.electric_field, dtype=np.float64)
    out_accel[:, : electric.shape[1]] += qom[:, None] * electric


def add_pressure_gradient_acceleration(
    out_accel: np.ndarray,
    static: ForceBatchStatic,
    state: ForceBatchState,
    active_idx: np.ndarray | None,
    samples: ForceBatchSamples,
    plan: ForcePipeline,
    t: float,
) -> None:
    del state, active_idx, plan, t
    if samples.fluid_acceleration is None or samples.gas_density is None:
        return
    rho_p = np.asarray(static.particle_density, dtype=np.float64).reshape(-1)
    rho_g = np.asarray(samples.gas_density, dtype=np.float64).reshape(-1)
    fluid = np.asarray(samples.fluid_acceleration, dtype=np.float64)
    valid = (
        np.isfinite(rho_p)
        & (rho_p > 0.0)
        & np.isfinite(rho_g)
        & (rho_g > 0.0)
        & np.all(np.isfinite(fluid), axis=1)
    )
    scale = rho_g / np.maximum(rho_p, 1.0e-300)
    out_accel[:, : fluid.shape[1]] += np.where(valid[:, None], scale[:, None] * fluid, 0.0)


def add_virtual_mass_acceleration(
    out_accel: np.ndarray,
    static: ForceBatchStatic,
    state: ForceBatchState,
    active_idx: np.ndarray | None,
    samples: ForceBatchSamples,
    plan: ForcePipeline,
    t: float,
) -> None:
    del active_idx, t
    if (
        samples.flow_time_derivative is None
        or samples.flow_velocity_gradient is None
        or samples.gas_density is None
    ):
        return
    params = plan.params
    rho_p = np.asarray(static.particle_density, dtype=np.float64).reshape(-1)
    rho_g = np.asarray(samples.gas_density, dtype=np.float64).reshape(-1)
    dudt = np.asarray(samples.flow_time_derivative, dtype=np.float64)
    grad_u = np.asarray(samples.flow_velocity_gradient, dtype=np.float64)
    vel = np.asarray(state.velocity, dtype=np.float64)
    particle_accel = dudt + np.einsum('nij,nj->ni', grad_u, vel[:, : dudt.shape[1]])
    valid = (
        np.isfinite(rho_p)
        & (rho_p > 0.0)
        & np.isfinite(rho_g)
        & (rho_g > 0.0)
        & np.all(np.isfinite(particle_accel), axis=1)
    )
    scale = max(float(params.virtual_mass_coefficient), 0.0) * rho_g / np.maximum(rho_p, 1.0e-300)
    out_accel[:, : particle_accel.shape[1]] += np.where(valid[:, None], scale[:, None] * particle_accel, 0.0)


def add_thermophoresis_acceleration(
    out_accel: np.ndarray,
    static: ForceBatchStatic,
    state: ForceBatchState,
    active_idx: np.ndarray | None,
    samples: ForceBatchSamples,
    plan: ForcePipeline,
    t: float,
) -> None:
    del state, active_idx, t
    if (
        samples.grad_T is None
        or samples.gas_density is None
        or samples.gas_mu is None
        or samples.gas_temperature is None
    ):
        return
    params = plan.params
    d = np.asarray(static.particle_diameter, dtype=np.float64).reshape(-1)
    mass = np.asarray(static.particle_mass, dtype=np.float64).reshape(-1)
    radius = 0.5 * np.maximum(d, 0.0)
    finite_mass = np.isfinite(mass) & (mass > 0.0)
    rho_g = np.asarray(samples.gas_density, dtype=np.float64).reshape(-1)
    mu_g = np.asarray(samples.gas_mu, dtype=np.float64).reshape(-1)
    temp_g = np.asarray(samples.gas_temperature, dtype=np.float64).reshape(-1)
    grad_T = np.asarray(samples.grad_T, dtype=np.float64)
    mol_mass = max(float(samples.gas_molecular_mass_kg), 1.0e-30)
    mean_free_path = (mu_g / np.maximum(rho_g, 1.0e-30)) * np.sqrt(
        np.pi * mol_mass / (2.0 * _K_BOLTZMANN * np.maximum(temp_g, 1.0))
    )
    kn = mean_free_path / np.maximum(radius, 1.0e-30)
    if str(params.thermophoresis_model).lower() == 'continuum':
        kn = np.zeros_like(kn)
    ratio = max(float(params.gas_thermal_conductivity_W_mK), 1.0e-30) / max(
        float(params.particle_thermal_conductivity_W_mK),
        1.0e-30,
    )
    factor = (
        float(params.thermophoresis_Cs)
        * (ratio + float(params.thermophoresis_Ct) * kn)
        / np.maximum(
            (1.0 + 3.0 * float(params.thermophoresis_Cm) * kn)
            * (1.0 + 2.0 * ratio + 2.0 * float(params.thermophoresis_Ct) * kn),
            1.0e-30,
        )
    )
    multiplier = np.where(
        np.isfinite(static.thermophoretic_coeff) & (np.asarray(static.thermophoretic_coeff) > 0.0),
        np.asarray(static.thermophoretic_coeff, dtype=np.float64),
        1.0,
    )
    tau_stokes = mass / np.maximum(3.0 * np.pi * mu_g * np.maximum(d, 1.0e-30), 1.0e-300)
    scale = -multiplier * factor * mu_g / np.maximum(rho_g * temp_g * tau_stokes, 1.0e-300)
    valid = finite_mass & np.isfinite(scale)
    out_accel[:, : grad_T.shape[1]] += np.where(valid[:, None], scale[:, None] * grad_T, 0.0)


def add_dielectrophoresis_acceleration(
    out_accel: np.ndarray,
    static: ForceBatchStatic,
    state: ForceBatchState,
    active_idx: np.ndarray | None,
    samples: ForceBatchSamples,
    plan: ForcePipeline,
    t: float,
) -> None:
    del state, active_idx, t
    if samples.grad_E2 is None:
        return
    params = plan.params
    d = np.asarray(static.particle_diameter, dtype=np.float64).reshape(-1)
    mass = np.asarray(static.particle_mass, dtype=np.float64).reshape(-1)
    radius = 0.5 * np.maximum(d, 0.0)
    finite_mass = np.isfinite(mass) & (mass > 0.0)
    epsp_arr = np.asarray(static.dep_particle_rel_permittivity, dtype=np.float64).reshape(-1)
    grad_E2 = np.asarray(samples.grad_E2, dtype=np.float64)
    epsp = np.where(
        np.isfinite(epsp_arr) & (epsp_arr > 0.0),
        epsp_arr,
        float(params.dep_particle_rel_permittivity),
    )
    epsp = np.where(np.isfinite(epsp) & (epsp > 0.0), epsp, 2.0)
    epsm = max(float(params.dep_medium_rel_permittivity), 1.0e-30)
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
    coeff = 2.0 * np.pi * _EPS0_F_M * epsm * radius**3 * cm_real / np.maximum(mass, 1.0e-300)
    valid = finite_mass & np.isfinite(coeff)
    out_accel[:, : grad_E2.shape[1]] += np.where(valid[:, None], coeff[:, None] * grad_E2, 0.0)


def add_lift_acceleration(
    out_accel: np.ndarray,
    static: ForceBatchStatic,
    state: ForceBatchState,
    active_idx: np.ndarray | None,
    samples: ForceBatchSamples,
    plan: ForcePipeline,
    t: float,
) -> None:
    del active_idx, t
    if (
        samples.vorticity_z is None
        or samples.flow_velocity is None
        or samples.gas_density is None
        or samples.gas_mu is None
    ):
        return
    params = plan.params
    d = np.asarray(static.particle_diameter, dtype=np.float64).reshape(-1)
    mass = np.asarray(static.particle_mass, dtype=np.float64).reshape(-1)
    radius = 0.5 * np.maximum(d, 0.0)
    finite_mass = np.isfinite(mass) & (mass > 0.0)
    rho_g = np.asarray(samples.gas_density, dtype=np.float64).reshape(-1)
    mu_g = np.asarray(samples.gas_mu, dtype=np.float64).reshape(-1)
    flow = np.asarray(samples.flow_velocity, dtype=np.float64)
    vel = np.asarray(state.velocity, dtype=np.float64)
    omega = np.asarray(samples.vorticity_z, dtype=np.float64).reshape(-1)
    omega_abs = np.abs(omega)
    slip = vel[:, :2] - flow[:, :2]
    nu = mu_g / np.maximum(rho_g, 1.0e-30)
    scale = (
        float(params.lift_coefficient)
        * mu_g
        * radius
        * radius
        / np.maximum(np.sqrt(nu * omega_abs), 1.0e-300)
        / np.maximum(mass, 1.0e-300)
    )
    valid = finite_mass & np.isfinite(scale) & (omega_abs > 1.0e-30)
    out_accel[:, 0] += np.where(valid, scale * slip[:, 1] * omega, 0.0)
    out_accel[:, 1] += np.where(valid, -scale * slip[:, 0] * omega, 0.0)


FORCE_EVALUATORS: Mapping[
    str,
    Callable[[np.ndarray, ForceBatchStatic, ForceBatchState, np.ndarray | None, ForceBatchSamples, ForcePipeline, float], None],
] = {
    'electric': add_electric_acceleration,
    'pressure_gradient': add_pressure_gradient_acceleration,
    'virtual_mass': add_virtual_mass_acceleration,
    'thermophoresis': add_thermophoresis_acceleration,
    'dielectrophoresis': add_dielectrophoresis_acceleration,
    'lift': add_lift_acceleration,
}


def evaluate_force_pipeline(
    out_accel: np.ndarray,
    static: ForceBatchStatic,
    state: ForceBatchState,
    active_idx: np.ndarray | None,
    samples: ForceBatchSamples,
    plan: ForcePipeline,
    t: float,
) -> np.ndarray:
    for name in plan.evaluator_names:
        evaluator = FORCE_EVALUATORS.get(str(name))
        if evaluator is None:
            raise ValueError(f"unknown force evaluator '{name}'")
        evaluator(out_accel, static, state, active_idx, samples, plan, float(t))
    return out_accel


__all__ = (
    'FORCE_EVALUATORS',
    'ForceBatchSamples',
    'ForceBatchState',
    'ForceBatchStatic',
    'ForcePipeline',
    'build_force_pipeline',
    'evaluate_force_pipeline',
)
