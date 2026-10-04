from __future__ import annotations

import numpy as np

from particle_tracer_unified.domain import StageFields
from particle_tracer_unified.solvers._force_evaluators import (
    add_lift_acceleration,
    add_thermophoresis_acceleration,
)
from particle_tracer_unified.solvers.force_runtime import (
    ForceBatchState,
    ForceBatchStatic,
    ForcePipeline,
)
from particle_tracer_unified.solvers.forces.runtime import ForceRuntimeParameters


def _static(*, diameter_m: float, mass_kg: float) -> ForceBatchStatic:
    return ForceBatchStatic(
        particle_diameter=np.asarray([diameter_m], dtype=np.float64),
        particle_density=np.asarray([1_000.0], dtype=np.float64),
        particle_mass=np.asarray([mass_kg], dtype=np.float64),
        dep_particle_rel_permittivity=np.asarray([np.nan], dtype=np.float64),
        thermophoretic_coeff=np.asarray([np.nan], dtype=np.float64),
    )


def test_continuum_thermophoresis_matches_comsol_reference_coefficient() -> None:
    diameter = 2.0e-6
    mass = 3.0e-15
    gas_density = 0.8
    viscosity = 1.7e-5
    temperature = 420.0
    gradient = np.asarray([[125.0, -40.0]], dtype=np.float64)
    params = ForceRuntimeParameters(
        thermophoresis_enabled=True,
        thermophoresis_model="continuum",
        gas_thermal_conductivity_W_mK=0.031,
        particle_thermal_conductivity_W_mK=1.7,
        thermophoresis_Cs=1.17,
    )
    fields = StageFields(
        points_m=np.zeros((1, 2), dtype=np.float64),
        time_s=0.0,
        values={
            "gas_density": np.asarray([gas_density]),
            "dynamic_viscosity": np.asarray([viscosity]),
            "temperature": np.asarray([temperature]),
            "temperature_gradient": gradient,
        },
        supported=np.ones(1, dtype=bool),
    )
    result = np.zeros((1, 2), dtype=np.float64)

    add_thermophoresis_acceleration(
        result,
        _static(diameter_m=diameter, mass_kg=mass),
        ForceBatchState(velocity=np.zeros((1, 2))),
        fields,
        ForcePipeline(evaluator_names=("thermophoresis",), params=params),
    )

    conductivity_ratio = (
        params.gas_thermal_conductivity_W_mK / params.particle_thermal_conductivity_W_mK
    )
    force_scale = (
        -6.0
        * np.pi
        * diameter
        * viscosity**2
        * params.thermophoresis_Cs
        * conductivity_ratio
        / (gas_density * (2.0 * conductivity_ratio + 1.0) * temperature)
    )
    np.testing.assert_allclose(
        result[0],
        force_scale * gradient[0] / mass,
        rtol=2.0e-15,
        atol=0.0,
    )


def test_saffman_lift_uses_fluid_minus_particle_slip_direction() -> None:
    diameter = 2.0e-6
    mass = 3.0e-15
    viscosity = 1.8e-5
    gas_density = 1.2
    coefficient = 6.46
    flow = np.asarray([[2.0, 0.0]], dtype=np.float64)
    particle_velocity = np.asarray([[1.0, 0.0]], dtype=np.float64)
    vorticity = np.asarray([[0.0, 0.0, -4.0]], dtype=np.float64)
    fields = StageFields(
        points_m=np.zeros((1, 2), dtype=np.float64),
        time_s=0.0,
        values={
            "gas_density": np.asarray([gas_density]),
            "dynamic_viscosity": np.asarray([viscosity]),
            "flow_velocity": flow,
            "vorticity": vorticity,
        },
        supported=np.ones(1, dtype=bool),
    )
    params = ForceRuntimeParameters(lift_enabled=True, lift_coefficient=coefficient)
    result = np.zeros((1, 2), dtype=np.float64)

    add_lift_acceleration(
        result,
        _static(diameter_m=diameter, mass_kg=mass),
        ForceBatchState(velocity=particle_velocity),
        fields,
        ForcePipeline(evaluator_names=("lift",), params=params),
    )

    radius = 0.5 * diameter
    omega = vorticity[0, -1]
    nu = viscosity / gas_density
    expected_scale = (
        coefficient * viscosity * radius**2 / np.sqrt(nu * abs(omega)) / mass
    )
    expected_cross = np.asarray(
        [
            (flow[0, 1] - particle_velocity[0, 1]) * omega,
            -(flow[0, 0] - particle_velocity[0, 0]) * omega,
        ]
    )
    np.testing.assert_allclose(
        result[0],
        expected_scale * expected_cross,
        rtol=2.0e-15,
        atol=0.0,
    )
