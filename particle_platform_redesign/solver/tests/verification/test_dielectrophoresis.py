from __future__ import annotations

import math

import numpy as np
import pytest

from chamber_particles.physics.catalog import resolve_physics_plan
from chamber_particles.physics.forces import (
    PhysicsEvaluationError,
    quasistatic_spherical_dep_acceleration,
    quasistatic_spherical_dep_acceleration_abs_upper,
)
from chamber_particles.physics.runtime import PrimitiveRange, prepare_physics_runtime

_REVISION = "quasistatic_spherical_gradient_e2_v1"
_EPSILON_0_F_M = 8.8541878128e-12
_RELATIVE_PERMITTIVITY = 2.5
_CM_FACTOR = 0.4


def test_dep_formula_zero_sign_and_particle_scaling() -> None:
    mass = np.asarray([2.0e-15, 2.0e-15, 4.0e-15])
    radius = np.asarray([1.0e-7, 2.0e-7, 1.0e-7])
    gradient = np.repeat(np.asarray([[3.0e12, -4.0e12]]), 3, axis=0)
    actual = quasistatic_spherical_dep_acceleration(
        mass_kg=mass,
        electrostatic_radius_m=radius,
        gradient_mean_e_squared_V2_m3=gradient,
        medium_relative_permittivity=_RELATIVE_PERMITTIVITY,
        real_clausius_mossotti_factor=_CM_FACTOR,
    )
    expected = np.asarray(
        [
            2.0
            * math.pi
            * _EPSILON_0_F_M
            * _RELATIVE_PERMITTIVITY
            * _CM_FACTOR
            * float(radius[row]) ** 3
            / float(mass[row])
            * gradient[row]
            for row in range(3)
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=3.0e-15, atol=0.0)
    np.testing.assert_allclose(actual[1], 8.0 * actual[0], rtol=3.0e-15, atol=0.0)
    np.testing.assert_allclose(actual[2], 0.5 * actual[0], rtol=3.0e-15, atol=0.0)

    negative = quasistatic_spherical_dep_acceleration(
        mass_kg=mass[:1],
        electrostatic_radius_m=radius[:1],
        gradient_mean_e_squared_V2_m3=gradient[:1],
        medium_relative_permittivity=_RELATIVE_PERMITTIVITY,
        real_clausius_mossotti_factor=-_CM_FACTOR,
    )
    np.testing.assert_allclose(negative, -actual[:1], rtol=0.0, atol=0.0)

    for factor, zero_gradient in ((_CM_FACTOR, True), (0.0, False)):
        zero = quasistatic_spherical_dep_acceleration(
            mass_kg=mass[:1],
            electrostatic_radius_m=radius[:1],
            gradient_mean_e_squared_V2_m3=(np.zeros((1, 2)) if zero_gradient else gradient[:1]),
            medium_relative_permittivity=_RELATIVE_PERMITTIVITY,
            real_clausius_mossotti_factor=factor,
        )
        np.testing.assert_array_equal(zero, np.zeros((1, 2)))

    angle = 0.37
    rotation = np.asarray([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    rotated = quasistatic_spherical_dep_acceleration(
        mass_kg=mass[:1],
        electrostatic_radius_m=radius[:1],
        gradient_mean_e_squared_V2_m3=gradient[:1] @ rotation.T,
        medium_relative_permittivity=_RELATIVE_PERMITTIVITY,
        real_clausius_mossotti_factor=_CM_FACTOR,
    )
    np.testing.assert_allclose(rotated, actual[:1] @ rotation.T, rtol=3.0e-15, atol=0.0)


def test_dep_component_bound_contains_the_declared_gradient_range() -> None:
    mass = np.asarray([1.5e-15, 2.5e-15])
    radius = np.asarray([4.0e-8, 7.0e-8])
    gradient_abs_upper = np.asarray([5.0e12, 7.0e12])
    relative_permittivity = 3.0
    cm_factor = -0.35
    bound = quasistatic_spherical_dep_acceleration_abs_upper(
        mass_kg=mass,
        electrostatic_radius_m=radius,
        gradient_mean_e_squared_abs_upper_V2_m3=gradient_abs_upper,
        medium_relative_permittivity=relative_permittivity,
        real_clausius_mossotti_factor=cm_factor,
    )
    expected = (
        2.0 * math.pi * _EPSILON_0_F_M * relative_permittivity * abs(cm_factor) * radius**3 / mass
    )[:, None] * gradient_abs_upper[None, :]
    assert bool((bound >= expected).all())
    np.testing.assert_allclose(bound, expected, rtol=2.0e-14, atol=0.0)

    rng = np.random.default_rng(20261001)
    for _ in range(64):
        gradient = rng.uniform(-gradient_abs_upper, gradient_abs_upper, size=(2, 2))
        acceleration = quasistatic_spherical_dep_acceleration(
            mass_kg=mass,
            electrostatic_radius_m=radius,
            gradient_mean_e_squared_V2_m3=gradient,
            medium_relative_permittivity=relative_permittivity,
            real_clausius_mossotti_factor=cm_factor,
        )
        assert bool((np.abs(acceleration) <= bound).all())


@pytest.mark.parametrize(
    ("coordinate_system", "components", "basis"),
    [
        ("cartesian_xy", ("x", "y"), "cartesian_xy"),
        ("axisymmetric_rz", ("r", "z"), "axisymmetric_rz"),
    ],
)
def test_dep_catalog_and_compiled_runtime_share_one_formula(
    coordinate_system: str,
    components: tuple[str, str],
    basis: str,
) -> None:
    plan = resolve_physics_plan(
        {"charge": {"model": "fixed"}, "dielectrophoresis": _model()},
        coordinate_system,  # type: ignore[arg-type]
    )
    requirements = {item.name: item for item in plan.required_fields}
    assert set(requirements) == {"grad_e2"}
    assert requirements["grad_e2"].components == components
    assert requirements["grad_e2"].stored_basis == basis
    assert requirements["grad_e2"].unit == "V^2/m^3"
    assert plan.resolved_models()["dielectrophoresis"] == {
        "model": "quasistatic_spherical",
        "revision": _REVISION,
    }

    mass = np.asarray([2.0e-15])
    radius = np.asarray([1.0e-7])
    gradient = np.asarray([[3.0e12, -4.0e12]])
    runtime = prepare_physics_runtime(
        plan=plan,
        coordinate_system=coordinate_system,  # type: ignore[arg-type]
        mass_kg=mass,
        drag_diameter_m=2.0 * radius,
        electrostatic_radius_m=radius,
        displaced_volume_m3=np.asarray([0.0]),
        charge_number=np.asarray([0.0]),
        primitive_ranges={"grad_e2": _constant_vector_range(3.0e12, -4.0e12)},
    )
    actual = runtime.evaluate(
        np.asarray([0], dtype=np.int64),
        np.zeros((1, 2)),
        np.asarray([0.0]),
        {"grad_e2": gradient},
    )
    expected = quasistatic_spherical_dep_acceleration(
        mass_kg=mass,
        electrostatic_radius_m=radius,
        gradient_mean_e_squared_V2_m3=gradient,
        medium_relative_permittivity=_RELATIVE_PERMITTIVITY,
        real_clausius_mossotti_factor=_CM_FACTOR,
    )
    if coordinate_system == "cartesian_xy":
        assert runtime.constant_acceleration_m_s2 is not None
        np.testing.assert_allclose(
            runtime.constant_acceleration_m_s2,
            expected,
            rtol=3.0e-15,
            atol=0.0,
        )
    else:
        assert runtime.constant_acceleration_m_s2 is None
    np.testing.assert_allclose(actual.acceleration_m_s2, expected, rtol=3.0e-15, atol=0.0)
    np.testing.assert_array_equal(actual.additive_acceleration_m_s2, actual.acceleration_m_s2)
    np.testing.assert_array_equal(actual.linear_drag_rate_s_inv, np.zeros(1))
    assert runtime.continuous_applicability(
        np.asarray([0], dtype=np.int64),
        np.zeros((1, 2)),
    ).tolist() == [True]


def test_dep_prepare_accepts_one_ulp_outward_rounding_at_certified_cap() -> None:
    cap_m = 2.0e-7
    plan = resolve_physics_plan(
        {"charge": {"model": "fixed"}, "dielectrophoresis": _model()},
        "cartesian_xy",
    )
    runtime = prepare_physics_runtime(
        plan=plan,
        coordinate_system="cartesian_xy",
        mass_kg=np.asarray([2.0e-15]),
        drag_diameter_m=np.asarray([4.0e-7]),
        electrostatic_radius_m=np.asarray([np.nextafter(cap_m, np.inf)]),
        displaced_volume_m3=np.asarray([0.0]),
        charge_number=np.asarray([0.0]),
        primitive_ranges={"grad_e2": _constant_vector_range(3.0e12, -4.0e12)},
    )

    assert runtime.constant_acceleration_m_s2 is not None


@pytest.mark.parametrize(
    "radius_m",
    [
        np.nextafter(np.nextafter(2.0e-7, np.inf), np.inf),
        2.1e-7,
    ],
)
def test_dep_prepare_rejects_radius_above_the_producer_certified_cap(
    radius_m: float,
) -> None:
    plan = resolve_physics_plan(
        {"charge": {"model": "fixed"}, "dielectrophoresis": _model()},
        "cartesian_xy",
    )
    with pytest.raises(
        PhysicsEvaluationError,
        match="exceeds maximum_point_dipole_radius_m",
    ):
        prepare_physics_runtime(
            plan=plan,
            coordinate_system="cartesian_xy",
            mass_kg=np.asarray([2.0e-15]),
            drag_diameter_m=np.asarray([4.2e-7]),
            electrostatic_radius_m=np.asarray([radius_m]),
            displaced_volume_m3=np.asarray([0.0]),
            charge_number=np.asarray([0.0]),
            primitive_ranges={"grad_e2": _constant_vector_range(3.0e12, -4.0e12)},
        )


def _model() -> dict[str, object]:
    return {
        "model": "quasistatic_spherical",
        "revision": _REVISION,
        "gradient_mean_e_squared_field": "grad_e2",
        "medium_relative_permittivity": _RELATIVE_PERMITTIVITY,
        "real_clausius_mossotti_factor": _CM_FACTOR,
        "maximum_point_dipole_radius_m": 2.0e-7,
    }


def _constant_vector_range(first: float, second: float) -> PrimitiveRange:
    values = np.asarray([first, second])
    return PrimitiveRange(values, values, values)
