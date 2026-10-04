from __future__ import annotations

from decimal import Decimal, getcontext

import numpy as np
import pytest

from chamber_particles.physics.forces import (
    electric_field_directed_image_ion_drag_global_bound,
    electric_field_directed_image_orbital_ion_drag,
    relative_flow_screened_collection_orbital_ion_drag,
    relative_flow_screened_continuous_applicability_batch,
    relative_flow_screened_ion_drag_global_bound,
)

_E = Decimal("1.602176634e-19")
_EPSILON_0 = Decimal("8.8541878128e-12")
_PI = Decimal("3.1415926535897932384626433832795028841971693993751")


def _decimal(value: float) -> Decimal:
    return Decimal(str(value))


def _relative_flow_oracle(
    *,
    mass: float,
    radius: float,
    charge: float,
    particle_velocity: tuple[float, float],
    density: float,
    ion_voltage: float,
    ion_velocity: tuple[float, float],
    ion_mass: float,
    screening_length: float,
    mean_free_path: float,
) -> tuple[tuple[float, float], float, float]:
    """Independent scalar transcription of the versioned relative-flow equation."""

    getcontext().prec = 70
    particle_mass = _decimal(mass)
    particle_radius = _decimal(radius)
    charge_number = _decimal(charge)
    ion_density = _decimal(density)
    thermal_voltage = _decimal(ion_voltage)
    effective_mass = _decimal(ion_mass)
    relative = (
        _decimal(ion_velocity[0]) - _decimal(particle_velocity[0]),
        _decimal(ion_velocity[1]) - _decimal(particle_velocity[1]),
    )
    speed_square = (
        relative[0] ** 2
        + relative[1] ** 2
        + Decimal(8) * _E * thermal_voltage / (_PI * effective_mass)
        + Decimal(1)
    )
    phi_one = _E / (
        Decimal(4)
        * _PI
        * _EPSILON_0
        * particle_radius
        * (Decimal(1) + particle_radius / max(particle_radius, _decimal(screening_length)))
    )
    screening = max(
        particle_radius,
        min(_decimal(screening_length), _decimal(mean_free_path)),
    )
    impact = (
        (charge_number**2 + Decimal("1e-20")).sqrt()
        * _E**2
        / (Decimal(4) * _PI * _EPSILON_0 * effective_mass * speed_square)
    )
    collection_square = min(
        screening**2,
        particle_radius**2
        * max(
            Decimal(0),
            Decimal(1)
            - Decimal(2) * _E * charge_number * phi_one / (effective_mass * speed_square),
        ),
    )
    logarithm = max(
        Decimal(0),
        Decimal("0.5") * ((screening**2 + impact**2) / (collection_square + impact**2)).ln(),
    )
    collection = _PI * collection_square
    orbital = Decimal(4) * _PI * impact**2 * logarithm
    factor = ion_density * effective_mass * speed_square.sqrt() * (collection + orbital)
    acceleration = tuple(float(factor * component / particle_mass) for component in relative)
    return acceleration, float(collection), float(orbital)


def _image_oracle(
    *,
    mass: float,
    radius: float,
    charge: float,
    density: float,
    electron_voltage: float,
    ion_voltage: float,
    ion_velocity: tuple[float, float],
    ion_mass: float,
    screening_length: float,
    electric_field: tuple[float, float],
) -> tuple[float, float]:
    """Independent scalar transcription of the producer-independent image equation."""

    getcontext().prec = 70
    particle_mass = _decimal(mass)
    particle_radius = _decimal(radius)
    charge_number = _decimal(charge)
    ion_density = _decimal(density)
    electron_thermal_voltage = _decimal(electron_voltage)
    ion_thermal_voltage = _decimal(ion_voltage)
    effective_mass = _decimal(ion_mass)
    ion_speed = (_decimal(ion_velocity[0]) ** 2 + _decimal(ion_velocity[1]) ** 2).sqrt()
    speed_square = (
        ion_speed**2 + Decimal(8) * _E * ion_thermal_voltage / (_PI * effective_mass) + Decimal(1)
    )
    phi_one = _E / (
        Decimal(4)
        * _PI
        * _EPSILON_0
        * particle_radius
        * (Decimal(1) + particle_radius / max(particle_radius, _decimal(screening_length)))
    )
    collection = (
        _PI
        * particle_radius**2
        * max(Decimal(0), Decimal(1) - charge_number * phi_one / ion_thermal_voltage)
    )
    image_impact = (
        _E**2 * charge_number / (Decimal(2) * _PI * _EPSILON_0 * effective_mass * speed_square)
    )
    image_screening = (_EPSILON_0 * electron_thermal_voltage / (_E * ion_density)).sqrt()
    logarithm = max(
        Decimal(1) + Decimal("1e-12"),
        image_screening / particle_radius,
    ).ln()
    orbital = _PI * image_impact**2 * logarithm
    magnitude = (
        ion_density * effective_mass * speed_square.sqrt() * ion_speed * (collection + orbital)
    )
    electric = (_decimal(electric_field[0]), _decimal(electric_field[1]))
    denominator = (electric[0] ** 2 + electric[1] ** 2 + Decimal(1)).sqrt()
    return tuple(
        float(magnitude * component / denominator / particle_mass) for component in electric
    )


def test_relative_flow_revision_matches_decimal_oracle_across_clamps_and_signs() -> None:
    charges = np.asarray([-180.0, 0.0, 80.0, -30.0])
    velocity = np.asarray([[15.0, -8.0], [30.0, 4.0], [-20.0, 10.0], [5.0, 5.0]])
    ion_velocity = np.asarray([[120.0, 25.0], [30.0, 4.0], [50.0, -40.0], [12.0, 9.0]])
    screening = np.asarray([2.0e-5, 5.0e-8, 8.0e-6, 4.0e-5])
    mean_free_path = np.asarray([7.0e-6, 2.0e-5, 1.0e-8, 9.0e-5])
    ion_mass = np.asarray([7.0e-26, 8.0e-26, 6.0e-26, 9.0e-26])
    radius = np.asarray([5.0e-8, 5.0e-8, 5.0e-8, 5.0e-8])
    mass = np.asarray([2.0e-18, 2.0e-18, 2.0e-18, 2.0e-18])
    density = np.asarray([8.0e14, 9.0e14, 1.1e15, 7.0e14])
    ion_voltage = np.asarray([0.03, 0.02, 0.08, 0.05])

    actual = relative_flow_screened_collection_orbital_ion_drag(
        mass_kg=mass,
        electrostatic_radius_m=radius,
        charge_number=charges,
        velocity_m_s=velocity,
        positive_ion_number_density_m3=density,
        positive_ion_thermal_voltage_V=ion_voltage,
        positive_ion_velocity_m_s=ion_velocity,
        effective_positive_ion_mass_kg=ion_mass,
        screening_length_m=screening,
        ion_neutral_mean_free_path_m=mean_free_path,
        maximum_relative_ion_speed_m_s=500.0,
    )
    expected = [
        _relative_flow_oracle(
            mass=float(mass[row]),
            radius=float(radius[row]),
            charge=float(charges[row]),
            particle_velocity=tuple(velocity[row]),
            density=float(density[row]),
            ion_voltage=float(ion_voltage[row]),
            ion_velocity=tuple(ion_velocity[row]),
            ion_mass=float(ion_mass[row]),
            screening_length=float(screening[row]),
            mean_free_path=float(mean_free_path[row]),
        )
        for row in range(charges.size)
    ]
    np.testing.assert_allclose(
        actual.acceleration_m_s2,
        np.asarray([item[0] for item in expected]),
        rtol=2.0e-14,
        atol=0.0,
    )
    np.testing.assert_allclose(
        actual.collection_cross_section_m2,
        np.asarray([item[1] for item in expected]),
        rtol=4.0e-15,
    )
    np.testing.assert_allclose(
        actual.orbital_cross_section_m2,
        np.asarray([item[2] for item in expected]),
        rtol=2.0e-13,
        atol=1.0e-300,
    )
    np.testing.assert_array_equal(actual.acceleration_m_s2[1], np.zeros(2))
    assert bool(actual.applicable.all())


def test_image_revision_matches_decimal_oracle_and_has_explicit_zero_limits() -> None:
    mass = np.full(3, 2.0e-18)
    radius = np.full(3, 5.0e-8)
    charge = np.asarray([-180.0, 50.0, -20.0])
    density = np.full(3, 9.0e14)
    electron_voltage = np.full(3, 3.5)
    ion_voltage = np.full(3, 0.04)
    ion_velocity = np.asarray([[120.0, -40.0], [0.0, 0.0], [30.0, 10.0]])
    ion_mass = np.full(3, 7.5e-26)
    screening = np.full(3, 2.0e-5)
    electric = np.asarray([[40.0, -30.0], [40.0, 5.0], [0.0, 0.0]])
    actual = electric_field_directed_image_orbital_ion_drag(
        mass_kg=mass,
        electrostatic_radius_m=radius,
        charge_number=charge,
        positive_ion_number_density_m3=density,
        electron_thermal_voltage_V=electron_voltage,
        positive_ion_thermal_voltage_V=ion_voltage,
        positive_ion_velocity_m_s=ion_velocity,
        effective_positive_ion_mass_kg=ion_mass,
        screening_length_m=screening,
        electric_field_V_m=electric,
    )
    expected = _image_oracle(
        mass=mass[0],
        radius=radius[0],
        charge=charge[0],
        density=density[0],
        electron_voltage=electron_voltage[0],
        ion_voltage=ion_voltage[0],
        ion_velocity=tuple(ion_velocity[0]),
        ion_mass=ion_mass[0],
        screening_length=screening[0],
        electric_field=tuple(electric[0]),
    )
    np.testing.assert_allclose(actual.acceleration_m_s2[0], expected, rtol=4.0e-15)
    np.testing.assert_array_equal(actual.acceleration_m_s2[1], np.zeros(2))
    np.testing.assert_array_equal(actual.acceleration_m_s2[2], np.zeros(2))
    assert np.dot(actual.acceleration_m_s2[0], electric[0]) > 0.0
    assert bool(actual.applicable.all())


def test_revision_global_bounds_and_relative_path_gate_are_conservative() -> None:
    mass = np.asarray([1.5e-18, 2.5e-18])
    radius = np.asarray([4.0e-8, 7.0e-8])
    charge_lower = np.asarray([-250.0, -80.0])
    charge_upper = np.asarray([100.0, 40.0])
    relative_bound = relative_flow_screened_ion_drag_global_bound(
        mass_kg=mass,
        electrostatic_radius_m=radius,
        charge_number_lower=charge_lower,
        charge_number_upper=charge_upper,
        positive_ion_number_density_upper_m3=1.2e15,
        positive_ion_thermal_voltage_lower_V=0.02,
        positive_ion_thermal_voltage_upper_V=0.08,
        effective_positive_ion_mass_lower_kg=6.0e-26,
        effective_positive_ion_mass_upper_kg=9.0e-26,
        screening_length_upper_m=4.0e-5,
        ion_neutral_mean_free_path_upper_m=7.0e-5,
        maximum_relative_ion_speed_m_s=300.0,
    )
    image_bound = electric_field_directed_image_ion_drag_global_bound(
        mass_kg=mass,
        electrostatic_radius_m=radius,
        charge_number_lower=charge_lower,
        charge_number_upper=charge_upper,
        positive_ion_number_density_lower_m3=7.0e14,
        positive_ion_number_density_upper_m3=1.2e15,
        electron_thermal_voltage_upper_V=4.0,
        positive_ion_thermal_voltage_lower_V=0.02,
        positive_ion_thermal_voltage_upper_V=0.08,
        positive_ion_velocity_abs_upper_m_s=np.asarray([180.0, 120.0]),
        effective_positive_ion_mass_lower_kg=6.0e-26,
        effective_positive_ion_mass_upper_kg=9.0e-26,
    )

    rng = np.random.default_rng(20261001)
    for _ in range(64):
        ion_velocity = rng.uniform([-180.0, -120.0], [180.0, 120.0], size=(2, 2))
        relative_direction = rng.normal(size=(2, 2))
        relative_direction /= np.linalg.norm(relative_direction, axis=1)[:, None]
        relative_speed = rng.uniform(0.0, 300.0, size=2)
        common = {
            "mass_kg": mass,
            "electrostatic_radius_m": radius,
            "charge_number": rng.uniform(charge_lower, charge_upper),
            "positive_ion_number_density_m3": rng.uniform(7.0e14, 1.2e15, size=2),
            "positive_ion_thermal_voltage_V": rng.uniform(0.02, 0.08, size=2),
            "positive_ion_velocity_m_s": ion_velocity,
            "effective_positive_ion_mass_kg": rng.uniform(6.0e-26, 9.0e-26, size=2),
            "screening_length_m": rng.uniform(8.0e-6, 4.0e-5, size=2),
        }
        relative = relative_flow_screened_collection_orbital_ion_drag(
            **common,
            velocity_m_s=ion_velocity - relative_direction * relative_speed[:, None],
            ion_neutral_mean_free_path_m=rng.uniform(9.0e-6, 7.0e-5, size=2),
            maximum_relative_ion_speed_m_s=300.0,
        )
        image = electric_field_directed_image_orbital_ion_drag(
            **common,
            electron_thermal_voltage_V=rng.uniform(2.0, 4.0, size=2),
            electric_field_V_m=rng.uniform(-500.0, 500.0, size=(2, 2)),
        )
        assert bool((np.abs(relative.acceleration_m_s2) <= relative_bound).all())
        assert bool((np.abs(image.acceleration_m_s2) <= image_bound).all())

    applicable, status = relative_flow_screened_continuous_applicability_batch(
        velocity_abs_upper_m_s=np.asarray([[20.0, 10.0], [500.0, 500.0]]),
        positive_ion_velocity_abs_upper_m_s=np.asarray([180.0, 120.0]),
        maximum_relative_ion_speed_m_s=300.0,
    )
    assert applicable.tolist() == [True, False]
    assert status.tolist() == [0, 0]


def test_image_formula_uses_vector_norm_not_saved_case_specific_speed_floor() -> None:
    result = electric_field_directed_image_orbital_ion_drag(
        mass_kg=np.asarray([1.0]),
        electrostatic_radius_m=np.asarray([1.0e-7]),
        charge_number=np.asarray([-1.0]),
        positive_ion_number_density_m3=np.asarray([1.0e14]),
        electron_thermal_voltage_V=np.asarray([2.0]),
        positive_ion_thermal_voltage_V=np.asarray([0.03]),
        positive_ion_velocity_m_s=np.asarray([[0.0, 0.0]]),
        effective_positive_ion_mass_kg=np.asarray([7.0e-26]),
        screening_length_m=np.asarray([1.0e-5]),
        electric_field_V_m=np.asarray([[100.0, 0.0]]),
    )
    assert result.acceleration_m_s2[0, 0] == pytest.approx(0.0, abs=0.0)
    assert result.acceleration_m_s2[0, 1] == pytest.approx(0.0, abs=0.0)


def test_relative_flow_revision_is_galilean_and_rotation_covariant() -> None:
    particle_velocity = np.asarray([[13.0, -7.0]])
    ion_velocity = np.asarray([[80.0, 25.0]])
    common = {
        "mass_kg": np.asarray([2.0e-18]),
        "electrostatic_radius_m": np.asarray([5.0e-8]),
        "charge_number": np.asarray([-180.0]),
        "positive_ion_number_density_m3": np.asarray([8.0e14]),
        "positive_ion_thermal_voltage_V": np.asarray([0.03]),
        "effective_positive_ion_mass_kg": np.asarray([7.0e-26]),
        "screening_length_m": np.asarray([2.0e-5]),
        "ion_neutral_mean_free_path_m": np.asarray([7.0e-6]),
        "maximum_relative_ion_speed_m_s": 500.0,
    }
    reference = relative_flow_screened_collection_orbital_ion_drag(
        **common,
        velocity_m_s=particle_velocity,
        positive_ion_velocity_m_s=ion_velocity,
    )
    shift = np.asarray([[140.0, -35.0]])
    shifted = relative_flow_screened_collection_orbital_ion_drag(
        **common,
        velocity_m_s=particle_velocity + shift,
        positive_ion_velocity_m_s=ion_velocity + shift,
    )
    rotation = np.asarray([[0.0, -1.0], [1.0, 0.0]])
    rotated = relative_flow_screened_collection_orbital_ion_drag(
        **common,
        velocity_m_s=particle_velocity @ rotation.T,
        positive_ion_velocity_m_s=ion_velocity @ rotation.T,
    )

    np.testing.assert_allclose(
        shifted.acceleration_m_s2,
        reference.acceleration_m_s2,
        rtol=2.0e-15,
        atol=0.0,
    )
    np.testing.assert_allclose(
        rotated.acceleration_m_s2,
        reference.acceleration_m_s2 @ rotation.T,
        rtol=2.0e-15,
        atol=0.0,
    )


def test_image_revision_applies_its_log_argument_floor() -> None:
    parameters = {
        "mass": 2.0e-18,
        "radius": 1.0e-3,
        "charge": -180.0,
        "density": 1.0e20,
        "electron_voltage": 1.0e-3,
        "ion_voltage": 0.03,
        "ion_velocity": (20.0, 4.0),
        "ion_mass": 7.0e-26,
        "screening_length": 2.0e-5,
        "electric_field": (40.0, -30.0),
    }
    actual = electric_field_directed_image_orbital_ion_drag(
        mass_kg=np.asarray([parameters["mass"]]),
        electrostatic_radius_m=np.asarray([parameters["radius"]]),
        charge_number=np.asarray([parameters["charge"]]),
        positive_ion_number_density_m3=np.asarray([parameters["density"]]),
        electron_thermal_voltage_V=np.asarray([parameters["electron_voltage"]]),
        positive_ion_thermal_voltage_V=np.asarray([parameters["ion_voltage"]]),
        positive_ion_velocity_m_s=np.asarray([parameters["ion_velocity"]]),
        effective_positive_ion_mass_kg=np.asarray([parameters["ion_mass"]]),
        screening_length_m=np.asarray([parameters["screening_length"]]),
        electric_field_V_m=np.asarray([parameters["electric_field"]]),
    )
    expected = _image_oracle(**parameters)

    np.testing.assert_allclose(actual.acceleration_m_s2[0], expected, rtol=5.0e-15)
    assert bool(actual.applicable[0])
