from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from tools.vv.comsol.evaluate_neutral_transport_closure import (
    AVOGADRO_PER_MOL,
    BOLTZMANN_J_K,
    STATUS_NOT_TESTED,
    finite_speed_rate_coefficient,
    linear_epstein_force,
    load_config,
    require_new_output_path,
    trajectory_thermophoresis_replay_status,
    validate_case_identity,
    waldmann_gradient_force,
    waldmann_heat_flux_force,
)


def test_linear_epstein_formula_uses_relative_velocity_and_molar_mass() -> None:
    radius = np.asarray([2.0e-8])
    density = np.asarray([0.012])
    temperature = np.asarray([330.0])
    relative_velocity = np.asarray([[4.0, -3.0]])
    molar_mass = 0.0768032
    diffuse_fraction = 0.9
    molecule_mass = molar_mass / AVOGADRO_PER_MOL
    c_bar = math.sqrt(8.0 * BOLTZMANN_J_K * temperature[0] / (math.pi * molecule_mass))
    friction = (
        4.0
        * math.pi
        / 3.0
        * radius[0] ** 2
        * density[0]
        * c_bar
        * (1.0 + diffuse_fraction * math.pi / 8.0)
    )

    force = linear_epstein_force(
        radius,
        density,
        temperature,
        relative_velocity,
        molar_mass_kg_per_mol=molar_mass,
        diffuse_reflection_fraction=diffuse_fraction,
    )

    np.testing.assert_allclose(force, friction * relative_velocity, rtol=2.0e-15)


def test_finite_speed_characterization_has_linear_limit_and_positive_correction() -> None:
    speed_ratio = np.asarray([0.0, 0.05, 0.5])
    diffuse_fraction = 0.9
    coefficient = finite_speed_rate_coefficient(speed_ratio, diffuse_fraction)
    linear = 1.0 + diffuse_fraction * math.pi / 8.0

    assert coefficient[0] == linear
    assert coefficient[1] > linear
    assert coefficient[2] > coefficient[1]


def test_waldmann_gradient_and_heat_flux_forms_are_equivalent() -> None:
    temperature = np.asarray([280.0, 450.0])
    conductivity = np.asarray([0.0165, 0.021])
    gradient = np.asarray([[120.0, -30.0], [-11.0, 70.0]])
    heat_flux = -conductivity[:, None] * gradient

    from_heat_flux = waldmann_heat_flux_force(
        50.0e-9,
        heat_flux,
        temperature,
        molar_mass_kg_per_mol=0.0768032,
    )
    from_gradient = waldmann_gradient_force(
        50.0e-9,
        conductivity,
        gradient,
        temperature,
        molar_mass_kg_per_mol=0.0768032,
    )

    np.testing.assert_allclose(from_heat_flux, from_gradient, rtol=2.0e-15)


def test_missing_trajectory_heat_flux_and_ppr_gradient_stays_not_tested() -> None:
    columns = frozenset({"local_gas_temperature_K", "thermophoretic_force_r_N"})

    assert trajectory_thermophoresis_replay_status(columns) == STATUS_NOT_TESTED


def test_config_has_exactly_twelve_ids_and_output_is_no_clobber(tmp_path: Path) -> None:
    config_path = Path(__file__).parents[1] / "cases/p18r_neutral_transport_closure_v1.json"
    config = load_config(config_path)
    case_ids = tuple(spec.case_id for spec in config.packages)

    validate_case_identity(case_ids, case_ids, expected_count=12)
    assert len(case_ids) == len(set(case_ids)) == 12
    output = tmp_path / "evidence"
    require_new_output_path(output)
    output.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        require_new_output_path(output)
