from __future__ import annotations

import numpy as np
import pytest

from particle_tracer_unified.solvers._segment_stage_dynamics import (
    _virtual_mass_factor,
)
from particle_tracer_unified.solvers.contact_sliding import displaced_fluid_factors
from particle_tracer_unified.solvers.forces.runtime import ForceRuntimeParameters


def test_extremely_small_positive_densities_are_used_without_a_floor() -> None:
    params = ForceRuntimeParameters(
        virtual_mass_enabled=True,
        virtual_mass_coefficient=0.5,
        gravity_buoyancy_enabled=True,
    )
    gas_density = np.asarray([2.0e-310, 3.0e-310])
    particle_density = np.asarray([1.0e-310, 6.0e-310])

    gravity_factor, inertia_factor = displaced_fluid_factors(
        params,
        gas_density,
        particle_density,
    )

    np.testing.assert_allclose(gravity_factor, np.asarray([-1.0, 0.5]), rtol=1e-14)
    np.testing.assert_allclose(inertia_factor, np.asarray([2.0, 1.25]), rtol=1e-14)
    assert _virtual_mass_factor(
        params, gas_density[0], particle_density[0]
    ) == pytest.approx(
        2.0,
        rel=1e-14,
    )


@pytest.mark.parametrize("invalid", [np.nan, np.inf, 0.0, -1.0])
def test_contact_factors_reject_invalid_gas_density_with_row(invalid: float) -> None:
    params = ForceRuntimeParameters(
        virtual_mass_enabled=True,
        gravity_buoyancy_enabled=True,
    )

    with pytest.raises(
        ValueError,
        match=(
            r"gravity_buoyancy/virtual_mass.*gas_density.*"
            r"invalid particle rows: \[1\]"
        ),
    ):
        displaced_fluid_factors(
            params,
            np.asarray([1.0, invalid]),
            np.asarray([1000.0, 1200.0]),
        )


@pytest.mark.parametrize("invalid", [np.nan, np.inf, 0.0, -1.0])
def test_contact_factors_reject_invalid_particle_density_with_row(
    invalid: float,
) -> None:
    params = ForceRuntimeParameters(gravity_buoyancy_enabled=True)

    with pytest.raises(
        ValueError,
        match=r"gravity_buoyancy.*particle_density.*invalid particle rows: \[0\]",
    ):
        displaced_fluid_factors(
            params,
            np.asarray([1.0, 1.2]),
            np.asarray([invalid, 1200.0]),
        )


def test_density_ratio_overflow_fails_instead_of_using_a_fallback() -> None:
    params = ForceRuntimeParameters(virtual_mass_enabled=True)

    with pytest.raises(
        ValueError,
        match=(
            r"virtual_mass.*gas_to_particle_density_ratio.*"
            r"invalid particle rows: \[0\]"
        ),
    ):
        displaced_fluid_factors(
            params,
            np.asarray([1.0]),
            np.asarray([np.nextafter(0.0, 1.0)]),
        )


@pytest.mark.parametrize("invalid", [np.nan, np.inf, 0.0, -0.5])
def test_virtual_mass_coefficient_must_be_strictly_positive(invalid: float) -> None:
    params = ForceRuntimeParameters(
        virtual_mass_enabled=True,
        virtual_mass_coefficient=invalid,
    )

    with pytest.raises(
        ValueError, match=r"virtual_mass.*coefficient.*strictly positive"
    ):
        displaced_fluid_factors(
            params,
            np.asarray([1.0]),
            np.asarray([1000.0]),
        )
    with pytest.raises(
        ValueError, match=r"virtual_mass.*coefficient.*strictly positive"
    ):
        _virtual_mass_factor(params, 1.0, 1000.0)


@pytest.mark.parametrize(
    ("quantity", "gas_density", "particle_density"),
    [
        ("gas_density", np.nan, 1000.0),
        ("particle_density", 1.0, 0.0),
    ],
)
def test_scalar_virtual_mass_rejects_invalid_density_as_particle_row_zero(
    quantity: str,
    gas_density: float,
    particle_density: float,
) -> None:
    params = ForceRuntimeParameters(virtual_mass_enabled=True)

    with pytest.raises(
        ValueError,
        match=rf"virtual_mass.*{quantity}.*invalid particle rows: \[0\]",
    ):
        _virtual_mass_factor(params, gas_density, particle_density)
