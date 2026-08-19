from __future__ import annotations

import numpy as np

from particle_tracer_unified.solvers._particle_geometry import (
    physical_sphere_diameter_m,
)


def test_physical_sphere_diameter_comes_from_mass_and_declared_density() -> None:
    physical_diameter = np.asarray([0.8e-6, 2.5e-6], dtype=np.float64)
    density = np.asarray([900.0, 2_200.0], dtype=np.float64)
    mass = density * np.pi * physical_diameter**3 / 6.0
    aerodynamic_diameter = np.asarray([1.4e-6, 1.1e-6], dtype=np.float64)

    result = physical_sphere_diameter_m(
        mass_kg=mass,
        density_kgm3=density,
        drag_diameter_m=aerodynamic_diameter,
    )

    np.testing.assert_allclose(result, physical_diameter, rtol=2.0e-15, atol=0.0)


def test_physical_sphere_diameter_defaults_to_drag_diameter_without_density() -> None:
    aerodynamic_diameter = np.asarray([0.8e-6, 2.5e-6], dtype=np.float64)

    result = physical_sphere_diameter_m(
        mass_kg=np.asarray([1.0e-15, 2.0e-15], dtype=np.float64),
        density_kgm3=np.full(2, np.nan, dtype=np.float64),
        drag_diameter_m=aerodynamic_diameter,
    )

    np.testing.assert_array_equal(result, aerodynamic_diameter)
