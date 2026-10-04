"""Particle geometry derived from canonical particle properties."""

from __future__ import annotations

import numpy as np


def physical_sphere_diameter_m(
    *,
    mass_kg: np.ndarray,
    density_kgm3: np.ndarray,
    drag_diameter_m: np.ndarray,
) -> np.ndarray:
    """Return the material-equivalent sphere diameter for each particle.

    A positive declared density owns physical geometry together with mass.  When
    it is unavailable, the drag diameter remains the legacy spherical default.
    """

    mass = np.asarray(mass_kg, dtype=np.float64)
    density = np.asarray(density_kgm3, dtype=np.float64)
    physical_diameter = np.asarray(drag_diameter_m, dtype=np.float64).copy()
    declared_density = np.isfinite(density) & (density > 0.0)
    physical_diameter[declared_density] = np.cbrt(
        6.0 * mass[declared_density] / (np.pi * density[declared_density])
    )
    return physical_diameter


__all__ = ("physical_sphere_diameter_m",)
