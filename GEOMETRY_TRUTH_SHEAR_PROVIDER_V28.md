# Geometry-truth-based shear provider in v28

The wall-shear provider now uses a clear priority order:

1. scalar `tauw`
2. scalar `u_tau`
3. vector `tauw_vec`, `traction`, `surface_traction`
4. direct probe estimate from geometry normal + local flow + viscosity

When vector traction-like quantities are available, the provider projects them into the wall tangent plane using the geometry normal. This makes the shear estimate part-aware and geometry-aware.

The current implementation supports synthetic box geometry and regular-grid flow fields. The same interface is intended for COMSOL-backed geometry/field providers.
