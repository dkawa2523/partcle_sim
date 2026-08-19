"""Public builders for precomputed NPZ providers."""

from ._precomputed_geometry import build_precomputed_geometry
from ._precomputed_regular import build_precomputed_field
from ._precomputed_triangle import build_precomputed_triangle_mesh_field

__all__ = [
    "build_precomputed_field",
    "build_precomputed_geometry",
    "build_precomputed_triangle_mesh_field",
]
