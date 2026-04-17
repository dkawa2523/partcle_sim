from __future__ import annotations

from typing import Any


SUPPORTED_COORDINATE_SYSTEMS = ("cartesian_xy", "axisymmetric_rz", "cartesian_xyz")


def default_coordinate_system(spatial_dim: int) -> str:
    dim = int(spatial_dim)
    if dim == 2:
        return "cartesian_xy"
    if dim == 3:
        return "cartesian_xyz"
    raise ValueError("spatial_dim must be 2 or 3")


def normalize_coordinate_system(value: Any, spatial_dim: int) -> str:
    dim = int(spatial_dim)
    raw = default_coordinate_system(dim) if value is None else str(value).strip()
    if not raw:
        raw = default_coordinate_system(dim)
    token = raw.lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "cartesian": default_coordinate_system(dim),
        "xy": "cartesian_xy",
        "cartesian_2d": "cartesian_xy",
        "cartesian_xy": "cartesian_xy",
        "axisymmetric": "axisymmetric_rz",
        "axisymmetric_2d": "axisymmetric_rz",
        "axisymmetric_rz": "axisymmetric_rz",
        "cylindrical_rz": "axisymmetric_rz",
        "r_z": "axisymmetric_rz",
        "rz": "axisymmetric_rz",
        "rz_axisymmetric": "axisymmetric_rz",
        "xyz": "cartesian_xyz",
        "cartesian_3d": "cartesian_xyz",
        "cartesian_xyz": "cartesian_xyz",
    }
    normalized = aliases.get(token)
    if normalized is None:
        supported = ", ".join(SUPPORTED_COORDINATE_SYSTEMS)
        raise ValueError(f"Unsupported coordinate_system={raw!r}; supported values are: {supported}")
    if dim == 2 and normalized == "cartesian_xyz":
        raise ValueError("coordinate_system=cartesian_xyz requires spatial_dim=3")
    if dim == 3 and normalized != "cartesian_xyz":
        raise ValueError("spatial_dim=3 currently supports coordinate_system=cartesian_xyz")
    return normalized


def is_axisymmetric_rz(coordinate_system: Any, spatial_dim: int) -> bool:
    return normalize_coordinate_system(coordinate_system, spatial_dim) == "axisymmetric_rz"
