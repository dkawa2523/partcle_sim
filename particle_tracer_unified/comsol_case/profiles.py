"""Declarative COMSOL export profiles and stable artifact names."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

SCHEMA_VERSION = 2
COMSOL_MANIFEST_NAME = "comsol_manifest.yaml"
CANONICAL_PARTICLES_NAME = "particles.csv"
CANONICAL_BOUNDARIES_NAME = "boundaries.csv"


@dataclass(frozen=True)
class BuildProfile:
    """Mapping from exported array names to physical quantities.

    Profiles describe only the exporter handoff.  Solver physics and wall
    behaviour remain explicit case inputs and manifest inventory.
    """

    name: str
    coordinate_system: str
    sample_axis_columns: tuple[str, str]
    required_sample_columns: tuple[str, ...]
    scalar_fields: Mapping[str, tuple[str, str]]


BUILD_PROFILES: Mapping[str, BuildProfile] = {
    "generic": BuildProfile(
        name="generic",
        coordinate_system="cartesian_xy",
        sample_axis_columns=("x", "y"),
        required_sample_columns=("ux", "uy"),
        scalar_fields={
            "mu": ("dynamic_viscosity", "Pa*s"),
            "rho": ("density", "kg/m^3"),
            "T": ("temperature", "K"),
            "pressure": ("pressure", "Pa"),
        },
    ),
    "icp_cf4_o2": BuildProfile(
        name="icp_cf4_o2",
        coordinate_system="axisymmetric_rz",
        sample_axis_columns=("r", "z"),
        required_sample_columns=("ux", "uy", "mu", "E_x", "E_y"),
        scalar_fields={
            "mu": ("dynamic_viscosity", "Pa*s"),
            "rho_g": ("density", "kg/m^3"),
            "T": ("temperature", "K"),
            "p": ("pressure", "Pa"),
            "ne": ("scalar", "1/m^3"),
            "Te": ("scalar", "eV"),
            "phi": ("scalar", "V"),
        },
    ),
}
