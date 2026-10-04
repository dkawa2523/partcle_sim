"""Value types and semantic names used by COMSOL manifests."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

ALLOWED_COORDINATE_SYSTEMS = {"cartesian_xy", "axisymmetric_rz", "cartesian_xyz"}
BUILTIN_FIELD_SEMANTICS = frozenset(
    {
        "velocity",
        "electric_field",
        "force",
        "acceleration",
        "density",
        "dynamic_viscosity",
        "temperature",
        "pressure",
    }
)
VECTOR_FIELD_SEMANTICS = frozenset(
    {"velocity", "electric_field", "force", "acceleration"}
)
BUILTIN_FIELD_SEMANTICS_CASEFOLD = frozenset(
    item.casefold() for item in BUILTIN_FIELD_SEMANTICS
)
GENERIC_SCALAR_SEMANTIC_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
EXPECTED_SI_UNITS = {
    "velocity": "m/s",
    "electric_field": "V/m",
    "force": "N",
    "acceleration": "m/s^2",
    "density": "kg/m^3",
    "dynamic_viscosity": "Pa*s",
    "temperature": "K",
    "pressure": "Pa",
}


def classify_field_semantic(value: str) -> str | None:
    """Return the storage kind for an exact built-in or generic scalar name."""

    semantic = str(value)
    if semantic in VECTOR_FIELD_SEMANTICS:
        return "vector"
    if semantic in BUILTIN_FIELD_SEMANTICS:
        return "scalar"
    if semantic == "scalar" or not GENERIC_SCALAR_SEMANTIC_RE.fullmatch(semantic):
        return None
    if semantic.casefold() in BUILTIN_FIELD_SEMANTICS_CASEFOLD:
        return None
    return "scalar"


def expected_axes(coordinate_system: str | None) -> tuple[str, ...]:
    return {
        "cartesian_xy": ("x", "y"),
        "axisymmetric_rz": ("r", "z"),
        "cartesian_xyz": ("x", "y", "z"),
    }.get(str(coordinate_system), ())


def field_target(semantic: str, component: str) -> str:
    axis = str(component)
    return {
        "velocity": f"u{axis}",
        "electric_field": f"E_{axis}",
        "density": "rho",
        "dynamic_viscosity": "mu",
        "temperature": "T",
        "pressure": "pressure",
    }.get(semantic, semantic if component == "value" else f"{semantic}_{component}")


@dataclass(frozen=True)
class ComsolArtifact:
    name: str
    path: str
    sha256: str
    format: str
    size_bytes: int | None = None

    def resolve(self, root_dir: Path) -> Path:
        candidate = Path(self.path)
        return (
            candidate if candidate.is_absolute() else (root_dir / candidate).resolve()
        )


@dataclass(frozen=True)
class ComsolFieldSpec:
    semantic_quantity: str
    components: Mapping[str, str]
    unit: str | None = None
    scale_to_si: float = 1.0
    artifact: str = "field"


__all__ = (
    "ALLOWED_COORDINATE_SYSTEMS",
    "BUILTIN_FIELD_SEMANTICS",
    "EXPECTED_SI_UNITS",
    "ComsolArtifact",
    "ComsolFieldSpec",
    "classify_field_semantic",
    "expected_axes",
    "field_target",
)
