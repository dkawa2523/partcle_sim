"""Compatibility rules that apply after adapter-specific physics resolution."""

from __future__ import annotations

from particle_tracer_unified.force_models import ForceModel


def validate_coordinate_force_compatibility(
    coordinate_system: str,
    force_model: ForceModel,
) -> None:
    """Reject force equations that are undefined in a coordinate contract."""

    if not isinstance(force_model, ForceModel):
        raise TypeError("force compatibility requires a typed ForceModel")
    if coordinate_system == "axisymmetric_rz" and force_model.lift.enabled:
        raise ValueError(
            "axisymmetric_rz no-swirl motion does not support the Cartesian lift force"
        )


__all__ = ("validate_coordinate_force_compatibility",)
