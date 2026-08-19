"""Resolve the small set of enabled features that carry experimental status."""

from __future__ import annotations

from .force_models import ForceModel


def _optional_feature_enabled(physics: object | None, name: str) -> bool:
    feature = getattr(physics, name, None)
    return feature is not None and bool(getattr(feature, "enabled", False))


def enabled_experimental_features(
    force_model: ForceModel | None, physics: object | None
) -> tuple[str, ...]:
    """Return enabled experimental features in stable artifact order."""

    enabled = {
        force.name
        for force in (() if force_model is None else force_model.definitions())
        if force.enabled and force.status == "experimental"
    }
    if _optional_feature_enabled(physics, "charge"):
        enabled.add("dynamic_charge")
    if _optional_feature_enabled(physics, "stochastic"):
        enabled.add("brownian_motion")
    return tuple(sorted(enabled))
