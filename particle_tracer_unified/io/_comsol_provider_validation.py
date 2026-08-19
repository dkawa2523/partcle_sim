"""Validate a loaded field provider against its COMSOL manifest."""

from __future__ import annotations

from typing import Any

import numpy as np

from .comsol_manifest import ComsolCaseManifest


def _provider_time_support(quantities: dict[str, Any]) -> tuple[float, float] | None:
    times = [
        values
        for series in quantities.values()
        if (values := np.asarray(series.times, dtype=np.float64)).size
    ]
    if not times:
        return None
    return (
        min(float(values[0]) for values in times),
        max(float(values[-1]) for values in times),
    )


def validate_comsol_runtime_provider(
    manifest: ComsolCaseManifest,
    field_provider: Any,
) -> None:
    if field_provider is None:
        return
    field = field_provider.field
    if int(getattr(field, "metadata", {}).get("field_ghost_cells", 0) or 0):
        raise ValueError(
            "COMSOL faithful mode requires field bundles without ghost cells"
        )

    quantities = getattr(field, "quantities", {})
    missing = sorted(set(manifest.field_quantity_mapping()) - set(quantities))
    if missing:
        raise ValueError(
            f"COMSOL field provider is missing manifest quantities: {missing}"
        )
    declared = manifest.time_support_s
    if declared is None:
        return
    actual = _provider_time_support(quantities)
    if actual is not None and not manifest.matches_time_support(actual):
        raise ValueError(
            "COMSOL manifest time.support_s does not match field provider: "
            f"{declared} != {actual}"
        )


__all__ = ("validate_comsol_runtime_provider",)
