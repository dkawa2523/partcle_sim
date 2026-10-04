from __future__ import annotations

from typing import Any

import numpy as np

from .experimental_features import enabled_experimental_features
from .preflight_types import ValidationIssue


def particle_issues(runtime: Any) -> list[ValidationIssue]:
    particles = getattr(runtime, "particles", None)
    if particles is None:
        return [
            ValidationIssue(
                "input.particles.missing", "Simulation case has no particle table"
            )
        ]

    issues: list[ValidationIssue] = []
    values_by_name = {
        "position": np.asarray(particles.position, dtype=np.float64),
        "velocity": np.asarray(particles.velocity, dtype=np.float64),
        "release_time_s": np.asarray(particles.release_time, dtype=np.float64),
        "mass_kg": np.asarray(particles.mass, dtype=np.float64),
        "drag_diameter_m": np.asarray(particles.diameter, dtype=np.float64),
        "charge_C": np.asarray(particles.charge, dtype=np.float64),
    }
    for name, values in values_by_name.items():
        if not np.all(np.isfinite(values)):
            issues.append(
                ValidationIssue(
                    f"input.particles.{name}.non_finite",
                    f"{name} contains non-finite values",
                )
            )
    invalid_value_checks = (
        (
            np.any(values_by_name["release_time_s"] < 0.0),
            "input.particles.release_time.negative",
            "release_time_s must be non-negative",
        ),
        (
            np.any(values_by_name["mass_kg"] <= 0.0),
            "input.particles.mass.non_positive",
            "mass_kg must be positive",
        ),
        (
            np.any(values_by_name["drag_diameter_m"] <= 0.0),
            "input.particles.drag_diameter.non_positive",
            "drag_diameter_m must be positive",
        ),
    )
    issues.extend(
        ValidationIssue(code, message)
        for invalid, code, message in invalid_value_checks
        if invalid
    )
    ids = np.asarray(particles.particle_id, dtype=np.int64)
    if np.unique(ids).size != ids.size:
        issues.append(
            ValidationIssue(
                "input.particles.id.duplicate", "particle_id values must be unique"
            )
        )
    return issues


def experimental_feature_report(
    case: Any,
    runtime: Any,
) -> tuple[list[str], list[ValidationIssue]]:
    force_catalog = getattr(runtime, "force_catalog", None)
    features = list(
        enabled_experimental_features(
            getattr(force_catalog, "model", None),
            getattr(case.config, "physics", None),
        )
    )
    if not features:
        return features, []
    return features, [
        ValidationIssue(
            "physics.experimental",
            "Enabled physics features are experimental until their v0.2 V&V "
            "suite is complete",
            severity="warning",
            context={"features": features},
        )
    ]


__all__ = (
    "experimental_feature_report",
    "particle_issues",
)
