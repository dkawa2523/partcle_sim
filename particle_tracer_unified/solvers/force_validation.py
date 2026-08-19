from __future__ import annotations

import numpy as np


def invalid_particle_rows(mask: np.ndarray) -> list[int]:
    invalid = np.asarray(mask, dtype=bool)
    if invalid.ndim > 1:
        invalid = np.any(invalid, axis=tuple(range(1, invalid.ndim)))
    return np.flatnonzero(invalid).tolist()


def require_batch_quantity(
    quantity: str,
    raw: np.ndarray,
    shape: tuple[int, ...],
    *,
    rule: str,
    forces: set[str],
) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float64)
    force_names = "/".join(sorted(forces))
    if values.shape != shape:
        raise ValueError(
            f"{force_names} input quantity {quantity!r} must have shape {shape}; "
            f"received {values.shape}"
        )
    if rule == "finite":
        invalid = ~np.isfinite(values)
        requirement = "finite"
    elif rule == "positive":
        invalid = ~np.isfinite(values) | (values <= 0.0)
        requirement = "finite and strictly positive"
    elif rule == "optional_positive":
        invalid = np.isinf(values) | (np.isfinite(values) & (values <= 0.0))
        requirement = "NaN (unspecified) or finite and strictly positive"
    else:
        raise AssertionError(f"unknown force input rule {rule!r}")
    if np.any(invalid):
        rows = invalid_particle_rows(invalid)
        raise ValueError(
            f"{force_names} input quantity {quantity!r} must be {requirement}; "
            f"invalid particle rows: {rows}"
        )
    return values


def require_force_parameter(
    force: str,
    quantity: str,
    raw: float,
    *,
    rule: str,
) -> float:
    value = float(raw)
    if rule == "positive":
        valid = np.isfinite(value) and value > 0.0
        requirement = "finite and strictly positive"
    elif rule == "nonnegative":
        valid = np.isfinite(value) and value >= 0.0
        requirement = "finite and nonnegative"
    elif rule == "optional_positive":
        valid = np.isnan(value) or (np.isfinite(value) and value > 0.0)
        requirement = "NaN (unspecified) or finite and strictly positive"
    else:
        raise AssertionError(f"unknown force parameter rule {rule!r}")
    if not valid:
        raise ValueError(
            f"{force} input parameter {quantity!r} must be {requirement}; "
            f"received {value!r}"
        )
    return value


def require_positive_density_ratio(
    *,
    forces: set[str],
    gas_density: np.ndarray,
    particle_density: np.ndarray,
) -> np.ndarray:
    particle = np.asarray(particle_density, dtype=np.float64)
    count = int(particle.size)
    particle = require_batch_quantity(
        "particle_density",
        particle,
        (count,),
        rule="positive",
        forces=forces,
    )
    gas = require_batch_quantity(
        "gas_density",
        gas_density,
        (count,),
        rule="positive",
        forces=forces,
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ratio = gas / particle
    return require_batch_quantity(
        "gas_to_particle_density_ratio",
        ratio,
        (count,),
        rule="positive",
        forces=forces,
    )


__all__ = (
    "invalid_particle_rows",
    "require_batch_quantity",
    "require_force_parameter",
    "require_positive_density_ratio",
)
