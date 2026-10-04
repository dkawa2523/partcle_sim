from __future__ import annotations

import math
from typing import cast

import numpy as np

from particle_tracer_unified.core.catalogs import (
    is_internal_pass_through,
    normalize_wall_law_name,
)
from particle_tracer_unified.core.datamodel import WallPartModel

from . import _stochastic_randomness
from ._stochastic_randomness import WallRandomContext

_ZERO_VELOCITY_OUTCOMES = {
    "stick": "stuck",
    "escape": "escaped",
    "absorb": "absorbed",
}


def _unit_vector(values: np.ndarray, *, context: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    magnitude = float(np.linalg.norm(vector))
    if np.any(~np.isfinite(vector)) or not np.isfinite(magnitude) or magnitude <= 0.0:
        raise ValueError(f"{context} must be a finite non-zero vector")
    return vector / magnitude


def _orthonormal_tangent_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = _unit_vector(normal, context="wall normal")
    reference = (
        np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(n[0]) < 0.9
        else np.array([0.0, 1.0, 0.0], dtype=np.float64)
    )
    tangent_1 = reference - np.dot(reference, n) * n
    tangent_1 = _unit_vector(tangent_1, context="first wall tangent")
    tangent_2 = _unit_vector(np.cross(n, tangent_1), context="second wall tangent")
    return tangent_1, tangent_2


def _sample_diffuse_reflection(
    rng: np.random.Generator | None,
    normal: np.ndarray,
    speed: float,
    context: WallRandomContext | None,
) -> np.ndarray:
    """Sample the cosine-weighted direction leaving a point-particle wall hit."""

    n = _unit_vector(normal, context="wall normal")
    if n.size == 2:
        tangent = _unit_vector(
            np.array([-n[1], n[0]], dtype=np.float64),
            context="wall tangent",
        )
        theta = math.asin(2.0 * _uniform_draw(rng, context, "diffuse_polar") - 1.0)
        direction = -math.cos(theta) * n + math.sin(theta) * tangent
        return speed * _unit_vector(direction, context="diffuse reflection direction")

    tangent_1, tangent_2 = _orthonormal_tangent_basis(n)
    u = _uniform_draw(rng, context, "diffuse_polar")
    phi = (
        2.0
        * math.pi
        * _uniform_draw(
            rng,
            context,
            "diffuse_azimuth",
        )
    )
    cos_theta = math.sqrt(1.0 - u)
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    direction = -cos_theta * n + sin_theta * (
        math.cos(phi) * tangent_1 + math.sin(phi) * tangent_2
    )
    return speed * _unit_vector(direction, context="diffuse reflection direction")


def sample_diffuse_reflection(
    rng: np.random.Generator,
    normal: np.ndarray,
    speed: float,
) -> np.ndarray:
    """Sample the cosine-weighted direction leaving a point-particle wall hit."""

    return _sample_diffuse_reflection(rng, normal, speed, None)


def _uniform_draw(
    rng: np.random.Generator | None,
    context: WallRandomContext | None,
    draw_kind: str,
) -> float:
    if context is None:
        return float(cast(np.random.Generator, rng).uniform(0.0, 1.0))
    return _stochastic_randomness.draw_wall_uniform(context, draw_kind)


def _probability_event(
    rng: np.random.Generator | None,
    context: WallRandomContext | None,
    *,
    probability: float,
    draw_kind: str,
) -> bool:
    if context is None:
        return bool(cast(np.random.Generator, rng).random() < probability)
    if probability <= 0.0:
        return False
    if probability >= 1.0:
        return True
    return bool(
        _stochastic_randomness.draw_wall_uniform(context, draw_kind) < probability
    )


def _terminal_wall_response(
    mode: str,
    velocity: np.ndarray,
    wall_model: WallPartModel,
) -> tuple[str, np.ndarray] | None:
    outcome = _ZERO_VELOCITY_OUTCOMES.get(mode)
    if outcome is not None:
        return outcome, np.zeros_like(velocity)
    if mode == "freeze":
        return "frozen", velocity.copy()
    if mode == "pass_through":
        outcome = (
            "passed_through" if is_internal_pass_through(wall_model) else "escaped"
        )
        return outcome, velocity.copy()
    return None


def _apply_wall_response(
    rng: np.random.Generator | None,
    velocity: np.ndarray,
    normal: np.ndarray,
    wall_model: WallPartModel,
    context: WallRandomContext | None,
) -> tuple[str, np.ndarray]:
    """Apply one catalogued wall law without geometry, state, or I/O effects."""

    mode = normalize_wall_law_name(wall_model.law_name, context="collision wall law")
    restitution = max(0.0, float(wall_model.restitution))
    diffuse_fraction = float(np.clip(wall_model.diffuse_fraction, 0.0, 1.0))
    stick_probability = float(np.clip(wall_model.stick_probability, 0.0, 1.0))
    n = _unit_vector(normal, context="wall normal")
    velocity_array = np.asarray(velocity, dtype=np.float64)
    speed = float(np.linalg.norm(velocity_array))
    normal_velocity = float(np.dot(velocity_array, n))

    terminal_response = _terminal_wall_response(mode, velocity_array, wall_model)
    if terminal_response is not None:
        return terminal_response
    if mode == "critical_sticking_velocity" and abs(normal_velocity) <= max(
        0.0, float(wall_model.critical_sticking_velocity_mps)
    ):
        return "stuck", np.zeros_like(velocity_array)
    if _probability_event(
        rng,
        context,
        probability=stick_probability,
        draw_kind="stick",
    ):
        return "stuck", np.zeros_like(velocity_array)
    if mode == "cosine_diffuse":
        return "reflected_diffuse", _sample_diffuse_reflection(
            rng,
            n,
            restitution * speed,
            context,
        )
    if mode == "mixed_specular_diffuse" and _probability_event(
        rng,
        context,
        probability=diffuse_fraction,
        draw_kind="diffuse_choice",
    ):
        return "reflected_diffuse", _sample_diffuse_reflection(
            rng,
            n,
            restitution * speed,
            context,
        )

    tangential_velocity = velocity_array - normal_velocity * n
    reflected_velocity = tangential_velocity - restitution * normal_velocity * n
    return "reflected_specular", reflected_velocity


def apply_wall_response(
    rng: np.random.Generator,
    velocity: np.ndarray,
    normal: np.ndarray,
    wall_model: WallPartModel,
) -> tuple[str, np.ndarray]:
    """Apply one catalogued wall law without geometry, state, or I/O effects."""

    return _apply_wall_response(rng, velocity, normal, wall_model, None)


def _apply_keyed_wall_response(
    context: WallRandomContext,
    velocity: np.ndarray,
    normal: np.ndarray,
    wall_model: WallPartModel,
) -> tuple[str, np.ndarray]:
    return _apply_wall_response(None, velocity, normal, wall_model, context)


__all__ = ("apply_wall_response", "sample_diffuse_reflection")
