"""Particle-identity based randomness for Brownian paths and wall events."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_UINT64_MASK = (1 << 64) - 1
_VELOCITY_DRAW = 1
_POSITION_DRAW = 2
_BRIDGE_SEED_DRAW = 3
_WALL_DRAW_KINDS = {
    "stick": 101,
    "diffuse_choice": 102,
    "diffuse_polar": 103,
    "diffuse_azimuth": 104,
}


@dataclass(frozen=True, slots=True)
class BrownianRandomContext:
    """Stable identities for one accepted free-flight cohort."""

    particle_id: np.ndarray
    cohort_index: np.ndarray
    macro_step_index: int


@dataclass(frozen=True, slots=True)
class WallRandomContext:
    """Stable identity of one accepted wall interaction."""

    seed: int
    particle_id: int
    macro_step_index: int
    cohort_index: int
    wall_event_ordinal: int


def _uint32_words(value: int) -> tuple[int, int]:
    unsigned = int(value) & _UINT64_MASK
    return unsigned & 0xFFFF_FFFF, unsigned >> 32


def _stream(
    *,
    seed: int,
    particle_id: int,
    macro_step_index: int,
    cohort_index: int,
    component: int,
    draw_kind: int,
) -> np.random.Generator:
    entropy = (
        *_uint32_words(seed),
        *_uint32_words(particle_id),
        *_uint32_words(macro_step_index),
        *_uint32_words(cohort_index),
        int(component),
        int(draw_kind),
    )
    return np.random.default_rng(np.random.SeedSequence(entropy))


def draw_particle_path_randomness(
    *,
    seed: int,
    particle_id: int,
    macro_step_index: int,
    cohort_index: int,
    leaf_count: int,
    dimension: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw one path without depending on active-array order or size.

    Each component and draw kind owns a keyed stream; the accepted-leaf ordinal
    is that stream's counter.  Consequently, another particle cannot shift an
    existing particle's Brownian draws.
    """

    leaves = int(leaf_count)
    dim = int(dimension)
    if leaves < 0 or dim <= 0:
        raise ValueError(
            "Brownian path shape must have nonnegative leaves and dimension"
        )
    shape = (leaves, dim)
    z_velocity = np.empty(shape, dtype=np.float64)
    z_position = np.empty(shape, dtype=np.float64)
    common = {
        "seed": int(seed),
        "particle_id": int(particle_id),
        "macro_step_index": int(macro_step_index),
        "cohort_index": int(cohort_index),
    }
    for component in range(dim):
        z_velocity[:, component] = _stream(
            **common,
            component=component,
            draw_kind=_VELOCITY_DRAW,
        ).normal(size=leaves)
        z_position[:, component] = _stream(
            **common,
            component=component,
            draw_kind=_POSITION_DRAW,
        ).normal(size=leaves)
    bridge_seeds = _stream(
        **common,
        component=0,
        draw_kind=_BRIDGE_SEED_DRAW,
    ).integers(
        0,
        np.iinfo(np.int64).max,
        size=leaves,
        dtype=np.int64,
    )
    return z_velocity, z_position, bridge_seeds


def draw_wall_uniform(context: WallRandomContext, draw_kind: str) -> float:
    """Draw one semantic wall variate without sharing mutable RNG state."""

    try:
        kind = _WALL_DRAW_KINDS[str(draw_kind)]
    except KeyError as exc:
        raise ValueError(f"unknown wall random draw kind {draw_kind!r}") from exc
    return float(
        _stream(
            seed=int(context.seed),
            particle_id=int(context.particle_id),
            macro_step_index=int(context.macro_step_index),
            cohort_index=int(context.cohort_index),
            component=int(context.wall_event_ordinal),
            draw_kind=int(kind),
        ).random()
    )


__all__ = (
    "BrownianRandomContext",
    "WallRandomContext",
    "draw_particle_path_randomness",
    "draw_wall_uniform",
)
