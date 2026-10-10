"""Stateless counter-based random numbers for wall laws and Brownian paths."""

from __future__ import annotations

import math

import numpy as np
from numba import njit
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]
type UInt64Array = NDArray[np.uint64]

RNG_ALGORITHM_REVISION = "philox4x32_10_v2"
BROWNIAN_RNG_REVISION = "philox4x32_10_brownian_interval_tree_v1"

WALL_PROBABILISTIC_STICK_STREAM = 0x57414C01
WALL_MAXWELL_DIFFUSE_STREAM = 0x57414C02
WALL_MAXWELL_NORMAL_STREAM = 0x57414C03
WALL_MAXWELL_TANGENTIAL_STREAM = 0x57414C04
BROWNIAN_ROOT_NORMAL_STREAM = 0x42524F01
BROWNIAN_SPLIT_NORMAL_STREAM = 0x42525301

_MASK32 = (1 << 32) - 1
_MASK64 = (1 << 64) - 1
_PHILOX_M0 = 0xD2511F53
_PHILOX_M1 = 0xCD9E8D57
_PHILOX_W0 = 0x9E3779B9
_PHILOX_W1 = 0xBB67AE85
_UNIT53 = 2.0**-53
_UNIT52 = 2.0**-52


def philox4x32_10(
    counter: tuple[int, int, int, int],
    key: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Return the Random123 Philox4x32-10 block for unsigned 32-bit words."""

    words = tuple(_uint32(value, "counter") for value in counter)
    key_words = tuple(_uint32(value, "key") for value in key)
    block = _philox4x32_10_kernel(
        np.uint64(words[0]),
        np.uint64(words[1]),
        np.uint64(words[2]),
        np.uint64(words[3]),
        np.uint64(key_words[0]),
        np.uint64(key_words[1]),
    )
    return int(block[0]), int(block[1]), int(block[2]), int(block[3])


def wall_uniform(
    seed: int,
    particle_id: int,
    physical_boundary_event_ordinal: int,
    stream: int,
) -> float:
    """Return one explicitly selected wall-stream draw in ``[0, 1)``."""

    _split_uint64(particle_id, "particle ID")
    _uint32(physical_boundary_event_ordinal, "physical boundary event ordinal")
    _uint32(stream, "wall stream")
    _seed_key(seed)
    return _wall_uniform_kernel(
        np.uint64(seed),
        np.uint64(particle_id),
        np.uint64(physical_boundary_event_ordinal),
        np.uint64(stream),
    )


def wall_uniform_batch(
    seed: int,
    particle_id: UInt64Array,
    physical_boundary_event_ordinal: UInt64Array,
    stream: int,
) -> FloatArray:
    """Evaluate one explicit wall stream for independent event rows."""

    _seed_key(seed)
    stream_word = _uint32(stream, "wall stream")
    particles = _nonnegative_uint64_vector(particle_id, "particle IDs")
    ordinals = _nonnegative_uint64_vector(
        physical_boundary_event_ordinal,
        "physical boundary event ordinals",
    )
    if particles.shape != ordinals.shape:
        raise ValueError("particle IDs and physical boundary event ordinals must align")
    if bool((ordinals > np.uint64(_MASK32)).any()):
        raise ValueError("physical boundary event ordinal must fit in an unsigned 32-bit integer")
    result = np.empty(particles.size, dtype=np.float64)
    _wall_uniform_batch_into_kernel(
        np.uint64(seed),
        particles,
        ordinals,
        np.uint64(stream_word),
        result,
    )
    return result


def wall_uniform_open_batch(
    seed: int,
    particle_id: UInt64Array,
    physical_boundary_event_ordinal: UInt64Array,
    stream: int,
) -> FloatArray:
    """Evaluate one explicit wall stream strictly inside ``(0, 1)``."""

    _seed_key(seed)
    stream_word = _uint32(stream, "wall stream")
    particles = _nonnegative_uint64_vector(particle_id, "particle IDs")
    ordinals = _nonnegative_uint64_vector(
        physical_boundary_event_ordinal,
        "physical boundary event ordinals",
    )
    if particles.shape != ordinals.shape:
        raise ValueError("particle IDs and physical boundary event ordinals must align")
    if bool((ordinals > np.uint64(_MASK32)).any()):
        raise ValueError("physical boundary event ordinal must fit in an unsigned 32-bit integer")
    result = np.empty(particles.size, dtype=np.float64)
    _wall_uniform_open_batch_into_kernel(
        np.uint64(seed),
        particles,
        ordinals,
        np.uint64(stream_word),
        result,
    )
    return result


def wall_standard_normal_batch(
    seed: int,
    particle_id: UInt64Array,
    physical_boundary_event_ordinal: UInt64Array,
    stream: int,
) -> FloatArray:
    """Evaluate one standard normal from each explicit wall stream counter."""

    _seed_key(seed)
    stream_word = _uint32(stream, "wall stream")
    particles = _nonnegative_uint64_vector(particle_id, "particle IDs")
    ordinals = _nonnegative_uint64_vector(
        physical_boundary_event_ordinal,
        "physical boundary event ordinals",
    )
    if particles.shape != ordinals.shape:
        raise ValueError("particle IDs and physical boundary event ordinals must align")
    if bool((ordinals > np.uint64(_MASK32)).any()):
        raise ValueError("physical boundary event ordinal must fit in an unsigned 32-bit integer")
    result = np.empty(particles.size, dtype=np.float64)
    _wall_standard_normal_batch_into_kernel(
        np.uint64(seed),
        particles,
        ordinals,
        np.uint64(stream_word),
        result,
    )
    return result


def brownian_normal_pair_batch(
    seed: int,
    particle_id: UInt64Array,
    macro_interval: int,
    root_interval: int,
    *,
    tree_level: int,
    tree_index: int,
    component: int,
    draw_kind: int,
) -> FloatArray:
    """Return two normal draws per physical Brownian interval and particle.

    The counter owns particle and macro-interval identity.  A two-stage
    Philox-derived key owns the independent root interval, binary interval-tree
    node, Cartesian component, and draw kind.
    Consequently particle ordering and future execution partitioning cannot
    change a draw.  ``(tree_level, tree_index)=(0, 0)`` denotes the root.
    """

    _seed_key(seed)
    particles = _nonnegative_uint64_vector(particle_id, "particle IDs")
    macro_low, macro_high = _split_uint64(macro_interval, "macro interval")
    root_low, root_high = _split_uint64(root_interval, "Brownian root interval")
    level = _uint32(tree_level, "Brownian tree level")
    if level > 63:
        raise ValueError("Brownian tree level must be in [0, 63]")
    tree_low, tree_high = _split_uint64(tree_index, "Brownian tree index")
    if tree_index >= (1 << level):
        raise ValueError("Brownian tree index is outside its level")
    component_word = _uint32(component, "Brownian component")
    if component_word > 1:
        raise ValueError("Brownian component must be 0 or 1")
    stream = _uint32(draw_kind, "Brownian draw kind")
    if stream not in {BROWNIAN_ROOT_NORMAL_STREAM, BROWNIAN_SPLIT_NORMAL_STREAM}:
        raise ValueError("Brownian draw kind is unsupported")
    if stream == BROWNIAN_ROOT_NORMAL_STREAM and (level != 0 or tree_index != 0):
        raise ValueError("Brownian root draws require tree node (0, 0)")

    root_key = philox4x32_10(
        (
            root_low,
            root_high,
            tree_low,
            tree_high,
        ),
        _seed_key(seed),
    )
    derived = philox4x32_10(
        (
            root_key[0],
            root_key[1],
            level,
            stream ^ component_word,
        ),
        (root_key[2], root_key[3]),
    )
    result = np.empty((particles.size, 2), dtype=np.float64)
    _brownian_normal_pair_batch_into_kernel(
        particles,
        np.uint64(macro_low),
        np.uint64(macro_high),
        np.uint64(derived[0]),
        np.uint64(derived[1]),
        result,
    )
    return result


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _philox4x32_10_kernel(
    c0: np.uint64,
    c1: np.uint64,
    c2: np.uint64,
    c3: np.uint64,
    k0: np.uint64,
    k1: np.uint64,
) -> tuple[np.uint64, np.uint64, np.uint64, np.uint64]:
    mask32 = np.uint64(_MASK32)
    for round_index in range(10):
        product0 = np.uint64(_PHILOX_M0) * c0
        product1 = np.uint64(_PHILOX_M1) * c2
        high0 = product0 >> np.uint64(32)
        high1 = product1 >> np.uint64(32)
        low0 = product0 & mask32
        low1 = product1 & mask32
        c0, c1, c2, c3 = (
            (high1 ^ c1 ^ k0) & mask32,
            low1,
            (high0 ^ c3 ^ k1) & mask32,
            low0,
        )
        if round_index != 9:
            k0 = (k0 + np.uint64(_PHILOX_W0)) & mask32
            k1 = (k1 + np.uint64(_PHILOX_W1)) & mask32
    return c0, c1, c2, c3


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _wall_uniform_kernel(
    seed: np.uint64,
    particle_id: np.uint64,
    physical_boundary_event_ordinal: np.uint64,
    stream: np.uint64,
) -> float:
    mask32 = np.uint64(_MASK32)
    block = _philox4x32_10_kernel(
        particle_id & mask32,
        particle_id >> np.uint64(32),
        physical_boundary_event_ordinal,
        stream,
        seed & mask32,
        seed >> np.uint64(32),
    )
    integer = ((block[0] << np.uint64(32)) | block[1]) >> np.uint64(11)
    return float(integer) * _UNIT53


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _wall_uniform_batch_into_kernel(
    seed: np.uint64,
    particle_id: UInt64Array,
    physical_boundary_event_ordinal: UInt64Array,
    stream: np.uint64,
    result: FloatArray,
) -> None:
    for row in range(particle_id.size):
        result[row] = _wall_uniform_kernel(
            seed,
            particle_id[row],
            physical_boundary_event_ordinal[row],
            stream,
        )


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _wall_uniform_open_batch_into_kernel(
    seed: np.uint64,
    particle_id: UInt64Array,
    physical_boundary_event_ordinal: UInt64Array,
    stream: np.uint64,
    result: FloatArray,
) -> None:
    mask32 = np.uint64(_MASK32)
    for row in range(particle_id.size):
        particle = particle_id[row]
        block = _philox4x32_10_kernel(
            particle & mask32,
            particle >> np.uint64(32),
            physical_boundary_event_ordinal[row],
            stream,
            seed & mask32,
            seed >> np.uint64(32),
        )
        integer = ((block[0] << np.uint64(32)) | block[1]) >> np.uint64(12)
        result[row] = (float(integer) + 0.5) * _UNIT52


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _wall_standard_normal_batch_into_kernel(
    seed: np.uint64,
    particle_id: UInt64Array,
    physical_boundary_event_ordinal: UInt64Array,
    stream: np.uint64,
    result: FloatArray,
) -> None:
    mask32 = np.uint64(_MASK32)
    for row in range(particle_id.size):
        particle = particle_id[row]
        block = _philox4x32_10_kernel(
            particle & mask32,
            particle >> np.uint64(32),
            physical_boundary_event_ordinal[row],
            stream,
            seed & mask32,
            seed >> np.uint64(32),
        )
        integer0 = ((block[0] << np.uint64(32)) | block[1]) >> np.uint64(12)
        integer1 = ((block[2] << np.uint64(32)) | block[3]) >> np.uint64(12)
        uniform0 = (float(integer0) + 0.5) * _UNIT52
        uniform1 = (float(integer1) + 0.5) * _UNIT52
        result[row] = math.sqrt(-2.0 * math.log(uniform0)) * math.cos(2.0 * math.pi * uniform1)


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _brownian_normal_pair_batch_into_kernel(
    particle_id: UInt64Array,
    macro_low: np.uint64,
    macro_high: np.uint64,
    key0: np.uint64,
    key1: np.uint64,
    result: FloatArray,
) -> None:
    mask32 = np.uint64(_MASK32)
    for row in range(particle_id.size):
        particle = particle_id[row]
        block = _philox4x32_10_kernel(
            particle & mask32,
            particle >> np.uint64(32),
            macro_low,
            macro_high,
            key0,
            key1,
        )
        integer0 = ((block[0] << np.uint64(32)) | block[1]) >> np.uint64(12)
        integer1 = ((block[2] << np.uint64(32)) | block[3]) >> np.uint64(12)
        uniform0 = (float(integer0) + 0.5) * _UNIT52
        uniform1 = (float(integer1) + 0.5) * _UNIT52
        radius = math.sqrt(-2.0 * math.log(uniform0))
        angle = 2.0 * math.pi * uniform1
        result[row, 0] = radius * math.cos(angle)
        result[row, 1] = radius * math.sin(angle)


def _seed_key(seed: int) -> tuple[int, int]:
    low, high = _split_uint64(seed, "seed")
    return low, high


def _split_uint64(value: int, label: str) -> tuple[int, int]:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _MASK64:
        raise ValueError(f"{label} must fit in an unsigned 64-bit integer")
    return value & _MASK32, (value >> 32) & _MASK32


def _uint32(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _MASK32:
        raise ValueError(f"{label} must fit in an unsigned 32-bit integer")
    return value


def _nonnegative_uint64_vector(value: UInt64Array, label: str) -> UInt64Array:
    result = np.asarray(value)
    if result.ndim != 1 or result.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{label} must be one integer vector")
    if result.dtype.kind == "i" and bool((result < 0).any()):
        raise ValueError(f"{label} must be nonnegative")
    return result.astype(np.uint64, copy=False)
