"""Bounded independent finite-volume reference for integrated OU barriers.

Dimensionless dx=v dt, dv=-v dt+sqrt(2) dW; x<0 is absorbing.
This test calculation does not call production OU, tree, or event routines.
Its first-order discretization requires measured refinement, not a claim that
the finest-grid difference is a rigorous error bound.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit


@dataclass(frozen=True)
class KramersResult:
    times: np.ndarray
    arrival_cdf: np.ndarray
    surviving_mass: np.ndarray
    position_leak: np.ndarray
    velocity_leak: np.ndarray
    minimum_cell_mass: float
    maximum_mass_error: float


def _delta_weights(centers: np.ndarray, value: float, width: float) -> np.ndarray:
    weights = np.maximum(0.0, 1.0 - np.abs(centers - value) / width)
    if not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=2.0e-13):
        raise ValueError("initial delta must lie inside the reference domain")
    return weights / weights.sum()


def _bernoulli(argument: np.ndarray) -> np.ndarray:
    result = np.ones_like(argument)
    nonzero = argument != 0.0
    result[nonzero] = argument[nonzero] / np.expm1(argument[nonzero])
    return result


@njit(cache=False)
def _advance(
    mass: np.ndarray,
    velocity: np.ndarray,
    right_rates: np.ndarray,
    left_rates: np.ndarray,
    dx: float,
    dt: float,
    sample_steps: np.ndarray,
    left_absorbing: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    nv, nx = mass.shape
    half = 0.5 * dt
    diagonal = 1.0 + half * (left_rates[:-1] + right_rates[1:])
    lower = -half * right_rates[1:-1]
    upper = -half * left_rates[1:-1]
    modified_upper = np.empty(nv - 1)
    inverse_pivot = np.empty(nv)
    inverse_pivot[0] = 1.0 / diagonal[0]
    modified_upper[0] = upper[0] * inverse_pivot[0]
    for j in range(1, nv):
        inverse_pivot[j] = 1.0 / (diagonal[j] - lower[j - 1] * modified_upper[j - 1])
        if j < nv - 1:
            modified_upper[j] = upper[j] * inverse_pivot[j]
    work = np.empty_like(mass)
    sample_count = sample_steps.size
    arrival = np.zeros(sample_count)
    survival = np.zeros(sample_count)
    x_leak = np.zeros(sample_count)
    v_leak = np.zeros(sample_count)
    absorbed = 0.0
    lost_x = 0.0
    lost_v = 0.0
    minimum = 0.0
    mass_error = 0.0
    sample = 0
    for step in range(1, sample_steps[-1] + 1):
        for split in range(2):
            for i in range(nx):
                work[0, i] = mass[0, i] * inverse_pivot[0]
                for j in range(1, nv):
                    work[j, i] = (mass[j, i] - lower[j - 1] * work[j - 1, i]) * inverse_pivot[j]
                mass[nv - 1, i] = work[nv - 1, i]
                for j in range(nv - 2, -1, -1):
                    mass[j, i] = work[j, i] - modified_upper[j] * mass[j + 1, i]
                lost_v += half * (left_rates[0] * mass[0, i] + right_rates[-1] * mass[-1, i])
            if split == 0:
                for j in range(nv):
                    courant = abs(velocity[j]) * dt / dx
                    if velocity[j] > 0.0:
                        absorbed += courant * mass[j, -1]
                        for i in range(nx - 1, 0, -1):
                            mass[j, i] = (1.0 - courant) * mass[j, i] + courant * mass[j, i - 1]
                        mass[j, 0] *= 1.0 - courant
                    else:
                        if left_absorbing:
                            absorbed += courant * mass[j, 0]
                        else:
                            lost_x += courant * mass[j, 0]
                        for i in range(nx - 1):
                            mass[j, i] = (1.0 - courant) * mass[j, i] + courant * mass[j, i + 1]
                        mass[j, -1] *= 1.0 - courant
        minimum = min(minimum, mass.min())
        total = mass.sum()
        mass_error = max(mass_error, abs(total + absorbed + lost_x + lost_v - 1.0))
        if step == sample_steps[sample]:
            arrival[sample] = absorbed
            survival[sample] = total
            x_leak[sample] = lost_x
            v_leak[sample] = lost_v
            sample += 1
    return arrival, survival, x_leak, v_leak, minimum, mass_error


def integrated_ou_barrier(
    *,
    nx: int,
    nv: int,
    dt: float,
    times: np.ndarray,
    length: float = 4.0,
    velocity_limit: float = 6.0,
    initial_position: float = -0.5,
    initial_velocity: float = 0.0,
    left_absorbing: bool = False,
) -> KramersResult:
    """Positive conservative Kramers calculation with separately counted leaks.

    The exponentially fitted velocity flux has nonnegative transition rates;
    backward Euler is an M-matrix solve. Spatial upwind transport uses zero
    incoming density, retaining outgoing v>0 flux at x=0. Delta data are
    represented by moment-preserving adjacent cell masses, refined with dx/dv.
    Artificial velocity ends are absorbing, so their loss is measured.
    For a folded signed OU radius, both position ends are physical absorbing
    barriers: left_absorbing counts their combined outgoing flux as arrival.
    """

    dx = length / nx
    dv = 2.0 * velocity_limit / nv
    position = -length + dx * (np.arange(nx) + 0.5)
    velocity = -velocity_limit + dv * (np.arange(nv) + 0.5)
    if np.max(np.abs(velocity)) * dt > dx:
        raise ValueError("reference spatial CFL exceeds one")
    steps = np.rint(times / dt).astype(np.int64)
    if not np.allclose(steps * dt, times, rtol=0.0, atol=2.0e-14) or np.any(np.diff(steps) <= 0):
        raise ValueError("reference observations must be distinct positive timestep endpoints")
    interface = np.linspace(-velocity_limit, velocity_limit, nv + 1)
    peclet = -interface * dv
    right = _bernoulli(-peclet) / dv**2
    left = _bernoulli(peclet) / dv**2
    mass = np.outer(
        _delta_weights(velocity, initial_velocity, dv),
        _delta_weights(position, initial_position, dx),
    )
    arrival, survival, x_leak, v_leak, minimum, mass_error = _advance(
        mass, velocity, right, left, dx, dt, steps, left_absorbing
    )
    return KramersResult(times.copy(), arrival, survival, x_leak, v_leak, minimum, mass_error)
