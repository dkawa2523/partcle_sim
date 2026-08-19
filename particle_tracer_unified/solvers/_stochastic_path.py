"""Exact integrated Ornstein-Uhlenbeck path representation and replay."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_DYADIC_BRIDGE_MAX_DEPTH = 24


@dataclass(frozen=True, slots=True)
class _IntegratedOuLeafPath:
    """Exact constant-coefficient integrated-OU path for one accepted leaf."""

    duration_s: float
    tau_eff_s: float
    thermal_velocity_variance_m2s2: float
    z_velocity: np.ndarray
    z_position: np.ndarray
    bridge_seed: int
    _bridge_states: dict[float, tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _bridge_rng: np.random.Generator = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        duration = float(self.duration_s)
        tau = float(self.tau_eff_s)
        theta = float(self.thermal_velocity_variance_m2s2)
        z_velocity = np.asarray(self.z_velocity, dtype=np.float64)
        z_position = np.asarray(self.z_position, dtype=np.float64)
        scalars = np.asarray((duration, tau, theta), dtype=np.float64)
        valid_scalars = np.isfinite(scalars) & (scalars >= 0.0)
        valid_scalars[1] = np.isfinite(tau) & (tau > 0.0)
        if not np.all(valid_scalars):
            raise ValueError(
                "Langevin path duration_s and thermal_velocity_variance_m2s2 must "
                "be finite and non-negative; tau_eff_s must be finite and positive"
            )
        if (
            z_velocity.shape != z_position.shape
            or np.any(~np.isfinite(z_velocity))
            or np.any(~np.isfinite(z_position))
        ):
            raise ValueError(
                "Langevin path normal vectors must be finite and have matching shapes"
            )
        object.__setattr__(self, "z_velocity", z_velocity)
        object.__setattr__(self, "z_position", z_position)
        seed = int(self.bridge_seed) % (1 << 64)
        object.__setattr__(self, "_bridge_rng", np.random.default_rng(seed))


def _integrated_ou_covariances(
    elapsed_s: float,
    tau_eff_s: float,
    thermal_velocity_variance_m2s2: float,
) -> tuple[float, float, float]:
    dt = max(float(elapsed_s), 0.0)
    tau = float(tau_eff_s)
    theta = max(float(thermal_velocity_variance_m2s2), 0.0)
    if dt <= 0.0 or theta <= 0.0:
        return 0.0, 0.0, 0.0
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("integrated Langevin motion requires finite tau_eff_s > 0")
    a = dt / tau
    one_minus_decay = -np.expm1(-a)
    var_v = theta * (-np.expm1(-2.0 * a))
    cov_xv = theta * tau * one_minus_decay * one_minus_decay
    if abs(a) <= 1.0e-3:
        a2 = a * a
        g = (
            (2.0 / 3.0) * a2 * a
            - 0.5 * a2 * a2
            + (7.0 / 30.0) * a2 * a2 * a
            - (1.0 / 12.0) * a2 * a2 * a2
        )
    else:
        g = 2.0 * a - 3.0 + 4.0 * np.exp(-a) - np.exp(-2.0 * a)
    var_x = theta * tau * tau * max(float(g), 0.0)
    return max(float(var_x), 0.0), max(float(var_v), 0.0), float(cov_xv)


def _ou_transition_matrix(delta_t_s: float, tau_eff_s: float) -> np.ndarray:
    """Transition for the normalized state ``(x / tau, v)``."""

    elapsed = max(float(delta_t_s), 0.0)
    tau = float(tau_eff_s)
    decay = np.exp(-elapsed / tau)
    return np.asarray(
        ((1.0, 1.0 - decay), (0.0, decay)),
        dtype=np.float64,
    )


def _ou_normalized_covariance(
    delta_t_s: float,
    tau_eff_s: float,
    thermal_velocity_variance_m2s2: float,
) -> np.ndarray:
    """Noise covariance for ``(x / tau, v)`` over one Markov interval."""

    tau = float(tau_eff_s)
    var_x, var_v, cov_xv = _integrated_ou_covariances(
        float(delta_t_s),
        tau,
        float(thermal_velocity_variance_m2s2),
    )
    return np.asarray(
        ((var_x / (tau * tau), cov_xv / tau), (cov_xv / tau, var_v)),
        dtype=np.float64,
    )


def _noise_from_covariance(
    covariance: np.ndarray,
    z_velocity: np.ndarray,
    z_position: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``(x / tau, v)`` noise with a velocity-first factorization."""

    matrix = np.asarray(covariance, dtype=np.float64)
    cov = 0.5 * (matrix + matrix.T)
    var_v = max(float(cov[1, 1]), 0.0)
    sigma_v = np.sqrt(var_v)
    coeff_xv = float(cov[0, 1]) / sigma_v if sigma_v > 0.0 else 0.0
    residual_var_x = max(float(cov[0, 0]) - coeff_xv * coeff_xv, 0.0)
    velocity_noise = sigma_v * np.asarray(z_velocity, dtype=np.float64)
    position_noise = coeff_xv * np.asarray(z_velocity, dtype=np.float64)
    position_noise = position_noise + np.sqrt(residual_var_x) * np.asarray(
        z_position,
        dtype=np.float64,
    )
    return np.asarray(position_noise, dtype=np.float64), np.asarray(
        velocity_noise, dtype=np.float64
    )


def _initialize_langevin_bridge(path: _IntegratedOuLeafPath) -> None:
    if path._bridge_states:
        return
    zero_x = np.zeros_like(np.asarray(path.z_position, dtype=np.float64))
    zero_v = np.zeros_like(np.asarray(path.z_velocity, dtype=np.float64))
    path._bridge_states[0.0] = (zero_x, zero_v)
    duration = float(path.duration_s)
    if duration <= 0.0:
        return
    covariance = _ou_normalized_covariance(
        duration,
        float(path.tau_eff_s),
        float(path.thermal_velocity_variance_m2s2),
    )
    endpoint_x_normalized, endpoint_v = _noise_from_covariance(
        covariance,
        np.asarray(path.z_velocity, dtype=np.float64),
        np.asarray(path.z_position, dtype=np.float64),
    )
    path._bridge_states[duration] = (
        float(path.tau_eff_s) * endpoint_x_normalized,
        endpoint_v,
    )


def _ou_bridge_gain_and_covariance(
    h_left_s: float,
    h_right_s: float,
    tau_eff_s: float,
    thermal_velocity_variance_m2s2: float,
) -> tuple[np.ndarray, np.ndarray]:
    tau = float(tau_eff_s)
    theta = float(thermal_velocity_variance_m2s2)
    transition_right = _ou_transition_matrix(float(h_right_s), tau)
    covariance_left = _ou_normalized_covariance(float(h_left_s), tau, theta)
    covariance_right = _ou_normalized_covariance(float(h_right_s), tau, theta)
    cross_covariance = covariance_left @ transition_right.T
    end_covariance = (
        transition_right @ covariance_left @ transition_right.T + covariance_right
    )
    try:
        gain = np.linalg.solve(end_covariance.T, cross_covariance.T).T
    except np.linalg.LinAlgError:
        gain = cross_covariance @ np.linalg.pinv(end_covariance)
    conditional_covariance = covariance_left - gain @ cross_covariance.T
    return gain, 0.5 * (conditional_covariance + conditional_covariance.T)


def _sample_langevin_bridge_state(
    path: _IntegratedOuLeafPath,
    *,
    left_time_s: float,
    right_time_s: float,
    query_time_s: float,
    random_key: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample an exact integrated-OU bridge state between cached endpoints."""

    left_time = float(left_time_s)
    right_time = float(right_time_s)
    query_time = float(query_time_s)
    left_x, left_v = path._bridge_states[left_time]
    right_x, right_v = path._bridge_states[right_time]
    tau = float(path.tau_eff_s)
    theta = float(path.thermal_velocity_variance_m2s2)
    if theta <= 0.0:
        return np.zeros_like(left_x), np.zeros_like(left_v)

    h_left = query_time - left_time
    h_right = right_time - query_time
    transition_left = _ou_transition_matrix(h_left, tau)
    transition_right = _ou_transition_matrix(h_right, tau)

    left_state = np.stack(
        (
            np.asarray(left_x, dtype=np.float64) / tau,
            np.asarray(left_v, dtype=np.float64),
        ),
        axis=0,
    )
    right_state = np.stack(
        (
            np.asarray(right_x, dtype=np.float64) / tau,
            np.asarray(right_v, dtype=np.float64),
        ),
        axis=0,
    )
    state_shape = left_state.shape[1:]
    left_flat = left_state.reshape(2, -1)
    right_flat = right_state.reshape(2, -1)
    prior_mean = transition_left @ left_flat
    end_mean = transition_right @ prior_mean
    gain, conditional_covariance = _ou_bridge_gain_and_covariance(
        h_left,
        h_right,
        tau,
        theta,
    )
    conditional_mean = prior_mean + gain @ (right_flat - end_mean)

    if random_key is None:
        bridge_rng = path._bridge_rng
    else:
        seed = int(path.bridge_seed) % (1 << 64)
        if len(random_key) == 3 and int(random_key[0]) == 0:
            depth = int(random_key[1])
            numerator = int(random_key[2]) % (1 << 64)
            key_words = [
                seed & 0xFFFF_FFFF,
                seed >> 32,
                depth,
                numerator & 0xFFFF_FFFF,
                numerator >> 32,
            ]
        else:
            key_words = [seed & 0xFFFF_FFFF, seed >> 32]
            for value in random_key:
                key_value = int(value) % (1 << 64)
                key_words.extend((key_value & 0xFFFF_FFFF, key_value >> 32))
        bridge_rng = np.random.default_rng(np.random.SeedSequence(tuple(key_words)))
    z_velocity = bridge_rng.normal(size=left_flat.shape[1]).reshape(state_shape)
    z_position = bridge_rng.normal(size=left_flat.shape[1]).reshape(state_shape)
    noise_x_normalized, noise_v = _noise_from_covariance(
        conditional_covariance,
        z_velocity,
        z_position,
    )
    sampled = conditional_mean.reshape((2, *state_shape))
    sampled_x = tau * (sampled[0] + noise_x_normalized)
    sampled_v = sampled[1] + noise_v
    return np.asarray(sampled_x, dtype=np.float64), np.asarray(
        sampled_v, dtype=np.float64
    )


def _conditional_position_standard_deviation(
    path: PiecewiseLangevinPath,
    left_time_s: float,
    right_time_s: float,
    query_time_s: float,
) -> float:
    left = float(left_time_s)
    right = float(right_time_s)
    query = float(query_time_s)
    if not 0.0 <= left < query < right <= path.duration_s:
        raise ValueError("OU bridge variance requires an interior ordered query")
    leaf_index = int(np.searchsorted(path.leaf_end_times_s, query, side="left"))
    leaf_index = min(leaf_index, len(path._leaves) - 1)
    leaf_start = (
        0.0 if leaf_index == 0 else float(path.leaf_end_times_s[leaf_index - 1])
    )
    leaf_end = float(path.leaf_end_times_s[leaf_index])
    allowance = 64.0 * max(abs(np.spacing(leaf_start)), abs(np.spacing(leaf_end)))
    if left < leaf_start - allowance or right > leaf_end + allowance:
        raise ValueError("OU bridge variance interval crosses a coefficient leaf")
    leaf = path._leaves[leaf_index]
    _gain, covariance = _ou_bridge_gain_and_covariance(
        query - left,
        right - query,
        float(leaf.tau_eff_s),
        float(leaf.thermal_velocity_variance_m2s2),
    )
    normalized_variance = max(float(covariance[0, 0]), 0.0)
    return float(leaf.tau_eff_s) * np.sqrt(normalized_variance)


def _float64_bits(value: float) -> int:
    scalar = np.asarray(float(value), dtype=np.float64)
    return int(scalar.view(np.uint64))


def _materialize_keyed_bridge_state(
    path: PiecewiseLangevinPath,
    *,
    left_time_s: float,
    right_time_s: float,
    query_time_s: float,
    random_key: tuple[int, ...],
) -> bool:
    left = float(left_time_s)
    right = float(right_time_s)
    query = float(query_time_s)
    if not 0.0 <= left < query < right <= path.duration_s:
        return False
    leaf_index = int(np.searchsorted(path.leaf_end_times_s, query, side="left"))
    leaf_index = min(leaf_index, len(path._leaves) - 1)
    leaf_start = (
        0.0 if leaf_index == 0 else float(path.leaf_end_times_s[leaf_index - 1])
    )
    leaf_end = float(path.leaf_end_times_s[leaf_index])
    allowance = 64.0 * max(abs(np.spacing(leaf_start)), abs(np.spacing(leaf_end)))
    if left < leaf_start - allowance or right > leaf_end + allowance:
        return False
    leaf = path._leaves[leaf_index]
    _initialize_langevin_bridge(leaf)
    left_local = max(0.0, left - leaf_start)
    right_local = min(float(leaf.duration_s), right - leaf_start)
    query_local = query - leaf_start
    if query_local in leaf._bridge_states:
        return True
    if left_local not in leaf._bridge_states or right_local not in leaf._bridge_states:
        return False
    sampled = _sample_langevin_bridge_state(
        leaf,
        left_time_s=left_local,
        right_time_s=right_local,
        query_time_s=query_local,
        random_key=tuple(random_key),
    )
    leaf._bridge_states[query_local] = sampled
    return True


def _bridge_state_is_cached(
    path: PiecewiseLangevinPath,
    path_time_s: float,
) -> bool:
    path_time = float(path_time_s)
    if not 0.0 <= path_time <= path.duration_s:
        return False
    leaf_index = int(np.searchsorted(path.leaf_end_times_s, path_time, side="left"))
    leaf_index = min(leaf_index, len(path._leaves) - 1)
    leaf_start = (
        0.0 if leaf_index == 0 else float(path.leaf_end_times_s[leaf_index - 1])
    )
    leaf = path._leaves[leaf_index]
    _initialize_langevin_bridge(leaf)
    local_time = min(float(leaf.duration_s), max(0.0, path_time - leaf_start))
    return bool(local_time in leaf._bridge_states)


def _dyadic_address(elapsed_s: float, duration_s: float) -> tuple[int, int] | None:
    fraction = float(elapsed_s) / float(duration_s)
    for depth in range(1, _DYADIC_BRIDGE_MAX_DEPTH + 1):
        scaled = np.ldexp(fraction, depth)
        numerator = round(scaled)
        if abs(scaled - numerator) > 8.0 * abs(np.spacing(scaled)):
            continue
        while depth > 1 and numerator % 2 == 0:
            numerator //= 2
            depth -= 1
        return int(depth), int(numerator)
    return None


def _dyadic_time(path: _IntegratedOuLeafPath, depth: int, numerator: int) -> float:
    return float(path.duration_s) * float(numerator) / float(1 << int(depth))


def _materialize_dyadic_fraction(
    path: _IntegratedOuLeafPath,
    depth: int,
    numerator: int,
) -> tuple[np.ndarray, np.ndarray]:
    denominator = 1 << int(depth)
    if numerator <= 0:
        return path._bridge_states[0.0]
    if numerator >= denominator:
        return path._bridge_states[float(path.duration_s)]
    while depth > 1 and numerator % 2 == 0:
        numerator //= 2
        depth -= 1
    query_time = _dyadic_time(path, depth, numerator)
    cached = path._bridge_states.get(query_time)
    if cached is not None:
        return cached
    left = _materialize_dyadic_fraction(path, depth, numerator - 1)
    right = _materialize_dyadic_fraction(path, depth, numerator + 1)
    left_time = _dyadic_time(path, depth, numerator - 1)
    right_time = _dyadic_time(path, depth, numerator + 1)
    path._bridge_states[left_time] = left
    path._bridge_states[right_time] = right
    sampled = _sample_langevin_bridge_state(
        path,
        left_time_s=left_time,
        right_time_s=right_time,
        query_time_s=query_time,
        random_key=(0, int(depth), int(numerator)),
    )
    path._bridge_states[query_time] = sampled
    return sampled


def _evaluate_leaf_path(
    path: _IntegratedOuLeafPath,
    elapsed_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one cached or conditionally sampled state on the OU path."""

    elapsed = float(np.clip(float(elapsed_s), 0.0, float(path.duration_s)))
    _initialize_langevin_bridge(path)
    cached = path._bridge_states.get(elapsed)
    if cached is None:
        address = _dyadic_address(elapsed, float(path.duration_s))
        if address is not None:
            cached = _materialize_dyadic_fraction(path, *address)
            path._bridge_states[elapsed] = cached
            return (
                np.asarray(cached[0], dtype=np.float64).copy(),
                np.asarray(cached[1], dtype=np.float64).copy(),
            )
        times = sorted(path._bridge_states)
        upper = int(
            np.searchsorted(np.asarray(times, dtype=np.float64), elapsed, side="right")
        )
        left_time = float(times[max(0, upper - 1)])
        right_time = float(times[min(len(times) - 1, upper)])
        if not left_time < elapsed < right_time:
            raise RuntimeError("Langevin bridge could not bracket requested time")
        cached = _sample_langevin_bridge_state(
            path,
            left_time_s=left_time,
            right_time_s=right_time,
            query_time_s=elapsed,
        )
        path._bridge_states[elapsed] = cached
    return (
        np.asarray(cached[0], dtype=np.float64).copy(),
        np.asarray(cached[1], dtype=np.float64).copy(),
    )


@dataclass(frozen=True, slots=True)
class PiecewiseLangevinPath:
    """Saved integrated-OU realization over accepted deterministic leaves."""

    leaf_end_times_s: np.ndarray
    tau_eff_s: np.ndarray
    thermal_velocity_variance_m2s2: np.ndarray
    z_velocity: np.ndarray
    z_position: np.ndarray
    bridge_seeds: np.ndarray
    _leaves: tuple[_IntegratedOuLeafPath, ...] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        ends = np.asarray(self.leaf_end_times_s, dtype=np.float64)
        tau = np.asarray(self.tau_eff_s, dtype=np.float64)
        thermal = np.asarray(self.thermal_velocity_variance_m2s2, dtype=np.float64)
        zv = np.asarray(self.z_velocity, dtype=np.float64)
        zp = np.asarray(self.z_position, dtype=np.float64)
        seeds = np.asarray(self.bridge_seeds, dtype=np.int64)
        if not all((ends.ndim == 1, ends.size > 0, np.all(np.isfinite(ends)))):
            raise ValueError(
                "piecewise Langevin leaf_end_times_s must be a finite non-empty vector"
            )
        if not all((ends[0] > 0.0, np.all(np.diff(ends) > 0.0))):
            raise ValueError(
                "piecewise Langevin leaf times must be strictly increasing and positive"
            )
        leaf_count = int(ends.size)
        if not all((tau.shape == (leaf_count,), thermal.shape == (leaf_count,))):
            raise ValueError(
                "piecewise Langevin coefficient arrays must match leaf count"
            )
        if not np.all(np.isfinite(tau) & (tau > 0.0)):
            raise ValueError("piecewise Langevin tau_eff_s must be finite and positive")
        if not np.all(np.isfinite(thermal) & (thermal >= 0.0)):
            raise ValueError(
                "piecewise Langevin thermal variance must be finite and non-negative"
            )
        if not all((zv.ndim == 2, zp.shape == zv.shape, zv.shape[:1] == (leaf_count,))):
            raise ValueError(
                "piecewise Langevin normal arrays must have shape "
                "(leaf_count, dimension)"
            )
        if not all((np.all(np.isfinite(zv)), np.all(np.isfinite(zp)))):
            raise ValueError("piecewise Langevin normal arrays must be finite")
        if seeds.shape != (leaf_count,):
            raise ValueError("piecewise Langevin bridge seeds must match leaf count")
        starts = np.concatenate(([0.0], ends[:-1]))
        leaves = tuple(
            _IntegratedOuLeafPath(
                duration_s=float(ends[index] - starts[index]),
                tau_eff_s=float(tau[index]),
                thermal_velocity_variance_m2s2=float(thermal[index]),
                z_velocity=zv[index],
                z_position=zp[index],
                bridge_seed=int(seeds[index]),
            )
            for index in range(leaf_count)
        )
        object.__setattr__(self, "leaf_end_times_s", ends)
        object.__setattr__(self, "tau_eff_s", tau)
        object.__setattr__(self, "thermal_velocity_variance_m2s2", thermal)
        object.__setattr__(self, "z_velocity", zv)
        object.__setattr__(self, "z_position", zp)
        object.__setattr__(self, "bridge_seeds", seeds)
        object.__setattr__(self, "_leaves", leaves)

    @property
    def duration_s(self) -> float:
        return float(self.leaf_end_times_s[-1])

    def state_at(self, elapsed_s: float) -> tuple[np.ndarray, np.ndarray]:
        elapsed = float(np.clip(float(elapsed_s), 0.0, self.duration_s))
        dimension = int(self.z_velocity.shape[1])
        position = np.zeros(dimension, dtype=np.float64)
        velocity = np.zeros(dimension, dtype=np.float64)
        leaf_start = 0.0
        for index, leaf in enumerate(self._leaves):
            leaf_end = float(self.leaf_end_times_s[index])
            local_end = min(elapsed, leaf_end) - leaf_start
            if local_end <= 0.0:
                break
            noise_position, noise_velocity = _evaluate_leaf_path(leaf, local_end)
            decay = np.exp(-local_end / float(leaf.tau_eff_s))
            carry = float(leaf.tau_eff_s) * (1.0 - decay)
            position = position + carry * velocity + noise_position
            velocity = decay * velocity + noise_velocity
            if elapsed <= leaf_end:
                break
            leaf_start = leaf_end
        return position.copy(), velocity.copy()

    def transition(self, start_s: float, end_s: float) -> tuple[float, float]:
        """Return ``A,B`` for ``v1=A*v0`` and ``x1=x0+B*v0``."""

        start = float(np.clip(float(start_s), 0.0, self.duration_s))
        end = float(np.clip(float(end_s), start, self.duration_s))
        velocity_factor = 1.0
        position_factor = 0.0
        current = start
        while current < end:
            index = int(np.searchsorted(self.leaf_end_times_s, current, side="right"))
            index = min(index, len(self._leaves) - 1)
            stop = min(end, float(self.leaf_end_times_s[index]))
            local_dt = stop - current
            tau = float(self.tau_eff_s[index])
            decay = np.exp(-local_dt / tau)
            carry = tau * (1.0 - decay)
            position_factor += carry * velocity_factor
            velocity_factor *= decay
            if stop <= current:
                raise RuntimeError("piecewise Langevin transition failed to advance")
            current = stop
        return float(velocity_factor), float(position_factor)

    def replay(self, start_s: float, end_s: float) -> tuple[np.ndarray, np.ndarray]:
        """Return saved interval innovation ``Y(end)-Phi(end,start)Y(start)``."""

        start = float(np.clip(float(start_s), 0.0, self.duration_s))
        end = float(np.clip(float(end_s), start, self.duration_s))
        x_start, v_start = self.state_at(start)
        x_end, v_end = self.state_at(end)
        velocity_factor, position_factor = self.transition(start, end)
        return (
            x_end - x_start - position_factor * v_start,
            v_end - velocity_factor * v_start,
        )

    def endpoint_covariance(self) -> np.ndarray:
        """Return exact per-component covariance after all accepted leaves."""

        covariance = np.zeros((2, 2), dtype=np.float64)
        leaf_start = 0.0
        for index, leaf_end in enumerate(self.leaf_end_times_s):
            duration = float(leaf_end) - leaf_start
            tau = float(self.tau_eff_s[index])
            decay = np.exp(-duration / tau)
            carry = tau * (1.0 - decay)
            transition = np.asarray(((1.0, carry), (0.0, decay)), dtype=np.float64)
            var_x, var_v, cov_xv = _integrated_ou_covariances(
                duration,
                tau,
                float(self.thermal_velocity_variance_m2s2[index]),
            )
            noise = np.asarray(((var_x, cov_xv), (cov_xv, var_v)), dtype=np.float64)
            covariance = transition @ covariance @ transition.T + noise
            leaf_start = float(leaf_end)
        return covariance
