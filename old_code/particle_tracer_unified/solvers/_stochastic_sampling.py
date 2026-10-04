"""Build stochastic paths from accepted motion leaves and OU coefficients."""

from __future__ import annotations

import numpy as np

from ._stochastic_coefficients import K_BOLTZMANN, resolve_coefficient_plans
from ._stochastic_config import StochasticMotionConfig
from ._stochastic_path import PiecewiseLangevinPath
from ._stochastic_randomness import (
    BrownianRandomContext,
    draw_particle_path_randomness,
)
from ._stochastic_temperature import ParticleLeafPlan
from .integrator_common import (
    DRAG_MODEL_EPSTEIN,
    DRAG_MODEL_NONE,
    DRAG_MODEL_STOKES,
    DRAG_MODEL_STOKES_CUNNINGHAM,
    drag_model_name_from_mode,
)
from .segment_motion import SegmentMotionBatchTrace

_LINEAR_BROWNIAN_DRAG_MODES = frozenset(
    (DRAG_MODEL_STOKES, DRAG_MODEL_STOKES_CUNNINGHAM, DRAG_MODEL_EPSTEIN)
)


def _sampling_disabled(
    config: StochasticMotionConfig,
    motion_batch: SegmentMotionBatchTrace,
    particle_indices: np.ndarray,
) -> bool:
    return bool(
        not config.enabled
        or float(motion_batch.request.duration_s) <= 0.0
        or particle_indices.size == 0
    )


def _require_linear_brownian_drag(drag_model_mode: int) -> None:
    mode = int(drag_model_mode)
    if mode == int(DRAG_MODEL_NONE):
        raise ValueError(
            "Brownian motion requires drag; drag_model=none is unsupported"
        )
    if mode not in _LINEAR_BROWNIAN_DRAG_MODES:
        name = drag_model_name_from_mode(mode)
        raise ValueError(
            "underdamped Brownian motion requires a slip-independent linear drag "
            f"law; drag_model={name} is velocity-dependent"
        )


def _draw_path_randomness(
    rng: np.random.Generator,
    total_leaves: int,
    dimension: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw velocity, position, then bridge seeds in the stable RNG order."""

    shape = (int(total_leaves), int(dimension))
    z_velocity = rng.normal(size=shape)
    z_position = rng.normal(size=shape)
    bridge_seeds = rng.integers(
        0,
        np.iinfo(np.int64).max,
        size=int(total_leaves),
        dtype=np.int64,
    )
    return z_velocity, z_position, bridge_seeds


def _draw_keyed_path_randomness(
    *,
    plans: list[ParticleLeafPlan],
    context: BrownianRandomContext,
    seed: int,
    dimension: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    particle_ids = np.asarray(context.particle_id, dtype=np.int64)
    cohort_indices = np.asarray(context.cohort_index, dtype=np.int64)
    draws = [
        draw_particle_path_randomness(
            seed=int(seed),
            particle_id=int(particle_ids[plan.particle_index]),
            macro_step_index=int(context.macro_step_index),
            cohort_index=int(cohort_indices[plan.particle_index]),
            leaf_count=int(plan.leaf_end_times_s.size),
            dimension=int(dimension),
        )
        for plan in plans
    ]
    return (
        np.concatenate([draw[0] for draw in draws], axis=0),
        np.concatenate([draw[1] for draw in draws], axis=0),
        np.concatenate([draw[2] for draw in draws], axis=0),
    )


def _sample_path_randomness(
    *,
    plans: list[ParticleLeafPlan],
    rng: np.random.Generator | None,
    context: BrownianRandomContext | None,
    seed: int,
    dimension: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total_leaves = sum(int(plan.leaf_end_times_s.size) for plan in plans)
    if context is not None:
        return _draw_keyed_path_randomness(
            plans=plans,
            context=context,
            seed=int(seed),
            dimension=int(dimension),
        )
    if rng is None:
        raise ValueError("Brownian sampling requires a random source")
    return _draw_path_randomness(rng, total_leaves, dimension)


def _build_paths(
    plans: list[ParticleLeafPlan],
    temperatures: list[np.ndarray],
    effective_masses: list[np.ndarray],
    z_velocity: np.ndarray,
    z_position: np.ndarray,
    bridge_seeds: np.ndarray,
) -> tuple[dict[int, PiecewiseLangevinPath], list[np.ndarray], list[np.ndarray]]:
    paths: dict[int, PiecewiseLangevinPath] = {}
    all_tau: list[np.ndarray] = []
    all_temperature: list[np.ndarray] = []
    offset = 0
    for plan, temperature, effective_mass in zip(
        plans, temperatures, effective_masses, strict=True
    ):
        leaf_count = int(plan.leaf_end_times_s.size)
        slc = slice(offset, offset + leaf_count)
        thermal = (
            K_BOLTZMANN
            * np.asarray(temperature, dtype=np.float64)
            / np.asarray(effective_mass, dtype=np.float64)
        )
        paths[int(plan.particle_index)] = PiecewiseLangevinPath(
            leaf_end_times_s=plan.leaf_end_times_s,
            tau_eff_s=plan.tau_mid_s,
            thermal_velocity_variance_m2s2=thermal,
            z_velocity=z_velocity[slc],
            z_position=z_position[slc],
            bridge_seeds=bridge_seeds[slc],
        )
        all_tau.append(plan.tau_mid_s)
        all_temperature.append(np.asarray(temperature, dtype=np.float64))
        offset += leaf_count
    return paths, all_tau, all_temperature


def _path_diagnostics(
    *,
    paths: dict[int, PiecewiseLangevinPath],
    particle_indices: np.ndarray,
    total_leaves: int,
    temperatures: list[np.ndarray],
    tau_values: list[np.ndarray],
    sampling_s: float,
    sampled_points: int,
    sample_calls: int,
) -> dict[str, object]:
    endpoint_velocity = np.asarray(
        [
            paths[int(index)].state_at(paths[int(index)].duration_s)[1]
            for index in particle_indices
        ],
        dtype=np.float64,
    )
    component_count = int(endpoint_velocity.size)
    sum_sq = float(np.sum(endpoint_velocity * endpoint_velocity))
    sigma_values = np.asarray(
        [
            np.sqrt(paths[int(index)].endpoint_covariance()[1, 1])
            for index in particle_indices
        ],
        dtype=np.float64,
    )
    return {
        "applied": True,
        "particle_count": int(particle_indices.size),
        "leaf_count": int(total_leaves),
        "component_count": component_count,
        "sum_sq": sum_sq,
        "rms_velocity_kick_mps": float(np.sqrt(sum_sq / component_count))
        if component_count
        else 0.0,
        "mean_sigma_v_mps": float(np.mean(sigma_values)) if sigma_values.size else 0.0,
        "max_sigma_v_mps": float(np.max(sigma_values)) if sigma_values.size else 0.0,
        "mean_temperature_K": float(np.mean(np.concatenate(temperatures))),
        "mean_tau_eff_s": float(np.mean(np.concatenate(tau_values))),
        "field_sampling_s": float(sampling_s),
        "field_sample_point_count": int(sampled_points),
        "field_sample_call_count": int(sample_calls),
    }


def sample_piecewise_langevin_paths(
    *,
    config: StochasticMotionConfig,
    rng: np.random.Generator | None,
    motion_batch: SegmentMotionBatchTrace,
    particle_indices: np.ndarray,
    minimum_substeps: np.ndarray,
    particle_mass: np.ndarray,
    gas_temperature_K: float,
    collect_diagnostics: bool = False,
    _random_context: BrownianRandomContext | None = None,
) -> tuple[dict[int, PiecewiseLangevinPath], dict[str, object]]:
    """Build one saved local-coefficient path after accepted leaves are final."""

    indices = np.asarray(particle_indices, dtype=np.int64)
    if _sampling_disabled(config, motion_batch, indices):
        return {}, {"applied": False, "particle_count": int(indices.size)}
    _require_linear_brownian_drag(int(motion_batch.request.drag_model_mode))
    sampled_plans, sampling_s, sampled_points, sample_calls = resolve_coefficient_plans(
        config=config,
        motion_batch=motion_batch,
        particle_indices=indices,
        minimum_substeps=minimum_substeps,
        particle_mass=np.asarray(particle_mass, dtype=np.float64),
        gas_temperature_K=float(gas_temperature_K),
        collect_diagnostics=bool(collect_diagnostics),
    )
    if not sampled_plans:
        return {}, {"applied": False, "particle_count": 0}
    plans = [sampled.plan for sampled in sampled_plans]
    resolved_indices = np.asarray(
        [plan.particle_index for plan in plans], dtype=np.int64
    )
    temperatures = [sampled.temperatures_K for sampled in sampled_plans]
    effective_masses = [sampled.effective_masses_kg for sampled in sampled_plans]
    total_leaves = int(sum(plan.leaf_end_times_s.size for plan in plans))
    dimension = int(motion_batch.request.spatial_dim)
    randomness = _sample_path_randomness(
        plans=plans,
        rng=rng,
        context=_random_context,
        seed=int(config.seed),
        dimension=dimension,
    )
    paths, tau_values, temperature_values = _build_paths(
        plans, temperatures, effective_masses, *randomness
    )
    if not collect_diagnostics:
        return paths, {"applied": True}
    return paths, _path_diagnostics(
        paths=paths,
        particle_indices=resolved_indices,
        total_leaves=total_leaves,
        temperatures=temperature_values,
        tau_values=tau_values,
        sampling_s=float(sampling_s),
        sampled_points=int(sampled_points),
        sample_calls=int(sample_calls),
    )
