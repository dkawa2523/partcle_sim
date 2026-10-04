"""Adaptive wall-crossing search on the saved dyadic OU bridge."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import pairwise

import numpy as np

from particle_tracer_unified.domain import BoundaryHit, BoundaryQuery

from ._segment_motion_scalar import SegmentMotionTrace
from ._stochastic_path import (
    _DYADIC_BRIDGE_MAX_DEPTH,
    PiecewiseLangevinPath,
    _bridge_state_is_cached,
    _conditional_position_standard_deviation,
    _dyadic_address,
    _float64_bits,
    _materialize_keyed_bridge_state,
)

_CLEARANCE_SIGMA_MULTIPLIER = 8.0


@dataclass(frozen=True, slots=True)
class StochasticCompositionResult:
    stage_points: np.ndarray
    prefetched_hit: BoundaryHit | None
    unresolved: bool


@dataclass(slots=True)
class _DyadicFirstPassageSearch:
    path: PiecewiseLangevinPath
    deterministic_trace: SegmentMotionTrace
    boundary_service: BoundaryQuery[object]
    stochastic_offset_s: float
    duration_s: float
    geometry_tolerance_m: float
    inside_reference_m: np.ndarray
    positions_by_time: dict[float, np.ndarray]

    def position_at(self, path_time_s: float) -> np.ndarray:
        path_time = float(path_time_s)
        cached = self.positions_by_time.get(path_time)
        if cached is not None:
            return cached
        elapsed = path_time - float(self.stochastic_offset_s)
        deterministic_position, _velocity = self.deterministic_trace.state_at(elapsed)
        noise_position, _noise_velocity = self.path.replay(
            float(self.stochastic_offset_s), path_time
        )
        dimension = int(self.deterministic_trace.request.spatial_dim)
        position = (
            np.asarray(deterministic_position, dtype=np.float64)[:dimension]
            + np.asarray(noise_position, dtype=np.float64)[:dimension]
        )
        self.positions_by_time[path_time] = position
        return position

    def clearance(self, position_m: np.ndarray) -> float:
        projection = self.boundary_service.nearest_projection(
            np.asarray(position_m, dtype=np.float64),
            np.asarray(self.inside_reference_m, dtype=np.float64),
        )
        if projection is None:
            return float("nan")
        return float(
            np.linalg.norm(
                np.asarray(position_m, dtype=np.float64)
                - np.asarray(projection.position, dtype=np.float64)
            )
        )

    def interval_is_safe(
        self,
        *,
        left_time_s: float,
        middle_time_s: float,
        right_time_s: float,
        positions: tuple[np.ndarray, np.ndarray, np.ndarray],
        child_hits: tuple[BoundaryHit | None, BoundaryHit | None],
    ) -> bool:
        if any(hit is not None for hit in child_hits):
            return False
        if not all(self.boundary_service.inside(point) for point in positions):
            return False
        clearances = np.asarray(
            [self.clearance(point) for point in positions], dtype=np.float64
        )
        if np.any(~np.isfinite(clearances)):
            return False
        sigma_x = _conditional_position_standard_deviation(
            self.path,
            float(left_time_s),
            float(right_time_s),
            float(middle_time_s),
        )
        threshold = float(self.geometry_tolerance_m) + (
            _CLEARANCE_SIGMA_MULTIPLIER * float(sigma_x)
        )
        return bool(float(np.min(clearances)) > threshold)

    def global_hit(
        self,
        hit: BoundaryHit,
        left_time_s: float,
        right_time_s: float,
        *,
        root_left_time_s: float,
        root_right_time_s: float,
    ) -> BoundaryHit | None:
        local_alpha = float(np.clip(float(hit.alpha_hint), 0.0, 1.0))
        path_time = float(left_time_s) + local_alpha * (
            float(right_time_s) - float(left_time_s)
        )
        if 0.0 < local_alpha < 1.0 and not _materialize_keyed_bridge_state(
            self.path,
            left_time_s=float(left_time_s),
            right_time_s=float(right_time_s),
            query_time_s=path_time,
            random_key=(
                2,
                _float64_bits(root_left_time_s),
                _float64_bits(root_right_time_s),
                _float64_bits(left_time_s),
                _float64_bits(right_time_s),
                _float64_bits(path_time),
            ),
        ):
            return None
        elapsed = path_time - float(self.stochastic_offset_s)
        return replace(
            hit,
            alpha_hint=float(np.clip(elapsed / float(self.duration_s), 0.0, 1.0)),
        )

    def search_interval(
        self,
        left_time_s: float,
        right_time_s: float,
        *,
        root_left_time_s: float,
        root_right_time_s: float,
        node_depth: int = 1,
        node_numerator: int = 1,
    ) -> tuple[BoundaryHit | None, bool]:
        left_time = float(left_time_s)
        right_time = float(right_time_s)
        middle_time = left_time + 0.5 * (right_time - left_time)
        canonical_depth = _path_dyadic_depth(self.path, middle_time)
        if canonical_depth is None and not _materialize_keyed_bridge_state(
            self.path,
            left_time_s=left_time,
            right_time_s=right_time,
            query_time_s=middle_time,
            random_key=(
                1,
                _float64_bits(root_left_time_s),
                _float64_bits(root_right_time_s),
                int(node_depth),
                int(node_numerator),
            ),
        ):
            return None, True
        left_position = self.position_at(left_time)
        middle_position = self.position_at(middle_time)
        right_position = self.position_at(right_time)
        left_hit = self.boundary_service.first_hit(left_position, middle_position)
        right_hit = self.boundary_service.first_hit(middle_position, right_position)
        child_hits = (left_hit, right_hit)
        if self.interval_is_safe(
            left_time_s=left_time,
            middle_time_s=middle_time,
            right_time_s=right_time,
            positions=(left_position, middle_position, right_position),
            child_hits=child_hits,
        ):
            return None, False
        reached_depth_limit = (
            canonical_depth >= _DYADIC_BRIDGE_MAX_DEPTH
            if canonical_depth is not None
            else node_depth >= _DYADIC_BRIDGE_MAX_DEPTH
        )
        if reached_depth_limit:
            if left_hit is not None:
                resolved_hit = self.global_hit(
                    left_hit,
                    left_time,
                    middle_time,
                    root_left_time_s=root_left_time_s,
                    root_right_time_s=root_right_time_s,
                )
                return resolved_hit, resolved_hit is None
            if right_hit is not None:
                resolved_hit = self.global_hit(
                    right_hit,
                    middle_time,
                    right_time,
                    root_left_time_s=root_left_time_s,
                    root_right_time_s=root_right_time_s,
                )
                return resolved_hit, resolved_hit is None
            return None, True
        hit, unresolved = self.search_interval(
            left_time,
            middle_time,
            root_left_time_s=root_left_time_s,
            root_right_time_s=root_right_time_s,
            node_depth=node_depth + 1,
            node_numerator=2 * node_numerator - 1,
        )
        if hit is not None or unresolved:
            return hit, unresolved
        return self.search_interval(
            middle_time,
            right_time,
            root_left_time_s=root_left_time_s,
            root_right_time_s=root_right_time_s,
            node_depth=node_depth + 1,
            node_numerator=2 * node_numerator + 1,
        )


def _path_dyadic_depth(
    path: PiecewiseLangevinPath,
    path_time_s: float,
) -> int | None:
    path_time = float(path_time_s)
    leaf_index = int(np.searchsorted(path.leaf_end_times_s, path_time, side="left"))
    leaf_index = min(leaf_index, len(path._leaves) - 1)
    leaf_start = (
        0.0 if leaf_index == 0 else float(path.leaf_end_times_s[leaf_index - 1])
    )
    leaf = path._leaves[leaf_index]
    address = _dyadic_address(path_time - leaf_start, float(leaf.duration_s))
    return None if address is None else int(address[0])


def _path_breakpoints(
    path: PiecewiseLangevinPath,
    start_time_s: float,
    end_time_s: float,
) -> np.ndarray:
    starts = np.asarray([float(start_time_s)], dtype=np.float64)
    internal = np.asarray(path.leaf_end_times_s, dtype=np.float64)
    internal = internal[
        (internal > float(start_time_s)) & (internal < float(end_time_s))
    ]
    ends = np.asarray([float(end_time_s)], dtype=np.float64)
    return np.concatenate((starts, internal, ends))


def _ensure_canonical_breakpoint(
    path: PiecewiseLangevinPath,
    path_time_s: float,
) -> bool:
    if _bridge_state_is_cached(path, float(path_time_s)):
        return True
    if _path_dyadic_depth(path, float(path_time_s)) is None:
        return False
    path.state_at(float(path_time_s))
    return _bridge_state_is_cached(path, float(path_time_s))


def _unresolved_result(
    deterministic_trace: SegmentMotionTrace,
) -> StochasticCompositionResult:
    return StochasticCompositionResult(
        stage_points=np.empty((0, deterministic_trace.request.spatial_dim)),
        prefetched_hit=None,
        unresolved=True,
    )


def _invalid_search_interval(
    path: PiecewiseLangevinPath,
    *,
    offset_s: float,
    duration_s: float,
) -> bool:
    return bool(
        duration_s <= 0.0 or offset_s < 0.0 or offset_s + duration_s > path.duration_s
    )


def search_piecewise_langevin_wall_crossing(
    *,
    path: PiecewiseLangevinPath,
    deterministic_trace: SegmentMotionTrace,
    boundary_service: BoundaryQuery[object],
    geometry_tolerance_m: float,
    stochastic_offset_s: float = 0.0,
) -> StochasticCompositionResult:
    """Search the saved OU path on an adaptive, clearance-bounded dyadic tree."""

    duration = float(deterministic_trace.request.duration_s)
    offset = float(stochastic_offset_s)
    end_time = offset + duration
    tolerance = max(float(geometry_tolerance_m), 0.0)
    if _invalid_search_interval(path, offset_s=offset, duration_s=duration):
        return _unresolved_result(deterministic_trace)
    breakpoints = _path_breakpoints(path, offset, end_time)
    if any(not _ensure_canonical_breakpoint(path, time_s) for time_s in breakpoints):
        return _unresolved_result(deterministic_trace)
    start_position, _start_velocity = deterministic_trace.state_at(0.0)
    search = _DyadicFirstPassageSearch(
        path=path,
        deterministic_trace=deterministic_trace,
        boundary_service=boundary_service,
        stochastic_offset_s=offset,
        duration_s=duration,
        geometry_tolerance_m=tolerance,
        inside_reference_m=np.asarray(start_position, dtype=np.float64),
        positions_by_time={},
    )
    search.position_at(offset)
    prefetched_hit = None
    unresolved = False
    for left_time, right_time in pairwise(breakpoints):
        left = float(left_time)
        right = float(right_time)
        prefetched_hit, unresolved = search.search_interval(
            left,
            right,
            root_left_time_s=left,
            root_right_time_s=right,
        )
        if prefetched_hit is not None or unresolved:
            break
    retained_times = sorted(
        time_s for time_s in search.positions_by_time if time_s > offset
    )
    stage_points = np.asarray(
        [search.positions_by_time[time_s] for time_s in retained_times],
        dtype=np.float64,
    )
    return StochasticCompositionResult(
        stage_points=stage_points,
        prefetched_hit=prefetched_hit,
        unresolved=bool(unresolved),
    )
