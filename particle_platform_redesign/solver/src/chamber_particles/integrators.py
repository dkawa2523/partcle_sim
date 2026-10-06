"""Batched fixed-step proposals independent of fields, geometry, and output."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from numba import njit
from numpy.typing import NDArray

from .numerical_status import (
    INTEGRATOR_NUMERICAL_FAILURE,
    NUMERICAL_STATUS_OK,
)

type FloatArray = NDArray[np.float64]
type Int64Array = NDArray[np.int64]
type BoolArray = NDArray[np.bool_]
type UInt8Array = NDArray[np.uint8]
type RowSelection = Int64Array | slice
type PathKind = Literal[
    "linear_exact",
    "quadratic_exact",
    "rk4_dense",
    "exponential_midpoint_reintegrated",
    "cubic_hermite",
]

STEP_PROPOSAL_REVISION = "coupled_fixed_step_proposal_v10"
RK4_ENCLOSURE_REVISION = "rk4_global_abs_enclosure_v2"
RK4_DENSE_PATH_REVISION = "rk4_position_hermite_state_extension_v3"
EXPONENTIAL_MIDPOINT_REVISION = "charge_stable_exponential_midpoint_v3"
EXPONENTIAL_MIDPOINT_ENCLOSURE_REVISION = "exponential_midpoint_local_stage_enclosure_v4"
# With unit roundoff u=eps/2, 16*eps=32u exceeds the dense expression's gamma_6,
# the scale construction's gamma_5, and the O(u^2) rounded-turn displacement.
_QUADRATIC_ROUNDOFF_EPS_FACTOR = 16.0
# The supported RK4 and exponential secant, endpoint, and endpoint chord
# expressions use fewer than 32 rounded operations per component.  The factor
# covers both dense values and the chord construction while keeping arithmetic
# error separate from the geometry tolerance.
_CURVED_CHORD_ROUNDOFF_EPS_FACTOR = 64.0
# The frozen-start predictor contains cancellation-prone ``expm1`` and affine
# combinations. Its local field box must contain every shortened predictor
# evaluated by the same compiled kernel, not only the exact mathematical path.
_FROZEN_PREDICTOR_ROUNDOFF_EPS_FACTOR = 128.0
# With unit roundoff u=eps/2, 8*eps=16u covers the final world-coordinate
# translation of the dense value, both translated chord endpoints, and the
# four rounded operations in the public endpoint-chord expression.  Relative
# Bezier/restriction arithmetic is covered separately below at its own scale.
_RK4_DENSE_WORLD_CHORD_EPS_FACTOR = 8.0
_SMALL_RELAXATION_ARGUMENT = 1.0e-3


@dataclass(frozen=True, slots=True)
class DynamicsEvaluation:
    """Stage acceleration, charge rate, and independent validity verdicts."""

    acceleration_m_s2: FloatArray
    charge_rate_number_s: FloatArray
    support_inside: BoolArray
    applicability_inside: BoolArray
    numerical_status: UInt8Array
    field_cell_id: Int64Array | None = None


class StageEvaluator(Protocol):
    """Batch callback evaluated at the actual state of every RK stage."""

    def __call__(
        self,
        particle_index: Int64Array,
        time_s: FloatArray,
        position_m: FloatArray,
        velocity_m_s: FloatArray,
        charge_number: FloatArray,
    ) -> DynamicsEvaluation: ...


@dataclass(frozen=True, slots=True)
class RelaxationEvaluation:
    """Linear relaxation and additive terms at one exponential stage."""

    linear_drag_rate_s_inv: FloatArray
    target_velocity_m_s: FloatArray
    additive_acceleration_m_s2: FloatArray
    charge_rate_number_s: FloatArray
    charge_rate_derivative_s_inv: FloatArray
    support_inside: BoolArray
    applicability_inside: BoolArray
    numerical_status: UInt8Array
    field_cell_id: Int64Array | None = None


class RelaxationStageEvaluator(Protocol):
    """Batch callback returning the decomposition required by exponential midpoint."""

    def __call__(
        self,
        particle_index: Int64Array,
        time_s: FloatArray,
        position_m: FloatArray,
        velocity_m_s: FloatArray,
        charge_number: FloatArray,
    ) -> RelaxationEvaluation: ...


class AccelerationAbsBounder(Protocol):
    """Return componentwise acceleration bounds valid at every field position.

    ``velocity_abs_upper_m_s`` is a nonnegative componentwise upper bound.  The
    returned array must bound ``abs(acceleration)`` for every velocity inside
    that box and for every spatial value the prepared field sampler can return.
    """

    def __call__(
        self,
        particle_index: Int64Array,
        velocity_abs_upper_m_s: FloatArray,
    ) -> tuple[FloatArray, UInt8Array]: ...


@dataclass(frozen=True, slots=True)
class CurvedPathEnclosure:
    """Componentwise enclosure of every shortened curved proposal state."""

    position_lower_m: FloatArray
    position_upper_m: FloatArray
    velocity_lower_m_s: FloatArray
    velocity_upper_m_s: FloatArray
    numerical_status: UInt8Array


@dataclass(frozen=True, slots=True)
class Rk4DenseEnclosure:
    """Componentwise enclosure of one root RK4 dense-path subinterval.

    The velocity interval contains both the dense physical velocity state and
    the physical-time derivative of the dense position polynomial.  Event
    localization may therefore use it as a path-speed bound even though these
    two cubic continuous-extension quantities are not identical internally.
    """

    position_lower_m: FloatArray
    position_upper_m: FloatArray
    velocity_lower_m_s: FloatArray
    velocity_upper_m_s: FloatArray
    charge_lower_number: FloatArray
    charge_upper_number: FloatArray
    position_control_origin_m: FloatArray
    relative_position_control_lower_m: FloatArray
    relative_position_control_upper_m: FloatArray
    numerical_status: UInt8Array


@dataclass(frozen=True, slots=True)
class _PositionInterval:
    """Componentwise interval owned by the integrator's path arithmetic."""

    lower_m: FloatArray
    upper_m: FloatArray


@dataclass(frozen=True, slots=True)
class ProposalSample:
    """Rows of one path sample or transient predictor state at a physical time."""

    particle_index: Int64Array
    position_m: FloatArray
    velocity_m_s: FloatArray
    charge_number: FloatArray
    support_inside: BoolArray
    applicability_inside: BoolArray
    numerical_status: UInt8Array


@dataclass(frozen=True, slots=True)
class FrozenStartPredictorEnclosure:
    """Frozen-start predictor endpoint and all-shortened-time enclosure."""

    sample: ProposalSample
    path: CurvedPathEnclosure
    charge_lower_number: FloatArray
    charge_upper_number: FloatArray


@dataclass(frozen=True, slots=True)
class _FrozenStartPredictorData:
    """One start evaluation shared by the predictor and its enclosure."""

    sample: ProposalSample
    elapsed_s: FloatArray
    target_velocity_m_s: FloatArray
    additive_acceleration_m_s2: FloatArray
    charge_rate_number_s: FloatArray


@dataclass(frozen=True, slots=True)
class _AdvanceResult:
    position_m: FloatArray
    velocity_m_s: FloatArray
    charge_number: FloatArray
    support_inside: BoolArray
    applicability_inside: BoolArray
    numerical_status: UInt8Array
    end_field_cell_id: Int64Array | None
    rk4_dense_controls: _Rk4DenseControls | None = None


@dataclass(frozen=True, slots=True)
class _CubicHermitePath:
    """Original endpoint data defining one immutable Hermite polynomial per row."""

    start_time_s: FloatArray
    target_time_s: FloatArray
    start_position_m: FloatArray
    start_velocity_m_s: FloatArray
    end_position_m: FloatArray
    end_velocity_m_s: FloatArray
    start_charge_number: FloatArray
    end_charge_number: FloatArray
    charge_root_time_s: FloatArray | None = None
    charge_root_number: FloatArray | None = None
    charge_affine_rate_number_s: FloatArray | None = None
    charge_rate_derivative_s_inv: FloatArray | None = None


@dataclass(frozen=True, slots=True)
class _Rk4DensePath:
    """Immutable Bernstein form of the classical-RK4 state path.

    Position is the cubic Hermite path owned jointly by the RK4 endpoint
    positions and physical endpoint velocities, exactly matching the event
    locator's path.  Velocity and charge retain cubic endpoint interpolants
    formed from their first/fourth RK stage rates.  Every endpoint is the
    unchanged fourth-order classical-RK4 endpoint.
    """

    start_time_s: FloatArray
    target_time_s: FloatArray
    position_controls_m: FloatArray
    velocity_controls_m_s: FloatArray
    charge_controls_number: FloatArray


@dataclass(frozen=True, slots=True)
class _Rk4DenseControls:
    """Mutable construction result kept private to one RK4 advance."""

    position_inner_m: FloatArray
    velocity_inner_m_s: FloatArray
    charge_inner_number: FloatArray


@dataclass(frozen=True, slots=True)
class StepProposal:
    """One fixed-step candidate with a reproducible physical-time evaluator."""

    particle_index: Int64Array
    start_time_s: FloatArray
    target_time_s: FloatArray
    start_position_m: FloatArray
    start_velocity_m_s: FloatArray
    start_charge_number: FloatArray
    end_position_m: FloatArray
    end_velocity_m_s: FloatArray
    end_charge_number: FloatArray
    end_field_cell_id: Int64Array | None
    support_inside: BoolArray
    applicability_inside: BoolArray
    numerical_status: UInt8Array
    path_kind: PathKind
    constant_acceleration_m_s2: FloatArray | None
    path_enclosure: CurvedPathEnclosure | None
    _evaluator: StageEvaluator | None
    _relaxation_evaluator: RelaxationStageEvaluator | None
    _cubic_hermite_path: _CubicHermitePath | None = None
    _rk4_dense_path: _Rk4DensePath | None = None

    def state_at(self, time_s: float) -> ProposalSample:
        """Evaluate selected rows on the proposal path without committing state."""

        if not np.isfinite(time_s) or (
            self.target_time_s.size and time_s > float(np.max(self.target_time_s))
        ):
            raise ValueError("proposal sample time is outside its upper bound")
        selected = np.flatnonzero(
            (self.start_time_s <= time_s) & (time_s <= self.target_time_s)
        ).astype("<i8", copy=False)
        return self.state_at_rows(time_s, selected)

    def state_at_rows(self, time_s: float, local_rows: Int64Array) -> ProposalSample:
        """Evaluate selected proposal rows at one time without committing state."""

        selected = np.asarray(local_rows, dtype=np.int64)
        if (
            not np.isfinite(time_s)
            or selected.ndim != 1
            or bool((selected < 0).any())
            or bool((selected >= self.particle_index.size).any())
        ):
            raise ValueError("proposal sample rows are invalid")
        if bool(
            ((self.start_time_s[selected] > time_s) | (time_s > self.target_time_s[selected])).any()
        ):
            raise ValueError("proposal sample is outside a selected row interval")
        if self.path_kind == "cubic_hermite":
            path = self._cubic_hermite_path
            if path is None:
                raise ValueError("cubic Hermite proposal has no original path")
            sample_time_s = np.full(selected.size, time_s, dtype=np.float64)
            position_m, velocity_m_s = _evaluate_cubic_hermite(path, selected, sample_time_s)
            charge_number = _evaluate_cubic_charge(path, selected, sample_time_s)
            return ProposalSample(
                particle_index=self.particle_index[selected],
                position_m=position_m,
                velocity_m_s=velocity_m_s,
                charge_number=charge_number,
                support_inside=self.support_inside[selected].copy(),
                applicability_inside=self.applicability_inside[selected].copy(),
                numerical_status=self.numerical_status[selected].copy(),
            )
        if self.path_kind == "rk4_dense":
            sample_time_s = np.full(selected.size, time_s, dtype=np.float64)
            return self.rk4_dense_state_at_rows(selected, sample_time_s)
        elapsed = time_s - self.start_time_s[selected]
        acceleration = (
            None
            if self.constant_acceleration_m_s2 is None
            else self.constant_acceleration_m_s2[selected]
        )
        result = _advance(
            self.path_kind,
            self._evaluator,
            self._relaxation_evaluator,
            self.particle_index[selected],
            self.start_time_s[selected],
            elapsed,
            self.start_position_m[selected],
            self.start_velocity_m_s[selected],
            self.start_charge_number[selected],
            acceleration,
        )
        return ProposalSample(
            particle_index=self.particle_index[selected],
            position_m=result.position_m,
            velocity_m_s=result.velocity_m_s,
            charge_number=result.charge_number,
            support_inside=result.support_inside,
            applicability_inside=result.applicability_inside,
            numerical_status=result.numerical_status,
        )

    def start_position(self) -> FloatArray:
        """Return the particle-local accepted state at the proposal start."""

        return self.start_position_m.copy()

    def end_position(self) -> FloatArray:
        """Return the candidate position at the macro-step boundary."""

        return self.end_position_m.copy()

    def end_velocity(self) -> FloatArray:
        """Return the candidate velocity at the macro-step boundary."""

        return self.end_velocity_m_s.copy()

    def end_charge(self) -> FloatArray:
        """Return the candidate charge number at the macro-step boundary."""

        return self.end_charge_number.copy()

    def cubic_charge_interval(self, local_rows: Int64Array) -> tuple[FloatArray, FloatArray]:
        """Return the exact monotone charge interval for selected Hermite rows.

        A deterministic Hermite path retains its endpoint-linear extension.
        A Langevin path supplies one root-owned affine relaxation with a
        nonpositive Jacobian, whose derivative keeps a constant sign.
        Endpoint minima and maxima are therefore exact in both cases.
        """

        selected = np.asarray(local_rows, dtype=np.int64)
        path = self._cubic_hermite_path
        if (
            self.path_kind != "cubic_hermite"
            or path is None
            or selected.ndim != 1
            or bool((selected < 0).any())
            or bool((selected >= self.particle_index.size).any())
        ):
            raise ValueError("cubic Hermite charge interval rows are invalid")
        start_charge = path.start_charge_number[selected]
        end_charge = path.end_charge_number[selected]
        return np.minimum(start_charge, end_charge), np.maximum(start_charge, end_charge)

    def quadratic_position_interval(self) -> _PositionInterval:
        """Enclose the exact quadratic path using the dense-state arithmetic."""

        if self.path_kind != "quadratic_exact" or self.constant_acceleration_m_s2 is None:
            raise ValueError("quadratic position interval requires an exact quadratic proposal")
        duration_s = self.target_time_s - self.start_time_s
        return _enclose_quadratic_position(
            self.start_position_m,
            self.start_velocity_m_s,
            self.constant_acceleration_m_s2,
            duration_s,
            self.end_position_m,
        )

    def rk4_dense_subinterval(
        self,
        local_rows: Int64Array,
        start_time_s: FloatArray,
        target_time_s: FloatArray,
    ) -> Rk4DenseEnclosure:
        """Enclose a root-path interval without reintegration or child chaining."""

        path = self._rk4_dense_path
        if self.path_kind != "rk4_dense" or path is None:
            raise ValueError("dense RK4 subinterval requires an RK4 proposal")
        selected = np.asarray(local_rows, dtype=np.int64)
        starts = np.asarray(start_time_s, dtype=np.float64)
        targets = np.asarray(target_time_s, dtype=np.float64)
        _validate_dense_subinterval(
            selected,
            starts,
            targets,
            self.start_time_s,
            self.target_time_s,
        )
        return _enclose_rk4_dense_subinterval(
            path,
            selected,
            starts,
            targets,
            self.numerical_status[selected],
        )

    def rk4_dense_chord_deviation(
        self,
        local_rows: Int64Array,
        start_time_s: FloatArray,
        target_time_s: FloatArray,
    ) -> FloatArray:
        """Bound position departure from each subinterval's endpoint chord.

        The bound is formed from the restricted cubic Bernstein controls of
        the immutable root path.  Unlike a velocity-span bound, it preserves
        the quadratic-in-duration curvature scale needed by event refinement.
        """

        path = self._rk4_dense_path
        if self.path_kind != "rk4_dense" or path is None:
            raise ValueError("dense RK4 chord deviation requires an RK4 proposal")
        selected = np.asarray(local_rows, dtype=np.int64)
        starts = np.asarray(start_time_s, dtype=np.float64)
        targets = np.asarray(target_time_s, dtype=np.float64)
        _validate_dense_subinterval(
            selected,
            starts,
            targets,
            self.start_time_s,
            self.target_time_s,
        )
        return _rk4_dense_chord_deviation(path, selected, starts, targets)

    def rk4_dense_state_at_rows(
        self,
        local_rows: Int64Array,
        time_s: FloatArray,
    ) -> ProposalSample:
        """Evaluate per-row times on the one immutable root RK4 path."""

        path = self._rk4_dense_path
        if self.path_kind != "rk4_dense" or path is None:
            raise ValueError("dense RK4 sampling requires an RK4 proposal")
        selected = np.asarray(local_rows, dtype=np.int64)
        sample_time_s = np.asarray(time_s, dtype=np.float64)
        _validate_dense_sample(selected, sample_time_s, self.start_time_s, self.target_time_s)
        position_m, velocity_m_s, charge_number = _evaluate_rk4_dense_path(
            path,
            selected,
            sample_time_s,
        )
        return ProposalSample(
            particle_index=self.particle_index[selected],
            position_m=position_m,
            velocity_m_s=velocity_m_s,
            charge_number=charge_number,
            support_inside=self.support_inside[selected].copy(),
            applicability_inside=self.applicability_inside[selected].copy(),
            numerical_status=self.numerical_status[selected].copy(),
        )


def cubic_hermite_step(
    particle_index: Int64Array,
    start_time_s: FloatArray,
    target_time_s: FloatArray,
    start_position_m: FloatArray,
    start_velocity_m_s: FloatArray,
    start_charge_number: FloatArray,
    end_position_m: FloatArray,
    end_velocity_m_s: FloatArray,
    *,
    end_charge_number: FloatArray | None = None,
    charge_root_time_s: FloatArray | None = None,
    charge_root_number: FloatArray | None = None,
    charge_affine_rate_number_s: FloatArray | None = None,
    charge_rate_derivative_s_inv: FloatArray | None = None,
    support_inside: BoolArray,
    applicability_inside: BoolArray,
    numerical_status: UInt8Array,
) -> StepProposal:
    """Construct endpoint-defined motion with one root-owned charge path.

    Each row is one immutable cubic polynomial over a positive interval.  This
    is the numerical path represented by a stochastic leaf; it does not claim
    to enclose an unresolved Gaussian bridge between its endpoints.
    """

    indices = np.asarray(particle_index, dtype=np.int64)
    starts = np.asarray(start_time_s, dtype=np.float64)
    targets = np.asarray(target_time_s, dtype=np.float64)
    start_position = np.asarray(start_position_m, dtype=np.float64)
    start_velocity = np.asarray(start_velocity_m_s, dtype=np.float64)
    charge = np.asarray(start_charge_number, dtype=np.float64)
    end_charge = (
        charge if end_charge_number is None else np.asarray(end_charge_number, dtype=np.float64)
    )
    end_position = np.asarray(end_position_m, dtype=np.float64)
    end_velocity = np.asarray(end_velocity_m_s, dtype=np.float64)
    support = np.asarray(support_inside)
    applicability = np.asarray(applicability_inside)
    status = np.asarray(numerical_status)
    count = int(indices.size)
    _validate_cubic_hermite_step_inputs(
        count,
        indices,
        starts,
        targets,
        start_position,
        start_velocity,
        charge,
        end_position,
        end_velocity,
        end_charge,
        support,
        applicability,
        status,
    )
    charge_path = _validate_cubic_charge_path_inputs(
        count,
        starts,
        charge_root_time_s,
        charge_root_number,
        charge_affine_rate_number_s,
        charge_rate_derivative_s_inv,
    )

    path = _CubicHermitePath(
        start_time_s=starts.copy(),
        target_time_s=targets.copy(),
        start_position_m=start_position.copy(),
        start_velocity_m_s=start_velocity.copy(),
        end_position_m=end_position.copy(),
        end_velocity_m_s=end_velocity.copy(),
        start_charge_number=charge.copy(),
        end_charge_number=end_charge.copy(),
        charge_root_time_s=(None if charge_path is None else charge_path[0]),
        charge_root_number=(None if charge_path is None else charge_path[1]),
        charge_affine_rate_number_s=(None if charge_path is None else charge_path[2]),
        charge_rate_derivative_s_inv=(None if charge_path is None else charge_path[3]),
    )
    rows = np.arange(count, dtype=np.int64)
    enclosure = _enclose_cubic_hermite_subinterval(path, rows, starts, targets)
    combined_status = status.copy()
    first_failure = (combined_status == NUMERICAL_STATUS_OK) & (
        enclosure.numerical_status != NUMERICAL_STATUS_OK
    )
    combined_status[first_failure] = enclosure.numerical_status[first_failure]
    _require_enclosure_contains_endpoint(
        enclosure,
        end_position,
        end_velocity,
        combined_status,
    )
    return StepProposal(
        particle_index=indices.copy(),
        start_time_s=starts.copy(),
        target_time_s=targets.copy(),
        start_position_m=start_position.copy(),
        start_velocity_m_s=start_velocity.copy(),
        start_charge_number=charge.copy(),
        end_position_m=end_position.copy(),
        end_velocity_m_s=end_velocity.copy(),
        end_charge_number=end_charge.copy(),
        end_field_cell_id=None,
        support_inside=support.copy(),
        applicability_inside=applicability.copy(),
        numerical_status=combined_status,
        path_kind="cubic_hermite",
        constant_acceleration_m_s2=None,
        path_enclosure=enclosure,
        _evaluator=None,
        _relaxation_evaluator=None,
        _cubic_hermite_path=path,
    )


def _validate_cubic_hermite_step_inputs(
    count: int,
    indices: Int64Array,
    starts: FloatArray,
    targets: FloatArray,
    start_position: FloatArray,
    start_velocity: FloatArray,
    start_charge: FloatArray,
    end_position: FloatArray,
    end_velocity: FloatArray,
    end_charge: FloatArray,
    support: BoolArray,
    applicability: BoolArray,
    status: UInt8Array,
) -> None:
    """Validate the immutable endpoint data owned by one Hermite path."""

    if indices.ndim != 1:
        raise ValueError("cubic Hermite particle index must have shape [N]")
    _validate_inputs(
        count,
        starts,
        targets,
        start_position,
        start_velocity,
        start_charge,
    )
    if end_position.shape != (count, 2) or end_velocity.shape != (count, 2):
        raise ValueError("cubic Hermite endpoint state must have shape [N, 2]")
    if end_charge.shape != (count,) or not bool(np.isfinite(end_charge).all()):
        raise ValueError("cubic Hermite endpoint charge must be finite with shape [N]")
    if (
        support.shape != (count,)
        or applicability.shape != (count,)
        or support.dtype != np.bool_
        or applicability.dtype != np.bool_
    ):
        raise ValueError("cubic Hermite validity verdicts must be bool arrays with shape [N]")
    if status.shape != (count,) or status.dtype != np.uint8:
        raise ValueError("cubic Hermite numerical status must be a uint8 array with shape [N]")
    if not bool(np.isfinite(end_position).all() and np.isfinite(end_velocity).all()):
        raise ValueError("cubic Hermite endpoint state must be finite")
    if bool((starts >= targets).any()):
        raise ValueError("cubic Hermite root intervals must be positive")


def _validate_cubic_charge_path_inputs(
    count: int,
    leaf_start_time_s: FloatArray,
    root_time_s: FloatArray | None,
    root_charge_number: FloatArray | None,
    affine_rate_number_s: FloatArray | None,
    rate_derivative_s_inv: FloatArray | None,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray] | None:
    values = (root_time_s, root_charge_number, affine_rate_number_s, rate_derivative_s_inv)
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError("cubic Hermite exponential charge metadata must be complete")
    root_time = np.asarray(root_time_s, dtype=np.float64)
    root_charge = np.asarray(root_charge_number, dtype=np.float64)
    affine_rate = np.asarray(affine_rate_number_s, dtype=np.float64)
    derivative = np.asarray(rate_derivative_s_inv, dtype=np.float64)
    if any(value.shape != (count,) for value in (root_time, root_charge, affine_rate, derivative)):
        raise ValueError("cubic Hermite exponential charge metadata must have shape [N]")
    if not bool(
        np.isfinite(root_time).all()
        and np.isfinite(root_charge).all()
        and np.isfinite(affine_rate).all()
        and np.isfinite(derivative).all()
        and (derivative <= 0.0).all()
        and (root_time <= leaf_start_time_s).all()
    ):
        raise ValueError("cubic Hermite exponential charge metadata is invalid")
    return root_time.copy(), root_charge.copy(), affine_rate.copy(), derivative.copy()


def restrict_cubic_hermite_proposal(
    proposal: StepProposal,
    local_rows: Int64Array,
    start_time_s: FloatArray,
    target_time_s: FloatArray,
) -> StepProposal:
    """Restrict selected rows without replacing their original polynomials."""

    path = proposal._cubic_hermite_path
    if proposal.path_kind != "cubic_hermite" or path is None:
        raise ValueError("restriction requires a cubic Hermite proposal")
    selected = np.asarray(local_rows, dtype=np.int64)
    if (
        selected.ndim != 1
        or bool((selected < 0).any())
        or bool((selected >= proposal.particle_index.size).any())
    ):
        raise ValueError("cubic Hermite restriction rows are invalid")
    starts = np.asarray(start_time_s, dtype=np.float64)
    targets = np.asarray(target_time_s, dtype=np.float64)
    count = int(selected.size)
    if starts.shape != (count,) or targets.shape != (count,):
        raise ValueError("cubic Hermite restriction times must have shape [N]")
    if not bool(np.isfinite(starts).all() and np.isfinite(targets).all()):
        raise ValueError("cubic Hermite restriction times must be finite")
    if bool(
        (
            (starts < proposal.start_time_s[selected])
            | (targets > proposal.target_time_s[selected])
            | (starts > targets)
        ).any()
    ):
        raise ValueError("cubic Hermite restriction is outside the proposal interval")

    original = _subset_cubic_hermite_path(path, selected)
    result_rows = np.arange(count, dtype=np.int64)
    start_position, start_velocity = _evaluate_cubic_hermite(
        original,
        result_rows,
        starts,
    )
    end_position, end_velocity = _evaluate_cubic_hermite(
        original,
        result_rows,
        targets,
    )
    enclosure = _enclose_cubic_hermite_subinterval(
        original,
        result_rows,
        starts,
        targets,
    )
    numerical_status = proposal.numerical_status[selected].copy()
    first_failure = (numerical_status == NUMERICAL_STATUS_OK) & (
        enclosure.numerical_status != NUMERICAL_STATUS_OK
    )
    numerical_status[first_failure] = enclosure.numerical_status[first_failure]
    _require_enclosure_contains_endpoint(
        enclosure,
        end_position,
        end_velocity,
        numerical_status,
    )
    start_charge = _evaluate_cubic_charge(original, result_rows, starts)
    end_charge = _evaluate_cubic_charge(original, result_rows, targets)
    return StepProposal(
        particle_index=proposal.particle_index[selected].copy(),
        start_time_s=starts.copy(),
        target_time_s=targets.copy(),
        start_position_m=start_position,
        start_velocity_m_s=start_velocity,
        start_charge_number=start_charge,
        end_position_m=end_position,
        end_velocity_m_s=end_velocity,
        end_charge_number=end_charge,
        end_field_cell_id=(
            None
            if proposal.end_field_cell_id is None
            else proposal.end_field_cell_id[selected].copy()
        ),
        support_inside=proposal.support_inside[selected].copy(),
        applicability_inside=proposal.applicability_inside[selected].copy(),
        numerical_status=numerical_status,
        path_kind="cubic_hermite",
        constant_acceleration_m_s2=None,
        path_enclosure=enclosure,
        _evaluator=None,
        _relaxation_evaluator=None,
        _cubic_hermite_path=original,
    )


def _subset_cubic_hermite_path(
    path: _CubicHermitePath,
    selected: Int64Array,
) -> _CubicHermitePath:
    return _CubicHermitePath(
        start_time_s=path.start_time_s[selected].copy(),
        target_time_s=path.target_time_s[selected].copy(),
        start_position_m=path.start_position_m[selected].copy(),
        start_velocity_m_s=path.start_velocity_m_s[selected].copy(),
        end_position_m=path.end_position_m[selected].copy(),
        end_velocity_m_s=path.end_velocity_m_s[selected].copy(),
        start_charge_number=path.start_charge_number[selected].copy(),
        end_charge_number=path.end_charge_number[selected].copy(),
        charge_root_time_s=(
            None if path.charge_root_time_s is None else path.charge_root_time_s[selected].copy()
        ),
        charge_root_number=(
            None if path.charge_root_number is None else path.charge_root_number[selected].copy()
        ),
        charge_affine_rate_number_s=(
            None
            if path.charge_affine_rate_number_s is None
            else path.charge_affine_rate_number_s[selected].copy()
        ),
        charge_rate_derivative_s_inv=(
            None
            if path.charge_rate_derivative_s_inv is None
            else path.charge_rate_derivative_s_inv[selected].copy()
        ),
    )


def _evaluate_cubic_hermite(
    path: _CubicHermitePath,
    rows: Int64Array,
    time_s: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    start_time = path.start_time_s[rows]
    duration = path.target_time_s[rows] - start_time
    unit_time = (time_s - start_time) / duration
    origin, relative_position, position_residual, velocity_controls = _cubic_hermite_root_controls(
        path, rows
    )
    with np.errstate(over="ignore", invalid="ignore"):
        position_m = origin + (
            _evaluate_bezier(relative_position, unit_time)
            + _evaluate_bezier(position_residual, unit_time)
        )
        weight = unit_time[:, None]
        first = _lerp_controls(velocity_controls[:, 0], velocity_controls[:, 1], unit_time)
        second = _lerp_controls(velocity_controls[:, 1], velocity_controls[:, 2], unit_time)
        velocity_m_s = (1.0 - weight) * first + weight * second
    at_start = time_s == start_time
    at_end = time_s == path.target_time_s[rows]
    position_m[at_start] = path.start_position_m[rows[at_start]]
    velocity_m_s[at_start] = path.start_velocity_m_s[rows[at_start]]
    position_m[at_end] = path.end_position_m[rows[at_end]]
    velocity_m_s[at_end] = path.end_velocity_m_s[rows[at_end]]
    return position_m, velocity_m_s


def _cubic_hermite_root_controls(
    path: _CubicHermitePath,
    rows: Int64Array,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    origin = path.start_position_m[rows]
    duration = (path.target_time_s[rows] - path.start_time_s[rows])[:, None]
    displacement, displacement_residual = _two_diff(
        path.end_position_m[rows],
        origin,
    )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        start_tangent = duration * path.start_velocity_m_s[rows] / 3.0
        end_tangent = duration * path.end_velocity_m_s[rows] / 3.0
        second_control, second_residual = _two_diff(displacement, end_tangent)
        zero = np.zeros_like(displacement)
        relative_position = np.stack(
            (zero, start_tangent, second_control, displacement),
            axis=1,
        )
        position_residual = np.stack(
            (
                zero,
                zero,
                second_residual + displacement_residual,
                displacement_residual,
            ),
            axis=1,
        )
        difference, difference_residual = _two_diff(
            relative_position[:, 1:],
            relative_position[:, :-1],
        )
        residual_difference = position_residual[:, 1:] - position_residual[:, :-1]
        velocity_controls = (
            3.0 * (difference + difference_residual + residual_difference) / duration[:, None]
        )
    velocity_controls[:, 0] = path.start_velocity_m_s[rows]
    velocity_controls[:, 2] = path.end_velocity_m_s[rows]
    return origin, relative_position, position_residual, velocity_controls


def _evaluate_cubic_charge(
    path: _CubicHermitePath,
    rows: Int64Array,
    time_s: FloatArray,
) -> FloatArray:
    """Evaluate the root-owned linear or affine-exponential charge path."""

    start_time = path.start_time_s[rows]
    duration = path.target_time_s[rows] - start_time
    start_charge = path.start_charge_number[rows]
    end_charge = path.end_charge_number[rows]
    if path.charge_root_time_s is None:
        unit_time = (time_s - start_time) / duration
        charge = start_charge + unit_time * (end_charge - start_charge)
    else:
        if (
            path.charge_root_number is None
            or path.charge_affine_rate_number_s is None
            or path.charge_rate_derivative_s_inv is None
        ):
            raise ValueError("cubic Hermite exponential charge metadata is incomplete")
        elapsed = time_s - path.charge_root_time_s[rows]
        multiplier = _affine_exponential_multiplier(
            path.charge_rate_derivative_s_inv[rows],
            elapsed,
        )
        charge = path.charge_root_number[rows] + multiplier * path.charge_affine_rate_number_s[rows]
    charge[time_s == start_time] = start_charge[time_s == start_time]
    at_end = time_s == path.target_time_s[rows]
    charge[at_end] = end_charge[at_end]
    return charge


def _validate_dense_sample(
    selected: Int64Array,
    time_s: FloatArray,
    proposal_start_time_s: FloatArray,
    proposal_target_time_s: FloatArray,
) -> None:
    if (
        selected.ndim != 1
        or time_s.shape != (selected.size,)
        or bool((selected < 0).any())
        or bool((selected >= proposal_start_time_s.size).any())
        or not bool(np.isfinite(time_s).all())
    ):
        raise ValueError("dense RK4 sample rows or times are invalid")
    if bool(
        (
            (time_s < proposal_start_time_s[selected]) | (time_s > proposal_target_time_s[selected])
        ).any()
    ):
        raise ValueError("dense RK4 sample is outside the proposal interval")


def _validate_dense_subinterval(
    selected: Int64Array,
    start_time_s: FloatArray,
    target_time_s: FloatArray,
    proposal_start_time_s: FloatArray,
    proposal_target_time_s: FloatArray,
) -> None:
    _validate_dense_sample(
        selected,
        start_time_s,
        proposal_start_time_s,
        proposal_target_time_s,
    )
    _validate_dense_sample(
        selected,
        target_time_s,
        proposal_start_time_s,
        proposal_target_time_s,
    )
    if bool((start_time_s > target_time_s).any()):
        raise ValueError("dense RK4 subinterval must have nonnegative duration")


def _freeze_rk4_dense_path(
    controls: _Rk4DenseControls,
    start_time_s: FloatArray,
    target_time_s: FloatArray,
    start_position_m: FloatArray,
    start_velocity_m_s: FloatArray,
    start_charge_number: FloatArray,
    end_position_m: FloatArray,
    end_velocity_m_s: FloatArray,
    end_charge_number: FloatArray,
) -> _Rk4DensePath:
    path = _Rk4DensePath(
        start_time_s=np.asarray(start_time_s, dtype=np.float64).copy(),
        target_time_s=np.asarray(target_time_s, dtype=np.float64).copy(),
        position_controls_m=np.stack(
            (
                start_position_m,
                controls.position_inner_m[:, 0],
                controls.position_inner_m[:, 1],
                end_position_m,
            ),
            axis=1,
        ),
        velocity_controls_m_s=np.stack(
            (
                start_velocity_m_s,
                controls.velocity_inner_m_s[:, 0],
                controls.velocity_inner_m_s[:, 1],
                end_velocity_m_s,
            ),
            axis=1,
        ),
        charge_controls_number=np.stack(
            (
                start_charge_number,
                controls.charge_inner_number[:, 0],
                controls.charge_inner_number[:, 1],
                end_charge_number,
            ),
            axis=1,
        ),
    )
    for value in (
        path.start_time_s,
        path.target_time_s,
        path.position_controls_m,
        path.velocity_controls_m_s,
        path.charge_controls_number,
    ):
        value.setflags(write=False)
    return path


def _dense_parameter(path: _Rk4DensePath, rows: Int64Array, time_s: FloatArray) -> FloatArray:
    start = path.start_time_s[rows]
    duration = path.target_time_s[rows] - start
    parameter = np.zeros(rows.size, dtype=np.float64)
    moving = duration > 0.0
    parameter[moving] = (time_s[moving] - start[moving]) / duration[moving]
    return np.clip(parameter, 0.0, 1.0)


def _evaluate_bezier(controls: FloatArray, parameter: FloatArray) -> FloatArray:
    first_01 = _lerp_controls(controls[:, 0], controls[:, 1], parameter)
    first_12 = _lerp_controls(controls[:, 1], controls[:, 2], parameter)
    first_23 = _lerp_controls(controls[:, 2], controls[:, 3], parameter)
    second_012 = _lerp_controls(first_01, first_12, parameter)
    second_123 = _lerp_controls(first_12, first_23, parameter)
    return _lerp_controls(second_012, second_123, parameter)


def _evaluate_rk4_dense_path(
    path: _Rk4DensePath,
    rows: Int64Array,
    time_s: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    parameter = _dense_parameter(path, rows, time_s)
    position_controls = path.position_controls_m[rows]
    position_origin = position_controls[:, :1]
    relative_position, position_origin_residual = _two_diff(
        position_controls,
        position_origin,
    )
    # Evaluate the same stored-control cubic as a root-origin-relative
    # polynomial.  The TwoDiff residual preserves the exact control
    # differences while avoiding loss from repeated interpolation of a large
    # common coordinate.  Dense position samples consequently use the same
    # arithmetic representation as their enclosure and chord certificate.
    with np.errstate(over="ignore", invalid="ignore"):
        position = position_origin[:, 0] + (
            _evaluate_bezier(relative_position, parameter)
            + _evaluate_bezier(position_origin_residual, parameter)
        )
    velocity = _evaluate_bezier(path.velocity_controls_m_s[rows], parameter)
    charge = _evaluate_bezier(path.charge_controls_number[rows, :, None], parameter)[:, 0]
    at_start = time_s == path.start_time_s[rows]
    at_end = time_s == path.target_time_s[rows]
    position[at_start] = path.position_controls_m[rows[at_start], 0]
    velocity[at_start] = path.velocity_controls_m_s[rows[at_start], 0]
    charge[at_start] = path.charge_controls_number[rows[at_start], 0]
    position[at_end] = path.position_controls_m[rows[at_end], 3]
    velocity[at_end] = path.velocity_controls_m_s[rows[at_end], 3]
    charge[at_end] = path.charge_controls_number[rows[at_end], 3]
    return position, velocity, charge


def _lerp_controls(left: FloatArray, right: FloatArray, parameter: FloatArray) -> FloatArray:
    weight_shape = (parameter.size,) + (1,) * (left.ndim - 1)
    weight = parameter.reshape(weight_shape)
    return (1.0 - weight) * left + weight * right


def _restrict_cubic_controls(
    controls: FloatArray,
    start_parameter: FloatArray,
    target_parameter: FloatArray,
) -> FloatArray:
    first_01 = _lerp_controls(controls[:, 0], controls[:, 1], start_parameter)
    first_12 = _lerp_controls(controls[:, 1], controls[:, 2], start_parameter)
    first_23 = _lerp_controls(controls[:, 2], controls[:, 3], start_parameter)
    second_012 = _lerp_controls(first_01, first_12, start_parameter)
    second_123 = _lerp_controls(first_12, first_23, start_parameter)
    start_value = _lerp_controls(second_012, second_123, start_parameter)
    denominator = 1.0 - start_parameter
    relative_target = np.zeros_like(start_parameter)
    moving = denominator > 0.0
    relative_target[moving] = (target_parameter[moving] - start_parameter[moving]) / denominator[
        moving
    ]
    relative_target = np.clip(relative_target, 0.0, 1.0)
    third_01 = _lerp_controls(start_value, second_123, relative_target)
    third_12 = _lerp_controls(second_123, first_23, relative_target)
    third_23 = _lerp_controls(first_23, controls[:, 3], relative_target)
    fourth_012 = _lerp_controls(third_01, third_12, relative_target)
    fourth_123 = _lerp_controls(third_12, third_23, relative_target)
    target_value = _lerp_controls(fourth_012, fourth_123, relative_target)
    restricted = np.stack((start_value, third_01, fourth_012, target_value), axis=1)
    at_root_end = start_parameter == 1.0
    restricted[at_root_end] = controls[at_root_end, 3, None]
    return restricted


def _restrict_quadratic_controls(
    controls: FloatArray,
    start_parameter: FloatArray,
    target_parameter: FloatArray,
) -> FloatArray:
    first_01 = _lerp_controls(controls[:, 0], controls[:, 1], start_parameter)
    first_12 = _lerp_controls(controls[:, 1], controls[:, 2], start_parameter)
    start_value = _lerp_controls(first_01, first_12, start_parameter)
    denominator = 1.0 - start_parameter
    relative_target = np.zeros_like(start_parameter)
    moving = denominator > 0.0
    relative_target[moving] = (target_parameter[moving] - start_parameter[moving]) / denominator[
        moving
    ]
    relative_target = np.clip(relative_target, 0.0, 1.0)
    second_01 = _lerp_controls(start_value, first_12, relative_target)
    second_12 = _lerp_controls(first_12, controls[:, 2], relative_target)
    target_value = _lerp_controls(second_01, second_12, relative_target)
    restricted = np.stack((start_value, second_01, target_value), axis=1)
    at_root_end = start_parameter == 1.0
    restricted[at_root_end] = controls[at_root_end, 2, None]
    return restricted


def _position_derivative_controls(
    path: _Rk4DensePath,
    rows: Int64Array,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    position = path.position_controls_m[rows]
    duration = path.target_time_s[rows] - path.start_time_s[rows]
    derivative = np.repeat(path.velocity_controls_m_s[rows, :1], 3, axis=1)
    scale = np.abs(derivative)
    absolute_error = np.zeros_like(derivative)
    moving = duration > 0.0
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        difference, residual = _two_diff(
            position[moving, 1:],
            position[moving, :-1],
        )
        derivative[moving] = 3.0 * difference / duration[moving, None, None]
        residual_contribution = np.nextafter(3.0 * np.abs(residual), np.inf)
        residual_contribution = np.nextafter(
            residual_contribution / duration[moving, None, None],
            np.inf,
        )
        residual_contribution[residual == 0.0] = 0.0
        absolute_error[moving] = residual_contribution
        scale[moving] = np.maximum(
            np.abs(derivative[moving]),
            residual_contribution,
        )
    return derivative, scale, absolute_error


def _two_diff(left: FloatArray, right: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Return the rounded difference and its exact binary64 residual.

    Knuth's TwoDiff decomposition makes ``difference + residual`` equal to the
    exact difference of the two stored controls when the finite subtraction
    does not overflow.  Unlike an ``abs(left) + abs(right)`` error estimate,
    its residual is independent of the selected coordinate origin.  Equal
    controls are assigned an exact zero explicitly, including the tiny-step
    case where a physical displacement is below the position ULP.
    """

    with np.errstate(over="ignore", invalid="ignore"):
        difference = left - right
        virtual_right = left - difference
        virtual_left = difference + virtual_right
        right_roundoff = virtual_right - right
        left_roundoff = left - virtual_left
        residual = left_roundoff + right_roundoff
    equal = left == right
    difference[equal] = 0.0
    residual[equal] = 0.0
    return difference, residual


def _outward_bezier_bounds(
    restricted_controls: FloatArray,
    root_controls: FloatArray,
    extra_scale: FloatArray | None = None,
    absolute_padding: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray]:
    lower_controls, upper_controls = _outward_bezier_control_bounds(
        restricted_controls,
        root_controls,
        extra_scale,
        absolute_padding,
    )
    return np.min(lower_controls, axis=1), np.max(upper_controls, axis=1)


def _outward_bezier_control_bounds(
    restricted_controls: FloatArray,
    root_controls: FloatArray,
    extra_scale: FloatArray | None = None,
    absolute_padding: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Bound every restricted control without collapsing control identity."""

    axis = 1
    scale = np.maximum(
        np.max(np.abs(restricted_controls), axis=axis),
        np.max(np.abs(root_controls), axis=axis),
    )
    if extra_scale is not None:
        scale = np.maximum(scale, np.max(extra_scale, axis=axis))
    eps_factor = 128.0 * np.finfo(np.float64).eps
    underflow = 128.0 * np.finfo(np.float64).smallest_subnormal
    with np.errstate(over="ignore", invalid="ignore"):
        padding = np.nextafter(scale * eps_factor + underflow, np.inf)
        if absolute_padding is not None:
            padding = np.nextafter(
                padding + np.max(absolute_padding, axis=axis),
                np.inf,
            )
        lower = np.nextafter(restricted_controls - padding[:, None, :], -np.inf)
        upper = np.nextafter(restricted_controls + padding[:, None, :], np.inf)
    return lower, upper


def _enclose_rk4_dense_subinterval(
    path: _Rk4DensePath,
    rows: Int64Array,
    start_time_s: FloatArray,
    target_time_s: FloatArray,
    inherited_status: UInt8Array,
) -> Rk4DenseEnclosure:
    start_parameter = _dense_parameter(path, rows, start_time_s)
    target_parameter = _dense_parameter(path, rows, target_time_s)
    root_position = path.position_controls_m[rows]
    root_velocity = path.velocity_controls_m_s[rows]
    root_charge = path.charge_controls_number[rows, :, None]
    position_origin = root_position[:, :1]
    relative_root_position, position_origin_residual = _two_diff(
        root_position,
        position_origin,
    )
    relative_position = _restrict_cubic_controls(
        relative_root_position,
        start_parameter,
        target_parameter,
    )
    velocity = _restrict_cubic_controls(root_velocity, start_parameter, target_parameter)
    charge = _restrict_cubic_controls(root_charge, start_parameter, target_parameter)
    derivative, derivative_scale, derivative_error = _position_derivative_controls(path, rows)
    derivative = _restrict_quadratic_controls(
        derivative,
        start_parameter,
        target_parameter,
    )
    relative_position_control_lower, relative_position_control_upper = (
        _outward_bezier_control_bounds(
            relative_position,
            relative_root_position,
            absolute_padding=np.abs(position_origin_residual),
        )
    )
    relative_position_lower = np.min(relative_position_control_lower, axis=1)
    relative_position_upper = np.max(relative_position_control_upper, axis=1)
    with np.errstate(over="ignore", invalid="ignore"):
        position_lower = np.nextafter(
            position_origin[:, 0] + relative_position_lower,
            -np.inf,
        )
        position_upper = np.nextafter(
            position_origin[:, 0] + relative_position_upper,
            np.inf,
        )
    velocity_controls = np.concatenate((velocity, derivative), axis=1)
    velocity_source = np.concatenate((root_velocity, derivative), axis=1)
    velocity_lower, velocity_upper = _outward_bezier_bounds(
        velocity_controls,
        velocity_source,
        derivative_scale,
        derivative_error,
    )
    charge_lower, charge_upper = _outward_bezier_bounds(charge, root_charge)
    root_charge_nonpositive = np.all(root_charge <= 0.0, axis=1)
    root_charge_nonnegative = np.all(root_charge >= 0.0, axis=1)
    charge_upper[root_charge_nonpositive] = np.minimum(
        charge_upper[root_charge_nonpositive],
        0.0,
    )
    charge_lower[root_charge_nonnegative] = np.maximum(
        charge_lower[root_charge_nonnegative],
        0.0,
    )
    numerical_status = inherited_status.copy()
    invalid = ~np.isfinite(position_lower).all(axis=1)
    invalid |= ~np.isfinite(position_upper).all(axis=1)
    invalid |= ~np.isfinite(velocity_lower).all(axis=1)
    invalid |= ~np.isfinite(velocity_upper).all(axis=1)
    invalid |= ~np.isfinite(charge_lower[:, 0]) | ~np.isfinite(charge_upper[:, 0])
    invalid |= ~np.isfinite(relative_position_control_lower).all(axis=(1, 2))
    invalid |= ~np.isfinite(relative_position_control_upper).all(axis=(1, 2))
    numerical_status[invalid & (numerical_status == NUMERICAL_STATUS_OK)] = (
        INTEGRATOR_NUMERICAL_FAILURE
    )
    failed = numerical_status != NUMERICAL_STATUS_OK
    if bool(failed.any()):
        position_lower[failed] = root_position[failed, 0]
        position_upper[failed] = root_position[failed, 0]
        velocity_lower[failed] = root_velocity[failed, 0]
        velocity_upper[failed] = root_velocity[failed, 0]
        charge_lower[failed, 0] = root_charge[failed, 0, 0]
        charge_upper[failed, 0] = root_charge[failed, 0, 0]
        relative_position_control_lower[failed] = 0.0
        relative_position_control_upper[failed] = 0.0
    position_control_origin = position_origin[:, 0].copy()
    position_control_origin[failed] = 0.0
    return Rk4DenseEnclosure(
        position_lower_m=position_lower,
        position_upper_m=position_upper,
        velocity_lower_m_s=velocity_lower,
        velocity_upper_m_s=velocity_upper,
        charge_lower_number=charge_lower[:, 0],
        charge_upper_number=charge_upper[:, 0],
        position_control_origin_m=position_control_origin,
        relative_position_control_lower_m=relative_position_control_lower,
        relative_position_control_upper_m=relative_position_control_upper,
        numerical_status=numerical_status,
    )


def _rk4_dense_chord_deviation(
    path: _Rk4DensePath,
    rows: Int64Array,
    start_time_s: FloatArray,
    target_time_s: FloatArray,
) -> FloatArray:
    """Return a conservative componentwise cubic-to-chord distance bound.

    Physical curvature is evaluated in root-origin-relative coordinates so it
    is translation invariant.  The returned float64 world-coordinate samples
    and their endpoint chord unavoidably perform a few operations at the
    absolute coordinate scale, which is covered by the separate world term.
    """

    start_parameter = _dense_parameter(path, rows, start_time_s)
    target_parameter = _dense_parameter(path, rows, target_time_s)
    root = path.position_controls_m[rows]
    relative_root, root_residual = _two_diff(root, root[:, :1])
    controls = _restrict_cubic_controls(relative_root, start_parameter, target_parameter)
    one_third = np.full(rows.size, 1.0 / 3.0, dtype=np.float64)
    two_thirds = np.full(rows.size, 2.0 / 3.0, dtype=np.float64)
    chord_one_third = _lerp_controls(controls[:, 0], controls[:, 3], one_third)
    chord_two_thirds = _lerp_controls(controls[:, 0], controls[:, 3], two_thirds)
    with np.errstate(over="ignore", invalid="ignore"):
        inner_deviation = np.stack(
            (
                controls[:, 1] - chord_one_third,
                controls[:, 2] - chord_two_thirds,
            ),
            axis=1,
        )
        arithmetic_scale = np.max(np.abs(relative_root), axis=1)
        arithmetic_scale = np.maximum(
            arithmetic_scale,
            np.max(np.abs(controls), axis=1),
        )
        subtraction_error = 4.0 * np.max(np.abs(root_residual), axis=1)
        world_scale = np.max(np.abs(root), axis=1)
        world_roundoff = (
            _RK4_DENSE_WORLD_CHORD_EPS_FACTOR * np.finfo(np.float64).eps * world_scale
            + _RK4_DENSE_WORLD_CHORD_EPS_FACTOR * np.finfo(np.float64).smallest_subnormal
        )
        padding = np.nextafter(
            256.0 * np.finfo(np.float64).eps * arithmetic_scale
            + subtraction_error
            + world_roundoff
            + 256.0 * np.finfo(np.float64).smallest_subnormal,
            np.inf,
        )
        bound = np.nextafter(
            np.max(np.abs(inner_deviation), axis=1) + padding,
            np.inf,
        )
    invalid = ~np.isfinite(bound).all(axis=1)
    if bool(invalid.any()):
        bound[invalid] = np.inf
    return bound


def _enclose_cubic_hermite_subinterval(
    path: _CubicHermitePath,
    rows: Int64Array,
    start_time_s: FloatArray,
    target_time_s: FloatArray,
) -> CurvedPathEnclosure:
    start_position, start_velocity = _evaluate_cubic_hermite(path, rows, start_time_s)
    root_duration = path.target_time_s[rows] - path.start_time_s[rows]
    start_parameter = np.clip(
        (start_time_s - path.start_time_s[rows]) / root_duration,
        0.0,
        1.0,
    )
    target_parameter = np.clip(
        (target_time_s - path.start_time_s[rows]) / root_duration,
        0.0,
        1.0,
    )
    origin, root_position, root_residual, root_velocity = _cubic_hermite_root_controls(
        path,
        rows,
    )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        relative_position = _restrict_cubic_controls(
            root_position,
            start_parameter,
            target_parameter,
        )
        relative_residual = _restrict_cubic_controls(
            root_residual,
            start_parameter,
            target_parameter,
        )
        velocity_controls = _restrict_quadratic_controls(
            root_velocity,
            start_parameter,
            target_parameter,
        )
        relative_lower, relative_upper = _outward_bezier_bounds(
            relative_position,
            root_position,
            absolute_padding=np.abs(relative_residual),
        )
        position_lower = np.nextafter(
            origin + relative_lower,
            -np.inf,
        )
        position_upper = np.nextafter(
            origin + relative_upper,
            np.inf,
        )
        velocity_lower, velocity_upper = _outward_bezier_bounds(
            velocity_controls,
            root_velocity,
        )
    numerical_status = np.full(rows.size, NUMERICAL_STATUS_OK, dtype=np.uint8)
    result = _normalize_enclosure_bounds(
        start_position,
        start_velocity,
        position_lower,
        position_upper,
        velocity_lower,
        velocity_upper,
        numerical_status,
    )
    _require_finite_enclosure(result)
    return result


def rk4_step(
    particle_index: Int64Array,
    start_time_s: FloatArray,
    target_time_s: FloatArray,
    start_position_m: FloatArray,
    start_velocity_m_s: FloatArray,
    start_charge_number: FloatArray,
    *,
    requires_stage_evaluation: bool,
    evaluator: StageEvaluator | None,
    constant_acceleration_m_s2: FloatArray | None = None,
    path_enclosure: CurvedPathEnclosure | None = None,
) -> StepProposal:
    """Construct one proposal, using an exact path only when it was certified.

    General RK4 stores one cubic state path while retaining the compiled
    full-step endpoint unchanged.  Sampling and applicability certificates
    use that immutable path without a second shortened integration.  A
    supplied global path enclosure remains owned by event localization; the
    dense path keeps its separate subinterval enclosure.
    """

    count = int(particle_index.size)
    targets = np.asarray(target_time_s, dtype=np.float64)
    _validate_inputs(
        count,
        start_time_s,
        targets,
        start_position_m,
        start_velocity_m_s,
        start_charge_number,
    )
    if requires_stage_evaluation and evaluator is None:
        raise ValueError("stage-evaluated RK4 requires a stage evaluator")
    acceleration = _validate_constant_acceleration(
        count,
        requires_stage_evaluation,
        constant_acceleration_m_s2,
    )
    enclosure = _copy_path_enclosure(
        path_enclosure,
        count,
        start_position_m,
        start_velocity_m_s,
    )
    path_kind: PathKind
    if acceleration is not None:
        path_kind = "quadratic_exact"
    elif requires_stage_evaluation:
        path_kind = "rk4_dense"
    else:
        path_kind = "linear_exact"
    elapsed = targets - start_time_s
    result = _advance_with_initial_status(
        path_kind,
        evaluator,
        None,
        particle_index,
        start_time_s,
        elapsed,
        start_position_m,
        start_velocity_m_s,
        start_charge_number,
        acceleration,
        None if enclosure is None else enclosure.numerical_status,
    )
    _require_enclosure_contains_endpoint(
        enclosure,
        result.position_m,
        result.velocity_m_s,
        result.numerical_status,
    )
    dense_path = None
    stored_enclosure = enclosure
    if path_kind == "rk4_dense":
        if result.rk4_dense_controls is None:
            raise ValueError("RK4 proposal did not produce dense controls")
        dense_path = _freeze_rk4_dense_path(
            result.rk4_dense_controls,
            start_time_s,
            targets,
            start_position_m,
            start_velocity_m_s,
            start_charge_number,
            result.position_m,
            result.velocity_m_s,
            result.charge_number,
        )
        dense_enclosure = _enclose_rk4_dense_subinterval(
            dense_path,
            np.arange(count, dtype=np.int64),
            start_time_s,
            targets,
            result.numerical_status,
        )
        result.numerical_status[:] = dense_enclosure.numerical_status
        if stored_enclosure is None:
            stored_enclosure = CurvedPathEnclosure(
                position_lower_m=dense_enclosure.position_lower_m,
                position_upper_m=dense_enclosure.position_upper_m,
                velocity_lower_m_s=dense_enclosure.velocity_lower_m_s,
                velocity_upper_m_s=dense_enclosure.velocity_upper_m_s,
                numerical_status=dense_enclosure.numerical_status.copy(),
            )
        _require_enclosure_contains_endpoint(
            stored_enclosure,
            result.position_m,
            result.velocity_m_s,
            result.numerical_status,
        )
    return StepProposal(
        particle_index=particle_index.copy(),
        start_time_s=start_time_s.copy(),
        target_time_s=targets.copy(),
        start_position_m=start_position_m.copy(),
        start_velocity_m_s=start_velocity_m_s.copy(),
        start_charge_number=start_charge_number.copy(),
        end_position_m=result.position_m,
        end_velocity_m_s=result.velocity_m_s,
        end_charge_number=result.charge_number,
        end_field_cell_id=(
            None if result.end_field_cell_id is None else result.end_field_cell_id.copy()
        ),
        support_inside=result.support_inside,
        applicability_inside=result.applicability_inside,
        numerical_status=result.numerical_status,
        path_kind=path_kind,
        constant_acceleration_m_s2=acceleration,
        path_enclosure=stored_enclosure,
        _evaluator=evaluator,
        _relaxation_evaluator=None,
        _rk4_dense_path=dense_path,
    )


def exponential_midpoint_step(
    particle_index: Int64Array,
    start_time_s: FloatArray,
    target_time_s: FloatArray,
    start_position_m: FloatArray,
    start_velocity_m_s: FloatArray,
    start_charge_number: FloatArray,
    *,
    evaluator: RelaxationStageEvaluator,
    path_enclosure: CurvedPathEnclosure | None = None,
) -> StepProposal:
    """Construct one linear-drag exponential-midpoint proposal."""

    count = int(particle_index.size)
    targets = np.asarray(target_time_s, dtype=np.float64)
    _validate_inputs(
        count,
        start_time_s,
        targets,
        start_position_m,
        start_velocity_m_s,
        start_charge_number,
    )
    enclosure = _copy_path_enclosure(
        path_enclosure,
        count,
        start_position_m,
        start_velocity_m_s,
    )
    elapsed = targets - start_time_s
    result = _advance_with_initial_status(
        "exponential_midpoint_reintegrated",
        None,
        evaluator,
        particle_index,
        start_time_s,
        elapsed,
        start_position_m,
        start_velocity_m_s,
        start_charge_number,
        None,
        None if enclosure is None else enclosure.numerical_status,
    )
    _require_enclosure_contains_endpoint(
        enclosure,
        result.position_m,
        result.velocity_m_s,
        result.numerical_status,
    )
    return StepProposal(
        particle_index=particle_index.copy(),
        start_time_s=start_time_s.copy(),
        target_time_s=targets.copy(),
        start_position_m=start_position_m.copy(),
        start_velocity_m_s=start_velocity_m_s.copy(),
        start_charge_number=start_charge_number.copy(),
        end_position_m=result.position_m,
        end_velocity_m_s=result.velocity_m_s,
        end_charge_number=result.charge_number,
        end_field_cell_id=(
            None if result.end_field_cell_id is None else result.end_field_cell_id.copy()
        ),
        support_inside=result.support_inside,
        applicability_inside=result.applicability_inside,
        numerical_status=result.numerical_status,
        path_kind="exponential_midpoint_reintegrated",
        constant_acceleration_m_s2=None,
        path_enclosure=enclosure,
        _evaluator=None,
        _relaxation_evaluator=evaluator,
    )


def exponential_frozen_start_predictor(
    particle_index: Int64Array,
    start_time_s: FloatArray,
    elapsed_s: FloatArray,
    start_position_m: FloatArray,
    start_velocity_m_s: FloatArray,
    start_charge_number: FloatArray,
    *,
    evaluator: RelaxationStageEvaluator,
) -> ProposalSample:
    """Predict a coefficient midpoint without sampling the predicted endpoint.

    This is the deterministic half-root predictor used by the stochastic
    exponential-midpoint composition. It evaluates the accepted root start
    once, freezes those coefficients, and advances by ``elapsed_s``. Elapsed
    time is authoritative even when ``start_time_s + elapsed_s`` rounds back
    to the start. The
    caller owns geometry admission of the returned point before any field or
    physics evaluation may occur there.
    """

    return _frozen_start_predictor_data(
        particle_index,
        start_time_s,
        elapsed_s,
        start_position_m,
        start_velocity_m_s,
        start_charge_number,
        evaluator=evaluator,
    ).sample


def exponential_frozen_start_predictor_enclosure(
    particle_index: Int64Array,
    start_time_s: FloatArray,
    elapsed_s: FloatArray,
    start_position_m: FloatArray,
    start_velocity_m_s: FloatArray,
    start_charge_number: FloatArray,
    *,
    evaluator: RelaxationStageEvaluator,
) -> FrozenStartPredictorEnclosure:
    """Return one predictor and a box for every shortened predictor time.

    The coefficients are evaluated once at the accepted start and then frozen.
    With nonnegative relaxation, velocity without additive acceleration is a
    convex combination of ``v0`` and ``u``; its additive memory is at most
    ``s``. Position uses nonnegative velocity weights summing to ``s`` and an
    additive memory no larger than ``s**2/2``. These facts bound every
    ``0 <= s <= H``. Absolute-term padding additionally contains float64
    cancellation in the compiled predictor and affine charge kernels.
    """

    data = _frozen_start_predictor_data(
        particle_index,
        start_time_s,
        elapsed_s,
        start_position_m,
        start_velocity_m_s,
        start_charge_number,
        evaluator=evaluator,
    )
    return _enclose_frozen_start_predictor(
        data,
        start_time_s,
        start_position_m,
        start_velocity_m_s,
        start_charge_number,
    )


def _frozen_start_predictor_data(
    particle_index: Int64Array,
    start_time_s: FloatArray,
    elapsed_s: FloatArray,
    start_position_m: FloatArray,
    start_velocity_m_s: FloatArray,
    start_charge_number: FloatArray,
    *,
    evaluator: RelaxationStageEvaluator,
) -> _FrozenStartPredictorData:
    """Evaluate one frozen start and retain the coefficients used to advance."""

    count = int(particle_index.size)
    elapsed = np.asarray(elapsed_s, dtype=np.float64)
    _validate_inputs(
        count,
        start_time_s,
        start_time_s,
        start_position_m,
        start_velocity_m_s,
        start_charge_number,
    )
    if elapsed.shape != (count,) or not bool(np.isfinite(elapsed).all() and (elapsed >= 0.0).all()):
        raise ValueError("frozen-start predictor elapsed time must be finite and nonnegative")
    numerical_status = np.full(count, NUMERICAL_STATUS_OK, dtype=np.uint8)
    support = np.ones(count, dtype=np.bool_)
    applicable = np.ones(count, dtype=np.bool_)
    drag_rate = np.zeros(count, dtype=np.float64)
    target_velocity = np.zeros((count, 2), dtype=np.float64)
    additive_acceleration = np.zeros((count, 2), dtype=np.float64)
    charge_rate = np.zeros(count, dtype=np.float64)
    charge_derivative = np.zeros(count, dtype=np.float64)
    rows, start = _evaluate_active_relaxation(
        evaluator,
        particle_index,
        start_time_s,
        start_position_m,
        start_velocity_m_s,
        start_charge_number,
        numerical_status,
    )
    if start is not None:
        support[rows] &= start.support_inside
        applicable[rows] &= start.applicability_inside
        drag_rate[rows] = start.linear_drag_rate_s_inv
        target_velocity[rows] = start.target_velocity_m_s
        additive_acceleration[rows] = start.additive_acceleration_m_s2
        charge_rate[rows] = start.charge_rate_number_s
        charge_derivative[rows] = start.charge_rate_derivative_s_inv
    position, velocity = _exponential_update_active(
        start_position_m,
        start_velocity_m_s,
        start_charge_number,
        drag_rate,
        target_velocity,
        additive_acceleration,
        elapsed,
        numerical_status,
    )
    charge = _stable_charge_update_active(
        start_charge_number,
        start_charge_number,
        charge_rate,
        charge_derivative,
        elapsed,
        numerical_status,
    )
    return _FrozenStartPredictorData(
        sample=ProposalSample(
            particle_index=particle_index.copy(),
            position_m=position,
            velocity_m_s=velocity,
            charge_number=charge,
            support_inside=support,
            applicability_inside=applicable,
            numerical_status=numerical_status,
        ),
        elapsed_s=elapsed.copy(),
        target_velocity_m_s=target_velocity,
        additive_acceleration_m_s2=additive_acceleration,
        charge_rate_number_s=charge_rate,
    )


def _enclose_frozen_start_predictor(
    data: _FrozenStartPredictorData,
    start_time_s: FloatArray,
    start_position_m: FloatArray,
    start_velocity_m_s: FloatArray,
    start_charge_number: FloatArray,
) -> FrozenStartPredictorEnclosure:
    """Enclose the compiled frozen-start predictor at its arithmetic scale."""

    sample = data.sample
    status = sample.numerical_status.copy()
    position_lower = start_position_m.copy()
    position_upper = start_position_m.copy()
    velocity_lower = start_velocity_m_s.copy()
    velocity_upper = start_velocity_m_s.copy()
    charge_lower = start_charge_number.copy()
    charge_upper = start_charge_number.copy()
    del start_time_s
    duration_s = data.elapsed_s
    moving = np.flatnonzero(duration_s > 0.0).astype("<i8", copy=False)
    if moving.size:
        duration = duration_s[moving, None]
        velocity = start_velocity_m_s[moving]
        target = data.target_velocity_m_s[moving]
        acceleration_abs = np.abs(data.additive_acceleration_m_s2[moving])
        with np.errstate(over="ignore", invalid="ignore"):
            velocity_additive = duration * acceleration_abs
            velocity_exact_lower = np.minimum(velocity, target) - velocity_additive
            velocity_exact_upper = np.maximum(velocity, target) + velocity_additive

            minimum_velocity = np.minimum(np.minimum(velocity, target), 0.0)
            maximum_velocity = np.maximum(np.maximum(velocity, target), 0.0)
            acceleration_displacement = 0.5 * duration * duration * acceleration_abs
            position_exact_lower = (
                start_position_m[moving] + duration * minimum_velocity - acceleration_displacement
            )
            position_exact_upper = (
                start_position_m[moving] + duration * maximum_velocity + acceleration_displacement
            )

            eps_factor = _FROZEN_PREDICTOR_ROUNDOFF_EPS_FACTOR * np.finfo(np.float64).eps
            underflow = (
                _FROZEN_PREDICTOR_ROUNDOFF_EPS_FACTOR * np.finfo(np.float64).smallest_subnormal
            )
            velocity_scale = (
                np.abs(velocity)
                + np.abs(target)
                + velocity_additive
                + np.abs(sample.velocity_m_s[moving])
            )
            velocity_padding = np.nextafter(eps_factor * velocity_scale + underflow, np.inf)
            position_scale = (
                np.abs(start_position_m[moving])
                + duration * (np.abs(velocity) + np.abs(target))
                + acceleration_displacement
                + np.abs(sample.position_m[moving])
            )
            position_padding = np.nextafter(eps_factor * position_scale + underflow, np.inf)

            velocity_lower[moving] = np.nextafter(
                np.minimum(velocity_exact_lower, sample.velocity_m_s[moving]) - velocity_padding,
                -np.inf,
            )
            velocity_upper[moving] = np.nextafter(
                np.maximum(velocity_exact_upper, sample.velocity_m_s[moving]) + velocity_padding,
                np.inf,
            )
            position_lower[moving] = np.nextafter(
                np.minimum(position_exact_lower, sample.position_m[moving]) - position_padding,
                -np.inf,
            )
            position_upper[moving] = np.nextafter(
                np.maximum(position_exact_upper, sample.position_m[moving]) + position_padding,
                np.inf,
            )

            charge_scale = (
                np.abs(start_charge_number[moving])
                + np.abs(sample.charge_number[moving])
                + duration_s[moving] * np.abs(data.charge_rate_number_s[moving])
            )
            charge_padding = np.nextafter(eps_factor * charge_scale + underflow, np.inf)
            charge_lower[moving] = np.nextafter(
                np.minimum(start_charge_number[moving], sample.charge_number[moving])
                - charge_padding,
                -np.inf,
            )
            charge_upper[moving] = np.nextafter(
                np.maximum(start_charge_number[moving], sample.charge_number[moving])
                + charge_padding,
                np.inf,
            )

    path = _normalize_enclosure_bounds(
        start_position_m,
        start_velocity_m_s,
        position_lower,
        position_upper,
        velocity_lower,
        velocity_upper,
        status,
    )
    invalid_charge = ~np.isfinite(charge_lower) | ~np.isfinite(charge_upper)
    invalid_charge |= charge_lower > charge_upper
    first_failure = (path.numerical_status == NUMERICAL_STATUS_OK) & invalid_charge
    path.numerical_status[first_failure] = INTEGRATOR_NUMERICAL_FAILURE
    failed = path.numerical_status != NUMERICAL_STATUS_OK
    charge_lower[failed] = start_charge_number[failed]
    charge_upper[failed] = start_charge_number[failed]
    return FrozenStartPredictorEnclosure(
        sample=sample,
        path=path,
        charge_lower_number=charge_lower,
        charge_upper_number=charge_upper,
    )


def enclose_rk4_path(
    particle_index: Int64Array,
    start_time_s: FloatArray,
    target_time_s: FloatArray,
    start_position_m: FloatArray,
    start_velocity_m_s: FloatArray,
    *,
    acceleration_abs_bounder: AccelerationAbsBounder,
) -> CurvedPathEnclosure:
    """Enclose all shortened-step RK4 stages and dense endpoint states.

    The enclosure follows the exact algebra used by ``_advance_positive_rk4``
    for every elapsed time from zero through each row's proposal duration.  It
    is deliberately independent of sampled stage points: the supplied bounder
    must provide a global componentwise acceleration bound for each velocity
    box.  Every arithmetic upper bound is rounded toward positive infinity;
    the final lower and upper state bounds are rounded outward.
    """

    _validate_enclosure_inputs(
        particle_index,
        start_time_s,
        target_time_s,
        start_position_m,
        start_velocity_m_s,
    )
    duration_s = target_time_s - start_time_s
    moving = np.flatnonzero(duration_s > 0.0)
    result = CurvedPathEnclosure(
        position_lower_m=start_position_m.copy(),
        position_upper_m=start_position_m.copy(),
        velocity_lower_m_s=start_velocity_m_s.copy(),
        velocity_upper_m_s=start_velocity_m_s.copy(),
        numerical_status=np.full(particle_index.size, NUMERICAL_STATUS_OK, dtype=np.uint8),
    )
    if not moving.size:
        _require_finite_enclosure(result)
        return result

    moving_particle_index = particle_index[moving]
    moving_position_m = start_position_m[moving]
    moving_velocity_m_s = start_velocity_m_s[moving]
    duration_s = duration_s[moving]
    numerical_status = np.full(moving.size, NUMERICAL_STATUS_OK, dtype=np.uint8)
    duration_column = duration_s[:, None]
    half_duration = _upper_product_batch(duration_column, 0.5, numerical_status)
    velocity_1_abs = np.abs(moving_velocity_m_s)
    acceleration_1_abs, numerical_status = _bounded_acceleration(
        acceleration_abs_bounder,
        moving_particle_index,
        velocity_1_abs,
        numerical_status,
    )

    velocity_2_delta = _upper_product_batch(half_duration, acceleration_1_abs, numerical_status)
    velocity_2_abs = _upper_sum_batch(velocity_1_abs, velocity_2_delta, numerical_status)
    acceleration_2_abs, numerical_status = _bounded_acceleration(
        acceleration_abs_bounder,
        moving_particle_index,
        velocity_2_abs,
        numerical_status,
    )

    velocity_3_delta = _upper_product_batch(half_duration, acceleration_2_abs, numerical_status)
    velocity_3_abs = _upper_sum_batch(velocity_1_abs, velocity_3_delta, numerical_status)
    acceleration_3_abs, numerical_status = _bounded_acceleration(
        acceleration_abs_bounder,
        moving_particle_index,
        velocity_3_abs,
        numerical_status,
    )

    velocity_4_delta = _upper_product_batch(duration_column, acceleration_3_abs, numerical_status)
    velocity_4_abs = _upper_sum_batch(velocity_1_abs, velocity_4_delta, numerical_status)
    acceleration_4_abs, numerical_status = _bounded_acceleration(
        acceleration_abs_bounder,
        moving_particle_index,
        velocity_4_abs,
        numerical_status,
    )

    weighted_velocity = _weighted_upper_sum_batch(
        velocity_1_abs,
        velocity_2_abs,
        velocity_3_abs,
        velocity_4_abs,
        numerical_status,
    )
    endpoint_position_radius = _upper_product_batch(
        _upper_quotient_batch(duration_column, 6.0, numerical_status),
        weighted_velocity,
        numerical_status,
    )
    stage_position_radius = np.maximum.reduce(
        (
            _upper_product_batch(half_duration, velocity_1_abs, numerical_status),
            _upper_product_batch(half_duration, velocity_2_abs, numerical_status),
            _upper_product_batch(duration_column, velocity_3_abs, numerical_status),
            endpoint_position_radius,
        )
    )

    weighted_acceleration = _weighted_upper_sum_batch(
        acceleration_1_abs,
        acceleration_2_abs,
        acceleration_3_abs,
        acceleration_4_abs,
        numerical_status,
    )
    endpoint_velocity_delta = _upper_product_batch(
        _upper_quotient_batch(duration_column, 6.0, numerical_status),
        weighted_acceleration,
        numerical_status,
    )
    velocity_radius = np.maximum.reduce(
        (
            velocity_2_delta,
            velocity_3_delta,
            velocity_4_delta,
            endpoint_velocity_delta,
        )
    )
    moving_enclosure = _curved_enclosure_from_radii(
        moving_position_m,
        moving_velocity_m_s,
        stage_position_radius,
        velocity_radius,
        numerical_status,
    )
    _require_finite_enclosure(moving_enclosure)
    result.position_lower_m[moving] = moving_enclosure.position_lower_m
    result.position_upper_m[moving] = moving_enclosure.position_upper_m
    result.velocity_lower_m_s[moving] = moving_enclosure.velocity_lower_m_s
    result.velocity_upper_m_s[moving] = moving_enclosure.velocity_upper_m_s
    result.numerical_status[moving] = moving_enclosure.numerical_status
    _require_finite_enclosure(result)
    return result


def enclose_exponential_midpoint_path(
    particle_index: Int64Array,
    start_time_s: FloatArray,
    target_time_s: FloatArray,
    start_position_m: FloatArray,
    start_velocity_m_s: FloatArray,
    *,
    linear_drag_rate_upper_s_inv: FloatArray,
    target_velocity_abs_upper_m_s: FloatArray,
    additive_acceleration_abs_bounder: AccelerationAbsBounder,
) -> CurvedPathEnclosure:
    """Enclose every shortened exponential-midpoint predictor and state.

    With nonnegative linear drag, the drag-only velocity is a convex
    combination of its start and target.  The non-drag bounder is first
    evaluated at ``abs(v0)``.  If ``V=max(abs(v0), abs(u))``, the start
    half-step predictor lies inside ``V+h*A0/2``; evaluating the bounder over
    that box therefore encloses the midpoint-frozen additive acceleration.
    With ``B=max(A0, Ahalf)``, every shortened state obeys
    ``abs(v)<=V+h*B`` and ``abs(x-x0)<=h*V+h**2*B/2``.  The stored velocity
    interval additionally uses ``lambda<=lambda_upper`` to retain any
    certifiable sign; it remains inside this absolute-value bound.  Every
    shortened position secant has the same convex relaxation form and an
    additive coefficient no larger than the velocity coefficient, so this
    velocity interval also supports the shared chord-deviation certificate.
    """

    count = _validate_enclosure_inputs(
        particle_index,
        start_time_s,
        target_time_s,
        start_position_m,
        start_velocity_m_s,
    )
    rate_upper = np.asarray(linear_drag_rate_upper_s_inv, dtype=np.float64)
    target_upper = np.asarray(target_velocity_abs_upper_m_s, dtype=np.float64)
    if rate_upper.shape != (count,):
        raise ValueError("linear drag rate upper bound must have shape [N]")
    if target_upper.shape != (count, 2):
        raise ValueError("exponential vector bounds must have shape [N, 2]")
    if not bool(
        np.isfinite(rate_upper).all()
        and np.isfinite(target_upper).all()
        and (rate_upper >= 0.0).all()
        and (target_upper >= 0.0).all()
    ):
        raise ValueError("exponential coefficient bounds must be finite and nonnegative")

    duration_s = target_time_s - start_time_s
    moving = np.flatnonzero(duration_s > 0.0)
    additive_start_abs = np.zeros((count, 2), dtype=np.float64)
    additive_midpoint_abs = np.zeros((count, 2), dtype=np.float64)
    numerical_status = np.full(count, NUMERICAL_STATUS_OK, dtype=np.uint8)
    if moving.size:
        duration = duration_s[moving, None]
        velocity_abs = np.abs(start_velocity_m_s[moving])
        moving_status = numerical_status[moving].copy()
        start_bound, moving_status = _bounded_acceleration(
            additive_acceleration_abs_bounder,
            particle_index[moving],
            velocity_abs,
            moving_status,
        )
        velocity_base_abs = np.maximum(velocity_abs, target_upper[moving])
        half_duration = _upper_product_batch(duration, 0.5, moving_status)
        midpoint_velocity_abs = _upper_sum_batch(
            velocity_base_abs,
            _upper_product_batch(half_duration, start_bound, moving_status),
            moving_status,
        )
        midpoint_bound, moving_status = _bounded_acceleration(
            additive_acceleration_abs_bounder,
            particle_index[moving],
            midpoint_velocity_abs,
            moving_status,
        )
        additive_start_abs[moving] = start_bound
        additive_midpoint_abs[moving] = midpoint_bound
        numerical_status[moving] = moving_status
    return enclose_exponential_midpoint_path_from_stage_bounds(
        particle_index,
        start_time_s,
        target_time_s,
        start_position_m,
        start_velocity_m_s,
        linear_drag_rate_upper_s_inv=rate_upper,
        target_velocity_abs_upper_m_s=target_upper,
        additive_start_abs_upper_m_s2=additive_start_abs,
        additive_midpoint_abs_upper_m_s2=additive_midpoint_abs,
        numerical_status=numerical_status,
    )


def enclose_exponential_midpoint_path_from_stage_bounds(
    particle_index: Int64Array,
    start_time_s: FloatArray,
    target_time_s: FloatArray,
    start_position_m: FloatArray,
    start_velocity_m_s: FloatArray,
    *,
    linear_drag_rate_upper_s_inv: FloatArray,
    target_velocity_abs_upper_m_s: FloatArray,
    additive_start_abs_upper_m_s2: FloatArray,
    additive_midpoint_abs_upper_m_s2: FloatArray,
    numerical_status: UInt8Array,
) -> CurvedPathEnclosure:
    """Assemble an exponential enclosure from independently proven stage bounds."""

    count = _validate_enclosure_inputs(
        particle_index,
        start_time_s,
        target_time_s,
        start_position_m,
        start_velocity_m_s,
    )
    rate_upper = np.asarray(linear_drag_rate_upper_s_inv, dtype=np.float64)
    target_upper = np.asarray(target_velocity_abs_upper_m_s, dtype=np.float64)
    additive_start = np.asarray(additive_start_abs_upper_m_s2, dtype=np.float64)
    additive_midpoint = np.asarray(additive_midpoint_abs_upper_m_s2, dtype=np.float64)
    status = np.asarray(numerical_status, dtype=np.uint8).copy()
    if (
        rate_upper.shape != (count,)
        or target_upper.shape != (count, 2)
        or additive_start.shape != (count, 2)
        or additive_midpoint.shape != (count, 2)
        or status.shape != (count,)
    ):
        raise ValueError("exponential stage bounds have an invalid shape")
    finite_nonnegative = all(
        bool(np.isfinite(value).all() and (value >= 0.0).all())
        for value in (rate_upper, target_upper, additive_start, additive_midpoint)
    )
    if not finite_nonnegative:
        raise ValueError("exponential stage bounds must be finite and nonnegative")

    result = CurvedPathEnclosure(
        position_lower_m=start_position_m.copy(),
        position_upper_m=start_position_m.copy(),
        velocity_lower_m_s=start_velocity_m_s.copy(),
        velocity_upper_m_s=start_velocity_m_s.copy(),
        numerical_status=status.copy(),
    )
    duration_s = target_time_s - start_time_s
    moving = np.flatnonzero(duration_s > 0.0)
    if not moving.size:
        return result
    duration = duration_s[moving, None]
    velocity = start_velocity_m_s[moving]
    moving_status = status[moving].copy()
    velocity_abs = np.abs(velocity)
    velocity_base_abs = np.maximum(velocity_abs, target_upper[moving])
    additive_upper = np.maximum(additive_start[moving], additive_midpoint[moving])
    rate_argument = _upper_product_batch(duration_s[moving], rate_upper[moving], moving_status)[
        :, None
    ]
    with np.errstate(under="ignore", invalid="ignore"):
        minimum_decay = np.maximum(0.0, np.nextafter(np.exp(-rate_argument), -np.inf))
    relaxed_fraction = np.nextafter(1.0 - minimum_decay, np.inf)
    decayed_velocity = minimum_decay * velocity
    decay_lower = np.nextafter(np.minimum(velocity, decayed_velocity), -np.inf)
    decay_upper = np.nextafter(np.maximum(velocity, decayed_velocity), np.inf)
    target_radius = _upper_product_batch(relaxed_fraction, target_upper[moving], moving_status)
    relaxed_lower = np.nextafter(decay_lower - target_radius, -np.inf)
    relaxed_upper = np.nextafter(decay_upper + target_radius, np.inf)
    additive_velocity = _upper_product_batch(duration, additive_upper, moving_status)
    velocity_lower = np.nextafter(
        np.minimum(velocity, relaxed_lower) - additive_velocity,
        -np.inf,
    )
    velocity_upper = np.nextafter(
        np.maximum(velocity, relaxed_upper) + additive_velocity,
        np.inf,
    )
    position_radius = _upper_sum_batch(
        _upper_product_batch(
            duration,
            velocity_base_abs,
            moving_status,
        ),
        _upper_product_batch(
            _upper_product_batch(duration, duration, moving_status),
            _upper_product_batch(additive_upper, 0.5, moving_status),
            moving_status,
        ),
        moving_status,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        position_lower = np.nextafter(start_position_m[moving] - position_radius, -np.inf)
        position_upper = np.nextafter(start_position_m[moving] + position_radius, np.inf)
    moving_enclosure = _normalize_enclosure_bounds(
        start_position_m[moving],
        start_velocity_m_s[moving],
        position_lower,
        position_upper,
        velocity_lower,
        velocity_upper,
        moving_status,
    )
    _require_finite_enclosure(moving_enclosure)
    result.position_lower_m[moving] = moving_enclosure.position_lower_m
    result.position_upper_m[moving] = moving_enclosure.position_upper_m
    result.velocity_lower_m_s[moving] = moving_enclosure.velocity_lower_m_s
    result.velocity_upper_m_s[moving] = moving_enclosure.velocity_upper_m_s
    result.numerical_status[moving] = moving_enclosure.numerical_status
    _require_finite_enclosure(result)
    return result


def curved_chord_deviation_bound(
    start_position_m: FloatArray,
    end_position_m: FloatArray,
    velocity_lower_m_s: FloatArray,
    velocity_upper_m_s: FloatArray,
    interval_s: float,
) -> FloatArray:
    """Bound one reintegrated curved path from its endpoint chord."""

    return curved_chord_deviation_bounds(
        np.asarray(start_position_m, dtype=np.float64).reshape(1, -1),
        np.asarray(end_position_m, dtype=np.float64).reshape(1, -1),
        np.asarray(velocity_lower_m_s, dtype=np.float64).reshape(1, -1),
        np.asarray(velocity_upper_m_s, dtype=np.float64).reshape(1, -1),
        np.asarray([interval_s], dtype=np.float64),
    )[0]


def curved_chord_deviation_bounds(
    start_position_m: FloatArray,
    end_position_m: FloatArray,
    velocity_lower_m_s: FloatArray,
    velocity_upper_m_s: FloatArray,
    interval_s: FloatArray,
) -> FloatArray:
    """Bound a batch of reintegrated curved paths from their endpoint chords.

    Each supported method certifies that every shortened position secant lies
    in the supplied velocity box.  Two such secants differ by at most the box
    width, so ``h * (v_upper - v_lower)`` bounds the exact-arithmetic
    curve-to-chord deviation.  The remaining terms cover the float64 position
    and chord expressions with outward rounding.
    """

    start = np.asarray(start_position_m, dtype=np.float64)
    end = np.asarray(end_position_m, dtype=np.float64)
    velocity_lower = np.asarray(velocity_lower_m_s, dtype=np.float64)
    velocity_upper = np.asarray(velocity_upper_m_s, dtype=np.float64)
    interval = np.asarray(interval_s, dtype=np.float64)
    count = int(interval.size)
    arrays = (start, end, velocity_lower, velocity_upper)
    if interval.ndim != 1 or any(value.shape != (count, 2) for value in arrays):
        raise ValueError("curved chord batch inputs must have shape [N, 2] and [N]")
    if not all(bool(np.isfinite(value).all()) for value in arrays):
        raise ValueError("curved chord inputs must be finite")
    if not bool(np.isfinite(interval).all()) or bool((interval <= 0.0).any()):
        raise ValueError("curved chord intervals must be finite and positive")
    if bool((velocity_lower > velocity_upper).any()):
        raise ValueError("curved chord velocity lower bounds exceed upper bounds")

    deviation = np.empty((count, 2), dtype=np.float64)
    valid = _compiled_curved_chord_deviation_bounds(
        start,
        end,
        velocity_lower,
        velocity_upper,
        interval,
        deviation,
    )
    if not valid:
        raise ValueError("curved chord deviation is not finite and nonnegative")
    return deviation


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def curved_chord_deviation_component(
    start: float,
    end: float,
    velocity_lower: float,
    velocity_upper: float,
    interval_s: float,
) -> tuple[bool, float]:
    """Return one outward curve-to-chord certificate component."""

    roundoff_factor = _CURVED_CHORD_ROUNDOFF_EPS_FACTOR * np.finfo(np.float64).eps
    underflow_unit = _CURVED_CHORD_ROUNDOFF_EPS_FACTOR * np.nextafter(0.0, np.inf)
    velocity_span = np.nextafter(velocity_upper - velocity_lower, np.inf)
    velocity_abs = max(abs(velocity_lower), abs(velocity_upper))
    motion_scale = np.nextafter(velocity_abs * interval_s, np.inf)
    curve_deviation = np.nextafter(velocity_span * interval_s, np.inf)
    normal_padding = np.nextafter(abs(start) * roundoff_factor, np.inf)
    normal_padding = np.nextafter(
        normal_padding + np.nextafter(abs(end) * roundoff_factor, np.inf),
        np.inf,
    )
    normal_padding = np.nextafter(
        normal_padding + np.nextafter(motion_scale * (2.0 * roundoff_factor), np.inf),
        np.inf,
    )
    underflow_padding = np.nextafter(underflow_unit, np.inf)
    underflow_padding = np.nextafter(
        underflow_padding + np.nextafter(interval_s * underflow_unit, np.inf),
        np.inf,
    )
    underflow_padding = np.nextafter(
        underflow_padding + np.nextafter(velocity_abs * underflow_unit, np.inf),
        np.inf,
    )
    underflow_padding = np.nextafter(
        underflow_padding + np.nextafter(abs(start) * underflow_unit, np.inf),
        np.inf,
    )
    underflow_padding = np.nextafter(
        underflow_padding + np.nextafter(abs(end) * underflow_unit, np.inf),
        np.inf,
    )
    deviation = np.nextafter(curve_deviation + normal_padding, np.inf)
    deviation = np.nextafter(deviation + underflow_padding, np.inf)
    return math.isfinite(deviation) and deviation >= 0.0, deviation


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _compiled_curved_chord_deviation_bounds(
    start_position_m: FloatArray,
    end_position_m: FloatArray,
    velocity_lower_m_s: FloatArray,
    velocity_upper_m_s: FloatArray,
    interval_s: FloatArray,
    deviation_m: FloatArray,
) -> bool:
    """Fill one serial batch without per-row Python dispatch."""

    valid = True
    for row in range(interval_s.size):
        for axis in range(2):
            component_valid, deviation = curved_chord_deviation_component(
                start_position_m[row, axis],
                end_position_m[row, axis],
                velocity_lower_m_s[row, axis],
                velocity_upper_m_s[row, axis],
                interval_s[row],
            )
            deviation_m[row, axis] = deviation
            valid = valid and component_valid
    return valid


def _validate_enclosure_inputs(
    particle_index: Int64Array,
    start_time_s: FloatArray,
    target_time_s: FloatArray,
    position_m: FloatArray,
    velocity_m_s: FloatArray,
) -> int:
    count = int(particle_index.size)
    if (
        particle_index.ndim != 1
        or start_time_s.shape != (count,)
        or target_time_s.shape != (count,)
    ):
        raise ValueError("enclosure indices, start times, and target times must have shape [N]")
    if position_m.shape != (count, 2) or velocity_m_s.shape != (count, 2):
        raise ValueError("enclosure position and velocity must have shape [N, 2]")
    if not bool(
        np.isfinite(start_time_s).all()
        and np.isfinite(target_time_s).all()
        and np.isfinite(position_m).all()
        and np.isfinite(velocity_m_s).all()
    ):
        raise ValueError("enclosure inputs must be finite")
    if bool((start_time_s > target_time_s).any()):
        raise ValueError("enclosure time interval is inconsistent")
    return count


def _bounded_acceleration(
    bounder: AccelerationAbsBounder,
    particle_index: Int64Array,
    velocity_abs_upper_m_s: FloatArray,
    numerical_status: UInt8Array,
) -> tuple[FloatArray, UInt8Array]:
    if not bool((numerical_status != NUMERICAL_STATUS_OK).any()):
        value, status = bounder(particle_index, velocity_abs_upper_m_s)
        value = np.asarray(value, dtype=np.float64)
        _validate_acceleration_bound(value, status, particle_index.size)
        numerical_status[:] = status
        return _expanded_acceleration_bound(value, numerical_status)
    result = np.zeros((particle_index.size, 2), dtype=np.float64)
    rows = np.flatnonzero(numerical_status == NUMERICAL_STATUS_OK).astype("<i8", copy=False)
    if not rows.size:
        return result, numerical_status
    value, status = bounder(particle_index[rows], velocity_abs_upper_m_s[rows])
    value = np.asarray(value, dtype=np.float64)
    _validate_acceleration_bound(value, status, rows.size)
    _merge_numerical_status(numerical_status, rows, status)
    expanded, local_status = _expanded_acceleration_bound(value, numerical_status[rows].copy())
    _merge_numerical_status(numerical_status, rows, local_status)
    survivors = rows[local_status == NUMERICAL_STATUS_OK]
    result[survivors] = expanded[local_status == NUMERICAL_STATUS_OK]
    return result, numerical_status


def _validate_acceleration_bound(
    value: FloatArray,
    status: UInt8Array,
    count: int,
) -> None:
    if value.shape != (count, 2) or status.shape != (count,):
        raise ValueError("acceleration bounder must return [N, 2] values and [N] status")
    if status.dtype != np.uint8:
        raise ValueError("acceleration bounder status must have dtype uint8")
    if not bool(np.isfinite(value).all()) or bool((value < 0.0).any()):
        raise ValueError("acceleration bounder must return finite nonnegative placeholders")


def _expanded_acceleration_bound(
    value: FloatArray,
    numerical_status: UInt8Array,
) -> tuple[FloatArray, UInt8Array]:
    with np.errstate(over="ignore", invalid="ignore"):
        expanded = np.nextafter(value, np.inf)
    invalid = ~np.isfinite(expanded).all(axis=1)
    first_failure = invalid & (numerical_status == NUMERICAL_STATUS_OK)
    numerical_status[first_failure] = INTEGRATOR_NUMERICAL_FAILURE
    expanded[numerical_status != NUMERICAL_STATUS_OK] = 0.0
    return expanded, numerical_status


def _upper_product_batch(
    left: FloatArray,
    right: FloatArray | float,
    numerical_status: UInt8Array,
) -> FloatArray:
    with np.errstate(over="ignore", invalid="ignore"):
        value = np.asarray(left, dtype=np.float64) * right
        result = np.nextafter(value, np.inf)
    _normalize_nonnegative_rows(result, numerical_status)
    return result


def _upper_sum_batch(
    left: FloatArray,
    right: FloatArray,
    numerical_status: UInt8Array,
) -> FloatArray:
    with np.errstate(over="ignore", invalid="ignore"):
        value = left + right
        result = np.nextafter(value, np.inf)
    _normalize_nonnegative_rows(result, numerical_status)
    return result


def _upper_quotient_batch(
    value: FloatArray,
    divisor: float,
    numerical_status: UInt8Array,
) -> FloatArray:
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        result = np.nextafter(value / divisor, np.inf)
    _normalize_nonnegative_rows(result, numerical_status)
    return result


def _weighted_upper_sum_batch(
    first: FloatArray,
    second: FloatArray,
    third: FloatArray,
    fourth: FloatArray,
    numerical_status: UInt8Array,
) -> FloatArray:
    result = _upper_sum_batch(
        first,
        _upper_product_batch(second, 2.0, numerical_status),
        numerical_status,
    )
    result = _upper_sum_batch(
        result,
        _upper_product_batch(third, 2.0, numerical_status),
        numerical_status,
    )
    return _upper_sum_batch(result, fourth, numerical_status)


def _normalize_nonnegative_rows(value: FloatArray, numerical_status: UInt8Array) -> None:
    if value.shape[0] != numerical_status.size:
        raise ValueError("enclosure arithmetic rows do not match numerical status")
    axes = tuple(range(1, value.ndim))
    invalid = ~np.isfinite(value) | (value < 0.0)
    invalid_rows = invalid if not axes else invalid.any(axis=axes)
    first_failure = (numerical_status == NUMERICAL_STATUS_OK) & invalid_rows
    numerical_status[first_failure] = INTEGRATOR_NUMERICAL_FAILURE
    value[numerical_status != NUMERICAL_STATUS_OK] = 0.0


def _curved_enclosure_from_radii(
    position_m: FloatArray,
    velocity_m_s: FloatArray,
    position_radius: FloatArray,
    velocity_radius: FloatArray,
    numerical_status: UInt8Array,
) -> CurvedPathEnclosure:
    return _normalize_enclosure_bounds(
        position_m,
        velocity_m_s,
        _outward_lower(position_m, position_radius),
        _outward_upper(position_m, position_radius),
        _outward_lower(velocity_m_s, velocity_radius),
        _outward_upper(velocity_m_s, velocity_radius),
        numerical_status,
    )


def _normalize_enclosure_bounds(
    start_position_m: FloatArray,
    start_velocity_m_s: FloatArray,
    position_lower_m: FloatArray,
    position_upper_m: FloatArray,
    velocity_lower_m_s: FloatArray,
    velocity_upper_m_s: FloatArray,
    numerical_status: UInt8Array,
) -> CurvedPathEnclosure:
    invalid = ~np.isfinite(position_lower_m).all(axis=1)
    invalid |= ~np.isfinite(position_upper_m).all(axis=1)
    invalid |= ~np.isfinite(velocity_lower_m_s).all(axis=1)
    invalid |= ~np.isfinite(velocity_upper_m_s).all(axis=1)
    invalid |= (position_lower_m > position_upper_m).any(axis=1)
    invalid |= (velocity_lower_m_s > velocity_upper_m_s).any(axis=1)
    first_failure = (numerical_status == NUMERICAL_STATUS_OK) & invalid
    numerical_status[first_failure] = INTEGRATOR_NUMERICAL_FAILURE
    failed = numerical_status != NUMERICAL_STATUS_OK
    position_lower_m[failed] = start_position_m[failed]
    position_upper_m[failed] = start_position_m[failed]
    velocity_lower_m_s[failed] = start_velocity_m_s[failed]
    velocity_upper_m_s[failed] = start_velocity_m_s[failed]
    return CurvedPathEnclosure(
        position_lower_m,
        position_upper_m,
        velocity_lower_m_s,
        velocity_upper_m_s,
        numerical_status.copy(),
    )


def _outward_lower(center: FloatArray, radius: FloatArray) -> FloatArray:
    with np.errstate(over="ignore", invalid="ignore"):
        return np.nextafter(center - radius, -np.inf)


def _outward_upper(center: FloatArray, radius: FloatArray) -> FloatArray:
    with np.errstate(over="ignore", invalid="ignore"):
        return np.nextafter(center + radius, np.inf)


def _copy_path_enclosure(
    enclosure: CurvedPathEnclosure | None,
    count: int,
    start_position_m: FloatArray,
    start_velocity_m_s: FloatArray,
) -> CurvedPathEnclosure | None:
    if enclosure is None:
        return None
    numerical_status = np.asarray(enclosure.numerical_status)
    if numerical_status.shape != (count,) or numerical_status.dtype != np.uint8:
        raise ValueError("path enclosure numerical status must be a uint8 array with shape [N]")
    result = CurvedPathEnclosure(
        np.asarray(enclosure.position_lower_m, dtype=np.float64).copy(),
        np.asarray(enclosure.position_upper_m, dtype=np.float64).copy(),
        np.asarray(enclosure.velocity_lower_m_s, dtype=np.float64).copy(),
        np.asarray(enclosure.velocity_upper_m_s, dtype=np.float64).copy(),
        numerical_status.copy(),
    )
    arrays = (
        result.position_lower_m,
        result.position_upper_m,
        result.velocity_lower_m_s,
        result.velocity_upper_m_s,
    )
    if any(value.shape != (count, 2) for value in arrays):
        raise ValueError("path enclosure arrays must have shape [N, 2]")
    _require_finite_enclosure(result)
    if bool((result.position_lower_m > result.position_upper_m).any()) or bool(
        (result.velocity_lower_m_s > result.velocity_upper_m_s).any()
    ):
        raise ValueError("path enclosure lower bounds exceed upper bounds")
    valid = result.numerical_status == NUMERICAL_STATUS_OK
    if bool(
        (start_position_m[valid] < result.position_lower_m[valid]).any()
        or (start_position_m[valid] > result.position_upper_m[valid]).any()
        or (start_velocity_m_s[valid] < result.velocity_lower_m_s[valid]).any()
        or (start_velocity_m_s[valid] > result.velocity_upper_m_s[valid]).any()
    ):
        raise ValueError("path enclosure does not contain the proposal start state")
    return result


def _require_enclosure_contains_endpoint(
    enclosure: CurvedPathEnclosure | None,
    position_m: FloatArray,
    velocity_m_s: FloatArray,
    numerical_status: UInt8Array,
) -> None:
    if enclosure is None:
        return
    valid = (enclosure.numerical_status == NUMERICAL_STATUS_OK) & (
        numerical_status == NUMERICAL_STATUS_OK
    )
    if bool(
        (position_m[valid] < enclosure.position_lower_m[valid]).any()
        or (position_m[valid] > enclosure.position_upper_m[valid]).any()
        or (velocity_m_s[valid] < enclosure.velocity_lower_m_s[valid]).any()
        or (velocity_m_s[valid] > enclosure.velocity_upper_m_s[valid]).any()
    ):
        raise ValueError("path enclosure does not contain the proposal endpoint")


def _require_finite_enclosure(enclosure: CurvedPathEnclosure) -> None:
    arrays = (
        enclosure.position_lower_m,
        enclosure.position_upper_m,
        enclosure.velocity_lower_m_s,
        enclosure.velocity_upper_m_s,
    )
    if not all(bool(np.isfinite(value).all()) for value in arrays):
        raise ValueError("RK4 path enclosure is not finite")
    if enclosure.numerical_status.shape != (enclosure.position_lower_m.shape[0],):
        raise ValueError("RK4 path enclosure numerical status has an invalid shape")
    if enclosure.numerical_status.dtype != np.uint8:
        raise ValueError("RK4 path enclosure numerical status must have dtype uint8")


def _advance(
    path_kind: PathKind,
    evaluator: StageEvaluator | None,
    relaxation_evaluator: RelaxationStageEvaluator | None,
    particle_index: Int64Array,
    start_time_s: FloatArray,
    elapsed_s: FloatArray,
    position_m: FloatArray,
    velocity_m_s: FloatArray,
    charge_number: FloatArray,
    constant_acceleration_m_s2: FloatArray | None,
) -> _AdvanceResult:
    if path_kind == "linear_exact":
        return _advance_linear(position_m, velocity_m_s, charge_number, elapsed_s)
    if path_kind == "exponential_midpoint_reintegrated":
        if relaxation_evaluator is None:
            raise ValueError("exponential-midpoint proposal lost its relaxation evaluator")
        return _advance_exponential_midpoint(
            relaxation_evaluator,
            particle_index,
            start_time_s,
            elapsed_s,
            position_m,
            velocity_m_s,
            charge_number,
        )
    if evaluator is None:
        raise ValueError("force-coupled proposal lost its stage evaluator")
    if path_kind == "quadratic_exact":
        if constant_acceleration_m_s2 is None:
            raise ValueError("quadratic proposal lost its certified acceleration")
        return _advance_quadratic(
            evaluator,
            particle_index,
            start_time_s,
            elapsed_s,
            position_m,
            velocity_m_s,
            charge_number,
            constant_acceleration_m_s2,
        )
    return _advance_rk4(
        evaluator,
        particle_index,
        start_time_s,
        elapsed_s,
        position_m,
        velocity_m_s,
        charge_number,
    )


def _advance_with_initial_status(
    path_kind: PathKind,
    evaluator: StageEvaluator | None,
    relaxation_evaluator: RelaxationStageEvaluator | None,
    particle_index: Int64Array,
    start_time_s: FloatArray,
    elapsed_s: FloatArray,
    position_m: FloatArray,
    velocity_m_s: FloatArray,
    charge_number: FloatArray,
    constant_acceleration_m_s2: FloatArray | None,
    initial_status: UInt8Array | None,
) -> _AdvanceResult:
    if initial_status is None:
        return _advance(
            path_kind,
            evaluator,
            relaxation_evaluator,
            particle_index,
            start_time_s,
            elapsed_s,
            position_m,
            velocity_m_s,
            charge_number,
            constant_acceleration_m_s2,
        )
    initial = np.asarray(initial_status)
    if initial.shape != (particle_index.size,) or initial.dtype != np.uint8:
        raise ValueError("initial numerical status must be a uint8 array with shape [N]")
    numerical_status = initial.copy()
    if not bool((numerical_status != NUMERICAL_STATUS_OK).any()):
        return _advance(
            path_kind,
            evaluator,
            relaxation_evaluator,
            particle_index,
            start_time_s,
            elapsed_s,
            position_m,
            velocity_m_s,
            charge_number,
            constant_acceleration_m_s2,
        )
    rows = np.flatnonzero(numerical_status == NUMERICAL_STATUS_OK).astype("<i8", copy=False)
    position = position_m.copy()
    velocity = velocity_m_s.copy()
    charge = charge_number.copy()
    support = np.ones(particle_index.size, dtype=np.bool_)
    applicable = np.ones(particle_index.size, dtype=np.bool_)
    end_field_cell_id: Int64Array | None = None
    dense_controls = (
        _degenerate_rk4_dense_controls(position_m, velocity_m_s, charge_number)
        if path_kind == "rk4_dense"
        else None
    )
    if rows.size:
        acceleration = (
            None if constant_acceleration_m_s2 is None else constant_acceleration_m_s2[rows]
        )
        advanced = _advance(
            path_kind,
            evaluator,
            relaxation_evaluator,
            particle_index[rows],
            start_time_s[rows],
            elapsed_s[rows],
            position_m[rows],
            velocity_m_s[rows],
            charge_number[rows],
            acceleration,
        )
        position[rows] = advanced.position_m
        velocity[rows] = advanced.velocity_m_s
        charge[rows] = advanced.charge_number
        support[rows] = advanced.support_inside
        applicable[rows] = advanced.applicability_inside
        _merge_numerical_status(numerical_status, rows, advanced.numerical_status)
        if advanced.end_field_cell_id is not None:
            end_field_cell_id = np.full(particle_index.size, -1, dtype=np.int64)
            end_field_cell_id[rows] = advanced.end_field_cell_id
        if dense_controls is not None:
            if advanced.rk4_dense_controls is None:
                raise ValueError("RK4 advance did not return dense controls")
            dense_controls.position_inner_m[rows] = advanced.rk4_dense_controls.position_inner_m
            dense_controls.velocity_inner_m_s[rows] = advanced.rk4_dense_controls.velocity_inner_m_s
            dense_controls.charge_inner_number[rows] = (
                advanced.rk4_dense_controls.charge_inner_number
            )
    return _AdvanceResult(
        position,
        velocity,
        charge,
        support,
        applicable,
        numerical_status,
        end_field_cell_id,
        dense_controls,
    )


def _evaluate_active_dynamics(
    evaluator: StageEvaluator,
    particle_index: Int64Array,
    time_s: FloatArray,
    position_m: FloatArray,
    velocity_m_s: FloatArray,
    charge_number: FloatArray,
    numerical_status: UInt8Array,
) -> tuple[RowSelection, DynamicsEvaluation | None]:
    if not bool((numerical_status != NUMERICAL_STATUS_OK).any()):
        stage = evaluator(
            particle_index,
            time_s,
            position_m,
            velocity_m_s,
            charge_number,
        )
        _validate_stage(stage, particle_index.size)
        numerical_status[:] = stage.numerical_status
        return slice(None), stage
    rows = np.flatnonzero(numerical_status == NUMERICAL_STATUS_OK).astype("<i8", copy=False)
    if not rows.size:
        return rows, None
    stage = evaluator(
        particle_index[rows],
        time_s[rows],
        position_m[rows],
        velocity_m_s[rows],
        charge_number[rows],
    )
    _validate_stage(stage, rows.size)
    _merge_numerical_status(numerical_status, rows, stage.numerical_status)
    return rows, stage


def _evaluate_active_relaxation(
    evaluator: RelaxationStageEvaluator,
    particle_index: Int64Array,
    time_s: FloatArray,
    position_m: FloatArray,
    velocity_m_s: FloatArray,
    charge_number: FloatArray,
    numerical_status: UInt8Array,
) -> tuple[RowSelection, RelaxationEvaluation | None]:
    if not bool((numerical_status != NUMERICAL_STATUS_OK).any()):
        stage = evaluator(
            particle_index,
            time_s,
            position_m,
            velocity_m_s,
            charge_number,
        )
        _validate_relaxation_stage(stage, particle_index.size)
        numerical_status[:] = stage.numerical_status
        return slice(None), stage
    rows = np.flatnonzero(numerical_status == NUMERICAL_STATUS_OK).astype("<i8", copy=False)
    if not rows.size:
        return rows, None
    stage = evaluator(
        particle_index[rows],
        time_s[rows],
        position_m[rows],
        velocity_m_s[rows],
        charge_number[rows],
    )
    _validate_relaxation_stage(stage, rows.size)
    _merge_numerical_status(numerical_status, rows, stage.numerical_status)
    return rows, stage


def _rk4_stage_state_active(
    start_position_m: FloatArray,
    start_velocity_m_s: FloatArray,
    start_charge_number: FloatArray,
    stage_velocity_m_s: FloatArray,
    stage_acceleration_m_s2: FloatArray,
    stage_charge_rate_number_s: FloatArray,
    elapsed_s: FloatArray,
    numerical_status: UInt8Array,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    if not bool((numerical_status != NUMERICAL_STATUS_OK).any()):
        candidate = _compiled_rk4_stage_state(
            start_position_m,
            start_velocity_m_s,
            start_charge_number,
            stage_velocity_m_s,
            stage_acceleration_m_s2,
            stage_charge_rate_number_s,
            elapsed_s,
        )
        rows = np.arange(numerical_status.size, dtype=np.int64)
        finite = _mark_nonfinite_state(numerical_status, rows, *candidate)
        if bool(finite.all()):
            return candidate
        position = start_position_m.copy()
        velocity = start_velocity_m_s.copy()
        charge = start_charge_number.copy()
        position[finite] = candidate[0][finite]
        velocity[finite] = candidate[1][finite]
        charge[finite] = candidate[2][finite]
        return position, velocity, charge
    position = start_position_m.copy()
    velocity = start_velocity_m_s.copy()
    charge = start_charge_number.copy()
    rows = np.flatnonzero(numerical_status == NUMERICAL_STATUS_OK).astype("<i8", copy=False)
    if not rows.size:
        return position, velocity, charge
    candidate = _compiled_rk4_stage_state(
        start_position_m[rows],
        start_velocity_m_s[rows],
        start_charge_number[rows],
        stage_velocity_m_s[rows],
        stage_acceleration_m_s2[rows],
        stage_charge_rate_number_s[rows],
        elapsed_s[rows],
    )
    finite = _mark_nonfinite_state(numerical_status, rows, *candidate)
    survivors = rows[finite]
    position[survivors] = candidate[0][finite]
    velocity[survivors] = candidate[1][finite]
    charge[survivors] = candidate[2][finite]
    return position, velocity, charge


def _exponential_update_active(
    start_position_m: FloatArray,
    start_velocity_m_s: FloatArray,
    charge_number: FloatArray,
    linear_drag_rate_s_inv: FloatArray,
    target_velocity_m_s: FloatArray,
    additive_acceleration_m_s2: FloatArray,
    step_s: FloatArray,
    numerical_status: UInt8Array,
) -> tuple[FloatArray, FloatArray]:
    if not bool((numerical_status != NUMERICAL_STATUS_OK).any()):
        candidate_position, candidate_velocity = _compiled_exponential_update(
            start_position_m,
            start_velocity_m_s,
            linear_drag_rate_s_inv,
            target_velocity_m_s,
            additive_acceleration_m_s2,
            step_s,
        )
        rows = np.arange(numerical_status.size, dtype=np.int64)
        finite = _mark_nonfinite_state(
            numerical_status,
            rows,
            candidate_position,
            candidate_velocity,
            charge_number,
        )
        if bool(finite.all()):
            return candidate_position, candidate_velocity
        position = start_position_m.copy()
        velocity = start_velocity_m_s.copy()
        position[finite] = candidate_position[finite]
        velocity[finite] = candidate_velocity[finite]
        return position, velocity
    position = start_position_m.copy()
    velocity = start_velocity_m_s.copy()
    rows = np.flatnonzero(numerical_status == NUMERICAL_STATUS_OK).astype("<i8", copy=False)
    if not rows.size:
        return position, velocity
    candidate_position, candidate_velocity = _compiled_exponential_update(
        start_position_m[rows],
        start_velocity_m_s[rows],
        linear_drag_rate_s_inv[rows],
        target_velocity_m_s[rows],
        additive_acceleration_m_s2[rows],
        step_s[rows],
    )
    finite = _mark_nonfinite_state(
        numerical_status,
        rows,
        candidate_position,
        candidate_velocity,
        charge_number[rows],
    )
    survivors = rows[finite]
    position[survivors] = candidate_position[finite]
    velocity[survivors] = candidate_velocity[finite]
    return position, velocity


def _advance_exponential_midpoint(
    evaluator: RelaxationStageEvaluator,
    particle_index: Int64Array,
    start_time_s: FloatArray,
    elapsed_s: FloatArray,
    position_m: FloatArray,
    velocity_m_s: FloatArray,
    charge_number: FloatArray,
) -> _AdvanceResult:
    """Advance positive rows and inspect the accepted endpoint."""

    count = int(particle_index.size)
    result_position = position_m.copy()
    result_velocity = velocity_m_s.copy()
    result_charge = charge_number.copy()
    support = np.ones(count, dtype=np.bool_)
    applicable = np.ones(count, dtype=np.bool_)
    numerical_status = np.full(count, NUMERICAL_STATUS_OK, dtype=np.uint8)
    moving = np.flatnonzero(elapsed_s > 0.0)
    if moving.size:
        advanced = _advance_positive_exponential_midpoint(
            evaluator,
            particle_index[moving],
            start_time_s[moving],
            elapsed_s[moving],
            position_m[moving],
            velocity_m_s[moving],
            charge_number[moving],
        )
        result_position[moving] = advanced.position_m
        result_velocity[moving] = advanced.velocity_m_s
        result_charge[moving] = advanced.charge_number
        support[moving] = advanced.support_inside
        applicable[moving] = advanced.applicability_inside
        numerical_status[moving] = advanced.numerical_status
    _require_finite_state(result_position, result_velocity, result_charge)
    rows, endpoint = _evaluate_active_relaxation(
        evaluator,
        particle_index,
        start_time_s + elapsed_s,
        result_position,
        result_velocity,
        result_charge,
        numerical_status,
    )
    end_field_cell_id: Int64Array | None = None
    if endpoint is not None:
        support[rows] &= endpoint.support_inside
        applicable[rows] &= endpoint.applicability_inside
        if endpoint.field_cell_id is not None:
            end_field_cell_id = np.full(count, -1, dtype=np.int64)
            end_field_cell_id[rows] = endpoint.field_cell_id
    return _AdvanceResult(
        result_position,
        result_velocity,
        result_charge,
        support,
        applicable,
        numerical_status,
        end_field_cell_id,
    )


def _advance_positive_exponential_midpoint(
    evaluator: RelaxationStageEvaluator,
    particle_index: Int64Array,
    start_time_s: FloatArray,
    step_s: FloatArray,
    position_m: FloatArray,
    velocity_m_s: FloatArray,
    charge_number: FloatArray,
) -> _AdvanceResult:
    """Use a start half-step predictor and midpoint-frozen analytic update."""

    count = int(particle_index.size)
    numerical_status = np.full(count, NUMERICAL_STATUS_OK, dtype=np.uint8)
    support = np.ones(count, dtype=np.bool_)
    applicable = np.ones(count, dtype=np.bool_)
    first_rate = np.zeros(count, dtype=np.float64)
    first_target = np.zeros((count, 2), dtype=np.float64)
    first_additive = np.zeros((count, 2), dtype=np.float64)
    first_charge_rate = np.zeros(count, dtype=np.float64)
    first_charge_derivative = np.zeros(count, dtype=np.float64)
    rows, first = _evaluate_active_relaxation(
        evaluator,
        particle_index,
        start_time_s,
        position_m,
        velocity_m_s,
        charge_number,
        numerical_status,
    )
    if first is not None:
        support[rows] &= first.support_inside
        applicable[rows] &= first.applicability_inside
        first_rate[rows] = first.linear_drag_rate_s_inv
        first_target[rows] = first.target_velocity_m_s
        first_additive[rows] = first.additive_acceleration_m_s2
        first_charge_rate[rows] = first.charge_rate_number_s
        first_charge_derivative[rows] = first.charge_rate_derivative_s_inv
    half_step = 0.5 * step_s
    midpoint_position, midpoint_velocity = _exponential_update_active(
        position_m,
        velocity_m_s,
        charge_number,
        first_rate,
        first_target,
        first_additive,
        half_step,
        numerical_status,
    )
    midpoint_charge = _stable_charge_update_active(
        charge_number,
        charge_number,
        first_charge_rate,
        first_charge_derivative,
        half_step,
        numerical_status,
    )
    midpoint_rate = np.zeros(count, dtype=np.float64)
    midpoint_target = np.zeros((count, 2), dtype=np.float64)
    midpoint_additive = np.zeros((count, 2), dtype=np.float64)
    midpoint_charge_rate = np.zeros(count, dtype=np.float64)
    midpoint_charge_derivative = np.zeros(count, dtype=np.float64)
    rows, midpoint = _evaluate_active_relaxation(
        evaluator,
        particle_index,
        start_time_s + half_step,
        midpoint_position,
        midpoint_velocity,
        midpoint_charge,
        numerical_status,
    )
    if midpoint is not None:
        support[rows] &= midpoint.support_inside
        applicable[rows] &= midpoint.applicability_inside
        midpoint_rate[rows] = midpoint.linear_drag_rate_s_inv
        midpoint_target[rows] = midpoint.target_velocity_m_s
        midpoint_additive[rows] = midpoint.additive_acceleration_m_s2
        midpoint_charge_rate[rows] = midpoint.charge_rate_number_s
        midpoint_charge_derivative[rows] = midpoint.charge_rate_derivative_s_inv
    end_charge = _stable_charge_update_active(
        charge_number,
        midpoint_charge,
        midpoint_charge_rate,
        midpoint_charge_derivative,
        step_s,
        numerical_status,
    )
    end_position, end_velocity = _exponential_update_active(
        position_m,
        velocity_m_s,
        end_charge,
        midpoint_rate,
        midpoint_target,
        midpoint_additive,
        step_s,
        numerical_status,
    )
    return _AdvanceResult(
        end_position,
        end_velocity,
        end_charge,
        support,
        applicable,
        numerical_status,
        None,
    )


def _stable_charge_update_active(
    start_charge_number: FloatArray,
    reference_charge_number: FloatArray,
    charge_rate_number_s: FloatArray,
    charge_rate_derivative_s_inv: FloatArray,
    elapsed_s: FloatArray,
    numerical_status: UInt8Array,
) -> FloatArray:
    """Advance one midpoint-frozen affine charge law with an exponential."""

    result = start_charge_number.copy()
    rows = np.flatnonzero(numerical_status == NUMERICAL_STATUS_OK).astype("<i8", copy=False)
    if not rows.size:
        return result
    start = start_charge_number[rows]
    reference = reference_charge_number[rows]
    rate = charge_rate_number_s[rows]
    derivative = charge_rate_derivative_s_inv[rows]
    elapsed = elapsed_s[rows]
    valid = (
        np.isfinite(start)
        & np.isfinite(reference)
        & np.isfinite(rate)
        & np.isfinite(derivative)
        & np.isfinite(elapsed)
        & (elapsed >= 0.0)
        & (derivative <= 0.0)
    )
    affine_start_rate = rate + derivative * (start - reference)
    multiplier = _affine_exponential_multiplier(derivative, elapsed)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        candidate = start + multiplier * affine_start_rate
    valid &= np.isfinite(candidate)
    numerical_status[rows[~valid]] = INTEGRATOR_NUMERICAL_FAILURE
    result[rows[valid]] = candidate[valid]
    return result


def _affine_exponential_multiplier(
    rate_derivative_s_inv: FloatArray,
    elapsed_s: FloatArray,
) -> FloatArray:
    """Return ``expm1(J h) / J`` with its exact ``J=0`` limit."""

    result = elapsed_s.copy()
    relaxing = rate_derivative_s_inv < 0.0
    argument = np.zeros_like(elapsed_s)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        argument[relaxing] = rate_derivative_s_inv[relaxing] * elapsed_s[relaxing]
        resolved = relaxing & np.isfinite(argument) & (argument != 0.0)
        result[resolved] = elapsed_s[resolved] * (np.expm1(argument[resolved]) / argument[resolved])
        saturated = relaxing & np.isneginf(argument)
        result[saturated] = -1.0 / rate_derivative_s_inv[saturated]
    return result


def affine_exponential_charge(
    root_charge_number: FloatArray,
    elapsed_s: FloatArray,
    affine_rate_number_s: FloatArray,
    rate_derivative_s_inv: FloatArray,
) -> FloatArray:
    """Evaluate the one root-owned affine-exponential charge representation."""

    root = np.asarray(root_charge_number, dtype=np.float64)
    elapsed = np.asarray(elapsed_s, dtype=np.float64)
    affine_rate = np.asarray(affine_rate_number_s, dtype=np.float64)
    derivative = np.asarray(rate_derivative_s_inv, dtype=np.float64)
    if not (root.shape == elapsed.shape == affine_rate.shape == derivative.shape):
        raise ValueError("affine-exponential charge columns must have matching shapes")
    if not bool(
        np.isfinite(root).all()
        and np.isfinite(elapsed).all()
        and np.isfinite(affine_rate).all()
        and np.isfinite(derivative).all()
        and (elapsed >= 0.0).all()
        and (derivative <= 0.0).all()
    ):
        raise ValueError("affine-exponential charge inputs are invalid")
    multiplier = _affine_exponential_multiplier(derivative, elapsed)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        result = root + multiplier * affine_rate
    # Brownian execution owns row-local numerical failure.  Returning an
    # arithmetic overflow here lets that caller retire only the affected row;
    # malformed inputs above remain fail-fast structural errors.
    return result


def _advance_linear(
    position_m: FloatArray,
    velocity_m_s: FloatArray,
    charge_number: FloatArray,
    elapsed_s: FloatArray,
) -> _AdvanceResult:
    with np.errstate(over="ignore", invalid="ignore"):
        candidate_position = position_m + elapsed_s[:, None] * velocity_m_s
    position = position_m.copy()
    finite = np.isfinite(candidate_position).all(axis=1)
    position[finite] = candidate_position[finite]
    exact_start = elapsed_s == 0.0
    if bool(exact_start.any()):
        position[exact_start] = position_m[exact_start]
    count = int(elapsed_s.size)
    numerical_status = np.full(count, NUMERICAL_STATUS_OK, dtype=np.uint8)
    numerical_status[~finite] = INTEGRATOR_NUMERICAL_FAILURE
    return _AdvanceResult(
        position,
        velocity_m_s.copy(),
        charge_number.copy(),
        np.ones(count, dtype=np.bool_),
        np.ones(count, dtype=np.bool_),
        numerical_status,
        None,
    )


def _advance_quadratic(
    evaluator: StageEvaluator,
    particle_index: Int64Array,
    start_time_s: FloatArray,
    elapsed_s: FloatArray,
    position_m: FloatArray,
    velocity_m_s: FloatArray,
    charge_number: FloatArray,
    acceleration_m_s2: FloatArray,
) -> _AdvanceResult:
    """Evaluate a prepare-certified constant-acceleration path exactly."""

    count = int(particle_index.size)
    numerical_status = np.full(count, NUMERICAL_STATUS_OK, dtype=np.uint8)
    half_elapsed = 0.5 * elapsed_s
    with np.errstate(over="ignore", invalid="ignore"):
        midpoint_position = _quadratic_position(
            position_m,
            velocity_m_s,
            acceleration_m_s2,
            half_elapsed,
        )
        midpoint_velocity = velocity_m_s + half_elapsed[:, None] * acceleration_m_s2
        end_position = _quadratic_position(
            position_m,
            velocity_m_s,
            acceleration_m_s2,
            elapsed_s,
        )
        end_velocity = velocity_m_s + elapsed_s[:, None] * acceleration_m_s2
    finite = (
        np.isfinite(midpoint_position).all(axis=1)
        & np.isfinite(midpoint_velocity).all(axis=1)
        & np.isfinite(end_position).all(axis=1)
        & np.isfinite(end_velocity).all(axis=1)
    )
    numerical_status[~finite] = INTEGRATOR_NUMERICAL_FAILURE
    midpoint_position[~finite] = position_m[~finite]
    midpoint_velocity[~finite] = velocity_m_s[~finite]
    end_position[~finite] = position_m[~finite]
    end_velocity[~finite] = velocity_m_s[~finite]
    support = np.ones(count, dtype=np.bool_)
    applicable = np.ones(count, dtype=np.bool_)
    end_field_cell_id: Int64Array | None = None
    stage_inputs = (
        (start_time_s, position_m, velocity_m_s),
        (start_time_s + half_elapsed, midpoint_position, midpoint_velocity),
        (start_time_s + elapsed_s, end_position, end_velocity),
    )
    for stage_index, (stage_time, stage_position, stage_velocity) in enumerate(stage_inputs):
        rows, stage = _evaluate_active_dynamics(
            evaluator,
            particle_index,
            stage_time,
            stage_position,
            stage_velocity,
            charge_number,
            numerical_status,
        )
        if stage is None:
            continue
        if not bool((stage.charge_rate_number_s == 0.0).all()):
            raise ValueError("quadratic exact path requires fixed charge")
        support[rows] &= stage.support_inside
        applicable[rows] &= stage.applicability_inside
        if stage_index == 2 and stage.field_cell_id is not None:
            end_field_cell_id = np.full(count, -1, dtype=np.int64)
            end_field_cell_id[rows] = stage.field_cell_id
    exact_start = elapsed_s == 0.0
    if bool(exact_start.any()):
        end_position[exact_start] = position_m[exact_start]
        end_velocity[exact_start] = velocity_m_s[exact_start]
    return _AdvanceResult(
        end_position,
        end_velocity,
        charge_number.copy(),
        support,
        applicable,
        numerical_status,
        end_field_cell_id,
    )


def _quadratic_position(
    start_position_m: FloatArray,
    start_velocity_m_s: FloatArray,
    acceleration_m_s2: FloatArray,
    elapsed_s: FloatArray,
) -> FloatArray:
    """Evaluate the constant-acceleration position with one operation order."""

    with np.errstate(over="ignore", invalid="ignore"):
        return (
            start_position_m
            + elapsed_s[:, None] * start_velocity_m_s
            + 0.5 * (elapsed_s * elapsed_s)[:, None] * acceleration_m_s2
        )


def _enclose_quadratic_position(
    start_position_m: FloatArray,
    start_velocity_m_s: FloatArray,
    acceleration_m_s2: FloatArray,
    duration_s: FloatArray,
    end_position_m: FloatArray,
) -> _PositionInterval:
    """Enclose every float64 dense-state value of an exact parabola."""

    lower_m = np.minimum(start_position_m, end_position_m)
    upper_m = np.maximum(start_position_m, end_position_m)
    for axis in range(2):
        axis_acceleration = acceleration_m_s2[:, axis]
        turn_time_s = np.full(axis_acceleration.shape, np.inf, dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            np.divide(
                -start_velocity_m_s[:, axis],
                axis_acceleration,
                out=turn_time_s,
                where=axis_acceleration != 0.0,
            )
        turning_rows = np.flatnonzero((turn_time_s > 0.0) & (turn_time_s < duration_s))
        if not turning_rows.size:
            continue
        turn_position = _quadratic_position(
            start_position_m[turning_rows],
            start_velocity_m_s[turning_rows],
            acceleration_m_s2[turning_rows],
            turn_time_s[turning_rows],
        )[:, axis]
        lower_m[turning_rows, axis] = np.minimum(
            lower_m[turning_rows, axis],
            turn_position,
        )
        upper_m[turning_rows, axis] = np.maximum(
            upper_m[turning_rows, axis],
            turn_position,
        )
    padding_m = _quadratic_position_roundoff_padding(
        start_position_m,
        start_velocity_m_s,
        acceleration_m_s2,
        duration_s,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        lower_m = np.nextafter(lower_m - padding_m, -np.inf)
        upper_m = np.nextafter(upper_m + padding_m, np.inf)
    if not bool(np.isfinite(lower_m).all() and np.isfinite(upper_m).all()):
        raise ValueError("quadratic position interval is not finite")
    return _PositionInterval(lower_m=lower_m, upper_m=upper_m)


def _quadratic_position_roundoff_padding(
    start_position_m: FloatArray,
    start_velocity_m_s: FloatArray,
    acceleration_m_s2: FloatArray,
    duration_s: FloatArray,
) -> FloatArray:
    """Bound evaluation and turning-time roundoff by the polynomial term scale."""

    duration = duration_s[:, None]
    with np.errstate(over="ignore", invalid="ignore"):
        absolute_term_sum = (
            np.abs(start_position_m)
            + duration * np.abs(start_velocity_m_s)
            + 0.5 * (duration * duration) * np.abs(acceleration_m_s2)
        )
        normal_padding_m = np.nextafter(
            _QUADRATIC_ROUNDOFF_EPS_FACTOR * np.finfo(np.float64).eps * absolute_term_sum,
            np.inf,
        )
        underflow_scale = np.nextafter(1.0 + np.abs(acceleration_m_s2), np.inf)
        underflow_padding_m = np.nextafter(
            _QUADRATIC_ROUNDOFF_EPS_FACTOR
            * np.finfo(np.float64).smallest_subnormal
            * underflow_scale,
            np.inf,
        )
        padding_m = np.nextafter(normal_padding_m + underflow_padding_m, np.inf)
    return padding_m


def _degenerate_rk4_dense_controls(
    position_m: FloatArray,
    velocity_m_s: FloatArray,
    charge_number: FloatArray,
) -> _Rk4DenseControls:
    return _Rk4DenseControls(
        position_inner_m=np.repeat(position_m[:, None, :], 2, axis=1),
        velocity_inner_m_s=np.repeat(velocity_m_s[:, None, :], 2, axis=1),
        charge_inner_number=np.repeat(charge_number[:, None], 2, axis=1),
    )


def _rk4_dense_inner_pair(
    start_value: FloatArray,
    end_value: FloatArray,
    start_derivative: FloatArray,
    end_derivative: FloatArray,
    step_s: FloatArray,
    numerical_status: UInt8Array,
) -> FloatArray:
    scale_shape = (step_s.size,) + (1,) * (start_value.ndim - 1)
    one_third_step = (step_s / 3.0).reshape(scale_shape)
    with np.errstate(over="ignore", invalid="ignore"):
        first = start_value + one_third_step * start_derivative
        second = end_value - one_third_step * end_derivative
    result = np.stack((first, second), axis=1)
    axes = tuple(range(1, result.ndim))
    invalid = ~np.isfinite(result).all(axis=axes)
    numerical_status[invalid & (numerical_status == NUMERICAL_STATUS_OK)] = (
        INTEGRATOR_NUMERICAL_FAILURE
    )
    failed = numerical_status != NUMERICAL_STATUS_OK
    result[failed, 0] = start_value[failed]
    result[failed, 1] = end_value[failed]
    return result


def _build_rk4_dense_controls(
    start_position_m: FloatArray,
    start_velocity_m_s: FloatArray,
    start_charge_number: FloatArray,
    end_position_m: FloatArray,
    end_velocity_m_s: FloatArray,
    end_charge_number: FloatArray,
    first_acceleration_m_s2: FloatArray,
    fourth_acceleration_m_s2: FloatArray,
    first_charge_rate_number_s: FloatArray,
    fourth_charge_rate_number_s: FloatArray,
    step_s: FloatArray,
    numerical_status: UInt8Array,
) -> _Rk4DenseControls:
    position_inner = _rk4_dense_inner_pair(
        start_position_m,
        end_position_m,
        start_velocity_m_s,
        end_velocity_m_s,
        step_s,
        numerical_status,
    )
    velocity_inner = _rk4_dense_inner_pair(
        start_velocity_m_s,
        end_velocity_m_s,
        first_acceleration_m_s2,
        fourth_acceleration_m_s2,
        step_s,
        numerical_status,
    )
    charge_inner = _rk4_dense_inner_pair(
        start_charge_number,
        end_charge_number,
        first_charge_rate_number_s,
        fourth_charge_rate_number_s,
        step_s,
        numerical_status,
    )
    return _Rk4DenseControls(position_inner, velocity_inner, charge_inner)


def _advance_rk4(
    evaluator: StageEvaluator,
    particle_index: Int64Array,
    start_time_s: FloatArray,
    elapsed_s: FloatArray,
    position_m: FloatArray,
    velocity_m_s: FloatArray,
    charge_number: FloatArray,
) -> _AdvanceResult:
    count = int(particle_index.size)
    result_position = position_m.copy()
    result_velocity = velocity_m_s.copy()
    result_charge = charge_number.copy()
    support = np.ones(count, dtype=np.bool_)
    applicable = np.ones(count, dtype=np.bool_)
    numerical_status = np.full(count, NUMERICAL_STATUS_OK, dtype=np.uint8)
    dense_controls = _degenerate_rk4_dense_controls(
        position_m,
        velocity_m_s,
        charge_number,
    )
    moving = np.flatnonzero(elapsed_s > 0.0)
    if moving.size:
        advanced = _advance_positive_rk4(
            evaluator,
            particle_index[moving],
            start_time_s[moving],
            elapsed_s[moving],
            position_m[moving],
            velocity_m_s[moving],
            charge_number[moving],
        )
        result_position[moving] = advanced.position_m
        result_velocity[moving] = advanced.velocity_m_s
        result_charge[moving] = advanced.charge_number
        support[moving] = advanced.support_inside
        applicable[moving] = advanced.applicability_inside
        numerical_status[moving] = advanced.numerical_status
        if advanced.rk4_dense_controls is None:
            raise ValueError("positive RK4 advance did not return dense controls")
        dense_controls.position_inner_m[moving] = advanced.rk4_dense_controls.position_inner_m
        dense_controls.velocity_inner_m_s[moving] = advanced.rk4_dense_controls.velocity_inner_m_s
        dense_controls.charge_inner_number[moving] = advanced.rk4_dense_controls.charge_inner_number
    _require_finite_state(result_position, result_velocity, result_charge)
    rows, endpoint = _evaluate_active_dynamics(
        evaluator,
        particle_index,
        start_time_s + elapsed_s,
        result_position,
        result_velocity,
        result_charge,
        numerical_status,
    )
    end_field_cell_id: Int64Array | None = None
    if endpoint is not None:
        support[rows] &= endpoint.support_inside
        applicable[rows] &= endpoint.applicability_inside
        if endpoint.field_cell_id is not None:
            end_field_cell_id = np.full(count, -1, dtype=np.int64)
            end_field_cell_id[rows] = endpoint.field_cell_id
    return _AdvanceResult(
        result_position,
        result_velocity,
        result_charge,
        support,
        applicable,
        numerical_status,
        end_field_cell_id,
        dense_controls,
    )


def _advance_positive_rk4(
    evaluator: StageEvaluator,
    particle_index: Int64Array,
    start_time_s: FloatArray,
    step_s: FloatArray,
    position_m: FloatArray,
    velocity_m_s: FloatArray,
    charge_number: FloatArray,
) -> _AdvanceResult:
    count = int(particle_index.size)
    numerical_status = np.full(count, NUMERICAL_STATUS_OK, dtype=np.uint8)
    support = np.ones(count, dtype=np.bool_)
    applicable = np.ones(count, dtype=np.bool_)
    acceleration = [np.zeros((count, 2), dtype=np.float64) for _ in range(4)]
    charge_rate = [np.zeros(count, dtype=np.float64) for _ in range(4)]
    half_step = 0.5 * step_s
    rows, first = _evaluate_active_dynamics(
        evaluator,
        particle_index,
        start_time_s,
        position_m,
        velocity_m_s,
        charge_number,
        numerical_status,
    )
    if first is not None:
        support[rows] &= first.support_inside
        applicable[rows] &= first.applicability_inside
        acceleration[0][rows] = first.acceleration_m_s2
        charge_rate[0][rows] = first.charge_rate_number_s
    position_2, velocity_2, charge_2 = _rk4_stage_state_active(
        position_m,
        velocity_m_s,
        charge_number,
        velocity_m_s,
        acceleration[0],
        charge_rate[0],
        half_step,
        numerical_status,
    )
    rows, second = _evaluate_active_dynamics(
        evaluator,
        particle_index,
        start_time_s + half_step,
        position_2,
        velocity_2,
        charge_2,
        numerical_status,
    )
    if second is not None:
        support[rows] &= second.support_inside
        applicable[rows] &= second.applicability_inside
        acceleration[1][rows] = second.acceleration_m_s2
        charge_rate[1][rows] = second.charge_rate_number_s
    position_3, velocity_3, charge_3 = _rk4_stage_state_active(
        position_m,
        velocity_m_s,
        charge_number,
        velocity_2,
        acceleration[1],
        charge_rate[1],
        half_step,
        numerical_status,
    )
    rows, third = _evaluate_active_dynamics(
        evaluator,
        particle_index,
        start_time_s + half_step,
        position_3,
        velocity_3,
        charge_3,
        numerical_status,
    )
    if third is not None:
        support[rows] &= third.support_inside
        applicable[rows] &= third.applicability_inside
        acceleration[2][rows] = third.acceleration_m_s2
        charge_rate[2][rows] = third.charge_rate_number_s
    position_4, velocity_4, charge_4 = _rk4_stage_state_active(
        position_m,
        velocity_m_s,
        charge_number,
        velocity_3,
        acceleration[2],
        charge_rate[2],
        step_s,
        numerical_status,
    )
    rows, fourth = _evaluate_active_dynamics(
        evaluator,
        particle_index,
        start_time_s + step_s,
        position_4,
        velocity_4,
        charge_4,
        numerical_status,
    )
    if fourth is not None:
        support[rows] &= fourth.support_inside
        applicable[rows] &= fourth.applicability_inside
        acceleration[3][rows] = fourth.acceleration_m_s2
        charge_rate[3][rows] = fourth.charge_rate_number_s

    end_position = position_m.copy()
    end_velocity = velocity_m_s.copy()
    end_charge = charge_number.copy()
    rows = np.flatnonzero(numerical_status == NUMERICAL_STATUS_OK).astype("<i8", copy=False)
    if rows.size:
        candidate = _compiled_rk4_endpoint(
            position_m[rows],
            velocity_m_s[rows],
            charge_number[rows],
            velocity_2[rows],
            velocity_3[rows],
            velocity_4[rows],
            acceleration[0][rows],
            acceleration[1][rows],
            acceleration[2][rows],
            acceleration[3][rows],
            charge_rate[0][rows],
            charge_rate[1][rows],
            charge_rate[2][rows],
            charge_rate[3][rows],
            step_s[rows],
        )
        finite = _mark_nonfinite_state(numerical_status, rows, *candidate)
        survivors = rows[finite]
        end_position[survivors] = candidate[0][finite]
        end_velocity[survivors] = candidate[1][finite]
        end_charge[survivors] = candidate[2][finite]
    dense_controls = _build_rk4_dense_controls(
        position_m,
        velocity_m_s,
        charge_number,
        end_position,
        end_velocity,
        end_charge,
        acceleration[0],
        acceleration[3],
        charge_rate[0],
        charge_rate[3],
        step_s,
        numerical_status,
    )
    return _AdvanceResult(
        end_position,
        end_velocity,
        end_charge,
        support,
        applicable,
        numerical_status,
        None,
        dense_controls,
    )


@njit(cache=True, fastmath=False, parallel=False, nogil=True, error_model="numpy")
def _compiled_exponential_update_into(
    start_position_m: FloatArray,
    start_velocity_m_s: FloatArray,
    linear_drag_rate_s_inv: FloatArray,
    target_velocity_m_s: FloatArray,
    additive_acceleration_m_s2: FloatArray,
    step_s: FloatArray,
    position: FloatArray,
    velocity: FloatArray,
) -> None:
    """Apply the midpoint-frozen update into caller-owned buffers."""

    count = step_s.size
    for row in range(count):
        step = step_s[row]
        rate = linear_drag_rate_s_inv[row]
        argument = 0.0
        if rate == 0.0:
            attenuation = 0.0
            velocity_memory = step
            target_memory = 0.0
            acceleration_memory = 0.5 * step * step
            small_argument = True
        else:
            argument = step * rate
            attenuation = -math.expm1(-argument)
            small_argument = argument < _SMALL_RELAXATION_ARGUMENT
            if small_argument:
                argument_2 = argument * argument
                argument_3 = argument_2 * argument
                argument_4 = argument_3 * argument
                argument_5 = argument_4 * argument
                argument_6 = argument_5 * argument
                phi_1 = (
                    1.0
                    - 0.5 * argument
                    + argument_2 / 6.0
                    - argument_3 / 24.0
                    + argument_4 / 120.0
                    - argument_5 / 720.0
                    + argument_6 / 5040.0
                )
                one_minus_phi_1 = (
                    0.5 * argument
                    - argument_2 / 6.0
                    + argument_3 / 24.0
                    - argument_4 / 120.0
                    + argument_5 / 720.0
                    - argument_6 / 5040.0
                )
                phi_2 = (
                    0.5
                    - argument / 6.0
                    + argument_2 / 24.0
                    - argument_3 / 120.0
                    + argument_4 / 720.0
                    - argument_5 / 5040.0
                    + argument_6 / 40320.0
                )
                velocity_memory = step * phi_1
                target_memory = step * one_minus_phi_1
                acceleration_memory = step * step * phi_2
            else:
                velocity_memory = attenuation / rate
                target_memory = step - velocity_memory
                acceleration_memory = target_memory / rate
        for axis in range(2):
            start_velocity = start_velocity_m_s[row, axis]
            target_velocity = target_velocity_m_s[row, axis]
            acceleration = additive_acceleration_m_s2[row, axis]
            position[row, axis] = (
                start_position_m[row, axis]
                + velocity_memory * start_velocity
                + target_memory * target_velocity
                + acceleration_memory * acceleration
            )
            if small_argument:
                velocity[row, axis] = (
                    start_velocity
                    + attenuation * (target_velocity - start_velocity)
                    + velocity_memory * acceleration
                )
            else:
                decay = math.exp(-(step * rate))
                velocity[row, axis] = (
                    decay * start_velocity
                    + attenuation * target_velocity
                    + velocity_memory * acceleration
                )


@njit(cache=True, fastmath=False, parallel=False, nogil=True, error_model="numpy")
def _compiled_exponential_update(
    start_position_m: FloatArray,
    start_velocity_m_s: FloatArray,
    linear_drag_rate_s_inv: FloatArray,
    target_velocity_m_s: FloatArray,
    additive_acceleration_m_s2: FloatArray,
    step_s: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Allocate one result and dispatch the sole exponential row kernel."""

    position = np.empty_like(start_position_m)
    velocity = np.empty_like(start_velocity_m_s)
    _compiled_exponential_update_into(
        start_position_m,
        start_velocity_m_s,
        linear_drag_rate_s_inv,
        target_velocity_m_s,
        additive_acceleration_m_s2,
        step_s,
        position,
        velocity,
    )
    return position, velocity


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _compiled_rk4_stage_state_into(
    start_position_m: FloatArray,
    start_velocity_m_s: FloatArray,
    start_charge_number: FloatArray,
    stage_velocity_m_s: FloatArray,
    stage_acceleration_m_s2: FloatArray,
    stage_charge_rate_number_s: FloatArray,
    elapsed_s: FloatArray,
    position: FloatArray,
    velocity: FloatArray,
    charge: FloatArray,
) -> None:
    """Advance one explicit RK stage into caller-owned buffers."""

    count = elapsed_s.size
    for row in range(count):
        step = elapsed_s[row]
        position[row, 0] = start_position_m[row, 0] + step * stage_velocity_m_s[row, 0]
        position[row, 1] = start_position_m[row, 1] + step * stage_velocity_m_s[row, 1]
        velocity[row, 0] = start_velocity_m_s[row, 0] + step * stage_acceleration_m_s2[row, 0]
        velocity[row, 1] = start_velocity_m_s[row, 1] + step * stage_acceleration_m_s2[row, 1]
        charge[row] = start_charge_number[row] + step * stage_charge_rate_number_s[row]


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _compiled_rk4_stage_state(
    start_position_m: FloatArray,
    start_velocity_m_s: FloatArray,
    start_charge_number: FloatArray,
    stage_velocity_m_s: FloatArray,
    stage_acceleration_m_s2: FloatArray,
    stage_charge_rate_number_s: FloatArray,
    elapsed_s: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Allocate one stage result and dispatch the sole RK row kernel."""

    position = np.empty_like(start_position_m)
    velocity = np.empty_like(start_velocity_m_s)
    charge = np.empty_like(start_charge_number)
    _compiled_rk4_stage_state_into(
        start_position_m,
        start_velocity_m_s,
        start_charge_number,
        stage_velocity_m_s,
        stage_acceleration_m_s2,
        stage_charge_rate_number_s,
        elapsed_s,
        position,
        velocity,
        charge,
    )
    return position, velocity, charge


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _compiled_rk4_endpoint_into(
    start_position_m: FloatArray,
    start_velocity_m_s: FloatArray,
    start_charge_number: FloatArray,
    velocity_2_m_s: FloatArray,
    velocity_3_m_s: FloatArray,
    velocity_4_m_s: FloatArray,
    acceleration_1_m_s2: FloatArray,
    acceleration_2_m_s2: FloatArray,
    acceleration_3_m_s2: FloatArray,
    acceleration_4_m_s2: FloatArray,
    charge_rate_1_number_s: FloatArray,
    charge_rate_2_number_s: FloatArray,
    charge_rate_3_number_s: FloatArray,
    charge_rate_4_number_s: FloatArray,
    step_s: FloatArray,
    position: FloatArray,
    velocity: FloatArray,
    charge: FloatArray,
) -> None:
    """Finish classical RK4 into caller-owned buffers in reference order."""

    count = step_s.size
    for row in range(count):
        weight = step_s[row] / 6.0
        for axis in range(2):
            velocity_sum = start_velocity_m_s[row, axis] + 2.0 * velocity_2_m_s[row, axis]
            velocity_sum = velocity_sum + 2.0 * velocity_3_m_s[row, axis]
            velocity_sum = velocity_sum + velocity_4_m_s[row, axis]
            acceleration_sum = acceleration_1_m_s2[row, axis] + 2.0 * acceleration_2_m_s2[row, axis]
            acceleration_sum = acceleration_sum + 2.0 * acceleration_3_m_s2[row, axis]
            acceleration_sum = acceleration_sum + acceleration_4_m_s2[row, axis]
            position[row, axis] = start_position_m[row, axis] + weight * velocity_sum
            velocity[row, axis] = start_velocity_m_s[row, axis] + weight * acceleration_sum
        charge_rate_sum = charge_rate_1_number_s[row] + 2.0 * charge_rate_2_number_s[row]
        charge_rate_sum = charge_rate_sum + 2.0 * charge_rate_3_number_s[row]
        charge_rate_sum = charge_rate_sum + charge_rate_4_number_s[row]
        charge[row] = start_charge_number[row] + weight * charge_rate_sum


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _compiled_rk4_endpoint(
    start_position_m: FloatArray,
    start_velocity_m_s: FloatArray,
    start_charge_number: FloatArray,
    velocity_2_m_s: FloatArray,
    velocity_3_m_s: FloatArray,
    velocity_4_m_s: FloatArray,
    acceleration_1_m_s2: FloatArray,
    acceleration_2_m_s2: FloatArray,
    acceleration_3_m_s2: FloatArray,
    acceleration_4_m_s2: FloatArray,
    charge_rate_1_number_s: FloatArray,
    charge_rate_2_number_s: FloatArray,
    charge_rate_3_number_s: FloatArray,
    charge_rate_4_number_s: FloatArray,
    step_s: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Allocate one endpoint result and dispatch the sole RK row kernel."""

    position = np.empty_like(start_position_m)
    velocity = np.empty_like(start_velocity_m_s)
    charge = np.empty_like(start_charge_number)
    _compiled_rk4_endpoint_into(
        start_position_m,
        start_velocity_m_s,
        start_charge_number,
        velocity_2_m_s,
        velocity_3_m_s,
        velocity_4_m_s,
        acceleration_1_m_s2,
        acceleration_2_m_s2,
        acceleration_3_m_s2,
        acceleration_4_m_s2,
        charge_rate_1_number_s,
        charge_rate_2_number_s,
        charge_rate_3_number_s,
        charge_rate_4_number_s,
        step_s,
        position,
        velocity,
        charge,
    )
    return position, velocity, charge


def _validate_constant_acceleration(
    count: int,
    requires_stage_evaluation: bool,
    value: FloatArray | None,
) -> FloatArray | None:
    if value is None:
        return None
    acceleration = np.asarray(value, dtype=np.float64)
    if not requires_stage_evaluation:
        raise ValueError("constant acceleration requires stage-evaluated motion")
    if acceleration.shape != (count, 2) or not bool(np.isfinite(acceleration).all()):
        raise ValueError("constant acceleration must be a finite array with shape [N, 2]")
    return acceleration.copy()


def _validate_stage(stage: DynamicsEvaluation, count: int) -> None:
    if (
        stage.acceleration_m_s2.shape != (count, 2)
        or stage.charge_rate_number_s.shape != (count,)
        or stage.support_inside.shape != (count,)
        or stage.applicability_inside.shape != (count,)
        or stage.numerical_status.shape != (count,)
        or (stage.field_cell_id is not None and stage.field_cell_id.shape != (count,))
    ):
        raise ValueError("stage evaluator returned inconsistent array shapes")
    if stage.numerical_status.dtype != np.uint8:
        raise ValueError("stage numerical status must have dtype uint8")
    if not bool(
        np.isfinite(stage.acceleration_m_s2).all() and np.isfinite(stage.charge_rate_number_s).all()
    ):
        raise ValueError("stage evaluator returned a non-finite derivative")


def _validate_relaxation_stage(stage: RelaxationEvaluation, count: int) -> None:
    shapes = (
        (stage.linear_drag_rate_s_inv.shape, (count,)),
        (stage.target_velocity_m_s.shape, (count, 2)),
        (stage.additive_acceleration_m_s2.shape, (count, 2)),
        (stage.charge_rate_number_s.shape, (count,)),
        (stage.charge_rate_derivative_s_inv.shape, (count,)),
        (stage.support_inside.shape, (count,)),
        (stage.applicability_inside.shape, (count,)),
        (stage.numerical_status.shape, (count,)),
    )
    invalid_field_cell = stage.field_cell_id is not None and stage.field_cell_id.shape != (count,)
    if any(actual != expected for actual, expected in shapes) or invalid_field_cell:
        raise ValueError("relaxation evaluator returned inconsistent array shapes")
    if stage.numerical_status.dtype != np.uint8:
        raise ValueError("relaxation numerical status must have dtype uint8")
    if not bool(
        np.isfinite(stage.linear_drag_rate_s_inv).all()
        and np.isfinite(stage.charge_rate_number_s).all()
        and np.isfinite(stage.charge_rate_derivative_s_inv).all()
        and np.isfinite(stage.target_velocity_m_s).all()
        and np.isfinite(stage.additive_acceleration_m_s2).all()
        and (stage.linear_drag_rate_s_inv >= 0.0).all()
        and (stage.charge_rate_derivative_s_inv <= 0.0).all()
    ):
        raise ValueError("relaxation evaluator returned an invalid coefficient")


def _merge_numerical_status(current: UInt8Array, rows: Int64Array, update: UInt8Array) -> None:
    if update.dtype != np.uint8:
        raise ValueError("stage numerical status must have dtype uint8")
    first_failure = (current[rows] == NUMERICAL_STATUS_OK) & (update != NUMERICAL_STATUS_OK)
    current[rows[first_failure]] = update[first_failure]


def _mark_nonfinite_state(
    status: UInt8Array,
    rows: Int64Array,
    position_m: FloatArray,
    velocity_m_s: FloatArray,
    charge_number: FloatArray,
) -> BoolArray:
    finite = (
        np.isfinite(position_m).all(axis=1)
        & np.isfinite(velocity_m_s).all(axis=1)
        & np.isfinite(charge_number)
    )
    failed = ~finite
    status[rows[failed]] = INTEGRATOR_NUMERICAL_FAILURE
    return finite


def _validate_inputs(
    count: int,
    start_time_s: FloatArray,
    target_time_s: FloatArray,
    position_m: FloatArray,
    velocity_m_s: FloatArray,
    charge_number: FloatArray,
) -> None:
    if (
        start_time_s.shape != (count,)
        or target_time_s.shape != (count,)
        or charge_number.shape != (count,)
    ):
        raise ValueError("proposal scalar columns do not match particle_index")
    if position_m.shape != (count, 2) or velocity_m_s.shape != (count, 2):
        raise ValueError("proposal position and velocity must have shape [N, 2]")
    if not bool(
        np.isfinite(start_time_s).all()
        and np.isfinite(target_time_s).all()
        and np.isfinite(position_m).all()
        and np.isfinite(velocity_m_s).all()
        and np.isfinite(charge_number).all()
    ):
        raise ValueError("proposal must contain only finite values")
    if bool((start_time_s > target_time_s).any()):
        raise ValueError("proposal time interval is inconsistent")


def _require_finite_state(
    position_m: FloatArray, velocity_m_s: FloatArray, charge_number: FloatArray
) -> None:
    if not bool(
        np.isfinite(position_m).all()
        and np.isfinite(velocity_m_s).all()
        and np.isfinite(charge_number).all()
    ):
        raise ValueError("proposal state is not finite at the requested time")
