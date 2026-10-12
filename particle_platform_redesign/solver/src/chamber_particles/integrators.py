"""Batched fixed-step proposals independent of fields, geometry, and output."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, NamedTuple, Protocol

import numpy as np
from numba import njit
from numpy.typing import NDArray

from .cpu import next_float64
from .numerical_status import (
    INTEGRATOR_NUMERICAL_FAILURE,
    NUMERICAL_STATUS_OK,
)

type FloatArray = NDArray[np.float64]
type Int64Array = NDArray[np.int64]
type BoolArray = NDArray[np.bool_]
type UInt8Array = NDArray[np.uint8]
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
# Ten numeric arrays in Rk4AdvanceWorkspace; persistent slot scratch is separate
# from proposal-owned endpoint and dense controls.
RK4_ADVANCE_WORK_BYTES_PER_PARTICLE = 209
# Ten frozen arrays retained by each full-interval dense enclosure.
RK4_DENSE_FULL_ENCLOSURE_BYTES_PER_PARTICLE = 225
STAGE_ADVANCE_WORK_BYTES_PER_PARTICLE = 65
RELAXATION_ADVANCE_WORK_BYTES_PER_PARTICLE = 169
# Sample buffers are borrowed until their next use; retained proposals/events
# keep their own copies. Prefix input and stage scratch must be distinct.
PROPOSAL_SAMPLE_WORK_BYTES_PER_PARTICLE = 59
PROPOSAL_PREFIX_WORK_BYTES_PER_PARTICLE = 80
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


class _CubicHermitePath(NamedTuple):
    """Original endpoint data defining one immutable Hermite polynomial per row."""

    start_time_s: FloatArray
    target_time_s: FloatArray
    start_position_m: FloatArray
    start_velocity_m_s: FloatArray
    end_position_m: FloatArray
    end_velocity_m_s: FloatArray
    start_charge_number: FloatArray
    end_charge_number: FloatArray
    charge_root_time_s: FloatArray
    charge_root_number: FloatArray
    charge_affine_rate_number_s: FloatArray
    charge_rate_derivative_s_inv: FloatArray


class _Rk4DensePath(NamedTuple):
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


# Empty numeric projections keep absent paths out of optional Numba types.
# They own no particle payload and are never sampled by the path dispatcher.
_EMPTY_PATH_VECTOR = np.empty(0, dtype=np.float64)
_EMPTY_PATH_STATE = np.empty((0, 2), dtype=np.float64)
_EMPTY_CUBIC_HERMITE_PATH = _CubicHermitePath(
    _EMPTY_PATH_VECTOR,
    _EMPTY_PATH_VECTOR,
    _EMPTY_PATH_STATE,
    _EMPTY_PATH_STATE,
    _EMPTY_PATH_STATE,
    _EMPTY_PATH_STATE,
    _EMPTY_PATH_VECTOR,
    _EMPTY_PATH_VECTOR,
    _EMPTY_PATH_VECTOR,
    _EMPTY_PATH_VECTOR,
    _EMPTY_PATH_VECTOR,
    _EMPTY_PATH_VECTOR,
)
_EMPTY_RK4_DENSE_PATH = _Rk4DensePath(
    _EMPTY_PATH_VECTOR,
    _EMPTY_PATH_VECTOR,
    np.empty((0, 4, 2), dtype=np.float64),
    np.empty((0, 4, 2), dtype=np.float64),
    np.empty((0, 4), dtype=np.float64),
)


@dataclass(frozen=True, slots=True)
class _Rk4DenseControls:
    """Mutable construction result kept private to one RK4 advance."""

    position_inner_m: FloatArray
    velocity_inner_m_s: FloatArray
    charge_inner_number: FloatArray


class Rk4AdvanceWorkspace(NamedTuple):
    """Exclusive bounded scratch for one native RK4 proposal batch."""

    active_rows: Int64Array
    particle_index: Int64Array
    time_s: FloatArray
    position_m: FloatArray
    velocity_m_s: FloatArray
    charge_number: FloatArray
    incoming_status: UInt8Array
    stage_velocity_m_s: FloatArray
    acceleration_m_s2: FloatArray
    charge_rate_number_s: FloatArray


class _Rk4AdvanceOutputs(NamedTuple):
    """Proposal-owned arrays retained after the scratch slot is reused."""

    position_m: FloatArray
    velocity_m_s: FloatArray
    charge_number: FloatArray
    support_inside: BoolArray
    applicability_inside: BoolArray
    numerical_status: UInt8Array
    field_cell_id: Int64Array
    position_inner_m: FloatArray
    velocity_inner_m_s: FloatArray
    charge_inner_number: FloatArray


class StageAdvanceWorkspace(NamedTuple):
    """Compact stage inputs retained by one exclusive advance slot."""

    active_rows: Int64Array
    particle_index: Int64Array
    time_s: FloatArray
    position_m: FloatArray
    velocity_m_s: FloatArray
    charge_number: FloatArray
    incoming_status: UInt8Array


class RelaxationAdvanceWorkspace(NamedTuple):
    """Bounded predictor state and frozen coefficients for exponential advance."""

    stage: StageAdvanceWorkspace
    midpoint_position_m: FloatArray
    midpoint_velocity_m_s: FloatArray
    midpoint_charge_number: FloatArray
    elapsed_s: FloatArray
    rate_s_inv: FloatArray
    target_velocity_m_s: FloatArray
    additive_acceleration_m_s2: FloatArray
    charge_rate_number_s: FloatArray
    charge_rate_derivative_s_inv: FloatArray


class _StageAdvanceOutputs(NamedTuple):
    """Proposal-owned state and independent stage verdicts."""

    position_m: FloatArray
    velocity_m_s: FloatArray
    charge_number: FloatArray
    support_inside: BoolArray
    applicability_inside: BoolArray
    numerical_status: UInt8Array
    field_cell_id: Int64Array


class ProposalSampleBuffer(NamedTuple):
    """Exclusive reusable samples, never retained by an accepted proposal."""

    particle_index: Int64Array
    state: _StageAdvanceOutputs


class ProposalPrefixWorkspace(NamedTuple):
    """Packed root inputs kept separate from the advance's mutable stages."""

    particle_index: Int64Array
    start_time_s: FloatArray
    elapsed_s: FloatArray
    position_m: FloatArray
    velocity_m_s: FloatArray
    charge_number: FloatArray
    acceleration_m_s2: FloatArray


class StepProposalProjection(NamedTuple):
    """Borrowed numeric proposal facts, excluding Python stage evaluators."""

    path_kind: PathKind
    particle_index: Int64Array
    start_time_s: FloatArray
    target_time_s: FloatArray
    start_position_m: FloatArray
    start_velocity_m_s: FloatArray
    start_charge_number: FloatArray
    support_inside: BoolArray
    applicability_inside: BoolArray
    numerical_status: UInt8Array
    constant_acceleration_m_s2: FloatArray
    cubic_hermite_path: _CubicHermitePath
    rk4_dense_path: _Rk4DensePath


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
    _rk4_dense_full_enclosure: tuple[_Rk4DensePath, Rk4DenseEnclosure] | None = None

    def numeric_projection(self) -> StepProposalProjection:
        """Borrow original roots for a native chain; no array payload is copied."""

        return StepProposalProjection(
            self.path_kind,
            self.particle_index,
            self.start_time_s,
            self.target_time_s,
            self.start_position_m,
            self.start_velocity_m_s,
            self.start_charge_number,
            self.support_inside,
            self.applicability_inside,
            self.numerical_status,
            (
                _EMPTY_PATH_STATE
                if self.constant_acceleration_m_s2 is None
                else self.constant_acceleration_m_s2
            ),
            _EMPTY_CUBIC_HERMITE_PATH
            if self._cubic_hermite_path is None
            else self._cubic_hermite_path,
            _EMPTY_RK4_DENSE_PATH if self._rk4_dense_path is None else self._rk4_dense_path,
        )

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
        sample_time_s = np.full(selected.size, time_s, dtype=np.float64)
        buffer = allocate_proposal_sample_buffer(selected.size)
        return self.state_at_times_into(selected, sample_time_s, buffer)

    def state_at_times_into(
        self,
        local_rows: Int64Array,
        time_s: FloatArray,
        buffer: ProposalSampleBuffer,
        prefix_workspace: ProposalPrefixWorkspace | None = None,
    ) -> ProposalSample:
        """Evaluate per-row times into exclusive scratch, valid until reuse.

        Stage evaluators retain their existing prepared native bindings. Dense
        paths borrow their original immutable roots, including after restriction.
        """

        selected = np.asarray(local_rows, dtype=np.int64)
        times = np.asarray(time_s, dtype=np.float64)
        _validate_dense_sample(selected, times, self.start_time_s, self.target_time_s)
        count = selected.size
        _validate_proposal_sample_buffer(buffer, count)
        outputs = _proposal_sample_outputs_view(buffer.state, count)
        projection = self.numeric_projection()
        if self.path_kind in ("cubic_hermite", "rk4_dense"):
            step_proposal_dense_state_into(projection, selected, times, outputs)
        else:
            if prefix_workspace is None:
                prefix_workspace = allocate_proposal_prefix_workspace(count)
            if prefix_workspace.particle_index.size < count:
                raise ValueError("proposal prefix exceeds its workspace capacity")
            step_proposal_prefix_into(projection, selected, times, prefix_workspace)
            _evaluate_proposal_prefix_into(self, prefix_workspace, count, outputs)
        _proposal_sample_indices_into(self.particle_index, selected, buffer.particle_index)
        return ProposalSample(
            particle_index=buffer.particle_index[:count],
            position_m=outputs.position_m,
            velocity_m_s=outputs.velocity_m_s,
            charge_number=outputs.charge_number,
            support_inside=outputs.support_inside,
            applicability_inside=outputs.applicability_inside,
            numerical_status=outputs.numerical_status,
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
        inherited_status = self.numerical_status[selected]
        full = self._rk4_dense_full_enclosure
        if (
            full is not None
            and full[0] is path
            and all(not value.flags.writeable for value in path)
            and np.array_equal(starts.view(np.uint64), path.start_time_s[selected].view(np.uint64))
            and np.array_equal(
                targets.view(np.uint64), path.target_time_s[selected].view(np.uint64)
            )
            and np.array_equal(inherited_status, full[1].numerical_status[selected])
        ):
            # Advanced row selection copies every column, including repeated rows.
            # Callers may modify these results without changing the frozen fact.
            enclosure = full[1]
            return Rk4DenseEnclosure(
                position_lower_m=enclosure.position_lower_m[selected],
                position_upper_m=enclosure.position_upper_m[selected],
                velocity_lower_m_s=enclosure.velocity_lower_m_s[selected],
                velocity_upper_m_s=enclosure.velocity_upper_m_s[selected],
                charge_lower_number=enclosure.charge_lower_number[selected],
                charge_upper_number=enclosure.charge_upper_number[selected],
                position_control_origin_m=enclosure.position_control_origin_m[selected],
                relative_position_control_lower_m=(
                    enclosure.relative_position_control_lower_m[selected]
                ),
                relative_position_control_upper_m=(
                    enclosure.relative_position_control_upper_m[selected]
                ),
                numerical_status=enclosure.numerical_status[selected],
            )
        return _enclose_rk4_dense_subinterval(
            path,
            selected,
            starts,
            targets,
            inherited_status,
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
        buffer = allocate_proposal_sample_buffer(selected.size)
        return self.state_at_times_into(selected, sample_time_s, buffer)


def allocate_proposal_sample_buffer(capacity: int) -> ProposalSampleBuffer:
    return ProposalSampleBuffer(
        np.empty(capacity, dtype=np.int64), _allocate_stage_advance_outputs(capacity)
    )


def allocate_proposal_prefix_workspace(capacity: int) -> ProposalPrefixWorkspace:
    return ProposalPrefixWorkspace(
        np.empty(capacity, dtype=np.int64),
        np.empty(capacity, dtype=np.float64),
        np.empty(capacity, dtype=np.float64),
        np.empty((capacity, 2), dtype=np.float64),
        np.empty((capacity, 2), dtype=np.float64),
        np.empty(capacity, dtype=np.float64),
        np.empty((capacity, 2), dtype=np.float64),
    )


def _validate_proposal_sample_buffer(buffer: ProposalSampleBuffer, count: int) -> None:
    if buffer.particle_index.size < count:
        raise ValueError("proposal sample exceeds its buffer capacity")
    arrays = (*buffer.state,)
    if any(array.shape[0] < count for array in arrays):
        raise ValueError("proposal sample state exceeds its buffer capacity")


@njit(cache=True, fastmath=False, parallel=False, nogil=True, error_model="numpy")
def _proposal_sample_outputs_view(state: _StageAdvanceOutputs, count: int) -> _StageAdvanceOutputs:
    return _StageAdvanceOutputs(
        state.position_m[:count],
        state.velocity_m_s[:count],
        state.charge_number[:count],
        state.support_inside[:count],
        state.applicability_inside[:count],
        state.numerical_status[:count],
        state.field_cell_id[:count],
    )


@njit(cache=True, fastmath=False, parallel=False, nogil=True, error_model="numpy")
def _proposal_sample_indices_into(
    particle_index: Int64Array, rows: Int64Array, result: Int64Array
) -> None:
    for row in range(rows.size):
        result[row] = particle_index[rows[row]]


@njit(cache=True, fastmath=False, parallel=False, nogil=True, error_model="numpy")
def step_proposal_prefix_into(
    proposal: StepProposalProjection,
    rows: Int64Array,
    time_s: FloatArray,
    workspace: ProposalPrefixWorkspace,
) -> None:
    """Pack selected original starts without aliasing mutable stage scratch."""

    acceleration = proposal.constant_acceleration_m_s2
    for row in range(rows.size):
        source = rows[row]
        workspace.particle_index[row] = proposal.particle_index[source]
        workspace.start_time_s[row] = proposal.start_time_s[source]
        workspace.elapsed_s[row] = time_s[row] - proposal.start_time_s[source]
        workspace.charge_number[row] = proposal.start_charge_number[source]
        for axis in range(2):
            workspace.position_m[row, axis] = proposal.start_position_m[source, axis]
            workspace.velocity_m_s[row, axis] = proposal.start_velocity_m_s[source, axis]
            workspace.acceleration_m_s2[row, axis] = (
                0.0 if acceleration.size == 0 else acceleration[source, axis]
            )


def _evaluate_proposal_prefix_into(
    proposal: StepProposal,
    workspace: ProposalPrefixWorkspace,
    count: int,
    outputs: _StageAdvanceOutputs,
) -> None:
    indices = workspace.particle_index[:count]
    starts = workspace.start_time_s[:count]
    elapsed = workspace.elapsed_s[:count]
    position = workspace.position_m[:count]
    velocity = workspace.velocity_m_s[:count]
    charge = workspace.charge_number[:count]
    if proposal.path_kind == "linear_exact":
        _compiled_linear_advance_into(position, velocity, charge, elapsed, outputs)
    elif proposal.path_kind == "quadratic_exact":
        if proposal._evaluator is None or proposal.constant_acceleration_m_s2 is None:
            raise ValueError("quadratic proposal lost its evaluator or acceleration")
        _advance_quadratic(
            proposal._evaluator,
            indices,
            starts,
            elapsed,
            position,
            velocity,
            charge,
            workspace.acceleration_m_s2[:count],
            outputs=outputs,
        )
    elif proposal.path_kind == "exponential_midpoint_reintegrated":
        if proposal._relaxation_evaluator is None:
            raise ValueError("exponential midpoint proposal has no relaxation evaluator")
        _run_relaxation_advance(
            proposal._relaxation_evaluator,
            indices,
            starts,
            elapsed,
            position,
            velocity,
            charge,
            predictor_only=False,
            outputs=outputs,
        )
    else:
        raise ValueError("proposal prefix requires an exact or relaxation path")


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
        charge_root_time_s=(_EMPTY_PATH_VECTOR if charge_path is None else charge_path[0]),
        charge_root_number=(_EMPTY_PATH_VECTOR if charge_path is None else charge_path[1]),
        charge_affine_rate_number_s=(_EMPTY_PATH_VECTOR if charge_path is None else charge_path[2]),
        charge_rate_derivative_s_inv=(
            _EMPTY_PATH_VECTOR if charge_path is None else charge_path[3]
        ),
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

    start_state = _allocate_stage_advance_outputs(count)
    end_state = _allocate_stage_advance_outputs(count)
    enclosure = CurvedPathEnclosure(
        np.empty((count, 2), dtype=np.float64),
        np.empty((count, 2), dtype=np.float64),
        np.empty((count, 2), dtype=np.float64),
        np.empty((count, 2), dtype=np.float64),
        np.empty(count, dtype=np.uint8),
    )
    cubic_hermite_restriction_into(
        proposal.numeric_projection(),
        selected,
        starts,
        targets,
        start_state,
        end_state,
        enclosure.position_lower_m,
        enclosure.position_upper_m,
        enclosure.velocity_lower_m_s,
        enclosure.velocity_upper_m_s,
        enclosure.numerical_status,
    )
    _require_finite_enclosure(enclosure)
    numerical_status = end_state.numerical_status
    _require_enclosure_contains_endpoint(
        enclosure, end_state.position_m, end_state.velocity_m_s, numerical_status
    )
    original = _subset_cubic_hermite_path(path, selected)
    return StepProposal(
        particle_index=proposal.particle_index[selected].copy(),
        start_time_s=starts.copy(),
        target_time_s=targets.copy(),
        start_position_m=start_state.position_m,
        start_velocity_m_s=start_state.velocity_m_s,
        start_charge_number=start_state.charge_number,
        end_position_m=end_state.position_m,
        end_velocity_m_s=end_state.velocity_m_s,
        end_charge_number=end_state.charge_number,
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
            path.charge_root_time_s
            if path.charge_root_time_s.size == 0
            else path.charge_root_time_s[selected].copy()
        ),
        charge_root_number=(
            path.charge_root_number
            if path.charge_root_number.size == 0
            else path.charge_root_number[selected].copy()
        ),
        charge_affine_rate_number_s=(
            path.charge_affine_rate_number_s
            if path.charge_affine_rate_number_s.size == 0
            else path.charge_affine_rate_number_s[selected].copy()
        ),
        charge_rate_derivative_s_inv=(
            path.charge_rate_derivative_s_inv
            if path.charge_rate_derivative_s_inv.size == 0
            else path.charge_rate_derivative_s_inv[selected].copy()
        ),
    )


@njit(cache=True, fastmath=False, parallel=False, nogil=True, error_model="numpy")
def _cubic_hermite_root_controls_scalar(
    start_position: float,
    end_position: float,
    start_velocity: float,
    end_velocity: float,
    duration: float,
) -> tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float],
]:
    displacement, displacement_residual = _dense_two_diff_scalar(end_position, start_position)
    start_tangent = duration * start_velocity / 3.0
    end_tangent = duration * end_velocity / 3.0
    second, second_residual = _dense_two_diff_scalar(displacement, end_tangent)
    residual = second_residual + displacement_residual
    difference, difference_residual = _dense_two_diff_scalar(second, start_tangent)
    middle_velocity = 3.0 * (difference + difference_residual + residual) / duration
    return (
        (0.0, start_tangent, second, displacement),
        (0.0, 0.0, residual, displacement_residual),
        (start_velocity, middle_velocity, end_velocity),
    )


@njit(cache=True, fastmath=False, parallel=False, nogil=True, error_model="numpy")
def _cubic_hermite_motion_into(
    path: _CubicHermitePath,
    rows: Int64Array,
    time_s: FloatArray,
    position: FloatArray,
    velocity: FloatArray,
) -> None:
    for row in range(rows.size):
        source = rows[row]
        duration = path.target_time_s[source] - path.start_time_s[source]
        parameter = (time_s[row] - path.start_time_s[source]) / duration
        for axis in range(2):
            relative, residual, speeds = _cubic_hermite_root_controls_scalar(
                path.start_position_m[source, axis],
                path.end_position_m[source, axis],
                path.start_velocity_m_s[source, axis],
                path.end_velocity_m_s[source, axis],
                duration,
            )
            position[row, axis] = path.start_position_m[source, axis] + (
                _dense_cubic_sample_scalar(relative, parameter)
                + _dense_cubic_sample_scalar(residual, parameter)
            )
            first = _dense_lerp_scalar(speeds[0], speeds[1], parameter)
            second = _dense_lerp_scalar(speeds[1], speeds[2], parameter)
            velocity[row, axis] = _dense_lerp_scalar(first, second, parameter)
            if time_s[row] == path.start_time_s[source]:
                position[row, axis] = path.start_position_m[source, axis]
                velocity[row, axis] = path.start_velocity_m_s[source, axis]
            if time_s[row] == path.target_time_s[source]:
                position[row, axis] = path.end_position_m[source, axis]
                velocity[row, axis] = path.end_velocity_m_s[source, axis]


@njit(cache=True, fastmath=False, parallel=False, nogil=True, error_model="numpy")
def _cubic_hermite_charge_into(
    path: _CubicHermitePath, rows: Int64Array, time_s: FloatArray, charge: FloatArray
) -> None:
    root_time = path.charge_root_time_s
    root_charge = path.charge_root_number
    affine_rate = path.charge_affine_rate_number_s
    derivative = path.charge_rate_derivative_s_inv
    for row in range(rows.size):
        source = rows[row]
        start_charge = path.start_charge_number[source]
        if root_time.size == 0:
            duration = path.target_time_s[source] - path.start_time_s[source]
            parameter = (time_s[row] - path.start_time_s[source]) / duration
            charge[row] = start_charge + parameter * (path.end_charge_number[source] - start_charge)
        else:
            if root_charge.size == 0 or affine_rate.size == 0 or derivative.size == 0:
                raise ValueError("cubic Hermite exponential charge metadata is incomplete")
            elapsed = time_s[row] - root_time[source]
            multiplier = _affine_exponential_multiplier_scalar(derivative[source], elapsed)
            charge[row] = root_charge[source] + multiplier * affine_rate[source]
        if time_s[row] == path.start_time_s[source]:
            charge[row] = start_charge
        if time_s[row] == path.target_time_s[source]:
            charge[row] = path.end_charge_number[source]


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


@njit(cache=True, fastmath=False, parallel=False, nogil=True, error_model="numpy")
def rk4_dense_state_at_into(
    path: _Rk4DensePath,
    rows: Int64Array,
    time_s: FloatArray,
    position: FloatArray,
    velocity: FloatArray,
    charge: FloatArray,
) -> None:
    """Sample immutable controls without selected-control or Bezier arrays."""

    for row in range(rows.size):
        source = rows[row]
        duration = path.target_time_s[source] - path.start_time_s[source]
        parameter = 0.0
        if duration > 0.0:
            parameter = (time_s[row] - path.start_time_s[source]) / duration
        parameter = np.minimum(np.maximum(parameter, 0.0), 1.0)
        for axis in range(2):
            origin = path.position_controls_m[source, 0, axis]
            first, first_residual = _dense_two_diff_scalar(origin, origin)
            second, second_residual = _dense_two_diff_scalar(
                path.position_controls_m[source, 1, axis], origin
            )
            third, third_residual = _dense_two_diff_scalar(
                path.position_controls_m[source, 2, axis], origin
            )
            fourth, fourth_residual = _dense_two_diff_scalar(
                path.position_controls_m[source, 3, axis], origin
            )
            position[row, axis] = origin + (
                _dense_cubic_sample_scalar((first, second, third, fourth), parameter)
                + _dense_cubic_sample_scalar(
                    (first_residual, second_residual, third_residual, fourth_residual), parameter
                )
            )
            velocity[row, axis] = _dense_cubic_sample_scalar(
                (
                    path.velocity_controls_m_s[source, 0, axis],
                    path.velocity_controls_m_s[source, 1, axis],
                    path.velocity_controls_m_s[source, 2, axis],
                    path.velocity_controls_m_s[source, 3, axis],
                ),
                parameter,
            )
            if time_s[row] == path.start_time_s[source]:
                position[row, axis] = path.position_controls_m[source, 0, axis]
                velocity[row, axis] = path.velocity_controls_m_s[source, 0, axis]
            if time_s[row] == path.target_time_s[source]:
                position[row, axis] = path.position_controls_m[source, 3, axis]
                velocity[row, axis] = path.velocity_controls_m_s[source, 3, axis]
        charge[row] = _dense_cubic_sample_scalar(
            (
                path.charge_controls_number[source, 0],
                path.charge_controls_number[source, 1],
                path.charge_controls_number[source, 2],
                path.charge_controls_number[source, 3],
            ),
            parameter,
        )
        if time_s[row] == path.start_time_s[source]:
            charge[row] = path.charge_controls_number[source, 0]
        if time_s[row] == path.target_time_s[source]:
            charge[row] = path.charge_controls_number[source, 3]


@njit(cache=True, fastmath=False, parallel=False, nogil=True, error_model="numpy")
def step_proposal_dense_state_into(
    proposal: StepProposalProjection,
    rows: Int64Array,
    time_s: FloatArray,
    outputs: _StageAdvanceOutputs,
) -> None:
    """Native dense path sampling and the proposal's independent verdicts."""

    if proposal.path_kind == "cubic_hermite":
        path = proposal.cubic_hermite_path
        if path.start_time_s.size == 0 and rows.size != 0:
            raise ValueError("cubic Hermite proposal has no original path")
        _cubic_hermite_motion_into(path, rows, time_s, outputs.position_m, outputs.velocity_m_s)
        _cubic_hermite_charge_into(path, rows, time_s, outputs.charge_number)
    else:
        dense = proposal.rk4_dense_path
        if proposal.path_kind != "rk4_dense":
            raise ValueError("dense sampling requires an RK4 or cubic Hermite proposal")
        rk4_dense_state_at_into(
            dense, rows, time_s, outputs.position_m, outputs.velocity_m_s, outputs.charge_number
        )
    for row in range(rows.size):
        source = rows[row]
        outputs.support_inside[row] = proposal.support_inside[source]
        outputs.applicability_inside[row] = proposal.applicability_inside[source]
        outputs.numerical_status[row] = proposal.numerical_status[source]
        outputs.field_cell_id[row] = -1


def _lerp_controls(left: FloatArray, right: FloatArray, parameter: FloatArray) -> FloatArray:
    weight_shape = (parameter.size,) + (1,) * (left.ndim - 1)
    weight = parameter.reshape(weight_shape)
    return (1.0 - weight) * left + weight * right


_DENSE_BOUND_EPS = 128.0 * np.finfo(np.float64).eps
_DENSE_BOUND_UNDERFLOW = 128.0 * np.finfo(np.float64).smallest_subnormal


@njit(cache=True, fastmath=False, parallel=False, error_model="numpy")
def _dense_lerp_scalar(left: float, right: float, parameter: float) -> float:
    # Keep the two products and their final addition separate: FMA or an
    # algebraic rewrite would change the stored-control restriction arithmetic.
    first = (1.0 - parameter) * left
    second = parameter * right
    return first + second


@njit(cache=True, fastmath=False, parallel=False, error_model="numpy")
def _dense_cubic_sample_scalar(
    controls: tuple[float, float, float, float], parameter: float
) -> float:
    first_01 = _dense_lerp_scalar(controls[0], controls[1], parameter)
    first_12 = _dense_lerp_scalar(controls[1], controls[2], parameter)
    first_23 = _dense_lerp_scalar(controls[2], controls[3], parameter)
    second_012 = _dense_lerp_scalar(first_01, first_12, parameter)
    second_123 = _dense_lerp_scalar(first_12, first_23, parameter)
    return _dense_lerp_scalar(second_012, second_123, parameter)


@njit(cache=True, fastmath=False, parallel=False, error_model="numpy")
def _dense_relative_target(start: float, target: float) -> float:
    denominator = 1.0 - start
    relative = 0.0
    if denominator > 0.0:
        relative = (target - start) / denominator
    return np.minimum(np.maximum(relative, 0.0), 1.0)


@njit(cache=True, fastmath=False, parallel=False, error_model="numpy")
def _dense_restrict_cubic_scalar(
    first: float,
    second: float,
    third: float,
    fourth: float,
    start: float,
    target: float,
) -> tuple[float, float, float, float]:
    first_01 = _dense_lerp_scalar(first, second, start)
    first_12 = _dense_lerp_scalar(second, third, start)
    first_23 = _dense_lerp_scalar(third, fourth, start)
    second_012 = _dense_lerp_scalar(first_01, first_12, start)
    second_123 = _dense_lerp_scalar(first_12, first_23, start)
    start_value = _dense_lerp_scalar(second_012, second_123, start)
    relative = _dense_relative_target(start, target)
    third_01 = _dense_lerp_scalar(start_value, second_123, relative)
    third_12 = _dense_lerp_scalar(second_123, first_23, relative)
    third_23 = _dense_lerp_scalar(first_23, fourth, relative)
    fourth_012 = _dense_lerp_scalar(third_01, third_12, relative)
    fourth_123 = _dense_lerp_scalar(third_12, third_23, relative)
    target_value = _dense_lerp_scalar(fourth_012, fourth_123, relative)
    if start == 1.0:
        return fourth, fourth, fourth, fourth
    return start_value, third_01, fourth_012, target_value


@njit(cache=True, fastmath=False, parallel=False, error_model="numpy")
def _dense_restrict_quadratic_scalar(
    first: float,
    second: float,
    third: float,
    start: float,
    target: float,
) -> tuple[float, float, float]:
    first_01 = _dense_lerp_scalar(first, second, start)
    first_12 = _dense_lerp_scalar(second, third, start)
    start_value = _dense_lerp_scalar(first_01, first_12, start)
    relative = _dense_relative_target(start, target)
    second_01 = _dense_lerp_scalar(start_value, first_12, relative)
    second_12 = _dense_lerp_scalar(first_12, third, relative)
    target_value = _dense_lerp_scalar(second_01, second_12, relative)
    if start == 1.0:
        return third, third, third
    return start_value, second_01, target_value


@njit(cache=True, fastmath=False, parallel=False, error_model="numpy")
def _dense_two_diff_scalar(left: float, right: float) -> tuple[float, float]:
    difference = left - right
    virtual_right = left - difference
    virtual_left = difference + virtual_right
    right_roundoff = virtual_right - right
    left_roundoff = left - virtual_left
    residual = left_roundoff + right_roundoff
    if left == right:
        return 0.0, 0.0
    return difference, residual


@njit(cache=True, fastmath=False, parallel=False, error_model="numpy")
def _dense_absolute_maximum(controls: tuple[float, ...]) -> float:
    maximum = abs(controls[0])
    for index in range(1, len(controls)):
        maximum = np.maximum(maximum, abs(controls[index]))
    return maximum


@njit(cache=True, fastmath=False, parallel=False, error_model="numpy")
def _dense_outward_padding(scale: float, absolute_padding: float, include_absolute: bool) -> float:
    padding = next_float64(scale * _DENSE_BOUND_EPS + _DENSE_BOUND_UNDERFLOW, True)
    if include_absolute:
        padding = next_float64(padding + absolute_padding, True)
    return padding


@njit(cache=True, fastmath=False, parallel=False, error_model="numpy")
def _dense_outward_extrema(controls: tuple[float, ...], padding: float) -> tuple[float, float]:
    lower = next_float64(controls[0] - padding, False)
    upper = next_float64(controls[0] + padding, True)
    for index in range(1, len(controls)):
        lower = np.minimum(lower, next_float64(controls[index] - padding, False))
        upper = np.maximum(upper, next_float64(controls[index] + padding, True))
    return lower, upper


@njit(cache=True, fastmath=False, parallel=False, error_model="numpy", nogil=True)
def _compiled_restrict_controls_into(
    controls: FloatArray,
    start: FloatArray,
    target: FloatArray,
    restricted: FloatArray,
) -> None:
    for row in range(controls.shape[0]):
        for component in range(controls.shape[2]):
            if controls.shape[1] == 4:
                cubic = _dense_restrict_cubic_scalar(
                    controls[row, 0, component],
                    controls[row, 1, component],
                    controls[row, 2, component],
                    controls[row, 3, component],
                    start[row],
                    target[row],
                )
                for control in range(4):
                    restricted[row, control, component] = cubic[control]
            else:
                quadratic = _dense_restrict_quadratic_scalar(
                    controls[row, 0, component],
                    controls[row, 1, component],
                    controls[row, 2, component],
                    start[row],
                    target[row],
                )
                for control in range(3):
                    restricted[row, control, component] = quadratic[control]


@njit(cache=True, fastmath=False, parallel=False, error_model="numpy", nogil=True)
def _compiled_two_diff_into(
    left: FloatArray,
    right: FloatArray,
    difference: FloatArray,
    residual: FloatArray,
) -> None:
    for index in range(left.size):
        difference[index], residual[index] = _dense_two_diff_scalar(left[index], right[index])


@njit(cache=True, fastmath=False, parallel=False, error_model="numpy", nogil=True)
def _compiled_outward_control_bounds_into(
    restricted: FloatArray,
    root: FloatArray,
    extra_scale: FloatArray,
    absolute_padding: FloatArray,
    include_extra: bool,
    include_absolute: bool,
    lower: FloatArray,
    upper: FloatArray,
) -> None:
    for row in range(restricted.shape[0]):
        for component in range(restricted.shape[2]):
            restricted_scale = abs(restricted[row, 0, component])
            for control in range(1, restricted.shape[1]):
                restricted_scale = np.maximum(
                    restricted_scale, abs(restricted[row, control, component])
                )
            root_scale = abs(root[row, 0, component])
            for control in range(1, root.shape[1]):
                root_scale = np.maximum(root_scale, abs(root[row, control, component]))
            scale = np.maximum(restricted_scale, root_scale)
            if include_extra:
                extra = extra_scale[row, 0, component]
                for control in range(1, extra_scale.shape[1]):
                    extra = np.maximum(extra, extra_scale[row, control, component])
                scale = np.maximum(scale, extra)
            absolute = 0.0
            if include_absolute:
                absolute = absolute_padding[row, 0, component]
                for control in range(1, absolute_padding.shape[1]):
                    absolute = np.maximum(absolute, absolute_padding[row, control, component])
            padding = _dense_outward_padding(scale, absolute, include_absolute)
            for control in range(restricted.shape[1]):
                lower[row, control, component] = next_float64(
                    restricted[row, control, component] - padding, False
                )
                upper[row, control, component] = next_float64(
                    restricted[row, control, component] + padding, True
                )


def _restrict_cubic_controls(
    controls: FloatArray,
    start_parameter: FloatArray,
    target_parameter: FloatArray,
) -> FloatArray:
    restricted = np.empty_like(controls)
    _compiled_restrict_controls_into(controls, start_parameter, target_parameter, restricted)
    return restricted


def _restrict_quadratic_controls(
    controls: FloatArray,
    start_parameter: FloatArray,
    target_parameter: FloatArray,
) -> FloatArray:
    restricted = np.empty_like(controls)
    _compiled_restrict_controls_into(controls, start_parameter, target_parameter, restricted)
    return restricted


def _two_diff(left: FloatArray, right: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Return the rounded difference and its exact binary64 residual.

    Knuth's TwoDiff decomposition makes ``difference + residual`` equal to the
    exact difference of the two stored controls when the finite subtraction
    does not overflow.  Unlike an ``abs(left) + abs(right)`` error estimate,
    its residual is independent of the selected coordinate origin.  Equal
    controls are assigned an exact zero explicitly, including the tiny-step
    case where a physical displacement is below the position ULP.
    """

    broadcast_left, broadcast_right = np.broadcast_arrays(left, right)
    difference = np.empty(broadcast_left.shape, dtype=np.float64)
    residual = np.empty_like(difference)
    _compiled_two_diff_into(
        broadcast_left.reshape(-1),
        broadcast_right.reshape(-1),
        difference.reshape(-1),
        residual.reshape(-1),
    )
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

    lower = np.empty_like(restricted_controls)
    upper = np.empty_like(restricted_controls)
    empty = np.empty((0, 0, 0), dtype=np.float64)
    _compiled_outward_control_bounds_into(
        restricted_controls,
        root_controls,
        empty if extra_scale is None else extra_scale,
        empty if absolute_padding is None else absolute_padding,
        extra_scale is not None,
        absolute_padding is not None,
        lower,
        upper,
    )
    return lower, upper


@njit(cache=True, fastmath=False, parallel=False, error_model="numpy")
def _dense_derivative_pair(right: float, left: float, duration: float) -> tuple[float, float]:
    difference, residual = _dense_two_diff_scalar(right, left)
    derivative = 3.0 * difference / duration
    residual_contribution = next_float64(3.0 * abs(residual), True)
    residual_contribution = next_float64(residual_contribution / duration, True)
    if residual == 0.0:
        residual_contribution = 0.0
    return derivative, residual_contribution


@njit(cache=True, fastmath=False, parallel=False, error_model="numpy")
def _dense_derivative_controls_scalar(
    position: tuple[float, float, float, float], initial_velocity: float, duration: float
) -> tuple[tuple[float, float, float], float, float]:
    if not (duration > 0.0):
        return (initial_velocity, initial_velocity, initial_velocity), abs(initial_velocity), 0.0
    first, first_error = _dense_derivative_pair(position[1], position[0], duration)
    second, second_error = _dense_derivative_pair(position[2], position[1], duration)
    third, third_error = _dense_derivative_pair(position[3], position[2], duration)
    error = np.maximum(np.maximum(first_error, second_error), third_error)
    scale = np.maximum(_dense_absolute_maximum((first, second, third)), error)
    return (first, second, third), scale, error


@njit(cache=True, fastmath=False, parallel=False, error_model="numpy")
def _dense_enclose_axis_scalar(
    position: tuple[float, float, float, float],
    velocity: tuple[float, float, float, float],
    duration: float,
    start: float,
    target: float,
) -> tuple[
    float,
    float,
    float,
    float,
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]:
    first, first_residual = _dense_two_diff_scalar(position[0], position[0])
    second, second_residual = _dense_two_diff_scalar(position[1], position[0])
    third, third_residual = _dense_two_diff_scalar(position[2], position[0])
    fourth, fourth_residual = _dense_two_diff_scalar(position[3], position[0])
    relative_root = (first, second, third, fourth)
    relative = _dense_restrict_cubic_scalar(first, second, third, fourth, start, target)
    relative_scale = np.maximum(
        _dense_absolute_maximum(relative), _dense_absolute_maximum(relative_root)
    )
    origin_error = _dense_absolute_maximum(
        (first_residual, second_residual, third_residual, fourth_residual)
    )
    relative_padding = _dense_outward_padding(relative_scale, origin_error, True)
    relative_lower = (
        next_float64(relative[0] - relative_padding, False),
        next_float64(relative[1] - relative_padding, False),
        next_float64(relative[2] - relative_padding, False),
        next_float64(relative[3] - relative_padding, False),
    )
    relative_upper = (
        next_float64(relative[0] + relative_padding, True),
        next_float64(relative[1] + relative_padding, True),
        next_float64(relative[2] + relative_padding, True),
        next_float64(relative[3] + relative_padding, True),
    )
    lower = np.minimum(
        np.minimum(np.minimum(relative_lower[0], relative_lower[1]), relative_lower[2]),
        relative_lower[3],
    )
    upper = np.maximum(
        np.maximum(np.maximum(relative_upper[0], relative_upper[1]), relative_upper[2]),
        relative_upper[3],
    )
    position_lower = next_float64(position[0] + lower, False)
    position_upper = next_float64(position[0] + upper, True)
    restricted_velocity = _dense_restrict_cubic_scalar(
        velocity[0], velocity[1], velocity[2], velocity[3], start, target
    )
    derivative, derivative_scale, derivative_error = _dense_derivative_controls_scalar(
        position, velocity[0], duration
    )
    derivative = _dense_restrict_quadratic_scalar(
        derivative[0], derivative[1], derivative[2], start, target
    )
    derivative_maximum = _dense_absolute_maximum(derivative)
    velocity_scale = np.maximum(
        np.maximum(_dense_absolute_maximum(restricted_velocity), derivative_maximum),
        np.maximum(_dense_absolute_maximum(velocity), derivative_maximum),
    )
    velocity_scale = np.maximum(velocity_scale, derivative_scale)
    velocity_padding = _dense_outward_padding(velocity_scale, derivative_error, True)
    velocity_lower, velocity_upper = _dense_outward_extrema(
        (
            restricted_velocity[0],
            restricted_velocity[1],
            restricted_velocity[2],
            restricted_velocity[3],
            derivative[0],
            derivative[1],
            derivative[2],
        ),
        velocity_padding,
    )
    return (
        position_lower,
        position_upper,
        velocity_lower,
        velocity_upper,
        relative_lower,
        relative_upper,
    )


@njit(cache=True, fastmath=False, parallel=False, error_model="numpy")
def _dense_enclose_charge_scalar(
    root: tuple[float, float, float, float], start: float, target: float
) -> tuple[float, float]:
    restricted = _dense_restrict_cubic_scalar(root[0], root[1], root[2], root[3], start, target)
    scale = np.maximum(_dense_absolute_maximum(restricted), _dense_absolute_maximum(root))
    padding = _dense_outward_padding(scale, 0.0, False)
    lower, upper = _dense_outward_extrema(restricted, padding)
    if root[0] <= 0.0 and root[1] <= 0.0 and root[2] <= 0.0 and root[3] <= 0.0:
        upper = np.minimum(upper, 0.0)
    if root[0] >= 0.0 and root[1] >= 0.0 and root[2] >= 0.0 and root[3] >= 0.0:
        lower = np.maximum(lower, 0.0)
    return lower, upper


@njit(cache=True, fastmath=False, parallel=False, error_model="numpy", nogil=True)
def _compiled_rk4_dense_enclosure_into(
    root_start: FloatArray,
    root_target: FloatArray,
    root_position: FloatArray,
    root_velocity: FloatArray,
    root_charge: FloatArray,
    rows: Int64Array,
    start: FloatArray,
    target: FloatArray,
    inherited_status: UInt8Array,
    position_lower: FloatArray,
    position_upper: FloatArray,
    velocity_lower: FloatArray,
    velocity_upper: FloatArray,
    charge_lower: FloatArray,
    charge_upper: FloatArray,
    position_origin: FloatArray,
    relative_lower: FloatArray,
    relative_upper: FloatArray,
    numerical_status: UInt8Array,
) -> None:
    """Restrict and certify frozen controls without particle-sized intermediates."""

    for row in range(rows.size):
        source = rows[row]
        duration = root_target[source] - root_start[source]
        finite = True
        for component in range(2):
            axis = _dense_enclose_axis_scalar(
                (
                    root_position[source, 0, component],
                    root_position[source, 1, component],
                    root_position[source, 2, component],
                    root_position[source, 3, component],
                ),
                (
                    root_velocity[source, 0, component],
                    root_velocity[source, 1, component],
                    root_velocity[source, 2, component],
                    root_velocity[source, 3, component],
                ),
                duration,
                start[row],
                target[row],
            )
            position_lower[row, component] = axis[0]
            position_upper[row, component] = axis[1]
            velocity_lower[row, component] = axis[2]
            velocity_upper[row, component] = axis[3]
            position_origin[row, component] = root_position[source, 0, component]
            finite &= (
                np.isfinite(axis[0])
                & np.isfinite(axis[1])
                & np.isfinite(axis[2])
                & np.isfinite(axis[3])
            )
            for control in range(4):
                relative_lower[row, control, component] = axis[4][control]
                relative_upper[row, control, component] = axis[5][control]
                finite &= np.isfinite(axis[4][control]) & np.isfinite(axis[5][control])
        charge_lower[row], charge_upper[row] = _dense_enclose_charge_scalar(
            (
                root_charge[source, 0],
                root_charge[source, 1],
                root_charge[source, 2],
                root_charge[source, 3],
            ),
            start[row],
            target[row],
        )
        finite &= np.isfinite(charge_lower[row]) & np.isfinite(charge_upper[row])
        numerical_status[row] = inherited_status[row]
        if (not finite) & (numerical_status[row] == NUMERICAL_STATUS_OK):
            numerical_status[row] = INTEGRATOR_NUMERICAL_FAILURE
        if numerical_status[row] != NUMERICAL_STATUS_OK:
            for component in range(2):
                position_lower[row, component] = root_position[source, 0, component]
                position_upper[row, component] = root_position[source, 0, component]
                velocity_lower[row, component] = root_velocity[source, 0, component]
                velocity_upper[row, component] = root_velocity[source, 0, component]
                position_origin[row, component] = 0.0
                for control in range(4):
                    relative_lower[row, control, component] = 0.0
                    relative_upper[row, control, component] = 0.0
            charge_lower[row] = root_charge[source, 0]
            charge_upper[row] = root_charge[source, 0]


def _enclose_rk4_dense_subinterval(
    path: _Rk4DensePath,
    rows: Int64Array,
    start_time_s: FloatArray,
    target_time_s: FloatArray,
    inherited_status: UInt8Array,
) -> Rk4DenseEnclosure:
    start_parameter = _dense_parameter(path, rows, start_time_s)
    target_parameter = _dense_parameter(path, rows, target_time_s)
    count = rows.size
    enclosure = Rk4DenseEnclosure(
        position_lower_m=np.empty((count, 2), dtype=np.float64),
        position_upper_m=np.empty((count, 2), dtype=np.float64),
        velocity_lower_m_s=np.empty((count, 2), dtype=np.float64),
        velocity_upper_m_s=np.empty((count, 2), dtype=np.float64),
        charge_lower_number=np.empty(count, dtype=np.float64),
        charge_upper_number=np.empty(count, dtype=np.float64),
        position_control_origin_m=np.empty((count, 2), dtype=np.float64),
        relative_position_control_lower_m=np.empty((count, 4, 2), dtype=np.float64),
        relative_position_control_upper_m=np.empty((count, 4, 2), dtype=np.float64),
        numerical_status=np.empty(count, dtype=np.uint8),
    )
    _compiled_rk4_dense_enclosure_into(
        path.start_time_s,
        path.target_time_s,
        path.position_controls_m,
        path.velocity_controls_m_s,
        path.charge_controls_number,
        rows,
        start_parameter,
        target_parameter,
        inherited_status,
        enclosure.position_lower_m,
        enclosure.position_upper_m,
        enclosure.velocity_lower_m_s,
        enclosure.velocity_upper_m_s,
        enclosure.charge_lower_number,
        enclosure.charge_upper_number,
        enclosure.position_control_origin_m,
        enclosure.relative_position_control_lower_m,
        enclosure.relative_position_control_upper_m,
        enclosure.numerical_status,
    )
    return enclosure


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


@njit(cache=True, fastmath=False, parallel=False, nogil=True, error_model="numpy")
def _cubic_hermite_enclose_axis_scalar(
    position: tuple[float, float, float, float],
    residual: tuple[float, float, float, float],
    velocity: tuple[float, float, float],
    origin: float,
    start: float,
    target: float,
) -> tuple[float, float, float, float]:
    restricted = _dense_restrict_cubic_scalar(
        position[0], position[1], position[2], position[3], start, target
    )
    restricted_residual = _dense_restrict_cubic_scalar(
        residual[0], residual[1], residual[2], residual[3], start, target
    )
    restricted_velocity = _dense_restrict_quadratic_scalar(
        velocity[0], velocity[1], velocity[2], start, target
    )
    position_scale = np.maximum(
        _dense_absolute_maximum(restricted), _dense_absolute_maximum(position)
    )
    position_padding = _dense_outward_padding(
        position_scale, _dense_absolute_maximum(restricted_residual), True
    )
    lower, upper = _dense_outward_extrema(restricted, position_padding)
    position_lower = next_float64(origin + lower, False)
    position_upper = next_float64(origin + upper, True)
    velocity_scale = np.maximum(
        _dense_absolute_maximum(restricted_velocity), _dense_absolute_maximum(velocity)
    )
    velocity_padding = _dense_outward_padding(velocity_scale, 0.0, False)
    velocity_lower, velocity_upper = _dense_outward_extrema(restricted_velocity, velocity_padding)
    return position_lower, position_upper, velocity_lower, velocity_upper


@njit(cache=True, fastmath=False, parallel=False, nogil=True, error_model="numpy")
def cubic_hermite_enclosure_into(
    path: _CubicHermitePath,
    rows: Int64Array,
    start_time_s: FloatArray,
    target_time_s: FloatArray,
    position_lower: FloatArray,
    position_upper: FloatArray,
    velocity_lower: FloatArray,
    velocity_upper: FloatArray,
    status: UInt8Array,
) -> None:
    """Restrict original controls with the existing outward-rounding owners."""

    for row in range(rows.size):
        source = rows[row]
        duration = path.target_time_s[source] - path.start_time_s[source]
        start = (start_time_s[row] - path.start_time_s[source]) / duration
        target = (target_time_s[row] - path.start_time_s[source]) / duration
        start = np.minimum(np.maximum(start, 0.0), 1.0)
        target = np.minimum(np.maximum(target, 0.0), 1.0)
        finite = True
        for axis in range(2):
            controls = _cubic_hermite_root_controls_scalar(
                path.start_position_m[source, axis],
                path.end_position_m[source, axis],
                path.start_velocity_m_s[source, axis],
                path.end_velocity_m_s[source, axis],
                duration,
            )
            bound = _cubic_hermite_enclose_axis_scalar(
                controls[0],
                controls[1],
                controls[2],
                path.start_position_m[source, axis],
                start,
                target,
            )
            position_lower[row, axis] = bound[0]
            position_upper[row, axis] = bound[1]
            velocity_lower[row, axis] = bound[2]
            velocity_upper[row, axis] = bound[3]
            finite &= (
                np.isfinite(bound[0])
                & np.isfinite(bound[1])
                & np.isfinite(bound[2])
                & np.isfinite(bound[3])
                & (bound[0] <= bound[1])
                & (bound[2] <= bound[3])
            )
        status[row] = NUMERICAL_STATUS_OK if finite else INTEGRATOR_NUMERICAL_FAILURE
        if not finite:
            _cubic_hermite_motion_into(
                path,
                rows[row : row + 1],
                start_time_s[row : row + 1],
                position_lower[row : row + 1],
                velocity_lower[row : row + 1],
            )
            for axis in range(2):
                position_upper[row, axis] = position_lower[row, axis]
                velocity_upper[row, axis] = velocity_lower[row, axis]


def _enclose_cubic_hermite_subinterval(
    path: _CubicHermitePath,
    rows: Int64Array,
    start_time_s: FloatArray,
    target_time_s: FloatArray,
) -> CurvedPathEnclosure:
    count = rows.size
    result = CurvedPathEnclosure(
        np.empty((count, 2), dtype=np.float64),
        np.empty((count, 2), dtype=np.float64),
        np.empty((count, 2), dtype=np.float64),
        np.empty((count, 2), dtype=np.float64),
        np.empty(count, dtype=np.uint8),
    )
    cubic_hermite_enclosure_into(
        path,
        rows,
        start_time_s,
        target_time_s,
        result.position_lower_m,
        result.position_upper_m,
        result.velocity_lower_m_s,
        result.velocity_upper_m_s,
        result.numerical_status,
    )
    _require_finite_enclosure(result)
    return result


@njit(cache=True, fastmath=False, parallel=False, nogil=True, error_model="numpy")
def cubic_hermite_restriction_into(
    proposal: StepProposalProjection,
    rows: Int64Array,
    starts: FloatArray,
    targets: FloatArray,
    start_state: _StageAdvanceOutputs,
    end_state: _StageAdvanceOutputs,
    position_lower: FloatArray,
    position_upper: FloatArray,
    velocity_lower: FloatArray,
    velocity_upper: FloatArray,
    enclosure_status: UInt8Array,
) -> None:
    """Evaluate a child interval while retaining the original root authority."""

    if proposal.path_kind != "cubic_hermite":
        raise ValueError("restriction requires a cubic Hermite proposal")
    step_proposal_dense_state_into(proposal, rows, starts, start_state)
    step_proposal_dense_state_into(proposal, rows, targets, end_state)
    cubic_hermite_enclosure_into(
        proposal.cubic_hermite_path,
        rows,
        starts,
        targets,
        position_lower,
        position_upper,
        velocity_lower,
        velocity_upper,
        enclosure_status,
    )
    for row in range(rows.size):
        if end_state.numerical_status[row] == NUMERICAL_STATUS_OK:
            end_state.numerical_status[row] = enclosure_status[row]


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
    dense_full_enclosure = None
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
        for value in (
            dense_enclosure.position_lower_m,
            dense_enclosure.position_upper_m,
            dense_enclosure.velocity_lower_m_s,
            dense_enclosure.velocity_upper_m_s,
            dense_enclosure.charge_lower_number,
            dense_enclosure.charge_upper_number,
            dense_enclosure.position_control_origin_m,
            dense_enclosure.relative_position_control_lower_m,
            dense_enclosure.relative_position_control_upper_m,
            dense_enclosure.numerical_status,
        ):
            value.setflags(write=False)
        dense_full_enclosure = (dense_path, dense_enclosure)
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
        _rk4_dense_full_enclosure=dense_full_enclosure,
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
    outputs, workspace, _ = _run_relaxation_advance(
        evaluator,
        particle_index,
        start_time_s,
        elapsed,
        start_position_m,
        start_velocity_m_s,
        start_charge_number,
        predictor_only=True,
    )
    return _FrozenStartPredictorData(
        sample=ProposalSample(
            particle_index=particle_index.copy(),
            position_m=outputs.position_m,
            velocity_m_s=outputs.velocity_m_s,
            charge_number=outputs.charge_number,
            support_inside=outputs.support_inside,
            applicability_inside=outputs.applicability_inside,
            numerical_status=outputs.numerical_status,
        ),
        elapsed_s=elapsed.copy(),
        target_velocity_m_s=workspace.target_velocity_m_s[:count].copy(),
        additive_acceleration_m_s2=workspace.additive_acceleration_m_s2[:count].copy(),
        charge_rate_number_s=workspace.charge_rate_number_s[:count].copy(),
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

    The enclosure follows the exact algebra used by ``_compiled_rk4_advance_into``
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
    underflow_unit = _CURVED_CHORD_ROUNDOFF_EPS_FACTOR * next_float64(0.0, True)
    velocity_span = next_float64(velocity_upper - velocity_lower, True)
    velocity_abs = max(abs(velocity_lower), abs(velocity_upper))
    motion_scale = next_float64(velocity_abs * interval_s, True)
    curve_deviation = next_float64(velocity_span * interval_s, True)
    normal_padding = next_float64(abs(start) * roundoff_factor, True)
    normal_padding = next_float64(
        normal_padding + next_float64(abs(end) * roundoff_factor, True),
        True,
    )
    normal_padding = next_float64(
        normal_padding + next_float64(motion_scale * (2.0 * roundoff_factor), True),
        True,
    )
    underflow_padding = next_float64(underflow_unit, True)
    underflow_padding = next_float64(
        underflow_padding + next_float64(interval_s * underflow_unit, True),
        True,
    )
    underflow_padding = next_float64(
        underflow_padding + next_float64(velocity_abs * underflow_unit, True),
        True,
    )
    underflow_padding = next_float64(
        underflow_padding + next_float64(abs(start) * underflow_unit, True),
        True,
    )
    underflow_padding = next_float64(
        underflow_padding + next_float64(abs(end) * underflow_unit, True),
        True,
    )
    deviation = next_float64(curve_deviation + normal_padding, True)
    deviation = next_float64(deviation + underflow_padding, True)
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


def allocate_stage_advance_workspace(capacity: int) -> StageAdvanceWorkspace:
    """Allocate one compact input bank, reusable across native stages."""

    return StageAdvanceWorkspace(
        np.empty(capacity, dtype=np.int64),
        np.empty(capacity, dtype=np.int64),
        np.empty(capacity, dtype=np.float64),
        np.empty((capacity, 2), dtype=np.float64),
        np.empty((capacity, 2), dtype=np.float64),
        np.empty(capacity, dtype=np.float64),
        np.empty(capacity, dtype=np.uint8),
    )


def allocate_relaxation_advance_workspace(capacity: int) -> RelaxationAdvanceWorkspace:
    """Allocate the compact bank, predictor state and reusable coefficients."""

    return RelaxationAdvanceWorkspace(
        allocate_stage_advance_workspace(capacity),
        np.empty((capacity, 2), dtype=np.float64),
        np.empty((capacity, 2), dtype=np.float64),
        np.empty(capacity, dtype=np.float64),
        np.empty(capacity, dtype=np.float64),
        np.empty(capacity, dtype=np.float64),
        np.empty((capacity, 2), dtype=np.float64),
        np.empty((capacity, 2), dtype=np.float64),
        np.empty(capacity, dtype=np.float64),
        np.empty(capacity, dtype=np.float64),
    )


def _allocate_stage_advance_outputs(
    count: int, *, retain_field_cells: bool = True
) -> _StageAdvanceOutputs:
    return _StageAdvanceOutputs(
        np.empty((count, 2), dtype=np.float64),
        np.empty((count, 2), dtype=np.float64),
        np.empty(count, dtype=np.float64),
        np.empty(count, dtype=np.bool_),
        np.empty(count, dtype=np.bool_),
        np.empty(count, dtype=np.uint8),
        np.empty(count if retain_field_cells else 0, dtype=np.int64),
    )


@dataclass(slots=True)
class _PythonRelaxationAdapter:
    """Project a verification evaluator onto the sole relaxation body."""

    evaluator: RelaxationStageEvaluator
    has_field_cell: bool = False


def _evaluate_python_relaxation_stage(
    context: _PythonRelaxationAdapter,
    stage_workspace: object,
    particle_index: Int64Array,
    time_s: FloatArray,
    position_m: FloatArray,
    velocity_m_s: FloatArray,
    charge_number: FloatArray,
    incoming_status: UInt8Array,
) -> tuple:
    del stage_workspace, incoming_status
    stage = context.evaluator(particle_index, time_s, position_m, velocity_m_s, charge_number)
    _validate_relaxation_stage(stage, particle_index.size)
    cells = stage.field_cell_id
    context.has_field_cell = cells is not None
    if cells is None:
        cells = np.full(particle_index.size, -1, dtype=np.int64)
    return (
        stage.linear_drag_rate_s_inv,
        stage.target_velocity_m_s,
        stage.additive_acceleration_m_s2,
        stage.charge_rate_number_s,
        stage.charge_rate_derivative_s_inv,
        stage.support_inside,
        stage.applicability_inside,
        stage.numerical_status,
        cells,
    )


@njit(cache=True, fastmath=False, parallel=False, nogil=True, error_model="numpy")
def _copy_stage_state_into(
    position: FloatArray, velocity: FloatArray, charge: FloatArray, outputs: _StageAdvanceOutputs
) -> None:
    for row in range(charge.size):
        outputs.charge_number[row] = charge[row]
        for axis in range(2):
            outputs.position_m[row, axis] = position[row, axis]
            outputs.velocity_m_s[row, axis] = velocity[row, axis]


@njit(cache=True, fastmath=False, parallel=False, nogil=True, error_model="numpy")
def _initialize_stage_advance_into(
    position: FloatArray, velocity: FloatArray, charge: FloatArray, outputs: _StageAdvanceOutputs
) -> None:
    _copy_stage_state_into(position, velocity, charge, outputs)
    for row in range(charge.size):
        outputs.support_inside[row] = True
        outputs.applicability_inside[row] = True
        outputs.numerical_status[row] = NUMERICAL_STATUS_OK
        if outputs.field_cell_id.size:
            outputs.field_cell_id[row] = -1


@njit(cache=True, fastmath=False, parallel=False, nogil=True, error_model="numpy")
def _pack_stage_advance_into(
    particle_index: Int64Array,
    start_time: FloatArray,
    step: FloatArray,
    position: FloatArray,
    velocity: FloatArray,
    charge: FloatArray,
    status: UInt8Array,
    positive_only: bool,
    fraction: float,
    workspace: StageAdvanceWorkspace,
) -> int:
    count = 0
    for row in range(particle_index.size):
        if status[row] != NUMERICAL_STATUS_OK or (positive_only and step[row] <= 0.0):
            continue
        workspace.active_rows[count] = row
        workspace.particle_index[count] = particle_index[row]
        workspace.time_s[count] = (
            start_time[row] if fraction == 0.0 else start_time[row] + fraction * step[row]
        )
        workspace.charge_number[count] = charge[row]
        workspace.incoming_status[count] = NUMERICAL_STATUS_OK
        for axis in range(2):
            workspace.position_m[count, axis] = position[row, axis]
            workspace.velocity_m_s[count, axis] = velocity[row, axis]
        count += 1
    return count


@njit(cache=True, fastmath=False, parallel=False, nogil=True, error_model="numpy")
def _collect_stage_verdicts_into(
    count: int,
    support: BoolArray,
    applicable: BoolArray,
    status: UInt8Array,
    cells: Int64Array,
    endpoint: bool,
    workspace: StageAdvanceWorkspace,
    outputs: _StageAdvanceOutputs,
) -> None:
    for local in range(count):
        row = workspace.active_rows[local]
        outputs.support_inside[row] = outputs.support_inside[row] and support[local]
        outputs.applicability_inside[row] = outputs.applicability_inside[row] and applicable[local]
        outputs.numerical_status[row] = status[local]
        if endpoint:
            outputs.field_cell_id[row] = cells[local]


@njit(cache=True, fastmath=False, parallel=False, nogil=True, error_model="numpy")
def _collect_relaxation_coefficients_into(
    count: int, values: tuple, workspace: RelaxationAdvanceWorkspace
) -> None:
    for local in range(count):
        row = workspace.stage.active_rows[local]
        workspace.rate_s_inv[row] = values[0][local]
        workspace.charge_rate_number_s[row] = values[3][local]
        workspace.charge_rate_derivative_s_inv[row] = values[4][local]
        for axis in range(2):
            workspace.target_velocity_m_s[row, axis] = values[1][local, axis]
            workspace.additive_acceleration_m_s2[row, axis] = values[2][local, axis]


@njit(cache=True, fastmath=False, parallel=False, nogil=True, error_model="numpy")
def _affine_exponential_multiplier_scalar(derivative: float, elapsed: float) -> float:
    result = elapsed
    if derivative < 0.0:
        argument = derivative * elapsed
        if np.isfinite(argument) and argument != 0.0:
            result = elapsed * (np.expm1(argument) / argument)
        elif np.isneginf(argument):
            result = -1.0 / derivative
    return result


@njit(cache=True, fastmath=False, parallel=False, nogil=True, error_model="numpy")
def _affine_exponential_multiplier_into(
    derivative: FloatArray, elapsed: FloatArray, result: FloatArray
) -> None:
    for row in range(elapsed.size):
        result.flat[row] = _affine_exponential_multiplier_scalar(
            derivative.flat[row], elapsed.flat[row]
        )


def _affine_exponential_multiplier(
    rate_derivative_s_inv: FloatArray, elapsed_s: FloatArray
) -> FloatArray:
    result = np.empty_like(elapsed_s)
    _affine_exponential_multiplier_into(rate_derivative_s_inv, elapsed_s, result)
    return result


@njit(cache=True, fastmath=False, parallel=False, nogil=True, error_model="numpy")
def stable_charge_update_into(
    start: FloatArray,
    reference: FloatArray,
    rate: FloatArray,
    derivative: FloatArray,
    elapsed: FloatArray,
    status: UInt8Array,
    result: FloatArray,
) -> None:
    """Apply one affine charge update without replacing an earlier failure."""

    for row in range(start.size):
        result[row] = start[row]
        if status[row] != NUMERICAL_STATUS_OK:
            continue
        valid = (
            np.isfinite(start[row])
            and np.isfinite(reference[row])
            and np.isfinite(rate[row])
            and np.isfinite(derivative[row])
            and np.isfinite(elapsed[row])
            and elapsed[row] >= 0.0
            and derivative[row] <= 0.0
        )
        affine_start_rate = rate[row] + derivative[row] * (start[row] - reference[row])
        multiplier = _affine_exponential_multiplier_scalar(derivative[row], elapsed[row])
        candidate = start[row] + multiplier * affine_start_rate
        if valid and np.isfinite(candidate):
            result[row] = candidate
        else:
            status[row] = INTEGRATOR_NUMERICAL_FAILURE


@njit(cache=True, fastmath=False, parallel=False, nogil=True, error_model="numpy")
def affine_exponential_charge_into(
    root: FloatArray,
    elapsed: FloatArray,
    rate: FloatArray,
    derivative: FloatArray,
    result: FloatArray,
) -> None:
    """Evaluate validated root-owned charge, retaining arithmetic overflow."""

    for row in range(root.size):
        multiplier = _affine_exponential_multiplier_scalar(derivative.flat[row], elapsed.flat[row])
        result.flat[row] = root.flat[row] + multiplier * rate.flat[row]


@njit(cache=True, fastmath=False, parallel=False, nogil=True, error_model="numpy")
def _relaxation_motion_into(
    position: FloatArray,
    velocity: FloatArray,
    charge: FloatArray,
    workspace: RelaxationAdvanceWorkspace,
    status: UInt8Array,
    result_position: FloatArray,
    result_velocity: FloatArray,
    skip_zero: bool,
) -> None:
    count = charge.size
    _compiled_exponential_update_into(
        position,
        velocity,
        workspace.rate_s_inv,
        workspace.target_velocity_m_s,
        workspace.additive_acceleration_m_s2,
        workspace.elapsed_s,
        result_position,
        result_velocity,
        count,
    )
    for row in range(count):
        failed = status[row] != NUMERICAL_STATUS_OK
        skip = skip_zero and workspace.elapsed_s[row] == 0.0
        finite = _rk4_state_is_finite(
            result_position[row, 0],
            result_position[row, 1],
            result_velocity[row, 0],
            result_velocity[row, 1],
            charge[row],
        )
        if not failed and not skip and not finite:
            status[row] = INTEGRATOR_NUMERICAL_FAILURE
        if failed or skip or not finite:
            for axis in range(2):
                result_position[row, axis] = position[row, axis]
                result_velocity[row, axis] = velocity[row, axis]


@njit(cache=True, fastmath=False, parallel=False, nogil=True, error_model="numpy")
def _initialize_relaxation_workspace_into(
    step: FloatArray, workspace: RelaxationAdvanceWorkspace, predictor_only: bool
) -> None:
    for row in range(step.size):
        workspace.elapsed_s[row] = step[row] if predictor_only else 0.5 * step[row]
        workspace.rate_s_inv[row] = 0.0
        workspace.charge_rate_number_s[row] = 0.0
        workspace.charge_rate_derivative_s_inv[row] = 0.0
        for axis in range(2):
            workspace.target_velocity_m_s[row, axis] = 0.0
            workspace.additive_acceleration_m_s2[row, axis] = 0.0


# First-class dispatcher identity is process-local; its disk-cache index can hold dead weakrefs.
@njit(cache=False, fastmath=False, parallel=False, nogil=True, error_model="numpy")
def _compiled_relaxation_advance_into(
    stage_evaluator: Callable,
    context: object,
    stage_workspace: object,
    particle_index: Int64Array,
    start_time: FloatArray,
    step: FloatArray,
    position: FloatArray,
    velocity: FloatArray,
    charge: FloatArray,
    workspace: RelaxationAdvanceWorkspace,
    outputs: _StageAdvanceOutputs,
    predictor_only: bool,
) -> bool:
    """Own start, coupled predictor, frozen midpoint and endpoint validation."""

    _initialize_stage_advance_into(position, velocity, charge, outputs)
    _initialize_relaxation_workspace_into(step, workspace, predictor_only)
    count_rows = particle_index.size
    midpoint_position = workspace.midpoint_position_m[:count_rows]
    midpoint_velocity = workspace.midpoint_velocity_m_s[:count_rows]
    midpoint_charge = workspace.midpoint_charge_number[:count_rows]
    stage = workspace.stage
    for phase in range(3):
        source_position, source_velocity, source_charge = position, velocity, charge
        if phase == 1:
            source_position, source_velocity, source_charge = (
                midpoint_position,
                midpoint_velocity,
                midpoint_charge,
            )
        elif phase == 2:
            source_position, source_velocity, source_charge = (
                outputs.position_m,
                outputs.velocity_m_s,
                outputs.charge_number,
            )
        count = _pack_stage_advance_into(
            particle_index,
            start_time,
            step,
            source_position,
            source_velocity,
            source_charge,
            outputs.numerical_status,
            phase < 2 and not predictor_only,
            0.5 * phase,
            stage,
        )
        if count:
            values = stage_evaluator(
                context,
                stage_workspace,
                stage.particle_index[:count],
                stage.time_s[:count],
                stage.position_m[:count],
                stage.velocity_m_s[:count],
                stage.charge_number[:count],
                stage.incoming_status[:count],
            )
            _collect_stage_verdicts_into(
                count, values[5], values[6], values[7], values[8], phase == 2, stage, outputs
            )
            if phase < 2:
                _collect_relaxation_coefficients_into(count, values, workspace)
        if phase == 2:
            return count > 0
        if phase == 0:
            _relaxation_motion_into(
                position,
                velocity,
                charge,
                workspace,
                outputs.numerical_status,
                midpoint_position,
                midpoint_velocity,
                not predictor_only,
            )
            stable_charge_update_into(
                charge,
                charge,
                workspace.charge_rate_number_s,
                workspace.charge_rate_derivative_s_inv,
                workspace.elapsed_s,
                outputs.numerical_status,
                midpoint_charge,
            )
            if predictor_only:
                _copy_stage_state_into(
                    midpoint_position, midpoint_velocity, midpoint_charge, outputs
                )
                return False
        else:
            for row in range(count_rows):
                workspace.elapsed_s[row] = step[row]
            stable_charge_update_into(
                charge,
                midpoint_charge,
                workspace.charge_rate_number_s,
                workspace.charge_rate_derivative_s_inv,
                workspace.elapsed_s,
                outputs.numerical_status,
                outputs.charge_number,
            )
            _relaxation_motion_into(
                position,
                velocity,
                outputs.charge_number,
                workspace,
                outputs.numerical_status,
                outputs.position_m,
                outputs.velocity_m_s,
                True,
            )
            for row in range(count_rows):
                if step[row] == 0.0:
                    outputs.charge_number[row] = charge[row]
    return False


def _run_relaxation_advance(
    evaluator: RelaxationStageEvaluator,
    particle_index: Int64Array,
    start_time_s: FloatArray,
    elapsed_s: FloatArray,
    position_m: FloatArray,
    velocity_m_s: FloatArray,
    charge_number: FloatArray,
    *,
    predictor_only: bool,
    outputs: _StageAdvanceOutputs | None = None,
) -> tuple[_StageAdvanceOutputs, RelaxationAdvanceWorkspace, bool]:
    count = particle_index.size
    owner = getattr(evaluator, "__self__", evaluator)
    workspace = getattr(owner, "native_relaxation_workspace", None)
    if workspace is None:
        workspace = allocate_relaxation_advance_workspace(count)
    if workspace.stage.active_rows.size < count:
        raise ValueError("relaxation advance exceeds its workspace capacity")
    if outputs is None:
        outputs = _allocate_stage_advance_outputs(count, retain_field_cells=not predictor_only)
    stage_evaluator = getattr(owner, "native_relaxation_stage_evaluator", None)
    if stage_evaluator is None:
        adapter = _PythonRelaxationAdapter(evaluator)
        numerical_body = getattr(_compiled_relaxation_advance_into, "py_func", None)
        assert numerical_body is not None
        endpoint = numerical_body(
            _evaluate_python_relaxation_stage,
            adapter,
            (),
            particle_index,
            start_time_s,
            elapsed_s,
            position_m,
            velocity_m_s,
            charge_number,
            workspace,
            outputs,
            predictor_only,
        )
        has_field_cell = endpoint and adapter.has_field_cell
    else:
        context = getattr(owner, "native_context", None)
        stage_workspace = getattr(owner, "native_workspace", None)
        if context is None or stage_workspace is None:
            raise ValueError("native relaxation stage evaluator lost its prepared bindings")
        has_field_cell = _compiled_relaxation_advance_into(
            stage_evaluator,
            context,
            stage_workspace,
            particle_index,
            start_time_s,
            elapsed_s,
            position_m,
            velocity_m_s,
            charge_number,
            workspace,
            outputs,
            predictor_only,
        )
    return outputs, workspace, has_field_cell


def _advance_exponential_midpoint(
    evaluator: RelaxationStageEvaluator,
    particle_index: Int64Array,
    start_time_s: FloatArray,
    elapsed_s: FloatArray,
    position_m: FloatArray,
    velocity_m_s: FloatArray,
    charge_number: FloatArray,
) -> _AdvanceResult:
    outputs, _, has_field_cell = _run_relaxation_advance(
        evaluator,
        particle_index,
        start_time_s,
        elapsed_s,
        position_m,
        velocity_m_s,
        charge_number,
        predictor_only=False,
    )
    return _AdvanceResult(
        outputs.position_m,
        outputs.velocity_m_s,
        outputs.charge_number,
        outputs.support_inside,
        outputs.applicability_inside,
        outputs.numerical_status,
        outputs.field_cell_id if has_field_cell else None,
    )


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
    result = np.empty_like(root)
    affine_exponential_charge_into(root, elapsed, affine_rate, derivative, result)
    # Brownian execution owns row-local numerical failure.  Returning an
    # arithmetic overflow here lets that caller retire only the affected row;
    # malformed inputs above remain fail-fast structural errors.
    return result


@njit(cache=True, fastmath=False, parallel=False, nogil=True, error_model="numpy")
def _quadratic_position_scalar(
    position: float, velocity: float, acceleration: float, step: float
) -> float:
    return position + step * velocity + 0.5 * (step * step) * acceleration


@njit(cache=True, fastmath=False, parallel=False, nogil=True, error_model="numpy")
def _compiled_linear_advance_into(
    position: FloatArray,
    velocity: FloatArray,
    charge: FloatArray,
    elapsed: FloatArray,
    outputs: _StageAdvanceOutputs,
) -> None:
    _initialize_stage_advance_into(position, velocity, charge, outputs)
    for row in range(charge.size):
        x = position[row, 0] + elapsed[row] * velocity[row, 0]
        y = position[row, 1] + elapsed[row] * velocity[row, 1]
        if np.isfinite(x) and np.isfinite(y):
            if elapsed[row] != 0.0:
                outputs.position_m[row, 0] = x
                outputs.position_m[row, 1] = y
        else:
            outputs.numerical_status[row] = INTEGRATOR_NUMERICAL_FAILURE


@njit(cache=True, fastmath=False, parallel=False, nogil=True, error_model="numpy")
def _initialize_quadratic_advance_into(
    position: FloatArray,
    velocity: FloatArray,
    charge: FloatArray,
    acceleration: FloatArray,
    elapsed: FloatArray,
    outputs: _StageAdvanceOutputs,
) -> None:
    _initialize_stage_advance_into(position, velocity, charge, outputs)
    for row in range(charge.size):
        finite = True
        for axis in range(2):
            x, v, a, step = (
                position[row, axis],
                velocity[row, axis],
                acceleration[row, axis],
                elapsed[row],
            )
            midpoint = _quadratic_position_scalar(x, v, a, 0.5 * step)
            midpoint_velocity = v + (0.5 * step) * a
            endpoint = _quadratic_position_scalar(x, v, a, step)
            endpoint_velocity = v + step * a
            finite = finite and np.isfinite(midpoint) and np.isfinite(midpoint_velocity)
            finite = finite and np.isfinite(endpoint) and np.isfinite(endpoint_velocity)
            outputs.position_m[row, axis] = endpoint
            outputs.velocity_m_s[row, axis] = endpoint_velocity
        if not finite:
            outputs.numerical_status[row] = INTEGRATOR_NUMERICAL_FAILURE
        if not finite or elapsed[row] == 0.0:
            for axis in range(2):
                outputs.position_m[row, axis] = position[row, axis]
                outputs.velocity_m_s[row, axis] = velocity[row, axis]


@njit(cache=True, fastmath=False, parallel=False, nogil=True, error_model="numpy")
def _prepare_quadratic_stage_into(
    phase: int,
    particle_index: Int64Array,
    start_time: FloatArray,
    step: FloatArray,
    position: FloatArray,
    velocity: FloatArray,
    charge: FloatArray,
    acceleration: FloatArray,
    workspace: StageAdvanceWorkspace,
    outputs: _StageAdvanceOutputs,
) -> int:
    source_position = outputs.position_m if phase == 2 else position
    source_velocity = outputs.velocity_m_s if phase == 2 else velocity
    count = _pack_stage_advance_into(
        particle_index,
        start_time,
        step,
        source_position,
        source_velocity,
        charge,
        outputs.numerical_status,
        False,
        0.5 * phase,
        workspace,
    )
    if phase == 1:
        for local in range(count):
            row = workspace.active_rows[local]
            half_step = 0.5 * step[row]
            for axis in range(2):
                workspace.position_m[local, axis] = _quadratic_position_scalar(
                    position[row, axis], velocity[row, axis], acceleration[row, axis], half_step
                )
                workspace.velocity_m_s[local, axis] = (
                    velocity[row, axis] + half_step * acceleration[row, axis]
                )
    return count


# First-class dispatcher identity is process-local; its disk-cache index can hold dead weakrefs.
@njit(cache=False, fastmath=False, parallel=False, nogil=True, error_model="numpy")
def _compiled_quadratic_advance_into(
    stage_evaluator: Callable,
    context: object,
    stage_workspace: object,
    particle_index: Int64Array,
    start_time: FloatArray,
    step: FloatArray,
    position: FloatArray,
    velocity: FloatArray,
    charge: FloatArray,
    acceleration: FloatArray,
    workspace: StageAdvanceWorkspace,
    outputs: _StageAdvanceOutputs,
) -> bool:
    """Keep analytic advance and all three actual-state inspections native."""

    _initialize_quadratic_advance_into(position, velocity, charge, acceleration, step, outputs)
    endpoint_evaluated = False
    for phase in range(3):
        count = _prepare_quadratic_stage_into(
            phase,
            particle_index,
            start_time,
            step,
            position,
            velocity,
            charge,
            acceleration,
            workspace,
            outputs,
        )
        if phase == 2:
            endpoint_evaluated = count > 0
        if count:
            values = stage_evaluator(
                context,
                stage_workspace,
                workspace.particle_index[:count],
                workspace.time_s[:count],
                workspace.position_m[:count],
                workspace.velocity_m_s[:count],
                workspace.charge_number[:count],
                workspace.incoming_status[:count],
            )
            if np.any(values[1] != 0.0):
                raise ValueError("quadratic exact path requires fixed charge")
            _collect_stage_verdicts_into(
                count, values[2], values[3], values[4], values[5], phase == 2, workspace, outputs
            )
    return endpoint_evaluated


def _advance_linear(
    position_m: FloatArray,
    velocity_m_s: FloatArray,
    charge_number: FloatArray,
    elapsed_s: FloatArray,
) -> _AdvanceResult:
    outputs = _allocate_stage_advance_outputs(charge_number.size, retain_field_cells=False)
    _compiled_linear_advance_into(position_m, velocity_m_s, charge_number, elapsed_s, outputs)
    return _AdvanceResult(
        outputs.position_m,
        outputs.velocity_m_s,
        outputs.charge_number,
        outputs.support_inside,
        outputs.applicability_inside,
        outputs.numerical_status,
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
    *,
    outputs: _StageAdvanceOutputs | None = None,
) -> _AdvanceResult:
    count = particle_index.size
    workspace = getattr(evaluator, "native_stage_advance_workspace", None)
    if workspace is None:
        workspace = allocate_stage_advance_workspace(count)
    if workspace.active_rows.size < count:
        raise ValueError("quadratic advance exceeds its workspace capacity")
    if outputs is None:
        outputs = _allocate_stage_advance_outputs(count)
    stage_evaluator = getattr(evaluator, "native_stage_evaluator", None)
    if stage_evaluator is None:
        adapter = _PythonStageAdapter(evaluator)
        numerical_body = getattr(_compiled_quadratic_advance_into, "py_func", None)
        assert numerical_body is not None
        endpoint = numerical_body(
            _evaluate_python_dynamics_stage,
            adapter,
            (),
            particle_index,
            start_time_s,
            elapsed_s,
            position_m,
            velocity_m_s,
            charge_number,
            acceleration_m_s2,
            workspace,
            outputs,
        )
        has_field_cell = endpoint and adapter.has_field_cell
    else:
        context = getattr(evaluator, "native_context", None)
        stage_workspace = getattr(evaluator, "native_workspace", None)
        if context is None or stage_workspace is None:
            raise ValueError("native quadratic stage evaluator lost its prepared bindings")
        has_field_cell = _compiled_quadratic_advance_into(
            stage_evaluator,
            context,
            stage_workspace,
            particle_index,
            start_time_s,
            elapsed_s,
            position_m,
            velocity_m_s,
            charge_number,
            acceleration_m_s2,
            workspace,
            outputs,
        )
    return _AdvanceResult(
        outputs.position_m,
        outputs.velocity_m_s,
        outputs.charge_number,
        outputs.support_inside,
        outputs.applicability_inside,
        outputs.numerical_status,
        outputs.field_cell_id if has_field_cell else None,
    )


@njit(cache=True, fastmath=False, parallel=False, nogil=True, error_model="numpy")
def _quadratic_position_into(
    position: FloatArray,
    velocity: FloatArray,
    acceleration: FloatArray,
    elapsed: FloatArray,
    result: FloatArray,
) -> None:
    for row in range(elapsed.size):
        for axis in range(2):
            result[row, axis] = _quadratic_position_scalar(
                position[row, axis], velocity[row, axis], acceleration[row, axis], elapsed[row]
            )


def _quadratic_position(
    start_position_m: FloatArray,
    start_velocity_m_s: FloatArray,
    acceleration_m_s2: FloatArray,
    elapsed_s: FloatArray,
) -> FloatArray:
    result = np.empty_like(start_position_m)
    _quadratic_position_into(
        start_position_m, start_velocity_m_s, acceleration_m_s2, elapsed_s, result
    )
    return result


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


def allocate_rk4_advance_workspace(capacity: int) -> Rk4AdvanceWorkspace:
    """Allocate scratch once for one exclusive bounded advance slot."""

    return Rk4AdvanceWorkspace(
        np.empty(capacity, dtype=np.int64),
        np.empty(capacity, dtype=np.int64),
        np.empty(capacity, dtype=np.float64),
        np.empty((capacity, 2), dtype=np.float64),
        np.empty((capacity, 2), dtype=np.float64),
        np.empty(capacity, dtype=np.float64),
        np.empty(capacity, dtype=np.uint8),
        np.empty((3, capacity, 2), dtype=np.float64),
        np.empty((4, capacity, 2), dtype=np.float64),
        np.empty((4, capacity), dtype=np.float64),
    )


@dataclass(slots=True)
class _PythonStageAdapter:
    """Project a verification evaluator onto the owned native stage bodies."""

    evaluator: StageEvaluator
    has_field_cell: bool = False


def _evaluate_python_dynamics_stage(
    context: _PythonStageAdapter,
    stage_workspace: object,
    particle_index: Int64Array,
    time_s: FloatArray,
    position_m: FloatArray,
    velocity_m_s: FloatArray,
    charge_number: FloatArray,
    incoming_status: UInt8Array,
) -> tuple[FloatArray, FloatArray, BoolArray, BoolArray, UInt8Array, Int64Array]:
    del stage_workspace, incoming_status
    stage = context.evaluator(particle_index, time_s, position_m, velocity_m_s, charge_number)
    _validate_stage(stage, particle_index.size)
    cells = stage.field_cell_id
    context.has_field_cell = cells is not None
    if cells is None:
        cells = np.full(particle_index.size, -1, dtype=np.int64)
    return (
        stage.acceleration_m_s2,
        stage.charge_rate_number_s,
        stage.support_inside,
        stage.applicability_inside,
        stage.numerical_status,
        cells,
    )


@njit(cache=True, fastmath=False, parallel=False, nogil=True, error_model="numpy")
def _rk4_stage_component(start: float, rate: float, step: float) -> float:
    return start + step * rate


@njit(cache=True, fastmath=False, parallel=False, nogil=True, error_model="numpy")
def _rk4_endpoint_component(
    start: float, first: float, second: float, third: float, fourth: float, weight: float
) -> float:
    total = first + 2.0 * second
    total = total + 2.0 * third
    total = total + fourth
    return start + weight * total


@njit(cache=True, fastmath=False, parallel=False, nogil=True, error_model="numpy")
def _rk4_state_is_finite(
    position_0: float, position_1: float, velocity_0: float, velocity_1: float, charge: float
) -> bool:
    return bool(
        np.isfinite(position_0)
        & np.isfinite(position_1)
        & np.isfinite(velocity_0)
        & np.isfinite(velocity_1)
        & np.isfinite(charge)
    )


@njit(cache=True, fastmath=False, parallel=False, nogil=True, error_model="numpy")
def _initialize_rk4_advance_into(
    position_m: FloatArray,
    velocity_m_s: FloatArray,
    charge_number: FloatArray,
    workspace: Rk4AdvanceWorkspace,
    outputs: _Rk4AdvanceOutputs,
) -> None:
    for row in range(charge_number.size):
        outputs.numerical_status[row] = NUMERICAL_STATUS_OK
        outputs.support_inside[row] = True
        outputs.applicability_inside[row] = True
        outputs.field_cell_id[row] = -1
        outputs.charge_number[row] = charge_number[row]
        for axis in range(2):
            outputs.position_m[row, axis] = position_m[row, axis]
            outputs.velocity_m_s[row, axis] = velocity_m_s[row, axis]
            for control in range(2):
                outputs.position_inner_m[row, control, axis] = position_m[row, axis]
                outputs.velocity_inner_m_s[row, control, axis] = velocity_m_s[row, axis]
        for control in range(2):
            outputs.charge_inner_number[row, control] = charge_number[row]
        for stage in range(4):
            workspace.charge_rate_number_s[stage, row] = 0.0
            for axis in range(2):
                workspace.acceleration_m_s2[stage, row, axis] = 0.0


@njit(cache=True, fastmath=False, parallel=False, nogil=True, error_model="numpy")
def _rk4_stage_is_active(stage: int, step: float, status: np.uint8) -> bool:
    return status == NUMERICAL_STATUS_OK and (stage == 4 or step > 0.0)


@njit(cache=True, fastmath=False, parallel=False, nogil=True, error_model="numpy")
def _prepare_rk4_stage_into(
    stage: int,
    particle_index: Int64Array,
    start_time_s: FloatArray,
    step_s: FloatArray,
    position_m: FloatArray,
    velocity_m_s: FloatArray,
    charge_number: FloatArray,
    workspace: Rk4AdvanceWorkspace,
    outputs: _Rk4AdvanceOutputs,
) -> int:
    count = 0
    for row in range(particle_index.size):
        if not _rk4_stage_is_active(stage, step_s[row], outputs.numerical_status[row]):
            continue
        source_position = position_m
        source_velocity = velocity_m_s
        source_charge = charge_number
        if stage == 4:
            source_position = outputs.position_m
            source_velocity = outputs.velocity_m_s
            source_charge = outputs.charge_number
        charge = source_charge[row]
        fraction = 0.5 if stage == 1 or stage == 2 else 1.0
        elapsed = fraction * step_s[row]
        for axis in range(2):
            position = source_position[row, axis]
            velocity = source_velocity[row, axis]
            if 0 < stage < 4:
                previous_velocity = velocity_m_s[row, axis]
                if stage > 1:
                    previous_velocity = workspace.stage_velocity_m_s[stage - 2, row, axis]
                position = _rk4_stage_component(position, previous_velocity, elapsed)
                velocity = _rk4_stage_component(
                    velocity, workspace.acceleration_m_s2[stage - 1, row, axis], elapsed
                )
            workspace.position_m[count, axis] = position
            workspace.velocity_m_s[count, axis] = velocity
        if 0 < stage < 4:
            charge = _rk4_stage_component(
                charge, workspace.charge_rate_number_s[stage - 1, row], elapsed
            )
        workspace.charge_number[count] = charge
        if not _rk4_state_is_finite(
            workspace.position_m[count, 0],
            workspace.position_m[count, 1],
            workspace.velocity_m_s[count, 0],
            workspace.velocity_m_s[count, 1],
            charge,
        ):
            outputs.numerical_status[row] = INTEGRATOR_NUMERICAL_FAILURE
            continue
        if 0 < stage < 4:
            for axis in range(2):
                workspace.stage_velocity_m_s[stage - 1, row, axis] = workspace.velocity_m_s[
                    count, axis
                ]
        workspace.active_rows[count] = row
        workspace.particle_index[count] = particle_index[row]
        workspace.incoming_status[count] = NUMERICAL_STATUS_OK
        workspace.time_s[count] = start_time_s[row] if stage == 0 else start_time_s[row] + elapsed
        count += 1
    return count


@njit(cache=True, fastmath=False, parallel=False, nogil=True, error_model="numpy")
def _collect_rk4_stage_into(
    stage: int,
    count: int,
    values: tuple[FloatArray, FloatArray, BoolArray, BoolArray, UInt8Array, Int64Array],
    workspace: Rk4AdvanceWorkspace,
    outputs: _Rk4AdvanceOutputs,
) -> None:
    acceleration, charge_rate, support, applicable, status, cells = values
    for selected in range(count):
        if not bool(
            np.isfinite(acceleration[selected, 0])
            & np.isfinite(acceleration[selected, 1])
            & np.isfinite(charge_rate[selected])
        ):
            raise ValueError("stage evaluator returned a non-finite derivative")
        row = workspace.active_rows[selected]
        outputs.numerical_status[row] = status[selected]
        outputs.support_inside[row] &= support[selected]
        outputs.applicability_inside[row] &= applicable[selected]
        if stage < 4:
            workspace.acceleration_m_s2[stage, row, 0] = acceleration[selected, 0]
            workspace.acceleration_m_s2[stage, row, 1] = acceleration[selected, 1]
            workspace.charge_rate_number_s[stage, row] = charge_rate[selected]
        else:
            outputs.field_cell_id[row] = cells[selected]


@njit(cache=True, fastmath=False, parallel=False, nogil=True, error_model="numpy")
def _rk4_dense_vector_pair_row_into(
    row: int,
    start: FloatArray,
    end: FloatArray,
    first_rate: FloatArray,
    fourth_rate: FloatArray,
    step: float,
    inner: FloatArray,
    status: UInt8Array,
) -> None:
    one_third_step = step / 3.0
    finite = True
    for axis in range(2):
        first = start[row, axis] + one_third_step * first_rate[row, axis]
        second = end[row, axis] - one_third_step * fourth_rate[row, axis]
        inner[row, 0, axis] = first
        inner[row, 1, axis] = second
        finite = finite & np.isfinite(first) & np.isfinite(second)
    if not finite and status[row] == NUMERICAL_STATUS_OK:
        status[row] = INTEGRATOR_NUMERICAL_FAILURE
    if status[row] != NUMERICAL_STATUS_OK:
        for axis in range(2):
            inner[row, 0, axis] = start[row, axis]
            inner[row, 1, axis] = end[row, axis]


@njit(cache=True, fastmath=False, parallel=False, nogil=True, error_model="numpy")
def _rk4_dense_charge_pair_row_into(
    row: int,
    start: FloatArray,
    end: FloatArray,
    first_rate: FloatArray,
    fourth_rate: FloatArray,
    step: float,
    inner: FloatArray,
    status: UInt8Array,
) -> None:
    one_third_step = step / 3.0
    first = start[row] + one_third_step * first_rate[row]
    second = end[row] - one_third_step * fourth_rate[row]
    inner[row, 0] = first
    inner[row, 1] = second
    if not bool(np.isfinite(first) & np.isfinite(second)) and status[row] == NUMERICAL_STATUS_OK:
        status[row] = INTEGRATOR_NUMERICAL_FAILURE
    if status[row] != NUMERICAL_STATUS_OK:
        inner[row, 0] = start[row]
        inner[row, 1] = end[row]


@njit(cache=True, fastmath=False, parallel=False, nogil=True, error_model="numpy")
def _finish_rk4_advance_into(
    position_m: FloatArray,
    velocity_m_s: FloatArray,
    charge_number: FloatArray,
    step_s: FloatArray,
    workspace: Rk4AdvanceWorkspace,
    outputs: _Rk4AdvanceOutputs,
) -> None:
    for row in range(step_s.size):
        if not (step_s[row] > 0.0):
            continue
        if outputs.numerical_status[row] == NUMERICAL_STATUS_OK:
            weight = step_s[row] / 6.0
            for axis in range(2):
                workspace.position_m[row, axis] = _rk4_endpoint_component(
                    position_m[row, axis],
                    velocity_m_s[row, axis],
                    workspace.stage_velocity_m_s[0, row, axis],
                    workspace.stage_velocity_m_s[1, row, axis],
                    workspace.stage_velocity_m_s[2, row, axis],
                    weight,
                )
                workspace.velocity_m_s[row, axis] = _rk4_endpoint_component(
                    velocity_m_s[row, axis],
                    workspace.acceleration_m_s2[0, row, axis],
                    workspace.acceleration_m_s2[1, row, axis],
                    workspace.acceleration_m_s2[2, row, axis],
                    workspace.acceleration_m_s2[3, row, axis],
                    weight,
                )
            charge = _rk4_endpoint_component(
                charge_number[row],
                workspace.charge_rate_number_s[0, row],
                workspace.charge_rate_number_s[1, row],
                workspace.charge_rate_number_s[2, row],
                workspace.charge_rate_number_s[3, row],
                weight,
            )
            if _rk4_state_is_finite(
                workspace.position_m[row, 0],
                workspace.position_m[row, 1],
                workspace.velocity_m_s[row, 0],
                workspace.velocity_m_s[row, 1],
                charge,
            ):
                for axis in range(2):
                    outputs.position_m[row, axis] = workspace.position_m[row, axis]
                    outputs.velocity_m_s[row, axis] = workspace.velocity_m_s[row, axis]
                outputs.charge_number[row] = charge
            else:
                outputs.numerical_status[row] = INTEGRATOR_NUMERICAL_FAILURE
        _rk4_dense_vector_pair_row_into(
            row,
            position_m,
            outputs.position_m,
            velocity_m_s,
            outputs.velocity_m_s,
            step_s[row],
            outputs.position_inner_m,
            outputs.numerical_status,
        )
        _rk4_dense_vector_pair_row_into(
            row,
            velocity_m_s,
            outputs.velocity_m_s,
            workspace.acceleration_m_s2[0],
            workspace.acceleration_m_s2[3],
            step_s[row],
            outputs.velocity_inner_m_s,
            outputs.numerical_status,
        )
        _rk4_dense_charge_pair_row_into(
            row,
            charge_number,
            outputs.charge_number,
            workspace.charge_rate_number_s[0],
            workspace.charge_rate_number_s[3],
            step_s[row],
            outputs.charge_inner_number,
            outputs.numerical_status,
        )


# First-class dispatcher identity is process-local; its disk-cache index can hold dead weakrefs.
@njit(cache=False, fastmath=False, parallel=False, nogil=True, error_model="numpy")
def _compiled_rk4_advance_into(
    stage_evaluator: Callable,
    context: object,
    stage_workspace: object,
    particle_index: Int64Array,
    start_time_s: FloatArray,
    step_s: FloatArray,
    position_m: FloatArray,
    velocity_m_s: FloatArray,
    charge_number: FloatArray,
    workspace: Rk4AdvanceWorkspace,
    outputs: _Rk4AdvanceOutputs,
) -> bool:
    """Keep four coupled stages and endpoint validation in one native call."""

    _initialize_rk4_advance_into(position_m, velocity_m_s, charge_number, workspace, outputs)
    endpoint_evaluated = False
    for stage in range(5):
        if stage == 4:
            _finish_rk4_advance_into(
                position_m, velocity_m_s, charge_number, step_s, workspace, outputs
            )
        count = _prepare_rk4_stage_into(
            stage,
            particle_index,
            start_time_s,
            step_s,
            position_m,
            velocity_m_s,
            charge_number,
            workspace,
            outputs,
        )
        if stage == 4:
            endpoint_evaluated = count > 0
        if count:
            values = stage_evaluator(
                context,
                stage_workspace,
                workspace.particle_index[:count],
                workspace.time_s[:count],
                workspace.position_m[:count],
                workspace.velocity_m_s[:count],
                workspace.charge_number[:count],
                workspace.incoming_status[:count],
            )
            _collect_rk4_stage_into(stage, count, values, workspace, outputs)
    return endpoint_evaluated


def _advance_rk4(
    evaluator: StageEvaluator,
    particle_index: Int64Array,
    start_time_s: FloatArray,
    elapsed_s: FloatArray,
    position_m: FloatArray,
    velocity_m_s: FloatArray,
    charge_number: FloatArray,
) -> _AdvanceResult:
    count = particle_index.size
    workspace = getattr(evaluator, "native_rk4_workspace", None)
    if workspace is None:
        workspace = allocate_rk4_advance_workspace(count)
    if workspace.active_rows.size < count:
        raise ValueError("RK4 advance exceeds its workspace capacity")
    outputs = _Rk4AdvanceOutputs(
        np.empty_like(position_m),
        np.empty_like(velocity_m_s),
        np.empty_like(charge_number),
        np.empty(count, dtype=np.bool_),
        np.empty(count, dtype=np.bool_),
        np.empty(count, dtype=np.uint8),
        np.empty(count, dtype=np.int64),
        np.empty((count, 2, 2), dtype=np.float64),
        np.empty((count, 2, 2), dtype=np.float64),
        np.empty((count, 2), dtype=np.float64),
    )
    stage_evaluator = getattr(evaluator, "native_stage_evaluator", None)
    has_field_cell = True
    if stage_evaluator is None:
        adapter = _PythonStageAdapter(evaluator)
        python_advance = getattr(_compiled_rk4_advance_into, "py_func", None)
        assert python_advance is not None
        endpoint_evaluated = python_advance(
            _evaluate_python_dynamics_stage,
            adapter,
            (),
            particle_index,
            start_time_s,
            elapsed_s,
            position_m,
            velocity_m_s,
            charge_number,
            workspace,
            outputs,
        )
        has_field_cell = endpoint_evaluated and adapter.has_field_cell
    else:
        native_context = getattr(evaluator, "native_context", None)
        native_workspace = getattr(evaluator, "native_workspace", None)
        if native_context is None or native_workspace is None:
            raise ValueError("native RK4 stage evaluator lost its prepared bindings")
        has_field_cell = _compiled_rk4_advance_into(
            stage_evaluator,
            native_context,
            native_workspace,
            particle_index,
            start_time_s,
            elapsed_s,
            position_m,
            velocity_m_s,
            charge_number,
            workspace,
            outputs,
        )
    return _AdvanceResult(
        outputs.position_m,
        outputs.velocity_m_s,
        outputs.charge_number,
        outputs.support_inside,
        outputs.applicability_inside,
        outputs.numerical_status,
        outputs.field_cell_id if has_field_cell else None,
        _Rk4DenseControls(
            outputs.position_inner_m, outputs.velocity_inner_m_s, outputs.charge_inner_number
        ),
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
    count: int,
) -> None:
    """Apply the midpoint-frozen update into caller-owned buffers."""

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
