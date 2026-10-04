"""Axisymmetric P1 finite-element solve for one reduced plasma closure.

The module is deliberately independent of YAML, HDF5, COMSOL, and the
particle engine.  It owns only the versioned charge closure and its nonlinear
Poisson discretisation.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from typing import Final

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]
type Int64Array = NDArray[np.int64]

ELEMENTARY_CHARGE_C: Final = 1.602176634e-19
VACUUM_PERMITTIVITY_F_M: Final = 8.8541878128e-12

_F8: Final = np.dtype("<f8")
_I8: Final = np.dtype("<i8")
_BARYCENTRIC_QUADRATURE: Final = np.asarray(
    [
        [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
        [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
        [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
    ],
    dtype=_F8,
)


class NonlinearSolveError(RuntimeError):
    """The reduced electrostatic problem did not reach its declared tolerance."""


@dataclass(frozen=True, slots=True)
class ClosureParameters:
    """Physical and explicit guard parameters for the v1 closure."""

    bulk_number_density_m3: float
    electron_temperature_V: float
    positive_ion_mass_kg: float
    bulk_potential_V: float
    sheath_smoothing_V: float
    density_floor_m3: float
    ion_speed_floor_m_s: float


@dataclass(frozen=True, slots=True)
class SolverSettings:
    """Numerical controls whose values are recorded in builder provenance."""

    continuation_ramps: tuple[float, ...]
    max_newton_iterations: int
    relative_residual_tolerance: float
    absolute_residual_tolerance_C: float
    max_linear_iterations: int
    linear_krylov_dimension: int
    linear_relative_tolerance: float
    minimum_line_search_factor: float


@dataclass(frozen=True, slots=True)
class ClosureState:
    """Pointwise state and analytic charge derivative for the v1 closure."""

    electron_number_density_m3: FloatArray
    positive_ion_number_density_m3: FloatArray
    positive_ion_density_derivative_m3_V: FloatArray
    space_charge_density_C_m3: FloatArray
    charge_derivative_C_m3_V: FloatArray
    positive_ion_speed_m_s: FloatArray


@dataclass(frozen=True, slots=True)
class ContinuationStepReport:
    """Convergence summary for one boundary-voltage continuation level."""

    ramp: float
    newton_iterations: int
    linear_iterations: int
    line_search_reductions: int
    absolute_residual_C: float
    relative_residual: float


@dataclass(frozen=True, slots=True)
class SolveReport:
    """Compact numerical evidence retained with the generated fields."""

    model_revision: str
    discretization_revision: str
    node_count: int
    cell_count: int
    continuation: tuple[ContinuationStepReport, ...]
    free_residual_linf_C: float
    relative_residual: float
    total_space_charge_C: float
    boundary_reaction_charge_C: float
    charge_balance_error_C: float
    electron_density_floor_nodes: int
    positive_ion_density_floor_nodes: int
    positive_ion_speed_floor_nodes: int


@dataclass(frozen=True, slots=True)
class ElectrostaticSolution:
    """Nodal reduced-plasma fields and the solve report."""

    potential_V: FloatArray
    electric_field_V_m: FloatArray
    electron_number_density_m3: FloatArray
    positive_ion_number_density_m3: FloatArray
    space_charge_density_C_m3: FloatArray
    positive_ion_speed_m_s: FloatArray
    positive_ion_density_gradient_m4: FloatArray
    report: SolveReport


@dataclass(frozen=True, slots=True)
class _PreparedMesh:
    nodes_m: FloatArray
    connectivity: Int64Array
    basis_gradient_m_inv: FloatArray
    stiffness_local_F: FloatArray
    stiffness_diagonal_F: FloatArray
    quadrature_weight_m3: FloatArray
    recovery_weight_m3: FloatArray


@dataclass(frozen=True, slots=True)
class _Residual:
    values_C: FloatArray
    source_C: FloatArray
    source_jacobian_local_F: FloatArray
    relative: float
    absolute_C: float


def evaluate_closure(potential_V: FloatArray, parameters: ClosureParameters) -> ClosureState:
    """Evaluate the Boltzmann-electron/Bohm-ion closure and ``d rho / d V``."""

    _validate_closure(parameters)
    potential = _finite_f8_vector(potential_V, "potential_V")
    raw = potential - parameters.bulk_potential_V
    gate, gate_derivative = _negative_sheath_gate(raw, parameters.sheath_smoothing_V)
    sheath = raw * gate
    sheath_derivative = gate + raw * gate_derivative

    exponent = sheath / parameters.electron_temperature_V
    if float(np.max(exponent)) > math.log(np.finfo(np.float64).max):
        raise ValueError("electron Boltzmann exponent exceeds the finite float64 range")
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        electron_unfloored = parameters.bulk_number_density_m3 * np.exp(exponent)
        electron = np.maximum(parameters.density_floor_m3, electron_unfloored)
        electron_derivative = np.zeros_like(electron)
        electron_active = electron_unfloored > parameters.density_floor_m3
        electron_derivative[electron_active] = (
            electron[electron_active]
            * sheath_derivative[electron_active]
            / parameters.electron_temperature_V
        )

        bohm_speed = _bohm_speed(parameters)
        sheath_energy = 2.0 * ELEMENTARY_CHARGE_C * sheath / parameters.positive_ion_mass_kg
        speed_squared_unbounded = bohm_speed * bohm_speed - sheath_energy
        speed_squared = np.maximum(
            parameters.ion_speed_floor_m_s**2,
            speed_squared_unbounded,
        )
        ion_speed = np.sqrt(speed_squared)
        ion_unfloored = parameters.bulk_number_density_m3 * bohm_speed / ion_speed
        ion = np.maximum(parameters.density_floor_m3, ion_unfloored)
        ion_derivative = np.zeros_like(ion)
        ion_active = (speed_squared_unbounded > parameters.ion_speed_floor_m_s**2) & (
            ion_unfloored > parameters.density_floor_m3
        )
        ion_derivative[ion_active] = (
            parameters.bulk_number_density_m3
            * bohm_speed
            * ELEMENTARY_CHARGE_C
            * sheath_derivative[ion_active]
            / (parameters.positive_ion_mass_kg * ion_speed[ion_active] ** 3)
        )
        charge = ELEMENTARY_CHARGE_C * (ion - electron)
        charge_derivative = ELEMENTARY_CHARGE_C * (ion_derivative - electron_derivative)
    _require_finite_closure_outputs(
        electron,
        ion,
        ion_derivative,
        charge,
        charge_derivative,
        ion_speed,
    )
    return ClosureState(
        _as_f8(electron),
        _as_f8(ion),
        _as_f8(ion_derivative),
        _as_f8(charge),
        _as_f8(charge_derivative),
        _as_f8(ion_speed),
    )


def solve_axisymmetric_p1(
    nodes_m: FloatArray,
    connectivity: Int64Array,
    dirichlet_node_ids: Int64Array,
    target_potential_V: FloatArray,
    closure: ClosureParameters,
    settings: SolverSettings,
) -> ElectrostaticSolution:
    """Solve the v1 reduced electrostatic model on one RZ P1 mesh."""

    _validate_closure(closure)
    _validate_settings(settings)
    mesh = _prepare_mesh(nodes_m, connectivity)
    fixed, target = _prepare_dirichlet(
        mesh.nodes_m.shape[0], dirichlet_node_ids, target_potential_V
    )
    free = np.flatnonzero(~fixed)
    potential = np.full(mesh.nodes_m.shape[0], closure.bulk_potential_V, dtype=_F8)
    reports: list[ContinuationStepReport] = []
    for ramp in settings.continuation_ramps:
        boundary = closure.bulk_potential_V + ramp * (target - closure.bulk_potential_V)
        potential[fixed] = boundary[fixed]
        step = _newton_step(mesh, potential, fixed, free, closure, settings, ramp)
        reports.append(step)

    final = _residual(mesh, potential, free, closure)
    nodal = evaluate_closure(potential, closure)
    electric = -_recover_nodal_gradient(mesh, potential)
    electric[mesh.nodes_m[:, 0] == 0.0, 0] = 0.0
    potential_gradient = _recover_nodal_gradient(mesh, potential)
    ion_density_gradient = nodal.positive_ion_density_derivative_m3_V[:, None] * potential_gradient
    total_charge = _total_charge(mesh, potential, closure)
    reaction_charge = -float(np.sum(final.values_C[fixed]))
    return ElectrostaticSolution(
        potential_V=_as_f8(potential),
        electric_field_V_m=_as_f8(electric),
        electron_number_density_m3=nodal.electron_number_density_m3,
        positive_ion_number_density_m3=nodal.positive_ion_number_density_m3,
        space_charge_density_C_m3=nodal.space_charge_density_C_m3,
        positive_ion_speed_m_s=nodal.positive_ion_speed_m_s,
        positive_ion_density_gradient_m4=_as_f8(ion_density_gradient),
        report=SolveReport(
            model_revision="boltzmann_bohm_sheath_c2_v1",
            discretization_revision="axisymmetric_p1_newton_gmres_v1",
            node_count=int(mesh.nodes_m.shape[0]),
            cell_count=int(mesh.connectivity.shape[0]),
            continuation=tuple(reports),
            free_residual_linf_C=final.absolute_C,
            relative_residual=final.relative,
            total_space_charge_C=total_charge,
            boundary_reaction_charge_C=reaction_charge,
            charge_balance_error_C=abs(total_charge - reaction_charge),
            electron_density_floor_nodes=int(
                np.count_nonzero(nodal.electron_number_density_m3 == closure.density_floor_m3)
            ),
            positive_ion_density_floor_nodes=int(
                np.count_nonzero(nodal.positive_ion_number_density_m3 == closure.density_floor_m3)
            ),
            positive_ion_speed_floor_nodes=int(
                np.count_nonzero(nodal.positive_ion_speed_m_s == closure.ion_speed_floor_m_s)
            ),
        ),
    )


def recover_axisymmetric_p1_nodal_gradient(
    nodes_m: FloatArray, connectivity: Int64Array, values: FloatArray
) -> FloatArray:
    """Return the mass-lumped nodal gradient of one RZ P1 scalar field."""

    return _recover_nodal_gradient(_prepare_mesh(nodes_m, connectivity), values)


def _recover_nodal_gradient(mesh: _PreparedMesh, values: FloatArray) -> FloatArray:

    nodal_values = _finite_f8_vector(values, "values")
    if nodal_values.shape != (mesh.nodes_m.shape[0],):
        raise ValueError("values length must match the mesh node count")
    cell_gradient = np.einsum(
        "ci,cij->cj",
        nodal_values[mesh.connectivity],
        mesh.basis_gradient_m_inv,
    )
    count = int(mesh.nodes_m.shape[0])
    mass = _scatter(mesh.connectivity, mesh.recovery_weight_m3, count)
    if bool((mass <= 0.0).any()):
        raise ValueError("every mesh node must belong to a positive-volume cell")
    gradient = np.empty((count, 2), dtype=_F8)
    for component in range(2):
        weighted = mesh.recovery_weight_m3 * cell_gradient[:, component, None]
        gradient[:, component] = _scatter(mesh.connectivity, weighted, count) / mass
    gradient[mesh.nodes_m[:, 0] == 0.0, 0] = 0.0
    return gradient


def _newton_step(
    mesh: _PreparedMesh,
    potential: FloatArray,
    fixed: NDArray[np.bool_],
    free: Int64Array,
    closure: ClosureParameters,
    settings: SolverSettings,
    ramp: float,
) -> ContinuationStepReport:
    linear_iterations = 0
    reductions = 0
    residual = _residual(mesh, potential, free, closure)
    if free.size == 0 or _converged(residual, settings):
        return _step_report(ramp, 0, linear_iterations, reductions, residual)
    for iteration in range(1, settings.max_newton_iterations + 1):
        diagonal = mesh.stiffness_diagonal_F - _local_diagonal(
            mesh.connectivity,
            residual.source_jacobian_local_F,
            int(mesh.nodes_m.shape[0]),
        )
        source_jacobian = residual.source_jacobian_local_F

        def operator(delta_free: FloatArray, jacobian: FloatArray = source_jacobian) -> FloatArray:
            delta = np.zeros_like(potential)
            delta[free] = delta_free
            applied = _apply_local(mesh.connectivity, mesh.stiffness_local_F, delta)
            applied -= _apply_local(
                mesh.connectivity,
                jacobian,
                delta,
            )
            return applied[free]

        delta, used = _gmres(
            operator,
            -residual.values_C[free],
            diagonal[free],
            settings,
        )
        linear_iterations += used
        residual, used_reductions = _line_search(
            mesh,
            potential,
            fixed,
            free,
            delta,
            residual,
            closure,
            settings,
            ramp,
        )
        reductions += used_reductions
        if _converged(residual, settings):
            return _step_report(ramp, iteration, linear_iterations, reductions, residual)
    raise NonlinearSolveError(
        "nonlinear solve did not converge at continuation ramp "
        f"{ramp:.17g}: residual={residual.absolute_C:.6e} C, "
        f"relative={residual.relative:.6e}"
    )


def _line_search(
    mesh: _PreparedMesh,
    potential: FloatArray,
    fixed: NDArray[np.bool_],
    free: Int64Array,
    delta: FloatArray,
    previous: _Residual,
    closure: ClosureParameters,
    settings: SolverSettings,
    ramp: float,
) -> tuple[_Residual, int]:
    factor = 1.0
    reductions = 0
    baseline = previous.absolute_C
    while factor >= settings.minimum_line_search_factor:
        candidate = potential.copy()
        candidate[free] += factor * delta
        candidate[fixed] = potential[fixed]
        tested = _residual(mesh, candidate, free, closure)
        if tested.absolute_C < baseline or _converged(tested, settings):
            potential[:] = candidate
            return tested, reductions
        factor *= 0.5
        reductions += 1
    raise NonlinearSolveError(f"Newton line search failed at continuation ramp {ramp:.17g}")


def _gmres(
    operator: Callable[[FloatArray], FloatArray],
    right_hand_side: FloatArray,
    diagonal: FloatArray,
    settings: SolverSettings,
) -> tuple[FloatArray, int]:
    solution = np.zeros_like(right_hand_side)
    scale = np.maximum(np.abs(diagonal), np.finfo(np.float64).tiny)
    scaled_right_hand_side = right_hand_side / scale
    initial_norm = float(np.linalg.norm(scaled_right_hand_side))
    if initial_norm == 0.0:
        return solution, 0
    threshold = settings.linear_relative_tolerance * initial_norm
    used = 0
    while used < settings.max_linear_iterations:
        residual = scaled_right_hand_side - operator(solution) / scale
        beta = float(np.linalg.norm(residual))
        if beta <= threshold:
            return solution, used
        cycle = min(
            settings.linear_krylov_dimension,
            settings.max_linear_iterations - used,
            int(right_hand_side.size),
        )
        basis = np.zeros((right_hand_side.size, cycle + 1), dtype=_F8)
        hessenberg = np.zeros((cycle + 1, cycle), dtype=_F8)
        cosine = np.zeros(cycle, dtype=_F8)
        sine = np.zeros(cycle, dtype=_F8)
        projected = np.zeros(cycle + 1, dtype=_F8)
        basis[:, 0] = residual / beta
        projected[0] = beta
        updated = False
        for column in range(cycle):
            vector = operator(basis[:, column]) / scale
            for row in range(column + 1):
                hessenberg[row, column] = np.dot(basis[:, row], vector)
                vector -= hessenberg[row, column] * basis[:, row]
            next_norm = float(np.linalg.norm(vector))
            hessenberg[column + 1, column] = next_norm
            if next_norm > 0.0:
                basis[:, column + 1] = vector / next_norm
            _apply_givens(hessenberg, cosine, sine, projected, column)
            used += 1
            if abs(float(projected[column + 1])) <= threshold:
                _update_gmres_solution(solution, basis, hessenberg, projected, column + 1)
                true_residual = scaled_right_hand_side - operator(solution) / scale
                if float(np.linalg.norm(true_residual)) <= threshold:
                    return solution, used
                updated = True
                break
            if next_norm == 0.0:
                _update_gmres_solution(solution, basis, hessenberg, projected, column + 1)
                updated = True
                break
        if not updated:
            _update_gmres_solution(solution, basis, hessenberg, projected, cycle)
    final_residual = scaled_right_hand_side - operator(solution) / scale
    if float(np.linalg.norm(final_residual)) <= threshold:
        return solution, used
    raise NonlinearSolveError(
        f"linear solve did not converge in {settings.max_linear_iterations} iterations"
    )


def _apply_givens(
    hessenberg: FloatArray,
    cosine: FloatArray,
    sine: FloatArray,
    projected: FloatArray,
    column: int,
) -> None:
    for row in range(column):
        first = hessenberg[row, column]
        second = hessenberg[row + 1, column]
        hessenberg[row, column] = cosine[row] * first + sine[row] * second
        hessenberg[row + 1, column] = -sine[row] * first + cosine[row] * second
    first = float(hessenberg[column, column])
    second = float(hessenberg[column + 1, column])
    magnitude = math.hypot(first, second)
    if magnitude == 0.0:
        cosine[column] = 1.0
        sine[column] = 0.0
    else:
        cosine[column] = first / magnitude
        sine[column] = second / magnitude
    hessenberg[column, column] = magnitude
    hessenberg[column + 1, column] = 0.0
    value = projected[column]
    projected[column] = cosine[column] * value
    projected[column + 1] = -sine[column] * value


def _update_gmres_solution(
    solution: FloatArray,
    basis: FloatArray,
    hessenberg: FloatArray,
    projected: FloatArray,
    dimension: int,
) -> None:
    try:
        coefficients = np.linalg.solve(hessenberg[:dimension, :dimension], projected[:dimension])
    except np.linalg.LinAlgError as error:
        raise NonlinearSolveError("GMRES Krylov system is singular") from error
    solution += basis[:, :dimension] @ coefficients


def _residual(
    mesh: _PreparedMesh,
    potential: FloatArray,
    free: Int64Array,
    closure: ClosureParameters,
) -> _Residual:
    source, source_jacobian = _source(mesh, potential, closure)
    stiffness = _apply_local(mesh.connectivity, mesh.stiffness_local_F, potential)
    values = stiffness - source
    absolute = _linf(values[free]) if free.size else 0.0
    scale = max(_linf(stiffness), _linf(source), np.finfo(np.float64).tiny)
    return _Residual(values, source, source_jacobian, absolute / scale, absolute)


def _source(
    mesh: _PreparedMesh,
    potential: FloatArray,
    closure: ClosureParameters,
) -> tuple[FloatArray, FloatArray]:
    local_potential = potential[mesh.connectivity]
    quadrature_potential = np.einsum("qi,ci->cq", _BARYCENTRIC_QUADRATURE, local_potential)
    state = evaluate_closure(_as_f8(quadrature_potential.reshape(-1)), closure)
    charge = state.space_charge_density_C_m3.reshape(quadrature_potential.shape)
    derivative = state.charge_derivative_C_m3_V.reshape(quadrature_potential.shape)
    weighted_charge = mesh.quadrature_weight_m3 * charge
    local_source = np.einsum("cq,qi->ci", weighted_charge, _BARYCENTRIC_QUADRATURE)
    local_jacobian = np.einsum(
        "cq,qi,qj->cij",
        mesh.quadrature_weight_m3 * derivative,
        _BARYCENTRIC_QUADRATURE,
        _BARYCENTRIC_QUADRATURE,
    )
    source = _scatter(mesh.connectivity, local_source, int(mesh.nodes_m.shape[0]))
    return source, _as_f8(local_jacobian)


def _total_charge(mesh: _PreparedMesh, potential: FloatArray, closure: ClosureParameters) -> float:
    quadrature_potential = np.einsum(
        "qi,ci->cq", _BARYCENTRIC_QUADRATURE, potential[mesh.connectivity]
    )
    state = evaluate_closure(_as_f8(quadrature_potential.reshape(-1)), closure)
    charge = state.space_charge_density_C_m3.reshape(quadrature_potential.shape)
    return float(np.sum(mesh.quadrature_weight_m3 * charge))


def _prepare_mesh(nodes_m: FloatArray, connectivity: Int64Array) -> _PreparedMesh:
    nodes = _finite_f8_matrix(nodes_m, "nodes_m", 2)
    cells = _i8_matrix(connectivity, "connectivity", 3)
    if nodes.shape[0] < 3 or cells.shape[0] == 0:
        raise ValueError("axisymmetric P1 mesh must contain nodes and cells")
    if bool((nodes[:, 0] < 0.0).any()):
        raise ValueError("axisymmetric P1 mesh requires r >= 0")
    if int(np.min(cells)) < 0 or int(np.max(cells)) >= nodes.shape[0]:
        raise ValueError("connectivity contains an out-of-range node index")
    local = nodes[cells]
    edge1 = local[:, 1] - local[:, 0]
    edge2 = local[:, 2] - local[:, 0]
    twice_area = edge1[:, 0] * edge2[:, 1] - edge1[:, 1] * edge2[:, 0]
    if bool((~np.isfinite(twice_area)).any()) or bool((twice_area <= 0.0).any()):
        raise ValueError("P1 cells must be finite, nondegenerate, and counter-clockwise")
    area = 0.5 * twice_area
    gradient = _basis_gradients(local, twice_area)
    centroid_radius = np.mean(local[:, :, 0], axis=1)
    stiffness = (
        2.0
        * math.pi
        * VACUUM_PERMITTIVITY_F_M
        * area[:, None, None]
        * centroid_radius[:, None, None]
        * np.einsum("cik,cjk->cij", gradient, gradient)
    )
    quadrature_radius = np.einsum("qi,ci->cq", _BARYCENTRIC_QUADRATURE, local[:, :, 0])
    quadrature_weight = 2.0 * math.pi * area[:, None] * quadrature_radius / 3.0
    radius_sum = np.sum(local[:, :, 0], axis=1)
    recovery_weight = area[:, None] * (radius_sum[:, None] + local[:, :, 0]) / 12.0
    diagonal = _local_diagonal(cells, stiffness, int(nodes.shape[0]))
    return _PreparedMesh(
        nodes,
        cells,
        _as_f8(gradient),
        _as_f8(stiffness),
        _as_f8(diagonal),
        _as_f8(quadrature_weight),
        _as_f8(recovery_weight),
    )


def _basis_gradients(local_nodes: FloatArray, twice_area: FloatArray) -> FloatArray:
    gradient = np.empty((local_nodes.shape[0], 3, 2), dtype=_F8)
    gradient[:, 0, 0] = local_nodes[:, 1, 1] - local_nodes[:, 2, 1]
    gradient[:, 0, 1] = local_nodes[:, 2, 0] - local_nodes[:, 1, 0]
    gradient[:, 1, 0] = local_nodes[:, 2, 1] - local_nodes[:, 0, 1]
    gradient[:, 1, 1] = local_nodes[:, 0, 0] - local_nodes[:, 2, 0]
    gradient[:, 2, 0] = local_nodes[:, 0, 1] - local_nodes[:, 1, 1]
    gradient[:, 2, 1] = local_nodes[:, 1, 0] - local_nodes[:, 0, 0]
    gradient /= twice_area[:, None, None]
    return gradient


def _prepare_dirichlet(
    node_count: int,
    node_ids: Int64Array,
    target_potential_V: FloatArray,
) -> tuple[NDArray[np.bool_], FloatArray]:
    ids = np.asarray(node_ids)
    target_values = np.asarray(target_potential_V)
    if ids.dtype.str != _I8.str or ids.ndim != 1 or not ids.flags.c_contiguous:
        raise ValueError("dirichlet_node_ids must be a contiguous <i8 vector")
    if ids.size == 0 or int(np.min(ids)) < 0 or int(np.max(ids)) >= node_count:
        raise ValueError("Dirichlet node IDs must be nonempty and in range")
    if np.unique(ids).size != ids.size:
        raise ValueError("Dirichlet node IDs must be unique")
    values = _finite_f8_vector(target_values, "target_potential_V")
    if values.shape != ids.shape:
        raise ValueError("target potential length must match Dirichlet node IDs")
    fixed = np.zeros(node_count, dtype=np.bool_)
    fixed[ids] = True
    target = np.zeros(node_count, dtype=_F8)
    target[ids] = values
    return fixed, target


def _negative_sheath_gate(raw_V: FloatArray, width_V: float) -> tuple[FloatArray, FloatArray]:
    with np.errstate(over="ignore", invalid="ignore"):
        scaled = -raw_V / width_V
    gate = np.empty_like(raw_V)
    derivative = np.zeros_like(raw_V)
    positive_potential = scaled <= -1.0
    negative_potential = scaled >= 1.0
    transition = ~(positive_potential | negative_potential)
    gate[positive_potential] = 0.0
    gate[negative_potential] = 1.0
    value = scaled[transition]
    gate[transition] = 0.5 + 0.9375 * value - 0.625 * value**3 + 0.1875 * value**5
    derivative[transition] = -0.9375 * (1.0 - value**2) ** 2 / width_V
    return gate, derivative


def _apply_local(
    connectivity: Int64Array, local_matrix: FloatArray, values: FloatArray
) -> FloatArray:
    local_result = np.einsum("cij,cj->ci", local_matrix, values[connectivity])
    return _scatter(connectivity, local_result, int(values.shape[0]))


def _local_diagonal(
    connectivity: Int64Array, local_matrix: FloatArray, node_count: int
) -> FloatArray:
    local = np.diagonal(local_matrix, axis1=1, axis2=2)
    return _scatter(connectivity, local, node_count)


def _scatter(connectivity: Int64Array, local: FloatArray, node_count: int) -> FloatArray:
    result = np.bincount(
        connectivity.reshape(-1),
        weights=local.reshape(-1),
        minlength=node_count,
    )
    return _as_f8(result)


def _converged(residual: _Residual, settings: SolverSettings) -> bool:
    return (
        residual.absolute_C <= settings.absolute_residual_tolerance_C
        or residual.relative <= settings.relative_residual_tolerance
    )


def _step_report(
    ramp: float,
    newton_iterations: int,
    linear_iterations: int,
    reductions: int,
    residual: _Residual,
) -> ContinuationStepReport:
    return ContinuationStepReport(
        ramp,
        newton_iterations,
        linear_iterations,
        reductions,
        residual.absolute_C,
        residual.relative,
    )


def _validate_closure(parameters: ClosureParameters) -> None:
    positive = {
        "bulk_number_density_m3": parameters.bulk_number_density_m3,
        "electron_temperature_V": parameters.electron_temperature_V,
        "positive_ion_mass_kg": parameters.positive_ion_mass_kg,
        "sheath_smoothing_V": parameters.sheath_smoothing_V,
        "density_floor_m3": parameters.density_floor_m3,
        "ion_speed_floor_m_s": parameters.ion_speed_floor_m_s,
    }
    for name, value in positive.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(parameters.bulk_potential_V):
        raise ValueError("bulk_potential_V must be finite")
    if parameters.density_floor_m3 >= parameters.bulk_number_density_m3:
        raise ValueError("density_floor_m3 must be below bulk_number_density_m3")
    bohm_speed = _bohm_speed(parameters)
    if parameters.ion_speed_floor_m_s >= bohm_speed:
        raise ValueError("ion_speed_floor_m_s must be below the bulk Bohm speed")


def _bohm_speed(parameters: ClosureParameters) -> float:
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        squared = (
            np.float64(ELEMENTARY_CHARGE_C)
            * np.float64(parameters.electron_temperature_V)
            / np.float64(parameters.positive_ion_mass_kg)
        )
    if math.isfinite(float(squared)) and squared > 0.0:
        return math.sqrt(float(squared))
    log_speed = 0.5 * (
        math.log(ELEMENTARY_CHARGE_C)
        + math.log(parameters.electron_temperature_V)
        - math.log(parameters.positive_ion_mass_kg)
    )
    if log_speed > math.log(np.finfo(np.float64).max):
        raise ValueError("bulk Bohm speed exceeds the finite float64 range")
    speed = math.exp(log_speed)
    if not math.isfinite(speed) or speed <= 0.0:
        raise ValueError("bulk Bohm speed is not finite and positive")
    return speed


def _require_finite_closure_outputs(*values: FloatArray) -> None:
    if any(not bool(np.isfinite(value).all()) for value in values):
        raise ValueError("closure evaluation produced a non-finite derived quantity")


def _validate_settings(settings: SolverSettings) -> None:
    counts = {
        "max_newton_iterations": settings.max_newton_iterations,
        "max_linear_iterations": settings.max_linear_iterations,
        "linear_krylov_dimension": settings.linear_krylov_dimension,
    }
    for name, value in counts.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    _validate_tolerances(settings)
    _validate_continuation(settings.continuation_ramps)


def _validate_tolerances(settings: SolverSettings) -> None:
    positive = {
        "relative_residual_tolerance": settings.relative_residual_tolerance,
        "absolute_residual_tolerance_C": settings.absolute_residual_tolerance_C,
        "linear_relative_tolerance": settings.linear_relative_tolerance,
        "minimum_line_search_factor": settings.minimum_line_search_factor,
    }
    for name, value in positive.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if settings.relative_residual_tolerance >= 1.0:
        raise ValueError("relative_residual_tolerance must be below one")
    if settings.linear_relative_tolerance >= 1.0:
        raise ValueError("linear_relative_tolerance must be below one")
    if settings.minimum_line_search_factor > 1.0:
        raise ValueError("minimum_line_search_factor must not exceed one")


def _validate_continuation(ramps: tuple[float, ...]) -> None:
    if not ramps or ramps[-1] != 1.0:
        raise ValueError("continuation_ramps must be nonempty and end at 1.0")
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in ramps):
        raise ValueError("continuation_ramps must contain finite values in [0, 1]")
    if any(second <= first for first, second in pairwise(ramps)):
        raise ValueError("continuation_ramps must be strictly increasing")


def _finite_f8_vector(value: object, label: str) -> FloatArray:
    array = np.asarray(value)
    if array.dtype.str != _F8.str or array.ndim != 1 or not array.flags.c_contiguous:
        raise ValueError(f"{label} must be a contiguous <f8 vector")
    if not bool(np.isfinite(array).all()):
        raise ValueError(f"{label} must contain only finite values")
    return array


def _finite_f8_matrix(value: object, label: str, width: int) -> FloatArray:
    array = np.asarray(value)
    if (
        array.dtype.str != _F8.str
        or array.ndim != 2
        or array.shape[1] != width
        or not array.flags.c_contiguous
    ):
        raise ValueError(f"{label} must be a contiguous <f8 matrix with width {width}")
    if not bool(np.isfinite(array).all()):
        raise ValueError(f"{label} must contain only finite values")
    return array


def _i8_matrix(value: object, label: str, width: int) -> Int64Array:
    array = np.asarray(value)
    if (
        array.dtype.str != _I8.str
        or array.ndim != 2
        or array.shape[1] != width
        or not array.flags.c_contiguous
    ):
        raise ValueError(f"{label} must be a contiguous <i8 matrix with width {width}")
    return array


def _as_f8(value: object) -> FloatArray:
    return np.ascontiguousarray(value, dtype=_F8)


def _linf(value: FloatArray) -> float:
    return float(np.max(np.abs(value))) if value.size else 0.0


__all__ = [
    "ClosureParameters",
    "ClosureState",
    "ContinuationStepReport",
    "ElectrostaticSolution",
    "NonlinearSolveError",
    "SolveReport",
    "SolverSettings",
    "evaluate_closure",
    "recover_axisymmetric_p1_nodal_gradient",
    "solve_axisymmetric_p1",
]
