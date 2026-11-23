"""Concrete ownership checks for the finite CPU particle and scratch banks."""

from pathlib import Path

import numpy as np
import pytest

from chamber_particles import load_case
from chamber_particles.cpu import ParticleState
from chamber_particles.engine import _prepare
from chamber_particles.execution import _process_private_arrays
from chamber_particles.integrators import (
    RELAXATION_ADVANCE_WORK_BYTES_PER_PARTICLE,
    RK4_ADVANCE_WORK_BYTES_PER_PARTICLE,
    STAGE_ADVANCE_WORK_BYTES_PER_PARTICLE,
    allocate_relaxation_advance_workspace,
    allocate_rk4_advance_workspace,
    allocate_stage_advance_workspace,
)
from chamber_particles.transport import (
    _allocate_brownian_pending_columns,
    _brownian_leaf_increments,
    _BrownianAdaptiveWorkspace,
)
from tests.scenarios.test_brownian_run import _brownian_case
from tests.scenarios.test_force_coupled_run import _continuous_charge_case
from tests.scenarios.test_stokes_cunningham_run import _local_applicability_case
from tests.verification.microcases import materialize_microcase


def _owned_bytes(arrays: object) -> int:
    """Count allocation roots rather than double-counting borrowed views."""
    from chamber_particles.execution import process_array_views

    roots: dict[int, np.ndarray] = {}
    for array in process_array_views(arrays):
        root = array
        while isinstance(root.base, np.ndarray):
            root = root.base
        roots[id(root)] = root
    return sum(array.nbytes for array in roots.values())


@pytest.mark.parametrize("count", [0, 1, 13])
def test_particle_and_advance_bytes_cover_their_concrete_columns(count: int) -> None:
    state = ParticleState.allocate(count)
    assert ParticleState.bytes_for(count) == _owned_bytes(state)
    for allocator, per_row in (
        (allocate_stage_advance_workspace, STAGE_ADVANCE_WORK_BYTES_PER_PARTICLE),
        (allocate_rk4_advance_workspace, RK4_ADVANCE_WORK_BYTES_PER_PARTICLE),
        (allocate_relaxation_advance_workspace, RELAXATION_ADVANCE_WORK_BYTES_PER_PARTICLE),
    ):
        assert _owned_bytes(allocator(count)) == count * per_row


@pytest.mark.parametrize(
    "case_id",
    ["C01", "C02", "C03", "C07", "C08", "C10", "local-field-ranges", "coupled-exponential-wall"],
)
def test_declared_process_private_workspaces_fit_the_shared_prepare_budget(
    tmp_path: Path, case_id: str
) -> None:
    if case_id == "local-field-ranges":
        case_path = _local_applicability_case(tmp_path / case_id, remote_velocity_m_s=100.0)
    elif case_id == "coupled-exponential-wall":
        case_path = _continuous_charge_case(tmp_path / case_id, integrator="exponential_midpoint")
    else:
        case_path = materialize_microcase(case_id, tmp_path / case_id).case_path
    run = _prepare(load_case(case_path))
    prepared = run.transport
    if case_id == "local-field-ranges":
        assert prepared.dynamics.field_range_workspace is not None
        assert prepared.dynamics.field_range_workspace.range_available.size > 0
    plan = run.memory_plan
    # This checks actual allocation roots reached by worker serialization,
    # including nested aliases, against the same public check/run memory plan.
    owned = _owned_bytes(tuple(_process_private_arrays(prepared)))
    components = plan.as_manifest()["components"]
    assert isinstance(components, dict)
    if case_id == "coupled-exponential-wall":
        assert prepared.capabilities.curved_events
        assert prepared.capabilities.path_kind == "exponential_midpoint_reintegrated"
        assert components["slab_certificate_work"] > 0
        assert components["slab_event_work"] > 0
        assert components["slab_proposal_replay_work"] > 0
    budget = sum(
        value
        for name, value in components.items()
        if name.startswith("slab_") or name == "geometry_query_scratch"
    )
    assert owned <= budget
    assert plan.run_peak_bytes == sum(components.values())


@pytest.mark.parametrize("count", [1, 13])
@pytest.mark.parametrize(("base_depth", "max_depth"), [(0, 0), (3, 3), (0, 8), (3, 8), (10, 10)])
def test_dynamic_brownian_tree_and_pending_banks_fit_the_declared_reserve(
    tmp_path: Path, count: int, base_depth: int, max_depth: int
) -> None:
    case_path = _brownian_case(
        tmp_path / "brownian",
        particle_count=count,
        end_s=0.4,
        dt_s=0.2,
        tree_depth=base_depth,
        adaptive_max_depth=max_depth,
        frame_times=None,
    )
    plan = _prepare(load_case(case_path)).memory_plan
    capacity = plan.slab_particles
    leaf = _brownian_leaf_increments(
        seed=1,
        particle_id=np.arange(capacity, dtype="<i8"),
        macro_interval=0,
        root_interval=np.zeros(capacity, dtype="<i8"),
        drag_rate_s_inv=np.ones(capacity, dtype="<f8"),
        thermal_velocity_variance_m2_s2=np.ones(capacity, dtype="<f8"),
        root_duration_s=np.ones(capacity, dtype="<f8"),
        tree_depth=base_depth,
        numerically_valid=np.ones(capacity, dtype=np.bool_),
    )
    # These banks coexist during a restarted root wave. The cursor's borrowed
    # numeric inputs are counted conservatively; aliases count only once.
    owned = _owned_bytes(
        (
            leaf,
            _BrownianAdaptiveWorkspace.allocate(capacity, max_depth - base_depth),
            _allocate_brownian_pending_columns(capacity),
            _allocate_brownian_pending_columns(capacity),
            np.zeros(1, dtype="<i8"),
        )
    )
    assert 0 < owned <= plan.components["slab_stochastic_tree_work"]
