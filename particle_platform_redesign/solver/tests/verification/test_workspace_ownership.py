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


@pytest.mark.parametrize("case_id", ["C01", "C02", "C03", "C07", "C10"])
def test_declared_process_private_workspaces_fit_the_shared_prepare_budget(
    tmp_path: Path, case_id: str
) -> None:
    paths = materialize_microcase(case_id, tmp_path / case_id)
    prepared = _prepare(load_case(paths.case_path))
    plan = prepared.memory_plan
    # This checks actual allocation roots reached by worker serialization,
    # including nested aliases, against the same public check/run memory plan.
    owned = _owned_bytes(tuple(_process_private_arrays(prepared)))
    components = plan.as_manifest()["components"]
    assert isinstance(components, dict)
    budget = sum(
        value
        for name, value in components.items()
        if name.startswith("slab_") or name == "geometry_query_scratch"
    )
    assert owned <= budget
    assert plan.run_peak_bytes == sum(components.values())
