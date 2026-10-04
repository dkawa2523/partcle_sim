from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from particle_tracer_unified.domain import StageFields
from particle_tracer_unified.solvers import (
    _force_evaluators,
    _force_pipeline,
    force_runtime,
)
from particle_tracer_unified.solvers._force_evaluators import _cm_factor_real
from particle_tracer_unified.solvers.force_runtime import (
    ForceBatchState,
    ForceBatchStatic,
    ForcePipeline,
    evaluate_force_pipeline,
)
from particle_tracer_unified.solvers.forces.runtime import ForceRuntimeParameters

_EPS0_F_M = 8.8541878128e-12


def test_force_runtime_facade_directly_reexports_owner_objects() -> None:
    assert force_runtime.ForceBatchStatic is _force_pipeline.ForceBatchStatic
    assert force_runtime.ForceBatchState is _force_pipeline.ForceBatchState
    assert force_runtime.ForcePipeline is _force_pipeline.ForcePipeline
    assert force_runtime.build_force_pipeline is _force_pipeline.build_force_pipeline
    assert force_runtime.FORCE_EVALUATORS is _force_evaluators.FORCE_EVALUATORS
    assert (
        force_runtime.evaluate_force_pipeline
        is _force_evaluators.evaluate_force_pipeline
    )


def _reference_cm_factor_real(
    particle_rel_permittivity: float,
    medium_rel_permittivity: float,
    particle_conductivity_sm: float,
    medium_conductivity_sm: float,
    frequency_hz: float,
) -> float:
    if frequency_hz == 0.0:
        return (particle_rel_permittivity - medium_rel_permittivity) / (
            particle_rel_permittivity + 2.0 * medium_rel_permittivity
        )
    omega = 2.0 * math.pi * frequency_hz
    particle = complex(
        particle_rel_permittivity,
        -particle_conductivity_sm / (omega * _EPS0_F_M),
    )
    medium = complex(
        medium_rel_permittivity,
        -medium_conductivity_sm / (omega * _EPS0_F_M),
    )
    return float(((particle - medium) / (particle + 2.0 * medium)).real)


@pytest.mark.parametrize(
    ("inputs", "expected_hex"),
    [
        ((3.9, 1.0, 0.2, 0.1, 13.56e6), "0x1.0007d68c54835p-2"),
        ((2.5, 4.0, 0.0, 0.0, 0.0), "-0x1.2492492492492p-3"),
        ((80.0, 2.0, 1.0e-12, 4.0e-4, 1.0), "-0x1.ffffffdef346cp-2"),
        ((1.0, 1.0, 0.5, 0.5, 1.0e9), "-0x0.0p+0"),
        ((10.0, 3.0, 0.4, 0.02, 2.45e9), "0x1.d1118a6e3e30bp-2"),
    ],
)
def test_ac_cm_factor_preserves_snapshot_bits(
    inputs: tuple[float, float, float, float, float],
    expected_hex: str,
) -> None:
    assert _cm_factor_real(*inputs).hex() == expected_hex


@given(
    particle_eps=st.floats(0.1, 100.0, allow_nan=False, allow_infinity=False),
    medium_eps=st.floats(0.1, 100.0, allow_nan=False, allow_infinity=False),
    particle_sigma=st.floats(0.0, 10.0, allow_nan=False, allow_infinity=False),
    medium_sigma=st.floats(0.0, 10.0, allow_nan=False, allow_infinity=False),
    frequency=st.floats(1.0, 1.0e12, allow_nan=False, allow_infinity=False),
)
def test_ac_cm_factor_matches_complex_permittivity_contract(
    particle_eps: float,
    medium_eps: float,
    particle_sigma: float,
    medium_sigma: float,
    frequency: float,
) -> None:
    result = _cm_factor_real(
        particle_eps,
        medium_eps,
        particle_sigma,
        medium_sigma,
        frequency,
    )

    assert isinstance(result, float)
    assert np.isfinite(result)
    assert result == _reference_cm_factor_real(
        particle_eps,
        medium_eps,
        particle_sigma,
        medium_sigma,
        frequency,
    )


def test_cm_factor_known_limits() -> None:
    assert _cm_factor_real(4.0, 4.0, 0.25, 0.25, 13.56e6) == 0.0
    assert _cm_factor_real(7.0, 2.0, 0.0, 0.0, 0.0) == pytest.approx(5.0 / 11.0)
    assert _cm_factor_real(7.0, 2.0, 0.0, 0.0, 1.0e9) == pytest.approx(5.0 / 11.0)


def test_ac_dep_evaluation_preserves_batch_shape_dtype_and_finiteness() -> None:
    count = 3
    out = np.zeros((count, 2), dtype=np.float64)
    static = ForceBatchStatic(
        particle_diameter=np.asarray([1.0e-6, 2.0e-6, 3.0e-6], dtype=np.float32),
        particle_density=np.ones(count, dtype=np.float32),
        particle_mass=np.asarray([1.0e-15, 3.0e-15, 8.0e-15], dtype=np.float32),
        dep_particle_rel_permittivity=np.asarray([3.9, np.nan, 8.0], dtype=np.float32),
        thermophoretic_coeff=np.full(count, np.nan, dtype=np.float32),
    )
    state = ForceBatchState(velocity=np.zeros((count, 2), dtype=np.float32))
    fields = StageFields(
        points_m=np.zeros((count, 2), dtype=np.float64),
        time_s=0.0,
        values={
            "electric_magnitude_squared_gradient": np.asarray(
                [[1.0, -2.0], [3.0, 4.0], [-5.0, 6.0]], dtype=np.float32
            )
        },
        supported=np.ones(count, dtype=bool),
    )
    plan = ForcePipeline(
        evaluator_names=("dielectrophoresis",),
        params=ForceRuntimeParameters(
            dielectrophoresis_enabled=True,
            dep_medium_rel_permittivity=1.2,
            dep_particle_rel_permittivity=4.5,
            dep_medium_conductivity_Sm=0.03,
            dep_particle_conductivity_Sm=0.2,
            dep_frequency_Hz=13.56e6,
        ),
    )

    result = evaluate_force_pipeline(out, static, state, fields, plan)

    assert result is out
    assert result.shape == (count, 2)
    assert result.dtype == np.dtype(np.float64)
    assert np.all(np.isfinite(result))
