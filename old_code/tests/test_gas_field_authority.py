from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from field_backend_helpers import (
    geometry_provider,
    regular_axes,
    regular_field_provider,
    regular_valid_mask,
)
from test_triangle_derived_strict_v02 import _triangle_runtime

from particle_tracer_unified.core.datamodel import QuantitySeriesND
from particle_tracer_unified.solvers import kernel2d_numba
from particle_tracer_unified.solvers.field_compilation import compile_runtime_backend
from particle_tracer_unified.solvers.integrator_common import (
    _EPSTEIN_DEFAULT_ACCOMMODATION_DELTA,
    DRAG_MODEL_SCHILLER_NAUMANN,
)


def _regular_runtime(
    quantities: dict[str, np.ndarray],
    *,
    valid_mask: np.ndarray | None = None,
) -> SimpleNamespace:
    axes = regular_axes(2)
    support = regular_valid_mask(2) if valid_mask is None else valid_mask
    return SimpleNamespace(
        geometry_provider=geometry_provider(
            axes,
            support,
            sdf=-np.ones(support.shape, dtype=np.float64),
            normal_components=(
                np.zeros(support.shape, dtype=np.float64),
                np.ones(support.shape, dtype=np.float64),
            ),
        ),
        field_provider=regular_field_provider(axes, support, quantities),
        gas=SimpleNamespace(
            density_kgm3=1.2,
            dynamic_viscosity_Pas=1.8e-5,
            temperature=300.0,
        ),
    )


@pytest.mark.parametrize(
    ("quantity", "invalid_value", "semantic_name"),
    [
        ("rho_g", 0.0, "gas_density"),
        ("mu", -1.0, "gas_mu"),
        ("T", np.nan, "gas_temperature"),
    ],
)
def test_compile_rejects_declared_invalid_gas_properties_inside_support(
    quantity: str,
    invalid_value: float,
    semantic_name: str,
) -> None:
    values = np.ones((3, 3), dtype=np.float64)
    values[1, 1] = invalid_value
    runtime = _regular_runtime({quantity: values})

    with pytest.raises(
        ValueError,
        match=rf"Field gas property {semantic_name} \({quantity}\).*finite and > 0",
    ):
        compile_runtime_backend(runtime, 2)


def test_compile_ignores_declared_invalid_gas_values_outside_support() -> None:
    support = regular_valid_mask(2)
    support[0, 0] = False
    density = np.ones((3, 3), dtype=np.float64)
    density[0, 0] = 0.0

    compiled = compile_runtime_backend(
        _regular_runtime({"rho_g": density}, valid_mask=support),
        2,
    )

    assert compiled.gas_density_source == "field:rho_g"
    assert compiled.gas_density[0, 0, 0] == 0.0


def test_compile_rejects_declared_gas_grid_outside_solver_support_shape() -> None:
    runtime = _regular_runtime({"rho_g": np.ones((2, 2), dtype=np.float64)})

    with pytest.raises(ValueError, match="shape must match solver support"):
        compile_runtime_backend(runtime, 2)


def test_absent_gas_properties_are_compiled_from_explicit_gas_defaults() -> None:
    compiled = compile_runtime_backend(_regular_runtime({}), 2)

    assert compiled.gas_density_source == "context:gas"
    assert compiled.gas_mu_source == "context:gas"
    assert compiled.gas_temperature_source == "context:gas"
    np.testing.assert_array_equal(compiled.gas_density, np.full((1, 3, 3), 1.2))
    np.testing.assert_array_equal(compiled.gas_mu, np.full((1, 3, 3), 1.8e-5))
    np.testing.assert_array_equal(compiled.gas_temperature, np.full((1, 3, 3), 300.0))


def test_triangle_compile_uses_the_same_declared_gas_property_authority() -> None:
    runtime, field = _triangle_runtime()
    quantities = dict(field.quantities)
    density = np.asarray(quantities["rho_g"].data, dtype=np.float64).copy()
    density[0, 0] = 0.0
    quantities["rho_g"] = QuantitySeriesND(
        "rho_g",
        "kg/m^3",
        quantities["rho_g"].times,
        density,
    )
    runtime, _field = _triangle_runtime(quantities=quantities)

    with pytest.raises(
        ValueError,
        match=r"Field gas property gas_density \(rho_g\).*finite and > 0",
    ):
        compile_runtime_backend(runtime, 2)


def test_regular_numba_stage_does_not_repair_invalid_compiled_field_values() -> None:
    axis = np.asarray([0.0, 1.0], dtype=np.float64)
    times = np.asarray([0.0], dtype=np.float64)
    shape = (1, 2, 2)
    stage = kernel2d_numba._regular_2d_stage(
        0,
        0.0,
        0.5,
        0.5,
        0.0,
        1.0e-6,
        0.0,
        0.0,
        1.0,
        1.0,
        1000.0,
        1.0,
        4.65e-26,
        DRAG_MODEL_SCHILLER_NAUMANN,
        _EPSTEIN_DEFAULT_ACCOMMODATION_DELTA,
        0.0,
        0.0,
        1.2,
        1.8e-5,
        300.0,
        axis,
        axis,
        times,
        np.zeros(shape, dtype=np.float64),
        np.zeros(shape, dtype=np.float64),
        np.zeros(1, dtype=np.float64),
        np.zeros(1, dtype=np.float64),
        0,
        np.zeros(shape, dtype=np.float64),
        np.full(shape, 1.8e-5, dtype=np.float64),
        np.full(shape, 300.0, dtype=np.float64),
        np.ones((2, 2), dtype=bool),
    )

    assert np.isnan(stage[6])
    assert getattr(kernel2d_numba._regular_2d_stage, "nopython_signatures", ())
