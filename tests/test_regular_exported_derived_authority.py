from __future__ import annotations

from dataclasses import replace

import numpy as np
from test_compiled_force_sampling import _runtime, _series, _varying_field_provider

from particle_tracer_unified.core.datamodel import FieldProviderND, QuantitySeriesND
from particle_tracer_unified.solvers.compiled_backend_types import (
    RegularRectilinearCompiledBackend,
)
from particle_tracer_unified.solvers.field_compilation import compile_runtime_backend
from particle_tracer_unified.solvers.forces import ForceRuntimeParameters


def _grid(value: float, count: int = 1) -> np.ndarray:
    return np.full((count, 3, 3), value, dtype=np.float64)


def _compile(
    extra: dict[str, QuantitySeriesND],
    force: ForceRuntimeParameters,
    *,
    velocity: bool = True,
) -> RegularRectilinearCompiledBackend:
    provider = _varying_field_provider()
    quantities = dict(provider.field.quantities)
    if not velocity:
        quantities.pop("ux")
        quantities.pop("uy")
    quantities.update(extra)
    provider = FieldProviderND(replace(provider.field, quantities=quantities))
    backend = compile_runtime_backend(_runtime(provider), 2, force_runtime=force)
    assert isinstance(backend, RegularRectilinearCompiledBackend)
    return backend


def test_complete_exported_aliases_override_finite_differences() -> None:
    exported = (
        ("dT_dx", "grad_T_x", 11.0),
        ("temperature_gradient_y", "grad_T_y", 12.0),
        ("dE2_dx", "grad_E2_x", 21.0),
        ("grad_esq_y", "grad_E2_y", 22.0),
        ("material_accel_x", "fluid_accel_x", 31.0),
        ("a_fluid_y", "fluid_accel_y", 32.0),
        ("omega_z", "vorticity_z", 41.0),
    )
    backend = _compile(
        {alias: _series(alias, _grid(value)) for alias, _, value in exported},
        ForceRuntimeParameters(
            thermophoresis_enabled=True,
            dielectrophoresis_enabled=True,
            pressure_gradient_enabled=True,
            virtual_mass_enabled=True,
            lift_enabled=True,
        ),
    )
    for _, target, expected in exported:
        np.testing.assert_array_equal(getattr(backend, target), _grid(expected))


def test_incomplete_exported_gradient_uses_atomic_fallback() -> None:
    backend = _compile(
        {"dT_dx": _series("dT_dx", _grid(99.0))},
        ForceRuntimeParameters(thermophoresis_enabled=True),
    )
    np.testing.assert_array_equal(backend.grad_T_x, _grid(20.0))
    np.testing.assert_array_equal(backend.grad_T_y, _grid(0.0))


def test_exported_fluid_acceleration_does_not_require_velocity() -> None:
    exported = {"fluid_acceleration_x": 7.0, "fluid_acceleration_y": -3.0}
    backend = _compile(
        {name: _series(name, _grid(value)) for name, value in exported.items()},
        ForceRuntimeParameters(pressure_gradient_enabled=True),
        velocity=False,
    )
    np.testing.assert_array_equal(backend.fluid_accel_x, _grid(7.0))
    np.testing.assert_array_equal(backend.fluid_accel_y, _grid(-3.0))


def test_exported_derived_quantity_can_define_transient_time_axis() -> None:
    values = np.concatenate((_grid(5.0), _grid(9.0)))
    vorticity = QuantitySeriesND("vorticity_z", "", np.asarray([0.0, 1.0]), values)
    backend = _compile(
        {"vorticity_z": vorticity}, ForceRuntimeParameters(lift_enabled=True)
    )
    np.testing.assert_array_equal(backend.times, [0.0, 1.0])
    np.testing.assert_array_equal(backend.vorticity_z, values)
