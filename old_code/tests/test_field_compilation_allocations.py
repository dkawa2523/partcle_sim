from __future__ import annotations

import gc
import tracemalloc
from types import SimpleNamespace

import numpy as np
from field_backend_helpers import geometry_provider, regular_field_provider

from particle_tracer_unified.solvers.compiled_backend_types import (
    RegularRectilinearCompiledBackend,
)
from particle_tracer_unified.solvers.field_compilation import compile_runtime_backend
from particle_tracer_unified.solvers.forces import ForceRuntimeParameters


def test_regular_compilation_does_not_allocate_a_second_full_backend() -> None:
    axis = np.linspace(0.0, 1.0, 21, dtype=np.float64)
    axes = (axis, axis, axis)
    shape = (axis.size,) * 3
    valid_mask = np.ones(shape, dtype=bool)
    velocity = np.broadcast_to(axis[:, None, None], shape).copy()
    runtime = SimpleNamespace(
        geometry_provider=geometry_provider(
            axes,
            valid_mask,
            sdf=-np.ones(shape, dtype=np.float64),
            normal_components=tuple(np.zeros(shape, dtype=np.float64) for _ in axes),
        ),
        field_provider=regular_field_provider(
            axes,
            valid_mask,
            quantities={
                "ux": velocity,
                "uy": 2.0 * velocity,
                "uz": -velocity,
            },
        ),
        gas=SimpleNamespace(
            density_kgm3=1.2,
            dynamic_viscosity_Pas=1.8e-5,
            temperature=300.0,
        ),
    )

    gc.collect()
    tracemalloc.start()
    backend = compile_runtime_backend(
        runtime,
        3,
        enable_electric=False,
        force_runtime=ForceRuntimeParameters(),
    )
    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert isinstance(backend, RegularRectilinearCompiledBackend)
    assert backend.uz is not None
    np.testing.assert_array_equal(backend.ux[0], velocity)
    np.testing.assert_array_equal(backend.uy[0], 2.0 * velocity)
    np.testing.assert_array_equal(backend.uz[0], -velocity)
    transient_bytes = peak_bytes - current_bytes
    assert transient_bytes <= 4 * velocity.nbytes + 32_768
