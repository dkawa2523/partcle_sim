from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from particle_tracer_unified.core.datamodel import TriangleMeshField2D

from .compiled_backend_types import CompiledRuntimeBackend
from .field_compilation_common import (
    gas_defaults,
)
from .field_compilation_common import (
    gas_property_quantity_names as _gas_property_quantity_names,
)
from .field_compilation_regular import compile_regular_backend
from .field_compilation_triangle import compile_triangle_backend
from .forces import ForceRuntimeParameters


def gas_property_quantity_names(field: Any) -> Mapping[str, str]:
    """Resolve canonical gas properties from the field quantity inventory."""

    return _gas_property_quantity_names(field)


def compile_runtime_backend(
    runtime: Any,
    spatial_dim: int,
    *,
    enable_electric: bool = True,
    force_runtime: ForceRuntimeParameters | None = None,
) -> CompiledRuntimeBackend:
    """Compile native or COMSOL fields into the solver's immutable representation."""

    if runtime.geometry_provider is None:
        raise ValueError("High-fidelity solver requires geometry_provider")
    force = force_runtime or ForceRuntimeParameters()
    defaults = gas_defaults(runtime)
    if runtime.field_provider is not None:
        field = runtime.field_provider.field
        if isinstance(field, TriangleMeshField2D):
            return compile_triangle_backend(
                field,
                spatial_dim,
                enable_electric,
                force,
                defaults,
            )
    return compile_regular_backend(
        runtime,
        spatial_dim,
        enable_electric,
        force,
        defaults,
    )


__all__ = ("compile_runtime_backend", "gas_property_quantity_names")
