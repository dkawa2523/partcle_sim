from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from particle_tracer_unified.core.field_backend import _validated_sample_input
from particle_tracer_unified.core.field_sampling import VALID_MASK_STATUS_CLEAN
from particle_tracer_unified.domain import FieldRequest, StageFields

from .base_field_sampling import (
    sample_compiled_electric_vectors,
    sample_compiled_flow_vectors,
    sample_compiled_gas_properties_vectors,
    sample_compiled_valid_mask_statuses,
)
from .compiled_backend_types import (
    CompiledRuntimeBackend,
    TriangleMesh2DCompiledBackend,
)

FLOW_VELOCITY = "flow_velocity"
ELECTRIC_FIELD = "electric_field"
GAS_DENSITY = "gas_density"
DYNAMIC_VISCOSITY = "dynamic_viscosity"
TEMPERATURE = "temperature"
VALID_MASK_STATUS = "valid_mask_status"

SUPPORTED_QUANTITIES = frozenset(
    {
        FLOW_VELOCITY,
        ELECTRIC_FIELD,
        GAS_DENSITY,
        DYNAMIC_VISCOSITY,
        TEMPERATURE,
        VALID_MASK_STATUS,
    }
)


def _validate_sample_request(
    points_m: np.ndarray,
    time_s: float,
    request: FieldRequest,
    spatial_dim: int,
) -> tuple[np.ndarray, float, tuple[str, ...]]:
    points, time_value = _validated_sample_input(points_m, time_s, spatial_dim)
    quantities = request.quantities
    unknown = tuple(name for name in quantities if name not in SUPPORTED_QUANTITIES)
    if unknown:
        allowed = ", ".join(sorted(SUPPORTED_QUANTITIES))
        raise KeyError(
            f"unsupported field quantities {unknown!r}; supported: {allowed}"
        )
    return points, time_value, quantities


def _sample_vector_values(
    backend: CompiledRuntimeBackend,
    spatial_dim: int,
    time_s: float,
    points_m: np.ndarray,
    quantities: tuple[str, ...],
    values: dict[str, np.ndarray],
    *,
    strict: bool,
) -> int:
    call_count = 0
    if FLOW_VELOCITY in quantities:
        values[FLOW_VELOCITY] = sample_compiled_flow_vectors(
            backend,
            spatial_dim,
            time_s,
            points_m,
        )
        call_count += 1

    if ELECTRIC_FIELD in quantities:
        electric = sample_compiled_electric_vectors(
            backend,
            spatial_dim,
            time_s,
            points_m,
        )
        call_count += 1
        if electric is None:
            if strict:
                raise ValueError("electric_field was requested but is unavailable")
        else:
            values[ELECTRIC_FIELD] = electric
    return call_count


def _sample_gas_values(
    backend: CompiledRuntimeBackend,
    spatial_dim: int,
    time_s: float,
    points_m: np.ndarray,
    quantities: tuple[str, ...],
    supported: np.ndarray,
    values: dict[str, np.ndarray],
    *,
    fallback_density_kgm3: float,
    fallback_dynamic_viscosity_Pas: float,
    fallback_temperature_K: float,
) -> int:
    gas_names = {
        GAS_DENSITY,
        DYNAMIC_VISCOSITY,
        TEMPERATURE,
    }.intersection(quantities)
    if not gas_names:
        return 0

    density, viscosity, temperature = sample_compiled_gas_properties_vectors(
        backend,
        spatial_dim,
        time_s,
        points_m,
        fallback_density_kgm3=fallback_density_kgm3,
        fallback_mu_pas=fallback_dynamic_viscosity_Pas,
        fallback_temperature_K=fallback_temperature_K,
    )
    if GAS_DENSITY in gas_names:
        values[GAS_DENSITY] = density
    if DYNAMIC_VISCOSITY in gas_names:
        values[DYNAMIC_VISCOSITY] = viscosity
    if TEMPERATURE in gas_names:
        values[TEMPERATURE] = temperature

    for name in gas_names:
        requested = np.asarray(values[name], dtype=np.float64)
        invalid_supported = supported & (~np.isfinite(requested) | (requested <= 0.0))
        if np.any(invalid_supported):
            raise ValueError(
                f"{name} was requested but neither the field nor an explicit gas "
                "fallback provides a positive finite value"
            )
    return 1


def _validate_available_values(
    quantities: tuple[str, ...],
    values: dict[str, np.ndarray],
    *,
    strict: bool,
) -> None:
    missing = tuple(name for name in quantities if name not in values)
    if missing and strict:
        raise ValueError(f"requested field quantities are unavailable: {missing!r}")


def _build_stage_fields(
    backend: CompiledRuntimeBackend,
    points_m: np.ndarray,
    time_s: float,
    values: dict[str, np.ndarray],
    supported: np.ndarray,
    statuses: np.ndarray,
    call_count: int,
) -> StageFields:
    return StageFields(
        points_m=points_m,
        time_s=time_s,
        values=values,
        supported=supported,
        metadata={
            "backend_kind": str(getattr(backend, "backend_kind", "")),
            "interpolation": "linear",
            "sample_call_count": int(call_count),
            "sample_point_count": int(points_m.shape[0]) * int(call_count),
            "valid_mask_status": statuses,
        },
    )


@dataclass(frozen=True, slots=True)
class CompiledSamplingBackend:
    """Semantic adapter over the existing compiled regular/triangle backend.

    The compiled arrays remain the hot-loop ABI.  This adapter is the single
    domain-facing representation and deliberately exposes fields, not forces:
    quantities that depend on particle mass, charge, or velocity are evaluated
    by the force pipeline after sampling.
    """

    compiled: CompiledRuntimeBackend
    spatial_dim: int
    fallback_density_kgm3: float
    fallback_dynamic_viscosity_Pas: float
    fallback_temperature_K: float
    strict: bool = True

    def __post_init__(self) -> None:
        backend = self.compiled
        dim = int(self.spatial_dim)
        if dim not in (2, 3):
            raise ValueError("spatial_dim must be 2 or 3")
        backend_dim = (
            2
            if isinstance(backend, TriangleMesh2DCompiledBackend)
            else len(backend.axes)
        )
        if backend_dim != dim:
            raise ValueError(
                f"compiled backend dimension {backend_dim} does not match "
                f"spatial_dim={dim}"
            )
        for name, value in (
            ("fallback_density_kgm3", self.fallback_density_kgm3),
            ("fallback_dynamic_viscosity_Pas", self.fallback_dynamic_viscosity_Pas),
            ("fallback_temperature_K", self.fallback_temperature_K),
        ):
            number = float(value)
            # NaN is the deliberate sentinel for "no fallback supplied".  It
            # is valid until that semantic gas quantity is actually requested;
            # infinities and non-positive physical values are never valid.
            if np.isinf(number) or (np.isfinite(number) and number <= 0.0):
                raise ValueError(
                    f"{name} must be positive or NaN when no fallback is supplied"
                )
        object.__setattr__(self, "spatial_dim", dim)

    @property
    def backend(self) -> CompiledRuntimeBackend:
        return self.compiled

    def sample(
        self,
        points_m: np.ndarray,
        time_s: float,
        request: FieldRequest,
    ) -> StageFields:
        points, time_value, quantities = _validate_sample_request(
            points_m,
            time_s,
            request,
            self.spatial_dim,
        )
        values: dict[str, np.ndarray] = {}
        statuses = sample_compiled_valid_mask_statuses(self.backend, points)
        supported = statuses == np.uint8(VALID_MASK_STATUS_CLEAN)
        if VALID_MASK_STATUS in quantities:
            values[VALID_MASK_STATUS] = statuses

        call_count = 1 + _sample_vector_values(
            self.backend,
            self.spatial_dim,
            time_value,
            points,
            quantities,
            values,
            strict=self.strict,
        )
        call_count += _sample_gas_values(
            self.backend,
            self.spatial_dim,
            time_value,
            points,
            quantities,
            supported,
            values,
            fallback_density_kgm3=float(self.fallback_density_kgm3),
            fallback_dynamic_viscosity_Pas=float(self.fallback_dynamic_viscosity_Pas),
            fallback_temperature_K=float(self.fallback_temperature_K),
        )
        _validate_available_values(quantities, values, strict=self.strict)
        return _build_stage_fields(
            self.backend,
            points,
            time_value,
            values,
            supported,
            statuses,
            call_count,
        )


__all__ = (
    "DYNAMIC_VISCOSITY",
    "ELECTRIC_FIELD",
    "FLOW_VELOCITY",
    "GAS_DENSITY",
    "SUPPORTED_QUANTITIES",
    "TEMPERATURE",
    "VALID_MASK_STATUS",
    "CompiledSamplingBackend",
)
