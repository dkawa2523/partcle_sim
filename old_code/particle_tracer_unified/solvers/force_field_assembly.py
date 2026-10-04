"""Public dispatch and particle validation for compiled force evaluation."""

from __future__ import annotations

import numpy as np

from particle_tracer_unified.core.field_sampling import (
    VALID_MASK_STATUS_CLEAN,
    VALID_MASK_STATUS_HARD_INVALID,
)
from particle_tracer_unified.domain import StageFields

from ._force_field_regular import _regular_force_fields
from ._force_field_triangle import _triangle_force_fields
from ._particle_geometry import physical_sphere_diameter_m
from .base_field_sampling import sample_compiled_valid_mask_statuses
from .compiled_backend_types import (
    CompiledRuntimeBackend,
    TriangleMesh2DCompiledBackend,
)
from .force_runtime import (
    ForceBatchState,
    ForceBatchStatic,
    build_force_pipeline,
    evaluate_force_pipeline,
)
from .forces import ForceRuntimeParameters


def _validated_stage_points(
    spatial_dim: int,
    t_eval: float,
    positions: np.ndarray,
    base_fields: StageFields | None,
) -> tuple[int, np.ndarray]:
    dim = int(spatial_dim)
    points = np.asarray(positions, dtype=np.float64)
    if dim not in (2, 3) or points.ndim != 2 or points.shape[1] != dim:
        raise ValueError(f"positions must have shape (particle, {dim})")
    if not np.all(np.isfinite(points)):
        raise ValueError("positions must contain only finite coordinates")
    if base_fields is None:
        return dim, points
    if base_fields.points_m.shape != points.shape or not np.array_equal(
        base_fields.points_m,
        points,
    ):
        raise ValueError("base_fields points must exactly match requested positions")
    if float(base_fields.time_s) != float(t_eval):
        raise ValueError("base_fields time must exactly match requested time")
    return dim, points


def _sample_backend_force_fields(
    compiled: CompiledRuntimeBackend,
    spatial_dim: int,
    t_eval: float,
    points: np.ndarray,
    *,
    params: ForceRuntimeParameters,
    include_electric: bool,
    flow_velocity: np.ndarray | None,
    fallback_density_kgm3: float,
    fallback_mu_pas: float,
    fallback_temperature_K: float,
    base_fields: StageFields | None,
) -> dict[str, np.ndarray]:
    if isinstance(compiled, TriangleMesh2DCompiledBackend):
        if int(spatial_dim) != 2:
            raise ValueError("triangle mesh force sampling is two-dimensional")
        return _triangle_force_fields(
            compiled,
            float(t_eval),
            points,
            params=params,
            include_electric=bool(include_electric),
            flow_velocity=flow_velocity,
            fallback_density_kgm3=float(fallback_density_kgm3),
            fallback_mu_pas=float(fallback_mu_pas),
            fallback_temperature_K=float(fallback_temperature_K),
            base_fields=base_fields,
        )
    if len(compiled.axes) != int(spatial_dim):
        raise ValueError("compiled backend and requested dimension differ")
    return _regular_force_fields(
        compiled,
        int(spatial_dim),
        float(t_eval),
        points,
        params=params,
        include_electric=bool(include_electric),
        flow_velocity=flow_velocity,
        fallback_density_kgm3=float(fallback_density_kgm3),
        fallback_mu_pas=float(fallback_mu_pas),
        fallback_temperature_K=float(fallback_temperature_K),
        base_fields=base_fields,
    )


def _stage_support(
    compiled: CompiledRuntimeBackend,
    points: np.ndarray,
    values: dict[str, np.ndarray],
    base_fields: StageFields | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object], dict[str, np.ndarray]]:
    if base_fields is None:
        statuses = sample_compiled_valid_mask_statuses(compiled, points)
        supported = statuses == np.uint8(VALID_MASK_STATUS_CLEAN)
        return supported, statuses, {}, values
    supported = np.asarray(base_fields.supported, dtype=bool)
    statuses = np.asarray(
        base_fields.metadata.get(
            "valid_mask_status",
            np.where(
                supported,
                np.uint8(VALID_MASK_STATUS_CLEAN),
                np.uint8(VALID_MASK_STATUS_HARD_INVALID),
            ),
        ),
        dtype=np.uint8,
    )
    return (
        supported,
        statuses,
        dict(base_fields.metadata),
        {**base_fields.values, **values},
    )


def _validate_positive_gas_fields(
    values: dict[str, np.ndarray],
    supported: np.ndarray,
) -> None:
    for name in ("gas_density", "dynamic_viscosity", "temperature"):
        if name not in values:
            continue
        sampled = np.asarray(values[name], dtype=np.float64).reshape(-1)
        invalid = supported & (~np.isfinite(sampled) | (sampled <= 0.0))
        if np.any(invalid):
            bad = np.flatnonzero(invalid).tolist()
            message = (
                f"{name} requires an explicit positive gas value; "
                f"invalid sample rows: {bad}"
            )
            raise ValueError(message)


def sample_compiled_stage_fields(
    compiled: CompiledRuntimeBackend,
    spatial_dim: int,
    t_eval: float,
    positions: np.ndarray,
    *,
    force_runtime: ForceRuntimeParameters | None = None,
    include_electric: bool = False,
    flow_velocity: np.ndarray | None = None,
    gas_density_kgm3: float = float("nan"),
    gas_mu_pas: float = float("nan"),
    gas_temperature_K: float = float("nan"),
    base_fields: StageFields | None = None,
) -> StageFields:
    """Assemble every semantic field requested by one force stage."""

    dim, points = _validated_stage_points(
        spatial_dim,
        float(t_eval),
        positions,
        base_fields,
    )
    params = force_runtime or ForceRuntimeParameters()
    values = _sample_backend_force_fields(
        compiled,
        dim,
        float(t_eval),
        points,
        params=params,
        include_electric=include_electric,
        flow_velocity=flow_velocity,
        fallback_density_kgm3=gas_density_kgm3,
        fallback_mu_pas=gas_mu_pas,
        fallback_temperature_K=gas_temperature_K,
        base_fields=base_fields,
    )
    supported, statuses, metadata, values = _stage_support(
        compiled,
        points,
        values,
        base_fields,
    )
    _validate_positive_gas_fields(values, supported)
    return StageFields(
        points_m=points,
        time_s=float(t_eval),
        values=values,
        supported=supported,
        metadata={
            **metadata,
            "backend_kind": compiled.backend_kind,
            "interpolation": "linear",
            "valid_mask_status": statuses,
        },
    )


def _particle_vector(
    raw: np.ndarray | None,
    count: int,
    name: str,
    *,
    default: float,
    positive_required: bool,
) -> np.ndarray:
    values = (
        np.full(count, float(default), dtype=np.float64)
        if raw is None
        else np.asarray(raw, dtype=np.float64).reshape(-1)
    )
    if values.shape != (count,):
        raise ValueError(f"{name} must have shape ({count},)")
    if positive_required and np.any(~np.isfinite(values) | (values <= 0.0)):
        bad = np.flatnonzero(~np.isfinite(values) | (values <= 0.0)).tolist()
        raise ValueError(f"{name} must be finite and positive; invalid rows: {bad}")
    return values


def _optional_charge_over_mass(
    raw: np.ndarray | None,
    count: int,
) -> np.ndarray | None:
    if raw is None:
        return None
    values = _particle_vector(
        raw,
        count,
        "electric_q_over_m",
        default=float("nan"),
        positive_required=False,
    )
    if np.any(~np.isfinite(values)):
        raise ValueError("electric_q_over_m must be finite")
    return values


def _force_batch_static(
    params: ForceRuntimeParameters,
    count: int,
    *,
    particle_diameter: np.ndarray | None,
    particle_density: np.ndarray | None,
    particle_mass: np.ndarray | None,
    dep_particle_rel_permittivity: np.ndarray | None,
    thermophoretic_coeff: np.ndarray | None,
) -> ForceBatchStatic:
    mass_dependent = bool(
        params.thermophoresis_enabled
        or params.dielectrophoresis_enabled
        or params.lift_enabled
    )
    density_dependent = bool(
        params.pressure_gradient_enabled or params.virtual_mass_enabled
    )
    drag_diameter = _particle_vector(
        particle_diameter,
        count,
        "particle_diameter",
        default=0.0,
        positive_required=mass_dependent,
    )
    particle_density_values = _particle_vector(
        particle_density,
        count,
        "particle_density",
        default=float("nan"),
        positive_required=density_dependent,
    )
    particle_mass_values = _particle_vector(
        particle_mass,
        count,
        "particle_mass",
        default=float("nan"),
        positive_required=mass_dependent,
    )
    physical_diameter = (
        physical_sphere_diameter_m(
            mass_kg=particle_mass_values,
            density_kgm3=particle_density_values,
            drag_diameter_m=drag_diameter,
        )
        if mass_dependent
        else drag_diameter
    )
    return ForceBatchStatic(
        particle_diameter=physical_diameter,
        particle_density=particle_density_values,
        particle_mass=particle_mass_values,
        dep_particle_rel_permittivity=_particle_vector(
            dep_particle_rel_permittivity,
            count,
            "dep_particle_rel_permittivity",
            default=float(params.dep_particle_rel_permittivity),
            positive_required=False,
        ),
        thermophoretic_coeff=_particle_vector(
            thermophoretic_coeff,
            count,
            "thermophoretic_coeff",
            default=float("nan"),
            positive_required=False,
        ),
    )


def _finite_vector_values(
    raw: np.ndarray,
    count: int,
    spatial_dim: int,
    label: str,
) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float64)
    expected_shape = (count, int(spatial_dim))
    if values.shape != expected_shape or not np.all(np.isfinite(values)):
        raise ValueError(
            f"{label} must contain finite values with shape {expected_shape}"
        )
    return values


def _particle_velocity_values(
    raw: np.ndarray | None,
    count: int,
    spatial_dim: int,
    *,
    required: bool,
) -> np.ndarray:
    if raw is None:
        if required:
            raise ValueError("virtual_mass and lift require particle velocity")
        return np.zeros((count, int(spatial_dim)), dtype=np.float64)
    return _finite_vector_values(raw, count, spatial_dim, "velocity")


def _optional_flow_velocity_values(
    raw: np.ndarray | None,
    count: int,
    spatial_dim: int,
) -> np.ndarray | None:
    if raw is None:
        return None
    return _finite_vector_values(raw, count, spatial_dim, "flow_velocity")


def sample_compiled_acceleration_vectors(
    compiled: CompiledRuntimeBackend,
    spatial_dim: int,
    t_eval: float,
    positions: np.ndarray,
    *,
    electric_q_over_m: np.ndarray | None = None,
    force_runtime: ForceRuntimeParameters | None = None,
    particle_diameter: np.ndarray | None = None,
    particle_density: np.ndarray | None = None,
    particle_mass: np.ndarray | None = None,
    dep_particle_rel_permittivity: np.ndarray | None = None,
    thermophoretic_coeff: np.ndarray | None = None,
    velocity: np.ndarray | None = None,
    flow_velocity: np.ndarray | None = None,
    gas_density_kgm3: float = float("nan"),
    gas_mu_pas: float = float("nan"),
    gas_temperature_K: float = float("nan"),
    gas_molecular_mass_kg: float = float("nan"),
    stage_fields: StageFields | None = None,
) -> np.ndarray:
    dim = int(spatial_dim)
    points = np.asarray(positions, dtype=np.float64)
    if dim not in (2, 3) or points.ndim != 2 or points.shape[1] != dim:
        raise ValueError(f"positions must have shape (particle, {dim})")
    count = int(points.shape[0])
    if count == 0:
        return np.zeros((0, dim), dtype=np.float64)

    params = force_runtime or ForceRuntimeParameters()
    charge_over_mass = _optional_charge_over_mass(electric_q_over_m, count)
    static = _force_batch_static(
        params,
        count,
        particle_diameter=particle_diameter,
        particle_density=particle_density,
        particle_mass=particle_mass,
        dep_particle_rel_permittivity=dep_particle_rel_permittivity,
        thermophoretic_coeff=thermophoretic_coeff,
    )
    velocity_values = _particle_velocity_values(
        velocity,
        count,
        dim,
        required=bool(params.virtual_mass_enabled or params.lift_enabled),
    )
    flow_values = _optional_flow_velocity_values(flow_velocity, count, dim)

    pipeline = build_force_pipeline(
        params,
        include_electric=charge_over_mass is not None,
        gas_molecular_mass_kg=float(gas_molecular_mass_kg),
    )
    if not pipeline.evaluator_names:
        return np.zeros((count, dim), dtype=np.float64)
    fields = sample_compiled_stage_fields(
        compiled,
        dim,
        float(t_eval),
        points,
        force_runtime=params,
        include_electric=charge_over_mass is not None,
        flow_velocity=flow_values,
        gas_density_kgm3=float(gas_density_kgm3),
        gas_mu_pas=float(gas_mu_pas),
        gas_temperature_K=float(gas_temperature_K),
        base_fields=stage_fields,
    )
    return evaluate_force_pipeline(
        np.zeros((count, dim), dtype=np.float64),
        static,
        ForceBatchState(
            velocity=velocity_values,
            charge_over_mass=charge_over_mass,
        ),
        fields,
        pipeline,
    )


def sample_compiled_acceleration_vector(
    compiled: CompiledRuntimeBackend,
    spatial_dim: int,
    t_eval: float,
    position: np.ndarray,
    *,
    electric_q_over_m: float | None = None,
    force_runtime: ForceRuntimeParameters | None = None,
    particle_diameter: float = 0.0,
    particle_density: float = float("nan"),
    particle_mass: float | None = None,
    dep_particle_rel_permittivity: float = float("nan"),
    thermophoretic_coeff: float = float("nan"),
    velocity: np.ndarray | None = None,
    flow_velocity: np.ndarray | None = None,
    gas_density_kgm3: float = float("nan"),
    gas_mu_pas: float = float("nan"),
    gas_temperature_K: float = float("nan"),
    gas_molecular_mass_kg: float = float("nan"),
    stage_fields: StageFields | None = None,
) -> np.ndarray:
    """Evaluate one point through the canonical batch force path."""

    dim = int(spatial_dim)
    point = np.asarray(position, dtype=np.float64)
    if point.shape != (dim,):
        raise ValueError(f"position must have shape ({dim},)")
    result = sample_compiled_acceleration_vectors(
        compiled,
        dim,
        float(t_eval),
        point.reshape(1, dim),
        electric_q_over_m=(
            None
            if electric_q_over_m is None
            else np.asarray([float(electric_q_over_m)], dtype=np.float64)
        ),
        force_runtime=force_runtime,
        particle_diameter=np.asarray([float(particle_diameter)], dtype=np.float64),
        particle_density=np.asarray([float(particle_density)], dtype=np.float64),
        particle_mass=(
            None
            if particle_mass is None
            else np.asarray([float(particle_mass)], dtype=np.float64)
        ),
        dep_particle_rel_permittivity=np.asarray(
            [float(dep_particle_rel_permittivity)],
            dtype=np.float64,
        ),
        thermophoretic_coeff=np.asarray(
            [float(thermophoretic_coeff)], dtype=np.float64
        ),
        velocity=(
            None
            if velocity is None
            else np.asarray(velocity, dtype=np.float64).reshape(1, dim)
        ),
        flow_velocity=(
            None
            if flow_velocity is None
            else np.asarray(flow_velocity, dtype=np.float64).reshape(1, dim)
        ),
        gas_density_kgm3=float(gas_density_kgm3),
        gas_mu_pas=float(gas_mu_pas),
        gas_temperature_K=float(gas_temperature_K),
        gas_molecular_mass_kg=float(gas_molecular_mass_kg),
        stage_fields=stage_fields,
    )
    return np.asarray(result[0], dtype=np.float64)


__all__ = (
    "sample_compiled_acceleration_vector",
    "sample_compiled_acceleration_vectors",
    "sample_compiled_stage_fields",
)
