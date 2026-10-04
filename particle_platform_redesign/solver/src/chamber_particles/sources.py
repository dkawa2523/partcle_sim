"""Deterministic realization and release ordering for table and surface sources."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .case import SimulationCase, SourceSpec
from .case_format import RealizedTableSource
from .geometry import PreparedGeometry
from .rng import SOURCE_FACET_DRAW, SOURCE_POSITION_DRAW, source_uniform_open_batch

type FloatArray = NDArray[np.float64]
type Int64Array = NDArray[np.int64]
type Int32Array = NDArray[np.int32]

SOURCE_ALGORITHM_REVISION = "table_surface_schedule_v1"
_SOURCE_REALIZATION_BATCH_SIZE = 65_536


@dataclass(frozen=True, slots=True)
class ParticleSchedule:
    """Particle rows in ID order plus a deterministic physical release order."""

    source_names: tuple[str, ...]
    particle_id: Int64Array
    source_id: Int32Array
    release_time_s: FloatArray
    position_m: FloatArray
    velocity_m_s: FloatArray
    charge_number: FloatArray
    mass_kg: FloatArray
    drag_diameter_m: FloatArray
    electrostatic_radius_m: FloatArray
    displaced_volume_m3: FloatArray
    model_weight: FloatArray
    material_id: Int32Array
    source_facet_id: Int64Array
    release_order: Int64Array

    @property
    def particle_count(self) -> int:
        return int(self.particle_id.size)


@dataclass(frozen=True, slots=True)
class _RealizedSource:
    particle_id: Int64Array
    release_time_s: FloatArray
    position_m: FloatArray
    velocity_m_s: FloatArray
    charge_number: FloatArray
    mass_kg: FloatArray
    drag_diameter_m: FloatArray
    electrostatic_radius_m: FloatArray
    displaced_volume_m3: FloatArray
    model_weight: FloatArray
    material_id: Int32Array
    source_facet_id: Int64Array


def source_particle_count(case: SimulationCase) -> int:
    """Return the validated source population without realizing particle arrays."""

    tables = {table.name: table for table in case.data.sources}
    count = 0
    for source in case.spec.sources:
        if source.kind == "table":
            table_name = source.parameters["table"]
            if not isinstance(table_name, str):
                raise TypeError("validated table source name must be text")
            count += int(tables[table_name].particle_id.size)
        elif source.kind == "surface":
            count += _integer_parameter(source.parameters, "count")
        else:
            raise ValueError(f"unsupported prepared source kind: {source.kind}")
    return count


def realize_sources(case: SimulationCase, geometry: PreparedGeometry) -> ParticleSchedule:
    """Realize all configured sources without changing particle identity."""

    tables = {table.name: table for table in case.data.sources}
    realized: list[_RealizedSource] = []
    for source_id, source in enumerate(case.spec.sources):
        if source.kind == "table":
            table_name = source.parameters["table"]
            if not isinstance(table_name, str):
                raise TypeError("validated table source name must be text")
            item = _realize_table(tables[table_name])
        elif source.kind == "surface":
            item = _realize_surface(case, geometry, source_id, source)
        else:
            raise ValueError(f"unsupported prepared source kind: {source.kind}")
        realized.append(item)

    particle_id = np.concatenate(tuple(item.particle_id for item in realized))
    # Cross-source validation makes IDs unique; only numeric ID order matters.
    particle_id.sort(kind="quicksort")
    particle_count = int(particle_id.size)
    ndim = int(realized[0].position_m.shape[1])

    source_id = np.empty(particle_count, dtype="<i4")
    release_time_s = np.empty(particle_count, dtype="<f8")
    position_m = np.empty((particle_count, ndim), dtype="<f8")
    velocity_m_s = np.empty((particle_count, ndim), dtype="<f8")
    charge_number = np.empty(particle_count, dtype="<f8")
    mass_kg = np.empty(particle_count, dtype="<f8")
    drag_diameter_m = np.empty(particle_count, dtype="<f8")
    electrostatic_radius_m = np.empty(particle_count, dtype="<f8")
    displaced_volume_m3 = np.empty(particle_count, dtype="<f8")
    model_weight = np.empty(particle_count, dtype="<f8")
    material_id = np.empty(particle_count, dtype="<i4")
    source_facet_id = np.empty(particle_count, dtype="<i8")

    for realized_source_id, item in enumerate(realized):
        resident_index = np.searchsorted(particle_id, item.particle_id)
        source_id[resident_index] = realized_source_id
        release_time_s[resident_index] = item.release_time_s
        position_m[resident_index] = item.position_m
        velocity_m_s[resident_index] = item.velocity_m_s
        charge_number[resident_index] = item.charge_number
        mass_kg[resident_index] = item.mass_kg
        drag_diameter_m[resident_index] = item.drag_diameter_m
        electrostatic_radius_m[resident_index] = item.electrostatic_radius_m
        displaced_volume_m3[resident_index] = item.displaced_volume_m3
        model_weight[resident_index] = item.model_weight
        material_id[resident_index] = item.material_id
        source_facet_id[resident_index] = item.source_facet_id

    release_order = np.asarray(np.lexsort((particle_id, release_time_s)), dtype=np.int64)
    return ParticleSchedule(
        source_names=tuple(source.name for source in case.spec.sources),
        particle_id=_read_only(particle_id),
        source_id=_read_only(source_id),
        release_time_s=_read_only(release_time_s),
        position_m=_read_only(position_m),
        velocity_m_s=_read_only(velocity_m_s),
        charge_number=_read_only(charge_number),
        mass_kg=_read_only(mass_kg),
        drag_diameter_m=_read_only(drag_diameter_m),
        electrostatic_radius_m=_read_only(electrostatic_radius_m),
        displaced_volume_m3=_read_only(displaced_volume_m3),
        model_weight=_read_only(model_weight),
        material_id=_read_only(material_id),
        source_facet_id=_read_only(source_facet_id),
        release_order=_read_only(release_order),
    )


def _realize_table(table: RealizedTableSource) -> _RealizedSource:
    count = int(table.particle_id.size)
    return _RealizedSource(
        particle_id=table.particle_id,
        release_time_s=table.release_time_s,
        position_m=table.position_m,
        velocity_m_s=table.velocity_m_s,
        charge_number=table.charge_number,
        mass_kg=table.mass_kg,
        drag_diameter_m=table.drag_diameter_m,
        electrostatic_radius_m=table.electrostatic_radius_m,
        displaced_volume_m3=table.displaced_volume_m3,
        model_weight=table.model_weight,
        material_id=table.material_id,
        source_facet_id=np.full(count, -1, dtype="<i8"),
    )


def _realize_surface(
    case: SimulationCase,
    geometry: PreparedGeometry,
    source_id: int,
    source: SourceSpec,
) -> _RealizedSource:
    count = _integer_parameter(source.parameters, "count")
    particle_id_start = _integer_parameter(source.parameters, "particle_id_start")
    particle = source.particle
    if particle is None:
        raise ValueError("surface source lost its particle properties")
    group_name = source.parameters["boundary_group"]
    if not isinstance(group_name, str):
        raise ValueError("surface source boundary group must be text")
    group_id = case.data.geometry.group_names.index(group_name)
    facets = np.flatnonzero(geometry.group_id == group_id).astype("<i8", copy=False)
    if not facets.size:
        raise ValueError(f"surface source {source.name!r} group has no material facets")

    position_model = _mapping_parameter(source.parameters, "position")
    facet_id, facet_parameter = _surface_positions(
        case,
        geometry,
        source_id,
        count,
        facets,
        position_model,
    )
    position_m = geometry.facet_start_m[facet_id] + facet_parameter[:, None] * (
        geometry.facet_end_m[facet_id] - geometry.facet_start_m[facet_id]
    )
    collapsed_to_start = np.equal(position_m, geometry.facet_start_m[facet_id]).all(axis=1)
    collapsed_to_end = np.equal(position_m, geometry.facet_end_m[facet_id]).all(axis=1)
    if bool((collapsed_to_start | collapsed_to_end).any()):
        raise ValueError("surface position rounded to a facet endpoint")
    velocity_m_s = _surface_velocities(
        geometry,
        facet_id,
        _mapping_parameter(source.parameters, "velocity"),
        count,
    )
    release_time_s = _surface_release_times(
        case,
        _mapping_parameter(source.parameters, "release"),
        count,
    )
    return _RealizedSource(
        particle_id=np.arange(
            particle_id_start,
            particle_id_start + count,
            dtype="<i8",
        ),
        release_time_s=release_time_s,
        position_m=np.asarray(position_m, dtype="<f8"),
        velocity_m_s=velocity_m_s,
        charge_number=_constant(count, particle.charge_number),
        mass_kg=_constant(count, particle.mass_kg),
        drag_diameter_m=_constant(count, particle.drag_diameter_m),
        electrostatic_radius_m=_constant(count, particle.electrostatic_radius_m),
        displaced_volume_m3=_constant(count, particle.displaced_volume_m3),
        model_weight=_constant(count, particle.model_weight),
        material_id=np.full(count, particle.material_id, dtype="<i4"),
        source_facet_id=facet_id,
    )


def _surface_positions(
    case: SimulationCase,
    geometry: PreparedGeometry,
    source_id: int,
    count: int,
    facets: Int64Array,
    model: Mapping[str, object],
) -> tuple[Int64Array, FloatArray]:
    model_id = model.get("model")
    if model_id == "edge_fraction":
        if set(model) != {"model", "fraction"} or facets.size != 1:
            raise ValueError("edge_fraction requires one-facet group and exactly fraction")
        fraction = _finite_number(model["fraction"], "surface position fraction")
        if not 0.0 < fraction < 1.0:
            raise ValueError("surface position fraction must be strictly inside (0, 1)")
        return (
            np.full(count, int(facets[0]), dtype="<i8"),
            np.full(count, fraction, dtype="<f8"),
        )
    if model_id != "uniform" or set(model) != {"model", "measure"}:
        raise ValueError("surface position supports edge_fraction or uniform with explicit measure")
    measure = model["measure"]
    if not isinstance(measure, str):
        raise ValueError("surface position measure must be text")
    weights = _facet_weights(case.data.coordinate_system, geometry, facets, measure)
    cumulative = np.cumsum(weights, dtype=np.float64)
    if not bool(np.isfinite(cumulative).all()) or cumulative[-1] <= 0.0:
        raise ValueError("surface source has no positive finite facet measure")
    if bool((np.diff(cumulative) <= 0.0).any()):
        raise ValueError("surface facet CDF is not strictly increasing at float64 precision")

    selected = np.empty(count, dtype="<i8")
    parameter = np.empty(count, dtype="<f8")
    for begin in range(0, count, _SOURCE_REALIZATION_BATCH_SIZE):
        end = min(begin + _SOURCE_REALIZATION_BATCH_SIZE, count)
        ordinal = np.arange(begin, end, dtype=np.uint64)
        facet_draw = source_uniform_open_batch(
            case.spec.solver.seed,
            source_id,
            ordinal,
            SOURCE_FACET_DRAW,
        )
        local_index = np.minimum(
            np.searchsorted(cumulative, facet_draw * cumulative[-1]),
            facets.size - 1,
        )
        selected[begin:end] = facets[local_index]
        position_draw = source_uniform_open_batch(
            case.spec.solver.seed,
            source_id,
            ordinal,
            SOURCE_POSITION_DRAW,
        )
        parameter[begin:end] = _facet_parameters(
            geometry,
            selected[begin:end],
            measure,
            position_draw,
        )
    return selected, parameter


def _facet_weights(
    coordinate_system: str,
    geometry: PreparedGeometry,
    facets: Int64Array,
    measure: str,
) -> FloatArray:
    if coordinate_system == "cartesian_xy":
        if measure != "line_length":
            raise ValueError("cartesian_xy uniform surface position requires line_length")
        return geometry.facet_length_m[facets].copy()
    if coordinate_system != "axisymmetric_rz":
        raise ValueError("unsupported surface-source coordinate system")
    if measure == "meridional_length":
        return geometry.facet_length_m[facets].copy()
    if measure != "revolved_area":
        raise ValueError(
            "axisymmetric_rz surface measure must be meridional_length or revolved_area"
        )
    radius_sum = geometry.facet_start_m[facets, 0] + geometry.facet_end_m[facets, 0]
    weights = geometry.facet_length_m[facets] * radius_sum
    positive = weights > 0.0
    if not bool(positive.all()):
        raise ValueError("revolved_area source group contains a zero-area facet")
    return weights


def _facet_parameters(
    geometry: PreparedGeometry,
    facet_id: Int64Array,
    measure: str,
    uniform: FloatArray,
) -> FloatArray:
    if measure != "revolved_area":
        return uniform
    radius0 = geometry.facet_start_m[facet_id, 0]
    radius1 = geometry.facet_end_m[facet_id, 0]
    discriminant = (1.0 - uniform) * radius0 * radius0 + uniform * radius1 * radius1
    radial = np.sqrt(discriminant)
    denominator = radius0 + radial
    if bool((denominator <= 0.0).any()):
        raise ValueError("revolved-area inverse CDF has no positive radial denominator")
    parameter = uniform * (radius0 + radius1) / denominator
    if not bool(np.isfinite(parameter).all()) or bool(
        ((parameter <= 0.0) | (parameter >= 1.0)).any()
    ):
        raise ValueError("revolved-area inverse CDF is unresolved")
    return np.asarray(parameter, dtype="<f8")


def _surface_velocities(
    geometry: PreparedGeometry,
    facet_id: Int64Array,
    model: Mapping[str, object],
    count: int,
) -> FloatArray:
    model_id = model.get("model")
    if model_id == "fixed":
        if set(model) != {"model", "value_m_s"}:
            raise ValueError("fixed surface velocity requires exactly value_m_s")
        value = model["value_m_s"]
        if not isinstance(value, tuple) or len(value) != 2:
            raise ValueError("fixed surface velocity value_m_s must contain two values")
        vector = np.asarray(
            [_finite_number(item, "surface fixed velocity") for item in value],
            dtype="<f8",
        )
        return np.repeat(vector[None, :], count, axis=0)
    if model_id == "normal":
        if set(model) != {"model", "direction", "speed_m_s"}:
            raise ValueError("normal surface velocity requires direction and speed_m_s")
        if model["direction"] != "into_domain":
            raise ValueError("normal surface velocity direction must be into_domain")
        speed = _finite_number(model["speed_m_s"], "surface normal speed_m_s")
        if speed <= 0.0:
            raise ValueError("surface normal speed_m_s must be positive")
        return np.asarray(-speed * geometry.facet_normal[facet_id], dtype="<f8")
    raise ValueError("surface velocity supports fixed or inward normal models")


def _surface_release_times(
    case: SimulationCase,
    model: Mapping[str, object],
    count: int,
) -> FloatArray:
    if model.get("model") != "fixed" or set(model) != {"model", "time_s"}:
        raise ValueError("surface release supports only fixed time_s")
    time_s = _finite_number(model["time_s"], "surface release time_s")
    if not case.spec.time.start_s <= time_s <= case.spec.time.end_s:
        raise ValueError("surface release time_s is outside the run interval")
    return np.full(count, time_s, dtype="<f8")


def _constant(count: int, value: float) -> FloatArray:
    return np.full(count, value, dtype="<f8")


def _mapping_parameter(parameters: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = parameters[name]
    if not isinstance(value, Mapping):
        raise ValueError(f"surface source {name} must be a mapping")
    return value


def _integer_parameter(parameters: Mapping[str, object], name: str) -> int:
    value = parameters[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"surface source {name} must be an integer")
    return value


def _finite_number(value: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{location} must be a finite number")
    return result


def _read_only[Scalar: np.generic](array: NDArray[Scalar]) -> NDArray[Scalar]:
    array.flags.writeable = False
    return array
