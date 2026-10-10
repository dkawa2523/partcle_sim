"""Release ordering for fully realized internal and surface particle schedules."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .case import SimulationCase
from .case_format import RealizedSource, RealizedSurfaceSource, RealizedTableSource
from .geometry import PreparedGeometry

type FloatArray = NDArray[np.float64]
type Int64Array = NDArray[np.int64]
type Int32Array = NDArray[np.int32]

SOURCE_ALGORITHM_REVISION = "realized_internal_surface_contact_schedule_v5"


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
    contact_radius_m: FloatArray
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
    contact_radius_m: FloatArray
    electrostatic_radius_m: FloatArray
    displaced_volume_m3: FloatArray
    model_weight: FloatArray
    material_id: Int32Array
    source_facet_id: Int64Array


def source_particle_count(case: SimulationCase) -> int:
    """Return the validated source population without allocating runtime arrays."""

    tables = {table.name: table for table in case.data.sources}
    return sum(
        int(tables[_table_name(source.parameters)].particle_id.size) for source in case.spec.sources
    )


def realize_sources(case: SimulationCase, geometry: PreparedGeometry) -> ParticleSchedule:
    """Resolve canonical rows into the sole runtime particle schedule."""

    tables = {table.name: table for table in case.data.sources}
    realized = [
        _realize_canonical(tables[_table_name(source.parameters)], geometry)
        for source in case.spec.sources
    ]

    particle_id = np.concatenate(tuple(item.particle_id for item in realized))
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
    contact_radius_m = np.empty(particle_count, dtype="<f8")
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
        contact_radius_m[resident_index] = item.contact_radius_m
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
        contact_radius_m=_read_only(contact_radius_m),
        electrostatic_radius_m=_read_only(electrostatic_radius_m),
        displaced_volume_m3=_read_only(displaced_volume_m3),
        model_weight=_read_only(model_weight),
        material_id=_read_only(material_id),
        source_facet_id=_read_only(source_facet_id),
        release_order=_read_only(release_order),
    )


def _table_name(parameters: Mapping[str, object]) -> str:
    table_name = parameters["table"]
    if not isinstance(table_name, str):
        raise TypeError("validated source table name must be text")
    return table_name


def _realize_canonical(source: RealizedSource, geometry: PreparedGeometry) -> _RealizedSource:
    count = int(source.particle_id.size)
    if isinstance(source, RealizedTableSource):
        position_m = source.position_m
        source_facet_id = np.full(count, -1, dtype="<i8")
    elif isinstance(source, RealizedSurfaceSource):
        source_facet_id = source.facet_id
        position_m = geometry.facet_start_m[source_facet_id] + source.facet_parameter[:, None] * (
            geometry.facet_end_m[source_facet_id] - geometry.facet_start_m[source_facet_id]
        )
        collapsed_to_start = np.equal(position_m, geometry.facet_start_m[source_facet_id]).all(
            axis=1
        )
        collapsed_to_end = np.equal(position_m, geometry.facet_end_m[source_facet_id]).all(axis=1)
        if bool((collapsed_to_start | collapsed_to_end).any()):
            raise ValueError("surface source position rounded to a facet endpoint")
        position_m = np.asarray(
            position_m
            - (source.contact_radius_m * geometry.facet_contact_enabled[source_facet_id])[:, None]
            * geometry.facet_normal[source_facet_id],
            dtype="<f8",
        )
        if not bool(np.isfinite(position_m).all()):
            raise ValueError("surface source contact-radius offset is not finite")
    else:
        raise TypeError("unsupported canonical source type")
    return _RealizedSource(
        particle_id=source.particle_id,
        release_time_s=source.release_time_s,
        position_m=position_m,
        velocity_m_s=source.velocity_m_s,
        charge_number=source.charge_number,
        mass_kg=source.mass_kg,
        drag_diameter_m=source.drag_diameter_m,
        contact_radius_m=source.contact_radius_m,
        electrostatic_radius_m=source.electrostatic_radius_m,
        displaced_volume_m3=source.displaced_volume_m3,
        model_weight=source.model_weight,
        material_id=source.material_id,
        source_facet_id=source_facet_id,
    )


def _read_only[Scalar: np.generic](array: NDArray[Scalar]) -> NDArray[Scalar]:
    array.flags.writeable = False
    return array


__all__ = [
    "SOURCE_ALGORITHM_REVISION",
    "ParticleSchedule",
    "realize_sources",
    "source_particle_count",
]
