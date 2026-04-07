from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np

from .datamodel import (
    MaterialRow,
    MaterialTable,
    PartWallRow,
    PartWallTable,
    ParticleTable,
    SourceResolutionParameters,
)
from .source_schema import (
    DEFAULT_SOURCE_EVENT_TAG,
    DEFAULT_SOURCE_LAW_NAME,
    SOURCE_RESOLUTION_SCALAR_DEFAULTS,
    SOURCE_RESOLUTION_SCALAR_FIELDS,
)
from .source_material_common import pick_float, pick_str
from .source_registry import get_source_law


def global_source_defaults(
    source_cfg: Mapping[str, Any],
    gas_temperature: float,
    gas_viscosity: float,
) -> Dict[str, float | str]:
    defaults: Dict[str, float | str] = {
        'source_law': str(source_cfg.get('default_law', DEFAULT_SOURCE_LAW_NAME)),
        'source_default_event_tag': str(source_cfg.get('source_default_event_tag', DEFAULT_SOURCE_EVENT_TAG)),
    }
    for field_name, static_default in SOURCE_RESOLUTION_SCALAR_DEFAULTS:
        fallback = static_default
        if field_name == 'source_temperature_K':
            fallback = gas_temperature
        elif field_name == 'source_dynamic_viscosity_Pas':
            fallback = gas_viscosity
        elif field_name == 'source_resuspension_utau_threshold_mps':
            fallback = source_cfg.get('source_resuspension_friction_velocity_threshold_mps', static_default)
        defaults[field_name] = float(source_cfg.get(field_name, fallback))
    return defaults


def resolve_source_parameters(
    particles: ParticleTable,
    walls: Optional[PartWallTable],
    materials: Optional[MaterialTable],
    source_cfg: Mapping[str, Any],
    gas_temperature: float,
    gas_viscosity: float,
) -> SourceResolutionParameters:
    defaults = global_source_defaults(source_cfg, gas_temperature, gas_viscosity)
    wall_lookup = walls.as_lookup() if walls is not None else {}
    material_lookup = materials.as_lookup() if materials is not None else {}
    n = particles.count

    resolved_material_id = np.zeros(n, dtype=np.int64)
    source_material_id = np.zeros(n, dtype=np.int64)
    resolved_law_name: list[str] = []
    resolved_law_code = np.zeros(n, dtype=np.int32)
    resolved_event_tag = np.full(n, '', dtype=object)

    scalar_arrays = {
        field_name: np.full(n, float(defaults[field_name]), dtype=np.float64)
        for field_name in SOURCE_RESOLUTION_SCALAR_FIELDS
    }
    law_usage: Dict[str, int] = {}
    material_usage: Dict[str, int] = {}
    unresolved = 0

    for i in range(n):
        src_pid = int(particles.source_part_id[i])
        wall_row: Optional[PartWallRow] = wall_lookup.get(src_pid) if src_pid > 0 else None
        input_mid = int(particles.material_id[i])
        source_mid = int(wall_row.material_id) if wall_row is not None else 0
        source_material_id[i] = source_mid
        material_row: Optional[MaterialRow] = None
        if input_mid > 0 and input_mid in material_lookup:
            material_row = material_lookup[input_mid]
            resolved_material_id[i] = input_mid
        elif source_mid > 0 and source_mid in material_lookup:
            material_row = material_lookup[source_mid]
            resolved_material_id[i] = source_mid
        else:
            unresolved += 1

        law_name = pick_str(
            particles.source_law_override[i],
            wall_row.source_law if wall_row is not None else None,
            material_row.source_law if material_row is not None else None,
            defaults['source_law'],
            default=DEFAULT_SOURCE_LAW_NAME,
        )
        law = get_source_law(law_name)
        resolved_law_name.append(law.name)
        resolved_law_code[i] = int(law.code)
        law_usage[law.name] = law_usage.get(law.name, 0) + 1
        resolved_event_tag[i] = pick_str(
            particles.source_event_tag[i],
            wall_row.source_default_event_tag if wall_row is not None else None,
            material_row.source_default_event_tag if material_row is not None else None,
            defaults['source_default_event_tag'],
            default='',
        )

        def pick(attr: str, default_val: float) -> float:
            particle_override = None
            if attr == 'source_speed_scale':
                particle_override = particles.source_speed_scale_override[i]
            return pick_float(
                particle_override,
                getattr(wall_row, attr) if wall_row is not None and hasattr(wall_row, attr) else np.nan,
                getattr(material_row, attr) if material_row is not None and hasattr(material_row, attr) else np.nan,
                defaults[attr],
                default=default_val,
            )

        for attr, arr in scalar_arrays.items():
            arr[i] = pick(attr, float(defaults[attr]))

        material_name = material_row.material_name if material_row is not None else 'global_default'
        material_usage[material_name] = material_usage.get(material_name, 0) + 1

    metadata = {
        'law_usage': law_usage,
        'material_usage': material_usage,
        'resolution_order': [
            'particle.source_law_override',
            'part_walls.source_law',
            'materials.source_law',
            'source.default_law',
        ],
        'unresolved_particle_count': int(unresolved),
        'global_defaults': defaults,
    }
    return SourceResolutionParameters(
        resolved_material_id=resolved_material_id,
        source_material_id=source_material_id,
        resolved_law_name=tuple(resolved_law_name),
        resolved_law_code=resolved_law_code,
        resolved_event_tag=resolved_event_tag,
        **{field_name: scalar_arrays[field_name] for field_name in SOURCE_RESOLUTION_SCALAR_FIELDS},
        metadata=metadata,
    )


__all__ = ('global_source_defaults', 'resolve_source_parameters')
