from __future__ import annotations

from pathlib import Path
from typing import Any, List, Mapping, Optional

import numpy as np
import pandas as pd

from ..core.datamodel import (
    MaterialRow,
    MaterialTable,
    PartWallRow,
    PartWallTable,
    ParticleTable,
    ProcessStepRow,
    ProcessStepTable,
    SourceEventRow,
    SourceEventTable,
)
from ..core.source_schema import SOURCE_CATALOG_FLOAT_ALIASES, SOURCE_CATALOG_TEXT_ALIASES


def _get_col(df: pd.DataFrame, candidates, default=None):
    for c in candidates:
        if c in df.columns:
            return df[c].to_numpy()
    if default is None:
        raise ValueError(f'Missing required columns; expected one of {candidates}')
    return np.full(len(df), default)


def _float_or_nan(row, *names):
    for name in names:
        if name in row and pd.notna(row[name]):
            return float(row[name])
    return float('nan')


def _str_or_empty(row, *names):
    for name in names:
        if name in row and pd.notna(row[name]):
            return str(row[name])
    return ''


def load_particles_csv(path: Path, spatial_dim: int, coordinate_system: str) -> ParticleTable:
    df = pd.read_csv(path)
    n = len(df)
    pid = _get_col(df, ['particle_id', 'id'], default=np.arange(n, dtype=np.int64)).astype(np.int64)
    x = _get_col(df, ['x', 'r'])
    y = _get_col(df, ['y', 'z'] if coordinate_system == 'axisymmetric_rz' else ['y'])
    z = _get_col(df, ['z'], default=0.0) if spatial_dim == 3 else None
    vx = _get_col(df, ['vx', 'vr'], default=0.0)
    vy = _get_col(df, ['vy', 'vz'] if coordinate_system == 'axisymmetric_rz' else ['vy'], default=0.0)
    vz = _get_col(df, ['vz'], default=0.0) if spatial_dim == 3 else None
    release_time = _get_col(df, ['release_time', 't0'], default=0.0)
    mass = _get_col(df, ['mass'], default=1e-15)
    diameter = _get_col(df, ['diameter', 'd', 'd_eq'], default=1e-6)
    density = _get_col(df, ['density', 'rho_p'], default=1000.0)
    charge = _get_col(df, ['charge', 'q'], default=0.0)
    source_part_id = _get_col(df, ['source_part_id', 'part_id_source', 'origin_part_id'], default=0).astype(np.int64)
    material_id = _get_col(df, ['material_id', 'particle_material_id'], default=0).astype(np.int64)
    source_event_tag = _get_col(df, ['source_event_tag', 'event_tag'], default='').astype(object)
    source_law_override = _get_col(df, ['source_law_override'], default='').astype(object)
    source_speed_scale_override = _get_col(df, ['source_speed_scale_override'], default=np.nan).astype(np.float64)
    stick_probability = _get_col(df, ['stick_probability', 'p_stick'], default=0.0)
    dep_rel = _get_col(df, ['dep_particle_rel_permittivity', 'epsr_particle'], default=2.0)
    thermo = _get_col(df, ['thermophoretic_coeff', 'thermo_coeff'], default=0.0)
    if spatial_dim == 2:
        position = np.stack([x, y], axis=1).astype(np.float64)
        velocity = np.stack([vx, vy], axis=1).astype(np.float64)
    else:
        position = np.stack([x, y, z], axis=1).astype(np.float64)
        velocity = np.stack([vx, vy, vz], axis=1).astype(np.float64)
    return ParticleTable(
        spatial_dim=spatial_dim,
        particle_id=pid.astype(np.int64),
        position=position,
        velocity=velocity,
        release_time=np.asarray(release_time, dtype=np.float64),
        mass=np.asarray(mass, dtype=np.float64),
        diameter=np.asarray(diameter, dtype=np.float64),
        density=np.asarray(density, dtype=np.float64),
        charge=np.asarray(charge, dtype=np.float64),
        source_part_id=np.asarray(source_part_id, dtype=np.int64),
        material_id=np.asarray(material_id, dtype=np.int64),
        source_event_tag=np.asarray(source_event_tag, dtype=object),
        source_law_override=np.asarray(source_law_override, dtype=object),
        source_speed_scale_override=np.asarray(source_speed_scale_override, dtype=np.float64),
        stick_probability=np.asarray(stick_probability, dtype=np.float64),
        dep_particle_rel_permittivity=np.asarray(dep_rel, dtype=np.float64),
        thermophoretic_coeff=np.asarray(thermo, dtype=np.float64),
        metadata={'path': str(Path(path).resolve())},
    )


def _source_material_row_kwargs(row, *, source_law_names: tuple[str, ...]) -> dict[str, Any]:
    kwargs = {'source_law': _str_or_empty(row, *source_law_names)}
    for field_name, aliases in SOURCE_CATALOG_TEXT_ALIASES:
        kwargs[field_name] = _str_or_empty(row, *aliases)
    for field_name, aliases in SOURCE_CATALOG_FLOAT_ALIASES:
        kwargs[field_name] = _float_or_nan(row, *aliases)
    return kwargs


def load_part_walls_csv(path: Path) -> PartWallTable:
    df = pd.read_csv(path)
    rows: List[PartWallRow] = []
    for _, row in df.iterrows():
        rows.append(
            PartWallRow(
                part_id=int(row.get('part_id', 1)),
                part_name=str(row.get('part_name', f'part_{int(row.get("part_id", 1))}')),
                material_id=int(row.get('material_id', 0) or 0),
                material_name=str(row.get('material_name', '')),
                **_source_material_row_kwargs(row, source_law_names=('source_law',)),
            )
        )
    return PartWallTable(rows=tuple(rows), metadata={'path': str(Path(path).resolve())})


def load_materials_csv(path: Path) -> MaterialTable:
    df = pd.read_csv(path)
    rows: List[MaterialRow] = []
    for _, row in df.iterrows():
        rows.append(
            MaterialRow(
                material_id=int(row.get('material_id', 0)),
                material_name=str(row.get('material_name', f'material_{int(row.get("material_id", 0))}')),
                **_source_material_row_kwargs(row, source_law_names=('source_law', 'source_law_default')),
            )
        )
    return MaterialTable(rows=tuple(rows), metadata={'path': str(Path(path).resolve())})


def load_source_events_csv(path: Path) -> SourceEventTable:
    df = pd.read_csv(path)
    transition_columns = sorted(
        col for col in ('bind_transition_from', 'bind_transition_to') if col in set(map(str, df.columns))
    )
    if transition_columns:
        raise ValueError(
            'source event transition bindings are no longer supported; '
            f'use absolute times or bind_step_name instead of {transition_columns}'
        )
    rows: List[SourceEventRow] = []
    for _, row in df.iterrows():
        rows.append(SourceEventRow(
            event_id=int(row.get('event_id', 0) or 0),
            event_name=str(row.get('event_name', f'event_{int(row.get("event_id", 0) or 0)}')),
            event_kind=str(row.get('event_kind', 'gaussian_burst') or 'gaussian_burst'),
            enabled=int(row.get('enabled', 1) or 0),
            applies_to_particle_id=int(row.get('applies_to_particle_id', 0) or 0),
            applies_to_source_part_id=int(row.get('applies_to_source_part_id', 0) or 0),
            applies_to_material_id=int(row.get('applies_to_material_id', 0) or 0),
            applies_to_source_law=_str_or_empty(row, 'applies_to_source_law'),
            applies_to_event_tag=_str_or_empty(row, 'applies_to_event_tag'),
            center_s=_float_or_nan(row, 'center_s'),
            sigma_s=_float_or_nan(row, 'sigma_s'),
            amplitude=_float_or_nan(row, 'amplitude'),
            period_s=_float_or_nan(row, 'period_s'),
            phase_s=_float_or_nan(row, 'phase_s'),
            start_s=_float_or_nan(row, 'start_s'),
            end_s=_float_or_nan(row, 'end_s'),
            gain_multiplier=_float_or_nan(row, 'gain_multiplier'),
            release_time_shift_s=_float_or_nan(row, 'release_time_shift_s'),
            min_factor=_float_or_nan(row, 'min_factor'),
            max_factor=_float_or_nan(row, 'max_factor'),
            bind_step_name=_str_or_empty(row, 'bind_step_name'),
            time_anchor=_str_or_empty(row, 'time_anchor') or 'absolute',
            time_offset_s=_float_or_nan(row, 'time_offset_s'),
            duration_s=_float_or_nan(row, 'duration_s'),
        ))
    return SourceEventTable(rows=tuple(rows), metadata={'path': str(Path(path).resolve())})


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, dict, Mapping, np.ndarray)):
        return False
    return bool(pd.isna(value))


def _mapping_pick(mapping: Mapping[str, Any], names: list[str], default: Any = None) -> Any:
    for name in names:
        if name in mapping and not _is_missing(mapping[name]):
            return mapping[name]
    return default


def _as_float(value: Any, default: float) -> float:
    if _is_missing(value):
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_int(value: Any, default: int) -> int:
    if _is_missing(value):
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _as_str(value: Any, default: str = '') -> str:
    if _is_missing(value):
        return str(default)
    s = str(value).strip()
    return s if s else str(default)


_PROCESS_STEP_COLUMNS = {
    'step_id',
    'step_name',
    'name',
    'start_s',
    'start',
    'end_s',
    'stop_s',
    't_end',
    'end',
    'output_segment_name',
    'segment_name',
}


def _build_process_step_row(step_values: Mapping[str, Any], metadata: Mapping[str, Any]) -> ProcessStepRow:
    return ProcessStepRow(
        step_id=_as_int(step_values.get('step_id'), 1),
        step_name=_as_str(step_values.get('step_name'), 'step_1'),
        start_s=_as_float(step_values.get('start_s'), 0.0),
        end_s=_as_float(step_values.get('end_s'), 0.0),
        output_segment_name=_as_str(step_values.get('output_segment_name'), ''),
        metadata=dict(metadata),
    )


def load_process_steps_csv(path: Path) -> ProcessStepTable:
    df = pd.read_csv(path)
    unsupported_columns = sorted(str(col) for col in df.columns if str(col) not in _PROCESS_STEP_COLUMNS)
    if unsupported_columns:
        raise ValueError(
            'process_steps.csv supports only time-label columns '
            '(step_id, step_name, start_s, end_s, output_segment_name); '
            f'unsupported columns: {unsupported_columns}'
        )
    rows: List[ProcessStepRow] = []
    for i, row in df.iterrows():
        raw = row.to_dict()
        step_values = {
            'step_id': _mapping_pick(raw, ['step_id'], i + 1),
            'step_name': _mapping_pick(raw, ['step_name', 'name'], f'step_{i+1}'),
            'start_s': _mapping_pick(raw, ['start_s', 'start'], 0.0),
            'end_s': _mapping_pick(raw, ['end_s', 'stop_s', 't_end', 'end'], 0.0),
            'output_segment_name': _mapping_pick(raw, ['output_segment_name', 'segment_name'], ''),
        }
        rows.append(_build_process_step_row(step_values, {}))
    rows = sorted(rows, key=lambda r: (r.start_s, r.end_s, r.step_id))
    return ProcessStepTable(rows=tuple(rows), metadata={'path': str(Path(path).resolve()), 'kind': 'csv'})
