from __future__ import annotations

from pathlib import Path
from typing import Any, List, Mapping, Optional

import numpy as np
import pandas as pd
import yaml

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
    with_process_step_explicit_fields,
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


def _nested_or_nan(item, *path):
    cur = item
    try:
        for key in path:
            if not isinstance(cur, Mapping) or key not in cur:
                return float('nan')
            cur = cur[key]
        if pd.notna(cur):
            return float(cur)
    except Exception:
        pass
    return float('nan')


def _nested_or_empty(item, *path):
    cur = item
    try:
        for key in path:
            if not isinstance(cur, Mapping) or key not in cur:
                return ''
            cur = cur[key]
        if cur is not None and str(cur).strip():
            return str(cur)
    except Exception:
        pass
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
            bind_transition_from=_str_or_empty(row, 'bind_transition_from'),
            bind_transition_to=_str_or_empty(row, 'bind_transition_to'),
            time_anchor=_str_or_empty(row, 'time_anchor') or 'absolute',
            time_offset_s=_float_or_nan(row, 'time_offset_s'),
            duration_s=_float_or_nan(row, 'duration_s'),
        ))
    return SourceEventTable(rows=tuple(rows), metadata={'path': str(Path(path).resolve())})


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


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
    except Exception:
        return float(default)


def _as_int(value: Any, default: int) -> int:
    if _is_missing(value):
        return int(default)
    try:
        return int(value)
    except Exception:
        return int(default)


def _as_str(value: Any, default: str = '') -> str:
    if _is_missing(value):
        return str(default)
    s = str(value).strip()
    return s if s else str(default)


_PROCESS_STEP_EXPLICIT_FIELD_SPECS = (
    ('physics_flow_scale', ('physics_flow_scale', 'flow_scale'), ('physics', 'flow_scale'), 'float', 1.0),
    ('physics_drag_tau_scale', ('physics_drag_tau_scale', 'drag_tau_scale'), ('physics', 'drag_tau_scale'), 'float', 1.0),
    ('physics_body_accel_scale', ('physics_body_accel_scale', 'body_accel_scale'), ('physics', 'body_accel_scale'), 'float', 1.0),
    ('wall_mode', ('wall_mode', 'mode'), ('wall', 'mode'), 'str', 'inherit'),
    ('wall_restitution', ('wall_restitution', 'restitution'), ('wall', 'restitution'), 'float', 1.0),
    ('wall_diffuse_fraction', ('wall_diffuse_fraction', 'diffuse_fraction'), ('wall', 'diffuse_fraction'), 'float', 0.0),
    ('wall_stick_probability_scale', ('wall_stick_probability_scale', 'stick_probability_scale'), ('wall', 'stick_probability_scale'), 'float', 1.0),
    ('wall_vcrit_scale', ('wall_vcrit_scale', 'vcrit_scale'), ('wall', 'vcrit_scale'), 'float', 1.0),
    ('source_law_override', ('source_law_override', 'law_override'), ('source', 'law_override'), 'str', ''),
    ('source_speed_scale', ('source_speed_scale', 'speed_scale'), ('source', 'speed_scale'), 'float', 1.0),
    ('source_release_time_shift_s', ('source_release_time_shift_s', 'release_time_shift_s'), ('source', 'release_time_shift_s'), 'float', 0.0),
    ('source_event_gain_scale', ('source_event_gain_scale', 'event_gain_scale'), ('source', 'event_gain_scale'), 'float', 1.0),
    ('source_enabled', ('source_enabled', 'source_on'), ('source', 'enabled'), 'int', 1),
    ('output_segment_name', ('output_segment_name', 'segment_name'), ('output', 'segment_name'), 'str', ''),
    ('output_save_every_override', ('output_save_every_override', 'save_every_override'), ('output', 'save_every_override'), 'int', 0),
    ('output_save_positions', ('output_save_positions', 'save_positions'), ('output', 'save_positions'), 'int', 1),
    ('output_write_wall_events', ('output_write_wall_events', 'write_wall_events'), ('output', 'write_wall_events'), 'int', 1),
    ('output_write_diagnostics', ('output_write_diagnostics', 'write_diagnostics'), ('output', 'write_diagnostics'), 'int', 1),
)


def _matches_legacy_step_default(value: Any, value_kind: str, legacy_default: Any) -> bool:
    if value_kind == 'float':
        return abs(_as_float(value, float(legacy_default)) - float(legacy_default)) <= 1.0e-12
    if value_kind == 'int':
        return _as_int(value, int(legacy_default)) == int(legacy_default)
    return _as_str(value, str(legacy_default)).strip().lower() == str(legacy_default).strip().lower()


def _mapping_has_explicit_value(mapping: Mapping[str, Any], names: tuple[str, ...], value_kind: str, legacy_default: Any) -> bool:
    for name in names:
        if name in mapping and not _is_missing(mapping[name]):
            return not _matches_legacy_step_default(mapping[name], value_kind, legacy_default)
    return False


def _nested_has_explicit_value(mapping: Mapping[str, Any], path: tuple[str, ...], value_kind: str, legacy_default: Any) -> bool:
    cur: Any = mapping
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return False
        cur = cur[key]
    return not _is_missing(cur) and not _matches_legacy_step_default(cur, value_kind, legacy_default)


def _collect_process_step_explicit_fields(mapping: Mapping[str, Any], *, allow_nested: bool) -> tuple[str, ...]:
    explicit_fields = []
    for field_name, flat_names, nested_path, value_kind, legacy_default in _PROCESS_STEP_EXPLICIT_FIELD_SPECS:
        if _mapping_has_explicit_value(mapping, flat_names, value_kind, legacy_default):
            explicit_fields.append(field_name)
            continue
        if allow_nested and _nested_has_explicit_value(mapping, nested_path, value_kind, legacy_default):
            explicit_fields.append(field_name)
    return tuple(explicit_fields)


def _build_process_step_row(step_values: Mapping[str, Any], metadata: Mapping[str, Any]) -> ProcessStepRow:
    return ProcessStepRow(
        step_id=_as_int(step_values.get('step_id'), 1),
        step_name=_as_str(step_values.get('step_name'), 'step_1'),
        start_s=_as_float(step_values.get('start_s'), 0.0),
        end_s=_as_float(step_values.get('end_s'), 0.0),
        step_kind=_as_str(step_values.get('step_kind'), ''),
        recipe_name=_as_str(step_values.get('recipe_name'), ''),
        physics_flow_scale=_as_float(step_values.get('physics_flow_scale'), 1.0),
        physics_drag_tau_scale=_as_float(step_values.get('physics_drag_tau_scale'), 1.0),
        physics_body_accel_scale=_as_float(step_values.get('physics_body_accel_scale'), 1.0),
        wall_mode=_as_str(step_values.get('wall_mode'), 'inherit'),
        wall_restitution=_as_float(step_values.get('wall_restitution'), 1.0),
        wall_diffuse_fraction=_as_float(step_values.get('wall_diffuse_fraction'), 0.0),
        wall_stick_probability_scale=_as_float(step_values.get('wall_stick_probability_scale'), 1.0),
        wall_vcrit_scale=_as_float(step_values.get('wall_vcrit_scale'), 1.0),
        source_law_override=_as_str(step_values.get('source_law_override'), ''),
        source_speed_scale=_as_float(step_values.get('source_speed_scale'), 1.0),
        source_release_time_shift_s=_as_float(step_values.get('source_release_time_shift_s'), 0.0),
        source_event_gain_scale=_as_float(step_values.get('source_event_gain_scale'), 1.0),
        source_enabled=_as_int(step_values.get('source_enabled'), 1),
        output_segment_name=_as_str(step_values.get('output_segment_name'), ''),
        output_save_every_override=_as_int(step_values.get('output_save_every_override'), 0),
        output_save_positions=_as_int(step_values.get('output_save_positions'), 1),
        output_write_wall_events=_as_int(step_values.get('output_write_wall_events'), 1),
        output_write_diagnostics=_as_int(step_values.get('output_write_diagnostics'), 1),
        metadata=dict(metadata),
    )


def load_process_steps_csv(path: Path) -> ProcessStepTable:
    df = pd.read_csv(path)
    rows: List[ProcessStepRow] = []
    for i, row in df.iterrows():
        raw = row.to_dict()
        step_values = {
            'step_id': _mapping_pick(raw, ['step_id'], i + 1),
            'step_name': _mapping_pick(raw, ['step_name', 'name'], f'step_{i+1}'),
            'start_s': _mapping_pick(raw, ['start_s', 'start'], 0.0),
            'end_s': _mapping_pick(raw, ['end_s', 'stop_s', 't_end', 'end'], 0.0),
            'step_kind': _mapping_pick(raw, ['step_kind', 'kind'], ''),
            'recipe_name': _mapping_pick(raw, ['recipe_name', 'recipe'], ''),
            'physics_flow_scale': _mapping_pick(raw, ['physics_flow_scale', 'flow_scale'], 1.0),
            'physics_drag_tau_scale': _mapping_pick(raw, ['physics_drag_tau_scale', 'drag_tau_scale'], 1.0),
            'physics_body_accel_scale': _mapping_pick(raw, ['physics_body_accel_scale', 'body_accel_scale'], 1.0),
            'wall_mode': _mapping_pick(raw, ['wall_mode', 'mode'], 'inherit'),
            'wall_restitution': _mapping_pick(raw, ['wall_restitution', 'restitution'], 1.0),
            'wall_diffuse_fraction': _mapping_pick(raw, ['wall_diffuse_fraction', 'diffuse_fraction'], 0.0),
            'wall_stick_probability_scale': _mapping_pick(raw, ['wall_stick_probability_scale', 'stick_probability_scale'], 1.0),
            'wall_vcrit_scale': _mapping_pick(raw, ['wall_vcrit_scale', 'vcrit_scale'], 1.0),
            'source_law_override': _mapping_pick(raw, ['source_law_override', 'law_override'], ''),
            'source_speed_scale': _mapping_pick(raw, ['source_speed_scale', 'speed_scale'], 1.0),
            'source_release_time_shift_s': _mapping_pick(raw, ['source_release_time_shift_s', 'release_time_shift_s'], 0.0),
            'source_event_gain_scale': _mapping_pick(raw, ['source_event_gain_scale', 'event_gain_scale'], 1.0),
            'source_enabled': _mapping_pick(raw, ['source_enabled', 'source_on'], 1),
            'output_segment_name': _mapping_pick(raw, ['output_segment_name', 'segment_name'], ''),
            'output_save_every_override': _mapping_pick(raw, ['output_save_every_override', 'save_every_override'], 0),
            'output_save_positions': _mapping_pick(raw, ['output_save_positions', 'save_positions'], 1),
            'output_write_wall_events': _mapping_pick(raw, ['output_write_wall_events', 'write_wall_events'], 1),
            'output_write_diagnostics': _mapping_pick(raw, ['output_write_diagnostics', 'write_diagnostics'], 1),
        }
        excluded = {
            'step_id', 'step_name', 'name', 'start_s', 'start', 'end_s', 'stop_s', 't_end', 'end',
            'step_kind', 'kind', 'recipe_name', 'recipe',
            'physics_flow_scale', 'flow_scale', 'physics_drag_tau_scale', 'drag_tau_scale', 'physics_body_accel_scale', 'body_accel_scale',
            'wall_mode', 'mode', 'wall_restitution', 'restitution', 'wall_diffuse_fraction', 'diffuse_fraction',
            'wall_stick_probability_scale', 'stick_probability_scale', 'wall_vcrit_scale', 'vcrit_scale',
            'source_law_override', 'law_override', 'source_speed_scale', 'speed_scale', 'source_release_time_shift_s', 'release_time_shift_s',
            'source_event_gain_scale', 'event_gain_scale', 'source_enabled', 'source_on',
            'output_segment_name', 'segment_name', 'output_save_every_override', 'save_every_override',
            'output_save_positions', 'save_positions', 'output_write_wall_events', 'write_wall_events',
            'output_write_diagnostics', 'write_diagnostics',
        }
        metadata = {k: v for k, v in raw.items() if k not in excluded and not _is_missing(v)}
        metadata = with_process_step_explicit_fields(metadata, _collect_process_step_explicit_fields(raw, allow_nested=False))
        rows.append(_build_process_step_row(step_values, metadata))
    rows = sorted(rows, key=lambda r: (r.start_s, r.end_s, r.step_id))
    return ProcessStepTable(rows=tuple(rows), metadata={'path': str(Path(path).resolve()), 'kind': 'csv'})


def load_recipe_manifest_yaml(path: Path) -> ProcessStepTable:
    with Path(path).open('r', encoding='utf-8') as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, Mapping):
        raise ValueError('recipe_manifest.yaml root must be a mapping')
    steps_raw = payload.get('steps', [])
    if not isinstance(steps_raw, list):
        raise ValueError('recipe_manifest.steps must be a list')
    rows: List[ProcessStepRow] = []
    for i, item in enumerate(steps_raw):
        if not isinstance(item, Mapping):
            raise ValueError(f'recipe_manifest.steps[{i}] must be a mapping')
        pf = _nested_or_nan(item, 'physics', 'flow_scale')
        pdt = _nested_or_nan(item, 'physics', 'drag_tau_scale')
        pba = _nested_or_nan(item, 'physics', 'body_accel_scale')
        wr = _nested_or_nan(item, 'wall', 'restitution')
        wdf = _nested_or_nan(item, 'wall', 'diffuse_fraction')
        ws = _nested_or_nan(item, 'wall', 'stick_probability_scale')
        wv = _nested_or_nan(item, 'wall', 'vcrit_scale')
        source_law_override = _nested_or_empty(item, 'source', 'law_override')
        source_speed_scale = _nested_or_nan(item, 'source', 'speed_scale')
        source_release_shift = _nested_or_nan(item, 'source', 'release_time_shift_s')
        source_gain = _nested_or_nan(item, 'source', 'event_gain_scale')
        source_enabled = _nested_or_nan(item, 'source', 'enabled')
        out_segment = _nested_or_empty(item, 'output', 'segment_name')
        out_save_every = _nested_or_nan(item, 'output', 'save_every_override')
        out_save_positions = _nested_or_nan(item, 'output', 'save_positions')
        out_write_wall = _nested_or_nan(item, 'output', 'write_wall_events')
        out_write_diag = _nested_or_nan(item, 'output', 'write_diagnostics')
        step_values = {
            'step_id': _mapping_pick(item, ['step_id'], i + 1),
            'step_name': _mapping_pick(item, ['step_name', 'name'], f'step_{i+1}'),
            'start_s': _mapping_pick(item, ['start_s', 'start'], 0.0),
            'end_s': _mapping_pick(item, ['end_s', 'end', 'stop_s'], 0.0),
            'step_kind': _mapping_pick(item, ['step_kind', 'kind'], ''),
            'recipe_name': _mapping_pick(item, ['recipe_name'], payload.get('recipe_name', '')),
            'physics_flow_scale': _mapping_pick(item, ['physics_flow_scale', 'flow_scale'], pf if np.isfinite(pf) else 1.0),
            'physics_drag_tau_scale': _mapping_pick(item, ['physics_drag_tau_scale', 'drag_tau_scale'], pdt if np.isfinite(pdt) else 1.0),
            'physics_body_accel_scale': _mapping_pick(item, ['physics_body_accel_scale', 'body_accel_scale'], pba if np.isfinite(pba) else 1.0),
            'wall_mode': _mapping_pick(item, ['wall_mode', 'mode'], _nested_or_empty(item, 'wall', 'mode') or 'inherit'),
            'wall_restitution': _mapping_pick(item, ['wall_restitution', 'restitution'], wr if np.isfinite(wr) else 1.0),
            'wall_diffuse_fraction': _mapping_pick(item, ['wall_diffuse_fraction', 'diffuse_fraction'], wdf if np.isfinite(wdf) else 0.0),
            'wall_stick_probability_scale': _mapping_pick(item, ['wall_stick_probability_scale', 'stick_probability_scale'], ws if np.isfinite(ws) else 1.0),
            'wall_vcrit_scale': _mapping_pick(item, ['wall_vcrit_scale', 'vcrit_scale'], wv if np.isfinite(wv) else 1.0),
            'source_law_override': _mapping_pick(item, ['source_law_override', 'law_override'], source_law_override),
            'source_speed_scale': _mapping_pick(item, ['source_speed_scale', 'speed_scale'], source_speed_scale if np.isfinite(source_speed_scale) else 1.0),
            'source_release_time_shift_s': _mapping_pick(item, ['source_release_time_shift_s', 'release_time_shift_s'], source_release_shift if np.isfinite(source_release_shift) else 0.0),
            'source_event_gain_scale': _mapping_pick(item, ['source_event_gain_scale', 'event_gain_scale'], source_gain if np.isfinite(source_gain) else 1.0),
            'source_enabled': _mapping_pick(item, ['source_enabled'], source_enabled if np.isfinite(source_enabled) else 1),
            'output_segment_name': _mapping_pick(item, ['output_segment_name', 'segment_name'], out_segment),
            'output_save_every_override': _mapping_pick(item, ['output_save_every_override', 'save_every_override'], out_save_every if np.isfinite(out_save_every) else 0),
            'output_save_positions': _mapping_pick(item, ['output_save_positions', 'save_positions'], out_save_positions if np.isfinite(out_save_positions) else 1),
            'output_write_wall_events': _mapping_pick(item, ['output_write_wall_events', 'write_wall_events'], out_write_wall if np.isfinite(out_write_wall) else 1),
            'output_write_diagnostics': _mapping_pick(item, ['output_write_diagnostics', 'write_diagnostics'], out_write_diag if np.isfinite(out_write_diag) else 1),
        }
        excluded = {
            'step_id', 'step_name', 'name', 'start_s', 'start', 'end_s', 'end', 'stop_s',
            'step_kind', 'kind', 'recipe_name',
            'physics_flow_scale', 'flow_scale', 'physics_drag_tau_scale', 'drag_tau_scale', 'physics_body_accel_scale', 'body_accel_scale',
            'wall_mode', 'mode', 'wall_restitution', 'restitution', 'wall_diffuse_fraction', 'diffuse_fraction',
            'wall_stick_probability_scale', 'stick_probability_scale', 'wall_vcrit_scale', 'vcrit_scale',
            'source_law_override', 'law_override', 'source_speed_scale', 'speed_scale', 'source_release_time_shift_s', 'release_time_shift_s',
            'source_event_gain_scale', 'event_gain_scale', 'source_enabled',
            'output_segment_name', 'segment_name', 'output_save_every_override', 'save_every_override',
            'output_save_positions', 'save_positions', 'output_write_wall_events', 'write_wall_events',
            'output_write_diagnostics', 'write_diagnostics',
            'physics', 'wall', 'source', 'output',
        }
        metadata = {k: v for k, v in item.items() if k not in excluded and not _is_missing(v)}
        metadata = with_process_step_explicit_fields(metadata, _collect_process_step_explicit_fields(item, allow_nested=True))
        rows.append(_build_process_step_row(step_values, metadata))
    rows = sorted(rows, key=lambda r: (r.start_s, r.end_s, r.step_id))
    return ProcessStepTable(rows=tuple(rows), metadata={'path': str(Path(path).resolve()), 'kind': 'recipe_manifest_yaml', 'recipe_name': str(payload.get('recipe_name', ''))})
