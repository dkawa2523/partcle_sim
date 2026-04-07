from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Mapping, Optional

from .datamodel import ProcessStepRow, ProcessStepTable, process_step_explicit_fields, with_process_step_explicit_fields


def _float_or_default(value: Any, default: float) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _int_or_default(value: Any, default: int) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def _str_or_default(value: Any, default: str) -> str:
    if value is None:
        return str(default)
    s = str(value).strip()
    return s if s else str(default)


def _mapping_value(mapping: Mapping[str, Any], key: str) -> tuple[bool, Any]:
    if isinstance(mapping, Mapping) and key in mapping and mapping[key] is not None:
        return True, mapping[key]
    return False, None


def _resolve_step_field(
    current: Any,
    *,
    is_explicit: bool,
    default_map: Mapping[str, Any],
    default_key: str,
    override_map: Mapping[str, Any],
    cast,
    fallback: Any,
) -> tuple[Any, bool]:
    has_override, override_value = _mapping_value(override_map, default_key)
    if has_override:
        return cast(override_value, fallback), True
    if is_explicit:
        return cast(current, fallback), True
    has_default, default_value = _mapping_value(default_map, default_key)
    if has_default:
        return cast(default_value, fallback), True
    return cast(fallback, fallback), False


def _merge_row(row: ProcessStepRow, step_defaults: Mapping[str, Any], step_override: Mapping[str, Any]) -> ProcessStepRow:
    phys_def = step_defaults.get('physics', {}) if isinstance(step_defaults.get('physics', {}), Mapping) else {}
    wall_def = step_defaults.get('wall', {}) if isinstance(step_defaults.get('wall', {}), Mapping) else {}
    source_def = step_defaults.get('source', {}) if isinstance(step_defaults.get('source', {}), Mapping) else {}
    out_def = step_defaults.get('output', {}) if isinstance(step_defaults.get('output', {}), Mapping) else {}
    phys_ovr = step_override.get('physics', {}) if isinstance(step_override.get('physics', {}), Mapping) else {}
    wall_ovr = step_override.get('wall', {}) if isinstance(step_override.get('wall', {}), Mapping) else {}
    source_ovr = step_override.get('source', {}) if isinstance(step_override.get('source', {}), Mapping) else {}
    out_ovr = step_override.get('output', {}) if isinstance(step_override.get('output', {}), Mapping) else {}

    segment_default = _str_or_default(out_def.get('segment_name', row.step_name), row.step_name)
    row_explicit = set(process_step_explicit_fields(row))
    specs = (
        ('physics_flow_scale', phys_def, phys_ovr, 'flow_scale', _float_or_default, row.physics_flow_scale),
        ('physics_drag_tau_scale', phys_def, phys_ovr, 'drag_tau_scale', _float_or_default, row.physics_drag_tau_scale),
        ('physics_body_accel_scale', phys_def, phys_ovr, 'body_accel_scale', _float_or_default, row.physics_body_accel_scale),
        ('wall_mode', wall_def, wall_ovr, 'mode', _str_or_default, row.wall_mode),
        ('wall_restitution', wall_def, wall_ovr, 'restitution', _float_or_default, row.wall_restitution),
        ('wall_diffuse_fraction', wall_def, wall_ovr, 'diffuse_fraction', _float_or_default, row.wall_diffuse_fraction),
        ('wall_stick_probability_scale', wall_def, wall_ovr, 'stick_probability_scale', _float_or_default, row.wall_stick_probability_scale),
        ('wall_vcrit_scale', wall_def, wall_ovr, 'vcrit_scale', _float_or_default, row.wall_vcrit_scale),
        ('source_law_override', source_def, source_ovr, 'law_override', _str_or_default, row.source_law_override),
        ('source_speed_scale', source_def, source_ovr, 'speed_scale', _float_or_default, row.source_speed_scale),
        ('source_release_time_shift_s', source_def, source_ovr, 'release_time_shift_s', _float_or_default, row.source_release_time_shift_s),
        ('source_event_gain_scale', source_def, source_ovr, 'event_gain_scale', _float_or_default, row.source_event_gain_scale),
        ('source_enabled', source_def, source_ovr, 'enabled', _int_or_default, row.source_enabled),
        ('output_segment_name', out_def, out_ovr, 'segment_name', _str_or_default, segment_default),
        ('output_save_every_override', out_def, out_ovr, 'save_every_override', _int_or_default, row.output_save_every_override),
        ('output_save_positions', out_def, out_ovr, 'save_positions', _int_or_default, row.output_save_positions),
        ('output_write_wall_events', out_def, out_ovr, 'write_wall_events', _int_or_default, row.output_write_wall_events),
        ('output_write_diagnostics', out_def, out_ovr, 'write_diagnostics', _int_or_default, row.output_write_diagnostics),
    )

    resolved_values: Dict[str, Any] = {}
    resolved_explicit = set(row_explicit.difference({name for name, *_ in specs}))
    for field_name, default_map, override_map, config_key, cast, fallback in specs:
        value, is_explicit = _resolve_step_field(
            getattr(row, field_name),
            is_explicit=field_name in row_explicit,
            default_map=default_map,
            default_key=config_key,
            override_map=override_map,
            cast=cast,
            fallback=fallback,
        )
        resolved_values[field_name] = value
        if is_explicit:
            resolved_explicit.add(field_name)
    return replace(
        row,
        **resolved_values,
        metadata=with_process_step_explicit_fields({**row.metadata, 'process_control_applied': True}, resolved_explicit),
    )


def apply_process_step_controls(process_steps: Optional[ProcessStepTable], process_cfg: Mapping[str, Any]) -> Optional[ProcessStepTable]:
    if process_steps is None:
        return None
    step_defaults = process_cfg.get('step_defaults', {}) if isinstance(process_cfg.get('step_defaults', {}), Mapping) else {}
    step_overrides = process_cfg.get('step_overrides', {}) if isinstance(process_cfg.get('step_overrides', {}), Mapping) else {}
    rows = []
    for row in process_steps.rows:
        override = step_overrides.get(row.step_name, {}) if isinstance(step_overrides, Mapping) else {}
        rows.append(_merge_row(row, step_defaults, override if isinstance(override, Mapping) else {}))
    meta = dict(process_steps.metadata)
    meta['step_defaults'] = step_defaults
    meta['step_override_names'] = sorted(step_overrides.keys()) if isinstance(step_overrides, Mapping) else []
    return ProcessStepTable(rows=tuple(rows), metadata=meta)


def process_step_control_summary(process_steps: Optional[ProcessStepTable]) -> Dict[str, Any]:
    if process_steps is None:
        return {'has_process_steps': False, 'step_count': 0}
    return {
        'has_process_steps': True,
        'step_count': len(process_steps.rows),
        'step_names': [r.step_name for r in process_steps.rows],
        'segments': [r.output_segment_name or r.step_name for r in process_steps.rows],
        'wall_modes': {r.step_name: r.wall_mode for r in process_steps.rows},
        'source_law_overrides': {r.step_name: r.source_law_override for r in process_steps.rows},
        'source_speed_scales': {r.step_name: float(r.source_speed_scale) for r in process_steps.rows},
        'physics_flow_scales': {r.step_name: float(r.physics_flow_scale) for r in process_steps.rows},
        'physics_drag_tau_scales': {r.step_name: float(r.physics_drag_tau_scale) for r in process_steps.rows},
        'wall_vcrit_scales': {r.step_name: float(r.wall_vcrit_scale) for r in process_steps.rows},
        'output_save_every_overrides': {r.step_name: int(r.output_save_every_override) for r in process_steps.rows},
        'output_save_positions': {r.step_name: int(r.output_save_positions) for r in process_steps.rows},
        'output_write_wall_events': {r.step_name: int(r.output_write_wall_events) for r in process_steps.rows},
        'output_write_diagnostics': {r.step_name: int(r.output_write_diagnostics) for r in process_steps.rows},
    }
