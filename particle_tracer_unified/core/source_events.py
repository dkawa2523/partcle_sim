from __future__ import annotations

from dataclasses import replace
from typing import Dict, Optional, Tuple

import numpy as np

from .datamodel import ProcessStepRow, ProcessStepTable, SourceEventRow, SourceEventTable


def _is_finite(v) -> bool:
    if v is None:
        return False
    try:
        value = float(v)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(value))


def _step_lookup(process_steps: Optional[ProcessStepTable]) -> Dict[str, ProcessStepRow]:
    if process_steps is None:
        return {}
    return process_steps.as_name_lookup()


def _anchor_time(row: SourceEventRow, process_steps: Optional[ProcessStepTable]) -> Tuple[Optional[float], Dict[str, object]]:
    meta = {
        'binding': 'absolute',
        'bind_step_name': row.bind_step_name,
        'time_anchor': row.time_anchor,
    }
    step_map = _step_lookup(process_steps)
    anchor = str(row.time_anchor or 'absolute').strip().lower()
    if row.bind_step_name and row.bind_step_name in step_map:
        step = step_map[row.bind_step_name]
        meta['binding'] = 'step'
        if anchor in {'step_start', 'start'}:
            return float(step.start_s), meta
        if anchor in {'step_end', 'end'}:
            return float(step.end_s), meta
        if anchor in {'step_center', 'center', 'mid'}:
            return 0.5 * float(step.start_s + step.end_s), meta
        if anchor == 'absolute':
            return 0.0, meta
        return float(step.start_s), meta
    if anchor == 'absolute':
        return 0.0, meta
    return None, meta


def _compile_event_row(row: SourceEventRow, process_steps: Optional[ProcessStepTable]) -> SourceEventRow:
    base, meta = _anchor_time(row, process_steps)
    if base is None:
        return replace(row, enabled=0, metadata={**row.metadata, **meta, 'compile_status': 'unresolved_binding'})
    offset = float(row.time_offset_s) if _is_finite(row.time_offset_s) else 0.0
    base = float(base + offset)
    kind = str(row.event_kind).strip().lower()
    new_center = row.center_s
    new_start = row.start_s
    new_end = row.end_s
    if kind in {'gaussian_burst', 'burst', 'periodic_burst', 'periodic'}:
        rel_center = float(row.center_s) if _is_finite(row.center_s) else 0.0
        new_center = base + rel_center
    elif kind in {'window_gate', 'gate'}:
        if row.bind_step_name and process_steps is not None and row.bind_step_name in process_steps.as_name_lookup() and not _is_finite(row.start_s) and not _is_finite(row.end_s):
            step = process_steps.as_name_lookup()[row.bind_step_name]
            new_start = float(step.start_s + offset)
            new_end = float(step.end_s + offset)
        else:
            rel_start = float(row.start_s) if _is_finite(row.start_s) else 0.0
            rel_end = float(row.end_s) if _is_finite(row.end_s) else float(row.duration_s if _is_finite(row.duration_s) else 0.0)
            new_start = base + rel_start
            new_end = base + rel_end
    else:
        if _is_finite(row.start_s):
            new_start = base + float(row.start_s)
        if _is_finite(row.end_s):
            new_end = base + float(row.end_s)
    return replace(
        row,
        center_s=float(new_center) if _is_finite(new_center) else new_center,
        start_s=float(new_start) if _is_finite(new_start) else new_start,
        end_s=float(new_end) if _is_finite(new_end) else new_end,
        metadata={**row.metadata, **meta, 'compile_status': 'compiled', 'compiled_anchor_s': base},
    )


def compile_source_events(events: Optional[SourceEventTable], process_steps: Optional[ProcessStepTable]) -> Optional[SourceEventTable]:
    if events is None:
        return None
    rows = tuple(_compile_event_row(r, process_steps) for r in events.rows)
    meta = dict(events.metadata)
    meta['compiled_from_process_steps'] = process_steps is not None
    if process_steps is not None:
        meta['process_steps_path'] = process_steps.metadata.get('path', '')
        meta['process_step_count'] = len(process_steps.rows)
    return SourceEventTable(rows=rows, metadata=meta)


def process_step_summary(process_steps: Optional[ProcessStepTable]) -> Dict[str, object]:
    if process_steps is None:
        return {'has_process_steps': False, 'process_step_count': 0}
    return {
        'has_process_steps': True,
        'process_step_count': len(process_steps.rows),
        'step_names': [r.step_name for r in process_steps.rows],
    }
