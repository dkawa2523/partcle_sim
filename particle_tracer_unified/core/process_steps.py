from __future__ import annotations

from typing import Dict, Optional

from .datamodel import ProcessStepTable


def process_step_control_summary(process_steps: Optional[ProcessStepTable]) -> Dict[str, object]:
    if process_steps is None:
        return {'has_process_steps': False, 'step_count': 0}
    return {
        'has_process_steps': True,
        'step_count': len(process_steps.rows),
        'step_names': [r.step_name for r in process_steps.rows],
        'segments': [r.output_segment_name or r.step_name for r in process_steps.rows],
    }
