from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .runtime_plan import (
    OutputPlan,
    OUTPUT_MODE_DEBUG,
    OUTPUT_MODE_MINIMAL,
)


@dataclass
class StepSummaryBuffer:
    time_s: List[float] = field(default_factory=list)
    step_name: List[str] = field(default_factory=list)
    segment_name: List[str] = field(default_factory=list)
    active_count: List[int] = field(default_factory=list)
    released_count: List[int] = field(default_factory=list)
    stuck_count: List[int] = field(default_factory=list)
    absorbed_count: List[int] = field(default_factory=list)
    contact_sliding_count: List[int] = field(default_factory=list)
    escaped_count: List[int] = field(default_factory=list)
    stopped_count: List[int] = field(default_factory=list)
    save_positions_enabled: List[int] = field(default_factory=list)
    write_wall_events_enabled: List[int] = field(default_factory=list)
    write_diagnostics_enabled: List[int] = field(default_factory=list)
    valid_mask_violation_count_step: List[int] = field(default_factory=list)
    valid_mask_mixed_stencil_count_step: List[int] = field(default_factory=list)
    valid_mask_hard_invalid_count_step: List[int] = field(default_factory=list)
    invalid_mask_stopped_count_step: List[int] = field(default_factory=list)

    def append(
        self,
        *,
        time_s: float,
        active_count: int,
        released_count: int,
        stopped_count: int = 0,
        step_name: str = '',
        segment_name: str = '',
        stuck_count: int = 0,
        absorbed_count: int = 0,
        contact_sliding_count: int = 0,
        escaped_count: int = 0,
        save_positions_enabled: int = 0,
        write_wall_events_enabled: int = 0,
        write_diagnostics_enabled: int = 0,
        valid_mask_violation_count_step: int = 0,
        valid_mask_mixed_stencil_count_step: int = 0,
        valid_mask_hard_invalid_count_step: int = 0,
        invalid_mask_stopped_count_step: int = 0,
    ) -> None:
        self.time_s.append(float(time_s))
        self.step_name.append(str(step_name))
        self.segment_name.append(str(segment_name))
        self.active_count.append(int(active_count))
        self.released_count.append(int(released_count))
        self.stuck_count.append(int(stuck_count))
        self.absorbed_count.append(int(absorbed_count))
        self.contact_sliding_count.append(int(contact_sliding_count))
        self.escaped_count.append(int(escaped_count))
        self.stopped_count.append(int(stopped_count))
        self.save_positions_enabled.append(int(save_positions_enabled))
        self.write_wall_events_enabled.append(int(write_wall_events_enabled))
        self.write_diagnostics_enabled.append(int(write_diagnostics_enabled))
        self.valid_mask_violation_count_step.append(int(valid_mask_violation_count_step))
        self.valid_mask_mixed_stencil_count_step.append(int(valid_mask_mixed_stencil_count_step))
        self.valid_mask_hard_invalid_count_step.append(int(valid_mask_hard_invalid_count_step))
        self.invalid_mask_stopped_count_step.append(int(invalid_mask_stopped_count_step))

    def as_runtime_step_rows(self) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        for i in range(self.sample_count):
            rows.append(
                {
                    'time_s': float(self.time_s[i]),
                    'step_name': str(self.step_name[i]),
                    'segment_name': str(self.segment_name[i]),
                    'released_count': int(self.released_count[i]),
                    'active_count': int(self.active_count[i]),
                    'stuck_count': int(self.stuck_count[i]),
                    'absorbed_count': int(self.absorbed_count[i]),
                    'contact_sliding_count': int(self.contact_sliding_count[i]),
                    'escaped_count': int(self.escaped_count[i]),
                    'save_positions_enabled': int(self.save_positions_enabled[i]),
                    'write_wall_events_enabled': int(self.write_wall_events_enabled[i]),
                    'write_diagnostics_enabled': int(self.write_diagnostics_enabled[i]),
                    'valid_mask_violation_count_step': int(self.valid_mask_violation_count_step[i]),
                    'valid_mask_mixed_stencil_count_step': int(
                        self.valid_mask_mixed_stencil_count_step[i]
                    ),
                    'valid_mask_hard_invalid_count_step': int(self.valid_mask_hard_invalid_count_step[i]),
                    'invalid_mask_stopped_count_step': int(self.invalid_mask_stopped_count_step[i]),
                }
            )
        return rows

    @property
    def sample_count(self) -> int:
        return int(len(self.time_s))


@dataclass(init=False)
class RuntimeBuffers:
    output: OutputPlan
    step_summary: StepSummaryBuffer | None = None

    def __init__(
        self,
        output: OutputPlan,
        *,
        step_summary: StepSummaryBuffer | None = None,
    ) -> None:
        self.output = output
        self.step_summary = (
            step_summary
            if step_summary is not None
            else StepSummaryBuffer() if output.write_step_summary else None
        )

    @property
    def minimal(self) -> bool:
        return self.output.mode == OUTPUT_MODE_MINIMAL

    @property
    def debug(self) -> bool:
        return self.output.mode == OUTPUT_MODE_DEBUG

    def summary(self) -> Dict[str, int | str]:
        out: Dict[str, int | str] = {
            'output_mode': self.output.mode,
            'output_minimal_enabled': int(self.minimal),
            'output_debug_enabled': int(self.debug),
            'step_summary_buffer_enabled': int(self.step_summary is not None),
        }
        if self.step_summary is not None:
            out['step_summary_count'] = self.step_summary.sample_count
        return out
