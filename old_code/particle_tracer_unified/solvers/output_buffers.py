from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class StepSummaryBuffer:
    time_s: list[float] = field(default_factory=list)
    step_name: list[str] = field(default_factory=list)
    segment_name: list[str] = field(default_factory=list)
    active_count: list[int] = field(default_factory=list)
    released_count: list[int] = field(default_factory=list)
    stuck_count: list[int] = field(default_factory=list)
    frozen_count: list[int] = field(default_factory=list)
    absorbed_count: list[int] = field(default_factory=list)
    contact_sliding_count: list[int] = field(default_factory=list)
    escaped_count: list[int] = field(default_factory=list)
    stopped_count: list[int] = field(default_factory=list)
    save_positions_enabled: list[int] = field(default_factory=list)
    write_wall_events_enabled: list[int] = field(default_factory=list)
    write_diagnostics_enabled: list[int] = field(default_factory=list)
    valid_mask_violation_count_step: list[int] = field(default_factory=list)
    valid_mask_mixed_stencil_count_step: list[int] = field(default_factory=list)
    valid_mask_hard_invalid_count_step: list[int] = field(default_factory=list)
    invalid_mask_stopped_count_step: list[int] = field(default_factory=list)

    def append(
        self,
        *,
        time_s: float,
        active_count: int,
        released_count: int,
        stopped_count: int = 0,
        step_name: str = "",
        segment_name: str = "",
        stuck_count: int = 0,
        frozen_count: int = 0,
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
        self.frozen_count.append(int(frozen_count))
        self.absorbed_count.append(int(absorbed_count))
        self.contact_sliding_count.append(int(contact_sliding_count))
        self.escaped_count.append(int(escaped_count))
        self.stopped_count.append(int(stopped_count))
        self.save_positions_enabled.append(int(save_positions_enabled))
        self.write_wall_events_enabled.append(int(write_wall_events_enabled))
        self.write_diagnostics_enabled.append(int(write_diagnostics_enabled))
        self.valid_mask_violation_count_step.append(
            int(valid_mask_violation_count_step)
        )
        self.valid_mask_mixed_stencil_count_step.append(
            int(valid_mask_mixed_stencil_count_step)
        )
        self.valid_mask_hard_invalid_count_step.append(
            int(valid_mask_hard_invalid_count_step)
        )
        self.invalid_mask_stopped_count_step.append(
            int(invalid_mask_stopped_count_step)
        )

    def as_runtime_step_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for i in range(self.sample_count):
            rows.append(
                {
                    "time_s": float(self.time_s[i]),
                    "step_name": str(self.step_name[i]),
                    "segment_name": str(self.segment_name[i]),
                    "released_count": int(self.released_count[i]),
                    "active_count": int(self.active_count[i]),
                    "stuck_count": int(self.stuck_count[i]),
                    "absorbed_count": int(self.absorbed_count[i]),
                    "contact_sliding_count": int(self.contact_sliding_count[i]),
                    "escaped_count": int(self.escaped_count[i]),
                    "save_positions_enabled": int(self.save_positions_enabled[i]),
                    "write_wall_events_enabled": int(self.write_wall_events_enabled[i]),
                    "write_diagnostics_enabled": int(self.write_diagnostics_enabled[i]),
                    "valid_mask_violation_count_step": int(
                        self.valid_mask_violation_count_step[i]
                    ),
                    "valid_mask_mixed_stencil_count_step": int(
                        self.valid_mask_mixed_stencil_count_step[i]
                    ),
                    "valid_mask_hard_invalid_count_step": int(
                        self.valid_mask_hard_invalid_count_step[i]
                    ),
                    "invalid_mask_stopped_count_step": int(
                        self.invalid_mask_stopped_count_step[i]
                    ),
                    "frozen_count": int(self.frozen_count[i]),
                }
            )
        return rows

    @property
    def sample_count(self) -> int:
        return len(self.time_s)


@dataclass
class DebugBuffers:
    """Own every row-oriented payload that exists only in debug mode."""

    trajectory_positions: list[np.ndarray] = field(default_factory=list)
    save_frames: list[dict[str, object]] = field(default_factory=list)
    wall_events: list[dict[str, object]] = field(default_factory=list)
    max_hit_events: list[dict[str, object]] = field(default_factory=list)
    step_summary: StepSummaryBuffer = field(default_factory=StepSummaryBuffer)

    def summary(self) -> dict[str, int | str]:
        return {
            "output_mode": "debug",
            "output_debug_enabled": 1,
            "step_summary_buffer_enabled": 1,
            "step_summary_count": self.step_summary.sample_count,
        }


__all__ = ("DebugBuffers", "StepSummaryBuffer")
