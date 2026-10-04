from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ForceContribution:
    name: str
    acceleration: np.ndarray
    force: np.ndarray | None = None
    enabled: bool = True
    physical_quantity: str = "acceleration"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        acc = np.asarray(self.acceleration, dtype=np.float64)
        row: dict[str, Any] = {
            "name": str(self.name),
            "enabled": int(bool(self.enabled)),
            "physical_quantity": str(self.physical_quantity),
            "accel_norm": float(np.linalg.norm(acc)) if acc.size else float("nan"),
        }
        for idx, component in enumerate(("x", "y", "z")[: acc.size]):
            row[f"accel_{component}"] = float(acc[idx])
        if self.force is not None:
            force = np.asarray(self.force, dtype=np.float64)
            row["force_norm"] = (
                float(np.linalg.norm(force)) if force.size else float("nan")
            )
            for idx, component in enumerate(("x", "y", "z")[: force.size]):
                row[f"force_{component}"] = float(force[idx])
        for key, value in dict(self.metadata).items():
            row[f"metadata_{key}"] = value
        return row


__all__ = ("ForceContribution",)
