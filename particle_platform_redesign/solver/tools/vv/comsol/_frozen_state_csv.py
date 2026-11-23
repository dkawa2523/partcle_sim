"""Read the shared active-state CSV envelope for local equation comparisons."""

from __future__ import annotations

import csv
import math
from collections import Counter
from collections.abc import Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def _particle_id(text: str, label: str) -> int:
    try:
        value = Decimal(text)
    except InvalidOperation as error:
        raise ValueError(f"{label}: particle_id must be an exact int64 integer") from error
    if not value.is_finite() or value != value.to_integral_value():
        raise ValueError(f"{label}: particle_id must be an exact int64 integer")
    bounds = np.iinfo(np.int64)
    if value < int(bounds.min) or value > int(bounds.max):
        raise ValueError(f"{label}: particle_id is outside the int64 range")
    return int(value)


def read_active_states(
    path: Path, required_columns: Sequence[str]
) -> tuple[NDArray[np.int64], dict[str, NDArray[np.float64]]]:
    """Preserve exact row IDs separately from finite physical float columns."""

    particle_ids: list[int] = []
    values: dict[str, list[float]] = {
        name: [] for name in required_columns if name != "particle_id"
    }
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or ()
        duplicates = sorted(name for name, count in Counter(columns).items() if count > 1)
        if duplicates:
            raise ValueError(f"{path}: duplicate column names: {duplicates}")
        missing = sorted(set(required_columns).difference(columns))
        if missing:
            raise ValueError(f"{path}: required columns are missing: {missing}")
        for row_number, row in enumerate(reader, start=2):
            label = f"{path}:{row_number}"
            active = float(row["active_state_flag"])
            if active not in (0.0, 1.0):
                raise ValueError(f"{label}: active_state_flag is not 0 or 1")
            if active == 0.0:
                continue
            particle_ids.append(_particle_id(row["particle_id"], label))
            for name, column in values.items():
                value = float(row[name])
                if not math.isfinite(value):
                    raise ValueError(f"{label}: {name} is not finite")
                column.append(value)
    if not particle_ids:
        raise ValueError(f"{path}: no active rows")
    return np.asarray(particle_ids, dtype=np.int64), {
        name: np.asarray(column, dtype=np.float64) for name, column in values.items()
    }
