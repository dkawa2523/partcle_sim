"""Bounded, result-only scientific summaries."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from os import PathLike

import numpy as np

from chamber_particles import open_result

ANALYSIS_REVISION = "result_fate_deposition_arrival_v2"

_LIFECYCLE_NAMES = {
    0: "pending",
    1: "active",
    2: "stuck",
    3: "escaped",
    4: "failed",
    5: "held",
}


@dataclass(slots=True)
class _ArrivalAccumulator:
    events: int = 0
    model_weight: float = 0.0
    weighted_time_s: float = 0.0
    first_time_s: float = float("inf")
    last_time_s: float = float("-inf")

    def add(
        self,
        *,
        events: int,
        model_weight: float,
        weighted_time_s: float,
        first_time_s: float,
        last_time_s: float,
    ) -> None:
        self.events += events
        self.model_weight += model_weight
        self.weighted_time_s += weighted_time_s
        self.first_time_s = min(self.first_time_s, first_time_s)
        self.last_time_s = max(self.last_time_s, last_time_s)


def summarize_result(
    path: str | PathLike[str],
    *,
    event_batch_rows: int = 4096,
) -> dict[str, object]:
    """Summarize particle fate and boundary arrivals without loading all events."""

    result = open_result(path)
    final = result.read_final()
    arrivals: dict[tuple[int, int, str], _ArrivalAccumulator] = {}
    for batch in result.iter_boundary_event_batches(batch_rows=event_batch_rows):
        _accumulate_arrivals(
            arrivals,
            batch.boundary_id,
            batch.material_id,
            batch.outcome,
            batch.time_s,
            batch.model_weight,
        )

    manifest = dict(result.manifest)
    manifest_json = json.dumps(
        manifest,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    arrival_rows = _arrival_rows(arrivals)
    return {
        "analysis_revision": ANALYSIS_REVISION,
        "parameters": {"event_batch_rows": event_batch_rows},
        "source_result": {
            "manifest_sha256": "sha256:" + hashlib.sha256(manifest_json).hexdigest(),
            "case_name": manifest.get("case_name"),
            "case_file_hash": manifest.get("case_file_hash"),
            "data_content_hash": manifest.get("data_content_hash"),
            "result_schema_version": manifest.get("result_schema_version"),
            "result_algorithm_revision": manifest.get("result_algorithm_revision"),
        },
        "fate": _fate_rows(final.lifecycle, final.model_weight),
        "deposition": [
            {
                "boundary_id": row["boundary_id"],
                "material_id": row["material_id"],
                "events": row["events"],
                "model_weight": row["model_weight"],
            }
            for row in arrival_rows
            if row["outcome"] == "stuck"
        ],
        "arrival": arrival_rows,
    }


def _fate_rows(lifecycle: np.ndarray, model_weight: np.ndarray) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for code in sorted(int(value) for value in np.unique(lifecycle)):
        selected = lifecycle == code
        rows.append(
            {
                "lifecycle": _LIFECYCLE_NAMES.get(code, f"unknown_{code}"),
                "particles": int(np.count_nonzero(selected)),
                "model_weight": float(np.sum(model_weight[selected], dtype=np.float64)),
            }
        )
    return rows


def _accumulate_arrivals(
    totals: dict[tuple[int, int, str], _ArrivalAccumulator],
    boundary_id: np.ndarray,
    material_id: np.ndarray,
    outcome: np.ndarray,
    time_s: np.ndarray,
    model_weight: np.ndarray,
) -> None:
    for outcome_value in np.unique(outcome):
        selected = outcome == outcome_value
        keys = np.column_stack((boundary_id[selected], material_id[selected]))
        unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
        selected_time = time_s[selected]
        selected_weight = model_weight[selected]
        counts = np.bincount(inverse)
        weights = np.bincount(inverse, weights=selected_weight)
        weighted_times = np.bincount(inverse, weights=selected_weight * selected_time)
        first = np.full(unique_keys.shape[0], np.inf, dtype=np.float64)
        last = np.full(unique_keys.shape[0], -np.inf, dtype=np.float64)
        np.minimum.at(first, inverse, selected_time)
        np.maximum.at(last, inverse, selected_time)
        for index, (boundary, material) in enumerate(unique_keys):
            key = (int(boundary), int(material), str(outcome_value))
            totals.setdefault(key, _ArrivalAccumulator()).add(
                events=int(counts[index]),
                model_weight=float(weights[index]),
                weighted_time_s=float(weighted_times[index]),
                first_time_s=float(first[index]),
                last_time_s=float(last[index]),
            )


def _arrival_rows(
    totals: dict[tuple[int, int, str], _ArrivalAccumulator],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (boundary_id, material_id, outcome), value in sorted(totals.items()):
        mean_time = value.weighted_time_s / value.model_weight if value.model_weight > 0.0 else None
        rows.append(
            {
                "boundary_id": boundary_id,
                "material_id": material_id,
                "outcome": outcome,
                "events": value.events,
                "model_weight": value.model_weight,
                "first_time_s": value.first_time_s,
                "last_time_s": value.last_time_s,
                "weighted_mean_time_s": mean_time,
            }
        )
    return rows


__all__ = ["ANALYSIS_REVISION", "summarize_result"]
