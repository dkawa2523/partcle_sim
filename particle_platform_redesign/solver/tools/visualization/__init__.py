"""Small dependency-free SVG views over one completed result."""

from __future__ import annotations

import hashlib
import heapq
import html
import json
import struct
from collections.abc import Callable, Iterable
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

from chamber_particles import open_result

VISUALIZATION_REVISION = "result_trajectory_boundary_svg_v3"

type _EventPoint = tuple[float, float, str, int]
type _RankedEvent = tuple[int, float, int, int, _EventPoint]

_WIDTH = 900
_HEIGHT = 620
_MARGIN = 70
_OUTCOME_COLOR = {
    "reflected": "#2563eb",
    "stuck": "#dc2626",
    "escaped": "#d97706",
    "held": "#7c3aed",
}


def render_result(
    path: str | PathLike[str],
    output_directory: str | PathLike[str],
    *,
    particle_id: int | None = None,
    event_batch_rows: int = 4096,
    max_event_points: int = 5000,
) -> dict[str, object]:
    """Render one representative trajectory and a bounded boundary-event map."""

    if max_event_points <= 0:
        raise ValueError("max_event_points must be positive")
    result = open_result(path)
    source = result.path.resolve()
    destination = Path(output_directory).expanduser().resolve()
    if destination == source or source in destination.parents:
        raise ValueError("visualization output must be outside the source result directory")
    destination.mkdir(parents=True, exist_ok=True)

    trajectory, selected_particle = _trajectory_points(result.iter_probes(), particle_id)
    trajectory_source = "probes"
    if not trajectory:
        trajectory, selected_particle = _trajectory_points(result.iter_frames(), particle_id)
        trajectory_source = "frames"
    if not trajectory or selected_particle is None:
        raise ValueError("result has no saved trajectory for the requested particle")

    counts = result.manifest.get("counts")
    event_count = int(counts.get("boundary_events", 0)) if isinstance(counts, dict) else 0
    event_points, trajectory_events = _boundary_points(
        result.iter_boundary_event_batches(batch_rows=event_batch_rows),
        selected_particle,
        max_event_points,
    )

    manifest_json = json.dumps(
        dict(result.manifest),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    source_digest = "sha256:" + hashlib.sha256(manifest_json).hexdigest()
    coordinate_system = str(result.manifest.get("data_coordinate_system", "cartesian_xy"))
    trajectory_path = destination / "trajectory.svg"
    event_path = destination / "boundary_events.svg"
    trajectory_path.write_text(
        _trajectory_svg(
            trajectory,
            trajectory_events,
            selected_particle,
            coordinate_system,
            source_digest,
        ),
        encoding="utf-8",
    )
    event_path.write_text(
        _event_svg(event_points, event_count, coordinate_system, source_digest),
        encoding="utf-8",
    )
    metadata_path = destination / "visualization.json"
    report: dict[str, object] = {
        "visualization_revision": VISUALIZATION_REVISION,
        "parameters": {
            "event_batch_rows": event_batch_rows,
            "max_event_points": max_event_points,
            "requested_particle_id": particle_id,
        },
        "source_manifest_sha256": source_digest,
        "trajectory_source": trajectory_source,
        "particle_id": selected_particle,
        "trajectory_points": len(trajectory),
        "boundary_events": event_count,
        "rendered_boundary_points": len(event_points),
        "event_selection": {
            "boundary_map": "smallest BLAKE2b-128 ranks of little-endian time_s/particle_id/event_ordinal",
            "trajectory": "first physical events by time_s/particle_id/event_ordinal",
            "display_order": "time_s/particle_id/event_ordinal",
            "max_points_per_view": max_event_points,
        },
        "trajectory_svg": str(trajectory_path),
        "boundary_events_svg": str(event_path),
        "metadata_json": str(metadata_path),
    }
    metadata_path.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _retain_event(heap: list[_RankedEvent], entry: _RankedEvent, limit: int) -> None:
    """Keep the smallest physical ranks using a bounded reversed-rank heap."""
    if len(heap) < limit:
        heapq.heappush(heap, entry)
    elif entry > heap[0]:
        heapq.heapreplace(heap, entry)


def _boundary_points(
    batches: Iterable[Any], selected_particle: int, limit: int
) -> tuple[list[_EventPoint], list[tuple[float, float, str]]]:
    sampled: list[_RankedEvent] = []
    selected: list[_RankedEvent] = []
    for batch in batches:
        for index in range(batch.particle_id.size):
            time_s = float(batch.time_s[index])
            particle_id = int(batch.particle_id[index])
            ordinal = int(batch.event_ordinal[index])
            identity = struct.pack("<dqI", time_s, particle_id, ordinal)
            rank = int.from_bytes(hashlib.blake2b(identity, digest_size=16).digest(), "little")
            point = (
                float(batch.position_m[index, 0]),
                float(batch.position_m[index, 1]),
                str(batch.outcome[index]),
                int(batch.boundary_id[index]),
            )
            key = (-time_s, -particle_id, -ordinal)
            _retain_event(sampled, (-rank, *key, point), limit)
            if particle_id == selected_particle:
                _retain_event(selected, (0, *key, point), limit)
    return (
        [entry[4] for entry in sorted(sampled, key=lambda entry: entry[1:4], reverse=True)],
        [entry[4][:3] for entry in sorted(selected, key=lambda entry: entry[1:4], reverse=True)],
    )


def _trajectory_points(
    frames: Iterable[Any], requested_particle: int | None
) -> tuple[list[tuple[float, float, float]], int | None]:
    points: list[tuple[float, float, float]] = []
    selected = requested_particle
    for frame in frames:
        if selected is None and frame.particle_id.size:
            selected = int(frame.particle_id[0])
        if selected is None:
            continue
        matches = np.flatnonzero(frame.particle_id == selected)
        if matches.size:
            row = int(matches[0])
            points.append(
                (
                    float(frame.time_s),
                    float(frame.position_m[row, 0]),
                    float(frame.position_m[row, 1]),
                )
            )
    return points, selected


def _trajectory_svg(
    trajectory: list[tuple[float, float, float]],
    events: list[tuple[float, float, str]],
    particle_id: int,
    coordinate_system: str,
    source_digest: str,
) -> str:
    spatial = [(x, y) for _, x, y in trajectory] + [(x, y) for x, y, _ in events]
    transform = _plot_transform(spatial)
    polyline = " ".join(
        f"{transform(x, y)[0]:.3f},{transform(x, y)[1]:.3f}" for x, y in spatial[: len(trajectory)]
    )
    event_marks = "".join(
        _circle(transform(x, y), 5.0, _OUTCOME_COLOR.get(outcome, "#6b7280"), outcome)
        for x, y, outcome in events
    )
    title = f"Representative trajectory: particle {particle_id}"
    body = (
        f'<polyline points="{polyline}" fill="none" stroke="#111827" stroke-width="2"/>'
        + event_marks
    )
    return _svg_document(title, body, coordinate_system, source_digest)


def _event_svg(
    events: list[tuple[float, float, str, int]],
    event_count: int,
    coordinate_system: str,
    source_digest: str,
) -> str:
    transform = _plot_transform([(x, y) for x, y, _, _ in events])
    marks = "".join(
        _circle(
            transform(x, y),
            3.5,
            _OUTCOME_COLOR.get(outcome, "#6b7280"),
            f"{outcome}; boundary_id={boundary_id}",
        )
        for x, y, outcome, boundary_id in events
    )
    if not marks:
        marks = '<text x="450" y="310" text-anchor="middle">No boundary events</text>'
    title = f"Boundary events: {len(events)} rendered of {event_count}"
    return _svg_document(title, marks, coordinate_system, source_digest)


def _plot_transform(
    points: list[tuple[float, float]],
) -> Callable[[float, float], tuple[float, float]]:
    if points:
        x_values, y_values = zip(*points, strict=True)
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)
    else:
        x_min = y_min = 0.0
        x_max = y_max = 1.0
    x_span = max(x_max - x_min, 1.0e-30)
    y_span = max(y_max - y_min, 1.0e-30)
    padding_x = 0.05 * x_span
    padding_y = 0.05 * y_span
    x_min -= padding_x
    x_span += 2.0 * padding_x
    y_min -= padding_y
    y_span += 2.0 * padding_y

    def transform(x: float, y: float) -> tuple[float, float]:
        plot_width = _WIDTH - 2 * _MARGIN
        plot_height = _HEIGHT - 2 * _MARGIN
        return (
            _MARGIN + (x - x_min) * plot_width / x_span,
            _HEIGHT - _MARGIN - (y - y_min) * plot_height / y_span,
        )

    return transform


def _circle(point: tuple[float, float], radius: float, color: str, label: str) -> str:
    x, y = point
    return (
        f'<circle cx="{x:.3f}" cy="{y:.3f}" r="{radius:.1f}" fill="{color}">'
        f"<title>{html.escape(label)}</title></circle>"
    )


def _svg_document(title: str, body: str, coordinate_system: str, source_digest: str) -> str:
    x_label, y_label = (
        ("r [m]", "z [m]") if coordinate_system == "axisymmetric_rz" else ("x [m]", "y [m]")
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{_WIDTH}" height="{_HEIGHT}" viewBox="0 0 {_WIDTH} {_HEIGHT}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{_WIDTH / 2:.0f}" y="30" text-anchor="middle" font-size="18">{html.escape(title)}</text>
  <rect x="{_MARGIN}" y="{_MARGIN}" width="{_WIDTH - 2 * _MARGIN}" height="{_HEIGHT - 2 * _MARGIN}" fill="none" stroke="#9ca3af"/>
  {body}
  <text x="{_WIDTH / 2:.0f}" y="{_HEIGHT - 18}" text-anchor="middle">{x_label}</text>
  <text x="20" y="{_HEIGHT / 2:.0f}" text-anchor="middle" transform="rotate(-90 20 {_HEIGHT / 2:.0f})">{y_label}</text>
  <text x="{_MARGIN}" y="{_HEIGHT - 4}" font-size="8" fill="#6b7280">{source_digest}</text>
</svg>
"""


__all__ = ["VISUALIZATION_REVISION", "render_result"]
