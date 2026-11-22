"""Bounded SVG result views with immutable geometry and explicit trajectory detail."""

from __future__ import annotations

import hashlib
import heapq
import html
import json
import math
import struct
from collections.abc import Callable, Iterable
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

from chamber_particles import open_result
from chamber_particles.output import ResultContext

from ..selection import select_particles

VISUALIZATION_REVISION = "result_context_population_svg_v4"
_WIDTH, _HEIGHT, _MARGIN = 900, 620, 70
_COLORS = {0: "#94a3b8", 1: "#2563eb", 2: "#dc2626", 3: "#d97706", 4: "#111827", 5: "#7c3aed"}
_OUTCOME_COLOR = {
    "reflected": "#2563eb",
    "stuck": "#dc2626",
    "escaped": "#d97706",
    "held": "#7c3aed",
    "transferred": "#0891b2",
}
type _EventPoint = tuple[float, float, str, int]
type _RankedEvent = tuple[int, float, int, int, _EventPoint]
type _Point = tuple[float, float, float, int] | None


def render_result(
    path: str | PathLike[str],
    output_directory: str | PathLike[str],
    *,
    particle_id: int | None = None,
    event_batch_rows: int = 4096,
    max_event_points: int = 5000,
    particle_ids: tuple[int, ...] | None = None,
    source_ids: tuple[int, ...] | None = None,
    max_particle_points: int = 5000,
    max_trajectory_points: int = 5000,
    detail: bool = False,
) -> dict[str, object]:
    """Render all selected particles as bounded occupancy; detail is explicit.

    Saved-position chords are broken at missing/nonfinite/terminal state and
    recorded interactions. They do not reconstruct unresolved stochastic paths.
    """
    _validate_limits((max_event_points, max_particle_points, max_trajectory_points))
    result = open_result(path)
    context, final = result.read_context(), result.read_final()
    selected = select_particles(final, context, source_ids=source_ids, particle_ids=particle_ids)
    ids = final.particle_id[selected]
    destination = Path(output_directory).expanduser().resolve()
    if destination == result.path or result.path in destination.parents:
        raise ValueError("visualization output must be outside the source result directory")
    destination.mkdir(parents=True, exist_ok=True)
    selected_particle = _detail_particle(ids, particle_id, detail)
    points, trajectory_source, truncated = _detail_points(
        result, selected_particle, max_trajectory_points
    )
    event_points, trajectory_events, event_total, breaks = _boundary_points(
        result.iter_boundary_event_batches(batch_rows=event_batch_rows),
        ids,
        selected_particle,
        max_event_points,
        points,
    )
    manifest_json = json.dumps(
        dict(result.manifest), allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode()
    digest = "sha256:" + hashlib.sha256(manifest_json).hexdigest()
    transform = _plot_transform(context)
    population_body, population_info = _population_body(
        final, selected, context, transform, max_particle_points
    )
    population_body += _subtitle(
        f"{population_info['finite_valid_positions']} finite positions / {population_info['selected_particles']} selected particles; {population_info['cells_per_axis']} x {population_info['cells_per_axis']} bins; dominant lifecycle color"
    )
    files = {
        "population_svg": destination / "population.svg",
        "trajectory_svg": destination / "trajectory.svg",
        "boundary_events_svg": destination / "boundary_events.svg",
    }
    files["population_svg"].write_text(
        _svg_document(
            "Final particle occupancy (binned computational counts)",
            population_body,
            context,
            transform,
            digest,
        ),
        encoding="utf-8",
    )
    trajectory_title = (
        f"Trajectory detail: particle {selected_particle}"
        if selected_particle is not None
        else "Trajectory detail not requested"
    )
    files["trajectory_svg"].write_text(
        _svg_document(
            trajectory_title,
            _trajectory_body(points, breaks, trajectory_events, transform),
            context,
            transform,
            digest,
        ),
        encoding="utf-8",
    )
    event_body = "".join(
        _circle(
            transform(x, y),
            3.5,
            _OUTCOME_COLOR.get(outcome, "#6b7280"),
            f"{outcome}; boundary_id={boundary}",
        )
        for x, y, outcome, boundary in event_points
    )
    if not event_body:
        event_body = _empty_message("No boundary interactions for this selection")
    files["boundary_events_svg"].write_text(
        _svg_document(
            f"Boundary interactions: {len(event_points)} rendered of {event_total}",
            event_body,
            context,
            transform,
            digest,
            legend="blue reflection; red stop; orange escape; purple hold; cyan periodic transfer",
        ),
        encoding="utf-8",
    )
    report = {
        "visualization_revision": VISUALIZATION_REVISION,
        "parameters": {
            "event_batch_rows": event_batch_rows,
            "max_event_points": max_event_points,
            "requested_particle_id": particle_id,
            "particle_ids": particle_ids,
            "source_ids": source_ids,
            "max_particle_points": max_particle_points,
            "max_trajectory_points": max_trajectory_points,
            "detail": detail,
        },
        "source_manifest_sha256": digest,
        "context": dict(result.manifest)["context"],
        "particle_id": selected_particle,
        "trajectory_source": trajectory_source,
        "trajectory_points": sum(point is not None for point in points),
        "trajectory_truncated": truncated,
        "boundary_events": event_total,
        "rendered_boundary_points": len(event_points),
        "population": population_info,
        "event_selection": {
            "boundary_map": "smallest BLAKE2b-128 ranks of time_s/particle_id/event_ordinal",
            "trajectory": "first physical events by time_s/particle_id/event_ordinal",
            "display_order": "time_s/particle_id/event_ordinal",
            "max_points_per_view": max_event_points,
        },
        "trajectory_meaning": "saved positions only; gaps and recorded interactions break chords; no reconstructed Brownian path",
        **{name: str(file) for name, file in files.items()},
        "metadata_json": str(destination / "visualization.json"),
    }
    (destination / "visualization.json").write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _validate_limits(values: tuple[int, ...]) -> None:
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("display point limits must be positive integers")


def _detail_particle(ids: np.ndarray, requested: int | None, detail: bool) -> int | None:
    if requested is not None:
        if isinstance(requested, bool) or not isinstance(requested, int) or requested not in ids:
            raise ValueError("requested particle_id is not in the selected population")
        return requested
    return int(ids[0]) if detail and ids.size else None


def _detail_points(
    result: Any, particle_id: int | None, limit: int
) -> tuple[list[_Point], str, bool]:
    if particle_id is None:
        return [], "unrequested", False
    points, truncated = _trajectory_points(result.iter_probes(), particle_id, limit)
    if any(point is not None for point in points):
        return points, "probes", truncated
    points, truncated = _trajectory_points(result.iter_frames(), particle_id, limit)
    return points, "frames" if points else "unavailable", truncated


def _trajectory_points(
    frames: Iterable[Any], particle_id: int, limit: int
) -> tuple[list[_Point], bool]:
    points: list[_Point] = []
    truncated = False
    for frame in frames:
        rows = np.flatnonzero(frame.particle_id == particle_id)
        point = None
        if rows.size:
            row = int(rows[0])
            position = frame.position_m[row]
            life = int(frame.lifecycle[row])
            if life in (1, 2, 5) and np.isfinite(position).all():
                point = (float(frame.time_s), float(position[0]), float(position[1]), life)
        if len(points) < limit:
            points.append(point)
        else:
            truncated = True
    return points, truncated


def _retain_event(heap: list[_RankedEvent], entry: _RankedEvent, limit: int) -> None:
    if len(heap) < limit:
        heapq.heappush(heap, entry)
    elif entry > heap[0]:
        heapq.heapreplace(heap, entry)


def _boundary_points(
    batches: Iterable[Any], ids: np.ndarray, particle: int | None, limit: int, points: list[_Point]
) -> tuple[list[_EventPoint], list[_EventPoint], int, set[int]]:
    sampled: list[_RankedEvent] = []
    detail: list[_RankedEvent] = []
    finite_times = [(i, point[0]) for i, point in enumerate(points) if point is not None]
    time_values = np.asarray([time for _, time in finite_times])
    breaks: set[int] = set()
    count = 0
    for batch in batches:
        for row in np.flatnonzero(np.isin(batch.particle_id, ids)):
            count += 1
            time, pid, ordinal = (
                float(batch.time_s[row]),
                int(batch.particle_id[row]),
                int(batch.event_ordinal[row]),
            )
            position = batch.position_m[row]
            if not np.isfinite(position).all():
                continue
            rank = int.from_bytes(
                hashlib.blake2b(struct.pack("<dqI", time, pid, ordinal), digest_size=16).digest(),
                "little",
            )
            point = (
                float(position[0]),
                float(position[1]),
                str(batch.outcome[row]),
                int(batch.boundary_id[row]),
            )
            key = (-time, -pid, -ordinal)
            _retain_event(sampled, (-rank, *key, point), limit)
            if pid == particle:
                _retain_event(detail, (0, *key, point), limit)
                after = int(np.searchsorted(time_values, time, side="left"))
                if after < len(finite_times):
                    breaks.add(finite_times[after][0])

    def ordered(heap: list[_RankedEvent]) -> list[_EventPoint]:
        return [entry[4] for entry in sorted(heap, key=lambda entry: entry[1:4], reverse=True)]

    return ordered(sampled), ordered(detail), count, breaks


def _population_body(
    final: Any,
    selected: np.ndarray,
    context: ResultContext,
    transform: Callable[[float, float], tuple[float, float]],
    limit: int,
) -> tuple[str, dict[str, object]]:
    valid = (
        selected
        & (final.kinematics_valid == 1)
        & np.isfinite(final.position_m).all(axis=1)
        & np.isin(final.lifecycle, (1, 2, 5))
    )
    bounds = context.geometry.nodes_m
    lower, upper = bounds.min(axis=0), bounds.max(axis=0)
    count = max(1, int(math.sqrt(limit)))
    edges = [np.linspace(lower[i], upper[i], count + 1) for i in range(2)]
    histograms = [
        np.histogram2d(
            final.position_m[valid & (final.lifecycle == code), 0],
            final.position_m[valid & (final.lifecycle == code), 1],
            bins=edges,
        )[0]
        for code in range(6)
    ]
    total = np.sum(histograms, axis=0)
    marks = []
    for i, j in zip(*np.nonzero(total), strict=True):
        x0, y0 = transform(float(edges[0][i]), float(edges[1][j + 1]))
        x1, y1 = transform(float(edges[0][i + 1]), float(edges[1][j]))
        dominant = int(np.argmax([h[i, j] for h in histograms]))
        label = "; ".join(
            f"lifecycle {code}: {int(h[i, j])}" for code, h in enumerate(histograms) if h[i, j]
        )
        opacity = 0.25 + 0.75 * math.log1p(total[i, j]) / math.log1p(float(total.max()))
        marks.append(
            f'<rect x="{x0:.3f}" y="{y0:.3f}" width="{x1 - x0:.3f}" height="{y1 - y0:.3f}" fill="{_COLORS[dominant]}" opacity="{opacity:.4f}"><title>{html.escape(label)}</title></rect>'
        )
    info: dict[str, object] = {
        "selected_particles": int(selected.sum()),
        "finite_valid_positions": int(valid.sum()),
        "omitted_nonspatial_or_pending": int(np.count_nonzero(selected & ~valid)),
        "binned_positions": int(total.sum()),
        "outside_geometry_bounds": int(valid.sum() - total.sum()),
        "rendered_cells": len(marks),
        "cells_per_axis": count,
        "meaning": "all finite active/stuck/held final positions binned; dominant lifecycle color; computational counts, not concentration",
    }
    return "".join(marks) if marks else _empty_message(
        "No finite active or retained particle positions"
    ), info


def _trajectory_body(
    points: list[_Point],
    breaks: set[int],
    events: list[_EventPoint],
    transform: Callable[[float, float], tuple[float, float]],
) -> str:
    marks = []
    previous: _Point = None
    for index, point in enumerate(points):
        if point is not None:
            _, x, y, life = point
            if previous is not None and index not in breaks and previous[3] == life == 1:
                x0, y0 = transform(previous[1], previous[2])
                x1, y1 = transform(x, y)
                marks.append(
                    f'<line x1="{x0:.3f}" y1="{y0:.3f}" x2="{x1:.3f}" y2="{y1:.3f}" stroke="#111827"/>'
                )
            marks.append(
                _circle(
                    transform(x, y),
                    2.2,
                    _COLORS[life],
                    f"saved state t={point[0]:.9g} s; lifecycle={life}",
                )
            )
        previous = point
    marks.extend(
        _circle(transform(x, y), 5, _OUTCOME_COLOR.get(outcome, "#6b7280"), outcome)
        for x, y, outcome, _ in events
    )
    return "".join(marks) if marks else _empty_message("No finite saved trajectory detail")


def _plot_transform(context: ResultContext) -> Callable[[float, float], tuple[float, float]]:
    lower, upper = context.geometry.nodes_m.min(axis=0), context.geometry.nodes_m.max(axis=0)
    center = (lower + upper) / 2
    span = upper - lower
    scale = min((_WIDTH - 2 * _MARGIN) / (1.1 * span[0]), (_HEIGHT - 2 * _MARGIN) / (1.1 * span[1]))

    def transform(x: float, y: float) -> tuple[float, float]:
        return (_WIDTH / 2 + (x - center[0]) * scale, _HEIGHT / 2 - (y - center[1]) * scale)

    return transform


def _geometry(
    context: ResultContext, transform: Callable[[float, float], tuple[float, float]]
) -> str:
    marks = []
    boundary = context.geometry.boundary
    for row, nodes in enumerate(boundary.line2):
        a, b = context.geometry.nodes_m[nodes]
        x0, y0 = transform(float(a[0]), float(a[1]))
        x1, y1 = transform(float(b[0]), float(b[1]))
        axis = context.axis_policy == "rz_chart_fold" and a[0] == b[0] == 0
        style = ' stroke-dasharray="5 4"' if axis else ""
        label = f"{context.geometry.group_names[int(boundary.group_id[row])]}; boundary_id={int(boundary.boundary_id[row])}; material_id={int(boundary.material_id[row])}"
        marks.append(
            f'<line x1="{x0:.3f}" y1="{y0:.3f}" x2="{x1:.3f}" y2="{y1:.3f}" stroke="#64748b" stroke-width="1.2"{style}><title>{html.escape(label)}</title></line>'
        )
    return "".join(marks)


def _circle(point: tuple[float, float], radius: float, color: str, label: str) -> str:
    return f'<circle cx="{point[0]:.3f}" cy="{point[1]:.3f}" r="{radius:.1f}" fill="{color}"><title>{html.escape(label)}</title></circle>'


def _empty_message(message: str) -> str:
    return f'<text x="450" y="310" text-anchor="middle">{html.escape(message)}</text>'


def _subtitle(message: str) -> str:
    return f'<text x="450" y="50" text-anchor="middle" font-size="11">{html.escape(message)}</text>'


def _axis_ticks(
    context: ResultContext, transform: Callable[[float, float], tuple[float, float]]
) -> str:
    lower, upper = context.geometry.nodes_m.min(axis=0), context.geometry.nodes_m.max(axis=0)
    marks = []
    for value in np.linspace(lower[0], upper[0], 5):
        x, _ = transform(float(value), float(lower[1]))
        marks.append(
            f'<line x1="{x:.3f}" y1="70" x2="{x:.3f}" y2="550" stroke="#e5e7eb"/><text x="{x:.3f}" y="565" text-anchor="middle" font-size="10">{value:.3g}</text>'
        )
    for value in np.linspace(lower[1], upper[1], 5):
        _, y = transform(float(lower[0]), float(value))
        marks.append(
            f'<line x1="70" y1="{y:.3f}" x2="830" y2="{y:.3f}" stroke="#e5e7eb"/><text x="60" y="{y + 3:.3f}" text-anchor="end" font-size="10">{value:.3g}</text>'
        )
    return "".join(marks)


def _svg_document(
    title: str,
    body: str,
    context: ResultContext,
    transform: Callable[[float, float], tuple[float, float]],
    digest: str,
    legend: str = "blue active; red stuck; purple held; computational counts, not concentration",
) -> str:
    x_label, y_label = (
        ("r [m]", "z [m]") if context.coordinate_system == "axisymmetric_rz" else ("x [m]", "y [m]")
    )
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{_WIDTH}" height="{_HEIGHT}" viewBox="0 0 {_WIDTH} {_HEIGHT}">
<rect width="100%" height="100%" fill="white"/>
<text x="450" y="30" text-anchor="middle" font-size="17">{html.escape(title)}</text>
    {_axis_ticks(context, transform)}{body}{_geometry(context, transform)}
<text x="450" y="{_HEIGHT - 36}" text-anchor="middle">{x_label}; equal spatial aspect; geometry from result context</text>
<text x="20" y="310" text-anchor="middle" transform="rotate(-90 20 310)">{y_label}</text>
<text x="70" y="{_HEIGHT - 20}" font-size="10">{html.escape(legend)}</text>
<text x="70" y="{_HEIGHT - 5}" font-size="8">{digest}</text></svg>'''


__all__ = ["VISUALIZATION_REVISION", "render_result"]
