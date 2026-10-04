"""Render external COMSOL/solver R-Z trajectory overlays without solver imports.

The input files are the normalized long-form CSVs produced by the external
COMSOL V&V workflow.  Particle identities are aligned before any figure is
written.  Finite coordinates are plotted independently on each side, while
numeric differences use the exact common finite pre-terminal active
``(particle_id, time_s)`` population used by the formal evaluator.  Held or
stuck tails never enter those metrics, and missing escaped suffixes are never
reconstructed. SVG is rendered directly with the Python standard library.
The deliverable is deliberately limited to physical-space trajectories: the
horizontal axis is ``r`` and the vertical axis is ``z``.  Summary PNGs are
optional browser screenshots of SVGs so the solver does not acquire a plotting
dependency.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import shutil
import struct
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

TOOL_REVISION: Final = "m3v_matched_rz_trajectory_plot_v4"
_REQUIRED_COLUMNS: Final = ("particle_id", "time_s", "r_m", "z_m")
_STATUS_COLUMNS: Final = ("lifecycle", "current_status")
_LIFECYCLES: Final = frozenset(("active", "held", "stuck", "escaped"))
_BLUE: Final = "#1D4ED8"
_ORANGE: Final = "#D97706"
_INK: Final = "#1F2937"
_MUTED: Final = "#64748B"
_GRID: Final = "#D7DEE8"
_BACKGROUND: Final = "#FFFFFF"
_M_TO_MM: Final = 1.0e3
_S_TO_MS: Final = 1.0e3


@dataclass(frozen=True)
class TrajectoryTable:
    """One normalized trajectory table containing only observed coordinates."""

    path: Path
    sha256: str
    particle_ids: np.ndarray
    times_by_particle: tuple[np.ndarray, ...]
    r_by_particle: tuple[np.ndarray, ...]
    z_by_particle: tuple[np.ndarray, ...]
    row_keys: tuple[tuple[int, float], ...]
    active_row_keys: tuple[tuple[int, float], ...]
    status_column: str
    input_row_count: int
    missing_coordinate_row_count: int


@dataclass(frozen=True)
class MatchedTrajectories:
    """Two tables with identical IDs and common finite active metric states."""

    candidate: TrajectoryTable
    reference: TrajectoryTable
    common_active_row_keys: tuple[tuple[int, float], ...]


@dataclass(frozen=True)
class PlotBox:
    x: float
    y: float
    width: float
    height: float


@dataclass
class _TrajectoryRows:
    """Validated CSV rows before conversion to per-particle NumPy arrays."""

    grouped: dict[int, list[tuple[float, float, float]]]
    lifecycle: dict[int, list[tuple[float, str]]]
    particle_ids: set[int]
    finite_keys: set[tuple[int, float]]
    active_keys: set[tuple[int, float]]
    status_column: str
    input_count: int
    missing_coordinate_count: int


class SvgCanvas:
    """Small purpose-built SVG writer for these external V&V figures."""

    def __init__(self, width: int, height: int, title: str) -> None:
        self.width = width
        self.height = height
        self._items = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
                f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
                f'aria-label="{html.escape(title, quote=True)}">'
            ),
            f"<title>{html.escape(title)}</title>",
            f'<rect width="100%" height="100%" fill="{_BACKGROUND}"/>',
            (
                "<style>text{font-family:Arial,Helvetica,sans-serif;fill:#1F2937}"
                ".axis{stroke:#475569;stroke-width:1}.grid{stroke:#D7DEE8;stroke-width:1}"
                ".label{font-size:13px}.small{font-size:11px;fill:#64748B}"
                ".title{font-size:24px;font-weight:700}.subtitle{font-size:14px;fill:#64748B}"
                "</style>"
            ),
        ]

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        stroke: str,
        width: float = 1.0,
        opacity: float = 1.0,
        dash: str | None = None,
    ) -> None:
        dash_attribute = f' stroke-dasharray="{dash}"' if dash else ""
        self._items.append(
            f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" '
            f'stroke="{stroke}" stroke-width="{width:.3f}" opacity="{opacity:.3f}"'
            f"{dash_attribute}/>"
        )

    def polyline(
        self,
        points: list[tuple[float, float]],
        *,
        stroke: str,
        width: float = 1.0,
        opacity: float = 1.0,
        dash: str | None = None,
    ) -> None:
        if not points:
            return
        encoded = " ".join(f"{x:.3f},{y:.3f}" for x, y in points)
        dash_attribute = f' stroke-dasharray="{dash}"' if dash else ""
        self._items.append(
            f'<polyline points="{encoded}" fill="none" stroke="{stroke}" '
            f'stroke-width="{width:.3f}" opacity="{opacity:.3f}" '
            f'stroke-linejoin="round" stroke-linecap="round"{dash_attribute}/>'
        )

    def rect(
        self,
        box: PlotBox,
        *,
        fill: str = "none",
        stroke: str = _GRID,
        width: float = 1.0,
    ) -> None:
        self._items.append(
            f'<rect x="{box.x:.3f}" y="{box.y:.3f}" width="{box.width:.3f}" '
            f'height="{box.height:.3f}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{width:.3f}"/>'
        )

    def circle(
        self,
        x: float,
        y: float,
        radius: float,
        *,
        fill: str,
        stroke: str = "none",
    ) -> None:
        self._items.append(
            f'<circle cx="{x:.3f}" cy="{y:.3f}" r="{radius:.3f}" fill="{fill}" stroke="{stroke}"/>'
        )

    def text(
        self,
        x: float,
        y: float,
        value: str,
        *,
        css_class: str = "label",
        anchor: str = "start",
        rotate: float | None = None,
    ) -> None:
        transform = f' transform="rotate({rotate:.1f} {x:.3f} {y:.3f})"' if rotate else ""
        self._items.append(
            f'<text x="{x:.3f}" y="{y:.3f}" class="{css_class}" '
            f'text-anchor="{anchor}"{transform}>{html.escape(value)}</text>'
        )

    def write(self, path: Path) -> None:
        path.write_text("\n".join([*self._items, "</svg>", ""]), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_float(value: str, path: Path, line_number: int, column: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{path}:{line_number}: invalid {column}") from error
    if not math.isfinite(parsed):
        raise ValueError(f"{path}:{line_number}: nonfinite {column}")
    return parsed


def _optional_coordinate(value: str, path: Path, line_number: int, column: str) -> float | None:
    stripped = value.strip()
    if not stripped:
        return None
    try:
        parsed = float(stripped)
    except ValueError as error:
        raise ValueError(f"{path}:{line_number}: invalid {column}") from error
    if math.isnan(parsed):
        return None
    if not math.isfinite(parsed):
        raise ValueError(f"{path}:{line_number}: nonfinite {column}")
    return parsed


def _trajectory_row(
    row: dict[str, str], path: Path, line_number: int, status_column: str
) -> tuple[int, float, float | None, float | None, str]:
    try:
        particle_id = int(row["particle_id"])
    except ValueError as error:
        raise ValueError(f"{path}:{line_number}: invalid particle_id") from error
    if particle_id < 0:
        raise ValueError(f"{path}:{line_number}: particle_id must be nonnegative")
    time_s = _finite_float(row["time_s"], path, line_number, "time_s")
    r_m = _optional_coordinate(row["r_m"], path, line_number, "r_m")
    z_m = _optional_coordinate(row["z_m"], path, line_number, "z_m")
    if (r_m is None) != (z_m is None):
        raise ValueError(f"{path}:{line_number}: r_m and z_m must both be observed or missing")
    status = row[status_column].strip()
    if status not in _LIFECYCLES:
        raise ValueError(f"{path}:{line_number}: unsupported lifecycle {status!r}")
    if status in {"active", "held", "stuck"} and r_m is None:
        raise ValueError(f"{path}:{line_number}: {status} position must be observed")
    if status == "escaped" and r_m is not None:
        raise ValueError(f"{path}:{line_number}: escaped position must remain unobserved")
    return particle_id, time_s, r_m, z_m, status


def _status_column(path: Path, fieldnames: list[str]) -> str:
    present = [name for name in _STATUS_COLUMNS if name in fieldnames]
    if len(present) != 1:
        raise ValueError(
            f"{path}: expected exactly one lifecycle column from {list(_STATUS_COLUMNS)}"
        )
    return present[0]


def _read_trajectory_rows(resolved: Path) -> _TrajectoryRows:
    """Parse one CSV and retain only coordinates that were directly observed."""

    grouped: dict[int, list[tuple[float, float, float]]] = {}
    lifecycle_rows: dict[int, list[tuple[float, str]]] = {}
    particle_ids_seen: set[int] = set()
    input_keys: set[tuple[int, float]] = set()
    finite_keys: set[tuple[int, float]] = set()
    active_keys: set[tuple[int, float]] = set()
    input_row_count = 0
    missing_coordinate_row_count = 0
    with resolved.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(_REQUIRED_COLUMNS).difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{resolved}: missing columns {sorted(missing)}")
        status_column = _status_column(resolved, list(reader.fieldnames or ()))
        for line_number, row in enumerate(reader, start=2):
            particle_id, time_s, r_m, z_m, status = _trajectory_row(
                row, resolved, line_number, status_column
            )
            key = (particle_id, time_s)
            if key in input_keys:
                raise ValueError(f"{resolved}:{line_number}: duplicate key {key}")
            input_keys.add(key)
            input_row_count += 1
            particle_ids_seen.add(particle_id)
            lifecycle_rows.setdefault(particle_id, []).append((time_s, status))
            if r_m is None:
                missing_coordinate_row_count += 1
                continue
            if z_m is None:
                raise AssertionError("coordinate-pair validation failed")
            finite_keys.add(key)
            if status == "active":
                active_keys.add(key)
            grouped.setdefault(particle_id, []).append((time_s, r_m, z_m))
    if not particle_ids_seen:
        raise ValueError(f"{resolved}: trajectory is empty")
    missing_particles = sorted(particle_ids_seen.difference(grouped))
    if missing_particles:
        raise ValueError(
            f"{resolved}: particles have no observed coordinates: {missing_particles[:3]}"
        )
    return _TrajectoryRows(
        grouped=grouped,
        lifecycle=lifecycle_rows,
        particle_ids=particle_ids_seen,
        finite_keys=finite_keys,
        active_keys=active_keys,
        status_column=status_column,
        input_count=input_row_count,
        missing_coordinate_count=missing_coordinate_row_count,
    )


def _validate_lifecycle(rows: _TrajectoryRows, path: Path) -> None:
    """Reject a particle that returns to active or changes terminal fate."""

    for particle_id, status_rows in rows.lifecycle.items():
        terminal_status: str | None = None
        for _, status in sorted(status_rows):
            if terminal_status is None and status == "active":
                continue
            if terminal_status is None:
                terminal_status = status
                continue
            if status != terminal_status:
                raise ValueError(
                    f"{path}: particle {particle_id} lifecycle changes after terminal state"
                )


def _coordinate_arrays(
    grouped: dict[int, list[tuple[float, float, float]]], particle_ids: np.ndarray
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    """Convert directly observed coordinates to ordered per-particle arrays."""

    times: list[np.ndarray] = []
    radial: list[np.ndarray] = []
    axial: list[np.ndarray] = []
    for particle_id in particle_ids:
        rows = sorted(grouped[int(particle_id)])
        times.append(np.asarray([row[0] for row in rows], dtype=np.float64))
        radial.append(np.asarray([row[1] for row in rows], dtype=np.float64))
        axial.append(np.asarray([row[2] for row in rows], dtype=np.float64))
    return tuple(times), tuple(radial), tuple(axial)


def read_trajectory(path: Path) -> TrajectoryTable:
    """Read one normalized long-form CSV and reject ambiguous row identity."""

    resolved = path.expanduser().resolve()
    rows = _read_trajectory_rows(resolved)
    _validate_lifecycle(rows, resolved)
    particle_ids = np.asarray(sorted(rows.particle_ids), dtype=np.int64)
    times, radial, axial = _coordinate_arrays(rows.grouped, particle_ids)
    return TrajectoryTable(
        path=resolved,
        sha256=_sha256(resolved),
        particle_ids=particle_ids,
        times_by_particle=times,
        r_by_particle=radial,
        z_by_particle=axial,
        row_keys=tuple(sorted(rows.finite_keys)),
        active_row_keys=tuple(sorted(rows.active_keys)),
        status_column=rows.status_column,
        input_row_count=rows.input_count,
        missing_coordinate_row_count=rows.missing_coordinate_count,
    )


def read_matched_trajectories(candidate_path: Path, reference_path: Path) -> MatchedTrajectories:
    """Read inputs and align IDs plus the common finite active metric population."""

    candidate = read_trajectory(candidate_path)
    reference = read_trajectory(reference_path)
    if not np.array_equal(candidate.particle_ids, reference.particle_ids):
        raise ValueError("particle IDs do not align")
    common = tuple(sorted(set(candidate.active_row_keys).intersection(reference.active_row_keys)))
    common_particles = {particle_id for particle_id, _ in common}
    missing_common = [
        int(particle_id)
        for particle_id in candidate.particle_ids
        if int(particle_id) not in common_particles
    ]
    if missing_common:
        raise ValueError(f"particles have no common finite active state: {missing_common[:3]}")
    return MatchedTrajectories(candidate, reference, common)


def _range_with_margin(values: np.ndarray, fraction: float = 0.05) -> tuple[float, float]:
    low = float(np.min(values))
    high = float(np.max(values))
    span = high - low
    if span == 0.0:
        scale = max(abs(low), 1.0)
        return low - scale * 1.0e-9, high + scale * 1.0e-9
    return low - fraction * span, high + fraction * span


def _project(
    x_values: np.ndarray,
    y_values: np.ndarray,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    box: PlotBox,
) -> list[tuple[float, float]]:
    x_scale = box.width / (x_range[1] - x_range[0])
    y_scale = box.height / (y_range[1] - y_range[0])
    return [
        (
            box.x + (float(x) - x_range[0]) * x_scale,
            box.y + box.height - (float(y) - y_range[0]) * y_scale,
        )
        for x, y in zip(x_values, y_values, strict=True)
    ]


def _axis_value(value: float, span: float) -> str:
    if abs(value) >= 1.0e6 or (value != 0.0 and abs(value) < 1.0e-3):
        return f"{value:.6e}"
    decimals = max(3, min(9, math.ceil(-math.log10(span)) + 2))
    return f"{value:.{decimals}f}"


def _draw_axes(
    canvas: SvgCanvas,
    box: PlotBox,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    *,
    x_label: str,
    y_label: str,
) -> None:
    canvas.rect(box, stroke="#94A3B8")
    for fraction in (0.25, 0.5, 0.75):
        x = box.x + fraction * box.width
        y = box.y + fraction * box.height
        canvas.line(x, box.y, x, box.y + box.height, stroke=_GRID)
        canvas.line(box.x, y, box.x + box.width, y, stroke=_GRID)
    x_span = x_range[1] - x_range[0]
    y_span = y_range[1] - y_range[0]
    canvas.text(
        box.x,
        box.y + box.height + 18,
        _axis_value(x_range[0], x_span),
        css_class="small",
    )
    canvas.text(
        box.x + box.width,
        box.y + box.height + 18,
        _axis_value(x_range[1], x_span),
        css_class="small",
        anchor="end",
    )
    canvas.text(
        box.x - 8,
        box.y + box.height,
        _axis_value(y_range[0], y_span),
        css_class="small",
        anchor="end",
    )
    canvas.text(
        box.x - 8,
        box.y + 10,
        _axis_value(y_range[1], y_span),
        css_class="small",
        anchor="end",
    )
    canvas.text(box.x + box.width / 2, box.y + box.height + 38, x_label, anchor="middle")
    canvas.text(
        box.x - 58,
        box.y + box.height / 2,
        y_label,
        anchor="middle",
        rotate=-90.0,
    )


def _draw_legend(canvas: SvgCanvas, x: float, y: float) -> None:
    canvas.line(x, y, x + 42, y, stroke=_ORANGE, width=2.5)
    canvas.text(x + 50, y + 5, "COMSOL reference", css_class="label")
    canvas.line(x + 190, y, x + 232, y, stroke=_BLUE, width=2.0, dash="8 5")
    canvas.text(x + 240, y + 5, "solver candidate", css_class="label")


def _all_values(series: tuple[np.ndarray, ...]) -> np.ndarray:
    return np.concatenate(series)


def _time_extent(table: TrajectoryTable) -> tuple[float, float]:
    values = _all_values(table.times_by_particle)
    return float(np.min(values)), float(np.max(values))


def _particle_end_extent(table: TrajectoryTable) -> tuple[float, float]:
    endings = np.asarray([times[-1] for times in table.times_by_particle], dtype=np.float64)
    return float(np.min(endings)), float(np.max(endings))


def _render_rz_overview(data: MatchedTrajectories, label: str, path: Path) -> None:
    canvas = SvgCanvas(1600, 1000, f"{label}: all-particle R-Z trajectories")
    canvas.text(80, 55, f"{label}: all-particle R-Z trajectories", css_class="title")
    canvas.text(
        80,
        82,
        (
            "Observed paths are drawn independently and are non-gating; "
            "COMSOL solid orange, solver dashed blue"
        ),
        css_class="subtitle",
    )
    _draw_legend(canvas, 1080, 62)
    box = PlotBox(150, 130, 1370, 790)
    r_all_mm = _M_TO_MM * np.concatenate(
        [_all_values(data.reference.r_by_particle), _all_values(data.candidate.r_by_particle)]
    )
    z_all_mm = _M_TO_MM * np.concatenate(
        [_all_values(data.reference.z_by_particle), _all_values(data.candidate.z_by_particle)]
    )
    r_range = _range_with_margin(r_all_mm, 0.025)
    z_range = _range_with_margin(z_all_mm, 0.025)
    _draw_axes(canvas, box, r_range, z_range, x_label="r (mm)", y_label="z (mm)")
    for index in range(len(data.reference.particle_ids)):
        canvas.polyline(
            _project(
                _M_TO_MM * data.reference.r_by_particle[index],
                _M_TO_MM * data.reference.z_by_particle[index],
                r_range,
                z_range,
                box,
            ),
            stroke=_ORANGE,
            width=1.35,
            opacity=0.48,
        )
        canvas.polyline(
            _project(
                _M_TO_MM * data.candidate.r_by_particle[index],
                _M_TO_MM * data.candidate.z_by_particle[index],
                r_range,
                z_range,
                box,
            ),
            stroke=_BLUE,
            width=1.0,
            opacity=0.72,
            dash="5 4",
        )
    candidate_time = _time_extent(data.candidate)
    reference_time = _time_extent(data.reference)
    canvas.text(
        150,
        946,
        (
            "Held/stuck tails may be displayed but are excluded from metrics; "
            "escaped NaN positions are unobserved and never reconstructed"
        ),
        css_class="small",
    )
    canvas.text(
        150,
        970,
        (
            f"solver observed coordinates t={candidate_time[0] * _S_TO_MS:.3f}.."
            f"{candidate_time[1] * _S_TO_MS:.3f} ms; "
            f"COMSOL observed coordinates t={reference_time[0] * _S_TO_MS:.3f}.."
            f"{reference_time[1] * _S_TO_MS:.3f} ms"
        ),
        css_class="small",
    )
    canvas.text(
        1520,
        970,
        (
            f"particles={len(data.candidate.particle_ids)}, "
            f"candidate path points={len(data.candidate.row_keys)}, "
            f"reference path points={len(data.reference.row_keys)}, "
            f"common active metric rows={len(data.common_active_row_keys)}"
        ),
        css_class="small",
        anchor="end",
    )
    canvas.write(path)


def _render_particle(data: MatchedTrajectories, label: str, index: int, path: Path) -> None:
    particle_id = int(data.candidate.particle_ids[index])
    reference_t = data.reference.times_by_particle[index]
    candidate_t = data.candidate.times_by_particle[index]
    candidate_r_mm = _M_TO_MM * data.candidate.r_by_particle[index]
    reference_r_mm = _M_TO_MM * data.reference.r_by_particle[index]
    candidate_z_mm = _M_TO_MM * data.candidate.z_by_particle[index]
    reference_z_mm = _M_TO_MM * data.reference.z_by_particle[index]

    canvas = SvgCanvas(1600, 1000, f"{label}: particle {particle_id} R-Z trajectory")
    canvas.text(80, 55, f"{label}: particle {particle_id} R-Z trajectory", css_class="title")
    canvas.text(
        80,
        82,
        (
            f"solver observed coordinates t={candidate_t[0] * _S_TO_MS:.3f}.."
            f"{candidate_t[-1] * _S_TO_MS:.3f} ms; "
            f"COMSOL observed coordinates t={reference_t[0] * _S_TO_MS:.3f}.."
            f"{reference_t[-1] * _S_TO_MS:.3f} ms; candidate markers show first/last "
            "observed coordinate"
        ),
        css_class="subtitle",
    )
    canvas.text(
        80,
        106,
        (
            "Observed path lines are independent and non-gating; "
            "escaped NaN positions are unobserved and never reconstructed"
        ),
        css_class="subtitle",
    )
    _draw_legend(canvas, 1080, 62)
    box = PlotBox(150, 140, 1370, 780)
    r_range = _range_with_margin(
        np.concatenate([reference_r_mm, candidate_r_mm]),
        0.08,
    )
    z_range = _range_with_margin(
        np.concatenate([reference_z_mm, candidate_z_mm]),
        0.08,
    )
    _draw_axes(canvas, box, r_range, z_range, x_label="r (mm)", y_label="z (mm)")
    reference_points = _project(reference_r_mm, reference_z_mm, r_range, z_range, box)
    candidate_points = _project(candidate_r_mm, candidate_z_mm, r_range, z_range, box)
    canvas.polyline(reference_points, stroke=_ORANGE, width=3.0)
    canvas.polyline(candidate_points, stroke=_BLUE, width=2.4, dash="10 6")
    start_x, start_y = candidate_points[0]
    end_x, end_y = candidate_points[-1]
    canvas.circle(start_x, start_y, 6.0, fill=_BACKGROUND, stroke=_INK)
    canvas.circle(end_x, end_y, 6.0, fill=_INK, stroke=_INK)
    canvas.text(
        150,
        970,
        (
            f"solver observed path points={len(candidate_t)}, "
            f"COMSOL observed path points={len(reference_t)}; "
            f"common finite active states={sum(1 for key in data.common_active_row_keys if key[0] == particle_id)}; "
            "metrics in SI"
        ),
        css_class="small",
    )
    canvas.write(path)


def _write_atlas(label: str, particle_files: list[tuple[int, Path]], output: Path) -> None:
    cards = []
    for particle_id, path in particle_files:
        relative = path.relative_to(output.parent).as_posix()
        cards.append(
            f'<article class="card" data-particle="{particle_id}">'
            f'<a href="{relative}">Particle {particle_id}</a>'
            f'<img loading="lazy" src="{relative}" alt="Particle {particle_id} trajectory comparison">'
            "</article>"
        )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(label)} trajectory atlas</title>
<style>
body{{margin:0;background:#f4f7fb;color:#1f2937;font:16px Arial,sans-serif}}
header{{position:sticky;top:0;z-index:2;padding:18px 28px;background:white;border-bottom:1px solid #d7dee8}}
h1{{margin:0 0 8px;font-size:24px}}p{{margin:4px 0;color:#64748b}}
nav a{{margin-right:18px;color:#1d4ed8}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:18px;padding:22px}}
.card{{background:white;border:1px solid #d7dee8;padding:12px}}
.card>a{{display:block;margin-bottom:8px;font-weight:700;color:#1d4ed8}}
.card img{{display:block;width:100%;height:auto}}
</style>
</head>
<body>
<header><h1>{html.escape(label)} trajectory atlas</h1>
<p>Every common particle ID is included. Each side's observed finite R-Z path is
drawn independently and is non-gating; held/stuck tails may be shown. Receipt
metrics use only exact common finite active states. Escaped NaN positions are
unobserved and never reconstructed. Each particle view uses its own range.</p>
<nav><a href="overview_rz.svg">All-particle R-Z overview</a></nav></header>
<main class="grid">{"".join(cards)}</main>
</body></html>
"""
    output.write_text(document, encoding="utf-8")


def _browser_path(explicit: Path | None) -> Path:
    if explicit is not None:
        candidate = explicit.expanduser().resolve()
        if not candidate.is_file():
            raise ValueError(f"browser executable does not exist: {candidate}")
        return candidate
    names = ("msedge", "microsoft-edge", "google-chrome", "chrome", "chromium")
    for name in names:
        discovered = shutil.which(name)
        if discovered:
            return Path(discovered).resolve()
    common = (
        Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"),
        Path("C:/Program Files/Microsoft/Edge/Application/msedge.exe"),
        Path("C:/Program Files/Google/Chrome/Application/chrome.exe"),
    )
    for candidate in common:
        if candidate.is_file():
            return candidate.resolve()
    raise RuntimeError("no Chrome/Edge browser found; rerun with --skip-png or --browser")


def _png_dimensions(path: Path) -> tuple[int, int]:
    data = path.read_bytes()[:24]
    if len(data) != 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError(f"browser did not write a valid PNG: {path}")
    return struct.unpack(">II", data[16:24])


def _wait_for_png(path: Path, timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.is_file() and path.stat().st_size >= 24:
            return
        time.sleep(0.05)
    raise RuntimeError(f"browser returned without writing PNG: {path}")


def _rasterize(svg_paths: list[Path], browser: Path) -> dict[str, str]:
    version = subprocess.run(
        [str(browser), "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    with tempfile.TemporaryDirectory(
        prefix="trajectory_plot_browser_", ignore_cleanup_errors=True
    ) as profile:
        for svg_path in svg_paths:
            png_path = svg_path.with_suffix(".png")
            command = [
                str(browser),
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                "--no-first-run",
                "--disable-background-mode",
                "--disable-sync",
                f"--user-data-dir={profile}",
                f"--screenshot={png_path}",
                "--window-size=1600,1000",
                svg_path.resolve().as_uri(),
            ]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=60)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"PNG rasterization failed for {svg_path}: {completed.stderr.strip()}"
                )
            _wait_for_png(png_path)
            dimensions = _png_dimensions(png_path)
            if dimensions != (1600, 1000):
                raise RuntimeError(f"unexpected PNG dimensions for {png_path}: {dimensions}")
    return {"executable": str(browser), "version": version}


def _coordinate_map(table: TrajectoryTable) -> dict[tuple[int, float], tuple[float, float]]:
    coordinates: dict[tuple[int, float], tuple[float, float]] = {}
    for index, raw_id in enumerate(table.particle_ids):
        particle_id = int(raw_id)
        for time_s, r_m, z_m in zip(
            table.times_by_particle[index],
            table.r_by_particle[index],
            table.z_by_particle[index],
            strict=True,
        ):
            coordinates[(particle_id, float(time_s))] = (float(r_m), float(z_m))
    return coordinates


def _metrics(data: MatchedTrajectories) -> dict[str, object]:
    candidate = _coordinate_map(data.candidate)
    reference = _coordinate_map(data.reference)
    values = np.asarray(
        [
            math.hypot(
                candidate[key][0] - reference[key][0],
                candidate[key][1] - reference[key][1],
            )
            for key in data.common_active_row_keys
        ],
        dtype=np.float64,
    )
    worst_index = int(np.argmax(values))
    worst_particle_id, worst_time_s = data.common_active_row_keys[worst_index]
    return {
        "position_difference_m": {
            "comparison_rows": int(values.size),
            "rms": float(np.sqrt(np.mean(values * values))),
            "maximum": float(np.max(values)),
            "p50": float(np.quantile(values, 0.50)),
            "p95": float(np.quantile(values, 0.95)),
            "p99": float(np.quantile(values, 0.99)),
            "worst_particle_id": worst_particle_id,
            "worst_time_s": worst_time_s,
        }
    }


def _artifact(path: Path, root: Path, role: str) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "role": role,
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def create_trajectory_figures(
    candidate_path: Path,
    reference_path: Path,
    output_directory: Path,
    *,
    label: str,
    render_png: bool,
    browser_path: Path | None = None,
) -> dict[str, object]:
    """Create one immutable external comparison atlas and its receipt."""

    output = output_directory.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    output.mkdir(parents=True)
    particle_directory = output / "particles"
    particle_directory.mkdir()
    data = read_matched_trajectories(candidate_path, reference_path)

    summary_path = output / "overview_rz.svg"
    _render_rz_overview(data, label, summary_path)

    particle_files: list[tuple[int, Path]] = []
    for index, raw_id in enumerate(data.candidate.particle_ids):
        particle_id = int(raw_id)
        path = particle_directory / f"particle_{particle_id:06d}.svg"
        _render_particle(data, label, index, path)
        particle_files.append((particle_id, path))
    atlas_path = output / "index.html"
    _write_atlas(label, particle_files, atlas_path)

    rasterizer: dict[str, str] | None = None
    if render_png:
        rasterizer = _rasterize([summary_path], _browser_path(browser_path))

    figures = [_artifact(summary_path, output, "all_particle_rz_svg")] + [
        _artifact(path, output, "particle_svg") for _, path in particle_files
    ]
    if render_png:
        figures.append(_artifact(summary_path.with_suffix(".png"), output, "all_particle_rz_png"))
    candidate_keys = set(data.candidate.row_keys)
    reference_keys = set(data.reference.row_keys)
    candidate_active_keys = set(data.candidate.active_row_keys)
    reference_active_keys = set(data.reference.active_row_keys)
    candidate_observed_counts = [len(times) for times in data.candidate.times_by_particle]
    reference_observed_counts = [len(times) for times in data.reference.times_by_particle]
    candidate_time = _time_extent(data.candidate)
    reference_time = _time_extent(data.reference)
    candidate_end = _particle_end_extent(data.candidate)
    reference_end = _particle_end_extent(data.reference)
    common_active_times = np.asarray(
        [key[1] for key in data.common_active_row_keys], dtype=np.float64
    )
    receipt: dict[str, object] = {
        "schema_version": 4,
        "tool_revision": TOOL_REVISION,
        "report_kind": "external_trajectory_figure_receipt",
        "tool": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "label": label,
        "scope": {
            "external_only": True,
            "solver_imported": False,
            "comparison": (
                "non-gating independently observed trajectories in physical R-Z space; "
                "numeric position differences on exact common finite active rows"
            ),
            "trajectory_rendering_population": (
                "each side's observed finite coordinates independently, including held/stuck tails"
            ),
            "missing_coordinate_policy": ("escaped_nan_unobserved_excluded_without_reconstruction"),
            "metric_population": (
                "exact candidate/reference intersection of finite pre-terminal active "
                "particle/time rows"
            ),
            "figure_role": "NON_GATING_DIAGNOSTIC",
            "physical_model_validity": "NOT_CLAIMED",
            "boundary_accuracy": "NOT_TESTED_BY_FIGURE",
        },
        "inputs": {
            "candidate": {
                "path": str(data.candidate.path),
                "sha256": data.candidate.sha256,
            },
            "reference": {
                "path": str(data.reference.path),
                "sha256": data.reference.sha256,
            },
        },
        "alignment": {
            "status": "PASS",
            "particle_identity": "EXACT",
            "observed_state_key": ["particle_id", "time_s"],
            "metric_state_key": ["particle_id", "time_s", "active_on_both_sides"],
            "particles": len(data.candidate.particle_ids),
            "first_particle_id": int(data.candidate.particle_ids[0]),
            "last_particle_id": int(data.candidate.particle_ids[-1]),
            "candidate_status_column": data.candidate.status_column,
            "reference_status_column": data.reference.status_column,
            "candidate_input_rows": data.candidate.input_row_count,
            "reference_input_rows": data.reference.input_row_count,
            "candidate_missing_coordinate_rows": data.candidate.missing_coordinate_row_count,
            "reference_missing_coordinate_rows": data.reference.missing_coordinate_row_count,
            "candidate_finite_rows": len(candidate_keys),
            "reference_finite_rows": len(reference_keys),
            "common_observed_finite_rows": len(candidate_keys.intersection(reference_keys)),
            "candidate_only_observed_finite_rows": len(candidate_keys.difference(reference_keys)),
            "reference_only_observed_finite_rows": len(reference_keys.difference(candidate_keys)),
            "candidate_active_finite_rows": len(candidate_active_keys),
            "reference_active_finite_rows": len(reference_active_keys),
            "candidate_terminal_finite_rows": len(candidate_keys.difference(candidate_active_keys)),
            "reference_terminal_finite_rows": len(reference_keys.difference(reference_active_keys)),
            "candidate_unobserved_escaped_rows": data.candidate.missing_coordinate_row_count,
            "reference_unobserved_escaped_rows": data.reference.missing_coordinate_row_count,
            "common_active_finite_rows": len(data.common_active_row_keys),
            "candidate_only_active_finite_rows": len(
                candidate_active_keys.difference(reference_active_keys)
            ),
            "reference_only_active_finite_rows": len(
                reference_active_keys.difference(candidate_active_keys)
            ),
            "candidate_observed_coordinate_rows_per_particle_min": min(candidate_observed_counts),
            "candidate_observed_coordinate_rows_per_particle_max": max(candidate_observed_counts),
            "reference_observed_coordinate_rows_per_particle_min": min(reference_observed_counts),
            "reference_observed_coordinate_rows_per_particle_max": max(reference_observed_counts),
            "candidate_observed_coordinate_time_s_min": candidate_time[0],
            "candidate_observed_coordinate_time_s_max": candidate_time[1],
            "reference_observed_coordinate_time_s_min": reference_time[0],
            "reference_observed_coordinate_time_s_max": reference_time[1],
            "candidate_last_observed_coordinate_time_s_min": candidate_end[0],
            "candidate_last_observed_coordinate_time_s_max": candidate_end[1],
            "reference_last_observed_coordinate_time_s_min": reference_end[0],
            "reference_last_observed_coordinate_time_s_max": reference_end[1],
            "common_active_time_s_min": float(np.min(common_active_times)),
            "common_active_time_s_max": float(np.max(common_active_times)),
        },
        "metrics": _metrics(data),
        "rendering": {
            "svg": "native_dependency_free",
            "png": rasterizer if rasterizer is not None else "SKIPPED",
            "style": {
                "reference": "solid orange",
                "candidate": "dashed blue",
                "redundant_encoding": True,
            },
            "display_coordinates": "millimetres",
            "source_and_metrics_coordinates": "SI metres and seconds",
            "path_comparison_is_gating": False,
        },
        "atlas": _artifact(atlas_path, output, "html_atlas"),
        "particle_svg_count": len(particle_files),
        "figures": figures,
    }
    receipt_path = output / "receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--skip-png", action="store_true")
    parser.add_argument("--browser", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    receipt = create_trajectory_figures(
        args.candidate,
        args.reference,
        args.output_directory,
        label=args.label,
        render_png=not args.skip_png,
        browser_path=args.browser,
    )
    print(json.dumps(receipt["alignment"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
