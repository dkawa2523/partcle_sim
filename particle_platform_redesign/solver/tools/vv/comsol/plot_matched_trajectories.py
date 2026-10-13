"""Render observed COMSOL/solver XY or RZ paths with optional canonical geometry.

Normalized CSVs own row identity, status and optional precontact eligibility.
The renderer never simulates missing positions or treats native disappeared as
candidate escaped. Geometry is read through the canonical format owner. SVG
rendering stays in this one external tool; browser PNG export remains optional.
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
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

TOOL_REVISION: Final = "matched_xy_rz_observed_trajectory_plot_v5"
_REQUIRED_COLUMNS: Final = ("particle_id", "time_s")
_STATUS_COLUMNS: Final = ("lifecycle", "current_status")
_LIFECYCLES: Final = frozenset(
    ("pending", "active", "held", "stuck", "escaped", "failed", "disappeared", "unobserved")
)
_TERMINAL: Final = frozenset(("held", "stuck", "escaped", "failed", "disappeared"))
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
    input_row_keys: tuple[tuple[int, float], ...]
    metric_row_keys: tuple[tuple[int, float], ...]
    precontact_column: str | None
    coordinate_system: str
    status_counts: dict[str, int]
    missing_coordinate_by_status: dict[str, int]


@dataclass(frozen=True)
class MatchedTrajectories:
    """Identical particle IDs, observed active rows and explicit metric eligibility."""

    candidate: TrajectoryTable
    reference: TrajectoryTable
    common_active_row_keys: tuple[tuple[int, float], ...]
    metric_row_keys: tuple[tuple[int, float], ...]
    metric_population: str


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
    input_keys: set[tuple[int, float]]
    metric_keys: set[tuple[int, float]]
    precontact_column: str | None
    coordinate_system: str
    status_counts: dict[str, int]
    missing_by_status: dict[str, int]


@dataclass(frozen=True)
class GeometryOverlay:
    path: Path
    sha256: str
    content_hash: str
    coordinate_system: str
    nodes_m: np.ndarray
    line2: np.ndarray
    group_names: tuple[str, ...]
    group_id: np.ndarray
    axis_line2: np.ndarray


def _load_geometry(path: Path | None, coordinate_system: str) -> GeometryOverlay | None:
    if path is None:
        return None
    from chamber_particles.case_format import read_with_info

    resolved = path.expanduser().resolve()
    bundle, identity = read_with_info(resolved)
    if bundle.coordinate_system != coordinate_system:
        raise ValueError("canonical geometry and trajectory coordinate systems do not align")
    geometry = bundle.geometry
    axis = _axis_seam_edges(
        geometry.nodes_m, geometry.tri3, geometry.quad4, geometry.boundary.line2, coordinate_system
    )
    return GeometryOverlay(
        resolved,
        _sha256(resolved),
        identity.content_hash,
        coordinate_system,
        geometry.nodes_m,
        geometry.boundary.line2,
        geometry.group_names,
        geometry.boundary.group_id,
        axis,
    )


def _axis_seam_edges(
    nodes: np.ndarray,
    triangles: np.ndarray | None,
    quads: np.ndarray | None,
    physical: np.ndarray,
    coordinate_system: str,
) -> np.ndarray:
    if coordinate_system != "axisymmetric_rz":
        return np.empty((0, 2), dtype=np.int64)
    incidence: Counter[tuple[int, int]] = Counter()
    cells: list[list[int]] = []
    if triangles is not None:
        cells.extend(triangles.tolist())
    if quads is not None:
        cells.extend(quads[:, [0, 1, 3, 2]].tolist())
    for cell in cells:
        for first, second in zip(cell, (*cell[1:], cell[0]), strict=True):
            incidence[(min(first, second), max(first, second))] += 1
    walls = set(map(tuple, np.sort(physical, axis=1).tolist()))
    edges = np.asarray(
        [edge for edge, count in incidence.items() if count == 1 and edge not in walls],
        dtype=np.int64,
    ).reshape(-1, 2)
    return edges[(nodes[edges, 0] == 0.0).all(axis=1)]


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
    row: dict[str, str],
    path: Path,
    line_number: int,
    status_column: str,
    coordinate_columns: tuple[str, str],
) -> tuple[int, float, float | None, float | None, str]:
    try:
        particle_id = int(row["particle_id"])
    except ValueError as error:
        raise ValueError(f"{path}:{line_number}: invalid particle_id") from error
    if particle_id < 0:
        raise ValueError(f"{path}:{line_number}: particle_id must be nonnegative")
    time_s = _finite_float(row["time_s"], path, line_number, "time_s")
    x_m = _optional_coordinate(row[coordinate_columns[0]], path, line_number, coordinate_columns[0])
    y_m = _optional_coordinate(row[coordinate_columns[1]], path, line_number, coordinate_columns[1])
    if (x_m is None) != (y_m is None):
        raise ValueError(f"{path}:{line_number}: coordinate pair must both be observed or missing")
    status = row[status_column].strip().lower()
    if status not in _LIFECYCLES:
        raise ValueError(f"{path}:{line_number}: unsupported lifecycle {status!r}")
    if status == "active" and x_m is None:
        raise ValueError(f"{path}:{line_number}: active position must be observed")
    return particle_id, time_s, x_m, y_m, status


def _status_column(path: Path, fieldnames: list[str]) -> str:
    present = [name for name in _STATUS_COLUMNS if name in fieldnames]
    if len(present) != 1:
        raise ValueError(f"{path}: expected exactly one lifecycle column")
    return present[0]


def _coordinate_columns(fieldnames: list[str]) -> tuple[str, tuple[str, str]]:
    candidates = [("axisymmetric_rz", ("r_m", "z_m")), ("cartesian_xy", ("x_m", "y_m"))]
    available = [
        (basis, columns) for basis, columns in candidates if set(columns).issubset(fieldnames)
    ]
    if len(available) != 1:
        raise ValueError("trajectory must contain exactly one explicit XY or RZ SI coordinate pair")
    return available[0]


def _precontact_value(row: dict[str, str], column: str | None) -> bool:
    if column is None:
        return True
    value = row[column].strip().lower()
    if value not in ("0", "1", "true", "false"):
        raise ValueError(f"{column} must contain explicit 0/1 or true/false values")
    return value in ("1", "true")


def _append_trajectory_row(
    rows: _TrajectoryRows,
    row: dict[str, str],
    resolved: Path,
    line_number: int,
    columns: tuple[str, str],
) -> None:
    if None in row or any(value is None for value in row.values()):
        raise ValueError("trajectory CSV row width does not match its header")
    particle_id, time_s, x_m, y_m, status = _trajectory_row(
        row, resolved, line_number, rows.status_column, columns
    )
    key = (particle_id, time_s)
    if key in rows.input_keys:
        raise ValueError(f"{resolved}:{line_number}: duplicate key {key}")
    rows.input_keys.add(key)
    rows.input_count += 1
    rows.particle_ids.add(particle_id)
    rows.status_counts[status] = rows.status_counts.get(status, 0) + 1
    rows.lifecycle.setdefault(particle_id, []).append((time_s, status))
    eligible = _precontact_value(row, rows.precontact_column)
    rows.grouped.setdefault(particle_id, []).append(
        (
            time_s,
            math.nan if x_m is None else x_m,
            math.nan if y_m is None else y_m,
        )
    )
    if x_m is None:
        rows.missing_coordinate_count += 1
        rows.missing_by_status[status] = rows.missing_by_status.get(status, 0) + 1
        return
    rows.finite_keys.add(key)
    if status == "active":
        rows.active_keys.add(key)
        if eligible:
            rows.metric_keys.add(key)


def _read_trajectory_rows(resolved: Path, precontact_column: str) -> _TrajectoryRows:
    with resolved.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        names = list(reader.fieldnames or ())
        if len(set(names)) != len(names) or not set(_REQUIRED_COLUMNS).issubset(names):
            raise ValueError("trajectory CSV requires unique identity/time column headers")
        basis, columns = _coordinate_columns(names)
        rows = _TrajectoryRows(
            {},
            {},
            set(),
            set(),
            set(),
            _status_column(resolved, names),
            0,
            0,
            set(),
            set(),
            precontact_column if precontact_column in names else None,
            basis,
            {},
            {},
        )
        for line_number, row in enumerate(reader, start=2):
            _append_trajectory_row(rows, row, resolved, line_number, columns)
    if not rows.input_keys:
        raise ValueError(f"{resolved}: trajectory is empty")
    return rows


def _validate_lifecycle(rows: _TrajectoryRows, path: Path) -> None:
    for particle_id, status_rows in rows.lifecycle.items():
        terminal: str | None = None
        for _, status in sorted(status_rows):
            if status == "unobserved":
                continue
            if terminal is not None and status != terminal:
                raise ValueError(
                    f"{path}: particle {particle_id} lifecycle changes after terminal state"
                )
            if status in _TERMINAL:
                terminal = status


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


def read_trajectory(
    path: Path, *, precontact_column: str = "precontact_eligible"
) -> TrajectoryTable:
    """Read one normalized long-form CSV and reject ambiguous row identity."""

    resolved = path.expanduser().resolve()
    rows = _read_trajectory_rows(resolved, precontact_column)
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
        input_row_keys=tuple(sorted(rows.input_keys)),
        metric_row_keys=tuple(sorted(rows.metric_keys)),
        precontact_column=rows.precontact_column,
        coordinate_system=rows.coordinate_system,
        status_counts=rows.status_counts,
        missing_coordinate_by_status=rows.missing_by_status,
    )


def read_matched_trajectories(
    candidate_path: Path, reference_path: Path, *, precontact_column: str = "precontact_eligible"
) -> MatchedTrajectories:
    candidate = read_trajectory(candidate_path, precontact_column=precontact_column)
    reference = read_trajectory(reference_path, precontact_column=precontact_column)
    if not np.array_equal(candidate.particle_ids, reference.particle_ids):
        raise ValueError("particle IDs do not align")
    if candidate.coordinate_system != reference.coordinate_system:
        raise ValueError("candidate/reference coordinate systems do not align")
    if candidate.precontact_column != reference.precontact_column:
        raise ValueError("precontact eligibility must be supplied on both sides or neither")
    common = tuple(sorted(set(candidate.active_row_keys).intersection(reference.active_row_keys)))
    metric = tuple(sorted(set(candidate.metric_row_keys).intersection(reference.metric_row_keys)))
    population = (
        "joint precontact finite active rows"
        if candidate.precontact_column is not None
        else "common finite active rows; contact eligibility is unobserved"
    )
    return MatchedTrajectories(candidate, reference, common, metric, population)


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
    canvas.text(x + 50, y + 5, "reference observations", css_class="label")
    canvas.line(x + 190, y, x + 232, y, stroke=_BLUE, width=2.0, dash="8 5")
    canvas.text(x + 240, y + 5, "candidate observations", css_class="label")


def _all_values(series: tuple[np.ndarray, ...]) -> np.ndarray:
    values = np.concatenate(series)
    return values[np.isfinite(values)]


def _observed_times(table: TrajectoryTable) -> np.ndarray:
    return np.asarray([key[1] for key in table.row_keys], dtype=np.float64)


def _time_extent(table: TrajectoryTable) -> tuple[float | None, float | None]:
    values = _observed_times(table)
    return (float(np.min(values)), float(np.max(values))) if values.size else (None, None)


def _particle_end_extent(table: TrajectoryTable) -> tuple[float | None, float | None]:
    endings = [
        max(key[1] for key in table.row_keys if key[0] == int(pid))
        for pid in table.particle_ids
        if any(key[0] == int(pid) for key in table.row_keys)
    ]
    return (min(endings), max(endings)) if endings else (None, None)


def _plot_ranges(
    data: MatchedTrajectories,
    geometry: GeometryOverlay | None,
    box: PlotBox,
) -> tuple[tuple[float, float], tuple[float, float]]:
    x = np.concatenate(
        [_all_values(data.candidate.r_by_particle), _all_values(data.reference.r_by_particle)]
    )
    y = np.concatenate(
        [_all_values(data.candidate.z_by_particle), _all_values(data.reference.z_by_particle)]
    )
    if geometry is not None:
        x = np.concatenate([x, geometry.nodes_m[:, 0]])
        y = np.concatenate([y, geometry.nodes_m[:, 1]])
    if x.size == 0:
        raise ValueError("no observed coordinates or geometry are available to render")
    x_range = _range_with_margin(_M_TO_MM * x, 0.025)
    y_range = _range_with_margin(_M_TO_MM * y, 0.025)
    scale = max((x_range[1] - x_range[0]) / box.width, (y_range[1] - y_range[0]) / box.height)
    x_center = (x_range[0] + x_range[1]) / 2
    y_center = (y_range[0] + y_range[1]) / 2
    return (
        (x_center - scale * box.width / 2, x_center + scale * box.width / 2),
        (y_center - scale * box.height / 2, y_center + scale * box.height / 2),
    )


def _draw_geometry(
    canvas: SvgCanvas,
    geometry: GeometryOverlay | None,
    box: PlotBox,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> None:
    if geometry is None:
        return
    for edge in geometry.line2:
        points = _project(
            _M_TO_MM * geometry.nodes_m[edge, 0],
            _M_TO_MM * geometry.nodes_m[edge, 1],
            x_range,
            y_range,
            box,
        )
        canvas.polyline(points, stroke="#475569", width=2.0, opacity=0.65)
    for edge in geometry.axis_line2:
        points = _project(
            _M_TO_MM * geometry.nodes_m[edge, 0],
            _M_TO_MM * geometry.nodes_m[edge, 1],
            x_range,
            y_range,
            box,
        )
        canvas.polyline(points, stroke="#94A3B8", width=1.5, dash="3 6")
    legend = "Canonical boundaries: grey"
    if geometry.coordinate_system == "axisymmetric_rz":
        legend += "; axis seam: dotted, nonmaterial"
    canvas.text(150, 114, legend, css_class="small")


def _draw_observed_path(
    canvas: SvgCanvas,
    table: TrajectoryTable,
    index: int,
    box: PlotBox,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    *,
    stroke: str,
    dash: str | None,
    markers: bool,
) -> None:
    x = table.r_by_particle[index]
    y = table.z_by_particle[index]
    observed = np.isfinite(x) & np.isfinite(y)
    endpoints = np.flatnonzero(observed)
    split = np.flatnonzero(np.diff(observed.astype(np.int8)) != 0) + 1
    for segment in np.split(np.arange(x.size), split):
        if segment.size and observed[segment[0]]:
            points = _project(_M_TO_MM * x[segment], _M_TO_MM * y[segment], x_range, y_range, box)
            canvas.polyline(points, stroke=stroke, width=1.8, opacity=0.65, dash=dash)
            if segment.size == 1:
                canvas.circle(*points[0], 2.5, fill=stroke)
    if markers and endpoints.size:
        points = _project(
            _M_TO_MM * x[endpoints[[0, -1]]],
            _M_TO_MM * y[endpoints[[0, -1]]],
            x_range,
            y_range,
            box,
        )
        canvas.circle(*points[0], 5.0, fill=_BACKGROUND, stroke=stroke)
        canvas.circle(*points[1], 5.0, fill=stroke)


def _coordinate_labels(coordinate_system: str) -> tuple[str, str, str]:
    return (
        ("r (mm)", "z (mm)", "R-Z")
        if coordinate_system == "axisymmetric_rz"
        else ("x (mm)", "y (mm)", "X-Y")
    )


def _render_overview(
    data: MatchedTrajectories,
    label: str,
    path: Path,
    geometry: GeometryOverlay | None,
) -> None:
    x_label, y_label, basis = _coordinate_labels(data.candidate.coordinate_system)
    canvas = SvgCanvas(1600, 1000, f"{label}: all-particle {basis} trajectories")
    canvas.text(80, 55, f"{label}: all-particle {basis} trajectories", css_class="title")
    canvas.text(
        80,
        82,
        "Observed paths are drawn independently and are non-gating; missing spans are not reconstructed",
        css_class="subtitle",
    )
    _draw_legend(canvas, 1080, 62)
    box = PlotBox(150, 140, 1370, 760)
    x_range, y_range = _plot_ranges(data, geometry, box)
    _draw_axes(canvas, box, x_range, y_range, x_label=x_label, y_label=y_label)
    _draw_geometry(canvas, geometry, box, x_range, y_range)
    for index in range(len(data.candidate.particle_ids)):
        _draw_observed_path(
            canvas,
            data.reference,
            index,
            box,
            x_range,
            y_range,
            stroke=_ORANGE,
            dash=None,
            markers=False,
        )
        _draw_observed_path(
            canvas,
            data.candidate,
            index,
            box,
            x_range,
            y_range,
            stroke=_BLUE,
            dash="5 4",
            markers=False,
        )
    canvas.text(
        150,
        945,
        f"Position metrics: {data.metric_population}; rows={len(data.metric_row_keys)}",
        css_class="small",
    )
    canvas.text(
        150,
        970,
        f"particles={len(data.candidate.particle_ids)}; candidate observed points={len(data.candidate.row_keys)}; reference observed points={len(data.reference.row_keys)}; equal geometric aspect",
        css_class="small",
    )
    canvas.write(path)


def _render_particle(
    data: MatchedTrajectories,
    label: str,
    index: int,
    path: Path,
    geometry: GeometryOverlay | None,
) -> None:
    particle_id = int(data.candidate.particle_ids[index])
    x_label, y_label, basis = _coordinate_labels(data.candidate.coordinate_system)
    canvas = SvgCanvas(1600, 1000, f"{label}: particle {particle_id} {basis} trajectory")
    canvas.text(80, 55, f"{label}: particle {particle_id} {basis} trajectory", css_class="title")
    canvas.text(
        80,
        82,
        "Complete observed paths; circles mark first/last observations, not inferred terminal locations",
        css_class="subtitle",
    )
    _draw_legend(canvas, 1080, 62)
    box = PlotBox(150, 140, 1370, 760)
    x_range, y_range = _plot_ranges(data, geometry, box)
    _draw_axes(canvas, box, x_range, y_range, x_label=x_label, y_label=y_label)
    _draw_geometry(canvas, geometry, box, x_range, y_range)
    _draw_observed_path(
        canvas,
        data.reference,
        index,
        box,
        x_range,
        y_range,
        stroke=_ORANGE,
        dash=None,
        markers=True,
    )
    _draw_observed_path(
        canvas,
        data.candidate,
        index,
        box,
        x_range,
        y_range,
        stroke=_BLUE,
        dash="10 6",
        markers=True,
    )
    count = sum(key[0] == particle_id for key in data.metric_row_keys)
    canvas.text(
        150,
        945,
        f"Position metrics: {data.metric_population}; this particle rows={count}",
        css_class="small",
    )
    canvas.text(
        150,
        970,
        (
            f"Observed candidate points={int(np.isfinite(data.candidate.r_by_particle[index]).sum())}, "
            f"reference points={int(np.isfinite(data.reference.r_by_particle[index]).sum())}; "
            "missing intervals and collision paths are not reconstructed"
        ),
        css_class="small",
    )
    canvas.write(path)


def _write_atlas(
    label: str, particle_files: list[tuple[int, Path]], output: Path, overview: str, population: str
) -> None:
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
<p>Every common particle ID is included. Each side's observed finite spatial path is
drawn independently and is non-gating; observed terminal tails may be shown.
Missing spans are never reconstructed. Receipt metrics use {html.escape(population)}.
All particle views share the same spatial range and equal geometric aspect.</p>
<nav><a href="{overview}">All-particle spatial overview</a></nav></header>
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
    with tempfile.TemporaryDirectory(
        prefix="trajectory_plot_browser_", ignore_cleanup_errors=True
    ) as profile:
        for svg_path in svg_paths:
            png_path = svg_path.with_suffix(".png").resolve()
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
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"PNG rasterization failed for {svg_path}: {completed.stderr.strip()}"
                )
            _wait_for_png(png_path)
            dimensions = _png_dimensions(png_path)
            if dimensions != (1600, 1000):
                raise RuntimeError(f"unexpected PNG dimensions for {png_path}: {dimensions}")
    return {
        "executable": str(browser),
        "executable_sha256": _sha256(browser),
        "version": "not queried; executable bytes are hash-bound",
    }


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
            for key in data.metric_row_keys
        ],
        dtype=np.float64,
    )
    if values.size == 0:
        return {
            "position_difference_m": {
                "comparison_rows": 0,
                "rms": None,
                "maximum": None,
                "reason": "no jointly eligible finite active observations",
            }
        }
    worst_index = int(np.argmax(values))
    worst_particle_id, worst_time_s = data.metric_row_keys[worst_index]
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


def _table_coverage(table: TrajectoryTable) -> dict[str, object]:
    finite = set(table.row_keys)
    active = set(table.active_row_keys)
    counts = [sum(key[0] == int(pid) for key in finite) for pid in table.particle_ids]
    return {
        "input_rows": table.input_row_count,
        "finite_coordinate_rows": len(finite),
        "missing_coordinate_rows": table.missing_coordinate_row_count,
        "observed_status_counts": table.status_counts,
        "missing_coordinate_rows_by_observed_status": table.missing_coordinate_by_status,
        "active_finite_rows": len(active),
        "eligible_metric_rows": len(table.metric_row_keys),
        "non_active_finite_rows": len(finite - active),
        "observed_coordinate_rows_per_particle_min": min(counts),
        "observed_coordinate_rows_per_particle_max": max(counts),
        "observed_coordinate_time_s": list(_time_extent(table)),
        "last_observed_coordinate_time_s_range": list(_particle_end_extent(table)),
        "precontact_column": table.precontact_column,
    }


def _alignment_receipt(data: MatchedTrajectories) -> dict[str, object]:
    candidate = set(data.candidate.input_row_keys)
    reference = set(data.reference.input_row_keys)
    finite_candidate = set(data.candidate.row_keys)
    finite_reference = set(data.reference.row_keys)
    return {
        "status": "PASS",
        "particle_identity": "EXACT",
        "observed_state_key": ["particle_id", "time_s"],
        "metric_population": data.metric_population,
        "particles": len(data.candidate.particle_ids),
        "candidate_only_input_keys": len(candidate - reference),
        "reference_only_input_keys": len(reference - candidate),
        "omitted_row_reason": "unobserved; absence of a CSV row does not identify an escape or disappearance cause",
        "common_observed_finite_rows": len(finite_candidate & finite_reference),
        "common_active_finite_rows": len(data.common_active_row_keys),
        "common_eligible_metric_rows": len(data.metric_row_keys),
        "candidate": _table_coverage(data.candidate),
        "reference": _table_coverage(data.reference),
    }


def _geometry_receipt(geometry: GeometryOverlay | None) -> dict[str, object] | None:
    if geometry is None:
        return None
    return {
        "path": str(geometry.path),
        "sha256": geometry.sha256,
        "canonical_content_hash": geometry.content_hash,
        "coordinate_system": geometry.coordinate_system,
        "unit": "m",
        "boundary_facet_count": int(geometry.line2.shape[0]),
        "boundary_group_names": list(geometry.group_names),
        "axis_seam_facet_count": int(geometry.axis_line2.shape[0]),
        "axis_seam_meaning": "dotted symmetry/support seam excluded from material boundary facets",
        "binding_scope": "explicit supplied canonical geometry; normalized CSVs do not contain result/case hashes, so result ownership is not inferred",
    }


def create_trajectory_figures(
    candidate_path: Path,
    reference_path: Path,
    output_directory: Path,
    *,
    label: str,
    render_png: bool,
    browser_path: Path | None = None,
    geometry_path: Path | None = None,
    precontact_column: str = "precontact_eligible",
) -> dict[str, object]:
    """Create observed spatial figures and an explicit coverage/metric receipt."""
    output = output_directory.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    data = read_matched_trajectories(
        candidate_path, reference_path, precontact_column=precontact_column
    )
    geometry = _load_geometry(geometry_path, data.candidate.coordinate_system)
    _plot_ranges(data, geometry, PlotBox(150, 140, 1370, 760))
    output.mkdir(parents=True)
    particle_directory = output / "particles"
    particle_directory.mkdir()
    overview_name = (
        "overview_rz.svg"
        if data.candidate.coordinate_system == "axisymmetric_rz"
        else "overview_xy.svg"
    )
    summary_path = output / overview_name
    _render_overview(data, label, summary_path, geometry)
    particle_files: list[tuple[int, Path]] = []
    for index, raw_id in enumerate(data.candidate.particle_ids):
        particle_id = int(raw_id)
        path = particle_directory / f"particle_{particle_id:06d}.svg"
        _render_particle(data, label, index, path, geometry)
        particle_files.append((particle_id, path))
    atlas_path = output / "index.html"
    _write_atlas(label, particle_files, atlas_path, overview_name, data.metric_population)
    rasterizer = _rasterize([summary_path], _browser_path(browser_path)) if render_png else None
    figures = [_artifact(summary_path, output, "all_particle_spatial_svg")]
    figures.extend(_artifact(path, output, "particle_svg") for _, path in particle_files)
    if render_png:
        figures.append(
            _artifact(summary_path.with_suffix(".png"), output, "all_particle_spatial_png")
        )
    x_range, y_range = _plot_ranges(data, geometry, PlotBox(150, 140, 1370, 760))
    receipt: dict[str, object] = {
        "schema_version": 5,
        "tool_revision": TOOL_REVISION,
        "report_kind": "external_trajectory_figure_receipt",
        "label": label,
        "tool": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "scope": {
            "external_only": True,
            "simulation_executed": False,
            "comparison": "observed spatial paths and descriptive position differences",
            "trajectory_rendering_population": "all observed finite positions; missing spans break path lines",
            "missing_coordinate_policy": "unobserved coordinates are excluded without reconstruction; reasons retain supplied status",
            "metric_population": data.metric_population,
            "precontact_eligibility_authority": "normalized input flags; renderer does not infer unsaved collision times",
            "native_disappeared_cause": "unknown; not equated to candidate escaped",
            "figure_role": "NON_GATING_DIAGNOSTIC",
            "physical_model_validity": "NOT_CLAIMED",
            "boundary_accuracy": "NOT_TESTED_BY_FIGURE",
        },
        "inputs": {
            "candidate": {"path": str(data.candidate.path), "sha256": data.candidate.sha256},
            "reference": {"path": str(data.reference.path), "sha256": data.reference.sha256},
            "geometry": _geometry_receipt(geometry),
        },
        "alignment": _alignment_receipt(data),
        "metrics": _metrics(data),
        "rendering": {
            "svg": "standard_library",
            "png": rasterizer if rasterizer is not None else "SKIPPED",
            "coordinate_system": data.candidate.coordinate_system,
            "coordinate_view": (
                "RZ meridional projection; no 3D orbit is reconstructed"
                if data.candidate.coordinate_system == "axisymmetric_rz"
                else "Cartesian XY observations; out-of-plane motion is not inferred"
            ),
            "display_unit": "mm",
            "source_and_metric_units": "m and s",
            "overview_and_particle_ranges_mm": {"axis0": list(x_range), "axis1": list(y_range)},
            "equal_geometric_aspect": True,
            "path_rule": "straight visual segments join consecutive observed CSV rows; missing and collision paths are not reconstructed",
            "style": {
                "reference": "solid orange",
                "candidate": "dashed blue",
                "geometry": "grey",
                "axis_seam": "dotted grey",
            },
        },
        "atlas": _artifact(atlas_path, output, "html_atlas"),
        "particle_svg_count": len(particle_files),
        "figures": figures,
    }
    (output / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
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
    parser.add_argument("--geometry", type=Path)
    parser.add_argument("--precontact-column", default="precontact_eligible")
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
        geometry_path=args.geometry,
        precontact_column=args.precontact_column,
    )
    print(json.dumps(receipt["alignment"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
