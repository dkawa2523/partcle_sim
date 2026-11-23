"""Bounded result-only population summaries and streaming exports."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from chamber_particles.lifecycle import ACTIVE, NAMES, PENDING, STUCK, TERMINAL
from chamber_particles.output import BoundaryEvents, FinalParticles, ResultContext, ResultView

from ..selection import ResultInput, result_labels, select_result

ANALYSIS_REVISION = "result_population_interaction_terminal_v4"


@dataclass(slots=True)
class _Interaction:
    events: int = 0
    model_weight: float = 0.0
    weighted_time_s: float = 0.0
    first_time_s: float = float("inf")
    last_time_s: float = float("-inf")


@dataclass(slots=True)
class _ParticleEvents:
    terminal_time_s: np.ndarray
    first_hit_time_s: np.ndarray
    terminal_boundary_id: np.ndarray
    terminal_material_id: np.ndarray


def summarize_result(
    path: ResultInput,
    *,
    event_batch_rows: int = 4096,
    source_ids: tuple[int, ...] | None = None,
    source_names: tuple[str, ...] | None = None,
    particle_ids: tuple[int, ...] | None = None,
    particle_ids_file: str | PathLike[str] | None = None,
    coordinate0_bin_edges_m: tuple[float, ...] | None = None,
    terminal_time_bin_edges_s: tuple[float, ...] | None = None,
) -> dict[str, object]:
    """Summarize exact cohorts in O(particles + batch rows + saved times) memory.

    Histogram edges are explicit physical inputs. Interactions include repeated
    hits; first material-wall hits and terminal events are separate quantities.
    """
    cohort = select_result(
        path,
        source_ids=source_ids,
        source_names=source_names,
        particle_ids=particle_ids,
        particle_ids_file=particle_ids_file,
    )
    result, context, final, selected = (
        cohort.result,
        cohort.context,
        cohort.final,
        cohort.selected,
    )
    events, interactions = _scan_events(result, cohort.selected_ids, event_batch_rows)
    report: dict[str, object] = {
        "analysis_revision": ANALYSIS_REVISION,
        "parameters": {
            "event_batch_rows": event_batch_rows,
            **cohort.selection_document(),
            "coordinate0_bin_edges_m": coordinate0_bin_edges_m,
            "terminal_time_bin_edges_s": terminal_time_bin_edges_s,
        },
        "source_result": result.source_identity(),
        "labels": result_labels(context),
        "population": {
            "selected_particles": int(selected.sum()),
            "model_weight": float(final.model_weight[selected].sum()),
            "count_unit": "computational_particles",
            "weight_meaning": "input model_weight; no inferred physical rate",
        },
        "fate": _fate_rows(final.lifecycle[selected], final.model_weight[selected]),
        "sources": _source_rows(final, selected, context),
        "boundary_interactions": _interaction_rows(interactions),
        "first_boundary_hit": _time_summary(events.first_hit_time_s, final.model_weight[selected]),
        "terminal": _terminal_rows(final, selected, events),
        "deposition": _deposition_rows(final, selected, events),
        "deposition_coverage": {
            "eligible_stuck_particles": int(np.count_nonzero(final.lifecycle[selected] == STUCK)),
            "missing_terminal_boundary": int(
                np.count_nonzero(
                    (final.lifecycle[selected] == STUCK) & (events.terminal_boundary_id < 0)
                )
            ),
        },
        "active_curve": _active_curve(result, final, selected, events),
        "histograms": _histograms(
            final,
            selected,
            events,
            coordinate0_bin_edges_m,
            terminal_time_bin_edges_s,
            "r" if context.coordinate_system == "axisymmetric_rz" else "x",
        ),
        "meanings": {
            "boundary_interactions": "all recorded wall/topology interactions including repeats",
            "first_boundary_hit": "first recorded material-wall hit per particle; excludes periodic transfers",
            "terminal": "stop, hold, escape or failure; not first arrival",
            "deposition": "one numerically stuck final particle counted once; not certified physical adhesion",
        },
    }
    report["verification"] = result.verification_document()
    return report


def _selected_event_rows(particle_id: np.ndarray, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not ids.size:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    index = np.searchsorted(ids, particle_id)
    valid = index < ids.size
    valid[valid] &= ids[index[valid]] == particle_id[valid]
    rows = np.flatnonzero(valid)
    return rows, index[rows]


def _scan_events(
    result: ResultView, ids: np.ndarray, batch_rows: int
) -> tuple[_ParticleEvents, dict[tuple[int, int, str, str], _Interaction]]:
    events = _ParticleEvents(
        np.full(ids.size, np.nan),
        np.full(ids.size, np.inf),
        np.full(ids.size, -1, dtype=np.int32),
        np.full(ids.size, -1, dtype=np.int32),
    )
    totals: dict[tuple[int, int, str, str], _Interaction] = {}
    for batch in result.iter_boundary_event_batches(batch_rows=batch_rows):
        rows, index = _selected_event_rows(batch.particle_id, ids)
        _accumulate_interactions(totals, batch, rows)
        wall = batch.interaction_kind[rows] == "wall"
        np.minimum.at(events.first_hit_time_s, index[wall], batch.time_s[rows[wall]])
        terminal = np.isin(batch.outcome[rows], ("stuck", "escaped", "held"))
        for row, destination in zip(rows[terminal], index[terminal], strict=True):
            if np.isfinite(events.terminal_time_s[destination]):
                raise ValueError("a particle has more than one terminal boundary event")
            events.terminal_time_s[destination] = batch.time_s[row]
            events.terminal_boundary_id[destination] = batch.boundary_id[row]
            events.terminal_material_id[destination] = batch.material_id[row]
    failures = result.read_failure_events()
    rows, index = _selected_event_rows(failures.particle_id, ids)
    events.terminal_time_s[index] = failures.time_s[rows]
    return events, totals


def _accumulate_interactions(
    totals: dict[tuple[int, int, str, str], _Interaction], batch: BoundaryEvents, rows: np.ndarray
) -> None:
    for row in rows:
        key = (
            int(batch.boundary_id[row]),
            int(batch.material_id[row]),
            str(batch.interaction_kind[row]),
            str(batch.outcome[row]),
        )
        value = totals.setdefault(key, _Interaction())
        weight, time = float(batch.model_weight[row]), float(batch.time_s[row])
        value.events += 1
        value.model_weight += weight
        value.weighted_time_s += weight * time
        value.first_time_s = min(value.first_time_s, time)
        value.last_time_s = max(value.last_time_s, time)


def _interaction_rows(
    totals: dict[tuple[int, int, str, str], _Interaction],
) -> list[dict[str, object]]:
    return [
        {
            "boundary_id": boundary,
            "material_id": material,
            "interaction_kind": kind,
            "outcome": outcome,
            "events": v.events,
            "model_weight": v.model_weight,
            "first_time_s": v.first_time_s,
            "last_time_s": v.last_time_s,
            "weighted_mean_time_s": v.weighted_time_s / v.model_weight
            if v.model_weight > 0
            else None,
        }
        for (boundary, material, kind, outcome), v in sorted(totals.items())
    ]


def _fate_rows(lifecycle: np.ndarray, weights: np.ndarray) -> list[dict[str, object]]:
    return [
        {
            "lifecycle": NAMES[int(code)],
            "particles": int(np.count_nonzero(lifecycle == code)),
            "model_weight": float(weights[lifecycle == code].sum()),
        }
        for code in np.unique(lifecycle)
    ]


def _source_rows(
    final: FinalParticles, selected: np.ndarray, context: ResultContext
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source in context.sources:
        source_id, name = source.source_id, source.name
        mask = selected & (final.source_id == source_id)
        rows.append(
            {
                "source_id": source_id,
                "name": name,
                "table": source.table,
                "kind": source.kind,
                "particles": int(mask.sum()),
                "model_weight": float(final.model_weight[mask].sum()),
                "fate": _fate_rows(final.lifecycle[mask], final.model_weight[mask]),
            }
        )
    return rows


def _time_summary(times: np.ndarray, weights: np.ndarray) -> dict[str, object]:
    finite = np.isfinite(times)
    values = times[finite]
    return {
        "particles": int(values.size),
        "model_weight": float(weights[finite].sum()),
        "time_s_quantiles_p10_p50_p90": list(map(float, np.quantile(values, [0.1, 0.5, 0.9])))
        if values.size
        else None,
        "weighted_mean_time_s": float(np.average(values, weights=weights[finite]))
        if weights[finite].sum() > 0
        else None,
        "quantile_meaning": "unweighted computational-particle quantiles",
    }


def _terminal_rows(
    final: FinalParticles, selected: np.ndarray, events: _ParticleEvents
) -> list[dict[str, object]]:
    lifecycle, weights = final.lifecycle[selected], final.model_weight[selected]
    return [
        {
            "lifecycle": NAMES[code],
            "eligible_particles": int(np.count_nonzero(lifecycle == code)),
            "missing_terminal_events": int(
                np.count_nonzero((lifecycle == code) & ~np.isfinite(events.terminal_time_s))
            ),
            **_time_summary(events.terminal_time_s[lifecycle == code], weights[lifecycle == code]),
        }
        for code in TERMINAL
    ]


def _deposition_rows(
    final: FinalParticles, selected: np.ndarray, events: _ParticleEvents
) -> list[dict[str, object]]:
    retained = (final.lifecycle[selected] == STUCK) & (events.terminal_boundary_id >= 0)
    keys = np.column_stack(
        (events.terminal_boundary_id[retained], events.terminal_material_id[retained])
    )
    weights = final.model_weight[selected][retained]
    return [
        {
            "boundary_id": int(boundary),
            "material_id": int(material),
            "particles": int(np.count_nonzero(np.all(keys == (boundary, material), axis=1))),
            "model_weight": float(weights[np.all(keys == (boundary, material), axis=1)].sum()),
        }
        for boundary, material in np.unique(keys, axis=0)
    ]


def _active_curve(
    result: ResultView, final: FinalParticles, selected: np.ndarray, events: _ParticleEvents
) -> dict[str, object]:
    ids, weights = final.particle_id[selected], final.model_weight[selected]
    rows = []
    for frame in result.iter_frames():
        source, index = _selected_event_rows(frame.particle_id, ids)
        active = frame.lifecycle[source] == ACTIVE
        rows.append(
            {
                "time_s": frame.time_s,
                "active_particles": int(active.sum()),
                "active_model_weight": float(weights[index[active]].sum()),
            }
        )
    if rows:
        return {"available": True, "source": "saved trajectory frames", "rows": rows}
    series = result.read_lifecycle_series()
    derived = _ledger_active_curve(result, final, selected, events, series.time_s)
    if derived["available"]:
        return derived
    if np.all(selected):
        return {
            "available": bool(series.time_s.size),
            "source": "online lifecycle counts; weights unavailable",
            "rows": [
                {"time_s": float(time), "active_particles": int(count), "active_model_weight": None}
                for time, count in zip(series.time_s, series.active, strict=True)
            ],
        }
    return derived


def _ledger_active_curve(
    result: ResultView,
    final: FinalParticles,
    selected: np.ndarray,
    events: _ParticleEvents,
    clock: np.ndarray,
) -> dict[str, object]:
    ids = final.particle_id[selected]
    release = result.read_release_events()
    rows, index = _selected_event_rows(release.particle_id, ids)
    occurrences = np.bincount(index, minlength=ids.size)
    expected_released = final.lifecycle[selected] != PENDING
    expected_terminal = np.isin(final.lifecycle[selected], TERMINAL)
    complete = np.array_equal(occurrences, expected_released.astype(np.int64))
    complete &= np.array_equal(np.isfinite(events.terminal_time_s), expected_terminal)
    if not complete or not clock.size:
        return {
            "available": False,
            "reason": "selected-cohort release/terminal ledger coverage or online clock is unavailable",
            "rows": [],
        }
    times = release.time_s[rows]
    weights = final.model_weight[selected]
    released_count, released_weight = _cumulative_events(times, weights[index], clock)
    terminal = np.isfinite(events.terminal_time_s)
    terminal_count, terminal_weight = _cumulative_events(
        events.terminal_time_s[terminal], weights[terminal], clock
    )
    count = released_count - terminal_count
    active_weight = np.where(count == 0, 0.0, released_weight - terminal_weight)
    return {
        "available": True,
        "source": "release/terminal ledgers at online lifecycle clock; right-continuous",
        "ledger_coverage": {
            "released_particles": int(rows.size),
            "terminal_particles": int(terminal.sum()),
        },
        "rows": [
            {
                "time_s": float(time),
                "active_particles": int(n),
                "active_model_weight": float(weight),
            }
            for time, n, weight in zip(clock, count, active_weight, strict=True)
        ],
    }


def _cumulative_events(
    times: np.ndarray, weights: np.ndarray, clock: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(times, kind="stable")
    count = np.searchsorted(times[order], clock, side="right")
    cumulative_weight = np.concatenate(([0.0], np.cumsum(weights[order], dtype=np.float64)))
    return count, cumulative_weight[count]


def _explicit_histogram(
    values: np.ndarray, weights: np.ndarray, edges: tuple[float, ...] | None
) -> dict[str, object] | None:
    if edges is None:
        return None
    bins = np.asarray(edges, dtype=np.float64)
    if bins.size < 2 or not np.isfinite(bins).all() or not np.all(np.diff(bins) > 0):
        raise ValueError(
            "histogram edges must be finite, strictly increasing and contain at least two values"
        )
    finite = np.isfinite(values)
    count, _ = np.histogram(values[finite], bins=bins)
    weighted, _ = np.histogram(values[finite], bins=bins, weights=weights[finite])
    return {
        "edges": list(map(float, bins)),
        "particles": list(map(int, count)),
        "model_weight": list(map(float, weighted)),
        "missing_values": int(np.count_nonzero(~finite)),
        "outside_range": int(np.count_nonzero(finite & ((values < bins[0]) | (values > bins[-1])))),
        "bin_convention": "left-closed/right-open; final bin includes right edge",
    }


def _histograms(
    final: FinalParticles,
    selected: np.ndarray,
    events: _ParticleEvents,
    radial_edges: tuple[float, ...] | None,
    time_edges: tuple[float, ...] | None,
    coordinate0_name: str,
) -> dict[str, object]:
    retained, weights = final.lifecycle[selected] == STUCK, final.model_weight[selected]
    terminal = _explicit_histogram(events.terminal_time_s, weights, time_edges)
    if terminal is not None:
        nonterminal = np.isin(final.lifecycle[selected], (PENDING, ACTIVE))
        terminal["excluded_nonterminal_particles"] = int(nonterminal.sum())
        terminal["missing_terminal_event_particles"] = int(
            np.count_nonzero(~nonterminal & ~np.isfinite(events.terminal_time_s))
        )
    return {
        "coordinate0_axis": {"name": coordinate0_name, "unit": "m"},
        "retained_final_coordinate0_m": _explicit_histogram(
            final.position_m[selected, 0][retained], weights[retained], radial_edges
        ),
        "terminal_time_s": terminal,
    }


def export_result(
    path: ResultInput,
    output_directory: str | PathLike[str],
    *,
    particle_ids: tuple[int, ...] | None = None,
    source_ids: tuple[int, ...] | None = None,
    source_names: tuple[str, ...] | None = None,
    particle_ids_file: str | PathLike[str] | None = None,
    event_batch_rows: int = 4096,
) -> dict[str, object]:
    """Stream selected rows to CSV while preserving context and result identity."""
    cohort = select_result(
        path,
        source_ids=source_ids,
        source_names=source_names,
        particle_ids=particle_ids,
        particle_ids_file=particle_ids_file,
    )
    result, final, ids = cohort.result, cohort.final, cohort.selected_ids
    destination = Path(output_directory).expanduser().resolve()
    if destination == result.path or result.path in destination.parents:
        raise ValueError("export output must be outside the source result directory")
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".result-export-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "export"
        staging.mkdir()
        counts = _export_tables(result, final, ids, staging, event_batch_rows)
        shutil.copyfile(result.path / "context.h5", staging / "context.h5")
        files = {file.name: _file_hash(file) for file in staging.iterdir()}
        if files["context.h5"] != result.manifest_document()["context"]["file_sha256"]:
            raise ValueError("result context changed after opening; export was not published")
        report: dict[str, object] = {
            "analysis_revision": ANALYSIS_REVISION,
            "source_result": result.source_identity(),
            "verification": result.verification_document(),
            "selection": {
                **cohort.selection_document(),
                "selected_particles": int(ids.size),
            },
            "labels": result_labels(cohort.context),
            "rows": counts,
            "units": "SI",
            "boundary_meaning": "recorded interactions, including repeats; not first arrivals",
            "row_order": "frames/probes by time; boundary interaction storage order with explicit physical event identity",
            "files": files,
        }
        (staging / "export.json").write_text(
            json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        staging.rename(destination)
    return report


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _csv_rows(path: Path, header: tuple[str, ...], rows: Iterable[tuple[Any, ...]]) -> int:
    count = 0
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def _state_rows(frames: Iterable[Any], ids: np.ndarray) -> Iterable[tuple[Any, ...]]:
    for frame in frames:
        rows, _ = _selected_event_rows(frame.particle_id, ids)
        for row in rows:
            yield (
                frame.time_s,
                int(frame.particle_id[row]),
                *map(float, frame.position_m[row]),
                *map(float, frame.velocity_m_s[row]),
                float(frame.charge_number[row]),
                int(frame.lifecycle[row]),
            )


def _boundary_rows(
    result: ResultView, ids: np.ndarray, batch_rows: int
) -> Iterable[tuple[Any, ...]]:
    for batch in result.iter_boundary_event_batches(batch_rows=batch_rows):
        rows, _ = _selected_event_rows(batch.particle_id, ids)
        for row in rows:
            yield (
                float(batch.time_s[row]),
                int(batch.particle_id[row]),
                int(batch.event_ordinal[row]),
                str(batch.interaction_kind[row]),
                int(batch.boundary_id[row]),
                str(batch.outcome[row]),
                *map(float, batch.position_m[row]),
                float(batch.model_weight[row]),
                int(batch.material_id[row]),
                int(batch.primary_facet_id[row]),
                int(batch.destination_facet_id[row]),
                float(batch.contact_radius_m[row]),
                *map(float, batch.position_post_m[row]),
                *map(float, batch.normal[row]),
                *map(float, batch.velocity_pre_m_s[row]),
                *map(float, batch.velocity_post_m_s[row]),
                float(batch.charge_number_pre[row]),
                float(batch.charge_number_post[row]),
                str(batch.law_id[row]),
                float(batch.localization_residual_m[row]),
                float(batch.position_budget_m[row]),
                float(batch.time_budget_s[row]),
                json.dumps(
                    batch.candidate_facet_id[
                        batch.candidate_offset[row] : batch.candidate_offset[row + 1]
                    ].tolist()
                ),
            )


def _release_rows(result: ResultView, ids: np.ndarray) -> Iterable[tuple[Any, ...]]:
    release = result.read_release_events()
    rows, _ = _selected_event_rows(release.particle_id, ids)
    for row in rows:
        yield (
            float(release.time_s[row]),
            int(release.particle_id[row]),
            int(release.event_ordinal[row]),
            int(release.source_id[row]),
        )


def _failure_rows(result: ResultView, ids: np.ndarray) -> Iterable[tuple[Any, ...]]:
    failure = result.read_failure_events()
    rows, _ = _selected_event_rows(failure.particle_id, ids)
    for row in rows:
        yield (
            float(failure.time_s[row]),
            int(failure.particle_id[row]),
            int(failure.event_ordinal[row]),
            int(failure.reason_code[row]),
        )


def _export_tables(
    result: ResultView, final: FinalParticles, ids: np.ndarray, destination: Path, batch_rows: int
) -> dict[str, int]:
    rows, _ = _selected_event_rows(final.particle_id, ids)
    final_rows = (
        (
            int(final.particle_id[row]),
            int(final.source_id[row]),
            float(final.time_s[row]),
            *map(float, final.position_m[row]),
            *map(float, final.velocity_m_s[row]),
            float(final.charge_number[row]),
            int(final.lifecycle[row]),
            int(final.kinematics_valid[row]),
            float(final.model_weight[row]),
            int(final.failure_reason_code[row]),
            float(final.mass_kg[row]),
            float(final.drag_diameter_m[row]),
            float(final.contact_radius_m[row]),
            float(final.electrostatic_radius_m[row]),
            float(final.displaced_volume_m3[row]),
            int(final.material_id[row]),
        )
        for row in rows
    )
    velocity_header = tuple(f"velocity{i}_m_s" for i in range(final.velocity_m_s.shape[1]))
    state_header = (
        "time_s",
        "particle_id",
        "coordinate0_m",
        "coordinate1_m",
        *velocity_header,
        "charge_number",
        "lifecycle",
    )
    return {
        "release_events": _csv_rows(
            destination / "release_events.csv",
            ("time_s", "particle_id", "event_ordinal", "source_id"),
            _release_rows(result, ids),
        ),
        "failure_events": _csv_rows(
            destination / "failure_events.csv",
            ("time_s", "particle_id", "event_ordinal", "reason_code"),
            _failure_rows(result, ids),
        ),
        "final": _csv_rows(
            destination / "final.csv",
            (
                "particle_id",
                "source_id",
                "time_s",
                "coordinate0_m",
                "coordinate1_m",
                *velocity_header,
                "charge_number",
                "lifecycle",
                "kinematics_valid",
                "model_weight",
                "failure_reason_code",
                "mass_kg",
                "drag_diameter_m",
                "contact_radius_m",
                "electrostatic_radius_m",
                "displaced_volume_m3",
                "material_id",
            ),
            final_rows,
        ),
        "frames": _csv_rows(
            destination / "frames.csv", state_header, _state_rows(result.iter_frames(), ids)
        ),
        "probes": _csv_rows(
            destination / "probes.csv", state_header, _state_rows(result.iter_probes(), ids)
        ),
        "boundary_interactions": _csv_rows(
            destination / "boundary_interactions.csv",
            (
                "time_s",
                "particle_id",
                "event_ordinal",
                "interaction_kind",
                "boundary_id",
                "outcome",
                "coordinate0_m",
                "coordinate1_m",
                "model_weight",
                "material_id",
                "primary_facet_id",
                "destination_facet_id",
                "contact_radius_m",
                "position_post0_m",
                "position_post1_m",
                "normal0",
                "normal1",
                "velocity_pre0_m_s",
                "velocity_pre1_m_s",
                "velocity_post0_m_s",
                "velocity_post1_m_s",
                "charge_number_pre",
                "charge_number_post",
                "law_id",
                "localization_residual_m",
                "position_budget_m",
                "time_budget_s",
                "candidate_facet_ids_json",
            ),
            _boundary_rows(result, ids, batch_rows),
        ),
    }


__all__ = ["ANALYSIS_REVISION", "export_result", "summarize_result"]
