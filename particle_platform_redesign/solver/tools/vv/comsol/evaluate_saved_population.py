"""Describe saved native RZ populations and optional completed public results.

Native status, terminal stopping time and disappeared cause have separate
meanings. This tool supplies no statistical or numerical agreement verdict.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from chamber_particles import open_result

TOOL_REVISION = "saved_rz_population_description_v1"
_NATIVE_NAMES = {0: "pending", 1: "active", 2: "frozen", 3: "stuck", 4: "disappeared_unknown"}
_RESULT_NAMES = {0: "pending", 1: "active", 2: "stuck", 3: "escaped", 4: "failed", 5: "held"}
_STATE_COLUMNS = ("r_m", "z_m", "vr_m_s", "vz_m_s", "charge_number")
_REQUIRED = {
    "particle_id",
    "time_s",
    *_STATE_COLUMNS,
    "native_status",
    "release_time_s",
    "stop_time_s",
    "native_final_status",
}
type _Row = tuple[int, float, list[float], int, float, float, int]


@dataclass(frozen=True)
class _Population:
    ids: np.ndarray
    times: np.ndarray
    states: np.ndarray
    status: np.ndarray
    observed: np.ndarray
    release_s: np.ndarray
    terminal_s: np.ndarray
    terminal_interval_s: np.ndarray
    names: dict[int, str]
    evidence: dict[str, Any]


def _artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _read_rows(path: Path) -> list[_Row]:
    rows: list[_Row] = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not _REQUIRED.issubset(reader.fieldnames or ()):
            raise ValueError("native CSV lacks required population columns")
        for row in reader:
            status, final = float(row["native_status"]), float(row["native_final_status"])
            if status not in _NATIVE_NAMES or final not in _NATIVE_NAMES:
                raise ValueError("native status code is unknown or nonfinite")
            time_s, release = float(row["time_s"]), float(row["release_time_s"])
            if not np.isfinite([time_s, release]).all():
                raise ValueError("native observation/release times must be finite")
            rows.append(
                (
                    int(row["particle_id"]),
                    time_s,
                    [float(row[key]) for key in _STATE_COLUMNS],
                    int(status),
                    release,
                    float(row["stop_time_s"]),
                    int(final),
                )
            )
    if not rows:
        raise ValueError("native CSV is empty")
    return rows


def _native(path: Path, terminal_times: bool) -> _Population:
    rows = _read_rows(path)
    metadata = _native_identity(rows, terminal_times)
    ids = np.asarray(sorted(metadata), dtype=np.int64)
    times = np.asarray(sorted({row[1] for row in rows}), dtype=np.float64)
    states = np.full((times.size, ids.size, 5), np.nan)
    status = np.full((times.size, ids.size), -1, dtype=np.int8)
    observed = np.zeros(status.shape, dtype=np.bool_)
    releases = np.asarray([metadata[int(pid)][0] for pid in ids])
    stops = np.asarray([metadata[int(pid)][1] for pid in ids])
    finals = np.asarray([metadata[int(pid)][2] for pid in ids], dtype=np.int8)
    id_index = {int(pid): index for index, pid in enumerate(ids)}
    clock = {float(time): index for index, time in enumerate(times)}
    for pid, time_s, state, code, _release, _stop, _final in rows:
        pi, ti = id_index[pid], clock[time_s]
        if observed[ti, pi]:
            raise ValueError("duplicate native particle/time row")
        states[ti, pi], status[ti, pi], observed[ti, pi] = state, code, True
    intervals = _native_terminal_intervals(times, status, observed, releases, finals)
    exact = _native_terminal_times(stops, intervals, finals, terminal_times)
    return _Population(
        ids,
        times,
        states,
        status,
        observed,
        releases,
        exact,
        intervals,
        _NATIVE_NAMES,
        {
            "trajectory": _artifact(path),
            "stop_time_meaning": "declared terminal stop"
            if terminal_times
            else "unknown; observation brackets only",
        },
    )


def _native_identity(rows: list[_Row], terminal_times: bool) -> dict[int, tuple[float, float, int]]:
    metadata: dict[int, tuple[float, float, int]] = {}
    for pid, _time, _state, _code, release, stop, final in rows:
        previous = metadata.get(pid)
        if previous is not None:
            if previous[0] != release or previous[2] != final:
                raise ValueError("native release/final identity changes across observations")
            if terminal_times and final in (2, 3, 4) and previous[1] != stop:
                raise ValueError("native terminal stop time changes across observations")
        metadata[pid] = release, stop, final
    return metadata


def _native_terminal_times(
    stops: np.ndarray, intervals: np.ndarray, finals: np.ndarray, enabled: bool
) -> np.ndarray:
    exact = np.full(finals.size, np.nan)
    if enabled:
        ended = np.isin(finals, (2, 3, 4))
        if not np.isfinite(stops[ended]).all() or not bool(
            ((stops[ended] >= intervals[ended, 0]) & (stops[ended] <= intervals[ended, 1])).all()
        ):
            raise ValueError("native terminal stop time is outside the observed transition bracket")
        exact[ended] = stops[ended]
    return exact


def _native_terminal_intervals(
    times: np.ndarray,
    status: np.ndarray,
    observed: np.ndarray,
    release: np.ndarray,
    finals: np.ndarray,
) -> np.ndarray:
    intervals = np.full((finals.size, 2), np.nan)
    for pi, final in enumerate(finals):
        codes = status[observed[:, pi], pi]
        if codes[-1] != final:
            raise ValueError("native final status does not match the final observed status")
        ended = np.flatnonzero(observed[:, pi] & np.isin(status[:, pi], (2, 3, 4)))
        if not ended.size:
            continue
        first = int(ended[0])
        if not bool((status[first:, pi][observed[first:, pi]] == final).all()):
            raise ValueError("native lifecycle changes after a terminal state")
        active = np.flatnonzero(observed[:first, pi] & (status[:first, pi] == 1))
        intervals[pi] = (times[active[-1]] if active.size else release[pi], times[first])
    return intervals


def _candidate(path: Path) -> _Population:
    result = open_result(path)
    manifest = result.manifest_document()
    if manifest.get("motion_mode") != "axisymmetric_rz_meridional":
        raise ValueError("candidate must use the native RZ meridional coordinate meaning")
    final = result.read_final()
    ids = final.particle_id
    frames = list(result.iter_frames())
    if not frames:
        raise ValueError("candidate requires saved full-population trajectory frames")
    times = np.asarray([frame.time_s for frame in frames])
    span = cast(dict[str, Any], manifest["time"])
    if times[0] != span["start_s"] or times[-1] != span["end_s"]:
        raise ValueError("candidate frames must cover its complete executed time window")
    states, status, observed = _candidate_frame_arrays(frames, ids)
    release = np.full(ids.size, np.nan)
    released = result.read_release_events()
    release[np.searchsorted(ids, released.particle_id)] = released.time_s
    stops, terminal_times, terminal_codes = _candidate_terminal_times(result, ids)
    _fill_candidate_lifecycle(
        status, observed, times, ids, release, terminal_times, terminal_codes, final.lifecycle
    )
    evidence = {
        "run_manifest": _artifact(path / "run.json"),
        "case_file_hash": manifest.get("case_file_hash"),
        "data_content_hash": manifest.get("data_content_hash"),
        "failure_reason_counts": manifest.get("failure_reason_counts"),
        "stop_time_meaning": "terminal boundary event; failures are separate",
    }
    return _Population(
        ids,
        times,
        states,
        status,
        observed,
        release,
        stops,
        np.column_stack((stops, stops)),
        _RESULT_NAMES,
        evidence,
    )


def _candidate_frame_arrays(
    frames: list[Any], ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states = np.full((len(frames), ids.size, 5), np.nan)
    status = np.full((len(frames), ids.size), -1, dtype=np.int8)
    observed = np.zeros(status.shape, dtype=np.bool_)
    for ti, frame in enumerate(frames):
        indices = np.searchsorted(ids, frame.particle_id)
        if bool((indices >= ids.size).any()) or not np.array_equal(ids[indices], frame.particle_id):
            raise ValueError("candidate frame contains IDs outside its final population")
        states[ti, indices] = np.column_stack(
            (frame.position_m, frame.velocity_m_s, frame.charge_number)
        )
        status[ti, indices], observed[ti, indices] = frame.lifecycle, True
    return states, status, observed


def _candidate_terminal_times(
    result: Any, ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stops = np.full(ids.size, np.nan)
    terminal_times = np.full(ids.size, np.inf)
    codes = np.full(ids.size, -1, dtype=np.int8)
    outcomes = {"stuck": 2, "escaped": 3, "held": 5}
    for batch in result.iter_boundary_event_batches():
        ended = np.isin(batch.outcome, ("stuck", "held", "escaped"))
        indices = np.searchsorted(ids, batch.particle_id[ended])
        stops[indices] = terminal_times[indices] = batch.time_s[ended]
        codes[indices] = [outcomes[str(outcome)] for outcome in batch.outcome[ended]]
    failures = result.read_failure_events()
    indices = np.searchsorted(ids, failures.particle_id)
    terminal_times[indices], codes[indices] = failures.time_s, 4
    return stops, terminal_times, codes


def _fill_candidate_lifecycle(
    status: np.ndarray,
    observed: np.ndarray,
    times: np.ndarray,
    ids: np.ndarray,
    release: np.ndarray,
    terminal_times: np.ndarray,
    terminal_codes: np.ndarray,
    final_status: np.ndarray,
) -> None:
    pending = (times[:, None] < release) | (~np.isfinite(release) & (final_status == 0))
    ended = times[:, None] >= terminal_times
    unexplained = ~observed & ~(pending | ended)
    if unexplained.any():
        ti, pi = np.argwhere(unexplained)[0]
        raise ValueError(
            f"missing active/pre-terminal candidate state for ID {ids[pi]} at {times[ti]} s"
        )
    status[~observed & pending] = 0
    known_terminal = ~observed & ended
    status[known_terminal] = np.broadcast_to(terminal_codes, status.shape)[known_terminal]
    if not np.array_equal(status[-1], final_status):
        raise ValueError("candidate final frame/terminal ledger disagrees with final lifecycle")


def _quantiles(values: np.ndarray) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return {"finite_count": 0, "p10": None, "p50": None, "p90": None}
    quantiles = np.quantile(finite, (0.1, 0.5, 0.9))
    return {
        "finite_count": int(finite.size),
        **{
            name: float(value) for name, value in zip(("p10", "p50", "p90"), quantiles, strict=True)
        },
    }


def _named_mask(table: _Population, name: str) -> np.ndarray:
    codes = [code for code, label in table.names.items() if label == name]
    return np.isin(table.status, codes)


def _mean_active_lifetime(table: _Population) -> float | None:
    terminal = [
        code for code, name in table.names.items() if name not in {"pending", "active", "failed"}
    ]
    supported = [code for code, name in table.names.items() if name != "failed"]
    ends = np.where(np.isin(table.status[-1], terminal), table.terminal_s, table.times[-1])
    if not bool(np.isin(table.status[-1], supported).all()) or not np.isfinite(ends).all():
        return None
    if not np.isfinite(table.release_s).all():
        return None
    durations = np.maximum(
        0.0, np.minimum(ends, table.times[-1]) - np.maximum(table.release_s, table.times[0])
    )
    return float(np.mean(durations))


def _describe(table: _Population) -> dict[str, Any]:
    active = _named_mask(table, "active")
    finite = np.isfinite(table.states).all(axis=2)
    capture = _named_mask(table, "stuck")[-1]
    series: list[dict[str, Any]] = []
    for ti, time_s in enumerate(table.times):
        series.append(
            {
                "time_s": float(time_s),
                "status_counts": {
                    name: int(np.sum(table.status[ti] == code))
                    for code, name in table.names.items()
                },
                "unobserved_state_row_count": int(np.sum(~table.observed[ti])),
                "unknown_lifecycle_count": int(np.sum(table.status[ti] < 0)),
                "active_speed_m_s": _quantiles(
                    np.linalg.norm(table.states[ti, active[ti], 2:4], axis=1)
                ),
                "active_charge_number": _quantiles(table.states[ti, active[ti], 4]),
            }
        )
    active_fraction = np.sum(active, axis=1) / table.ids.size
    return {
        "registered_particle_count": int(table.ids.size),
        "observation_count": int(table.times.size),
        "window_s": [float(table.times[0]), float(table.times[-1])],
        "coverage": {
            "expected_rows": int(table.observed.size),
            "observed_rows": int(table.observed.sum()),
            "missing_rows": int((~table.observed).sum()),
            "lifecycle_known_rows": int((table.status >= 0).sum()),
            "lifecycle_unknown_rows": int((table.status < 0).sum()),
            "nonfinite_state_rows": int((~finite).sum()),
            "nonfinite_observed_state_rows": int((table.observed & ~finite).sum()),
            "nonfinite_active_state_rows": int((active & ~finite).sum()),
        },
        "final_counts": series[-1]["status_counts"],
        "final_fractions": {
            name: count / table.ids.size for name, count in series[-1]["status_counts"].items()
        },
        "saved_active_fraction_integral_s": float(np.trapezoid(active_fraction, table.times)),
        "restricted_mean_active_lifetime_s": _mean_active_lifetime(table),
        "terminal_stop_time_s_by_outcome": {
            name: _quantiles(table.terminal_s[_named_mask(table, name)[-1]])
            for name in table.names.values()
            if name not in {"pending", "active", "failed"}
        },
        "capture_terminal_time_s": _quantiles(table.terminal_s[capture]),
        "capture_terminal_observation_brackets_s": {
            "lower": _quantiles(table.terminal_interval_s[capture, 0]),
            "upper": _quantiles(table.terminal_interval_s[capture, 1]),
        },
        "series": series,
        "evidence": table.evidence,
    }


def _compare(left: _Population, right: _Population) -> dict[str, Any]:
    if not np.array_equal(left.ids, right.ids):
        raise ValueError("candidate/reference stable ID populations differ")
    if not np.array_equal(left.times[[0, -1]], right.times[[0, -1]]):
        raise ValueError("candidate/reference must cover the same complete observation window")
    times, li, ri = np.intersect1d(left.times, right.times, return_indices=True)
    mask = _named_mask(left, "active")[li] & _named_mask(right, "active")[ri]
    mask &= np.isfinite(left.states[li]).all(axis=2) & np.isfinite(right.states[ri]).all(axis=2)
    difference = left.states[li] - right.states[ri]
    raw = {}
    for name, values in (
        ("position_m", np.linalg.norm(difference[:, :, :2], axis=2)),
        ("velocity_m_s", np.linalg.norm(difference[:, :, 2:4], axis=2)),
        ("charge_number", np.abs(difference[:, :, 4])),
    ):
        selected = values[mask]
        raw[name] = {
            "rows": int(selected.size),
            "rms": float(np.sqrt(np.mean(selected**2))) if selected.size else None,
            "maximum": float(selected.max()) if selected.size else None,
        }
    population = {}
    for name in ("active", "stuck"):
        delta = np.sum(_named_mask(left, name)[li], axis=1) - np.sum(
            _named_mask(right, name)[ri], axis=1
        )
        population[name] = {
            "candidate_minus_reference_counts": delta.tolist(),
            "maximum_abs_fraction_difference": float(np.max(np.abs(delta)) / left.ids.size)
            if delta.size
            else None,
        }
    return {
        "common_exact_observation_times_s": times.tolist(),
        "candidate_unmatched_clock_count": int(left.times.size - times.size),
        "reference_unmatched_clock_count": int(right.times.size - times.size),
        "population_curves": population,
        "initial_state": _initial_state_difference(left, right),
        "same_ID_difference_weighting": "equal finite-active ID/time rows; not time-weighted on the nonuniform clock",
        "same_ID_finite_active_differences_descriptive_only": raw,
    }


def _initial_state_difference(left: _Population, right: _Population) -> dict[str, Any]:
    difference = left.states[0] - right.states[0]
    valid = np.isfinite(left.states[0]).all(axis=1) & np.isfinite(right.states[0]).all(axis=1)
    release_valid = np.isfinite(left.release_s) & np.isfinite(right.release_s)
    return {
        "meaning": "states at common window start; release times are compared separately",
        "paired_finite_rows": int(valid.sum()),
        "maximum_abs_component_difference": np.max(np.abs(difference[valid]), axis=0).tolist()
        if valid.any()
        else None,
        "components": ["r_m", "z_m", "vr_m_s", "vz_m_s", "charge_number"],
        "paired_finite_release_times": int(release_valid.sum()),
        "maximum_abs_release_time_difference_s": float(
            np.max(np.abs(left.release_s[release_valid] - right.release_s[release_valid]))
        )
        if release_valid.any()
        else None,
    }


def evaluate(
    reference: Path, *, reference_terminal_times: bool = False, candidate: Path | None = None
) -> dict[str, Any]:
    native = _native(reference, reference_terminal_times)
    report: dict[str, Any] = {
        "tool_revision": TOOL_REVISION,
        "tool": _artifact(Path(__file__)),
        "report_kind": "descriptive_saved_population_comparison",
        "qualified_solver_parity": "NOT_TESTED",
        "physical_validation": "NOT_CERTIFIED",
        "interpretation": {
            "weights": "computational particle counts; physical flux/weight not inferred",
            "native_disappeared": "unknown cause; not equated to candidate escaped",
            "terminal_time": "capture/terminal stop, never first wall arrival",
            "clock": "exact common stored times only; no interpolation or output-step inference",
            "omitted_candidate_rows": "only lifecycle is recovered after a recorded terminal/failure event, or before release; kinematics remain missing",
            "random_paths": "same IDs/seeds do not establish coupled Brownian paths",
            "allowance": "none; raw descriptive differences, no fitted threshold or agreement verdict",
            "survivors": "speed/charge quantiles condition on each participant's own active cohort",
            "lifetime": "release to known terminal stop, censored at window edges; unknown stops/failures leave the mean untested",
        },
        "reference": _describe(native),
    }
    if candidate is not None:
        actual = _candidate(candidate)
        report.update(candidate=_describe(actual), comparison=_compare(actual, native))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument(
        "--reference-terminal-times",
        action="store_true",
        help="Use native st only when independently bound as terminal stopping time.",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = evaluate(
        args.reference,
        reference_terminal_times=args.reference_terminal_times,
        candidate=args.candidate,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
