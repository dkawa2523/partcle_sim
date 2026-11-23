"""Describe time-grouped saved native RZ populations and public results.

Native status, terminal stopping time and disappeared cause have separate
meanings. This tool supplies no statistical or numerical agreement verdict.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, cast

import numpy as np

from chamber_particles import open_result
from chamber_particles.lifecycle import NAMES

TOOL_REVISION = "saved_rz_population_description_v2"
_NATIVE_NAMES = {0: "pending", 1: "active", 2: "frozen", 3: "stuck", 4: "disappeared_unknown"}
_RESULT_NAMES = dict(NAMES)
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
class _Frame:
    time_s: float
    states: np.ndarray
    status: np.ndarray
    observed: np.ndarray


@dataclass(frozen=True)
class _Population:
    ids: np.ndarray
    frames: Iterator[_Frame]
    release_s: np.ndarray
    terminal_s: np.ndarray
    terminal_interval_s: np.ndarray
    names: dict[int, str]
    evidence: dict[str, Any]


@dataclass
class _NativeParticle:
    release: float
    stop: float
    final: int
    last_status: int
    last_active: float | None = None
    interval: tuple[float, float] | None = None


def _artifact(path: Path) -> dict[str, str]:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path.resolve()), "sha256": digest}


def _text_lines(stream: BinaryIO, digest: Any) -> Iterator[str]:
    first = True
    for line in stream:
        digest.update(line)
        yield line.decode("utf-8-sig" if first else "utf-8")
        first = False


def _row(document: dict[str, Any]) -> _Row:
    if None in document or any(value is None for value in document.values()):
        raise ValueError("native CSV row has an inconsistent column count")
    status, final = float(document["native_status"]), float(document["native_final_status"])
    if status not in _NATIVE_NAMES or final not in _NATIVE_NAMES:
        raise ValueError("native status code is unknown or nonfinite")
    time_s, release = float(document["time_s"]), float(document["release_time_s"])
    if not np.isfinite([time_s, release]).all():
        raise ValueError("native observation/release times must be finite")
    pid = int(document["particle_id"])
    if not 0 <= pid <= np.iinfo(np.int64).max:
        raise ValueError("native particle_id must be a nonnegative int64 integer")
    return (
        pid,
        time_s,
        [float(document[key]) for key in _STATE_COLUMNS],
        int(status),
        release,
        float(document["stop_time_s"]),
        int(final),
    )


def _read_rows(stream: BinaryIO, digest: Any) -> Iterator[_Row]:
    reader = csv.DictReader(_text_lines(stream, digest))
    columns = reader.fieldnames or ()
    duplicates = sorted(name for name, count in Counter(columns).items() if count > 1)
    if duplicates:
        raise ValueError(f"native CSV has duplicate column names: {duplicates}")
    if not _REQUIRED.issubset(columns):
        raise ValueError("native CSV lacks required population columns")
    previous_time: float | None = None
    group_ids: set[int] = set()
    for document in reader:
        row = _row(document)
        pid, time_s = row[:2]
        if previous_time is not None and time_s < previous_time:
            raise ValueError(
                "native CSV time groups must be ascending; reorder rows by time_s "
                "before evaluation (particle IDs within each group may be unordered)"
            )
        if time_s != previous_time:
            group_ids.clear()
        if pid in group_ids:
            raise ValueError("duplicate native particle/time row")
        group_ids.add(pid)
        previous_time = time_s
        yield row
    if previous_time is None:
        raise ValueError("native CSV is empty")


def _update_native(particle: _NativeParticle, row: _Row, terminal_times: bool) -> None:
    _pid, time_s, _state, code, release, stop, final = row
    if particle.release != release or particle.final != final:
        raise ValueError("native release/final identity changes across observations")
    if terminal_times and final in (2, 3, 4) and particle.stop != stop:
        raise ValueError("native terminal stop time changes across observations")
    if particle.interval is not None and code != final:
        raise ValueError("native lifecycle changes after a terminal state")
    if code == 1:
        particle.last_active = time_s
    elif code in (2, 3, 4) and particle.interval is None:
        if code != final:
            raise ValueError("native lifecycle changes after a terminal state")
        lower = particle.last_active if particle.last_active is not None else release
        particle.interval = lower, time_s
    particle.last_status = code


def _native(stream: BinaryIO, path: Path, terminal_times: bool) -> _Population:
    metadata: dict[int, _NativeParticle] = {}
    digest = hashlib.sha256()
    for row in _read_rows(stream, digest):
        pid, _time, _state, code, release, stop, final = row
        particle = metadata.setdefault(pid, _NativeParticle(release, stop, final, code))
        _update_native(particle, row, terminal_times)
    ids = np.asarray(sorted(metadata), dtype=np.int64)
    particles = [metadata[int(pid)] for pid in ids]
    if any(particle.last_status != particle.final for particle in particles):
        raise ValueError("native final status does not match the final observed status")
    intervals = np.asarray(
        [
            particle.interval if particle.interval is not None else (np.nan, np.nan)
            for particle in particles
        ]
    )
    finals = np.asarray([particle.final for particle in particles])
    stops = np.asarray([particle.stop for particle in particles])
    exact = _native_terminal_times(stops, intervals, finals, terminal_times)
    expected = digest.hexdigest()
    stream.seek(0)
    return _Population(
        ids,
        _native_frames(stream, ids, expected),
        np.asarray([particle.release for particle in particles]),
        exact,
        intervals,
        _NATIVE_NAMES,
        {
            "trajectory": {"path": str(path.resolve()), "sha256": expected},
            "stop_time_meaning": "declared terminal stop"
            if terminal_times
            else "unknown; observation brackets only",
        },
    )


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


def _empty_frame(time_s: float, size: int) -> _Frame:
    return _Frame(
        time_s,
        np.full((size, 5), np.nan),
        np.full(size, -1, dtype=np.int8),
        np.zeros(size, dtype=np.bool_),
    )


def _native_frames(stream: BinaryIO, ids: np.ndarray, expected: str) -> Iterator[_Frame]:
    digest = hashlib.sha256()
    frame: _Frame | None = None
    id_index = {int(pid): index for index, pid in enumerate(ids)}
    for pid, time_s, state, code, _release, _stop, _final in _read_rows(stream, digest):
        if frame is None or time_s != frame.time_s:
            if frame is not None:
                yield frame
            frame = _empty_frame(time_s, ids.size)
        if pid not in id_index:
            raise ValueError("native CSV changed during evaluation")
        index = id_index[pid]
        frame.states[index], frame.status[index], frame.observed[index] = state, code, True
    if digest.hexdigest() != expected:
        raise ValueError("native CSV changed during evaluation")
    if frame is not None:
        yield frame


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


def _candidate(path: Path) -> _Population:
    result = open_result(path)
    manifest = result.manifest_document()
    if manifest.get("motion_mode") != "axisymmetric_rz_meridional":
        raise ValueError("candidate must use the native RZ meridional coordinate meaning")
    final = result.read_final()
    ids = final.particle_id
    release = np.full(ids.size, np.nan)
    released = result.read_release_events()
    release[np.searchsorted(ids, released.particle_id)] = released.time_s
    stops, terminal_times, terminal_codes = _candidate_terminal_times(result, ids)
    span = cast(dict[str, Any], manifest["time"])
    frames = _candidate_frames(
        result, ids, release, terminal_times, terminal_codes, final.lifecycle, span
    )
    return _Population(
        ids,
        frames,
        release,
        stops,
        np.column_stack((stops, stops)),
        _RESULT_NAMES,
        {
            "result": result.source_identity(),
            "run_manifest": _artifact(path / "run.json"),
            "case_file_hash": manifest.get("case_file_hash"),
            "data_content_hash": manifest.get("data_content_hash"),
            "failure_reason_counts": manifest.get("failure_reason_counts"),
            "stop_time_meaning": "terminal boundary event; failures are separate",
        },
    )


def _candidate_frames(
    result: Any,
    ids: np.ndarray,
    release: np.ndarray,
    terminal_times: np.ndarray,
    terminal_codes: np.ndarray,
    final_status: np.ndarray,
    span: dict[str, Any],
) -> Iterator[_Frame]:
    previous: _Frame | None = None
    for saved in result.iter_frames():
        if previous is None and saved.time_s != span["start_s"]:
            raise ValueError("candidate frames must cover its complete executed time window")
        if previous is not None and saved.time_s <= previous.time_s:
            raise ValueError("candidate saved clocks must be strictly ascending")
        frame = _empty_frame(saved.time_s, ids.size)
        indices = np.searchsorted(ids, saved.particle_id)
        if bool((indices >= ids.size).any()) or not np.array_equal(ids[indices], saved.particle_id):
            raise ValueError("candidate frame contains IDs outside its final population")
        frame.states[indices] = np.column_stack(
            (saved.position_m, saved.velocity_m_s, saved.charge_number)
        )
        frame.status[indices], frame.observed[indices] = saved.lifecycle, True
        _fill_candidate_lifecycle(frame, ids, release, terminal_times, terminal_codes, final_status)
        previous = frame
        yield frame
    if previous is None:
        raise ValueError("candidate requires saved full-population trajectory frames")
    if previous.time_s != span["end_s"]:
        raise ValueError("candidate frames must cover its complete executed time window")
    if not np.array_equal(previous.status, final_status):
        raise ValueError("candidate final frame/terminal ledger disagrees with final lifecycle")


def _fill_candidate_lifecycle(
    frame: _Frame,
    ids: np.ndarray,
    release: np.ndarray,
    terminal_times: np.ndarray,
    terminal_codes: np.ndarray,
    final_status: np.ndarray,
) -> None:
    pending = (frame.time_s < release) | (~np.isfinite(release) & (final_status == 0))
    ended = frame.time_s >= terminal_times
    unexplained = ~frame.observed & ~(pending | ended)
    if unexplained.any():
        index = int(np.flatnonzero(unexplained)[0])
        raise ValueError(
            f"missing active/pre-terminal candidate state for ID {ids[index]} at {frame.time_s} s"
        )
    frame.status[~frame.observed & pending] = 0
    known_terminal = ~frame.observed & ended
    frame.status[known_terminal] = terminal_codes[known_terminal]


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


def _mask(population: _Population, frame: _Frame, name: str) -> np.ndarray:
    codes = [code for code, label in population.names.items() if label == name]
    return np.isin(frame.status, codes)


def _mean_active_lifetime(population: _Population, first: _Frame, last: _Frame) -> float | None:
    terminal = [
        code
        for code, name in population.names.items()
        if name not in {"pending", "active", "failed"}
    ]
    supported = [code for code, name in population.names.items() if name != "failed"]
    ends = np.where(np.isin(last.status, terminal), population.terminal_s, last.time_s)
    if not bool(np.isin(last.status, supported).all()) or not np.isfinite(ends).all():
        return None
    if not np.isfinite(population.release_s).all():
        return None
    durations = np.maximum(
        0.0, np.minimum(ends, last.time_s) - np.maximum(population.release_s, first.time_s)
    )
    return float(np.mean(durations))


@dataclass
class _Description:
    population: _Population
    series: list[dict[str, Any]] = field(default_factory=list)
    coverage: Counter[str] = field(default_factory=Counter)
    first: _Frame | None = None
    last: _Frame | None = None

    def add(self, frame: _Frame) -> None:
        population = self.population
        active, finite = _mask(population, frame, "active"), np.isfinite(frame.states).all(axis=1)
        counts = {
            "expected_rows": frame.status.size,
            "observed_rows": frame.observed.sum(),
            "missing_rows": (~frame.observed).sum(),
            "lifecycle_known_rows": (frame.status >= 0).sum(),
            "lifecycle_unknown_rows": (frame.status < 0).sum(),
            "nonfinite_state_rows": (~finite).sum(),
            "nonfinite_observed_state_rows": (frame.observed & ~finite).sum(),
            "nonfinite_active_state_rows": (active & ~finite).sum(),
        }
        self.coverage.update({name: int(value) for name, value in counts.items()})
        self.series.append(
            {
                "time_s": frame.time_s,
                "status_counts": {
                    name: int(np.sum(frame.status == code))
                    for code, name in population.names.items()
                },
                "unobserved_state_row_count": int((~frame.observed).sum()),
                "unknown_lifecycle_count": int((frame.status < 0).sum()),
                "active_speed_m_s": _quantiles(np.linalg.norm(frame.states[active, 2:4], axis=1)),
                "active_charge_number": _quantiles(frame.states[active, 4]),
            }
        )
        if self.first is None:
            self.first = frame
        self.last = frame

    def document(self) -> dict[str, Any]:
        assert self.first is not None and self.last is not None
        population, first, last = self.population, self.first, self.last
        capture = _mask(population, last, "stuck")
        fractions = [row["status_counts"]["active"] / population.ids.size for row in self.series]
        times = [row["time_s"] for row in self.series]
        return {
            "registered_particle_count": int(population.ids.size),
            "observation_count": len(self.series),
            "window_s": [first.time_s, last.time_s],
            "coverage": dict(self.coverage),
            "final_counts": self.series[-1]["status_counts"],
            "final_fractions": {
                name: count / population.ids.size
                for name, count in self.series[-1]["status_counts"].items()
            },
            "saved_active_fraction_integral_s": float(np.trapezoid(fractions, times)),
            "restricted_mean_active_lifetime_s": _mean_active_lifetime(population, first, last),
            "terminal_stop_time_s_by_outcome": {
                name: _quantiles(population.terminal_s[_mask(population, last, name)])
                for name in population.names.values()
                if name not in {"pending", "active", "failed"}
            },
            "capture_terminal_time_s": _quantiles(population.terminal_s[capture]),
            "capture_terminal_observation_brackets_s": {
                "lower": _quantiles(population.terminal_interval_s[capture, 0]),
                "upper": _quantiles(population.terminal_interval_s[capture, 1]),
            },
            "series": self.series,
            "evidence": population.evidence,
        }


@dataclass
class _Comparison:
    times: list[float] = field(default_factory=list)
    curves: dict[str, list[int]] = field(default_factory=lambda: {"active": [], "stuck": []})
    squared: np.ndarray = field(default_factory=lambda: np.zeros(3))
    maximum: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rows: int = 0

    def add(self, left: _Population, right: _Population, a: _Frame, b: _Frame) -> None:
        mask = _mask(left, a, "active") & _mask(right, b, "active")
        mask &= np.isfinite(a.states).all(axis=1) & np.isfinite(b.states).all(axis=1)
        difference = a.states[mask] - b.states[mask]
        values = np.column_stack(
            (
                np.linalg.norm(difference[:, :2], axis=1),
                np.linalg.norm(difference[:, 2:4], axis=1),
                np.abs(difference[:, 4]),
            )
        )
        self.rows += values.shape[0]
        self.squared += np.sum(values**2, axis=0)
        if values.size:
            self.maximum = np.maximum(self.maximum, values.max(axis=0))
        self.times.append(a.time_s)
        for name in self.curves:
            self.curves[name].append(int(_mask(left, a, name).sum() - _mask(right, b, name).sum()))

    def document(self, left: _Description, right: _Description) -> dict[str, Any]:
        assert left.first is not None and right.first is not None
        assert left.last is not None and right.last is not None
        if (left.first.time_s, left.last.time_s) != (right.first.time_s, right.last.time_s):
            raise ValueError("candidate/reference must cover the same complete observation window")
        raw = {
            name: {
                "rows": self.rows,
                "rms": float(np.sqrt(self.squared[i] / self.rows)) if self.rows else None,
                "maximum": float(self.maximum[i]) if self.rows else None,
            }
            for i, name in enumerate(("position_m", "velocity_m_s", "charge_number"))
        }
        return {
            "common_exact_observation_times_s": self.times,
            "candidate_unmatched_clock_count": len(left.series) - len(self.times),
            "reference_unmatched_clock_count": len(right.series) - len(self.times),
            "population_curves": {
                name: {
                    "candidate_minus_reference_counts": values,
                    "maximum_abs_fraction_difference": max(map(abs, values))
                    / left.population.ids.size
                    if values
                    else None,
                }
                for name, values in self.curves.items()
            },
            "initial_state": _initial_state_difference(left, right),
            "same_ID_difference_weighting": "equal finite-active ID/time rows; not time-weighted on the nonuniform clock",
            "same_ID_finite_active_differences_descriptive_only": raw,
        }


def _initial_state_difference(left: _Description, right: _Description) -> dict[str, Any]:
    assert left.first is not None and right.first is not None
    difference = left.first.states - right.first.states
    valid = np.isfinite(left.first.states).all(axis=1) & np.isfinite(right.first.states).all(axis=1)
    a, b = left.population.release_s, right.population.release_s
    release_valid = np.isfinite(a) & np.isfinite(b)
    return {
        "meaning": "states at common window start; release times are compared separately",
        "paired_finite_rows": int(valid.sum()),
        "maximum_abs_component_difference": np.max(np.abs(difference[valid]), axis=0).tolist()
        if valid.any()
        else None,
        "components": ["r_m", "z_m", "vr_m_s", "vz_m_s", "charge_number"],
        "paired_finite_release_times": int(release_valid.sum()),
        "maximum_abs_release_time_difference_s": float(
            np.max(np.abs(a[release_valid] - b[release_valid]))
        )
        if release_valid.any()
        else None,
    }


def _merge_frames(
    left: Iterator[_Frame], right: Iterator[_Frame]
) -> Iterator[tuple[_Frame | None, _Frame | None]]:
    a, b = next(left, None), next(right, None)
    while a is not None or b is not None:
        ta, tb = a.time_s if a is not None else np.inf, b.time_s if b is not None else np.inf
        if ta < tb:
            yield a, None
            a = next(left, None)
        elif tb < ta:
            yield None, b
            b = next(right, None)
        else:
            yield a, b
            a, b = next(left, None), next(right, None)


def _collect(native: _Population, actual: _Population | None) -> dict[str, Any]:
    reference = _Description(native)
    if actual is None:
        for frame in native.frames:
            reference.add(frame)
        return {"reference": reference.document()}
    if not np.array_equal(actual.ids, native.ids):
        raise ValueError("candidate/reference stable ID populations differ")
    candidate, comparison = _Description(actual), _Comparison()
    for a, b in _merge_frames(actual.frames, native.frames):
        if a is not None and b is not None:
            comparison.add(actual, native, a, b)
        if a is not None:
            candidate.add(a)
        if b is not None:
            reference.add(b)
    return {
        "reference": reference.document(),
        "candidate": candidate.document(),
        "comparison": comparison.document(candidate, reference),
    }


def evaluate(
    reference: Path, *, reference_terminal_times: bool = False, candidate: Path | None = None
) -> dict[str, Any]:
    with reference.open("rb") as stream:
        native = _native(stream, reference, reference_terminal_times)
        actual = _candidate(candidate) if candidate is not None else None
        descriptions = _collect(native, actual)
    return {
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
            "native_order": "ascending time groups; IDs within each group may be unordered; two passes of one opened input",
            "omitted_candidate_rows": "only lifecycle is recovered after a recorded terminal/failure event, or before release; kinematics remain missing",
            "random_paths": "same IDs/seeds do not establish coupled Brownian paths",
            "allowance": "none; raw descriptive differences, no fitted threshold or agreement verdict",
            "survivors": "speed/charge quantiles condition on each participant's own active cohort",
            "lifetime": "release to known terminal stop, censored at window edges; unknown stops/failures leave the mean untested",
        },
        **descriptions,
    }


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
