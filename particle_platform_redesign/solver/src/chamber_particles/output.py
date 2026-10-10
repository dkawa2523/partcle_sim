"""Atomic result writing and lazy completed-result access."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import h5py
import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]
type Int64Array = NDArray[np.int64]
type Int32Array = NDArray[np.int32]
type UInt64Array = NDArray[np.uint64]
type UInt32Array = NDArray[np.uint32]
type UInt16Array = NDArray[np.uint16]
type UInt8Array = NDArray[np.uint8]
type StringArray = NDArray[np.str_]
type _PathInput = str | PathLike[str]

RESULT_SCHEMA_VERSION: Final = 3
RESULT_ALGORITHM_REVISION: Final = "durable_segmented_result_v6"
CHECKPOINT_SCHEMA_VERSION: Final = 2
RESULT_WRITER_RESERVE_BYTES: Final = 3 * 1024 * 1024

_RAW_CHUNK_CACHE_BYTES: Final = 64 * 1024
_DEFAULT_BOUNDARY_BATCH_ROWS: Final = 4096
_FINAL_VALIDATION_BLOCK_ROWS: Final = 4096

_LATEST_FORMAT_VERSION: Final = 1
_SEGMENT_DIRECTORY: Final = Path("segments")
_CHECKPOINT_DIRECTORY: Final = Path("checkpoints")


class ResultWriteError(RuntimeError):
    """A result directory cannot be created or finalized safely."""


class ResultOpenError(RuntimeError):
    """A completed result has an unsupported or inconsistent representation."""


class IncompleteResult(RuntimeError):
    """A result lacks its successful completion marker."""


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Small completion record returned instead of in-memory trajectories."""

    output_path: Path
    particle_count: int
    release_event_count: int
    boundary_event_count: int
    failure_event_count: int
    series_count: int
    frame_count: int
    frame_row_count: int
    probe_count: int
    probe_row_count: int
    macro_step_count: int


@dataclass(frozen=True, slots=True)
class CheckpointState:
    """Engine-owned mutable state captured only at an accepted macro barrier."""

    macro_time_s: float
    macro_step_count: int
    release_cursor: int
    frame_cursor: int
    probe_cursor: int
    position_m: FloatArray
    velocity_m_s: FloatArray
    charge_number: FloatArray
    lifecycle: UInt8Array
    failure_reason_code: UInt16Array
    terminal_time_s: FloatArray
    event_ordinal: UInt32Array
    physical_boundary_event_ordinal: UInt32Array
    exact_origin_time_s: FloatArray
    exact_origin_position_m: FloatArray
    exact_origin_velocity_m_s: FloatArray
    start_contact_state: UInt8Array
    active_particle_index: Int64Array
    last_field_cell: Int64Array | None
    accepted_particle_pieces: int
    candidate_queries: int
    refinements: int
    maximum_refinement_depth: int
    wall_interactions: int
    residual_splits: int
    axis_crossings: int


@dataclass(frozen=True, slots=True)
class _ResultCounts:
    release_events: int = 0
    boundary_events: int = 0
    failure_events: int = 0
    series: int = 0
    frames: int = 0
    frame_rows: int = 0
    probes: int = 0
    probe_rows: int = 0

    def add(self, other: _ResultCounts) -> _ResultCounts:
        return _ResultCounts(
            release_events=self.release_events + other.release_events,
            boundary_events=self.boundary_events + other.boundary_events,
            failure_events=self.failure_events + other.failure_events,
            series=self.series + other.series,
            frames=self.frames + other.frames,
            frame_rows=self.frame_rows + other.frame_rows,
            probes=self.probes + other.probes,
            probe_rows=self.probe_rows + other.probe_rows,
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "release_events": self.release_events,
            "boundary_events": self.boundary_events,
            "failure_events": self.failure_events,
            "series": self.series,
            "frames": self.frames,
            "frame_rows": self.frame_rows,
            "probes": self.probes,
            "probe_rows": self.probe_rows,
        }


@dataclass(frozen=True, slots=True)
class _LatestCommit:
    commit_id: int
    segment_index: int
    checkpoint_name: str
    checkpoint_sha256: str
    counts: _ResultCounts


@dataclass(frozen=True, slots=True)
class FinalParticles:
    """Particle rows stored at the requested run end time."""

    particle_id: Int64Array
    source_id: Int32Array
    time_s: FloatArray
    position_m: FloatArray
    velocity_m_s: FloatArray
    charge_number: FloatArray
    lifecycle: UInt8Array
    kinematics_valid: UInt8Array
    failure_reason_code: UInt16Array
    mass_kg: FloatArray
    drag_diameter_m: FloatArray
    contact_radius_m: FloatArray
    electrostatic_radius_m: FloatArray
    displaced_volume_m3: FloatArray
    model_weight: FloatArray
    material_id: Int32Array


@dataclass(frozen=True, slots=True)
class ReleaseEvents:
    """Canonical release-event rows in physical event order."""

    time_s: FloatArray
    particle_id: Int64Array
    event_ordinal: UInt32Array
    source_id: Int32Array


@dataclass(frozen=True, slots=True)
class BoundaryEvents:
    """Localized boundary-interaction rows and simultaneous source facets."""

    time_s: FloatArray
    particle_id: Int64Array
    event_ordinal: UInt32Array
    interaction_kind: StringArray
    primary_facet_id: Int64Array
    destination_facet_id: Int64Array
    boundary_id: Int32Array
    material_id: Int32Array
    contact_radius_m: FloatArray
    position_m: FloatArray
    position_post_m: FloatArray
    normal: FloatArray
    velocity_pre_m_s: FloatArray
    velocity_post_m_s: FloatArray
    charge_number_pre: FloatArray
    charge_number_post: FloatArray
    model_weight: FloatArray
    law_id: StringArray
    outcome: StringArray
    localization_residual_m: FloatArray
    position_budget_m: FloatArray
    time_budget_s: FloatArray
    candidate_offset: Int64Array
    candidate_facet_id: Int64Array


@dataclass(frozen=True, slots=True)
class FailureEvents:
    """Particle-local failures in logical event order."""

    time_s: FloatArray
    particle_id: Int64Array
    event_ordinal: UInt32Array
    reason_code: UInt16Array


@dataclass(frozen=True, slots=True)
class LifecycleSeries:
    """Small online population counts at selected physical times."""

    time_s: FloatArray
    pending: UInt64Array
    active: UInt64Array
    stuck: UInt64Array
    held: UInt64Array
    escaped: UInt64Array
    failed: UInt64Array


@dataclass(frozen=True, slots=True)
class TrajectoryFrame:
    """Released particles at one explicitly requested output time."""

    time_s: float
    particle_id: Int64Array
    position_m: FloatArray
    velocity_m_s: FloatArray
    charge_number: FloatArray
    lifecycle: UInt8Array


@dataclass(frozen=True, slots=True)
class ProbeFrame:
    """State rows for explicitly selected particle IDs at one requested time."""

    time_s: float
    particle_id: Int64Array
    position_m: FloatArray
    velocity_m_s: FloatArray
    charge_number: FloatArray
    lifecycle: UInt8Array


class _SegmentSink:
    """One uncommitted epoch segment, owned only by the synchronous writer."""

    def __init__(self, partial_path: Path, particle_capacity: int, segment_index: int) -> None:
        self._segment_index = segment_index
        name = f"epoch-{segment_index:06d}"
        self._temp_path = partial_path / _SEGMENT_DIRECTORY / f"{name}.tmp.h5"
        self._final_path = partial_path / _SEGMENT_DIRECTORY / f"{name}.h5"
        chunk_rows = max(1, min(particle_capacity, 4096))
        self._segment = h5py.File(
            self._temp_path,
            "w",
            rdcc_nbytes=_RAW_CHUNK_CACHE_BYTES,
            rdcc_nslots=521,
            rdcc_w0=1.0,
        )
        self._segment.attrs["result_schema_version"] = RESULT_SCHEMA_VERSION
        self._segment.attrs["segment_index"] = segment_index
        self._segment.create_group("events/release")
        self._event_time = self._resizable("events/release/time_s", "<f8", chunk_rows)
        self._event_particle = self._resizable("events/release/particle_id", "<i8", chunk_rows)
        self._event_ordinal = self._resizable("events/release/event_ordinal", "<u4", chunk_rows)
        self._event_source = self._resizable("events/release/source_id", "<i4", chunk_rows)
        self._segment.create_group("events/boundary")
        self._boundary_time = self._resizable("events/boundary/time_s", "<f8", chunk_rows)
        self._boundary_particle = self._resizable("events/boundary/particle_id", "<i8", chunk_rows)
        self._boundary_ordinal = self._resizable("events/boundary/event_ordinal", "<u4", chunk_rows)
        self._boundary_interaction_kind = self._resizable_string(
            "events/boundary/interaction_kind", chunk_rows
        )
        self._boundary_primary_facet = self._resizable(
            "events/boundary/primary_facet_id", "<i8", chunk_rows
        )
        self._boundary_destination_facet = self._resizable(
            "events/boundary/destination_facet_id", "<i8", chunk_rows
        )
        self._boundary_id = self._resizable("events/boundary/boundary_id", "<i4", chunk_rows)
        self._boundary_material = self._resizable("events/boundary/material_id", "<i4", chunk_rows)
        self._boundary_contact_radius = self._resizable(
            "events/boundary/contact_radius_m", "<f8", chunk_rows
        )
        self._boundary_position = self._resizable_2d(
            "events/boundary/position_m", "<f8", chunk_rows, width=2
        )
        self._boundary_position_post = self._resizable_2d(
            "events/boundary/position_post_m", "<f8", chunk_rows, width=2
        )
        self._boundary_normal = self._resizable_2d(
            "events/boundary/normal", "<f8", chunk_rows, width=2
        )
        self._boundary_velocity_pre = self._resizable_2d(
            "events/boundary/velocity_pre_m_s", "<f8", chunk_rows, width=2
        )
        self._boundary_velocity_post = self._resizable_2d(
            "events/boundary/velocity_post_m_s", "<f8", chunk_rows, width=2
        )
        self._boundary_charge_pre = self._resizable(
            "events/boundary/charge_number_pre", "<f8", chunk_rows
        )
        self._boundary_charge_post = self._resizable(
            "events/boundary/charge_number_post", "<f8", chunk_rows
        )
        self._boundary_weight = self._resizable("events/boundary/model_weight", "<f8", chunk_rows)
        self._boundary_law = self._resizable_string("events/boundary/law_id", chunk_rows)
        self._boundary_outcome = self._resizable_string("events/boundary/outcome", chunk_rows)
        self._boundary_residual = self._resizable(
            "events/boundary/localization_residual_m", "<f8", chunk_rows
        )
        self._boundary_position_budget = self._resizable(
            "events/boundary/position_budget_m", "<f8", chunk_rows
        )
        self._boundary_time_budget = self._resizable(
            "events/boundary/time_budget_s", "<f8", chunk_rows
        )
        self._boundary_candidate_offset = self._segment.create_dataset(
            "events/boundary/candidate_offset",
            shape=(1,),
            maxshape=(None,),
            chunks=(chunk_rows,),
            dtype="<i8",
        )
        self._boundary_candidate_offset[0] = 0
        self._boundary_candidate_facet = self._resizable(
            "events/boundary/candidate_facet_id", "<i8", chunk_rows
        )
        self._segment.create_group("events/failure")
        self._failure_time = self._resizable("events/failure/time_s", "<f8", chunk_rows)
        self._failure_particle = self._resizable("events/failure/particle_id", "<i8", chunk_rows)
        self._failure_ordinal = self._resizable("events/failure/event_ordinal", "<u4", chunk_rows)
        self._failure_reason = self._resizable("events/failure/reason_code", "<u2", chunk_rows)
        self._segment.create_group("series")
        self._series_time = self._resizable("series/time_s", "<f8", 64)
        self._series_pending = self._resizable("series/pending", "<u8", 64)
        self._series_active = self._resizable("series/active", "<u8", 64)
        self._series_stuck = self._resizable("series/stuck", "<u8", 64)
        self._series_held = self._resizable("series/held", "<u8", 64)
        self._series_escaped = self._resizable("series/escaped", "<u8", 64)
        self._series_failed = self._resizable("series/failed", "<u8", 64)
        self._segment.create_group("frames")
        self._frame_time = self._resizable("frames/time_s", "<f8", 64)
        self._frame_offset = self._segment.create_dataset(
            "frames/offset", shape=(1,), maxshape=(None,), chunks=(64,), dtype="<i8"
        )
        self._frame_offset[0] = 0
        self._frame_particle = self._resizable("frames/particle_id", "<i8", chunk_rows)
        self._frame_position = self._resizable_2d("frames/position_m", "<f8", chunk_rows, width=2)
        self._frame_velocity = self._resizable_2d("frames/velocity_m_s", "<f8", chunk_rows, width=2)
        self._frame_charge = self._resizable("frames/charge_number", "<f8", chunk_rows)
        self._frame_lifecycle = self._resizable("frames/lifecycle", "<u1", chunk_rows)
        self._segment.create_group("probes")
        self._probe_time = self._resizable("probes/time_s", "<f8", 64)
        self._probe_offset = self._segment.create_dataset(
            "probes/offset", shape=(1,), maxshape=(None,), chunks=(64,), dtype="<i8"
        )
        self._probe_offset[0] = 0
        self._probe_particle = self._resizable("probes/particle_id", "<i8", chunk_rows)
        self._probe_position = self._resizable_2d("probes/position_m", "<f8", chunk_rows, width=2)
        self._probe_velocity = self._resizable_2d("probes/velocity_m_s", "<f8", chunk_rows, width=2)
        self._probe_charge = self._resizable("probes/charge_number", "<f8", chunk_rows)
        self._probe_lifecycle = self._resizable("probes/lifecycle", "<u1", chunk_rows)
        self._closed = False

    @property
    def release_event_count(self) -> int:
        return int(self._event_time.shape[0])

    @property
    def boundary_event_count(self) -> int:
        return int(self._boundary_time.shape[0])

    @property
    def failure_event_count(self) -> int:
        return int(self._failure_time.shape[0])

    @property
    def series_count(self) -> int:
        return int(self._series_time.shape[0])

    @property
    def frame_count(self) -> int:
        return int(self._frame_time.shape[0])

    @property
    def frame_row_count(self) -> int:
        return int(self._frame_particle.shape[0])

    @property
    def probe_count(self) -> int:
        return int(self._probe_time.shape[0])

    @property
    def probe_row_count(self) -> int:
        return int(self._probe_particle.shape[0])

    def __enter__(self) -> _SegmentSink:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def write_release_events(self, events: ReleaseEvents) -> None:
        count = int(events.particle_id.size)
        if not (
            events.time_s.shape == events.event_ordinal.shape == events.source_id.shape == (count,)
        ):
            raise ResultWriteError("release-event columns have inconsistent lengths")
        self._append(self._event_time, events.time_s)
        self._append(self._event_particle, events.particle_id)
        self._append(self._event_ordinal, events.event_ordinal)
        self._append(self._event_source, events.source_id)

    def write_boundary_events(self, events: BoundaryEvents) -> None:
        """Append one batch, translating its local ragged offsets to segment offsets."""

        count = int(events.particle_id.size)
        _validate_boundary_event_shapes(events, count)
        candidate_start = int(self._boundary_candidate_facet.shape[0])
        scalar_columns = (
            (self._boundary_time, events.time_s),
            (self._boundary_particle, events.particle_id),
            (self._boundary_ordinal, events.event_ordinal),
            (self._boundary_primary_facet, events.primary_facet_id),
            (self._boundary_destination_facet, events.destination_facet_id),
            (self._boundary_id, events.boundary_id),
            (self._boundary_material, events.material_id),
            (self._boundary_contact_radius, events.contact_radius_m),
            (self._boundary_charge_pre, events.charge_number_pre),
            (self._boundary_charge_post, events.charge_number_post),
            (self._boundary_weight, events.model_weight),
            (self._boundary_residual, events.localization_residual_m),
            (self._boundary_position_budget, events.position_budget_m),
            (self._boundary_time_budget, events.time_budget_s),
        )
        vector_columns = (
            (self._boundary_position, events.position_m),
            (self._boundary_position_post, events.position_post_m),
            (self._boundary_normal, events.normal),
            (self._boundary_velocity_pre, events.velocity_pre_m_s),
            (self._boundary_velocity_post, events.velocity_post_m_s),
        )
        for dataset, values in scalar_columns + vector_columns:
            self._append(dataset, values)
        self._append_string(self._boundary_interaction_kind, events.interaction_kind)
        self._append_string(self._boundary_law, events.law_id)
        self._append_string(self._boundary_outcome, events.outcome)
        self._append(self._boundary_candidate_facet, events.candidate_facet_id)
        global_offsets = np.asarray(events.candidate_offset[1:] + candidate_start, dtype="<i8")
        self._append(self._boundary_candidate_offset, global_offsets)

    def write_failure_events(self, events: FailureEvents) -> None:
        count = int(events.particle_id.size)
        if not (
            events.time_s.shape
            == events.event_ordinal.shape
            == events.reason_code.shape
            == (count,)
        ):
            raise ResultWriteError("failure-event columns have inconsistent lengths")
        if not bool(np.isfinite(events.time_s).all()):
            raise ResultWriteError("failure-event times must be finite")
        if events.reason_code.dtype != np.dtype("<u2") or bool((events.reason_code == 0).any()):
            raise ResultWriteError("failure-event reason codes must be nonzero uint16 values")
        self._append(self._failure_time, events.time_s)
        self._append(self._failure_particle, events.particle_id)
        self._append(self._failure_ordinal, events.event_ordinal)
        self._append(self._failure_reason, events.reason_code)

    def write_lifecycle_series(self, series: LifecycleSeries) -> None:
        count = int(series.time_s.size)
        columns = (
            series.pending,
            series.active,
            series.stuck,
            series.held,
            series.escaped,
            series.failed,
        )
        if any(column.shape != (count,) for column in columns):
            raise ResultWriteError("lifecycle-series columns have inconsistent lengths")
        if not bool(np.isfinite(series.time_s).all()):
            raise ResultWriteError("lifecycle-series times must be finite")
        if count and self.series_count and float(series.time_s[0]) <= float(self._series_time[-1]):
            raise ResultWriteError("lifecycle-series times must be globally strictly increasing")
        if bool((np.diff(series.time_s) <= 0.0).any()):
            raise ResultWriteError("lifecycle-series times must be strictly increasing")
        self._append(self._series_time, series.time_s)
        for dataset, values in zip(
            (
                self._series_pending,
                self._series_active,
                self._series_stuck,
                self._series_held,
                self._series_escaped,
                self._series_failed,
            ),
            columns,
            strict=True,
        ):
            self._append(dataset, values)

    def write_frame(self, frame: TrajectoryFrame) -> None:
        count = int(frame.particle_id.size)
        _validate_state_frame(frame, count, "trajectory frame")
        self._append(self._frame_time, np.asarray([frame.time_s], dtype="<f8"))
        self._append(self._frame_particle, frame.particle_id)
        self._append(self._frame_position, frame.position_m)
        self._append(self._frame_velocity, frame.velocity_m_s)
        self._append(self._frame_charge, frame.charge_number)
        self._append(self._frame_lifecycle, frame.lifecycle)
        offset = np.asarray([self.frame_row_count], dtype="<i8")
        self._append(self._frame_offset, offset)

    def write_probe(self, frame: ProbeFrame) -> None:
        count = int(frame.particle_id.size)
        _validate_state_frame(frame, count, "probe")
        if self.probe_count and frame.time_s <= float(self._probe_time[-1]):
            raise ResultWriteError("probe times must be strictly increasing")
        self._append(self._probe_time, np.asarray([frame.time_s], dtype="<f8"))
        self._append(self._probe_particle, frame.particle_id)
        self._append(self._probe_position, frame.position_m)
        self._append(self._probe_velocity, frame.velocity_m_s)
        self._append(self._probe_charge, frame.charge_number)
        self._append(self._probe_lifecycle, frame.lifecycle)
        offset = np.asarray([self.probe_row_count], dtype="<i8")
        self._append(self._probe_offset, offset)

    def commit(self) -> _ResultCounts:
        """Close and atomically publish this segment, returning its local counts."""

        counts = _ResultCounts(
            release_events=self.release_event_count,
            boundary_events=self.boundary_event_count,
            failure_events=self.failure_event_count,
            series=self.series_count,
            frames=self.frame_count,
            frame_rows=self.frame_row_count,
            probes=self.probe_count,
            probe_rows=self.probe_row_count,
        )
        self._segment.attrs["commit_id"] = self._segment_index
        self._segment.attrs["closed"] = np.uint8(1)
        self.close()
        _sync_file(self._temp_path)
        os.replace(self._temp_path, self._final_path)
        return counts

    def close(self) -> None:
        if not self._closed:
            self._segment.close()
            self._closed = True

    def _resizable(self, path: str, dtype: str, chunk_rows: int) -> h5py.Dataset:
        return self._segment.create_dataset(
            path, shape=(0,), maxshape=(None,), chunks=(chunk_rows,), dtype=dtype
        )

    def _resizable_2d(self, path: str, dtype: str, chunk_rows: int, *, width: int) -> h5py.Dataset:
        return self._segment.create_dataset(
            path,
            shape=(0, width),
            maxshape=(None, width),
            chunks=(chunk_rows, width),
            dtype=dtype,
        )

    def _resizable_string(self, path: str, chunk_rows: int) -> h5py.Dataset:
        return self._segment.create_dataset(
            path,
            shape=(0,),
            maxshape=(None,),
            chunks=(chunk_rows,),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )

    @staticmethod
    def _append(dataset: h5py.Dataset, values: NDArray[Any]) -> None:
        start = int(dataset.shape[0])
        stop = start + int(values.shape[0])
        dataset.resize(stop, axis=0)
        dataset[start:stop] = values

    @staticmethod
    def _append_string(dataset: h5py.Dataset, values: StringArray) -> None:
        encoded = np.asarray([str(value) for value in values.tolist()], dtype=object)
        _SegmentSink._append(dataset, encoded)


class ResultWriter:
    """Durable synchronous epoch writer with one HDF5 owner."""

    def __init__(
        self,
        output: _PathInput,
        particle_capacity: int,
        resume_identity: Mapping[str, object],
    ) -> None:
        if particle_capacity < 0:
            raise ResultWriteError("particle capacity must be nonnegative")
        self.output_path = Path(output).expanduser().resolve()
        self.partial_path = self.output_path.parent / f"{self.output_path.name}.partial"
        self._particle_capacity = particle_capacity
        self._identity_json, self._identity_value = _canonical_identity(resume_identity)
        self._identity_hash = _sha256_bytes(self._identity_json.encode("utf-8"))
        self._counts = _ResultCounts()
        self._next_segment_index = 0
        self._resume_state: CheckpointState | None = None
        self._completed_summary: RunSummary | None = None
        self._sink: _SegmentSink | None = None
        self._closed = False

        if self.output_path.exists():
            raise ResultWriteError(f"result already exists: {self.output_path}")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.partial_path.exists():
            try:
                self._resume_existing_partial()
            except ResultOpenError as error:
                raise ResultWriteError("partial result is not safely resumable") from error
        else:
            self._create_partial()

    @property
    def resume_state(self) -> CheckpointState | None:
        return self._resume_state

    @property
    def completed_summary(self) -> RunSummary | None:
        return self._completed_summary

    def __enter__(self) -> ResultWriter:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def begin_epoch(self) -> None:
        """Open the next uncommitted segment."""

        self._require_running()
        if self._sink is not None:
            raise ResultWriteError("cannot begin an epoch while another segment is open")
        self._sink = _SegmentSink(
            self.partial_path,
            self._particle_capacity,
            self._next_segment_index,
        )

    def write_release_events(self, events: ReleaseEvents) -> None:
        self._write_segment("release", events)

    def write_boundary_events(self, events: BoundaryEvents) -> None:
        self._write_segment("boundary", events)

    def write_failure_events(self, events: FailureEvents) -> None:
        self._write_segment("failure", events)

    def write_lifecycle_series(self, series: LifecycleSeries) -> None:
        self._write_segment("series", series)

    def write_frame(self, frame: TrajectoryFrame) -> None:
        self._write_segment("frame", frame)

    def write_probe(self, frame: ProbeFrame) -> None:
        self._write_segment("probe", frame)

    def commit_epoch(self, checkpoint: CheckpointState) -> None:
        """Commit one segment/checkpoint pair and advance LATEST last."""

        self._require_running()
        _validate_checkpoint_state(checkpoint, self._particle_capacity)
        sink = self._require_open_segment()
        try:
            local_counts = sink.commit()
        except BaseException:
            sink.close()
            self._sink = None
            raise
        self._sink = None
        committed_counts = self._counts.add(local_counts)
        commit_id = self._next_segment_index
        checkpoint_name = "A.h5" if commit_id % 2 == 0 else "B.h5"
        checkpoint_path = self.partial_path / _CHECKPOINT_DIRECTORY / checkpoint_name
        checkpoint_temp = checkpoint_path.with_name(checkpoint_path.stem + ".tmp.h5")
        segment_path = _segment_path(self.partial_path, commit_id)
        segment_sha256 = _sha256_file(segment_path)
        _write_checkpoint(
            checkpoint_temp,
            checkpoint,
            commit_id=commit_id,
            segment_index=commit_id,
            segment_sha256=segment_sha256,
            counts=committed_counts,
            resume_identity_hash=self._identity_hash,
        )
        _sync_file(checkpoint_temp)
        os.replace(checkpoint_temp, checkpoint_path)
        checkpoint_sha256 = _sha256_file(checkpoint_path)
        latest = _LatestCommit(
            commit_id,
            commit_id,
            checkpoint_name,
            checkpoint_sha256,
            committed_counts,
        )
        _write_latest(self.partial_path, latest)
        self._counts = committed_counts
        self._resume_state = checkpoint
        self._next_segment_index += 1

    def finalize(
        self,
        final: FinalParticles,
        manifest: Mapping[str, object],
        *,
        macro_step_count: int,
    ) -> RunSummary:
        """Publish a complete result after the final epoch was committed."""

        self._require_running()
        if macro_step_count < 0:
            raise ResultWriteError("macro-step count must be nonnegative")
        particle_count = self._particle_capacity
        if final.particle_id.size != particle_count:
            raise ResultWriteError("final particle count does not match the prepared capacity")
        _validate_final_shapes(final, particle_count)
        try:
            latest = _read_latest(self.partial_path)
        except ResultOpenError as error:
            raise ResultWriteError("final committed epoch cannot be verified") from error
        if latest is None or latest.segment_index + 1 != self._next_segment_index:
            raise ResultWriteError("finalization requires a committed final epoch")
        if self._resume_state is None or self._resume_state.macro_step_count != macro_step_count:
            raise ResultWriteError("final checkpoint does not match the completed macro-step count")
        final_temp = self.partial_path / "final.tmp.h5"
        _write_final(final_temp, final)
        try:
            with h5py.File(final_temp, "r", rdcc_nbytes=_RAW_CHUNK_CACHE_BYTES) as handle:
                _validate_final_file(handle, particle_count)
        except (ResultOpenError, OSError, KeyError) as error:
            raise ResultWriteError(f"final result artifact is inconsistent: {error}") from error
        _sync_file(final_temp)
        os.replace(final_temp, self.partial_path / "final.h5")
        complete_manifest = dict(manifest)
        complete_manifest.update(
            {
                "status": "complete",
                "result_schema_version": RESULT_SCHEMA_VERSION,
                "result_algorithm_revision": RESULT_ALGORITHM_REVISION,
                "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
                "segment_count": self._next_segment_index,
                "latest_commit_id": latest.commit_id,
                "resume_identity": self._identity_value,
                "resume_identity_hash": self._identity_hash,
                "counts": {
                    "particles": particle_count,
                    **self._counts.as_dict(),
                    "macro_steps": macro_step_count,
                },
            }
        )
        _atomic_json(self.partial_path / "run.json", complete_manifest)
        _atomic_bytes(self.partial_path / "_SUCCESS", b"")
        if self.output_path.exists():
            raise ResultWriteError(f"result appeared during publication: {self.output_path}")
        os.rename(self.partial_path, self.output_path)
        self._closed = True
        return _summary_from_counts(
            self.output_path,
            particle_count,
            self._counts,
            macro_step_count,
        )

    def close(self) -> None:
        if self._closed:
            return
        if self._sink is not None:
            self._sink.close()
            self._sink = None
        self._closed = True

    def _create_partial(self) -> None:
        self.partial_path.mkdir(parents=False, exist_ok=False)
        (self.partial_path / _SEGMENT_DIRECTORY).mkdir()
        (self.partial_path / _CHECKPOINT_DIRECTORY).mkdir()
        _atomic_json(
            self.partial_path / "run.json",
            {
                "status": "running",
                "result_schema_version": RESULT_SCHEMA_VERSION,
                "result_algorithm_revision": RESULT_ALGORITHM_REVISION,
                "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
                "resume_identity": self._identity_value,
                "resume_identity_hash": self._identity_hash,
            },
        )

    def _resume_existing_partial(self) -> None:
        if not self.partial_path.is_dir():
            raise ResultWriteError(f"partial result is not a directory: {self.partial_path}")
        manifest = _read_manifest(self.partial_path)
        _require_resume_identity(manifest, self._identity_json, self._identity_hash)
        if (self.partial_path / "_SUCCESS").is_file():
            view = _open_completed_path(self.partial_path)
            summary = _summary_from_manifest(self.output_path, view.manifest)
            os.rename(self.partial_path, self.output_path)
            self._completed_summary = summary
            self._closed = True
            return
        latest = _read_latest(self.partial_path)
        if latest is None:
            # Nothing is committed yet.  Reusing epoch zero is an exact restart:
            # its temp/final segment and checkpoint names are atomically replaced.
            return
        segment_paths = _committed_segment_paths(self.partial_path, latest)
        actual_counts = _validate_segments(segment_paths)
        if actual_counts != latest.counts:
            raise ResultWriteError("committed segment counts do not match LATEST")
        checkpoint_path = self.partial_path / _CHECKPOINT_DIRECTORY / latest.checkpoint_name
        if _sha256_file(checkpoint_path) != latest.checkpoint_sha256:
            raise ResultWriteError("checkpoint content hash does not match LATEST")
        state, metadata_counts = _read_checkpoint(
            checkpoint_path,
            expected_identity_hash=self._identity_hash,
            expected_commit_id=latest.commit_id,
            expected_segment_index=latest.segment_index,
        )
        if metadata_counts != latest.counts:
            raise ResultWriteError("checkpoint counts do not match LATEST")
        if _sha256_file(segment_paths[-1]) != _checkpoint_segment_hash(checkpoint_path):
            raise ResultWriteError("latest segment content hash does not match checkpoint")
        self._counts = latest.counts
        self._next_segment_index = latest.segment_index + 1
        self._resume_state = state

    def _require_running(self) -> None:
        if self._completed_summary is not None:
            raise ResultWriteError("result was already completed before publication recovery")
        if self._closed:
            raise ResultWriteError("result writer is closed")

    def _require_open_segment(self) -> _SegmentSink:
        self._require_running()
        if self._sink is None:
            raise ResultWriteError("no epoch segment is open")
        return self._sink

    def _write_segment(self, operation: str, payload: object) -> None:
        sink = self._require_open_segment()
        try:
            _dispatch_segment_write(sink, operation, payload)
        except BaseException:
            sink.close()
            self._sink = None
            raise


@dataclass(frozen=True, slots=True)
class ResultView:
    """Lazy read-only view over committed segments of one logical result."""

    path: Path
    manifest: Mapping[str, object]
    segment_paths: tuple[Path, ...]
    complete: bool

    def read_final(self) -> FinalParticles:
        if not self.complete:
            raise ResultOpenError("an incomplete recovery result has no authoritative final state")
        particle_count = _summary_from_manifest(self.path, self.manifest).particle_count
        with h5py.File(self.path / "final.h5", "r") as handle:
            _validate_final_file(handle, particle_count)
            group = handle["particles"]
            return FinalParticles(
                particle_id=_read_only(group["particle_id"][...]),
                source_id=_read_only(group["source_id"][...]),
                time_s=_read_only(group["time_s"][...]),
                position_m=_read_only(group["position_m"][...]),
                velocity_m_s=_read_only(group["velocity_m_s"][...]),
                charge_number=_read_only(group["charge_number"][...]),
                lifecycle=_read_only(group["lifecycle"][...]),
                kinematics_valid=_read_only(group["kinematics_valid"][...]),
                failure_reason_code=_read_only(group["failure_reason_code"][...]),
                mass_kg=_read_only(group["mass_kg"][...]),
                drag_diameter_m=_read_only(group["drag_diameter_m"][...]),
                contact_radius_m=_read_only(group["contact_radius_m"][...]),
                electrostatic_radius_m=_read_only(group["electrostatic_radius_m"][...]),
                displaced_volume_m3=_read_only(group["displaced_volume_m3"][...]),
                model_weight=_read_only(group["model_weight"][...]),
                material_id=_read_only(group["material_id"][...]),
            )

    def read_release_events(self) -> ReleaseEvents:
        return _read_release_segments(self.segment_paths)

    def read_boundary_events(self) -> BoundaryEvents:
        return _read_boundary_batches(self.iter_boundary_event_batches())

    def iter_boundary_event_batches(
        self,
        *,
        batch_rows: int = _DEFAULT_BOUNDARY_BATCH_ROWS,
    ) -> Iterator[BoundaryEvents]:
        """Yield storage-order boundary-event batches with bounded row counts.

        Batches preserve each event and its simultaneous-facet candidate range,
        but are not globally sorted across writer waves.  Use
        :meth:`read_boundary_events` when canonical physical-event order is
        required and the complete table fits in memory.
        """

        if isinstance(batch_rows, bool) or not isinstance(batch_rows, int) or batch_rows <= 0:
            raise ValueError("boundary-event batch_rows must be a positive integer")
        for path in self.segment_paths:
            yield from _iter_boundary_event_batches(path, batch_rows)

    def read_failure_events(self) -> FailureEvents:
        return _read_failure_segments(self.segment_paths)

    def read_lifecycle_series(self) -> LifecycleSeries:
        return _read_series_segments(self.segment_paths)

    def iter_frames(self) -> Iterator[TrajectoryFrame]:
        for path in self.segment_paths:
            yield from _iter_trajectory_frames(path)

    def iter_probes(self) -> Iterator[ProbeFrame]:
        for path in self.segment_paths:
            yield from _iter_probe_frames(path)


def open_result_store(path: _PathInput, *, recovery: bool = False) -> ResultView:
    """Open a complete result, or explicitly the committed prefix of a partial run."""

    result_path = Path(path).expanduser().resolve()
    if result_path.is_dir() and (result_path / "_SUCCESS").is_file():
        return _open_completed_path(result_path)
    if not recovery:
        raise IncompleteResult(f"result is not complete: {result_path}")
    partial_path = result_path.parent / f"{result_path.name}.partial"
    if not partial_path.is_dir():
        raise IncompleteResult(f"result has no recoverable partial directory: {result_path}")
    manifest = _read_manifest(partial_path)
    latest = _read_latest(partial_path)
    if latest is None:
        raise IncompleteResult(f"result has no committed checkpoint: {result_path}")
    checkpoint_path = partial_path / _CHECKPOINT_DIRECTORY / latest.checkpoint_name
    if _sha256_file(checkpoint_path) != latest.checkpoint_sha256:
        raise ResultOpenError("checkpoint content hash does not match LATEST")
    identity_hash = manifest.get("resume_identity_hash")
    if not isinstance(identity_hash, str) or not _is_sha256(identity_hash):
        raise ResultOpenError("partial result resume identity hash is invalid")
    checkpoint_counts = _validate_checkpoint_reference(
        checkpoint_path,
        identity_hash,
        latest.commit_id,
        latest.segment_index,
    )
    if checkpoint_counts != latest.counts:
        raise ResultOpenError("checkpoint counts do not match LATEST")
    paths = _committed_segment_paths(partial_path, latest)
    actual_counts = _validate_segments(paths)
    if actual_counts != latest.counts:
        raise ResultOpenError("committed segment counts do not match LATEST")
    if _sha256_file(paths[-1]) != _checkpoint_segment_hash(checkpoint_path):
        raise ResultOpenError("latest segment content hash does not match checkpoint")
    recovery_manifest = dict(manifest)
    recovery_manifest.update(
        {
            "status": "recovery",
            "segment_count": len(paths),
            "latest_commit_id": latest.commit_id,
            "counts": latest.counts.as_dict(),
        }
    )
    return ResultView(
        partial_path,
        MappingProxyType(recovery_manifest),
        paths,
        False,
    )


def _dispatch_segment_write(sink: _SegmentSink, operation: str, payload: object) -> None:
    if operation == "release" and isinstance(payload, ReleaseEvents):
        sink.write_release_events(payload)
    elif operation == "boundary" and isinstance(payload, BoundaryEvents):
        sink.write_boundary_events(payload)
    elif operation == "failure" and isinstance(payload, FailureEvents):
        sink.write_failure_events(payload)
    elif operation == "series" and isinstance(payload, LifecycleSeries):
        sink.write_lifecycle_series(payload)
    elif operation == "frame" and isinstance(payload, TrajectoryFrame):
        sink.write_frame(payload)
    elif operation == "probe" and isinstance(payload, ProbeFrame):
        sink.write_probe(payload)
    else:
        raise ResultWriteError(f"invalid writer operation: {operation}")


def _canonical_identity(identity: Mapping[str, object]) -> tuple[str, dict[str, object]]:
    try:
        encoded = json.dumps(
            dict(identity),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        decoded = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ResultWriteError("resume identity must be finite JSON data") from error
    if not isinstance(decoded, dict) or not decoded:
        raise ResultWriteError("resume identity must be one non-empty object")
    return encoded, decoded


def _require_resume_identity(
    manifest: Mapping[str, object],
    identity_json: str,
    identity_hash: str,
) -> None:
    stored = manifest.get("resume_identity")
    stored_hash = manifest.get("resume_identity_hash")
    try:
        stored_json = json.dumps(
            stored,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ResultWriteError("partial result has an invalid resume identity") from error
    if stored_json != identity_json or stored_hash != identity_hash:
        raise ResultWriteError("partial result resume identity does not match this run")
    if manifest.get("result_schema_version") != RESULT_SCHEMA_VERSION:
        raise ResultWriteError("partial result schema version is incompatible")
    if manifest.get("result_algorithm_revision") != RESULT_ALGORITHM_REVISION:
        raise ResultWriteError("partial result algorithm revision is incompatible")
    if manifest.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ResultWriteError("partial checkpoint schema version is incompatible")


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    try:
        payload = json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    except (TypeError, ValueError) as error:
        raise ResultWriteError(f"cannot encode JSON artifact: {path.name}") from error
    _atomic_bytes(path, payload.encode("utf-8"))


def _atomic_bytes(path: Path, payload: bytes) -> None:
    temporary = _temporary_path(path)
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as error:
        raise ResultWriteError(f"cannot atomically write {path.name}") from error


def _temporary_path(path: Path) -> Path:
    if path.suffix:
        return path.with_name(f"{path.stem}.tmp{path.suffix}")
    return path.with_name(path.name + ".tmp")


def _sync_file(path: Path) -> None:
    try:
        with path.open("r+b") as handle:
            os.fsync(handle.fileno())
    except OSError as error:
        raise ResultWriteError(f"cannot synchronize {path.name}") from error


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
    except OSError as error:
        raise OSError(f"cannot hash committed artifact: {path}") from error
    return "sha256:" + digest.hexdigest()


def _read_manifest(path: Path) -> dict[str, object]:
    try:
        value = json.loads((path / "run.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ResultOpenError("run.json cannot be read") from error
    if not isinstance(value, dict):
        raise ResultOpenError("run.json must contain one object")
    return value


def _write_latest(path: Path, latest: _LatestCommit) -> None:
    _atomic_json(
        path / "LATEST",
        {
            "format_version": _LATEST_FORMAT_VERSION,
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "commit_id": latest.commit_id,
            "segment_index": latest.segment_index,
            "checkpoint": latest.checkpoint_name,
            "checkpoint_sha256": latest.checkpoint_sha256,
            "counts": latest.counts.as_dict(),
        },
    )


def _read_latest(path: Path) -> _LatestCommit | None:
    latest_path = path / "LATEST"
    if not latest_path.is_file():
        return None
    try:
        value = json.loads(latest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ResultOpenError("LATEST cannot be read") from error
    if not isinstance(value, dict):
        raise ResultOpenError("LATEST must contain one object")
    if (
        value.get("format_version") != _LATEST_FORMAT_VERSION
        or value.get("result_schema_version") != RESULT_SCHEMA_VERSION
        or value.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION
    ):
        raise ResultOpenError("LATEST version information is incompatible")
    commit_id = _json_nonnegative_integer(value.get("commit_id"), "LATEST.commit_id")
    segment_index = _json_nonnegative_integer(value.get("segment_index"), "LATEST.segment_index")
    if commit_id != segment_index:
        raise ResultOpenError("LATEST commit and segment indices must match")
    checkpoint_name = value.get("checkpoint")
    expected_checkpoint = "A.h5" if commit_id % 2 == 0 else "B.h5"
    if checkpoint_name != expected_checkpoint:
        raise ResultOpenError("LATEST checkpoint generation is invalid")
    checkpoint_sha256 = value.get("checkpoint_sha256")
    if not isinstance(checkpoint_sha256, str) or not _is_sha256(checkpoint_sha256):
        raise ResultOpenError("LATEST checkpoint hash is invalid")
    counts = _counts_from_mapping(value.get("counts"), "LATEST.counts")
    return _LatestCommit(
        commit_id,
        segment_index,
        expected_checkpoint,
        checkpoint_sha256,
        counts,
    )


def _json_nonnegative_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ResultOpenError(f"{label} must be a nonnegative integer")
    return int(value)


def _counts_from_mapping(value: object, label: str) -> _ResultCounts:
    if not isinstance(value, dict):
        raise ResultOpenError(f"{label} must be one object")
    expected = {
        "release_events",
        "boundary_events",
        "failure_events",
        "series",
        "frames",
        "frame_rows",
        "probes",
        "probe_rows",
    }
    if set(value) != expected:
        raise ResultOpenError(f"{label} has invalid keys")
    parsed = {key: _json_nonnegative_integer(value[key], f"{label}.{key}") for key in expected}
    return _ResultCounts(**parsed)


def _is_sha256(value: str) -> bool:
    prefix = "sha256:"
    digest = value[len(prefix) :] if value.startswith(prefix) else ""
    return len(digest) == 64 and all(character in "0123456789abcdef" for character in digest)


def _segment_path(path: Path, segment_index: int) -> Path:
    return path / _SEGMENT_DIRECTORY / f"epoch-{segment_index:06d}.h5"


def _committed_segment_paths(path: Path, latest: _LatestCommit) -> tuple[Path, ...]:
    paths = tuple(_segment_path(path, index) for index in range(latest.segment_index + 1))
    if any(not segment.is_file() for segment in paths):
        raise ResultOpenError("a segment referenced by LATEST is missing")
    return paths


def _write_checkpoint(
    path: Path,
    state: CheckpointState,
    *,
    commit_id: int,
    segment_index: int,
    segment_sha256: str,
    counts: _ResultCounts,
    resume_identity_hash: str,
) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["checkpoint_schema_version"] = CHECKPOINT_SCHEMA_VERSION
        handle.attrs["result_schema_version"] = RESULT_SCHEMA_VERSION
        handle.attrs["result_algorithm_revision"] = RESULT_ALGORITHM_REVISION
        handle.attrs["commit_id"] = commit_id
        handle.attrs["segment_index"] = segment_index
        handle.attrs["segment_sha256"] = segment_sha256
        handle.attrs["resume_identity_hash"] = resume_identity_hash
        handle.attrs["macro_time_s"] = state.macro_time_s
        handle.attrs["macro_step_count"] = state.macro_step_count
        handle.attrs["release_cursor"] = state.release_cursor
        handle.attrs["frame_cursor"] = state.frame_cursor
        handle.attrs["probe_cursor"] = state.probe_cursor
        handle.attrs["accepted_particle_pieces"] = state.accepted_particle_pieces
        handle.attrs["candidate_queries"] = state.candidate_queries
        handle.attrs["refinements"] = state.refinements
        handle.attrs["maximum_refinement_depth"] = state.maximum_refinement_depth
        handle.attrs["wall_interactions"] = state.wall_interactions
        handle.attrs["residual_splits"] = state.residual_splits
        handle.attrs["axis_crossings"] = state.axis_crossings
        count_group = handle.create_group("counts")
        for name, count in counts.as_dict().items():
            count_group.attrs[name] = count
        state_group = handle.create_group("state")
        arrays = (
            ("position_m", state.position_m, "<f8"),
            ("velocity_m_s", state.velocity_m_s, "<f8"),
            ("charge_number", state.charge_number, "<f8"),
            ("lifecycle", state.lifecycle, "<u1"),
            ("failure_reason_code", state.failure_reason_code, "<u2"),
            ("terminal_time_s", state.terminal_time_s, "<f8"),
            ("event_ordinal", state.event_ordinal, "<u4"),
            (
                "physical_boundary_event_ordinal",
                state.physical_boundary_event_ordinal,
                "<u4",
            ),
            ("exact_origin_time_s", state.exact_origin_time_s, "<f8"),
            ("exact_origin_position_m", state.exact_origin_position_m, "<f8"),
            ("exact_origin_velocity_m_s", state.exact_origin_velocity_m_s, "<f8"),
            ("start_contact_state", state.start_contact_state, "<u1"),
            ("active_particle_index", state.active_particle_index, "<i8"),
        )
        for name, array, dtype in arrays:
            state_group.create_dataset(name, data=array, dtype=dtype)
        if state.last_field_cell is not None:
            state_group.create_dataset("last_field_cell", data=state.last_field_cell, dtype="<i8")
        handle.flush()


def _read_checkpoint(
    path: Path,
    *,
    expected_identity_hash: str,
    expected_commit_id: int,
    expected_segment_index: int,
) -> tuple[CheckpointState, _ResultCounts]:
    try:
        with h5py.File(path, "r") as handle:
            _validate_checkpoint_metadata(
                handle,
                expected_identity_hash,
                expected_commit_id,
                expected_segment_index,
            )
            counts = _checkpoint_counts(handle)
            group = handle["state"]
            if not isinstance(group, h5py.Group):
                raise ResultOpenError("checkpoint state group is missing")
            last_field = group.get("last_field_cell")
            state = CheckpointState(
                macro_time_s=float(handle.attrs["macro_time_s"]),
                macro_step_count=int(handle.attrs["macro_step_count"]),
                release_cursor=int(handle.attrs["release_cursor"]),
                frame_cursor=int(handle.attrs["frame_cursor"]),
                probe_cursor=int(handle.attrs["probe_cursor"]),
                position_m=np.asarray(group["position_m"][...], dtype="<f8"),
                velocity_m_s=np.asarray(group["velocity_m_s"][...], dtype="<f8"),
                charge_number=np.asarray(group["charge_number"][...], dtype="<f8"),
                lifecycle=np.asarray(group["lifecycle"][...], dtype="<u1"),
                failure_reason_code=np.asarray(group["failure_reason_code"][...], dtype="<u2"),
                terminal_time_s=np.asarray(group["terminal_time_s"][...], dtype="<f8"),
                event_ordinal=np.asarray(group["event_ordinal"][...], dtype="<u4"),
                physical_boundary_event_ordinal=np.asarray(
                    group["physical_boundary_event_ordinal"][...], dtype="<u4"
                ),
                exact_origin_time_s=np.asarray(group["exact_origin_time_s"][...], dtype="<f8"),
                exact_origin_position_m=np.asarray(
                    group["exact_origin_position_m"][...], dtype="<f8"
                ),
                exact_origin_velocity_m_s=np.asarray(
                    group["exact_origin_velocity_m_s"][...], dtype="<f8"
                ),
                start_contact_state=np.asarray(group["start_contact_state"][...], dtype="<u1"),
                active_particle_index=np.asarray(group["active_particle_index"][...], dtype="<i8"),
                last_field_cell=(
                    np.asarray(last_field[...], dtype="<i8")
                    if isinstance(last_field, h5py.Dataset)
                    else None
                ),
                accepted_particle_pieces=int(handle.attrs["accepted_particle_pieces"]),
                candidate_queries=int(handle.attrs["candidate_queries"]),
                refinements=int(handle.attrs["refinements"]),
                maximum_refinement_depth=int(handle.attrs["maximum_refinement_depth"]),
                wall_interactions=int(handle.attrs["wall_interactions"]),
                residual_splits=int(handle.attrs["residual_splits"]),
                axis_crossings=int(handle.attrs["axis_crossings"]),
            )
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ResultOpenError("checkpoint cannot be read") from error
    _validate_checkpoint_state(state, int(state.position_m.shape[0]))
    return state, counts


def _validate_checkpoint_metadata(
    handle: h5py.File,
    expected_identity_hash: str,
    expected_commit_id: int,
    expected_segment_index: int,
) -> None:
    expected = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "result_algorithm_revision": RESULT_ALGORITHM_REVISION,
        "commit_id": expected_commit_id,
        "segment_index": expected_segment_index,
        "resume_identity_hash": expected_identity_hash,
    }
    if any(handle.attrs.get(name) != value for name, value in expected.items()):
        raise ResultOpenError("checkpoint identity or revision is incompatible")
    hash_value = handle.attrs.get("segment_sha256")
    if not isinstance(hash_value, str) or not _is_sha256(hash_value):
        raise ResultOpenError("checkpoint segment hash is invalid")


def _checkpoint_counts(handle: h5py.File) -> _ResultCounts:
    group = handle.get("counts")
    if not isinstance(group, h5py.Group):
        raise ResultOpenError("checkpoint counts are missing")
    return _counts_from_mapping(dict(group.attrs.items()), "checkpoint.counts")


def _validate_checkpoint_reference(
    path: Path,
    identity_hash: str,
    commit_id: int,
    segment_index: int,
) -> _ResultCounts:
    try:
        with h5py.File(path, "r") as handle:
            _validate_checkpoint_metadata(handle, identity_hash, commit_id, segment_index)
            return _checkpoint_counts(handle)
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ResultOpenError("checkpoint reference cannot be read") from error


def _checkpoint_segment_hash(path: Path) -> str:
    try:
        with h5py.File(path, "r") as handle:
            value = handle.attrs["segment_sha256"]
    except (OSError, KeyError) as error:
        raise ResultOpenError("checkpoint segment hash cannot be read") from error
    if not isinstance(value, str) or not _is_sha256(value):
        raise ResultOpenError("checkpoint segment hash is invalid")
    return value


def _validate_checkpoint_state(state: CheckpointState, particle_count: int) -> None:
    if not math.isfinite(state.macro_time_s):
        raise ResultWriteError("checkpoint macro time must be finite")
    _validate_checkpoint_counters(state)
    _validate_checkpoint_arrays(state, particle_count)
    _validate_checkpoint_active_index(state.active_particle_index, particle_count)


def _validate_checkpoint_counters(state: CheckpointState) -> None:
    integers = (
        state.macro_step_count,
        state.release_cursor,
        state.frame_cursor,
        state.probe_cursor,
        state.accepted_particle_pieces,
        state.candidate_queries,
        state.refinements,
        state.maximum_refinement_depth,
        state.wall_interactions,
        state.residual_splits,
        state.axis_crossings,
    )
    if any(isinstance(value, bool) or value < 0 for value in integers):
        raise ResultWriteError("checkpoint counters must be nonnegative integers")


def _validate_checkpoint_arrays(state: CheckpointState, particle_count: int) -> None:
    vectors = (
        state.position_m,
        state.velocity_m_s,
        state.exact_origin_position_m,
        state.exact_origin_velocity_m_s,
    )
    if any(
        array.shape != (particle_count, 2) or array.dtype != np.dtype("<f8") for array in vectors
    ):
        raise ResultWriteError("checkpoint vector arrays have invalid layouts")
    scalars = (
        (state.charge_number, "<f8"),
        (state.lifecycle, "<u1"),
        (state.failure_reason_code, "<u2"),
        (state.terminal_time_s, "<f8"),
        (state.event_ordinal, "<u4"),
        (state.physical_boundary_event_ordinal, "<u4"),
        (state.exact_origin_time_s, "<f8"),
        (state.start_contact_state, "<u1"),
    )
    if any(
        array.shape != (particle_count,) or array.dtype != np.dtype(dtype)
        for array, dtype in scalars
    ):
        raise ResultWriteError("checkpoint scalar arrays have invalid layouts")
    if state.last_field_cell is not None and (
        state.last_field_cell.shape != (particle_count,)
        or state.last_field_cell.dtype != np.dtype("<i8")
    ):
        raise ResultWriteError("checkpoint field-cell hint has an invalid layout")
    finite = (
        state.position_m,
        state.velocity_m_s,
        state.charge_number,
        state.exact_origin_time_s,
        state.exact_origin_position_m,
        state.exact_origin_velocity_m_s,
    )
    if any(not bool(np.isfinite(array).all()) for array in finite):
        raise ResultWriteError("checkpoint physical state must be finite")


def _validate_checkpoint_active_index(active: Int64Array, particle_count: int) -> None:
    if active.ndim != 1 or active.dtype != np.dtype("<i8"):
        raise ResultWriteError("checkpoint active index has an invalid layout")
    if active.size and (
        int(active[0]) < 0
        or int(active[-1]) >= particle_count
        or bool((np.diff(active) <= 0).any())
    ):
        raise ResultWriteError("checkpoint active index must be sorted and unique")


def _open_completed_path(path: Path) -> ResultView:
    if not path.is_dir() or not (path / "_SUCCESS").is_file():
        raise IncompleteResult(f"result is not complete: {path}")
    manifest = _read_manifest(path)
    if manifest.get("status") != "complete":
        raise IncompleteResult(f"result manifest is not complete: {path}")
    if manifest.get("result_schema_version") != RESULT_SCHEMA_VERSION:
        raise ResultOpenError("unsupported result schema version")
    if manifest.get("result_algorithm_revision") != RESULT_ALGORITHM_REVISION:
        raise ResultOpenError("unsupported result algorithm revision")
    summary = _summary_from_manifest(path, manifest)
    latest = _read_latest(path)
    if latest is None:
        raise ResultOpenError("completed result has no LATEST commit")
    paths = _committed_segment_paths(path, latest)
    counts = _validate_segments(paths)
    if counts != latest.counts:
        raise ResultOpenError("committed segment counts do not match LATEST")
    segment_count = manifest.get("segment_count")
    if segment_count != len(paths) or manifest.get("latest_commit_id") != latest.commit_id:
        raise ResultOpenError("completed manifest does not match LATEST")
    manifest_counts = manifest.get("counts")
    if not isinstance(manifest_counts, dict):
        raise ResultOpenError("completed manifest counts are missing")
    for name, count in counts.as_dict().items():
        if manifest_counts.get(name) != count:
            raise ResultOpenError("completed manifest counts do not match segments")
    try:
        with h5py.File(path / "final.h5", "r") as final:
            _validate_final_file(final, summary.particle_count)
    except (OSError, KeyError) as error:
        raise ResultOpenError("final result artifact cannot be read") from error
    return ResultView(path, MappingProxyType(manifest), paths, True)


def _summary_from_counts(
    path: Path,
    particle_count: int,
    counts: _ResultCounts,
    macro_step_count: int,
) -> RunSummary:
    return RunSummary(
        output_path=path,
        particle_count=particle_count,
        release_event_count=counts.release_events,
        boundary_event_count=counts.boundary_events,
        failure_event_count=counts.failure_events,
        series_count=counts.series,
        frame_count=counts.frames,
        frame_row_count=counts.frame_rows,
        probe_count=counts.probes,
        probe_row_count=counts.probe_rows,
        macro_step_count=macro_step_count,
    )


def _summary_from_manifest(path: Path, manifest: Mapping[str, object]) -> RunSummary:
    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        raise ResultOpenError("completed manifest counts are missing")
    durable_counts = _counts_from_mapping(
        {name: counts.get(name) for name in _ResultCounts().as_dict()},
        "run.counts",
    )
    particles = _json_nonnegative_integer(counts.get("particles"), "run.counts.particles")
    macro_steps = _json_nonnegative_integer(counts.get("macro_steps"), "run.counts.macro_steps")
    return _summary_from_counts(path, particles, durable_counts, macro_steps)


def _read_release_segments(paths: tuple[Path, ...]) -> ReleaseEvents:
    times: list[np.ndarray] = []
    particles: list[np.ndarray] = []
    ordinals: list[np.ndarray] = []
    sources: list[np.ndarray] = []
    for path in paths:
        with h5py.File(path, "r") as handle:
            group = handle["events/release"]
            times.append(group["time_s"][...])
            particles.append(group["particle_id"][...])
            ordinals.append(group["event_ordinal"][...])
            sources.append(group["source_id"][...])
    time_values = _join_arrays(times, "<f8")
    particle_values = _join_arrays(particles, "<i8")
    ordinal_values = _join_arrays(ordinals, "<u4")
    source_values = _join_arrays(sources, "<i4")
    order = np.lexsort((particle_values, time_values))
    return ReleaseEvents(
        _read_only(time_values[order]),
        _read_only(particle_values[order]),
        _read_only(ordinal_values[order]),
        _read_only(source_values[order]),
    )


def _read_boundary_batches(batches: Iterator[BoundaryEvents]) -> BoundaryEvents:
    numeric: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "time_s",
            "particle_id",
            "event_ordinal",
            "primary_facet_id",
            "destination_facet_id",
            "boundary_id",
            "material_id",
            "contact_radius_m",
            "position_m",
            "position_post_m",
            "normal",
            "velocity_pre_m_s",
            "velocity_post_m_s",
            "charge_number_pre",
            "charge_number_post",
            "model_weight",
            "localization_residual_m",
            "position_budget_m",
            "time_budget_s",
        )
    }
    interaction_kinds: list[np.ndarray] = []
    laws: list[np.ndarray] = []
    outcomes: list[np.ndarray] = []
    candidate_lengths: list[np.ndarray] = []
    candidate_chunks: list[np.ndarray] = []
    for events in batches:
        for name in numeric:
            numeric[name].append(getattr(events, name))
        interaction_kinds.append(events.interaction_kind)
        laws.append(events.law_id)
        outcomes.append(events.outcome)
        candidate_lengths.append(np.diff(events.candidate_offset))
        candidate_chunks.append(events.candidate_facet_id)
    values = _join_boundary_numeric(numeric)
    interaction_kind_values = _join_strings(interaction_kinds)
    law_values = _join_strings(laws)
    outcome_values = _join_strings(outcomes)
    order = np.lexsort((values["event_ordinal"], values["particle_id"], values["time_s"]))
    lengths = _join_arrays(candidate_lengths, "<i8")
    candidates = _join_arrays(candidate_chunks, "<i8")
    offsets, candidate_values = _reorder_boundary_candidates(lengths, candidates, order)
    return BoundaryEvents(
        time_s=_read_only(values["time_s"][order]),
        particle_id=_read_only(values["particle_id"][order]),
        event_ordinal=_read_only(values["event_ordinal"][order]),
        interaction_kind=_read_only(interaction_kind_values[order]),
        primary_facet_id=_read_only(values["primary_facet_id"][order]),
        destination_facet_id=_read_only(values["destination_facet_id"][order]),
        boundary_id=_read_only(values["boundary_id"][order]),
        material_id=_read_only(values["material_id"][order]),
        contact_radius_m=_read_only(values["contact_radius_m"][order]),
        position_m=_read_only(values["position_m"][order]),
        position_post_m=_read_only(values["position_post_m"][order]),
        normal=_read_only(values["normal"][order]),
        velocity_pre_m_s=_read_only(values["velocity_pre_m_s"][order]),
        velocity_post_m_s=_read_only(values["velocity_post_m_s"][order]),
        charge_number_pre=_read_only(values["charge_number_pre"][order]),
        charge_number_post=_read_only(values["charge_number_post"][order]),
        model_weight=_read_only(values["model_weight"][order]),
        law_id=_read_only(law_values[order]),
        outcome=_read_only(outcome_values[order]),
        localization_residual_m=_read_only(values["localization_residual_m"][order]),
        position_budget_m=_read_only(values["position_budget_m"][order]),
        time_budget_s=_read_only(values["time_budget_s"][order]),
        candidate_offset=_read_only(offsets),
        candidate_facet_id=_read_only(candidate_values),
    )


def _iter_boundary_event_batches(path: Path, batch_rows: int) -> Iterator[BoundaryEvents]:
    with h5py.File(path, "r") as handle:
        group = handle["events/boundary"]
        row_count = int(group["time_s"].shape[0])
        for start in range(0, row_count, batch_rows):
            stop = min(start + batch_rows, row_count)
            offsets = np.asarray(group["candidate_offset"][start : stop + 1], dtype="<i8")
            candidate_start = int(offsets[0])
            candidate_stop = int(offsets[-1])
            yield BoundaryEvents(
                time_s=_read_only(group["time_s"][start:stop]),
                particle_id=_read_only(group["particle_id"][start:stop]),
                event_ordinal=_read_only(group["event_ordinal"][start:stop]),
                interaction_kind=_read_only(
                    np.asarray(group["interaction_kind"].asstr()[start:stop], dtype=np.str_)
                ),
                primary_facet_id=_read_only(group["primary_facet_id"][start:stop]),
                destination_facet_id=_read_only(group["destination_facet_id"][start:stop]),
                boundary_id=_read_only(group["boundary_id"][start:stop]),
                material_id=_read_only(group["material_id"][start:stop]),
                contact_radius_m=_read_only(group["contact_radius_m"][start:stop]),
                position_m=_read_only(group["position_m"][start:stop]),
                position_post_m=_read_only(group["position_post_m"][start:stop]),
                normal=_read_only(group["normal"][start:stop]),
                velocity_pre_m_s=_read_only(group["velocity_pre_m_s"][start:stop]),
                velocity_post_m_s=_read_only(group["velocity_post_m_s"][start:stop]),
                charge_number_pre=_read_only(group["charge_number_pre"][start:stop]),
                charge_number_post=_read_only(group["charge_number_post"][start:stop]),
                model_weight=_read_only(group["model_weight"][start:stop]),
                law_id=_read_only(np.asarray(group["law_id"].asstr()[start:stop], dtype=np.str_)),
                outcome=_read_only(np.asarray(group["outcome"].asstr()[start:stop], dtype=np.str_)),
                localization_residual_m=_read_only(group["localization_residual_m"][start:stop]),
                position_budget_m=_read_only(group["position_budget_m"][start:stop]),
                time_budget_s=_read_only(group["time_budget_s"][start:stop]),
                candidate_offset=_read_only(offsets - candidate_start),
                candidate_facet_id=_read_only(
                    group["candidate_facet_id"][candidate_start:candidate_stop]
                ),
            )


def _reorder_boundary_candidates(
    lengths: np.ndarray,
    candidates: np.ndarray,
    order: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros(order.size + 1, dtype="<i8")
    if not order.size:
        return offsets, np.empty(0, dtype="<i8")
    ordered_lengths = lengths[order]
    offsets[1:] = np.cumsum(ordered_lengths, dtype="<i8")
    source_rows = np.repeat(np.arange(order.size, dtype="<i8"), lengths)
    canonical_rank = np.empty(order.size, dtype="<i8")
    canonical_rank[order] = np.arange(order.size, dtype="<i8")
    candidate_order = np.argsort(canonical_rank[source_rows], kind="stable")
    return offsets, candidates[candidate_order].astype("<i8", copy=False)


def _join_boundary_numeric(chunks: Mapping[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    dtypes = {
        "time_s": "<f8",
        "particle_id": "<i8",
        "event_ordinal": "<u4",
        "primary_facet_id": "<i8",
        "destination_facet_id": "<i8",
        "boundary_id": "<i4",
        "material_id": "<i4",
        "contact_radius_m": "<f8",
        "position_m": "<f8",
        "position_post_m": "<f8",
        "normal": "<f8",
        "velocity_pre_m_s": "<f8",
        "velocity_post_m_s": "<f8",
        "charge_number_pre": "<f8",
        "charge_number_post": "<f8",
        "model_weight": "<f8",
        "localization_residual_m": "<f8",
        "position_budget_m": "<f8",
        "time_budget_s": "<f8",
    }
    vectors = {
        "position_m",
        "position_post_m",
        "normal",
        "velocity_pre_m_s",
        "velocity_post_m_s",
    }
    return {
        name: _join_arrays(values, dtypes[name], (2,) if name in vectors else ())
        for name, values in chunks.items()
    }


def _read_failure_segments(paths: tuple[Path, ...]) -> FailureEvents:
    times: list[np.ndarray] = []
    particles: list[np.ndarray] = []
    ordinals: list[np.ndarray] = []
    reasons: list[np.ndarray] = []
    for path in paths:
        with h5py.File(path, "r") as handle:
            group = handle["events/failure"]
            times.append(group["time_s"][...])
            particles.append(group["particle_id"][...])
            ordinals.append(group["event_ordinal"][...])
            reasons.append(group["reason_code"][...])
    time_values = _join_arrays(times, "<f8")
    particle_values = _join_arrays(particles, "<i8")
    ordinal_values = _join_arrays(ordinals, "<u4")
    reason_values = _join_arrays(reasons, "<u2")
    order = np.lexsort((ordinal_values, particle_values, time_values))
    return FailureEvents(
        _read_only(time_values[order]),
        _read_only(particle_values[order]),
        _read_only(ordinal_values[order]),
        _read_only(reason_values[order]),
    )


def _read_series_segments(paths: tuple[Path, ...]) -> LifecycleSeries:
    chunks: dict[str, list[np.ndarray]] = {
        name: [] for name in ("time_s", "pending", "active", "stuck", "held", "escaped", "failed")
    }
    for path in paths:
        with h5py.File(path, "r") as handle:
            group = handle["series"]
            for name in chunks:
                chunks[name].append(group[name][...])
    return LifecycleSeries(
        time_s=_read_only(_join_arrays(chunks["time_s"], "<f8")),
        pending=_read_only(_join_arrays(chunks["pending"], "<u8")),
        active=_read_only(_join_arrays(chunks["active"], "<u8")),
        stuck=_read_only(_join_arrays(chunks["stuck"], "<u8")),
        held=_read_only(_join_arrays(chunks["held"], "<u8")),
        escaped=_read_only(_join_arrays(chunks["escaped"], "<u8")),
        failed=_read_only(_join_arrays(chunks["failed"], "<u8")),
    )


def _iter_trajectory_frames(path: Path) -> Iterator[TrajectoryFrame]:
    with h5py.File(path, "r") as handle:
        group = handle["frames"]
        times = group["time_s"][...]
        offsets = group["offset"][...]
        for index, time_s in enumerate(times):
            start = int(offsets[index])
            stop = int(offsets[index + 1])
            yield TrajectoryFrame(
                float(time_s),
                _read_only(group["particle_id"][start:stop]),
                _read_only(group["position_m"][start:stop]),
                _read_only(group["velocity_m_s"][start:stop]),
                _read_only(group["charge_number"][start:stop]),
                _read_only(group["lifecycle"][start:stop]),
            )


def _iter_probe_frames(path: Path) -> Iterator[ProbeFrame]:
    with h5py.File(path, "r") as handle:
        group = handle["probes"]
        times = group["time_s"][...]
        offsets = group["offset"][...]
        for index, time_s in enumerate(times):
            start = int(offsets[index])
            stop = int(offsets[index + 1])
            yield ProbeFrame(
                float(time_s),
                _read_only(group["particle_id"][start:stop]),
                _read_only(group["position_m"][start:stop]),
                _read_only(group["velocity_m_s"][start:stop]),
                _read_only(group["charge_number"][start:stop]),
                _read_only(group["lifecycle"][start:stop]),
            )


def _join_arrays(
    chunks: list[np.ndarray],
    dtype: str,
    trailing_shape: tuple[int, ...] = (),
) -> np.ndarray:
    if not chunks:
        return np.empty((0, *trailing_shape), dtype=dtype)
    return np.concatenate(chunks, axis=0).astype(dtype, copy=False)


def _join_strings(chunks: list[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.empty(0, dtype=np.str_)
    return np.concatenate(chunks).astype(np.str_, copy=False)


def _write_final(path: Path, final: FinalParticles) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["result_schema_version"] = RESULT_SCHEMA_VERSION
        group = handle.create_group("particles")
        columns = (
            ("particle_id", "<i8"),
            ("source_id", "<i4"),
            ("time_s", "<f8"),
            ("position_m", "<f8"),
            ("velocity_m_s", "<f8"),
            ("charge_number", "<f8"),
            ("lifecycle", "<u1"),
            ("kinematics_valid", "<u1"),
            ("failure_reason_code", "<u2"),
            ("mass_kg", "<f8"),
            ("drag_diameter_m", "<f8"),
            ("contact_radius_m", "<f8"),
            ("electrostatic_radius_m", "<f8"),
            ("displaced_volume_m3", "<f8"),
            ("model_weight", "<f8"),
            ("material_id", "<i4"),
        )
        for name, dtype in columns:
            group.create_dataset(name, data=getattr(final, name), dtype=dtype)


def _validate_final_shapes(final: FinalParticles, count: int) -> None:
    vectors = (final.position_m, final.velocity_m_s)
    scalars = (
        final.particle_id,
        final.source_id,
        final.time_s,
        final.charge_number,
        final.lifecycle,
        final.kinematics_valid,
        final.failure_reason_code,
        final.mass_kg,
        final.drag_diameter_m,
        final.contact_radius_m,
        final.electrostatic_radius_m,
        final.displaced_volume_m3,
        final.model_weight,
        final.material_id,
    )
    if any(array.shape != (count, 2) for array in vectors):
        raise ResultWriteError("final vector columns must have shape [N, 2]")
    if any(array.shape != (count,) for array in scalars):
        raise ResultWriteError("final scalar columns have inconsistent lengths")
    if bool(((final.kinematics_valid != 0) & (final.kinematics_valid != 1)).any()):
        raise ResultWriteError("final kinematics_valid values must be 0 or 1")
    if final.failure_reason_code.dtype != np.dtype("<u2"):
        raise ResultWriteError("final failure reason codes must use uint16")
    failed = final.lifecycle == np.uint8(4)
    if bool((final.failure_reason_code[failed] == 0).any()) or bool(
        (final.failure_reason_code[~failed] != 0).any()
    ):
        raise ResultWriteError("final failure reason must be nonzero only for failed particles")
    finite_columns = (
        final.time_s,
        final.position_m,
        final.velocity_m_s,
        final.charge_number,
        final.mass_kg,
        final.drag_diameter_m,
        final.contact_radius_m,
        final.electrostatic_radius_m,
        final.displaced_volume_m3,
        final.model_weight,
    )
    if any(not bool(np.isfinite(array).all()) for array in finite_columns):
        raise ResultWriteError("final floating-point columns must be finite")
    if bool((final.contact_radius_m < 0.0).any()):
        raise ResultWriteError("final contact_radius_m must be nonnegative")


def _validate_state_frame(
    frame: TrajectoryFrame | ProbeFrame,
    count: int,
    label: str,
) -> None:
    if not math.isfinite(frame.time_s):
        raise ResultWriteError(f"{label} time must be finite")
    if frame.position_m.shape != (count, 2) or frame.velocity_m_s.shape != (count, 2):
        raise ResultWriteError(f"{label} position and velocity must have shape [N, 2]")
    if frame.charge_number.shape != (count,) or frame.lifecycle.shape != (count,):
        raise ResultWriteError(f"{label} scalar columns have inconsistent lengths")
    if not bool(
        np.isfinite(frame.position_m).all()
        and np.isfinite(frame.velocity_m_s).all()
        and np.isfinite(frame.charge_number).all()
    ):
        raise ResultWriteError(f"{label} floating-point columns must be finite")


def _validate_boundary_event_shapes(events: BoundaryEvents, count: int) -> None:
    vectors = (
        events.position_m,
        events.position_post_m,
        events.normal,
        events.velocity_pre_m_s,
        events.velocity_post_m_s,
    )
    scalars = (
        events.time_s,
        events.event_ordinal,
        events.interaction_kind,
        events.primary_facet_id,
        events.destination_facet_id,
        events.boundary_id,
        events.material_id,
        events.contact_radius_m,
        events.charge_number_pre,
        events.charge_number_post,
        events.model_weight,
        events.law_id,
        events.outcome,
        events.localization_residual_m,
        events.position_budget_m,
        events.time_budget_s,
    )
    if any(array.shape != (count, 2) for array in vectors):
        raise ResultWriteError("boundary-event vector columns must have shape [N, 2]")
    if any(array.shape != (count,) for array in scalars):
        raise ResultWriteError("boundary-event scalar columns have inconsistent lengths")
    _validate_boundary_candidate_shapes(events, count)
    _validate_boundary_numeric_values(events)
    string_columns = (events.interaction_kind, events.law_id, events.outcome)
    if any(array.dtype.kind != "U" for array in string_columns):
        raise ResultWriteError("boundary-event string arrays must use Unicode dtype")
    if not _boundary_interaction_rows_are_valid(
        events.interaction_kind,
        events.destination_facet_id,
        events.position_m,
        events.position_post_m,
        events.law_id,
        events.outcome,
    ):
        raise ResultWriteError("boundary-event interaction columns are inconsistent")


def _validate_boundary_candidate_shapes(events: BoundaryEvents, count: int) -> None:
    offsets = events.candidate_offset
    candidate_count = int(events.candidate_facet_id.size)
    if offsets.shape != (count + 1,) or offsets.dtype != np.dtype("<i8"):
        raise ResultWriteError("boundary-event candidate offsets have invalid layout")
    if (
        int(offsets[0]) != 0
        or bool((np.diff(offsets) <= 0).any())
        or int(offsets[-1]) != candidate_count
    ):
        raise ResultWriteError("each boundary event must have a non-empty candidate range")
    if events.candidate_facet_id.dtype != np.dtype("<i8") or not _candidate_ranges_are_valid(
        offsets,
        events.candidate_facet_id,
        events.primary_facet_id,
    ):
        raise ResultWriteError("boundary-event candidates must be sorted and contain primary facet")


def _validate_boundary_numeric_values(events: BoundaryEvents) -> None:
    finite_columns = (
        events.time_s,
        events.position_m,
        events.position_post_m,
        events.normal,
        events.velocity_pre_m_s,
        events.velocity_post_m_s,
        events.contact_radius_m,
        events.charge_number_pre,
        events.charge_number_post,
        events.model_weight,
        events.localization_residual_m,
        events.position_budget_m,
        events.time_budget_s,
    )
    if any(not bool(np.isfinite(array).all()) for array in finite_columns):
        raise ResultWriteError("boundary-event floating-point columns must be finite")
    if bool((events.contact_radius_m < 0.0).any()):
        raise ResultWriteError("boundary-event contact_radius_m must be nonnegative")
    if bool(
        (events.localization_residual_m < 0.0).any()
        or (events.position_budget_m <= 0.0).any()
        or (events.time_budget_s <= 0.0).any()
        or (events.localization_residual_m > events.position_budget_m).any()
    ):
        raise ResultWriteError("boundary-event residual and budgets are inconsistent")


def _validate_segments(paths: tuple[Path, ...]) -> _ResultCounts:
    counts = _ResultCounts()
    last_series_time = -math.inf
    last_frame_time = -math.inf
    last_probe_time = -math.inf
    try:
        for expected_index, path in enumerate(paths):
            with h5py.File(path, "r") as segment:
                _validate_segment_file(segment, expected_index)
                local = _segment_counts(segment)
                counts = counts.add(local)
                last_series_time = _validate_cross_segment_times(
                    segment["series/time_s"], last_series_time, "lifecycle-series"
                )
                last_frame_time = _validate_cross_segment_times(
                    segment["frames/time_s"], last_frame_time, "trajectory-frame"
                )
                last_probe_time = _validate_cross_segment_times(
                    segment["probes/time_s"], last_probe_time, "probe"
                )
    except (OSError, KeyError) as error:
        raise ResultOpenError("committed result segment cannot be read") from error
    return counts


def _validate_cross_segment_times(
    dataset: h5py.Dataset,
    previous: float,
    label: str,
) -> float:
    values = dataset[...]
    if not bool(np.isfinite(values).all()) or bool((np.diff(values) <= 0.0).any()):
        raise ResultOpenError(f"{label} times must be finite and strictly increasing")
    if values.size and float(values[0]) <= previous:
        raise ResultOpenError(f"{label} times must increase across segments")
    return float(values[-1]) if values.size else previous


def _segment_counts(segment: h5py.File) -> _ResultCounts:
    return _ResultCounts(
        release_events=int(segment["events/release/time_s"].shape[0]),
        boundary_events=int(segment["events/boundary/time_s"].shape[0]),
        failure_events=int(segment["events/failure/time_s"].shape[0]),
        series=int(segment["series/time_s"].shape[0]),
        frames=int(segment["frames/time_s"].shape[0]),
        frame_rows=int(segment["frames/particle_id"].shape[0]),
        probes=int(segment["probes/time_s"].shape[0]),
        probe_rows=int(segment["probes/particle_id"].shape[0]),
    )


def _validate_segment_file(segment: h5py.File, expected_index: int) -> None:
    if segment.attrs.get("result_schema_version") != RESULT_SCHEMA_VERSION:
        raise ResultOpenError("segment schema version is invalid")
    if (
        segment.attrs.get("segment_index") != expected_index
        or segment.attrs.get("commit_id") != expected_index
        or segment.attrs.get("closed") != 1
    ):
        raise ResultOpenError("result segment is not closed")
    release = segment["events/release"]
    boundary = segment["events/boundary"]
    failure = segment["events/failure"]
    series = segment["series"]
    frames = segment["frames"]
    probes = segment["probes"]
    if (
        not isinstance(release, h5py.Group)
        or not isinstance(boundary, h5py.Group)
        or not isinstance(failure, h5py.Group)
        or not isinstance(series, h5py.Group)
        or not isinstance(frames, h5py.Group)
        or not isinstance(probes, h5py.Group)
    ):
        raise ResultOpenError("result segment groups are invalid")
    _validate_release_group(release)
    _validate_boundary_group(boundary)
    _validate_failure_group(failure)
    _validate_series_group(series)
    _validate_frame_group(frames)
    _validate_probe_group(probes)


def _validate_release_group(release: h5py.Group) -> None:
    release_rows = _one_dimensional_rows(release, "time_s", "<f8")
    release_columns = {
        "particle_id": "<i8",
        "event_ordinal": "<u4",
        "source_id": "<i4",
    }
    if any(
        not _has_layout(release, name, (release_rows,), dtype)
        for name, dtype in release_columns.items()
    ):
        raise ResultOpenError("release-event columns have inconsistent shapes")


def _validate_boundary_group(boundary: h5py.Group) -> None:
    row_count = _one_dimensional_rows(boundary, "time_s", "<f8")
    expected_shapes = {
        "time_s": ((row_count,), "<f8"),
        "particle_id": ((row_count,), "<i8"),
        "event_ordinal": ((row_count,), "<u4"),
        "primary_facet_id": ((row_count,), "<i8"),
        "destination_facet_id": ((row_count,), "<i8"),
        "boundary_id": ((row_count,), "<i4"),
        "material_id": ((row_count,), "<i4"),
        "contact_radius_m": ((row_count,), "<f8"),
        "position_m": ((row_count, 2), "<f8"),
        "position_post_m": ((row_count, 2), "<f8"),
        "normal": ((row_count, 2), "<f8"),
        "velocity_pre_m_s": ((row_count, 2), "<f8"),
        "velocity_post_m_s": ((row_count, 2), "<f8"),
        "charge_number_pre": ((row_count,), "<f8"),
        "charge_number_post": ((row_count,), "<f8"),
        "model_weight": ((row_count,), "<f8"),
        "localization_residual_m": ((row_count,), "<f8"),
        "position_budget_m": ((row_count,), "<f8"),
        "time_budget_s": ((row_count,), "<f8"),
    }
    if any(
        not _has_layout(boundary, name, shape, dtype)
        for name, (shape, dtype) in expected_shapes.items()
    ):
        raise ResultOpenError("boundary-event columns have inconsistent shapes")
    contact_radius = boundary["contact_radius_m"]
    for start in range(0, row_count, _DEFAULT_BOUNDARY_BATCH_ROWS):
        values = contact_radius[start : start + _DEFAULT_BOUNDARY_BATCH_ROWS]
        if not bool(np.isfinite(values).all()) or bool((values < 0.0).any()):
            raise ResultOpenError("boundary-event contact_radius_m must be finite and nonnegative")
    string_columns = ("interaction_kind", "law_id", "outcome")
    if any(not _has_utf8_layout(boundary, name, row_count) for name in string_columns):
        raise ResultOpenError("boundary-event string columns have invalid layouts")
    _validate_boundary_interaction_group(boundary, row_count)
    offsets = boundary.get("candidate_offset")
    candidates = boundary.get("candidate_facet_id")
    if not isinstance(offsets, h5py.Dataset) or not isinstance(candidates, h5py.Dataset):
        raise ResultOpenError("boundary-event candidate indices are missing")
    if offsets.shape != (row_count + 1,) or offsets.dtype != np.dtype("<i8"):
        raise ResultOpenError("boundary-event candidate offsets have invalid layouts")
    if candidates.ndim != 1 or candidates.dtype != np.dtype("<i8"):
        raise ResultOpenError("boundary-event candidate facets have invalid layouts")
    _validate_boundary_candidates(boundary, row_count)


def _validate_boundary_candidates(boundary: h5py.Group, row_count: int) -> None:
    offsets = boundary["candidate_offset"]
    candidates = boundary["candidate_facet_id"]
    primary = boundary["primary_facet_id"]
    if int(offsets[0]) != 0 or int(offsets[-1]) != candidates.shape[0]:
        raise ResultOpenError("boundary-event candidate offsets are inconsistent")
    for start in range(0, row_count, _DEFAULT_BOUNDARY_BATCH_ROWS):
        stop = min(start + _DEFAULT_BOUNDARY_BATCH_ROWS, row_count)
        offset_values = np.asarray(offsets[start : stop + 1], dtype="<i8")
        if bool((np.diff(offset_values) <= 0).any()):
            raise ResultOpenError("boundary-event candidate offsets are inconsistent")
        candidate_start = int(offset_values[0])
        candidate_stop = int(offset_values[-1])
        if not _candidate_ranges_are_valid(
            offset_values - candidate_start,
            candidates[candidate_start:candidate_stop],
            primary[start:stop],
        ):
            raise ResultOpenError("boundary-event candidates are inconsistent")


def _validate_boundary_interaction_group(boundary: h5py.Group, row_count: int) -> None:
    for start in range(0, row_count, _DEFAULT_BOUNDARY_BATCH_ROWS):
        stop = min(start + _DEFAULT_BOUNDARY_BATCH_ROWS, row_count)
        if not _boundary_interaction_rows_are_valid(
            np.asarray(boundary["interaction_kind"].asstr()[start:stop], dtype=np.str_),
            boundary["destination_facet_id"][start:stop],
            boundary["position_m"][start:stop],
            boundary["position_post_m"][start:stop],
            np.asarray(boundary["law_id"].asstr()[start:stop], dtype=np.str_),
            np.asarray(boundary["outcome"].asstr()[start:stop], dtype=np.str_),
        ):
            raise ResultOpenError("boundary-event interaction columns are inconsistent")


def _boundary_interaction_rows_are_valid(
    interaction_kind: NDArray[Any],
    destination_facet_id: NDArray[Any],
    position_m: NDArray[Any],
    position_post_m: NDArray[Any],
    law_id: NDArray[Any],
    outcome: NDArray[Any],
) -> bool:
    if not bool(np.isfinite(position_m).all() and np.isfinite(position_post_m).all()):
        return False
    wall = interaction_kind == "wall"
    periodic = interaction_kind == "periodic_translation"
    if not bool((wall | periodic).all()):
        return False
    if bool((destination_facet_id[wall] != -1).any()):
        return False
    if bool((destination_facet_id[periodic] < 0).any()):
        return False
    if not bool(np.array_equal(position_m[wall], position_post_m[wall])):
        return False
    return bool(((law_id[periodic] == "") & (outcome[periodic] == "transferred")).all())


def _validate_failure_group(failure: h5py.Group) -> None:
    row_count = _one_dimensional_rows(failure, "time_s", "<f8")
    expected = {
        "particle_id": "<i8",
        "event_ordinal": "<u4",
        "reason_code": "<u2",
    }
    if any(not _has_layout(failure, name, (row_count,), dtype) for name, dtype in expected.items()):
        raise ResultOpenError("failure-event columns have inconsistent shapes")
    if not bool(np.isfinite(failure["time_s"][...]).all()):
        raise ResultOpenError("failure-event times must be finite")
    if row_count and bool((failure["reason_code"][...] == 0).any()):
        raise ResultOpenError("failure-event reason codes must be nonzero")


def _validate_series_group(series: h5py.Group) -> None:
    row_count = _one_dimensional_rows(series, "time_s", "<f8")
    names = ("pending", "active", "stuck", "held", "escaped", "failed")
    if any(not _has_layout(series, name, (row_count,), "<u8") for name in names):
        raise ResultOpenError("lifecycle-series columns have inconsistent shapes")
    times = series["time_s"][...]
    if not bool(np.isfinite(times).all()) or bool((np.diff(times) <= 0.0).any()):
        raise ResultOpenError("lifecycle-series times must be finite and strictly increasing")


def _validate_frame_group(frames: h5py.Group) -> None:
    _validate_ragged_state_group(frames, "trajectory frame")


def _validate_probe_group(probes: h5py.Group) -> None:
    _validate_ragged_state_group(probes, "probe")
    times = probes["time_s"][...]
    if not bool(np.isfinite(times).all()) or bool((np.diff(times) <= 0.0).any()):
        raise ResultOpenError("probe times must be finite and strictly increasing")


def _validate_ragged_state_group(group: h5py.Group, label: str) -> None:
    frame_count = _one_dimensional_rows(group, "time_s", "<f8")
    offsets = group["offset"]
    rows = group["particle_id"]
    if not isinstance(offsets, h5py.Dataset) or not isinstance(rows, h5py.Dataset):
        raise ResultOpenError(f"{label} indices are invalid")
    if offsets.dtype != np.dtype("<i8") or rows.dtype != np.dtype("<i8"):
        raise ResultOpenError(f"{label} indices have invalid dtypes")
    if offsets.shape != (frame_count + 1,):
        raise ResultOpenError(f"{label} offsets are inconsistent")
    offset_values = offsets[...]
    if (
        int(offset_values[0]) != 0
        or bool((np.diff(offset_values) < 0).any())
        or int(offset_values[-1]) != rows.shape[0]
    ):
        raise ResultOpenError(f"{label} offsets are inconsistent")
    frame_rows = int(rows.shape[0])
    expected_shapes = {
        "particle_id": ((frame_rows,), "<i8"),
        "position_m": ((frame_rows, 2), "<f8"),
        "velocity_m_s": ((frame_rows, 2), "<f8"),
        "charge_number": ((frame_rows,), "<f8"),
        "lifecycle": ((frame_rows,), "<u1"),
    }
    if any(
        not _has_layout(group, name, shape, dtype)
        for name, (shape, dtype) in expected_shapes.items()
    ):
        raise ResultOpenError(f"{label} columns have inconsistent shapes")


def _validate_final_file(final: h5py.File, expected_particle_count: int) -> None:
    if final.attrs.get("result_schema_version") != RESULT_SCHEMA_VERSION:
        raise ResultOpenError("final schema version is invalid")
    particles = final["particles"]
    if not isinstance(particles, h5py.Group):
        raise ResultOpenError("final particles group is invalid")
    particle_count = _one_dimensional_rows(particles, "particle_id", "<i8")
    if particle_count != expected_particle_count:
        raise ResultOpenError("final particle count does not match the completed manifest")
    expected_shapes = {
        "particle_id": ((particle_count,), "<i8"),
        "source_id": ((particle_count,), "<i4"),
        "time_s": ((particle_count,), "<f8"),
        "position_m": ((particle_count, 2), "<f8"),
        "velocity_m_s": ((particle_count, 2), "<f8"),
        "charge_number": ((particle_count,), "<f8"),
        "lifecycle": ((particle_count,), "<u1"),
        "kinematics_valid": ((particle_count,), "<u1"),
        "failure_reason_code": ((particle_count,), "<u2"),
        "mass_kg": ((particle_count,), "<f8"),
        "drag_diameter_m": ((particle_count,), "<f8"),
        "contact_radius_m": ((particle_count,), "<f8"),
        "electrostatic_radius_m": ((particle_count,), "<f8"),
        "displaced_volume_m3": ((particle_count,), "<f8"),
        "model_weight": ((particle_count,), "<f8"),
        "material_id": ((particle_count,), "<i4"),
    }
    if any(
        not _has_layout(particles, name, shape, dtype)
        for name, (shape, dtype) in expected_shapes.items()
    ):
        raise ResultOpenError("final particle columns have inconsistent shapes")
    _validate_final_particle_blocks(particles, particle_count)


def _validate_final_particle_blocks(particles: h5py.Group, particle_count: int) -> None:
    """Check final row values without allocating a particle-sized validation array."""

    previous_id = -1
    for begin in range(0, particle_count, _FINAL_VALIDATION_BLOCK_ROWS):
        end = min(begin + _FINAL_VALIDATION_BLOCK_ROWS, particle_count)
        particle_ids = particles["particle_id"][begin:end]
        if int(particle_ids[0]) <= previous_id or bool(
            (particle_ids[1:] <= particle_ids[:-1]).any()
        ):
            raise ResultOpenError("final particle_id must be nonnegative and strictly increasing")
        previous_id = int(particle_ids[-1])
        contact_radius = particles["contact_radius_m"][begin:end]
        if not bool(np.isfinite(contact_radius).all()) or bool((contact_radius < 0.0).any()):
            raise ResultOpenError("final contact_radius_m must be finite and nonnegative")
        validity = particles["kinematics_valid"][begin:end]
        if bool(((validity != 0) & (validity != 1)).any()):
            raise ResultOpenError("final kinematics_valid values must be 0 or 1")
        lifecycle = particles["lifecycle"][begin:end]
        reasons = particles["failure_reason_code"][begin:end]
        failed = lifecycle == np.uint8(4)
        if bool((reasons[failed] == 0).any()) or bool((reasons[~failed] != 0).any()):
            raise ResultOpenError("final failure reason does not match lifecycle")


def _one_dimensional_rows(group: h5py.Group, name: str, dtype: str) -> int:
    dataset = group[name]
    if (
        not isinstance(dataset, h5py.Dataset)
        or dataset.ndim != 1
        or dataset.dtype != np.dtype(dtype)
    ):
        raise ResultOpenError(f"result dataset {name!r} must be one-dimensional")
    return int(dataset.shape[0])


def _has_layout(group: h5py.Group, name: str, shape: tuple[int, ...], dtype: str) -> bool:
    value = group.get(name)
    return (
        isinstance(value, h5py.Dataset) and value.shape == shape and value.dtype == np.dtype(dtype)
    )


def _has_utf8_layout(group: h5py.Group, name: str, rows: int) -> bool:
    value = group.get(name)
    if not isinstance(value, h5py.Dataset) or value.shape != (rows,):
        return False
    string_info = h5py.check_string_dtype(value.dtype)
    return string_info is not None and string_info.encoding == "utf-8"


def _candidate_ranges_are_valid(
    offsets: NDArray[Any],
    candidates: NDArray[Any],
    primary: NDArray[Any],
) -> bool:
    row_count = int(primary.size)
    if offsets.shape != (row_count + 1,):
        return False
    lengths = np.diff(offsets)
    if bool((lengths <= 0).any()) or int(offsets[0]) != 0 or int(offsets[-1]) != candidates.size:
        return False
    if row_count == 0:
        return candidates.size == 0
    increasing = np.diff(candidates) > 0
    increasing[offsets[1:-1] - 1] = True
    if not bool(increasing.all()):
        return False
    matches_primary = candidates == np.repeat(primary, lengths)
    contains_primary = np.logical_or.reduceat(matches_primary, offsets[:-1])
    return bool(contains_primary.all())


def _read_only(array: NDArray[Any]) -> NDArray[Any]:
    result = np.asarray(array)
    result.flags.writeable = False
    return result
