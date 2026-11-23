"""Persistent process workers, shared arrays and bounded event handoff."""

from __future__ import annotations

import copyreg
import io
import mmap
import multiprocessing
import os
import pickle
import traceback
import weakref
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from dataclasses import fields as dataclass_fields
from multiprocessing.connection import Connection, wait
from multiprocessing.process import BaseProcess
from multiprocessing.shared_memory import SharedMemory
from types import MappingProxyType
from typing import cast

import numpy as np

from .cpu import (
    ActiveIndex,
    CpuMemoryPlan,
    ParticleState,
)
from .output import (
    BoundaryEvents,
    FailureEvents,
    ResultWriter,
)
from .transport import (
    EngineError,
    PreparedTransport,
    _advance_active_slabs,
    _advance_particle_slab,
    _allocate_replay_buffer,
    _EventSink,
    _EventStatistics,
    _ReplayBuffer,
    _SlabResult,
)


class _ProcessEventSink:
    """Synchronous bounded handoff; workers never open the result store."""

    def __init__(self, connection: Connection) -> None:
        self.connection = connection

    def write_boundary_events(self, events: BoundaryEvents) -> None:
        self._handoff("boundary", events)

    def write_failure_events(self, events: FailureEvents) -> None:
        self._handoff("failure", events)

    def _handoff(self, kind: str, events: BoundaryEvents | FailureEvents) -> None:
        self.connection.send((kind, events))
        if self.connection.recv() != ("accepted",):
            raise EngineError("particle event handoff was not acknowledged")


@dataclass(slots=True)
class _ProcessWorker:
    process: BaseProcess
    connection: Connection


def validate_worker_count(workers: int) -> None:
    """Resolve one execution-resource request before either check or run."""
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise EngineError("workers must be a positive integer")
    if workers > (os.cpu_count() or 1):
        raise EngineError("workers exceeds the available logical CPU count")


def _process_private_arrays(prepared: PreparedTransport) -> Iterator[np.ndarray]:
    for owner in prepared.private_workspaces:
        yield from process_array_views(owner)


def _share_process_state(state: ParticleState, store: ProcessArrayStore) -> None:
    """Move the explicit resident columns into parent-owned storage."""
    for item in dataclass_fields(ParticleState):
        name = item.name
        setattr(state, name, store.share(getattr(state, name)))


def _process_state(state: ParticleState) -> ParticleState:
    return ParticleState(*(getattr(state, item.name) for item in dataclass_fields(ParticleState)))


def _process_particle_worker(connection: Connection, payload: bytes) -> None:
    """Use the existing slab engine with exclusive scratch in one interpreter."""
    reader = ProcessArrayReader()
    replay_reader = ProcessArrayReader()
    try:
        prepared, state = cast(tuple[PreparedTransport, ParticleState], reader.loads(payload))
        sink = _ProcessEventSink(connection)
        connection.send(("ready", None))
        while True:
            command = connection.recv()
            if command[0] == "stop":
                return
            _, rows, interval, start, end, replay_payload = command
            result = _process_worker_slab(
                prepared, state, rows, interval, start, end, replay_payload, replay_reader, sink
            )
            connection.send(("done", result.statistics))
    except BaseException:
        try:
            connection.send(("error", traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        replay_reader.close()
        reader.close()
        connection.close()


def _process_worker_slab(
    prepared: PreparedTransport,
    state: ParticleState,
    rows: np.ndarray,
    interval: int,
    start: float,
    end: float,
    replay_payload: bytes,
    reader: ProcessArrayReader,
    sink: _EventSink,
) -> _SlabResult:
    replay = cast(_ReplayBuffer, reader.loads(replay_payload))
    try:
        return _advance_particle_slab(
            prepared,
            rows,
            interval,
            start,
            end,
            state,
            replay,
            sink,
        )
    finally:
        reader.release_views()


def _receive_process_message(worker: _ProcessWorker) -> tuple[str, object]:
    ready = wait([worker.connection, worker.process.sentinel])
    if worker.connection not in ready:
        raise EngineError(f"particle worker exited with code {worker.process.exitcode}")
    try:
        kind, payload = worker.connection.recv()
    except (EOFError, OSError) as error:
        raise EngineError("particle worker connection closed before completion") from error
    if kind == "error":
        raise EngineError(f"particle worker failed: {payload}")
    return kind, payload


class ProcessTeam:
    """One run owns fixed processes, shared state, and ordered macro barriers."""

    def __init__(
        self,
        prepared: PreparedTransport,
        state: ParticleState,
        active_index: ActiveIndex,
        event_statistics: _EventStatistics,
        *,
        memory_plan: CpuMemoryPlan,
        probe_particle_index: np.ndarray,
    ) -> None:
        self.prepared = prepared
        self.memory_plan = memory_plan
        self.probe_particle_index = probe_particle_index
        self.state = state
        self.active_index = active_index
        self.event_statistics = event_statistics
        self.workers: list[_ProcessWorker] = []
        self.store: ProcessArrayStore | None = None
        self.replay_store: ProcessArrayStore | None = None
        self.replay_storage: np.ndarray | None = None
        self.finished = False

    def __enter__(self) -> ProcessTeam:
        try:
            if self.memory_plan.worker_count > 1:
                self._start()
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def _start(self) -> None:
        prepared = self.prepared
        self.store = ProcessArrayStore()
        _share_process_state(self.state, self.store)
        hint = prepared.dynamics.last_field_cell
        if hint is not None:
            hint = self.store.share(hint)
            prepared.dynamics.last_field_cell = hint
            prepared.dynamics.native_context = prepared.dynamics.native_context._replace(
                accepted_field_cell=hint,
            )
        payload = self.store.dumps(
            (prepared, _process_state(self.state)),
            private_arrays=_process_private_arrays(prepared),
        )
        plan = self.memory_plan.reserve_process_padding(
            self.store.page_padding_bytes + 7 * mmap.PAGESIZE
        )
        self.memory_plan = plan
        components = self.memory_plan.components
        reserve = sum(value for name, value in components.items() if name.startswith("process_"))
        required = self.store.shared_bytes + (
            self.memory_plan.worker_count * self.store.private_bytes
        )
        if required > reserve:
            raise EngineError("process storage exceeds its predicted array reserve")
        context = multiprocessing.get_context("spawn")
        for _ in range(self.memory_plan.worker_count):
            parent, child = context.Pipe()
            process = context.Process(target=_process_particle_worker, args=(child, payload))
            # Windows PipeConnection implements the same documented Connection interface.
            worker = _ProcessWorker(process, cast(Connection, parent))
            self.workers.append(worker)
            try:
                process.start()
            except RuntimeError as error:
                raise EngineError(
                    "process workers require an importable main module and a guarded script entry"
                ) from error
            finally:
                child.close()
        pending = {worker.connection: worker for worker in self.workers}
        while pending:
            ready = self._ready_connections(pending)
            for connection in tuple(pending):
                if connection in ready:
                    kind, _ = _receive_process_message(pending.pop(connection))
                    if kind != "ready":
                        raise EngineError("particle worker did not complete preparation")
        self.store.release_private_templates()

    def _ready_connections(self, pending: dict[Connection, _ProcessWorker]) -> Sequence[object]:
        ready = wait([*pending, *(worker.process.sentinel for worker in self.workers)])
        for worker in self.workers:
            if worker.process.sentinel in ready and worker.connection not in ready:
                raise EngineError(f"particle worker exited with code {worker.process.exitcode}")
        return ready

    def allocate_replay(
        self, time_s: tuple[float, ...], *, include_all_particles: bool
    ) -> _ReplayBuffer:
        """Initialize macro views in one run-owned shared replay allocation."""
        if self.workers and time_s and self.replay_storage is None:
            self.replay_store = ProcessArrayStore()
            self.replay_storage = self.replay_store.share(
                np.empty(self.memory_plan.replay_work_bytes, dtype=np.uint8)
            )
        return _allocate_replay_buffer(
            time_s,
            self.prepared.schedule,
            self.probe_particle_index,
            include_all_particles=include_all_particles,
            storage=self.replay_storage,
        )

    def advance(
        self,
        interval: int,
        start: float,
        end: float,
        replay: _ReplayBuffer,
        writer: ResultWriter,
    ) -> None:
        prepared, state = self.prepared, self.state
        if not self.workers:
            _advance_active_slabs(
                prepared,
                self.active_index.rows,
                interval,
                start,
                end,
                state,
                replay,
                self.event_statistics,
                writer,
            )
            return
        if not replay.time_s.size:
            # The replay owner allocates no rows or payload without requested times.
            payload = pickle.dumps(replay, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            if self.replay_store is None:
                raise EngineError("process replay storage was not prepared")
            payload = self.replay_store.describe(replay)
        self._advance_processes(interval, start, end, payload, writer)

    def release_replay(self) -> None:
        if self.replay_store is not None:
            self.replay_store.release_views()

    def _advance_processes(
        self,
        interval: int,
        start: float,
        end: float,
        replay_payload: bytes,
        writer: ResultWriter,
    ) -> None:
        rows = self.active_index.rows
        width = self.memory_plan.active_slab_particles(rows.size)
        pending: dict[Connection, _ProcessWorker] = {}
        next_row = 0
        for worker in self.workers:
            if next_row < rows.size:
                self._send_slab(worker, rows, next_row, width, interval, start, end, replay_payload)
                pending[worker.connection] = worker
                next_row += width
        while pending:
            ready = self._ready_connections(pending)
            for worker in tuple(pending.values()):
                if worker.connection not in ready:
                    continue
                if not self._receive_handoff(worker, writer):
                    continue
                if next_row < rows.size:
                    self._send_slab(
                        worker, rows, next_row, width, interval, start, end, replay_payload
                    )
                    next_row += width
                else:
                    del pending[worker.connection]

    @staticmethod
    def _send_slab(
        worker: _ProcessWorker,
        rows: np.ndarray,
        offset: int,
        width: int,
        interval: int,
        start: float,
        end: float,
        replay_payload: bytes,
    ) -> None:
        worker.connection.send(
            ("slab", rows[offset : offset + width], interval, start, end, replay_payload)
        )

    def _receive_handoff(self, worker: _ProcessWorker, writer: ResultWriter) -> bool:
        kind, payload = _receive_process_message(worker)
        if kind == "done":
            self.event_statistics.merge(cast(_EventStatistics, payload))
            return True
        if kind == "boundary":
            writer.write_boundary_events(cast(BoundaryEvents, payload))
        elif kind == "failure":
            writer.write_failure_events(cast(FailureEvents, payload))
        else:
            raise EngineError("unexpected particle worker handoff")
        worker.connection.send(("accepted",))
        return False

    def finish(self) -> None:
        if self.finished:
            return
        for worker in self.workers:
            worker.connection.send(("stop",))
        for worker in self.workers:
            worker.process.join(timeout=30.0)
            if worker.process.is_alive() or worker.process.exitcode != 0:
                raise EngineError("particle worker did not shut down successfully")
        self.finished = True

    def __exit__(self, *_: object) -> None:
        for worker in self.workers:
            if worker.process.is_alive():
                worker.process.terminate()
            if worker.process.pid is not None:
                worker.process.join(timeout=5.0)
            worker.connection.close()
            worker.process.close()
        self.replay_storage = None
        if self.replay_store is not None:
            self.replay_store.close()
            self.replay_store = None
        if self.store is not None:
            self.store.close()


@dataclass(frozen=True, slots=True)
class _ProcessArrayStorage:
    name: str
    nbytes: int
    capacity: int
    private: bool


@dataclass(frozen=True, slots=True)
class _ProcessArrayDescriptor:
    storage: _ProcessArrayStorage
    view_id: int
    shape: tuple[int, ...]
    dtype: str
    strides: tuple[int, ...]
    offset: int
    writeable: bool


@dataclass(slots=True)
class _ProcessArrayRoot:
    storage: _ProcessArrayStorage
    handle: SharedMemory
    bytes_view: np.ndarray


def process_array_views(obj: object) -> Iterator[np.ndarray]:
    """Visit prepared numeric arrays once without inferring scratch ownership."""

    pending = [obj]
    visited: set[int] = set()
    while pending:
        value = pending.pop()
        identity = id(value)
        if identity in visited:
            continue
        visited.add(identity)
        if isinstance(value, np.ndarray):
            yield value
        elif is_dataclass(value) and not isinstance(value, type):
            pending.extend(getattr(value, field.name) for field in fields(value))
        elif isinstance(value, Mapping):
            pending.extend(value.values())
        elif isinstance(value, (tuple, list)):
            pending.extend(value)


def _process_array_dtype(dtype: np.dtype) -> None:
    if dtype.kind not in "biufc" or dtype.hasobject or dtype.fields or dtype.subdtype:
        raise ValueError("process arrays require primitive numeric dtypes")


def _process_array_base(array: np.ndarray) -> np.ndarray:
    root = array
    while isinstance(root.base, np.ndarray):
        root = root.base
    return root


def _process_buffer_owner(array: np.ndarray) -> object:
    owner = array.base
    while isinstance(owner, memoryview):
        owner = owner.obj
    return owner


def _process_array_bounds(descriptor: _ProcessArrayDescriptor) -> None:
    dtype = np.dtype(descriptor.dtype)
    _process_array_dtype(dtype)
    if len(descriptor.shape) != len(descriptor.strides) or any(
        length < 0 for length in descriptor.shape
    ):
        raise ValueError("process array shape and strides are inconsistent")
    if any(length == 0 for length in descriptor.shape):
        if not 0 <= descriptor.offset <= descriptor.storage.nbytes:
            raise ValueError("empty process array offset exceeds its root")
        return
    lower = upper = descriptor.offset
    for length, stride in zip(descriptor.shape, descriptor.strides, strict=True):
        extent = (length - 1) * stride
        lower += min(0, extent)
        upper += max(0, extent)
    if lower < 0 or upper + dtype.itemsize > descriptor.storage.nbytes:
        raise ValueError("process array view exceeds its root storage")


def _process_mapping_proxy(value: Mapping) -> MappingProxyType:
    return MappingProxyType(value)


def _reduce_process_mapping_proxy(value: MappingProxyType) -> tuple:
    return _process_mapping_proxy, (dict(value),)


class _ProcessArrayPickler(pickle.Pickler):
    def __init__(self, stream: io.BytesIO, store: ProcessArrayStore) -> None:
        super().__init__(stream, protocol=pickle.HIGHEST_PROTOCOL)
        self.store = store
        self.dispatch_table = {
            **copyreg.dispatch_table,
            MappingProxyType: _reduce_process_mapping_proxy,
        }

    def persistent_id(self, obj: object) -> _ProcessArrayDescriptor | None:
        if isinstance(obj, np.ndarray):
            return self.store._descriptor(obj)
        return None


class _ProcessArrayUnpickler(pickle.Unpickler):
    def __init__(self, stream: io.BytesIO, reader: ProcessArrayReader) -> None:
        super().__init__(stream)
        self.reader = reader

    def persistent_load(self, pid: object) -> np.ndarray:
        if not isinstance(pid, _ProcessArrayDescriptor):
            raise pickle.UnpicklingError("unknown process array descriptor")
        return self.reader._view(pid)


class ProcessArrayStore:
    """Parent-owned shared roots and metadata-only prepared-object transport.

    ``shared_bytes`` counts page-rounded named storage, including private
    templates. ``private_bytes`` counts one reader's private copies. Original
    arrays still belong to the caller; use returned shared views for mutation.
    Close only after all readers and users of the shared arrays have finished.
    """

    def __init__(self) -> None:
        self._roots: dict[str, _ProcessArrayRoot] = {}
        self._source_roots: dict[int, tuple[weakref.ReferenceType, str]] = {}
        self._shared_owners: dict[int, str] = {}
        self._private_roots: dict[int, weakref.ReferenceType] = {}
        self._views: dict[int, tuple[weakref.ReferenceType, _ProcessArrayDescriptor]] = {}
        self._next_view_id = 0
        self._closed = False
        self._sealed = False

    @property
    def shared_bytes(self) -> int:
        return sum(root.storage.capacity for root in self._roots.values())

    @property
    def private_bytes(self) -> int:
        return sum(
            max(1, root.storage.nbytes) for root in self._roots.values() if root.storage.private
        )

    @property
    def page_padding_bytes(self) -> int:
        return sum(root.storage.capacity - root.storage.nbytes for root in self._roots.values())

    def _open(self) -> None:
        if self._closed:
            raise RuntimeError("process array store is closed")

    def _describing(self) -> None:
        self._open()
        if self._sealed:
            raise RuntimeError("process array preparation is sealed")

    def _known_root(self, root: np.ndarray) -> tuple[_ProcessArrayRoot, int] | None:
        source = self._source_roots.get(id(root))
        if source is not None and source[0]() is root:
            return self._roots[source[1]], int(root.ctypes.data)
        name = self._shared_owners.get(id(_process_buffer_owner(root)))
        if name is not None:
            record = self._roots[name]
            return record, int(record.bytes_view.ctypes.data)
        return None

    def _root(self, root: np.ndarray) -> tuple[_ProcessArrayRoot, int]:
        known = self._known_root(root)
        if known is not None:
            return known
        _process_array_dtype(root.dtype)
        if not root.flags.owndata or not (root.flags.c_contiguous or root.flags.f_contiguous):
            raise ValueError("process array roots must own contiguous C or Fortran storage")
        capacity = max(1, (root.nbytes + mmap.PAGESIZE - 1) // mmap.PAGESIZE) * mmap.PAGESIZE
        private = self._private_roots.get(id(root))
        handle = SharedMemory(create=True, size=capacity)
        try:
            bytes_view = np.ndarray((capacity,), dtype=np.uint8, buffer=handle.buf)
            bytes_view.fill(0)
            bytes_view[: root.nbytes] = root.ravel(order="K").view(np.uint8)
        except BaseException:
            handle.close()
            handle.unlink()
            raise
        storage = _ProcessArrayStorage(
            handle.name, int(root.nbytes), capacity, private is not None and private() is root
        )
        record = _ProcessArrayRoot(storage, handle, bytes_view)
        self._roots[handle.name] = record
        self._source_roots[id(root)] = (weakref.ref(root), handle.name)
        self._source_roots[id(bytes_view)] = (weakref.ref(bytes_view), handle.name)
        self._shared_owners[id(_process_buffer_owner(bytes_view))] = handle.name
        return record, int(root.ctypes.data)

    def _descriptor(self, array: np.ndarray) -> _ProcessArrayDescriptor:
        self._describing()
        _process_array_dtype(array.dtype)
        root = _process_array_base(array)
        record, address = self._root(root)
        previous = self._views.get(id(array))
        view_id = (
            previous[1].view_id
            if previous is not None and previous[0]() is array
            else self._next_view_id
        )
        descriptor = _ProcessArrayDescriptor(
            record.storage,
            view_id,
            tuple(array.shape),
            array.dtype.str,
            tuple(array.strides),
            int(array.ctypes.data) - address,
            bool(array.flags.writeable),
        )
        _process_array_bounds(descriptor)
        if previous is not None and previous[0]() is array and previous[1] != descriptor:
            raise ValueError("process array layout changed after description")
        if previous is None or previous[0]() is not array:
            self._next_view_id += 1
        self._views[id(array)] = (weakref.ref(array), descriptor)
        return descriptor

    def share(self, array: np.ndarray) -> np.ndarray:
        """Initialize a root once and return its shared view without changing IDs."""

        descriptor = self._descriptor(array)
        if descriptor.storage.private:
            raise ValueError("private process arrays cannot become shared resident state")
        result = self._make_view(descriptor)
        return result

    def _make_view(self, descriptor: _ProcessArrayDescriptor) -> np.ndarray:
        result = np.ndarray(
            descriptor.shape,
            dtype=np.dtype(descriptor.dtype),
            buffer=self._roots[descriptor.storage.name].bytes_view,
            offset=descriptor.offset,
            strides=descriptor.strides,
        )
        result.setflags(write=descriptor.writeable)
        return result

    def dumps(self, obj: object, *, private_arrays: Iterable[np.ndarray] = ()) -> bytes:
        """Serialize facts with explicit private root ownership; no ndarray payload."""

        self._describing()
        try:
            for array in private_arrays:
                root = _process_array_base(array)
                known = self._known_root(root)
                if known is not None and not known[0].storage.private:
                    raise ValueError("shared process root cannot become private scratch")
                self._private_roots[id(root)] = weakref.ref(root)
            stream = io.BytesIO()
            _ProcessArrayPickler(stream, self).dump(obj)
            return stream.getvalue()
        except BaseException:
            self.close()
            raise

    def describe(self, obj: object) -> bytes:
        return self.dumps(obj)

    def release_views(self) -> None:
        """Forget completed-payload descriptors while preserving its shared roots."""

        self._describing()
        self._views.clear()

    def release_private_templates(self) -> None:
        """Seal preparation after every reader has copied its private scratch.

        Shared state and facts stay mapped until normal store closure. The
        detached reader copies keep their layout and aliases after their
        startup templates disappear; no new reader may load the old payload.
        """

        self._open()
        if self._sealed:
            return
        self._sealed = True
        names = {name for name, root in self._roots.items() if root.storage.private}
        roots = tuple(self._roots.pop(name) for name in names)
        self._source_roots = {
            identity: value
            for identity, value in self._source_roots.items()
            if value[1] not in names
        }
        self._shared_owners = {
            identity: name for identity, name in self._shared_owners.items() if name not in names
        }
        self._private_roots.clear()
        self._views = {
            identity: value
            for identity, value in self._views.items()
            if value[1].storage.name not in names
        }
        self._close_roots(roots)

    @staticmethod
    def _close_roots(roots: tuple[_ProcessArrayRoot, ...]) -> None:
        for root in roots:
            del root.bytes_view
            try:
                root.handle.close()
            finally:
                try:
                    root.handle.unlink()
                except FileNotFoundError:
                    pass

    def close(self) -> None:
        """Unlink parent-owned roots; closed stores cannot be reused."""

        if self._closed:
            return
        self._closed = True
        roots = tuple(self._roots.values())
        self._roots.clear()
        self._views.clear()
        self._source_roots.clear()
        self._private_roots.clear()
        self._shared_owners.clear()
        self._close_roots(roots)

    def __enter__(self) -> ProcessArrayStore:
        self._open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class ProcessArrayReader:
    """Child-owned mappings; explicit scratch roots are copied once per reader."""

    def __init__(self) -> None:
        self._storage: dict[str, _ProcessArrayStorage] = {}
        self._buffers: dict[str, np.ndarray] = {}
        self._handles: dict[str, SharedMemory] = {}
        self._views: dict[tuple[str, int], tuple[_ProcessArrayDescriptor, np.ndarray]] = {}
        self._closed = False

    def _buffer(self, storage: _ProcessArrayStorage) -> np.ndarray:
        if storage.name in self._buffers:
            if self._storage[storage.name] != storage:
                raise ValueError("process array root metadata changed")
            return self._buffers[storage.name]
        if storage.nbytes < 0 or storage.capacity < max(1, storage.nbytes):
            raise ValueError("process array root capacity is invalid")
        handle = SharedMemory(name=storage.name)
        try:
            if handle.size < storage.capacity:
                raise ValueError("process array shared mapping is too small")
            buffer = np.ndarray((storage.capacity,), dtype=np.uint8, buffer=handle.buf)
            if storage.private:
                buffer = buffer[: max(1, storage.nbytes)].copy()
                handle.close()
            else:
                self._handles[storage.name] = handle
        except BaseException:
            handle.close()
            raise
        self._storage[storage.name] = storage
        self._buffers[storage.name] = buffer
        return buffer

    def _view(self, descriptor: _ProcessArrayDescriptor) -> np.ndarray:
        _process_array_bounds(descriptor)
        key = descriptor.storage.name, descriptor.view_id
        if key in self._views:
            previous, array = self._views[key]
            if previous != descriptor:
                raise ValueError("process array view metadata changed")
            return array
        array = np.ndarray(
            descriptor.shape,
            dtype=np.dtype(descriptor.dtype),
            buffer=self._buffer(descriptor.storage),
            offset=descriptor.offset,
            strides=descriptor.strides,
        )
        array.setflags(write=descriptor.writeable)
        self._views[key] = descriptor, array
        return array

    def loads(self, payload: bytes) -> object:
        """Read trusted parent metadata; a failed load closes this reader."""

        if self._closed:
            raise RuntimeError("process array reader is closed")
        try:
            return _ProcessArrayUnpickler(io.BytesIO(payload), self).load()
        except BaseException:
            self.close()
            raise

    def release_views(self) -> None:
        """Forget completed-task descriptors, retaining the named root mappings.

        The engine calls this only when every consumer of the old payload has
        finished. New views may then use the same root with another layout;
        private copies and shared mapping ownership stay with this reader.
        """

        if self._closed:
            raise RuntimeError("process array reader is closed")
        self._views.clear()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._views.clear()
        self._buffers.clear()
        self._storage.clear()
        handles = tuple(self._handles.values())
        self._handles.clear()
        for handle in handles:
            handle.close()

    def __enter__(self) -> ProcessArrayReader:
        if self._closed:
            raise RuntimeError("process array reader is closed")
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
