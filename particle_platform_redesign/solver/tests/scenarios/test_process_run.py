from __future__ import annotations

import json
import multiprocessing
import os
from dataclasses import dataclass, fields, replace
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

import chamber_particles.cpu as cpu_module
import chamber_particles.engine as engine_module
import chamber_particles.output as output_module
from chamber_particles import (
    IncompleteResultError,
    SimulationError,
    load_case,
    open_result,
    simulate,
)
from chamber_particles.case import SimulationCase
from chamber_particles.case_format import RealizedSource, write
from chamber_particles.output import ResultView
from tests.scenarios.test_brownian_run import _brownian_first_passage_case
from tests.scenarios.test_durable_result import _assert_record_identity
from tests.scenarios.test_force_coupled_run import _continuous_charge_case
from tests.scenarios.test_periodic_run import _materialize_periodic_case
from tests.verification.microcases import materialize_microcase


@pytest.mark.parametrize("workers", (2, 4))
def test_process_brownian_terminal_and_continuing_rows_keep_full_payload(
    tmp_path: Path, workers: int
) -> None:
    case = _mixed_brownian_case(tmp_path / "case")
    reference_path, parallel_path = tmp_path / "serial", tmp_path / "parallel"
    simulate(case, reference_path, workers=1)
    simulate(case, parallel_path, workers=workers)
    reference, parallel = open_result(reference_path), open_result(parallel_path)
    _assert_science_identity(reference, parallel)

    first_frame = list(parallel.iter_frames())[1]
    assert bool((first_frame.lifecycle == 5).any())
    assert bool((first_frame.lifecycle == 1).any())
    final = parallel.read_final()
    np.testing.assert_array_equal(final.particle_id, np.arange(1, 33, dtype="<i8"))
    terminal = np.arange(32) % 5 != 0
    np.testing.assert_array_equal(final.lifecycle[terminal], 5)
    assert bool((final.velocity_m_s[terminal, 0] > 0.0).all())
    assert parallel.read_failure_events().particle_id.size == 0


@pytest.mark.parametrize(
    ("kind", "workers", "observed"),
    (("periodic", 2, True), ("charge", 4, True), ("periodic", 4, False), ("charge", 2, False)),
)
def test_process_shared_fields_charge_and_periodic_events_keep_payload(
    tmp_path: Path, kind: str, workers: int, observed: bool
) -> None:
    if kind == "periodic":
        case = _materialize_periodic_case(
            tmp_path / "case",
            position_m=(0.25, 0.5),
            velocity_m_s=(3.0, 0.0),
            end_s=0.6,
            dt_s=0.3,
        )
        times = (0.0, 0.125, 0.3, 0.45, 0.6)
    else:
        case = load_case(_continuous_charge_case(tmp_path / "case", dt_s=0.25))
        times = (0.0, 0.125, 0.25, 0.5)
    source = case.data.sources[0]
    selected = np.arange(32) % source.particle_id.size
    source = replace(
        source,
        **{
            item.name: value[selected]
            for item in fields(source)
            if isinstance(value := getattr(source, item.name), np.ndarray)
        },
    )
    source = replace(source, particle_id=np.arange(1, 33, dtype="<i8"))
    case = _write_observed_case(case, source, times, observed=observed)
    reference_path, parallel_path = tmp_path / "serial", tmp_path / "parallel"
    simulate(case, reference_path, workers=1)
    simulate(case, parallel_path, workers=workers)
    reference, parallel = open_result(reference_path), open_result(parallel_path)
    _assert_science_identity(reference, parallel)
    if not observed:
        assert list(parallel.iter_frames()) == []
        assert list(parallel.iter_probes()) == []
    assert parallel.read_failure_events().particle_id.size == 0
    if kind == "periodic":
        events = parallel.read_boundary_events()
        assert events.particle_id.size >= 32
        np.testing.assert_array_equal(events.interaction_kind, "periodic_translation")
    else:
        assert bool((parallel.read_final().charge_number != 0.0).all())


def test_process_checkpoint_resumes_with_a_different_worker_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(engine_module, "_DURABLE_COMMIT_MINIMUM_WORK", 1)
    monkeypatch.setattr(engine_module, "_DURABLE_COMMIT_WORK_PER_PARTICLE", 0)
    case = _mixed_brownian_case(tmp_path / "case")
    reference_path, resumed_path = tmp_path / "serial", tmp_path / "resumed"
    simulate(case, reference_path, workers=1)
    original_replace = os.replace

    def replace_then_fail(source: object, destination: object) -> None:
        original_replace(source, destination)
        if Path(destination).name == "LATEST":
            raise OSError("injected interruption after process macro commit")

    with monkeypatch.context() as patch:
        patch.setattr(output_module.os, "replace", replace_then_fail)
        with pytest.raises(SimulationError):
            simulate(case, resumed_path, workers=2)
    recovered = open_result(resumed_path, recovery=True)
    np.testing.assert_array_equal(recovered.read_lifecycle_series().time_s, [0.5])
    simulate(case, resumed_path, workers=4)
    _assert_science_identity(open_result(reference_path), open_result(resumed_path))


def test_abrupt_child_exit_preserves_only_the_last_complete_macro_and_can_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(engine_module, "_DURABLE_COMMIT_MINIMUM_WORK", 1)
    monkeypatch.setattr(engine_module, "_DURABLE_COMMIT_WORK_PER_PARTICLE", 0)
    case = _materialize_periodic_case(
        tmp_path / "case",
        position_m=(0.25, 0.5),
        velocity_m_s=(3.0, 0.0),
        end_s=0.6,
        dt_s=0.3,
    )
    source = case.data.sources[0]
    source = replace(
        source,
        **{
            item.name: np.repeat(value, 32, axis=0)
            for item in fields(source)
            if isinstance(value := getattr(source, item.name), np.ndarray)
        },
    )
    source = replace(source, particle_id=np.arange(1, 33, dtype="<i8"))
    case = _write_observed_case(case, source, (0.0, 0.125, 0.3, 0.45, 0.6))
    reference_path, interrupted_path = tmp_path / "serial", tmp_path / "interrupted"
    simulate(case, reference_path, workers=1)

    marker = tmp_path / "abrupt-child.json"
    shared_names: list[str] = []

    def record_shared_memory(*args: Any, **kwargs: Any) -> SharedMemory:
        handle = SharedMemory(*args, **kwargs)
        if kwargs.get("create"):
            shared_names.append(handle.name)
        return handle

    with monkeypatch.context() as patch:
        patch.setattr(engine_module, "_process_particle_worker", _AbruptBoundaryWorker(marker))
        patch.setattr(cpu_module, "SharedMemory", record_shared_memory)
        with pytest.raises(SimulationError):
            simulate(case, interrupted_path, workers=2)

    fault = json.loads(marker.read_text(encoding="utf-8"))
    assert fault["pid"] != os.getpid()
    assert fault["exit_code"] == 27
    assert 0.3 < fault["event_time_s"] <= 0.6
    assert multiprocessing.active_children() == []
    assert shared_names
    for name in set(shared_names):
        with pytest.raises(FileNotFoundError):
            SharedMemory(name=name)
    assert not interrupted_path.exists()
    assert (
        not interrupted_path.with_name(f"{interrupted_path.name}.partial")
        .joinpath("_SUCCESS")
        .exists()
    )
    with pytest.raises(IncompleteResultError):
        open_result(interrupted_path)
    recovered = open_result(interrupted_path, recovery=True)
    np.testing.assert_array_equal(recovered.read_lifecycle_series().time_s, [0.3])
    assert bool((recovered.read_boundary_events().time_s <= 0.3).all())
    assert max(frame.time_s for frame in recovered.iter_frames()) == 0.3
    assert max(frame.time_s for frame in recovered.iter_probes()) == 0.3

    simulate(case, interrupted_path, workers=4)
    assert multiprocessing.active_children() == []
    _assert_science_identity(open_result(reference_path), open_result(interrupted_path))


@pytest.mark.parametrize("workers", (0, True, 1.5))
def test_invalid_process_count_fails_before_output_creation(
    tmp_path: Path, workers: object
) -> None:
    case = load_case(materialize_microcase("C01", tmp_path / "case").case_path)
    output = tmp_path / "result"
    with pytest.raises(SimulationError):
        simulate(case, output, workers=workers)
    assert not output.exists()
    assert not output.with_name(f"{output.name}.partial").exists()


def _mixed_brownian_case(directory: Path) -> SimulationCase:
    path = _brownian_first_passage_case(
        directory, particle_count=32, tree_depth=4, adaptive_max_depth=6, seed=82017
    )
    case = load_case(path)
    source = case.data.sources[0]
    velocity = source.velocity_m_s.copy()
    velocity[np.arange(32) % 5 != 0, 0] = 0.5
    source = replace(source, velocity_m_s=velocity)
    return _write_observed_case(case, source, (0.0, 1.0 / 64, 0.125, 0.5, 0.75, 1.0), hold=True)


def _write_observed_case(
    case: SimulationCase,
    source: RealizedSource,
    times: tuple[float, ...],
    *,
    hold: bool = False,
    observed: bool = True,
) -> SimulationCase:
    data_path = case.case_path.with_name("process.h5")
    info = write(data_path, replace(case.data, sources=(source,)))
    document = yaml.safe_load(case.case_path.read_text(encoding="utf-8"))
    document["case"]["data_path"] = data_path.name
    document["case"]["expected_content_hash"] = info.content_hash
    document["time"]["end_s"] = times[-1]
    if hold:
        document["time"]["dt_s"] = 0.5
        document["boundaries"][0]["law"] = "hold"
    document["output"] = {
        "trajectories": {"selection": "all", "schedule": {"explicit_times_s": list(times)}},
        "probes": {
            "particle_ids": source.particle_id.tolist(),
            "schedule": {"explicit_times_s": list(times)},
        },
    }
    if not observed:
        document["output"] = {"trajectories": None}
    case_path = case.case_path.with_name("process.yaml")
    case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return load_case(case_path)


def _assert_science_identity(reference: ResultView, actual: ResultView) -> None:
    assert reference.manifest["resume_identity"] == actual.manifest["resume_identity"]
    for name in ("counts", "lifecycle_counts", "boundary_interactions", "failure_reason_counts"):
        assert reference.manifest[name] == actual.manifest[name]
    for reader in (
        "read_final",
        "read_release_events",
        "read_boundary_events",
        "read_failure_events",
        "read_lifecycle_series",
    ):
        _assert_record_identity(getattr(reference, reader)(), getattr(actual, reader)())
    for method in ("iter_frames", "iter_probes"):
        for first, second in zip(
            getattr(reference, method)(), getattr(actual, method)(), strict=True
        ):
            _assert_record_identity(first, second)


@dataclass(frozen=True)
class _AbruptBoundaryWorker:
    marker: Path

    def __call__(self, connection: Any, payload: bytes) -> None:
        # Spawn imports the ordinary production worker; this target is only test material.
        engine_module._process_particle_worker(
            _ExitAfterBoundaryHandoff(connection, self.marker), payload
        )


class _ExitAfterBoundaryHandoff:
    def __init__(self, connection: Any, marker: Path) -> None:
        self.connection = connection
        self.marker = marker

    def send(self, message: Any) -> None:
        self.connection.send(message)
        if message[0] == "boundary":
            event = message[1]
            selected = (event.particle_id == 1) & (event.time_s > 0.3)
            if bool(selected.any()):
                self.marker.write_text(
                    json.dumps(
                        {
                            "pid": os.getpid(),
                            "exit_code": 27,
                            "event_time_s": float(event.time_s[selected][0]),
                        }
                    ),
                    encoding="utf-8",
                )
                os._exit(27)

    def recv(self) -> Any:
        return self.connection.recv()

    def close(self) -> None:
        self.connection.close()
