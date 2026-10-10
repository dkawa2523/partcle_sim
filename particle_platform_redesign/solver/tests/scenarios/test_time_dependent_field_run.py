from __future__ import annotations

import os
from dataclasses import fields as dataclass_fields
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

import chamber_particles.engine as engine_module
import chamber_particles.output as output_module
from chamber_particles import SimulationError, load_case, open_result, simulate
from chamber_particles.case_format import write
from chamber_particles.fields import FIELD_TIME_REVISION
from tests.verification.microcases import materialize_microcase

_ELEMENTARY_CHARGE_C = 1.602176634e-19
_KNOT_S = 0.15
_END_S = 0.5


def test_linear_snapshot_field_uses_stage_time_and_splits_at_knot(tmp_path: Path) -> None:
    case_path = _time_field_case(tmp_path / "time-field", snapshot_end_s=_END_S)

    summary = simulate(load_case(case_path), tmp_path / "result")
    result = open_result(tmp_path / "result")
    final = result.read_final()
    source = load_case(case_path).data.sources[0]
    electric_impulse, electric_position_kernel = _piecewise_linear_integrals(_END_S)
    charge_to_mass = source.charge_number * _ELEMENTARY_CHARGE_C / source.mass_kg
    expected_velocity = source.velocity_m_s.copy()
    expected_velocity[:, 0] += charge_to_mass * electric_impulse
    expected_position = source.position_m + _END_S * source.velocity_m_s
    expected_position[:, 0] += charge_to_mass * electric_position_kernel

    np.testing.assert_allclose(final.position_m, expected_position, rtol=0.0, atol=2.0e-14)
    np.testing.assert_allclose(final.velocity_m_s, expected_velocity, rtol=0.0, atol=2.0e-14)
    frames = list(result.iter_frames())
    assert [frame.time_s for frame in frames] == [0.0, 0.25, _END_S]
    for frame in frames:
        impulse, position_kernel = _piecewise_linear_integrals(frame.time_s)
        frame_velocity = source.velocity_m_s.copy()
        frame_velocity[:, 0] += charge_to_mass * impulse
        frame_position = source.position_m + frame.time_s * source.velocity_m_s
        frame_position[:, 0] += charge_to_mass * position_kernel
        np.testing.assert_allclose(frame.position_m, frame_position, rtol=0.0, atol=2.0e-14)
        np.testing.assert_allclose(frame.velocity_m_s, frame_velocity, rtol=0.0, atol=2.0e-14)
    assert summary.macro_step_count == 4
    assert result.manifest["field_time_revision"] == FIELD_TIME_REVISION
    assert result.manifest["time"]["field_snapshot_splits_s"] == [_KNOT_S]
    assert result.manifest["resolved"]["required_fields"] == [
        {
            "name": "electric_field",
            "layout": "uniform",
            "unit": "V/m",
            "components": ["x", "y"],
            "stored_basis": "cartesian_xy",
            "time_interpolation": "linear",
            "snapshot_count": 3,
            "snapshot_range_s": [0.0, _END_S],
        }
    ]


def test_snapshot_range_outside_run_fails_closed_before_motion(tmp_path: Path) -> None:
    case_path = _time_field_case(tmp_path / "short-field", snapshot_end_s=0.4)

    with pytest.raises(SimulationError, match="snapshot range does not cover the run interval"):
        simulate(load_case(case_path), tmp_path / "rejected-result")


def test_snapshot_knot_on_fixed_grid_does_not_duplicate_macro_boundary(tmp_path: Path) -> None:
    case_path = _time_field_case(
        tmp_path / "aligned-knot",
        snapshot_end_s=_END_S,
        snapshot_knot_s=0.2,
    )

    summary = simulate(load_case(case_path), tmp_path / "aligned-result")
    result = open_result(tmp_path / "aligned-result")

    assert summary.macro_step_count == 3
    assert result.manifest["time"]["field_snapshot_splits_s"] == []


def test_non_grid_snapshot_knot_resume_matches_uninterrupted_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = load_case(_time_field_case(tmp_path / "resume-field", snapshot_end_s=_END_S))
    uninterrupted_path = tmp_path / "uninterrupted"
    interrupted_path = tmp_path / "interrupted"

    with monkeypatch.context() as cadence:
        cadence.setattr(engine_module, "_DURABLE_COMMIT_MINIMUM_WORK", 1)
        cadence.setattr(engine_module, "_DURABLE_COMMIT_WORK_PER_PARTICLE", 1)
        simulate(case, uninterrupted_path)
        with monkeypatch.context() as interruption:
            interruption.setattr(
                output_module.os,
                "replace",
                _fail_after_segment("epoch-000001.h5"),
            )
            with pytest.raises(SimulationError):
                simulate(case, interrupted_path)
        simulate(case, interrupted_path)

    uninterrupted = open_result(uninterrupted_path)
    resumed = open_result(interrupted_path)
    for key in (
        "time",
        "resume_identity",
        "resolved",
        "counts",
        "lifecycle_counts",
        "boundary_interactions",
    ):
        assert uninterrupted.manifest[key] == resumed.manifest[key]
    assert {
        key: value for key, value in uninterrupted.manifest.items() if key.endswith("_revision")
    } == {key: value for key, value in resumed.manifest.items() if key.endswith("_revision")}
    uninterrupted_records = [
        uninterrupted.read_final(),
        uninterrupted.read_release_events(),
        uninterrupted.read_boundary_events(),
        uninterrupted.read_failure_events(),
        uninterrupted.read_lifecycle_series(),
        *uninterrupted.iter_frames(),
    ]
    resumed_records = [
        resumed.read_final(),
        resumed.read_release_events(),
        resumed.read_boundary_events(),
        resumed.read_failure_events(),
        resumed.read_lifecycle_series(),
        *resumed.iter_frames(),
    ]
    assert len(uninterrupted_records) == len(resumed_records)
    for expected, actual in zip(uninterrupted_records, resumed_records, strict=True):
        _assert_record_identity(expected, actual)


def _fail_after_segment(expected_name: str) -> Any:
    real_replace = os.replace

    def replace_and_fail(source: Any, destination: Any) -> None:
        destination_path = Path(destination)
        real_replace(source, destination)
        if destination_path.parent.name == "segments" and destination_path.name == expected_name:
            raise OSError("injected time-field interruption after segment replace")

    return replace_and_fail


def _assert_record_identity(expected: Any, actual: Any) -> None:
    assert type(expected) is type(actual)
    for field in dataclass_fields(expected):
        expected_value = getattr(expected, field.name)
        actual_value = getattr(actual, field.name)
        if isinstance(expected_value, np.ndarray):
            assert isinstance(actual_value, np.ndarray)
            assert expected_value.dtype == actual_value.dtype
            assert expected_value.shape == actual_value.shape
            assert expected_value.tobytes(order="C") == actual_value.tobytes(order="C")
        else:
            assert expected_value == actual_value


def _time_field_case(
    directory: Path,
    *,
    snapshot_end_s: float,
    snapshot_knot_s: float = _KNOT_S,
) -> Path:
    paths = materialize_microcase("C04", directory)
    case = load_case(paths.case_path)
    static = case.data.fields[0]
    snapshot_x = np.asarray([0.0, 0.6, 1.0], dtype="<f8")
    snapshots = np.empty((3, static.values.shape[0], 2), dtype="<f8")
    snapshots[:, :, 0] = snapshot_x[:, None]
    snapshots[:, :, 1] = 0.0
    field = replace(
        static,
        values=snapshots,
        time_s=np.asarray([0.0, snapshot_knot_s, snapshot_end_s], dtype="<f8"),
    )
    data_path = directory / "time-field.h5"
    info = write(data_path, replace(case.data, fields=(field,)))

    document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
    document["case"]["data_path"] = data_path.name
    document["case"]["expected_content_hash"] = info.content_hash
    document["time"] = {"start_s": 0.0, "end_s": _END_S, "dt_s": 0.2}
    case_path = directory / "time-field.yaml"
    case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return case_path


def _piecewise_linear_integrals(end_s: float) -> tuple[float, float]:
    impulse = 0.0
    first_moment = 0.0
    for start_s, snapshot_end_s, start_value, end_value in (
        (0.0, _KNOT_S, 0.0, 0.6),
        (_KNOT_S, _END_S, 0.6, 1.0),
    ):
        if end_s <= start_s:
            break
        full_width_s = snapshot_end_s - start_s
        width_s = min(end_s, snapshot_end_s) - start_s
        slope = (end_value - start_value) / full_width_s
        segment_impulse = start_value * width_s + 0.5 * slope * width_s**2
        impulse += segment_impulse
        first_moment += (
            start_s * segment_impulse + 0.5 * start_value * width_s**2 + slope * width_s**3 / 3.0
        )
    return impulse, end_s * impulse - first_moment
