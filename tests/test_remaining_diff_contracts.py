from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from particle_tracer_unified.compare._common import (
    as_long_reference,
    finite_float,
    first_present,
    merge_with_reference,
    row_particle_id,
    row_point_id,
    row_position,
    row_time,
    row_velocity,
)
from particle_tracer_unified.preflight_physics import (
    _dielectrophoresis_particle_issues,
    _force_field_issues,
    _gas_requirements,
    _particle_density_issues,
    _positive_number,
)
from particle_tracer_unified.solvers._contact_geometry import _build_contact_frame_3d
from particle_tracer_unified.solvers.contact_sliding import advance_contact_relaxation


def test_compare_row_adapters_preserve_alias_and_missing_value_contracts() -> None:
    row = {
        "sample_id": 7,
        "time_s": 0.25,
        "r": 1.0,
        "z": 2.0,
        "vr": -3.0,
        "vz": 4.0,
        "blank": "  ",
    }

    assert first_present(row, ("blank", "sample_id")) == 7
    assert row_time(row) == pytest.approx(0.25)
    assert row_point_id(row, 99) == 7
    assert row_particle_id({"pid": np.int64(8)}) == 8
    assert row_particle_id({"particle_id": np.nan}) is None
    np.testing.assert_array_equal(row_position(row, 2), [1.0, 2.0])
    velocity = row_velocity(row, 2)
    assert velocity is not None
    np.testing.assert_array_equal(velocity, [-3.0, 4.0])
    assert row_velocity({"x": 1.0, "y": 2.0}, 2) is None
    assert finite_float("1.5") == pytest.approx(1.5)
    assert finite_float("invalid", default=-1.0) == pytest.approx(-1.0)
    assert finite_float(np.inf, default=-2.0) == pytest.approx(-2.0)


def test_reference_frames_normalize_long_and_wide_schemas() -> None:
    long = pd.DataFrame(
        {
            "point_id": [1],
            "component": ["x"],
            "reference_value": [2.0],
        }
    )
    normalized = as_long_reference(long)
    assert normalized.to_dict("records") == [
        {"point_id": 1, "component": "x", "comsol_value": 2.0}
    ]

    wide = pd.DataFrame(
        {
            "sample_id": [2],
            "time_s": [0.5],
            "x": [3.0],
            "y": [-4.0],
        }
    )
    expanded = as_long_reference(wide, value_name="reference")
    assert expanded.to_dict("records") == [
        {
            "point_id": 2.0,
            "time_s": 0.5,
            "component": "x",
            "reference": 3.0,
        },
        {
            "point_id": 2.0,
            "time_s": 0.5,
            "component": "y",
            "reference": -4.0,
        },
    ]

    with pytest.raises(ValueError, match="point_id/sample_id"):
        as_long_reference(pd.DataFrame({"x": [1.0]}))


def test_reference_merge_uses_field_identity_and_reports_error_direction() -> None:
    sampled = pd.DataFrame(
        {
            "point_id": [1, 1],
            "field": ["electric", "flow"],
            "component": ["x", "x"],
            "python_value": [2.0, -1.0],
        }
    )
    reference = pd.DataFrame(
        {
            "point_id": [1, 1],
            "field": ["electric", "flow"],
            "component": ["x", "x"],
            "comsol_value": [1.0, 1.0],
        }
    )

    merged = merge_with_reference(sampled, reference)

    np.testing.assert_array_equal(merged["abs_error"], [1.0, 2.0])
    np.testing.assert_array_equal(merged["rel_error"], [1.0, 2.0])
    assert merged["sign_match"].tolist() == [True, False]


def test_reference_merge_uses_time_and_rejects_ambiguous_keys() -> None:
    sampled = pd.DataFrame(
        {
            "point_id": [1, 1],
            "time": [0.0, 1.0],
            "component": ["x", "x"],
            "python_value": [2.0, 3.0],
        }
    )
    reference = pd.DataFrame(
        {
            "point_id": [1, 1],
            "t": [0.0, 1.0],
            "component": ["x", "x"],
            "comsol_value": [2.0, 4.0],
        }
    )

    merged = merge_with_reference(sampled, reference)

    assert merged.shape[0] == 2
    assert merged["time_s"].tolist() == [0.0, 1.0]
    assert merged["abs_error"].tolist() == [0.0, 1.0]

    with pytest.raises(ValueError, match="duplicate comparison keys"):
        merge_with_reference(pd.concat([sampled, sampled.iloc[[0]]]), reference)


class _ForceSpec:
    def __init__(
        self,
        *,
        enabled: bool = True,
        model: str = "",
        force: object | None = None,
        config: object | None = None,
    ) -> None:
        self.enabled = enabled
        self.model = model
        self.force = force
        self.config = {} if config is None else config


def test_preflight_physics_reports_field_and_stochastic_requirements() -> None:
    pressure_issues = _force_field_issues(
        {"pressure_gradient": _ForceSpec()},
        (),
        (),
        "",
        ("", ""),
    )
    assert [issue.code for issue in pressure_issues] == ["physics.force.field.missing"]
    assert pressure_issues[0].context == {
        "feature": "pressure_gradient",
        "missing": ["flow_velocity|fluid_material_acceleration"],
    }

    physics = SimpleNamespace(stochastic=SimpleNamespace(enabled=True))
    requirements, density_features, issue = _gas_requirements(
        physics,
        {"gravity": _ForceSpec(force=SimpleNamespace(buoyancy=True))},
    )
    assert requirements == {"gravity_buoyancy": ("density_kgm3",)}
    assert density_features == {"gravity_buoyancy"}
    assert issue is not None
    assert issue.code == "physics.stochastic.drag"


def test_preflight_physics_accepts_explicit_valid_particle_properties() -> None:
    assert not _positive_number("not-a-number")
    runtime = SimpleNamespace(
        particles=SimpleNamespace(
            density=np.asarray([1200.0]),
            dep_particle_rel_permittivity=np.asarray([3.9]),
            particle_id=np.asarray([1]),
        )
    )
    assert _particle_density_issues(runtime, {"virtual_mass"}) == []
    assert (
        _dielectrophoresis_particle_issues(
            runtime,
            {"dielectrophoresis": _ForceSpec(config={})},
        )
        == []
    )
    assert (
        _dielectrophoresis_particle_issues(
            runtime,
            {
                "dielectrophoresis": _ForceSpec(
                    config={"particle_rel_permittivity": 2.5}
                )
            },
        )
        == []
    )


def test_contact_relaxation_rejects_nonfinite_or_negative_duration() -> None:
    for duration in (np.nan, np.inf, -1.0):
        with pytest.raises(ValueError, match="duration must be finite"):
            advance_contact_relaxation(
                np.asarray([0.0]),
                np.asarray([0.0]),
                np.asarray([0.0]),
                np.asarray([1.0]),
                duration,
            )


def test_degenerate_3d_contact_frame_is_rejected_without_state_mutation() -> None:
    state = SimpleNamespace(
        contact_edge_index=np.asarray([0], dtype=np.int64),
        contact_normal=np.zeros((1, 3), dtype=np.float64),
        x=np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64),
        v=np.asarray([[4.0, 5.0, 6.0]], dtype=np.float64),
    )
    execution = SimpleNamespace(
        state=state,
        plan=SimpleNamespace(
            boundary=SimpleNamespace(contact_offset_m=1.0e-9),
        ),
    )
    surface = SimpleNamespace(
        triangles=np.zeros((1, 3, 3), dtype=np.float64),
        normals=np.zeros((1, 3), dtype=np.float64),
    )
    diagnostics: dict[str, object] = {}
    position_before = state.x.copy()
    velocity_before = state.v.copy()

    frame = _build_contact_frame_3d(
        cast(Any, execution),
        np.asarray([0], dtype=np.int64),
        surface,
        diagnostics,
    )

    assert frame is None
    assert diagnostics["contact_frame_fail_count"] == 1
    np.testing.assert_array_equal(state.x, position_before)
    np.testing.assert_array_equal(state.v, velocity_before)
