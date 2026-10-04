from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from particle_tracer_unified.compare._common import json_safe
from particle_tracer_unified.compare._first_step_forces import (
    _external_acceleration,
    _force_contribution_frame,
    _named_force_component,
    _SampledForceState,
    _stokes_tau,
)
from particle_tracer_unified.compare._first_step_metrics import (
    _first_step_error_frame,
    _with_force_total_update_consistency,
)
from particle_tracer_unified.compare._first_step_report import _add_dt_sweep_ratios
from particle_tracer_unified.solvers.forces import ForceRuntimeParameters
from particle_tracer_unified.solvers.integrator_common import (
    DRAG_MODEL_EPSTEIN,
    DRAG_MODEL_NONE,
)
from tools import visualization_common as visual


def _first_step_context() -> Any:
    particles = SimpleNamespace(
        count=2,
        particle_id=np.asarray([1, 2], dtype=np.int64),
        source_part_id=np.asarray([7, 8], dtype=np.int32),
        position=np.asarray([[1.0, 2.0], [-1.0, 0.0]], dtype=np.float64),
        velocity=np.asarray([[0.5, -0.5], [1.0, 0.0]], dtype=np.float64),
    )
    return SimpleNamespace(
        particles=particles,
        coordinate_system="cartesian_xy",
        spatial_dim=2,
    )


def _final_particle_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "particle_id": 1,
                "x_m": 2.0,
                "y_m": 4.0,
                "vx_mps": 1.0,
                "vy_mps": -1.0,
            }
        ]
    )


def test_first_step_error_frame_preserves_columns_without_reference() -> None:
    actual = _first_step_error_frame(
        _first_step_context(),
        _final_particle_frame(),
        reference=None,
    )

    assert actual.columns.tolist() == [
        "particle_id",
        "source_part_id",
        "field_status",
        "notes",
        "x0",
        "vx0",
        "x1_solver",
        "vx1_solver",
        "x1_ref",
        "vx1_ref",
        "y0",
        "vy0",
        "y1_solver",
        "vy1_solver",
        "y1_ref",
        "vy1_ref",
        "position_error_m",
        "velocity_error_mps",
        "speed_ratio",
    ]
    assert actual[["x1_solver", "y1_solver"]].iloc[1].isna().all()
    assert (
        actual[["position_error_m", "velocity_error_mps", "speed_ratio"]]
        .isna()
        .all()
        .all()
    )


def test_first_step_error_frame_preserves_reference_metrics_and_order(
    tmp_path: Any,
) -> None:
    reference = tmp_path / "reference.csv"
    pd.DataFrame(
        [
            {
                "particle_id": 1,
                "x1_ref": 1.0,
                "y1_ref": 1.0,
                "vx1_ref": 0.5,
                "vy1_ref": -0.5,
            },
            {
                "particle_id": 2,
                "x1_ref": 0.0,
                "y1_ref": 0.0,
                "vx1_ref": 0.0,
                "vy1_ref": 0.0,
            },
        ]
    ).to_csv(reference, index=False)

    actual = _first_step_error_frame(
        _first_step_context(),
        _final_particle_frame(),
        reference=reference,
    )

    assert actual.columns.tolist() == [
        "particle_id",
        "source_part_id",
        "field_status",
        "notes",
        "x0",
        "vx0",
        "x1_solver",
        "vx1_solver",
        "y0",
        "vy0",
        "y1_solver",
        "vy1_solver",
        "x1_ref",
        "vx1_ref",
        "y1_ref",
        "vy1_ref",
        "position_error_m",
        "velocity_error_mps",
        "speed_ratio",
    ]
    np.testing.assert_allclose(
        actual[["position_error_m", "velocity_error_mps", "speed_ratio"]]
        .iloc[0]
        .to_numpy(dtype=np.float64),
        [np.sqrt(10.0), np.sqrt(0.5), 2.0],
        rtol=0.0,
        atol=1.0e-15,
    )
    assert (
        actual[["position_error_m", "velocity_error_mps", "speed_ratio"]]
        .iloc[1]
        .isna()
        .all()
    )


def test_first_step_error_frame_preserves_reference_validation_error(
    tmp_path: Any,
) -> None:
    reference = tmp_path / "reference.csv"
    pd.DataFrame([{"x1_ref": 1.0}]).to_csv(reference, index=False)

    with pytest.raises(
        ValueError,
        match=r"^reference CSV must contain particle_id$",
    ):
        _first_step_error_frame(
            _first_step_context(),
            _final_particle_frame(),
            reference=reference,
        )


def test_dt_sweep_ratios_preserve_missing_zero_and_nonfinite_values() -> None:
    rows: list[dict[str, Any]] = [
        {
            "force_update_velocity_residual_mps": {"max": 2.0},
            "force_update_position_residual_m": {"max": 0.0},
            "force_euler_velocity_residual_mps": {"max": np.nan},
            "force_euler_position_residual_m": {"max": np.inf},
        },
        {
            "force_update_velocity_residual_mps": {"max": 1.0},
            "force_update_position_residual_m": {"max": 4.0},
            "force_euler_velocity_residual_mps": {"max": 3.0},
            "force_euler_position_residual_m": {"max": 5.0},
        },
        {
            "force_update_position_residual_m": {"max": 2.0},
            "force_euler_velocity_residual_mps": {"max": np.inf},
            "force_euler_position_residual_m": {"max": np.nan},
        },
        {
            "force_update_velocity_residual_mps": {"max": 2.0},
            "force_euler_velocity_residual_mps": {"max": 1.0},
            "force_euler_position_residual_m": {"max": 1.0},
        },
    ]

    _add_dt_sweep_ratios(rows)

    ratio_names = [
        "force_update_velocity_residual_max_ratio_vs_previous",
        "force_update_position_residual_max_ratio_vs_previous",
        "force_euler_velocity_residual_max_ratio_vs_previous",
        "force_euler_position_residual_max_ratio_vs_previous",
    ]
    assert [rows[0][name] for name in ratio_names] == [None, None, None, None]
    assert [rows[1][name] for name in ratio_names] == [0.5, None, None, 0.0]
    assert rows[2][ratio_names[0]] is None
    assert rows[2][ratio_names[1]] == 0.5
    assert np.isinf(rows[2][ratio_names[2]])
    assert np.isnan(rows[2][ratio_names[3]])
    assert [rows[3][name] for name in ratio_names] == [2.0, None, 0.0, None]

    round_trip = json.loads(json.dumps(json_safe(rows)))
    assert np.isinf(round_trip[2][ratio_names[2]])
    assert np.isnan(round_trip[2][ratio_names[3]])


def test_force_total_consistency_preserves_2d_columns_and_values() -> None:
    first_step = pd.DataFrame(
        [
            {
                "particle_id": 1,
                "source_part_id": 7,
                "x0": 1.0,
                "y0": -2.0,
                "vx0": 0.5,
                "vy0": -0.25,
                "x1_solver": 1.3,
                "y1_solver": -2.2,
                "vx1_solver": 0.9,
                "vy1_solver": -0.4,
            },
            {
                "particle_id": 2,
                "source_part_id": 8,
                "x0": 0.0,
                "y0": 0.0,
                "vx0": 0.0,
                "vy0": 0.0,
                "x1_solver": 0.0,
                "y1_solver": 0.0,
                "vx1_solver": 0.0,
                "vy1_solver": 0.0,
            },
        ]
    )
    forces = pd.DataFrame(
        [
            {
                "particle_id": 1,
                "total_ax": 2.0,
                "total_ay": -1.0,
                "drag_ax": 1.5,
                "drag_ay": -0.5,
                "drag_tau_eff_s": 0.5,
            }
        ]
    )

    actual = _with_force_total_update_consistency(
        first_step,
        forces,
        axes=("x", "y"),
        dt=0.25,
    )

    assert actual.columns.tolist() == [
        "particle_id",
        "source_part_id",
        "x0",
        "y0",
        "vx0",
        "vy0",
        "x1_solver",
        "y1_solver",
        "vx1_solver",
        "vy1_solver",
        "x1_force_total",
        "vx1_force_total",
        "x1_force_total_euler",
        "vx1_force_total_euler",
        "y1_force_total",
        "vy1_force_total",
        "y1_force_total_euler",
        "vy1_force_total_euler",
        "force_total_update_velocity_residual_mps",
        "force_total_update_position_residual_m",
        "force_total_euler_velocity_residual_mps",
        "force_total_euler_position_residual_m",
    ]
    np.testing.assert_allclose(
        actual.loc[0, "x1_force_total":"force_total_euler_position_residual_m"],
        [
            1.1782653298563166,
            0.8934693402873666,
            1.25,
            1.0,
            -2.0891326649281585,
            -0.4467346701436833,
            -2.125,
            -0.5,
            0.047188758298148656,
            0.16465386694798098,
            0.14142135623730948,
            0.0901387818865999,
        ],
        rtol=0.0,
        atol=1.0e-15,
    )
    assert actual.loc[1, "x1_force_total":].isna().all()


def test_force_total_consistency_preserves_3d_predictor_values() -> None:
    first_step = pd.DataFrame(
        [
            {
                "particle_id": 3,
                "source_part_id": 9,
                "x0": 1.0,
                "y0": -2.0,
                "z0": 0.25,
                "vx0": 0.5,
                "vy0": -0.25,
                "vz0": 0.125,
                "x1_solver": 1.3,
                "y1_solver": -2.2,
                "z1_solver": 0.4,
                "vx1_solver": 0.9,
                "vy1_solver": -0.4,
                "vz1_solver": 0.3,
            }
        ]
    )
    forces = pd.DataFrame(
        [
            {
                "particle_id": 3,
                "total_ax": 2.0,
                "total_ay": -1.0,
                "total_az": 0.75,
                "drag_ax": 1.5,
                "drag_ay": -0.5,
                "drag_az": 0.25,
                "drag_tau_eff_s": 0.5,
            }
        ]
    )

    actual = _with_force_total_update_consistency(
        first_step,
        forces,
        axes=("x", "y", "z"),
        dt=0.25,
    )

    value_columns = [
        "x1_force_total",
        "vx1_force_total",
        "y1_force_total",
        "vy1_force_total",
        "z1_force_total",
        "vz1_force_total",
        "force_total_update_velocity_residual_mps",
        "force_total_update_position_residual_m",
        "force_total_euler_velocity_residual_mps",
        "force_total_euler_position_residual_m",
    ]
    np.testing.assert_allclose(
        actual[value_columns].iloc[0].to_numpy(dtype=np.float64),
        [
            1.1782653298563166,
            0.8934693402873666,
            -2.0891326649281585,
            -0.4467346701436833,
            0.3012244986961188,
            0.2725510026077625,
            0.05459144958288024,
            0.1920091028018111,
            0.1419727086450068,
            0.11528666716060464,
        ],
        rtol=0.0,
        atol=1.0e-15,
    )


def test_force_total_consistency_preserves_validation_error() -> None:
    with pytest.raises(
        ValueError,
        match=r"^force contribution frame must contain particle_id$",
    ):
        _with_force_total_update_consistency(
            pd.DataFrame([{"particle_id": 1}]),
            pd.DataFrame([{"total_ax": 0.0}]),
            axes=("x", "y"),
            dt=1.0,
        )


def test_force_total_consistency_uses_euler_fallback_for_invalid_drag() -> None:
    first_step = pd.DataFrame(
        [
            {
                "particle_id": 1,
                "x0": 1.0,
                "y0": 2.0,
                "vx0": 0.5,
                "vy0": -0.5,
                "x1_solver": np.nan,
                "y1_solver": 1.75,
                "vx1_solver": 1.0,
                "vy1_solver": -1.0,
            }
        ]
    )
    forces = pd.DataFrame(
        [
            {
                "particle_id": 1,
                "total_ax": 2.0,
                "total_ay": -1.0,
                "drag_ax": 1.0,
                "drag_ay": -0.5,
                "drag_tau_eff_s": np.nan,
            }
        ]
    )

    actual = _with_force_total_update_consistency(
        first_step,
        forces,
        axes=("x", "y"),
        dt=0.25,
    )

    np.testing.assert_allclose(
        actual[["x1_force_total", "y1_force_total"]].iloc[0].to_numpy(dtype=np.float64),
        [1.25, 1.8125],
    )
    position_residual = actual["force_total_update_position_residual_m"].to_numpy(
        dtype=np.float64
    )
    assert np.isnan(position_residual[0])


def test_force_frame_preserves_missing_particles_error() -> None:
    context: Any = SimpleNamespace(particles=None)

    with pytest.raises(ValueError, match=r"^Simulation requires particles$"):
        _force_contribution_frame(context)


def test_force_helpers_preserve_special_drag_and_body_behaviors() -> None:
    particles: Any = SimpleNamespace(count=2)
    static: Any = SimpleNamespace(
        mass_kg=np.asarray([1.0, 1.0]),
        diameter_m=np.asarray([0.1, 0.1]),
        density_kgm3=np.asarray([2.0, 0.0]),
    )
    viscosity = np.asarray([1.0, 1.0], dtype=np.float64)
    assert np.isinf(
        _stokes_tau(
            particles,
            static,
            viscosity,
            int(DRAG_MODEL_NONE),
        )
    ).all()
    assert np.isnan(
        _stokes_tau(
            particles,
            static,
            viscosity,
            int(DRAG_MODEL_EPSTEIN),
        )
    ).all()

    sampled = _SampledForceState(
        positions=np.zeros((2, 2), dtype=np.float64),
        velocities=np.zeros((2, 2), dtype=np.float64),
        time_s=0.0,
        status_codes=np.zeros(2, dtype=np.int32),
        flow=np.zeros((2, 2), dtype=np.float64),
        gas_density=np.asarray([1.0, 3.0], dtype=np.float64),
        gas_viscosity=viscosity,
        gas_temperature=np.asarray([300.0, 300.0], dtype=np.float64),
    )
    context: Any = SimpleNamespace(
        plan=SimpleNamespace(body_acceleration_mps2=(2.0,)),
    )
    buoyant_runtime = ForceRuntimeParameters(gravity_buoyancy_enabled=True)
    np.testing.assert_allclose(
        _external_acceleration(
            context,
            particles,
            static,
            sampled,
            buoyant_runtime,
            2,
        ),
        [[1.0, 0.0], [2.0, 0.0]],
    )

    calls: list[dict[str, Any]] = []

    def sample_component(**kwargs: Any) -> np.ndarray:
        calls.append(kwargs)
        return np.ones((2, 2), dtype=np.float64)

    compiled: Any = object()
    result = _named_force_component(
        enabled=True,
        name="lift",
        compiled=compiled,
        context=context,
        sampled=sampled,
        force_runtime=ForceRuntimeParameters(),
        zeros=np.zeros((2, 2), dtype=np.float64),
        component_sampler=sample_component,
    )

    np.testing.assert_array_equal(result, np.ones((2, 2), dtype=np.float64))
    assert len(calls) == 1
    assert calls[0]["force_runtime"].lift_enabled is True


def test_domain_part_polygons_preserve_element_and_part_order() -> None:
    vertices = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=np.float32,
    )
    triangles = np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
    quads = np.asarray([[0, 1, 3, 2]], dtype=np.int64)

    polygons, part_ids = visual._domain_part_polygons(
        vertices,
        triangles,
        np.asarray([7, 8], dtype=np.int64),
        quads,
        np.asarray([9], dtype=np.int64),
    )

    assert [polygon.tolist() for polygon in polygons] == [
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    ]
    assert all(polygon.dtype == np.float64 for polygon in polygons)
    assert part_ids.dtype == np.int32
    assert part_ids.tolist() == [7, 8, 9]


def test_domain_part_polygons_preserve_empty_and_default_part_behavior() -> None:
    empty_polygons, empty_ids = visual._domain_part_polygons(None)
    assert empty_polygons == []
    assert empty_ids.dtype == np.int32
    assert empty_ids.size == 0

    vertices = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=np.float64,
    )
    polygons, part_ids = visual._domain_part_polygons(
        vertices,
        np.asarray([[0, 1, 2]], dtype=np.int32),
        np.asarray([7, 8], dtype=np.int32),
        np.asarray([0, 1, 3, 2], dtype=np.int32),
        np.asarray([9], dtype=np.int32),
    )

    assert len(polygons) == 1
    assert part_ids.tolist() == [0]
