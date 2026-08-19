from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from particle_tracer_unified.core.boundary_core import (
    runtime_bounds,
    sample_geometry_normal,
    sample_geometry_part_id,
    sample_geometry_sdf,
    sample_geometry_sdf_points_2d,
)
from particle_tracer_unified.solvers._stochastic_path import _IntegratedOuLeafPath
from particle_tracer_unified.solvers.terminal_outcome import terminal_segment_outcome

ROOT = Path(__file__).resolve().parents[1]


def _runtime(
    *, geometry: object | None = None, field: object | None = None
) -> SimpleNamespace:
    return SimpleNamespace(
        geometry_provider=(
            SimpleNamespace(geometry=geometry) if geometry is not None else None
        ),
        field_provider=SimpleNamespace(field=field) if field is not None else None,
    )


def _grid_geometry(
    *,
    sdf: np.ndarray | None = None,
    normal_components: tuple[np.ndarray, np.ndarray] | None = None,
    part_ids: np.ndarray | None = None,
) -> SimpleNamespace:
    axes = (
        np.asarray([0.0, 1.0], dtype=np.float64),
        np.asarray([0.0, 1.0], dtype=np.float64),
    )
    return SimpleNamespace(
        axes=axes,
        spatial_dim=2,
        sdf=(
            np.asarray([[-1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
            if sdf is None
            else sdf
        ),
        normal_components=(
            (
                np.full((2, 2), 3.0, dtype=np.float64),
                np.full((2, 2), 4.0, dtype=np.float64),
            )
            if normal_components is None
            else normal_components
        ),
        nearest_boundary_part_id_map=(
            np.full((2, 2), 7.0, dtype=np.float64) if part_ids is None else part_ids
        ),
    )


def test_runtime_plan_type_dependencies_point_to_type_owners() -> None:
    source = (
        ROOT / "particle_tracer_unified" / "solvers" / "runtime_plan.py"
    ).read_text(encoding="utf-8")
    assert "from ._charge_model_types import ChargeModelConfig" in source
    assert "from ._stochastic_config import StochasticMotionConfig" in source
    assert "from .charge_model import ChargeModelConfig" not in source
    assert "from .stochastic_motion import StochasticMotionConfig" not in source


def test_runtime_bounds_uses_geometry_field_axes_or_triangle_vertices() -> None:
    axes = (
        np.asarray([-2.0, 1.0], dtype=np.float64),
        np.asarray([3.0, 8.0], dtype=np.float64),
    )
    expected_min = np.asarray([-2.0, 3.0], dtype=np.float64)
    expected_max = np.asarray([1.0, 8.0], dtype=np.float64)

    for runtime in (
        _runtime(geometry=SimpleNamespace(axes=axes)),
        _runtime(field=SimpleNamespace(axes=axes)),
    ):
        mins, maxs = runtime_bounds(runtime)
        np.testing.assert_array_equal(mins, expected_min)
        np.testing.assert_array_equal(maxs, expected_max)

    vertices = np.asarray([[-3.0, 2.0], [4.0, -5.0], [1.0, 6.0]])
    mins, maxs = runtime_bounds(_runtime(field=SimpleNamespace(mesh_vertices=vertices)))
    np.testing.assert_array_equal(mins, [-3.0, -5.0])
    np.testing.assert_array_equal(maxs, [4.0, 6.0])


def test_runtime_bounds_rejects_missing_or_unsupported_providers() -> None:
    with pytest.raises(
        ValueError, match="requires geometry_provider or field_provider"
    ):
        runtime_bounds(_runtime())
    with pytest.raises(ValueError, match="does not expose axes or mesh_vertices"):
        runtime_bounds(_runtime(field=SimpleNamespace()))


def test_geometry_sampling_preserves_provider_and_bounds_fallbacks() -> None:
    geometry = _grid_geometry()
    runtime = _runtime(geometry=geometry)

    np.testing.assert_allclose(
        sample_geometry_normal(runtime, np.asarray([0.5, 0.5])),
        [0.6, 0.8],
        rtol=0.0,
        atol=np.finfo(np.float64).eps,
    )
    assert sample_geometry_part_id(runtime, np.asarray([0.5, 0.5])) == 7
    assert sample_geometry_sdf(runtime, np.asarray([0.5, 0.5])) == 0.0

    zero_normals = (
        np.zeros((2, 2), dtype=np.float64),
        np.zeros((2, 2), dtype=np.float64),
    )
    np.testing.assert_array_equal(
        sample_geometry_normal(
            _runtime(geometry=_grid_geometry(normal_components=zero_normals)),
            np.asarray([0.5, 0.5]),
        ),
        [0.0, 1.0],
    )
    assert (
        sample_geometry_part_id(
            _runtime(
                geometry=_grid_geometry(
                    part_ids=np.full((2, 2), np.nan, dtype=np.float64)
                )
            ),
            np.asarray([0.5, 0.5]),
        )
        == 0
    )
    assert (
        sample_geometry_part_id(
            _runtime(
                geometry=_grid_geometry(
                    part_ids=np.full((2, 2), -3.0, dtype=np.float64)
                )
            ),
            np.asarray([0.5, 0.5]),
        )
        == 0
    )


def test_geometry_sampling_without_geometry_uses_field_bounds() -> None:
    axes = (
        np.asarray([0.0, 2.0], dtype=np.float64),
        np.asarray([0.0, 4.0], dtype=np.float64),
    )
    runtime = _runtime(field=SimpleNamespace(axes=axes))

    assert sample_geometry_part_id(runtime, np.asarray([0.1, 2.0])) == 0
    np.testing.assert_array_equal(
        sample_geometry_normal(runtime, np.asarray([0.1, 2.0])), [-1.0, 0.0]
    )
    assert sample_geometry_sdf(runtime, np.asarray([3.0, 6.0])) == pytest.approx(
        np.sqrt(5.0)
    )


def test_batch_sdf_sampling_returns_nan_for_unavailable_geometry() -> None:
    points = np.asarray([[0.25, 0.25], [0.75, 0.75]], dtype=np.float64)
    assert np.all(np.isnan(sample_geometry_sdf_points_2d(_runtime(), points)))
    assert np.all(
        np.isnan(
            sample_geometry_sdf_points_2d(
                _runtime(geometry=SimpleNamespace(spatial_dim=3)), points
            )
        )
    )
    malformed = _grid_geometry(sdf=np.zeros((1, 3), dtype=np.float64))
    assert np.all(
        np.isnan(sample_geometry_sdf_points_2d(_runtime(geometry=malformed), points))
    )
    with pytest.raises(ValueError, match="requires shape"):
        sample_geometry_sdf_points_2d(_runtime(), np.zeros((2, 3)))


def test_terminal_segment_outcome_copies_and_freezes_accepted_position() -> None:
    position = np.asarray([1.0, 2.0], dtype=np.float32)
    outcome = terminal_segment_outcome(
        accepted_elapsed_s=0.25,
        segment_duration_s=0.5,
        position=position,
        reason=" escaped ",
    )

    position[:] = -1.0
    assert outcome.accepted_elapsed_s == 0.25
    assert outcome.reason == "escaped"
    assert outcome.position.dtype == np.float64
    assert outcome.position.tolist() == [1.0, 2.0]
    assert not outcome.position.flags.writeable


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"segment_duration_s": np.nan}, "duration"),
        ({"segment_duration_s": -1.0}, "duration"),
        ({"accepted_elapsed_s": np.inf}, "accepted elapsed"),
        ({"accepted_elapsed_s": -1.0}, "accepted elapsed"),
        ({"accepted_elapsed_s": 2.0}, "accepted elapsed"),
        ({"position": np.asarray([])}, "finite coordinate vector"),
        ({"position": np.asarray([[1.0]])}, "finite coordinate vector"),
        ({"position": np.asarray([np.nan])}, "finite coordinate vector"),
        ({"reason": "  "}, "reason must be non-empty"),
    ],
)
def test_terminal_segment_outcome_rejects_invalid_contract_values(
    updates: dict[str, Any], message: str
) -> None:
    values: dict[str, Any] = {
        "accepted_elapsed_s": 0.5,
        "segment_duration_s": 1.0,
        "position": np.asarray([0.0, 1.0]),
        "reason": "stopped",
    }
    values.update(updates)
    with pytest.raises(ValueError, match=message):
        terminal_segment_outcome(**values)


def test_integrated_ou_leaf_normalizes_arrays_and_unsigned_seed() -> None:
    z_velocity = np.asarray([1.0, 2.0], dtype=np.float32)
    z_position = np.asarray([3.0, 4.0], dtype=np.float32)
    leaf = _IntegratedOuLeafPath(
        duration_s=1.0,
        tau_eff_s=2.0,
        thermal_velocity_variance_m2s2=3.0,
        z_velocity=z_velocity,
        z_position=z_position,
        bridge_seed=-1,
    )

    assert leaf.z_velocity.dtype == np.float64
    assert leaf.z_position.dtype == np.float64
    np.testing.assert_array_equal(leaf.z_velocity, [1.0, 2.0])
    np.testing.assert_array_equal(leaf.z_position, [3.0, 4.0])
    expected_rng = np.random.default_rng((1 << 64) - 1)
    assert leaf._bridge_rng.random() == expected_rng.random()


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"duration_s": np.nan}, "duration_s"),
        ({"duration_s": -1.0}, "duration_s"),
        ({"tau_eff_s": np.inf}, "tau_eff_s"),
        ({"tau_eff_s": 0.0}, "tau_eff_s"),
        ({"thermal_velocity_variance_m2s2": np.nan}, "variance"),
        ({"thermal_velocity_variance_m2s2": -1.0}, "variance"),
        ({"z_position": np.zeros(3)}, "matching shapes"),
        ({"z_velocity": np.asarray([np.inf, 0.0])}, "must be finite"),
        ({"z_position": np.asarray([0.0, np.nan])}, "must be finite"),
    ],
)
def test_integrated_ou_leaf_rejects_invalid_contract_values(
    updates: dict[str, Any], message: str
) -> None:
    values: dict[str, Any] = {
        "duration_s": 1.0,
        "tau_eff_s": 2.0,
        "thermal_velocity_variance_m2s2": 3.0,
        "z_velocity": np.zeros(2),
        "z_position": np.zeros(2),
        "bridge_seed": 7,
    }
    values.update(updates)
    with pytest.raises(ValueError, match=message):
        _IntegratedOuLeafPath(**values)
