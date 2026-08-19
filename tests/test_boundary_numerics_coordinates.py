from __future__ import annotations

import inspect
import math
from types import SimpleNamespace

import numpy as np
import pytest

from particle_tracer_unified.core.boundary_numerics import (
    BOUNDARY_NUMERICS_POLICY_VERSION,
    BoundaryNumerics,
    resolve_boundary_numerics,
    scaled_classification_tolerance,
)
from particle_tracer_unified.core.coordinate_systems import (
    axis_names_for_coordinate_system,
    axisymmetric_rz_geometry_report,
    axisymmetric_rz_report_from_metadata,
    default_coordinate_system,
    normalize_coordinate_system,
    ring_area_weight,
    validate_axisymmetric_rz_radial_axis,
)


def _provider(
    *,
    axes: tuple[np.ndarray, ...],
    boundary_edges: object = None,
    boundary_triangles: object = None,
    boundary_loops_2d: tuple[np.ndarray, ...] = (),
) -> SimpleNamespace:
    geometry = SimpleNamespace(
        axes=axes,
        boundary_edges=boundary_edges,
        boundary_triangles=boundary_triangles,
        boundary_loops_2d=boundary_loops_2d,
    )
    return SimpleNamespace(geometry=geometry)


def test_boundary_numeric_public_signatures_are_stable() -> None:
    assert str(inspect.signature(resolve_boundary_numerics)) == (
        "(geometry_provider: 'object') -> 'BoundaryNumerics'"
    )
    assert str(inspect.signature(scaled_classification_tolerance)) == (
        "(coordinates_m: 'np.ndarray', resolution_length_m: 'float') "
        "-> 'tuple[float, float]'"
    )
    assert str(inspect.signature(axisymmetric_rz_geometry_report)) == (
        "(*, coordinate_system: 'Any', spatial_dim: 'int', "
        "axes: 'Sequence[Sequence[float]]', boundary_edges: 'Any' = None, "
        "boundary_edge_part_ids: 'Any' = None) -> 'dict[str, Any]'"
    )


def test_boundary_policy_preserves_float64_values_and_summary_order() -> None:
    axes = (
        np.asarray([0.0, 0.25, 1.0], dtype=np.float64),
        np.asarray([-2.0, 0.0, 2.0], dtype=np.float64),
    )
    edges = np.asarray(
        [
            [[0.0, -1.0], [0.0, 0.0]],
            [[1.0e-12, 0.0], [1.0e-12, 1.0]],
            [[0.5, 0.0], [0.5, 1.0]],
        ],
        dtype=np.float64,
    )

    policy = resolve_boundary_numerics(_provider(axes=axes, boundary_edges=edges))

    assert policy == BoundaryNumerics(
        policy_version=BOUNDARY_NUMERICS_POLICY_VERSION,
        reference_length_m=4.0,
        resolution_length_m=0.25,
        coordinate_roundoff_m=5.684341886080802e-14,
        classification_tolerance_m=2.5e-11,
        contact_offset_m=2.5e-09,
        radial_axis_tolerance_m=2.5e-11,
    )
    assert list(policy.summary()) == [
        "policy_version",
        "reference_length_m",
        "resolution_length_m",
        "coordinate_roundoff_m",
        "classification_tolerance_m",
        "contact_offset_m",
        "radial_axis_tolerance_m",
    ]


def test_boundary_policy_uses_triangle_altitudes_as_resolved_lengths() -> None:
    axes = tuple(np.asarray([0.0, 20.0], dtype=np.float64) for _ in range(3))
    triangles = np.asarray(
        [[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]]],
        dtype=np.float64,
    )

    policy = resolve_boundary_numerics(
        _provider(axes=axes, boundary_triangles=triangles)
    )

    assert policy.reference_length_m == 20.0
    assert policy.resolution_length_m == 2.4


def test_boundary_policy_falls_back_to_loops_for_malformed_primitives() -> None:
    axes = (
        np.asarray([0.0, 1.0], dtype=np.float64),
        np.asarray([0.0, 1.0], dtype=np.float64),
    )
    loop = np.asarray(
        [[0.0, 0.0], [0.125, 0.0], [0.125, 0.5]],
        dtype=np.float64,
    )

    policy = resolve_boundary_numerics(
        _provider(
            axes=axes,
            boundary_edges=np.zeros((2, 2), dtype=np.float64),
            boundary_triangles=np.zeros((2, 3), dtype=np.float64),
            boundary_loops_2d=(np.zeros((1, 2)), loop),
        )
    )

    assert policy.resolution_length_m == 0.125


@pytest.mark.parametrize(
    ("provider", "message"),
    [
        (SimpleNamespace(), "require a geometry provider"),
        (_provider(axes=()), "require geometry axes"),
        (
            _provider(axes=(np.asarray([1.0, 0.0]),)),
            "finite geometry axes with positive span",
        ),
        (
            _provider(axes=(np.asarray([0.0, np.nan, 1.0]),)),
            "could not resolve a positive geometry length",
        ),
    ],
)
def test_boundary_policy_preserves_validation_failures(
    provider: SimpleNamespace,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_boundary_numerics(provider)


@pytest.mark.parametrize(
    ("coordinates", "resolution", "message"),
    [
        (np.asarray([], dtype=np.float64), 1.0, "finite coordinates"),
        (np.asarray([np.inf]), 1.0, "finite coordinates"),
        (np.asarray([1.0]), 0.0, "positive resolution"),
        (np.asarray([1.0]), np.nan, "positive resolution"),
    ],
)
def test_scaled_tolerance_preserves_validation_order(
    coordinates: np.ndarray,
    resolution: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        scaled_classification_tolerance(coordinates, resolution)


@pytest.mark.parametrize(
    ("coordinate", "resolution"),
    [(0.0, 0.25), (1.0e9, 0.25), (-1.0e-120, 1.0e-125)],
)
def test_scaled_tolerance_is_exactly_64_ulps_or_resolution_fraction(
    coordinate: float,
    resolution: float,
) -> None:
    roundoff, tolerance = scaled_classification_tolerance(
        np.asarray([coordinate], dtype=np.float64),
        resolution,
    )
    expected_roundoff = 64.0 * float(
        np.spacing(np.float64(max(resolution, abs(coordinate))))
    )

    assert roundoff == expected_roundoff
    assert tolerance == max(1.0e-10 * resolution, expected_roundoff)


def test_rz_report_preserves_schema_order_and_axis_edge_diagnostics() -> None:
    axes = (
        np.asarray([0.0, 0.25, 1.0], dtype=np.float64),
        np.asarray([-2.0, 0.0, 2.0], dtype=np.float64),
    )
    edges = np.asarray(
        [
            [[0.0, -1.0], [0.0, 0.0]],
            [[1.0e-12, 0.0], [1.0e-12, 1.0]],
            [[0.5, 0.0], [0.5, 1.0]],
        ],
        dtype=np.float64,
    )

    report = axisymmetric_rz_geometry_report(
        coordinate_system="axisymmetric_rz",
        spatial_dim=2,
        axes=tuple(axis.tolist() for axis in axes),
        boundary_edges=edges,
        boundary_edge_part_ids=np.asarray([9, 7, 5], dtype=np.int32),
    )

    assert list(report) == [
        "coordinate_system",
        "axis_names",
        "semantics",
        "radial_axis_name",
        "axial_axis_name",
        "radial_axis_min_m",
        "radial_axis_max_m",
        "radial_axis_nonnegative",
        "r0_on_grid",
        "r0_detection_tolerance_m",
        "r0_axis_boundary_edge_count",
        "r0_axis_boundary_edge_indices",
        "r0_axis_boundary_part_ids",
        "velocity_components",
        "v_theta_dynamics",
        "source_ring_weighting_policy",
        "ring_area_weight_formula",
        "radial_ring_area_weight",
    ]
    assert report["r0_detection_tolerance_m"] == 2.5e-11
    assert report["r0_axis_boundary_edge_indices"] == [0, 1]
    assert report["r0_axis_boundary_part_ids"] == [7, 9]
    assert report["radial_ring_area_weight"] == {
        "count": 3,
        "min": 0.0,
        "max": 2.0 * math.pi,
        "sum": 2.5 * math.pi,
    }


@pytest.mark.parametrize(
    ("value", "spatial_dim", "normalized", "axis_names"),
    [
        (None, 2, "cartesian_xy", ("x", "y")),
        ("  ", 2, "cartesian_xy", ("x", "y")),
        ("R Z", 2, "axisymmetric_rz", ("r", "z")),
        ("cartesian-3d", 3, "cartesian_xyz", ("x", "y", "z")),
    ],
)
def test_coordinate_normalization_preserves_aliases_and_axis_names(
    value: object,
    spatial_dim: int,
    normalized: str,
    axis_names: tuple[str, ...],
) -> None:
    assert normalize_coordinate_system(value, spatial_dim) == normalized
    assert axis_names_for_coordinate_system(value, spatial_dim) == axis_names


def test_coordinate_normalization_preserves_dimension_errors() -> None:
    assert default_coordinate_system(3) == "cartesian_xyz"
    with pytest.raises(ValueError, match="spatial_dim must be 2 or 3"):
        default_coordinate_system(4)
    with pytest.raises(ValueError, match="Unsupported coordinate_system='polar'"):
        normalize_coordinate_system("polar", 2)
    with pytest.raises(ValueError, match="cartesian_xyz requires spatial_dim=3"):
        normalize_coordinate_system("xyz", 2)
    with pytest.raises(ValueError, match="spatial_dim=3 currently supports"):
        normalize_coordinate_system("axisymmetric_rz", 3)


@pytest.mark.parametrize(
    ("edges", "part_ids", "expected_indices"),
    [
        (None, None, []),
        (np.zeros((2, 2), dtype=np.float64), np.asarray([3, 4]), []),
        (
            np.asarray([[[1.0, 0.0], [1.0, 1.0]]], dtype=np.float64),
            None,
            [],
        ),
        (
            np.asarray([[[0.0, 0.0], [0.0, 1.0]]], dtype=np.float64),
            np.asarray([], dtype=np.int32),
            [0],
        ),
    ],
)
def test_rz_report_ignores_absent_or_incomplete_edge_metadata(
    edges: object,
    part_ids: object,
    expected_indices: list[int],
) -> None:
    report = axisymmetric_rz_geometry_report(
        coordinate_system="axisymmetric_rz",
        spatial_dim=2,
        axes=([0.0, 1.0], [-1.0, 1.0]),
        boundary_edges=edges,
        boundary_edge_part_ids=part_ids,
    )

    assert report["r0_axis_boundary_edge_indices"] == expected_indices
    assert report["r0_axis_boundary_part_ids"] == []


def test_rz_report_returns_before_inspecting_axes_for_cartesian_geometry() -> None:
    assert (
        axisymmetric_rz_geometry_report(
            coordinate_system="cartesian_xy",
            spatial_dim=2,
            axes=(),
            boundary_edges="not an array",
        )
        == {}
    )
    validate_axisymmetric_rz_radial_axis("cartesian_xy", 2, [])


@pytest.mark.parametrize(
    ("axes", "message"),
    [
        (([0.0, 1.0],), "requires r and z axes"),
        (([], [0.0, 1.0]), "non-empty 1D axis"),
        (([0.0, np.inf], [0.0, 1.0]), "only finite values"),
        (([-1.0, 0.0], [0.0, 1.0]), "must be non-negative"),
        (([0.0, 0.0], [0.0, 1.0]), "positive radial grid spacing"),
    ],
)
def test_rz_report_preserves_axis_validation_failures(
    axes: tuple[list[float], ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        axisymmetric_rz_geometry_report(
            coordinate_system="axisymmetric_rz",
            spatial_dim=2,
            axes=axes,
        )


def test_rz_report_uses_coordinate_dimension_validation_first() -> None:
    with pytest.raises(ValueError, match="spatial_dim must be 2 or 3"):
        axisymmetric_rz_geometry_report(
            coordinate_system="axisymmetric_rz",
            spatial_dim=4,
            axes=([0.0, 1.0], [0.0, 1.0]),
        )


def test_ring_weight_and_metadata_helpers_preserve_value_types() -> None:
    assert ring_area_weight(0.5) == math.pi
    weights = ring_area_weight(np.asarray([0.0, 0.5, 1.0], dtype=np.float32))
    assert isinstance(weights, np.ndarray)
    assert weights.dtype == np.float64
    np.testing.assert_array_equal(weights, np.asarray([0.0, math.pi, 2.0 * math.pi]))

    source = {"axisymmetric_rz": {"r0_on_grid": 1}}
    report = axisymmetric_rz_report_from_metadata(source)
    assert report == {"r0_on_grid": 1}
    assert report is not source["axisymmetric_rz"]
    assert axisymmetric_rz_report_from_metadata(None) == {}
    assert axisymmetric_rz_report_from_metadata({"axisymmetric_rz": "invalid"}) == {}


@pytest.mark.parametrize("radius", [np.nan, np.inf, -1.0])
def test_ring_weight_rejects_nonfinite_or_negative_radius(radius: float) -> None:
    with pytest.raises(ValueError, match="must be"):
        ring_area_weight(radius)
