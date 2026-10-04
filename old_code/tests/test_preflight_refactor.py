from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from particle_tracer_unified import load_case, validate_case
from particle_tracer_unified._preflight_boundary import boundary_samples
from particle_tracer_unified._preflight_initial_state import (
    initial_particle_support_report,
)
from particle_tracer_unified.core.field_backend import ProviderSamplingBackend
from particle_tracer_unified.core.field_sampling import VALID_MASK_STATUS_CLEAN

ROOT = Path(__file__).resolve().parents[1]
MINIMAL_CASE = ROOT / "examples" / "v02_minimal" / "run_config.yaml"


def test_preflight_report_keeps_check_issue_and_schema_order() -> None:
    report = validate_case(load_case(MINIMAL_CASE), detail="full").to_dict()

    assert list(report) == [
        "artifact_type",
        "schema_version",
        "detail",
        "passed",
        "error_count",
        "warning_count",
        "issues",
        "checks",
    ]
    assert list(report["checks"]) == [
        "boundary_coverage",
        "experimental_features",
        "drag_regime",
        "initial_particles",
        "provider_boundary_support",
    ]
    assert report["issues"] == [
        {
            "code": "physics.drag.regime.transition",
            "severity": "warning",
            "message": (
                "The declared drag law starts in a transition range and requires "
                "model review"
            ),
            "context": {
                "model": "stokes",
                "scope": "initial_release_state",
                "reason_counts": {"knudsen_requires_rarefaction_review": 2},
            },
        }
    ]


def test_preflight_keeps_provider_call_and_boundary_candidate_order(
    monkeypatch: Any,
) -> None:
    calls: list[tuple[np.ndarray, float, tuple[str, ...]]] = []
    original_sample = ProviderSamplingBackend.sample

    def record_sample(
        backend: ProviderSamplingBackend,
        points_m: np.ndarray,
        time_s: float,
        request: Any,
    ) -> Any:
        calls.append(
            (
                np.asarray(points_m, dtype=np.float64).copy(),
                float(time_s),
                tuple(request.quantities),
            )
        )
        return original_sample(backend, points_m, time_s, request)

    monkeypatch.setattr(ProviderSamplingBackend, "sample", record_sample)
    validate_case(load_case(MINIMAL_CASE), detail="full")

    assert [(time_s, quantities) for _, time_s, quantities in calls] == [
        (0.0, ("ux", "uy", "mu")),
        (0.0, ("valid_mask_status",)),
        (0.025, ("valid_mask_status",)),
        (0.0, ("valid_mask_status",)),
    ]
    np.testing.assert_array_equal(
        calls[0][0],
        np.asarray([[-0.5, -0.2], [-0.5, 0.2]], dtype=np.float64),
    )
    np.testing.assert_array_equal(calls[1][0], calls[0][0][[0]])
    np.testing.assert_array_equal(calls[2][0], calls[0][0][[1]])
    np.testing.assert_array_equal(
        calls[3][0],
        np.asarray(
            [
                [0.0, -0.9292893218813455],
                [0.9292893218813455, 0.0],
                [0.0, 0.9292893218813455],
                [-0.9292893218813455, 0.0],
            ],
            dtype=np.float64,
        ),
    )


def test_boundary_samples_keep_geometry_order_shape_and_dtype() -> None:
    runtime = load_case(MINIMAL_CASE)._context

    points, normals, part_ids = boundary_samples(runtime)

    np.testing.assert_array_equal(
        points,
        np.asarray(
            [[0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
            dtype=np.float64,
        ),
    )
    np.testing.assert_array_equal(
        normals,
        np.asarray(
            [[-0.0, 1.0], [-1.0, 0.0], [-0.0, -1.0], [1.0, 0.0]],
            dtype=np.float64,
        ),
    )
    np.testing.assert_array_equal(part_ids, np.asarray([10, 20, 20, 10]))
    assert points.dtype == np.dtype(np.float64)
    assert normals.dtype == np.dtype(np.float64)
    assert part_ids.dtype == np.dtype(np.int64)


@pytest.mark.parametrize(
    ("config_name", "inside_position", "boundary_position", "outside_position"),
    [
        (
            "v02_minimal",
            np.asarray([0.0, 0.0]),
            np.asarray([1.0, 0.0]),
            np.asarray([1.1, 0.0]),
        ),
        (
            "v02_minimal_3d",
            np.asarray([0.0, 0.0, 0.0]),
            np.asarray([1.0, 0.0, 0.0]),
            np.asarray([1.1, 0.0, 0.0]),
        ),
    ],
)
def test_initial_particle_geometry_classifies_inside_boundary_and_outside(
    config_name: str,
    inside_position: np.ndarray,
    boundary_position: np.ndarray,
    outside_position: np.ndarray,
) -> None:
    context = load_case(ROOT / "examples" / config_name / "run_config.yaml")._context

    def report_for(position: np.ndarray) -> dict[str, Any]:
        positions = np.asarray(context.particles.position, dtype=np.float64).copy()
        positions[:] = inside_position
        positions[0] = position
        runtime = replace(
            context,
            particles=replace(context.particles, position=positions),
        )
        return initial_particle_support_report(runtime, include_violations=True)

    inside = report_for(inside_position)
    boundary = report_for(boundary_position)
    outside = report_for(outside_position)

    assert inside["geometry_status_counts"]["strict_inside"] == context.particles.count
    assert inside["geometry_passed"] is True
    assert boundary["geometry_status_counts"]["on_boundary"] == 1
    assert boundary["geometry_passed"] is False
    assert boundary["geometry_violations"][0]["status"] == "on_boundary"
    assert outside["geometry_status_counts"]["outside"] == 1
    assert outside["geometry_passed"] is False
    assert outside["geometry_violations"][0]["status"] == "outside"


def test_preflight_rejects_unprojected_boundary_release() -> None:
    case = load_case(MINIMAL_CASE)
    positions = np.asarray(case._context.particles.position, dtype=np.float64).copy()
    positions[0] = np.asarray([1.0, 0.0])
    context = replace(
        case._context,
        particles=replace(case._context.particles, position=positions),
    )

    report = validate_case(replace(case, _context=context), detail="full").to_dict()

    geometry_issue = next(
        issue for issue in report["issues"] if issue["code"] == "input.initial_geometry"
    )
    # The moved particle sits on a boundary that is not the entity its own
    # source_part_id declares, so it stays a violation.  Only a release on its
    # own entity is accepted.
    assert geometry_issue["context"] == {
        "status_counts": {
            "strict_inside": 1,
            "on_release_boundary": 0,
            "on_boundary": 1,
            "outside": 0,
        }
    }
    initial = report["checks"]["initial_particles"]
    assert initial["field_support_passed"] is True
    assert initial["geometry_passed"] is False
    assert initial["passed"] is False


def test_initial_support_groups_equal_release_times_in_stable_order(
    monkeypatch: Any,
) -> None:
    config = ROOT / "examples" / "v02_minimal_3d" / "run_config.yaml"
    context = load_case(config)._context
    release_times = np.asarray([0.5, 0.25, 0.5], dtype=np.float64)
    particles = replace(context.particles, release_time=release_times)
    runtime = replace(context, particles=particles)
    calls: list[tuple[float, np.ndarray]] = []

    def sample_statuses(
        _field_provider: Any,
        points: np.ndarray,
        time_s: float,
    ) -> np.ndarray:
        calls.append((float(time_s), np.asarray(points).copy()))
        return np.full(points.shape[0], int(VALID_MASK_STATUS_CLEAN), dtype=np.uint8)

    monkeypatch.setattr(
        "particle_tracer_unified._preflight_initial_state.sample_support_statuses",
        sample_statuses,
    )

    report = initial_particle_support_report(runtime, include_violations=False)

    assert report["checked_release_times_s"] == [0.25, 0.5]
    assert [time_s for time_s, _points in calls] == [0.25, 0.5]
    np.testing.assert_array_equal(calls[0][1], particles.position[[1]])
    np.testing.assert_array_equal(calls[1][1], particles.position[[0, 2]])


def test_initial_geometry_keeps_3d_surface_fail_fast_contract() -> None:
    config = ROOT / "examples" / "v02_minimal_3d" / "run_config.yaml"
    context = load_case(config)._context
    geometry = replace(
        context.geometry_provider.geometry,
        boundary_triangles=None,
        boundary_triangle_part_ids=None,
    )
    runtime = replace(
        context,
        geometry_provider=replace(context.geometry_provider, geometry=geometry),
    )

    with pytest.raises(
        ValueError,
        match=r"3D initial-particle validation requires geometry\.boundary_triangles",
    ):
        initial_particle_support_report(runtime, include_violations=False)


def test_initial_geometry_accepts_3d_surface_without_optional_part_ids() -> None:
    config = ROOT / "examples" / "v02_minimal_3d" / "run_config.yaml"
    context = load_case(config)._context
    geometry = replace(
        context.geometry_provider.geometry,
        boundary_triangle_part_ids=None,
    )
    runtime = replace(
        context,
        geometry_provider=replace(context.geometry_provider, geometry=geometry),
    )

    report = initial_particle_support_report(runtime, include_violations=False)

    assert report["geometry_passed"] is True


def test_initial_geometry_requires_resolved_positive_tolerance() -> None:
    context = load_case(MINIMAL_CASE)._context
    boundary = replace(context.plan.boundary, classification_tolerance_m=0.0)
    runtime = replace(context, plan=replace(context.plan, boundary=boundary))

    with pytest.raises(
        ValueError,
        match="requires a positive boundary classification tolerance",
    ):
        initial_particle_support_report(runtime, include_violations=False)
