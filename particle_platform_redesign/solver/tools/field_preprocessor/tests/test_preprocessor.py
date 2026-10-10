from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import yaml
from tests.verification.microcases import build_microcase
from tools.field_preprocessor.workflow import preprocess_from_configuration

from chamber_particles.case_format import (
    DataBundle,
    FieldData,
    P1TriLayout,
    Q1QuadLayout,
    RealizedTableSource,
    RegularLayout,
    read,
    write,
)
from chamber_particles.fields import FieldLocationError


def _p1_bundle(
    axes: list[float], *, hat: tuple[float, float] | None = None, rz: bool = False
) -> DataBundle:
    base = build_microcase("C06").data
    x, y = np.meshgrid(axes, [0.0, 1.0] if hat is None else axes, indexing="ij")
    nodes = np.ascontiguousarray(np.column_stack((x.ravel(), y.ravel())), dtype="<f8")
    ny = y.shape[1]
    cells = []
    for i in range(x.shape[0] - 1):
        for j in range(ny - 1):
            n = i * ny + j
            cells.extend(((n, n + ny, n + ny + 1), (n, n + ny + 1, n + 1)))
    layout = P1TriLayout(
        "p1", nodes, np.asarray(cells, dtype="<i8"), np.ones(len(cells), dtype="<u1")
    )
    values = 1.0 + 2.0 * nodes[:, 0] + 3.0 * nodes[:, 1]
    if hat is not None:
        values = ((nodes[:, 0] == hat[0]) & (nodes[:, 1] == hat[1])).astype(float)
    field = FieldData(
        "source",
        layout.name,
        "node",
        ("value",),
        "scalar",
        np.ascontiguousarray(values[:, None], dtype="<f8"),
        "1",
    )
    return DataBundle(
        "axisymmetric_rz" if rz else "cartesian_xy",
        base.provenance_json,
        base.geometry,
        (layout,),
        (field,),
        base.sources,
    )


def _configuration(source: Path) -> dict[str, Any]:
    return {
        "format_version": 2,
        "input": {"data_path": source.name},
        "target": {
            "kind": "regular",
            "name": "cache",
            "axis0": {"start_m": 0.0, "stop_m": 1.0, "count": 2},
            "axis1": {"start_m": 0.0, "stop_m": 1.0, "count": 2},
        },
        "validation": {"memory_limit_mb": 128.0, "workspace_rows": 32, "max_patch_work": 100_000},
        "fields": [
            {
                "source": "source",
                "output": "source_cache",
                "limits": {
                    "value_relative_l2": 10.0,
                    "gradient_relative_l2": 10.0,
                    "boundary_value_relative_l2": 10.0,
                },
            }
        ],
    }


def _run(
    tmp_path: Path,
    bundle: DataBundle,
    document: dict[str, Any] | None = None,
    *,
    source_time_diagnostic: bool = False,
) -> dict[str, Any]:
    source = tmp_path / "source.h5"
    write(source, bundle)
    configuration = tmp_path / "preprocessor.yaml"
    configuration.write_text(
        yaml.safe_dump(document or _configuration(source), sort_keys=False), encoding="utf-8"
    )
    return cast(
        dict[str, Any],
        preprocess_from_configuration(
            configuration,
            tmp_path / "cache.h5",
            report_path=tmp_path / "cache.json",
            source_time_diagnostic_requested=source_time_diagnostic,
        ),
    )


def _metrics(summary: dict[str, Any]) -> dict[str, Any]:
    return cast(dict[str, Any], summary["resampled_fields"][0]["metrics"])


def _assert_no_publication(tmp_path: Path) -> None:
    assert not (tmp_path / "cache.h5").exists()
    assert not (tmp_path / "cache.json").exists()


def test_static_p1_regular_publication_preserves_source_and_uses_true_norms(tmp_path: Path) -> None:
    bundle = _p1_bundle([0.0, 0.2, 1.0])
    summary = _run(tmp_path, bundle)
    cached = read(tmp_path / "cache.h5")
    np.testing.assert_array_equal(cached.fields[0].values, bundle.fields[0].values)
    np.testing.assert_array_equal(cached.geometry.nodes_m, bundle.geometry.nodes_m)
    np.testing.assert_array_equal(cached.sources[0].particle_id, bundle.sources[0].particle_id)
    assert cached.fields[1].time_s is None
    assert cached.fields[1].components == bundle.fields[0].components
    assert cached.fields[1].stored_basis == bundle.fields[0].stored_basis
    assert json.loads((tmp_path / "cache.json").read_text()) == summary
    assert summary["validation_resources"]["workspace_rows"] == 32
    for name in ("value", "gradient", "boundary_value"):
        assert _metrics(summary)[name]["relative_l2"] < 2e-14
        assert _metrics(summary)[name]["relative_l2_upper"] < 1e-10
    assert _metrics(summary)["value"]["measure"] == pytest.approx(1.0)
    assert _metrics(summary)["gradient"]["squared_reference_integral"] == pytest.approx(13.0)


@pytest.mark.parametrize("rz", [False, True])
def test_nonuniform_piecewise_linear_field_matches_independent_weighted_integrals(
    tmp_path: Path, rz: bool
) -> None:
    c = 0.2
    bundle = _p1_bundle([0.0, c, 1.0], rz=rz)
    layout = cast(P1TriLayout, bundle.layouts[0])
    field = replace(
        bundle.fields[0], values=np.ascontiguousarray(np.abs(layout.nodes_m[:, 0] - c)[:, None])
    )
    summary = _run(tmp_path, replace(bundle, fields=(field,)))
    metrics = _metrics(summary)
    if rz:
        error = (
            2
            * math.pi
            * ((1 - c) ** 2 * c**4 + 4 * c * c * (1 / 12 - c * c / 2 + 2 * c**3 / 3 - c**4 / 4))
        )
        reference = 2 * math.pi * (1 / 4 - 2 * c / 3 + c * c / 2)
        gradient_error = 4 * math.pi * ((1 - c) ** 2 * c * c + c * c * (1 - c * c))
        gradient_reference = math.pi
        boundary_error, boundary_reference = 2 * error, 2 * reference + 2 * math.pi * (1 - c) ** 2
    else:
        error = 4 * c * c * (1 - c) ** 2 / 3
        reference = (c**3 + (1 - c) ** 3) / 3
        gradient_error, gradient_reference = 4 * c * (1 - c), 1.0
        boundary_error, boundary_reference = 2 * error, 2 * reference + c * c + (1 - c) ** 2
    for name, expected_error, expected_reference in (
        ("value", error, reference),
        ("gradient", gradient_error, gradient_reference),
        ("boundary_value", boundary_error, boundary_reference),
    ):
        assert metrics[name]["squared_error_integral"] == pytest.approx(expected_error, rel=2e-13)
        assert metrics[name]["squared_reference_integral"] == pytest.approx(
            expected_reference, rel=2e-13
        )
        assert metrics[name]["relative_l2_upper"] >= metrics[name]["relative_l2"]


@pytest.mark.parametrize("hat,metric", [((0.1, 0.1), "value"), ((0.1, 0.0), "boundary_value")])
def test_source_local_hat_cannot_pass_a_coarse_target_gate(
    tmp_path: Path, hat: tuple[float, float], metric: str
) -> None:
    document = _configuration(tmp_path / "source.h5")
    document["fields"][0]["limits"][f"{metric}_relative_l2"] = 0.5
    with pytest.raises(ValueError, match=rf"{metric} relative L2 .* exceeds limit"):
        _run(tmp_path, _p1_bundle([0.0, 0.09, 0.1, 0.11, 1.0], hat=hat), document)
    _assert_no_publication(tmp_path)


@pytest.mark.parametrize("rz", [False, True])
def test_regular_tensor_hat_matches_analytic_norm_and_rejects_coarse_cache(
    tmp_path: Path, rz: bool
) -> None:
    base = _p1_bundle([0.0, 1.0], rz=rz)
    axes = np.asarray([0.0, 0.09, 0.1, 0.11, 1.0], dtype="<f8")
    layout = RegularLayout("regular_source", axes, axes, np.ones((4, 4), dtype="<u1"))
    values = np.zeros((25, 1), dtype="<f8")
    values[12] = 1.0
    bundle = replace(
        base,
        layouts=(layout,),
        fields=(replace(base.fields[0], layout=layout.name, values=values),),
    )
    summary = _run(tmp_path, bundle)
    metric = _metrics(summary)
    expected = (0.02 / 3) ** 2 * (2 * math.pi * 0.1 if rz else 1.0)
    assert metric["value"]["squared_reference_integral"] == pytest.approx(expected, rel=2e-13)
    assert metric["value"]["squared_error_integral"] == pytest.approx(expected, rel=2e-13)
    assert metric["value"]["relative_l2"] == pytest.approx(1.0)
    other = tmp_path / "rejected"
    other.mkdir()
    document = _configuration(other / "source.h5")
    document["fields"][0]["limits"]["value_relative_l2"] = 0.5
    with pytest.raises(ValueError, match=r"value relative L2 .* exceeds limit"):
        _run(other, bundle, document)
    _assert_no_publication(other)


@pytest.mark.parametrize("kind", ["q1", "p1_target"])
def test_unimplemented_capabilities_fail_without_publication(tmp_path: Path, kind: str) -> None:
    bundle = _p1_bundle([0.0, 1.0])
    document = _configuration(tmp_path / "source.h5")
    reason = "validation_unavailable_for_layout_pair"
    if kind == "q1":
        nodes = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.1, 1.0], [0.0, 1.0]])
        layout = Q1QuadLayout(
            "warped", nodes, np.asarray([[0, 1, 2, 3]], dtype="<i8"), np.ones(1, dtype="<u1")
        )
        bundle = replace(
            bundle, layouts=(layout,), fields=(replace(bundle.fields[0], layout=layout.name),)
        )
        reason = "validation_unavailable_for_warped_q1"
    else:
        document["target"] = {"kind": "existing", "layout": "p1"}
    with pytest.raises(
        ValueError, match=reason if kind != "p1_target" else "already uses the target layout"
    ):
        _run(tmp_path, bundle, document)
    _assert_no_publication(tmp_path)


@pytest.mark.parametrize("problem", ["gap", "overlap_gap"])
def test_support_gap_and_overlap_are_checked_separately(tmp_path: Path, problem: str) -> None:
    bundle = _p1_bundle([0.0, 0.5, 1.0], hat=(0.5, 0.5))
    layout = cast(P1TriLayout, bundle.layouts[0])
    if problem == "gap":
        support = layout.cell_support.copy()
        support[0] = 0
        layout = replace(layout, cell_support=support)
        reason = "extends outside source support"
    else:
        connectivity = layout.connectivity.copy()
        connectivity[1] = connectivity[0]
        layout = replace(layout, connectivity=connectivity)
        reason = "positive-area overlap"
    with pytest.raises(ValueError, match=reason):
        _run(tmp_path, replace(bundle, layouts=(layout,)))
    _assert_no_publication(tmp_path)


@pytest.mark.parametrize(
    "setting,value,reason",
    [
        ("memory_limit_mb", 0.5, "memory preflight"),
        ("max_patch_work", 1, "max_patch_work exhausted"),
        ("workspace_rows", 15, "at least 16"),
    ],
)
def test_validation_resource_bounds_stop_before_publication(
    tmp_path: Path, setting: str, value: float, reason: str
) -> None:
    document = _configuration(tmp_path / "source.h5")
    document["validation"][setting] = value
    with pytest.raises(ValueError, match=reason):
        _run(tmp_path, _p1_bundle([0.0, 1.0]), document)
    _assert_no_publication(tmp_path)


def test_resource_preflight_precedes_large_target_allocation(tmp_path: Path) -> None:
    document = _configuration(tmp_path / "source.h5")
    document["target"]["axis0"]["count"] = 10**10
    with pytest.raises(ValueError, match="memory preflight"):
        _run(tmp_path, _p1_bundle([0.0, 1.0]), document)
    _assert_no_publication(tmp_path)


def test_patch_work_is_shared_across_selected_fields(tmp_path: Path) -> None:
    bundle = _p1_bundle([0.0, 1.0])
    bundle = replace(bundle, fields=(*bundle.fields, replace(bundle.fields[0], name="second")))
    document = _configuration(tmp_path / "source.h5")
    document["fields"].append(
        {**document["fields"][0], "source": "second", "output": "second_cache"}
    )
    document["validation"]["max_patch_work"] = 16
    with pytest.raises(ValueError, match="max_patch_work exhausted"):
        _run(tmp_path, bundle, document)
    _assert_no_publication(tmp_path)


def test_exact_zero_reference_is_certified_without_epsilon_denominator(tmp_path: Path) -> None:
    bundle = _p1_bundle([0.0, 1.0])
    bundle = replace(
        bundle, fields=(replace(bundle.fields[0], values=np.zeros_like(bundle.fields[0].values)),)
    )
    document = _configuration(tmp_path / "source.h5")
    document["fields"][0]["limits"] = dict.fromkeys(document["fields"][0]["limits"], 0.0)
    metrics = _metrics(_run(tmp_path, bundle, document))
    for name in ("value", "gradient", "boundary_value"):
        assert metrics[name]["relative_l2_upper"] == 0.0


def test_shifted_bilinear_gradient_upper_covers_rounded_quadrature_points(tmp_path: Path) -> None:
    # In normalized coordinates (s,t), the diagonal P1 interpolant is min(s,t)
    # and its four-node regular cache is s*t. Direct integration gives gradient
    # squared error 1/3 and reference 1, independently of this scale/translation.
    origin = float(-(2**20))
    width = 2.0
    bundle = _p1_bundle([0.0, 1.0], hat=(1.0, 1.0))
    layout = cast(P1TriLayout, bundle.layouts[0])
    source = bundle.sources[0]
    assert isinstance(source, RealizedTableSource)
    bundle = replace(
        bundle,
        layouts=(replace(layout, nodes_m=width * layout.nodes_m + origin),),
        geometry=replace(bundle.geometry, nodes_m=width * bundle.geometry.nodes_m + origin),
        sources=(replace(source, position_m=width * source.position_m + origin),),
    )
    document = _configuration(tmp_path / "source.h5")
    for axis in ("axis0", "axis1"):
        document["target"][axis].update(start_m=origin, stop_m=origin + width)
    accepted = tmp_path / "reported"
    accepted.mkdir()
    metrics = _metrics(_run(accepted, bundle, document))["gradient"]
    exact_relative = math.sqrt(1.0 / 3.0)
    assert metrics["relative_l2_upper"] >= exact_relative
    assert metrics["reference_l2_lower"] <= 1.0
    # Before the coordinate/Hessian allowance, the reported upper was
    # 0.5773502691826807 and incorrectly permitted this below-truth limit.
    document["fields"][0]["limits"]["gradient_relative_l2"] = 0.577350269186
    with pytest.raises(ValueError, match=r"gradient relative L2 .* exceeds limit"):
        _run(tmp_path, bundle, document)
    _assert_no_publication(tmp_path)


def test_huge_numeric_is_a_location_bearing_configuration_error(tmp_path: Path) -> None:
    document = _configuration(tmp_path / "source.h5")
    document["validation"]["memory_limit_mb"] = 10**1000
    with pytest.raises(ValueError, match=r"validation.memory_limit_mb.*finite"):
        _run(tmp_path, _p1_bundle([0.0, 1.0]), document)
    _assert_no_publication(tmp_path)


def test_nonrepresentable_norm_is_unresolved_instead_of_zero_error(tmp_path: Path) -> None:
    bundle = _p1_bundle([0.0, 1.0])
    bundle = replace(
        bundle, fields=(replace(bundle.fields[0], values=bundle.fields[0].values * 1e-200),)
    )
    with pytest.raises(ValueError, match=r"unresolved_validation.*underflow"):
        _run(tmp_path, bundle)
    _assert_no_publication(tmp_path)


def test_unresolved_sliver_is_not_discarded_to_publish_a_cache(tmp_path: Path) -> None:
    bundle = _p1_bundle([index / 10 for index in range(11)], hat=(0.5, 0.5))
    document = _configuration(tmp_path / "source.h5")
    document["target"]["axis0"]["count"] = 11
    document["target"]["axis1"]["count"] = 11
    document["validation"]["max_patch_work"] = 1_000_000
    with pytest.raises(ValueError, match=r"unresolved_validation.*production locator"):
        _run(tmp_path, bundle, document)
    _assert_no_publication(tmp_path)


def test_unselected_time_field_is_preserved_without_claiming_its_cache_certification(
    tmp_path: Path,
) -> None:
    bundle = _p1_bundle([0.0, 1.0])
    field = bundle.fields[0]
    time = replace(
        field,
        name="unselected",
        values=np.stack((field.values, 2 * field.values)),
        time_s=np.asarray([0.0, 1.0], dtype="<f8"),
    )
    _run(tmp_path, replace(bundle, fields=(field, time)))
    cached = read(tmp_path / "cache.h5")
    preserved = next(item for item in cached.fields if item.name == time.name)
    np.testing.assert_array_equal(preserved.time_s, time.time_s)
    np.testing.assert_array_equal(preserved.values, time.values)


def test_rz_source_axis_regularity_is_required_for_static_p1_cache(tmp_path: Path) -> None:
    bundle = _p1_bundle([0.0, 1.0], rz=True)
    layout = cast(P1TriLayout, bundle.layouts[0])
    field = replace(
        bundle.fields[0],
        components=("r", "z"),
        stored_basis="axisymmetric_rz",
        unit="m/s",
        values=np.column_stack((np.ones(layout.nodes_m.shape[0]), layout.nodes_m[:, 1])),
    )
    with pytest.raises(FieldLocationError, match="zero radial component on the axis"):
        _run(tmp_path, replace(bundle, fields=(field,)))
    _assert_no_publication(tmp_path)


@pytest.mark.parametrize("rz", [False, True])
def test_regular_bilinear_source_uses_analytic_weighted_gradient_norm(
    tmp_path: Path, rz: bool
) -> None:
    base = _p1_bundle([0.0, 1.0], rz=rz)
    axes0, axes1 = np.asarray([0.0, 0.25, 1.0]), np.asarray([0.0, 0.5, 1.0])
    layout = RegularLayout("regular_source", axes0, axes1, np.ones((2, 2), dtype="<u1"))
    x, y = np.meshgrid(axes0, axes1, indexing="ij")
    values = (1 + 2 * x + 3 * y + 4 * x * y).reshape(-1, 1)
    field = replace(base.fields[0], layout=layout.name, values=values)
    summary = _run(tmp_path, replace(base, layouts=(layout,), fields=(field,)))
    gradient = _metrics(summary)["gradient"]
    assert gradient["squared_reference_integral"] == pytest.approx(
        151 * math.pi / 3 if rz else 131 / 3, rel=2e-13
    )
    assert gradient["relative_l2_upper"] < 1e-10
    assert summary["resampled_fields"][0]["certified_pair"] == "static_regular_to_full_regular"


def test_affine_q1_source_matches_independent_sheared_polynomial_integrals(tmp_path: Path) -> None:
    base = _p1_bundle([0.0, 1.0])
    nodes = np.asarray([[-1.0, -1.0], [3.0, -1.0], [4.0, 3.0], [0.0, 3.0]])
    layout = Q1QuadLayout(
        "affine_q1", nodes, np.asarray([[0, 1, 2, 3]], dtype="<i8"), np.ones(1, dtype="<u1")
    )
    # Reference field u*v = (x+3/4-y/4)(y+1)/16. Its cache drops -y²/64.
    field = replace(
        base.fields[0], layout=layout.name, values=np.asarray([[0.0], [0.0], [1.0], [0.0]])
    )
    summary = _run(tmp_path, replace(base, layouts=(layout,), fields=(field,)))
    metrics = _metrics(summary)
    assert metrics["value"]["squared_error_integral"] == pytest.approx(1 / 122880, rel=3e-13)
    assert metrics["gradient"]["squared_error_integral"] == pytest.approx(1 / 12288, rel=3e-13)
    assert metrics["boundary_value"]["squared_error_integral"] == pytest.approx(
        1 / 61440, rel=3e-13
    )
    assert summary["resampled_fields"][0]["certified_pair"] == "static_affine_q1_to_full_regular"


@pytest.mark.parametrize("rz", [False, True])
def test_affine_q1_bilinear_cache_preserves_true_gradient_norm(tmp_path: Path, rz: bool) -> None:
    base = _p1_bundle([0.0, 1.0], rz=rz)
    nodes = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    layout = Q1QuadLayout(
        "affine_q1", nodes, np.asarray([[0, 1, 2, 3]], dtype="<i8"), np.ones(1, dtype="<u1")
    )
    field = replace(
        base.fields[0], layout=layout.name, values=np.asarray([[0.0], [0.0], [1.0], [0.0]])
    )
    summary = _run(tmp_path, replace(base, layouts=(layout,), fields=(field,)))
    assert _metrics(summary)["gradient"]["squared_reference_integral"] == pytest.approx(
        5 * math.pi / 6 if rz else 2 / 3, rel=2e-13
    )
    assert _metrics(summary)["gradient"]["relative_l2_upper"] < 1e-10


@pytest.mark.parametrize("rz", [False, True])
@pytest.mark.parametrize("kind", ["p1", "regular", "affine_q1"])
def test_linear_time_cache_keeps_knots_and_weights_nonuniform_intervals(
    tmp_path: Path, rz: bool, kind: str
) -> None:
    bundle = _p1_bundle([0.0, 0.25, 1.0], rz=rz)
    field = bundle.fields[0]
    if kind != "p1":
        old = cast(P1TriLayout, bundle.layouts[0])
        layout: RegularLayout | Q1QuadLayout = (
            RegularLayout(
                "regular_source",
                np.asarray([0.0, 0.25, 1.0]),
                np.asarray([0.0, 1.0]),
                np.ones((2, 1), dtype="<u1"),
            )
            if kind == "regular"
            else Q1QuadLayout(
                "affine_q1",
                old.nodes_m,
                np.asarray([[0, 2, 3, 1], [2, 4, 5, 3]], dtype="<i8"),
                np.ones(2, dtype="<u1"),
            )
        )
        bundle = replace(bundle, layouts=(layout,))
        field = replace(field, layout=layout.name)
    timed = replace(
        field,
        time_s=np.asarray([0.0, 0.25, 2.0]),
        values=np.stack((field.values, 2 * field.values, 4 * field.values)),
    )
    summary = _run(tmp_path, replace(bundle, fields=(timed,)))
    cache = read(tmp_path / "cache.h5").fields[1]
    np.testing.assert_array_equal(cache.time_s, timed.time_s)
    assert cache.values.shape == (3, 4, 1)
    metrics = _metrics(summary)
    expected_factor = 203 / 24  # Exact time average of piecewise-linear amplitude².
    assert metrics["value"]["squared_reference_integral"] == pytest.approx(
        expected_factor * (47 * math.pi / 3 if rz else 40 / 3), rel=3e-13
    )
    assert metrics["gradient"]["squared_reference_integral"] == pytest.approx(
        expected_factor * 13 * (math.pi if rz else 1), rel=3e-13
    )
    assert metrics["value"]["relative_l2_upper"] < 1e-10
    assert metrics["time_scope"] == "maximum_ratio_on_every_linear_time_interval"
    assert len(metrics["time_intervals"]) == 2


@pytest.mark.parametrize("rz", [False, True])
def test_time_interval_gate_rejects_reference_cancellation_hidden_by_snapshots(
    tmp_path: Path, rz: bool
) -> None:
    bundle = _p1_bundle([0.0, 0.25, 1.0], rz=rz)
    layout = cast(P1TriLayout, bundle.layouts[0])
    shape = np.abs(layout.nodes_m[:, 0] - 0.25)[:, None]
    field = replace(
        bundle.fields[0],
        time_s=np.asarray([0.0, 1.0]),
        values=np.stack((100 + shape, -100 + shape)),
    )
    document = _configuration(tmp_path / "source.h5")
    document["fields"][0]["limits"]["value_relative_l2"] = 0.1
    # Snapshot value errors are <0.003; at t=1/2 the constant cancels.
    # The independent minimum reference variance gives max E²/R²=36/37
    # in XY and 324/431 in RZ, at an interior time near 1/2.
    with pytest.raises(ValueError, match=r"value relative L2 .* exceeds limit"):
        _run(tmp_path, replace(bundle, fields=(field,)), document)
    _assert_no_publication(tmp_path)
    accepted = tmp_path / "accepted"
    accepted.mkdir()
    relaxed = _configuration(accepted / "source.h5")
    relaxed["fields"][0]["limits"]["value_relative_l2"] = 2.0
    summary = _run(accepted, replace(bundle, fields=(field,)), relaxed)
    actual = _metrics(summary)["value"]
    exact_maximum = math.sqrt(324 / 431 if rz else 36 / 37)
    assert actual["relative_l2"] == pytest.approx(exact_maximum, rel=2e-9)
    assert exact_maximum <= actual["relative_l2_upper"] <= 2.0


def test_time_zero_reference_with_roundoff_is_unresolved_not_regularized(tmp_path: Path) -> None:
    bundle = _p1_bundle([0.0, 1.0])
    field = bundle.fields[0]
    timed = replace(
        field, time_s=np.asarray([0.0, 1.0]), values=np.stack((field.values, -field.values))
    )
    with pytest.raises(ValueError, match="unresolved_validation"):
        _run(tmp_path, replace(bundle, fields=(timed,)))
    _assert_no_publication(tmp_path)


def test_one_ulp_warp_is_not_reclassified_as_affine_for_publication(tmp_path: Path) -> None:
    base = _p1_bundle([0.0, 1.0])
    nodes = np.asarray([[0.0, 0.0], [1.0, 0.0], [np.nextafter(1.0, math.inf), 1.0], [0.0, 1.0]])
    layout = Q1QuadLayout(
        "tiny_warp", nodes, np.asarray([[0, 1, 2, 3]], dtype="<i8"), np.ones(1, dtype="<u1")
    )
    field = replace(base.fields[0], layout=layout.name)
    with pytest.raises(ValueError, match="validation_unavailable_for_warped_q1"):
        _run(tmp_path, replace(base, layouts=(layout,), fields=(field,)))
    _assert_no_publication(tmp_path)


def test_time_cache_resident_snapshot_array_is_in_memory_preflight(tmp_path: Path) -> None:
    bundle = _p1_bundle([0.0, 1.0])
    field = replace(
        bundle.fields[0],
        time_s=np.arange(400.0),
        values=np.repeat(bundle.fields[0].values[None], 400, axis=0),
    )
    document = _configuration(tmp_path / "source.h5")
    document["validation"]["memory_limit_mb"] = 8.0
    document["target"]["axis0"]["count"] = 100
    document["target"]["axis1"]["count"] = 100
    with pytest.raises(ValueError, match="validation memory preflight"):
        _run(tmp_path, replace(bundle, fields=(field,)), document)
    _assert_no_publication(tmp_path)


@pytest.mark.parametrize("rz", [False, True])
def test_saved_time_pulse_is_reported_even_when_spatial_cache_is_exact(
    tmp_path: Path, rz: bool
) -> None:
    bundle = _p1_bundle([0.0, 0.25, 1.0], rz=rz)
    field = bundle.fields[0]
    timed = replace(
        field,
        time_s=np.asarray([0.0, 0.25, 2.0]),
        values=np.stack((field.values, 3 * field.values, field.values)),
    )
    summary = _run(tmp_path, replace(bundle, fields=(timed,)), source_time_diagnostic=True)
    report = summary["resampled_fields"][0]
    assert summary["status"] == "complete"
    assert report["metrics"]["value"]["relative_l2_upper"] < 1e-10
    diagnostic = report["source_time_diagnostic"]
    assert diagnostic["status"] == "EVALUATED"
    assert diagnostic["continuous_time_fidelity"] == "NOT_TESTED"
    omission = diagnostic["omitted_knots"][0]
    assert omission["right_weight"] == pytest.approx(0.125)
    for name in ("value", "gradient", "boundary_value"):
        assert omission["metrics"][name]["relative_l2"] == pytest.approx(2 / 3)
    value = omission["metrics"]["value"]
    assert value["squared_error_integral"] == pytest.approx(188 * math.pi / 3 if rz else 160 / 3)
    assert value["squared_reference_integral"] == pytest.approx(141 * math.pi if rz else 120)
    gradient = omission["metrics"]["gradient"]
    assert gradient["squared_error_integral"] == pytest.approx(52 * (math.pi if rz else 1))
    np.testing.assert_array_equal(read(tmp_path / "cache.h5").fields[1].time_s, timed.time_s)


def test_nonuniform_time_affine_vector_uses_physical_time_weights(tmp_path: Path) -> None:
    bundle = _p1_bundle([0.0, 0.25, 1.0])
    scalar = bundle.fields[0].values
    vector = np.concatenate((scalar, -2 * scalar), axis=1)
    times = np.asarray([0.0, 0.125, 2.0, 3.0])
    field = replace(
        bundle.fields[0],
        stored_basis="cartesian_xy",
        components=("x", "y"),
        time_s=times,
        values=np.stack([(1 + 2 * time) * vector for time in times]),
    )
    summary = _run(tmp_path, replace(bundle, fields=(field,)), source_time_diagnostic=True)
    diagnostic = summary["resampled_fields"][0]["source_time_diagnostic"]
    assert diagnostic["status"] == "EVALUATED"
    assert len(diagnostic["omitted_knots"]) == 2
    for omission in diagnostic["omitted_knots"]:
        for name in ("value", "gradient", "boundary_value"):
            assert omission["metrics"][name]["relative_l2"] < 1e-14
            assert omission["metrics"][name]["relative_l2_upper"] < 1e-10


@pytest.mark.parametrize(
    "snapshot_count,status", [(0, "NOT_APPLICABLE"), (2, "INSUFFICIENT_SNAPSHOTS")]
)
def test_source_time_diagnostic_does_not_certify_unobserved_times(
    tmp_path: Path, snapshot_count: int, status: str
) -> None:
    bundle = _p1_bundle([0.0, 1.0])
    if snapshot_count:
        bundle = replace(
            bundle,
            fields=(
                replace(
                    bundle.fields[0],
                    time_s=np.asarray([0.0, 1.0]),
                    values=np.stack((bundle.fields[0].values,) * 2),
                ),
            ),
        )
    summary = _run(tmp_path, bundle, source_time_diagnostic=True)
    diagnostic = summary["resampled_fields"][0]["source_time_diagnostic"]
    assert diagnostic["status"] == status
    assert diagnostic["continuous_time_fidelity"] == "NOT_TESTED"
    assert diagnostic["omitted_knots"] == []


def test_source_time_diagnostic_resource_limit_does_not_veto_valid_cache(tmp_path: Path) -> None:
    bundle = _p1_bundle([0.0, 0.25, 1.0])
    field = replace(
        bundle.fields[0],
        time_s=np.asarray([0.0, 0.25, 2.0]),
        values=np.stack(
            (bundle.fields[0].values, 3 * bundle.fields[0].values, bundle.fields[0].values)
        ),
    )
    bundle = replace(bundle, fields=(field,))
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    before = _run(baseline, bundle)
    cache_work = before["validation_resources"]["patch_work"]
    assert before["resampled_fields"][0]["source_time_diagnostic"]["status"] == "NOT_REQUESTED"
    requested = tmp_path / "requested"
    requested.mkdir()
    document = _configuration(requested / "source.h5")
    document["validation"]["max_patch_work"] = cache_work
    after = _run(requested, bundle, document, source_time_diagnostic=True)
    assert after["status"] == "complete"
    assert after["validation_resources"]["patch_work"] == cache_work
    assert after["resampled_fields"][0]["metrics"] == before["resampled_fields"][0]["metrics"]
    assert after["resampled_fields"][0]["source_time_diagnostic"]["status"] == "RESOURCE_LIMITED"
    np.testing.assert_array_equal(
        read(requested / "cache.h5").fields[1].values, read(baseline / "cache.h5").fields[1].values
    )


@pytest.mark.parametrize("amplitude", [0.0, 1.0])
def test_saved_knots_cannot_certify_hidden_time_variation(tmp_path: Path, amplitude: float) -> None:
    bundle = _p1_bundle([0.0, 1.0])
    values = np.full_like(bundle.fields[0].values, amplitude)
    field = replace(
        bundle.fields[0],
        time_s=np.asarray([0.0, 1.0, 2.0]),
        values=np.stack((values, values, values)),
    )
    summary = _run(tmp_path, replace(bundle, fields=(field,)), source_time_diagnostic=True)
    diagnostic = summary["resampled_fields"][0]["source_time_diagnostic"]
    assert diagnostic["status"] == "EVALUATED"
    assert diagnostic["continuous_time_fidelity"] == "NOT_TESTED"
    assert diagnostic["omitted_knots"][0]["metrics"]["value"]["relative_l2"] == 0.0
    # amplitude + sin(pi*t)^2 has exactly these saved values, but an unsaved pulse at t=1/2.
    assert amplitude + math.sin(math.pi * 0.5) ** 2 == amplitude + 1.0
    assert diagnostic["omitted_knots"][0]["metrics"]["gradient"]["relative_l2"] == 0.0


def test_diagnostic_memory_refusal_preserves_cache_acceptance(tmp_path: Path) -> None:
    bundle = _p1_bundle([0.0, 1.0])
    field = replace(
        bundle.fields[0],
        time_s=np.arange(10.0),
        values=np.repeat(bundle.fields[0].values[None], 10, axis=0),
    )
    bundle = replace(bundle, fields=(field,))
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    before = _run(baseline, bundle)
    requested = tmp_path / "requested"
    requested.mkdir()
    document = _configuration(requested / "source.h5")
    document["validation"]["memory_limit_mb"] = (
        before["validation_resources"]["planned_owned_array_bytes"] + 1024
    ) / 1024**2
    after = _run(requested, bundle, document, source_time_diagnostic=True)
    assert after["status"] == "complete"
    assert after["resampled_fields"][0]["source_time_diagnostic"]["status"] == "RESOURCE_LIMITED"
    assert after["resampled_fields"][0]["metrics"] == before["resampled_fields"][0]["metrics"]
    assert after["source_time_diagnostic_resources"]["attempted_patch_work"] == 0


@pytest.mark.parametrize("kind", ["regular_gap", "q1_gap", "q1_overlap"])
def test_added_source_layouts_keep_exact_coverage_and_multiplicity_gates(
    tmp_path: Path, kind: str
) -> None:
    base = _p1_bundle([0.0, 0.5, 1.0])
    old = cast(P1TriLayout, base.layouts[0])
    if kind == "regular_gap":
        layout: RegularLayout | Q1QuadLayout = RegularLayout(
            "regular_source",
            np.asarray([0.0, 0.5, 1.0]),
            np.asarray([0.0, 1.0]),
            np.asarray([[0], [1]], dtype="<u1"),
        )
    else:
        cells = np.asarray([[0, 2, 3, 1], [2, 4, 5, 3]], dtype="<i8")
        support = np.asarray([0, 1], dtype="<u1")
        if kind == "q1_overlap":
            cells[1] = cells[0]
            support[:] = 1
        layout = Q1QuadLayout("affine_q1", old.nodes_m, cells, support)
    field = replace(base.fields[0], layout=layout.name)
    with pytest.raises(
        ValueError,
        match="positive-area overlap" if kind == "q1_overlap" else "extends outside source support",
    ):
        _run(tmp_path, replace(base, layouts=(layout,), fields=(field,)))
    _assert_no_publication(tmp_path)
