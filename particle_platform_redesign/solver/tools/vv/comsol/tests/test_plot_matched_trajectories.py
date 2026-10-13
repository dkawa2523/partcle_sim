from __future__ import annotations

import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pytest
from tools.vv.comsol.plot_matched_trajectories import (
    create_trajectory_figures,
    read_matched_trajectories,
)

from chamber_particles.case_format import BoundaryData, DataBundle, GeometryData, write


def _write_rows(
    path: Path,
    rows: list[list[object]],
    *,
    xy: bool = False,
    precontact: bool = False,
    native: bool = False,
) -> None:
    columns = [
        "particle_id",
        "time_s",
        *(("x_m", "y_m") if xy else ("r_m", "z_m")),
        "current_status" if native else "lifecycle",
    ]
    if precontact:
        columns.append("precontact_eligible")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


def _geometry(path: Path, *, xy: bool = False) -> None:
    nodes = np.asarray([[0, 0], [0.2, 0], [0.2, 0.04], [0, 0.04]], dtype="<f8")
    lines = [[0, 1], [1, 2], [2, 3], [3, 0]] if xy else [[0, 1], [1, 2], [2, 3]]
    count = len(lines)
    boundary = BoundaryData(
        np.asarray(lines, dtype="<i8"),
        np.arange(count, dtype="<i4"),
        np.zeros(count, dtype="<i4"),
        np.zeros(count, dtype="<i4"),
        np.ones(count, dtype="<u1"),
        np.asarray([0, 0, 1, 1][:count], dtype="<i8"),
        np.ones(count, dtype="<i1"),
    )
    coordinates: Literal["cartesian_xy", "axisymmetric_rz"] = (
        "cartesian_xy" if xy else "axisymmetric_rz"
    )
    provenance = {
        "producer": "manufactured rectangular chamber",
        "producer_version": "1",
        "source_sha256": "sha256:" + "0" * 64,
        "field_semantics_revision": "geometry_only",
        "producer_metadata": {},
    }
    write(
        path,
        DataBundle(
            coordinates,
            json.dumps(provenance),
            GeometryData(
                nodes,
                boundary,
                ("wall",),
                tri3=np.asarray([[0, 1, 2], [0, 2, 3]], dtype="<i8"),
                tri3_domain_id=np.ones(2, dtype="<i4"),
            ),
        ),
    )


def _create(
    tmp_path: Path,
    candidate_rows: list[list[object]],
    reference_rows: list[list[object]],
    *,
    precontact: bool = False,
    geometry: bool = False,
    xy: bool = False,
) -> tuple[dict[str, object], Path]:
    candidate = tmp_path / "candidate.csv"
    reference = tmp_path / "reference.csv"
    _write_rows(candidate, candidate_rows, precontact=precontact, xy=xy)
    _write_rows(reference, reference_rows, precontact=precontact, xy=xy, native=True)
    geometry_path = tmp_path / "geometry.h5" if geometry else None
    if geometry_path is not None:
        _geometry(geometry_path, xy=xy)
    output = tmp_path / "figures"
    return create_trajectory_figures(
        candidate,
        reference,
        output,
        label="manufactured comparison",
        render_png=False,
        geometry_path=geometry_path,
    ), output


def _alignment(receipt: dict[str, object]) -> dict[str, object]:
    return cast(dict[str, object], receipt["alignment"])


def _position(receipt: dict[str, object]) -> dict[str, object]:
    return cast(dict[str, dict[str, object]], receipt["metrics"])["position_difference_m"]


@pytest.mark.parametrize(
    ("xy", "coordinate_system", "physical_facets", "axis_facets"),
    [(False, "axisymmetric_rz", 3, 1), (True, "cartesian_xy", 4, 0)],
)
def test_geometry_receipt_binds_units_identity_and_nonmaterial_axis(
    tmp_path: Path, xy: bool, coordinate_system: str, physical_facets: int, axis_facets: int
) -> None:
    rows: list[list[object]] = [[1, 0, 0, 0, "active"], [1, 0.001, 0.1, 0.02, "active"]]
    receipt, _ = _create(tmp_path, rows, rows, geometry=True, xy=xy)
    geometry = cast(dict[str, object], cast(dict[str, object], receipt["inputs"])["geometry"])
    assert geometry["unit"] == "m"
    assert geometry["coordinate_system"] == coordinate_system
    assert geometry["axis_seam_facet_count"] == axis_facets
    assert geometry["boundary_facet_count"] == physical_facets
    assert geometry["sha256"] == hashlib.sha256((tmp_path / "geometry.h5").read_bytes()).hexdigest()


@pytest.mark.parametrize("xy", [False, True])
def test_spatial_overlay_preserves_coordinate_alignment_si_scale_and_equal_aspect(
    tmp_path: Path, xy: bool
) -> None:
    rows: list[list[object]] = [[1, 0, 0, 0, "active"], [1, 0.001, 0.1, 0.02, "active"]]
    receipt, output = _create(tmp_path, rows, rows, geometry=True, xy=xy)
    rendering = cast(dict[str, object], receipt["rendering"])
    ranges = cast(dict[str, list[float]], rendering["overview_and_particle_ranges_mm"])
    assert (ranges["axis0"][1] - ranges["axis0"][0]) / 1370 == pytest.approx(
        (ranges["axis1"][1] - ranges["axis1"][0]) / 760
    )
    assert ranges["axis0"][0] < 0 and ranges["axis0"][1] > 200
    svg = ET.parse(output / "particles/particle_000001.svg")
    boundaries = [
        item
        for item in svg.iter()
        if item.tag.endswith("polyline") and item.attrib.get("stroke") == "#475569"
    ]
    starts = [
        item
        for item in svg.iter()
        if item.tag.endswith("circle") and item.attrib.get("stroke") == "#1D4ED8"
    ]
    x, y = map(float, boundaries[0].attrib["points"].split()[0].split(","))
    assert (float(starts[0].attrib["cx"]), float(starts[0].attrib["cy"])) == (x, y)
    assert _position(receipt)["maximum"] == 0
    ET.parse(output / ("overview_xy.svg" if xy else "overview_rz.svg"))


def test_joint_precontact_metrics_exclude_postreflection_but_preserve_observed_path(
    tmp_path: Path,
) -> None:
    candidate: list[list[object]] = [
        [1, 0, 0.1, 0.02, "active", 1],
        [1, 0.001, 0.11, 0.02, "active", 1],
        [1, 0.002, 9, 9, "active", 0],
    ]
    reference: list[list[object]] = [
        [1, 0, 0.1, 0.02, "active", 1],
        [1, 0.001, 0.11, 0.02, "active", 0],
        [1, 0.002, -9, -9, "active", 0],
    ]
    receipt, output = _create(tmp_path, candidate, reference, precontact=True)
    assert _alignment(receipt)["common_active_finite_rows"] == 3
    assert _alignment(receipt)["common_eligible_metric_rows"] == 1
    assert _position(receipt)["comparison_rows"] == 1
    assert _position(receipt)["maximum"] == 0
    svg = ET.parse(output / "particles/particle_000001.svg")
    candidate_paths = [
        item
        for item in svg.iter()
        if item.tag.endswith("polyline") and item.attrib.get("stroke") == "#1D4ED8"
    ]
    assert len(candidate_paths[0].attrib["points"].split()) == 3
    assert "joint precontact" in (output / "overview_rz.svg").read_text(encoding="utf-8")


def test_status_and_missing_coverage_do_not_infer_native_disappearance_cause(
    tmp_path: Path,
) -> None:
    candidate: list[list[object]] = [
        [1, 0, 0.1, 0.02, "active"],
        [1, 0.001, "NaN", "", "escaped"],
        [2, 0, "", "", "pending"],
        [2, 0.001, "", "", "failed"],
    ]
    reference: list[list[object]] = [
        [1, 0, 0.1, 0.02, "active"],
        [1, 0.001, "", "", "disappeared"],
        [1, 0.002, "", "", "disappeared"],
        [2, 0, "", "", "pending"],
        [2, 0.001, "", "", "unobserved"],
    ]
    receipt, output = _create(tmp_path, candidate, reference, geometry=True)
    alignment = _alignment(receipt)
    candidate_coverage = cast(dict[str, object], alignment["candidate"])
    reference_coverage = cast(dict[str, object], alignment["reference"])
    assert candidate_coverage["missing_coordinate_rows_by_observed_status"] == {
        "escaped": 1,
        "pending": 1,
        "failed": 1,
    }
    assert reference_coverage["missing_coordinate_rows_by_observed_status"] == {
        "disappeared": 2,
        "pending": 1,
        "unobserved": 1,
    }
    assert alignment["reference_only_input_keys"] == 1
    assert "unobserved_escaped_rows" not in str(alignment)
    assert receipt["particle_svg_count"] == 2
    assert _position(receipt)["comparison_rows"] == 1
    cause = cast(str, cast(dict[str, object], receipt["scope"])["native_disappeared_cause"])
    assert cause.startswith("unknown")
    ET.parse(output / "particles/particle_000002.svg")


def test_internal_missing_span_breaks_the_visual_path(tmp_path: Path) -> None:
    rows: list[list[object]] = [
        [1, 0, 0.1, 0.02, "active"],
        [1, 0.001, "", "", "unobserved"],
        [1, 0.002, 0.2, 0.04, "active"],
    ]
    receipt, output = _create(tmp_path, rows, rows)
    assert _position(receipt)["comparison_rows"] == 2
    svg = ET.parse(output / "particles/particle_000001.svg")
    paths = [
        item
        for item in svg.iter()
        if item.tag.endswith("polyline") and item.attrib.get("stroke") == "#1D4ED8"
    ]
    assert len(paths) == 2
    assert all(len(item.attrib["points"].split()) == 1 for item in paths)


def test_no_common_eligible_rows_is_reported_without_fabricated_error(tmp_path: Path) -> None:
    rows: list[list[object]] = [[1, 0, "", "", "pending", 0]]
    receipt, _ = _create(tmp_path, rows, rows, precontact=True, geometry=True)
    assert _position(receipt) == {
        "comparison_rows": 0,
        "rms": None,
        "maximum": None,
        "reason": "no jointly eligible finite active observations",
    }


def test_observed_held_tail_is_displayed_but_excluded_from_position_metrics(tmp_path: Path) -> None:
    candidate: list[list[object]] = [[1, 0, 0.1, 0.02, "active"], [1, 0.001, 9.0, 9.0, "held"]]
    reference: list[list[object]] = [[1, 0, 0.1, 0.02, "active"], [1, 0.001, -9.0, -9.0, "stuck"]]
    receipt, output = _create(tmp_path, candidate, reference)
    assert _position(receipt)["comparison_rows"] == 1
    assert _position(receipt)["maximum"] == 0
    coverage = cast(dict[str, object], _alignment(receipt)["candidate"])
    assert coverage["observed_status_counts"] == {"active": 1, "held": 1}
    assert coverage["non_active_finite_rows"] == 1
    svg = ET.parse(output / "particles/particle_000001.svg")
    paths = [
        item
        for item in svg.iter()
        if item.tag.endswith("polyline") and item.attrib.get("stroke") == "#1D4ED8"
    ]
    assert len(paths[0].attrib["points"].split()) == 2


@pytest.mark.parametrize(
    "failure", ["identity", "basis", "one_sided_eligibility", "active_missing", "terminal_return"]
)
def test_ambiguous_identity_basis_eligibility_or_lifecycle_is_rejected(
    tmp_path: Path, failure: str
) -> None:
    candidate = tmp_path / "candidate.csv"
    reference = tmp_path / "reference.csv"
    rows: list[list[object]] = [[1, 0, 0.1, 0.02, "active"]]
    _write_rows(candidate, rows)
    _write_rows(reference, rows, native=True)
    if failure == "identity":
        _write_rows(reference, [[2, 0, 0.1, 0.02, "active"]], native=True)
    elif failure == "basis":
        _write_rows(reference, rows, native=True, xy=True)
    elif failure == "one_sided_eligibility":
        _write_rows(candidate, [[1, 0, 0.1, 0.02, "active", 1]], precontact=True)
    elif failure == "active_missing":
        _write_rows(candidate, [[1, 0, "", "", "active"]])
    else:
        _write_rows(
            candidate, [*rows, [1, 0.001, "", "", "escaped"], [1, 0.002, 0.1, 0.02, "active"]]
        )
    with pytest.raises(ValueError):
        read_matched_trajectories(candidate, reference)


def test_geometry_basis_mismatch_is_rejected_before_publishing_figures(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.csv"
    reference = tmp_path / "reference.csv"
    _write_rows(candidate, [[1, 0, 0.1, 0.02, "active"]])
    _write_rows(reference, [[1, 0, 0.1, 0.02, "active"]], native=True)
    geometry = tmp_path / "geometry.h5"
    _geometry(geometry, xy=True)
    output = tmp_path / "figures"
    with pytest.raises(ValueError, match="geometry and trajectory"):
        create_trajectory_figures(
            candidate, reference, output, label="mismatch", render_png=False, geometry_path=geometry
        )
    assert not output.exists()
