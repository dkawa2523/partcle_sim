from __future__ import annotations

import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import cast

import pytest
from tools.vv.comsol.plot_matched_trajectories import (
    TOOL_REVISION,
    create_trajectory_figures,
    read_matched_trajectories,
)


def _write_trajectory(
    path: Path,
    *,
    offset_m: float = 0.0,
    particle_ids: tuple[int, ...] = (1, 4),
    status_column: str = "lifecycle",
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["particle_id", "time_s", "r_m", "z_m", status_column])
        for particle_id in particle_ids:
            for frame in range(3):
                time_s = frame * 1.0e-5
                writer.writerow(
                    [
                        particle_id,
                        time_s,
                        0.1 + particle_id * 1.0e-3 + time_s + offset_m,
                        0.02 + particle_id * 2.0e-3 + 2.0 * time_s - offset_m,
                        "active",
                    ]
                )


def test_external_plot_receipt_covers_every_aligned_particle(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.csv"
    reference = tmp_path / "reference.csv"
    _write_trajectory(candidate, offset_m=1.0e-12)
    _write_trajectory(reference, status_column="current_status")
    output = tmp_path / "figures"

    receipt = create_trajectory_figures(
        candidate,
        reference,
        output,
        label="synthetic matched case",
        render_png=False,
    )

    scope = cast(dict[str, object], receipt["scope"])
    rendering = cast(dict[str, object], receipt["rendering"])
    figures = cast(list[object], receipt["figures"])
    observed = {
        "tool_revision": receipt["tool_revision"],
        "alignment": receipt["alignment"],
        "scope_types": (type(scope), type(rendering), type(figures)),
        "external_only": scope["external_only"],
        "solver_imported": scope["solver_imported"],
        "figure_role": scope["figure_role"],
        "metric_population": scope["metric_population"],
        "png": rendering["png"],
        "particle_files": {path.name for path in (output / "particles").glob("*.svg")},
        "schema_version": receipt["schema_version"],
        "particle_svg_count": receipt["particle_svg_count"],
        "figure_count": len(figures),
    }
    assert observed == {
        "tool_revision": TOOL_REVISION,
        "alignment": {
            "status": "PASS",
            "particle_identity": "EXACT",
            "observed_state_key": ["particle_id", "time_s"],
            "metric_state_key": ["particle_id", "time_s", "active_on_both_sides"],
            "particles": 2,
            "first_particle_id": 1,
            "last_particle_id": 4,
            "candidate_status_column": "lifecycle",
            "reference_status_column": "current_status",
            "candidate_input_rows": 6,
            "reference_input_rows": 6,
            "candidate_missing_coordinate_rows": 0,
            "reference_missing_coordinate_rows": 0,
            "candidate_finite_rows": 6,
            "reference_finite_rows": 6,
            "common_observed_finite_rows": 6,
            "candidate_only_observed_finite_rows": 0,
            "reference_only_observed_finite_rows": 0,
            "candidate_active_finite_rows": 6,
            "reference_active_finite_rows": 6,
            "candidate_terminal_finite_rows": 0,
            "reference_terminal_finite_rows": 0,
            "candidate_unobserved_escaped_rows": 0,
            "reference_unobserved_escaped_rows": 0,
            "common_active_finite_rows": 6,
            "candidate_only_active_finite_rows": 0,
            "reference_only_active_finite_rows": 0,
            "candidate_observed_coordinate_rows_per_particle_min": 3,
            "candidate_observed_coordinate_rows_per_particle_max": 3,
            "reference_observed_coordinate_rows_per_particle_min": 3,
            "reference_observed_coordinate_rows_per_particle_max": 3,
            "candidate_observed_coordinate_time_s_min": 0.0,
            "candidate_observed_coordinate_time_s_max": 2.0e-5,
            "reference_observed_coordinate_time_s_min": 0.0,
            "reference_observed_coordinate_time_s_max": 2.0e-5,
            "candidate_last_observed_coordinate_time_s_min": 2.0e-5,
            "candidate_last_observed_coordinate_time_s_max": 2.0e-5,
            "reference_last_observed_coordinate_time_s_min": 2.0e-5,
            "reference_last_observed_coordinate_time_s_max": 2.0e-5,
            "common_active_time_s_min": 0.0,
            "common_active_time_s_max": 2.0e-5,
        },
        "scope_types": (dict, dict, list),
        "external_only": True,
        "solver_imported": False,
        "figure_role": "NON_GATING_DIAGNOSTIC",
        "metric_population": (
            "exact candidate/reference intersection of finite pre-terminal active "
            "particle/time rows"
        ),
        "png": "SKIPPED",
        "particle_files": {"particle_000001.svg", "particle_000004.svg"},
        "schema_version": 4,
        "particle_svg_count": 2,
        "figure_count": 3,
    }
    ET.parse(output / "overview_rz.svg")
    ET.parse(output / "particles" / "particle_000004.svg")
    overview = (output / "overview_rz.svg").read_text(encoding="utf-8")
    particle = (output / "particles" / "particle_000004.svg").read_text(encoding="utf-8")
    assert (
        "r (mm)" in overview,
        "z (mm)" in overview,
        "R-Z trajectory" in particle,
        (
            "candidate path points=6, reference path points=6, common active metric rows=6"
            in overview
        ),
        "Observed paths are drawn independently and are non-gating" in overview,
        "solver observed path points=3, COMSOL observed path points=3" in particle,
        "common finite active states=3" in particle,
        "aligned states=" not in overview,
        (output / "position_difference_heatmap.svg").exists(),
    ) == (True, True, True, True, True, True, True, True, False)

    persisted = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    expected_hash = hashlib.sha256(candidate.read_bytes()).hexdigest()
    assert (
        persisted["inputs"]["candidate"]["sha256"],
        "Particle 4" in (output / "index.html").read_text(encoding="utf-8"),
        "independently and is non-gating" in (output / "index.html").read_text(encoding="utf-8"),
    ) == (expected_hash, True, True)


def test_external_plot_rejects_unaligned_particle_ids(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.csv"
    reference = tmp_path / "reference.csv"
    _write_trajectory(candidate)
    _write_trajectory(reference, particle_ids=(1, 5))

    with pytest.raises(ValueError, match="particle IDs do not align"):
        read_matched_trajectories(candidate, reference)


def test_external_plot_metrics_exclude_finite_terminal_tails(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.csv"
    reference = tmp_path / "reference.csv"
    with candidate.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["particle_id", "time_s", "r_m", "z_m", "lifecycle"])
        writer.writerows(
            [
                [1, 0.0, 0.100001, 0.019999, "active"],
                [1, 1.0e-5, 0.110001, 0.020999, "active"],
                [1, 2.0e-5, 9.0, 9.0, "stuck"],
            ]
        )
    with reference.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["particle_id", "time_s", "r_m", "z_m", "current_status"])
        writer.writerows(
            [
                [1, 0.0, 0.1, 0.02, "active"],
                [1, 1.0e-5, 0.11, 0.021, "active"],
                [1, 2.0e-5, -9.0, -9.0, "held"],
            ]
        )

    output = tmp_path / "terminal_tail_figures"
    receipt = create_trajectory_figures(
        candidate,
        reference,
        output,
        label="finite terminal tails",
        render_png=False,
    )

    alignment = cast(dict[str, object], receipt["alignment"])
    metrics = cast(dict[str, object], receipt["metrics"])
    position = cast(dict[str, object], metrics["position_difference_m"])
    assert position["comparison_rows"] == 2
    assert float(cast(float, position["maximum"])) == pytest.approx(2.0**0.5 * 1.0e-6)
    assert (
        alignment["candidate_finite_rows"],
        alignment["reference_finite_rows"],
        alignment["common_observed_finite_rows"],
        alignment["candidate_active_finite_rows"],
        alignment["reference_active_finite_rows"],
        alignment["common_active_finite_rows"],
        alignment["candidate_terminal_finite_rows"],
        alignment["reference_terminal_finite_rows"],
    ) == (3, 3, 3, 2, 2, 2, 1, 1)
    overview = (output / "overview_rz.svg").read_text(encoding="utf-8")
    particle = (output / "particles" / "particle_000001.svg").read_text(encoding="utf-8")
    assert (
        "candidate path points=3, reference path points=3, common active metric rows=2" in overview
    )
    assert "common finite active states=2" in particle


@pytest.mark.parametrize(
    ("candidate_rows", "message"),
    [
        ([[1, 0.0, "NaN", "", "active"]], "active position must be observed"),
        (
            [
                [1, 0.0, 0.1, 0.02, "active"],
                [1, 1.0e-5, "NaN", "", "escaped"],
                [1, 2.0e-5, 0.2, 0.03, "active"],
            ],
            "lifecycle changes after terminal state",
        ),
    ],
)
def test_external_plot_rejects_invalid_missing_coordinate_lifecycle(
    tmp_path: Path,
    candidate_rows: list[list[object]],
    message: str,
) -> None:
    candidate = tmp_path / "candidate.csv"
    reference = tmp_path / "reference.csv"
    with candidate.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["particle_id", "time_s", "r_m", "z_m", "lifecycle"])
        writer.writerows(candidate_rows)
    _write_trajectory(reference, particle_ids=(1,), status_column="current_status")

    with pytest.raises(ValueError, match=message):
        read_matched_trajectories(candidate, reference)


def test_external_plot_preserves_ragged_and_dense_missing_escape_suffixes(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate.csv"
    reference = tmp_path / "reference.csv"
    candidate_header = ["particle_id", "time_s", "r_m", "z_m", "lifecycle"]
    reference_header = ["particle_id", "time_s", "r_m", "z_m", "current_status"]
    with candidate.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(candidate_header)
        for particle_id in (1, 4):
            last_frame = 3 if particle_id == 1 else 1
            for frame in range(last_frame + 1):
                writer.writerow(
                    [
                        particle_id,
                        frame * 1.0e-5,
                        0.1 + particle_id * 1.0e-3,
                        0.02 + frame * 1.0e-4,
                        "active",
                    ]
                )
    with reference.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(reference_header)
        for particle_id in (1, 4):
            for frame in range(4):
                if particle_id == 4 and frame >= 2:
                    writer.writerow([particle_id, frame * 1.0e-5, "NaN", "", "escaped"])
                else:
                    writer.writerow(
                        [
                            particle_id,
                            frame * 1.0e-5,
                            0.1 + particle_id * 1.0e-3,
                            0.02 + frame * 1.0e-4,
                            "active",
                        ]
                    )

    output = tmp_path / "ragged_dense_figures"
    receipt = create_trajectory_figures(
        candidate,
        reference,
        output,
        label="ragged candidate and dense missing reference",
        render_png=False,
    )

    alignment = cast(dict[str, object], receipt["alignment"])
    metrics = cast(dict[str, object], receipt["metrics"])
    position = cast(dict[str, object], metrics["position_difference_m"])
    assert alignment == {
        "status": "PASS",
        "particle_identity": "EXACT",
        "observed_state_key": ["particle_id", "time_s"],
        "metric_state_key": ["particle_id", "time_s", "active_on_both_sides"],
        "particles": 2,
        "first_particle_id": 1,
        "last_particle_id": 4,
        "candidate_status_column": "lifecycle",
        "reference_status_column": "current_status",
        "candidate_input_rows": 6,
        "reference_input_rows": 8,
        "candidate_missing_coordinate_rows": 0,
        "reference_missing_coordinate_rows": 2,
        "candidate_finite_rows": 6,
        "reference_finite_rows": 6,
        "common_observed_finite_rows": 6,
        "candidate_only_observed_finite_rows": 0,
        "reference_only_observed_finite_rows": 0,
        "candidate_active_finite_rows": 6,
        "reference_active_finite_rows": 6,
        "candidate_terminal_finite_rows": 0,
        "reference_terminal_finite_rows": 0,
        "candidate_unobserved_escaped_rows": 0,
        "reference_unobserved_escaped_rows": 2,
        "common_active_finite_rows": 6,
        "candidate_only_active_finite_rows": 0,
        "reference_only_active_finite_rows": 0,
        "candidate_observed_coordinate_rows_per_particle_min": 2,
        "candidate_observed_coordinate_rows_per_particle_max": 4,
        "reference_observed_coordinate_rows_per_particle_min": 2,
        "reference_observed_coordinate_rows_per_particle_max": 4,
        "candidate_observed_coordinate_time_s_min": 0.0,
        "candidate_observed_coordinate_time_s_max": 3.0000000000000004e-5,
        "reference_observed_coordinate_time_s_min": 0.0,
        "reference_observed_coordinate_time_s_max": 3.0000000000000004e-5,
        "candidate_last_observed_coordinate_time_s_min": 1.0e-5,
        "candidate_last_observed_coordinate_time_s_max": 3.0000000000000004e-5,
        "reference_last_observed_coordinate_time_s_min": 1.0e-5,
        "reference_last_observed_coordinate_time_s_max": 3.0000000000000004e-5,
        "common_active_time_s_min": 0.0,
        "common_active_time_s_max": 3.0000000000000004e-5,
    }
    assert position["comparison_rows"] == 6
    assert cast(dict[str, object], receipt["scope"])["missing_coordinate_policy"] == (
        "escaped_nan_unobserved_excluded_without_reconstruction"
    )
    assert receipt["particle_svg_count"] == 2
    assert not (output / "position_difference_heatmap.svg").exists()
    ET.parse(output / "overview_rz.svg")
    ET.parse(output / "particles" / "particle_000004.svg")
    overview = (output / "overview_rz.svg").read_text(encoding="utf-8")
    assert "escaped NaN positions are unobserved and never reconstructed" in overview
