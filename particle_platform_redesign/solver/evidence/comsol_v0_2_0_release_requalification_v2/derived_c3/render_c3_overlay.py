"""Task-specific rendering of saved C3 comparison artifacts; no simulation."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from tools.vv.comsol.plot_matched_trajectories import (
    PlotBox,
    SvgCanvas,
    read_matched_trajectories,
)

from chamber_particles import load_case, open_result


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    out = Path(__file__).resolve().parent
    root = out.parent
    candidate = root / "c3_prepared/candidate/dt_0p625us"
    native = root / "saved_native/c3_dt_1p25us"
    case = load_case(candidate / "case.yaml")
    result = open_result(candidate / "result")
    assert result.complete
    assert case.case_file_hash == result.manifest["case_file_hash"]
    assert case.content_hash == result.manifest["data_content_hash"]
    assert result.manifest["data_coordinate_system"] == "axisymmetric_rz"
    paired = read_matched_trajectories(
        candidate / "trajectory.csv", native / "trajectory_reference.csv"
    )
    ids = paired.candidate.particle_ids
    selected = ids[np.linspace(0, len(ids) - 1, 12, dtype=int)]
    geometry = case.data.geometry
    lower = geometry.nodes_m.min(axis=0)
    upper = geometry.nodes_m.max(axis=0)
    span = upper - lower
    pad = span.max() * 0.025
    lower -= pad
    upper += pad
    span = upper - lower
    scale = min(1020 / span[0], 420 / span[1])
    width, height = span * scale
    origin_x = 60 + (1020 - width) / 2
    origin_y = 94 + (420 - height) / 2
    canvas = SvgCanvas(1180, 710, "C3 deterministic R-Z comparison with canonical geometry")
    canvas.text(40, 36, "C3: v46 candidate and saved COMSOL reference", css_class="title")
    canvas.text(
        40,
        63,
        "Candidate 0.625 us / saved COMSOL 1.25 us; 12 fixed IDs of 287; equal geometric aspect",
        css_class="subtitle",
    )

    def xy(point: np.ndarray) -> tuple[float, float]:
        return (
            float(origin_x + (point[0] - lower[0]) * scale),
            float(origin_y + height - (point[1] - lower[1]) * scale),
        )

    box = PlotBox(float(origin_x), float(origin_y), float(width), float(height))
    canvas.rect(box, stroke="#94A3B8")
    for dim in range(2):
        for value in np.linspace(lower[dim], upper[dim], 6):
            point = lower.copy()
            point[dim] = value
            x, y = xy(point)
            if dim == 0:
                canvas.line(x, origin_y, x, origin_y + height, stroke="#E2E8F0")
                canvas.text(x, origin_y + height + 21, f"{value * 1000:.1f}", anchor="middle")
            else:
                canvas.line(origin_x, y, origin_x + width, y, stroke="#E2E8F0")
                canvas.text(origin_x - 9, y + 4, f"{value * 1000:.1f}", anchor="end")
    canvas.text(origin_x + width / 2, origin_y + height + 44, "r [mm]", anchor="middle")
    canvas.text(origin_x - 48, origin_y + height / 2, "z [mm]", rotate=-90, anchor="middle")
    colors = ("#047857", "#7C3AED", "#64748B", "#475569", "#BE123C", "#0891B2")
    for edge, group in zip(geometry.boundary.line2, geometry.boundary.group_id, strict=True):
        a, b = (xy(p) for p in geometry.nodes_m[edge])
        canvas.line(*a, *b, stroke=colors[int(group)], width=2.1, opacity=0.8)
    axis_a, axis_b = xy(np.array([0.0, lower[1]])), xy(np.array([0.0, upper[1]]))
    canvas.line(*axis_a, *axis_b, stroke="#94A3B8", dash="3,4")
    events = result.read_boundary_events()
    selected_indices = [int(np.searchsorted(ids, value)) for value in selected]
    for index in selected_indices:
        particle_id = int(ids[index])
        for table, color, dash in (
            (paired.candidate, "#1D4ED8", None),
            (paired.reference, "#D97706", "5,3"),
        ):
            points = np.column_stack((table.r_by_particle[index], table.z_by_particle[index]))
            canvas.polyline([xy(p) for p in points], stroke=color, width=1.5, dash=dash)
        start = np.array(
            [paired.candidate.r_by_particle[index][0], paired.candidate.z_by_particle[index][0]]
        )
        canvas.circle(*xy(start), 3.5, fill="#16A34A", stroke="#166534")
        for row in np.flatnonzero(events.particle_id == particle_id):
            x, y = xy(events.position_m[row])
            canvas.rect(PlotBox(x - 3, y - 3, 6, 6), fill="#DC2626", stroke="#991B1B")
    legend_y = 576
    for index, label in enumerate(geometry.group_names):
        x = 40 + (index % 3) * 360
        y = legend_y + (index // 3) * 22
        canvas.line(x, y - 4, x + 28, y - 4, stroke=colors[index], width=2.1)
        canvas.text(x + 36, y, label, css_class="small")
    canvas.text(
        40,
        628,
        "Blue solid: v46 candidate; orange dashed: saved COMSOL; green circle: start; red square: candidate boundary event.",
        css_class="small",
    )
    canvas.text(
        40,
        650,
        "Dotted r=0 guide is the symmetry seam. Saved stuck tails remain at their observed positions; no trajectory is recomputed.",
        css_class="small",
    )
    canvas.text(
        40,
        674,
        "Figure is illustrative. Formal numeric gates use all 287 particles and a registered empirical fine-pair allowance.",
        css_class="small",
    )
    figure = out / "c3_geometry_overlay.svg"
    canvas.write(figure)
    report = {
        "kind": "saved_c3_version_requalification_geometry_overlay_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_sha256": {
            str(p.relative_to(root)): sha(p)
            for p in (
                candidate / "case.yaml",
                case.data_path,
                candidate / "trajectory.csv",
                native / "trajectory_reference.csv",
                candidate / "events.csv",
                root / "c3_version_requalification_evaluation.json",
                root / "registration/execution_preregistration.json",
            )
        },
        "source_result_directory": str(result.path.relative_to(root)),
        "source_result_files_sha256": {
            str(path.relative_to(root)): sha(path)
            for path in sorted(result.path.rglob("*"))
            if path.is_file()
        },
        "renderer_sha256": sha(Path(__file__)),
        "case_and_result_hash_match": True,
        "data_content_hash": case.content_hash,
        "coordinates": "R-Z meridional; stored SI metres; display millimetres",
        "geometry_boundary_segments": len(geometry.boundary.line2),
        "axis_limits_m": [lower.tolist(), upper.tolist()],
        "equal_aspect": True,
        "sampling": "12 evenly spaced indices in sorted stable particle IDs; illustrative only",
        "selected_particle_ids": selected.tolist(),
        "full_population": len(ids),
        "time_range_s": [0.0, 0.03],
        "trajectory_recomputation": False,
        "new_native_solve": False,
        "common_active_rows_in_full_tables": len(paired.common_active_row_keys),
        "figure_sha256": sha(figure),
        "limits": [
            "Meridional 2D, not reconstructed 3D",
            "No boundary or equivalence claim from visual overlap",
            "Numeric evaluation remains authoritative",
        ],
    }
    (out / "visualization_receipt.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"figure": str(figure), "selected_ids": selected.tolist()}))


if __name__ == "__main__":
    main()
