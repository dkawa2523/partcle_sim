from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools import (
    export_boundary_diagnostics_visuals,
    export_mechanics_visuals,
    export_result_graphs,
    export_trajectory_animation,
    export_visualizations,
)


def _write_boundary_case(case_dir: Path) -> None:
    generated = case_dir / "generated"
    generated.mkdir(parents=True)
    axis = np.asarray([0.0, 1.0], dtype=np.float64)
    shape = (2, 2)
    vertices = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=np.float64,
    )
    edges = np.asarray(
        [
            [vertices[0], vertices[1]],
            [vertices[1], vertices[2]],
            [vertices[2], vertices[3]],
            [vertices[3], vertices[0]],
        ],
        dtype=np.float64,
    )
    np.savez(
        generated / "comsol_geometry_2d.npz",
        axis_0=axis,
        axis_1=axis,
        sdf=np.asarray([[-0.1, 0.1], [-0.1, 0.1]], dtype=np.float64),
        normal_0=np.ones(shape, dtype=np.float64),
        normal_1=np.zeros(shape, dtype=np.float64),
        valid_mask=np.asarray([[True, True], [False, True]], dtype=bool),
        boundary_edges=edges,
        boundary_edge_part_ids=np.full(4, 7, dtype=np.int32),
        mesh_vertices=vertices,
        mesh_quads=np.asarray([[0, 1, 2, 3]], dtype=np.int32),
        mesh_quad_part_ids=np.asarray([7], dtype=np.int32),
    )
    np.savez(
        generated / "comsol_field_2d.npz",
        ux=np.ones(shape, dtype=np.float64),
        uy=np.zeros(shape, dtype=np.float64),
        valid_mask=np.ones(shape, dtype=bool),
    )


def _stub_exporter(calls: list[str], name: str, directory: Path, suffix: str):
    def export(*args, **kwargs) -> Path:
        del args, kwargs
        calls.append(name)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{name}{suffix}").write_text("artifact", encoding="utf-8")
        return directory

    return export


@pytest.mark.parametrize(
    ("selector", "expected"),
    [
        ("", ["graphs"]),
        ("DEFAULT", ["graphs"]),
        ("all", ["graphs", "animations", "mechanics", "boundary"]),
    ],
)
def test_visualization_module_selectors(selector: str, expected: list[str]) -> None:
    assert export_visualizations._parse_modules(selector) == expected


def test_visualization_module_selector_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="Unsupported module: obsolete"):
        export_visualizations._parse_modules("graphs,obsolete")


def test_visualization_route_keeps_canonical_module_and_artifact_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    root = tmp_path / "visualizations"
    monkeypatch.setattr(
        export_result_graphs,
        "export_result_graphs",
        _stub_exporter(calls, "graphs", root / "graphs", ".png"),
    )
    monkeypatch.setattr(
        export_trajectory_animation,
        "export_trajectory_animations",
        _stub_exporter(calls, "animations", root / "animations", ".gif"),
    )
    monkeypatch.setattr(
        export_mechanics_visuals,
        "export_mechanics_visuals",
        _stub_exporter(calls, "mechanics", root / "mechanics", ".csv"),
    )
    monkeypatch.setattr(
        export_boundary_diagnostics_visuals,
        "export_boundary_diagnostics",
        _stub_exporter(
            calls,
            "boundary",
            root / "boundary_diagnostics",
            ".json",
        ),
    )

    index_path = export_visualizations.export_visualizations(
        tmp_path,
        case_dir=tmp_path / "case",
        modules=("boundary", "graphs", "mechanics", "animations", "graphs"),
    )

    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert calls == ["graphs", "animations", "mechanics", "boundary"]
    assert list(payload["modules"]) == calls
    assert payload["modules"]["graphs"]["files"] == ["graphs.png"]
    assert payload["modules"]["animations"]["files"] == ["animations.gif"]
    assert payload["modules"]["mechanics"]["files"] == ["mechanics.csv"]
    assert payload["modules"]["boundary"]["files"] == ["boundary.json"]


@pytest.mark.parametrize("modules", [("mechanics",), ("boundary",)])
def test_geometry_visualization_routes_require_a_case(
    tmp_path: Path, modules: tuple[str, ...]
) -> None:
    with pytest.raises(ValueError, match="case_dir is required"):
        export_visualizations.export_visualizations(tmp_path, modules=modules)


def test_visualization_route_does_not_hide_programming_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args, **kwargs) -> Path:
        del args, kwargs
        raise RuntimeError("programming error")

    monkeypatch.setattr(
        export_trajectory_animation, "export_trajectory_animations", fail
    )

    with pytest.raises(RuntimeError, match="programming error"):
        export_visualizations.export_visualizations(
            tmp_path,
            modules=("animations",),
            best_effort_animations=True,
        )


def test_visualization_route_records_only_recoverable_animation_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args, **kwargs) -> Path:
        del args, kwargs
        raise export_trajectory_animation.AnimationInputError("missing trajectory")

    monkeypatch.setattr(
        export_trajectory_animation, "export_trajectory_animations", fail
    )

    index_path = export_visualizations.export_visualizations(
        tmp_path,
        modules=("animations",),
        best_effort_animations=True,
    )

    module = json.loads(index_path.read_text(encoding="utf-8"))["modules"]["animations"]
    assert list(module) == ["status", "dir", "files", "error", "action"]
    assert module["status"] == "failed"
    assert module["files"] == []
    assert module["error"] == "missing trajectory"

    with pytest.raises(
        export_trajectory_animation.AnimationInputError,
        match="missing trajectory",
    ):
        export_visualizations.export_visualizations(
            tmp_path,
            modules=("animations",),
            best_effort_animations=False,
        )


def test_visualize_cli_forwards_public_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}
    index_path = tmp_path / "visualization_index.json"

    def export(output_dir: Path, **options: object) -> Path:
        captured.update(output_dir=output_dir, **options)
        return index_path

    monkeypatch.setattr(export_visualizations, "export_visualizations", export)

    assert (
        export_visualizations.main(
            [
                "--output-dir",
                str(tmp_path),
                "--case-dir",
                str(tmp_path / "case"),
                "--modules",
                "boundary,graphs",
                "--clean",
                "--strict-visualizations",
                "--skip-all-particles-animation",
            ]
        )
        == 0
    )
    assert captured["modules"] == "boundary,graphs"
    assert captured["clean"] is True
    assert captured["best_effort_animations"] is False
    assert captured["animation_write_all_particles"] is False
    assert capsys.readouterr().out == f"wrote visualization index: {index_path}\n"


def test_boundary_diagnostics_export_preserves_public_artifacts(
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "case"
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    _write_boundary_case(case_dir)
    pd.DataFrame(
        {
            "x_m": [0.25, 0.75],
            "y_m": [0.25, 0.75],
            "final_state": ["active_free_flight", "invalid_mask_stopped"],
        }
    ).to_csv(output_dir / "final_particles.csv", index=False)

    artifact_dir = export_boundary_diagnostics_visuals.export_boundary_diagnostics(
        case_dir,
        output_dir,
        normal_band_m=1.0,
        quiver_stride=1,
    )

    expected_pngs = [
        "01_recognized_boundary_geometry.png",
        "02_recognized_domain_mask.png",
        "03_signed_distance_field.png",
        "04_boundary_normals_near_wall.png",
        "05_flow_speed_vectors_over_geometry.png",
        "06_mixed_stencil_hotspots.png",
        "07_hard_invalid_stop_hotspots.png",
    ]
    report = json.loads(
        (artifact_dir / "boundary_diagnostics_report.json").read_text(encoding="utf-8")
    )
    assert report["boundary_part_ids"] == [7]
    assert report["domain_grid_shape"] == [2, 2]
    assert report["invalid_mask_stopped_point_count"] == 1
    assert report["files"] == [
        *expected_pngs,
        "domain_part_medium_summary.csv",
    ]
    assert all((artifact_dir / name).is_file() for name in expected_pngs)


def test_boundary_diagnostics_reports_missing_input_at_its_owner(
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "case"
    output_dir = tmp_path / "run"

    with pytest.raises(FileNotFoundError, match="Geometry npz not found"):
        export_boundary_diagnostics_visuals.export_boundary_diagnostics(
            case_dir, output_dir
        )

    _write_boundary_case(case_dir)
    (case_dir / "generated" / "comsol_field_2d.npz").unlink()
    with pytest.raises(FileNotFoundError, match="Field npz not found"):
        export_boundary_diagnostics_visuals.export_boundary_diagnostics(
            case_dir, output_dir
        )
