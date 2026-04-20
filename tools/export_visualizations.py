from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from tools.export_boundary_diagnostics_visuals import export_boundary_diagnostics
from tools.export_mechanics_visuals import export_mechanics_visuals
from tools.export_result_graphs import export_result_graphs
from tools.export_trajectory_animation import export_trajectory_animations
from tools.visualization_common import (
    build_run_health_summary,
    ensure_visualization_dirs,
    list_files,
    write_run_summary,
    write_visualization_index,
)


_ALLOWED_MODULES = ("graphs", "animations", "mechanics", "boundary")
_DEFAULT_MODULES = ("graphs",)
_COMPACT_SUMMARY_FILES = (
    "plasma_background_summary.csv",
    "charge_model_summary.csv",
)


def _parse_modules(raw: str | Iterable[str]) -> list[str]:
    if isinstance(raw, str):
        parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
    else:
        parts = [str(p).strip().lower() for p in raw if str(p).strip()]
    if not parts or "standard" in parts or "default" in parts:
        return list(_DEFAULT_MODULES)
    if "all" in parts:
        return list(_ALLOWED_MODULES)
    out: list[str] = []
    for name in parts:
        if name not in _ALLOWED_MODULES:
            raise ValueError(f"Unsupported module: {name}. Supported: {', '.join(_ALLOWED_MODULES)}")
        if name not in out:
            out.append(name)
    return out


def export_visualizations(
    output_dir: Path,
    *,
    case_dir: Path | None = None,
    modules: Iterable[str] = _DEFAULT_MODULES,
    clean: bool = False,
    sample_trajectories: int = 300,
    animation_fps: int = 6,
    animation_sample_count: int = 450,
    animation_interpolate_factor: int = 1,
    overlay_wall_events: bool = False,
    interpolate_wall_event_positions: bool = False,
    mechanics_sample_trajectories: int = 500,
    mechanics_quiver_stride: int = 12,
    boundary_normal_band_m: float = 2.5e-3,
    boundary_quiver_stride: int = 10,
) -> Path:
    output_dir = Path(output_dir).resolve()
    selected = _parse_modules(list(modules))
    dirs = ensure_visualization_dirs(output_dir, clean=bool(clean))

    if ("mechanics" in selected or "boundary" in selected) and case_dir is None:
        raise ValueError("case_dir is required when modules include mechanics or boundary")

    module_records: dict[str, dict[str, object]] = {}
    if "graphs" in selected:
        graph_dir = export_result_graphs(output_dir=output_dir, case_dir=case_dir, sample_trajectories=int(sample_trajectories))
        module_records["graphs"] = {
            "dir": str(graph_dir.resolve()),
            "files": list_files(graph_dir, (".png", ".csv", ".json")),
        }
    if "animations" in selected:
        anim_dir = export_trajectory_animations(
            output_dir=output_dir,
            case_dir=case_dir,
            fps=int(animation_fps),
            sample_count=int(animation_sample_count),
            interpolate_factor=int(animation_interpolate_factor),
            overlay_wall_events=bool(overlay_wall_events),
            interpolate_wall_event_positions=bool(interpolate_wall_event_positions),
        )
        module_records["animations"] = {
            "dir": str(anim_dir.resolve()),
            "files": list_files(anim_dir, (".gif", ".json")),
        }
    if "mechanics" in selected:
        mechanics_dir = export_mechanics_visuals(
            case_dir=Path(case_dir).resolve(),
            output_dir=output_dir,
            sample_trajectories=int(mechanics_sample_trajectories),
            quiver_stride=max(1, int(mechanics_quiver_stride)),
        )
        module_records["mechanics"] = {
            "dir": str(mechanics_dir.resolve()),
            "files": list_files(mechanics_dir, (".png", ".csv", ".json")),
        }
    if "boundary" in selected:
        boundary_dir = export_boundary_diagnostics(
            case_dir=Path(case_dir).resolve(),
            output_dir=output_dir,
            normal_band_m=float(boundary_normal_band_m),
            quiver_stride=max(1, int(boundary_quiver_stride)),
        )
        module_records["boundary"] = {
            "dir": str(boundary_dir.resolve()),
            "files": list_files(boundary_dir, (".png", ".json")),
        }

    summary_files = {
        name: str((output_dir / name).resolve())
        for name in _COMPACT_SUMMARY_FILES
        if (output_dir / name).exists()
    }
    payload = {
        "output_dir": str(output_dir),
        "visualizations_root": str(dirs["root"].resolve()),
        "clean": bool(clean),
        "health_summary": build_run_health_summary(output_dir),
        "summary_files": summary_files,
        "modules": module_records,
    }
    summary_path = write_run_summary(output_dir, payload)
    payload["run_summary_md"] = str(summary_path.resolve())
    return write_visualization_index(output_dir, payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export visualizations into output_dir/visualizations with unified layout.")
    parser.add_argument("--output-dir", required=True, help="Simulation output directory")
    parser.add_argument("--case-dir", default="", help="Case directory for geometry-based modules")
    parser.add_argument(
        "--modules",
        default="standard",
        help="Comma-separated module list: graphs,animations,mechanics,boundary; standard writes graphs only, all includes GIFs",
    )
    parser.add_argument("--clean", action="store_true", help="Remove legacy output dirs (graphs/animations/visuals) under output-dir")
    parser.add_argument("--sample-trajectories", type=int, default=300, help="Sample trajectories for graphs")
    parser.add_argument("--animation-fps", type=int, default=6, help="Animation FPS")
    parser.add_argument("--animation-sample-count", type=int, default=450, help="Sample particles for trail animation")
    parser.add_argument("--animation-interpolate-factor", type=int, default=1, help="Frame interpolation factor")
    parser.add_argument("--overlay-wall-events", action="store_true", help="Overlay wall events on sampled trail GIF")
    parser.add_argument(
        "--interpolate-wall-event-positions",
        action="store_true",
        help="Linearly interpolate wall-event positions",
    )
    parser.add_argument("--mechanics-sample-trajectories", type=int, default=500, help="Sample trajectories for mechanics overlay")
    parser.add_argument("--mechanics-quiver-stride", type=int, default=12, help="Quiver stride for mechanics")
    parser.add_argument("--boundary-normal-band-m", type=float, default=2.5e-3, help="Near-wall normal band width [m]")
    parser.add_argument("--boundary-quiver-stride", type=int, default=10, help="Quiver stride for boundary diagnostics")
    args = parser.parse_args()

    modules = _parse_modules(args.modules)
    index_path = export_visualizations(
        output_dir=Path(args.output_dir),
        case_dir=Path(args.case_dir).resolve() if args.case_dir else None,
        modules=modules,
        clean=bool(args.clean),
        sample_trajectories=int(args.sample_trajectories),
        animation_fps=int(args.animation_fps),
        animation_sample_count=int(args.animation_sample_count),
        animation_interpolate_factor=int(args.animation_interpolate_factor),
        overlay_wall_events=bool(args.overlay_wall_events),
        interpolate_wall_event_positions=bool(args.interpolate_wall_event_positions),
        mechanics_sample_trajectories=int(args.mechanics_sample_trajectories),
        mechanics_quiver_stride=max(1, int(args.mechanics_quiver_stride)),
        boundary_normal_band_m=float(args.boundary_normal_band_m),
        boundary_quiver_stride=max(1, int(args.boundary_quiver_stride)),
    )
    print(f"wrote visualization index: {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
