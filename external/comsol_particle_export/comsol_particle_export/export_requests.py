from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


WALL_EVENT_PROBE_EXPRESSIONS = (
    "fpt.bnd",
    "fpt.bid",
    "fpt.boundary",
    "fpt.wall",
    "fpt.wallid",
    "fpt.event",
    "fpt.status",
    "fpt.pstatus",
    "fpt.st",
    "fpt.fs",
    "particlestatus",
    "fpt.particlestatus",
    "fpt.freeze",
    "fpt.stick",
    "fpt.nx",
    "fpt.ny",
)

WALL_STATUS_RECOMPUTE_EXPRESSIONS = (
    "fpt.st",
    "fpt.fs",
    "fpt.bnd",
    "fpt.bid",
    "fpt.wallid",
    "fpt.nx",
    "fpt.ny",
    "x",
    "y",
    "fpt.vx",
    "fpt.vy",
)

RELEASE_PROPERTY_PROBE_EXPRESSIONS = (
    "fpt.dp",
    "fpt.rp",
    "fpt.rhop",
    "fpt.mp",
    "fpt.qp",
    "fpt.source",
    "fpt.relid",
    "fpt.inl",
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sanitize(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_]+", "_", str(value).strip())
    return text.strip("_") or "expr"


def _first_column(frame: pd.DataFrame, names: Sequence[str]) -> str | None:
    lower = {str(col).strip().lower(): str(col) for col in frame.columns}
    for name in names:
        found = lower.get(str(name).strip().lower())
        if found is not None:
            return found
    return None


def _required_coordinate_scale(field_manifest: Mapping[str, Any]) -> float:
    if "coordinate_scale_m_per_model_unit" not in field_manifest:
        raise ValueError(
            "coordinate_scale_m_per_model_unit is required for COMSOL parity export requests; "
            "implicit 1.0 scale is not allowed"
        )
    try:
        scale = float(field_manifest.get("coordinate_scale_m_per_model_unit"))
    except (TypeError, ValueError):
        scale = float("nan")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("coordinate_scale_m_per_model_unit must be a positive finite value")
    return scale


def _trajectory_times(trajectory_csv: Path | None, trajectory_report: Mapping[str, Any]) -> list[float]:
    if trajectory_csv is not None and trajectory_csv.exists():
        columns = pd.read_csv(trajectory_csv, nrows=0).columns
        dummy = pd.DataFrame(columns=columns)
        time_col = _first_column(dummy, ("time_s", "time", "t"))
        if time_col is not None:
            values = pd.read_csv(trajectory_csv, usecols=[time_col])[time_col]
            arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
            finite = np.unique(arr[np.isfinite(arr)])
            if finite.size:
                return [float(v) for v in finite]
    count = int(trajectory_report.get("trajectory_time_count", 0) or 0)
    t_min = trajectory_report.get("time_min_s")
    t_max = trajectory_report.get("time_max_s")
    if count > 1 and t_min is not None and t_max is not None:
        return [float(v) for v in np.linspace(float(t_min), float(t_max), count)]
    if count == 1 and t_min is not None:
        return [float(t_min)]
    return []


def _trajectory_export_ready(trajectory_csv: Path | None) -> bool:
    if trajectory_csv is None or not trajectory_csv.exists():
        return False
    try:
        columns = pd.read_csv(trajectory_csv, nrows=0).columns
    except Exception:
        return False
    if len(columns) == 0:
        return False
    dummy = pd.DataFrame(columns=columns)
    required = (
        _first_column(dummy, ("particle_id", "id", "particle")),
        _first_column(dummy, ("time_s", "time", "t")),
        _first_column(dummy, ("x", "x_m", "qx")),
        _first_column(dummy, ("y", "y_m", "qy")),
        _first_column(dummy, ("v_x", "vx", "u", "fpt.vx")),
        _first_column(dummy, ("v_y", "vy", "v", "fpt.vy")),
    )
    return all(column is not None for column in required)


def _remove_stale_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _base_data_export_config(
    *,
    case_name: str,
    field_manifest: Mapping[str, Any],
    particle_manifest: Mapping[str, Any],
    data_export_filename: str,
    expressions: Sequence[str],
) -> dict[str, Any]:
    scale = _required_coordinate_scale(field_manifest)
    return {
        "case_name": case_name,
        "mode": "inventory",
        "export_mesh": False,
        "export_fields": False,
        "dataset": str(field_manifest.get("dataset", "dset1")),
        "mesh_tag": str(field_manifest.get("mesh_tag", "mesh1")),
        "spatial_dim": int(field_manifest.get("spatial_dim", 2) or 2),
        "axis_names": list(field_manifest.get("axis_names", ["x", "y"]) or ["x", "y"]),
        "coordinate_model_unit": str(field_manifest.get("coordinate_model_unit", "m")),
        "coordinate_scale_m_per_model_unit": scale,
        "export_data_table": True,
        "export_data_table_required": False,
        "data_export_dataset": str(particle_manifest.get("data_export_dataset", "part1")),
        "data_export_filename": str(data_export_filename),
        "data_export_innerinput": "all",
        "data_export_expr": [str(expr) for expr in expressions],
        "required": [],
    }


def _mesh_time_config(
    *,
    case_name: str,
    field_manifest: Mapping[str, Any],
    time_values: Sequence[float],
) -> dict[str, Any]:
    scale = _required_coordinate_scale(field_manifest)
    expression_mapping = field_manifest.get("expression_mapping", {})
    expressions = dict(expression_mapping) if isinstance(expression_mapping, Mapping) else {}
    required = [name for name in ("ux", "uy", "mu") if name in expressions] or list(expressions.keys())
    config: dict[str, Any] = {
        "case_name": case_name,
        "mode": "fields",
        "export_mesh": True,
        "export_fields": True,
        "export_grid_field_samples": False,
        "export_mesh_field_samples": True,
        "mesh_field_samples_filename": "mesh_field_samples_time_resolved.csv",
        "dataset": str(field_manifest.get("dataset", "dset1")),
        "mesh_tag": str(field_manifest.get("mesh_tag", "mesh1")),
        "spatial_dim": int(field_manifest.get("spatial_dim", 2) or 2),
        "axis_names": list(field_manifest.get("axis_names", ["x", "y"]) or ["x", "y"]),
        "coordinate_model_unit": str(field_manifest.get("coordinate_model_unit", "m")),
        "coordinate_scale_m_per_model_unit": scale,
        "time_values": [float(v) for v in time_values],
        "solnums": [],
        "required": required,
    }
    for name, expr in expressions.items():
        config[str(name)] = [str(expr)]
    return config


def write_reextract_request_bundle(
    *,
    case_name: str,
    field_manifest: Mapping[str, Any],
    particle_manifest: Mapping[str, Any],
    trajectory_report: Mapping[str, Any],
    out_dir: str | Path,
    trajectory_csv: str | Path | None = None,
    needs_time_resolved_field: bool = False,
    needs_wall_events: bool = False,
    needs_release_properties: bool = False,
) -> dict[str, Any]:
    """Write COMSOL batch config requests for missing parity evidence.

    The files are intentionally plain exporter configs. They do not add COMSOL
    dependencies to the solver core, and unknown wall/property expressions are
    emitted as one-expression probes so a failed probe does not invalidate the
    whole extraction pass.
    """

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    requests: list[dict[str, Any]] = []
    trajectory_path = Path(trajectory_csv) if trajectory_csv is not None else None
    times = _trajectory_times(trajectory_path, trajectory_report)
    needs_trajectory_baseline = not _trajectory_export_ready(trajectory_path)

    if needs_time_resolved_field:
        config = _mesh_time_config(case_name=case_name, field_manifest=field_manifest, time_values=times)
        path = root / "mesh_field_time_resolved_config.json"
        _write_json(path, config)
        requests.append(
            {
                "kind": "mesh_field_time_resolved",
                "name": "mesh_field_time_resolved",
                "config": str(path),
                "reason": "non-particle physics contains time-dependent expressions but field export has one context",
                "time_count": int(len(times)),
                "output_filename": "mesh_field_samples_time_resolved.csv",
            }
        )
    else:
        _remove_stale_path(root / "mesh_field_time_resolved_config.json")

    if needs_trajectory_baseline:
        trajectory_config = _base_data_export_config(
            case_name=case_name,
            field_manifest=field_manifest,
            particle_manifest=particle_manifest,
            data_export_filename="comsol_particle_xy_velocity_probe.csv",
            expressions=("x", "y", "fpt.vx", "fpt.vy"),
        )
        trajectory_config_path = root / "particle_trajectory_xy_velocity_config.json"
        _write_json(trajectory_config_path, trajectory_config)
        requests.append(
            {
                "kind": "particle_trajectory_xy_velocity",
                "name": "particle_trajectory_xy_velocity",
                "config": str(trajectory_config_path),
                "reason": "canonical particle trajectory/release extraction baseline",
                "output_filename": "comsol_particle_xy_velocity_probe.csv",
            }
        )
    else:
        _remove_stale_path(root / "particle_trajectory_xy_velocity_config.json")

    wall_probe_paths = []
    if needs_wall_events:
        recompute_config = _base_data_export_config(
            case_name=case_name,
            field_manifest=field_manifest,
            particle_manifest=particle_manifest,
            data_export_filename="comsol_wall_status_recomputed.csv",
            expressions=WALL_STATUS_RECOMPUTE_EXPRESSIONS,
        )
        recompute_config.update(
            {
                "run_study": True,
                "study_tag": str(particle_manifest.get("study", field_manifest.get("study", "std1")) or "std1"),
                "enable_particle_status_data": True,
                "enable_wall_extra_steps": True,
                "enable_particle_release_statistics": True,
                "require_runtime_option_application": True,
                "data_export_innerinput": "all",
            }
        )
        recompute_path = root / "wall_status_recompute_config.json"
        _write_json(recompute_path, recompute_config)
        requests.append(
            {
                "kind": "wall_status_recompute",
                "name": "wall_status_recompute",
                "config": str(recompute_path),
                "reason": "rerun COMSOL with particle status data and extra wall-interaction time steps before exporting wall/status variables",
                "output_filename": "comsol_wall_status_recomputed.csv",
            }
        )
        probe_dir = root / "wall_event_expression_probes"
        for expr in WALL_EVENT_PROBE_EXPRESSIONS:
            config = _base_data_export_config(
                case_name=case_name,
                field_manifest=field_manifest,
                particle_manifest=particle_manifest,
                data_export_filename=f"probe_{_sanitize(expr)}.csv",
                expressions=(expr,),
            )
            path = probe_dir / f"probe_{_sanitize(expr)}.json"
            _write_json(path, config)
            wall_probe_paths.append(str(path))
            requests.append(
                {
                    "kind": "wall_event_expression_probe",
                    "name": f"wall_event_probe_{_sanitize(expr)}",
                    "config": str(path),
                    "reason": "single-expression wall-event probe",
                    "candidate_expression": str(expr),
                    "output_filename": f"probe_{_sanitize(expr)}.csv",
                }
            )
        probe_summary = root / "wall_event_expression_probes.json"
        _write_json(
            probe_summary,
            {
                "kind": "wall_event_expression_probe_group",
                "configs": wall_probe_paths,
                "reason": "COMSOL wall-hit/outcome variables are case/version dependent; probe one expression per run",
                "candidate_expressions": list(WALL_EVENT_PROBE_EXPRESSIONS),
            },
        )
        requests.append(
            {
                "kind": "probe_group_summary",
                "name": "wall_event_expression_probes",
                "config": str(probe_summary),
                "reason": "index of one-expression wall-event probes",
                "member_count": int(len(wall_probe_paths)),
            }
        )
    else:
        _remove_stale_path(root / "wall_status_recompute_config.json")
        _remove_stale_path(root / "wall_event_expression_probes.json")
        _remove_stale_path(root / "wall_event_expression_probes")

    property_probe_paths = []
    if needs_release_properties:
        probe_dir = root / "release_property_expression_probes"
        for expr in RELEASE_PROPERTY_PROBE_EXPRESSIONS:
            config = _base_data_export_config(
                case_name=case_name,
                field_manifest=field_manifest,
                particle_manifest=particle_manifest,
                data_export_filename=f"probe_{_sanitize(expr)}.csv",
                expressions=(expr,),
            )
            path = probe_dir / f"probe_{_sanitize(expr)}.json"
            _write_json(path, config)
            property_probe_paths.append(str(path))
            requests.append(
                {
                    "kind": "release_property_expression_probe",
                    "name": f"release_property_probe_{_sanitize(expr)}",
                    "config": str(path),
                    "reason": "single-expression release property/source probe",
                    "candidate_expression": str(expr),
                    "output_filename": f"probe_{_sanitize(expr)}.csv",
                }
            )
        probe_summary = root / "release_property_expression_probes.json"
        _write_json(
            probe_summary,
            {
                "kind": "release_property_expression_probe_group",
                "configs": property_probe_paths,
                "reason": "canonical release table is missing row-level source/property columns",
                "candidate_expressions": list(RELEASE_PROPERTY_PROBE_EXPRESSIONS),
            },
        )
        requests.append(
            {
                "kind": "probe_group_summary",
                "name": "release_property_expression_probes",
                "config": str(probe_summary),
                "reason": "index of one-expression release property/source probes",
                "member_count": int(len(property_probe_paths)),
            }
        )
    else:
        _remove_stale_path(root / "release_property_expression_probes.json")
        _remove_stale_path(root / "release_property_expression_probes")

    runnable_requests = [
        request
        for request in requests
        if str(request.get("kind")) != "probe_group_summary" and str(request.get("config", "")).strip()
    ]
    default_mph = str(field_manifest.get("mph_path", particle_manifest.get("mph_path", "data/micromixer_particle_tracing.mph")))
    script = _reextract_script(default_mph=default_mph, requests=runnable_requests)
    script_path = root / "run_reextract_requests.ps1"
    _write_text(script_path, script)
    summary = {
        "case_name": str(case_name),
        "request_count": int(len(requests)),
        "runnable_config_count": int(len(runnable_requests)),
        "requests": requests,
        "run_script": str(script_path),
        "run_hint": (
            "Run run_reextract_requests.ps1 from the repository root with "
            "-ComsolExe and optionally -Mph/-OutRoot."
        ),
    }
    _write_json(root / "reextract_request_summary.json", summary)
    return summary


def _ps_single_quoted(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _reextract_script(*, default_mph: str, requests: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "param(",
        "  [Parameter(Mandatory=$true)][string]$ComsolExe,",
        f"  [string]$Mph = {_ps_single_quoted(default_mph)},",
        "  [string]$OutRoot = '_external_exports\\micromixer_particle_tracing_reextract'",
        ")",
        "$ErrorActionPreference = 'Stop'",
        "$Runner = 'external\\comsol_particle_export\\run_export.ps1'",
        "$Requests = @(",
    ]
    for idx, request in enumerate(requests):
        name = str(request.get("name", request.get("kind", "request")))
        config = str(request.get("config", ""))
        safe_name = _sanitize(name)
        suffix = "," if idx < len(requests) - 1 else ""
        lines.append(
            "  @{ Name = "
            + _ps_single_quoted(safe_name)
            + "; Config = "
            + _ps_single_quoted(config)
            + "; OutDir = (Join-Path $OutRoot "
            + _ps_single_quoted(safe_name)
            + ") }"
            + suffix
        )
    lines.extend(
        [
            ")",
            "foreach ($Request in $Requests) {",
            "  Write-Host \"[COMSOL export] $($Request.Name)\"",
            "  & $Runner -ComsolExe $ComsolExe -Mph $Mph -Config $Request.Config -OutDir $Request.OutDir",
            "}",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = ("write_reextract_request_bundle",)
