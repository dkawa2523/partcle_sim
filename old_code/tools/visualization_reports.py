"""Own visualization directories, indexes, and human-readable run reports."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tools.state_contract import STATE_ORDER


def ensure_visualization_dirs(output_dir: Path, clean: bool = False) -> dict[str, Path]:
    base = Path(output_dir)
    if clean:
        for legacy in ("graphs", "animations", "visuals"):
            legacy_dir = base / legacy
            if legacy_dir.exists() and legacy_dir.is_dir():
                shutil.rmtree(legacy_dir)
        existing_root = base / "visualizations"
        if existing_root.exists() and existing_root.is_dir():
            shutil.rmtree(existing_root)
    root = base / "visualizations"
    dirs = {
        "root": root,
        "graphs": root / "graphs",
        "animations": root / "animations",
        "mechanics": root / "mechanics",
        "boundary_diagnostics": root / "boundary_diagnostics",
        "reports": root / "reports",
    }
    dirs["root"].mkdir(parents=True, exist_ok=True)
    dirs["reports"].mkdir(parents=True, exist_ok=True)
    return dirs


def write_visualization_index(output_dir: Path, payload: Mapping[str, object]) -> Path:
    dirs = ensure_visualization_dirs(output_dir, clean=False)
    index_path = dirs["reports"] / "visualization_index.json"
    index_path.write_text(json.dumps(dict(payload), indent=2), encoding="utf-8")
    return index_path


def read_optional_json_object(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _int_value(report: Mapping[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(report.get(key, default))
    except (TypeError, ValueError):
        return int(default)


def _optional_float(report: Mapping[str, Any], key: str) -> float | None:
    try:
        return float(report[key]) if key in report else None
    except (TypeError, ValueError):
        return None


def _format_optional_seconds(value: Any) -> str:
    try:
        if value is None:
            return "not_recorded"
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "not_recorded"


def build_run_health_summary(output_dir: Path) -> dict[str, object]:
    base = Path(output_dir)
    report = read_optional_json_object(base / "run_summary.json")
    debug = read_optional_json_object(base / "debug_diagnostics.json")
    raw_diagnostics = debug.get("collision", {})
    diagnostics = raw_diagnostics if isinstance(raw_diagnostics, Mapping) else {}
    timing_raw = report.get("timing_s", report.get("timing", {}))
    timing = timing_raw if isinstance(timing_raw, Mapping) else {}
    raw_memory = report.get("memory_estimate_bytes", report.get("memory", {}))
    memory = raw_memory if isinstance(raw_memory, Mapping) else {}
    raw_state_counts = report.get("final_state_counts", {})
    state_counts = (
        {name: _int_value(raw_state_counts, name) for name in STATE_ORDER}
        if isinstance(raw_state_counts, Mapping)
        else {}
    )
    health: dict[str, Any] = {
        "particle_count": _int_value(
            report, "particle_count", default=sum(state_counts.values())
        ),
        "released_count": _int_value(report, "released_count"),
        "invalid_mask_stopped_count": _int_value(report, "invalid_mask_stopped_count"),
        "numerical_boundary_stopped_count": _int_value(
            report, "numerical_boundary_stopped_count"
        ),
        "stuck_count": _int_value(report, "stuck_count"),
        "absorbed_count": _int_value(report, "absorbed_count"),
        "field_support_exit_count": _int_value(report, "field_support_exit_count"),
        "max_hits_reached_count": _int_value(report, "max_hits_reached_count"),
        "unresolved_crossing_count": _int_value(report, "unresolved_crossing_count"),
        "nearest_projection_fallback_count": _int_value(
            diagnostics, "nearest_projection_fallback_count"
        ),
        "contact_sliding_particle_count": int(state_counts.get("contact_sliding", 0)),
        "contact_endpoint_stopped_count": int(
            state_counts.get("contact_endpoint_stopped", 0)
        ),
        "contact_tangent_step_count": _int_value(
            diagnostics, "contact_tangent_step_count"
        ),
        "nonfinite_position_count": _int_value(report, "nonfinite_position_count"),
        "nonfinite_velocity_count": _int_value(report, "nonfinite_velocity_count"),
        "solver_core_s": _optional_float(timing, "solver_core_s"),
        "estimated_numpy_bytes": _int_value(memory, "estimated_numpy_bytes"),
        "final_state_counts": state_counts,
    }
    failure_keys = (
        "invalid_mask_stopped_count",
        "numerical_boundary_stopped_count",
        "max_hits_reached_count",
        "unresolved_crossing_count",
        "nearest_projection_fallback_count",
        "nonfinite_position_count",
        "nonfinite_velocity_count",
    )
    failed = any(int(health.get(key, 0)) > 0 for key in failure_keys)
    health["status"] = "pass" if not failed else "review"
    return health


_RUN_SUMMARY_HEALTH_KEYS = (
    "invalid_mask_stopped_count",
    "numerical_boundary_stopped_count",
    "max_hits_reached_count",
    "unresolved_crossing_count",
    "nearest_projection_fallback_count",
    "contact_sliding_particle_count",
    "contact_endpoint_stopped_count",
    "nonfinite_position_count",
    "nonfinite_velocity_count",
)


def _run_summary_health(payload: Mapping[str, object]) -> dict[str, object]:
    health_payload = payload.get("health_summary", {})
    return dict(health_payload) if isinstance(health_payload, Mapping) else {}


def _run_summary_module_names(payload: Mapping[str, object]) -> list[Any]:
    modules = payload.get("modules", {})
    return sorted(modules.keys()) if isinstance(modules, Mapping) else []


def _run_summary_header(output_dir: Path, health: Mapping[str, object]) -> list[str]:
    return [
        "# Run Summary",
        "",
        f"- status: {health.get('status', 'unknown')}",
        f"- output_dir: {Path(output_dir).resolve()}",
        f"- particles: {health.get('particle_count', 0)}",
        f"- released: {health.get('released_count', 0)}",
        f"- solver_core_s: {_format_optional_seconds(health.get('solver_core_s'))}",
        f"- estimated_numpy_bytes: {health.get('estimated_numpy_bytes', 0)}",
        "",
        "## Solver Health",
        "",
    ]


def _append_run_health(lines: list[str], health: Mapping[str, object]) -> None:
    for key in _RUN_SUMMARY_HEALTH_KEYS:
        lines.append(f"- {key}: {health.get(key, 0)}")


def _append_final_states(lines: list[str], health: Mapping[str, object]) -> None:
    state_counts = health.get("final_state_counts", {})
    if isinstance(state_counts, Mapping) and state_counts:
        lines.extend(["", "## Final States", ""])
        for name, count in state_counts.items():
            lines.append(f"- {name}: {count}")


def _append_compact_summary_files(
    lines: list[str], payload: Mapping[str, object]
) -> None:
    summary_files = payload.get("summary_files", {})
    if isinstance(summary_files, Mapping) and summary_files:
        lines.extend(["", "## Compact Summary Files", ""])
        for name, path in sorted(summary_files.items()):
            lines.append(f"- {name}: {path}")


def _append_visualization_modules(lines: list[str], module_names: list[Any]) -> None:
    lines.extend(["", "## Visualization Modules", ""])
    if module_names:
        for name in module_names:
            lines.append(f"- {name}")
    else:
        lines.append("- none")


def write_run_summary(output_dir: Path, payload: Mapping[str, object]) -> Path:
    dirs = ensure_visualization_dirs(output_dir, clean=False)
    health = _run_summary_health(payload)
    module_names = _run_summary_module_names(payload)
    lines = _run_summary_header(output_dir, health)
    _append_run_health(lines, health)
    _append_final_states(lines, health)
    _append_compact_summary_files(lines, payload)
    _append_visualization_modules(lines, module_names)
    summary_path = dirs["reports"] / "run_summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path
