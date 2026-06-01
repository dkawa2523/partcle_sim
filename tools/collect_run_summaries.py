from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


FINAL_STATE_COLUMNS = (
    "active_free_flight",
    "contact_sliding",
    "contact_endpoint_stopped",
    "stuck",
    "absorbed",
    "escaped",
    "invalid_mask_stopped",
    "numerical_boundary_stopped",
    "inactive",
)

OUTPUT_COLUMNS = (
    "run_name",
    "output_dir",
    "status",
    "particle_count",
    "released_count",
    "coordinate_system",
    "integrator",
    "valid_mask_policy",
    "drag_model",
    "acceleration_source",
    "solver_core_s",
    "step_loop_s",
    "freeflight_s",
    "collision_classify_s",
    "collider_resolution_s",
    "charge_model_s",
    "stochastic_motion_s",
    "estimated_numpy_bytes",
    *FINAL_STATE_COLUMNS,
    "unresolved_crossing_count",
    "max_hits_reached_count",
    "boundary_event_contract_passed",
    "drag_density_source",
    "drag_temperature_source",
    "drag_fallback_density_kgm3",
    "drag_fallback_temperature_K",
    "stochastic_enabled",
    "stochastic_stride",
    "plasma_source",
    "electron_density_m3",
    "ion_density_m3",
    "electron_temperature_eV",
    "ion_temperature_eV",
    "pressure_Pa",
    "gas_temperature_K",
    "neutral_density_m3",
    "debye_length_m",
    "electron_collision_frequency_s",
    "ion_collision_frequency_s",
    "effective_electron_collision_frequency_s",
    "conductivity_Sm",
    "charge_enabled",
    "charge_mode",
    "charge_background_source",
    "charge_plasma_background_source",
    "charge_update_stride",
    "charge_final_mean_charge_e",
    "charge_last_mean_floating_potential_V",
    "charge_last_mean_tau_q_s",
    "charge_last_response_regime",
    "charge_last_radius_over_debye",
)

SHARD_COMPACT_ARTIFACTS = (
    "solver_report.json",
    "prepared_runtime_summary.json",
    "wall_summary_by_part.csv",
    "source_model_summary.json",
    "source_particle_diagnostics.csv",
    "first_step_compare_summary.json",
    "first_step_summary.json",
    "collision_diagnostics.json",
    "comparison_summary.json",
    "compare_summary.json",
)

COUNTER_KEYS = (
    "unresolved_crossing_count",
    "max_hits_reached_count",
    "nearest_projection_fallback_count",
    "bisection_fallback_count",
    "numerical_boundary_stopped_count",
    "boundary_event_failure_count",
    "invalid_mask_stopped_count",
    "valid_mask_mixed_stencil_count",
    "valid_mask_hard_invalid_count",
    "source_surface_release_skip_count",
    "source_surface_release_skip_blocked_count",
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _read_scalar_summary(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = csv.DictReader(handle)
            out: dict[str, str] = {}
            for row in rows:
                key = str(row.get("quantity", "")).strip()
                if key:
                    out[key] = str(row.get("value", "")).strip()
            return out
    except OSError:
        return {}


def _get(mapping: Mapping[str, Any], path: str, default: Any = "") -> Any:
    current: Any = mapping
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _first(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value == "":
            continue
        return value
    return ""


def _count_final_particles(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    counts = {key: 0 for key in FINAL_STATE_COLUMNS}
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("active") in {"1", "True", "true"}:
                    counts["active_free_flight"] += 1
                for key in FINAL_STATE_COLUMNS:
                    if key == "active_free_flight":
                        continue
                    if row.get(key) in {"1", "True", "true"}:
                        counts[key] += 1
        return counts
    except OSError:
        return {}


def _sum_counts(counts: Mapping[str, Any]) -> int:
    total = 0
    for value in counts.values():
        try:
            total += int(float(value))
        except (TypeError, ValueError):
            continue
    return total


def _as_number(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out and out not in (float("inf"), float("-inf")) else None


def _sum_numeric_values(values: Iterable[Any]) -> int | float:
    total = 0.0
    has_float = False
    for value in values:
        number = _as_number(value)
        if number is None:
            continue
        has_float = has_float or not float(number).is_integer()
        total += number
    return float(total) if has_float else int(total)


def _shared_value(values: Iterable[Any]) -> Any:
    cleaned = [value for value in values if value not in (None, "")]
    if not cleaned:
        return ""
    first = cleaned[0]
    return first if all(value == first for value in cleaned) else "mixed"


def _shared_list(values: Iterable[Any]) -> list[Any]:
    cleaned = [list(value) for value in values if isinstance(value, list)]
    if not cleaned:
        return []
    first = cleaned[0]
    return first if all(value == first for value in cleaned) else []


def _sum_mapping_values(mappings: Iterable[Mapping[str, Any]], key: str) -> int | float:
    return _sum_numeric_values(mapping.get(key, 0) for mapping in mappings)


def _sum_named_count_maps(mappings: Iterable[Mapping[str, Any]], key: str) -> dict[str, int | float]:
    out: dict[str, int | float] = {}
    for mapping in mappings:
        raw = mapping.get(key, {})
        if not isinstance(raw, Mapping):
            continue
        for name, value in raw.items():
            out[str(name)] = _sum_numeric_values((out.get(str(name), 0), value))
    return out


def _status(row: Mapping[str, Any]) -> str:
    failure_keys = (
        "invalid_mask_stopped",
        "numerical_boundary_stopped",
        "unresolved_crossing_count",
        "max_hits_reached_count",
    )
    for key in failure_keys:
        try:
            if int(float(row.get(key, 0) or 0)) > 0:
                return "review"
        except (TypeError, ValueError):
            return "review"
    if str(row.get("boundary_event_contract_passed", "1")) in {"0", "False", "false"}:
        return "review"
    return "pass"


def collect_run_summary(output_dir: Path) -> dict[str, Any]:
    base = output_dir.resolve()
    report = _read_json(base / "solver_report.json")
    diagnostics = _read_json(base / "collision_diagnostics.json")
    plasma = _read_scalar_summary(base / "plasma_background_summary.csv")
    charge = _read_scalar_summary(base / "charge_model_summary.csv")

    report_counts = _get(report, "final_state_counts", {})
    final_counts = report_counts if isinstance(report_counts, Mapping) else {}
    if not final_counts:
        final_counts = _count_final_particles(base / "final_particles.csv")

    row: dict[str, Any] = {
        "run_name": base.name,
        "output_dir": str(base),
        "particle_count": _first(report.get("particle_count"), diagnostics.get("particle_count"), _sum_counts(final_counts)),
        "released_count": _first(report.get("released_count"), diagnostics.get("released_count")),
        "coordinate_system": report.get("coordinate_system", ""),
        "integrator": report.get("integrator", ""),
        "valid_mask_policy": report.get("valid_mask_policy", ""),
        "drag_model": report.get("drag_model", ""),
        "acceleration_source": report.get("acceleration_source", ""),
        "solver_core_s": _get(report, "timing_s.solver_core_s"),
        "step_loop_s": _get(report, "timing_s.step_loop_s"),
        "freeflight_s": _get(report, "timing_s.freeflight_s"),
        "collision_classify_s": _get(report, "timing_s.collision_classify_s"),
        "collider_resolution_s": _get(report, "timing_s.collider_resolution_s"),
        "charge_model_s": _get(report, "timing_s.charge_model_s"),
        "stochastic_motion_s": _get(report, "timing_s.stochastic_motion_s"),
        "estimated_numpy_bytes": _get(report, "memory_estimate_bytes.estimated_numpy_bytes"),
        "unresolved_crossing_count": _first(report.get("unresolved_crossing_count"), diagnostics.get("unresolved_crossing_count")),
        "max_hits_reached_count": _first(report.get("max_hits_reached_count"), diagnostics.get("max_hits_reached_count")),
        "boundary_event_contract_passed": _first(report.get("boundary_event_contract_passed"), _get(report, "boundary_event_contract.passed")),
        "drag_density_source": _get(report, "drag_gas_properties.density_source"),
        "drag_temperature_source": _get(report, "drag_gas_properties.temperature_source"),
        "drag_fallback_density_kgm3": _get(report, "drag_gas_properties.fallback_density_kgm3"),
        "drag_fallback_temperature_K": _get(report, "drag_gas_properties.fallback_temperature_K"),
        "stochastic_enabled": _get(report, "stochastic_motion.enabled"),
        "stochastic_stride": _get(report, "stochastic_motion.stride"),
    }
    state_count_fallbacks = {
        "contact_sliding": report.get("contact_sliding_particle_count"),
        "contact_endpoint_stopped": report.get("contact_endpoint_stopped_count"),
        "stuck": _first(report.get("stuck_count"), diagnostics.get("stuck_count")),
        "absorbed": _first(report.get("absorbed_count"), diagnostics.get("absorbed_count")),
        "escaped": report.get("escaped_count"),
        "invalid_mask_stopped": _first(report.get("invalid_mask_stopped_count"), diagnostics.get("invalid_mask_stopped_count")),
        "numerical_boundary_stopped": _first(
            report.get("numerical_boundary_stopped_count"),
            diagnostics.get("numerical_boundary_stopped_count"),
        ),
    }
    for key in FINAL_STATE_COLUMNS:
        row[key] = _first(final_counts.get(key), state_count_fallbacks.get(key))

    row.update(
        {
            "plasma_source": _first(plasma.get("source"), _get(report, "plasma_background.source")),
            "electron_density_m3": _first(plasma.get("electron_density_m3"), _get(report, "plasma_background.electron_density_m3")),
            "ion_density_m3": _first(plasma.get("ion_density_m3"), _get(report, "plasma_background.ion_density_m3")),
            "electron_temperature_eV": _first(plasma.get("electron_temperature_eV"), _get(report, "plasma_background.electron_temperature_eV")),
            "ion_temperature_eV": _first(plasma.get("ion_temperature_eV"), _get(report, "plasma_background.ion_temperature_eV")),
            "pressure_Pa": _first(plasma.get("pressure_Pa"), _get(report, "plasma_background.pressure_Pa")),
            "gas_temperature_K": _first(plasma.get("gas_temperature_K"), _get(report, "plasma_background.gas_temperature_K")),
            "neutral_density_m3": _first(plasma.get("neutral_density_m3"), _get(report, "plasma_background.neutral_density_m3")),
            "debye_length_m": _first(plasma.get("debye_length_m"), _get(report, "plasma_background.debye_length_m")),
            "electron_collision_frequency_s": _first(plasma.get("electron_collision_frequency_s"), _get(report, "plasma_background.electron_collision_frequency_s")),
            "ion_collision_frequency_s": _first(plasma.get("ion_collision_frequency_s"), _get(report, "plasma_background.ion_collision_frequency_s")),
            "effective_electron_collision_frequency_s": _first(
                plasma.get("effective_electron_collision_frequency_s"),
                _get(report, "plasma_background.effective_electron_collision_frequency_s"),
            ),
            "conductivity_Sm": _first(plasma.get("conductivity_Sm"), _get(report, "plasma_background.conductivity_Sm")),
            "charge_enabled": _first(charge.get("enabled"), _get(report, "charge_model.enabled")),
            "charge_mode": _first(charge.get("mode"), _get(report, "charge_model.mode")),
            "charge_background_source": _first(charge.get("background_source"), _get(report, "charge_model.background_source")),
            "charge_plasma_background_source": _first(
                charge.get("plasma_background_source"),
                _get(report, "charge_model.plasma_background_source"),
            ),
            "charge_update_stride": _first(charge.get("update_stride"), _get(report, "charge_model.update_stride")),
            "charge_final_mean_charge_e": _first(charge.get("final_mean_charge_e"), _get(report, "charge_model.final_mean_charge_e")),
            "charge_last_mean_floating_potential_V": _first(
                charge.get("last_mean_floating_potential_V"),
                _get(report, "charge_model.last_mean_floating_potential_V"),
            ),
            "charge_last_mean_tau_q_s": _first(charge.get("last_mean_tau_q_s"), _get(report, "charge_model.last_mean_tau_q_s")),
            "charge_last_response_regime": _first(
                charge.get("last_charge_response_regime"),
                _get(report, "charge_model.last_charge_response_regime"),
            ),
            "charge_last_radius_over_debye": _first(
                charge.get("last_mean_particle_radius_over_debye"),
                _get(report, "charge_model.last_mean_particle_radius_over_debye"),
            ),
        }
    )
    row["status"] = _status(row)
    return {key: row.get(key, "") for key in OUTPUT_COLUMNS}


def collect_run_summaries(output_dirs: list[Path], output_csv: Path) -> Path:
    rows = [collect_run_summary(path) for path in output_dirs]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OUTPUT_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    return output_csv


def _read_source_particle_diagnostics(output_dir: Path) -> tuple[list[str], list[dict[str, str]]]:
    path = output_dir / "source_particle_diagnostics.csv"
    if not path.exists():
        return [], []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = []
        for row in reader:
            item = {str(key): str(value) for key, value in row.items()}
            item["shard_name"] = output_dir.name
            item["shard_output_dir"] = str(output_dir.resolve())
            rows.append(item)
    return fieldnames, rows


def _write_root_source_particle_diagnostics(output_dirs: list[Path], root_artifacts_dir: Path) -> tuple[Path | None, int]:
    rows: list[dict[str, str]] = []
    source_fields: list[str] = []
    seen_fields = {"shard_name", "shard_output_dir"}
    for output_dir in output_dirs:
        fieldnames, shard_rows = _read_source_particle_diagnostics(output_dir.resolve())
        for field in fieldnames:
            if field not in seen_fields:
                seen_fields.add(field)
                source_fields.append(field)
        rows.extend(shard_rows)
    if not rows:
        return None, 0

    output_path = root_artifacts_dir / "source_particle_diagnostics.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["shard_name", "shard_output_dir", *source_fields]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return output_path, len(rows)


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _shard_jsons(output_dirs: list[Path], filename: str) -> list[tuple[Path, dict[str, Any]]]:
    rows: list[tuple[Path, dict[str, Any]]] = []
    for output_dir in output_dirs:
        path = output_dir.resolve() / filename
        payload = _read_json(path)
        if payload:
            rows.append((path, payload))
    return rows


def _source_model_summaries(output_dirs: list[Path]) -> list[tuple[Path, dict[str, Any], str]]:
    rows: list[tuple[Path, dict[str, Any], str]] = []
    for output_dir in output_dirs:
        base = output_dir.resolve()
        source_path = base / "source_model_summary.json"
        source_payload = _read_json(source_path)
        if source_payload:
            rows.append((source_path, source_payload, "file"))
            continue
        prepared_path = base / "prepared_runtime_summary.json"
        prepared = _read_json(prepared_path)
        embedded = prepared.get("source_model_summary", {}) if isinstance(prepared, Mapping) else {}
        if isinstance(embedded, Mapping) and embedded:
            rows.append((prepared_path, dict(embedded), "embedded"))
    return rows


def _aggregate_solver_reports(output_dirs: list[Path], root_artifacts_dir: Path) -> Path | None:
    rows = _shard_jsons(output_dirs, "solver_report.json")
    if not rows:
        return None
    reports = [payload for _path, payload in rows]
    final_state_counts: dict[str, int | float] = {}
    for key in FINAL_STATE_COLUMNS:
        values = []
        for report in reports:
            counts = report.get("final_state_counts", {})
            if isinstance(counts, Mapping) and key in counts:
                values.append(counts.get(key, 0))
            elif key in report:
                values.append(report.get(key, 0))
            elif key == "active_free_flight":
                values.append(report.get("active_count", 0))
        final_state_counts[key] = _sum_numeric_values(values)
    timing_keys = sorted(
        {
            str(key)
            for report in reports
            if isinstance(report.get("timing_s", {}), Mapping)
            for key in report.get("timing_s", {})
        }
    )
    timing_s = {
        key: _sum_numeric_values(
            report.get("timing_s", {}).get(key, 0) if isinstance(report.get("timing_s", {}), Mapping) else 0
            for report in reports
        )
        for key in timing_keys
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "source_kind": "sharded_root_aggregate",
        "shard_count": int(len(output_dirs)),
        "aggregated_shard_count": int(len(rows)),
        "shard_output_dirs": [str(path.parent) for path, _payload in rows],
        "shard_solver_report_paths": [str(path) for path, _payload in rows],
        "particle_count": _sum_mapping_values(reports, "particle_count"),
        "released_count": _sum_mapping_values(reports, "released_count"),
        "coordinate_system": _shared_value(report.get("coordinate_system", "") for report in reports),
        "axis_names": _shared_list(report.get("axis_names", []) for report in reports),
        "integrator": _shared_value(report.get("integrator", "") for report in reports),
        "valid_mask_policy": _shared_value(report.get("valid_mask_policy", "") for report in reports),
        "drag_model": _shared_value(report.get("drag_model", "") for report in reports),
        "final_state_counts": final_state_counts,
        "timing_s": timing_s,
    }
    for key in COUNTER_KEYS:
        payload[key] = _sum_mapping_values(reports, key)
    return _write_json(root_artifacts_dir / "solver_report.json", payload)


def _aggregate_prepared_summaries(output_dirs: list[Path], root_artifacts_dir: Path) -> Path | None:
    rows = _shard_jsons(output_dirs, "prepared_runtime_summary.json")
    if not rows:
        return None
    summaries = [payload for _path, payload in rows]
    payload = {
        "schema_version": 1,
        "source_kind": "sharded_root_aggregate",
        "shard_count": int(len(output_dirs)),
        "aggregated_shard_count": int(len(rows)),
        "shard_output_dirs": [str(path.parent) for path, _payload in rows],
        "shard_prepared_runtime_summary_paths": [str(path) for path, _payload in rows],
        "coordinate_system": _shared_value(summary.get("coordinate_system", "") for summary in summaries),
        "axis_names": _shared_list(summary.get("axis_names", []) for summary in summaries),
        "spatial_dim": _shared_value(summary.get("spatial_dim", "") for summary in summaries),
        "particles": _sum_mapping_values(summaries, "particles"),
        "has_geometry_provider": _shared_value(summary.get("has_geometry_provider", "") for summary in summaries),
        "has_field_provider": _shared_value(summary.get("has_field_provider", "") for summary in summaries),
        "notes": ["This is a root aggregate wrapper; per-shard prepared summaries remain authoritative."],
    }
    return _write_json(root_artifacts_dir / "prepared_runtime_summary.json", payload)


def _aggregate_wall_summary_by_part(output_dirs: list[Path], root_artifacts_dir: Path) -> Path | None:
    grouped: dict[tuple[str, str, str], int] = {}
    for output_dir in output_dirs:
        path = output_dir.resolve() / "wall_summary_by_part.csv"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                key = (
                    str(row.get("part_id", "")).strip(),
                    str(row.get("outcome", "")).strip(),
                    str(row.get("wall_mode", "")).strip(),
                )
                try:
                    count = int(float(row.get("count", 0) or 0))
                except (TypeError, ValueError):
                    count = 0
                grouped[key] = grouped.get(key, 0) + count
    if not grouped:
        return None
    output_path = root_artifacts_dir / "wall_summary_by_part.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["part_id", "outcome", "wall_mode", "count"])
        writer.writeheader()
        for part_id, outcome, wall_mode in sorted(grouped):
            writer.writerow(
                {
                    "part_id": part_id,
                    "outcome": outcome,
                    "wall_mode": wall_mode,
                    "count": int(grouped[(part_id, outcome, wall_mode)]),
                }
            )
    return output_path


def _aggregate_source_model_summaries(output_dirs: list[Path], root_artifacts_dir: Path) -> Path | None:
    rows = _source_model_summaries(output_dirs)
    if not rows:
        return None
    summaries = [payload for _path, payload, _kind in rows]
    numeric_counts: dict[str, int | float] = {}
    for summary in summaries:
        for key, value in summary.items():
            if "count" not in str(key):
                continue
            number = _as_number(value)
            if number is not None:
                numeric_counts[str(key)] = _sum_numeric_values((numeric_counts.get(str(key), 0), number))
    payload = {
        "schema_version": 1,
        "source_kind": "sharded_root_aggregate",
        "shard_count": int(len(output_dirs)),
        "aggregated_shard_count": int(len(rows)),
        "source_model_summary_paths": [str(path) for path, _payload, _kind in rows],
        "source_model_summary_sources": [
            {"path": str(path), "source": kind}
            for path, _payload, kind in rows
        ],
        **numeric_counts,
        "law_usage": _sum_named_count_maps(summaries, "law_usage"),
        "source_provenance_counts": _sum_named_count_maps(summaries, "source_provenance_counts"),
        "notes": ["Known numeric *count* fields are summed; per-shard source summaries remain authoritative."],
    }
    return _write_json(root_artifacts_dir / "source_model_summary.json", payload)


def _aggregate_first_step_summaries(output_dirs: list[Path], root_artifacts_dir: Path) -> Path | None:
    rows = _shard_jsons(output_dirs, "first_step_compare_summary.json")
    if not rows:
        rows = _shard_jsons(output_dirs, "first_step_summary.json")
    if not rows:
        return None
    summaries = [payload for _path, payload in rows]
    payload = {
        "schema_version": 1,
        "source_kind": "sharded_root_aggregate",
        "shard_count": int(len(output_dirs)),
        "aggregated_shard_count": int(len(rows)),
        "shard_first_step_summary_paths": [str(path) for path, _payload in rows],
        "particle_count": _sum_mapping_values(summaries, "particle_count"),
        "compared_particle_count": _sum_mapping_values(summaries, "compared_particle_count"),
        "stochastic_policy": _shared_value(summary.get("stochastic_policy", "") for summary in summaries),
        "notes": ["Root first-step summary lists shard summaries; error distributions are not reweighted here."],
    }
    return _write_json(root_artifacts_dir / "first_step_compare_summary.json", payload)


def _aggregate_collision_diagnostics(output_dirs: list[Path], root_artifacts_dir: Path) -> Path | None:
    rows = _shard_jsons(output_dirs, "collision_diagnostics.json")
    if not rows:
        return None
    diagnostics = [payload for _path, payload in rows]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "source_kind": "sharded_root_aggregate",
        "shard_count": int(len(output_dirs)),
        "aggregated_shard_count": int(len(rows)),
        "shard_collision_diagnostics_paths": [str(path) for path, _payload in rows],
    }
    for key in COUNTER_KEYS:
        payload[key] = _sum_mapping_values(diagnostics, key)
    for key in (
        "invalid_mask_stop_reason_counts",
        "numerical_boundary_stop_reason_counts",
        "source_surface_release_skip_blocked_reasons",
    ):
        values = _sum_named_count_maps(diagnostics, key)
        if values:
            payload[key] = values
    return _write_json(root_artifacts_dir / "collision_diagnostics.json", payload)


def _write_root_aggregate_artifacts(output_dirs: list[Path], root_artifacts_dir: Path) -> dict[str, str]:
    generated: dict[str, str] = {}
    for name, writer in (
        ("solver_report.json", _aggregate_solver_reports),
        ("prepared_runtime_summary.json", _aggregate_prepared_summaries),
        ("wall_summary_by_part.csv", _aggregate_wall_summary_by_part),
        ("source_model_summary.json", _aggregate_source_model_summaries),
        ("first_step_compare_summary.json", _aggregate_first_step_summaries),
        ("collision_diagnostics.json", _aggregate_collision_diagnostics),
    ):
        path = writer(output_dirs, root_artifacts_dir)
        if path is not None:
            generated[name] = str(path.resolve())
    return generated


def _shard_artifact_manifest(output_dirs: list[Path]) -> list[dict[str, Any]]:
    shards: list[dict[str, Any]] = []
    for output_dir in output_dirs:
        base = output_dir.resolve()
        artifacts: dict[str, Any] = {}
        for filename in SHARD_COMPACT_ARTIFACTS:
            path = base / filename
            artifacts[filename] = {
                "exists": bool(path.exists()),
                "path": str(path) if path.exists() else "",
            }
        shards.append(
            {
                "shard_name": base.name,
                "output_dir": str(base),
                "artifacts": artifacts,
            }
        )
    return shards


def collect_shard_root_artifacts(
    output_dirs: list[Path],
    root_artifacts_dir: Path,
    *,
    output_csv: Path | None = None,
) -> tuple[Path, Path]:
    root_artifacts_dir = root_artifacts_dir.resolve()
    root_artifacts_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = collect_run_summaries(output_dirs, output_csv or (root_artifacts_dir / "run_summary_compare.csv"))
    source_diag_path, source_diag_rows = _write_root_source_particle_diagnostics(output_dirs, root_artifacts_dir)
    aggregate_artifacts = _write_root_aggregate_artifacts(output_dirs, root_artifacts_dir)
    shards = _shard_artifact_manifest(output_dirs)
    comparison_summary_paths = [
        artifact["path"]
        for shard in shards
        for artifact in (shard.get("artifacts", {}) or {}).values()
        if isinstance(artifact, Mapping)
        and bool(artifact.get("exists"))
        and str(artifact.get("path", "")).endswith(("comparison_summary.json", "compare_summary.json"))
    ]
    generated_artifacts: dict[str, Any] = {
        "run_summary_compare.csv": str(summary_csv.resolve()),
    }
    generated_artifacts.update(aggregate_artifacts)
    if source_diag_path is not None:
        generated_artifacts["source_particle_diagnostics.csv"] = str(source_diag_path.resolve())

    manifest = {
        "schema_version": 1,
        "root_artifacts_dir": str(root_artifacts_dir),
        "shard_count": len(output_dirs),
        "source_particle_diagnostics_rows": int(source_diag_rows),
        "generated_artifacts": generated_artifacts,
        "comparison_summary_paths": comparison_summary_paths,
        "shards": shards,
        "notes": [
            "Per-shard comparison summaries are listed, not merged.",
            "Write ensemble comparison_summary.json at the root with a compare tool.",
        ],
    }
    manifest_path = root_artifacts_dir / "shard_artifacts_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return summary_csv, manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect compact CSV summaries from one or more solver output directories.")
    parser.add_argument("output_dirs", nargs="+", type=Path, help="Solver output directories containing solver_report.json.")
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument(
        "--root-artifacts-dir",
        type=Path,
        default=None,
        help="Optional root directory for sharded-run comparison artifacts.",
    )
    args = parser.parse_args()
    if args.root_artifacts_dir is not None:
        summary_csv, manifest_path = collect_shard_root_artifacts(
            args.output_dirs,
            args.root_artifacts_dir,
            output_csv=args.output_csv,
        )
        print(f"wrote {summary_csv.resolve()}")
        print(f"wrote {manifest_path.resolve()}")
        return 0
    out = collect_run_summaries(args.output_dirs, args.output_csv or Path("run_summary_compare.csv"))
    print(f"wrote {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
