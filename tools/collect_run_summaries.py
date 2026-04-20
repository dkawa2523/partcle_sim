from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect compact CSV summaries from one or more solver output directories.")
    parser.add_argument("output_dirs", nargs="+", type=Path, help="Solver output directories containing solver_report.json.")
    parser.add_argument("--output-csv", type=Path, default=Path("run_summary_compare.csv"))
    args = parser.parse_args()
    out = collect_run_summaries(args.output_dirs, args.output_csv)
    print(f"wrote {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
