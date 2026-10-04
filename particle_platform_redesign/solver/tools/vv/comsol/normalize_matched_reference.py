"""Normalize the read-only COMSOL matched-case exports and quantify convergence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

COLUMNS = (
    "particle_id",
    "time_s",
    "r_m",
    "z_m",
    "velocity_r_m_per_s",
    "velocity_z_m_per_s",
    "charge_number_e",
    "current_status_code",
    "final_status_code",
    "stop_or_event_time_s",
    "electric_force_r_N",
    "electric_force_z_N",
    "epstein_force_r_N",
    "epstein_force_z_N",
    "gravity_buoyancy_force_r_N",
    "gravity_buoyancy_force_z_N",
    "acceleration_r_m_per_s2",
    "acceleration_z_m_per_s2",
    "gas_velocity_r_m_per_s",
    "gas_velocity_z_m_per_s",
    "gas_temperature_K",
    "gas_density_kg_per_m3",
    "gas_dynamic_viscosity_Pa_s",
    "electric_field_r_V_per_m",
    "electric_field_z_V_per_m",
    "particle_mass_kg",
)
STEP_DIRECTORIES = ("dt_10us", "dt_5us", "dt_2p5us")
OUTPUT_TIMES = 41


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_wide(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for line in stream:
            if line.startswith("%") or not line.strip():
                continue
            values = next(csv.reader([line]))
            expected = len(COLUMNS) * OUTPUT_TIMES
            if len(values) != expected:
                raise ValueError(f"{path}: expected {expected} values, found {len(values)}")
            rows.append([float(value) for value in values])
    if len(rows) != 287:
        raise ValueError(f"{path}: expected 287 particle rows, found {len(rows)}")
    return rows


def _tidy(wide: list[list[float]]) -> list[list[float]]:
    width = len(COLUMNS)
    records = [row[offset : offset + width] for row in wide for offset in range(0, len(row), width)]
    expected = 287 * OUTPUT_TIMES
    if len(records) != expected:
        raise ValueError(f"expected {expected} records, found {len(records)}")
    return records


def _validate_records(records: list[list[float]], source: Path) -> None:
    by_particle: dict[int, list[float]] = {}
    for record in records:
        if not all(math.isfinite(value) for value in record):
            raise ValueError(f"{source}: nonfinite value in the pre-event reference window")
        particle = round(record[0])
        by_particle.setdefault(particle, []).append(record[1])
        if record[6] != -1.0:
            raise ValueError(f"{source}: charge is not fixed at -1 for particle {particle}")
        if round(record[7]) != 1:
            raise ValueError(f"{source}: terminal state appeared in the pre-event window")
        if record[25] <= 0.0:
            raise ValueError(f"{source}: nonpositive particle mass")
    if sorted(by_particle) != list(range(1, 288)):
        raise ValueError(f"{source}: particle IDs are not exactly 1..287")
    expected_times = [index * 1e-5 for index in range(OUTPUT_TIMES)]
    for particle, times in by_particle.items():
        if len(times) != OUTPUT_TIMES or any(
            abs(actual - expected) > 1e-15
            for actual, expected in zip(times, expected_times, strict=True)
        ):
            raise ValueError(f"{source}: unexpected time grid for particle {particle}")


def _lifecycle(status: float) -> str:
    return {1: "active", 2: "frozen", 3: "stuck", 4: "escaped"}.get(round(status), "unknown")


def _write_csv(path: Path, header: tuple[str, ...], rows: list[tuple[Any, ...]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _write_artifacts(directory: Path, records: list[list[float]]) -> dict[str, Any]:
    trajectory: list[tuple[Any, ...]] = []
    forces: list[tuple[Any, ...]] = []
    fields: list[tuple[Any, ...]] = []
    events: list[tuple[Any, ...]] = []
    status_counts: dict[str, int] = {}
    by_particle: dict[int, list[list[float]]] = {}
    for record in records:
        particle = round(record[0])
        time = record[1]
        state = _lifecycle(record[7])
        status_counts[state] = status_counts.get(state, 0) + 1
        by_particle.setdefault(particle, []).append(record)
        trajectory.append((particle, time, *record[2:6], record[6], state))
        forces.append((particle, time, *record[10:18]))
        fields.append(
            (
                f"p{particle}_t{time:.17g}",
                *record[2:4],
                *record[18:25],
                "inside",
                3,
            )
        )

    for particle, history in by_particle.items():
        terminal = next((row for row in history if round(row[7]) != 1), None)
        if terminal is None:
            continue
        event_time = terminal[9]
        event_type = _lifecycle(terminal[7])
        events.append(
            (
                particle,
                0,
                event_time,
                terminal[2],
                terminal[3],
                event_type,
                "saved_state_observation",
                event_type,
            )
        )

    _write_csv(
        directory / "trajectory_reference.csv",
        (
            "particle_id",
            "time_s",
            "r_m",
            "z_m",
            "velocity_r_m_per_s",
            "velocity_z_m_per_s",
            "charge_number_e",
            "lifecycle",
        ),
        trajectory,
    )
    _write_csv(
        directory / "force_reference.csv",
        (
            "particle_id",
            "time_s",
            "electric_force_r_N",
            "electric_force_z_N",
            "epstein_force_r_N",
            "epstein_force_z_N",
            "gravity_buoyancy_force_r_N",
            "gravity_buoyancy_force_z_N",
            "acceleration_r_m_per_s2",
            "acceleration_z_m_per_s2",
        ),
        forces,
    )
    _write_csv(
        directory / "field_probe_reference.csv",
        (
            "probe_id",
            "r_m",
            "z_m",
            "gas_velocity_r_m_per_s",
            "gas_velocity_z_m_per_s",
            "gas_temperature_K",
            "gas_density_kg_per_m3",
            "gas_dynamic_viscosity_Pa_s",
            "electric_field_r_V_per_m",
            "electric_field_z_V_per_m",
            "support",
            "domain_id",
        ),
        fields,
    )
    _write_csv(
        directory / "event_observations.csv",
        (
            "particle_id",
            "event_ordinal",
            "event_time_s",
            "observed_r_m",
            "observed_z_m",
            "event_type",
            "boundary_semantic",
            "outcome",
        ),
        events,
    )
    return {
        "particles": len(by_particle),
        "records": len(records),
        "output_times": OUTPUT_TIMES,
        "status_record_counts": status_counts,
        "event_observation_count": len(events),
        "all_continuous_and_active": not events and status_counts == {"active": len(records)},
    }


def _record_map(records: list[list[float]]) -> dict[tuple[int, float], list[float]]:
    return {(round(row[0]), row[1]): row for row in records}


def _error_metrics(coarse: list[list[float]], fine: list[list[float]]) -> dict[str, float]:
    coarse_map = _record_map(coarse)
    fine_map = _record_map(fine)
    if coarse_map.keys() != fine_map.keys():
        raise ValueError("convergence histories do not have identical particle/time keys")
    position_sq: list[float] = []
    velocity_sq: list[float] = []
    position_norm_sq: list[float] = []
    velocity_norm_sq: list[float] = []
    position_abs: list[float] = []
    velocity_abs: list[float] = []
    for key, coarse_row in coarse_map.items():
        fine_row = fine_map[key]
        dp = math.hypot(coarse_row[2] - fine_row[2], coarse_row[3] - fine_row[3])
        dv = math.hypot(coarse_row[4] - fine_row[4], coarse_row[5] - fine_row[5])
        position_abs.append(dp)
        velocity_abs.append(dv)
        position_sq.append(dp * dp)
        velocity_sq.append(dv * dv)
        position_norm_sq.append(fine_row[2] ** 2 + fine_row[3] ** 2)
        velocity_norm_sq.append(fine_row[4] ** 2 + fine_row[5] ** 2)
    count = len(position_sq)
    pos_rms = math.sqrt(math.fsum(position_sq) / count)
    vel_rms = math.sqrt(math.fsum(velocity_sq) / count)
    return {
        "position_rms_m": pos_rms,
        "position_max_m": max(position_abs),
        "position_relative_l2": math.sqrt(math.fsum(position_sq) / math.fsum(position_norm_sq)),
        "velocity_rms_m_per_s": vel_rms,
        "velocity_max_m_per_s": max(velocity_abs),
        "velocity_relative_l2": math.sqrt(math.fsum(velocity_sq) / math.fsum(velocity_norm_sq)),
    }


def _observed_order(coarse_fine: float, fine_finer: float) -> float | None:
    if coarse_fine <= 0.0 or fine_finer <= 0.0:
        return None
    return math.log(coarse_fine / fine_finer, 2.0)


def _write_convergence(root: Path, records: dict[str, list[list[float]]]) -> dict[str, Any]:
    ten_five = _error_metrics(records["dt_10us"], records["dt_5us"])
    five_two = _error_metrics(records["dt_5us"], records["dt_2p5us"])
    result: dict[str, Any] = {"dt_10us_vs_5us": ten_five, "dt_5us_vs_2p5us": five_two}
    for quantity, key in (
        ("position", "position_rms_m"),
        ("velocity", "velocity_rms_m_per_s"),
    ):
        order = _observed_order(ten_five[key], five_two[key])
        estimate = None
        if order is not None and order > 0.0:
            estimate = five_two[key] / math.expm1(order * math.log(2.0))
        result[f"{quantity}_observed_order"] = order
        result[f"{quantity}_fine_grid_richardson_estimate"] = estimate
    (root / "comsol_self_convergence.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def normalize(root: Path) -> None:
    all_records: dict[str, list[list[float]]] = {}
    summaries: dict[str, Any] = {}
    for name in STEP_DIRECTORIES:
        directory = root / name
        raw = directory / "history_raw_wide.csv"
        records = _tidy(_read_wide(raw))
        _validate_records(records, raw)
        all_records[name] = records
        summary = _write_artifacts(directory, records)
        summary["raw_sha256"] = _sha256(raw)
        summaries[name] = summary
    convergence = _write_convergence(root, all_records)
    receipt = {
        "case": "Case A 100 nm deterministic common physics",
        "coordinate_system": "axisymmetric_rz_no_swirl",
        "reference_physics": ["electric", "Epstein", "gravity_buoyancy"],
        "disabled_physics": [
            "Brownian",
            "dynamic_charge",
            "ion_drag",
            "thermophoresis",
            "Saffman_lift",
            "free_molecular_lift",
            "dielectrophoresis",
        ],
        "charge_number_e": -1,
        "time_window_s": [0.0, 4e-4],
        "boundary_claim": "NOT_TESTED; window ends before first audited saved-run event",
        "integrator_stage_claim": "NOT_EXPORTED; COMSOL public result has accepted output states only",
        "runs": summaries,
        "self_convergence": convergence,
    }
    (root / "reference_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    normalize(args.root.resolve())


if __name__ == "__main__":
    main()
