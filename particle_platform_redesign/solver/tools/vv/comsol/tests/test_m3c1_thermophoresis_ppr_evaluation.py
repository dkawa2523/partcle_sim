from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pytest
from tools.vv.comsol import evaluate_m3c1_thermophoresis_ppr as ppr


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_wide(
    path: Path,
    columns: tuple[str, ...],
    particle_rows: list[list[dict[str, float]]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["% synthetic wide fixture"])
        for frames in particle_rows:
            writer.writerow([record[name] for record in frames for name in columns])


def _particle_record(particle_id: int, time_s: float) -> dict[str, float]:
    temperature = 300.0 + particle_id
    conductivity = 0.02
    gradient_r = 100.0 + particle_id
    gradient_z = -200.0 + 1e4 * time_s
    heat_flux_r = -conductivity * gradient_r
    heat_flux_z = -conductivity * gradient_z
    diameter = 1e-7
    molar_mass = 0.0768032
    thermal_speed = math.sqrt(
        8.0 * ppr.MOLAR_GAS_CONSTANT_J_PER_MOL_K * temperature / (math.pi * molar_mass)
    )
    factor = ppr.WALDMANN_COEFFICIENT * (0.5 * diameter) ** 2 / thermal_speed
    return {
        "particle_id": float(particle_id),
        "time_s": time_s,
        "r_m": 0.01 * particle_id + time_s,
        "z_m": 0.02 * particle_id - time_s,
        "current_status_code": 1.0,
        "thermophoretic_force_r_N": factor * heat_flux_r,
        "thermophoretic_force_z_N": factor * heat_flux_z,
        "gas_temperature_K": temperature,
        "gas_thermal_conductivity_W_per_mK": conductivity,
        "unrecovered_temperature_gradient_r_K_per_m": 0.9 * gradient_r,
        "unrecovered_temperature_gradient_z_K_per_m": 0.9 * gradient_z,
        "ppr_temperature_gradient_r_K_per_m": gradient_r,
        "ppr_temperature_gradient_z_K_per_m": gradient_z,
        "ppr_heat_flux_r_W_per_m2": heat_flux_r,
        "ppr_heat_flux_z_W_per_m2": heat_flux_z,
        "particle_diameter_m": diameter,
        "background_gas_molar_mass_kg_per_mol": molar_mass,
    }


def _state_record(record: dict[str, float]) -> dict[str, float]:
    result = dict.fromkeys(ppr.V6_STATE_COLUMNS, 0.0)
    result.update(
        {
            "particle_id": record["particle_id"],
            "time_s": record["time_s"],
            "r_m": record["r_m"],
            "z_m": record["z_m"],
            "current_status_code": 1.0,
            "final_status_code": 1.0,
            "particle_mass_kg": 1e-18,
        }
    )
    return result


def _force_record(record: dict[str, float]) -> dict[str, float]:
    result = dict.fromkeys(ppr.V6_FORCE_COLUMNS, 0.0)
    result.update(
        {
            "particle_id": record["particle_id"],
            "time_s": record["time_s"],
            "thermophoretic_force_r_N": record["thermophoretic_force_r_N"],
            "thermophoretic_force_z_N": record["thermophoretic_force_z_N"],
        }
    )
    return result


def _mesh_record(index: int, domain: float) -> dict[str, float]:
    conductivity = 0.02
    gradient_r = 10.0 + index
    gradient_z = -20.0 - index
    result = dict.fromkeys(ppr.MESH_COLUMNS, 0.0)
    result.update(
        {
            "r_m": 0.001 * index,
            "z_m": 0.002 * index,
            "domain_id": domain,
            "gas_temperature_K": 300.0,
            "gas_thermal_conductivity_W_per_mK": conductivity,
            "unrecovered_temperature_gradient_r_K_per_m": 0.8 * gradient_r,
            "unrecovered_temperature_gradient_z_K_per_m": 0.8 * gradient_z,
            "ppr_temperature_gradient_r_K_per_m": gradient_r,
            "ppr_temperature_gradient_z_K_per_m": gradient_z,
            "ppr_heat_flux_r_W_per_m2": -conductivity * gradient_r,
            "ppr_heat_flux_z_W_per_m2": -conductivity * gradient_z,
        }
    )
    if domain != 3.0:
        for name in ppr.MESH_COLUMNS[3:]:
            result[name] = math.nan
    return result


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    solver = tmp_path / "solver"
    output = solver / "_out_m3c1" / "caseA_100nm_thermophoresis_ppr_v1"
    step = output / "dt_0p15625us"
    step.mkdir(parents=True)
    reference = solver / "_out_m3c0b" / "caseA_100nm_theory_pre_event_v6"
    reference_step = reference / "dt_0p15625us"
    reference_step.mkdir(parents=True)

    times = [0.0, 1e-5, 2e-5]
    particles = [
        [_particle_record(particle_id, time_s) for time_s in times] for particle_id in (1, 2)
    ]
    _write_wide(step / ppr.PARTICLE_RAW_TABLE, ppr.PARTICLE_COLUMNS, particles)
    _write_wide(
        reference_step / "state_raw_wide.csv",
        ppr.V6_STATE_COLUMNS,
        [[_state_record(record) for record in frames] for frames in particles],
    )
    _write_wide(
        reference_step / "force_raw_wide.csv",
        ppr.V6_FORCE_COLUMNS,
        [[_force_record(record) for record in frames] for frames in particles],
    )
    with (step / ppr.MESH_RAW_TABLE).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=ppr.MESH_COLUMNS, lineterminator="\n")
        writer.writerow({name: f"% {name}" for name in ppr.MESH_COLUMNS})
        writer.writerows(
            _mesh_record(index, domain) for index, domain in enumerate((1.4, 3.0, 3.0, 4.0), 1)
        )

    source_hash = "ab" * 32
    state_hash = _sha256(reference_step / "state_raw_wide.csv")
    force_hash = _sha256(reference_step / "force_raw_wide.csv")
    config: dict[str, Any] = {
        "schema_version": 1,
        "evaluation_id": "M3-C1-caseA-100nm-thermophoresis-PPR-export",
        "evaluation_revision": 1,
        "classification": "external_comsol_reference_correction",
        "source_model": {
            "sha256": source_hash,
            "load_mode": "ModelUtil.loadCopy",
            "saved": False,
        },
        "preserved_reference": {
            "root": "_out_m3c0b/caseA_100nm_theory_pre_event_v6",
            "step_directory": "dt_0p15625us",
            "state_raw_sha256": state_hash,
            "force_raw_sha256": force_hash,
        },
        "case": {
            "fixed_rk4_step_s": 1.5625e-7,
            "step_directory": "dt_0p15625us",
            "particle_count": 2,
            "output_times": 3,
            "time_end_s": 2e-5,
        },
        "thermophoretic_feature": {
            "selected_domain": 3,
            "model": "Waldmann",
            "UsePPR": True,
            "temperature_input": "root.comp1.AS_Tg",
            "thermal_conductivity": "k_mix",
            "background_gas_molar_mass": "Mmix",
            "background_gas_molar_mass_kg_per_mol": 0.0768032,
            "recovered_temperature_gradient": {
                "r": "ppr(d(root.comp1.AS_Tg,r))",
                "z": "ppr(d(root.comp1.AS_Tg,z))",
            },
            "recovered_heat_flux": {
                "r": "-k_mix*ppr(d(root.comp1.AS_Tg,r))",
                "z": "-k_mix*ppr(d(root.comp1.AS_Tg,z))",
            },
            "waldmann_replay": {
                "mean_thermal_speed": "sqrt(8*R_const*root.comp1.AS_Tg/(pi*Mmix))",
                "force_from_heat_flux": (
                    "(32/15)*(d0/2)^2*q/sqrt(8*R_const*root.comp1.AS_Tg/(pi*Mmix))"
                ),
            },
        },
        "acceptance": {
            "expected_particle_records": 6,
            "maximum_v6_coordinate_absolute_difference_m": 1e-14,
            "maximum_waldmann_force_component_scale_normalized_residual": 1e-10,
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    staged_config = output / "config.json"
    staged_config.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    provenance = {
        "source_sha256_before": source_hash,
        "source_sha256_after": source_hash,
        "source_unchanged": True,
        "source_load_mode": "ModelUtil.loadCopy",
        "model_saved": False,
        "preserved_m3c0b_v6_state_sha256_before": state_hash,
        "preserved_m3c0b_v6_state_sha256_after": state_hash,
        "preserved_m3c0b_v6_force_sha256_before": force_hash,
        "preserved_m3c0b_v6_force_sha256_after": force_hash,
        "preserved_m3c0b_v6_unchanged": True,
        "process_count": 1,
        "steps_run": 1,
        "staged_config": staged_config.name,
    }
    (output / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    (output / "run_status.json").write_text(
        json.dumps({"status": "COMPLETE", "failure": ""}), encoding="utf-8"
    )
    return output, config_path


def test_validates_saved_rows_waldmann_replay_and_dataset_points(tmp_path: Path) -> None:
    output, config = _fixture(tmp_path)

    report = ppr.evaluate(output, config, write_artifacts=False)

    assert report["overall_status"] == "PASS"
    assert report["particle_saved_states"]["records"] == 6
    assert report["coordinate_identity_to_locked_v6"]["maximum_absolute_difference_m"] == 0.0
    assert report["waldmann_ppr_replay"]["force_replay"]["status"] == "PASS"
    points = report["background_dataset_mesh_point_samples"]
    assert points["status"] == "PASS"
    assert points["selected_domain_rows"] == 2
    assert points["all_columns_nonfinite_rows"] == 2
    assert points["fractional_smoothed_domain_rows"] == 1
    assert points["native_fe_node_identity"] == "NOT_TESTED_NO_NODE_IDS"


def test_failed_force_replay_is_not_promoted(tmp_path: Path) -> None:
    output, config = _fixture(tmp_path)
    raw = output / "dt_0p15625us" / ppr.PARTICLE_RAW_TABLE
    lines = raw.read_text(encoding="utf-8").splitlines()
    values = next(csv.reader([lines[1]]))
    values[ppr.PARTICLE_COLUMNS.index("thermophoretic_force_r_N")] = str(
        2.0 * float(values[ppr.PARTICLE_COLUMNS.index("thermophoretic_force_r_N")])
    )
    lines[1] = ",".join(values)
    raw.write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = ppr.evaluate(output, config, write_artifacts=False)

    assert report["waldmann_ppr_replay"]["force_replay"]["status"] == "FAIL"
    assert report["overall_status"] == "FAIL"
    assert report["claims"]["saved_row_ppr_primitive_closure"] == "FAIL"


def test_writes_no_clobber_normalized_particle_and_domain3_tables(tmp_path: Path) -> None:
    output, config = _fixture(tmp_path)

    report = ppr.evaluate(output, config)

    assert report["normalized_particle_table"]["rows"] == 6
    assert report["normalized_domain3_point_table"]["rows"] == 2
    assert (output / ppr.PARTICLE_TIDY_TABLE).is_file()
    assert (output / ppr.DOMAIN3_TABLE).is_file()
    assert (output / ppr.REPORT_FILE).is_file()
    with pytest.raises(FileExistsError, match="refusing to clobber"):
        ppr.evaluate(output, config)
