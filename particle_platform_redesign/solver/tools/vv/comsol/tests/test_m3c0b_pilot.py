from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest
from tools.vv.comsol import m3c0b_pilot


def _receipt(step_s: float) -> str:
    return (
        "COMSOL startup text\n"
        f"M3C0B|configuration|step_s={step_s:.17g}|brownian_active=false"
        "|saffman_active=false|dynamic_charge_active=true|physics=fptas"
        "|study=stdAS100|solution=solPilot|output_times=3|particle_rows=2"
        "|source_model=source_copy.mph|model_saved=false|store_particle_status=true"
        "|store_extra=false|wall_accuracy_order=1|integrator=classical_rk4"
        "|integrator_order=4|relative_tolerance=1e-8|background_study=stdASf"
        "|background_solution=sol26"
        "|deterministic_contributions=electric,relative_flow_ion_drag,epstein_drag,"
        "waldmann_thermophoresis,free_molecular_lift_sensitivity,dielectrophoresis,"
        "gravity_buoyancy\n"
    )


def _record(particle_id: int, time_s: float, error: float, *, status: int) -> list[float]:
    values = [0.0] * len(m3c0b_pilot.COLUMNS)
    values[0:12] = [
        float(particle_id),
        time_s,
        0.1 + particle_id * 1e-3 + 0.5 * time_s + error,
        0.02 - 0.25 * time_s + 2.0 * error,
        0.3 + error,
        -0.2 + error,
        -1.0 + 0.5 * error,
        float(status),
        4.0 if particle_id == 2 else 1.0,
        0.025 if particle_id == 2 else 0.0,
        4.0 + error,
        2.0,
    ]
    values[12] = 2.0
    values[14] = 3.0
    values[16] = -1.0
    values[18] = 0.5
    values[20] = 0.25
    values[22] = -0.125
    values[24] = 0.375
    values[26] = 5.0
    values[27] = 0.0
    values[28] = 2.5
    values[29] = 0.0
    values[32] = 300.0
    values[33] = 1.0
    values[34] = 2e-5
    values[35] = 0.02
    values[36] = 1e-3
    values[42] = 1e15
    values[43] = 1e15
    values[44] = 6e-26
    values[47] = 0.03
    values[50] = 1e-3
    values[51] = 2e-3
    values[57] = 1e-7
    if status == 4:
        for index in (*range(2, 7), *range(10, 58)):
            values[index] = float("nan")
    return values


def _write_step(directory: Path, step_s: float, error: float) -> None:
    directory.mkdir(parents=True)
    rows: dict[str, list[list[float]]] = {
        "state": [],
        "force": [],
        "neutral": [],
        "electric": [],
        "plasma": [],
    }
    for particle_id in (1, 2):
        wide = {name: [] for name in rows}
        for time_s in (0.0, 0.01, 0.03):
            record = _record(particle_id, time_s, error, status=1)
            wide["state"].extend(record[:12])
            wide["force"].extend((record[0], record[1], *record[12:26]))
            wide["neutral"].extend((record[0], record[1], *record[30:37], *record[52:58]))
            wide["electric"].extend((record[0], record[1], *record[37:42], record[48]))
            wide["plasma"].extend((record[0], record[1], *record[42:48], *record[49:52]))
        for name in rows:
            rows[name].append(wide[name])
    for name, table_rows in rows.items():
        with (directory / f"{name}_raw_wide.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["% synthetic M3-C0b fixture"])
            writer.writerows(table_rows)


def _small_pilot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    monkeypatch.setattr(m3c0b_pilot, "EXPECTED_PARTICLES", 2)
    monkeypatch.setattr(m3c0b_pilot, "EXPECTED_OUTPUT_TIMES", 3)
    monkeypatch.setattr(m3c0b_pilot, "EXPECTED_END_S", 0.03)
    errors = {"dt_0p625us": 16e-6, "dt_0p3125us": 1e-6, "dt_0p15625us": 0.0625e-6}
    for name, step_s in m3c0b_pilot.STEP_S.items():
        _write_step(tmp_path / name, step_s, errors[name])
    (tmp_path / "comsol_process.log").write_text(
        "".join(_receipt(step_s) for step_s in m3c0b_pilot.STEP_S.values()),
        encoding="utf-8",
    )
    config = {
        "evaluation_revision": 6,
        "source_model": {"relative_path": "model/reference.mph", "sha256": "00" * 32},
        "case": {
            "workflow": "caseA",
            "diameter_m": 1e-7,
            "ion_drag_revision": "relative_flow_screened_collection_orbital_v1",
            "particle_count": 2,
            "output_times": 3,
            "time_end_s": 0.03,
            "fixed_rk4_steps_s": list(m3c0b_pilot.STEP_S.values()),
        },
        "physics": {
            "brownian_active": False,
            "saffman_active": False,
            "dynamic_charge_active": True,
            "deterministic_contributions": list(m3c0b_pilot.EXPECTED_DETERMINISTIC_CONTRIBUTIONS),
        },
        "raw_export": {
            "revision": 3,
            "tables": list(m3c0b_pilot.RAW_EXPORT_TABLES),
            "join_key": ["particle_id", "time_s"],
        },
        "acceptance": {
            "minimum_observed_order": 0.75,
            "maximum_fine_pair_position_displacement_relative_l2": 1.0,
            "maximum_fine_pair_velocity_relative_l2": 1.0,
            "maximum_fine_pair_charge_relative_l2": 1.0,
            "require_all_records_active": True,
        },
        "numerics": {
            "integrator": "classical_rk4",
            "integrator_order": 4,
            "relative_tolerance": 1e-8,
            "wall_accuracy_order": 1,
            "store_particle_status": True,
            "store_extra": False,
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return tmp_path, config_path


def _first_csv_record(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as stream:
        return next(csv.DictReader(stream))


def _assert_rectangular_csv(path: Path) -> None:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        assert all(len(row) == len(header) for row in reader)


def _assert_all_artifact_widths(root: Path) -> None:
    for step in m3c0b_pilot.STEP_S:
        directory = root / step
        summary = json.loads((directory / "step_summary.json").read_text(encoding="utf-8"))
        for artifact in summary["artifacts"]:
            _assert_rectangular_csv(directory / artifact)


def _translate_raw_positions(root: Path, offset_m: float) -> None:
    width = len(m3c0b_pilot.STATE_RAW_COLUMNS)
    for step in m3c0b_pilot.STEP_S:
        path = root / step / "state_raw_wide.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.reader(stream))
        for row in rows[1:]:
            for frame in range(3):
                for component in (2, 3):
                    index = frame * width + component
                    row[index] = str(float(row[index]) + offset_m)
        with path.open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream, lineterminator="\n").writerows(rows)


def _assert_convergence(manifest: dict[str, Any]) -> None:
    assert manifest["brownian_active"] is False
    assert set(manifest["runs"]) == set(m3c0b_pilot.STEP_S)
    convergence = manifest["self_convergence"]
    assert convergence["status"] == "PASS"
    assert convergence["observed_order_base_2"]["position_all_records"] == pytest.approx(4.0)
    assert convergence["dt_0p625us_vs_0p3125us"]["status_agreement_fraction"] == 1.0
    assert convergence["dt_0p625us_vs_0p3125us"]["position_finite_records"] == 6
    assert manifest["configuration"]["raw_export"]["revision"] == 3
    events = convergence["dt_0p625us_vs_0p3125us"]["first_nonactive_event_comparison"]
    assert events["both_event_count"] == 0
    assert events["type_agreement_fraction"] is None
    assert events["event_time_rms_s"] is None
    assert events["event_time_max_s"] is None
    assert events["missing_on_one_side_count"] == 0


def _assert_fine_artifacts(fine: Path) -> None:
    assert (fine / "trajectory_reference.csv").read_text(encoding="utf-8").count("\n") == 7
    assert (fine / "event_observations.csv").read_text(encoding="utf-8").count("\n") == 1
    receipt = json.loads((fine / "run_receipt.json").read_text(encoding="utf-8"))
    assert receipt["configuration"]["brownian_active"] == "false"
    summary = json.loads((fine / "step_summary.json").read_text(encoding="utf-8"))
    assert summary["particles_with_observed_nonactive_state"] == 0
    assert set(summary["raw"]) == {"state", "force", "neutral", "electric", "plasma"}
    force = _first_csv_record(fine / "force_reference.csv")
    rhs = _first_csv_record(fine / "rhs_reference.csv")
    assert "gravity_buoyancy_force_r_N" in force
    assert "gravity_force_r_N" not in force
    assert float(force["total_force_r_N"]) == 5.0
    assert float(rhs["velocity_rate_r_m_per_s2"]) == 2.5


def test_normalizes_three_step_pilot_and_characterizes_convergence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, config = _small_pilot(tmp_path, monkeypatch)

    combined = list(m3c0b_pilot.iter_combined_records(root / "dt_0p15625us"))

    manifest = m3c0b_pilot.normalize(root, config)

    _assert_convergence(manifest)
    _assert_fine_artifacts(root / "dt_0p15625us")
    _assert_all_artifact_widths(root)
    assert len(combined) == 6
    trajectory = _first_csv_record(root / "dt_0p15625us/trajectory_reference.csv")
    force = _first_csv_record(root / "dt_0p15625us/force_reference.csv")
    assert combined[0]["r_m"] == float(trajectory["r_m"])
    assert combined[0]["charge_number_e"] == float(trajectory["charge_number_e"])
    assert combined[0]["epstein_force_r_N"] == float(force["epstein_force_r_N"])


def test_revision_six_records_explicit_numerics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, config = _small_pilot(tmp_path, monkeypatch)

    manifest = m3c0b_pilot.normalize(root, config)

    assert manifest["tool_revision"] == "m3c0b_deterministic_pilot_v6"
    assert manifest["configuration"]["numerics"] == {
        "integrator": "classical_rk4",
        "integrator_order": 4,
        "relative_tolerance": 1e-8,
        "wall_accuracy_order": 1,
        "store_particle_status": True,
        "store_extra": False,
    }
    receipt = json.loads((root / "dt_0p15625us/run_receipt.json").read_text(encoding="utf-8"))
    assert receipt["configuration"]["relative_tolerance"] == "1e-8"


def test_position_displacement_metric_is_translation_invariant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base, base_config = _small_pilot(tmp_path / "base", monkeypatch)
    translated, translated_config = _small_pilot(tmp_path / "translated", monkeypatch)
    _translate_raw_positions(translated, 10.0)

    base_result = m3c0b_pilot.normalize(base, base_config)["self_convergence"]
    translated_result = m3c0b_pilot.normalize(translated, translated_config)["self_convergence"]

    pair = "dt_0p3125us_vs_0p15625us"
    metric = "position_displacement_relative_l2"
    assert translated_result[pair][metric] == pytest.approx(base_result[pair][metric], rel=1e-8)


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        ("|integrator=classical_rk4", "", "missing receipt keys: integrator"),
        ("relative_tolerance=1e-8", "relative_tolerance=1e-4", "relative_tolerance"),
    ],
)
def test_revision_six_rejects_missing_or_wrong_numerics_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    old: str,
    new: str,
    expected: str,
) -> None:
    root, config = _small_pilot(tmp_path, monkeypatch)
    log = root / "comsol_process.log"
    log.write_text(log.read_text(encoding="utf-8").replace(old, new, 1), encoding="utf-8")

    with pytest.raises(ValueError, match=expected):
        m3c0b_pilot.normalize(root, config)


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        ("brownian_active=false", "brownian_active=true", "brownian_active=false"),
        ("store_extra=false", "store_extra=true", "store_extra=false"),
    ],
)
def test_rejects_inconsistent_configuration_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    old: str,
    new: str,
    expected: str,
) -> None:
    root, config = _small_pilot(tmp_path, monkeypatch)
    log = root / "comsol_process.log"
    text = log.read_text(encoding="utf-8")
    log.write_text(text.replace(old, new, 1), encoding="utf-8")

    with pytest.raises(ValueError, match=expected):
        m3c0b_pilot.normalize(root, config)


def test_rejects_noncanonical_particle_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, config = _small_pilot(tmp_path, monkeypatch)
    raw = root / "dt_0p625us/force_raw_wide.csv"
    lines = raw.read_text(encoding="utf-8").splitlines()
    values = next(csv.reader([lines[2]]))
    for frame in range(3):
        values[frame * len(m3c0b_pilot.FORCE_RAW_COLUMNS)] = "3"
    lines[2] = ",".join(values)
    raw.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="particle ID/time mismatch"):
        m3c0b_pilot.normalize(root, config)
