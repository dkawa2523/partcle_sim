from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
from tools.vv.comsol import normalize_m3c1_common_p1_reference as common
from tools.vv.comsol.prepare_m3c1_common_p1_tables import COMPONENT_EXPORTS


def test_initial_primitive_contract_covers_the_prepared_22_components_in_order() -> None:
    assert len(common.PRIMITIVE_PROBE_COLUMNS) == 22
    assert tuple(raw for raw, _ in common.PRIMITIVE_PROBE_COLUMNS) == common.PRIMITIVE_COLUMNS[2:]
    assert tuple(probe for _, probe in common.PRIMITIVE_PROBE_COLUMNS) == tuple(
        export.probe_column for export in COMPONENT_EXPORTS
    )


def _state(particle: int, frame: int, *, offset_r: float = 0.0) -> list[float]:
    time = frame * 1.0e-5
    r = 0.1 + particle * 0.01 + offset_r
    z = 0.02 + particle * 0.001
    vr = 0.01 * particle
    vz = 0.02 * particle
    return [particle, time, r, z, vr, vz, -1.0, 1.0, 1.0, 0.0, 3.0, 2.0, vr, vz, -1.0]


def _force(particle: int, frame: int) -> list[float]:
    time = frame * 1.0e-5
    components = [1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    return [particle, time, *components, 1.0, 2.0, 0.5, 1.0]


def _primitive(particle: int, frame: int, *, offset: float = 0.0) -> list[float]:
    values = [float(index + 1) for index in range(22)]
    values[0] += offset
    return [particle, frame * 1.0e-5, *values]


def _write_wide(path: Path, rows: list[list[list[float]]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        for particle_rows in rows:
            writer.writerow([value for record in particle_rows for value in record])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_prepared_table_validation(root: Path) -> None:
    artifact_paths = [root / "common_p1_release_probes.csv"]
    for index in range(25):
        path = root / f"prepared_{index:02d}.txt"
        path.write_text(f"artifact {index}\n", encoding="utf-8")
        artifact_paths.append(path)
    artifacts = {
        path.name: {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
        for path in artifact_paths
    }
    receipt_path = root / "common_p1_table_receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tool_revision": "m3c1_full_physics_common_p1_tables_v1",
                "component_count": 22,
                "release": {"functions": [{}, {}, {}]},
                "artifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )
    total_size = sum(path.stat().st_size for path in artifact_paths)
    phase = {"status": "PASS", "artifact_count": 26, "total_size_bytes": total_size}
    validation_path = root / "prepared_table_validation.json"
    validation_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "PASS",
                "criterion": "test fixture receipt hash and size match",
                "receipt_sha256": _sha256(receipt_path),
                "pre_comsol": phase,
                "post_comsol": phase,
                "artifacts": [{"path": name, **record} for name, record in artifacts.items()],
            }
        ),
        encoding="utf-8",
    )


def _configuration_line(step: float) -> str:
    fields = {
        "step_s": f"{step:.17g}",
        "brownian_active": "false",
        "saffman_active": "false",
        "dynamic_charge_active": "true",
        "native_thermophoresis_active": "false",
        "common_heat_flux_force_active": "true",
        "field_source": "canonical_exact_connectivity_P1_sectionwise",
        "initial_state_source": "candidate_realized_source_table",
        "primitive_function_count": "22",
        "deterministic_contributions": common.DETERMINISTIC_CONTRIBUTIONS,
        "integrator": "classical_rk4",
        "integrator_order": "4",
        "relative_tolerance": "1e-8",
        "wall_accuracy_order": "1",
        "store_particle_status": "true",
        "store_extra": "false",
        "physics": "fptas",
        "study": "synthetic",
        "solution": "synthetic",
        "output_times": "3",
        "particle_rows": "2",
        "source_model": "source_copy.mph",
        "model_saved": "false",
    }
    return "M3C1_COMMON_P1|configuration|" + "|".join(
        f"{key}={value}" for key, value in fields.items()
    )


def _write_fixture(
    root: Path, *, initial_offset: float = 0.0, primitive_initial_offset: float = 0.0
) -> Path:
    root.mkdir()
    config = {
        "evaluation_revision": 1,
        "case": {
            "particle_count": 2,
            "output_times": 3,
            "time_start_s": 0.0,
            "time_end_s": 2.0e-5,
            "output_interval_s": 1.0e-5,
            "fixed_rk4_steps_s": list(common.STEP_S.values()),
        },
        "validation": {
            "require_all_records_active": True,
            "initial_state_roundoff_multiplier": 4096.0,
            "initial_primitive_roundoff_multiplier": 4096.0,
        },
        "raw_export": {"revision": 1, "tables": list(common.RAW_TABLES)},
    }
    config_path = root / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with (root / "common_p1_release_probes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "particle_id",
                "r_m",
                "z_m",
                "velocity_r_m_per_s",
                "velocity_z_m_per_s",
                "charge_number",
                *(probe_column for _, probe_column in common.PRIMITIVE_PROBE_COLUMNS),
            )
        )
        for particle in (1, 2):
            row = _state(particle, 0)
            writer.writerow((particle, row[2], row[3], row[4], row[5], row[6], *range(1, 23)))
    _write_prepared_table_validation(root)
    (root / "comsol_process.log").write_text(
        "\n".join(_configuration_line(step) for step in common.STEP_S.values()),
        encoding="utf-8",
    )
    for name in common.STEP_S:
        directory = root / name
        directory.mkdir()
        state_rows = [
            [
                _state(particle, frame, offset_r=initial_offset if frame == 0 else 0.0)
                for frame in range(3)
            ]
            for particle in (1, 2)
        ]
        force_rows = [[_force(particle, frame) for frame in range(3)] for particle in (1, 2)]
        primitive_rows = [
            [
                _primitive(
                    particle,
                    frame,
                    offset=primitive_initial_offset if particle == 1 and frame == 0 else 0.0,
                )
                for frame in range(3)
            ]
            for particle in (1, 2)
        ]
        _write_wide(directory / "state_raw_wide.csv", state_rows)
        _write_wide(directory / "force_raw_wide.csv", force_rows)
        _write_wide(directory / "primitive_raw_wide.csv", primitive_rows)
    return config_path


def _small_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(common, "EXPECTED_PARTICLES", 2)
    monkeypatch.setattr(common, "EXPECTED_FRAMES", 3)
    monkeypatch.setattr(common, "EXPECTED_END_S", 2.0e-5)


def test_normalizes_common_p1_reference_and_checks_initial_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _small_protocol(monkeypatch)
    root = tmp_path / "reference"
    config = _write_fixture(root)

    summary = common.normalize(root, config)

    assert summary["status"] == "COMPLETE"
    assert summary["scope"] == {
        "particles": 2,
        "frames": 3,
        "time_window_s": [0.0, 2.0e-5],
        "output_interval_s": 1.0e-5,
    }
    assert summary["prepared_table_artifact_validation"]["status"] == "PASS"
    assert summary["prepared_table_artifact_validation"]["artifact_count"] == 26
    assert all(run["rows"] == 6 for run in summary["runs"].values())
    assert all(run["initial_state"]["status"] == "PASS" for run in summary["runs"].values())
    assert all(run["initial_primitives"]["status"] == "PASS" for run in summary["runs"].values())
    assert all(
        run["initial_primitives"]["checked_value_count"] == 44 for run in summary["runs"].values()
    )
    assert all(run["event_count"] == 0 for run in summary["runs"].values())


def test_normalizes_one_staged_campaign_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _small_protocol(monkeypatch)
    root = tmp_path / "reference"
    config_path = _write_fixture(root)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["case"]["diameter_m"] = 1.0e-8
    contribution = "relative_flow_screened_collection_orbital_ion_drag"
    contributions = common.DETERMINISTIC_CONTRIBUTIONS.split(",")
    contributions[1] = contribution
    config["physics"] = {"deterministic_contributions": contributions}
    config_path.write_text(json.dumps(config), encoding="utf-8")
    spec = {
        "case_id": "caseA_10nm_relative_flow",
        "diameter_nm": "10",
        "ion_drag_revision": "relative_flow_screened_collection_orbital_aggregate_ion_v1",
        "deterministic_contribution_name": contribution,
    }
    (root / "run_spec.properties").write_text(
        "".join(f"{key}={value}\n" for key, value in spec.items()),
        encoding="ascii",
    )
    log_path = root / "comsol_process.log"
    configuration_lines = log_path.read_text(encoding="utf-8").replace(
        common.DETERMINISTIC_CONTRIBUTIONS,
        ",".join(contributions),
    )
    campaign_line = "M3C1_COMMON_P1|campaign_spec|" + "|".join(
        f"{key}={value}" for key, value in spec.items()
    )
    log_path.write_text(configuration_lines + "\n" + campaign_line, encoding="utf-8")

    summary = common.normalize(root, config_path)

    assert summary["status"] == "COMPLETE"
    assert summary["campaign_spec"] == spec


def test_rejects_comsol_initial_state_that_differs_from_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _small_protocol(monkeypatch)
    root = tmp_path / "reference"
    config = _write_fixture(root, initial_offset=1.0e-6)

    with pytest.raises(ValueError, match="t=0 state differs"):
        common.normalize(root, config)


def test_rejects_comsol_initial_primitive_that_differs_from_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _small_protocol(monkeypatch)
    root = tmp_path / "reference"
    config = _write_fixture(root, primitive_initial_offset=1.0e-6)

    with pytest.raises(ValueError, match="t=0 primitive gas_density_kg_per_m3 differs"):
        common.normalize(root, config)


def test_rejects_prepared_table_changed_after_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _small_protocol(monkeypatch)
    root = tmp_path / "reference"
    config = _write_fixture(root)
    (root / "prepared_00.txt").write_text("changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="prepared-table hash or size differs"):
        common.normalize(root, config)
