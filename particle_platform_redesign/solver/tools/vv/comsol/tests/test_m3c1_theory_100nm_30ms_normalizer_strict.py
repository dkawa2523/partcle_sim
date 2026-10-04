from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import pytest
from tools.vv.comsol import normalize_m3c1_theory_100nm_30ms as normalize


def _protocol() -> normalize.Protocol:
    return normalize.Protocol(
        config_path=Path("formal.json"),
        config_sha256=normalize.EXPECTED_SHARED_CONFIG_SHA256,
        particle_count=1,
        output_times_s=(0.0, 0.1),
        steps_s_by_workflow={"caseA": (0.04, 0.02, 0.01), "caseP": (0.04, 0.02, 0.01)},
        charge_lipschitz_s_inv_by_workflow={"caseA": 1.0, "caseP": 1.0},
        maximum_dt_charge_lipschitz=0.5,
        initial_ulp_multiplier=4096.0,
        roundoff_multiplier=4096.0,
    )


def _state(
    time_s: float,
    status: int = 1,
    final: int = 1,
    state: tuple[float, float, float, float, float] = (0.1, 0.2, 0.0, 0.0, -1.0),
    stop_time_s: float | None = None,
) -> normalize.StateRecord:
    return normalize.StateRecord(
        particle_id=1,
        time_s=time_s,
        state=state,
        current_status=status,
        final_status=final,
        stop_time_s=stop_time_s,
        mass_kg=2.0,
        sampled_release=(0.0, 0.0, -1.0),
    )


def _write_wide(path: Path, records: list[list[float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerow(
            [value for record in records for value in record]
        )


@pytest.mark.parametrize(
    "column_count", [normalize.FORCE_COLUMN_COUNT, normalize.PRIMITIVE_COLUMN_COUNT]
)
def test_active_auxiliary_nonfinite_is_rejected(tmp_path: Path, column_count: int) -> None:
    path = tmp_path / "raw.csv"
    first = [1.0, 0.0, *([1.0] * (column_count - 2))]
    second = [1.0, 0.1, *([1.0] * (column_count - 2))]
    second[-1] = math.nan
    _write_wide(path, [first, second])

    with pytest.raises(ValueError, match="active auxiliary record is nonfinite"):
        normalize._read_auxiliary_table(
            path, _protocol(), column_count, [[_state(0.0), _state(0.1)]]
        )


def test_force_sum_and_acceleration_identity_are_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "force.csv"
    contributions_r = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
    contributions_z = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5)
    values: list[float] = [1.0, 0.0]
    for radial, axial in zip(contributions_r, contributions_z, strict=True):
        values.extend((radial, axial))
    total_r = sum(contributions_r)
    total_z = sum(contributions_z)
    values.extend((total_r, total_z, total_r / 2.0, total_z / 2.0))
    histories = [[_state(0.0)]]
    forces = {1: [tuple(values)]}

    result = normalize._validate_force_consistency(
        path, histories, forces, roundoff_multiplier=4096.0
    )
    assert result["status"] == "PASS"

    corrupted = values.copy()
    corrupted[16] += 1.0e-6
    with pytest.raises(ValueError, match="total force differs"):
        normalize._validate_force_consistency(
            path, histories, {1: [tuple(corrupted)]}, roundoff_multiplier=4096.0
        )

    corrupted = values.copy()
    corrupted[18] += 1.0e-6
    with pytest.raises(ValueError, match="acceleration differs"):
        normalize._validate_force_consistency(
            path, histories, {1: [tuple(corrupted)]}, roundoff_multiplier=4096.0
        )


def test_retained_terminal_tail_is_invariant_but_escape_stays_sparse(tmp_path: Path) -> None:
    path = tmp_path / "state.csv"
    held = [
        _state(0.0, final=2),
        _state(0.1, status=2, final=2, stop_time_s=0.08),
        _state(
            0.2,
            status=2,
            final=2,
            state=(0.1, 0.2, 0.0, 0.0, -0.9),
            stop_time_s=0.08,
        ),
    ]
    with pytest.raises(ValueError, match="terminal charge tail changes"):
        normalize._validate_status_history(held, path, 4096.0)

    escaped = [
        _state(0.0, final=4),
        _state(
            0.1,
            status=4,
            final=4,
            state=(math.nan,) * 5,
            stop_time_s=0.08,
        ),
    ]
    for record in escaped:
        normalize._validate_record_state(record, path)
    normalize._validate_status_history(escaped, path, 4096.0)


def _formal_component_receipt(workflow: str) -> dict[str, object]:
    components = [
        dict(
            zip(
                ("field", "component", "file", "function", "probe_column", "unit"),
                values,
                strict=True,
            )
        )
        for values in normalize.PREPARED_COMPONENT_SPECS
    ]
    fields = [
        {
            "field": field,
            "components": list(components),
            "association": "node",
            "stored_basis": basis,
            "unit": unit,
        }
        for field, components, basis, unit in normalize.PREPARED_FIELD_SPECS
    ]
    functions = [
        {
            "file": filename,
            "function": function,
            "representation": "rectangular_grid_point_table_r_z_value",
            "source_column": column,
            "unit": unit,
        }
        for filename, function, column, unit in normalize.PREPARED_RELEASE_SPECS
    ]
    candidate_sha, content_hash = normalize.EXPECTED_CANDIDATE_IDENTITIES[workflow]
    return {
        "schema_version": 1,
        "tool_revision": "m3c1_full_physics_common_p1_tables_v1",
        "component_count": 22,
        "field_count": 17,
        "components": components,
        "fields": fields,
        "release": {
            "functions": functions,
            "grid": {"r_count": 41, "z_count": 7},
            "initial_state_columns": [
                "particle_id",
                "r_m",
                "z_m",
                "velocity_r_m_per_s",
                "velocity_z_m_per_s",
                "charge_number",
            ],
            "ordering": "particle_id_1_to_287_and_lexicographic_r_then_z",
            "probe_count": 287,
            "probe_field_sampling": "canonical_candidate_P1_sampler_at_realized_release_positions",
            "probe_file": "common_p1_release_probes.csv",
            "source_name": "particles",
        },
        "layout": {
            "type": "P1TriLayout",
            "name": "plasma",
            "node_count": 1987,
            "cell_count": 3779,
            "connectivity_width": 3,
            "connectivity_indexing_in_candidate": "zero_based",
            "connectivity_indexing_in_sectionwise_files": "one_based",
            "interpolation_contract": "first_order_triangular_sectionwise",
        },
        "candidate": {"file_sha256": candidate_sha, "content_hash": content_hash},
    }


def test_prepared_schema_requires_exact_components_and_candidate_identity() -> None:
    receipt = _formal_component_receipt("caseP")
    assert len(normalize._validate_prepared_schema(receipt, "caseP")) == 26

    receipt["components"] = list(receipt["components"])[1:]  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="component coverage differs"):
        normalize._validate_prepared_schema(receipt, "caseP")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_executed_code_receipt_binds_command_and_staged_java(tmp_path: Path) -> None:
    config = (
        Path(normalize.__file__).resolve().parent / "cases" / ("m3c1_theory_100nm_30ms_v2.json")
    )
    protocol, _ = normalize._load_protocol(config)
    contract = normalize._load_comsol_contract()
    directory = tmp_path / "executed_code_receipt"
    directory.mkdir()
    values = {
        "case_name": "caseP",
        "physics_tag": "fpt",
        "background_study": "std2",
        "background_solution": "sol2",
        "source_dataset": "part_P_100nm",
        "particle_geometry": "pgeom_fpt",
        "position_dof_r": "qr",
        "position_dof_z": "qz",
        "charge_state": "ZP",
        "run_keys": "coarse,medium,fine",
        "fixed_rk4_steps_s": "4.6875E-08,2.34375E-08,1.171875E-08",
        "charge_lipschitz_s_inv": "10403456.669581516",
        "maximum_dt_charge_lipschitz": "0.5",
        "common_config_sha256": normalize.EXPECTED_SHARED_CONFIG_SHA256,
        "comsol_config_sha256": normalize.EXPECTED_COMSOL_CONFIG_SHA256,
    }
    (directory / "run_spec.properties").write_text(
        "".join(f"{key}={value}\n" for key, value in values.items()), encoding="ascii"
    )
    (directory / "RunM3C1Theory100nm30ms.java").write_bytes(normalize._expected_staged_java(values))
    (directory / "RunM3C1Theory100nm30ms.class").write_bytes(b"compiled")
    (directory / "RunM3C1Theory100nm30ms.class.status").write_text(
        "123\nRunning\n", encoding="utf-8"
    )
    runner = Path(normalize.__file__).resolve().parent / (
        "run_m3c1_theory_100nm_30ms_reference.ps1"
    )
    assert _sha256(runner) == normalize.EXPECTED_EXECUTED_RUNNER_SHA256
    (directory / "run_m3c1_theory_100nm_30ms_reference.executed.ps1").write_bytes(
        runner.read_bytes()
    )
    artifacts = {
        path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in directory.iterdir()
    }
    receipt = {
        "schema_version": 1,
        "status": "CAPTURED_WHILE_PROCESS_ACTIVE",
        "workflow": "caseP",
        "active_process": {
            "pid": 123,
            "executable": "C:/Program Files/COMSOL/COMSOL64/Multiphysics_copy1/bin/win64/comsolbatch.exe",
            "input_class": "../RunM3C1Theory100nm30ms.class",
            "flags": ["-nosave", "-np", "1"],
            "batch_log": "../comsol_process.log",
        },
        "artifacts": artifacts,
    }
    receipt_path = directory / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    result = normalize._verify_executed_code_receipt(tmp_path, protocol, "caseP", contract)
    assert result["status"] == "PASS"

    receipt["active_process"]["flags"] = ["-nosave", "-np", "2"]  # type: ignore[index]
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="active COMSOL command differs"):
        normalize._verify_executed_code_receipt(tmp_path, protocol, "caseP", contract)
