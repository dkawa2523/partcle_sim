from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from tools.vv.comsol import evaluate_m3c1_field_localization as localization
from tools.vv.comsol.tests.common_p1_fixture import rematerialize_saved_localization_input, sha256

SOLVER_ROOT = Path(__file__).resolve().parents[4]
CASE = Path(__file__).resolve().parents[1] / "cases" / "m3c1_caseA_100nm_field_localization_v1.json"


@pytest.fixture(scope="module")
def current_configuration(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("localization-current-writer")
    payload = json.loads(CASE.read_text(encoding="utf-8"))
    record = payload["artifacts"]["candidate_input"]
    input_path = directory / "candidate_input.h5"
    info = rematerialize_saved_localization_input(SOLVER_ROOT / record["path"], input_path)
    record.update(path=str(input_path), sha256=sha256(input_path), content_hash=info.content_hash)
    config_path = directory / "localization.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return config_path


@pytest.fixture(scope="module")
def locked_report(current_configuration: Path) -> dict[str, Any]:
    return localization.evaluate(SOLVER_ROOT, current_configuration)


def _relative_l2(rows: list[dict[str, Any]], scope: str, quantity_key: str, quantity: str) -> float:
    match = [row for row in rows if row["scope"] == scope and row[quantity_key] == quantity]
    assert len(match) == 1
    return float(match[0]["relative_l2"])


def test_locked_slice_localizes_the_first_difference_before_integration(
    locked_report: dict[str, Any],
) -> None:
    assert locked_report["overall_status"] == "PASS_LOCALIZED"
    assert locked_report["first_difference_layer"] == "FIELD_REPRESENTATION_OR_SAMPLING"
    assert locked_report["comsol_rerun_performed"] is False
    assert all(gate["status"] == "PASS" for gate in locked_report["gates"])

    fields = locked_report["field_metrics"]
    rhs = locked_report["rhs_metrics"]
    assert len(fields) == 34
    assert len(rhs) == 18
    assert _relative_l2(fields, "identical_t0_state", "field", "electric_field") == pytest.approx(
        0.028620728726031667
    )
    assert _relative_l2(
        fields, "identical_t0_state", "field", "gradient_mean_e_squared"
    ) == pytest.approx(0.1259955413662082)
    assert _relative_l2(rhs, "identical_t0_state", "quantity", "charge_rate") == pytest.approx(
        0.049101396439728405
    )
    assert _relative_l2(
        rhs, "identical_t0_state", "quantity", "total_acceleration"
    ) == pytest.approx(0.03989452875541078)

    axis = locked_report["axis_projection_exclusion"]
    assert axis["minimum_t0_sample_r_m"] > axis["axis_incident_cell_maximum_r_m"]
    ppr = locked_report["ppr_node_binding"]
    assert ppr["matched_nodes"] == ppr["unique_reference_nodes"] == 1987
    assert ppr["after_axis_policy_relative_l2"] <= 1.0e-12


def test_configuration_rejects_result_tuned_acceptance(
    tmp_path: Path, current_configuration: Path
) -> None:
    payload = json.loads(current_configuration.read_text(encoding="utf-8"))
    payload["acceptance"]["roundoff_relative_l2"] = 1.0
    path = tmp_path / "result_tuned.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="acceptance policy differs"):
        localization.evaluate(SOLVER_ROOT, path)


def test_configuration_rejects_changed_locked_artifact(
    tmp_path: Path, current_configuration: Path
) -> None:
    payload = json.loads(current_configuration.read_text(encoding="utf-8"))
    payload["artifacts"]["candidate_input"]["sha256"] = "0" * 64
    path = tmp_path / "changed_hash.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="locked artifact hash differs"):
        localization.evaluate(SOLVER_ROOT, path)


def test_report_writer_is_no_clobber(locked_report: dict[str, Any], tmp_path: Path) -> None:
    output = tmp_path / "localization"
    localization.write_outputs(copy.deepcopy(locked_report), output)

    assert (output / localization.REPORT_FILE).is_file()
    with pytest.raises(FileExistsError):
        localization.write_outputs(copy.deepcopy(locked_report), output)
