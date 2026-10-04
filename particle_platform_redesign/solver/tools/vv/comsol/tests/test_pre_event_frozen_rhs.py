from __future__ import annotations

import math
from pathlib import Path

import pytest
from tools.vv.comsol.evaluate_pre_event_frozen_rhs import (
    load_config,
    write_report,
)

CASE = Path(__file__).resolve().parents[1] / "cases" / "m3c1_caseA_100nm_frozen_rhs_v1.json"


def test_frozen_protocol_keeps_formula_model_and_applicability_claims_separate() -> None:
    config = load_config(CASE)

    assert config.step_directory == "dt_0p15625us"
    assert config.expected_records == 13_202
    assert config.raw_sha256.keys() == {
        "state_raw_wide.csv",
        "force_raw_wide.csv",
        "neutral_raw_wide.csv",
        "electric_raw_wide.csv",
        "plasma_raw_wide.csv",
    }
    assert config.thermophoretic_primitive_authority.startswith("unrecovered_")
    assert config.known_constant_convention_models == {
        "dynamic_charge",
        "relative_flow_ion_drag",
        "dielectrophoresis",
    }
    assert config.epstein_delta == pytest.approx(1.0 + 0.9 * math.pi / 8.0)


def test_report_writer_is_no_clobber(tmp_path: Path) -> None:
    output = tmp_path / "evidence"
    write_report({"status": "PASS"}, output)

    with pytest.raises(FileExistsError):
        write_report({"status": "FAIL"}, output)
