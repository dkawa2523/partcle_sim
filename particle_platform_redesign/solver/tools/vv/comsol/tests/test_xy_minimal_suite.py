from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

TOOL = Path(__file__).resolve().parents[1] / "evaluate_xy_minimal_suite.py"


def _load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("evaluate_xy_minimal_suite", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _load_tool()
EVENT_TIME_S = tool.EVENT_TIME_S
_analytic = tool._analytic
_validate_lineage = tool._validate_lineage


def test_checked_xy_evidence_is_source_and_raw_bound() -> None:
    solver_root = Path(__file__).resolve().parents[4]
    evidence = solver_root / "evidence" / "xy" / "cartesian_minimal_suite_v1"
    lineage = _validate_lineage(evidence / "raw", evidence / "run_receipt.json")
    summary = json.loads(
        (evidence / "evaluation" / "comparison_summary.json").read_text(encoding="utf-8")
    )

    assert lineage.configuration_record_count == 21
    assert lineage.comsol_version == "COMSOL 6.4.0.429"
    assert summary["scientific_status"] == "PASS"
    assert set(summary["judgments"].values()) == {"PASS"}


def test_specular_analytic_reference_preserves_tangential_motion() -> None:
    trajectory = _analytic("specular")
    after = trajectory.time_s > EVENT_TIME_S

    np.testing.assert_allclose(
        trajectory.position_m[after, 1],
        0.4 + 0.1 * trajectory.time_s[after],
        rtol=0.0,
        atol=2.0e-16,
    )
    np.testing.assert_array_equal(trajectory.velocity_m_s[after, 0], -0.8)
