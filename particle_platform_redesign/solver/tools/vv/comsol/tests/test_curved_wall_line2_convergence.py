"""Analytic convergence check for a smooth wall represented by line2 facets."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

TOOL = Path(__file__).resolve().parents[1] / "evaluate_curved_wall_line2_convergence.py"


def _load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("curved_wall_line2_convergence", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _load_tool()
FACET_COUNTS = tool.FACET_COUNTS
PARTICLE_COUNT = tool.PARTICLE_COUNT
evaluate = tool.evaluate


def test_circle_line2_first_hit_and_reflection_converge(tmp_path: Path) -> None:
    output = tmp_path / "curved_wall"

    summary = evaluate(output)

    assert summary["scientific_status"] == "PASS"
    assert summary["facets"] == list(FACET_COUNTS)
    assert summary["particle_count_per_level"] == PARTICLE_COUNT
    assert all(summary["gates"].values())

    levels = summary["levels"]
    for metric in (
        "rms_time_error_s",
        "rms_point_error_m",
        "rms_normal_error",
        "rms_reflected_velocity_error_m_s",
    ):
        error = [level[metric] for level in levels]
        assert error == sorted(error, reverse=True)

    orders = summary["observed_orders"]
    assert orders["rms_time"] == pytest.approx(2.0, abs=0.05)
    assert orders["rms_point"] == pytest.approx(2.0, abs=0.05)
    assert orders["rms_normal"] == pytest.approx(1.0, abs=0.05)
    assert orders["rms_reflected_velocity"] == pytest.approx(1.0, abs=0.05)

    rows = list(csv.DictReader((output / "metrics.csv").open(encoding="utf-8")))
    assert [int(row["facets"]) for row in rows] == list(FACET_COUNTS)
    persisted = json.loads((output / "comparison_summary.json").read_text(encoding="utf-8"))
    assert persisted == summary
    assert "COMSOL curved-geometry equivalence" in persisted["not_tested"]
