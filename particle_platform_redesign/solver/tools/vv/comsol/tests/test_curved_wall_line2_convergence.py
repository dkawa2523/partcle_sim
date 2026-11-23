"""Analytic convergence check for a smooth wall represented by line2 facets."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import cast

import pytest
from tools.vv.comsol.evaluate_curved_wall_line2_convergence import (
    FACET_COUNTS,
    PARTICLE_COUNT,
    evaluate,
)


def test_circle_line2_first_hit_and_reflection_converge(tmp_path: Path) -> None:
    output = tmp_path / "curved_wall"

    summary = evaluate(output)

    assert summary["scientific_status"] == "PASS"
    assert summary["facets"] == list(FACET_COUNTS)
    assert summary["particle_count_per_level"] == PARTICLE_COUNT
    gates = cast(dict[str, bool], summary["gates"])
    assert all(gates.values())

    levels = cast(list[dict[str, object]], summary["levels"])
    for metric in (
        "rms_time_error_s",
        "rms_point_error_m",
        "rms_normal_error",
        "rms_reflected_velocity_error_m_s",
    ):
        error = [cast(float, level[metric]) for level in levels]
        assert error == sorted(error, reverse=True)

    orders = cast(dict[str, float], summary["observed_orders"])
    assert orders["rms_time"] == pytest.approx(2.0, abs=0.05)
    assert orders["rms_point"] == pytest.approx(2.0, abs=0.05)
    assert orders["rms_normal"] == pytest.approx(1.0, abs=0.05)
    assert orders["rms_reflected_velocity"] == pytest.approx(1.0, abs=0.05)

    rows = list(csv.DictReader((output / "metrics.csv").open(encoding="utf-8")))
    assert [int(row["facets"]) for row in rows] == list(FACET_COUNTS)
    persisted = json.loads((output / "comparison_summary.json").read_text(encoding="utf-8"))
    assert persisted == summary
    assert "COMSOL curved-geometry equivalence" in persisted["not_tested"]
