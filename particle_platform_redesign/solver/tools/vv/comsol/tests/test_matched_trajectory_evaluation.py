from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from tools.vv.comsol.evaluate_matched_trajectory import (
    characterize,
    compare,
    register_budget,
)


def _write_trajectory(path: Path, *, dt_s: float, coefficient: float) -> None:
    error_scale = coefficient * dt_s * dt_s
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "particle_id",
                "time_s",
                "r_m",
                "z_m",
                "velocity_r_m_per_s",
                "velocity_z_m_per_s",
                "charge_number_e",
                "lifecycle",
            ]
        )
        for frame in range(41):
            time_s = frame * 1.0e-5
            error = error_scale * frame / 40.0
            for particle_id in range(1, 288):
                writer.writerow(
                    [
                        particle_id,
                        time_s,
                        0.14 + particle_id * 1.0e-5 + 0.01 * time_s + error,
                        0.023 + particle_id * 2.0e-6 + 0.02 * time_s - 0.5 * error,
                        0.01 + particle_id * 1.0e-6 + error,
                        0.02 - particle_id * 1.0e-6 - 0.25 * error,
                        -1.0,
                        "active",
                    ]
                )


def _characterize_side(tmp_path: Path, label: str, coefficient: float) -> tuple[Path, Path]:
    paths: list[Path] = []
    for name, dt_s in (("10us", 1.0e-5), ("5us", 5.0e-6), ("2p5us", 2.5e-6)):
        path = tmp_path / f"{label}_{name}.csv"
        _write_trajectory(path, dt_s=dt_s, coefficient=coefficient)
        paths.append(path)
    report = characterize(label, *paths)
    output = tmp_path / f"{label}_self.json"
    output.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    return output, paths[-1]


def test_budget_is_registered_before_locked_cross_comparison(tmp_path: Path) -> None:
    candidate_report, candidate_fine = _characterize_side(tmp_path, "candidate", 1.0)
    reference_report, reference_fine = _characterize_side(tmp_path, "comsol_reference", 1.5)

    budget = register_budget(candidate_report, reference_report)
    assert budget["status"] == "REGISTERED"
    budget_path = tmp_path / "budget.json"
    budget_path.write_text(json.dumps(budget, sort_keys=True), encoding="utf-8")

    comparison = compare(budget_path, candidate_fine, reference_fine)

    assert comparison["status"] == "PASS"
    claim = comparison["accuracy_claim"]
    assert isinstance(claim, dict)
    assert claim["same_accuracy_for_hashed_case_and_time_window"].startswith("SUPPORTED")
    assert claim["comsol_universal_equal_accuracy"] == "NOT_CLAIMED"
    assert claim["boundary_accuracy"] == "NOT_TESTED_PRE_EVENT_WINDOW"

    with candidate_fine.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="hash differs"):
        compare(budget_path, candidate_fine, reference_fine)
