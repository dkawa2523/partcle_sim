from __future__ import annotations

from pathlib import Path

import pytest

from particle_tracer_unified import load_case, simulate, validate_case, write_result

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = (
    ROOT / "examples" / "v02_minimal" / "run_config.yaml",
    ROOT / "examples" / "v02_minimal_3d" / "run_config.yaml",
)


@pytest.mark.parametrize("config_path", EXAMPLES, ids=("regular_2d", "regular_3d"))
def test_canonical_examples_pass_preflight_and_write_only_standard_artifacts(
    config_path: Path,
    tmp_path: Path,
) -> None:
    case = load_case(config_path)
    report = validate_case(case, detail="summary")
    assert report.passed, report.to_dict()
    initial = report.checks["initial_particles"]
    assert initial["support_scope"] == "spatial_only"
    assert initial["sample_time_scope"] == "particle_release_time"
    assert initial["geometry_passed"] is True
    boundary_support = report.checks["provider_boundary_support"]
    assert boundary_support["support_scope"] == "spatial_only"
    assert "checked_times_s" not in boundary_support

    result = simulate(case)
    output = tmp_path / config_path.parent.name
    write_result(result, output)

    assert {path.name for path in output.iterdir()} == {
        "final_particles.csv",
        "run_summary.json",
        "wall_summary.csv",
    }
