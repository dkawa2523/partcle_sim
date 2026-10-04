from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from particle_tracer_unified._migration.charge import _migrate_charge
from particle_tracer_unified._migration.legacy import _source_generation_findings
from particle_tracer_unified.compare import near_wall_nohit
from particle_tracer_unified.compare.near_wall_nohit import analyze_near_wall_nohit
from tools import _reference_compare_inputs as comparison_inputs
from tools import _reference_compare_metrics as comparison_metrics


def test_source_generation_findings_keep_stable_evidence_order() -> None:
    config = {
        "paths": {"source_events_csv": "legacy-events.csv"},
        "source": {
            "law": "generated-volume",
            "default_law": "explicit_csv",
            "preprocess": {
                "boundary_release": "yes",
                "normal_velocity_policy": "flip",
            },
        },
    }
    particles = pd.DataFrame(
        {
            "source_law": ["explicit_csv", "volume"],
            "source_law_default": [None, "explicit_csv"],
            "source_law_override": ["boundary", None],
            "source_event_tag": ["", "seed-2"],
        }
    )

    assert _source_generation_findings(config, [("particles.csv", particles)]) == [
        "paths.source_events_csv",
        "source.law=generated-volume",
        "source.preprocess.boundary_release",
        "source.preprocess.normal_velocity_policy=flip",
        "particles.csv:row 3:source_law=volume",
        "particles.csv:row 2:source_law_override=boundary",
        "particles.csv:row 3:source_event_tag",
    ]


def test_charge_migration_keeps_materialized_default_and_warning_order() -> None:
    warnings: list[str] = []

    result = _migrate_charge(
        {
            "charge_model": {
                "enabled": "yes",
                "mode": "te-relaxation",
            }
        },
        warnings,
    )

    assert result == {
        "enabled": True,
        "mode": "te_relaxation",
        "parameters": {
            "te_relaxation_alpha": 2.5,
            "relaxation_time_s": 1.0e-6,
        },
    }
    assert warnings == [
        "solver.charge_model.te_relaxation_alpha was absent; "
        "materialized the legacy default 2.5",
        "solver.charge_model.relaxation_time_s was absent; "
        "materialized the legacy default 1e-6 s",
    ]


def _write_near_wall_particles(output_dir: Path) -> None:
    output_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "particle_id": 1,
                "source_part_id": 10,
                "final_state": "active_free_flight",
                "x_m": 1.0e-8,
                "y_m": 0.5,
                "vx_mps": -1.0,
                "vy_mps": 0.0,
                "nearest_boundary_part_id": 10,
                "nearest_boundary_distance_m": 1.0e-8,
                "sdf_m": -1.0e-8,
                "inside_geometry": 1,
                "field_support_status": "hard_invalid",
            },
            {
                "particle_id": 2,
                "source_part_id": 10,
                "final_state": "contact_sliding",
                "x_m": 2.0e-8,
                "y_m": 0.5,
                "vx_mps": 0.0,
                "vy_mps": 0.0,
                "nearest_boundary_part_id": 10,
                "nearest_boundary_distance_m": 2.0e-8,
                "sdf_m": -2.0e-8,
                "inside_geometry": 1,
                "field_support_status": "valid",
            },
            {
                "particle_id": 3,
                "source_part_id": 10,
                "final_state": "stuck",
                "x_m": 3.0e-8,
                "y_m": 0.5,
                "vx_mps": 0.0,
                "vy_mps": 0.0,
                "nearest_boundary_part_id": 10,
                "nearest_boundary_distance_m": 3.0e-8,
                "sdf_m": -3.0e-8,
                "inside_geometry": 1,
                "field_support_status": "valid",
            },
        ]
    ).to_csv(output_dir / "final_particles.csv", index=False)
    pd.DataFrame([{"particle_id": 2, "part_id": 10}]).to_csv(
        output_dir / "wall_events.csv", index=False
    )


def test_near_wall_analysis_filters_before_classification_and_summary(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "run"
    analysis_dir = tmp_path / "analysis"
    _write_near_wall_particles(output_dir)

    summary = analyze_near_wall_nohit(
        output_dir=output_dir,
        threshold_m=1.0e-6,
        analysis_output_dir=analysis_dir,
    )

    rows = pd.read_csv(analysis_dir / "near_wall_nohit_particles.csv")
    assert rows["particle_id"].tolist() == [1]
    assert rows["classification"].tolist() == ["field_support_issue"]
    assert summary["active_particle_count"] == 2
    assert summary["near_wall_active_count"] == 2
    assert summary["suspicious_particle_count"] == 1
    assert summary["classification_counts"] == {"field_support_issue": 1}
    assert summary["nearest_boundary_part_counts"] == [{"part_id": "10", "count": 1}]


def test_near_wall_analysis_samples_missing_geometry_from_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    pd.DataFrame(
        [
            {
                "particle_id": 9,
                "source_part_id": 4,
                "final_state": "active_free_flight",
                "x_m": 2.0e-8,
                "y_m": 0.25,
                "vx_mps": -3.0,
                "vy_mps": 0.0,
            }
        ]
    ).to_csv(output_dir / "final_particles.csv", index=False)
    pd.DataFrame(columns=["particle_id", "part_id"]).to_csv(
        output_dir / "wall_events.csv", index=False
    )
    runtime = SimpleNamespace(
        spatial_dim=2,
        geometry_provider=SimpleNamespace(
            geometry=SimpleNamespace(boundary_edges=None)
        ),
    )
    monkeypatch.setattr(near_wall_nohit, "_load_runtime", lambda _path: runtime)
    monkeypatch.setattr(
        near_wall_nohit,
        "sample_geometry_sdf",
        lambda _runtime, position: -float(position[0]),
    )
    monkeypatch.setattr(
        near_wall_nohit,
        "sample_geometry_part_id",
        lambda _runtime, _position: 41,
    )
    monkeypatch.setattr(
        near_wall_nohit,
        "sample_geometry_normal",
        lambda _runtime, _position: np.array([1.0, 0.0]),
    )
    monkeypatch.setattr(
        near_wall_nohit,
        "inside_geometry",
        lambda _runtime, _position, *, on_boundary_tol_m: on_boundary_tol_m == 0.0,
    )

    summary = analyze_near_wall_nohit(
        output_dir=output_dir,
        config_path=tmp_path / "config.yaml",
        threshold_m=1.0e-6,
        analysis_output_dir=tmp_path / "analysis",
    )

    rows = pd.read_csv(tmp_path / "analysis" / "near_wall_nohit_particles.csv")
    assert summary["geometry_available"] == 1
    assert rows["nearest_boundary_part_id"].tolist() == [41]
    assert rows["nearest_boundary_distance_m"].tolist() == pytest.approx([2.0e-8])
    assert rows["sdf_m"].tolist() == pytest.approx([-2.0e-8])
    assert rows["normal_velocity_mps"].tolist() == pytest.approx([-3.0])
    assert rows["inside_geometry"].tolist() == [1]


def test_reference_compare_summary_keeps_failure_and_pair_semantics() -> None:
    args = Namespace(
        reference_scope="full",
        override_t_end=0.25,
        artifact_mode="debug",
    )
    reference = {
        "run": "reference",
        "runtime_s": 1.0,
        "class_match_ratio_vs_reference": 1.0,
        "boundary_event_failure_count": 1,
    }
    runs = [
        {
            "run": "base",
            "runtime_s": 2.0,
            "class_match_ratio_vs_reference": 0.75,
            "boundary_event_failure_count": 0,
        },
        {
            "run": "candidate",
            "runtime_s": 3.0,
            "class_match_ratio_vs_reference": 0.5,
            "diagnostic_hard_invalid_failed": True,
            "boundary_event_failure_count": 0,
        },
    ]

    summary, exit_code = comparison_metrics.comparison_summary(
        args,
        timestamp="20260812_000000_000000",
        comparison_dir=Path("comparison"),
        reference=reference,
        runs=runs,
    )

    assert exit_code == 1
    assert summary["overrides"] == {"t_end": 0.25, "artifact_mode": "debug"}
    assert summary["boundary_event_failures"] == ["reference"]
    assert summary["diagnostic_hard_invalid_failures"] == ["candidate"]
    assert summary["pair_delta"]["base_run"] == "base"
    assert summary["pair_delta"]["candidate_run"] == "candidate"


def test_reference_compare_materializes_execution_configs_in_staging(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.yaml"
    candidate = tmp_path / "candidate.yaml"
    for path in (reference, candidate):
        path.write_text("output:\n  mode: debug\n", encoding="utf-8")
    staging = tmp_path / "staging"
    published = tmp_path / "published"
    args = Namespace(override_t_end=0.5, artifact_mode="standard")

    execution_reference, reported_reference, runs = comparison_inputs.execution_configs(
        args,
        reference_config=reference,
        run_specs=[("candidate", candidate)],
        staging_dir=staging,
        comparison_dir=published,
    )

    assert execution_reference == staging / "configs" / "reference_reference.yaml"
    assert reported_reference == published / "configs" / execution_reference.name
    assert runs == [
        (
            "candidate",
            staging / "configs" / "candidate.yaml",
            published / "configs" / "candidate.yaml",
        )
    ]
    materialized = comparison_inputs.load_yaml_mapping(execution_reference)
    assert materialized["time"]["t_end"] == 0.5
    assert materialized["output"] == {"mode": "standard"}


def test_near_wall_validation_and_global_diagnostic_fallbacks(
    tmp_path: Path,
) -> None:
    assert near_wall_nohit._int_or_default(None, 7) == 7
    assert near_wall_nohit._int_or_default("invalid", 7) == 7
    with pytest.raises(ValueError, match="finite non-negative"):
        near_wall_nohit._resolve_threshold(-1.0, {}, {})
    with pytest.raises(ValueError, match="--threshold-m is required"):
        near_wall_nohit._resolve_threshold(
            None,
            {"classification_tolerance_m": "invalid"},
            {"execution": "invalid"},
        )

    assert (
        near_wall_nohit._field_support_status(
            {}, {"valid_mask_hard_invalid_count": 1}, {}
        )
        == "global_hard_invalid_seen"
    )
    assert (
        near_wall_nohit._field_support_status(
            {}, {"valid_mask_mixed_stencil_count": 1}, {}
        )
        == "global_mixed_stencil_seen"
    )
    assert (
        near_wall_nohit._classify_row(
            row={"final_state": "numerical_boundary_stopped"},
            wall_events_available=True,
            diagnostics={},
            field_support_status="valid",
        )[0]
        == "unresolved_crossing_numerical_boundary_issue"
    )

    missing_output = tmp_path / "missing"
    missing_output.mkdir()
    with pytest.raises(FileNotFoundError):
        near_wall_nohit._load_analysis_inputs(missing_output, None)
    pd.DataFrame([{"final_state": "active_free_flight"}]).to_csv(
        missing_output / "final_particles.csv", index=False
    )
    with pytest.raises(ValueError, match="particle_id"):
        near_wall_nohit._load_analysis_inputs(missing_output, None)
