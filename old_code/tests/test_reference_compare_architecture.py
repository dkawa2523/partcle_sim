from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tools import _reference_compare_inputs as comparison_inputs
from tools import _reference_compare_metrics as comparison_metrics
from tools import _reference_compare_runs as comparison_runs
from tools import compare_against_reference

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = REPO_ROOT / "examples" / "v02_minimal" / "run_config.yaml"


@pytest.mark.parametrize(
    ("name", "expected_signature"),
    [
        (
            "class_match_ratio",
            "(candidate_final: 'pd.DataFrame', reference_final: 'pd.DataFrame') "
            "-> 'tuple[float, int]'",
        ),
        (
            "class_transition_summary",
            "(candidate_final: 'pd.DataFrame', reference_final: 'pd.DataFrame', *, "
            "top_n: 'int' = 12) -> 'dict[str, Any]'",
        ),
        (
            "geometry_feature_delta_summary",
            "(candidate_final: 'pd.DataFrame', reference_final: 'pd.DataFrame', "
            "runtime, *, top_n: 'int' = 12) -> 'dict[str, Any]'",
        ),
    ],
)
def test_public_comparison_functions_are_direct_metric_exports(
    name: str,
    expected_signature: str,
) -> None:
    public_function = getattr(compare_against_reference, name)
    assert public_function is getattr(comparison_metrics, name)
    assert str(inspect.signature(public_function)) == expected_signature


def test_pair_delta_preserves_key_order_and_numeric_semantics() -> None:
    counters = (
        "unresolved_crossing_count",
        "max_hits_reached_count",
        "nearest_projection_fallback_count",
        "boundary_event_failure_count",
        "stuck_count",
        "invalid_mask_stopped_count",
        "valid_mask_mixed_stencil_count",
        "valid_mask_hard_invalid_count",
    )
    base: dict[str, Any] = dict(
        zip(counters, (3, 4, 5, 7, 11, 13, 17, 19), strict=True)
    )
    candidate: dict[str, Any] = dict(
        zip(counters, (5, 7, 9, 12, 17, 20, 25, 28), strict=True)
    )
    base.update(run="base", runtime_s=2.0, class_match_ratio_vs_reference=0.75)
    candidate.update(run="candidate", runtime_s=2.5, class_match_ratio_vs_reference=1.0)

    delta = comparison_metrics.pair_delta(base, candidate)

    assert list(delta) == [
        "base_run",
        "candidate_run",
        "runtime_increase_ratio",
        "class_match_ratio_delta",
        *(f"{name}_delta" for name in counters),
    ]
    assert list(delta.values()) == [
        "base",
        "candidate",
        0.25,
        0.25,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
    ]


def test_summary_preserves_schema_and_failure_priority() -> None:
    args = argparse.Namespace(
        reference_scope="full", override_t_end=None, artifact_mode=None
    )
    reference = {
        "run": "reference",
        "diagnostic_hard_invalid_failed": True,
        "boundary_event_failure_count": 1,
    }
    candidate = {
        "run": "candidate",
        "diagnostic_hard_invalid_failed": True,
        "boundary_event_failure_count": 2,
    }

    summary, exit_code = comparison_metrics.comparison_summary(
        args,
        timestamp="20260814_000000_000000",
        comparison_dir=Path("comparison"),
        reference=reference,
        runs=[candidate],
    )

    assert list(summary) == [
        "artifact_type",
        "schema_version",
        "timestamp",
        "comparison_dir",
        "reference_scope",
        "overrides",
        "reference",
        "runs",
        "diagnostic_hard_invalid_failures",
        "boundary_event_failures",
    ]
    assert summary["diagnostic_hard_invalid_failures"] == [
        "reference",
        "candidate",
    ]
    assert summary["boundary_event_failures"] == ["reference", "candidate"]
    assert exit_code == 1


def test_direct_cli_run_preserves_published_artifact_schema(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_root = tmp_path / "comparison"
    summary_alias = tmp_path / "summary.json"
    exit_code = compare_against_reference.main(
        [
            "--reference-config",
            str(EXAMPLE),
            "--run",
            f"same={EXAMPLE}",
            "--output-root",
            str(output_root),
            "--summary-json",
            str(summary_alias),
        ]
    )

    printed = json.loads(capsys.readouterr().out)
    canonical = Path(printed["comparison_dir"]) / "comparison_summary.json"
    assert exit_code == 0
    assert list(printed) == [
        "artifact_type",
        "schema_version",
        "timestamp",
        "comparison_dir",
        "reference_scope",
        "overrides",
        "reference",
        "runs",
    ]
    assert json.loads(canonical.read_text(encoding="utf-8")) == printed
    assert json.loads(summary_alias.read_text(encoding="utf-8")) == printed
    assert printed["runs"][0]["class_match_ratio_vs_reference"] == 1.0
    assert not list(output_root.glob(".*.staging-*"))


def test_class_and_geometry_metrics_keep_numeric_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference_classes = pd.DataFrame(
        {"particle_id": [1, 2, 3], "final_state": ["stuck", "escaped", "absorbed"]}
    )
    candidate_classes = pd.DataFrame(
        {"particle_id": [1, 2, 4], "final_state": ["stuck", "absorbed", "escaped"]}
    )
    assert comparison_metrics.class_match_ratio(
        candidate_classes, reference_classes
    ) == (0.5, 2)
    transitions = comparison_metrics.class_transition_summary(
        candidate_classes, reference_classes, top_n=1
    )
    assert transitions["mismatch_count"] == 1
    assert transitions["top_mismatches"][0] == {
        "reference_class": "escaped",
        "candidate_class": "absorbed",
        "count": 1,
    }

    geometry = SimpleNamespace(
        axes=(np.array([0.0, 1.0, 3.0]), np.array([0.0, 2.0])),
        boundary_edges=np.ones((1, 2, 2)),
    )
    runtime = SimpleNamespace(
        spatial_dim=2, geometry_provider=SimpleNamespace(geometry=geometry)
    )
    monkeypatch.setattr(
        comparison_metrics,
        "sample_geometry_sdf",
        lambda _runtime, position: float(position[0] - 1.0),
    )
    monkeypatch.setattr(
        comparison_metrics,
        "sample_geometry_part_id",
        lambda _runtime, position: 10 + int(position[0] >= 1.0),
    )
    monkeypatch.setattr(
        comparison_metrics,
        "nearest_boundary_edge_features_2d",
        lambda _runtime, _positions: (
            np.array([21, 22], dtype=np.int32),
            np.array([0.01, np.nan]),
        ),
    )
    reference = pd.DataFrame(
        {
            "particle_id": [1, 2],
            "final_state": ["stuck", "escaped"],
            "x_m": [0.1, 2.0],
            "y_m": [0.0, 0.0],
            "vx_mps": [1.0, 2.0],
        }
    )
    candidate = reference.copy()
    candidate[["final_state", "x_m", "vx_mps"]] = [
        ["absorbed", 0.2, 2.0],
        ["escaped", 2.5, 2.0],
    ]

    features = comparison_metrics.geometry_feature_delta_summary(
        candidate, reference, runtime
    )

    assert list(features) == [
        "compared_particles",
        "near_boundary_threshold_m",
        "position_error_m",
        "sdf_error_m",
        "abs_sdf_error_m",
        "nearest_boundary_distance_error_m",
        "speed_error_mps",
        "outside_geometry_count_reference",
        "outside_geometry_count_candidate",
        "outside_geometry_count_delta",
        "near_boundary_count_reference",
        "near_boundary_count_candidate",
        "near_boundary_count_delta",
        "nearest_part_transition_summary",
        "mismatched_state_feature_summary",
    ]
    assert features["near_boundary_threshold_m"] == 1.0
    assert features["position_error_m"]["mean"] == pytest.approx(0.3)
    assert features["mismatched_state_feature_summary"]["count"] == 1
    positions = comparison_metrics.final_position_array(candidate, 2)
    assert positions.shape == (2, 2)
    assert positions.dtype == np.float64


def test_input_validation_keeps_error_priority_and_normalization(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "existing.yaml"
    existing.write_text("inputs: unchanged\n", encoding="utf-8")
    missing_reference = tmp_path / "missing-reference.yaml"
    missing_candidate = tmp_path / "missing-candidate.yaml"
    args = argparse.Namespace(
        reference_config=missing_reference,
        run=[("candidate", missing_candidate)],
        output_root=tmp_path / "output",
    )
    with pytest.raises(FileNotFoundError) as reference_error:
        comparison_inputs.resolve_comparison_inputs(args, tmp_path)
    assert (
        str(reference_error.value) == f"reference config not found: {missing_reference}"
    )
    args.reference_config = existing
    with pytest.raises(FileNotFoundError) as candidate_error:
        comparison_inputs.resolve_comparison_inputs(args, tmp_path)
    assert str(candidate_error.value) == (
        f"run config not found for candidate: {missing_candidate}"
    )
    with pytest.raises(ValueError, match=r"^invalid run name 'reference'"):
        comparison_inputs.validate_run_specs(
            [("reference", Path("first")), ("reference", Path("second"))]
        )

    destination = comparison_inputs.write_config_variant(
        source_config=existing,
        output_config=tmp_path / "normalized.yaml",
        override_t_end=None,
        artifact_mode=None,
    )
    assert comparison_inputs.load_yaml_mapping(destination) == {"inputs": "unchanged"}
    assert comparison_inputs.parse_named_run(" run-1 = config.yaml ") == (
        "run-1",
        Path("config.yaml"),
    )


def test_run_artifacts_keep_legacy_defaults_and_relocation(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    pd.DataFrame(
        {
            "particle_id": range(6),
            "final_state": [
                "stuck",
                "absorbed",
                "escaped",
                "invalid_mask_stopped",
                "numerical_boundary_stopped",
                None,
            ],
            "released": [1, 1, 1, 0, 0, 0],
        }
    ).to_csv(output_dir / "final_particles.csv", index=False)
    (output_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "timing_s": "legacy-invalid",
                "memory_estimate_bytes": None,
                "unresolved_crossing_count": 2,
                "max_hits_reached_count": 3,
            }
        ),
        encoding="utf-8",
    )

    summary = comparison_runs._summarize_run(output_dir, runtime_s=1.25)

    assert summary["released_count"] == 3
    assert summary["solver_core_s"] == summary["estimated_numpy_bytes"] == 0
    assert summary["boundary_event_failure_count"] == 6
    source, destination = tmp_path / "source", tmp_path / "published"
    assert comparison_runs.relocate_value(
        [str(source), str(source / "file"), "unrelated", 7], source, destination
    ) == [str(destination), str(destination / "file"), "unrelated", 7]
    with pytest.raises(FileNotFoundError):
        comparison_runs._load_json(tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError):
        comparison_runs._load_final_particles(tmp_path)


def test_atomic_publication_and_staging_cleanup(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    output_root.mkdir()
    with comparison_runs.staging_directory(
        output_root, output_root / "comparison"
    ) as staging:
        staging_path = staging
        (staging / "partial").write_text("partial", encoding="utf-8")
    assert not staging_path.exists()

    destination = output_root / "summary.json"
    comparison_runs.write_json_atomic(destination, {"second": 2, "first": 1})
    assert destination.read_text(encoding="utf-8").endswith("\n")
    assert list(json.loads(destination.read_text(encoding="utf-8"))) == [
        "second",
        "first",
    ]
    comparison_runs.write_summary_alias(
        None,
        repo_root=tmp_path,
        output_root=output_root,
        comparison_dir=output_root / "comparison",
        summary={"status": "complete"},
    )
    default_alias = output_root / "comparison_summary.json"
    assert json.loads(default_alias.read_text(encoding="utf-8")) == {
        "status": "complete"
    }


def test_main_rejects_existing_timestamped_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FixedDatetime:
        @staticmethod
        def now() -> SimpleNamespace:
            return SimpleNamespace(strftime=lambda _pattern: "fixed")

    output_root = tmp_path / "output"
    (output_root / "compare_fixed").mkdir(parents=True)
    monkeypatch.setattr(compare_against_reference, "datetime", FixedDatetime)
    with pytest.raises(FileExistsError, match="comparison directory already exists"):
        compare_against_reference.main(
            [
                "--reference-config",
                str(EXAMPLE),
                "--run",
                f"same={EXAMPLE}",
                "--output-root",
                str(output_root),
            ]
        )
