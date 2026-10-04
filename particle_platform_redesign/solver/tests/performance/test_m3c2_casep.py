"""Direct checks for the manual Case-P owner-profile harness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.performance.m3c2_casep import _profile_owner, main, profile_decision
from tests.verification.microcases import materialize_microcase


def test_smoke_exercises_isolated_public_api_profile(tmp_path: Path) -> None:
    fixture = materialize_microcase("C01", tmp_path / "case")
    report_path = tmp_path / "report.json"

    main(
        [
            "--suite",
            "smoke",
            "--fixture-case",
            str(fixture.case_path),
            "--json",
            str(report_path),
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["benchmark"] == "m3c2_casep_owner_profile_v1"
    assert report["suite"] == "smoke"
    _assert_smoke_authority(report["authority"])
    _assert_smoke_observation(report["observations"][0])
    assert report["decision"]["owner_discovery_consistent"] is False
    assert report["decision"]["optimization_authorized"] is False


def _assert_smoke_authority(authority: dict[str, Any]) -> None:
    assert authority["seeds"] == [319032, 319047, 319063]
    assert authority["numerical_setting"] == {
        "brownian_interval_tree_depth": 3,
        "dt_s": 2.0e-5,
        "geometry_rtol": 1.0e-8,
        "purpose": "accepted_final",
    }


def _assert_smoke_observation(observation: dict[str, Any]) -> None:
    assert observation["fixture"] is True
    assert observation["environment"]["numba_cache_dir_private"] is True
    assert observation["environment"]["numba_cache_dir_initially_empty"] is True
    assert observation["environment"]["numba_num_threads"] == "1"
    assert observation["accepted_baseline"] is None
    digests = {
        observation[run]["scientific_payload_sha256"] for run in ("warmup", "measured", "profile")
    }
    assert len(digests) == 1
    assert observation["identity"]["exact_scientific_payload"] is True
    assert observation["identity"]["exact_work_counts"] is True
    assert observation["identity"]["registered_baseline_available"] is False
    assert observation["measured"]["timing_s"]["public_end_to_end_wall_s"] > 0.0
    assert observation["measured"]["timing_s"]["public_end_to_end_process_s"] >= 0.0
    assert observation["measured"]["memory"]["peak_rss_after_bytes"] > 0
    assert sum(observation["profile"]["owner_shares"].values()) == pytest.approx(1.0)
    assert observation["profile_overhead"]["public_end_to_end_wall_ratio"] > 0.0


def test_287_particle_discovery_never_authorizes_optimization() -> None:
    observations = [
        _decision_observation(seed, owner="ou_rng_charge", share=share)
        for seed, share in zip((319032, 319047, 319063), (0.31, 0.33, 0.29), strict=True)
    ]

    decision = profile_decision(observations)

    assert decision["owner_discovery_consistent"] is True
    assert decision["dominant_owner"] == "ou_rng_charge"
    assert decision["owner_share_threshold_met_in_every_seed"] is True
    assert decision["10k_confirmation_required"] is True
    assert decision["minimum_optimization_particle_count"] == 10_000
    assert decision["optimization_authorized"] is False
    reasons = decision["reasons"]
    assert isinstance(reasons, list)
    assert "10000_particle_accepted_accuracy_confirmation_required" in reasons

    observations[-1]["profile"]["dominant_owner_share"] = 0.24
    below_threshold = profile_decision(observations)
    assert below_threshold["owner_discovery_consistent"] is False
    assert below_threshold["owner_share_threshold_met_in_every_seed"] is False
    assert below_threshold["optimization_authorized"] is False


def test_state_commit_is_not_misclassified_as_writer_time() -> None:
    engine = "C:/workspace/chamber_particles/engine.py"
    geometry = "C:/workspace/chamber_particles/geometry.py"
    fields = "C:/workspace/chamber_particles/fields.py"
    output = "C:/workspace/chamber_particles/output.py"

    assert _profile_owner(engine, "_commit_curved_clear_wave") == "engine_orchestration"
    assert _profile_owner(engine, "_locate_curved_wave_batches") == "event_broad_localize"
    assert _profile_owner(engine, "_write_failure_events") == "writer"
    assert _profile_owner(geometry, "count_aabb_candidates") == "event_broad_localize"
    assert _profile_owner(fields, "sample_batch") == "field_locate_sample"
    assert _profile_owner(fields, "prepare_required_fields") == "field_prepare_or_bounds"
    assert _profile_owner(output, "write_frame") == "writer"
    assert _profile_owner(output, "_sha256_file") == "output_shared_unattributed"
    assert _profile_owner("~", "<method 'reduce'>") == "native_unattributed"


def _decision_observation(seed: int, *, owner: str, share: float) -> dict[str, Any]:
    return {
        "seed": seed,
        "fixture": False,
        "profile": {
            "dominant_owner": owner,
            "dominant_owner_share": share,
        },
        "identity": {
            "exact_scientific_payload": True,
            "exact_work_counts": True,
            "exact_case_identity": True,
            "exact_algorithm_revisions": True,
            "registered_baseline_available": True,
            "exact_registered_baseline_science": True,
            "exact_registered_baseline_work_counts": True,
            "exact_registered_baseline_case_identity": True,
            "exact_registered_baseline_algorithm_revisions": True,
        },
        "measured": {"work_counts": {"particle_macro_roots": 430_500}},
        "profile_overhead": {
            "public_end_to_end_wall_ratio": 1.25,
            "public_end_to_end_process_ratio": 1.24,
            "excluded_from_baseline_timing": True,
        },
    }
