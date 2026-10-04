"""M3-C2 anchor preflight must reject the existing single saved-run histories."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, cast

import pytest
from tools.vv.comsol.m3c2_anchor_preflight import build_preflight

CONFIG = Path(__file__).parents[1] / "cases/m3c2_theory_caseA_100nm_anchor_v1.json"
REPOSITORY_ROOT = Path(__file__).parents[6]


@pytest.fixture(scope="module")
def preflight(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, object], Path]:
    output = tmp_path_factory.mktemp("m3c2") / "preflight"
    manifest = build_preflight(CONFIG, REPOSITORY_ROOT, output)
    return manifest, output


def test_preflight_rejects_saved_semantics_and_never_reports_pass(
    preflight: tuple[dict[str, object], Path],
) -> None:
    manifest, output = preflight

    assert {
        "overall_status": manifest["overall_status"],
        "comparison_decision": manifest["comparison_decision"],
        "accuracy_claim": manifest["accuracy_claim"],
        "planned_run_count": manifest["planned_run_count"],
        "run_matrix_authority": manifest["run_matrix_authority"],
        "campaign_lock_status": manifest["campaign_lock_status"],
    } == {
        "overall_status": "BLOCKED_MISSING_MEANING_MATCHED_COHORT",
        "comparison_decision": "NOT_AUTHORIZED",
        "accuracy_claim": "NOT_EVALUATED",
        "planned_run_count": 64,
        "run_matrix_authority": "PROVISIONAL_SEED_ALLOCATION_ONLY",
        "campaign_lock_status": "NOT_CREATED",
    }
    mismatch_codes = cast(list[str], manifest["semantic_mismatch_codes"])
    gates = cast(list[dict[str, Any]], manifest["gates"])
    assert {
        "COMSOL_STEPWISE_BROWNIAN_REQUIRES_ENSEMBLE_CONVERGENCE",
        "SAVED_RANDOM_STREAM_IDENTITY_NOT_PROVEN",
        "NATIVE_FIELD_NOT_COMMON_P1",
    }.issubset(mismatch_codes)
    assert [gate["status"] for gate in gates[-5:]] == [
        "FAIL",
        "FAIL",
        "NOT_RUN",
        "NOT_RUN",
        "BLOCKED",
    ]

    report = json.loads((output / "comparison_manifest.json").read_text(encoding="utf-8"))
    assert {
        "overall_status": report["overall_status"],
        "golden_truth": report["golden_truth"],
        "brownian_degrees_of_freedom": report["future_companion_requirements"][
            "brownian_degrees_of_freedom"
        ],
        "comsol_out_of_plane_degrees_of_freedom": report["future_companion_requirements"][
            "comsol_out_of_plane_degrees_of_freedom"
        ],
        "comsol_random_number_mode": report["future_companion_requirements"][
            "comsol_random_number_mode"
        ],
    } == {
        "overall_status": "BLOCKED_MISSING_MEANING_MATCHED_COHORT",
        "golden_truth": "NOT_CLAIMED",
        "brownian_degrees_of_freedom": 2,
        "comsol_out_of_plane_degrees_of_freedom": False,
        "comsol_random_number_mode": "UserDefined",
    }


def test_saved_package_and_seed_receipts_are_explicit(
    preflight: tuple[dict[str, object], Path],
) -> None:
    _, output = preflight

    with (output / "source_package_audit.csv").open(encoding="utf-8", newline="") as stream:
        audits = list(csv.DictReader(stream))
    assert len(audits) == 2
    assert {row["configured_seed_parameter_value"] for row in audits} == {"1.0", "21.0"}
    assert {row["effective_random_stream_identity"] for row in audits} == {"UNVERIFIED"}
    assert all(row["meaning_matched_saved_cohort"] == "False" for row in audits)
    assert all(int(row["brownian_phi_reported_nonzero_rows"]) > 0 for row in audits)

    with (output / "seed_plan_receipt.csv").open(encoding="utf-8", newline="") as stream:
        seed_receipts = list(csv.DictReader(stream))
    assert len(seed_receipts) == 2
    assert {row["replica_count"] for row in seed_receipts} == {"32"}
    assert {row["unique_seed_count"] for row in seed_receipts} == {"32"}
    assert {row["validation_status"] for row in seed_receipts} == {"PASS"}


def test_run_matrix_has_32_unique_seeds_for_each_required_participant(
    preflight: tuple[dict[str, object], Path],
) -> None:
    _, output = preflight

    with (output / "run_matrix.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert {row["participant"] for row in rows} == {
        "comsol_common_p1_rz_companion",
        "solver_b03_candidate",
    }
    for participant in {row["participant"] for row in rows}:
        participant_rows = [row for row in rows if row["participant"] == participant]
        assert len(participant_rows) == 32
        assert len({row["campaign_seed"] for row in participant_rows}) == 32
        assert {row["execution_status"] for row in participant_rows} == {"PLANNED_NOT_RUN"}


def test_failed_upstream_model_lock_takes_blocker_precedence(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    original_lock = (
        REPOSITORY_ROOT
        / "particle_platform_redesign/solver/evidence/m3c0/reference_lock_v1/model_identity.csv"
    )
    rows = list(csv.DictReader(original_lock.read_text(encoding="utf-8").splitlines()))
    target = next(row for row in rows if row["variant"] == "formal_iondrag_theory_consistent")
    target["status"] = "FAIL"
    failed_lock = tmp_path / "model_identity.csv"
    with failed_lock.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    config["m3c0_locks"]["model_identity"] = str(failed_lock)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    manifest = build_preflight(config_path, REPOSITORY_ROOT, tmp_path / "output")

    assert manifest["overall_status"] == "BLOCKED_SOURCE_IDENTITY_FAILURE"
    artifact_rows = list(
        csv.DictReader(
            (tmp_path / "output/artifact_hashes.csv").read_text(encoding="utf-8").splitlines()
        )
    )
    assert [row["status"] for row in artifact_rows if row["kind"] == "source_model"] == ["FAIL"]
