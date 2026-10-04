"""M3-C2A pilot contract locks meaning but never authorizes execution."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, cast

import pytest
from tools.vv.comsol.lock_m3c2_caseA_100nm_pilot_contract import lock_contract

CONFIG = Path(__file__).parents[1] / "cases/m3c2_caseA_100nm_stochastic_pilot_v1.json"
CASE_P_CONFIG = Path(__file__).parents[1] / "cases/m3c2_caseP_100nm_stochastic_pilot_v1.json"
CASE_P_ALLOCATION = (
    Path(__file__).parents[1] / "cases/m3c2_caseP_100nm_final_seed_allocation_v1.json"
)
CASE_P_POLICY = Path(__file__).parents[1] / "cases/m3c2_caseP_100nm_ensemble_evaluation_v4.json"
CASE_P_RECIPE = Path(__file__).parents[1] / "cases/m3c2_caseP_100nm_candidate_macro_pilot_v1.json"
CASE_P_RECEIPT = (
    Path(__file__).parents[4] / "evidence/m3c2/caseP_100nm_pilot_contract_v1/contract_receipt.json"
)
CASE_P_AUTHORIZATION = (
    Path(__file__).parents[4]
    / "evidence/m3c2/caseP_100nm_pilot_execution_v1/execution_authorization.json"
)
REPOSITORY_ROOT = Path(__file__).parents[6]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def locked_receipt(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[dict[str, object], Path]:
    output = tmp_path_factory.mktemp("m3c2_contract") / "receipt"
    return lock_contract(CONFIG, REPOSITORY_ROOT, output), output


@pytest.fixture(scope="module")
def case_p_locked_receipt(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[dict[str, object], Path]:
    output = tmp_path_factory.mktemp("m3c2_case_p_contract") / "receipt"
    return lock_contract(CASE_P_CONFIG, REPOSITORY_ROOT, output), output


def test_contract_receipt_does_not_authorize_or_claim_accuracy(
    locked_receipt: tuple[dict[str, object], Path],
) -> None:
    receipt, _ = locked_receipt
    assert receipt["contract_validation_status"] == ("PASS_INPUT_IDENTITY_AND_SEMANTICS_LOCKED")
    assert receipt["execution_status"] == "NOT_AUTHORIZED"
    assert receipt["pilot_status"] == "NOT_RUN"
    assert receipt["solver_accuracy_claim"] == "NOT_EVALUATED"
    assert receipt["comparison_decision"] == "NOT_EVALUATED"
    assert receipt["comsol_or_candidate_executed"] is False
    semantics = cast(dict[str, Any], receipt["stochastic_semantics"])
    assert semantics["source_out_of_plane"] is False
    assert semantics["source_random_number_args"] == "GenerateUnique"
    assert semantics["required_runner_random_number_args"] == "UserDefined"
    assert semantics["required_seed_authority"] == "fptas.bf1.i"
    seed_isolation = cast(dict[str, Any], receipt["seed_isolation"])
    assert seed_isolation["final_replicas_per_participant"] == 32
    assert seed_isolation["all_pilot_seeds_disjoint_from_final"] is True
    assert seed_isolation["participant_pilot_seed_sets_disjoint"] is True


def test_stored_receipt_locks_all_nine_inputs(
    locked_receipt: tuple[dict[str, object], Path],
) -> None:
    receipt, output = locked_receipt
    stored = json.loads((output / "contract_receipt.json").read_text(encoding="utf-8"))
    assert stored == receipt
    rows = list(
        csv.DictReader((output / "artifact_hashes.csv").read_text(encoding="utf-8").splitlines())
    )
    assert len(rows) == 9
    assert {row["status"] for row in rows} == {"PASS"}


def test_changed_artifact_hash_fails_before_receipt(tmp_path: Path) -> None:
    tampered = json.loads(CONFIG.read_text(encoding="utf-8"))
    tampered["input_artifacts"][0]["sha256"] = "0" * 64
    tampered_path = tmp_path / "tampered.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="locked artifact identity differs: source_mph"):
        lock_contract(tampered_path, REPOSITORY_ROOT, tmp_path / "must_not_exist")
    assert not (tmp_path / "must_not_exist").exists()


def _assert_case_p_identity_and_semantics(receipt: dict[str, object]) -> None:
    assert receipt["contract_validation_status"] == "PASS_INPUT_IDENTITY_AND_SEMANTICS_LOCKED"
    assert receipt["execution_status"] == "NOT_AUTHORIZED"
    assert receipt["campaign"] == {
        "case_id": "formal_iondrag_theory_consistent/caseP_100nm",
        "evaluation_case_id": "M3-C2A_caseP_100nm_common-P1",
        "output_slug": "caseP_100nm",
        "final_registration_kind": "m3c2_caseP_100nm_final_campaign",
        "candidate_case_name_prefix": "m3c2_caseP_100nm",
    }
    applicability = cast(dict[str, Any], receipt["physical_applicability"])
    assert applicability["status"] == "NOT_CERTIFIED_NEGATIVE_ION_CURRENT_OMITTED"
    assert applicability["locked_charge_model_includes_negative_ion_current"] is False
    semantics = cast(dict[str, Any], receipt["stochastic_semantics"])
    assert semantics["required_seed_authority"] == "fpt.bf1.i"
    assert semantics["source_particle_dataset"] == "part_P_100nm"


def _assert_case_p_seed_receipt(receipt: dict[str, object], output: Path) -> None:
    seeds = cast(dict[str, Any], receipt["seed_isolation"])
    assert seeds["final_replicas_per_participant"] == 32
    assert seeds["final_unique_seed_labels"] == 64
    assert seeds["final_participant_seed_sets_disjoint"] is True
    assert seeds["all_pilot_seeds_disjoint_from_audited_existing"] is True
    assert seeds["comsol_pilot_seeds"] == [919000, 919001, 919002, 919003]
    assert seeds["candidate_pilot_seeds"] == [919004, 919005, 919006, 919007]
    rows = list(
        csv.DictReader((output / "artifact_hashes.csv").read_text(encoding="utf-8").splitlines())
    )
    assert len(rows) == 9
    assert {row["status"] for row in rows} == {"PASS"}


def test_case_p_contract_locks_identity_semantics_and_applicability(
    case_p_locked_receipt: tuple[dict[str, object], Path],
) -> None:
    receipt, output = case_p_locked_receipt
    _assert_case_p_identity_and_semantics(receipt)
    _assert_case_p_seed_receipt(receipt, output)


def test_case_p_seed_allocation_is_single_owner_and_globally_disjoint() -> None:
    allocation = json.loads(CASE_P_ALLOCATION.read_text(encoding="utf-8"))
    contract = json.loads(CASE_P_CONFIG.read_text(encoding="utf-8"))
    policy = json.loads(CASE_P_POLICY.read_text(encoding="utf-8"))
    allocated = allocation["participant_seed_sets"]
    comsol = set(allocated["comsol"])
    candidate = set(allocated["candidate"])
    pilot = contract["seed_plan"]["pilot"]
    assert len(comsol) == len(candidate) == 32
    assert not comsol & candidate
    assert not (comsol | candidate) & set(pilot["comsol_seeds"] + pilot["candidate_seeds"])
    assert min(comsol | candidate) > 318383
    assert contract["seed_plan"]["final_cohort"]["seed_source"]["path"].endswith(
        CASE_P_ALLOCATION.name
    )
    assert "seed_plan" not in policy["final"]
    assert policy["final"]["seed_allocation"]["path"] == CASE_P_ALLOCATION.name


def test_case_p_policy_preregisters_informative_full_population_rz_gate() -> None:
    policy = json.loads(CASE_P_POLICY.read_text(encoding="utf-8"))
    final = policy["final"]
    distribution = final["rz_distribution_gate"]
    assert final["familywise_error_control"]["allocation"] == {
        "terminal_population": 0.025,
        "rz_distribution": 0.025,
    }
    assert distribution["category_count"] == 83
    assert distribution["participant_union_count"] == 2
    assert distribution["terminal_categories"] == ["stuck", "held", "escaped"]
    expected_radius = math.sqrt(
        (
            distribution["category_count"] * math.log(2.0)
            + math.log(
                distribution["participant_union_count"]
                * distribution["output_time_count"]
                / distribution["familywise_alpha"]
            )
        )
        / (2.0 * distribution["observations_per_participant_per_time"])
    )
    assert distribution["one_sample_critical_radius"] == pytest.approx(expected_radius)
    assert distribution["two_sample_critical_radius"] == pytest.approx(2.0 * expected_radius)
    assert distribution["maximum_empirical_tv_that_can_pass"] == pytest.approx(
        distribution["margin_total_variation"] - 2.0 * expected_radius
    )
    assert policy["claim_policy"]["negative_ion_current_in_locked_charge_model"] is False


def _assert_case_p_recipe_authorization(recipe: dict[str, Any], contract: dict[str, Any]) -> None:
    assert recipe["evaluation_plan"]["policy"]["sha256"] == _sha256(CASE_P_POLICY)
    assert recipe["execution_authorization"] == {
        "path": (
            "particle_platform_redesign/solver/evidence/m3c2/"
            "caseP_100nm_pilot_execution_v1/execution_authorization.json"
        ),
        "sha256": _sha256(CASE_P_AUTHORIZATION),
    }
    authorization = json.loads(CASE_P_AUTHORIZATION.read_text(encoding="utf-8"))
    assert authorization["status"] == "AUTHORIZED_BY_EXPLICIT_USER_DIRECTION"
    assert authorization["contract"] == recipe["contract"]
    assert authorization["contract_receipt"] == recipe["contract_receipt"]
    assert authorization["campaign"] == contract["campaign"]
    assert authorization["purpose"] == "pilot"
    assert authorization["participants"] == ["comsol", "candidate"]
    assert recipe["physical_applicability"]["status"] == (
        "NOT_CERTIFIED_NEGATIVE_ION_CURRENT_OMITTED"
    )


def test_case_p_candidate_recipe_resolves_locked_contract_seeds_and_policy() -> None:
    recipe = json.loads(CASE_P_RECIPE.read_text(encoding="utf-8"))
    contract = json.loads(CASE_P_CONFIG.read_text(encoding="utf-8"))
    assert "campaign" not in recipe
    assert recipe["recipe_id"] == "M3-C2A-caseP-100nm-qualified-macro-pilot"
    assert recipe["contract"] == {
        "path": (
            "particle_platform_redesign/solver/tools/vv/comsol/cases/"
            "m3c2_caseP_100nm_stochastic_pilot_v1.json"
        ),
        "sha256": _sha256(CASE_P_CONFIG),
    }
    assert recipe["contract_receipt"]["sha256"] == _sha256(CASE_P_RECEIPT)
    seed_source = recipe["pilot"]["seed_source"]
    assert seed_source == {
        **recipe["contract"],
        "json_pointer": "/seed_plan/pilot/candidate_seeds",
    }
    assert contract["seed_plan"]["pilot"]["candidate_seeds"] == [
        919004,
        919005,
        919006,
        919007,
    ]
    assert [
        (level["name"], level["dt_s"], level["brownian_interval_tree_depth"])
        for level in recipe["pilot"]["levels"]
    ] == [
        ("macro_coarse", 2e-5, 3),
        ("macro_medium", 1e-5, 3),
        ("macro_fine", 5e-6, 3),
        ("path_fine", 5e-6, 4),
    ]
    _assert_case_p_recipe_authorization(recipe, contract)
