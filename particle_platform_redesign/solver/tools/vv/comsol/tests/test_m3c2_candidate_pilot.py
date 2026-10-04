"""Focused checks for the M3-C2A candidate pilot runner."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml
from tools.vv.comsol import run_m3c2_candidate_pilot as candidate

from chamber_particles import load_case

RECIPE = Path(__file__).parents[1] / "cases/m3c2_caseA_100nm_candidate_pilot_v1.json"
DIAGNOSTIC_RECIPE = Path(__file__).parents[1] / "cases/m3c2_caseA_100nm_candidate_pilot_v2.json"
SENSITIVITY_RECIPE = Path(__file__).parents[1] / "cases/m3c2_caseA_100nm_candidate_pilot_v3.json"
CROSS_BAND_RECIPE = Path(__file__).parents[1] / "cases/m3c2_caseA_100nm_candidate_pilot_v4.json"
QUALIFIED_MACRO_RECIPE = (
    Path(__file__).parents[1] / "cases/m3c2_caseA_100nm_candidate_macro_pilot_v5.json"
)
CASE_P_RECIPE = Path(__file__).parents[1] / "cases/m3c2_caseP_100nm_candidate_macro_pilot_v1.json"
CASE_P_RECIPE_V2 = (
    Path(__file__).parents[1] / "cases/m3c2_caseP_100nm_candidate_macro_pilot_v2.json"
)
CASE_P_EVENT_TOLERANCE_RECIPE = (
    Path(__file__).parents[1] / "cases/m3c2_caseP_100nm_event_tolerance_pre_final_v1.json"
)
CASE_P_POLICY = Path(__file__).parents[1] / "cases/m3c2_caseP_100nm_ensemble_evaluation_v4.json"
CASE_P_POLICY_V5 = Path(__file__).parents[1] / "cases/m3c2_caseP_100nm_ensemble_evaluation_v5.json"
FINAL_COMSOL_SEEDS = list(range(318160, 318192))
FINAL_CANDIDATE_SEEDS = list(range(318192, 318224))
CASE_P_PILOT_COMSOL_SEEDS = list(range(919000, 919004))
CASE_P_PILOT_CANDIDATE_SEEDS = list(range(919004, 919008))
CASE_P_FINAL_COMSOL_SEEDS = list(range(319000, 319032))
CASE_P_FINAL_CANDIDATE_SEEDS = list(range(319032, 319064))
CASE_P_CAMPAIGN = {
    "case_id": "formal_iondrag_theory_consistent/caseP_100nm",
    "evaluation_case_id": "M3-C2A_caseP_100nm_common-P1",
    "output_slug": "caseP_100nm",
    "final_registration_kind": "m3c2_caseP_100nm_final_campaign",
    "candidate_case_name_prefix": "m3c2_caseP_100nm",
}


def _mapping(value: object) -> dict[str, Any]:
    assert isinstance(value, dict)
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _final_registration_fixture(
    tmp_path: Path,
    campaign_identity: dict[str, str] | None = None,
    pilot_seed_sets: dict[str, list[int]] | None = None,
    final_seed_sets: dict[str, list[int]] | None = None,
    final_seed_source: dict[str, object] | None = None,
    policy_revision: int = 3,
) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    identity = campaign_identity or candidate.LEGACY_CAMPAIGN_IDENTITY
    pilot_seeds = pilot_seed_sets or {
        "comsol": [918160, 918161, 918162, 918163],
        "candidate": [918164, 918165, 918166, 918167],
    }
    seed_plan = final_seed_sets or {
        "comsol": FINAL_COMSOL_SEEDS,
        "candidate": FINAL_CANDIDATE_SEEDS,
    }
    policy_final: dict[str, object] = {"replicas_per_participant": 32}
    if final_seed_source is None:
        policy_final["seed_plan"] = seed_plan
    else:
        policy_final["seed_allocation"] = final_seed_source
    policy = tmp_path / "evaluation_policy.json"
    _write_json(
        policy,
        {
            "policy_kind": "m3c2_stochastic_ensemble",
            "policy_revision": policy_revision,
            "final": policy_final,
        },
    )
    pilot = tmp_path / "pilot_evaluation.json"
    _write_json(
        pilot,
        {
            "schema_version": policy_revision,
            "tool_revision": f"m3c2_stochastic_ensemble_evaluator_v{policy_revision}",
            "phase": "pilot",
            "status": "PASS",
            "policy": {"sha256": candidate._sha256(policy), "revision": policy_revision},
            "seed_sets": pilot_seeds,
            "scope_fingerprint": {
                "sha256": "a" * 64,
                "common_design_sha256": "b" * 64,
            },
            "pilot_configuration_screening": {
                "candidate": {"largest_screened_macro_level": "macro_coarse"}
            },
        },
    )
    selection = tmp_path / "selection_receipt.json"
    _write_json(
        selection,
        {
            "schema_version": 1,
            "receipt_kind": candidate.FINAL_AUTHORIZATION_KIND,
            "status": "AUTHORIZED_FOR_CONFIRMATORY_FINAL",
            "policy_sha256": candidate._sha256(policy),
            "pilot_report_sha256": candidate._sha256(pilot),
            "pilot_scope_sha256": "a" * 64,
            "common_design_sha256": "b" * 64,
            "seed_plan_sha256": candidate._object_sha256(seed_plan),
            "selected_final_levels": {
                "comsol": {
                    "level_id": "dt_20us",
                    "ordinal": 0,
                    "numerical_setting": {
                        "integrator": "classical_rk4",
                        "fixed_step_s": 2.0e-5,
                        "purpose": "accepted_final",
                    },
                },
                "candidate": {
                    "level_id": "macro_coarse",
                    "ordinal": 0,
                    "numerical_setting": {
                        "dt_s": 2.0e-5,
                        "brownian_interval_tree_depth": 3,
                        "geometry_rtol": 1.0e-8,
                        "purpose": "accepted_final",
                    },
                },
            },
        },
    )
    registration = tmp_path / "final_registration.json"
    registration_document: dict[str, object] = {
        "schema_version": 1,
        "registration_kind": identity["final_registration_kind"],
        "case_id": identity["case_id"],
        "purpose": "final",
        "comsol_numerical_setting": {
            "integrator": "classical_rk4",
            "fixed_step_s": 2.0e-5,
        },
        "execution_authorization": {
            "status": "AUTHORIZED",
            "evaluation_policy": {
                "path": policy.name,
                "sha256": candidate._sha256(policy),
            },
            "pilot_evaluation": {
                "path": pilot.name,
                "sha256": candidate._sha256(pilot),
            },
            "selection_receipt": {
                "path": selection.name,
                "sha256": candidate._sha256(selection),
            },
        },
    }
    if final_seed_source is None:
        registration_document["participant_seed_sets"] = seed_plan
    else:
        registration_document["participant_seed_source"] = final_seed_source
    _write_json(registration, registration_document)
    return registration, selection


def _final_seed_allocation_fixture(root: Path) -> tuple[dict[str, object], dict[str, list[int]]]:
    root.mkdir(parents=True, exist_ok=True)
    seed_sets = {
        "comsol": CASE_P_FINAL_COMSOL_SEEDS,
        "candidate": CASE_P_FINAL_CANDIDATE_SEEDS,
    }
    allocation = root / "m3c2_caseP_100nm_final_seed_allocation_v1.json"
    _write_json(
        allocation,
        {
            "schema_version": 1,
            "allocation_kind": "m3c2_final_seed_allocation",
            "allocation_id": "M3-C2A-caseP-100nm-final-seeds",
            "allocation_revision": 1,
            "case_id": CASE_P_CAMPAIGN["case_id"],
            "purpose": "final",
            "replicas_per_participant": 32,
            "participant_seed_sets": seed_sets,
            "execution_status": "PLANNED_NOT_RUN",
            "authorization": "NOT_AUTHORIZED_PENDING_ACCEPTED_PILOT",
            "seed_audit": {},
        },
    )
    return (
        {
            "path": allocation.name,
            "sha256": candidate._sha256(allocation),
            "json_pointers": {
                "comsol": "/participant_seed_sets/comsol",
                "candidate": "/participant_seed_sets/candidate",
            },
        },
        seed_sets,
    )


def _configured_recipe_fixture(
    tmp_path: Path,
    *,
    campaign_in_recipe: bool = True,
    final_seed_source: dict[str, object] | None = None,
    evaluation_policy_path: Path | None = None,
    recipe_revision: int = 1,
) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    contract = json.loads(
        (Path(__file__).parents[1] / "cases/m3c2_caseA_100nm_stochastic_pilot_v1.json").read_text(
            encoding="utf-8"
        )
    )
    contract["contract_id"] = "M3-C2A-caseP-100nm-stochastic-pilot"
    contract["campaign"] = CASE_P_CAMPAIGN
    contract["seed_plan"]["pilot"]["comsol_seeds"] = CASE_P_PILOT_COMSOL_SEEDS
    contract["seed_plan"]["pilot"]["candidate_seeds"] = CASE_P_PILOT_CANDIDATE_SEEDS
    contract["seed_plan"]["final_cohort"]["case_id"] = CASE_P_CAMPAIGN["case_id"]
    if final_seed_source is not None:
        contract["seed_plan"]["final_cohort"]["seed_source"] = final_seed_source
    contract_path = tmp_path / "configured_contract.json"
    _write_json(contract_path, contract)

    recipe = json.loads(QUALIFIED_MACRO_RECIPE.read_text(encoding="utf-8"))
    recipe["recipe_id"] = "M3-C2A-caseP-100nm-qualified-macro-pilot"
    recipe["recipe_revision"] = recipe_revision
    if campaign_in_recipe:
        recipe["campaign"] = CASE_P_CAMPAIGN
    contract_record = {
        "path": str(contract_path.resolve()),
        "sha256": candidate._sha256(contract_path),
    }
    recipe["contract"] = contract_record
    receipt = json.loads(
        (
            Path(__file__).parents[4]
            / "evidence/m3c2/caseA_100nm_pilot_contract_v1/contract_receipt.json"
        ).read_text(encoding="utf-8")
    )
    receipt["contract_sha256"] = contract_record["sha256"]
    receipt["campaign"] = CASE_P_CAMPAIGN
    receipt_path = tmp_path / "configured_contract_receipt.json"
    _write_json(receipt_path, receipt)
    receipt_record = {
        "path": str(receipt_path.resolve()),
        "sha256": candidate._sha256(receipt_path),
    }
    recipe["contract_receipt"] = receipt_record
    authorization_path = tmp_path / "configured_execution_authorization.json"
    _write_json(
        authorization_path,
        {
            "schema_version": 1,
            "authorization_kind": "m3c2_pilot_execution_authorization",
            "status": "AUTHORIZED_BY_EXPLICIT_USER_DIRECTION",
            "contract": contract_record,
            "contract_receipt": receipt_record,
            "campaign": CASE_P_CAMPAIGN,
            "purpose": "pilot",
            "participants": ["comsol", "candidate"],
        },
    )
    recipe["execution_authorization"] = {
        "path": str(authorization_path.resolve()),
        "sha256": candidate._sha256(authorization_path),
    }
    recipe["pilot"].pop("candidate_seeds")
    recipe["pilot"]["seed_source"] = {
        **contract_record,
        "json_pointer": "/seed_plan/pilot/candidate_seeds",
    }
    policy_path = (evaluation_policy_path or CASE_P_POLICY).resolve()
    recipe["evaluation_plan"]["policy"] = {
        "path": str(policy_path),
        "sha256": candidate._sha256(policy_path),
    }
    repository_root = candidate._repository_root()
    for key in ("canonical_input", "deterministic_case_template"):
        record = recipe[key]
        record["path"] = str((repository_root / record["path"]).resolve())
    recipe_path = tmp_path / "configured_recipe.json"
    _write_json(recipe_path, recipe)
    return recipe_path


def _write_complete_receipts(output: Path, cells: dict[str, Any]) -> None:
    report = json.loads((output / candidate.PREPARE_REPORT).read_text(encoding="utf-8"))
    for value in cells.values():
        planned = _mapping(value)
        cell = output / "levels" / str(planned["level"]) / f"seed_{planned['seed']}"
        result = cell / "result"
        result.mkdir()
        _write_json(result / "run.json", {"status": "complete"})
        for name in ("trajectory", "events", "failures"):
            (cell / f"{name}.csv").write_text(f"{name}\n", encoding="utf-8")
        _write_json(cell / "performance.json", {"measurement_status": "NOT_MEASURED"})

        def relative(path: Path) -> str:
            return path.relative_to(output).as_posix()

        _write_json(
            cell / "run_receipt.json",
            {
                "status": "COMPLETE",
                "tool_revision": report["tool_revision"],
                "participant": "candidate",
                "level": planned["level"],
                "seed": planned["seed"],
                "dt_s": planned["dt_s"],
                "brownian_interval_tree_depth": planned["brownian_interval_tree_depth"],
                "geometry_rtol": planned["geometry_rtol"],
                "case": planned["case"],
                "case_sha256": planned["case_sha256"],
                "result": relative(result),
                "result_manifest_sha256": candidate._sha256(result / "run.json"),
                "trajectory": relative(cell / "trajectory.csv"),
                "trajectory_sha256": candidate._sha256(cell / "trajectory.csv"),
                "events": relative(cell / "events.csv"),
                "events_sha256": candidate._sha256(cell / "events.csv"),
                "failures": relative(cell / "failures.csv"),
                "failures_sha256": candidate._sha256(cell / "failures.csv"),
                "performance": relative(cell / "performance.json"),
                "performance_sha256": candidate._sha256(cell / "performance.json"),
            },
        )


def _assert_prepared_final_campaign(report: dict[str, object], output: Path) -> dict[str, Any]:
    assert report["tool_revision"] == "m3c2_candidate_campaign_runner_v4"
    assert report["purpose"] == "final"
    assert report["final_report"] == candidate.FINAL_CAMPAIGN_REPORT
    cells = _mapping(report["cells"])
    assert list(cells) == [f"macro_coarse/{seed}" for seed in FINAL_CANDIDATE_SEEDS]
    assert not set(FINAL_COMSOL_SEEDS).intersection(FINAL_CANDIDATE_SEEDS)
    first = _mapping(cells[f"macro_coarse/{FINAL_CANDIDATE_SEEDS[0]}"])
    assert (first["dt_s"], first["brownian_interval_tree_depth"]) == (2.0e-5, 3)
    assert first["geometry_rtol"] == 1.0e-8
    case = yaml.safe_load((output / str(first["case"])).read_text(encoding="utf-8"))
    assert case["solver"]["seed"] == FINAL_CANDIDATE_SEEDS[0]
    assert case["physics"]["noise"]["interval_tree_depth"] == 3
    assert candidate._load_prepared(output)["purpose"] == "final"
    return cells


def _assert_final_campaign_manifest(manifest: dict[str, object], output: Path) -> None:
    assert manifest["purpose"] == "final"
    assert manifest["comparison_status"] == ("READY_FOR_INDEPENDENT_ENSEMBLE_FINAL_EVALUATION")
    assert len(_mapping(manifest["levels"])["macro_coarse"]["replicas"]) == 32
    assert (output / candidate.FINAL_CAMPAIGN_REPORT).is_file()
    assert not (output / candidate.FINAL_REPORT).exists()


def test_recipe_separates_macro_step_and_path_depth_sensitivity() -> None:
    recipe = json.loads(RECIPE.read_text(encoding="utf-8"))

    candidate._validate_recipe(recipe)

    levels = recipe["pilot"]["levels"]
    assert [(level["dt_s"], level["brownian_interval_tree_depth"]) for level in levels] == [
        (2.0e-5, 3),
        (1.0e-5, 3),
        (5.0e-6, 3),
        (5.0e-6, 4),
    ]
    assert len(candidate._output_times()) == 121
    assert candidate._output_times()[-1] == 0.03


def test_prepare_uses_the_locked_common_input_and_public_case_schema(tmp_path: Path) -> None:
    output = tmp_path / "candidate-pilot"

    report = candidate.prepare(RECIPE.resolve(), output)

    cells = _mapping(report["cells"])
    assert report["campaign_identity"] == candidate.LEGACY_CAMPAIGN_IDENTITY
    assert len(cells) == 16
    assert report["input_content_hash"] == (
        "sha256:d30e9048cf8e142c3689508f0de2f20c503e7e787c80809e9fbe1806c8b45a1c"
    )
    coarse_path = output / str(_mapping(cells["macro_coarse/918164"])["case"])
    path_fine = output / str(_mapping(cells["path_fine/918164"])["case"])
    coarse = yaml.safe_load(coarse_path.read_text(encoding="utf-8"))
    refined = yaml.safe_load(path_fine.read_text(encoding="utf-8"))
    assert coarse["solver"]["integrator"] == "ou_langevin"
    assert coarse["case"]["name"] == "m3c2_caseA_100nm_macro_coarse_seed_918164"
    assert coarse["solver"]["seed"] == 918164
    assert coarse["physics"]["noise"]["interval_tree_depth"] == 3
    assert refined["physics"]["noise"]["interval_tree_depth"] == 4
    assert {item["boundary_group"]: item["law"] for item in coarse["boundaries"]}[
        "gas_inlet"
    ] == "hold"
    assert (
        load_case(coarse_path).case_file_hash
        == _mapping(cells["macro_coarse/918164"])["case_file_hash"]
    )


def test_v2_predeclares_one_scale_adequate_event_tolerance_diagnostic(
    tmp_path: Path,
) -> None:
    recipe = json.loads(DIAGNOSTIC_RECIPE.read_text(encoding="utf-8"))
    candidate._validate_recipe(recipe)
    assert recipe["claim_policy"]["comsol_fit"] is False
    output = tmp_path / "candidate-diagnostic"

    report = candidate.prepare(DIAGNOSTIC_RECIPE.resolve(), output)

    cells = _mapping(report["cells"])
    assert list(cells) == ["event_tolerance_diagnostic/918164"]
    case = yaml.safe_load(
        (output / str(_mapping(cells["event_tolerance_diagnostic/918164"])["case"])).read_text(
            encoding="utf-8"
        )
    )
    assert case["time"]["dt_s"] == 5e-6
    assert case["physics"]["noise"]["interval_tree_depth"] == 3
    assert case["solver"]["event"]["geometry_rtol"] == 1e-8


def test_v3_gate_is_frozen_before_tolerance_sensitivity_execution(tmp_path: Path) -> None:
    recipe = json.loads(SENSITIVITY_RECIPE.read_text(encoding="utf-8"))

    candidate._validate_recipe(recipe)
    report = candidate.prepare(SENSITIVITY_RECIPE.resolve(), tmp_path / "sensitivity")

    gate = recipe["acceptance_gate"]
    cell = _mapping(_mapping(report["cells"])["event_tolerance_sensitivity/918164"])
    assert cell["geometry_rtol"] == 1e-9
    assert gate["normalized_ratio_limit"] == 4.0
    assert gate["required_zero_failures_both"] is True
    assert gate["required_exact_identity"] == [
        "particle_id_and_event_ordinal",
        "candidate_facet_id_and_candidate_offset",
        "primary_facet_id",
        "boundary_id",
        "material_id",
        "law_id",
        "outcome",
        "final_particle_fate",
    ]


def test_v4_uses_unseen_seed_and_two_predeclared_cross_band_cells(tmp_path: Path) -> None:
    recipe = json.loads(CROSS_BAND_RECIPE.read_text(encoding="utf-8"))

    candidate._validate_recipe(recipe)
    report = candidate.prepare(CROSS_BAND_RECIPE.resolve(), tmp_path / "cross-band")

    cells = _mapping(report["cells"])
    assert list(cells) == [
        "event_tolerance_reference/918165",
        "event_tolerance_candidate/918165",
    ]
    assert [
        _mapping(cells[key])["geometry_rtol"]
        for key in (
            "event_tolerance_reference/918165",
            "event_tolerance_candidate/918165",
        )
    ] == [1e-8, 1e-9]
    assert recipe["acceptance_gate"]["policy_revision"] == ("cross_band_stopping_time_allowance_v1")


def test_v5_qualified_macro_recipe_prepares_sixteen_cells(tmp_path: Path) -> None:
    recipe = json.loads(QUALIFIED_MACRO_RECIPE.read_text(encoding="utf-8"))

    candidate._validate_recipe(recipe)
    report = candidate.prepare(QUALIFIED_MACRO_RECIPE.resolve(), tmp_path / "macro-v5")

    cells = _mapping(report["cells"])
    assert len(cells) == 16
    assert {float(_mapping(cell)["geometry_rtol"]) for cell in cells.values()} == {1e-8}
    assert recipe["claim_policy"]["macro_convergence"] == (
        "NOT_ESTABLISHED_UNTIL_INDEPENDENT_EVALUATION"
    )


@pytest.mark.parametrize("recipe_revision", [1, 2])
def test_casep_pre_final_event_tolerance_recipe_prepares_exact_two_cells(
    tmp_path: Path, recipe_revision: int
) -> None:
    recipe_path = CASE_P_EVENT_TOLERANCE_RECIPE.resolve()
    if recipe_revision == 2:
        recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
        recipe["recipe_revision"] = 2
        recipe_path = tmp_path / "casep-event-tolerance-v2.json"
        _write_json(recipe_path, recipe)
    report = candidate.prepare(recipe_path, tmp_path / "casep-event-tolerance")

    cells = _mapping(report["cells"])
    assert list(cells) == [
        "geometry_rtol_reference/919008",
        "geometry_rtol_candidate/919008",
    ]
    assert [
        (
            _mapping(cell)["dt_s"],
            _mapping(cell)["brownian_interval_tree_depth"],
            _mapping(cell)["geometry_rtol"],
        )
        for cell in cells.values()
    ] == [(2.0e-5, 3, 1.0e-8), (2.0e-5, 3, 1.0e-9)]


@pytest.mark.parametrize(
    "damage",
    ["recipe_id", "revision", "seed", "level", "dt", "depth", "rtol", "purpose"],
)
def test_casep_pre_final_seed_exception_is_exact(tmp_path: Path, damage: str) -> None:
    recipe = json.loads(CASE_P_EVENT_TOLERANCE_RECIPE.read_text(encoding="utf-8"))
    pilot = _mapping(recipe["pilot"])
    levels = pilot["levels"]
    assert isinstance(levels, list)
    first = _mapping(levels[0])
    if damage == "recipe_id":
        recipe["recipe_id"] = "another-configured-recipe"
    elif damage == "revision":
        recipe["recipe_revision"] = 3
    elif damage == "seed":
        pilot["candidate_seeds"] = [919009]
    elif damage == "level":
        first["name"] = "another_reference"
    elif damage == "dt":
        first["dt_s"] = 1.0e-5
    elif damage == "depth":
        first["brownian_interval_tree_depth"] = 4
    elif damage == "rtol":
        first["geometry_rtol"] = 1.0e-7
    else:
        first["purpose"] = "another_purpose"
    recipe_path = tmp_path / "tampered_recipe.json"
    _write_json(recipe_path, recipe)

    with pytest.raises(ValueError, match=r"seeds differ|recipe matrix differs"):
        candidate.prepare(recipe_path, tmp_path / "rejected")


@pytest.mark.parametrize(
    "damage",
    ["level", "seed", "dt", "depth", "rtol", "case_hash", "case_artifact", "status", "artifact"],
)
def test_finalize_rechecks_completed_receipt_against_planned_cell(
    tmp_path: Path, damage: str
) -> None:
    output = tmp_path / "candidate-pilot"
    report = candidate.prepare(QUALIFIED_MACRO_RECIPE.resolve(), output)
    cells = _mapping(report["cells"])
    _write_complete_receipts(output, cells)
    planned = _mapping(next(iter(cells.values())))
    cell = output / "levels" / str(planned["level"]) / f"seed_{planned['seed']}"
    receipt_path = cell / "run_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    field_damage = {
        "level": ("level", "wrong-level"),
        "seed": ("seed", int(planned["seed"]) + 1),
        "dt": ("dt_s", float(planned["dt_s"]) / 2.0),
        "depth": (
            "brownian_interval_tree_depth",
            int(planned["brownian_interval_tree_depth"]) + 1,
        ),
        "rtol": ("geometry_rtol", float(planned["geometry_rtol"]) / 10.0),
        "case_hash": ("case_sha256", "0" * 64),
        "status": ("status", "BLOCKED"),
        "artifact": ("trajectory_sha256", "0" * 64),
    }
    if damage == "case_artifact":
        (output / str(planned["case"])).write_text("changed\n", encoding="utf-8")
    else:
        field, value = field_damage[damage]
        receipt[field] = value
        _write_json(receipt_path, receipt)

    with pytest.raises(ValueError, match=r"receipt|case identity"):
        candidate.finalize(output)

    assert not (output / str(report["final_report"])).exists()


@pytest.mark.parametrize("campaign_in_recipe", [True, False])
def test_configured_campaign_identity_can_be_recipe_or_contract_driven(
    tmp_path: Path, campaign_in_recipe: bool
) -> None:
    recipe = _configured_recipe_fixture(
        tmp_path / "configuration", campaign_in_recipe=campaign_in_recipe
    )
    output = tmp_path / "configured-pilot"

    report = candidate.prepare(recipe, output)

    assert report["campaign_identity"] == CASE_P_CAMPAIGN
    cells = _mapping(report["cells"])
    first_seed = CASE_P_PILOT_CANDIDATE_SEEDS[0]
    first = _mapping(cells[f"macro_coarse/{first_seed}"])
    case = yaml.safe_load((output / str(first["case"])).read_text(encoding="utf-8"))
    assert case["case"]["name"] == f"m3c2_caseP_100nm_macro_coarse_seed_{first_seed}"
    _write_complete_receipts(output, cells)
    manifest = candidate.finalize(output)
    assert manifest["campaign_identity"] == CASE_P_CAMPAIGN
    assert manifest["evaluation_policy_sha256"] == candidate._sha256(CASE_P_POLICY)


def test_registered_case_p_recipe_prepares_contract_driven_pilot(tmp_path: Path) -> None:
    output = tmp_path / "case-p-pilot"

    report = candidate.prepare(CASE_P_RECIPE.resolve(), output)

    assert report["tool_revision"] == candidate.TOOL_REVISION
    assert report["campaign_identity"] == CASE_P_CAMPAIGN
    assert report["evaluation_policy_sha256"] == candidate._sha256(CASE_P_POLICY)
    assert (
        report["pilot_authorization"]
        == json.loads(CASE_P_RECIPE.read_text(encoding="utf-8"))["execution_authorization"]
    )
    assert _mapping(report["campaign"])["candidate_seeds"] == CASE_P_PILOT_CANDIDATE_SEEDS
    cells = _mapping(report["cells"])
    assert len(cells) == 16
    first = _mapping(cells[f"macro_coarse/{CASE_P_PILOT_CANDIDATE_SEEDS[0]}"])
    case = yaml.safe_load((output / str(first["case"])).read_text(encoding="utf-8"))
    assert case["case"]["name"] == "m3c2_caseP_100nm_macro_coarse_seed_919004"
    assert report["claim"] == "candidate pilot runner locked; numerical accuracy not evaluated"
    assert candidate._load_prepared(output)["pilot_authorization"] == report["pilot_authorization"]
    _write_complete_receipts(output, cells)
    manifest = candidate.finalize(output)
    assert manifest["evaluation_policy_sha256"] == candidate._sha256(CASE_P_POLICY)


def test_registered_case_p_v2_recipe_uses_v5_policy_without_case_a_contingency(
    tmp_path: Path,
) -> None:
    output = tmp_path / "case-p-v2-pilot"
    report = candidate.prepare(CASE_P_RECIPE_V2.resolve(), output)

    assert report["evaluation_policy_sha256"] == candidate._sha256(CASE_P_POLICY_V5)
    assert len(_mapping(report["cells"])) == 16
    assert _mapping(report["campaign"])["candidate_seeds"] == CASE_P_PILOT_CANDIDATE_SEEDS
    cells = _mapping(report["cells"])
    assert len(cells) == 16
    first = _mapping(cells[f"macro_coarse/{CASE_P_PILOT_CANDIDATE_SEEDS[0]}"])
    case = yaml.safe_load((output / str(first["case"])).read_text(encoding="utf-8"))
    assert case["case"]["name"] == "m3c2_caseP_100nm_macro_coarse_seed_919004"
    assert report["claim"] == "candidate pilot runner locked; numerical accuracy not evaluated"
    assert candidate._load_prepared(output)["pilot_authorization"] == report["pilot_authorization"]
    _write_complete_receipts(output, cells)
    manifest = candidate.finalize(output)
    assert manifest["evaluation_policy_sha256"] == candidate._sha256(CASE_P_POLICY_V5)


def test_configured_recipe_rejects_evaluation_policy_hash_drift(tmp_path: Path) -> None:
    recipe_path = _configured_recipe_fixture(tmp_path / "configuration")
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    recipe["evaluation_plan"]["policy"]["sha256"] = "0" * 64
    _write_json(recipe_path, recipe)

    with pytest.raises(ValueError, match="evaluation policy identity differs"):
        candidate.prepare(recipe_path, tmp_path / "must-not-exist")


def test_registered_case_p_recipe_requires_locked_authorization_before_run(tmp_path: Path) -> None:
    recipe_path = tmp_path / "not-authorized.json"
    recipe = json.loads(CASE_P_RECIPE.read_text(encoding="utf-8"))
    recipe["execution_authorization"] = {
        "status": "NOT_AUTHORIZED",
        "required_before_execution": ["lock_shared_pilot_authorization"],
    }
    _write_json(recipe_path, recipe)
    output = tmp_path / "case-p-pilot"
    candidate.prepare(recipe_path, output)

    with pytest.raises(ValueError, match="not authorized for execution"):
        candidate.run_cell(output, "macro_coarse", CASE_P_PILOT_CANDIDATE_SEEDS[0])


@pytest.mark.parametrize("damage", ["campaign", "contract", "content"])
def test_registered_case_p_preparation_rechecks_locked_authority(
    tmp_path: Path, damage: str
) -> None:
    output = tmp_path / "case-p-pilot"
    candidate.prepare(CASE_P_RECIPE.resolve(), output)
    report_path = output / candidate.PREPARE_REPORT
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if damage == "campaign":
        report["campaign_identity"]["case_id"] = "wrong-case"
    elif damage == "contract":
        report["contract_sha256"] = "c" * 64
    else:
        report["input_content_hash"] = "sha256:" + "c" * 64
    _write_json(report_path, report)

    with pytest.raises(ValueError, match="prepared campaign"):
        candidate._load_prepared(output)


@pytest.mark.parametrize(("recipe_revision", "policy_revision"), [(1, 4), (2, 5)])
def test_configured_final_registration_uses_campaign_kind_and_case_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recipe_revision: int,
    policy_revision: int,
) -> None:
    authority_root = tmp_path / "authority"
    contract_seed_source, final_seed_sets = _final_seed_allocation_fixture(authority_root)
    registration_seed_source = {
        **contract_seed_source,
        "path": (Path("..") / "authority" / str(contract_seed_source["path"])).as_posix(),
    }
    registration, _ = _final_registration_fixture(
        tmp_path / "registration",
        CASE_P_CAMPAIGN,
        {
            "comsol": CASE_P_PILOT_COMSOL_SEEDS,
            "candidate": CASE_P_PILOT_CANDIDATE_SEEDS,
        },
        final_seed_sets,
        registration_seed_source,
        policy_revision,
    )
    recipe = _configured_recipe_fixture(
        tmp_path / "configuration",
        final_seed_source=contract_seed_source,
        evaluation_policy_path=registration.parent / "evaluation_policy.json",
        recipe_revision=recipe_revision,
    )
    monkeypatch.setattr(candidate, "_repository_root", lambda: authority_root.resolve())
    monkeypatch.setattr(candidate, "_solver_project_root", lambda: tmp_path.resolve())
    output = tmp_path / "configured-final"

    report = candidate.prepare(recipe, output, registration)

    assert report["campaign_identity"] == CASE_P_CAMPAIGN
    assert _mapping(report["final_registration"])["evaluation_authority"] == {
        "evaluation_policy_revision": policy_revision,
        "pilot_evaluator_revision": f"m3c2_stochastic_ensemble_evaluator_v{policy_revision}",
    }
    cells = _mapping(report["cells"])
    first = _mapping(cells[f"macro_coarse/{CASE_P_FINAL_CANDIDATE_SEEDS[0]}"])
    case = yaml.safe_load((output / str(first["case"])).read_text(encoding="utf-8"))
    assert case["case"]["name"].startswith("m3c2_caseP_100nm_")


def test_configured_final_rejects_another_recipe_evaluation_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = _configured_recipe_fixture(tmp_path / "configuration")
    registration, _ = _final_registration_fixture(
        tmp_path / "registration", CASE_P_CAMPAIGN, policy_revision=4
    )
    monkeypatch.setattr(candidate, "_solver_project_root", lambda: tmp_path.resolve())

    with pytest.raises(ValueError, match="another recipe evaluation policy"):
        candidate.prepare(recipe, tmp_path / "must-not-exist", registration)


def test_configured_campaign_rejects_recipe_contract_identity_drift(tmp_path: Path) -> None:
    recipe_path = _configured_recipe_fixture(tmp_path / "configuration")
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    recipe["campaign"]["evaluation_case_id"] = "wrong-case"
    _write_json(recipe_path, recipe)

    with pytest.raises(ValueError, match="recipe and contract campaign identities differ"):
        candidate.prepare(recipe_path, tmp_path / "must-not-exist")


def test_new_campaign_identity_is_contract_mandatory() -> None:
    recipe = json.loads(RECIPE.read_text(encoding="utf-8"))
    contract = json.loads(
        (Path(__file__).parents[1] / "cases/m3c2_caseA_100nm_stochastic_pilot_v1.json").read_text(
            encoding="utf-8"
        )
    )
    recipe["campaign"] = CASE_P_CAMPAIGN

    with pytest.raises(ValueError, match="contracts must define campaign identity"):
        candidate._campaign_identity(recipe, contract)


def test_registered_final_prepares_one_selected_level_and_32_independent_seeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registration, _ = _final_registration_fixture(tmp_path)
    monkeypatch.setattr(candidate, "_solver_project_root", lambda: tmp_path.resolve())
    output = tmp_path / "candidate-final"

    report = candidate.prepare(QUALIFIED_MACRO_RECIPE.resolve(), output, registration)

    cells = _assert_prepared_final_campaign(report, output)

    _write_complete_receipts(output, cells)
    manifest = candidate.finalize(output)
    _assert_final_campaign_manifest(manifest, output)


def test_final_registration_rejects_overlap_and_changed_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registration, selection = _final_registration_fixture(tmp_path)
    monkeypatch.setattr(candidate, "_solver_project_root", lambda: tmp_path.resolve())
    document = json.loads(registration.read_text(encoding="utf-8"))
    document["participant_seed_sets"]["candidate"][0] = FINAL_COMSOL_SEEDS[0]
    _write_json(registration, document)
    with pytest.raises(ValueError, match="seed sets must be disjoint"):
        candidate.prepare(
            QUALIFIED_MACRO_RECIPE.resolve(), tmp_path / "overlap-output", registration
        )

    registration, selection = _final_registration_fixture(tmp_path / "changed")
    monkeypatch.setattr(candidate, "_solver_project_root", lambda: selection.parent.resolve())
    selection.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="registered SHA-256"):
        candidate.prepare(
            QUALIFIED_MACRO_RECIPE.resolve(), tmp_path / "changed-output", registration
        )

    registration, _ = _final_registration_fixture(tmp_path / "policy-mismatch")
    pilot = registration.parent / "pilot_evaluation.json"
    pilot_document = json.loads(pilot.read_text(encoding="utf-8"))
    pilot_document["policy"]["revision"] = 8
    _write_json(pilot, pilot_document)
    registration_document = json.loads(registration.read_text(encoding="utf-8"))
    registration_document["execution_authorization"]["pilot_evaluation"]["sha256"] = (
        candidate._sha256(pilot)
    )
    _write_json(registration, registration_document)
    monkeypatch.setattr(candidate, "_solver_project_root", lambda: pilot.parent.resolve())
    with pytest.raises(ValueError, match="another ensemble policy"):
        candidate.prepare(
            QUALIFIED_MACRO_RECIPE.resolve(), tmp_path / "policy-output", registration
        )


@pytest.mark.parametrize(
    ("legacy_form", "message"),
    [
        ("nested", "pilot configuration screening must be a mapping"),
        ("qualified_field", "largest screened candidate macro level"),
        ("revision_2", "passing revision-3 pilot evaluation"),
    ],
)
def test_final_registration_rejects_obsolete_pilot_screening_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    legacy_form: str,
    message: str,
) -> None:
    registration, selection = _final_registration_fixture(tmp_path)
    pilot = registration.parent / "pilot_evaluation.json"
    pilot_document = json.loads(pilot.read_text(encoding="utf-8"))
    screening = pilot_document["pilot_configuration_screening"]
    if legacy_form == "nested":
        pilot_document.pop("pilot_configuration_screening")
        pilot_document["decision"] = {"pilot_configuration_screening": screening}
    elif legacy_form == "qualified_field":
        candidate_screen = screening["candidate"]
        candidate_screen["largest_qualified_macro_level"] = candidate_screen.pop(
            "largest_screened_macro_level"
        )
    else:
        pilot_document["schema_version"] = 2
        pilot_document["tool_revision"] = "m3c2_stochastic_ensemble_evaluator_v2"
    _write_json(pilot, pilot_document)

    selection_document = json.loads(selection.read_text(encoding="utf-8"))
    selection_document["pilot_report_sha256"] = candidate._sha256(pilot)
    _write_json(selection, selection_document)
    registration_document = json.loads(registration.read_text(encoding="utf-8"))
    authorization = registration_document["execution_authorization"]
    authorization["pilot_evaluation"]["sha256"] = candidate._sha256(pilot)
    authorization["selection_receipt"]["sha256"] = candidate._sha256(selection)
    _write_json(registration, registration_document)

    monkeypatch.setattr(candidate, "_solver_project_root", lambda: tmp_path.resolve())
    with pytest.raises(ValueError, match=message):
        candidate.prepare(
            QUALIFIED_MACRO_RECIPE.resolve(), tmp_path / "nested-output", registration
        )


def test_read_only_result_manifest_is_accepted_and_performance_recovery_is_explicit(
    tmp_path: Path,
) -> None:
    mapped = candidate._mapping(MappingProxyType({"status": "complete"}), "manifest")
    result = tmp_path / "result"
    result.mkdir()
    (result / "run.json").write_text("{}", encoding="utf-8")

    performance = candidate._performance_record(
        result,
        MappingProxyType({"counts": {"particles": 287}}),
        None,
        None,
    )

    assert mapped == {"status": "complete"}
    assert performance["wall_time_s"] is None
    assert performance["peak_rss_bytes"] is None
    assert performance["measurement_status"] == "NOT_MEASURED_OR_NOT_RECOVERABLE"
    assert performance["output_bytes_scope"] == "solver_result_directory"


def test_dense_projection_only_fills_post_escape_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(candidate, "PARTICLE_COUNT", 2)
    monkeypatch.setattr(candidate, "OUTPUT_COUNT", 3)
    monkeypatch.setattr(candidate, "_output_times", lambda: [0.0, 0.5, 1.0])

    def frame(time_s: float, particle_ids: list[int], lifecycle: list[int]) -> SimpleNamespace:
        count = len(particle_ids)
        return SimpleNamespace(
            time_s=time_s,
            particle_id=np.asarray(particle_ids),
            position_m=np.column_stack((np.arange(count), np.arange(count) + 1.0)),
            velocity_m_s=np.ones((count, 2)),
            charge_number=np.full(count, -1.0),
            lifecycle=np.asarray(lifecycle),
        )

    frames = [frame(0.0, [1, 2], [1, 1]), frame(0.5, [1], [1]), frame(1.0, [1], [2])]
    events = SimpleNamespace(
        particle_id=np.asarray([2]),
        outcome=np.asarray(["escaped"]),
        time_s=np.asarray([0.25]),
    )
    result = SimpleNamespace(
        iter_frames=lambda: iter(frames),
        read_boundary_events=lambda: events,
    )
    path = tmp_path / "trajectory.csv"

    assert candidate._write_trajectory(path, result) == 6

    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    escaped = [row for row in rows if row["particle_id"] == "2" and float(row["time_s"]) > 0]
    assert len(escaped) == 2
    assert all(row["lifecycle"] == "escaped" for row in escaped)
    assert all(math.isnan(float(row["r_m"])) for row in escaped)
    stuck = next(row for row in rows if row["particle_id"] == "1" and row["time_s"] == "1")
    assert stuck["lifecycle"] == "stuck"

    missing_without_escape = SimpleNamespace(
        iter_frames=lambda: iter(frames),
        read_boundary_events=lambda: SimpleNamespace(
            particle_id=np.empty(0), outcome=np.empty(0), time_s=np.empty(0)
        ),
    )
    with pytest.raises(ValueError, match="interior or non-escape"):
        invalid = tmp_path / "invalid.csv"
        candidate._write_trajectory(invalid, missing_without_escape)
    assert not invalid.exists()


def test_failed_run_projection_is_sparse_and_explicit(tmp_path: Path) -> None:
    frame = SimpleNamespace(
        time_s=0.0,
        particle_id=np.asarray([1, 2]),
        position_m=np.asarray([[0.1, 0.2], [0.3, 0.4]]),
        velocity_m_s=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        charge_number=np.asarray([-1.0, -2.0]),
        lifecycle=np.asarray([1, 4]),
    )
    result = SimpleNamespace(iter_frames=lambda: iter([frame]))
    path = tmp_path / "trajectory.blocked.csv"

    assert candidate._write_observed_trajectory(path, result) == 2

    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["lifecycle"] for row in rows] == ["active", "failed"]
    assert candidate.BLOCKED_TRAJECTORY_REVISION == "observed_rows_only_failed_run_v1"
