"""Prepare and execute a registered M3-C2A candidate campaign.

This external V&V runner reuses the locked canonical P1 input and the ordinary
``load_case -> simulate -> open_result`` API.  It owns no solver physics and
does not introduce a COMSOL-specific production path.  With no registration it
preserves the historical Case-A pilot workflow.  New campaigns carry one
contract-locked identity block.  A final campaign additionally needs one
post-pilot registration whose locked selection receipt names the single
qualified candidate level and the independent 32-seed cohort.
"""

from __future__ import annotations

import argparse
import copy
import csv
import ctypes
import hashlib
import importlib
import json
import math
import os
import shutil
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, Literal, cast

import yaml

from chamber_particles import load_case, open_result, simulate
from chamber_particles.case_format import read_with_info

TOOL_REVISION: Final = "m3c2_candidate_campaign_runner_v4"
LEGACY_TOOL_REVISION: Final = "m3c2_caseA_100nm_candidate_campaign_runner_v3"
TRAJECTORY_NORMALIZATION_REVISION: Final = "dense_schedule_escape_nan_suffix_v1"
BLOCKED_TRAJECTORY_REVISION: Final = "observed_rows_only_failed_run_v1"
RECIPE_COPY: Final = "candidate_pilot_recipe.json"
FINAL_REGISTRATION_COPY: Final = "candidate_final_registration.json"
PREPARE_REPORT: Final = "prepare_report.json"
FINAL_REPORT: Final = "candidate_pilot_manifest.json"
FINAL_CAMPAIGN_REPORT: Final = "candidate_campaign_manifest.json"
PARTICLE_COUNT: Final = 287
OUTPUT_COUNT: Final = 121
END_TIME_S: Final = 0.03
FINAL_REPLICAS: Final = 32
FINAL_REGISTRATION_KIND: Final = "m3c2_caseA_100nm_final_campaign"
FINAL_AUTHORIZATION_KIND: Final = "m3c2_post_pilot_final_authorization"
FINAL_EVALUATION_POLICY_REVISION: Final = 3
SUPPORTED_EVALUATION_POLICY_REVISIONS: Final = {3, 4, 5}
CASE_ID: Final = "formal_iondrag_theory_consistent/caseA_100nm"
EVALUATION_CASE_ID: Final = "M3-C2A_caseA_100nm_common-P1"
CASEP_EVENT_TOLERANCE_RECIPE_ID: Final = "M3-C2A-caseP-100nm-event-tolerance-pre-final"
CASEP_CONTRACT_ID: Final = "M3-C2A-caseP-100nm-stochastic-pilot"
CASEP_CASE_ID: Final = "formal_iondrag_theory_consistent/caseP_100nm"
CASEP_EVALUATION_CASE_ID: Final = "M3-C2A_caseP_100nm_common-P1"
CAMPAIGN_IDENTITY_KEYS: Final = {
    "case_id",
    "evaluation_case_id",
    "output_slug",
    "final_registration_kind",
    "candidate_case_name_prefix",
}
LEGACY_CAMPAIGN_IDENTITY: Final = {
    "case_id": CASE_ID,
    "evaluation_case_id": EVALUATION_CASE_ID,
    "output_slug": "caseA_100nm",
    "final_registration_kind": FINAL_REGISTRATION_KIND,
    "candidate_case_name_prefix": "m3c2_caseA_100nm",
}
type Purpose = Literal["pilot", "final"]
_LIFECYCLE: Final = {
    0: "pending",
    1: "active",
    2: "stuck",
    3: "escaped",
    4: "failed",
    5: "held",
}
_TRAJECTORY_HEADER: Final = (
    "particle_id",
    "time_s",
    "r_m",
    "z_m",
    "velocity_r_m_per_s",
    "velocity_z_m_per_s",
    "charge_number_e",
    "lifecycle",
)
_EVENT_HEADER: Final = (
    "particle_id",
    "event_ordinal",
    "event_time_s",
    "hit_r_m",
    "hit_z_m",
    "normal_r",
    "normal_z",
    "pre_velocity_r_m_per_s",
    "pre_velocity_z_m_per_s",
    "post_velocity_r_m_per_s",
    "post_velocity_z_m_per_s",
    "event_type",
    "boundary_semantic",
    "law",
    "outcome",
)
_FAILURE_HEADER: Final = (
    "particle_id",
    "event_ordinal",
    "event_time_s",
    "reason_code",
)


def _mapping(value: object, location: str) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be a mapping")
    return dict(value)


def _sequence(value: object, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{location} must be a list")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _load_json(path: Path, location: str) -> dict[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), location)


def _object_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _integer_list(value: object, location: str) -> list[int]:
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise ValueError(f"{location} must be an integer list")
    result = cast(list[int], value)
    if len(result) != len(set(result)) or any(seed < 0 or seed >= 1 << 64 for seed in result):
        raise ValueError(f"{location} must contain unique unsigned-64-bit seeds")
    return result


def _campaign_identity_record(value: object, location: str) -> dict[str, str]:
    record = _mapping(value, location)
    if set(record) != CAMPAIGN_IDENTITY_KEYS:
        raise ValueError(f"{location} must contain exactly the registered identity fields")
    if any(
        not isinstance(record[key], str) or not record[key] or record[key] != record[key].strip()
        for key in CAMPAIGN_IDENTITY_KEYS
    ):
        raise ValueError(f"{location} values must be nonempty strings")
    identity = {key: cast(str, record[key]) for key in CAMPAIGN_IDENTITY_KEYS}
    safe_characters = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-")
    for key in ("output_slug", "candidate_case_name_prefix"):
        if any(character not in safe_characters for character in identity[key]):
            raise ValueError(f"{location}.{key} must be a path-safe identifier")
    return identity


def _campaign_identity(
    recipe: dict[str, Any], contract: dict[str, Any]
) -> tuple[dict[str, str], bool]:
    recipe_value = recipe.get("campaign")
    contract_value = contract.get("campaign")
    if recipe_value is None and contract_value is None:
        final_cohort = _mapping(
            _mapping(contract.get("seed_plan"), "contract seed plan").get("final_cohort"),
            "contract final cohort",
        )
        if (
            contract.get("contract_id") != "M3-C2A-caseA-100nm-stochastic-pilot"
            or final_cohort.get("case_id") != CASE_ID
        ):
            raise ValueError("new M3-C2 contracts must define campaign identity")
        return dict(LEGACY_CAMPAIGN_IDENTITY), False
    if contract_value is None:
        raise ValueError("new M3-C2 contracts must define campaign identity")
    recipe_identity = (
        None if recipe_value is None else _campaign_identity_record(recipe_value, "recipe campaign")
    )
    contract_identity = _campaign_identity_record(contract_value, "contract campaign")
    if recipe_identity is not None and recipe_identity != contract_identity:
        raise ValueError("recipe and contract campaign identities differ")
    return dict(recipe_identity or contract_identity), True


def _recipe_candidate_seeds(recipe: dict[str, Any], contract: dict[str, Any]) -> list[int]:
    pilot = _mapping(recipe.get("pilot"), "pilot")
    inline = pilot.get("candidate_seeds")
    source_value = pilot.get("seed_source")
    if inline is not None and source_value is not None:
        raise ValueError("pilot candidate seeds must have one owner")
    if inline is not None:
        return _integer_list(inline, "pilot candidate seeds")
    source = _mapping(source_value, "pilot seed source")
    if set(source) != {"path", "sha256", "json_pointer"}:
        raise ValueError("pilot seed source has unexpected keys")
    contract_record = _mapping(recipe.get("contract"), "contract")
    if (
        source.get("path") != contract_record.get("path")
        or source.get("sha256") != contract_record.get("sha256")
        or source.get("json_pointer") != "/seed_plan/pilot/candidate_seeds"
    ):
        raise ValueError("pilot seed source must point to the locked contract candidate seeds")
    contract_pilot = _mapping(
        _mapping(contract.get("seed_plan"), "contract seed plan").get("pilot"),
        "contract pilot",
    )
    return _integer_list(contract_pilot.get("candidate_seeds"), "contract candidate seeds")


def _solver_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _registration_artifact(
    registration_path: Path,
    project_root: Path,
    value: object,
    location: str,
) -> Path:
    record = _mapping(value, location)
    if set(record) != {"path", "sha256"}:
        raise ValueError(f"{location} has unexpected keys")
    relative = Path(str(record["path"]))
    if relative.is_absolute() or str(relative) in {"", "."}:
        raise ValueError(f"{location}.path must be relative to the registration")
    path = (registration_path.parent / relative).resolve()
    if not path.is_relative_to(project_root) or not path.is_file():
        raise ValueError(f"{location} is missing or outside the solver project")
    if _sha256(path) != str(record["sha256"]).lower():
        raise ValueError(f"{location} differs from its registered SHA-256")
    return path


def _locked_path(root: Path, record: object, location: str) -> Path:
    item = _mapping(record, location)
    if set(item) != {"path", "sha256"} and set(item) != {
        "path",
        "sha256",
        "content_hash",
    }:
        raise ValueError(f"{location} has unexpected keys")
    path = root / str(item["path"])
    if not path.is_file() or _sha256(path) != item["sha256"]:
        raise ValueError(f"{location} identity differs: {path}")
    return path


def _recipe_evaluation_policy_sha256(
    recipe: dict[str, Any], root: Path, *, required: bool
) -> str | None:
    plan_value = recipe.get("evaluation_plan")
    if plan_value is None:
        if required:
            raise ValueError("configured candidate recipe must lock an evaluation policy")
        return None
    plan = _mapping(plan_value, "evaluation plan")
    policy_value = plan.get("policy")
    if policy_value is None:
        if required:
            raise ValueError("configured candidate recipe must lock an evaluation policy")
        return None
    policy = _mapping(policy_value, "evaluation policy")
    if set(policy) != {"path", "sha256"}:
        raise ValueError("evaluation policy must contain exactly path and sha256")
    path = _locked_path(root, policy, "evaluation policy")
    return _sha256(path)


def _configured_authorization_record(recipe: dict[str, Any]) -> dict[str, Any]:
    authorization = _mapping(
        recipe.get("execution_authorization"), "configured execution authorization"
    )
    if "status" in authorization:
        if set(authorization) - {"status", "required_before_execution"}:
            raise ValueError("configured execution authorization has unexpected keys")
        if authorization.get("status") not in {"AUTHORIZED", "NOT_AUTHORIZED"}:
            raise ValueError("configured execution authorization status is invalid")
        if "required_before_execution" in authorization:
            blockers = _sequence(
                authorization["required_before_execution"],
                "configured execution authorization blockers",
            )
            if any(not isinstance(value, str) or not value.strip() for value in blockers):
                raise ValueError("configured execution authorization blockers are invalid")
        return authorization
    if set(authorization) != {"path", "sha256"}:
        raise ValueError("configured execution authorization reference is invalid")
    return authorization


def _configured_authorization(
    recipe: dict[str, Any], root: Path, campaign_identity: dict[str, str]
) -> tuple[str, dict[str, str] | None]:
    authorization = _configured_authorization_record(recipe)
    if "status" in authorization:
        return cast(str, authorization["status"]), None
    path = _locked_path(root, authorization, "configured execution authorization")
    receipt = _load_json(path, "configured execution authorization")
    status = receipt.get("status")
    if status not in {"AUTHORIZED", "AUTHORIZED_BY_EXPLICIT_USER_DIRECTION"}:
        raise ValueError("configured execution authorization receipt is not authorized")
    identity = (
        receipt.get("schema_version"),
        receipt.get("authorization_kind"),
        receipt.get("purpose"),
        receipt.get("participants"),
    )
    if identity != (1, "m3c2_pilot_execution_authorization", "pilot", ["comsol", "candidate"]):
        raise ValueError("configured execution authorization identity or scope is invalid")
    if receipt.get("contract") != recipe.get("contract"):
        raise ValueError("configured execution authorization refers to another contract")
    if receipt.get("contract_receipt") != recipe.get("contract_receipt"):
        raise ValueError("configured execution authorization refers to another contract receipt")
    if receipt.get("campaign") != campaign_identity:
        raise ValueError("configured execution authorization campaign differs")
    source = {
        "path": str(authorization["path"]),
        "sha256": str(authorization["sha256"]).lower(),
    }
    return cast(str, status), source


def _seed_allocation_source(
    value: object, location: str, base: Path, allowed_root: Path
) -> tuple[Path, str, dict[str, Any]]:
    record = _mapping(value, location)
    if set(record) != {"path", "sha256", "json_pointers"}:
        raise ValueError(f"{location} has unexpected keys")
    pointers = _mapping(record.get("json_pointers"), f"{location} JSON pointers")
    if pointers != {
        "comsol": "/participant_seed_sets/comsol",
        "candidate": "/participant_seed_sets/candidate",
    }:
        raise ValueError(f"{location} JSON pointers are invalid")
    relative = Path(str(record.get("path", "")))
    if relative.is_absolute() or str(relative) in {"", "."}:
        raise ValueError(f"{location}.path must be relative")
    path = (base / relative).resolve()
    expected = str(record.get("sha256", "")).lower()
    if not path.is_relative_to(allowed_root) or not path.is_file() or _sha256(path) != expected:
        raise ValueError(f"{location} differs from its registered SHA-256")
    return path, expected, _load_json(path, "final seed allocation")


def _validate_final_allocation_identity(
    allocation: dict[str, Any],
    contract_final: dict[str, Any],
    campaign_identity: dict[str, str],
) -> None:
    identity = (
        allocation.get("schema_version"),
        allocation.get("allocation_kind"),
        allocation.get("case_id"),
        allocation.get("purpose"),
        allocation.get("replicas_per_participant"),
    )
    expected = (
        1,
        "m3c2_final_seed_allocation",
        campaign_identity["case_id"],
        "final",
        FINAL_REPLICAS,
    )
    if identity != expected:
        raise ValueError("final seed allocation identity or scope is invalid")
    if contract_final.get("case_id") != campaign_identity["case_id"]:
        raise ValueError("contract final cohort differs from the seed allocation")
    if int(contract_final.get("replicas_per_participant", -1)) != FINAL_REPLICAS:
        raise ValueError("contract final cohort differs from the seed allocation")


def _referenced_final_seed_participants(
    source_values: tuple[object, object, object],
    registration: dict[str, Any],
    policy_final: dict[str, Any],
    contract_final: dict[str, Any],
    campaign_identity: dict[str, str],
    registration_path: Path,
    policy_path: Path,
) -> dict[str, Any]:
    if any(value is None for value in source_values):
        raise ValueError("final seed allocation must be locked by every campaign authority")
    if registration.get("participant_seed_sets") is not None:
        raise ValueError("referenced final seeds must have one owner")
    if policy_final.get("seed_plan") is not None:
        raise ValueError("referenced final seeds must have one owner")

    project_root = _solver_project_root()
    repository_root = _repository_root()
    resolved = [
        _seed_allocation_source(
            source_values[0],
            "registration participant seed source",
            registration_path.parent,
            project_root,
        ),
        _seed_allocation_source(
            source_values[1],
            "evaluation policy seed allocation",
            policy_path.parent,
            project_root,
        ),
        _seed_allocation_source(
            source_values[2],
            "contract final seed source",
            repository_root,
            repository_root,
        ),
    ]
    if len({(path, digest) for path, digest, _ in resolved}) != 1:
        raise ValueError("campaign authorities reference different final seed allocations")
    allocation = resolved[0][2]
    _validate_final_allocation_identity(allocation, contract_final, campaign_identity)
    return _mapping(allocation.get("participant_seed_sets"), "allocated participant seed sets")


def _legacy_final_seed_participants(
    registration: dict[str, Any],
    campaign_identity: dict[str, str],
    *,
    explicit_campaign_identity: bool,
) -> dict[str, Any]:
    if explicit_campaign_identity or campaign_identity != LEGACY_CAMPAIGN_IDENTITY:
        raise ValueError("new campaigns must reference one final seed allocation")
    return _mapping(registration.get("participant_seed_sets"), "registration participant seed sets")


def _validate_legacy_policy_seeds(
    policy_final: dict[str, Any], comsol_seeds: list[int], candidate_seeds: list[int]
) -> None:
    policy_seed_plan = _mapping(policy_final.get("seed_plan"), "evaluation policy seed plan")
    if _integer_list(policy_seed_plan.get("comsol"), "policy COMSOL seeds") != comsol_seeds:
        raise ValueError("final registration seed sets differ from the registered policy")
    if (
        _integer_list(policy_seed_plan.get("candidate"), "policy candidate seeds")
        != candidate_seeds
    ):
        raise ValueError("final registration seed sets differ from the registered policy")


def _validated_final_seed_values(
    participants: dict[str, Any],
    contract: dict[str, Any],
    policy_final: dict[str, Any],
    *,
    referenced: bool,
) -> tuple[list[int], list[int], set[int]]:
    if set(participants) != {"comsol", "candidate"}:
        raise ValueError("final registration must contain exactly COMSOL and candidate seeds")
    comsol_seeds = _integer_list(participants["comsol"], "final COMSOL seeds")
    candidate_seeds = _integer_list(participants["candidate"], "final candidate seeds")
    if any(seed > 2_147_483_647 for seed in comsol_seeds):
        raise ValueError("final COMSOL seeds must fit the Java integer range")
    if len(comsol_seeds) != FINAL_REPLICAS or len(candidate_seeds) != FINAL_REPLICAS:
        raise ValueError("final campaign requires exactly 32 seeds per participant")
    if set(comsol_seeds).intersection(candidate_seeds):
        raise ValueError("final participant seed sets must be disjoint")
    pilot = _mapping(
        _mapping(contract.get("seed_plan"), "contract seed plan").get("pilot"),
        "contract pilot seed plan",
    )
    pilot_seeds = set(_integer_list(pilot.get("comsol_seeds"), "pilot COMSOL seeds"))
    pilot_seeds.update(_integer_list(pilot.get("candidate_seeds"), "pilot candidate seeds"))
    if pilot_seeds.intersection(comsol_seeds):
        raise ValueError("final seeds must be disjoint from both pilot participants")
    if pilot_seeds.intersection(candidate_seeds):
        raise ValueError("final seeds must be disjoint from both pilot participants")
    if not referenced:
        _validate_legacy_policy_seeds(policy_final, comsol_seeds, candidate_seeds)
    if int(policy_final.get("replicas_per_participant", -1)) != FINAL_REPLICAS:
        raise ValueError("final registration replica count differs from the registered policy")
    return comsol_seeds, candidate_seeds, pilot_seeds


def _final_seed_sets(
    registration: dict[str, Any],
    policy: dict[str, Any],
    contract: dict[str, Any],
    campaign_identity: dict[str, str],
    registration_path: Path,
    policy_path: Path,
    *,
    explicit_campaign_identity: bool,
) -> tuple[list[int], list[int], set[int]]:
    policy_final = _mapping(policy.get("final"), "evaluation policy final section")
    contract_final = _mapping(
        _mapping(contract.get("seed_plan"), "contract seed plan").get("final_cohort"),
        "contract final cohort",
    )
    source_values = (
        registration.get("participant_seed_source"),
        policy_final.get("seed_allocation"),
        contract_final.get("seed_source"),
    )
    referenced = any(value is not None for value in source_values)
    if referenced:
        participants = _referenced_final_seed_participants(
            source_values,
            registration,
            policy_final,
            contract_final,
            campaign_identity,
            registration_path,
            policy_path,
        )
    else:
        participants = _legacy_final_seed_participants(
            registration,
            campaign_identity,
            explicit_campaign_identity=explicit_campaign_identity,
        )
    return _validated_final_seed_values(participants, contract, policy_final, referenced=referenced)


def _selected_candidate_level(
    recipe: dict[str, Any],
    pilot_evaluation: dict[str, Any],
    selection: dict[str, Any],
) -> dict[str, object]:
    selected = _mapping(selection.get("selected_final_levels"), "selected final levels")
    if set(selected) != {"comsol", "candidate"}:
        raise ValueError("selection receipt must select exactly COMSOL and candidate levels")
    candidate = _mapping(selected["candidate"], "selected candidate level")
    if int(candidate.get("ordinal", -1)) != 0:
        raise ValueError("selected candidate final level ordinal must be zero")
    level_id = str(candidate.get("level_id", ""))
    numerical = _mapping(candidate.get("numerical_setting"), "candidate numerical setting")
    if (
        set(numerical)
        != {
            "dt_s",
            "brownian_interval_tree_depth",
            "geometry_rtol",
            "purpose",
        }
        or numerical.get("purpose") != "accepted_final"
    ):
        raise ValueError("selected candidate final numerical setting is invalid")
    pilot_levels = {
        str(level["name"]): level
        for value in _sequence(_mapping(recipe["pilot"], "pilot")["levels"], "pilot levels")
        for level in [_mapping(value, "pilot level")]
        if level.get("purpose") == "macro_step_convergence"
    }
    if level_id not in pilot_levels:
        raise ValueError("selected candidate level is not a macro level from the qualified pilot")
    qualified = pilot_levels[level_id]
    for key in ("dt_s", "brownian_interval_tree_depth", "geometry_rtol"):
        if numerical.get(key) != qualified.get(key):
            raise ValueError("selected candidate numerical setting differs from its pilot level")
    screening = _mapping(
        pilot_evaluation.get("pilot_configuration_screening"),
        "pilot configuration screening",
    )
    candidate_screen = _mapping(screening.get("candidate"), "candidate pilot screening")
    if candidate_screen.get("largest_screened_macro_level") != level_id:
        raise ValueError("selection is not the largest screened candidate macro level")
    return {"name": level_id, **numerical}


def _validate_comsol_selection(registration: dict[str, Any], selection: dict[str, Any]) -> None:
    registered = _mapping(
        registration.get("comsol_numerical_setting"), "registered COMSOL numerical setting"
    )
    if (
        set(registered) != {"integrator", "fixed_step_s"}
        or registered.get("integrator") != "classical_rk4"
    ):
        raise ValueError("registered COMSOL numerical setting is invalid")
    selected = _mapping(selection.get("selected_final_levels"), "selected final levels")
    comsol = _mapping(selected.get("comsol"), "selected COMSOL level")
    numerical = _mapping(comsol.get("numerical_setting"), "selected COMSOL numerical setting")
    expected = {**registered, "purpose": "accepted_final"}
    if int(comsol.get("ordinal", -1)) != 0 or numerical != expected:
        raise ValueError("selected COMSOL setting differs from the final registration")


def _final_authority_documents(
    registration: dict[str, Any],
    registration_path: Path,
    *,
    explicit_campaign_identity: bool,
) -> tuple[
    dict[str, Path],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, object],
]:
    authorization = _mapping(
        registration.get("execution_authorization"), "final execution authorization"
    )
    if authorization.get("status") != "AUTHORIZED":
        raise ValueError("final campaign is not authorized for execution")
    artifacts = {
        name: _registration_artifact(
            registration_path,
            _solver_project_root(),
            authorization.get(name),
            f"execution authorization {name}",
        )
        for name in ("evaluation_policy", "pilot_evaluation", "selection_receipt")
    }
    policy = _load_json(artifacts["evaluation_policy"], "final evaluation policy")
    policy_revision = policy.get("policy_revision")
    required_revisions = (
        {4, 5} if explicit_campaign_identity else {FINAL_EVALUATION_POLICY_REVISION}
    )
    if (
        policy.get("policy_kind") != "m3c2_stochastic_ensemble"
        or policy_revision not in SUPPORTED_EVALUATION_POLICY_REVISIONS
        or policy_revision not in required_revisions
    ):
        required = " or ".join(str(value) for value in sorted(required_revisions))
        raise ValueError(f"final campaign requires M3-C2 ensemble policy revision {required}")
    evaluator_revision = f"m3c2_stochastic_ensemble_evaluator_v{policy_revision}"
    pilot = _load_json(artifacts["pilot_evaluation"], "pilot evaluation")
    if (
        pilot.get("schema_version") != policy_revision
        or pilot.get("tool_revision") != evaluator_revision
        or pilot.get("phase") != "pilot"
        or pilot.get("status") != "PASS"
    ):
        raise ValueError(
            f"final campaign requires the passing revision-{policy_revision} pilot evaluation"
        )
    pilot_policy = _mapping(pilot.get("policy"), "pilot evaluation policy")
    if (
        pilot_policy.get("sha256") != _sha256(artifacts["evaluation_policy"])
        or pilot_policy.get("revision") != policy_revision
    ):
        raise ValueError("pilot evaluation uses another ensemble policy")
    selection = _load_json(artifacts["selection_receipt"], "final selection receipt")
    if (
        selection.get("schema_version"),
        selection.get("receipt_kind"),
        selection.get("status"),
    ) != (1, FINAL_AUTHORIZATION_KIND, "AUTHORIZED_FOR_CONFIRMATORY_FINAL"):
        raise ValueError("final selection receipt is not an accepted authorization")
    authority_identity: dict[str, object] = {
        "evaluation_policy_revision": policy_revision,
        "pilot_evaluator_revision": evaluator_revision,
    }
    return artifacts, policy, pilot, selection, authority_identity


def _validate_final_registration(
    recipe: dict[str, Any],
    registration_path: Path,
    contract_path: Path,
    campaign_identity: dict[str, str],
    evaluation_policy_sha256: str | None,
    *,
    explicit_campaign_identity: bool,
) -> tuple[list[int], dict[str, object], dict[str, Path], dict[str, object]]:
    if not explicit_campaign_identity and int(recipe.get("recipe_revision", -1)) != 5:
        raise ValueError("final campaign requires the qualified revision-5 pilot recipe")
    registration = _load_json(registration_path, "final campaign registration")
    if (
        registration.get("schema_version"),
        registration.get("registration_kind"),
        registration.get("case_id"),
        registration.get("purpose"),
    ) != (
        1,
        campaign_identity["final_registration_kind"],
        campaign_identity["case_id"],
        "final",
    ):
        raise ValueError("final campaign registration identity is invalid")
    artifacts, policy, pilot_evaluation, selection, authority_identity = _final_authority_documents(
        registration,
        registration_path,
        explicit_campaign_identity=explicit_campaign_identity,
    )
    _validate_final_evaluation_policy(artifacts, evaluation_policy_sha256)
    contract = _load_json(contract_path, "M3-C2 pilot contract")
    comsol_seeds, candidate_seeds, locked_pilot_seeds = _final_seed_sets(
        registration,
        policy,
        contract,
        campaign_identity,
        registration_path,
        artifacts["evaluation_policy"],
        explicit_campaign_identity=explicit_campaign_identity,
    )
    pilot_seed_sets = _mapping(pilot_evaluation.get("seed_sets"), "pilot evaluation seed sets")
    if set(pilot_seed_sets) != {"comsol", "candidate"}:
        raise ValueError("pilot evaluation seed sets are invalid")
    observed_pilot_seeds = {
        seed
        for participant in ("comsol", "candidate")
        for seed in _integer_list(pilot_seed_sets.get(participant), f"pilot {participant} seeds")
    }
    if observed_pilot_seeds != locked_pilot_seeds:
        raise ValueError("pilot evaluation cohort differs from the locked pilot contract")
    if observed_pilot_seeds.intersection(comsol_seeds + candidate_seeds):
        raise ValueError("final seeds overlap the evaluated pilot cohort")
    seed_plan = {"comsol": comsol_seeds, "candidate": candidate_seeds}
    pilot_scope = _mapping(pilot_evaluation.get("scope_fingerprint"), "pilot scope fingerprint")
    if (
        selection.get("policy_sha256") != _sha256(artifacts["evaluation_policy"])
        or selection.get("pilot_report_sha256") != _sha256(artifacts["pilot_evaluation"])
        or selection.get("pilot_scope_sha256") != pilot_scope.get("sha256")
        or selection.get("common_design_sha256") != pilot_scope.get("common_design_sha256")
        or selection.get("seed_plan_sha256") != _object_sha256(seed_plan)
    ):
        raise ValueError("final selection receipt does not bind policy, pilot scope, and seed plan")
    _validate_comsol_selection(registration, selection)
    level = _selected_candidate_level(recipe, pilot_evaluation, selection)
    return candidate_seeds, level, artifacts, authority_identity


def _validate_final_evaluation_policy(
    artifacts: dict[str, Path], evaluation_policy_sha256: str | None
) -> None:
    if (
        evaluation_policy_sha256 is not None
        and _sha256(artifacts["evaluation_policy"]) != evaluation_policy_sha256
    ):
        raise ValueError("final registration uses another recipe evaluation policy")


def _output_times() -> list[float]:
    return (
        [index * 1.0e-5 for index in range(51)]
        + [index * 1.0e-4 for index in range(6, 51)]
        + [index * 1.0e-3 for index in range(6, 31)]
    )


def _same_schedule(actual: object, expected: list[float]) -> bool:
    values = _sequence(actual, "output times")
    if len(values) != len(expected):
        return False
    return all(
        abs(float(value) - target) <= 8.0 * math.ulp(max(abs(float(value)), abs(target)))
        for value, target in zip(values, expected, strict=True)
    )


def _validate_configured_seed_plan(
    pilot: dict[str, Any], resolved_candidate_seeds: list[int] | None
) -> None:
    if resolved_candidate_seeds is not None:
        if not resolved_candidate_seeds:
            raise ValueError("configured candidate pilot seed set must be nonempty")
        return
    if pilot.get("seed_source") is None:
        _integer_list(pilot.get("candidate_seeds"), "pilot candidate seeds")
        return
    source = _mapping(pilot.get("seed_source"), "pilot seed source")
    if set(source) != {"path", "sha256", "json_pointer"}:
        raise ValueError("pilot seed source has unexpected keys")


def _configured_level_name(value: object, index: int) -> str:
    level = _mapping(value, f"pilot levels[{index}]")
    name = str(level.get("name", "")).strip()
    purpose = str(level.get("purpose", "")).strip()
    if not name:
        raise ValueError("configured candidate pilot level is invalid")
    if not purpose:
        raise ValueError("configured candidate pilot level is invalid")
    dt_s = float(cast(Any, level.get("dt_s", math.nan)))
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("configured candidate pilot level is invalid")
    depth = level.get("brownian_interval_tree_depth")
    if isinstance(depth, bool) or not isinstance(depth, int):
        raise ValueError("configured candidate pilot level is invalid")
    if not 0 <= depth <= 10:
        raise ValueError("configured candidate pilot level is invalid")
    if "geometry_rtol" in level:
        geometry_rtol = float(cast(Any, level["geometry_rtol"]))
        if not math.isfinite(geometry_rtol) or geometry_rtol <= 0.0:
            raise ValueError("configured candidate geometry_rtol must be positive")
    return name


def _validate_configured_execution(recipe: dict[str, Any]) -> None:
    execution = _mapping(recipe.get("execution"), "execution")
    if float(execution.get("time_end_s", math.nan)) != END_TIME_S:
        raise ValueError("configured candidate execution scope differs from this runner")
    if int(execution.get("output_count", -1)) != OUTPUT_COUNT:
        raise ValueError("configured candidate execution scope differs from this runner")


def _validate_configured_recipe(
    recipe: dict[str, Any], resolved_candidate_seeds: list[int] | None
) -> None:
    recipe_id = recipe.get("recipe_id")
    revision = recipe.get("recipe_revision")
    if not isinstance(recipe_id, str) or not recipe_id.strip():
        raise ValueError("configured candidate recipe_id must be a nonempty string")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        raise ValueError("configured candidate recipe_revision must be a positive integer")
    if recipe.get("campaign") is not None:
        _campaign_identity_record(recipe["campaign"], "recipe campaign")
    _configured_authorization_record(recipe)
    pilot = _mapping(recipe.get("pilot"), "pilot")
    _validate_configured_seed_plan(pilot, resolved_candidate_seeds)
    levels = _sequence(pilot.get("levels"), "pilot levels")
    if not levels:
        raise ValueError("configured candidate pilot must contain at least one level")
    names = [_configured_level_name(value, index) for index, value in enumerate(levels)]
    if len(names) != len(set(names)):
        raise ValueError("configured candidate pilot level names must be unique")
    _validate_configured_execution(recipe)


def _validate_recipe(
    recipe: dict[str, Any],
    *,
    explicit_campaign_identity: bool = False,
    resolved_candidate_seeds: list[int] | None = None,
) -> None:
    if recipe.get("schema_version") != 1:
        raise ValueError("unsupported candidate pilot recipe revision")
    if explicit_campaign_identity or recipe.get("campaign") is not None:
        _validate_configured_recipe(recipe, resolved_candidate_seeds)
        return
    revision = int(recipe.get("recipe_revision", -1))
    identities = {
        1: "M3-C2A-caseA-100nm-candidate-pilot",
        2: "M3-C2A-caseA-100nm-event-tolerance-diagnostic",
        3: "M3-C2A-caseA-100nm-event-tolerance-sensitivity",
        4: "M3-C2A-caseA-100nm-event-tolerance-cross-band-sensitivity",
        5: "M3-C2A-caseA-100nm-qualified-macro-pilot",
    }
    if recipe.get("recipe_id") != identities.get(revision):
        raise ValueError("unexpected candidate pilot recipe identity")
    pilot = _mapping(recipe.get("pilot"), "pilot")
    seeds = [int(value) for value in _sequence(pilot.get("candidate_seeds"), "pilot seeds")]
    expected_seeds = {
        1: [918164, 918165, 918166, 918167],
        2: [918164],
        3: [918164],
        4: [918165],
        5: [918164, 918165, 918166, 918167],
    }.get(revision)
    if seeds != expected_seeds:
        raise ValueError("candidate pilot seeds differ from the revision plan")
    levels = _sequence(pilot.get("levels"), "pilot levels")
    names = [str(_mapping(level, "level").get("name")) for level in levels]
    triples = [
        (
            float(_mapping(level, "level")["dt_s"]),
            int(_mapping(level, "level")["brownian_interval_tree_depth"]),
        )
        for level in levels
    ]
    expected = {
        1: (
            ["macro_coarse", "macro_medium", "macro_fine", "path_fine"],
            [(2.0e-5, 3), (1.0e-5, 3), (5.0e-6, 3), (5.0e-6, 4)],
        ),
        2: (["event_tolerance_diagnostic"], [(5.0e-6, 3)]),
        3: (["event_tolerance_sensitivity"], [(5.0e-6, 3)]),
        4: (
            ["event_tolerance_reference", "event_tolerance_candidate"],
            [(5.0e-6, 3), (5.0e-6, 3)],
        ),
        5: (
            ["macro_coarse", "macro_medium", "macro_fine", "path_fine"],
            [(2.0e-5, 3), (1.0e-5, 3), (5.0e-6, 3), (5.0e-6, 4)],
        ),
    }
    if (names, triples) != expected.get(revision):
        raise ValueError("candidate pilot step/depth matrix differs")
    expected_rtols = {
        2: [1e-8],
        3: [1e-9],
        4: [1e-8, 1e-9],
        5: [1e-8, 1e-8, 1e-8, 1e-8],
    }.get(revision)
    if expected_rtols is not None:
        geometry_rtols = [
            float(cast(Any, _mapping(level, "diagnostic level").get("geometry_rtol")))
            for level in levels
        ]
        if geometry_rtols != expected_rtols:
            raise ValueError("event-tolerance diagnostic geometry_rtol differs")


def _validate_v2_evidence(recipe: dict[str, Any], root: Path, superseded: Path) -> list[Path]:
    evidence = [
        _locked_path(root, item, f"contingency evidence {index}")
        for index, item in enumerate(
            _sequence(recipe.get("contingency_evidence"), "contingency evidence")
        )
    ]
    if len(evidence) != 3:
        raise ValueError("numerical contingency must lock the three rejected-level receipts")
    for path in evidence:
        receipt = _load_json(path, "contingency receipt")
        reasons = _mapping(receipt.get("failure_reason_counts"), "failure reasons")
        if receipt.get("status") != "BLOCKED" or int(reasons.get("indeterminate_event", 0)) < 1:
            raise ValueError(
                f"contingency receipt is not rejected numerical-event evidence: {path}"
            )
    return [superseded, *evidence]


def _validate_v3_evidence(recipe: dict[str, Any], root: Path, superseded: Path) -> list[Path]:
    authorization = _locked_path(
        root, recipe.get("execution_authorization"), "execution authorization"
    )
    evidence = [
        _locked_path(root, item, f"baseline diagnostic evidence {index}")
        for index, item in enumerate(
            _sequence(recipe.get("baseline_diagnostic"), "baseline diagnostic")
        )
    ]
    if len(evidence) != 2:
        raise ValueError("tolerance sensitivity must lock baseline receipt and manifest")
    receipt = _load_json(evidence[0], "baseline diagnostic receipt")
    reasons = _mapping(receipt.get("failure_reason_counts"), "failure reasons")
    if receipt.get("status") != "COMPLETE" or any(int(value) for value in reasons.values()):
        raise ValueError("baseline tolerance diagnostic is not failure-free")
    return [superseded, authorization, *evidence]


def _validate_v4_evidence(recipe: dict[str, Any], root: Path, superseded: Path) -> list[Path]:
    authorization = _locked_path(
        root, recipe.get("execution_authorization"), "execution authorization"
    )
    invalid_gate_path = _locked_path(
        root, recipe.get("v3_invalid_gate_evidence"), "v3 invalid-gate evidence"
    )
    evaluator_path = _locked_path(root, recipe.get("evaluator"), "cross-band evaluator")
    invalid_gate = _load_json(invalid_gate_path, "v3 invalid-gate evidence")
    gates = _mapping(invalid_gate.get("gates"), "v3 invalid-gate evidence gates")
    if invalid_gate.get("status") != "FAIL" or gates.get("time_budget") is not False:
        raise ValueError("v3 evidence does not record the superseded time-budget gate failure")
    return [superseded, authorization, invalid_gate_path, evaluator_path]


def _validate_v5_evidence(recipe: dict[str, Any], root: Path) -> list[Path]:
    base_recipe = _locked_path(root, recipe.get("base_recipe"), "base macro recipe")
    decision_path = _locked_path(
        root, recipe.get("tolerance_qualification"), "tolerance qualification"
    )
    decision = _load_json(decision_path, "tolerance qualification")
    if (
        decision.get("status") != "PASS"
        or decision.get("decision")
        != "GEOMETRY_RTOL_1E-8_OPERATIONALLY_QUALIFIED_FOR_TRANSVERSE_TERMINAL_EVENTS_IN_THIS_CASE"
    ):
        raise ValueError("operational event tolerance is not qualified")
    return [base_recipe, decision_path]


def _validate_contingency(recipe: dict[str, Any], root: Path) -> list[Path]:
    revision = int(recipe["recipe_revision"])
    if revision == 5:
        return _validate_v5_evidence(recipe, root)
    if revision not in {2, 3, 4}:
        return []
    superseded = _locked_path(root, recipe.get("supersedes"), "superseded recipe")
    if revision == 2:
        return _validate_v2_evidence(recipe, root, superseded)
    if revision == 3:
        return _validate_v3_evidence(recipe, root, superseded)
    return _validate_v4_evidence(recipe, root, superseded)


def _campaign_evidence_paths(
    recipe: dict[str, Any], root: Path, *, explicit_campaign_identity: bool
) -> list[Path]:
    if explicit_campaign_identity:
        return []
    return _validate_contingency(recipe, root)


def _contract_common_input_receipt(contract: dict[str, Any], input_path: Path) -> dict[str, object]:
    common = _mapping(contract.get("common_p1_input"), "contract common input")
    geometry = _mapping(common.get("geometry"), "contract common input geometry")
    release = _mapping(common.get("release"), "contract common input release")
    scope = _mapping(contract.get("scope"), "contract scope")
    if common.get("file_sha256") != _sha256(input_path):
        raise ValueError("contract common input file identity differs")
    return {
        "boundary_lines": geometry.get("boundary_lines"),
        "content_hash": common.get("content_hash"),
        "file_sha256": common.get("file_sha256"),
        "nodes": geometry.get("nodes"),
        "particle_count": scope.get("particle_count"),
        "release_time_s": release.get("release_time_s"),
        "triangles": geometry.get("triangles"),
    }


def _validate_contract_receipt(
    receipt: dict[str, Any],
    contract: dict[str, Any],
    contract_path: Path,
    input_path: Path,
    campaign_identity: dict[str, str],
    *,
    explicit_campaign_identity: bool,
) -> None:
    if receipt.get("contract_validation_status") != "PASS_INPUT_IDENTITY_AND_SEMANTICS_LOCKED":
        raise ValueError("M3-C2 execution contract has not passed")
    if receipt.get("comsol_or_candidate_executed") is not False:
        raise ValueError("contract receipt is not the pre-execution receipt")
    if receipt.get("contract_sha256") != _sha256(contract_path):
        raise ValueError("contract receipt refers to another M3-C2 contract")
    expected_campaign: object = campaign_identity if explicit_campaign_identity else None
    if receipt.get("campaign") != expected_campaign:
        raise ValueError("contract receipt campaign identity differs")
    expected_input = _contract_common_input_receipt(contract, input_path)
    if receipt.get("common_p1_input") != expected_input:
        raise ValueError("contract receipt common input identity differs")


def _casep_event_tolerance_seed_exception(
    recipe: dict[str, Any],
    contract: dict[str, Any],
    campaign_identity: dict[str, str],
    recipe_seeds: list[int],
) -> bool:
    if recipe.get("recipe_id") != CASEP_EVENT_TOLERANCE_RECIPE_ID:
        return False
    expected_pilot = {
        "candidate_seeds": [919008],
        "levels": [
            {
                "name": "geometry_rtol_reference",
                "dt_s": 2.0e-5,
                "brownian_interval_tree_depth": 3,
                "geometry_rtol": 1.0e-8,
                "purpose": "caseP_geometry_tolerance_reference_on_unseen_seed",
            },
            {
                "name": "geometry_rtol_candidate",
                "dt_s": 2.0e-5,
                "brownian_interval_tree_depth": 3,
                "geometry_rtol": 1.0e-9,
                "purpose": "caseP_geometry_tolerance_sensitivity_on_unseen_seed",
            },
        ],
    }
    if (
        recipe.get("recipe_revision") not in {1, 2}
        or contract.get("contract_id") != CASEP_CONTRACT_ID
        or campaign_identity.get("case_id") != CASEP_CASE_ID
        or campaign_identity.get("evaluation_case_id") != CASEP_EVALUATION_CASE_ID
        or _mapping(recipe.get("pilot"), "pilot") != expected_pilot
        or recipe_seeds != [919008]
    ):
        raise ValueError("Case-P pre-final event-tolerance recipe matrix differs")
    return True


def _validate_contract(
    recipe: dict[str, Any], root: Path
) -> tuple[
    Path,
    Path,
    Path,
    Path,
    dict[str, str],
    bool,
    list[int],
    dict[str, str] | None,
]:
    contract_path = _locked_path(root, recipe.get("contract"), "contract")
    receipt_path = _locked_path(root, recipe.get("contract_receipt"), "contract receipt")
    input_path = _locked_path(root, recipe.get("canonical_input"), "canonical input")
    template_path = _locked_path(
        root, recipe.get("deterministic_case_template"), "deterministic template"
    )
    contract = _load_json(contract_path, "M3-C2 contract")
    receipt = _load_json(receipt_path, "M3-C2 contract receipt")
    campaign_identity, explicit_campaign_identity = _campaign_identity(recipe, contract)
    pilot_authorization = None
    if explicit_campaign_identity:
        _, pilot_authorization = _configured_authorization(recipe, root, campaign_identity)
    pilot = _mapping(contract.get("seed_plan"), "contract seed plan").get("pilot")
    contract_seeds = _integer_list(
        _mapping(pilot, "contract pilot").get("candidate_seeds"), "contract candidate seeds"
    )
    recipe_seeds = _recipe_candidate_seeds(recipe, contract)
    revision = int(recipe["recipe_revision"])
    expected_recipe_seeds = {
        2: [918164],
        3: [918164],
        4: [918165],
        5: contract_seeds,
    }.get(revision)
    casep_event_tolerance = _casep_event_tolerance_seed_exception(
        recipe, contract, campaign_identity, recipe_seeds
    )
    if explicit_campaign_identity or revision == 1:
        seeds_match = contract_seeds == recipe_seeds or casep_event_tolerance
    else:
        seeds_match = recipe_seeds == expected_recipe_seeds
    if not seeds_match:
        raise ValueError("candidate seeds differ from the locked M3-C2 contract")
    if explicit_campaign_identity:
        scope = _mapping(contract.get("scope"), "contract scope")
        if (
            int(scope.get("particle_count", -1)) != PARTICLE_COUNT
            or int(scope.get("output_count", -1)) != OUTPUT_COUNT
            or float(scope.get("time_end_s", math.nan)) != END_TIME_S
        ):
            raise ValueError("configured M3-C2 contract scope differs from this runner")
    _validate_contract_receipt(
        receipt,
        contract,
        contract_path,
        input_path,
        campaign_identity,
        explicit_campaign_identity=explicit_campaign_identity,
    )
    _, input_info = read_with_info(input_path)
    expected_content_hash = _mapping(recipe["canonical_input"], "canonical input")["content_hash"]
    if input_info.content_hash != expected_content_hash:
        raise ValueError("canonical input logical content hash differs")
    return (
        contract_path,
        receipt_path,
        input_path,
        template_path,
        campaign_identity,
        explicit_campaign_identity,
        recipe_seeds,
        pilot_authorization,
    )


def _validate_template(document: dict[str, Any]) -> None:
    physics = _mapping(document.get("physics"), "template physics")
    expected_models = {
        "charge",
        "drag",
        "electric",
        "ion_drag",
        "thermophoresis",
        "dielectrophoresis",
        "lift",
        "gravity_buoyancy",
    }
    if set(physics) != expected_models:
        raise ValueError("deterministic template physics matrix differs")
    boundaries = {
        str(item["boundary_group"]): str(item["law"])
        for value in _sequence(document.get("boundaries"), "template boundaries")
        for item in [_mapping(value, "boundary")]
    }
    expected_boundaries = {
        "wafer": "stick",
        "grounded_wall": "stick",
        "focus_transition": "stick",
        "outer_dielectric": "stick",
        "gas_inlet": "hold",
        "pump_outlet": "escape",
    }
    if boundaries != expected_boundaries:
        raise ValueError("deterministic template boundary mapping differs")
    trajectories = _mapping(_mapping(document["output"], "output")["trajectories"], "frames")
    times = _mapping(trajectories["schedule"], "schedule")["explicit_times_s"]
    if not _same_schedule(times, _output_times()):
        raise ValueError("deterministic template output schedule differs")


def _case_document(
    template: dict[str, Any],
    content_hash: str,
    level: Mapping[str, object],
    seed: int,
    case_name_prefix: str,
) -> dict[str, Any]:
    document = copy.deepcopy(template)
    level_name = str(level["name"])
    document["case"] = {
        "name": f"{case_name_prefix}_{level_name}_seed_{seed}",
        "data_path": "../../../candidate_input.h5",
        "expected_content_hash": content_hash,
    }
    document["time"] = {
        "start_s": 0.0,
        "end_s": END_TIME_S,
        "dt_s": float(cast(Any, level["dt_s"])),
    }
    solver = _mapping(document["solver"], "solver")
    solver["integrator"] = "ou_langevin"
    solver["seed"] = seed
    if "geometry_rtol" in level:
        _mapping(solver["event"], "solver event")["geometry_rtol"] = float(
            cast(Any, level["geometry_rtol"])
        )
    physics = _mapping(document["physics"], "physics")
    physics["noise"] = {
        "model": "inertial_langevin_fdt",
        "revision": "inertial_langevin_fdt_epstein_linear_rz_meridional_projected_v1",
        "interval_tree_depth": int(cast(Any, level["brownian_interval_tree_depth"])),
    }
    document["resources"] = {"memory_limit_mb": 512}
    _mapping(_mapping(document["output"], "output")["trajectories"], "frames")["schedule"] = {
        "explicit_times_s": _output_times()
    }
    return document


def _case_cell(output: Path, level_name: str, seed: int) -> Path:
    return output / "levels" / level_name / f"seed_{seed}"


def prepare(
    recipe_path: Path,
    output: Path,
    registration_path: Path | None = None,
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"candidate campaign output already exists: {output}")
    recipe = _load_json(recipe_path, "candidate pilot recipe")
    root = _repository_root()
    (
        contract_path,
        receipt_path,
        input_path,
        template_path,
        campaign_identity,
        explicit_campaign_identity,
        candidate_seeds,
        pilot_authorization,
    ) = _validate_contract(recipe, root)
    _validate_recipe(
        recipe,
        explicit_campaign_identity=explicit_campaign_identity,
        resolved_candidate_seeds=candidate_seeds,
    )
    evaluation_policy_sha256 = _recipe_evaluation_policy_sha256(
        recipe, root, required=explicit_campaign_identity
    )
    contingency_paths = _campaign_evidence_paths(
        recipe, root, explicit_campaign_identity=explicit_campaign_identity
    )
    purpose: Purpose = "final" if registration_path is not None else "pilot"
    pilot = _mapping(recipe["pilot"], "pilot")
    campaign = {
        "candidate_seeds": candidate_seeds,
        "levels": _sequence(pilot.get("levels"), "pilot levels"),
    }
    authorization_paths: dict[str, Path] = {}
    authorization_identity: dict[str, object] = {}
    if registration_path is not None:
        registration_path = registration_path.resolve()
        seeds, level, authorization_paths, authorization_identity = _validate_final_registration(
            recipe,
            registration_path,
            contract_path,
            campaign_identity,
            evaluation_policy_sha256,
            explicit_campaign_identity=explicit_campaign_identity,
        )
        campaign = {"candidate_seeds": seeds, "levels": [level]}
    template = _mapping(yaml.safe_load(template_path.read_text(encoding="utf-8")), "template")
    _validate_template(template)
    output.mkdir(parents=True)
    shutil.copyfile(recipe_path, output / RECIPE_COPY)
    if registration_path is not None:
        shutil.copyfile(registration_path, output / FINAL_REGISTRATION_COPY)
    copied_input = output / "candidate_input.h5"
    shutil.copyfile(input_path, copied_input)
    if _sha256(copied_input) != _sha256(input_path):
        raise ValueError("copied canonical input differs")
    input_info = read_with_info(copied_input)[1]
    cells: dict[str, dict[str, object]] = {}
    for level_value in _sequence(campaign["levels"], "levels"):
        level = _mapping(level_value, "level")
        level_name = str(level["name"])
        for seed_value in _sequence(campaign["candidate_seeds"], "seeds"):
            seed = int(seed_value)
            cell = _case_cell(output, level_name, seed)
            cell.mkdir(parents=True)
            case_path = cell / "case.yaml"
            case_document = _case_document(
                template,
                input_info.content_hash,
                level,
                seed,
                campaign_identity["candidate_case_name_prefix"],
            )
            case_path.write_text(
                yaml.safe_dump(
                    case_document,
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            case = load_case(case_path)
            cells[f"{level_name}/{seed}"] = {
                "level": level_name,
                "seed": seed,
                "dt_s": level["dt_s"],
                "brownian_interval_tree_depth": level["brownian_interval_tree_depth"],
                "purpose": level["purpose"],
                "geometry_rtol": _mapping(case_document["solver"], "solver")["event"][
                    "geometry_rtol"
                ],
                "case": str(case_path.relative_to(output)).replace("\\", "/"),
                "case_sha256": _sha256(case_path),
                "case_file_hash": case.case_file_hash,
                "status": "PREPARED_NOT_RUN",
            }
    report: dict[str, object] = {
        "status": "PREPARED",
        "tool_revision": TOOL_REVISION,
        "purpose": purpose,
        "recipe": RECIPE_COPY,
        "recipe_sha256": _sha256(output / RECIPE_COPY),
        "final_report": FINAL_REPORT if purpose == "pilot" else FINAL_CAMPAIGN_REPORT,
        "campaign": campaign,
        "campaign_identity": campaign_identity,
        **(
            {"evaluation_policy_sha256": evaluation_policy_sha256}
            if evaluation_policy_sha256 is not None
            else {}
        ),
        **({"pilot_authorization": pilot_authorization} if pilot_authorization is not None else {}),
        "contract_sha256": _sha256(contract_path),
        "contract_receipt_sha256": _sha256(receipt_path),
        "contingency_evidence": [
            {
                "path": str(path.relative_to(root)).replace("\\", "/"),
                "sha256": _sha256(path),
            }
            for path in contingency_paths
        ],
        "final_registration": (
            None
            if registration_path is None
            else {
                "path": FINAL_REGISTRATION_COPY,
                "sha256": _sha256(output / FINAL_REGISTRATION_COPY),
                "authorization_artifacts": {
                    name: {
                        "path": str(path.relative_to(_solver_project_root())).replace("\\", "/"),
                        "sha256": _sha256(path),
                    }
                    for name, path in authorization_paths.items()
                },
                "evaluation_authority": authorization_identity,
            }
        ),
        "input": copied_input.name,
        "input_sha256": _sha256(copied_input),
        "input_content_hash": input_info.content_hash,
        "particle_count": PARTICLE_COUNT,
        "output_count": OUTPUT_COUNT,
        "time_end_s": END_TIME_S,
        "public_api_path": ["load_case", "simulate", "open_result"],
        "cells": cells,
        "claim": (
            "candidate pilot runner locked; numerical accuracy not evaluated"
            if purpose == "pilot"
            else "registered final candidate execution; accuracy not evaluated by runner"
        ),
    }
    _write_json(output / PREPARE_REPORT, report)
    return report


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _replace_json(path: Path, value: object) -> None:
    replacement = path.with_suffix(path.suffix + ".replacement")
    if replacement.exists():
        raise FileExistsError(f"stale JSON replacement exists: {replacement}")
    _write_json(replacement, value)
    replacement.replace(path)


def _number(value: object) -> str:
    return format(float(cast(Any, value)), ".17g")


def _schedule_index(value: float, schedule: list[float]) -> int:
    index = min(range(len(schedule)), key=lambda item: abs(schedule[item] - value))
    scale = max(abs(value), abs(schedule[index]))
    if abs(schedule[index] - value) > 8.0 * math.ulp(scale):
        raise ValueError(f"result frame is outside the locked output schedule: {value:.17g}")
    return index


def _escape_times(result: Any) -> dict[int, float]:
    events = result.read_boundary_events()
    escaped: dict[int, float] = {}
    for row, particle_id_value in enumerate(events.particle_id):
        if str(events.outcome[row]) != "escaped":
            continue
        particle_id = int(particle_id_value)
        if particle_id in escaped:
            raise ValueError(f"particle has more than one escape event: {particle_id}")
        escaped[particle_id] = float(events.time_s[row])
    return escaped


def _require_dense_schedule(
    observed: Mapping[tuple[int, int], tuple[str, ...]],
    escaped: Mapping[int, float],
    schedule: list[float],
) -> None:
    for time_index, time_s in enumerate(schedule):
        for particle_id in range(1, PARTICLE_COUNT + 1):
            if (time_index, particle_id) in observed:
                continue
            escape_time = escaped.get(particle_id)
            if escape_time is None or time_s < escape_time:
                raise ValueError(
                    "result has an interior or non-escape trajectory gap: "
                    f"particle={particle_id}, time={time_s:.17g}"
                )


def _write_trajectory(path: Path, result: Any) -> int:
    schedule = _output_times()
    escaped = _escape_times(result)
    observed: dict[tuple[int, int], tuple[str, ...]] = {}
    for frame in result.iter_frames():
        time_index = _schedule_index(float(frame.time_s), schedule)
        for row, particle_id_value in enumerate(frame.particle_id):
            particle_id = int(particle_id_value)
            if particle_id < 1 or particle_id > PARTICLE_COUNT:
                raise ValueError(f"result frame has an unexpected particle ID: {particle_id}")
            lifecycle = _LIFECYCLE[int(frame.lifecycle[row])]
            if lifecycle == "failed":
                raise ValueError(f"failed particle cannot be normalized: {particle_id}")
            event_time = escaped.get(particle_id)
            if event_time is not None and schedule[time_index] >= event_time:
                raise ValueError(f"result contains a state at or after escape: {particle_id}")
            key = (time_index, particle_id)
            if key in observed:
                raise ValueError(f"duplicate result frame row: {key}")
            observed[key] = (
                _number(frame.position_m[row, 0]),
                _number(frame.position_m[row, 1]),
                _number(frame.velocity_m_s[row, 0]),
                _number(frame.velocity_m_s[row, 1]),
                _number(frame.charge_number[row]),
                lifecycle,
            )
    _require_dense_schedule(observed, escaped, schedule)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(_TRAJECTORY_HEADER)
        for time_index, time_s in enumerate(schedule):
            for particle_id in range(1, PARTICLE_COUNT + 1):
                state = observed.get((time_index, particle_id))
                if state is None:
                    state = ("nan", "nan", "nan", "nan", "nan", "escaped")
                writer.writerow((particle_id, _number(time_s), *state))
    return PARTICLE_COUNT * OUTPUT_COUNT


def _write_observed_trajectory(path: Path, result: Any) -> int:
    """Write only durable observed rows from a failed run for forensic use."""

    rows = 0
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(_TRAJECTORY_HEADER)
        for frame in result.iter_frames():
            for row, particle_id_value in enumerate(frame.particle_id):
                lifecycle = _LIFECYCLE[int(frame.lifecycle[row])]
                writer.writerow(
                    (
                        int(particle_id_value),
                        _number(frame.time_s),
                        _number(frame.position_m[row, 0]),
                        _number(frame.position_m[row, 1]),
                        _number(frame.velocity_m_s[row, 0]),
                        _number(frame.velocity_m_s[row, 1]),
                        _number(frame.charge_number[row]),
                        lifecycle,
                    )
                )
                rows += 1
    return rows


def _write_events(path: Path, result: Any) -> int:
    events = result.read_boundary_events()
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(_EVENT_HEADER)
        for row, particle_id in enumerate(events.particle_id):
            writer.writerow(
                (
                    int(particle_id),
                    int(events.event_ordinal[row]),
                    _number(events.time_s[row]),
                    _number(events.position_m[row, 0]),
                    _number(events.position_m[row, 1]),
                    _number(events.normal[row, 0]),
                    _number(events.normal[row, 1]),
                    _number(events.velocity_pre_m_s[row, 0]),
                    _number(events.velocity_pre_m_s[row, 1]),
                    _number(events.velocity_post_m_s[row, 0]),
                    _number(events.velocity_post_m_s[row, 1]),
                    "material_boundary",
                    int(events.boundary_id[row]),
                    str(events.law_id[row]),
                    str(events.outcome[row]),
                )
            )
    return int(events.particle_id.size)


def _write_failures(path: Path, result: Any) -> int:
    failures = result.read_failure_events()
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(_FAILURE_HEADER)
        for row, particle_id in enumerate(failures.particle_id):
            writer.writerow(
                (
                    int(particle_id),
                    int(failures.event_ordinal[row]),
                    _number(failures.time_s[row]),
                    int(failures.reason_code[row]),
                )
            )
    return int(failures.particle_id.size)


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _peak_rss_bytes() -> int | None:
    if os.name == "nt":

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = cast(Any, ctypes.windll.kernel32)
        psapi = cast(Any, ctypes.windll.psapi)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.c_ulong,
        )
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        if not psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            return None
        return int(counters.PeakWorkingSetSize)
    resource_api = cast(Any, importlib.import_module("resource"))
    scale = 1 if sys.platform == "darwin" else 1024
    return int(resource_api.getrusage(resource_api.RUSAGE_SELF).ru_maxrss) * scale


def _validate_prepared_registration(
    prepared: Path, report: dict[str, Any], purpose: object
) -> None:
    registration = report.get("final_registration")
    if purpose == "final":
        registered = _mapping(registration, "prepared final registration")
        if registered.get("path") != FINAL_REGISTRATION_COPY or _sha256(
            prepared / FINAL_REGISTRATION_COPY
        ) != registered.get("sha256"):
            raise ValueError("prepared final campaign registration differs")
        project_root = _solver_project_root()
        artifacts = _mapping(registered.get("authorization_artifacts"), "authorization artifacts")
        if set(artifacts) != {"evaluation_policy", "pilot_evaluation", "selection_receipt"}:
            raise ValueError("prepared final authorization artifacts are incomplete")
        for name, value in artifacts.items():
            artifact = _mapping(value, f"authorization artifact {name}")
            path = (project_root / str(artifact.get("path", ""))).resolve()
            if (
                not path.is_relative_to(project_root)
                or not path.is_file()
                or _sha256(path) != artifact.get("sha256")
            ):
                raise ValueError("prepared final authorization artifact differs")
        if registered.get("evaluation_authority") is not None:
            authority = _mapping(
                registered["evaluation_authority"], "prepared evaluation authority"
            )
            revision = authority.get("evaluation_policy_revision")
            if (
                set(authority) != {"evaluation_policy_revision", "pilot_evaluator_revision"}
                or revision not in SUPPORTED_EVALUATION_POLICY_REVISIONS
                or authority.get("pilot_evaluator_revision")
                != f"m3c2_stochastic_ensemble_evaluator_v{revision}"
            ):
                raise ValueError("prepared evaluation authority is invalid")
    elif registration is not None:
        raise ValueError("pilot preparation must not contain a final registration")


def _prepared_evaluation_policy_sha256(
    recipe: dict[str, Any],
    report: dict[str, Any],
    repository_root: Path,
    *,
    required: bool,
) -> str | None:
    evaluation_policy_sha256 = _recipe_evaluation_policy_sha256(
        recipe, repository_root, required=required
    )
    declared_policy_sha256 = report.get("evaluation_policy_sha256")
    if declared_policy_sha256 is not None and declared_policy_sha256 != evaluation_policy_sha256:
        raise ValueError("prepared evaluation policy differs from the locked recipe")
    return evaluation_policy_sha256


def _validate_prepared_campaign_authority(prepared: Path, report: dict[str, Any]) -> str | None:
    recipe = _load_json(prepared / RECIPE_COPY, "prepared candidate recipe")
    repository_root = _repository_root()
    contract_path = _locked_path(repository_root, recipe.get("contract"), "prepared contract")
    contract = _load_json(contract_path, "prepared M3-C2 contract")
    campaign_identity, explicit_identity = _campaign_identity(recipe, contract)
    if report.get("tool_revision") == LEGACY_TOOL_REVISION:
        if explicit_identity or report.get("campaign_identity") is not None:
            raise ValueError("legacy preparation cannot carry a configured campaign identity")
        return None
    if report.get("campaign_identity") != campaign_identity:
        raise ValueError("prepared campaign identity differs from the locked contract")
    if report.get("contract_sha256") != _sha256(contract_path):
        raise ValueError("prepared campaign contract identity differs")
    receipt_path = _locked_path(
        repository_root, recipe.get("contract_receipt"), "prepared contract receipt"
    )
    if report.get("contract_receipt_sha256") != _sha256(receipt_path):
        raise ValueError("prepared campaign contract receipt identity differs")
    common_input = _mapping(contract.get("common_p1_input"), "prepared contract common input")
    if report.get("input_sha256") != common_input.get("file_sha256"):
        raise ValueError("prepared campaign input identity differs from the locked contract")
    if report.get("input_content_hash") != common_input.get("content_hash"):
        raise ValueError("prepared campaign input content differs from the locked contract")
    input_info = read_with_info(prepared / "candidate_input.h5")[1]
    if report.get("input_content_hash") != input_info.content_hash:
        raise ValueError("prepared campaign input logical content differs")
    if explicit_identity:
        status, source = _configured_authorization(recipe, repository_root, campaign_identity)
        if status not in {"AUTHORIZED", "AUTHORIZED_BY_EXPLICIT_USER_DIRECTION"}:
            raise ValueError("configured candidate campaign is not authorized for execution")
        if source is None or report.get("pilot_authorization") != source:
            raise ValueError("prepared pilot authorization differs from the locked authority")
    elif report.get("pilot_authorization") is not None:
        raise ValueError("legacy preparation must not carry a pilot authorization")
    return _prepared_evaluation_policy_sha256(
        recipe, report, repository_root, required=explicit_identity
    )


def _load_prepared(prepared: Path) -> dict[str, Any]:
    report_path = prepared / PREPARE_REPORT
    report = _load_json(report_path, "prepare report")
    if report.get("status") != "PREPARED" or report.get("tool_revision") not in {
        TOOL_REVISION,
        LEGACY_TOOL_REVISION,
    }:
        raise ValueError("prepared candidate campaign does not belong to this runner")
    purpose = report.get("purpose")
    expected_report = FINAL_REPORT if purpose == "pilot" else FINAL_CAMPAIGN_REPORT
    if purpose not in {"pilot", "final"} or report.get("final_report") != expected_report:
        raise ValueError("prepared candidate campaign purpose is invalid")
    if report.get("recipe") != RECIPE_COPY or _sha256(prepared / RECIPE_COPY) != report.get(
        "recipe_sha256"
    ):
        raise ValueError("prepared candidate campaign recipe differs")
    if report.get("campaign_identity") is not None:
        _campaign_identity_record(report["campaign_identity"], "prepared campaign identity")
    _validate_prepared_registration(prepared, report, purpose)
    if _sha256(prepared / "candidate_input.h5") != report.get("input_sha256"):
        raise ValueError("prepared candidate campaign input differs")
    evaluation_policy_sha256 = _validate_prepared_campaign_authority(prepared, report)
    if evaluation_policy_sha256 is not None:
        report["evaluation_policy_sha256"] = evaluation_policy_sha256
    return report


def _cell_context(
    prepared: Path, level: str, seed: int
) -> tuple[dict[str, Any], Path, dict[str, Path]]:
    report = _load_prepared(prepared)
    cells = _mapping(report.get("cells"), "prepared cells")
    key = f"{level}/{seed}"
    if key not in cells:
        raise ValueError(f"candidate campaign cell is not locked: {key}")
    planned = dict(_mapping(cells[key], f"cell {key}"))
    planned["_runner_tool_revision"] = report["tool_revision"]
    cell = _case_cell(prepared, level, seed)
    paths = {
        "result": cell / "result",
        "trajectory": cell / "trajectory.csv",
        "events": cell / "events.csv",
        "failures": cell / "failures.csv",
        "performance": cell / "performance.json",
        "receipt": cell / "run_receipt.json",
    }
    return planned, prepared / str(planned["case"]), paths


def _performance_record(
    result_path: Path,
    manifest: Mapping[str, object],
    wall_time_s: float | None,
    peak_rss_bytes: int | None,
) -> dict[str, object]:
    return {
        "wall_time_s": wall_time_s,
        "peak_rss_bytes": peak_rss_bytes,
        "output_bytes": _directory_bytes(result_path),
        "output_bytes_scope": "solver_result_directory",
        "particle_count": PARTICLE_COUNT,
        "output_frames": OUTPUT_COUNT,
        "stage_times_s": None,
        "measurement_status": (
            "MEASURED_PROCESS_LIFETIME_HIGH_WATER"
            if wall_time_s is not None and peak_rss_bytes is not None
            else "NOT_MEASURED_OR_NOT_RECOVERABLE"
        ),
        "peak_rss_scope": "process_lifetime_high_water_after_run",
        "trajectory_normalization_revision": TRAJECTORY_NORMALIZATION_REVISION,
        "counts": manifest.get("counts"),
        "event_refinement": manifest.get("event_refinement"),
        "boundary_interactions": manifest.get("boundary_interactions"),
        "memory_plan": manifest.get("memory_plan"),
    }


def _cell_receipt(
    prepared: Path,
    level: str,
    seed: int,
    planned: Mapping[str, object],
    case_path: Path,
    paths: Mapping[str, Path],
    manifest: Mapping[str, object],
    row_counts: tuple[int, int, int],
    *,
    recovered_after_postprocess_failure: bool,
    trajectory_normalization_revision: str = TRAJECTORY_NORMALIZATION_REVISION,
) -> dict[str, object]:
    trajectory_rows, event_rows, failure_rows = row_counts
    counts = _mapping(manifest.get("counts"), "result counts")
    complete = (
        manifest.get("status") == "complete"
        and int(counts.get("particles", -1)) == PARTICLE_COUNT
        and int(counts.get("frames", -1)) == OUTPUT_COUNT
        and int(counts.get("failure_events", -1)) == 0
        and failure_rows == 0
        and trajectory_rows == PARTICLE_COUNT * OUTPUT_COUNT
        and trajectory_normalization_revision == TRAJECTORY_NORMALIZATION_REVISION
    )
    return {
        "status": "COMPLETE" if complete else "BLOCKED",
        "tool_revision": planned.get("_runner_tool_revision", TOOL_REVISION),
        "participant": "candidate",
        "level": level,
        "seed": seed,
        "dt_s": planned["dt_s"],
        "brownian_interval_tree_depth": planned["brownian_interval_tree_depth"],
        "geometry_rtol": planned.get("geometry_rtol", 1.0e-12),
        "case": str(case_path.relative_to(prepared)).replace("\\", "/"),
        "case_sha256": _sha256(case_path),
        "result": str(paths["result"].relative_to(prepared)).replace("\\", "/"),
        "result_manifest_sha256": _sha256(paths["result"] / "run.json"),
        "trajectory": str(paths["trajectory"].relative_to(prepared)).replace("\\", "/"),
        "trajectory_sha256": _sha256(paths["trajectory"]),
        "trajectory_rows": trajectory_rows,
        "trajectory_normalization_revision": trajectory_normalization_revision,
        "events": str(paths["events"].relative_to(prepared)).replace("\\", "/"),
        "events_sha256": _sha256(paths["events"]),
        "event_rows": event_rows,
        "failures": str(paths["failures"].relative_to(prepared)).replace("\\", "/"),
        "failures_sha256": _sha256(paths["failures"]),
        "failure_rows": failure_rows,
        "performance": str(paths["performance"].relative_to(prepared)).replace("\\", "/"),
        "performance_sha256": _sha256(paths["performance"]),
        "engine_algorithm_revision": manifest.get("engine_algorithm_revision"),
        "brownian_rng_revision": manifest.get("brownian_rng_revision"),
        "joint_ou_revision": manifest.get("joint_ou_revision"),
        "resolved_physics_models": _mapping(manifest.get("resolved"), "resolved").get(
            "physics_models"
        ),
        "result_counts": counts,
        "failure_reason_counts": manifest.get("failure_reason_counts"),
        "recovered_after_postprocess_failure": recovered_after_postprocess_failure,
    }


def run_cell(prepared: Path, level: str, seed: int) -> dict[str, object]:
    planned, case_path, paths = _cell_context(prepared, level, seed)
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"candidate campaign cell output already exists: {existing}")
    if _sha256(case_path) != planned["case_sha256"]:
        raise ValueError("candidate campaign case differs from the prepare receipt")
    case = load_case(case_path)
    start = time.perf_counter()
    simulate(case, paths["result"])
    elapsed_s = time.perf_counter() - start
    peak_rss_bytes = _peak_rss_bytes()
    result = open_result(paths["result"])
    manifest = _mapping(result.manifest, "result manifest")
    counts = _mapping(manifest.get("counts"), "result counts")
    projection_paths = dict(paths)
    normalization_revision = TRAJECTORY_NORMALIZATION_REVISION
    if int(counts.get("failure_events", -1)) > 0:
        projection_paths["trajectory"] = paths["trajectory"].with_name("trajectory.blocked.csv")
        normalization_revision = BLOCKED_TRAJECTORY_REVISION
        trajectory_rows = _write_observed_trajectory(projection_paths["trajectory"], result)
    else:
        trajectory_rows = _write_trajectory(paths["trajectory"], result)
    event_rows = _write_events(paths["events"], result)
    failure_rows = _write_failures(paths["failures"], result)
    _write_json(
        paths["performance"],
        _performance_record(paths["result"], manifest, elapsed_s, peak_rss_bytes),
    )
    receipt = _cell_receipt(
        prepared,
        level,
        seed,
        planned,
        case_path,
        projection_paths,
        manifest,
        (trajectory_rows, event_rows, failure_rows),
        recovered_after_postprocess_failure=False,
        trajectory_normalization_revision=normalization_revision,
    )
    _write_json(paths["receipt"], receipt)
    return receipt


def _csv_rows(path: Path, expected_header: tuple[str, ...]) -> int:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        if next(reader, None) != list(expected_header):
            raise ValueError(f"projection header differs: {path}")
        return sum(1 for row in reader if row)


def recover_cell(prepared: Path, level: str, seed: int) -> dict[str, object]:
    """Finish receipts after a completed run whose Python postprocess stopped."""

    planned, case_path, paths = _cell_context(prepared, level, seed)
    if not paths["result"].is_dir():
        raise FileNotFoundError(f"candidate cell result is missing: {paths['result']}")
    if paths["performance"].exists() or paths["receipt"].exists():
        raise FileExistsError("candidate cell receipt or performance record already exists")
    result = open_result(paths["result"])
    manifest = _mapping(result.manifest, "result manifest")
    counts = _mapping(manifest.get("counts"), "result counts")
    if int(counts.get("failure_events", -1)) > 0:
        blocked_trajectory = paths["trajectory"].with_name("trajectory.blocked.csv")
        conflicting = [
            str(path)
            for path in (blocked_trajectory, paths["events"], paths["failures"])
            if path.exists()
        ]
        if conflicting:
            raise FileExistsError(f"blocked recovery artifact already exists: {conflicting}")
        trajectory_rows = _write_observed_trajectory(blocked_trajectory, result)
        event_rows = _write_events(paths["events"], result)
        failure_rows = _write_failures(paths["failures"], result)
        projection_paths = dict(paths)
        projection_paths["trajectory"] = blocked_trajectory
        _write_json(
            paths["performance"],
            _performance_record(paths["result"], manifest, None, None),
        )
        receipt = _cell_receipt(
            prepared,
            level,
            seed,
            planned,
            case_path,
            projection_paths,
            manifest,
            (trajectory_rows, event_rows, failure_rows),
            recovered_after_postprocess_failure=True,
            trajectory_normalization_revision=BLOCKED_TRAJECTORY_REVISION,
        )
        _write_json(paths["receipt"], receipt)
        return receipt
    required = ("trajectory", "events", "failures")
    missing = [str(paths[name]) for name in required if not paths[name].exists()]
    if missing:
        raise FileNotFoundError(f"completed candidate cell artifact is missing: {missing}")
    row_counts = (
        _csv_rows(paths["trajectory"], _TRAJECTORY_HEADER),
        _csv_rows(paths["events"], _EVENT_HEADER),
        _csv_rows(paths["failures"], _FAILURE_HEADER),
    )
    _write_json(paths["performance"], _performance_record(paths["result"], manifest, None, None))
    receipt = _cell_receipt(
        prepared,
        level,
        seed,
        planned,
        case_path,
        paths,
        manifest,
        row_counts,
        recovered_after_postprocess_failure=True,
    )
    _write_json(paths["receipt"], receipt)
    return receipt


def renormalize_cell(prepared: Path, level: str, seed: int) -> dict[str, object]:
    """Supersede a sparse derived projection while preserving its prior hashes."""

    planned, case_path, paths = _cell_context(prepared, level, seed)
    required = ("result", "trajectory", "events", "failures", "performance", "receipt")
    missing = [str(paths[name]) for name in required if not paths[name].exists()]
    if missing:
        raise FileNotFoundError(f"candidate cell artifact is missing: {missing}")
    old_receipt = _load_json(paths["receipt"], "old cell receipt")
    if old_receipt.get("status") != "COMPLETE":
        raise ValueError("only a completed candidate cell may be renormalized")
    if old_receipt.get("trajectory_normalization_revision") is not None:
        raise ValueError("candidate cell already uses the dense trajectory normalization")
    old_performance = _load_json(paths["performance"], "old performance")
    elapsed = old_performance.get("elapsed_public_api_s")
    wall_time_s = None if elapsed is None else float(elapsed)
    result = open_result(paths["result"])
    manifest = _mapping(result.manifest, "result manifest")
    replacement = paths["trajectory"].with_suffix(".csv.replacement")
    if replacement.exists():
        raise FileExistsError(f"stale trajectory replacement exists: {replacement}")
    trajectory_rows = _write_trajectory(replacement, result)
    event_rows = _csv_rows(paths["events"], _EVENT_HEADER)
    failure_rows = _csv_rows(paths["failures"], _FAILURE_HEADER)
    performance = _performance_record(paths["result"], manifest, wall_time_s, None)
    temporary_paths = dict(paths)
    temporary_paths["trajectory"] = replacement
    receipt = _cell_receipt(
        prepared,
        level,
        seed,
        planned,
        case_path,
        temporary_paths,
        manifest,
        (trajectory_rows, event_rows, failure_rows),
        recovered_after_postprocess_failure=bool(
            old_receipt.get("recovered_after_postprocess_failure")
        ),
    )
    receipt["trajectory"] = str(paths["trajectory"].relative_to(prepared)).replace("\\", "/")
    receipt["performance"] = str(paths["performance"].relative_to(prepared)).replace("\\", "/")
    supersession = {
        "status": "COMPLETE",
        "revision": TRAJECTORY_NORMALIZATION_REVISION,
        "durable_result_unchanged": True,
        "result_manifest_sha256": _sha256(paths["result"] / "run.json"),
        "superseded": {
            "trajectory_sha256": _sha256(paths["trajectory"]),
            "trajectory_rows": old_receipt.get("trajectory_rows"),
            "performance_sha256": _sha256(paths["performance"]),
            "receipt_sha256": _sha256(paths["receipt"]),
        },
        "replacement": {
            "trajectory_sha256": _sha256(replacement),
            "trajectory_rows": trajectory_rows,
        },
    }
    replacement.replace(paths["trajectory"])
    _replace_json(paths["performance"], performance)
    receipt["performance_sha256"] = _sha256(paths["performance"])
    _replace_json(paths["receipt"], receipt)
    supersession["replacement"]["performance_sha256"] = _sha256(paths["performance"])
    supersession["replacement"]["receipt_sha256"] = _sha256(paths["receipt"])
    _write_json(paths["receipt"].with_name("normalization_supersession.json"), supersession)
    return receipt


def classify_level_performance_overlap(
    prepared: Path, level: str, reason: str
) -> dict[str, object]:
    """Mark one completed level's measured timing as externally contaminated."""
    prepared_report = _load_prepared(prepared)
    final_report = prepared / str(prepared_report["final_report"])
    if final_report.exists():
        raise FileExistsError("performance classification must precede finalization")
    campaign = _mapping(prepared_report["campaign"], "prepared campaign")
    level_names = {
        str(_mapping(item, "level")["name"]) for item in _sequence(campaign["levels"], "levels")
    }
    if level not in level_names:
        raise ValueError(f"level is absent from the prepared recipe: {level}")
    cells: list[dict[str, object]] = []
    for seed_value in _sequence(campaign["candidate_seeds"], "seeds"):
        seed = int(seed_value)
        cell = _case_cell(prepared, level, seed)
        performance_path = cell / "performance.json"
        receipt_path = cell / "run_receipt.json"
        supersession_path = cell / "performance_classification_supersession.json"
        if supersession_path.exists():
            raise FileExistsError(f"performance classification already exists: {supersession_path}")
        performance = _load_json(performance_path, "cell performance")
        receipt = _load_json(receipt_path, "cell receipt")
        old_performance_hash = _sha256(performance_path)
        old_receipt_hash = _sha256(receipt_path)
        if (
            receipt.get("status") != "COMPLETE"
            or receipt.get("performance_sha256") != old_performance_hash
        ):
            raise ValueError(f"completed cell performance identity differs: {level}/{seed}")
        performance["measurement_status"] = "NON_AUTHORITATIVE_EXTERNAL_WORKLOAD_OVERLAP"
        performance["non_authoritative_reason"] = reason
        _replace_json(performance_path, performance)
        receipt["performance_sha256"] = _sha256(performance_path)
        _replace_json(receipt_path, receipt)
        supersession = {
            "status": "COMPLETE",
            "classification": "performance_only_supersession_science_payload_unchanged",
            "reason": reason,
            "science_payload_unchanged": True,
            "performance_classification": "NON_AUTHORITATIVE_EXTERNAL_WORKLOAD_OVERLAP",
            "superseded": {
                "performance_sha256": old_performance_hash,
                "run_receipt_sha256": old_receipt_hash,
            },
            "replacement": {
                "performance_sha256": _sha256(performance_path),
                "run_receipt_sha256": _sha256(receipt_path),
            },
        }
        _write_json(supersession_path, supersession)
        cells.append(
            {
                "seed": seed,
                "supersession": str(supersession_path.relative_to(prepared)).replace("\\", "/"),
                "supersession_sha256": _sha256(supersession_path),
            }
        )
    report: dict[str, object] = {
        "status": "COMPLETE",
        "level": level,
        "classification": "NON_AUTHORITATIVE_EXTERNAL_WORKLOAD_OVERLAP",
        "science_payload_unchanged": True,
        "reason": reason,
        "cells": cells,
    }
    _write_json(
        prepared / "levels" / level / "performance_classification_supersession.json", report
    )
    return report


def _validate_receipt_artifact(
    receipt: dict[str, Any],
    prepared: Path,
    path_field: str,
    sha256_field: str,
    artifact_path: Path,
    *,
    hashed_path: Path | None = None,
) -> None:
    expected_path = str(artifact_path.relative_to(prepared)).replace("\\", "/")
    digest_path = artifact_path if hashed_path is None else hashed_path
    if (
        receipt.get(path_field) != expected_path
        or not digest_path.is_file()
        or receipt.get(sha256_field) != _sha256(digest_path)
    ):
        raise ValueError(f"candidate cell receipt {path_field} identity differs")


def _validate_receipt_plan(
    receipt: dict[str, Any], report: dict[str, Any], planned: dict[str, Any], level: str, seed: int
) -> None:
    expected = {
        "status": "COMPLETE",
        "tool_revision": report["tool_revision"],
        "participant": "candidate",
        "level": level,
        "seed": seed,
        "dt_s": planned["dt_s"],
        "brownian_interval_tree_depth": planned["brownian_interval_tree_depth"],
        "geometry_rtol": planned.get("geometry_rtol", 1.0e-12),
        "case": planned["case"],
        "case_sha256": planned["case_sha256"],
    }
    differing = [field for field, value in expected.items() if receipt.get(field) != value]
    if differing:
        raise ValueError(
            f"candidate cell receipt differs from the planned cell: {', '.join(differing)}"
        )


def _validated_cell_receipt(
    prepared: Path,
    report: dict[str, Any],
    cells: dict[str, Any],
    level: str,
    seed: int,
) -> tuple[dict[str, Any], Path]:
    key = f"{level}/{seed}"
    planned = _mapping(cells.get(key), f"planned cell {key}")
    receipt_path = _case_cell(prepared, level, seed) / "run_receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(f"candidate campaign receipt is missing: {receipt_path}")
    receipt = _load_json(receipt_path, "cell receipt")
    _validate_receipt_plan(receipt, report, planned, level, seed)
    case_path = prepared / str(planned["case"])
    if not case_path.is_file() or _sha256(case_path) != planned["case_sha256"]:
        raise ValueError("candidate cell case identity differs from the planned cell")
    cell = receipt_path.parent
    _validate_receipt_artifact(
        receipt,
        prepared,
        "result",
        "result_manifest_sha256",
        cell / "result",
        hashed_path=cell / "result" / "run.json",
    )
    for name in ("trajectory", "events", "failures", "performance"):
        _validate_receipt_artifact(
            receipt,
            prepared,
            name,
            f"{name}_sha256",
            cell / (f"{name}.json" if name == "performance" else f"{name}.csv"),
        )
    return receipt, receipt_path


def finalize(prepared: Path) -> dict[str, object]:
    report = _load_prepared(prepared)
    manifest_path = prepared / str(report["final_report"])
    if manifest_path.exists():
        raise FileExistsError(f"candidate campaign manifest already exists: {manifest_path}")
    campaign = _mapping(report["campaign"], "prepared campaign")
    cells = _mapping(report.get("cells"), "prepared cells")
    purpose = cast(Purpose, report["purpose"])
    levels: dict[str, dict[str, object]] = {}
    for level_value in _sequence(campaign["levels"], "levels"):
        level = _mapping(level_value, "level")
        name = str(level["name"])
        replicas: list[dict[str, object]] = []
        for seed_value in _sequence(campaign["candidate_seeds"], "seeds"):
            seed = int(seed_value)
            receipt, receipt_path = _validated_cell_receipt(prepared, report, cells, name, seed)
            replicas.append(
                {
                    **receipt,
                    "receipt": str(receipt_path.relative_to(prepared)).replace("\\", "/"),
                    "receipt_sha256": _sha256(receipt_path),
                }
            )
        levels[name] = {
            "dt_s": level["dt_s"],
            "brownian_interval_tree_depth": level["brownian_interval_tree_depth"],
            "geometry_rtol": level.get("geometry_rtol", 1.0e-12),
            "purpose": level["purpose"],
            "replicas": replicas,
        }
    final: dict[str, object] = {
        "status": "COMPLETE",
        "tool_revision": report["tool_revision"],
        "participant": "candidate",
        "purpose": purpose,
        **(
            {"campaign_identity": report["campaign_identity"]}
            if report.get("campaign_identity") is not None
            else {}
        ),
        **(
            {"evaluation_policy_sha256": report["evaluation_policy_sha256"]}
            if report.get("evaluation_policy_sha256") is not None
            else {}
        ),
        **(
            {"pilot_authorization": report["pilot_authorization"]}
            if purpose == "pilot" and report.get("pilot_authorization") is not None
            else {}
        ),
        **(
            {
                "campaign_binding": {
                    "contract_sha256": report["contract_sha256"],
                    "input_sha256": report["input_sha256"],
                    "input_content_hash": report["input_content_hash"],
                }
            }
            if report.get("tool_revision") == TOOL_REVISION
            else {}
        ),
        "prepare_report": PREPARE_REPORT,
        "prepare_report_sha256": _sha256(prepared / PREPARE_REPORT),
        "input_sha256": report["input_sha256"],
        "input_content_hash": report["input_content_hash"],
        "particle_count": PARTICLE_COUNT,
        "output_count": OUTPUT_COUNT,
        "time_end_s": END_TIME_S,
        "levels": levels,
        "comparison_status": f"READY_FOR_INDEPENDENT_ENSEMBLE_{purpose.upper()}_EVALUATION",
        "accuracy_claim": "NOT_EVALUATED_BY_RUNNER",
        **({"final_registration": report["final_registration"]} if purpose == "final" else {}),
    }
    _write_json(manifest_path, final)
    return final


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("recipe", type=Path)
    prepare_parser.add_argument("output", type=Path)
    prepare_parser.add_argument(
        "--registration",
        type=Path,
        help="post-pilot final registration; omit it to prepare the historical pilot",
    )
    run_parser = subparsers.add_parser("run-cell")
    run_parser.add_argument("prepared", type=Path)
    run_parser.add_argument("level")
    run_parser.add_argument("seed", type=int)
    recovery_parser = subparsers.add_parser("recover-cell")
    recovery_parser.add_argument("prepared", type=Path)
    recovery_parser.add_argument("level")
    recovery_parser.add_argument("seed", type=int)
    normalization_parser = subparsers.add_parser("renormalize-cell")
    normalization_parser.add_argument("prepared", type=Path)
    normalization_parser.add_argument("level")
    normalization_parser.add_argument("seed", type=int)
    overlap_parser = subparsers.add_parser("classify-level-performance-overlap")
    overlap_parser.add_argument("prepared", type=Path)
    overlap_parser.add_argument("level")
    overlap_parser.add_argument("reason")
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("prepared", type=Path)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    if arguments.command == "prepare":
        result = prepare(
            arguments.recipe.resolve(),
            arguments.output.resolve(),
            None if arguments.registration is None else arguments.registration.resolve(),
        )
    elif arguments.command == "run-cell":
        result = run_cell(arguments.prepared.resolve(), arguments.level, arguments.seed)
    elif arguments.command == "recover-cell":
        result = recover_cell(arguments.prepared.resolve(), arguments.level, arguments.seed)
    elif arguments.command == "renormalize-cell":
        result = renormalize_cell(arguments.prepared.resolve(), arguments.level, arguments.seed)
    elif arguments.command == "classify-level-performance-overlap":
        result = classify_level_performance_overlap(
            arguments.prepared.resolve(), arguments.level, arguments.reason
        )
    else:
        result = finalize(arguments.prepared.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") in {"PREPARED", "COMPLETE"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
