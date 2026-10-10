"""Prepare and normalize no-save M3-C2A COMSOL stochastic trajectories.

This external V&V adapter converts COMSOL's one-wide-row-per-particle export
into the participant/level/replica schema consumed by the stochastic
evaluator.  It also turns either the locked pilot contract or a separately
registered final campaign into the only request file read by the Java runner.
It validates the output schedule, study isolation, and realized initial state,
but does not decide solver accuracy or tune an acceptance threshold.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Final, cast

from tools.vv.comsol.actual_run_receipt import (
    inventory_artifact,
    materialize_actual_run_receipts,
    normalize_terminal_event,
    read_actual_run_receipt,
)

TOOL_REVISION: Final = "m3c2_comsol_campaign_normalizer_v5"
EXECUTION_REQUEST_REVISION: Final = "m3c2_comsol_execution_request_v3"
LEGACY_CONTRACT_ID: Final = "M3-C2A-caseA-100nm-stochastic-pilot"
PILOT_AUTHORIZATION_KIND: Final = "m3c2_pilot_execution_authorization"
PILOT_AUTHORIZATION_STATUS: Final = "AUTHORIZED_BY_EXPLICIT_USER_DIRECTION"
PILOT_AUTHORIZATION_PARTICIPANTS: Final = ["comsol", "candidate"]
FINAL_AUTHORIZATION_KIND: Final = "m3c2_post_pilot_final_authorization"
FINAL_AUTHORIZATION_STATUS: Final = "AUTHORIZED_FOR_CONFIRMATORY_FINAL"
LEGACY_CAMPAIGN: Final = {
    "case_id": "formal_iondrag_theory_consistent/caseA_100nm",
    "evaluation_case_id": "M3-C2A_caseA_100nm_common-P1",
    "output_slug": "caseA_100nm",
    "final_registration_kind": "m3c2_caseA_100nm_final_campaign",
    "candidate_case_name_prefix": "m3c2_caseA_100nm",
}
LEGACY_COMSOL: Final = {
    "physics_tag": "fptas",
    "background_study": "stdASf",
    "background_study_step": "stat",
    "background_solution": "sol26",
    "shared_variable_tag": "varAS",
    "viscosity_expression": "root.comp1.AS_muB",
    "temperature_expression": "root.comp1.AS_Tg",
    "pressure_expression": ("m3c1_rhog(r,z)*k_B_const*m3c1_Tg(r,z)/1.2753471408396638e-25[kg]"),
    "seed_parameter": "AS_brownian_seed",
    "sole_seed_authority": "fptas.bf1.i",
    "position_expressions": ["q3r", "q3z"],
    "velocity_expressions": ["fptas.vr", "fptas.vz"],
    "charge_state_expression": "ZAS",
    "particle_geometry": "pgeom_fptas",
}
REQUEST_FILE: Final = "m3c2_pilot_request.csv"
EXECUTION_REQUEST_FILE: Final = "m3c2_execution_request.json"
PILOT_MANIFEST_FILE: Final = "comsol_pilot_manifest.json"
FINAL_MANIFEST_FILE: Final = "comsol_campaign_manifest.json"
PILOT_STEPS_NS: Final = (20000, 10000, 5000)
FINAL_REPLICAS: Final = 32
EXPECTED_PARTICLES: Final = 287
EXPECTED_FRAMES: Final = 121
STATE_COLUMNS: Final = (
    "particle_id",
    "time_s",
    "r_m",
    "z_m",
    "velocity_r_m_per_s",
    "velocity_z_m_per_s",
    "charge_number_e",
    "current_status_code",
    "final_status_code",
    "stop_or_event_time_s",
)
TRAJECTORY_COLUMNS: Final = (
    "particle_id",
    "time_s",
    "r_m",
    "z_m",
    "velocity_r_m_per_s",
    "velocity_z_m_per_s",
    "charge_number_e",
    "lifecycle",
)
EVENT_COLUMNS: Final = (
    "particle_id",
    "event_time_s",
    "event_type",
    "outcome",
    "boundary_semantic",
)
STATUS: Final = {1: "active", 2: "held", 3: "stuck", 4: "escaped"}
CONFIGURATION_PREFIX: Final = "M3C2_COMSOL|configuration|"
SOLVE_PREFIX: Final = "M3C2_COMSOL|solve_pass|"
TIME_END_S: Final = 0.03
TIME_TOLERANCE_S: Final = 3.0e-13
TIME_END_NS: Final = 30_000_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return cast(dict[str, Any], value)


def _campaign_identity(contract: dict[str, Any]) -> dict[str, str]:
    raw = contract.get("campaign")
    if raw is None:
        if contract.get("contract_id") != LEGACY_CONTRACT_ID:
            raise ValueError("new M3-C2 contracts must define campaign identity")
        return dict(LEGACY_CAMPAIGN)
    campaign = _mapping(raw, "contract.campaign")
    names = tuple(LEGACY_CAMPAIGN)
    if set(campaign) != set(names) or any(
        not isinstance(campaign.get(name), str)
        or not campaign[name]
        or campaign[name] != campaign[name].strip()
        for name in names
    ):
        raise ValueError("contract.campaign must contain the five campaign identity fields")
    return {name: cast(str, campaign[name]) for name in names}


def _campaign_binding(contract: dict[str, Any], contract_path: Path) -> dict[str, str]:
    common = _mapping(contract.get("common_p1_input"), "contract.common_p1_input")
    binding = {
        "contract_sha256": _sha256(contract_path),
        "input_sha256": str(common.get("file_sha256", "")).lower(),
        "input_content_hash": str(common.get("content_hash", "")),
    }
    if len(binding["input_sha256"]) != 64 or not binding["input_content_hash"].startswith(
        "sha256:"
    ):
        raise ValueError("contract common-P1 identity is incomplete")
    return binding


def _comsol_semantics(contract: dict[str, Any]) -> dict[str, Any]:
    stochastic = _mapping(contract.get("stochastic_physics"), "contract.stochastic_physics")
    raw = stochastic.get("comsol")
    if not isinstance(raw, dict):
        raise ValueError("contract.stochastic_physics.comsol must be a mapping")
    source = cast(dict[str, Any], raw)
    if contract.get("contract_id") == LEGACY_CONTRACT_ID:
        merged: dict[str, Any] = {**LEGACY_COMSOL, **source}
    else:
        merged = dict(source)
    required = tuple(LEGACY_COMSOL)
    if any(name not in merged for name in required):
        raise ValueError("contract COMSOL semantics are incomplete")
    if merged.get("random_number_args") != "UserDefined":
        raise ValueError("contract COMSOL random-number authority is not UserDefined")
    physics = str(merged["physics_tag"])
    if merged.get("sole_seed_authority") != f"{physics}.bf1.i":
        raise ValueError("contract COMSOL seed authority is inconsistent")
    positions = merged["position_expressions"]
    velocities = merged["velocity_expressions"]
    if not isinstance(positions, list) or len(positions) != 2:
        raise ValueError("contract COMSOL position expressions must contain two entries")
    if not isinstance(velocities, list) or len(velocities) != 2:
        raise ValueError("contract COMSOL velocity expressions must contain two entries")
    return merged


def _pilot_steps_ns(contract: dict[str, Any]) -> tuple[int, ...]:
    numerical = _mapping(contract.get("numerical_policy"), "contract.numerical_policy")
    values = numerical.get("comsol_pilot_fixed_steps_s")
    if values is None and contract.get("contract_id") == LEGACY_CONTRACT_ID:
        return PILOT_STEPS_NS
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError("contract COMSOL pilot must register exactly three fixed steps")
    result = tuple(_step_ns(value, "COMSOL pilot fixed step") for value in values)
    if tuple(sorted(result, reverse=True)) != result or len(set(result)) != 3:
        raise ValueError("contract COMSOL pilot steps must be unique and descending")
    return result


def _integer_list(value: object, name: str) -> list[int]:
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise ValueError(f"{name} must be an integer list")
    result = cast(list[int], value)
    if any(seed < 0 or seed > 2_147_483_647 for seed in result):
        raise ValueError(f"{name} contains a seed outside the COMSOL Java integer range")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must contain unique seeds")
    return result


def _step_ns(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number")
    seconds = float(value)
    nanoseconds = round(seconds * 1.0e9)
    if (
        not math.isfinite(seconds)
        or seconds <= 0.0
        or nanoseconds <= 0
        or not math.isclose(seconds, nanoseconds * 1.0e-9, rel_tol=0.0, abs_tol=1.0e-18)
    ):
        raise ValueError(f"{name} must be exactly representable as integer nanoseconds")
    if TIME_END_NS % nanoseconds != 0:
        raise ValueError(f"{name} must divide the 30 ms campaign horizon")
    return nanoseconds


def _step_label(step_ns: int) -> str:
    if step_ns % 1000 == 0:
        return f"dt_{step_ns // 1000}us"
    return f"dt_{step_ns}ns"


def _request_directory(seed: int, step_ns: int) -> str:
    return f"levels/{_step_label(step_ns)}/replicas/seed_{seed}"


def _json(path: Path, name: str) -> dict[str, Any]:
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8-sig")), name)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {name}: {path}") from error


def _authorization_artifact(
    value: object,
    name: str,
    registration_path: Path,
    project_root: Path,
) -> dict[str, str]:
    artifact = _mapping(value, name)
    relative = Path(str(artifact.get("path", "")))
    expected = str(artifact.get("sha256", "")).lower()
    if relative.is_absolute() or str(relative) in {"", "."}:
        raise ValueError(f"{name}.path must be relative to the registration directory")
    resolved = (registration_path.parent / relative).resolve()
    if not resolved.is_relative_to(project_root) or not resolved.is_file():
        raise ValueError(f"{name} is missing or outside the solver project")
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError(f"{name}.sha256 is invalid")
    if _sha256(resolved) != expected:
        raise ValueError(f"{name} differs from its registered SHA-256")
    return {"path": relative.as_posix(), "sha256": expected}


def _selected_comsol_final_step_ns(
    registration: dict[str, Any],
    registration_path: Path,
    authorization_artifacts: dict[str, dict[str, str]],
) -> int:
    registered = _mapping(
        registration.get("comsol_numerical_setting"),
        "registration.comsol_numerical_setting",
    )
    if (
        set(registered) != {"integrator", "fixed_step_s"}
        or registered.get("integrator") != "classical_rk4"
    ):
        raise ValueError("final COMSOL numerical setting is invalid")
    selection_record = authorization_artifacts["selection_receipt"]
    selection_path = (registration_path.parent / selection_record["path"]).resolve()
    selection = _json(selection_path, "M3-C2 final selection receipt")
    if (
        selection.get("schema_version"),
        selection.get("receipt_kind"),
        selection.get("status"),
    ) != (1, FINAL_AUTHORIZATION_KIND, FINAL_AUTHORIZATION_STATUS):
        raise ValueError("final selection receipt is not an accepted authorization")
    if (
        selection.get("policy_sha256") != authorization_artifacts["evaluation_policy"]["sha256"]
        or selection.get("pilot_report_sha256")
        != authorization_artifacts["pilot_evaluation"]["sha256"]
    ):
        raise ValueError("final selection receipt uses different authorization artifacts")
    selected_levels = _mapping(selection.get("selected_final_levels"), "selected final levels")
    if set(selected_levels) != {"comsol", "candidate"}:
        raise ValueError("selection receipt must select exactly COMSOL and candidate levels")
    selected = _mapping(selected_levels["comsol"], "selected COMSOL final level")
    numerical = _mapping(selected.get("numerical_setting"), "selected COMSOL numerical setting")
    expected = {**registered, "purpose": "accepted_final"}
    step_ns = _step_ns(numerical.get("fixed_step_s"), "selected COMSOL fixed_step_s")
    if (
        set(selected) != {"level_id", "ordinal", "numerical_setting"}
        or int(selected.get("ordinal", -1)) != 0
        or selected.get("level_id") != _step_label(step_ns)
        or numerical != expected
    ):
        raise ValueError("registered COMSOL final setting differs from the pilot selection")
    return step_ns


def _solver_project_root(contract_path: Path) -> Path:
    for parent in contract_path.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "uv.lock").is_file():
            return parent.resolve()
    raise ValueError("cannot locate the solver project root from the pilot contract")


def _repository_root(contract_path: Path) -> Path:
    solver_root = _solver_project_root(contract_path)
    if solver_root.parent.name == "particle_platform_redesign":
        return solver_root.parent.parent.resolve()
    return solver_root.parent.resolve()


def _repository_artifact(
    value: object, name: str, repository_root: Path
) -> tuple[Path, dict[str, str]]:
    record = _mapping(value, name)
    if set(record) != {"path", "sha256"}:
        raise ValueError(f"{name} must contain exactly path and sha256")
    relative = Path(str(record.get("path", "")))
    expected = str(record.get("sha256", "")).lower()
    if relative.is_absolute() or str(relative) in {"", "."}:
        raise ValueError(f"{name}.path must be repository-relative")
    resolved = (repository_root / relative).resolve()
    if not resolved.is_relative_to(repository_root) or not resolved.is_file():
        raise ValueError(f"{name} is missing or outside the repository")
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError(f"{name}.sha256 is invalid")
    if _sha256(resolved) != expected:
        raise ValueError(f"{name} differs from its registered SHA-256")
    return resolved, {"path": relative.as_posix(), "sha256": expected}


def _validate_pilot_authorization_document(
    authorization: dict[str, Any],
    contract: dict[str, Any],
    contract_path: Path,
    repository_root: Path,
    *,
    require_contract_path_match: bool,
) -> None:
    campaign = _campaign_identity(contract)
    required = {
        "schema_version",
        "authorization_kind",
        "authorization_id",
        "status",
        "contract",
        "contract_receipt",
        "campaign",
        "purpose",
        "participants",
    }
    if not required.issubset(authorization):
        raise ValueError("pilot authorization is missing required consumer fields")
    identity = (
        authorization.get("schema_version"),
        authorization.get("authorization_kind"),
        authorization.get("status"),
        authorization.get("purpose"),
        authorization.get("participants"),
    )
    expected_identity = (
        1,
        PILOT_AUTHORIZATION_KIND,
        PILOT_AUTHORIZATION_STATUS,
        "pilot",
        PILOT_AUTHORIZATION_PARTICIPANTS,
    )
    if identity != expected_identity:
        raise ValueError("pilot authorization identity, purpose, or participants differ")
    authorization_id = authorization.get("authorization_id")
    if (
        not isinstance(authorization_id, str)
        or not authorization_id
        or authorization_id != authorization_id.strip()
    ):
        raise ValueError("pilot authorization_id must be a nonempty string")
    authorized_campaign = _mapping(authorization.get("campaign"), "pilot authorization campaign")
    if authorized_campaign != campaign:
        raise ValueError("pilot authorization campaign differs from the contract")

    authorized_contract, contract_reference = _repository_artifact(
        authorization.get("contract"), "pilot authorization contract", repository_root
    )
    if require_contract_path_match and authorized_contract != contract_path.resolve():
        raise ValueError("pilot authorization names another execution contract")
    contract_sha256 = _sha256(contract_path)
    if contract_reference["sha256"] != contract_sha256:
        raise ValueError("pilot authorization contract SHA-256 differs")

    receipt_path, _ = _repository_artifact(
        authorization.get("contract_receipt"),
        "pilot authorization contract receipt",
        repository_root,
    )
    receipt = _json(receipt_path, "pilot authorization contract receipt")
    if (
        receipt.get("contract_validation_status") != "PASS_INPUT_IDENTITY_AND_SEMANTICS_LOCKED"
        or receipt.get("comsol_or_candidate_executed") is not False
        or receipt.get("contract_sha256") != contract_sha256
        or receipt.get("campaign") != campaign
    ):
        raise ValueError("pilot authorization contract receipt differs from the contract")


def _pilot_authorization_reference(
    contract: dict[str, Any],
    contract_path: Path,
    authorization_path: Path | None,
    mode: str,
) -> dict[str, str] | None:
    explicit_campaign = contract.get("campaign") is not None
    required = mode == "FullPilot" and explicit_campaign
    if not required:
        if authorization_path is not None:
            raise ValueError(
                "--pilot-authorization is only valid for an explicit FullPilot campaign"
            )
        return None
    if authorization_path is None:
        raise ValueError("explicit FullPilot requires --pilot-authorization")
    repository_root = _repository_root(contract_path)
    authorization_path = authorization_path.resolve()
    if not authorization_path.is_relative_to(repository_root) or not authorization_path.is_file():
        raise ValueError("pilot authorization is missing or outside the repository")
    resolved, reference = _repository_artifact(
        {
            "path": authorization_path.relative_to(repository_root).as_posix(),
            "sha256": _sha256(authorization_path),
        },
        "pilot authorization",
        repository_root,
    )
    authorization = _json(resolved, "pilot authorization")
    _validate_pilot_authorization_document(
        authorization,
        contract,
        contract_path,
        repository_root,
        require_contract_path_match=True,
    )
    return reference


def _validate_staged_pilot_authorization(
    root: Path, request: dict[str, Any]
) -> dict[str, str] | None:
    contract_record = _mapping(request.get("execution_contract"), "execution contract")
    contract_path = root / str(contract_record.get("filename", ""))
    contract = _json(contract_path, "staged M3-C2 contract")
    reference_value = request.get("pilot_authorization")
    required = request.get("mode") == "FullPilot" and contract.get("campaign") is not None
    if not required:
        if reference_value is not None:
            raise ValueError("execution request has an unexpected pilot authorization")
        return None
    if reference_value is None:
        raise ValueError("explicit FullPilot execution request has no pilot authorization")

    repository_root = _repository_root(contract_path)
    source_path, reference = _repository_artifact(
        reference_value, "execution-request pilot authorization", repository_root
    )
    staged_path = root / source_path.name
    if not staged_path.is_file() or _sha256(staged_path) != reference["sha256"]:
        raise ValueError("staged pilot authorization differs from the execution request")
    authorization = _json(staged_path, "staged pilot authorization")
    _validate_pilot_authorization_document(
        authorization,
        contract,
        contract_path,
        repository_root,
        require_contract_path_match=False,
    )
    return reference


def _pilot_request(contract_path: Path, mode: str) -> dict[str, Any]:
    contract = _json(contract_path, "M3-C2 pilot contract")
    seed_plan = _mapping(contract.get("seed_plan"), "contract.seed_plan")
    pilot = _mapping(seed_plan.get("pilot"), "contract.seed_plan.pilot")
    comsol_seeds = _integer_list(pilot.get("comsol_seeds"), "pilot COMSOL seeds")
    candidate_seeds = _integer_list(pilot.get("candidate_seeds"), "pilot candidate seeds")
    if len(comsol_seeds) != 4 or len(candidate_seeds) != 4:
        raise ValueError("pilot contract must contain four seeds per participant")
    if set(comsol_seeds).intersection(candidate_seeds):
        raise ValueError("pilot participant seed sets must be disjoint")
    seeds = comsol_seeds[:1] if mode == "RunnerValidation" else comsol_seeds
    pilot_steps = _pilot_steps_ns(contract)
    steps_ns = pilot_steps[:1] if mode == "RunnerValidation" else pilot_steps
    return {
        "purpose": "pilot",
        "seeds": seeds,
        "steps_ns": list(steps_ns),
        "expected_replicas_per_level": len(seeds),
        "normalized_manifest": PILOT_MANIFEST_FILE,
        "registration": {
            "kind": "locked_pilot_contract",
            "filename": contract_path.name,
            "sha256": _sha256(contract_path),
        },
        "seed_isolation": {
            "other_participant_seeds": candidate_seeds,
            "excluded_pilot_seeds": [],
            "participant_seed_sets_disjoint": True,
        },
    }


def _seed_allocation_source(
    value: object, name: str, base: Path, allowed_root: Path
) -> tuple[Path, str, dict[str, Any]]:
    record = _mapping(value, name)
    if set(record) != {"path", "sha256", "json_pointers"}:
        raise ValueError(f"{name} has unexpected keys")
    pointers = _mapping(record.get("json_pointers"), f"{name}.json_pointers")
    if pointers != {
        "comsol": "/participant_seed_sets/comsol",
        "candidate": "/participant_seed_sets/candidate",
    }:
        raise ValueError(f"{name} JSON pointers are invalid")
    relative = Path(str(record.get("path", "")))
    if relative.is_absolute() or str(relative) in {"", "."}:
        raise ValueError(f"{name}.path must be relative")
    path = (base / relative).resolve()
    expected = str(record.get("sha256", "")).lower()
    if not path.is_relative_to(allowed_root) or not path.is_file() or _sha256(path) != expected:
        raise ValueError(f"{name} differs from its registered SHA-256")
    return path, expected, _json(path, "final seed allocation")


def _referenced_final_seed_participants(
    registration: dict[str, Any],
    registration_path: Path,
    policy_final: dict[str, Any],
    policy_path: Path,
    contract_final: dict[str, Any],
    contract_path: Path,
    campaign: dict[str, str],
) -> dict[str, Any]:
    sources = (
        registration.get("participant_seed_source"),
        policy_final.get("seed_allocation"),
        contract_final.get("seed_source"),
    )
    if any(source is None for source in sources):
        raise ValueError("final seed allocation must be locked by every campaign authority")
    if (
        registration.get("participant_seed_sets") is not None
        or policy_final.get("seed_plan") is not None
    ):
        raise ValueError("referenced final seeds must have one owner")
    project_root = _solver_project_root(contract_path)
    repository_root = _repository_root(contract_path)
    resolved = (
        _seed_allocation_source(
            sources[0],
            "registration participant seed source",
            registration_path.parent,
            project_root,
        ),
        _seed_allocation_source(
            sources[1],
            "evaluation policy seed allocation",
            policy_path.parent,
            project_root,
        ),
        _seed_allocation_source(
            sources[2],
            "contract final seed source",
            repository_root,
            repository_root,
        ),
    )
    if len({(path, digest) for path, digest, _ in resolved}) != 1:
        raise ValueError("campaign authorities reference different final seed allocations")
    allocation = resolved[0][2]
    if (
        allocation.get("schema_version"),
        allocation.get("allocation_kind"),
        allocation.get("case_id"),
        allocation.get("purpose"),
        allocation.get("replicas_per_participant"),
    ) != (1, "m3c2_final_seed_allocation", campaign["case_id"], "final", FINAL_REPLICAS):
        raise ValueError("final seed allocation identity or scope is invalid")
    if (
        contract_final.get("case_id") != campaign["case_id"]
        or int(contract_final.get("replicas_per_participant", -1)) != FINAL_REPLICAS
    ):
        raise ValueError("contract final cohort differs from the seed allocation")
    return _mapping(allocation.get("participant_seed_sets"), "allocated participant seed sets")


def _final_seed_participants(
    registration: dict[str, Any],
    registration_path: Path,
    authorization_artifacts: dict[str, dict[str, str]],
    contract: dict[str, Any],
    contract_path: Path,
    campaign: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    policy_record = authorization_artifacts["evaluation_policy"]
    policy_path = (registration_path.parent / policy_record["path"]).resolve()
    policy = _json(policy_path, "M3-C2 evaluation policy")
    policy_final = _mapping(policy.get("final"), "evaluation policy final section")
    seed_plan = _mapping(contract.get("seed_plan"), "contract.seed_plan")
    contract_final_value = seed_plan.get("final_cohort")
    referenced = any(
        value is not None
        for value in (
            registration.get("participant_seed_source"),
            policy_final.get("seed_allocation"),
            (
                None
                if contract_final_value is None
                else _mapping(contract_final_value, "contract final cohort").get("seed_source")
            ),
        )
    )
    if referenced:
        contract_final = _mapping(contract_final_value, "contract final cohort")
        participants = _referenced_final_seed_participants(
            registration,
            registration_path,
            policy_final,
            policy_path,
            contract_final,
            contract_path,
            campaign,
        )
    else:
        if campaign != LEGACY_CAMPAIGN:
            raise ValueError("new campaigns must reference one final seed allocation")
        participants = _mapping(
            registration.get("participant_seed_sets"),
            "registration.participant_seed_sets",
        )
    return participants, policy_final, referenced


def _validated_final_seed_sets(
    participants: dict[str, Any],
    policy_final: dict[str, Any],
    contract: dict[str, Any],
    *,
    referenced: bool,
) -> tuple[list[int], list[int], set[int]]:
    if set(participants) != {"comsol", "candidate"}:
        raise ValueError("final registration must contain exactly COMSOL and candidate seeds")
    comsol_seeds = _integer_list(participants.get("comsol"), "final COMSOL seeds")
    candidate_seeds = _integer_list(participants.get("candidate"), "final candidate seeds")
    if len(comsol_seeds) != FINAL_REPLICAS or len(candidate_seeds) != FINAL_REPLICAS:
        raise ValueError("final campaign requires exactly 32 seeds per participant")
    if set(comsol_seeds).intersection(candidate_seeds):
        raise ValueError("final participant seed sets must be disjoint")
    pilot = _mapping(
        _mapping(contract.get("seed_plan"), "contract.seed_plan").get("pilot"),
        "contract.seed_plan.pilot",
    )
    pilot_seeds = set(_integer_list(pilot.get("comsol_seeds"), "pilot COMSOL seeds"))
    pilot_seeds.update(_integer_list(pilot.get("candidate_seeds"), "pilot candidate seeds"))
    if pilot_seeds.intersection(comsol_seeds) or pilot_seeds.intersection(candidate_seeds):
        raise ValueError("final seeds must be disjoint from both pilot participants")
    if not referenced:
        policy_seeds = _mapping(policy_final.get("seed_plan"), "evaluation policy seed plan")
        if (
            _integer_list(policy_seeds.get("comsol"), "policy COMSOL seeds") != comsol_seeds
            or _integer_list(policy_seeds.get("candidate"), "policy candidate seeds")
            != candidate_seeds
        ):
            raise ValueError("final registration seed sets differ from the registered policy")
    if int(policy_final.get("replicas_per_participant", -1)) != FINAL_REPLICAS:
        raise ValueError("final registration replica count differs from the registered policy")
    return comsol_seeds, candidate_seeds, pilot_seeds


def _final_request(
    contract_path: Path,
    registration_path: Path | None,
) -> dict[str, Any]:
    if registration_path is None:
        raise ValueError("FinalCampaign requires --registration")
    contract = _json(contract_path, "M3-C2 pilot contract")
    campaign = _campaign_identity(contract)
    registration = _json(registration_path, "M3-C2 final campaign registration")
    if (
        registration.get("schema_version") != 1
        or registration.get("registration_kind") != campaign["final_registration_kind"]
        or registration.get("case_id") != campaign["case_id"]
        or registration.get("purpose") != "final"
    ):
        raise ValueError("final campaign registration identity is invalid")
    authorization = _mapping(
        registration.get("execution_authorization"),
        "registration.execution_authorization",
    )
    if authorization.get("status") != "AUTHORIZED":
        raise ValueError("final campaign is not authorized for execution")
    authorization_artifacts = {
        key: _authorization_artifact(
            authorization.get(key),
            f"registration.execution_authorization.{key}",
            registration_path,
            _solver_project_root(contract_path),
        )
        for key in ("evaluation_policy", "pilot_evaluation", "selection_receipt")
    }
    participants, policy_final, referenced = _final_seed_participants(
        registration,
        registration_path,
        authorization_artifacts,
        contract,
        contract_path,
        campaign,
    )
    comsol_seeds, candidate_seeds, pilot_seeds = _validated_final_seed_sets(
        participants, policy_final, contract, referenced=referenced
    )
    step_ns = _selected_comsol_final_step_ns(
        registration, registration_path, authorization_artifacts
    )
    return {
        "purpose": "final",
        "seeds": comsol_seeds,
        "steps_ns": [step_ns],
        "expected_replicas_per_level": FINAL_REPLICAS,
        "normalized_manifest": FINAL_MANIFEST_FILE,
        "registration": {
            "kind": campaign["final_registration_kind"],
            "filename": registration_path.name,
            "sha256": _sha256(registration_path),
            "authorization_artifacts": authorization_artifacts,
        },
        "seed_isolation": {
            "other_participant_seeds": candidate_seeds,
            "excluded_pilot_seeds": sorted(pilot_seeds),
            "participant_seed_sets_disjoint": True,
        },
    }


def prepare_request(
    root: Path,
    mode: str,
    contract_path: Path,
    registration_path: Path | None = None,
    pilot_authorization_path: Path | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    if mode not in {"RunnerValidation", "FullPilot", "FinalCampaign"}:
        raise ValueError(f"unsupported COMSOL execution mode: {mode}")
    if mode != "FinalCampaign" and registration_path is not None:
        raise ValueError("--registration is only valid for FinalCampaign")
    resolved_contract = contract_path.resolve()
    contract = _json(resolved_contract, "M3-C2 pilot contract")
    campaign = _campaign_identity(contract)
    binding = _campaign_binding(contract, resolved_contract)
    comsol_semantics = _comsol_semantics(contract)
    pilot_authorization = _pilot_authorization_reference(
        contract,
        resolved_contract,
        None if pilot_authorization_path is None else pilot_authorization_path.resolve(),
        mode,
    )
    selected = (
        _final_request(
            contract_path.resolve(),
            None if registration_path is None else registration_path.resolve(),
        )
        if mode == "FinalCampaign"
        else _pilot_request(contract_path.resolve(), mode)
    )
    request_rows = [
        {
            "seed": seed,
            "step_ns": step_ns,
            "directory": _request_directory(seed, step_ns),
        }
        for step_ns in selected["steps_ns"]
        for seed in selected["seeds"]
    ]
    execution = {
        "schema_version": 1,
        "tool_revision": EXECUTION_REQUEST_REVISION,
        "mode": mode,
        "purpose": selected["purpose"],
        "case_id": campaign["case_id"],
        "campaign_identity": campaign,
        "campaign_binding": binding,
        **({"pilot_authorization": pilot_authorization} if pilot_authorization is not None else {}),
        "execution_contract": {
            "filename": resolved_contract.name,
            "sha256": binding["contract_sha256"],
        },
        "comsol_semantics": comsol_semantics,
        "participant": "comsol",
        "seeds": selected["seeds"],
        "steps_ns": selected["steps_ns"],
        "expected_replicas_per_level": selected["expected_replicas_per_level"],
        "request_rows": request_rows,
        "normalized_manifest": selected["normalized_manifest"],
        "registration": selected["registration"],
        "seed_isolation": selected["seed_isolation"],
    }
    request_path = root / REQUEST_FILE
    with request_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("seed", "step_ns", "relative_directory"))
        for row in request_rows:
            writer.writerow((row["seed"], row["step_ns"], row["directory"]))
    _write_json(root / EXECUTION_REQUEST_FILE, execution)
    return execution


def _output_times() -> list[float]:
    early = [index * 1.0e-5 for index in range(51)]
    middle = [6.0e-4 + index * 1.0e-4 for index in range(45)]
    late = [6.0e-3 + index * 1.0e-3 for index in range(25)]
    result = early + middle + late
    if len(result) != EXPECTED_FRAMES:
        raise AssertionError("invalid locked output schedule")
    return result


def _fields(line: str, prefix: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in line[line.index(prefix) + len(prefix) :].strip().split("|"):
        if "=" not in token:
            raise ValueError(f"malformed COMSOL receipt token: {token!r}")
        key, value = token.split("=", 1)
        if not key or key in result:
            raise ValueError(f"duplicate or empty COMSOL receipt key: {key!r}")
        result[key] = value
    return result


def _solve_for_map(value: str | None, name: str) -> dict[str, bool]:
    if value is None or not value:
        raise ValueError(f"missing COMSOL {name} solve-for receipt")
    if value == "none":
        return {}
    result: dict[str, bool] = {}
    for token in value.split(","):
        if "=" not in token:
            raise ValueError(f"malformed COMSOL {name} solve-for token: {token!r}")
        tag, raw = token.split("=", 1)
        if not tag or tag in result or raw not in {"true", "false"}:
            raise ValueError(f"invalid COMSOL {name} solve-for token: {token!r}")
        result[tag] = raw == "true"
    return result


def _study_isolation(values: dict[str, str], physics_tag: str = "fptas") -> dict[str, object]:
    physics = _solve_for_map(values.get("solve_for_physics"), "physics")
    multiphysics = _solve_for_map(values.get("solve_for_multiphysics"), "multiphysics")
    enabled_physics = sorted(tag for tag, enabled in physics.items() if enabled)
    enabled_multiphysics = sorted(tag for tag, enabled in multiphysics.items() if enabled)
    if values.get("solve_for_assertion") != "PASS" or enabled_physics != [physics_tag]:
        raise ValueError(f"COMSOL study is not isolated to {physics_tag}")
    if enabled_multiphysics:
        raise ValueError("COMSOL study has an enabled multiphysics coupling")
    return {
        "status": "PASS",
        "enabled_physics": enabled_physics,
        "enabled_multiphysics": enabled_multiphysics,
        "physics_solve_for": dict(sorted(physics.items())),
        "multiphysics_solve_for": dict(sorted(multiphysics.items())),
    }


def _receipts(
    process_log: Path,
    comsol_semantics: dict[str, Any] | None = None,
) -> tuple[dict[tuple[int, int], dict[str, str]], dict[tuple[int, int], float]]:
    semantics = LEGACY_COMSOL if comsol_semantics is None else comsol_semantics
    text = process_log.read_text(encoding="utf-8", errors="replace")
    configuration_lines = sorted(
        {
            line[line.index(CONFIGURATION_PREFIX) :].strip()
            for line in text.splitlines()
            if CONFIGURATION_PREFIX in line
        }
    )
    solve_lines = sorted(
        {
            line[line.index(SOLVE_PREFIX) :].strip()
            for line in text.splitlines()
            if SOLVE_PREFIX in line
        }
    )
    configurations: dict[tuple[int, int], dict[str, str]] = {}
    for line in configuration_lines:
        values = _fields(line, CONFIGURATION_PREFIX)
        key = (int(values["seed"]), round(float(values["step_s"]) * 1.0e9))
        if key in configurations:
            raise ValueError(f"duplicate COMSOL configuration receipt: {key}")
        expected = {
            "random_number_args": "UserDefined",
            "seed_authority": str(semantics["sole_seed_authority"]),
            "brownian_seed_expression": str(semantics["seed_parameter"]),
            "brownian_active": "true",
            "out_of_plane": "false",
            "brownian_viscosity": str(semantics["viscosity_expression"]),
            "brownian_temperature": str(semantics["temperature_expression"]),
            "saffman_active": "false",
            "dynamic_charge_active": "true",
            "field_source": "canonical_exact_connectivity_P1_sectionwise",
            "initial_state_source": "candidate_realized_source_table",
            "integrator": "classical_rk4",
            "integrator_order": "4",
            "relative_tolerance": "1e-2",
            "wall_accuracy_order": "1",
            "output_times": str(EXPECTED_FRAMES),
            "particle_rows": str(EXPECTED_PARTICLES),
            "time_end_s": "0.03",
            "source_model": "source_copy.mph",
            "model_saved": "false",
            "solve_for_assertion": "PASS",
        }
        for name, expected_value in expected.items():
            if values.get(name) != expected_value:
                raise ValueError(
                    f"COMSOL receipt {key} expected {name}={expected_value}, "
                    f"found {values.get(name)!r}"
                )
        _study_isolation(values, str(semantics["physics_tag"]))
        configurations[key] = values
    durations: dict[tuple[int, int], float] = {}
    for line in solve_lines:
        values = _fields(line, SOLVE_PREFIX)
        key = (int(values["seed"]), round(float(values["step_s"]) * 1.0e9))
        seconds = float(values["seconds"])
        if key in durations or not math.isfinite(seconds) or seconds <= 0.0:
            raise ValueError(f"duplicate or invalid COMSOL solve receipt: {key}")
        durations[key] = seconds
    return configurations, durations


def _request_rows(
    path: Path,
    expected: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["seed", "step_ns", "relative_directory"]:
            raise ValueError(f"{path}: invalid request header")
        for row in reader:
            seed = int(row["seed"])
            step_ns = int(row["step_ns"])
            directory = row["relative_directory"]
            if seed < 0 or seed > 2_147_483_647:
                raise ValueError(f"{path}: seed is outside the COMSOL Java integer range")
            if step_ns <= 0 or TIME_END_NS % step_ns != 0:
                raise ValueError(f"{path}: step must be positive and divide the 30 ms horizon")
            if Path(directory).is_absolute() or ".." in Path(directory).parts:
                raise ValueError(f"{path}: unsafe relative directory {directory}")
            if directory != _request_directory(seed, step_ns):
                raise ValueError(f"{path}: request directory does not match seed and step")
            result.append({"seed": seed, "step_ns": step_ns, "directory": directory})
    if not result or len({(row["seed"], row["step_ns"]) for row in result}) != len(result):
        raise ValueError(f"{path}: empty or duplicate execution request")
    if expected is not None and result != expected:
        raise ValueError(f"{path}: CSV rows differ from the registered execution request")
    return result


def _validate_execution_seed_scope(
    request: dict[str, Any],
    mode: str,
    seeds: list[int],
    steps_ns: list[int],
) -> None:
    expected_shapes = {
        "RunnerValidation": (1, 1),
        "FullPilot": (4, 3),
        "FinalCampaign": (FINAL_REPLICAS, 1),
    }
    if (len(seeds), len(steps_ns)) != expected_shapes[mode]:
        raise ValueError(f"{mode} execution request has the wrong seed/step shape")
    if request.get("expected_replicas_per_level") != len(seeds):
        raise ValueError("execution-request replica count differs from its seed cohort")
    isolation = _mapping(request.get("seed_isolation"), "execution-request seed isolation")
    other = set(_integer_list(isolation.get("other_participant_seeds"), "other seeds"))
    excluded = set(_integer_list(isolation.get("excluded_pilot_seeds"), "excluded seeds"))
    if (
        isolation.get("participant_seed_sets_disjoint") is not True
        or set(seeds).intersection(other)
        or set(seeds).intersection(excluded)
    ):
        raise ValueError("execution-request seed isolation is invalid")


def _validate_staged_registration(root: Path, request: dict[str, Any]) -> None:
    registration = _mapping(request.get("registration"), "execution-request registration")
    filename = str(registration.get("filename", ""))
    expected = str(registration.get("sha256", "")).lower()
    if not filename or Path(filename).name != filename:
        raise ValueError("execution-request registration filename is unsafe")
    path = root / filename
    if len(expected) != 64 or not path.is_file() or _sha256(path) != expected:
        raise ValueError("staged execution registration differs from its SHA-256")


def _validate_campaign_binding(root: Path, request: dict[str, Any]) -> dict[str, str]:
    binding_raw = _mapping(request.get("campaign_binding"), "execution-request campaign binding")
    if set(binding_raw) != {"contract_sha256", "input_sha256", "input_content_hash"}:
        raise ValueError("execution-request campaign binding has unexpected fields")
    binding = {name: str(value) for name, value in binding_raw.items()}
    contract_record = _mapping(request.get("execution_contract"), "execution contract")
    if set(contract_record) != {"filename", "sha256"}:
        raise ValueError("execution contract reference has unexpected fields")
    filename = str(contract_record["filename"])
    contract_path = root / filename
    if (
        Path(filename).name != filename
        or str(contract_record["sha256"]).lower() != binding["contract_sha256"]
        or not contract_path.is_file()
        or _sha256(contract_path) != binding["contract_sha256"]
    ):
        raise ValueError("staged execution contract differs from campaign binding")
    contract = _json(contract_path, "staged M3-C2 contract")
    if _campaign_binding(contract, contract_path) != binding:
        raise ValueError("staged contract common-P1 identity differs from campaign binding")
    table_receipt = _json(root / "common_p1_table_receipt.json", "common-P1 table receipt")
    candidate = _mapping(table_receipt.get("candidate"), "common-P1 table candidate")
    if {
        "input_sha256": str(candidate.get("file_sha256", "")).lower(),
        "input_content_hash": str(candidate.get("content_hash", "")),
    } != {
        "input_sha256": binding["input_sha256"],
        "input_content_hash": binding["input_content_hash"],
    }:
        raise ValueError("prepared common-P1 input differs from campaign binding")
    return binding


def _execution_request(root: Path) -> dict[str, Any]:
    path = root / EXECUTION_REQUEST_FILE
    request = _json(path, "M3-C2 COMSOL execution request")
    campaign = _mapping(request.get("campaign_identity"), "execution-request campaign identity")
    if set(campaign) != set(LEGACY_CAMPAIGN) or request.get("case_id") != campaign.get("case_id"):
        raise ValueError("M3-C2 COMSOL campaign identity is invalid")
    identity = (
        request.get("schema_version"),
        request.get("tool_revision"),
        request.get("participant"),
    )
    if identity != (1, EXECUTION_REQUEST_REVISION, "comsol"):
        raise ValueError("M3-C2 COMSOL execution request identity is invalid")
    semantics = _mapping(request.get("comsol_semantics"), "execution-request COMSOL semantics")
    physics = str(semantics.get("physics_tag", ""))
    if semantics.get("sole_seed_authority") != f"{physics}.bf1.i":
        raise ValueError("M3-C2 COMSOL execution request seed authority is invalid")
    mode_policy = {
        "RunnerValidation": ("pilot", PILOT_MANIFEST_FILE),
        "FullPilot": ("pilot", PILOT_MANIFEST_FILE),
        "FinalCampaign": ("final", FINAL_MANIFEST_FILE),
    }
    mode = str(request.get("mode", ""))
    if mode not in mode_policy or (
        request.get("purpose"),
        request.get("normalized_manifest"),
    ) != mode_policy.get(mode):
        raise ValueError("M3-C2 COMSOL execution-request mode is inconsistent")
    expected = request.get("request_rows")
    if not isinstance(expected, list) or not expected:
        raise ValueError("M3-C2 COMSOL execution request has no request rows")
    rows = cast(list[dict[str, Any]], expected)
    seeds = _integer_list(request.get("seeds"), "execution-request seeds")
    steps_ns = _integer_list(request.get("steps_ns"), "execution-request steps")
    _validate_execution_seed_scope(request, mode, seeds, steps_ns)
    _validate_staged_registration(root, request)
    _validate_campaign_binding(root, request)
    _validate_staged_pilot_authorization(root, request)
    expected_rows = [
        {
            "seed": seed,
            "step_ns": step_ns,
            "directory": _request_directory(seed, step_ns),
        }
        for step_ns in steps_ns
        for seed in seeds
    ]
    if rows != expected_rows:
        raise ValueError("execution-request rows do not match its seed and step authorities")
    _request_rows(root / REQUEST_FILE, rows)
    return request


def _release_state(path: Path) -> dict[int, tuple[float, float, float, float, float]]:
    result: dict[int, tuple[float, float, float, float, float]] = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {
            "particle_id",
            "r_m",
            "z_m",
            "velocity_r_m_per_s",
            "velocity_z_m_per_s",
            "charge_number",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path}: release-probe columns are incomplete")
        for row in reader:
            particle_id = int(row["particle_id"])
            values = tuple(
                float(row[name])
                for name in (
                    "r_m",
                    "z_m",
                    "velocity_r_m_per_s",
                    "velocity_z_m_per_s",
                    "charge_number",
                )
            )
            if particle_id in result or not all(math.isfinite(value) for value in values):
                raise ValueError(f"{path}: duplicate or nonfinite release {particle_id}")
            result[particle_id] = values  # type: ignore[assignment]
    if set(result) != set(range(1, EXPECTED_PARTICLES + 1)):
        raise ValueError(f"{path}: release particle IDs must be exactly 1..287")
    return result


def _raw_rows(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for line in stream:
            if line.startswith("%") or not line.strip():
                continue
            rows.append([float(value) for value in next(csv.reader([line]))])
    return rows


def _integer(value: float, context: str) -> int:
    rounded = round(value)
    if not math.isfinite(value) or abs(value - rounded) > 1.0e-9:
        raise ValueError(f"{context}: expected integer-valued data, found {value}")
    return rounded


def _initial_limit(observed: float, expected: float, component_scale: float) -> float:
    return 4096.0 * math.ulp(max(abs(observed), abs(expected), component_scale, 1.0e-300))


def _finite_state(record: list[float]) -> bool:
    return all(math.isfinite(record[index]) for index in range(7))


def _status(record: list[float], index: int, raw_path: Path) -> int:
    status_code = _integer(record[index], str(raw_path))
    if status_code not in STATUS:
        raise ValueError(f"{raw_path}: unknown status {status_code}")
    return status_code


def _terminal_candidate(record: list[float], raw_path: Path) -> tuple[str, float] | None:
    if not _finite_state(record):
        return None
    status_code = _status(record, 7, raw_path)
    final_status_code = _status(record, 8, raw_path)
    outcome_code = status_code if status_code != 1 else final_status_code
    if outcome_code == 1:
        return None
    event_time = record[9]
    if not math.isfinite(event_time) or event_time < 0.0:
        raise ValueError(f"{raw_path}: invalid terminal event time")
    if event_time > TIME_END_S + TIME_TOLERANCE_S:
        if status_code != 1:
            raise ValueError(f"{raw_path}: nonactive state beyond time horizon")
        return None
    return STATUS[outcome_code], event_time


def _infer_terminal(records: list[list[float]], raw_path: Path) -> tuple[str, float] | None:
    terminal: tuple[str, float] | None = None
    for record in records:
        candidate = _terminal_candidate(record, raw_path)
        if candidate is None:
            continue
        if terminal is not None and terminal != candidate:
            raise ValueError(f"{raw_path}: inconsistent terminal state")
        terminal = candidate
    return terminal


def _write_missing_escape(
    writer: Any,
    record: list[float],
    terminal: tuple[str, float] | None,
    particle_id: int,
    frame: int,
    expected_time: float,
    raw_path: Path,
) -> None:
    if any(math.isfinite(record[index]) for index in range(2, 7)):
        raise ValueError(f"{raw_path}: partially finite state for particle {particle_id}")
    if (
        terminal is None
        or terminal[0] != "escaped"
        or expected_time + TIME_TOLERANCE_S < terminal[1]
    ):
        raise ValueError(
            f"{raw_path}: missing active/pre-terminal state for particle "
            f"{particle_id}, frame {frame}"
        )
    writer.writerow(
        (particle_id, expected_time, math.nan, math.nan, math.nan, math.nan, math.nan, "escaped")
    )


def _validate_initial_state(
    record: list[float],
    expected: tuple[float, float, float, float, float],
    scales: tuple[float, ...],
    particle_id: int,
    raw_path: Path,
) -> None:
    observed = (record[2], record[3], record[4], record[5], record[6])
    for component, (actual, reference) in enumerate(zip(observed, expected, strict=True)):
        if abs(actual - reference) > _initial_limit(actual, reference, scales[component]):
            raise ValueError(
                f"{raw_path}: t=0 state differs from realized source for particle {particle_id}"
            )


def _write_particle_trajectory(
    writer: Any,
    records: list[list[float]],
    expected_times: list[float],
    terminal: tuple[str, float] | None,
    particle_id: int,
    release: tuple[float, float, float, float, float],
    release_scales: tuple[float, ...],
    raw_path: Path,
) -> str:
    last_lifecycle = "active"
    for frame, (record, expected_time) in enumerate(zip(records, expected_times, strict=True)):
        if not _finite_state(record):
            _write_missing_escape(
                writer, record, terminal, particle_id, frame, expected_time, raw_path
            )
            last_lifecycle = "escaped"
            continue
        if _integer(record[0], str(raw_path)) != particle_id:
            raise ValueError(f"{raw_path}: particle ID changed within a wide row")
        if not math.isclose(record[1], expected_time, rel_tol=0.0, abs_tol=TIME_TOLERANCE_S):
            raise ValueError(
                f"{raw_path}: particle {particle_id} frame {frame} has unexpected time"
            )
        last_lifecycle = STATUS[_status(record, 7, raw_path)]
        writer.writerow((particle_id, *record[1:7], last_lifecycle))
        if frame == 0:
            _validate_initial_state(record, release, release_scales, particle_id, raw_path)
    return last_lifecycle


def _normalize_replica(
    directory: Path,
    release: dict[int, tuple[float, float, float, float, float]],
    seed: int,
    step_ns: int,
    seconds: float,
    peak_rss_bytes: int,
    configuration: dict[str, str],
    physics_tag: str = "fptas",
    boundary_meaning: Path | None = None,
) -> dict[str, Any]:
    raw_path = directory / "trajectory_raw_wide.csv"
    raw_rows = _raw_rows(raw_path)
    if len(raw_rows) != EXPECTED_PARTICLES:
        raise ValueError(f"{raw_path}: expected 287 particle rows, found {len(raw_rows)}")
    width = len(STATE_COLUMNS) * EXPECTED_FRAMES
    expected_times = _output_times()
    release_scales = tuple(
        max(abs(values[component]) for values in release.values()) for component in range(5)
    )
    ids: set[int] = set()
    lifecycle_counts: Counter[str] = Counter()
    event_rows: list[tuple[object, ...]] = []
    event_evidence: list[dict[str, Any]] = []
    actual = read_actual_run_receipt(directory, boundary_meaning)
    trajectory_rows = 0
    with (directory / "trajectory.csv").open("x", newline="", encoding="utf-8") as trajectory:
        writer = csv.writer(trajectory, lineterminator="\n")
        writer.writerow(TRAJECTORY_COLUMNS)
        for raw in raw_rows:
            if len(raw) != width:
                raise ValueError(f"{raw_path}: expected row width {width}, found {len(raw)}")
            particle_id = _integer(raw[0], str(raw_path))
            if particle_id in ids:
                raise ValueError(f"{raw_path}: duplicate particle {particle_id}")
            ids.add(particle_id)
            records = [
                raw[frame * len(STATE_COLUMNS) : (frame + 1) * len(STATE_COLUMNS)]
                for frame in range(EXPECTED_FRAMES)
            ]
            terminal = _infer_terminal(records, raw_path)
            last_lifecycle = _write_particle_trajectory(
                writer,
                records,
                expected_times,
                terminal,
                particle_id,
                release[particle_id],
                release_scales,
                raw_path,
            )
            trajectory_rows += EXPECTED_FRAMES
            final_lifecycle = terminal[0] if terminal is not None else last_lifecycle
            lifecycle_counts[final_lifecycle] += 1
            if terminal is not None:
                outcome, event_time = terminal
                event, evidence = normalize_terminal_event(actual, particle_id, outcome, event_time)
                event_rows.append(event)
                event_evidence.append(evidence)
    if ids != set(range(1, EXPECTED_PARTICLES + 1)):
        raise ValueError(f"{raw_path}: particle IDs must be exactly 1..287")
    if trajectory_rows != EXPECTED_PARTICLES * EXPECTED_FRAMES:
        raise ValueError(
            f"{raw_path}: expected {EXPECTED_PARTICLES * EXPECTED_FRAMES} normalized rows, "
            f"found {trajectory_rows}"
        )

    with (directory / "events.csv").open("x", newline="", encoding="utf-8") as events:
        writer = csv.writer(events, lineterminator="\n")
        writer.writerow(EVENT_COLUMNS)
        writer.writerows(sorted(event_rows))
    output_bytes = sum(
        path.stat().st_size
        for path in (raw_path, directory / "trajectory.csv", directory / "events.csv")
    )
    performance = {
        "schema_version": 1,
        "participant": "comsol",
        "seed": seed,
        "fixed_rk4_step_s": step_ns * 1.0e-9,
        "wall_time_s": seconds,
        "peak_rss_bytes": peak_rss_bytes,
        "output_bytes": output_bytes,
        "particle_count": EXPECTED_PARTICLES,
        "output_frames": EXPECTED_FRAMES,
        "stage_times_s": {"solve": seconds},
        "solve_wall_seconds": seconds,
        "process_count": 1,
        "peak_rss_measurement_scope": "maximum_working_set_of_comsolbatch_process_for_batch",
        "scope": {"particles": EXPECTED_PARTICLES, "time_end_s": 0.03},
    }
    _write_json(directory / "performance.json", performance)
    return {
        "seed": seed,
        "status": "COMPLETE",
        "trajectory": {
            "path": "trajectory.csv",
            "sha256": _sha256(directory / "trajectory.csv"),
        },
        "trajectory_rows": trajectory_rows,
        "events": {
            "path": "events.csv",
            "sha256": _sha256(directory / "events.csv"),
        },
        "event_count": len(event_rows),
        "actual_run_readback": actual.artifact,
        "terminal_boundary_evidence": event_evidence,
        "boundary_behavior": "NOT_TESTED" if not event_rows else "REQUIRES_MEANING_PREFLIGHT",
        "final_lifecycle_counts": dict(sorted(lifecycle_counts.items())),
        "performance": {
            "path": "performance.json",
            "sha256": _sha256(directory / "performance.json"),
        },
        "raw": {
            "path": "trajectory_raw_wide.csv",
            "sha256": _sha256(raw_path),
        },
        "study_isolation": _study_isolation(configuration, physics_tag),
    }


def _prefix_artifact_paths(summary: dict[str, Any], directory: str) -> None:
    for name in ("trajectory", "events", "performance", "raw"):
        artifact = summary[name]
        artifact["path"] = (Path(directory) / artifact["path"]).as_posix()
    if "path" in summary["actual_run_readback"]:
        artifact = summary["actual_run_readback"]
        artifact["path"] = (Path(directory) / artifact["path"]).as_posix()


def normalize(root: Path) -> dict[str, Any]:
    root = root.resolve()
    execution = _execution_request(root)
    campaign = _mapping(execution["campaign_identity"], "execution-request campaign identity")
    comsol_semantics = _mapping(execution["comsol_semantics"], "execution-request COMSOL semantics")
    physics_tag = str(comsol_semantics["physics_tag"])
    requests = _request_rows(root / REQUEST_FILE, execution["request_rows"])
    configurations, durations = _receipts(root / "comsol_process.log", comsol_semantics)
    materialize_actual_run_receipts(root)
    process_metrics = json.loads(
        (root / "comsol_process_metrics.json").read_text(encoding="utf-8-sig")
    )
    peak_rss_bytes = int(process_metrics.get("peak_rss_bytes", 0))
    if peak_rss_bytes <= 0:
        raise ValueError("COMSOL process peak RSS was not measured")
    request_keys = {(row["seed"], row["step_ns"]) for row in requests}
    if set(configurations) != request_keys or set(durations) != request_keys:
        raise ValueError("COMSOL receipts do not exactly cover the requested pilot runs")
    release = _release_state(root / "common_p1_release_probes.csv")
    replicas_by_step: dict[int, list[dict[str, Any]]] = {}
    for request in requests:
        key = (request["seed"], request["step_ns"])
        directory = root / request["directory"]
        if configurations[key].get("directory") != request["directory"]:
            raise ValueError(f"COMSOL configuration directory differs for request {key}")
        summary = _normalize_replica(
            directory,
            release,
            request["seed"],
            request["step_ns"],
            durations[key],
            peak_rss_bytes,
            configurations[key],
            physics_tag,
            root / "boundary_meaning.json",
        )
        _prefix_artifact_paths(summary, request["directory"])
        replicas_by_step.setdefault(request["step_ns"], []).append(summary)
    levels = []
    for ordinal, step_ns in enumerate(execution["steps_ns"]):
        replicas = replicas_by_step.get(step_ns)
        if replicas is None:
            continue
        levels.append(
            {
                "level_id": _step_label(step_ns),
                "ordinal": ordinal,
                "numerical_setting": {
                    "integrator": "classical_rk4",
                    "fixed_step_s": step_ns * 1.0e-9,
                    "purpose": (
                        "macro_step_convergence"
                        if execution["purpose"] == "pilot"
                        else "accepted_final"
                    ),
                },
                "replicas": sorted(replicas, key=lambda replica: replica["seed"]),
            }
        )
    manifest = {
        "schema_version": 1,
        "manifest_kind": "m3c2_participant",
        "tool_revision": TOOL_REVISION,
        "meaning_preflight_inventory": inventory_artifact(root),
        "source_model_sha256": (
            _sha256(root / "source_copy.mph") if (root / "source_copy.mph").is_file() else None
        ),
        "status": "COMPLETE_NORMALIZED_NOT_EVALUATED",
        "participant": "comsol",
        "case_id": campaign["evaluation_case_id"],
        "campaign_identity": campaign,
        "campaign_binding": _validate_campaign_binding(root, execution),
        **(
            {"pilot_authorization": execution["pilot_authorization"]}
            if execution.get("pilot_authorization") is not None
            else {}
        ),
        "purpose": execution["purpose"],
        "execution_mode": execution["mode"],
        "coordinate_system": "axisymmetric_rz_no_swirl",
        "common_observation_times": _output_times(),
        "levels": levels,
        "claim_policy": {
            "solver_accuracy": "NOT_EVALUATED",
            "step_convergence": "NOT_EVALUATED",
            "comsol_golden_truth": False,
            "pathwise_comparison": False,
        },
        "execution_request": {
            "path": EXECUTION_REQUEST_FILE,
            "sha256": _sha256(root / EXECUTION_REQUEST_FILE),
            "registration": execution["registration"],
            "seed_isolation": execution["seed_isolation"],
        },
        "study_isolation": {
            "status": "PASS",
            "required_enabled_physics": [physics_tag],
            "required_enabled_multiphysics": [],
            "validated_for_every_replica": True,
        },
    }
    _write_json(root / execution["normalized_manifest"], manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-request")
    prepare.add_argument("output_directory", type=Path)
    prepare.add_argument(
        "--mode",
        required=True,
        choices=("RunnerValidation", "FullPilot", "FinalCampaign"),
    )
    prepare.add_argument("--contract", required=True, type=Path)
    prepare.add_argument("--registration", type=Path)
    prepare.add_argument("--pilot-authorization", type=Path)
    normalize_parser = subparsers.add_parser("normalize")
    normalize_parser.add_argument("output_directory", type=Path)
    arguments = parser.parse_args()
    if arguments.command == "prepare-request":
        prepare_request(
            arguments.output_directory,
            arguments.mode,
            arguments.contract,
            arguments.registration,
            arguments.pilot_authorization,
        )
    else:
        normalize(arguments.output_directory)


if __name__ == "__main__":
    main()
