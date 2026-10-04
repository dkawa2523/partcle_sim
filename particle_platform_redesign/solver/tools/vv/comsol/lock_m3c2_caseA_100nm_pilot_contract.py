"""Validate and receipt a supported M3-C2A 100 nm execution contract.

This tool only locks already-existing input identity and the intended pilot
semantics.  It does not run COMSOL or the candidate solver, authorize either
runner, or make an accuracy claim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

from chamber_particles.case_format import read_with_info

TOOL_REVISION: Final = "m3c2_stochastic_pilot_contract_locker_v2"
EXPECTED_ROLES: Final = {
    "source_mph",
    "common_p1_input",
    "deterministic_30ms_contract",
    "common_p1_configuration",
    "common_p1_evidence",
    "m3c2_preflight_configuration",
    "m3c2_preflight_evidence",
    "m3c2_final_seed_allocation",
    "m3c2_model_semantics_probe",
}
CAMPAIGN_KEYS: Final = {
    "case_id",
    "evaluation_case_id",
    "output_slug",
    "final_registration_kind",
    "candidate_case_name_prefix",
}
CASE_P_CAMPAIGN: Final = {
    "case_id": "formal_iondrag_theory_consistent/caseP_100nm",
    "evaluation_case_id": "M3-C2A_caseP_100nm_common-P1",
    "output_slug": "caseP_100nm",
    "final_registration_kind": "m3c2_caseP_100nm_final_campaign",
    "candidate_case_name_prefix": "m3c2_caseP_100nm",
}


@dataclass(frozen=True, slots=True)
class LockedArtifact:
    role: str
    path: str
    size_bytes: int
    sha256: str
    resolved: Path


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return cast(dict[str, Any], value)


def _list(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return cast(list[Any], value)


def _load_json(path: Path, name: str) -> dict[str, Any]:
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), name)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {name}: {path}") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _load_contract(path: Path) -> dict[str, Any]:
    contract = _load_json(path, "execution contract")
    scope = _mapping(contract.get("scope"), "scope")
    workflow = str(scope.get("workflow"))
    identity = (
        contract.get("schema_version"),
        contract.get("contract_id"),
        contract.get("contract_revision"),
        contract.get("classification"),
    )
    expected_id = {
        "caseA": "M3-C2A-caseA-100nm-stochastic-pilot",
        "caseP": "M3-C2A-caseP-100nm-stochastic-pilot",
    }.get(workflow)
    expected = (1, expected_id, 1, "external_vv_execution_contract_not_run")
    if identity != expected:
        raise ValueError("unexpected M3-C2A execution-contract identity")
    _validate_campaign_identity(contract, workflow)
    authorization = _mapping(contract.get("execution_authorization"), "authorization")
    if authorization.get("status") != "NOT_AUTHORIZED":
        raise ValueError("contract must remain NOT_AUTHORIZED")
    if authorization.get("solver_accuracy_claim") != "NOT_EVALUATED":
        raise ValueError("contract must not make a solver-accuracy claim")
    return contract


def _validate_campaign_identity(contract: dict[str, Any], workflow: str) -> None:
    raw = contract.get("campaign")
    if workflow == "caseA" and raw is None:
        return
    campaign = _mapping(raw, "campaign")
    if set(campaign) != CAMPAIGN_KEYS:
        raise ValueError("campaign must contain exactly the five registered identity keys")
    if workflow != "caseP" or campaign != CASE_P_CAMPAIGN:
        raise ValueError("campaign identity differs from the supported Case-P registration")


def _lock_artifacts(
    contract: dict[str, Any], repository_root: Path
) -> tuple[dict[str, LockedArtifact], list[dict[str, object]]]:
    artifacts: dict[str, LockedArtifact] = {}
    receipt_rows: list[dict[str, object]] = []
    for index, raw in enumerate(_list(contract.get("input_artifacts"), "input_artifacts")):
        record = _mapping(raw, f"input_artifacts[{index}]")
        artifact = LockedArtifact(
            role=str(record.get("role")),
            path=str(record.get("path")),
            size_bytes=int(record.get("size_bytes", -1)),
            sha256=str(record.get("sha256")),
            resolved=(repository_root / str(record.get("path"))).resolve(),
        )
        if artifact.role in artifacts:
            raise ValueError(f"duplicate artifact role: {artifact.role}")
        if not artifact.resolved.is_file():
            raise ValueError(f"locked artifact is missing: {artifact.path}")
        actual_size = artifact.resolved.stat().st_size
        actual_hash = _sha256(artifact.resolved)
        if actual_size != artifact.size_bytes or actual_hash != artifact.sha256:
            raise ValueError(f"locked artifact identity differs: {artifact.role}")
        artifacts[artifact.role] = artifact
        receipt_rows.append(
            {
                "role": artifact.role,
                "path": artifact.path,
                "size_bytes": actual_size,
                "sha256": actual_hash,
                "status": "PASS",
            }
        )
    if set(artifacts) != EXPECTED_ROLES:
        raise ValueError("input_artifacts roles are incomplete or unexpected")
    return artifacts, receipt_rows


def _validate_scope_and_deterministic_contract(
    contract: dict[str, Any], artifacts: dict[str, LockedArtifact]
) -> None:
    deterministic = _load_json(
        artifacts["deterministic_30ms_contract"].resolved, "deterministic 30 ms contract"
    )
    scope = _mapping(contract.get("scope"), "scope")
    matrix = _mapping(deterministic.get("matrix"), "deterministic matrix")
    required_scope = {
        "theory_variant": deterministic.get("theory_variant"),
        "particle_diameter_m": matrix.get("particle_diameter_m"),
        "particle_count": matrix.get("particle_count"),
        "time_end_s": matrix.get("time_end_s"),
        "output_count": matrix.get("output_count"),
        "output_schedule_segments": matrix.get("output_schedule_segments"),
    }
    for name, expected in required_scope.items():
        if scope.get(name) != expected:
            raise ValueError(f"scope.{name} differs from the deterministic contract")
    workflow = str(scope.get("workflow"))
    workflows = _mapping(deterministic.get("workflows"), "deterministic workflows")
    if workflow not in {"caseA", "caseP"} or workflow not in workflows:
        raise ValueError("scope workflow is not supported by the deterministic contract")
    if scope.get("coordinate_system") != "axisymmetric_rz_no_swirl":
        raise ValueError("scope must be axisymmetric R-Z without swirl")

    physics = _mapping(deterministic.get("physics"), "deterministic physics")
    locked = _mapping(contract.get("deterministic_physics"), "deterministic_physics")
    revision_keys = (
        "drag_revision",
        "electric_revision",
        "ion_drag_revision",
        "thermophoresis_revision",
        "dielectrophoresis_revision",
        "lift_revision",
        "gravity_buoyancy_revision",
    )
    if any(locked.get(name) != physics.get(name) for name in revision_keys):
        raise ValueError("one or more deterministic physics revisions differ")
    charge = _mapping(locked.get("dynamic_charge"), "dynamic_charge")
    if not charge.get("active") or charge.get("revision") != physics.get("charge_revision"):
        raise ValueError("dynamic-charge revision is not locked")
    if locked.get("saffman_active") != physics.get("saffman_active"):
        raise ValueError("Saffman activation differs from the deterministic contract")

    expected_boundary = dict(_mapping(deterministic.get("boundaries"), "boundaries"))
    expected_boundary["axis"] = "coordinate_crossing_not_material_event"
    if contract.get("boundary") != expected_boundary:
        raise ValueError("boundary semantics differ from the deterministic contract")
    _validate_comsol_configuration(contract, artifacts, workflow)


def _validate_comsol_configuration(
    contract: dict[str, Any], artifacts: dict[str, LockedArtifact], workflow: str
) -> None:
    if workflow == "caseA":
        return
    configuration = _load_json(
        artifacts["common_p1_configuration"].resolved, "common-P1 COMSOL configuration"
    )
    configured = _mapping(
        _mapping(configuration.get("workflows"), "COMSOL workflows").get(workflow),
        f"COMSOL workflow {workflow}",
    )
    stochastic = _mapping(contract.get("stochastic_physics"), "stochastic_physics")
    comsol = _mapping(stochastic.get("comsol"), "COMSOL stochastic physics")
    expected = (
        configured.get("physics_tag"),
        configured.get("background_study"),
        configured.get("background_solution"),
        configured.get("source_dataset"),
        configured.get("particle_geometry"),
        configured.get("position_dofs"),
        configured.get("charge_state"),
    )
    observed = (
        comsol.get("physics_tag"),
        comsol.get("background_study"),
        comsol.get("background_solution"),
        comsol.get("particle_dataset"),
        comsol.get("particle_geometry"),
        comsol.get("position_expressions"),
        comsol.get("charge_state_expression"),
    )
    if observed != expected:
        raise ValueError("COMSOL workflow semantics differ from the deterministic inventory")
    velocity = _list(comsol.get("velocity_expressions"), "velocity expressions")
    expected_velocity = [f"{configured['physics_tag']}.vr", f"{configured['physics_tag']}.vz"]
    if velocity != expected_velocity:
        raise ValueError("COMSOL velocity expressions differ from the particle physics")
    case_p_runner = (
        comsol.get("background_study_step"),
        comsol.get("shared_variable_tag"),
        comsol.get("temperature_expression"),
        comsol.get("pressure_expression"),
        comsol.get("seed_parameter"),
    )
    expected_runner = (
        "ftper",
        "varDustField",
        "m3c1_Tg(r,z)",
        "m3c1_rhog(r,z)*k_B_const*m3c1_Tg(r,z)/1.2753471408396638e-25[kg]",
        "P_brownian_seed",
    )
    if case_p_runner != expected_runner:
        raise ValueError("Case-P common-P1 Brownian runner expressions differ")


def _validate_common_p1_input(
    contract: dict[str, Any], artifacts: dict[str, LockedArtifact]
) -> dict[str, object]:
    locked = _mapping(contract.get("common_p1_input"), "common_p1_input")
    artifact = artifacts["common_p1_input"]
    if locked.get("path") != artifact.path or locked.get("file_sha256") != artifact.sha256:
        raise ValueError("common-P1 file identity is inconsistent inside the contract")
    data, info = read_with_info(artifact.resolved)
    if info.content_hash != locked.get("content_hash"):
        raise ValueError("common-P1 logical content hash differs")
    if str(data.coordinate_system) != "axisymmetric_rz":
        raise ValueError("common-P1 input is not axisymmetric R-Z")
    if len(data.sources) != 1:
        raise ValueError("common-P1 input must contain exactly one source")
    source = data.sources[0]
    release = _mapping(locked.get("release"), "release")
    release_facts = (
        source.name,
        len(source.particle_id),
        int(source.particle_id.min()),
        int(source.particle_id.max()),
        float(source.release_time_s.min()),
        float(source.release_time_s.max()),
    )
    expected_release = (
        release.get("source_name"),
        287,
        release.get("particle_id_min"),
        release.get("particle_id_max"),
        release.get("release_time_s"),
        release.get("release_time_s"),
    )
    if release_facts != expected_release:
        raise ValueError("common-P1 release table differs from the contract")
    geometry = _mapping(locked.get("geometry"), "geometry")
    if data.geometry.tri3 is None:
        raise ValueError("common-P1 geometry has no triangles")
    geometry_facts = (
        data.geometry.nodes_m.shape[0],
        data.geometry.tri3.shape[0],
        data.geometry.boundary.line2.shape[0],
    )
    expected_geometry = (
        geometry.get("nodes"),
        geometry.get("triangles"),
        geometry.get("boundary_lines"),
    )
    if geometry_facts != expected_geometry:
        raise ValueError("common-P1 geometry shape differs from the contract")
    evidence = _load_json(artifacts["common_p1_evidence"].resolved, "common-P1 evidence")
    workflow = str(_mapping(contract.get("scope"), "scope").get("workflow"))
    if workflow == "caseA":
        _validate_case_a_common_p1_evidence(evidence, artifact, info.content_hash)
    else:
        _validate_case_p_common_p1_preparation(evidence, artifact, info.content_hash)
    return {
        "file_sha256": artifact.sha256,
        "content_hash": info.content_hash,
        "particle_count": len(source.particle_id),
        "release_time_s": float(source.release_time_s[0]),
        "nodes": geometry_facts[0],
        "triangles": geometry_facts[1],
        "boundary_lines": geometry_facts[2],
    }


def _validate_case_a_common_p1_evidence(
    evidence: dict[str, Any], artifact: LockedArtifact, content_hash: str
) -> None:
    canonical = _mapping(
        _mapping(
            _mapping(evidence.get("source_artifacts"), "source_artifacts").get("candidate"),
            "candidate evidence",
        ).get("canonical_input"),
        "canonical input evidence",
    )
    if canonical.get("file_sha256") != artifact.sha256:
        raise ValueError("common-P1 evidence file hash differs")
    if canonical.get("recomputed_content_hash") != content_hash:
        raise ValueError("common-P1 evidence content hash differs")
    if evidence.get("overall_decision") != (
        "PASS_LOCKED_SAME_FIELD_COMMON_CANONICAL_P1_CASE_AND_WINDOW"
    ):
        raise ValueError("common-P1 prerequisite evidence did not pass")


def _validate_case_p_common_p1_preparation(
    evidence: dict[str, Any], artifact: LockedArtifact, content_hash: str
) -> None:
    workflow = _mapping(
        _mapping(evidence.get("workflows"), "prepared workflows").get("caseP"),
        "prepared Case-P workflow",
    )
    source = _mapping(
        _mapping(workflow.get("preparation_receipt"), "preparation receipt").get("source"),
        "prepared source receipt",
    )
    identity = (
        evidence.get("status"),
        workflow.get("status"),
        workflow.get("input_sha256"),
        workflow.get("input_content_hash"),
        workflow.get("output_count"),
        source.get("particle_count"),
        source.get("particle_id_min"),
        source.get("particle_id_max"),
    )
    expected = (
        "PREPARED",
        "PREPARED",
        artifact.sha256,
        content_hash,
        121,
        287,
        1,
        287,
    )
    if identity != expected:
        raise ValueError("Case-P common-P1 preparation evidence differs")


def _validate_stochastic_semantics(
    contract: dict[str, Any], artifacts: dict[str, LockedArtifact]
) -> dict[str, object]:
    preflight = _load_json(artifacts["m3c2_preflight_evidence"].resolved, "M3-C2 preflight")
    if preflight.get("overall_status") != "BLOCKED_MISSING_MEANING_MATCHED_COHORT":
        raise ValueError("M3-C2 preflight status differs")
    if preflight.get("comparison_decision") != "NOT_AUTHORIZED":
        raise ValueError("M3-C2 preflight unexpectedly authorizes comparison")
    requirements = _mapping(
        preflight.get("future_companion_requirements"), "future companion requirements"
    )
    stochastic = _mapping(contract.get("stochastic_physics"), "stochastic_physics")
    comsol = _mapping(stochastic.get("comsol"), "COMSOL stochastic physics")
    candidate = _mapping(stochastic.get("candidate"), "candidate stochastic physics")
    if requirements.get("comsol_out_of_plane_degrees_of_freedom") is not False:
        raise ValueError("preflight does not require R-Z-only COMSOL motion")
    expected = (
        stochastic.get("degrees_of_freedom"),
        comsol.get("out_of_plane_degrees_of_freedom"),
        comsol.get("feature_kind"),
        comsol.get("viscosity_meaning"),
        comsol.get("random_number_args"),
        candidate.get("brownian_revision"),
    )
    required = (
        ["r", "z"],
        False,
        "built_in_brownian_force",
        requirements.get("comsol_brownian_viscosity"),
        requirements.get("comsol_random_number_mode"),
        requirements.get("candidate_brownian_revision"),
    )
    if expected != required:
        raise ValueError("stochastic target semantics differ from the preflight")
    expected_seed_authority = f"{comsol.get('physics_tag')}.{comsol.get('feature_tag')}.i"
    if comsol.get("sole_seed_authority") != expected_seed_authority:
        raise ValueError("COMSOL bf1.i must be the sole seed authority")
    return _validate_source_model_semantics(contract, artifacts)


def _validate_source_model_semantics(
    contract: dict[str, Any], artifacts: dict[str, LockedArtifact]
) -> dict[str, object]:
    probe = _load_json(artifacts["m3c2_model_semantics_probe"].resolved, "model probe")
    if not probe.get("load_copy") or probe.get("study_run") or probe.get("model_save"):
        raise ValueError("model-semantics probe did not use read-only loadCopy")
    source = _mapping(contract.get("source_model"), "source_model")
    source_artifact = artifacts["source_mph"]
    source_policy = (
        source.get("path"),
        source.get("sha256"),
        source.get("load_policy"),
        source.get("save_policy"),
        source.get("study_execution_in_contract_lock"),
    )
    expected_source_policy = (
        source_artifact.path,
        source_artifact.sha256,
        "ModelUtil.loadCopy",
        "never_save_source_or_loaded_copy",
        False,
    )
    if source_policy != expected_source_policy:
        raise ValueError("source MPH loadCopy/no-save policy differs")
    if str(probe.get("source_mph_sha256")).lower() != source.get("sha256"):
        raise ValueError("model-semantics probe source hash differs")
    interfaces = [
        _mapping(value, "probe interface")
        for value in _list(probe.get("interfaces"), "probe interfaces")
    ]
    workflow = str(_mapping(contract.get("scope"), "scope").get("workflow"))
    selected = [row for row in interfaces if row.get("case") == f"{workflow}_100nm"]
    if len(selected) != 1:
        raise ValueError("model-semantics probe must contain one selected-case record")
    observed = selected[0]
    stochastic = _mapping(contract.get("stochastic_physics"), "stochastic_physics")
    comsol = _mapping(stochastic.get("comsol"), "COMSOL stochastic physics")
    legacy = {
        "physics_tag": "fptas",
        "feature_tag": "bf1",
        "viscosity_expression": "root.comp1.AS_muB",
        "particle_dataset": "part_AS_100nm",
        "position_expressions": ["q3r", "q3z"],
    }
    source_values = comsol if workflow == "caseP" else legacy
    position_expressions = [
        str(value)
        for value in _list(source_values.get("position_expressions"), "position expressions")
    ]
    source_semantics = (
        observed.get("physics_tag"),
        observed.get("include_out_of_plane"),
        observed.get("brownian_feature"),
        observed.get("brownian_mu"),
        observed.get("random_number_args"),
        observed.get("brownian_i"),
        observed.get("particle_dataset"),
        observed.get("particle_position_dofs"),
    )
    expected_source = (
        source_values.get("physics_tag"),
        "0",
        source_values.get("feature_tag"),
        source_values.get("viscosity_expression"),
        "GenerateUnique",
        "brownian_seed" if workflow == "caseP" else "AS_brownian_seed",
        source_values.get("particle_dataset"),
        f"[comp1.{position_expressions[0]}, comp1.{position_expressions[1]}]",
    )
    if source_semantics != expected_source:
        raise ValueError("source MPH stochastic semantics differ from the read-only probe")
    return {
        "source_out_of_plane": False,
        "source_random_number_args": "GenerateUnique",
        "required_runner_random_number_args": "UserDefined",
        "required_seed_authority": comsol.get("sole_seed_authority"),
        "brownian_feature": comsol.get("feature_tag"),
        "brownian_viscosity_expression": comsol.get("viscosity_expression"),
        "source_particle_dataset": observed.get("particle_dataset"),
        "source_position_dofs": observed.get("particle_position_dofs"),
    }


def _legacy_final_seed_labels(
    final: dict[str, Any], artifact: LockedArtifact
) -> tuple[set[int], int, bool, set[int]]:
    if final.get("path") != artifact.path:
        raise ValueError("final-cohort path differs from the locked artifact")
    rows = _read_csv(artifact.resolved)
    rows = [row for row in rows if row.get("case_id") == final.get("case_id")]
    participants = {row["participant"] for row in rows}
    if participants != {"comsol_common_p1_rz_companion", "solver_b03_candidate"}:
        raise ValueError("final cohort participants differ")
    expected_count = int(final.get("replicas_per_participant", -1))
    for participant in participants:
        selected = [row for row in rows if row["participant"] == participant]
        if len(selected) != expected_count:
            raise ValueError(f"final cohort count differs for {participant}")
        if {row["execution_status"] for row in selected} != {final.get("execution_status")}:
            raise ValueError(f"final cohort status differs for {participant}")
    return {int(row["campaign_seed"]) for row in rows}, expected_count, False, set()


def _json_pointer(document: object, pointer: str) -> object:
    if not pointer.startswith("/"):
        raise ValueError("JSON pointer must be absolute")
    current = document
    for raw in pointer[1:].split("/"):
        key = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and key in current:
            current = current[key]
        elif isinstance(current, list) and key.isdigit() and int(key) < len(current):
            current = current[int(key)]
        else:
            raise ValueError(f"JSON pointer does not resolve: {pointer}")
    return current


def _seed_values(document: object, pointer: str, name: str) -> list[int]:
    return [int(value) for value in _list(_json_pointer(document, pointer), name)]


def _audited_existing_seeds(allocation: dict[str, Any], repository_root: Path) -> set[int]:
    audit = _mapping(allocation.get("seed_audit"), "seed audit")
    if audit.get("completed_before_allocation") is not True:
        raise ValueError("final seed audit must be completed before allocation")
    existing: set[int] = set()
    for index, raw in enumerate(_list(audit.get("authorities"), "seed authorities")):
        authority = _mapping(raw, f"seed authorities[{index}]")
        path = (repository_root / str(authority.get("path"))).resolve()
        if not path.is_file() or _sha256(path) != authority.get("sha256"):
            raise ValueError("seed-audit authority identity differs")
        if path.suffix.lower() == ".csv":
            rows = _read_csv(path)
            existing.update(int(row["seed"]) for row in rows)
            continue
        document = _load_json(path, "seed-audit authority")
        for pointer in _list(authority.get("json_pointers"), "seed authority pointers"):
            existing.update(_seed_values(document, str(pointer), "audited seed values"))
    return existing


def _validate_final_allocation_identity(final: dict[str, Any], allocation: dict[str, Any]) -> int:
    identity = (
        allocation.get("schema_version"),
        allocation.get("allocation_kind"),
        allocation.get("case_id"),
        allocation.get("purpose"),
        allocation.get("execution_status"),
        allocation.get("authorization"),
    )
    expected = (
        1,
        "m3c2_final_seed_allocation",
        final.get("case_id"),
        "final",
        final.get("execution_status"),
        "NOT_AUTHORIZED_PENDING_ACCEPTED_PILOT",
    )
    if identity != expected:
        raise ValueError("final seed allocation identity differs")
    expected_count = int(final.get("replicas_per_participant", -1))
    if allocation.get("replicas_per_participant") != expected_count:
        raise ValueError("final seed allocation replica count differs")
    return expected_count


def _participant_final_seed_sets(
    allocation: dict[str, Any], pointers: dict[str, Any], expected_count: int
) -> dict[str, list[int]]:
    participant_sets = {
        participant: _seed_values(allocation, str(pointer), f"{participant} final seeds")
        for participant, pointer in pointers.items()
    }
    counts_are_valid = all(
        len(values) == expected_count and len(set(values)) == expected_count
        for values in participant_sets.values()
    )
    if not counts_are_valid:
        raise ValueError("final participant seed count or uniqueness differs")
    if set(participant_sets["comsol"]) & set(participant_sets["candidate"]):
        raise ValueError("final participant seed sets overlap")
    return participant_sets


def _validate_final_seed_audit(
    allocation: dict[str, Any],
    participant_sets: dict[str, list[int]],
    final_seeds: set[int],
    repository_root: Path,
) -> set[int]:
    existing = _audited_existing_seeds(allocation, repository_root)
    if final_seeds & existing:
        raise ValueError("final seeds collide with an audited existing campaign seed")
    audit = _mapping(allocation.get("seed_audit"), "seed audit")
    selected = _mapping(audit.get("selected_seed_ranges"), "selected seed ranges")
    reported_ranges = {
        participant: [min(values), max(values)] for participant, values in participant_sets.items()
    }
    if selected != reported_ranges or audit.get("collision_count") != 0:
        raise ValueError("final seed-audit summary differs from the allocation")
    if audit.get("participant_sets_disjoint") is not True:
        raise ValueError("final seed audit did not certify participant isolation")
    return existing


def _referenced_final_seed_labels(
    final: dict[str, Any], artifact: LockedArtifact, repository_root: Path
) -> tuple[set[int], int, bool, set[int]]:
    source = _mapping(final.get("seed_source"), "final seed source")
    if source.get("path") != artifact.path or source.get("sha256") != artifact.sha256:
        raise ValueError("final seed source differs from the locked artifact")
    pointers = _mapping(source.get("json_pointers"), "final seed pointers")
    if set(pointers) != {"comsol", "candidate"}:
        raise ValueError("final seed pointers must contain exactly both participants")
    allocation = _load_json(artifact.resolved, "final seed allocation")
    expected_count = _validate_final_allocation_identity(final, allocation)
    participant_sets = _participant_final_seed_sets(allocation, pointers, expected_count)
    comsol = set(participant_sets["comsol"])
    candidate = set(participant_sets["candidate"])
    final_seeds = comsol | candidate
    existing = _validate_final_seed_audit(
        allocation, participant_sets, final_seeds, repository_root
    )
    return final_seeds, expected_count, True, existing


def _final_seed_labels(
    final: dict[str, Any], artifact: LockedArtifact, repository_root: Path
) -> tuple[set[int], int, bool, set[int]]:
    if "seed_source" in final:
        return _referenced_final_seed_labels(final, artifact, repository_root)
    return _legacy_final_seed_labels(final, artifact)


def _pilot_seed_sets(pilot: dict[str, Any], final_seeds: set[int]) -> tuple[list[int], list[int]]:
    comsol_seeds = [int(value) for value in _list(pilot.get("comsol_seeds"), "comsol seeds")]
    candidate_seeds = [
        int(value) for value in _list(pilot.get("candidate_seeds"), "candidate seeds")
    ]
    if not comsol_seeds or not candidate_seeds:
        raise ValueError("both pilot seed sets must be nonempty")
    if len(set(comsol_seeds)) != len(comsol_seeds):
        raise ValueError("COMSOL pilot seeds are not unique")
    if len(set(candidate_seeds)) != len(candidate_seeds):
        raise ValueError("candidate pilot seeds are not unique")
    if set(comsol_seeds) & set(candidate_seeds):
        raise ValueError("pilot participant seed sets overlap")
    if (set(comsol_seeds) | set(candidate_seeds)) & final_seeds:
        raise ValueError("pilot and final-cohort seeds overlap")
    if pilot.get("execution_status") != "NOT_AUTHORIZED":
        raise ValueError("pilot seed plan must remain NOT_AUTHORIZED")
    return comsol_seeds, candidate_seeds


def _validate_seed_isolation(
    contract: dict[str, Any], artifacts: dict[str, LockedArtifact], repository_root: Path
) -> dict[str, object]:
    plan = _mapping(contract.get("seed_plan"), "seed_plan")
    final = _mapping(plan.get("final_cohort"), "final_cohort")
    final_seeds, expected_count, final_participants_disjoint, audited_existing = _final_seed_labels(
        final, artifacts["m3c2_final_seed_allocation"], repository_root
    )
    pilot = _mapping(plan.get("pilot"), "pilot")
    comsol_seeds, candidate_seeds = _pilot_seed_sets(pilot, final_seeds)
    if (set(comsol_seeds) | set(candidate_seeds)) & audited_existing:
        raise ValueError("pilot seeds collide with an audited existing campaign seed")
    return {
        "final_replicas_per_participant": expected_count,
        "final_unique_seed_labels": len(final_seeds),
        "final_participant_seed_sets_disjoint": final_participants_disjoint,
        "comsol_pilot_seeds": comsol_seeds,
        "candidate_pilot_seeds": candidate_seeds,
        "all_pilot_seeds_disjoint_from_final": True,
        "all_pilot_seeds_disjoint_from_audited_existing": True,
        "participant_pilot_seed_sets_disjoint": True,
    }


def _validate_case_p_applicability(contract: dict[str, Any]) -> dict[str, Any] | None:
    workflow = str(_mapping(contract.get("scope"), "scope").get("workflow"))
    if workflow == "caseA":
        return None
    applicability = _mapping(contract.get("physical_applicability"), "physical applicability")
    expected = {
        "status": "NOT_CERTIFIED_NEGATIVE_ION_CURRENT_OMITTED",
        "caseP_contains_negative_ions": True,
        "locked_charge_model_includes_negative_ion_current": False,
        "permitted_interpretation": "same_form_numerical_sensitivity_comparison_only",
        "forbidden_interpretation": "physical_validation_of_caseP_charging_or_trajectory_truth",
    }
    if applicability != expected:
        raise ValueError("Case-P physical-applicability limitation differs")
    claims = _mapping(contract.get("claim_policy"), "claim policy")
    if (
        claims.get("physical_applicability") != applicability["status"]
        or claims.get("comparison_scope") != "locked_same_form_numerical_sensitivity_only"
    ):
        raise ValueError("Case-P claim policy exceeds the locked model applicability")
    return applicability


def _validate_numerical_policy(contract: dict[str, Any]) -> dict[str, Any]:
    numerical = _mapping(contract.get("numerical_policy"), "numerical_policy")
    if numerical.get("step_policy") != (
        "independent_solver_observable_convergence_not_equal_fixed_step"
    ):
        raise ValueError("both solvers must use independent observable convergence")
    if numerical.get("comparison_kind") != "independent_seed_ensemble_observables_not_pathwise":
        raise ValueError("comparison must be an independent-seed ensemble")
    if numerical.get("pathwise_equality_claim") is not False:
        raise ValueError("pathwise equality must not be claimed")
    workflow = str(_mapping(contract.get("scope"), "scope").get("workflow"))
    if workflow == "caseP" and numerical.get("comsol_pilot_fixed_steps_s") != [
        2e-5,
        1e-5,
        5e-6,
    ]:
        raise ValueError("Case-P COMSOL pilot step series differs")
    return numerical


def lock_contract(config_path: Path, repository_root: Path, output: Path) -> dict[str, object]:
    contract = _load_contract(config_path)
    artifacts, artifact_rows = _lock_artifacts(contract, repository_root)
    _validate_scope_and_deterministic_contract(contract, artifacts)
    common_input = _validate_common_p1_input(contract, artifacts)
    stochastic = _validate_stochastic_semantics(contract, artifacts)
    seeds = _validate_seed_isolation(contract, artifacts, repository_root)
    numerical = _validate_numerical_policy(contract)
    applicability = _validate_case_p_applicability(contract)
    workflow = str(_mapping(contract.get("scope"), "scope").get("workflow"))

    gates: list[dict[str, str]] = [
        {"gate": "input_artifact_identity", "status": "PASS"},
        {"gate": "common_p1_content_release_geometry", "status": "PASS"},
        {"gate": "deterministic_physics_and_boundary", "status": "PASS"},
        {"gate": "rz_brownian_target_semantics", "status": "PASS"},
        {"gate": "pilot_seed_isolation", "status": "PASS"},
    ]
    if applicability is not None:
        gates.append({"gate": "physical_applicability_scope", "status": "PASS"})
    gates.append({"gate": "runner_recipes_and_pilot", "status": "BLOCKED_NOT_LOCKED_OR_RUN"})
    receipt: dict[str, object] = {
        "schema_version": 1,
        "evidence_id": f"M3-C2A-{workflow}-100nm-pilot-contract-receipt-v1",
        "tool_revision": TOOL_REVISION,
        "contract_sha256": _sha256(config_path),
        "tool_sha256": _sha256(Path(__file__).resolve()),
        "contract_validation_status": "PASS_INPUT_IDENTITY_AND_SEMANTICS_LOCKED",
        "execution_status": "NOT_AUTHORIZED",
        "pilot_status": "NOT_RUN",
        "solver_accuracy_claim": "NOT_EVALUATED",
        "comparison_decision": "NOT_EVALUATED",
        "solver_core_changed": False,
        "comsol_or_candidate_executed": False,
        "common_p1_input": common_input,
        "stochastic_semantics": stochastic,
        "seed_isolation": seeds,
        "numerical_policy": numerical,
        "remaining_blockers": _mapping(
            contract.get("execution_authorization"), "execution_authorization"
        ).get("required_before_execution"),
        "gates": gates,
    }
    if applicability is not None:
        receipt["campaign"] = _mapping(contract.get("campaign"), "campaign")
        receipt["physical_applicability"] = applicability
    output.mkdir(parents=True, exist_ok=False)
    with (output / "artifact_hashes.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("role", "path", "size_bytes", "sha256", "status"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(artifact_rows)
    (output / "contract_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(
        f"# M3-C2A {workflow} 100 nm pilot execution contract\n\n"
        "Contract validation: `PASS_INPUT_IDENTITY_AND_SEMANTICS_LOCKED`.\n\n"
        "Execution: `NOT_AUTHORIZED`; pilot: `NOT_RUN`; solver accuracy: "
        "`NOT_EVALUATED`. This receipt only binds existing source/common-P1 identity, "
        "physics and boundary meaning, R-Z Brownian intent, output observations, and "
        "pilot/final seed separation. No COMSOL or candidate calculation was run. "
        "Executable no-save runner recipes and the observable-convergence pilot must "
        "be locked and accepted before execution can be authorized.\n"
        + (
            "\nPhysical applicability remains "
            "`NOT_CERTIFIED_NEGATIVE_ION_CURRENT_OMITTED`; this contract permits only "
            "a same-form numerical sensitivity comparison.\n"
            if workflow == "caseP"
            else ""
        ),
        encoding="utf-8",
    )
    return receipt


def _default_repository_root() -> Path:
    return Path(__file__).resolve().parents[5]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repository-root", type=Path, default=_default_repository_root())
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    receipt = lock_contract(
        args.config.resolve(), args.repository_root.resolve(), args.output.resolve()
    )
    print(
        json.dumps(
            {
                "contract_validation_status": receipt["contract_validation_status"],
                "execution_status": receipt["execution_status"],
                "output": str(args.output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
