"""Focused fail-closed checks for Case-P RNG and final tolerance bindings."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from tools.vv.comsol import evaluate_m3c2_stochastic_ensemble as evaluator

CANDIDATE_EVENT_COLUMNS = (
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


def _solver_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


@pytest.mark.parametrize(
    "current_runner", ["m3c2_candidate_campaign_runner_v5", "m3c2_candidate_campaign_runner_v6"]
)
def test_casep_candidate_manifest_uses_current_campaign_identity_schema(
    current_runner: str,
) -> None:
    root = _solver_root()
    manifest = _json(
        root / "_out_m3c2" / "caseP_100nm_candidate_pilot_v2" / "candidate_pilot_manifest.json"
    )

    assert (
        evaluator._casep_candidate_participant(manifest, "pilot")
        == "m3c2_candidate_campaign_runner_v4"
    )

    changed = copy.deepcopy(manifest)
    changed["campaign_identity"]["evaluation_case_id"] = "another-case"
    with pytest.raises(ValueError, match="identity differs"):
        evaluator._casep_candidate_participant(changed, "pilot")

    current = copy.deepcopy(manifest)
    current["tool_revision"] = current_runner
    current["brownian_noise_revision"] = evaluator.CASEP_CURRENT_NOISE_REVISION
    replicas = [replica for level in current["levels"].values() for replica in level["replicas"]]
    for replica in replicas:
        replica["resolved_physics_models"]["noise"]["revision"] = (
            evaluator.CASEP_CURRENT_NOISE_REVISION
        )
    assert evaluator._casep_candidate_participant(current, "pilot") == current_runner

    mismatched = copy.deepcopy(manifest)
    mismatched_replicas = [
        replica for level in mismatched["levels"].values() for replica in level["replicas"]
    ]
    for replica in mismatched_replicas:
        replica["resolved_physics_models"]["noise"]["revision"] = (
            evaluator.CASEP_CURRENT_NOISE_REVISION
        )
    with pytest.raises(ValueError, match="runner and Brownian model revisions differ"):
        evaluator._casep_candidate_participant(mismatched, "pilot")

    replicas[0]["resolved_physics_models"]["noise"]["revision"] = (
        evaluator.CASEP_HISTORICAL_NOISE_REVISION
    )
    with pytest.raises(ValueError, match="mixes Brownian model revisions"):
        evaluator._casep_candidate_participant(current, "pilot")


def test_changed_evaluator_uses_v6_and_rejects_unknown_runner() -> None:
    policy = cast(evaluator.Policy, SimpleNamespace(revision=5))
    historical = cast(
        evaluator.Campaign,
        SimpleNamespace(candidate_tool_revision="m3c2_candidate_campaign_runner_v4"),
    )
    current = cast(
        evaluator.Campaign,
        SimpleNamespace(candidate_tool_revision="m3c2_candidate_campaign_runner_v5"),
    )

    assert evaluator.HISTORICAL_TOOL_REVISION_V5 == "m3c2_stochastic_ensemble_evaluator_v5"
    assert evaluator._tool_revision(policy, historical) == evaluator.TOOL_REVISION_V6
    assert evaluator._tool_revision(policy, current) == evaluator.TOOL_REVISION_V6

    unknown = cast(evaluator.Campaign, SimpleNamespace(candidate_tool_revision=None))
    with pytest.raises(ValueError, match="recognized candidate runner revision"):
        evaluator._tool_revision(policy, unknown)


def test_casep_v5_projection_has_one_strict_policy_only_transition() -> None:
    root = _solver_root()
    campaign_path = (
        root / "_out_m3c2" / "caseP_100nm_pilot_campaign_v5_projection_v1" / "campaign.json"
    )
    campaign = _json(campaign_path)
    source_policy_sha256 = evaluator._participant_projection_policy_sha256(
        campaign, campaign_path.parent
    )

    assert source_policy_sha256 == (
        "6388084a526053144550a5914a8000cae3a6f1a2899d784cc1cb875a30050eae"
    )

    changed = copy.deepcopy(campaign)
    changed["policy_projection"]["changed_fields"] = ["participants"]
    with pytest.raises(ValueError, match="projection declaration differs"):
        evaluator._participant_projection_policy_sha256(changed, campaign_path.parent)


def _final_campaign(
    tmp_path: Path,
    *,
    event_rows: int,
    event_columns: tuple[str, ...] = CANDIDATE_EVENT_COLUMNS,
) -> Any:
    event_path = tmp_path / "events.csv"
    rows = [",".join(event_columns)]
    event = {
        "particle_id": "1",
        "event_ordinal": "1",
        "event_time_s": "0.01",
        "hit_r_m": "0",
        "hit_z_m": "0",
        "normal_r": "1",
        "normal_z": "0",
        "pre_velocity_r_m_per_s": "0",
        "pre_velocity_z_m_per_s": "0",
        "post_velocity_r_m_per_s": "0",
        "post_velocity_z_m_per_s": "0",
        "event_type": "wall",
        "boundary_semantic": "wall",
        "law": "freeze",
        "outcome": "stuck",
    }
    rows.extend(",".join(event[column] for column in event_columns) for _ in range(event_rows))
    event_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    replica = SimpleNamespace(event_path=event_path)
    return SimpleNamespace(participants={"candidate": (SimpleNamespace(replicas=(replica,)),)})


def _qualification() -> dict[str, object]:
    return {
        "schema_version": 1,
        "qualification_kind": "m3c2_caseP_geometry_rtol_pre_final",
        "tool_revision": "m3c2_caseP_event_tolerance_pre_final_evaluator_v2",
        "status": "PASS",
        "classification": "PASS_NO_EVENT_OPERATIONAL_BRIDGE",
        "validity_condition": "zero_candidate_boundary_events",
        "seed": 919008,
        "selected_setting": {"dt_s": 2.0e-5, "brownian_interval_tree_depth": 3},
        "reference": {"geometry_rtol": 1.0e-8},
        "candidate": {"geometry_rtol": 1.0e-9},
    }


def _selected() -> dict[str, dict[str, object]]:
    return {
        "candidate": {
            "numerical_setting": {
                "dt_s": 2.0e-5,
                "brownian_interval_tree_depth": 3,
                "geometry_rtol": 1.0e-8,
                "purpose": "accepted_final",
            }
        }
    }


def test_zero_event_qualification_rejects_eventful_candidate_final(tmp_path: Path) -> None:
    assert (
        evaluator._validate_casep_final_qualification(
            _qualification(), _selected(), _final_campaign(tmp_path, event_rows=0)
        )
        == 0
    )

    with pytest.raises(ValueError, match="zero-event geometry-tolerance qualification"):
        evaluator._validate_casep_final_qualification(
            _qualification(), _selected(), _final_campaign(tmp_path, event_rows=1)
        )


def test_zero_event_qualification_rejects_incomplete_candidate_header(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="columns are incomplete"):
        evaluator._validate_casep_final_qualification(
            _qualification(),
            _selected(),
            _final_campaign(
                tmp_path,
                event_rows=0,
                event_columns=tuple(
                    column for column in CANDIDATE_EVENT_COLUMNS if column != "outcome"
                ),
            ),
        )


def test_casep_final_receipt_hash_locks_rng_rtol_and_seed_evidence(tmp_path: Path) -> None:
    root = _solver_root()
    receipt_path = (
        root / "evidence" / "m3c2" / "caseP_100nm_final_campaign_v1" / "selection_receipt.json"
    )
    receipt = _json(receipt_path)
    allocation_path = (
        root
        / "tools"
        / "vv"
        / "comsol"
        / "cases"
        / "m3c2_caseP_100nm_final_seed_allocation_v1.json"
    )
    allocation = _json(allocation_path)
    seed_plan = cast(dict[str, list[int]], allocation["participant_seed_sets"])
    evidence_path = root / "evidence" / "m3c2" / "rng_noninteraction_v2" / "evidence_manifest.json"
    policy = cast(
        evaluator.Policy,
        SimpleNamespace(
            evidence_manifest_path=evidence_path.resolve(),
            evidence_manifest_sha256=receipt["rng_noninteraction_evidence"]["sha256"],
        ),
    )

    report = evaluator._validate_casep_final_authorization_evidence(
        receipt,
        receipt_path.parent,
        _final_campaign(tmp_path, event_rows=0),
        policy,
        _selected(),
        seed_plan,
    )

    assert report["candidate_final_boundary_event_rows"] == 0
    assert report["physical_applicability"] == "NOT_CERTIFIED_NEGATIVE_ION_CURRENT_OMITTED"

    changed = copy.deepcopy(receipt)
    changed["geometry_tolerance_qualification"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 differs"):
        evaluator._validate_casep_final_authorization_evidence(
            changed,
            receipt_path.parent,
            _final_campaign(tmp_path, event_rows=0),
            policy,
            _selected(),
            seed_plan,
        )
