"""Focused checks for the M3-C2 participant-to-campaign adapter."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from tools.vv.comsol import assemble_m3c2_campaign as assembler
from tools.vv.comsol import evaluate_m3c2_stochastic_ensemble as evaluator

CASE_P_CAMPAIGN = {
    "case_id": "formal_iondrag_theory_consistent/caseP_100nm",
    "evaluation_case_id": "M3-C2A_caseP_100nm_common-P1",
    "output_slug": "caseP_100nm",
    "final_registration_kind": "m3c2_caseP_100nm_final_campaign",
    "candidate_case_name_prefix": "m3c2_caseP_100nm",
}
CASE_P_POLICY_SHA256 = "e" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _artifact(path: Path, value: object) -> dict[str, str]:
    if isinstance(value, str):
        path.write_text(value, encoding="utf-8")
    else:
        _write_json(path, value)
    return {"path": path.name, "sha256": _sha256(path)}


def _performance(status: str | None) -> dict[str, object]:
    result: dict[str, object] = {
        "wall_time_s": 2.5,
        "peak_rss_bytes": 4096,
        "output_bytes": 1024,
        "particle_count": 287,
        "output_frames": 121,
        "stage_times_s": {},
    }
    if status is not None:
        result["measurement_status"] = status
    if status == "NON_AUTHORITATIVE_EXTERNAL_WORKLOAD_OVERLAP":
        result["non_authoritative_reason"] = "overlapped another measured workload"
    return result


def _replica(
    root: Path,
    participant: str,
    level_id: str,
    seed: int,
    performance_status: str | None,
) -> dict[str, object]:
    stem = f"{participant}_{level_id}_{seed}"
    trajectory = _artifact(root / f"{stem}_trajectory.csv", "trajectory\n")
    events = _artifact(root / f"{stem}_events.csv", "events\n")
    performance = _artifact(root / f"{stem}_performance.json", _performance(performance_status))
    if participant == "comsol":
        return {
            "seed": seed,
            "trajectory": trajectory,
            "trajectory_rows": 287 * 121,
            "events": events,
            "event_count": 0,
            "performance": performance,
        }
    return {
        "status": "COMPLETE",
        "participant": "candidate",
        "seed": seed,
        "trajectory": trajectory["path"],
        "trajectory_sha256": trajectory["sha256"],
        "trajectory_rows": 287 * 121,
        "events": events["path"],
        "events_sha256": events["sha256"],
        "event_rows": 0,
        "performance": performance["path"],
        "performance_sha256": performance["sha256"],
    }


def _inputs(
    root: Path,
    *,
    purpose: str = "pilot",
    performance_status: str | None = "MEASURED_PROCESS_LIFETIME_HIGH_WATER",
    campaign_identity: dict[str, str] | None = None,
) -> tuple[Path, Path, Path]:
    root.mkdir()
    canonical = root / "canonical.h5"
    canonical.write_bytes(b"locked canonical input")
    count = 4 if purpose == "pilot" else 32
    comsol_seeds = list(range(10, 10 + count))
    candidate_seeds = list(range(100, 100 + count))
    comsol_level_ids = ["dt_20us", "dt_10us", "dt_5us"] if purpose == "pilot" else ["dt_5us"]
    comsol_levels = [
        {
            "level_id": level_id,
            "ordinal": ordinal,
            "numerical_setting": {"fixed_step_s": 2.0e-5 / 2**ordinal},
            "replicas": [
                _replica(root, "comsol", level_id, seed, performance_status)
                for seed in comsol_seeds
            ],
        }
        for ordinal, level_id in enumerate(comsol_level_ids)
    ]
    comsol: dict[str, Any] = {
        "schema_version": 1,
        "manifest_kind": "m3c2_participant",
        "tool_revision": assembler.LEGACY_COMSOL_TOOL_REVISION,
        "case_id": assembler.LEGACY_CAMPAIGN_IDENTITY["case_id"],
        "status": "COMPLETE_NORMALIZED_NOT_EVALUATED",
        "participant": "comsol",
        "common_observation_times": list(assembler.OUTPUT_TIMES_S),
        "levels": comsol_levels,
    }
    if purpose == "pilot":
        candidate_level_specs = [
            ("macro_coarse", 2.0e-5, "macro_step_convergence"),
            ("macro_medium", 1.0e-5, "macro_step_convergence"),
            ("macro_fine", 5.0e-6, "macro_step_convergence"),
            ("path_fine", 5.0e-6, assembler.PATH_SENSITIVITY_PURPOSE),
        ]
    else:
        candidate_level_specs = [("accepted", 5.0e-6, "accepted_final")]
    candidate_levels = {
        level_id: {
            "dt_s": dt_s,
            "brownian_interval_tree_depth": 4 if level_id == "path_fine" else 3,
            "purpose": level_purpose,
            "replicas": [
                _replica(root, "candidate", level_id, seed, performance_status)
                for seed in candidate_seeds
            ],
        }
        for level_id, dt_s, level_purpose in candidate_level_specs
    }
    candidate: dict[str, Any] = {
        "status": "COMPLETE",
        "tool_revision": assembler.LEGACY_CANDIDATE_TOOL_REVISION,
        "participant": "candidate",
        "comparison_status": (f"READY_FOR_INDEPENDENT_ENSEMBLE_{purpose.upper()}_EVALUATION"),
        "input_sha256": _sha256(canonical),
        "input_content_hash": "sha256:" + "a" * 64,
        "particle_count": 287,
        "output_count": 121,
        "time_end_s": 0.03,
        "levels": candidate_levels,
    }
    if campaign_identity is not None:
        campaign_binding = {
            "contract_sha256": "b" * 64,
            "input_sha256": _sha256(canonical),
            "input_content_hash": "sha256:" + "a" * 64,
        }
        pilot_authorization = {
            "path": (
                "particle_platform_redesign/solver/evidence/m3c2/"
                "caseP_100nm_pilot_execution_v1/execution_authorization.json"
            ),
            "sha256": "d" * 64,
        }
        comsol["tool_revision"] = assembler.COMSOL_TOOL_REVISION
        comsol["case_id"] = campaign_identity["evaluation_case_id"]
        comsol["campaign_identity"] = campaign_identity
        comsol["campaign_binding"] = campaign_binding
        comsol["pilot_authorization"] = pilot_authorization
        candidate["tool_revision"] = assembler.CANDIDATE_TOOL_REVISION
        candidate["campaign_identity"] = campaign_identity
        candidate["campaign_binding"] = campaign_binding
        candidate["evaluation_policy_sha256"] = CASE_P_POLICY_SHA256
        candidate["pilot_authorization"] = pilot_authorization
    comsol_path = root / "comsol_manifest.json"
    candidate_path = root / "candidate_manifest.json"
    _write_json(comsol_path, comsol)
    _write_json(candidate_path, candidate)
    return comsol_path, candidate_path, canonical


def _stub_canonical_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    data = SimpleNamespace(
        geometry=SimpleNamespace(
            nodes_m=np.asarray([[-0.25, 4.0], [1.5, -2.0], [0.5, 3.0]], dtype=np.float64)
        ),
        sources=(SimpleNamespace(name="particles", particle_id=np.arange(1, 288)),),
    )
    monkeypatch.setattr(
        assembler,
        "read_with_info",
        lambda _: (data, SimpleNamespace(content_hash="sha256:" + "a" * 64)),
    )
    monkeypatch.setattr(
        assembler,
        "_validate_trajectory_scope",
        lambda *_: 287 * 121,
    )
    monkeypatch.setattr(assembler, "_validate_event_scope", lambda *_: 0)


def _all_replicas(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        replica
        for participant in campaign["participants"].values()
        for level in participant["levels"]
        for replica in level["replicas"]
    ]


def test_pilot_emits_exact_evaluator_schema_and_omits_contaminated_performance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    comsol, candidate, canonical = _inputs(tmp_path / "inputs")
    raw = json.loads(candidate.read_text(encoding="utf-8"))
    contaminated = raw["levels"]["macro_medium"]["replicas"][0]
    performance = candidate.parent / contaminated["performance"]
    _write_json(performance, _performance("NON_AUTHORITATIVE_EXTERNAL_WORKLOAD_OVERLAP"))
    contaminated["performance_sha256"] = _sha256(performance)
    _write_json(candidate, raw)
    _stub_canonical_reader(monkeypatch)
    output = tmp_path / "campaign" / "pilot.json"

    campaign = assembler.assemble_campaign(
        comsol,
        candidate,
        canonical,
        "pilot",
        output,
        candidate_path_reference_level_id="macro_medium",
    )

    assert set(campaign) == {
        "schema_version",
        "manifest_kind",
        "case_id",
        "purpose",
        "scope",
        "participant_manifests",
        "participants",
    }
    assert campaign["case_id"] == assembler.CASE_ID
    assert campaign["scope"] == {
        "particle_ids": list(range(1, 288)),
        "output_times_s": list(assembler.OUTPUT_TIMES_S),
        "geometry_bounds_m": [-0.25, 1.5, -2.0, 4.0],
        "canonical_input": {
            "path": "../inputs/canonical.h5",
            "sha256": _sha256(canonical),
            "content_hash": "sha256:" + "a" * 64,
        },
    }
    participant_manifests = cast(dict[str, Any], campaign["participant_manifests"])
    assert set(participant_manifests) == {"comsol", "candidate"}
    assert participant_manifests["comsol"]["sha256"] == _sha256(comsol)
    assert participant_manifests["candidate"]["sha256"] == _sha256(candidate)
    participants = cast(dict[str, Any], campaign["participants"])
    assert set(participants) == {"comsol", "candidate"}
    candidate_levels = participants["candidate"]["levels"]
    assert [level["level_id"] for level in candidate_levels] == [
        "macro_coarse",
        "macro_medium",
        "macro_fine",
        "path_fine",
    ]
    assert candidate_levels[-1]["numerical_setting"]["reference_level_id"] == "macro_medium"
    assert all(
        set(replica) == {"seed", "trajectory", "events"} for replica in _all_replicas(campaign)
    )
    assert json.loads(output.read_text(encoding="utf-8")) == campaign
    first = _all_replicas(campaign)[0]
    assert (output.parent / first["trajectory"]["path"]).resolve().is_file()
    monkeypatch.setattr(
        evaluator,
        "_load_replica",
        lambda raw, *_: SimpleNamespace(seed=int(raw["seed"])),
    )
    policy = evaluator.Policy(
        path=output,
        sha256="",
        confidence=0.95,
        bootstrap_resamples=100,
        bootstrap_seed=1,
        quantiles=(0.5,),
        radial_bins=2,
        axial_bins=2,
        pilot_replicas=4,
        pilot_levels=3,
        max_stabilization_ratio=0.8,
        final_replicas=32,
        margins={},
        numerical_fraction=0.25,
        path_sensitivity_fraction=0.25,
    )
    loaded = evaluator._load_campaign(output, policy)
    assert loaded.purpose == "pilot"


def test_performance_is_all_or_none_when_every_cell_is_comparable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    comsol, candidate, canonical = _inputs(tmp_path / "inputs")
    _stub_canonical_reader(monkeypatch)

    campaign = assembler.assemble_campaign(
        comsol,
        candidate,
        canonical,
        "pilot",
        tmp_path / "campaign.json",
        candidate_path_reference_level_id="macro_fine",
    )

    assert all(
        set(replica) == {"seed", "trajectory", "events", "performance"}
        for replica in _all_replicas(campaign)
    )


def test_explicit_participant_identity_drives_evaluator_case_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    comsol, candidate, canonical = _inputs(tmp_path / "inputs", campaign_identity=CASE_P_CAMPAIGN)
    _stub_canonical_reader(monkeypatch)

    campaign = assembler.assemble_campaign(
        comsol,
        candidate,
        canonical,
        "pilot",
        tmp_path / "campaign.json",
        candidate_path_reference_level_id="macro_fine",
    )

    assert campaign["case_id"] == CASE_P_CAMPAIGN["evaluation_case_id"]
    assert campaign["campaign_binding"] == {
        "contract_sha256": "b" * 64,
        "input_sha256": _sha256(canonical),
        "input_content_hash": "sha256:" + "a" * 64,
    }
    assert campaign["evaluation_policy_sha256"] == CASE_P_POLICY_SHA256
    authorization = cast(dict[str, Any], campaign["pilot_authorization"])
    assert authorization["sha256"] == "d" * 64


@pytest.mark.parametrize("damage", ["missing", "uppercase"])
def test_explicit_participant_requires_recipe_evaluation_policy_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    comsol, candidate, canonical = _inputs(tmp_path / "inputs", campaign_identity=CASE_P_CAMPAIGN)
    candidate_raw = json.loads(candidate.read_text(encoding="utf-8"))
    if damage == "missing":
        candidate_raw.pop("evaluation_policy_sha256")
    else:
        candidate_raw["evaluation_policy_sha256"] = CASE_P_POLICY_SHA256.upper()
    _write_json(candidate, candidate_raw)
    _stub_canonical_reader(monkeypatch)
    output = tmp_path / "must-not-exist.json"

    with pytest.raises(ValueError, match="evaluation policy SHA-256"):
        assembler.assemble_campaign(
            comsol,
            candidate,
            canonical,
            "pilot",
            output,
            candidate_path_reference_level_id="macro_fine",
        )

    assert not output.exists()


@pytest.mark.parametrize("damage", ["missing", "mismatch"])
def test_explicit_participant_identity_must_be_complete_and_equal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    comsol, candidate, canonical = _inputs(tmp_path / "inputs", campaign_identity=CASE_P_CAMPAIGN)
    candidate_raw = json.loads(candidate.read_text(encoding="utf-8"))
    if damage == "missing":
        candidate_raw.pop("campaign_identity")
    else:
        candidate_raw["campaign_identity"]["case_id"] = "wrong-case"
    _write_json(candidate, candidate_raw)
    _stub_canonical_reader(monkeypatch)
    output = tmp_path / "must-not-exist.json"

    with pytest.raises(ValueError, match="campaign identit"):
        assembler.assemble_campaign(
            comsol,
            candidate,
            canonical,
            "pilot",
            output,
            candidate_path_reference_level_id="macro_fine",
        )

    assert not output.exists()


def test_identity_free_fallback_is_limited_to_historical_case_a(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    comsol, candidate, canonical = _inputs(tmp_path / "inputs")
    candidate_raw = json.loads(candidate.read_text(encoding="utf-8"))
    candidate_raw["tool_revision"] = assembler.CANDIDATE_TOOL_REVISION
    _write_json(candidate, candidate_raw)
    _stub_canonical_reader(monkeypatch)

    with pytest.raises(ValueError, match="must define campaign identity"):
        assembler.assemble_campaign(
            comsol,
            candidate,
            canonical,
            "pilot",
            tmp_path / "must-not-exist.json",
            candidate_path_reference_level_id="macro_fine",
        )


def test_explicit_participant_binding_rejects_mixed_contracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    comsol, candidate, canonical = _inputs(tmp_path / "inputs", campaign_identity=CASE_P_CAMPAIGN)
    candidate_raw = json.loads(candidate.read_text(encoding="utf-8"))
    candidate_raw["campaign_binding"]["contract_sha256"] = "c" * 64
    _write_json(candidate, candidate_raw)
    _stub_canonical_reader(monkeypatch)

    with pytest.raises(ValueError, match="campaign bindings differ"):
        assembler.assemble_campaign(
            comsol,
            candidate,
            canonical,
            "pilot",
            tmp_path / "must-not-exist.json",
            candidate_path_reference_level_id="macro_fine",
        )


def test_explicit_pilot_rejects_mixed_authorizations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    comsol, candidate, canonical = _inputs(tmp_path / "inputs", campaign_identity=CASE_P_CAMPAIGN)
    candidate_raw = json.loads(candidate.read_text(encoding="utf-8"))
    candidate_raw["pilot_authorization"]["sha256"] = "e" * 64
    _write_json(candidate, candidate_raw)
    _stub_canonical_reader(monkeypatch)

    with pytest.raises(ValueError, match="pilot authorizations differ"):
        assembler.assemble_campaign(
            comsol,
            candidate,
            canonical,
            "pilot",
            tmp_path / "must-not-exist.json",
            candidate_path_reference_level_id="macro_fine",
        )


def test_final_uses_one_32_seed_level_and_forbids_path_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    comsol, candidate, canonical = _inputs(tmp_path / "inputs", purpose="final")
    _stub_canonical_reader(monkeypatch)
    output = tmp_path / "final.json"

    campaign = assembler.assemble_campaign(comsol, candidate, canonical, "final", output)

    participants = cast(dict[str, Any], campaign["participants"])
    assert all(len(participant["levels"]) == 1 for participant in participants.values())
    assert all(
        len(participant["levels"][0]["replicas"]) == 32 for participant in participants.values()
    )
    with pytest.raises(ValueError, match="only valid for a pilot"):
        assembler.assemble_campaign(
            comsol,
            candidate,
            canonical,
            "final",
            tmp_path / "rejected.json",
            candidate_path_reference_level_id="accepted",
        )


@pytest.mark.parametrize("damage", ["status", "seed_overlap", "bad_hash", "scope"])
def test_invalid_participant_evidence_fails_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    comsol, candidate, canonical = _inputs(tmp_path / "inputs")
    comsol_raw = json.loads(comsol.read_text(encoding="utf-8"))
    candidate_raw = json.loads(candidate.read_text(encoding="utf-8"))
    if damage == "status":
        candidate_raw["status"] = "BLOCKED"
    elif damage == "seed_overlap":
        for level in candidate_raw["levels"].values():
            for index, replica in enumerate(level["replicas"]):
                replica["seed"] = 10 + index
    elif damage == "bad_hash":
        candidate_raw["levels"]["macro_coarse"]["replicas"][0]["trajectory_sha256"] = "0" * 64
    else:
        comsol_raw["common_observation_times"][-1] = 0.031
    _write_json(comsol, comsol_raw)
    _write_json(candidate, candidate_raw)
    _stub_canonical_reader(monkeypatch)
    output = tmp_path / "must_not_exist.json"

    with pytest.raises(ValueError):
        assembler.assemble_campaign(
            comsol,
            candidate,
            canonical,
            "pilot",
            output,
            candidate_path_reference_level_id="macro_fine",
        )

    assert not output.exists()


def test_scope_scanners_require_complete_schedule_and_bounded_unique_events(
    tmp_path: Path,
) -> None:
    trajectory = tmp_path / "trajectory.csv"
    with trajectory.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(assembler.TRAJECTORY_COLUMNS))
        writer.writeheader()
        for particle_id in (1, 2):
            for time_s in (0.0, 0.5):
                writer.writerow(
                    {
                        "particle_id": particle_id,
                        "time_s": time_s,
                        "r_m": 0.1,
                        "z_m": 0.2,
                        "velocity_r_m_per_s": 0.0,
                        "velocity_z_m_per_s": 0.0,
                        "charge_number_e": -1.0,
                        "lifecycle": "active",
                    }
                )
    events = tmp_path / "events.csv"
    events.write_text(
        "particle_id,event_time_s,outcome,boundary_semantic\n2,0.25,stuck,wall\n",
        encoding="utf-8",
    )

    assert assembler._validate_trajectory_scope(trajectory, (1, 2), (0.0, 0.5)) == 4
    assert assembler._validate_event_scope(events, (1, 2), 0.5) == 1

    rows = trajectory.read_text(encoding="utf-8").splitlines()
    trajectory.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="complete particle/time scope"):
        assembler._validate_trajectory_scope(trajectory, (1, 2), (0.0, 0.5))
