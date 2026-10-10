"""Focused synthetic checks for the M3-C2 ensemble evaluator."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from tools.vv.comsol import evaluate_m3c2_stochastic_ensemble as evaluator


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _artifact(path: Path) -> dict[str, object]:
    return {"path": path.name, "sha256": _sha256(path)}


def _policy(path: Path, *, revision: int = 3) -> Path:
    margins = {
        "mean_position": 0.01,
        "covariance": 0.05,
        "quantile": 0.01,
        "occupancy_tv": 0.2,
        "fate_probability": 0.1,
        "first_arrival_cdf": 0.1,
    }
    pilot: dict[str, object] = {
        "replicas_per_participant": 4,
        "minimum_levels": 3,
        "numerical_margin_fraction": 0.25 if revision >= 3 else 0.5,
        "path_sensitivity_margin_fraction": 0.25,
        "maximum_fine_to_coarse_ratio": 0.6,
    }
    final: dict[str, object] = {"replicas_per_participant": 4}
    value: dict[str, object] = {
        "schema_version": 1,
        "policy_kind": "m3c2_stochastic_ensemble",
        "policy_revision": revision,
        "bootstrap": {"confidence": 0.9, "resamples": 100, "seed": 4471},
        "quantiles": [0.25, 0.5, 0.75],
        "occupancy": {"radial_bins": 2, "axial_bins": 2},
        "pilot": pilot,
        "final": final,
    }
    if revision == 1:
        final["equivalence_margins"] = margins
    else:
        evidence = path.parent / "rng_noninteraction_evidence.json"
        _write_json(
            evidence,
            {
                "schema_version": 1,
                "manifest_kind": "m3c2_rng_noninteraction_evidence",
                "status": "PASS_DOCUMENTED_INDEPENDENT_PARTICLE_STREAMS_AND_ONE_WAY_DYNAMICS",
                "statistical_assumptions": {
                    "independent_unit": "seed_x_fixed_source_particle_trajectory"
                },
            },
        )
        pilot["screening_margins"] = margins
        if revision >= 3:
            pilot.update(
                {
                    "selection_method": (
                        "direct_each_macro_level_to_registered_finest_population_"
                        "terminal_point_summary"
                    ),
                    "gating_observables": ["stuck", "held", "escaped", "any_terminal"],
                    "terminal_screening_margin": 0.2,
                    "registered_finest_reference": "last_macro_level_by_ordinal",
                    "candidate_path_reference": "registered_finest_macro_level",
                }
            )
        final.update(
            {
                "fixed_design": {
                    "case_id": "M3-C2A_caseA_100nm_common-P1",
                    "particle_count": 3,
                    "output_time_count": 3,
                },
                "terminal_population_gate": {
                    "method": "two_sample_union_hoeffding",
                    "curves": ["stuck", "held", "escaped", "any_terminal"],
                    "familywise_alpha": 0.1,
                    "margin": 0.8,
                },
                "seed_plan": {
                    "comsol": [1000, 1001, 1002, 1003],
                    "candidate": [1010, 1011, 1012, 1013],
                },
            }
        )
        value["independence_evidence"] = _artifact(evidence)
    _write_json(path, value)
    return path


def _synthetic_v4_campaign_policy(path: Path) -> tuple[Path, evaluator.Policy]:
    policy_path = _policy(path)
    policy = replace(evaluator._load_policy(policy_path), revision=4)
    return policy_path, policy


def _trajectory(
    path: Path,
    *,
    seed: int,
    bias: float,
    participant_shift: float,
    origin_shift: float = 0.0,
    swap_drift: bool = False,
) -> None:
    fieldnames = (
        "particle_id",
        "time_s",
        "r_m",
        "z_m",
        "velocity_r_m_per_s",
        "velocity_z_m_per_s",
        "charge_number_e",
        "lifecycle",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for particle_id, initial_r in enumerate((0.2, 0.4, 0.7), start=1):
            direction = -1.0 if swap_drift and particle_id in {1, 2} else 1.0
            seed_offset = (seed % 4 - 1.5) * 1.0e-4
            for time_s in (0.0, 0.5, 1.0):
                displacement = direction * (0.02 + bias + participant_shift) * time_s
                initial_row = time_s == 0.0
                writer.writerow(
                    {
                        "particle_id": particle_id,
                        "time_s": time_s,
                        "r_m": initial_r + origin_shift + displacement + seed_offset * time_s,
                        "z_m": 0.2 + 0.1 * particle_id + 0.01 * time_s,
                        "velocity_r_m_per_s": (
                            0.02 if initial_row else direction * (0.02 + bias + participant_shift)
                        ),
                        "velocity_z_m_per_s": 0.01,
                        "charge_number_e": -10.0,
                        "lifecycle": "active",
                    }
                )


def _events(path: Path) -> None:
    path.write_text("particle_id,event_time_s,outcome,boundary_semantic\n", encoding="utf-8")


def _numerical_setting(purpose: str, ordinal: int, path_reference_index: int) -> dict[str, object]:
    if purpose != "pilot":
        return {"step_s": 0.0025, "purpose": "accepted_final"}
    if ordinal != 3:
        return {"step_s": 0.01 / 2**ordinal, "purpose": "macro_step_convergence"}
    return {
        "step_s": 0.01 / 2**path_reference_index,
        "purpose": "brownian_first_passage_path_depth_sensitivity",
        "reference_level_id": f"level_{path_reference_index}",
    }


def _campaign(
    root: Path,
    *,
    purpose: str,
    seed_start: int,
    candidate_shift: float = 0.0,
    candidate_origin_shift: float = 0.0,
    swap_candidate_drift: bool = False,
    measured_performance: bool = False,
    path_reference_index: int = 2,
    evaluation_policy_sha256: str | None = None,
) -> Path:
    canonical = root / "canonical_input.h5"
    canonical.write_bytes(b"synthetic locked canonical input")
    participant_manifests: dict[str, object] = {}
    participants: dict[str, object] = {}
    for participant_index, participant in enumerate(("comsol", "candidate")):
        participant_manifest = root / f"{purpose}_{participant}_participant.json"
        _write_json(participant_manifest, {"participant": participant, "purpose": purpose})
        participant_manifests[participant] = _artifact(participant_manifest)
        levels: list[dict[str, object]] = []
        seeds = [seed_start + participant_index * 10 + index for index in range(4)]
        level_biases = (0.01, 0.002, 0.0) if purpose == "pilot" else (0.0,)
        if purpose == "pilot" and participant == "candidate":
            level_biases = (*level_biases, level_biases[path_reference_index])
        for ordinal, bias in enumerate(level_biases):
            replicas: list[dict[str, object]] = []
            for seed in seeds:
                stem = f"{purpose}_{participant}_l{ordinal}_s{seed}"
                trajectory = root / f"{stem}_trajectory.csv"
                events = root / f"{stem}_events.csv"
                _trajectory(
                    trajectory,
                    seed=seed,
                    bias=bias,
                    participant_shift=candidate_shift if participant == "candidate" else 0.0,
                    origin_shift=(candidate_origin_shift if participant == "candidate" else 0.0),
                    swap_drift=swap_candidate_drift and participant == "candidate",
                )
                _events(events)
                replicas.append(
                    {
                        "seed": seed,
                        "trajectory": _artifact(trajectory),
                        "events": _artifact(events),
                    }
                )
                if measured_performance:
                    performance = root / f"{stem}_performance.json"
                    _write_json(
                        performance,
                        {
                            "wall_time_s": 10.0,
                            "peak_rss_bytes": 100_000_000,
                            "output_bytes": 2048,
                            "particle_count": 3,
                            "output_frames": 3,
                            "stage_times_s": {"field": 3.0, "writer": 1.0},
                        },
                    )
                    replicas[-1]["performance"] = _artifact(performance)
            levels.append(
                {
                    "level_id": f"level_{ordinal}",
                    "ordinal": ordinal,
                    "numerical_setting": _numerical_setting(purpose, ordinal, path_reference_index),
                    "replicas": replicas,
                }
            )
        participants[participant] = {"levels": levels}
    manifest = {
        "schema_version": 1,
        "manifest_kind": "m3c2_campaign",
        "case_id": "M3-C2A_caseA_100nm_common-P1",
        "purpose": purpose,
        "scope": {
            "particle_ids": [1, 2, 3],
            "output_times_s": [0.0, 0.5, 1.0],
            "geometry_bounds_m": [0.0, 1.0, 0.0, 1.0],
            "canonical_input": {
                **_artifact(canonical),
                "content_hash": f"sha256:{_sha256(canonical)}",
            },
        },
        "participant_manifests": participant_manifests,
        "participants": participants,
    }
    if evaluation_policy_sha256 is not None:
        canonical_sha256 = _sha256(canonical)
        manifest["campaign_binding"] = {
            "contract_sha256": "b" * 64,
            "input_sha256": canonical_sha256,
            "input_content_hash": f"sha256:{canonical_sha256}",
        }
        manifest["evaluation_policy_sha256"] = evaluation_policy_sha256
    path = root / f"{purpose}_campaign.json"
    _write_json(path, manifest)
    return path


def _authorization(policy: Path, pilot_report: Path, final_campaign: Path) -> Path:
    loaded_policy = evaluator._load_policy(policy.resolve())
    loaded_campaign = evaluator._load_campaign(final_campaign.resolve(), loaded_policy)
    scope = evaluator._scope_fingerprint(loaded_campaign, loaded_policy)
    pilot = json.loads(pilot_report.read_text(encoding="utf-8"))
    seed_plan = evaluator._campaign_seed_plan(loaded_campaign)
    receipt = {
        "schema_version": 1,
        "receipt_kind": "m3c2_post_pilot_final_authorization",
        "status": "AUTHORIZED_FOR_CONFIRMATORY_FINAL",
        "policy_sha256": loaded_policy.sha256,
        "pilot_report_sha256": _sha256(pilot_report),
        "pilot_scope_sha256": pilot["scope_fingerprint"]["sha256"],
        "common_design_sha256": scope["common_design_sha256"],
        "selected_final_levels": evaluator._selected_final_levels(loaded_campaign),
        "seed_plan_sha256": evaluator._object_sha256(seed_plan),
    }
    path = final_campaign.parent / "final_authorization.json"
    _write_json(path, receipt)
    return path


def _evaluate_final(
    policy: Path,
    final_campaign: Path,
    output: Path,
    pilot_report: Path,
    *,
    authorization: Path | None = None,
) -> dict[str, object]:
    receipt = authorization or _authorization(policy, pilot_report, final_campaign)
    return evaluator.evaluate(
        policy,
        final_campaign,
        output,
        pilot_report_path=pilot_report,
        authorization_receipt_path=receipt,
    )


def _rewrite_sparse_escape(campaign: Path) -> None:
    raw = json.loads(campaign.read_text(encoding="utf-8"))
    for participant in raw["participants"].values():
        for level in participant["levels"]:
            for replica in level["replicas"]:
                trajectory = campaign.parent / replica["trajectory"]["path"]
                with trajectory.open(encoding="utf-8", newline="") as stream:
                    rows = list(csv.DictReader(stream))
                with trajectory.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]), lineterminator="\n")
                    writer.writeheader()
                    writer.writerows(
                        row
                        for row in rows
                        if not (row["particle_id"] == "3" and float(row["time_s"]) >= 0.5)
                    )
                events = campaign.parent / replica["events"]["path"]
                events.write_text(
                    "particle_id,event_time_s,outcome,boundary_semantic\n"
                    "3,0.25,escaped,pump_outlet\n",
                    encoding="utf-8",
                )
                replica["trajectory"]["sha256"] = _sha256(trajectory)
                replica["events"]["sha256"] = _sha256(events)
    _write_json(campaign, raw)


def _rewrite_candidate_all_stuck(campaign: Path) -> None:
    raw = json.loads(campaign.read_text(encoding="utf-8"))
    level = raw["participants"]["candidate"]["levels"][0]
    for replica in level["replicas"]:
        trajectory = campaign.parent / replica["trajectory"]["path"]
        with trajectory.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            if float(row["time_s"]) >= 0.5:
                row["lifecycle"] = "stuck"
        with trajectory.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        events = campaign.parent / replica["events"]["path"]
        events.write_text(
            "particle_id,event_time_s,outcome,boundary_semantic\n"
            "1,0.25,stuck,wall\n2,0.25,stuck,wall\n3,0.25,stuck,wall\n",
            encoding="utf-8",
        )
        replica["trajectory"]["sha256"] = _sha256(trajectory)
        replica["events"]["sha256"] = _sha256(events)
    _write_json(campaign, raw)


def _shift_candidate_t0_velocity(campaign: Path, difference: float) -> None:
    raw = json.loads(campaign.read_text(encoding="utf-8"))
    level = raw["participants"]["candidate"]["levels"][0]
    for replica in level["replicas"]:
        trajectory = campaign.parent / replica["trajectory"]["path"]
        with trajectory.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            if float(row["time_s"]) == 0.0:
                row["velocity_r_m_per_s"] = str(float(row["velocity_r_m_per_s"]) + difference)
        with trajectory.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        replica["trajectory"]["sha256"] = _sha256(trajectory)
    _write_json(campaign, raw)


def _reverse_level_replicas(campaign: Path, participant: str, level_index: int) -> None:
    raw = json.loads(campaign.read_text(encoding="utf-8"))
    raw["participants"][participant]["levels"][level_index]["replicas"].reverse()
    _write_json(campaign, raw)


def _pilot(policy: Path, root: Path, *, seed_start: int = 100) -> tuple[dict[str, object], Path]:
    campaign = _campaign(root, purpose="pilot", seed_start=seed_start)
    output = root / "pilot_output"
    report = evaluator.evaluate(policy, campaign, output)
    return report, output / "evaluation_manifest.json"


def _assert_pilot_screening(pilot: dict[str, object]) -> None:
    assert pilot["status"] == "PASS"
    screening = cast(dict[str, Any], pilot["pilot_configuration_screening"])
    candidate = cast(dict[str, Any], screening["candidate"])
    assert candidate["selection_method"].startswith("direct_each_macro_level")
    assert candidate["largest_screened_macro_level"] == "level_0"
    assert candidate["continuous_weak_convergence_established"] is False
    assert candidate["adjacent_terminal_diagnostics"]["role"].startswith("descriptive_only")
    assert all(
        comparison["terminal_gate"]["status"] == "PASS"
        for comparison in candidate["comparisons_to_finest"]
    )
    assert pilot["bootstrap_execution"] == "NOT_RUN_POLICY_REVISION_3"
    source_identity = cast(dict[str, Any], pilot["source_identity"])
    assert source_identity["status"] == "PASS"


def _assert_zero_difference_final(final: dict[str, object], output: Path) -> None:
    gate = cast(dict[str, Any], final["terminal_population_gate"])
    assert final["status"] == "PASS"
    assert final["resampling_unit"] == "seed_x_fixed_source_particle_trajectory"
    assert gate["maximum_absolute_difference"] == 0.0
    assert gate["critical_radius"] > 0.0
    assert gate["observations_per_participant"] == 12
    assert gate["simultaneous_endpoint_count"] == 12
    assert gate["active_curve_role"].endswith("not_double_counted")
    assert (output / "rz_ensemble_trajectory.svg").is_file()


def test_registered_v2_and_v3_policies_lock_the_same_final_design() -> None:
    cases = Path(__file__).resolve().parents[1] / "cases"
    v2_path = cases / "m3c2_caseA_100nm_ensemble_evaluation_v2.json"
    v3_path = cases / "m3c2_caseA_100nm_ensemble_evaluation_v3.json"
    v2 = evaluator._load_policy(v2_path)
    v3 = evaluator._load_policy(v3_path)
    historical = evaluator._load_policy(cases / "m3c2_caseA_100nm_ensemble_evaluation_v1.json")
    expected_radius = math.sqrt(math.log(2.0 * 121 * 4 / 0.05) / (32 * 287))
    assert (v2.revision, v3.revision) == (2, 3)
    assert v3.expected_particle_count == v2.expected_particle_count == 287
    assert v3.expected_output_count == v2.expected_output_count == 121
    assert v3.final_replicas == v2.final_replicas == 32
    assert v3.terminal_margin == v2.terminal_margin == 0.05
    assert v3.margins == v2.margins == historical.margins
    assert expected_radius == pytest.approx(0.0327841444359473, rel=0.0, abs=1.0e-16)
    assert v3.final_seed_plan == v2.final_seed_plan
    assert v3.final_seed_plan is not None
    assert v3.final_seed_plan["comsol"] == tuple(range(318160, 318192))
    assert v3.final_seed_plan["candidate"] == tuple(range(318192, 318224))
    raw_v3 = json.loads(v3_path.read_text(encoding="utf-8"))
    assert raw_v3["pilot"]["terminal_screening_margin"] == 0.0125
    assert raw_v3["pilot"]["maximum_fine_to_coarse_ratio_role"].endswith("not_a_v3_gate")


def test_v4_full_population_partition_counts_each_unit_once(tmp_path: Path) -> None:
    policy = evaluator.Policy(
        path=tmp_path / "policy.json",
        sha256="0" * 64,
        confidence=0.95,
        bootstrap_resamples=100,
        bootstrap_seed=1,
        quantiles=(0.5,),
        radial_bins=2,
        axial_bins=2,
        pilot_replicas=2,
        pilot_levels=3,
        max_stabilization_ratio=0.6,
        final_replicas=2,
        margins={
            "mean_position": 0.1,
            "covariance": 0.1,
            "quantile": 0.1,
            "occupancy_tv": 0.1,
            "fate_probability": 0.1,
            "first_arrival_cdf": 0.1,
        },
        numerical_fraction=0.25,
        path_sensitivity_fraction=0.25,
        revision=4,
        terminal_alpha=0.025,
        terminal_margin=0.1,
        rz_distribution_alpha=0.025,
        rz_distribution_margin=0.25,
        rz_one_sample_radius=0.05,
        rz_two_sample_radius=0.1,
        rz_category_count=7,
        rz_participant_union_count=2,
        rz_screening_margin=0.0625,
        expected_case_id="synthetic",
        expected_particle_count=2,
        expected_output_count=2,
        final_seed_plan={"comsol": (1, 2), "candidate": (3, 4)},
    )
    campaign = evaluator.Campaign(
        path=tmp_path / "campaign.json",
        sha256="1" * 64,
        purpose="final",
        particle_ids=np.array([1, 2]),
        times_s=np.array([0.0, 1.0]),
        bounds_m=(0.0, 1.0, 0.0, 1.0),
        geometry_scale_m=1.0,
        participants={},
        case_id="synthetic",
        canonical_input=None,
        participant_manifests=None,
        campaign_binding=None,
        evaluation_policy_sha256=None,
    )
    positions = np.array(
        [
            [[[0.1, 0.1], [0.9, 0.9]], [[0.1, 0.1], [math.nan, math.nan]]],
            [[[0.1, 0.1], [0.9, 0.9]], [[0.9, 0.9], [math.nan, math.nan]]],
        ]
    )
    fate = np.zeros((2, 2, 2, 4), dtype=np.float64)
    fate[:, 0, :, 0] = 1.0
    fate[:, 1, 0, 0] = 1.0
    fate[:, 1, 1, 1] = 1.0
    metrics = {
        "position_m": positions,
        "fate": fate,
        "first_arrival": 1.0 - fate[..., 0],
    }
    partition = evaluator._full_population_rz_fate_partition(metrics, campaign, policy)
    assert partition.shape == (2, 7)
    assert np.sum(partition, axis=1) == pytest.approx([1.0, 1.0])
    assert partition[1, -3:].tolist() == pytest.approx([0.5, 0.0, 0.0])
    same = evaluator._rz_distribution_gate(metrics, metrics, campaign, policy)
    assert same["status"] == "PASS"
    assert same["maximum_empirical_total_variation"] == 0.0

    shifted = {**metrics, "position_m": positions.copy()}
    shifted["position_m"][:, :, 0] = [0.9, 0.1]
    different = evaluator._rz_distribution_gate(metrics, shifted, campaign, policy)
    assert different["status"] == "FAIL"
    assert cast(float, different["maximum_empirical_total_variation"]) > 0.25


def test_registered_casep_v4_policy_locks_two_familywise_gates_and_seed_pointer() -> None:
    path = (
        Path(__file__).resolve().parents[1] / "cases/m3c2_caseP_100nm_ensemble_evaluation_v4.json"
    )
    policy = evaluator._load_policy(path)
    expected_one_sample = math.sqrt(
        (83 * math.log(2.0) + math.log(2 * 121 / 0.025)) / (2 * 32 * 287)
    )
    assert policy.revision == 4
    assert policy.rz_category_count == 83
    assert policy.rz_participant_union_count == 2
    assert policy.rz_one_sample_radius == pytest.approx(expected_one_sample, abs=5.0e-16)
    assert policy.rz_two_sample_radius == pytest.approx(2 * expected_one_sample, abs=5.0e-16)
    assert policy.rz_distribution_margin == 0.15
    assert policy.rz_screening_margin == 0.0375
    assert policy.terminal_alpha == policy.rz_distribution_alpha == 0.025
    assert policy.final_seed_plan is not None
    assert policy.final_seed_plan["comsol"] == tuple(range(319000, 319032))
    assert policy.final_seed_plan["candidate"] == tuple(range(319032, 319064))


@pytest.mark.parametrize("damage", ["missing", "mismatch"])
def test_v4_campaign_enforces_recipe_evaluation_policy_hash(tmp_path: Path, damage: str) -> None:
    _, policy = _synthetic_v4_campaign_policy(tmp_path / "policy.json")
    campaign_path = _campaign(
        tmp_path,
        purpose="pilot",
        seed_start=100,
        evaluation_policy_sha256=policy.sha256,
    )
    raw = json.loads(campaign_path.read_text(encoding="utf-8"))
    if damage == "missing":
        raw.pop("evaluation_policy_sha256")
    else:
        raw["evaluation_policy_sha256"] = "f" * 64
    _write_json(campaign_path, raw)

    with pytest.raises(ValueError, match=r"evaluation policy SHA-256|another evaluation policy"):
        evaluator._load_campaign(campaign_path, policy)


def test_casep_campaign_reloads_exact_participant_manifest_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    comsol_manifest = tmp_path / "comsol_manifest.json"
    candidate_manifest = tmp_path / "candidate_manifest.json"
    _write_json(comsol_manifest, {"marker": "reloaded-comsol"})
    _write_json(candidate_manifest, {"marker": "reloaded-candidate"})
    records = {
        "comsol": _artifact(comsol_manifest),
        "candidate": _artifact(candidate_manifest),
    }
    participants = {
        "comsol": {"participant": "comsol", "levels": []},
        "candidate": {
            "participant": "candidate",
            "levels": [
                {
                    "level_id": "path_fine",
                    "numerical_setting": {
                        "purpose": evaluator.PATH_SENSITIVITY_PURPOSE,
                        "reference_level_id": "macro_fine",
                    },
                }
            ],
        },
    }
    campaign = {
        "case_id": evaluator.CASEP_EVALUATION_CASE_ID,
        "campaign_binding": {"contract_sha256": "a" * 64},
        "evaluation_policy_sha256": "b" * 64,
        "pilot_authorization": {"path": "authorization.json", "sha256": "c" * 64},
    }
    projection = {**campaign, "participants": participants}

    def reload_projection(
        comsol: dict[str, Any],
        candidate: dict[str, Any],
        _comsol_root: Path,
        _candidate_root: Path,
        purpose: str,
        _project_root: Path,
        path_reference: str | None,
    ) -> dict[str, Any]:
        assert comsol == {"marker": "reloaded-comsol"}
        assert candidate == {"marker": "reloaded-candidate"}
        assert (purpose, path_reference) == ("pilot", "macro_fine")
        return projection

    monkeypatch.setattr(evaluator, "_participant_manifest_projection", reload_projection)
    evaluator._validate_participant_manifest_projection(
        records, tmp_path, "pilot", campaign, participants
    )

    tampered = json.loads(json.dumps(participants))
    tampered["candidate"]["unexpected"] = True
    with pytest.raises(ValueError, match="differ from their hash-locked manifests"):
        evaluator._validate_participant_manifest_projection(
            records, tmp_path, "pilot", campaign, tampered
        )


def test_v4_scope_and_final_authorization_preserve_campaign_binding(tmp_path: Path) -> None:
    policy_path, policy = _synthetic_v4_campaign_policy(tmp_path / "policy.json")
    pilot_path = _campaign(
        tmp_path,
        purpose="pilot",
        seed_start=100,
        evaluation_policy_sha256=policy.sha256,
    )
    pilot_campaign = evaluator._load_campaign(pilot_path, policy)
    pilot_scope = evaluator._scope_fingerprint(pilot_campaign, policy)
    pilot_payload = cast(dict[str, Any], pilot_scope["payload"])
    assert pilot_payload["campaign_binding"] == pilot_campaign.campaign_binding
    pilot_report_path = tmp_path / "pilot_report.json"
    pilot_report = {"scope_fingerprint": pilot_scope}
    _write_json(pilot_report_path, pilot_report)

    final_path = _campaign(
        tmp_path,
        purpose="final",
        seed_start=1000,
        evaluation_policy_sha256=policy.sha256,
    )
    raw = json.loads(final_path.read_text(encoding="utf-8"))
    raw["campaign_binding"]["contract_sha256"] = "c" * 64
    _write_json(final_path, raw)
    final_campaign = evaluator._load_campaign(final_path, policy)
    final_scope = evaluator._scope_fingerprint(final_campaign, policy)
    authorization = tmp_path / "final_authorization.json"
    _write_json(
        authorization,
        {
            "schema_version": 1,
            "receipt_kind": "m3c2_post_pilot_final_authorization",
            "status": "AUTHORIZED_FOR_CONFIRMATORY_FINAL",
        },
    )

    with pytest.raises(ValueError, match="pilot and final campaign bindings differ"):
        evaluator._verify_final_authorization(
            authorization,
            campaign=final_campaign,
            policy=policy,
            pilot_report_path=pilot_report_path,
            pilot_report=pilot_report,
            final_scope=final_scope,
        )

    assert policy_path.is_file()


def test_v3_one_fate_flip_is_one_population_unit_and_material_change_fails() -> None:
    seeds = 4
    particles = 287
    left_fate = np.zeros((seeds, 2, particles, 4), dtype=np.float64)
    left_fate[..., 0] = 1.0
    left_arrival = np.zeros((seeds, 2, particles), dtype=np.float64)
    right_fate = left_fate.copy()
    right_arrival = left_arrival.copy()
    right_fate[0, 1, 0, 0] = 0.0
    right_fate[0, 1, 0, 1] = 1.0
    right_arrival[0, 1, 0] = 1.0
    left = np.mean(
        evaluator._terminal_population_values({"fate": left_fate, "first_arrival": left_arrival}),
        axis=(0, 2),
    )
    right = np.mean(
        evaluator._terminal_population_values({"fate": right_fate, "first_arrival": right_arrival}),
        axis=(0, 2),
    )
    report = evaluator._population_terminal_point_report(left, right)
    assert report["maximum_absolute_difference"] == pytest.approx(1.0 / (seeds * particles))
    assert cast(float, report["maximum_absolute_difference"]) < 0.01
    assert (
        evaluator._v3_terminal_screening_gate(report, base_margin=0.05, fraction=0.25)["status"]
        == "PASS"
    )

    material = right.copy()
    material[1, 0] = 0.2
    material[1, 3] = 0.2
    material_report = evaluator._population_terminal_point_report(left, material)
    assert (
        evaluator._v3_terminal_screening_gate(material_report, base_margin=0.05, fraction=0.25)[
            "status"
        ]
        == "FAIL"
    )


def test_pilot_then_all_zero_final_has_nonzero_finite_sample_radius(tmp_path: Path) -> None:
    policy = _policy(tmp_path / "policy.json")
    pilot_campaign = _campaign(tmp_path, purpose="pilot", seed_start=100, measured_performance=True)
    _reverse_level_replicas(pilot_campaign, "candidate", 1)
    pilot_output = tmp_path / "pilot_output"
    pilot = evaluator.evaluate(policy, pilot_campaign, pilot_output)
    _assert_pilot_screening(pilot)

    final_campaign = _campaign(tmp_path, purpose="final", seed_start=1000)
    final_output = tmp_path / "final_output"
    final = _evaluate_final(
        policy,
        final_campaign,
        final_output,
        pilot_output / "evaluation_manifest.json",
    )
    _assert_zero_difference_final(final, final_output)


def test_v3_pilot_and_final_do_not_execute_legacy_bootstrap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_bootstrap(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("V3 must not execute legacy bootstrap calculations")

    monkeypatch.setattr(evaluator, "_compare_metric_sets", unexpected_bootstrap)
    monkeypatch.setattr(evaluator, "_compare_descriptive_metric_sets", unexpected_bootstrap)
    policy = _policy(tmp_path / "policy.json")
    _, pilot_report = _pilot(policy, tmp_path, seed_start=150)
    final_campaign = _campaign(tmp_path, purpose="final", seed_start=1000)
    final = _evaluate_final(policy, final_campaign, tmp_path / "no_bootstrap_final", pilot_report)
    assert final["status"] == "PASS"
    assert final["bootstrap_execution"] == "NOT_RUN_POLICY_REVISION_3"


def test_continuous_per_id_differences_are_descriptive_only(tmp_path: Path) -> None:
    policy = _policy(tmp_path / "policy.json")
    _, pilot_report = _pilot(policy, tmp_path, seed_start=200)
    final_campaign = _campaign(
        tmp_path,
        purpose="final",
        seed_start=1000,
        swap_candidate_drift=True,
    )
    report = _evaluate_final(policy, final_campaign, tmp_path / "descriptive", pilot_report)
    descriptive = cast(dict[str, Any], report["descriptive_continuous_metrics"])
    assert report["status"] == "PASS"
    assert descriptive["mean_position"]["maximum_absolute_difference"] > 0.0
    assert descriptive["mean_position"]["evidence_role"] == ("descriptive_non_gating_point_summary")
    assert descriptive["mean_position"]["bootstrap_used"] is False
    assert "bootstrap_critical_radius" not in descriptive["mean_position"]


def test_auxiliary_occupancy_cannot_fail_final_decision(tmp_path: Path) -> None:
    policy = _policy(tmp_path / "policy.json")
    _, pilot_report = _pilot(policy, tmp_path, seed_start=250)
    final_campaign = _campaign(tmp_path, purpose="final", seed_start=1000, candidate_shift=0.2)
    report = _evaluate_final(policy, final_campaign, tmp_path / "occupancy", pilot_report)
    occupancy = cast(dict[str, Any], report["descriptive_continuous_metrics"])["occupancy"]
    assert report["status"] == "PASS"
    assert occupancy["maximum_total_variation"] > 0.0
    assert occupancy["evidence_role"] == ("auxiliary_descriptive_non_gating_point_summary")


def test_deliberate_terminal_population_difference_fails_final(tmp_path: Path) -> None:
    policy = _policy(tmp_path / "policy.json")
    _, pilot_report = _pilot(policy, tmp_path, seed_start=300)
    final_campaign = _campaign(tmp_path, purpose="final", seed_start=1000)
    _rewrite_candidate_all_stuck(final_campaign)
    report = _evaluate_final(policy, final_campaign, tmp_path / "terminal_fail", pilot_report)
    gate = cast(dict[str, Any], report["terminal_population_gate"])
    assert report["status"] == "FAIL"
    assert gate["status"] == "FAIL"
    assert gate["maximum_absolute_difference"] == 1.0


def test_path_sensitivity_uses_explicit_nonfinest_reference(tmp_path: Path) -> None:
    policy = _policy(tmp_path / "policy.json", revision=2)
    campaign = _campaign(tmp_path, purpose="pilot", seed_start=375, path_reference_index=1)
    report = evaluator.evaluate(policy, campaign, tmp_path / "explicit_path_reference")
    screening = cast(dict[str, Any], report["pilot_configuration_screening"])
    assert screening["candidate"]["status"] == "PASS"
    assert screening["candidate"]["path_sensitivity"]["reference_level"] == "level_1"
    assert screening["candidate"]["reference_level"] == "level_2"


def test_v3_path_sensitivity_requires_finest_macro_reference(tmp_path: Path) -> None:
    policy = _policy(tmp_path / "policy.json")
    campaign = _campaign(tmp_path, purpose="pilot", seed_start=380, path_reference_index=1)
    with pytest.raises(ValueError, match="must reference the finest macro level"):
        evaluator.evaluate(policy, campaign, tmp_path / "wrong_path_reference")


def test_final_seed_sets_must_be_disjoint(tmp_path: Path) -> None:
    policy = _policy(tmp_path / "policy.json")
    campaign = _campaign(tmp_path, purpose="final", seed_start=1000)
    raw = json.loads(campaign.read_text(encoding="utf-8"))
    candidate = raw["participants"]["candidate"]["levels"][0]["replicas"]
    for index, replica in enumerate(candidate):
        replica["seed"] = 1000 + index
    _write_json(campaign, raw)
    with pytest.raises(ValueError, match="seed sets must be disjoint"):
        evaluator.evaluate(policy, campaign, tmp_path / "overlap")


def test_pilot_and_final_seed_sets_must_be_disjoint(tmp_path: Path) -> None:
    policy = _policy(tmp_path / "policy.json")
    _, pilot_report = _pilot(policy, tmp_path, seed_start=1000)
    campaign = _campaign(tmp_path, purpose="final", seed_start=1000)
    with pytest.raises(ValueError, match="pilot and final seed sets overlap"):
        evaluator.evaluate(
            policy,
            campaign,
            tmp_path / "phase_seed_overlap",
            pilot_report_path=pilot_report,
        )


def test_v1_policy_cannot_authorize_confirmatory_final(tmp_path: Path) -> None:
    policy = _policy(tmp_path / "policy_v1.json", revision=1)
    campaign = _campaign(tmp_path, purpose="final", seed_start=1000)
    with pytest.raises(ValueError, match="revision 1 cannot authorize"):
        evaluator.evaluate(policy, campaign, tmp_path / "v1_final")


def test_final_requires_post_pilot_authorization_receipt(tmp_path: Path) -> None:
    policy = _policy(tmp_path / "policy.json")
    _, pilot_report = _pilot(policy, tmp_path, seed_start=450)
    campaign = _campaign(tmp_path, purpose="final", seed_start=1000)
    with pytest.raises(ValueError, match="requires --authorization-receipt"):
        evaluator.evaluate(
            policy,
            campaign,
            tmp_path / "missing_authorization",
            pilot_report_path=pilot_report,
        )


def test_authorization_selected_settings_tamper_fails_closed(tmp_path: Path) -> None:
    policy = _policy(tmp_path / "policy.json")
    _, pilot_report = _pilot(policy, tmp_path, seed_start=475)
    campaign = _campaign(tmp_path, purpose="final", seed_start=1000)
    receipt = _authorization(policy, pilot_report, campaign)
    raw = json.loads(receipt.read_text(encoding="utf-8"))
    raw["selected_final_levels"]["candidate"]["numerical_setting"]["step_s"] = 0.005
    _write_json(receipt, raw)
    with pytest.raises(ValueError, match="selected levels differ"):
        _evaluate_final(
            policy,
            campaign,
            tmp_path / "tampered_authorization",
            pilot_report,
            authorization=receipt,
        )


def test_t0_roundoff_difference_passes_source_identity(tmp_path: Path) -> None:
    policy = _policy(tmp_path / "policy.json")
    _, pilot_report = _pilot(policy, tmp_path, seed_start=485)
    campaign = _campaign(tmp_path, purpose="final", seed_start=1000)
    _shift_candidate_t0_velocity(campaign, 2.0e-15)
    report = _evaluate_final(policy, campaign, tmp_path / "source_roundoff", pilot_report)
    identity = cast(dict[str, Any], report["source_identity"])
    assert identity["status"] == "PASS"
    assert identity["maximum_absolute_dynamic_difference_by_component"]["velocity_r_m_per_s"] > 0.0
    assert identity["maximum_roundoff_budget_fraction"] < 1.0


def test_material_t0_source_change_fails_even_when_terminal_gate_passes(
    tmp_path: Path,
) -> None:
    policy = _policy(tmp_path / "policy.json")
    _, pilot_report = _pilot(policy, tmp_path, seed_start=490)
    campaign = _campaign(tmp_path, purpose="final", seed_start=1000)
    _shift_candidate_t0_velocity(campaign, 1.0e-4)
    report = _evaluate_final(policy, campaign, tmp_path / "source_mismatch", pilot_report)
    terminal_gate = cast(dict[str, Any], report["terminal_population_gate"])
    source_identity = cast(dict[str, Any], report["source_identity"])
    assert terminal_gate["status"] == "PASS"
    assert source_identity["status"] == "FAIL"
    assert report["status"] == "FAIL"


def test_changed_artifact_hash_fails_before_output(tmp_path: Path) -> None:
    policy = _policy(tmp_path / "policy.json")
    campaign = _campaign(tmp_path, purpose="pilot", seed_start=500)
    raw = json.loads(campaign.read_text(encoding="utf-8"))
    raw["participants"]["comsol"]["levels"][0]["replicas"][0]["events"]["sha256"] = "0" * 64
    _write_json(campaign, raw)
    output = tmp_path / "must_not_exist"
    with pytest.raises(ValueError, match="events SHA-256 differs"):
        evaluator.evaluate(policy, campaign, output)
    assert not output.exists()


def test_sparse_escaped_suffix_is_inferred_only_from_terminal_event(tmp_path: Path) -> None:
    policy = _policy(tmp_path / "policy.json")
    campaign = _campaign(tmp_path, purpose="pilot", seed_start=550)
    _rewrite_sparse_escape(campaign)
    report = evaluator.evaluate(policy, campaign, tmp_path / "sparse_escape")
    assert report["status"] == "PASS"


@pytest.mark.parametrize("event_type", ["terminal_status", "terminal_boundary"])
def test_population_status_does_not_invent_unobserved_boundary_identity(
    tmp_path: Path, event_type: str
) -> None:
    policy = _policy(tmp_path / "policy.json")
    campaign = _campaign(tmp_path, purpose="pilot", seed_start=550)
    _rewrite_sparse_escape(campaign)
    raw = json.loads(campaign.read_text(encoding="utf-8"))
    for participant in raw["participants"].values():
        for level in participant["levels"]:
            for replica in level["replicas"]:
                events = campaign.parent / replica["events"]["path"]
                events.write_text(
                    "particle_id,event_time_s,event_type,outcome,boundary_semantic\n"
                    f"3,0.25,{event_type},escaped,\n",
                    encoding="utf-8",
                )
                replica["events"]["sha256"] = _sha256(events)
    _write_json(campaign, raw)
    if event_type == "terminal_boundary":
        with pytest.raises(ValueError, match="boundary_semantic is empty"):
            evaluator.evaluate(policy, campaign, tmp_path / "population")
    else:
        report = evaluator.evaluate(policy, campaign, tmp_path / "population")
        assert report["status"] == "PASS"


def test_missing_active_row_without_terminal_event_fails_closed(tmp_path: Path) -> None:
    policy = _policy(tmp_path / "policy.json")
    campaign = _campaign(tmp_path, purpose="pilot", seed_start=600)
    raw = json.loads(campaign.read_text(encoding="utf-8"))
    artifact = raw["participants"]["comsol"]["levels"][0]["replicas"][0]["trajectory"]
    trajectory = campaign.parent / artifact["path"]
    with trajectory.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    with trajectory.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            row for row in rows if not (row["particle_id"] == "3" and row["time_s"] == "1.0")
        )
    artifact["sha256"] = _sha256(trajectory)
    _write_json(campaign, raw)
    with pytest.raises(ValueError, match="only an escaped suffix may be omitted"):
        evaluator.evaluate(policy, campaign, tmp_path / "missing_active")


def test_overlapped_external_workload_is_not_measured_performance(tmp_path: Path) -> None:
    performance = tmp_path / "performance.json"
    _write_json(
        performance,
        {
            "wall_time_s": 12.0,
            "peak_rss_bytes": 123456,
            "output_bytes": 2048,
            "particle_count": 287,
            "output_frames": 121,
            "measurement_status": "NON_AUTHORITATIVE_EXTERNAL_WORKLOAD_OVERLAP",
            "non_authoritative_reason": "overlapped independent workload",
            "memory_plan": {"planned_bytes": 4096},
        },
    )
    loaded = evaluator._load_performance(_artifact(performance), tmp_path, "performance")
    assert loaded is not None
    assert loaded["measurement_status"] == "NOT_MEASURED"
    assert loaded["reason"] == "overlapped independent workload"
