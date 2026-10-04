from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1]
JAVA = TOOLS / "comsol" / "RunM3C2StochasticCampaign.java"
VALIDATION_JAVA = TOOLS / "comsol" / "RunM3C2StochasticRunnerValidation.java"
RUNNER = TOOLS / "run_m3c2_comsol_campaign.ps1"
NORMALIZER = TOOLS / "normalize_m3c2_comsol_pilot.py"


def _normalizer_module():
    spec = importlib.util.spec_from_file_location("m3c2_comsol_normalizer", NORMALIZER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_java_runner_locks_stochastic_meaning_and_no_save_policy() -> None:
    source = JAVA.read_text(encoding="utf-8")
    assert source.count("ModelUtil.loadCopy(") == 1
    assert ".save(" not in source
    assert 'physics.prop("RandomNumberArgs").set("RandomNumberArgs", "UserDefined")' in source
    assert 'brownian.set("i", RunM3C2StochasticRequest.seedParameter())' in source
    assert 'brownian.set("mu", RunM3C2StochasticRequest.viscosityExpression())' in source
    assert (
        'brownian.set("minput_temperature", RunM3C2StochasticRequest.temperatureExpression())'
        in source
    )
    assert "RunM3C2StochasticRequest.positionRExpression()" in source
    assert "RunM3C2StochasticRequest.positionZExpression()" in source
    assert "range(6e-3[s],1e-3[s],3e-2[s])" in source
    assert "RunM3C2StochasticCampaign.runValidation();" in VALIDATION_JAVA.read_text(
        encoding="utf-8"
    )


def test_java_runner_uses_compiled_request_and_fails_closed_on_study_isolation() -> None:
    source = JAVA.read_text(encoding="utf-8")
    assert "RunM3C2StochasticRequest.rows()" in source
    assert "Files.newBufferedReader" not in source
    assert "java.nio.file" not in source
    assert "MAX_REQUESTS = 32" in source
    assert 'step.solveFor("/physics/" + tag)' in source
    assert 'step.solveFor("/multiphysics/" + tag)' in source
    assert "try { step.setSolveFor" not in source
    assert "class SolveForReceipt" not in source
    assert "class StudyRun" not in source
    assert '"solve_for_assertion", "PASS"' in source


def test_wrapper_has_one_explicit_validation_and_authority_inputs() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert '[ValidateSet("RunnerValidation", "FullPilot", "FinalCampaign")]' in source
    assert '"prepare-request"' in source
    assert '"--registration", $CampaignRegistration' in source
    assert '"--pilot-authorization", $PilotAuthorization' in source
    assert "$PilotAuthorizationRequired" in source
    assert "$PilotAuthorizationRelative" in source
    assert (
        "Copy-Item -LiteralPath $PilotAuthorization -Destination $StagedPilotAuthorization"
        in source
    )


def test_wrapper_compiles_the_locked_full_pilot_recipe() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "pilot_authorization = $(" in source
    assert "$ExecutionRequest.steps_ns" in source
    assert '"-inputfile", "`"$ClassFile`""' in source
    assert '"-nosave"' in source
    assert '"-error", "on"' in source
    assert '"-np", "1"' in source
    assert '"RunM3C2StochasticRequest.java"' in source
    assert "$RequestRowsJava = $JavaRequestRows -join" in source
    assert "& $Compiler $StagedRequestJava" in source
    assert "generated_request_java_sha256 = $RequestJavaHash" in source
    assert "-classpathadd $OutputDirectory $StagedValidationJava" in source


def test_wrapper_isolates_and_retains_comsol_internal_diagnostics() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert 'Join-Path $OutputDirectory ".comsol_preferences"' in source
    assert "security.external.filepermission" not in source
    assert '"-prefsdir", "`"$ComsolPreferencesDirectory`""' in source
    assert 'Join-Path $OutputDirectory "comsol_internal.log"' in source
    assert 'Where-Object { $_.Name -notlike "*_render.log" }' in source
    assert 'GetFileName($PreferencesFullPath) -ne ".comsol_preferences"' in source
    assert "Remove-Item -LiteralPath $PreferencesFullPath -Recurse" in source


def test_wrapper_bounds_measurement_and_preserves_inputs() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "Start-Sleep -Milliseconds 250" in source
    assert "sample_history_retained = $false" in source
    assert "-WindowStyle Hidden" in source
    assert "SourceHashAfter -ne $SourceHashBefore" in source
    assert "Confirm-PreparedTableArtifacts" in source
    assert "normalize_m3c2_comsol_pilot.py" in source
    assert "ThreadPoolExecutor" not in source
    assert "Start-Job" not in source


def test_normalizer_schedule_and_exchange_schema_are_locked() -> None:
    module = _normalizer_module()
    times = module._output_times()
    assert len(times) == 121
    assert times[0] == 0.0
    assert times[50] == 5.0e-4
    assert times[51] == 6.0e-4
    assert times[95] == 5.0e-3
    assert times[96] == 6.0e-3
    assert times[-1] == 3.0e-2
    assert module.TRAJECTORY_COLUMNS == (
        "particle_id",
        "time_s",
        "r_m",
        "z_m",
        "velocity_r_m_per_s",
        "velocity_z_m_per_s",
        "charge_number_e",
        "lifecycle",
    )
    assert module.EVENT_COLUMNS == (
        "particle_id",
        "event_time_s",
        "event_type",
        "outcome",
        "boundary_semantic",
    )


def test_normalizer_accepts_only_safe_registered_request_rows(tmp_path: Path) -> None:
    module = _normalizer_module()
    request = tmp_path / "m3c2_pilot_request.csv"
    with request.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("seed", "step_ns", "relative_directory"))
        writer.writerow((918160, 20000, "levels/dt_20us/replicas/seed_918160"))
    assert module._request_rows(request) == [
        {
            "seed": 918160,
            "step_ns": 20000,
            "directory": "levels/dt_20us/replicas/seed_918160",
        }
    ]
    request.write_text(
        "seed,step_ns,relative_directory\n318160,20000,levels/dt_20us/replicas/seed_318161\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match seed and step"):
        module._request_rows(request)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _request_fixture(
    tmp_path: Path, *, referenced_seeds: bool = False
) -> tuple[Path, Path, Path, Path]:
    project = tmp_path / "solver"
    project.mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    (project / "uv.lock").write_text("fixture\n", encoding="utf-8")
    canonical = project / "candidate_input.h5"
    canonical.write_bytes(b"synthetic canonical input")
    canonical_sha256 = _sha256(canonical)
    final_seed_sets = {
        "comsol": list(range(318160, 318192)),
        "candidate": list(range(318192, 318224)),
    }
    seed_reference: dict[str, object] | None = None
    contract_seed_reference: dict[str, object] | None = None
    if referenced_seeds:
        allocation = project / "final_seed_allocation.json"
        allocation.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "allocation_kind": "m3c2_final_seed_allocation",
                    "case_id": "formal_iondrag_theory_consistent/caseA_100nm",
                    "purpose": "final",
                    "replicas_per_participant": 32,
                    "participant_seed_sets": final_seed_sets,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        seed_reference = {
            "path": allocation.name,
            "sha256": _sha256(allocation),
            "json_pointers": {
                "comsol": "/participant_seed_sets/comsol",
                "candidate": "/participant_seed_sets/candidate",
            },
        }
        contract_seed_reference = {**seed_reference, "path": f"solver/{allocation.name}"}
    contract = project / "pilot_contract.json"
    contract.write_text(
        json.dumps(
            {
                "contract_id": "M3-C2A-caseA-100nm-stochastic-pilot",
                "common_p1_input": {
                    "file_sha256": canonical_sha256,
                    "content_hash": f"sha256:{canonical_sha256}",
                },
                "campaign": {
                    "case_id": "formal_iondrag_theory_consistent/caseA_100nm",
                    "evaluation_case_id": "M3-C2A_caseA_100nm_common-P1",
                    "output_slug": "caseA_100nm",
                    "final_registration_kind": "m3c2_caseA_100nm_final_campaign",
                    "candidate_case_name_prefix": "m3c2_caseA_100nm",
                },
                "stochastic_physics": {
                    "comsol": {
                        "physics_tag": "fptas",
                        "background_study": "stdASf",
                        "background_study_step": "stat",
                        "background_solution": "sol26",
                        "shared_variable_tag": "varAS",
                        "viscosity_expression": "root.comp1.AS_muB",
                        "temperature_expression": "root.comp1.AS_Tg",
                        "pressure_expression": "m3c1_pressure",
                        "random_number_args": "UserDefined",
                        "seed_parameter": "AS_brownian_seed",
                        "sole_seed_authority": "fptas.bf1.i",
                        "position_expressions": ["q3r", "q3z"],
                        "velocity_expressions": ["fptas.vr", "fptas.vz"],
                        "charge_state_expression": "ZAS",
                        "particle_geometry": "pgeom_fptas",
                    }
                },
                "numerical_policy": {"comsol_pilot_fixed_steps_s": [2.0e-5, 1.0e-5, 5.0e-6]},
                "seed_plan": {
                    "pilot": {
                        "comsol_seeds": [918160, 918161, 918162, 918163],
                        "candidate_seeds": [918164, 918165, 918166, 918167],
                    },
                    **(
                        {
                            "final_cohort": {
                                "case_id": "formal_iondrag_theory_consistent/caseA_100nm",
                                "replicas_per_participant": 32,
                                "seed_source": contract_seed_reference,
                            }
                        }
                        if contract_seed_reference is not None
                        else {}
                    ),
                },
            }
        ),
        encoding="utf-8",
    )
    contract_document = json.loads(contract.read_text(encoding="utf-8"))
    contract_receipt = project / "contract_receipt.json"
    contract_receipt.write_text(
        json.dumps(
            {
                "contract_validation_status": "PASS_INPUT_IDENTITY_AND_SEMANTICS_LOCKED",
                "comsol_or_candidate_executed": False,
                "contract_sha256": _sha256(contract),
                "campaign": contract_document["campaign"],
            }
        ),
        encoding="utf-8",
    )
    pilot_authorization = project / "pilot_authorization.json"
    pilot_authorization.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "authorization_kind": "m3c2_pilot_execution_authorization",
                "authorization_id": "fixture-opaque-pilot-authorization",
                "status": "AUTHORIZED_BY_EXPLICIT_USER_DIRECTION",
                "contract": {
                    "path": "solver/pilot_contract.json",
                    "sha256": _sha256(contract),
                },
                "contract_receipt": {
                    "path": "solver/contract_receipt.json",
                    "sha256": _sha256(contract_receipt),
                },
                "campaign": contract_document["campaign"],
                "purpose": "pilot",
                "participants": ["comsol", "candidate"],
                "provenance": {"test_fixture": True},
            }
        ),
        encoding="utf-8",
    )
    registration = project / "final_registration.json"
    evaluation_policy = project / "evaluation_policy.json"
    evaluation_policy.write_text(
        json.dumps(
            {
                "policy_kind": "m3c2_stochastic_ensemble",
                "policy_revision": 4,
                "final": {
                    "replicas_per_participant": 32,
                    **(
                        {"seed_allocation": seed_reference}
                        if seed_reference is not None
                        else {"seed_plan": final_seed_sets}
                    ),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    pilot_evaluation = project / "pilot_evaluation.json"
    pilot_evaluation.write_text(
        json.dumps({"phase": "pilot", "status": "PASS"}) + "\n",
        encoding="utf-8",
    )
    selection_receipt = project / "selection_receipt.json"
    selection_receipt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "receipt_kind": "m3c2_post_pilot_final_authorization",
                "status": "AUTHORIZED_FOR_CONFIRMATORY_FINAL",
                "policy_sha256": _sha256(evaluation_policy),
                "pilot_report_sha256": _sha256(pilot_evaluation),
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
                            "purpose": "accepted_final",
                        },
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    final_authorization_artifacts = {
        name: {"path": artifact.name, "sha256": _sha256(artifact)}
        for name, artifact in {
            "evaluation_policy": evaluation_policy,
            "pilot_evaluation": pilot_evaluation,
            "selection_receipt": selection_receipt,
        }.items()
    }
    registration.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "registration_kind": "m3c2_caseA_100nm_final_campaign",
                "case_id": "formal_iondrag_theory_consistent/caseA_100nm",
                "purpose": "final",
                **(
                    {"participant_seed_source": seed_reference}
                    if seed_reference is not None
                    else {"participant_seed_sets": final_seed_sets}
                ),
                "comsol_numerical_setting": {
                    "integrator": "classical_rk4",
                    "fixed_step_s": 2.0e-5,
                },
                "execution_authorization": {
                    "status": "AUTHORIZED",
                    **final_authorization_artifacts,
                },
            }
        ),
        encoding="utf-8",
    )
    output = project / "output"
    output.mkdir()
    return contract, registration, pilot_authorization, output


def test_request_preparation_preserves_pilot_and_registers_final_campaign(
    tmp_path: Path,
) -> None:
    module = _normalizer_module()
    contract, registration, pilot_authorization, output = _request_fixture(tmp_path)
    pilot_output = output / "pilot"
    pilot_output.mkdir()
    pilot = module.prepare_request(
        pilot_output,
        "FullPilot",
        contract,
        pilot_authorization_path=pilot_authorization,
    )
    assert pilot["seeds"] == [918160, 918161, 918162, 918163]
    assert pilot["steps_ns"] == [20000, 10000, 5000]
    assert len(pilot["request_rows"]) == 12
    assert pilot["normalized_manifest"] == "comsol_pilot_manifest.json"
    assert pilot["pilot_authorization"] == {
        "path": "solver/pilot_authorization.json",
        "sha256": _sha256(pilot_authorization),
    }
    assert pilot["campaign_binding"] == {
        "contract_sha256": _sha256(contract),
        "input_sha256": json.loads(contract.read_text())["common_p1_input"]["file_sha256"],
        "input_content_hash": json.loads(contract.read_text())["common_p1_input"]["content_hash"],
    }
    (pilot_output / contract.name).write_bytes(contract.read_bytes())
    (pilot_output / pilot_authorization.name).write_bytes(pilot_authorization.read_bytes())
    (pilot_output / "common_p1_table_receipt.json").write_text(
        json.dumps(
            {
                "candidate": {
                    "file_sha256": pilot["campaign_binding"]["input_sha256"],
                    "content_hash": pilot["campaign_binding"]["input_content_hash"],
                }
            }
        ),
        encoding="utf-8",
    )
    assert module._execution_request(pilot_output) == pilot

    final_output = output / "final"
    final_output.mkdir()
    final = module.prepare_request(final_output, "FinalCampaign", contract, registration)
    assert final["seeds"] == list(range(318160, 318192))
    assert final["steps_ns"] == [20000]
    assert len(final["request_rows"]) == 32
    assert final["normalized_manifest"] == "comsol_campaign_manifest.json"
    assert final["registration"]["sha256"] == _sha256(registration)
    assert set(final["registration"]["authorization_artifacts"]) == {
        "evaluation_policy",
        "pilot_evaluation",
        "selection_receipt",
    }
    (final_output / registration.name).write_bytes(registration.read_bytes())
    (final_output / contract.name).write_bytes(contract.read_bytes())
    (final_output / "common_p1_table_receipt.json").write_text(
        json.dumps(
            {
                "candidate": {
                    "file_sha256": final["campaign_binding"]["input_sha256"],
                    "content_hash": final["campaign_binding"]["input_content_hash"],
                }
            }
        ),
        encoding="utf-8",
    )
    assert module._execution_request(final_output) == final


def test_final_request_resolves_one_hash_locked_seed_allocation(tmp_path: Path) -> None:
    module = _normalizer_module()
    contract, registration, _, output = _request_fixture(tmp_path, referenced_seeds=True)

    final = module.prepare_request(output, "FinalCampaign", contract, registration)

    assert final["seeds"] == list(range(318160, 318192))
    assert final["seed_isolation"]["other_participant_seeds"] == list(range(318192, 318224))


@pytest.mark.parametrize("damage", ["pointer", "inline_duplicate"])
def test_final_request_rejects_noncanonical_seed_source(tmp_path: Path, damage: str) -> None:
    module = _normalizer_module()
    contract, registration, _, output = _request_fixture(tmp_path, referenced_seeds=True)
    document = json.loads(registration.read_text(encoding="utf-8"))
    if damage == "pointer":
        document["participant_seed_source"]["json_pointers"]["comsol"] = "/wrong"
    else:
        document["participant_seed_sets"] = {
            "comsol": list(range(318160, 318192)),
            "candidate": list(range(318192, 318224)),
        }
    registration.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match=r"JSON pointers|one owner"):
        module.prepare_request(output, "FinalCampaign", contract, registration)


def test_final_request_rejects_seed_overlap_and_changed_authorization(
    tmp_path: Path,
) -> None:
    module = _normalizer_module()
    contract, registration, _, output = _request_fixture(tmp_path)
    document = json.loads(registration.read_text(encoding="utf-8"))
    document["participant_seed_sets"]["candidate"][0] = document["participant_seed_sets"]["comsol"][
        0
    ]
    registration.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="seed sets must be disjoint"):
        module.prepare_request(output, "FinalCampaign", contract, registration)

    contract, registration, _, output = _request_fixture(tmp_path / "changed")
    (registration.parent / "selection_receipt.json").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="registered SHA-256"):
        module.prepare_request(output, "FinalCampaign", contract, registration)

    contract, registration, _, output = _request_fixture(tmp_path / "changed-setting")
    document = json.loads(registration.read_text(encoding="utf-8"))
    document["comsol_numerical_setting"]["fixed_step_s"] = 1.0e-5
    registration.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from the pilot selection"):
        module.prepare_request(output, "FinalCampaign", contract, registration)


def test_study_isolation_receipt_is_fail_closed() -> None:
    module = _normalizer_module()
    receipt = {
        "solve_for_assertion": "PASS",
        "solve_for_physics": "es=false,fptas=true,ht=false",
        "solve_for_multiphysics": "emh=false,nonisothermal=false",
    }
    isolation = module._study_isolation(receipt)
    assert isolation["enabled_physics"] == ["fptas"]
    assert isolation["enabled_multiphysics"] == []
    for damaged in (
        {**receipt, "solve_for_physics": "es=true,fptas=true"},
        {**receipt, "solve_for_multiphysics": "emh=true"},
        {**receipt, "solve_for_assertion": "NOT_CHECKED"},
    ):
        with pytest.raises(ValueError, match=r"not isolated|enabled multiphysics"):
            module._study_isolation(damaged)


def _process_receipt(multiphysics: str) -> str:
    fields = {
        "seed": "918160",
        "step_s": "2e-5",
        "random_number_args": "UserDefined",
        "seed_authority": "fptas.bf1.i",
        "brownian_seed_expression": "AS_brownian_seed",
        "brownian_active": "true",
        "out_of_plane": "false",
        "brownian_viscosity": "root.comp1.AS_muB",
        "brownian_temperature": "root.comp1.AS_Tg",
        "saffman_active": "false",
        "dynamic_charge_active": "true",
        "field_source": "canonical_exact_connectivity_P1_sectionwise",
        "initial_state_source": "candidate_realized_source_table",
        "integrator": "classical_rk4",
        "integrator_order": "4",
        "relative_tolerance": "1e-2",
        "wall_accuracy_order": "1",
        "output_times": "121",
        "particle_rows": "287",
        "time_end_s": "0.03",
        "directory": "levels/dt_20us/replicas/seed_918160",
        "source_model": "source_copy.mph",
        "model_saved": "false",
        "solve_for_assertion": "PASS",
        "solve_for_physics": "es=false,fptas=true,ht=false",
        "solve_for_multiphysics": multiphysics,
    }
    configuration = "M3C2_COMSOL|configuration|" + "|".join(
        f"{name}={value}" for name, value in fields.items()
    )
    solve = "M3C2_COMSOL|solve_pass|seed=918160|step_s=2e-5|seconds=2.0"
    return f"{configuration}\n{solve}\n"


def test_receipt_parser_rejects_nonisolated_study(tmp_path: Path) -> None:
    module = _normalizer_module()
    process_log = tmp_path / "comsol_process.log"
    process_log.write_text(_process_receipt("none"), encoding="utf-8")
    configurations, durations = module._receipts(process_log)
    assert set(configurations) == {(918160, 20000)}
    assert durations == {(918160, 20000): 2.0}

    process_log.write_text(_process_receipt("emh=true"), encoding="utf-8")
    with pytest.raises(ValueError, match="enabled multiphysics"):
        module._receipts(process_log)


def _terminal_fields(particle_id: int) -> tuple[float, float]:
    event_time = 0.0015
    if particle_id == 1:
        return 4.0, event_time
    if particle_id == 2:
        return 4.0, 0.030005487330439843
    return 1.0, math.nan


def _synthetic_record(
    module,
    particle_id: int,
    time_s: float,
    release: dict[int, tuple[float, float, float, float, float]],
) -> list[float]:
    event_time = 0.0015
    if particle_id == 1 and time_s >= event_time:
        return [math.nan] * len(module.STATE_COLUMNS)
    final_status, stop_time = _terminal_fields(particle_id)
    return [
        particle_id,
        time_s,
        release[particle_id][0],
        release[particle_id][1],
        0.0,
        0.0,
        0.0,
        1.0,
        final_status,
        stop_time,
    ]


def _write_synthetic_raw(
    module,
    path: Path,
    release: dict[int, tuple[float, float, float, float, float]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        for particle_id in range(1, 288):
            wide: list[float] = []
            for time_s in module._output_times():
                wide.extend(_synthetic_record(module, particle_id, time_s, release))
            writer.writerow(wide)


def _assert_normalized_trajectory(path: Path) -> None:
    event_time = 0.0015
    with path.open(newline="", encoding="utf-8") as stream:
        trajectory = list(csv.DictReader(stream))
    escaped = [row for row in trajectory if row["particle_id"] == "1"]
    suffix = [row for row in escaped if float(row["time_s"]) >= event_time]
    predicted_after_horizon = [row for row in trajectory if row["particle_id"] == "2"]
    assert len(escaped) == 121
    assert suffix and all(row["lifecycle"] == "escaped" for row in suffix)
    assert all(math.isnan(float(row["r_m"])) for row in suffix)
    assert len(predicted_after_horizon) == 121
    assert all(row["lifecycle"] == "active" for row in predicted_after_horizon)


def _assert_normalized_events_and_performance(path: Path) -> None:
    with (path / "events.csv").open(newline="", encoding="utf-8") as stream:
        events = list(csv.DictReader(stream))
    performance = json.loads((path / "performance.json").read_text(encoding="utf-8"))
    assert [row["particle_id"] for row in events] == ["1"]
    assert {
        "wall_time_s": performance["wall_time_s"],
        "peak_rss_bytes": performance["peak_rss_bytes"],
        "particle_count": performance["particle_count"],
        "output_frames": performance["output_frames"],
    } == {
        "wall_time_s": 2.0,
        "peak_rss_bytes": 4096,
        "particle_count": 287,
        "output_frames": 121,
    }
    assert performance["output_bytes"] > 0


def test_normalizer_materializes_escape_and_ignores_post_horizon_prediction(
    tmp_path: Path,
) -> None:
    module = _normalizer_module()
    release = {
        particle_id: (particle_id * 1.0e-6, 0.01, 0.0, 0.0, 0.0) for particle_id in range(1, 288)
    }
    raw_path = tmp_path / "trajectory_raw_wide.csv"
    _write_synthetic_raw(module, raw_path, release)
    summary = module._normalize_replica(
        tmp_path,
        release,
        seed=918160,
        step_ns=20000,
        seconds=2.0,
        peak_rss_bytes=4096,
        configuration={
            "solve_for_assertion": "PASS",
            "solve_for_physics": "es=false,fptas=true,ht=false",
            "solve_for_multiphysics": "none",
        },
    )
    assert {
        "trajectory_rows": summary["trajectory_rows"],
        "event_count": summary["event_count"],
        "final_lifecycle_counts": summary["final_lifecycle_counts"],
        "trajectory_path": summary["trajectory"]["path"],
        "trajectory_hash_length": len(summary["trajectory"]["sha256"]),
        "study_isolation": summary["study_isolation"]["status"],
    } == {
        "trajectory_rows": 287 * 121,
        "event_count": 1,
        "final_lifecycle_counts": {"active": 286, "escaped": 1},
        "trajectory_path": "trajectory.csv",
        "trajectory_hash_length": 64,
        "study_isolation": "PASS",
    }
    module._prefix_artifact_paths(summary, "levels/dt_20us/replicas/seed_918160")
    assert summary["trajectory"]["path"] == "levels/dt_20us/replicas/seed_918160/trajectory.csv"
    _assert_normalized_trajectory(tmp_path / "trajectory.csv")
    _assert_normalized_events_and_performance(tmp_path)
