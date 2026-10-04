from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import pytest
from tools.vv.comsol import evaluate_m3c1_theory_100nm_30ms as evaluate


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _config(root: Path) -> Path:
    path = root / "config.json"
    _write_json(
        path,
        {
            "schema_version": 1,
            "evaluation_id": "M3-C1-theory-consistent-100nm-30ms",
            "evaluation_revision": 2,
            "matrix": {
                "particle_count": 2,
                "time_end_s": 0.002,
                "run_keys": ["coarse", "medium", "fine"],
                "output_schedule_segments": [{"start_s": 0.0, "stop_s": 0.002, "step_s": 0.001}],
                "output_count": 3,
            },
            "workflows": {
                "caseA": {
                    "fixed_rk4_steps_s": [0.0004, 0.0002, 0.0001],
                    "charge_lipschitz_s_inv": 100.0,
                },
                "caseP": {
                    "fixed_rk4_steps_s": [0.001, 0.0005, 0.00025],
                    "charge_lipschitz_s_inv": 100.0,
                },
            },
            "acceptance": {
                "initial_state_ulp_multiplier": 4096.0,
                "minimum_rms_order": 0.75,
                "roundoff_multiplier": 4096.0,
                "cross_envelope_safety_factor": 2.0,
                "maximum_dt_charge_lipschitz": 0.5,
            },
        },
    )
    return path


def _state_record(particle: int, frame: int, offset: float, status: int, final: int) -> list[float]:
    time_s = frame * 0.001
    initial_r = 0.1 + particle * 0.01
    initial_z = 0.02 + particle * 0.002
    r_m = initial_r + (0.1 + offset) * time_s
    z_m = initial_z + (0.2 - offset) * time_s
    vr = 0.1 + offset * time_s
    vz = 0.2 - offset * time_s
    charge = -float(particle) + offset * time_s
    if status == 4:
        r_m = z_m = vr = vz = charge = math.nan
    return [
        particle,
        time_s,
        r_m,
        z_m,
        vr,
        vz,
        charge,
        status,
        final,
        math.nan if final == 1 else 0.0015,
        offset,
        1.0e-18,
        0.1,
        0.2,
        -float(particle),
    ]


def _reference_event_row(particle: int, offset: float) -> list[object]:
    held = particle == 1
    if held:
        r_m = 0.1 + particle * 0.01 + (0.1 + offset) * 0.002
        z_m = 0.02 + particle * 0.002 + (0.2 - offset) * 0.002
        charge = -particle + offset * 0.002
        observed = (r_m, z_m, charge)
    else:
        observed = ("", "", "")
    return [
        particle,
        0,
        0.0015,
        0.002,
        "held" if held else "escaped",
        "gas_inlet" if held else "pump_outlet",
        "terminal_state_directly_observed" if held else "NOT_DIRECTLY_OBSERVED",
        *observed,
    ]


def _write_reference_case(
    root: Path,
    config: Path,
    workflow: str,
    offsets: tuple[float, float, float],
    *,
    terminal_events: bool = True,
) -> None:
    root.mkdir(parents=True)
    input_sha = f"input-{workflow}"
    content_hash = f"content-{workflow}"
    config_hash = _sha256(config)
    raw_config = json.loads(config.read_text(encoding="utf-8"))
    workflow_config = raw_config["workflows"][workflow]
    steps = tuple(float(value) for value in workflow_config["fixed_rk4_steps_s"])
    normalized_runs: dict[str, object] = {}
    for run_key, step_s, offset in zip(evaluate.RUN_KEYS, steps, offsets, strict=True):
        trajectory_path = root / f"reference_trajectory_{run_key}.csv"
        with trajectory_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(evaluate.TRAJECTORY_COLUMNS)
            for frame in range(3):
                for particle in (1, 2):
                    final = 1 if not terminal_events else (2 if particle == 1 else 4)
                    current = final if terminal_events and frame == 2 else 1
                    state = _state_record(particle, frame, offset, current, final)
                    payload: tuple[object, ...]
                    if current == 4:
                        payload = ("", "", "", "", "")
                    else:
                        payload = tuple(state[2:7])
                    status = {1: "active", 2: "held", 4: "escaped"}[current]
                    final_status = {1: "active", 2: "held", 4: "escaped"}[final]
                    writer.writerow(
                        (
                            particle,
                            state[1],
                            *payload,
                            status,
                            final_status,
                            "" if final == 1 else 0.0015,
                        )
                    )
        event_path = root / f"reference_events_{run_key}.csv"
        with event_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(evaluate.EVENT_COLUMNS)
            if terminal_events:
                writer.writerows(_reference_event_row(particle, offset) for particle in (1, 2))
        normalized_runs[run_key] = {
            "step_s": step_s,
            "trajectory": trajectory_path.name,
            "trajectory_sha256": _sha256(trajectory_path),
            "event_ledger": event_path.name,
            "event_ledger_sha256": _sha256(event_path),
        }
    _write_json(
        root / "reference_normalization_report.json",
        {
            "schema_version": 1,
            "tool_revision": "m3c1_theory_100nm_30ms_reference_normalizer_v3",
            "status": "COMPLETE",
            "workflow": workflow,
            "config": {"path": str(config.resolve()), "sha256": config_hash},
            "common_p1_receipt": {
                "status": "PASS",
                "candidate_input_sha256": input_sha,
                "canonical_content_hash": content_hash,
            },
            "runs": normalized_runs,
        },
    )


def _candidate_event_row(particle: int, offset: float) -> list[object]:
    held = particle == 1
    fate = "held" if held else "escaped"
    law = "hold" if held else "escape"
    r_m = 0.1 + particle * 0.01 + (0.1 + offset) * 0.002
    z_m = 0.02 + particle * 0.002 + (0.2 - offset) * 0.002
    return [
        particle,
        1,
        0.0015,
        r_m,
        z_m,
        1.0,
        0.0,
        0.1,
        0.2,
        0.1,
        0.2,
        -particle,
        -particle + offset * 0.002,
        37 if held else 35,
        0,
        law,
        fate,
        1.0e-14,
        1.0e-12,
        1.0e-12,
    ]


def _write_candidate_case(
    root: Path,
    workflow: str,
    offsets: tuple[float, float, float],
    *,
    terminal_events: bool = True,
) -> None:
    directory = root / workflow
    directory.mkdir(parents=True)
    for label, offset in zip(evaluate.STEP_LABELS, offsets, strict=True):
        trajectory_path = directory / f"candidate_trajectory_{label}.csv"
        with trajectory_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(evaluate.CANDIDATE_COLUMNS)
            for frame in range(3):
                for particle in (1, 2):
                    if not terminal_events:
                        row = _state_record(particle, frame, offset, 1, 1)
                        writer.writerow((particle, row[1], *row[2:7], "active"))
                        continue
                    final = "held" if particle == 1 else "escaped"
                    if final == "escaped" and frame == 2:
                        continue
                    status = final if frame == 2 else "active"
                    row = _state_record(particle, frame, offset, 1, 1)
                    writer.writerow((particle, row[1], *row[2:7], status))
        event_path = directory / f"candidate_events_{label}.csv"
        with event_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(evaluate.CANDIDATE_EVENT_COLUMNS)
            if terminal_events:
                writer.writerows(_candidate_event_row(particle, offset) for particle in (1, 2))


def _write_candidate_report(root: Path, config: Path) -> None:
    candidate_workflows = {}
    for workflow in evaluate.WORKFLOWS:
        runs = {}
        workflow_root = root / workflow
        for run_key, step_s in zip(
            evaluate.RUN_KEYS,
            evaluate._load_config(config).steps_s_by_workflow[workflow],
            strict=True,
        ):
            trajectory_path = workflow_root / f"candidate_trajectory_{run_key}.csv"
            event_path = workflow_root / f"candidate_events_{run_key}.csv"
            runs[run_key] = {
                "status": "COMPLETE",
                "workflow": workflow,
                "step_label": run_key,
                "dt_s": step_s,
                "trajectory": trajectory_path.name,
                "trajectory_sha256": _sha256(trajectory_path),
                "events": event_path.name,
                "events_sha256": _sha256(event_path),
            }
        candidate_workflows[workflow] = {
            "status": "COMPLETE",
            "input_content_hash": f"content-{workflow}",
            "input_sha256": f"input-{workflow}",
            "runs": runs,
        }
    _write_json(
        root / "candidate_run_report.json",
        {
            "status": "COMPLETE",
            "configuration_sha256": _sha256(config),
            "workflows": candidate_workflows,
        },
    )


def _rewrite_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _make_terminal_suffixes_sparse(root: Path, event_times: tuple[float, float, float]) -> None:
    for workflow in evaluate.WORKFLOWS:
        for label, event_time in zip(evaluate.STEP_LABELS, event_times, strict=True):
            trajectory_path = root / workflow / f"candidate_trajectory_{label}.csv"
            with trajectory_path.open(encoding="utf-8", newline="") as stream:
                trajectory_rows = list(csv.DictReader(stream, strict=True))
            _rewrite_csv(
                trajectory_path,
                [
                    row
                    for row in trajectory_rows
                    if not (row["particle_id"] == "2" and float(row["time_s"]) >= 0.002)
                ],
            )
            event_path = root / workflow / f"candidate_events_{label}.csv"
            with event_path.open(encoding="utf-8", newline="") as stream:
                event_rows = list(csv.DictReader(stream, strict=True))
            for row in event_rows:
                row["event_time_s"] = str(event_time)
            _rewrite_csv(event_path, event_rows)


def _matrix(
    tmp_path: Path,
    *,
    candidate_offsets: tuple[float, float, float] = (4.0e-4, 1.0e-4, 2.5e-5),
    terminal_events: bool = True,
) -> tuple[Path, Path, Path]:
    config = _config(tmp_path)
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    for workflow in evaluate.WORKFLOWS:
        _write_reference_case(
            reference / workflow,
            config,
            workflow,
            (8.0e-4, 2.0e-4, 5.0e-5),
            terminal_events=terminal_events,
        )
        _write_candidate_case(
            candidate,
            workflow,
            candidate_offsets,
            terminal_events=terminal_events,
        )
    _write_candidate_report(candidate, config)
    return config, candidate, reference


def _step_sizes(report: dict[str, object], workflow: str) -> list[float]:
    workflows = report["workflows"]
    assert isinstance(workflows, dict)
    workflow_report = workflows[workflow]
    assert isinstance(workflow_report, dict)
    sequence = workflow_report["step_sequence"]
    assert isinstance(sequence, list)
    steps: list[float] = []
    for row in sequence:
        assert isinstance(row, dict)
        steps.append(float(row["step_s"]))
    return steps


def _assert_formula_claims(result: dict[str, object]) -> None:
    claims = result["claim_status"]
    assert isinstance(claims, dict)
    assert claims["configured_formula_identity"] == {
        "status": "SHARED_DECLARATION_CONFIRMED",
        "basis": "shared_configuration_and_common_p1_input_identity",
        "pointwise_rhs_parity_implied": False,
    }
    assert claims["pointwise_rhs_formula_parity"] == "NOT_SEPARATELY_ESTABLISHED"
    assert "numerical_formula_parity" not in claims
    assert claims["caseP_product_field_adapter"] == "NOT_TESTED"


def _assert_nonempty_event_outputs(output: Path) -> None:
    case_a = json.loads((output / "caseA_metrics.json").read_text(encoding="utf-8"))
    case_p = json.loads((output / "caseP_metrics.json").read_text(encoding="utf-8"))
    assert [row["step_s"] for row in case_a["step_sequence"]] == [0.0004, 0.0002, 0.0001]
    assert [row["step_s"] for row in case_p["step_sequence"]] == [0.001, 0.0005, 0.00025]
    assert case_a["event_metrics"]["escape_position"] == {
        "status": "NOT_DIRECTLY_OBSERVED",
        "count": 1,
        "gated": False,
    }
    assert case_a["trajectory_metrics"]["trajectory_rms_definition"] == {
        "aggregation": (
            "unweighted_over_shared_active_particle_records_at_configured_scheduled_frames"
        ),
        "configured_scheduled_frame_count": 3,
        "time_interval_weighted": False,
        "time_integrated": False,
        "continuous_trajectory_metric": False,
    }
    readme = (output / "README.md").read_text(encoding="utf-8")
    assert "unweighted across shared-active particle samples" in readme
    assert "neither time-integrated nor continuous-trajectory RMS" in readme
    gate_rows = list(csv.DictReader((output / "gates.csv").open(encoding="utf-8")))
    exact_event_metrics = {"identity", "fate", "order", "group", "semantic"}
    exact_event_gates = [
        row
        for row in gate_rows
        if row["category"] == "event" and row["metric"] in exact_event_metrics
    ]
    assert len(exact_event_gates) == 5 * len(evaluate.WORKFLOWS)
    assert all(row["status"] == "PASS" for row in exact_event_gates)
    population_path = output / "caseA_population_survival.csv"
    assert sum(1 for _ in csv.DictReader(population_path.open())) == 3


def _assert_zero_event_characterization(report: dict[str, object]) -> None:
    assert report["status"] == "PASS"
    workflows = report["workflows"]
    assert isinstance(workflows, dict)
    for workflow in evaluate.WORKFLOWS:
        workflow_report = workflows[workflow]
        assert isinstance(workflow_report, dict)
        events = workflow_report["event_self_convergence"]
        assert events["status"] == "NOT_APPLICABLE_NO_EVENTS"
        assert events["event_counts"] == {"coarse": 0, "medium": 0, "fine": 0}
        assert events["boundary_accuracy_evidence"] is False


def _assert_zero_event_outputs(output: Path) -> None:
    for workflow in evaluate.WORKFLOWS:
        metrics = json.loads((output / f"{workflow}_metrics.json").read_text(encoding="utf-8"))
        assert metrics["event_metrics"]["evidence_classification"] == {
            "status": "NOT_APPLICABLE_NO_EVENTS",
            "candidate_event_count": 0,
            "reference_event_count": 0,
            "boundary_accuracy_evidence": False,
        }
        assert metrics["status"] == "PASS"
        assert metrics["final_fates"]["particle_identity_exact"] is True
    gate_rows = list(csv.DictReader((output / "gates.csv").open(encoding="utf-8")))
    zero_event_gates = [row for row in gate_rows if row["category"] == "event"]
    assert len(zero_event_gates) == 5 * len(evaluate.WORKFLOWS)
    assert {row["metric"] for row in zero_event_gates} == {
        "identity",
        "fate",
        "order",
        "group",
        "semantic",
    }
    assert all(row["status"] == "NOT_APPLICABLE_NO_EVENTS" for row in zero_event_gates)
    assert all(row["observed"] == "NO_EVENTS" for row in zero_event_gates)
    assert all(row["limit"] == "NOT_APPLICABLE" for row in zero_event_gates)
    assert all("vacuous zero-event" in row["notes"] for row in zero_event_gates)


def test_formal_phases_pass_and_keep_escape_positions_unobserved(tmp_path: Path) -> None:
    config, candidate, reference = _matrix(tmp_path)
    candidate_characterization = tmp_path / "candidate_characterization.json"
    reference_characterization = tmp_path / "reference_characterization.json"
    budget = tmp_path / "comparison_budget.json"
    evaluate.characterize(config, "candidate", candidate, candidate_characterization)
    evaluate.characterize(config, "reference", reference, reference_characterization)
    registration = evaluate.register(
        config, candidate_characterization, reference_characterization, budget
    )
    assert registration["status"] == "REGISTERED"
    assert registration["registration_policy"]["safety_factor"] == 2.0
    assert registration["registration_policy"]["post_t0_cross_state_values_read"] is False
    candidate_report = json.loads(candidate_characterization.read_text(encoding="utf-8"))
    assert _step_sizes(candidate_report, "caseA") == [0.0004, 0.0002, 0.0001]
    assert _step_sizes(candidate_report, "caseP") == [0.001, 0.0005, 0.00025]

    output = tmp_path / "evidence"
    result = evaluate.compare(config, budget, output)

    assert result["status"] == "PASS"
    assert result["failed_gate_count"] == 0
    assert result["tool_revision"] == "m3c1_theory_100nm_30ms_evaluator_v4"
    _assert_formula_claims(result)
    _assert_nonempty_event_outputs(output)


def test_zero_events_are_not_applicable_without_blocking_state_and_fate(tmp_path: Path) -> None:
    config, candidate, reference = _matrix(tmp_path, terminal_events=False)
    candidate_characterization = tmp_path / "candidate_characterization.json"
    reference_characterization = tmp_path / "reference_characterization.json"
    candidate_report = evaluate.characterize(
        config, "candidate", candidate, candidate_characterization
    )
    reference_report = evaluate.characterize(
        config, "reference", reference, reference_characterization
    )

    _assert_zero_event_characterization(candidate_report)
    _assert_zero_event_characterization(reference_report)

    budget = tmp_path / "comparison_budget.json"
    assert (
        evaluate.register(config, candidate_characterization, reference_characterization, budget)[
            "status"
        ]
        == "REGISTERED"
    )
    output = tmp_path / "evidence"
    result = evaluate.compare(config, budget, output)

    assert result["status"] == "PASS"
    _assert_zero_event_outputs(output)


def test_sparse_terminal_suffixes_use_events_and_common_active_states(tmp_path: Path) -> None:
    config, candidate, reference = _matrix(tmp_path)
    _make_terminal_suffixes_sparse(candidate, (0.0012, 0.0014, 0.0015))
    _write_candidate_report(candidate, config)
    candidate_characterization = tmp_path / "candidate_characterization.json"
    reference_characterization = tmp_path / "reference_characterization.json"
    candidate_report = evaluate.characterize(
        config, "candidate", candidate, candidate_characterization
    )
    evaluate.characterize(config, "reference", reference, reference_characterization)

    assert candidate_report["status"] == "PASS"
    convergence = candidate_report["workflows"]["caseA"]["state_self_convergence"]
    assert convergence["shared_pre_terminal_active_records"] == 4
    assert convergence["shared_pre_terminal_active_records_by_frame"] == [2, 2, 0]
    assert convergence["coarse_medium"]["position"]["count"] == 4
    assert convergence["medium_fine"]["position"]["count"] == 4
    assert convergence["comparison_population"] == (
        "exact three-run intersection of observed active particle records at configured scheduled "
        "frames; RMS is unweighted by time interval"
    )
    assert convergence["trajectory_coverage"]["coarse"] == {
        "observed_records": 5,
        "active_records": 4,
        "terminal_particles": 2,
    }

    budget = tmp_path / "comparison_budget.json"
    registration = evaluate.register(
        config, candidate_characterization, reference_characterization, budget
    )
    assert registration["status"] == "REGISTERED"
    output = tmp_path / "evidence"
    result = evaluate.compare(config, budget, output)
    assert result["status"] == "PASS"
    case_a = json.loads((output / "caseA_metrics.json").read_text(encoding="utf-8"))
    assert case_a["shared_pre_terminal_active_records_by_frame"] == [2, 2, 0]
    rows = list(csv.DictReader((output / "caseA_population_survival.csv").open()))
    assert rows[-1]["candidate_held"] == "1"
    assert rows[-1]["candidate_escaped"] == "1"


def test_stick_terminal_tail_is_retained_and_validated_by_event(tmp_path: Path) -> None:
    config, candidate, _ = _matrix(tmp_path)
    _make_terminal_suffixes_sparse(candidate, (0.0012, 0.0014, 0.0015))
    for workflow in evaluate.WORKFLOWS:
        for label in evaluate.STEP_LABELS:
            event_path = candidate / workflow / f"candidate_events_{label}.csv"
            with event_path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream, strict=True))
            particle_one = next(row for row in rows if row["particle_id"] == "1")
            particle_one["law"] = "stick"
            particle_one["outcome"] = "stuck"
            _rewrite_csv(event_path, rows)
            trajectory_path = candidate / workflow / f"candidate_trajectory_{label}.csv"
            with trajectory_path.open(encoding="utf-8", newline="") as stream:
                trajectory_rows = list(csv.DictReader(stream, strict=True))
            terminal = next(
                row
                for row in trajectory_rows
                if row["particle_id"] == "1" and float(row["time_s"]) == 0.002
            )
            terminal["lifecycle"] = "stuck"
            _rewrite_csv(trajectory_path, trajectory_rows)
    _write_candidate_report(candidate, config)

    report = evaluate.characterize(
        config, "candidate", candidate, tmp_path / "candidate_characterization.json"
    )

    assert report["status"] == "PASS"
    assert (
        report["workflows"]["caseA"]["state_self_convergence"]["trajectory_coverage"]["fine"][
            "terminal_particles"
        ]
        == 2
    )


def test_stick_terminal_tail_cannot_be_omitted(tmp_path: Path) -> None:
    config, candidate, _ = _matrix(tmp_path)
    event_path = candidate / "caseA" / "candidate_events_coarse.csv"
    with event_path.open(encoding="utf-8", newline="") as stream:
        event_rows = list(csv.DictReader(stream, strict=True))
    particle_one = next(row for row in event_rows if row["particle_id"] == "1")
    particle_one["law"] = "stick"
    particle_one["outcome"] = "stuck"
    _rewrite_csv(event_path, event_rows)
    trajectory_path = candidate / "caseA" / "candidate_trajectory_coarse.csv"
    with trajectory_path.open(encoding="utf-8", newline="") as stream:
        trajectory_rows = list(csv.DictReader(stream, strict=True))
    _rewrite_csv(
        trajectory_path,
        [
            row
            for row in trajectory_rows
            if not (row["particle_id"] == "1" and float(row["time_s"]) == 0.002)
        ],
    )
    _write_candidate_report(candidate, config)

    with pytest.raises(ValueError, match="retained terminal trajectory has a missing suffix"):
        evaluate.characterize(
            config, "candidate", candidate, tmp_path / "candidate_characterization.json"
        )


def test_escape_event_must_follow_last_present_active_frame(tmp_path: Path) -> None:
    config, candidate, _ = _matrix(tmp_path)
    event_path = candidate / "caseA" / "candidate_events_coarse.csv"
    with event_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, strict=True))
    escaped = next(row for row in rows if row["particle_id"] == "2")
    escaped["event_time_s"] = "0.001"
    _rewrite_csv(event_path, rows)
    _write_candidate_report(candidate, config)

    with pytest.raises(ValueError, match="lifecycle disagrees at terminal time"):
        evaluate.characterize(
            config, "candidate", candidate, tmp_path / "candidate_characterization.json"
        )


def test_interior_trajectory_hole_is_rejected(tmp_path: Path) -> None:
    config, candidate, _ = _matrix(tmp_path)
    path = candidate / "caseA" / "candidate_trajectory_coarse.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, strict=True))
    _rewrite_csv(
        path,
        [row for row in rows if not (row["particle_id"] == "1" and float(row["time_s"]) == 0.001)],
    )
    _write_candidate_report(candidate, config)

    with pytest.raises(ValueError, match="interior trajectory hole"):
        evaluate.characterize(
            config, "candidate", candidate, tmp_path / "candidate_characterization.json"
        )


def test_terminal_particle_must_still_be_present_at_t0(tmp_path: Path) -> None:
    config, candidate, _ = _matrix(tmp_path)
    path = candidate / "caseA" / "candidate_trajectory_coarse.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, strict=True))
    _rewrite_csv(
        path,
        [row for row in rows if not (row["particle_id"] == "1" and float(row["time_s"]) == 0.0)],
    )
    _write_candidate_report(candidate, config)

    with pytest.raises(ValueError, match="all particles must be present and active at t0"):
        evaluate.characterize(
            config, "candidate", candidate, tmp_path / "candidate_characterization.json"
        )


def test_nonterminal_missing_suffix_is_rejected(tmp_path: Path) -> None:
    config, candidate, _ = _matrix(tmp_path)
    trajectory_path = candidate / "caseA" / "candidate_trajectory_coarse.csv"
    with trajectory_path.open(encoding="utf-8", newline="") as stream:
        trajectory_rows = list(csv.DictReader(stream, strict=True))
    _rewrite_csv(
        trajectory_path,
        [
            row
            for row in trajectory_rows
            if not (row["particle_id"] == "1" and float(row["time_s"]) == 0.002)
        ],
    )
    event_path = candidate / "caseA" / "candidate_events_coarse.csv"
    with event_path.open(encoding="utf-8", newline="") as stream:
        event_rows = list(csv.DictReader(stream, strict=True))
    _rewrite_csv(event_path, [row for row in event_rows if row["particle_id"] != "1"])
    _write_candidate_report(candidate, config)

    with pytest.raises(ValueError, match="non-terminal trajectory has a missing suffix"):
        evaluate.characterize(
            config, "candidate", candidate, tmp_path / "candidate_characterization.json"
        )


def test_failed_self_convergence_blocks_budget_registration(tmp_path: Path) -> None:
    config, candidate, reference = _matrix(tmp_path, candidate_offsets=(4.0e-4, 3.0e-4, 2.0e-4))
    candidate_characterization = tmp_path / "candidate_characterization.json"
    reference_characterization = tmp_path / "reference_characterization.json"
    candidate_report = evaluate.characterize(
        config, "candidate", candidate, candidate_characterization
    )
    evaluate.characterize(config, "reference", reference, reference_characterization)

    assert candidate_report["status"] == "FAIL"
    registration = evaluate.register(
        config,
        candidate_characterization,
        reference_characterization,
        tmp_path / "comparison_budget.json",
    )
    assert registration["status"] == "BLOCKED"
