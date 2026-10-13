"""Campaign-only readback of new RK4 and preserved native/exp-midpoint paths."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from itertools import chain
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from chamber_particles import load_case, open_result

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
OLD = CAMPAIGN.parent / "comsol_speed10000_comparison_2026_10_13"
HELPER_PATH = (
    OLD / "original_comparison/particle_trajectory_review/analyze_particle_trajectories.py"
)
HELPER_SHA = "ef4fe8bd8b0cb715d953711c2a2fe6aaf57670cdbbea58ab2a8a33fabacb163e"
INPUTS: dict[str, dict[str, Any]] = {}
OUTPUTS: dict[str, dict[str, Any]] = {}
COMPARISONS = {
    "rk4_coarse_minus_medium": ("rk4_coarse", "rk4_medium"),
    "rk4_medium_minus_fine": ("rk4_medium", "rk4_fine"),
    "rk4_fine_minus_native10": ("rk4_fine", "native10"),
    "rk4_fine_minus_native5": ("rk4_fine", "native5"),
    "native5_minus_native10": ("native5", "native10"),
    "old_exp_fine_minus_rk4_fine": ("old_exp_fine", "rk4_fine"),
    "old_exp_fine_minus_rk4_medium": ("old_exp_fine", "rk4_medium"),
}


def receipt(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def read_json(path: Path) -> dict[str, Any]:
    INPUTS[str(path.resolve())] = receipt(path)
    return json.loads(path.read_text(encoding="utf-8"))


def verify(binding: dict[str, Any]) -> Path:
    path = Path(binding["path"])
    actual = receipt(path)
    if actual["sha256"] != binding["sha256"]:
        raise ValueError(f"Changed bound source: {path}")
    INPUTS[str(path.resolve())] = actual
    return path


def helpers() -> ModuleType:
    verify({"path": str(HELPER_PATH), "sha256": HELPER_SHA})
    module = ModuleType("bound_saved_trajectory_math")
    module.__file__ = str(HELPER_PATH)
    exec(
        compile(HELPER_PATH.read_text(encoding="utf-8"), str(HELPER_PATH), "exec"), module.__dict__
    )
    return module


def publish(path: Path, data: bytes) -> None:
    if not path.resolve().is_relative_to(HERE):
        raise ValueError("Outputs must remain in the owned trajectory_review directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_bytes() != data:
        temporary = path.with_suffix(path.suffix + ".writing")
        temporary.write_bytes(data)
        temporary.replace(path)
    OUTPUTS[str(path.relative_to(HERE))] = receipt(path)


def write_csv(path: Path, rows: Any) -> None:
    import io

    iterator = iter(rows)
    first = next(iterator)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, list(first), lineterminator="\n")
    writer.writeheader()
    writer.writerow(first)
    writer.writerows(iterator)
    publish(path, stream.getvalue().encode("utf-8"))


def write_json(path: Path, value: Any) -> None:
    publish(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def native_counter_cutoff(series: dict[str, Any]) -> None:
    counter = series["material_counter"]
    finite = np.isfinite(counter)
    if np.any(np.abs(counter[finite] - np.rint(counter[finite])) > 1e-10):
        raise ValueError("Native material counter is not a validated integer count")
    rounded = np.where(finite, np.rint(counter), np.nan)
    lower, upper = np.full(287, np.inf), np.full(287, np.inf)
    for slot in range(287):
        hit = np.flatnonzero(np.isfinite(rounded[:, slot]) & (rounded[:, slot] >= 1))
        if len(hit):
            index = int(hit[0])
            lower[slot], upper[slot] = series["times"][max(0, index - 1)], series["times"][index]
    series["material_counter"] = rounded
    series["first_contact_lower"], series["first_contact_upper"] = lower, upper
    series["precontact_cutoff"] = np.minimum(lower, series["terminal_time"])


def read_series(
    case_plan: dict[str, Any], old: dict[str, Any], audit: dict[str, Any], h: Any
) -> dict[str, Any]:
    name = case_plan["case"]
    mass = audit["cases"][name]["native_constants"]["rho_p*pi*d0^3/6"]
    values = {
        label: h.native_series(
            name, label, old["reference_characterization"][f"{name}_dt_{step}us"], mass
        )
        for label, step in (("native10", 10), ("native5", 5))
    }
    for item in values.values():
        native_counter_cutoff(item)
    times = values["native10"]["times"]
    if not np.allclose(times, case_plan["observation_times_s"], rtol=0, atol=1e-17):
        raise ValueError("Registered and native observation clocks differ")
    old_exp = next(
        row
        for row in old["candidate_runs"]
        if row["case"] == name and row["label"] == "full_v64_h0p25us"
    )
    values["old_exp_fine"] = h.candidate_series(name, "old_exp_fine", old_exp, times)
    values["old_exp_fine"]["source_bindings"] = {
        key: old_exp["source"][key] for key in ("case", "data", "run")
    }
    for label, step in (("native10", 10), ("native5", 5)):
        values[label]["source_bindings"] = old["reference_characterization"][f"{name}_dt_{step}us"][
            "source"
        ]
    for item in case_plan["runs"]:
        output = Path(item["output"])
        if not (output / "_SUCCESS").exists():
            continue
        run_receipt = read_json(output.parent / f"{item['level']}_execution_receipt.json")
        case = load_case(item["case"])
        binding = {
            "source": {
                "case": receipt(case.case_path),
                "data": receipt(case.data_path),
                "run": receipt(output / "run.json"),
            }
        }
        for key in ("case", "data", "run"):
            if binding["source"][key]["sha256"] != run_receipt[f"{key}_sha256"]:
                raise ValueError("Run payload differs from completed execution receipt")
        label = f"rk4_{item['level']}"
        values[label] = h.candidate_series(name, label, binding, times)
        values[label]["source_bindings"] = {
            **binding["source"],
            "execution_receipt": receipt(output.parent / f"{item['level']}_execution_receipt.json"),
        }
        manifest = open_result(output).manifest
        if (
            manifest["resolved"]["integrator"] != "rk4_fixed"
            or values[label]["run_dt_s"] != item["step_s"]
        ):
            raise ValueError("Candidate method/step differs from registered run")
    for label, item in values.items():
        if not np.allclose(item["times"], times, rtol=0, atol=1e-17):
            raise ValueError("Participant observation clocks differ")
        for quantity in ("position", "velocity", "charge"):
            if not np.array_equal(item[quantity][0], values["native10"][quantity][0]):
                raise ValueError(f"Initial source differs: {name}/{label}/{quantity}")
    INPUTS.update(h.INPUTS)
    for label, series in values.items():
        if series["kind"] == "candidate":
            case = load_case(series["source_bindings"]["case"]["path"])
            result = open_result(Path(series["source_bindings"]["run"]["path"]).parent)
            series["events"] = detailed_events(result, case, label)
    return values


def detailed_events(result: Any, case: Any, label: str) -> list[dict[str, Any]]:
    events = result.read_boundary_events()
    rows = []
    for index, pid in enumerate(events.particle_id):
        outcome = str(events.outcome[index])
        rows.append(
            {
                "participant": label,
                "particle_id": int(pid),
                "time_s": float(events.time_s[index]),
                "event_ordinal": int(events.event_ordinal[index]),
                "outcome": outcome,
                "primary_native_boundary_id": int(
                    case.data.geometry.boundary.boundary_id[int(events.primary_facet_id[index])]
                ),
                "material_encounter": outcome in ("stuck", "reflected"),
                "r_m": float(events.position_m[index, 0]),
                "z_m": float(events.position_m[index, 1]),
                "vr_pre_m_s": float(events.velocity_pre_m_s[index, 0]),
                "vz_pre_m_s": float(events.velocity_pre_m_s[index, 1]),
                "vr_post_m_s": float(events.velocity_post_m_s[index, 0]),
                "vz_post_m_s": float(events.velocity_post_m_s[index, 1]),
                "Z_pre": float(events.charge_number_pre[index]),
                "Z_post": float(events.charge_number_post[index]),
                "normal_r": float(events.normal[index, 0]),
                "normal_z": float(events.normal[index, 1]),
                "localization_residual_m": float(events.localization_residual_m[index]),
                "position_budget_m": float(events.position_budget_m[index]),
                "time_budget_s": float(events.time_budget_s[index]),
            }
        )
    return rows


def normalized_rows(series: dict[str, Any], h: Any) -> Any:
    for row in h.state_rows(series):
        slot = row["particle_id"] - 1
        valid = (
            row["state_observed"]
            and row["current_status"] == "active"
            and row["position_finite"]
            and row["velocity_finite"]
            and row["charge_finite"]
        )
        yield {
            "particle_id": row["particle_id"],
            "time_s": row["time_s"],
            "r_m": row["r_m"],
            "z_m": row["z_m"],
            "velocity_r_m_per_s": row["vr_m_s"],
            "velocity_z_m_per_s": row["vz_m_s"],
            "charge_number": row["Z"],
            "lifecycle": row["current_status"],
            "precontact_eligible": int(valid and row["time_s"] < series["precontact_cutoff"][slot]),
            "observed": int(row["state_observed"]),
            "material_counter_value": row["material_counter_value"],
            "material_counter_observed": int(row["material_counter_observed"]),
            "first_material_contact_lower_s": row["first_material_contact_lower_s"],
            "first_material_contact_upper_s": row["first_material_contact_upper_s"],
            "first_candidate_boundary_event_s": row["first_candidate_boundary_event_s"],
            "terminal_time_s": row["terminal_time_s"],
        }


def all_states_rows(values: dict[str, Any], h: Any) -> Any:
    for value in values.values():
        for row in normalized_rows(value, h):
            yield {"participant": value["label"], **row}


def population_rows(series: dict[str, Any]) -> list[dict[str, Any]]:
    counter = np.maximum.accumulate(
        np.where(np.isfinite(series["material_counter"]), series["material_counter"], 0), axis=0
    )
    rows = []
    for index, time in enumerate(series["times"]):
        counts = Counter(series["status"][index].tolist())
        rows.append(
            {
                "participant": series["label"],
                "time_s": float(time),
                "released_denominator": 287,
                **{
                    state: counts[state]
                    for state in (
                        "pending",
                        "active",
                        "stuck",
                        "held",
                        "escaped",
                        "disappeared",
                        "failed",
                        "unobserved",
                    )
                },
                "state_observed": int(series["observed"][index].sum()),
                "material_encounters": int(counter[index].sum()),
                "ever_material_hit": int((counter[index] >= 1).sum()),
                "nonstick_material_encounters": int(counter[index].sum() - counts["stuck"]),
            }
        )
    return rows


def first_hit_rows(left: dict[str, Any], right: dict[str, Any], h: Any) -> list[dict[str, Any]]:
    first = []
    for series in (left, right):
        by_id = {}
        for event in series["events"]:
            if event["material_encounter"]:
                by_id.setdefault(event["particle_id"], event)
        first.append(by_id)
    rows = []
    for slot in range(287):
        a, b = left["first_contact_upper"][slot], right["first_contact_upper"][slot]
        exact = left["kind"] == right["kind"] == "candidate"
        rows.append(
            {
                "particle_id": slot + 1,
                "left_first_material_lower_s": h.finite_value(left["first_contact_lower"][slot]),
                "left_first_material_upper_s": h.finite_value(a),
                "right_first_material_lower_s": h.finite_value(right["first_contact_lower"][slot]),
                "right_first_material_upper_s": h.finite_value(b),
                "candidate_exact_first_hit_pair": bool(exact and np.isfinite(a) and np.isfinite(b)),
                "absolute_candidate_first_hit_time_difference_s": float(abs(a - b))
                if exact and np.isfinite(a) and np.isfinite(b)
                else None,
            }
        )
        aa, bb = first[0].get(slot + 1), first[1].get(slot + 1)
        row = rows[-1]
        row.update(
            {
                "left_primary_boundary_id": aa["primary_native_boundary_id"] if aa else None,
                "right_primary_boundary_id": bb["primary_native_boundary_id"] if bb else None,
                "same_candidate_first_surface_outcome": (
                    aa["primary_native_boundary_id"],
                    aa["outcome"],
                )
                == (bb["primary_native_boundary_id"], bb["outcome"])
                if aa and bb
                else None,
                "candidate_first_hit_position_vector_difference_m": float(
                    np.hypot(aa["r_m"] - bb["r_m"], aa["z_m"] - bb["z_m"])
                )
                if aa and bb
                else None,
                "candidate_first_hit_velocity_pre_vector_difference_m_s": float(
                    np.hypot(
                        aa["vr_pre_m_s"] - bb["vr_pre_m_s"], aa["vz_pre_m_s"] - bb["vz_pre_m_s"]
                    )
                )
                if aa and bb
                else None,
                "candidate_first_hit_absolute_Z_pre_difference": abs(aa["Z_pre"] - bb["Z_pre"])
                if aa and bb
                else None,
            }
        )
    return rows


def aggregate_norms(series: dict[str, Any], arrays: dict[str, Any], h: Any) -> dict[str, Any]:
    rows = {}
    for scope in ("precontact", "transport", "postcontact"):
        rows[scope] = {}
        for quantity in h.FLAG_SCALES:
            mask = arrays[scope] & np.isfinite(arrays[quantity])
            selected = arrays[quantity][mask]
            rows[scope][quantity] = {
                "retained_particle_observation_rows": int(mask.sum()),
                "full_denominator": 287 * len(series["times"]),
                "rms": float(np.sqrt(np.mean(selected**2))) if selected.size else None,
                "maximum": float(selected.max()) if selected.size else None,
                "weighting": "equal retained saved particle-observation rows; not time weighted",
            }
    return rows


def common_three_level_sensitivity(values: dict[str, Any], h: Any) -> dict[str, Any]:
    if "rk4_fine" not in values:
        return {"status": "WAITING_FOR_FINE"}
    a = h.comparison_arrays(values["rk4_coarse"], values["rk4_medium"])
    b = h.comparison_arrays(values["rk4_medium"], values["rk4_fine"])
    mask = a["precontact"] & b["precontact"]
    result = {
        "shared_precontact_rows": int(mask.sum()),
        "denominator": 287 * 121,
        "step_ratios": [2.0, 2.0],
        "definition": "Both RMS use exactly the same3-level jointly observed active precontact mask and equal saved-row weighting. Observed ratios are diagnostics on nonsmooth P1 fields, not formal RK4 order or error bounds.",
    }
    for quantity in h.FLAG_SCALES:
        aa, bb = a[quantity][mask], b[quantity][mask]
        rms_a, rms_b = float(np.sqrt(np.mean(aa**2))), float(np.sqrt(np.mean(bb**2)))
        result[quantity] = {
            "coarse_medium_rms": rms_a,
            "medium_fine_rms": rms_b,
            "observed_rms_order_diagnostic": float(np.log2(rms_a / rms_b))
            if rms_a > 0 and rms_b > 0
            else None,
        }
    return result


def comparison(
    case: str, name: str, left: dict[str, Any], right: dict[str, Any], h: Any
) -> dict[str, Any]:
    arrays = h.comparison_arrays(left, right)
    particles = [h.per_particle_metrics(left, right, arrays, slot) for slot in range(287)]
    populations = h.time_population_rows(left["times"], arrays)
    first_hits = first_hit_rows(left, right, h)
    terminal = h.terminal_identity_summary(left, right)
    write_csv(HERE / f"{case}_{name}_all_times.csv", h.difference_time_rows(left, right, arrays))
    write_csv(HERE / f"{case}_{name}_particles.csv", particles)
    write_csv(HERE / f"{case}_{name}_population_metrics.csv", populations)
    write_csv(HERE / f"{case}_{name}_10us_particles.csv", h.ten_us_ID_rows(left, right, arrays))
    write_csv(
        HERE / f"{case}_{name}_first_saved_divergence.csv",
        h.first_export_scale_timeline(left, right, arrays),
    )
    write_csv(HERE / f"{case}_{name}_terminal_confusion.csv", terminal["named_status_confusion"])
    write_csv(HERE / f"{case}_{name}_first_hit_intervals.csv", first_hits)
    numeric_hits = [
        row["absolute_candidate_first_hit_time_difference_s"]
        for row in first_hits
        if row["absolute_candidate_first_hit_time_difference_s"] is not None
    ]
    return {
        "left": left["label"],
        "right": right["label"],
        "per_ID_count": len(particles),
        "all_time_pair_records": len(left["times"]) * 287,
        "initial_x_v_Z_exact": True,
        "aggregate_norm_metrics": aggregate_norms(left, arrays, h),
        "terminal_ID_comparison": terminal,
        "material_arrival_ID_comparison": h.arrival_ID_summary(left, right),
        "candidate_event_identity_difference_IDs": [
            row["particle_id"]
            for row in particles
            if row["event_sequence_identity"].startswith("DIFFERENT")
        ],
        "candidate_event_sequence_review_scope": "All287 candidate IDs, primary boundary/outcome/ordinal identity only; event times/states are separately measured"
        if left["kind"] == right["kind"] == "candidate"
        else "NOT_TESTED: native complete event ledger is unavailable; empty differing-ID list does not establish event identity",
        "candidate_integration_steps_s": {
            "left": left.get("run_dt_s"),
            "right": right.get("run_dt_s"),
        },
        "maximum_precontact_difference_over_all_saved_times": {
            quantity: h.maximum_precontact_packet(particles, quantity) for quantity in h.FLAG_SCALES
        },
        "precontact_target_exceedance_ID_counts": {
            quantity: sum(
                row[f"precontact_{quantity}_first_above_fixed_flag_s"] is not None
                for row in particles
            )
            for quantity in h.FLAG_SCALES
        },
        "candidate_first_hit_maximum_time_difference_s": max(numeric_hits)
        if numeric_hits
        else None,
        "candidate_first_hit_pairs_over_1us": sum(value > 1e-6 for value in numeric_hits),
        "selected_population_observations": [
            min(populations, key=lambda row: abs(row["time_s"] - time))
            for time in (0, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 0.03)
        ],
        "PID11": particles[10],
        "PID30": particles[29],
        "interpretation": "Native comparisons are original-vs-analogue raw differences, not qualified parity. Candidate step and method changes are empirical sensitivities, not certified error bounds. Postcontact is descriptive; event identity equality does not mean equal event times or hit states.",
    }


def event_rows(values: dict[str, Any]) -> Any:
    for item in values.values():
        yield from item["events"]


def review_case(
    case_plan: dict[str, Any], old: dict[str, Any], audit: dict[str, Any], h: Any
) -> dict[str, Any]:
    name = case_plan["case"]
    values = read_series(case_plan, old, audit, h)
    for label, series in values.items():
        write_csv(HERE / name / f"{label}.csv", normalized_rows(series, h))
    write_csv(HERE / f"{name}_all_states.csv", all_states_rows(values, h))
    write_csv(HERE / f"{name}_candidate_boundary_events.csv", event_rows(values))
    write_csv(
        HERE / f"{name}_population.csv",
        chain.from_iterable(population_rows(series) for series in values.values()),
    )
    print(f"NORMALIZED_SERIES_READY {name} {','.join(values)}", flush=True)
    summaries = {
        key: comparison(name, key, values[left], values[right], h)
        for key, (left, right) in COMPARISONS.items()
        if left in values and right in values
    }
    pid_states = [row for row in all_states_rows(values, h) if row["particle_id"] in (11, 30)]
    write_csv(HERE / f"{name}_PID11_PID30_all_states.csv", pid_states)
    result = {
        "case": name,
        "analysis_state": "COMPLETE" if "rk4_fine" in values else "WAITING_FOR_FINE",
        "participants": list(values),
        "coverage": {key: h.series_coverage(value) for key, value in values.items()},
        "comparisons": summaries,
        "common_three_level_self_sensitivity": common_three_level_sensitivity(values, h),
        "source_bindings": {key: value["source_bindings"] for key, value in values.items()},
        "expected_all_states_records_when_complete": 6 * 287 * 121,
        "native_counter_authority": "Validated integer-counter conservation at all121 observations; exact native hit times/surfaces and disappeared causes remain unknown.",
        "known_original_model_differences": [
            "original Brownian versus deterministic analogue",
            "original linear versus candidate finite-speed Epstein drag",
            "original Waldmann omitted in candidate",
            "native FE versus transferred P1 fields",
            "native gas-domain/solution selection qualification remains unresolved",
        ],
        "qualified_original_solver_parity": "NOT_TESTED",
        "numerical_sensitivity_is_error_bound": False,
    }
    write_json(HERE / f"{name}_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("caseA", "caseP", "both"), default="both")
    args = parser.parse_args()
    plan = read_json(CAMPAIGN / "scientific_run_plan.json")
    read_json(HERE / "analysis_plan.json")
    read_json(HERE / "same_step_method_amendment.json")
    old = read_json(OLD / "evaluation/final_v64/evaluation_summary.json")
    audit = read_json(OLD / "original_comparison/equation_audit/equation_audit_summary.json")
    proof = read_json(OLD / "particle_trajectory_review/reviewer_notes.json")
    for record in proof["native_semantic_bindings"].values():
        path = CAMPAIGN.parents[3] / record["raw_owner"]["workspace_relative_path"]
        verify({"path": str(path), "sha256": record["raw_owner"]["sha256"]})
    h = helpers()
    cases = {
        item["case"]: review_case(item, old, audit, h)
        for item in plan["cases"]
        if args.case == "both" or args.case == item["case"]
    }
    summary = {
        "analysis_state": "COMPLETE"
        if all(row["analysis_state"] == "COMPLETE" for row in cases.values())
        else "WAITING_FOR_FINE",
        "cases": cases,
        "provisional_computational_targets": plan["empirical_resolution_targets"],
        "executor": receipt(Path(__file__)),
        "pure_helper": receipt(HELPER_PATH),
        "inputs": INPUTS,
        "outputs": OUTPUTS,
        "scope": "All287 IDs/case at121 real saved times, physical-model differences separate from numerical sensitivity; no simulation or historical writes.",
    }
    write_json(HERE / "trajectory_review_summary.json", summary)
    print(
        json.dumps(
            {
                "analysis_state": summary["analysis_state"],
                "cases": {key: value["analysis_state"] for key, value in cases.items()},
                "input_files": len(INPUTS),
                "output_files": len(OUTPUTS),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
