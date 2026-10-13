"""Independent review of normalized saved CSV rows; no solver/evaluator imports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

PARTICIPANTS = ("native10", "native5", "old_exp_fine", "rk4_coarse", "rk4_medium", "rk4_fine")
COLUMNS = ("r_m", "z_m", "velocity_r_m_per_s", "velocity_z_m_per_s", "charge_number")
QUANTITIES = {
    "position_vector_m": slice(0, 2),
    "velocity_vector_m_s": slice(2, 4),
    "charge_number": slice(4, 5),
}
PAIRS = {
    "rk4_coarse_minus_medium": ("rk4_coarse", "rk4_medium"),
    "rk4_medium_minus_fine": ("rk4_medium", "rk4_fine"),
    "rk4_fine_minus_native10": ("rk4_fine", "native10"),
    "rk4_fine_minus_native5": ("rk4_fine", "native5"),
    "native5_minus_native10": ("native5", "native10"),
    "old_exp_fine_minus_rk4_fine": ("old_exp_fine", "rk4_fine"),
    "old_exp_fine_minus_rk4_medium": ("old_exp_fine", "rk4_medium"),
}
POPULATION_COLUMNS = {
    "position_vector_m": "precontact_position_vector_m",
    "velocity_vector_m_s": "precontact_velocity_vector_m_s",
    "charge_number": "precontact_charge_number",
}


def _hash(path: Path) -> str:
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def _read_clock(path: Path) -> tuple[list[int], list[float], int]:
    ids: set[int] = set()
    times: set[float] = set()
    names: set[str] = set()
    count = 0
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            ids.add(int(row["particle_id"]))
            times.add(float(row["time_s"]))
            names.add(row["participant"])
            count += 1
    if names != set(PARTICIPANTS):
        raise ValueError(f"unexpected participant set: {names}")
    return sorted(ids), sorted(times), count


def _fill(path: Path, ids: list[int], times: list[float]) -> dict:
    shape = (len(PARTICIPANTS), len(times), len(ids))
    arrays = {
        "state": np.full((*shape, 5), np.nan),
        "observed": np.zeros(shape, dtype=bool),
        "precontact": np.zeros(shape, dtype=bool),
        "present": np.zeros(shape, dtype=bool),
        "lifecycle": np.full(shape, "", dtype="U24"),
    }
    participant_index = {name: index for index, name in enumerate(PARTICIPANTS)}
    id_index = {value: index for index, value in enumerate(ids)}
    time_index = {value: index for index, value in enumerate(times)}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (
                participant_index[row["participant"]],
                time_index[float(row["time_s"])],
                id_index[int(row["particle_id"])],
            )
            if arrays["present"][key]:
                raise ValueError(f"duplicate input key: {key}")
            arrays["present"][key] = True
            arrays["state"][key] = [float(row[name]) if row[name] else np.nan for name in COLUMNS]
            arrays["observed"][key] = row["observed"] == "1"
            arrays["precontact"][key] = row["precontact_eligible"] == "1"
            arrays["lifecycle"][key] = row["lifecycle"]
    return arrays


def _load(path: Path) -> tuple[dict, dict]:
    before = _hash(path)
    ids, times, count = _read_clock(path)
    arrays = _fill(path, ids, times)
    if _hash(path) != before:
        raise ValueError("input changed during read-only review")
    if ids != list(range(1, 288)) or len(times) != 121:
        raise ValueError("campaign cohort is not the declared actual 287 IDs x 121 clock")
    if count != len(PARTICIPANTS) * len(ids) * len(times) or not arrays["present"].all():
        raise ValueError("normalized CSV is missing an explicit ID/time record")
    arrays["ids"] = np.asarray(ids)
    arrays["times"] = np.asarray(times)
    return arrays, {
        "path": str(path.resolve()),
        "sha256": before,
        "records": count,
        "particle_ids": ids,
        "observation_times_s": times,
        "explicit_key_grid_complete": True,
        "meaning": "Explicit normalized keys are not proof of observed coordinate states.",
    }


def _coverage(arrays: dict, index: int) -> dict:
    observed = arrays["observed"][index]
    state_finite = np.isfinite(arrays["state"][index]).all(axis=-1)
    active = arrays["lifecycle"][index] == "active"
    return {
        "IDs": int(arrays["ids"].size),
        "times": int(arrays["times"].size),
        "state_records_expected": int(observed.size),
        "state_records_observed": int(observed.sum()),
        "state_records_missing": int((~observed).sum()),
        "active_nonfinite_or_missing": int((active & (~observed | ~state_finite)).sum()),
        "final_named_status": dict(Counter(arrays["lifecycle"][index, -1].tolist())),
    }


def _initial(arrays: dict, left: int, right: int) -> dict:
    a, b = arrays["state"][left, 0], arrays["state"][right, 0]
    full = bool(
        arrays["observed"][[left, right], 0].all() and np.isfinite(a).all() and np.isfinite(b).all()
    )
    return {
        "full_287_initial_rows_observed_finite": full,
        "exact_x_v_Z_equality": full and bool(np.array_equal(a, b)),
        "maximum_absolute_component_difference": dict(
            zip(COLUMNS, np.max(np.abs(a - b), axis=0).tolist(), strict=True)
        ),
    }


def _pair_data(arrays: dict, left: int, right: int) -> tuple[dict, np.ndarray]:
    states = arrays["state"]
    finite = np.isfinite(states[[left, right]]).all(axis=-1).all(axis=0)
    observed = arrays["observed"][[left, right]].all(axis=0)
    active = (arrays["lifecycle"][[left, right]] == "active").all(axis=0)
    precontact = arrays["precontact"][[left, right]].all(axis=0)
    mask = finite & observed & active & precontact
    difference = states[left] - states[right]
    magnitudes = {
        name: np.sqrt(np.sum(difference[..., section] ** 2, axis=-1))
        for name, section in QUANTITIES.items()
    }
    return magnitudes, mask


def _metric(magnitude: np.ndarray, mask: np.ndarray, arrays: dict) -> dict:
    values = magnitude[mask]
    if not values.size:
        return {"count": 0, "rms": None, "maximum": None, "maximum_location": None}
    maximum_key = np.unravel_index(np.argmax(np.where(mask, magnitude, -np.inf)), mask.shape)
    return {
        "count": int(values.size),
        "rms": float(np.sqrt(np.mean(values**2))),
        "maximum": float(np.max(values)),
        "maximum_location": {
            "particle_id": int(arrays["ids"][maximum_key[1]]),
            "time_s": float(arrays["times"][maximum_key[0]]),
        },
    }


def _time_metric(magnitude: np.ndarray, mask: np.ndarray, index: int, arrays: dict) -> dict:
    selected = mask[index]
    values = magnitude[index, selected]
    if not values.size:
        return {"count": 0, "rms": None, "maximum": None, "maximum_particle_id": None}
    maximum_index = int(np.argmax(np.where(selected, magnitude[index], -np.inf)))
    return {
        "count": int(values.size),
        "rms": float(np.sqrt(np.mean(values**2))),
        "maximum": float(np.max(values)),
        "maximum_particle_id": int(arrays["ids"][maximum_index]),
    }


def _close(actual: float | None, expected: float | None) -> bool:
    if actual is None or expected is None:
        return actual is expected
    return bool(np.isclose(actual, expected, rtol=2e-13, atol=2e-18))


def _aggregate_matches(summary: dict, report: dict) -> dict:
    return {
        quantity: actual["count"]
        == summary["aggregate_norm_metrics"]["precontact"][quantity][
            "retained_particle_observation_rows"
        ]
        and report["full_saved_denominator"]
        == summary["aggregate_norm_metrics"]["precontact"][quantity]["full_denominator"]
        and _close(actual["rms"], summary["aggregate_norm_metrics"]["precontact"][quantity]["rms"])
        and _close(
            actual["maximum"], summary["aggregate_norm_metrics"]["precontact"][quantity]["maximum"]
        )
        for quantity, actual in report["all_saved_joint_precontact"].items()
    }


def _verify_pair(
    summary: dict, report: dict, magnitudes: dict, mask: np.ndarray, arrays: dict
) -> dict:
    initial_matches = report["initial"]["exact_x_v_Z_equality"] == summary["initial_x_v_Z_exact"]
    maxima = {}
    for quantity, actual in report["all_saved_joint_precontact"].items():
        expected = summary["maximum_precontact_difference_over_all_saved_times"][quantity]
        maxima[quantity] = (
            _close(actual["maximum"], expected["value"])
            and actual["maximum_location"]["particle_id"] == expected["particle_id"]
            and actual["maximum_location"]["time_s"] == expected["time_s"]
        )
    population = {}
    for row in summary["selected_population_observations"]:
        index = int(np.flatnonzero(arrays["times"] == row["time_s"])[0])
        row_checks = {}
        for quantity, prefix in POPULATION_COLUMNS.items():
            actual = _time_metric(magnitudes[quantity], mask, index, arrays)
            row_checks[quantity] = (
                actual["count"] == row[prefix + "_valid_particles"]
                and _close(actual["rms"], row[prefix + "_rms"])
                and _close(actual["maximum"], row[prefix + "_maximum"])
                and actual["maximum_particle_id"] == row[prefix + "_maximum_particle_id"]
            )
        population[str(row["time_s"])] = row_checks
    aggregate_matches = _aggregate_matches(summary, report)
    return {
        "initial_equality_summary_matches": initial_matches,
        "all_saved_maximum_and_ID_time_match": maxima,
        "selected_time_population_count_rms_max_ID_match": population,
        "aggregate_precontact_count_denominator_rms_max_match": aggregate_matches,
        "all_checked_values_match": initial_matches
        and all(maxima.values())
        and all(aggregate_matches.values())
        and all(all(row.values()) for row in population.values()),
    }


def _three_level_review(arrays: dict, expected: dict) -> dict:
    coarse = PARTICIPANTS.index("rk4_coarse")
    medium = PARTICIPANTS.index("rk4_medium")
    fine = PARTICIPANTS.index("rk4_fine")
    coarse_medium, left_mask = _pair_data(arrays, coarse, medium)
    medium_fine, right_mask = _pair_data(arrays, medium, fine)
    mask = left_mask & right_mask
    result = {
        "shared_precontact_rows": int(mask.sum()),
        "denominator": int(mask.size),
        "step_ratio_basis": "Declared twofold ratios in the bound saved summary; CSV states alone do not authenticate actual integration partitions.",
        "step_ratios": expected["step_ratios"],
    }
    matches = {}
    for quantity in QUANTITIES:
        first_rms = _metric(coarse_medium[quantity], mask, arrays)["rms"]
        second_rms = _metric(medium_fine[quantity], mask, arrays)["rms"]
        ratio = first_rms / second_rms
        diagnostic = float(np.log2(ratio))
        result[quantity] = {
            "coarse_medium_rms": first_rms,
            "medium_fine_rms": second_rms,
            "observed_rms_ratio": ratio,
            "log2_ratio_diagnostic": diagnostic,
            "meaning": "Observed saved-row diagnostic; not formal convergence order or an error bound.",
        }
        matches[quantity] = (
            _close(first_rms, expected[quantity]["coarse_medium_rms"])
            and _close(second_rms, expected[quantity]["medium_fine_rms"])
            and _close(
                diagnostic,
                expected[quantity]["observed_rms_order_diagnostic"],
            )
        )
    result["saved_summary_matches"] = matches
    result["all_checked_saved_summary_values_match"] = (
        all(matches.values())
        and result["shared_precontact_rows"] == expected["shared_precontact_rows"]
        and result["denominator"] == expected["denominator"]
    )
    return result


def _pair_review(arrays: dict, left_name: str, right_name: str, summary: dict) -> dict:
    left, right = PARTICIPANTS.index(left_name), PARTICIPANTS.index(right_name)
    magnitudes, mask = _pair_data(arrays, left, right)
    time_index = int(np.flatnonzero(arrays["times"] == 1e-5)[0])
    report = {
        "left": left_name,
        "right": right_name,
        "initial": _initial(arrays, left, right),
        "all_saved_joint_precontact": {
            name: _metric(magnitude, mask, arrays) for name, magnitude in magnitudes.items()
        },
        "at_10us_joint_precontact": {
            name: _time_metric(magnitude, mask, time_index, arrays)
            for name, magnitude in magnitudes.items()
        },
        "full_saved_denominator": int(mask.size),
        "joint_precontact_rows": int(mask.sum()),
    }
    report["saved_summary_verification"] = _verify_pair(summary, report, magnitudes, mask, arrays)
    return report


def _case_review(directory: Path, name: str) -> dict:
    arrays, identity = _load(directory / f"{name}_all_states.csv")
    summary_path = directory / f"{name}_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    coverage = {
        participant: _coverage(arrays, index) for index, participant in enumerate(PARTICIPANTS)
    }
    coverage_matches = {
        name: all(value == summary["coverage"][name][key] for key, value in row.items())
        for name, row in coverage.items()
    }
    initial = {name: _initial(arrays, PARTICIPANTS.index(name), 0) for name in PARTICIPANTS}
    pairs = {
        pair_name: _pair_review(arrays, left, right, summary["comparisons"][pair_name])
        for pair_name, (left, right) in PAIRS.items()
    }
    common_three_level = _three_level_review(arrays, summary["common_three_level_self_sensitivity"])
    return {
        "case": name,
        "normalized_csv": identity,
        "saved_summary": {"path": str(summary_path.resolve()), "sha256": _hash(summary_path)},
        "coverage_and_final_lifecycle": coverage,
        "initial_against_native10": initial,
        "pairs": pairs,
        "common_three_level_self_sensitivity": common_three_level,
        "coverage_and_final_lifecycle_summary_matches": coverage_matches,
        "all_checked_saved_summary_values_match": all(coverage_matches.values())
        and common_three_level["all_checked_saved_summary_values_match"]
        and all(
            pair["saved_summary_verification"]["all_checked_values_match"]
            for pair in pairs.values()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory_directory", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()
    cases = {name: _case_review(args.trajectory_directory, name) for name in ("caseA", "caseP")}
    result = {
        "schema_version": 1,
        "review_date_local": "2026-10-14",
        "reviewer": "original_model_mapping",
        "review_kind": "independent_normalized_csv_numeric_review",
        "imports": "Python standard library and numpy only; no repository analysis/evaluator/solver imports.",
        "reproducer_sha256": _hash(Path(__file__)),
        "status": "PASS"
        if all(case["all_checked_saved_summary_values_match"] for case in cases.values())
        else "FAIL",
        "definitions": {
            "mask": "Both rows observed, all x/v/Z components finite, lifecycle active, and explicitly precontact_eligible.",
            "RMS": "sqrt(mean(Euclidean vector difference squared)) or scalar absolute Z; equal weight per retained ID/time row.",
            "denominator": "Actual explicit 287 IDs x 121 saved clock, including missing or terminal rows.",
            "missing": "No fill, interpolation, padding or substitute state.",
            "native_disappeared": "Kept distinct from candidate escaped; no inferred boundary cause.",
        },
        "cases": cases,
        "not_claimed": [
            "Physical or statistical equivalence between original native and deterministic candidate.",
            "Shared Brownian or wall RNG histories from matching IDs or seeds.",
            "Converged trajectory accuracy from equal terminal fate counts.",
            "Error bounds from coarse/medium/fine sampled changes.",
            "Independent re-verification of native status/storage meaning beyond the supplied normalized CSV.",
        ],
    }
    with args.output_json.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "output": str(args.output_json)}))


if __name__ == "__main__":
    main()
