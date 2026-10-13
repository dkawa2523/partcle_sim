"""Use the generic evaluator on the registered, observed A/P release window."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.vv.comsol.evaluate_matched_trajectory import characterize, compare_pair

HERE = Path(__file__).resolve().parent


def write_json(path: Path, record: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)


def main() -> None:
    plan = json.loads((HERE / "scientific_run_plan.json").read_text(encoding="utf-8"))
    output = HERE / "early_window"
    output.mkdir(exist_ok=False)
    for case in plan["cases"]:
        name = case["case"]
        common = {
            "coordinate_mode": "axisymmetric_rz_meridional",
            "particle_ids": case["source_ids"],
            "observation_times_s": [0.0, 1.0e-5],
        }
        levels = {
            run["level"]: {
                "step_s": run["step_s"],
                "case_path": run["case"],
                "result_path": run["output"],
            }
            for run in case["runs"]
        }
        series_path = output / f"{name}_series.json"
        write_json(series_path, {**common, "label": f"{name} RK4 self", "levels": levels})
        write_json(output / f"{name}_self.json", characterize(series_path))
        candidate_path = output / f"{name}_candidate.json"
        write_json(candidate_path, {**common, "label": f"{name} RK4 fine", "level": levels["fine"]})
        native_csv = HERE / "trajectory_review" / name / "native10.csv"
        reference_path = output / f"{name}_reference.json"
        write_json(
            reference_path,
            {
                **common,
                "label": f"{name} original native saved RK4 10us, Brownian enabled",
                "level": {
                    "step_s": 1.0e-5,
                    "csv_path": str(native_csv),
                    "columns": {
                        "particle_id": "particle_id",
                        "time": "time_s",
                        "position": ["r_m", "z_m"],
                        "velocity": ["velocity_r_m_per_s", "velocity_z_m_per_s"],
                        "charge": "charge_number",
                        "lifecycle": "lifecycle",
                    },
                },
            },
        )
        policy_path = output / f"{name}_policy.json"
        write_json(
            policy_path,
            {
                "comparison_question": "Original saved native versus deterministic analogue: observed release-to-10us same-ID x/v/Z differences",
                "comparison_intent": "DESCRIPTIVE; numerical sensitivity targets from the pre-execution plan are not validated physical-equivalence tolerances",
                "physical_budgets": {
                    "position": {"rms": 1.0e-4, "maximum": 1.0e-4},
                    "velocity": {"rms": 1.0, "maximum": 1.0},
                    "charge": {"rms": 0.1, "maximum": 0.1},
                },
                "known_model_differences": [
                    "Native linear slip drag versus finite-speed Maxwell mean drag",
                    "Native Brownian force enabled versus deterministic candidate",
                    "Native Waldmann force versus candidate no thermophoresis",
                    "Native FE/recovery versus transferred P1 nodal fields and explicit axis parity",
                    "Candidate prescribed particle-surface temperature equal to local gas temperature is an explicit closure",
                ],
            },
        )
        write_json(
            output / f"{name}_original_raw_pair.json",
            compare_pair(candidate_path, reference_path, policy_path),
        )
    files = sorted(output.glob("*.json"))
    write_json(
        output / "receipt.json",
        {
            "scope": "Two saved release/10us observations, all actual stable IDs; full-window trajectory/event analysis is separate",
            "files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
            "plan_sha256": hashlib.sha256(
                (HERE / "scientific_run_plan.json").read_bytes()
            ).hexdigest(),
            "evaluator_sha256": hashlib.sha256(
                (HERE.parent.parent / "tools/vv/comsol/evaluate_matched_trajectory.py").read_bytes()
            ).hexdigest(),
        },
    )


if __name__ == "__main__":
    main()
