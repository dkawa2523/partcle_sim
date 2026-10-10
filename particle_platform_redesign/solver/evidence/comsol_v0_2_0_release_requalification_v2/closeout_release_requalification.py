"""Record this release's saved-reference requalification; never run a simulation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from tools.vv.comsol.meaning_preflight import load_inventory
from tools.vv.comsol.run_m3c2_candidate_pilot import installed_executor_identity

BASE = Path(__file__).resolve().parent
PROJECT = BASE.parents[1]
PRIOR = BASE.parent / "comsol_v0_2_0_release_requalification_v1"
ORIGINAL = BASE.parent / "comsol_binding_recert_2026_10_09_v1"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference(path: Path) -> dict[str, str]:
    return {"path": os.path.relpath(path, BASE).replace("\\", "/"), "sha256": digest(path)}


def save(path: Path, record: object) -> None:
    if path.exists():
        raise ValueError(f"Do not overwrite closed evidence: {path}")
    path.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def native_integrity() -> dict[str, object]:
    ledger = load_inventory(BASE / "saved_native_frozen_ledger.json")
    checked = 0
    for directory in ledger["directories"].values():
        root = (BASE / directory["source_root"]).resolve()
        for name, expected in directory["files"].items():
            path = root / name
            if path.stat().st_size != expected["size"] or digest(path) != expected["sha256"]:
                raise ValueError(f"Frozen native reference changed: {path}")
            checked += 1
    closure = load_inventory(BASE / "publication_saved_native_closure.json")
    external_checked = 0
    for name, expected in closure["external_raw_inputs"].items():
        if digest((PROJECT / name).resolve()) != expected["sha256"]:
            raise ValueError(f"Frozen external source/raw input changed: {name}")
        external_checked += 1
    return {
        "status": "PASS",
        "checked_original_files": checked,
        "checked_original_external_inputs_including_source_mph": external_checked,
        "ledger": reference(BASE / "saved_native_frozen_ledger.json"),
    }


def population_record(case: str) -> dict[str, object]:
    directory = BASE / f"case{case}_final_evaluation"
    report = load_inventory(directory / "evaluation_manifest.json")
    return {
        "status": report["status"],
        "evaluation": reference(directory / "evaluation_manifest.json"),
        "terminal_population_gate": report["terminal_population_gate"],
        "rz_distribution_gate": {
            key: value
            for key, value in report["rz_distribution_gate"].items()
            if key != "maximum_empirical_total_variation_by_time"
        },
        "interpretation": "Previously observed cohorts, fixed v45 selection and statistical formula used as a current-version regression gate. This creates neither a fresh unseen confirmation nor a new confidence family.",
        "boundary_limit": "CaseA Disappear is population status/time only; its cause and boundary ID remain unobserved."
        if case == "A"
        else "CaseP has no observed Brownian wall events; wall behavior is NOT_TESTED.",
    }


def four_evaluations(c3: dict[str, object]) -> dict[str, object]:
    return {
        "initial_conditions": {
            "status": "PASS",
            "scope": "The immutable canonical releases and saved native configuration/provenance are reused unchanged. Current candidate checks bind all particle IDs, releases, position, velocity, diameter, mass, charge, enabled models and configured boundary profile to registered artifacts.",
            "mass_limit": "Density/diameter getter expressions plus immutable-source provenance and the registered 100 nm override; fresh per-replica numerical native mass is NOT_TESTED.",
        },
        "fields_and_right_hand_side": {
            "status": "NOT_TESTED",
            "supported_subscope": "Unchanged exact-P1 common primitives, initial SI beta/FDT algebra, configured force/charge expression parity, and the saved named native total Ftr/Ftz at matched output states.",
            "unobserved": [
                "individual native force contributions",
                "auxiliary-charge assembled RHS",
                "all internal integration stages",
                "native FE field truth or arbitrary continuous fields",
            ],
            "interpretation": "The aggregate item remains NOT_TESTED because the complete native component/stage assembly was not observed. Algebra reconstruction is not promoted to that observation.",
        },
        "boundary_behavior": {
            "status": "NOT_TESTED",
            "retained_stick_event_subscope": c3["boundary_events_fine_vs_comsol"],
            "scope": "The deterministic retained Stick case compares 141 observed first-event IDs, semantic group, outcome and time. The current candidate reports actual canonical boundary IDs.",
            "unobserved": [
                "native first-hit location",
                "native pre/post velocity",
                "native remaining-substep response",
                "CaseA Disappear boundary ID/cause",
                "CaseP Brownian wall-positive behavior",
            ],
            "axis_limit": "The separate saved axis control is unchanged historical evidence; this release performs no new native axis or wall solve.",
        },
        "time_trajectory": {
            "status": c3["status"],
            "deterministic_scope": "Three current-candidate steps versus three saved-native steps at the registered common observed times. Position/charge use finite lifecycle states, velocity uses common active states, and lifecycle/validity/event identity are separate gates.",
            "deterministic_comparison": c3["candidate_fine_vs_comsol"],
            "candidate_self_convergence": c3["self_convergence"],
            "stochastic_scope": "CaseA/CaseP fixed full-population RZ occupancy and fate/time curves use independent ensemble arms. No pathwise equality, velocity/charge ensemble law, continuous-SDE bias or time-stage noise law is certified.",
            "allowance_limit": "The registered four-times fine-pair allowance is empirical numerical sensitivity, not a rigorous truncation bound. Inherited absolute floors alone are separately reported by the evaluator.",
        },
    }


def release_closure(independent_review: Path) -> dict[str, object]:
    saved = load_inventory(BASE / "publication_saved_native_closure.json")
    closure = copy.deepcopy(saved)
    closure["status"] = "PUBLISHED_AUTHORITY_AND_EXTERNAL_INPUTS_CLASSIFIED"
    closure["frozen_saved_native_closure"] = reference(
        BASE / "publication_saved_native_closure.json"
    )
    authority = closure["retained_git_authority"]
    external = closure["external_raw_inputs"]
    local = closure["local_regenerated_metadata"]
    for root in (PRIOR, BASE):
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.name == "publication_release_closure.json":
                continue
            if "__pycache__" in path.relative_to(root).parts:
                continue
            name = os.path.relpath(path, PROJECT).replace("\\", "/")
            if name in authority or name in external or name in local:
                continue
            if "result" in path.relative_to(root).parts or path.suffix.lower() not in {
                ".json",
                ".yaml",
                ".py",
                ".md",
                ".svg",
                ".png",
            }:
                external[name] = {
                    "classification": "EXTERNAL_RAW_INPUT",
                    "distributed_in_git": False,
                    "sha256": digest(path),
                }
            elif path.name != "coordination_status.json":
                authority[name] = {"classification": "GIT_AUTHORITY", "sha256": digest(path)}
    for name in ("comparison_manifest.json", "comparison_summary.json"):
        path = ORIGINAL / name
        authority[os.path.relpath(path, PROJECT).replace("\\", "/")] = {
            "classification": "GIT_AUTHORITY",
            "sha256": digest(path),
        }
    for path in (
        independent_review,
        PROJECT.parent / "reviews/v0_2_release_source_normalization_2026-10-10.json",
        PROJECT.parent / "reviews/v0_2_release_index_source_2026-10-10.json",
        BASE.parent / "comsol_v0_2_0_release_caseA_consumer_manifest.json",
        BASE.parent / "comsol_v0_2_0_release_caseP_consumer_manifest.json",
    ):
        authority[os.path.relpath(path, PROJECT).replace("\\", "/")] = {
            "classification": "GIT_AUTHORITY",
            "sha256": digest(path),
        }
    closure["scientific_reproduction"] = (
        "Use the fixed registered commands and provide every hash-matched external CSV/HDF5/MPH/result input. Git metadata is auditable without distributing proprietary/native raw datasets; it is not a self-contained numerical dataset bundle. No native rerun is needed for this version requalification. Full preflight conditions are regenerated with the listed existing command."
    )
    closure["counts"] = {
        "git_authority": len(authority),
        "external_raw_inputs": len(external),
        "local_regenerated_metadata": len(local),
    }
    return closure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--independent-review", type=Path, required=True)
    arguments = parser.parse_args()
    registry = load_inventory(BASE / "registration/execution_preregistration.json")
    executor = installed_executor_identity()
    if executor != registry["candidate_executor"]:
        raise ValueError("The actual final executor drifted from the fixed registration")
    for name, expected in registry["source_files"].items():
        if digest(PROJECT / name) != expected:
            raise ValueError(f"The registered source/lock changed: {name}")
    original_integrity = native_integrity()
    c3 = load_inventory(BASE / "c3_version_requalification_evaluation.json")
    c2 = {case: population_record(case) for case in ("A", "P")}
    summary = {
        "schema_version": 1,
        "status": "PASS"
        if c3["status"] == "PASS" and all(item["status"] == "PASS" for item in c2.values())
        else "FAIL",
        "closed_at_utc": datetime.now(UTC).isoformat(),
        "scope": "0.2.0 current-v46/event-v22 same-canonical-field version requalification on previously observed cohorts against unchanged saved COMSOL 6.4.0.429 references.",
        "new_native_solve": False,
        "fresh_unseen_confirmation": False,
        "new_confidence_family": False,
        "candidate_executor": executor,
        "c2": c2,
        "c3_registered_empirical_gate_status": c3["status"],
        "four_evaluations": four_evaluations(c3),
        "frozen_original_native_integrity": original_integrity,
        "broader_guarantees": {
            "general_3d": "NOT_APPLICABLE",
            "experimental_truth": "NOT_TESTED",
            "continuous_stochastic_bias_and_noise_law": "NOT_TESTED",
            "universal_COMSOL_equivalence": "NOT_TESTED",
        },
    }
    manifest = {
        "schema_version": 1,
        "registered_acceptance_changed": False,
        "registration": reference(BASE / "registration/execution_preregistration.json"),
        "independent_pre_run_audit_snapshot": reference(
            BASE / "independent_pre_run_audit_snapshot.json"
        ),
        "independent_result_review": reference(arguments.independent_review.resolve()),
        "source_normalization": reference(
            PROJECT.parent / "reviews/v0_2_release_source_normalization_2026-10-10.json"
        ),
        "frozen_native_projection": reference(BASE / "saved_native_projection_manifest.json"),
        "frozen_native_closure": reference(BASE / "publication_saved_native_closure.json"),
        "c3_evaluation": reference(BASE / "c3_version_requalification_evaluation.json"),
        "c3_configured_rhs_diagnostic": reference(BASE / "c3_configured_rhs_diagnostic.json"),
        "spatial_visualization": reference(BASE / "derived_c3/visualization_receipt.json"),
        "c2_evaluations": {case: c2[case]["evaluation"] for case in ("A", "P")},
        "actual_c2_consumer_manifest_views": {
            case: reference(
                BASE.parent / f"comsol_v0_2_0_release_case{case}_consumer_manifest.json"
            )
            for case in ("A", "P")
        },
        "consumer_manifest_projection_receipts": {
            case: reference(BASE / f"case{case}_consumer_manifest_projection_receipt.json")
            for case in ("A", "P")
        },
        "registered_view_operational_rejection": reference(
            BASE / "consumer_manifest_containment_rejection.json"
        ),
        "closeout_recipe": reference(Path(__file__).resolve()),
        "entrypoint_note": "The first C3 direct-script command stopped before case load with ModuleNotFoundError; its unchanged log is retained externally. The canonical python -m tools.vv.comsol.run_m3c3_casep_three_current_candidate invocation executed all three registered cells. No numerical setting changed.",
    }
    save(BASE / "comparison_summary.json", summary)
    manifest["comparison_summary"] = reference(BASE / "comparison_summary.json")
    save(BASE / "comparison_manifest.json", manifest)
    save(
        BASE / "publication_release_closure.json",
        release_closure(arguments.independent_review.resolve()),
    )
    print(
        json.dumps(
            {
                name: reference(BASE / name)
                for name in (
                    "comparison_manifest.json",
                    "comparison_summary.json",
                    "publication_release_closure.json",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
