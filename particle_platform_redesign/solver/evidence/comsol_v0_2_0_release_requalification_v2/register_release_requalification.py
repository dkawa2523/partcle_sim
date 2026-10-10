"""Fix this release's LF-only amendment before any new candidate outcomes.

Run as ``python -m evidence.comsol_v0_2_0_release_requalification_v2.register_release_requalification``
from the solver uv project after the root performance-complete signal.
This evidence recipe neither changes production code nor runs simulations.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from tools.vv.comsol.meaning_preflight import load_inventory
from tools.vv.comsol.run_m3c2_candidate_pilot import installed_executor_identity

BASE = Path(__file__).resolve().parent
PROJECT = BASE.parents[1]
PRIOR = BASE.parent / "comsol_v0_2_0_release_requalification_v1"
ORIGINAL = BASE.parent / "comsol_binding_recert_2026_10_09_v1"
INDEX_WITNESS = PROJECT.parent / "reviews/v0_2_release_index_source_2026-10-10.json"
INDEX_WITNESS_SHA256 = "1a56482f45cec3286d19fd67419d2912f3cd08cbfa238d75cd81b931a4097788"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference(path: Path, basis: Path = BASE) -> dict[str, str]:
    import os

    return {"path": os.path.relpath(path, basis).replace("\\", "/"), "sha256": digest(path)}


def save(path: Path, record: object) -> None:
    if path.exists():
        raise ValueError(f"Do not overwrite registered evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-performance-complete-utc", required=True)
    arguments = parser.parse_args()
    release_time = datetime.fromisoformat(arguments.root_performance_complete_utc)
    if release_time.tzinfo is None or release_time > datetime.now(UTC):
        raise ValueError("Require the already received, timezone-aware root completion time")
    if digest(INDEX_WITNESS) != INDEX_WITNESS_SHA256:
        raise ValueError("The predeclared LF/index witness changed")
    index = load_inventory(INDEX_WITNESS)
    entries = index["source_entries"]
    source_files = {name: digest(PROJECT / name) for name in entries}
    if any(source_files[name] != item["index_sha256"] for name, item in entries.items()):
        raise ValueError("Wait for all 27 actual working source files to equal the final LF index")
    if any(not item["python_ast_identical"] for item in entries.values()):
        raise ValueError("The declared normalization is not Python-AST preserving")
    old_registry_path = PRIOR / "registration/execution_preregistration.json"
    old_registry = load_inventory(old_registry_path)
    if list(PRIOR.rglob("run_receipt.json")) or list(PRIOR.rglob("result_manifest.json")):
        raise ValueError("The superseded CRLF preparation must remain unexecuted")
    if any(
        (BASE / "c3_prepared/candidate" / level / name).exists()
        for level in ("dt_2p5us", "dt_1p25us", "dt_0p625us")
        for name in ("result", "run_receipt.json", "trajectory.csv", "events.csv")
    ):
        raise ValueError("The new C3 candidate outputs must not exist before registration")
    if any((BASE / f"case{case}_candidate_final").exists() for case in ("A", "P")):
        raise ValueError("The new C2 preparation must not exist before this registration")
    executor = installed_executor_identity()
    if executor["uv_lock_sha256"] != index["uv_lock_sha256"]:
        raise ValueError("The dependency lock changed")
    source_files["uv.lock"] = executor["uv_lock_sha256"]
    registry = copy.deepcopy(old_registry)
    registry["registered_at_utc"] = datetime.now(UTC).isoformat()
    registry["candidate_executor"] = executor
    registry["source_files"] = source_files
    registry["execution_condition"] = (
        "Root performance window released; independent pre-run audit required before scientific execution."
    )
    registry["saved_native_projection"] = reference(BASE / "saved_native_projection_manifest.json")
    registry["publication_saved_native_closure"] = reference(
        BASE / "publication_saved_native_closure.json"
    )
    registry["registration_recipe"] = reference(Path(__file__).resolve())
    registry["lf_only_observation_free_amendment"] = {
        "path_basis": "release evidence root",
        "prior_unexecuted_crlf_registration": reference(old_registry_path),
        "root_index_witness": reference(INDEX_WITNESS),
        "root_performance_complete_utc": arguments.root_performance_complete_utc,
        "source_file_changed_since_crlf_preparation": ["src/chamber_particles/engine.py"],
        "python_ast_unchanged": True,
        "numeric_inputs_models_seeds_thresholds_bins_unchanged": True,
        "historical_pointer_repair": "/producer_observation_amendment/original_pilot_registration/path",
        "new_candidate_outcomes_observed": False,
        "new_native_solve": False,
    }
    for case in ("A", "P"):
        prefix = f"m3c2_case{case}_100nm_axis_normalized"
        old_registration = PRIOR / "registration"
        registration = BASE / "registration"
        for suffix in ("historical_pilot_path_projection", "fixed_selection_path_projection"):
            source = old_registration / f"{prefix}_{suffix}.json"
            target = registration / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise ValueError(f"Do not overwrite authority: {target}")
            shutil.copyfile(source, target)
        recipe_path = registration / f"{prefix}_candidate_recipe_v46.json"
        recipe = load_inventory(old_registration / recipe_path.name)
        prior_executor = recipe["execution"]["expected_executor"]
        assert {k: v for k, v in executor.items() if k != "source_sha256"} == {
            k: v for k, v in prior_executor.items() if k != "source_sha256"
        }
        recipe["execution"]["expected_executor"] = executor
        save(recipe_path, recipe)
        restored_recipe = copy.deepcopy(recipe)
        restored_recipe["execution"]["expected_executor"] = prior_executor
        assert restored_recipe == load_inventory(old_registration / recipe_path.name)
        registration_path = registration / f"{prefix}_v46_registration.json"
        record = load_inventory(old_registration / registration_path.name)
        record["registered_at_utc"] = registry["registered_at_utc"]
        record["locked_inputs"]["recipe"] = reference(recipe_path, registration)
        historical = ORIGINAL / "registration_axis_normalized/m3c2_execution_preregistration.json"
        record["producer_observation_amendment"]["original_pilot_registration"] = reference(
            historical, registration
        )
        record["version_requalification"]["final_lf_source_amendment"] = registry[
            "lf_only_observation_free_amendment"
        ]
        save(registration_path, record)
        projection_path = registration / f"case{case}_authority_projection_receipt.json"
        projection = load_inventory(old_registration / projection_path.name)
        projection["reference_path_basis"] = {
            "projected_canonical_input_path": "new campaign root",
            "unchanged_historical_pilot_artifact_paths": "original historical campaign.path directory",
            "receipt_artifact_paths": "this registration directory",
        }
        projection["lf_only_recipe_amendment"] = {
            "prior_recipe": reference(old_registration / recipe_path.name, registration),
            "current_recipe": reference(recipe_path, registration),
            "changed_json_pointers": ["/execution/expected_executor/source_sha256"],
            "all_other_recipe_values_identical": True,
        }
        save(projection_path, projection)
        output = BASE / f"case{case}_candidate_final"
        subprocess.run(
            [
                sys.executable,
                str(PROJECT / "tools/vv/comsol/run_m3c2_candidate_pilot.py"),
                "prepare",
                str(recipe_path),
                str(output),
                "--registration",
                str(registration_path),
            ],
            cwd=PROJECT,
            check=True,
            capture_output=True,
            text=True,
        )
        item = registry["c2"][case]
        for key, path in {
            "recipe": recipe_path,
            "registration": registration_path,
            "prepared": output / "prepare_report.json",
            "authority_projection_receipt": projection_path,
        }.items():
            item[key] = reference(path)
        native = BASE / "saved_native" / f"case{case}"
        item["native_manifest"] = reference(native / "comsol_campaign_manifest.json")
        item["native_meaning_inventory"] = reference(native / "meaning_inventory.json")
        item["native_publication_projection_receipt"] = reference(
            native / "publication_projection_receipt.json"
        )
    for item, directory in zip(
        registry["c3"]["saved_references"],
        ("c3_dt_5us", "c3_dt_2p5us", "c3_dt_1p25us"),
        strict=True,
    ):
        root = BASE / "saved_native" / directory
        item["root"] = str(root.relative_to(BASE)).replace("\\", "/")
        for key, name in {
            "trajectory": "trajectory_reference.csv",
            "events": "events_reference.csv",
            "meaning_inventory": "meaning_inventory.json",
            "normalization_summary": "normalization_summary.json",
            "run_receipt": "run_receipt.json",
            "publication_projection_receipt": "publication_projection_receipt.json",
        }.items():
            item[key] = reference(root / name)
    for item in registry["c3"]["staged_authorities"]:
        assert digest(BASE / item["staged"]["path"]) == item["staged"]["sha256"]
    assert installed_executor_identity() == executor
    save(BASE / "registration/execution_preregistration.json", registry)
    print(
        json.dumps(reference(BASE / "registration/execution_preregistration.json"), sort_keys=True)
    )


if __name__ == "__main__":
    main()
