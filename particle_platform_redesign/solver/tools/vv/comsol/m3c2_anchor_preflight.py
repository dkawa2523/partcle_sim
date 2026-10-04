"""Audit M3-C2 anchor inventory and reserve seeds without running either solver.

The existing COMSOL histories are audited as provenance evidence only.  A
single native-field Brownian history whose effective random stream is not
proven is never promoted into the meaning-matched RZ ensemble comparison
planned here.  This preflight is not the later campaign lock: steps, exact
physics revisions, and executable COMSOL/candidate recipes still have to be
fixed before any run is authorized.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

Record = dict[str, str]
TOOL_REVISION = "m3c2_anchor_preflight_v1"


@dataclass(frozen=True)
class Anchor:
    case_id: str
    role: str
    package: str
    background_dataset: str
    brownian_feature_tag: str
    brownian_seed_parameter: str
    brownian_seed_expression: str


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return cast(dict[str, Any], value)


def _list(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return cast(list[Any], value)


def _load_config(path: Path) -> dict[str, Any]:
    raw = _mapping(json.loads(path.read_text(encoding="utf-8")), "configuration")
    if raw.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    return raw


def _anchors(config: dict[str, Any]) -> tuple[Anchor, ...]:
    rows: list[Anchor] = []
    for index, raw_anchor in enumerate(_list(config["anchors"], "anchors")):
        anchor = _mapping(raw_anchor, f"anchors[{index}]")
        rows.append(
            Anchor(
                case_id=str(anchor["case_id"]),
                role=str(anchor["role"]),
                package=str(anchor["package"]),
                background_dataset=str(anchor["background_dataset"]),
                brownian_feature_tag=str(anchor["brownian_feature_tag"]),
                brownian_seed_parameter=str(anchor["brownian_seed_parameter"]),
                brownian_seed_expression=str(anchor["brownian_seed_expression"]),
            )
        )
    if not rows:
        raise ValueError("at least one anchor is required")
    return tuple(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[Record]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _write_csv(path: Path, rows: list[Record], columns: tuple[str, ...]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _validate_seed_rows(
    rows: list[Record], case_ids: tuple[str, ...], expected_replicas: int, expected_status: str
) -> dict[str, list[Record]]:
    selected: dict[str, list[Record]] = defaultdict(list)
    for row in rows:
        if row.get("case_id") in case_ids:
            selected[row["case_id"]].append(row)
    all_seeds: list[int] = []
    for case_id in case_ids:
        case_rows = selected[case_id]
        replicas = [int(row["replica"]) for row in case_rows]
        seeds = [int(row["seed"]) for row in case_rows]
        statuses = {row["execution_status"] for row in case_rows}
        if len(case_rows) != expected_replicas:
            raise ValueError(f"{case_id} must have exactly {expected_replicas} seed rows")
        if sorted(replicas) != list(range(expected_replicas)):
            raise ValueError(f"{case_id} replica ordinals must be complete and unique")
        if len(set(seeds)) != expected_replicas:
            raise ValueError(f"{case_id} seeds must be unique")
        if statuses != {expected_status}:
            raise ValueError(f"{case_id} seed status differs from {expected_status}")
        all_seeds.extend(seeds)
    if len(set(all_seeds)) != len(all_seeds):
        raise ValueError("anchor seed sets must not overlap")
    return dict(selected)


def _locked_artifact_rows(
    repository_root: Path,
    config: dict[str, Any],
    anchors: tuple[Anchor, ...],
) -> tuple[list[Record], bool]:
    locks = _mapping(config["m3c0_locks"], "m3c0_locks")
    artifact_lock = repository_root / str(locks["artifact_hashes"])
    model_lock = repository_root / str(locks["model_identity"])
    locked = {(row["case_id"], row["path"]): row["sha256"] for row in _read_csv(artifact_lock)}
    selected_paths = tuple(
        str(value)
        for value in _list(config["locked_package_artifacts"], "locked_package_artifacts")
    )
    output: list[Record] = []
    all_match = True
    for anchor in anchors:
        package = repository_root / anchor.package
        for relative_path in selected_paths:
            path = package / relative_path
            expected = locked.get((anchor.case_id, relative_path), "<missing-lock>")
            actual = _sha256(path) if path.is_file() else "<missing-artifact>"
            match = actual == expected
            all_match &= match
            output.append(
                {
                    "owner": anchor.case_id,
                    "kind": "package_artifact",
                    "path": relative_path,
                    "expected_sha256": expected,
                    "actual_sha256": actual,
                    "status": "PASS" if match else "FAIL",
                }
            )
    model = _mapping(config["model"], "model")
    model_row = next(
        (row for row in _read_csv(model_lock) if row["variant"] == str(model["variant"])), None
    )
    model_path = repository_root / str(model["path"])
    expected_model = "<missing-lock>" if model_row is None else model_row["expected_sha256"]
    actual_model = _sha256(model_path) if model_path.is_file() else "<missing-artifact>"
    model_lock_valid = (
        model_row is not None
        and model_row.get("status") == "PASS"
        and model_row.get("actual_sha256") == expected_model
    )
    model_match = model_lock_valid and actual_model == expected_model
    all_match &= model_match
    output.append(
        {
            "owner": str(model["variant"]),
            "kind": "source_model",
            "path": str(model["path"]),
            "expected_sha256": expected_model,
            "actual_sha256": actual_model,
            "status": "PASS" if model_match else "FAIL",
        }
    )
    return output, all_match


def _history_brownian_summary(path: Path) -> tuple[int, int, int, int, tuple[str, ...]]:
    rows = 0
    evaluated_rows = 0
    nonzero_magnitude = 0
    nonzero_phi = 0
    components: tuple[str, ...] = ()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        header = set(reader.fieldnames or ())
        components = tuple(
            component
            for component in ("r", "phi", "z")
            if f"Brownian_force_{component}_N" in header
        )
        required = {
            "Brownian_force_r_N",
            "Brownian_force_phi_N",
            "Brownian_force_z_N",
            "Brownian_force_magnitude_N",
            "active_state_flag",
        }
        if not required.issubset(header):
            raise ValueError(f"Brownian history columns missing from {path}")
        for row in reader:
            rows += 1
            if float(row["active_state_flag"]) != 1.0:
                continue
            evaluated_rows += 1
            values = tuple(
                float(row[column]) for column in sorted(required) if column != "active_state_flag"
            )
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"nonfinite active Brownian history value in {path}")
            nonzero_magnitude += float(row["Brownian_force_magnitude_N"]) != 0.0
            nonzero_phi += float(row["Brownian_force_phi_N"]) != 0.0
    if evaluated_rows == 0:
        raise ValueError(f"Brownian history has no active samples in {path}")
    return rows, evaluated_rows, nonzero_magnitude, nonzero_phi, components


def _audit_anchor(repository_root: Path, anchor: Anchor) -> tuple[Record, list[str]]:
    package = repository_root / anchor.package
    manifest = {row["key"]: row["value"] for row in _read_csv(package / "manifest.csv")}
    parameters = {
        row["parameter"]: row for row in _read_csv(package / "config/global_parameters.csv")
    }
    features = _read_csv(package / "config/particle_physics_feature_settings.csv")
    seed_rows = [
        row
        for row in features
        if row["feature_tag"] == anchor.brownian_feature_tag and row["property"] == "i"
    ]
    if len(seed_rows) != 1:
        raise ValueError(f"{anchor.case_id} must have exactly one Brownian seed property")
    seed_parameter = parameters.get(anchor.brownian_seed_parameter)
    if seed_parameter is None:
        raise ValueError(f"{anchor.case_id} Brownian seed parameter is missing")
    history = package / "results/particle_history_full_tidy.csv"
    history_rows, evaluated_rows, nonzero_magnitude, nonzero_phi, components = (
        _history_brownian_summary(history)
    )
    mismatches: list[str] = []
    actual_background_dataset = manifest.get("background_dataset", "<missing>")
    if actual_background_dataset == anchor.background_dataset:
        mismatches.append("NATIVE_FIELD_NOT_COMMON_P1")
    else:
        mismatches.append("BACKGROUND_DATASET_DIFFERS_FROM_LOCK")
    mismatches.append("COMSOL_STEPWISE_BROWNIAN_REQUIRES_ENSEMBLE_CONVERGENCE")
    mismatches.append("SAVED_RANDOM_STREAM_IDENTITY_NOT_PROVEN")
    mismatches.append("ONE_SAVED_RUN_NOT_32_SEED_COHORT")
    seed_expression = seed_rows[0]["value"]
    feature_match = seed_expression == anchor.brownian_seed_expression
    if not feature_match:
        mismatches.append("BROWNIAN_SEED_FEATURE_DIFFERS_FROM_LOCK")
    return (
        {
            "case_id": anchor.case_id,
            "role": anchor.role,
            "background_dataset": actual_background_dataset,
            "brownian_feature_tag": anchor.brownian_feature_tag,
            "brownian_seed_expression": seed_expression,
            "configured_seed_parameter_value": seed_parameter["evaluated_SI_value"],
            "effective_random_stream_identity": "UNVERIFIED",
            "history_rows": str(history_rows),
            "brownian_evaluated_active_rows": str(evaluated_rows),
            "brownian_nonzero_rows": str(nonzero_magnitude),
            "brownian_phi_reported_nonzero_rows": str(nonzero_phi),
            "brownian_reported_components": ";".join(components),
            "saved_run_count": "1",
            "meaning_matched_saved_cohort": "False",
            "mismatch_codes": ";".join(mismatches),
        },
        mismatches,
    )


def _run_matrix(config: dict[str, Any], seed_rows: dict[str, list[Record]]) -> list[Record]:
    participants = [
        _mapping(value, "participant") for value in _list(config["participants"], "participants")
    ]
    rows: list[Record] = []
    for case_id in (
        str(value) for value in _list(config["campaign_case_ids"], "campaign_case_ids")
    ):
        for participant in participants:
            rows.extend(
                {
                    "case_id": case_id,
                    "participant": str(participant["name"]),
                    "replica": seed["replica"],
                    "campaign_seed": seed["seed"],
                    "field_contract": str(participant["field_contract"]),
                    "stochastic_contract": str(participant["stochastic_contract"]),
                    "comparison_role": "independent_seed_cluster_not_pathwise_pair",
                    "execution_status": "PLANNED_NOT_RUN",
                }
                for seed in sorted(seed_rows[case_id], key=lambda row: int(row["replica"]))
            )
    return rows


def _seed_receipts(seed_rows: dict[str, list[Record]]) -> list[Record]:
    receipts: list[Record] = []
    for case_id, rows in seed_rows.items():
        replicas = [int(row["replica"]) for row in rows]
        seeds = [int(row["seed"]) for row in rows]
        receipts.append(
            {
                "case_id": case_id,
                "replica_count": str(len(replicas)),
                "unique_seed_count": str(len(set(seeds))),
                "replica_min": str(min(replicas)),
                "replica_max": str(max(replicas)),
                "seed_min": str(min(seeds)),
                "seed_max": str(max(seeds)),
                "execution_status": rows[0]["execution_status"],
                "validation_status": "PASS",
            }
        )
    return receipts


def build_preflight(config_path: Path, repository_root: Path, output: Path) -> dict[str, object]:
    config = _load_config(config_path)
    anchors = _anchors(config)
    future_requirements = _mapping(
        config["required_companion_semantics"], "required_companion_semantics"
    )
    seed_config = _mapping(config["seed_plan"], "seed_plan")
    seed_path = repository_root / str(seed_config["path"])
    expected_replicas = int(seed_config["expected_replicas_per_case"])
    seed_rows = _validate_seed_rows(
        _read_csv(seed_path),
        tuple(anchor.case_id for anchor in anchors),
        expected_replicas,
        str(seed_config["expected_execution_status"]),
    )
    artifact_rows, artifacts_match = _locked_artifact_rows(repository_root, config, anchors)
    audits: list[Record] = []
    mismatch_codes: set[str] = set()
    for anchor in anchors:
        audit, mismatches = _audit_anchor(repository_root, anchor)
        audits.append(audit)
        mismatch_codes.update(mismatches)
    audit_integrity = not bool(
        mismatch_codes
        & {
            "BACKGROUND_DATASET_DIFFERS_FROM_LOCK",
            "BROWNIAN_SEED_FEATURE_DIFFERS_FROM_LOCK",
        }
    )
    matrix = _run_matrix(config, seed_rows)
    seed_receipts = _seed_receipts(seed_rows)
    overall_status = (
        "BLOCKED_MISSING_MEANING_MATCHED_COHORT"
        if artifacts_match and audit_integrity
        else "BLOCKED_SOURCE_IDENTITY_FAILURE"
    )
    gates = [
        {
            "gate": "M3C2-PF-01-seed-plan",
            "status": "PASS",
            "reason": f"{expected_replicas} complete, unique, nonoverlapping planned seeds per anchor",
        },
        {
            "gate": "M3C2-PF-02-artifact-identity",
            "status": "PASS" if artifacts_match else "FAIL",
            "reason": "source model and selected package artifacts equal the M3-C0 SHA-256 lock",
        },
        {
            "gate": "M3C2-PF-03-saved-feature-audit",
            "status": "PASS" if audit_integrity else "FAIL",
            "reason": "saved Brownian feature, configured seed expression, reported components, and active finite history were inspected",
        },
        {
            "gate": "M3C2-PF-04-saved-cohort-completeness",
            "status": "FAIL",
            "reason": "one saved history per anchor is not the preregistered 32-seed cohort",
        },
        {
            "gate": "M3C2-PF-05-saved-semantic-parity",
            "status": "FAIL",
            "reason": "native fields, one saved run, and an unverified effective random stream cannot authorize an ensemble comparison",
        },
        {
            "gate": "M3C2-PF-06-common-p1-companion",
            "status": "NOT_RUN",
            "reason": "32-seed COMSOL RZ-projected common-P1 companion is planned in run_matrix.csv",
        },
        {
            "gate": "M3C2-PF-07-b03-candidate",
            "status": "NOT_RUN",
            "reason": "32-seed B03 exact-OU common-P1 candidate is planned in run_matrix.csv",
        },
        {
            "gate": "M3C2-PF-08-comparison-authorization",
            "status": "BLOCKED",
            "reason": "both meaning-matched cohorts must exist before statistical evaluation",
        },
    ]
    output.mkdir(parents=True, exist_ok=False)
    _write_csv(
        output / "artifact_hashes.csv",
        artifact_rows,
        ("owner", "kind", "path", "expected_sha256", "actual_sha256", "status"),
    )
    _write_csv(
        output / "source_package_audit.csv",
        audits,
        (
            "case_id",
            "role",
            "background_dataset",
            "brownian_feature_tag",
            "brownian_seed_expression",
            "configured_seed_parameter_value",
            "effective_random_stream_identity",
            "history_rows",
            "brownian_evaluated_active_rows",
            "brownian_nonzero_rows",
            "brownian_phi_reported_nonzero_rows",
            "brownian_reported_components",
            "saved_run_count",
            "meaning_matched_saved_cohort",
            "mismatch_codes",
        ),
    )
    _write_csv(
        output / "seed_plan_receipt.csv",
        seed_receipts,
        (
            "case_id",
            "replica_count",
            "unique_seed_count",
            "replica_min",
            "replica_max",
            "seed_min",
            "seed_max",
            "execution_status",
            "validation_status",
        ),
    )
    _write_csv(
        output / "run_matrix.csv",
        matrix,
        (
            "case_id",
            "participant",
            "replica",
            "campaign_seed",
            "field_contract",
            "stochastic_contract",
            "comparison_role",
            "execution_status",
        ),
    )
    _write_csv(output / "gates.csv", gates, ("gate", "status", "reason"))
    manifest: dict[str, object] = {
        "schema_version": 1,
        "evaluation_id": str(config["evaluation_id"]),
        "evaluation_revision": int(config["evaluation_revision"]),
        "tool_revision": TOOL_REVISION,
        "generated_utc": datetime.now(UTC).isoformat(),
        "configuration_sha256": _sha256(config_path),
        "tool_sha256": _sha256(Path(__file__).resolve()),
        "seed_plan_sha256": _sha256(seed_path),
        "overall_status": overall_status,
        "comparison_decision": "NOT_AUTHORIZED",
        "accuracy_claim": "NOT_EVALUATED",
        "golden_truth": "NOT_CLAIMED",
        "solver_core_changed": False,
        "anchor_cases": [anchor.case_id for anchor in anchors],
        "campaign_cases": list(config["campaign_case_ids"]),
        "planned_replicas_per_participant": expected_replicas,
        "planned_run_count": len(matrix),
        "run_matrix_authority": "PROVISIONAL_SEED_ALLOCATION_ONLY",
        "campaign_lock_status": "NOT_CREATED",
        "saved_reference_run_count_per_anchor": 1,
        "future_companion_requirements": future_requirements,
        "semantic_mismatch_codes": sorted(mismatch_codes),
        "gates": gates,
    }
    (output / "comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# M3-C2 100 nm anchor preflight\n\n"
        f"Status: `{overall_status}`.\n\n"
        "The M3-C0 seed plan contains 32 unique seeds for both audited anchors. "
        "The saved Case A/P histories are exact-hash-locked provenance, but each is "
        "one native-field run whose effective random stream is not proven. The "
        "reported Brownian table has r/phi/z columns and nonzero phi values, but the "
        "source particle interface has no out-of-plane motion DOF; those columns are "
        "diagnostic output, not evidence of three-degree-of-freedom motion. The saved "
        "configured seed parameters also do not prove the random stream used by the "
        "GenerateUnique interface mode. The histories therefore cannot authorize a "
        "meaning-matched ensemble comparison against B03. The read-only model basis "
        "is [`../model_semantics_probe_v1/`](../model_semantics_probe_v1/README.md).\n\n"
        "`run_matrix.csv` is a provisional seed allocation for the first Case A 100 nm "
        "campaign: 32 common-P1 RZ-projected COMSOL companion replicas and 32 B03 "
        "candidate replicas. It is not an executable campaign lock; exact step, physics, "
        "boundary, input-hash, and pilot contracts remain to be fixed. "
        "No COMSOL model, dataset artifact, or solver-core file was changed.\n",
        encoding="utf-8",
    )
    return manifest


def _default_repository_root() -> Path:
    return Path(__file__).resolve().parents[5]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=_default_repository_root())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_preflight(
        args.config.resolve(), args.repository_root.resolve(), args.output.resolve()
    )
    print(json.dumps({"overall_status": manifest["overall_status"], "output": str(args.output)}))
    # Exit zero means that the fail-closed preflight report was generated, not
    # that the campaign is ready. Consumers must inspect overall_status.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
