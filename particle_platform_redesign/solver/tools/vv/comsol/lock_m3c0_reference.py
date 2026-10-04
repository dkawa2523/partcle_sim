"""Freeze the existing 12-package reference identity and expose M3-C0 gaps.

This is an external V&V tool.  It does not import the production solver and it
does not launch or modify COMSOL.  The current dataset is intentionally treated
as evidence to audit, not as a golden physics definition.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import yaml

Record = dict[str, str]


@dataclass(frozen=True)
class LockConfig:
    source_path: Path
    evaluation_id: str
    evaluation_revision: int
    expected_comsol_version: str
    dataset_relative_path: str
    expected_particles: int
    expected_times: int
    expected_rows: int
    variants: dict[str, str]
    cases: tuple[str, ...]
    model_files: dict[str, tuple[str, str]]
    required_package_files: tuple[str, ...]
    required_field_semantics: dict[str, str]
    candidate_steps_s: tuple[float, ...]
    replicas_per_package: int
    seed_base: int


@dataclass(frozen=True)
class PackageEvidence:
    inventory: Record
    artifacts: list[Record]
    formulas: list[Record]
    fields: list[Record]
    structure_ok: bool
    semantics_ok: bool
    brownian_off: bool
    freeze_seen: bool
    disappear_seen: bool


@dataclass(frozen=True)
class GateFacts:
    package_ok: bool
    semantics_ok: bool
    all_brownian_off: bool
    freeze_seen: bool
    disappear_seen: bool
    formula_lock_ok: bool


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return cast(dict[str, object], value)


def load_config(path: Path) -> LockConfig:
    raw = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), "configuration")
    if raw.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    dataset = _mapping(raw["dataset"], "dataset")
    variants = _mapping(raw["variants"], "variants")
    models = _mapping(raw["model_files"], "model_files")
    semantics = _mapping(raw["required_field_semantics"], "required_field_semantics")
    campaign = _mapping(raw["stochastic_campaign"], "stochastic_campaign")
    model_files: dict[str, tuple[str, str]] = {}
    for variant, value in models.items():
        model = _mapping(value, f"model_files.{variant}")
        model_files[variant] = (str(model["path"]), str(model["sha256"]).lower())
    return LockConfig(
        source_path=path.resolve(),
        evaluation_id=str(raw["evaluation_id"]),
        evaluation_revision=int(cast(int, raw["evaluation_revision"])),
        expected_comsol_version=str(raw["expected_comsol_version"]),
        dataset_relative_path=str(dataset["relative_path"]),
        expected_particles=int(cast(int, dataset["expected_particles"])),
        expected_times=int(cast(int, dataset["expected_times"])),
        expected_rows=int(cast(int, dataset["expected_rows"])),
        variants={str(key): str(value) for key, value in variants.items()},
        cases=tuple(str(value) for value in cast(list[object], raw["cases"])),
        model_files=model_files,
        required_package_files=tuple(
            str(value) for value in cast(list[object], raw["required_package_files"])
        ),
        required_field_semantics={str(key): str(value) for key, value in semantics.items()},
        candidate_steps_s=tuple(
            float(cast(float | int | str, value))
            for value in cast(list[object], raw["candidate_steps_s"])
        ),
        replicas_per_package=int(cast(int, campaign["replicas_per_package"])),
        seed_base=int(cast(int, campaign["seed_base"])),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_text_sha256(path: Path) -> str:
    """Hash COMSOL CSV payload while excluding its nonphysical export date."""
    digest = hashlib.sha256()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for line in stream:
            if line.startswith("% Date,"):
                continue
            digest.update(line.replace("\r\n", "\n").encode())
    return digest.hexdigest()


def _read_csv(path: Path) -> list[Record]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _write_csv(path: Path, rows: list[Record], columns: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _manifest(package: Path) -> dict[str, str]:
    return {row["key"]: row["value"] for row in _read_csv(package / "manifest.csv")}


def _normalize_expression(value: str) -> str:
    return re.sub(r"\s+", "", value)


def _project_config(package: Path) -> dict[str, str]:
    """Return physical settings with labels and downstream force sums excluded."""
    projected: dict[str, str] = {}
    config = package / "config"
    schemas = {
        "global_parameters.csv": ("parameter", "expression", "evaluated_SI_value"),
        "physical_constants.csv": ("symbol", "SI_value", "unit"),
        "boundary_conditions_and_event_codes.csv": (
            "feature_tag",
            "boundary_ids",
            "condition",
            "status_code",
            "physical_reflection_enabled",
        ),
        "background_field_column_dictionary.csv": ("column", "COMSOL_expression", "unit"),
    }
    for filename, columns in schemas.items():
        for row in _read_csv(config / filename):
            key = "/".join(row[column] for column in columns[:-1])
            projected[f"{filename}:{key}"] = _normalize_expression(row[columns[-1]])
    for row in _read_csv(config / "particle_physics_feature_settings.csv"):
        key = "/".join(
            (row["physics_tag"], row["feature_tag"], row["selected_entities"], row["property"])
        )
        projected[f"particle_physics_feature_settings.csv:{key}"] = _normalize_expression(
            row["value"]
        )
    ignored_study_properties = {"outputInterface", "physselection"}
    for row in _read_csv(config / "study_and_solver_settings.csv"):
        if row["property"] in ignored_study_properties:
            continue
        key = "/".join((row["study_tag"], row["feature_tag"], row["property"]))
        projected[f"study_and_solver_settings.csv:{key}"] = _normalize_expression(row["value"])
    for row in _read_csv(config / "particle_output_column_dictionary.csv"):
        column = row["column"]
        if column.startswith(("ion_drag_force_", "sum_exported_")):
            continue
        projected[f"particle_output_column_dictionary.csv:{column}/unit"] = row["unit"]
        projected[f"particle_output_column_dictionary.csv:{column}/expression"] = (
            _normalize_expression(row["COMSOL_expression"])
        )
    return projected


def _allowed_variant_difference(key: str) -> bool:
    return (
        key.startswith("particle_physics_feature_settings.csv:")
        and "/idf/" in key
        and key.endswith("/F")
    )


def _variant_differences(first: Path, second: Path) -> list[Record]:
    left = _project_config(first)
    right = _project_config(second)
    rows: list[Record] = []
    for key in sorted(left.keys() | right.keys()):
        if left.get(key) == right.get(key):
            continue
        allowed = _allowed_variant_difference(key)
        rows.append(
            {
                "setting": key,
                "theory_value": left.get(key, "<missing>"),
                "image_value": right.get(key, "<missing>"),
                "classification": "ION_DRAG_ALLOWED" if allowed else "CONFOUNDER",
            }
        )
    return rows


def _history_summary(path: Path) -> tuple[int, int, int, Counter[str], bool, bool, bool]:
    rows = 0
    particles: set[str] = set()
    times: set[str] = set()
    last_status: dict[str, str] = {}
    brownian_nonzero = False
    freeze_seen = False
    disappear_seen = False
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            rows += 1
            particle = row["particle_id"]
            particles.add(particle)
            times.add(row["time_s"])
            status = row["current_status_code"]
            last_status[particle] = status
            freeze_seen |= status == "2"
            disappear_seen |= status == "4"
            brownian = float(row["Brownian_force_magnitude_N"])
            brownian_nonzero |= brownian != 0.0
    return (
        rows,
        len(particles),
        len(times),
        Counter(last_status.values()),
        brownian_nonzero,
        freeze_seen,
        disappear_seen,
    )


def _formula_rows(case_id: str, package: Path) -> list[Record]:
    wanted = {("auxq", "R"), ("idf", "F"), ("depf", "F"), ("liftfm", "F"), ("bf1", "i")}
    rows: list[Record] = []
    for row in _read_csv(package / "config/particle_physics_feature_settings.csv"):
        if (row["feature_tag"], row["property"]) not in wanted:
            continue
        rows.append(
            {
                "case_id": case_id,
                "physics_tag": row["physics_tag"],
                "feature_tag": row["feature_tag"],
                "property": row["property"],
                "selected_entities": row["selected_entities"],
                "normalized_value": _normalize_expression(row["value"]),
            }
        )
    return rows


def _field_rows(case_id: str, package: Path, required: dict[str, str]) -> tuple[list[Record], bool]:
    dictionary = {
        row["column"]: row
        for row in _read_csv(package / "config/background_field_column_dictionary.csv")
    }
    rows: list[Record] = []
    complete = True
    for column, expected_unit in required.items():
        source = dictionary.get(column)
        passed = source is not None and source["unit"] == expected_unit
        complete &= passed
        rows.append(
            {
                "case_id": case_id,
                "canonical_column": column,
                "COMSOL_expression": "<missing>" if source is None else source["COMSOL_expression"],
                "unit": "<missing>" if source is None else source["unit"],
                "name_unit_status": "PASS" if passed else "FAIL",
                "solution_averaging_recovery_topology": "NOT_EXPORTED",
            }
        )
    return rows, complete


def _gate(gate: str, status: str, reason: str) -> Record:
    return {"gate": gate, "status": status, "reason": reason}


def _model_identity(config: LockConfig, dataset: Path) -> tuple[list[Record], bool]:
    model_rows: list[Record] = []
    models_ok = True
    for variant, (relative_path, expected_hash) in config.model_files.items():
        path = dataset / relative_path
        actual_hash = _sha256(path) if path.is_file() else "<missing>"
        passed = actual_hash == expected_hash
        models_ok &= passed
        model_rows.append(
            {
                "variant": variant,
                "path": relative_path,
                "expected_sha256": expected_hash,
                "actual_sha256": actual_hash,
                "status": "PASS" if passed else "FAIL",
            }
        )
    return model_rows, models_ok


def _missing_package(case_id: str, model_revision: str, missing: list[str]) -> PackageEvidence:
    inventory = {
        "case_id": case_id,
        "model_revision": model_revision,
        "rows": "0",
        "particles": "0",
        "times": "0",
        "terminal_status_counts": "",
        "brownian_off": "UNKNOWN",
        "freeze_seen": "False",
        "disappear_seen": "False",
        "status": "FAIL_MISSING:" + ";".join(missing),
    }
    return PackageEvidence(inventory, [], [], [], False, False, False, False, False)


def _artifact_rows(config: LockConfig, case_id: str, package: Path) -> list[Record]:
    rows: list[Record] = []
    for relative_path in config.required_package_files:
        path = package / relative_path
        rows.append(
            {
                "case_id": case_id,
                "path": relative_path,
                "sha256": _sha256(path),
                "normalized_payload_sha256": (
                    _normalized_text_sha256(path)
                    if relative_path.startswith("input_fields/")
                    else "NOT_APPLICABLE"
                ),
                "bytes": str(path.stat().st_size),
            }
        )
    return rows


def _collect_package(
    config: LockConfig,
    variant: str,
    model_revision: str,
    case_name: str,
    package: Path,
) -> PackageEvidence:
    case_id = f"{variant}/{case_name}"
    missing = [name for name in config.required_package_files if not (package / name).is_file()]
    if missing:
        return _missing_package(case_id, model_revision, missing)
    metadata = _manifest(package)
    history = _history_summary(package / "results/particle_history_full_tidy.csv")
    rows, particles, times, statuses, brownian_nonzero, freeze_seen, disappear_seen = history
    structure_ok = (
        metadata.get("model_variant") == variant
        and metadata.get("case") == case_name[4]
        and metadata.get("particle_diameter_nm") == case_name.split("_")[1].removesuffix("nm")
        and rows == config.expected_rows
        and particles == config.expected_particles
        and times == config.expected_times
    )
    inventory = {
        "case_id": case_id,
        "model_revision": model_revision,
        "rows": str(rows),
        "particles": str(particles),
        "times": str(times),
        "terminal_status_counts": ";".join(f"{key}:{statuses[key]}" for key in sorted(statuses)),
        "brownian_off": str(not brownian_nonzero),
        "freeze_seen": str(freeze_seen),
        "disappear_seen": str(disappear_seen),
        "status": "PASS" if structure_ok else "FAIL",
    }
    fields, semantics_ok = _field_rows(case_id, package, config.required_field_semantics)
    return PackageEvidence(
        inventory=inventory,
        artifacts=_artifact_rows(config, case_id, package),
        formulas=_formula_rows(case_id, package),
        fields=fields,
        structure_ok=structure_ok,
        semantics_ok=semantics_ok,
        brownian_off=not brownian_nonzero,
        freeze_seen=freeze_seen,
        disappear_seen=disappear_seen,
    )


def _collect_packages(
    config: LockConfig, packages_root: Path
) -> tuple[list[PackageEvidence], dict[tuple[str, str], Path]]:
    evidence: list[PackageEvidence] = []
    paths: dict[tuple[str, str], Path] = {}
    for variant, model_revision in config.variants.items():
        for case_name in config.cases:
            package = packages_root / variant / case_name / "external_reproduction"
            paths[(variant, case_name)] = package
            evidence.append(_collect_package(config, variant, model_revision, case_name, package))
    return evidence, paths


def _hash_difference(
    case_name: str, relative_path: str, first: Path, second: Path, *, normalized: bool
) -> Record | None:
    hash_function = _normalized_text_sha256 if normalized else _sha256
    left = hash_function(first / relative_path)
    right = hash_function(second / relative_path)
    if left == right:
        return None
    return {
        "case": case_name,
        "setting": relative_path,
        "theory_value": left,
        "image_value": right,
        "classification": "CONFOUNDER",
    }


def _projected_csv_sha256(path: Path, columns: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for row in _read_csv(path):
        digest.update("\x1f".join(row[column] for column in columns).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _pair_differences(
    config: LockConfig, package_paths: dict[tuple[str, str], Path]
) -> tuple[list[Record], bool]:
    variant_names = tuple(config.variants)
    if len(variant_names) != 2:
        raise ValueError("M3-C0a requires exactly two ion-drag variants")
    rows: list[Record] = []
    geometry = (
        "geometry/mesh_vertices_si.csv",
        "geometry/boundary_edges.csv",
        "geometry/domain_triangles.csv",
        "geometry/domain_quadrilaterals.csv",
    )
    fields = (
        "input_fields/background_fields_mesh_points.csv",
        "input_fields/background_fields_regular_grid_301x301.csv",
    )
    for case_name in config.cases:
        first = package_paths[(variant_names[0], case_name)]
        second = package_paths[(variant_names[1], case_name)]
        if not first.is_dir() or not second.is_dir():
            rows.append(
                {
                    "case": case_name,
                    "setting": "package_directory",
                    "theory_value": str(first.is_dir()),
                    "image_value": str(second.is_dir()),
                    "classification": "CONFOUNDER",
                }
            )
            continue
        differences = _variant_differences(first, second)
        for row in differences:
            row["case"] = case_name
        rows.extend(differences)
        rows.extend(
            difference
            for path in geometry
            if (difference := _hash_difference(case_name, path, first, second, normalized=False))
            is not None
        )
        rows.extend(
            difference
            for path in fields
            if (difference := _hash_difference(case_name, path, first, second, normalized=True))
            is not None
        )
        initial_state = "results/release_state_t0_tidy.csv"
        initial_columns = (
            "particle_id",
            "time_s",
            "r_m",
            "z_m",
            "velocity_r_m_per_s",
            "velocity_phi_m_per_s",
            "velocity_z_m_per_s",
            "charge_number_e",
            "particle_diameter_m",
            "particle_mass_kg",
        )
        left = _projected_csv_sha256(first / initial_state, initial_columns)
        right = _projected_csv_sha256(second / initial_state, initial_columns)
        if left != right:
            rows.append(
                {
                    "case": case_name,
                    "setting": initial_state + ":initial_state_projection",
                    "theory_value": left,
                    "image_value": right,
                    "classification": "CONFOUNDER",
                }
            )
    return rows, all(row["classification"] == "ION_DRAG_ALLOWED" for row in rows)


def _seed_rows(config: LockConfig, package_rows: list[Record]) -> list[Record]:
    rows: list[Record] = []
    for package_index, package in enumerate(package_rows):
        rows.extend(
            {
                "case_id": package["case_id"],
                "replica": str(replica),
                "seed": str(
                    config.seed_base + package_index * config.replicas_per_package + replica
                ),
                "execution_status": "PLANNED_NOT_RUN",
            }
            for replica in range(config.replicas_per_package)
        )
    return rows


def _step_rows(config: LockConfig) -> list[Record]:
    return [
        {
            "step_s": f"{step:.17g}",
            "role": "candidate",
            "admissibility": "NOT_TESTED_BROWNIAN_OFF_FULL_PHYSICS",
        }
        for step in config.candidate_steps_s
    ]


def _gate_facts(packages: list[PackageEvidence]) -> GateFacts:
    return GateFacts(
        package_ok=len(packages) == 12 and all(row.structure_ok for row in packages),
        semantics_ok=bool(packages) and all(row.semantics_ok for row in packages),
        all_brownian_off=bool(packages) and all(row.brownian_off for row in packages),
        freeze_seen=any(row.freeze_seen for row in packages),
        disappear_seen=any(row.disappear_seen for row in packages),
        formula_lock_ok=bool(packages) and all(row.formulas and row.artifacts for row in packages),
    )


def _pass_fail(value: bool) -> str:
    return "PASS" if value else "FAIL"


def _pass_not_tested(value: bool) -> str:
    return "PASS" if value else "NOT_TESTED"


def _gate_rows(
    config: LockConfig,
    *,
    models_ok: bool,
    packages: list[PackageEvidence],
    pair_ok: bool,
) -> list[Record]:
    facts = _gate_facts(packages)
    return [
        _gate("M3C0A-01-model-identity", _pass_fail(models_ok), "MPH hashes"),
        _gate(
            "M3C0A-02-package-structure",
            _pass_fail(facts.package_ok),
            "12 packages, 287 particles, 121 times and required files",
        ),
        _gate(
            "M3C0A-03-formula-parameter-lock",
            _pass_fail(facts.formula_lock_ok),
            "exact config and reference artifacts hashed; selected charge/force rows normalized",
        ),
        _gate(
            "M3C0A-04-field-name-unit-lock",
            _pass_fail(facts.semantics_ok),
            "required canonical primitive names, COMSOL expressions and units",
        ),
        _gate(
            "M3C0A-05-derived-field-full-provenance",
            "NOT_TESTED",
            "solution tag, averaging definition, recovery method and topology id were not exported",
        ),
        _gate(
            "M3C0A-06-ion-drag-only-pair",
            _pass_fail(pair_ok),
            "all non-ion-drag physical settings and field/geometry payloads must match",
        ),
        _gate(
            "M3C0A-07-brownian-off-references",
            _pass_not_tested(facts.all_brownian_off),
            "all off"
            if facts.all_brownian_off
            else "existing histories contain nonzero Brownian force",
        ),
        _gate(
            "M3C0A-08-admissible-step-series",
            "NOT_TESTED",
            "candidate 10/5/2.5 us series requires both-solver full-physics characterization",
        ),
        _gate(
            "M3C0A-09-freeze-positive-example",
            _pass_not_tested(facts.freeze_seen),
            "observed" if facts.freeze_seen else "no status-code 2 sample in existing histories",
        ),
        _gate(
            "M3C0A-10-disappear-observation",
            _pass_not_tested(facts.disappear_seen),
            "status-code 4 is present; dedicated pre/post-hit microcase remains future evidence",
        ),
        _gate(
            "M3C0A-11-seed-cohort",
            "PASS",
            f"{config.replicas_per_package} unique preregistered seeds per package",
        ),
    ]


def _write_tables(
    output: Path,
    model_rows: list[Record],
    packages: list[PackageEvidence],
    difference_rows: list[Record],
    step_rows: list[Record],
    seed_rows: list[Record],
    gates: list[Record],
) -> None:
    package_rows = [row.inventory for row in packages]
    artifact_rows = [item for row in packages for item in row.artifacts]
    formula_rows = [item for row in packages for item in row.formulas]
    field_rows = [item for row in packages for item in row.fields]
    _write_csv(
        output / "model_identity.csv",
        model_rows,
        ("variant", "path", "expected_sha256", "actual_sha256", "status"),
    )
    _write_csv(
        output / "package_inventory.csv",
        package_rows,
        (
            "case_id",
            "model_revision",
            "rows",
            "particles",
            "times",
            "terminal_status_counts",
            "brownian_off",
            "freeze_seen",
            "disappear_seen",
            "status",
        ),
    )
    _write_csv(
        output / "artifact_hashes.csv",
        artifact_rows,
        ("case_id", "path", "sha256", "normalized_payload_sha256", "bytes"),
    )
    _write_csv(
        output / "formula_lock.csv",
        formula_rows,
        (
            "case_id",
            "physics_tag",
            "feature_tag",
            "property",
            "selected_entities",
            "normalized_value",
        ),
    )
    _write_csv(
        output / "field_semantics.csv",
        field_rows,
        (
            "case_id",
            "canonical_column",
            "COMSOL_expression",
            "unit",
            "name_unit_status",
            "solution_averaging_recovery_topology",
        ),
    )
    _write_csv(
        output / "variant_pair_differences.csv",
        difference_rows,
        ("case", "setting", "theory_value", "image_value", "classification"),
    )
    _write_csv(
        output / "candidate_steps.csv",
        step_rows,
        ("step_s", "role", "admissibility"),
    )
    _write_csv(
        output / "seed_cohort.csv",
        seed_rows,
        ("case_id", "replica", "seed", "execution_status"),
    )
    _write_csv(output / "gates.csv", gates, ("gate", "status", "reason"))


def _write_manifest(
    config: LockConfig,
    dataset: Path,
    output: Path,
    package_count: int,
    seed_count: int,
    gates: list[Record],
) -> dict[str, object]:
    ready = all(row["status"] == "PASS" for row in gates)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "evaluation_id": config.evaluation_id,
        "evaluation_revision": config.evaluation_revision,
        "configuration_sha256": _sha256(config.source_path),
        "tool_sha256": _sha256(Path(__file__).resolve()),
        "expected_comsol_version": config.expected_comsol_version,
        "generated_utc": datetime.now(UTC).isoformat(),
        "classification": "EXTERNAL_VV_REFERENCE_READINESS_LOCK",
        "overall_status": "READY_FOR_P18" if ready else "PARTIAL_RERUN_REQUIRED",
        "golden_truth": "NOT_CLAIMED",
        "solver_core_changed": False,
        "dataset_root": str(dataset.resolve()),
        "packages": package_count,
        "seed_rows": seed_count,
        "gates": gates,
        "limitations": [
            "existing 12 histories are Brownian-on single-seed references",
            "accepted RK steps and RK stages are not exported",
            "positive Freeze behavior is not exercised",
            "derived-field solution/averaging/recovery/topology provenance is incomplete",
            "Case-P ion-drag variants contain a non-ion-drag lift confounder",
        ],
    }
    (output / "comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = "\n".join(
        [
            "# M3-C0a offline reference lock",
            "",
            f"Overall status: `{manifest['overall_status']}`.",
            "",
            "This artifact freezes the existing 12-package identity, formulas, parameters,",
            "field names/units, variant differences, candidate steps, and future seed cohort.",
            "It is not a COMSOL-accuracy certificate and does not complete M3-C0.",
            "",
            "The remaining blocking work is a new read-only-copy COMSOL campaign with",
            "Brownian disabled, an ion-drag-only paired configuration, admissible step",
            "characterization, RK/accepted-step probes, and positive Freeze/Disappear",
            "microcases.  Missing evidence is never inferred from saved frames.",
            "",
        ]
    )
    (output / "README.md").write_text(summary, encoding="utf-8")
    return manifest


def build_lock(config: LockConfig, repository_root: Path, output: Path) -> dict[str, object]:
    dataset = repository_root / config.dataset_relative_path
    output.mkdir(parents=True, exist_ok=False)
    model_rows, models_ok = _model_identity(config, dataset)
    packages, package_paths = _collect_packages(config, dataset / "cases")
    difference_rows, pair_ok = _pair_differences(config, package_paths)
    package_rows = [row.inventory for row in packages]
    seed_rows = _seed_rows(config, package_rows)
    step_rows = _step_rows(config)
    gates = _gate_rows(config, models_ok=models_ok, packages=packages, pair_ok=pair_ok)
    _write_tables(output, model_rows, packages, difference_rows, step_rows, seed_rows, gates)
    return _write_manifest(config, dataset, output, len(packages), len(seed_rows), gates)


def _default_repository_root() -> Path:
    return Path(__file__).resolve().parents[5]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=_default_repository_root())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    manifest = build_lock(config, args.repository_root.resolve(), args.output.resolve())
    print(json.dumps({"overall_status": manifest["overall_status"], "output": str(args.output)}))
    return 2 if args.strict and manifest["overall_status"] != "READY_FOR_P18" else 0


if __name__ == "__main__":
    raise SystemExit(main())
