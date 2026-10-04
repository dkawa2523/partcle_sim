"""Audit saved neutral-transport evidence without running COMSOL or solver code.

This tool is intentionally limited to the twelve locked M3-V packages.  It
replays the native linear Epstein force, characterizes the existing P15-E and
P16 formulae with the saved effective mixture molar mass, and records where
the saved exports cannot support a physical-applicability or trajectory claim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]
type JsonValue = str | int | float | bool | list["JsonValue"] | dict[str, "JsonValue"] | None
type Record = dict[str, str | int | float]

BOLTZMANN_J_K = 1.380649e-23
AVOGADRO_PER_MOL = 6.02214076e23
MOLAR_GAS_CONSTANT_J_PER_MOL_K = BOLTZMANN_J_K * AVOGADRO_PER_MOL
STATUS_PASS = "PASS"
STATUS_NOT_APPLICABLE = "NOT_APPLICABLE"
STATUS_NOT_TESTED = "NOT_TESTED"
ALLOWED_EVIDENCE_STATUSES = frozenset({STATUS_PASS, STATUS_NOT_APPLICABLE, STATUS_NOT_TESTED})

HISTORY_COLUMNS = (
    "particle_id",
    "time_s",
    "active_state_flag",
    "particle_radius_m",
    "velocity_r_m_per_s",
    "velocity_z_m_per_s",
    "Epstein_drag_force_r_N",
    "Epstein_drag_force_z_N",
    "local_gas_velocity_r_m_per_s",
    "local_gas_velocity_z_m_per_s",
    "local_gas_temperature_K",
    "local_gas_density_kg_per_m3",
    "local_gas_mean_free_path_m",
)
TRAJECTORY_HEAT_FLUX_COLUMNS = frozenset(
    {
        "local_translational_heat_flux_r_W_per_m2",
        "local_translational_heat_flux_z_W_per_m2",
    }
)
TRAJECTORY_PPR_GRADIENT_COLUMNS = frozenset(
    {
        "local_ppr_temperature_gradient_r_K_per_m",
        "local_ppr_temperature_gradient_z_K_per_m",
    }
)
FIELD_COLUMNS = (
    "inside_model_domain",
    "gas_temperature_K",
    "thermal_conductivity_W_per_mK",
    "gas_mean_free_path_m",
    "temperature_gradient_r_K_per_m",
    "temperature_gradient_z_K_per_m",
)


@dataclass(frozen=True, slots=True)
class PackageSpec:
    case_id: str
    hashes: dict[str, str]


@dataclass(frozen=True, slots=True)
class ClosureConfig:
    source: Path
    dataset_relative_path: str
    expected_comsol_version: str
    expected_rows: int
    expected_particles: int
    expected_times: int
    neutral_molar_mass_kg_per_mol: float
    diffuse_reflection_fraction: float
    thermal_conductivity_W_per_m_K: float
    minimum_mean_free_path_over_radius: float
    maximum_low_speed_ratio: float
    force_residual_limit: float
    equivalence_residual_limit: float
    source_models: dict[str, str]
    case_profiles: dict[str, dict[str, Any]]
    packages: tuple[PackageSpec, ...]


@dataclass(frozen=True, slots=True)
class HistoryMetrics:
    rows: int
    particles: int
    times: int
    active_rows: int
    history_columns: frozenset[str]
    linear_residual_p90: float
    linear_residual_max: float
    speed_ratio_s_p90: float
    speed_ratio_s_max: float
    knudsen_over_radius_min: float
    finite_linear_difference_p90: float
    finite_linear_difference_max: float
    low_speed_ratio_p90: float
    low_speed_ratio_max: float
    low_speed_coverage: float
    saved_row_p16_coverage: float


@dataclass(frozen=True, slots=True)
class BackgroundMetrics:
    finite_rows: int
    fourier_parameter_min: float
    fourier_parameter_p90: float
    fourier_parameter_max: float
    waldmann_equivalence_residual_max: float


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return {str(key): item for key, item in value.items()}


def _string_mapping(value: object, name: str) -> dict[str, str]:
    mapping = _mapping(value, name)
    if not all(isinstance(item, str) for item in mapping.values()):
        raise ValueError(f"{name} values must be strings")
    return {key: str(item) for key, item in mapping.items()}


def load_config(path: Path) -> ClosureConfig:
    raw = _mapping(json.loads(path.read_text(encoding="utf-8")), "configuration")
    structure = _mapping(raw["structure"], "structure")
    neutral = _mapping(raw["effective_neutral_sensitivity"], "effective_neutral_sensitivity")
    hashes = _mapping(raw["hashes"], "hashes")
    case_ids_value = raw["case_ids"]
    if not isinstance(case_ids_value, list) or not all(
        isinstance(case_id, str) for case_id in case_ids_value
    ):
        raise ValueError("case_ids must be a list of strings")
    case_ids = tuple(str(case_id) for case_id in case_ids_value)
    validate_case_identity(case_ids, case_ids, expected_count=12)

    manifests = _string_mapping(hashes["manifest_by_case_id"], "manifest hashes")
    histories = _string_mapping(hashes["history_by_case_id"], "history hashes")
    fields = _string_mapping(hashes["field_by_case_id"], "field hashes")
    global_parameters = _string_mapping(
        hashes["global_parameters_by_diameter_nm"], "global-parameter hashes"
    )
    settings = _string_mapping(hashes["settings_by_variant_case"], "settings hashes")
    validation = _string_mapping(hashes["validation_by_case"], "validation hashes")
    packages = tuple(
        PackageSpec(
            case_id=case_id,
            hashes=_package_hashes(
                case_id,
                manifests,
                histories,
                fields,
                global_parameters,
                settings,
                validation,
            ),
        )
        for case_id in case_ids
    )
    return ClosureConfig(
        source=path,
        dataset_relative_path=str(raw["dataset_relative_path"]),
        expected_comsol_version=str(raw["expected_comsol_version"]),
        expected_rows=int(structure["rows"]),
        expected_particles=int(structure["particles"]),
        expected_times=int(structure["times"]),
        neutral_molar_mass_kg_per_mol=float(neutral["molar_mass_kg_per_mol"]),
        diffuse_reflection_fraction=float(neutral["diffuse_reflection_fraction"]),
        thermal_conductivity_W_per_m_K=float(neutral["thermal_conductivity_W_per_m_K"]),
        minimum_mean_free_path_over_radius=float(neutral["minimum_mean_free_path_over_radius"]),
        maximum_low_speed_ratio=float(neutral["maximum_low_speed_ratio"]),
        force_residual_limit=float(neutral["force_vector_relative_residual_limit"]),
        equivalence_residual_limit=float(neutral["algebraic_equivalence_residual_limit"]),
        source_models=_string_mapping(raw["source_models"], "source_models"),
        case_profiles={
            key: _mapping(value, f"case profile {key}")
            for key, value in _mapping(raw["case_profiles"], "case_profiles").items()
        },
        packages=packages,
    )


def _case_parts(case_id: str) -> tuple[str, str, str]:
    variant, case_name = case_id.split("/", maxsplit=1)
    profile, diameter = case_name.split("_", maxsplit=1)
    return variant, profile, diameter.removesuffix("nm")


def _package_hashes(
    case_id: str,
    manifests: dict[str, str],
    histories: dict[str, str],
    fields: dict[str, str],
    global_parameters: dict[str, str],
    settings: dict[str, str],
    validation: dict[str, str],
) -> dict[str, str]:
    variant, profile, diameter = _case_parts(case_id)
    return {
        "manifest.csv": manifests[case_id],
        "validation/package_validation.csv": validation[profile],
        "config/global_parameters.csv": global_parameters[diameter],
        "config/particle_physics_feature_settings.csv": settings[f"{variant}/{profile}"],
        "input_fields/background_fields_regular_grid_301x301.csv": fields[case_id],
        "results/particle_history_full_tidy.csv": histories[case_id],
    }


def validate_case_identity(
    expected_case_ids: tuple[str, ...],
    observed_case_ids: tuple[str, ...],
    *,
    expected_count: int,
) -> None:
    expected = set(expected_case_ids)
    observed = set(observed_case_ids)
    if len(expected_case_ids) != expected_count or len(expected) != expected_count:
        raise ValueError(f"configuration must identify exactly {expected_count} unique cases")
    if len(observed_case_ids) != expected_count or len(observed) != expected_count:
        raise ValueError(f"dataset must contain exactly {expected_count} unique cases")
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise ValueError(f"case identity mismatch: missing={missing}, unexpected={unexpected}")


def require_new_output_path(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def linear_epstein_force(
    radius_m: FloatArray,
    gas_density_kg_per_m3: FloatArray,
    gas_temperature_K: FloatArray,
    relative_velocity_m_per_s: FloatArray,
    *,
    molar_mass_kg_per_mol: float,
    diffuse_reflection_fraction: float,
) -> FloatArray:
    molecular_mass_kg = molar_mass_kg_per_mol / AVOGADRO_PER_MOL
    mean_thermal_speed = np.sqrt(
        8.0 * BOLTZMANN_J_K * gas_temperature_K / (math.pi * molecular_mass_kg)
    )
    delta = 1.0 + diffuse_reflection_fraction * math.pi / 8.0
    friction = (
        4.0 * math.pi / 3.0 * radius_m**2 * gas_density_kg_per_m3 * mean_thermal_speed * delta
    )
    return friction[:, None] * relative_velocity_m_per_s


def _specular_speed_factor_scalar(speed_ratio_s: float) -> float:
    if speed_ratio_s <= 0.1:
        squared = speed_ratio_s * speed_ratio_s
        return 1.0 + squared / 5.0 - squared**2 / 70.0 + squared**3 / 630.0 - squared**4 / 5544.0
    inverse = 1.0 / speed_ratio_s
    exponential = math.exp(-(speed_ratio_s**2))
    return 3.0 / 16.0 * (2.0 + inverse**2) * exponential + 3.0 * math.sqrt(math.pi) / 32.0 * (
        4.0 * speed_ratio_s + 4.0 * inverse - inverse**3
    ) * math.erf(speed_ratio_s)


def finite_speed_rate_coefficient(
    speed_ratio_s: FloatArray, diffuse_reflection_fraction: float
) -> FloatArray:
    flat = np.asarray(speed_ratio_s, dtype=np.float64).ravel()
    specular = np.fromiter(
        (_specular_speed_factor_scalar(float(value)) for value in flat),
        dtype=np.float64,
        count=flat.size,
    ).reshape(np.shape(speed_ratio_s))
    return specular + diffuse_reflection_fraction * math.pi / 8.0


def waldmann_heat_flux_force(
    radius_m: float,
    heat_flux_W_per_m2: FloatArray,
    gas_temperature_K: FloatArray,
    *,
    molar_mass_kg_per_mol: float,
) -> FloatArray:
    molecular_mass_kg = molar_mass_kg_per_mol / AVOGADRO_PER_MOL
    mean_thermal_speed = np.sqrt(
        8.0 * BOLTZMANN_J_K * gas_temperature_K / (math.pi * molecular_mass_kg)
    )
    return (32.0 / 15.0) * radius_m**2 * heat_flux_W_per_m2 / mean_thermal_speed[:, None]


def waldmann_gradient_force(
    radius_m: float,
    thermal_conductivity_W_per_m_K: FloatArray,
    temperature_gradient_K_per_m: FloatArray,
    gas_temperature_K: FloatArray,
    *,
    molar_mass_kg_per_mol: float,
) -> FloatArray:
    coefficient = (
        -8.0
        * math.sqrt(2.0 * math.pi)
        / 15.0
        * radius_m**2
        * thermal_conductivity_W_per_m_K
        * np.sqrt(molar_mass_kg_per_mol / (MOLAR_GAS_CONSTANT_J_PER_MOL_K * gas_temperature_K))
    )
    return coefficient[:, None] * temperature_gradient_K_per_m


def trajectory_thermophoresis_replay_status(columns: frozenset[str]) -> str:
    has_heat_flux = TRAJECTORY_HEAT_FLUX_COLUMNS <= columns
    has_ppr_gradient = TRAJECTORY_PPR_GRADIENT_COLUMNS <= columns
    return STATUS_PASS if has_heat_flux or has_ppr_gradient else STATUS_NOT_TESTED


def _relative_vector_residual(first: FloatArray, second: FloatArray) -> FloatArray:
    absolute = np.linalg.norm(first - second, axis=1)
    scale = np.maximum(
        np.maximum(np.linalg.norm(first, axis=1), np.linalg.norm(second, axis=1)),
        np.finfo(np.float64).tiny,
    )
    return absolute / scale


def _read_numeric_columns(path: Path, names: tuple[str, ...]) -> tuple[FloatArray, frozenset[str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream))
    missing = sorted(set(names) - set(header))
    if missing:
        raise ValueError(f"{path} lacks required columns: {missing}")
    indices = tuple(header.index(name) for name in names)
    values = np.loadtxt(path, delimiter=",", skiprows=1, usecols=indices, ndmin=2)
    return np.asarray(values, dtype=np.float64), frozenset(header)


def _history_metrics(path: Path, config: ClosureConfig) -> HistoryMetrics:
    values, columns = _read_numeric_columns(path, HISTORY_COLUMNS)
    particle_id = values[:, 0]
    time_s = values[:, 1]
    active = values[:, 2] == 1.0
    active_values = values[active]
    if active_values.size == 0 or not np.isfinite(active_values).all():
        raise ValueError(f"active history primitives are missing or nonfinite: {path}")
    radius = active_values[:, 3]
    particle_velocity = active_values[:, 4:6]
    exported_force = active_values[:, 6:8]
    relative_velocity = active_values[:, 8:10] - particle_velocity
    temperature = active_values[:, 10]
    density = active_values[:, 11]
    mean_free_path = active_values[:, 12]
    if not bool(
        (radius > 0.0).all()
        and (temperature > 0.0).all()
        and (density > 0.0).all()
        and (mean_free_path > 0.0).all()
    ):
        raise ValueError(f"active history contains nonpositive neutral primitives: {path}")

    reconstructed = linear_epstein_force(
        radius,
        density,
        temperature,
        relative_velocity,
        molar_mass_kg_per_mol=config.neutral_molar_mass_kg_per_mol,
        diffuse_reflection_fraction=config.diffuse_reflection_fraction,
    )
    residual = _relative_vector_residual(reconstructed, exported_force)
    molecular_mass = config.neutral_molar_mass_kg_per_mol / AVOGADRO_PER_MOL
    c0 = np.sqrt(2.0 * BOLTZMANN_J_K * temperature / molecular_mass)
    c_bar = 2.0 * c0 / math.sqrt(math.pi)
    relative_speed = np.linalg.norm(relative_velocity, axis=1)
    speed_ratio_s = relative_speed / c0
    low_speed_ratio = relative_speed / c_bar
    knudsen_over_radius = mean_free_path / radius
    finite_coefficient = finite_speed_rate_coefficient(
        speed_ratio_s, config.diffuse_reflection_fraction
    )
    linear_coefficient = 1.0 + config.diffuse_reflection_fraction * math.pi / 8.0
    finite_linear_difference = np.abs(finite_coefficient / linear_coefficient - 1.0)
    low_speed_gate = low_speed_ratio <= config.maximum_low_speed_ratio
    p16_gate = low_speed_gate & (knudsen_over_radius >= config.minimum_mean_free_path_over_radius)
    return HistoryMetrics(
        rows=int(values.shape[0]),
        particles=int(np.unique(particle_id).size),
        times=int(np.unique(time_s).size),
        active_rows=int(active_values.shape[0]),
        history_columns=columns,
        linear_residual_p90=float(np.percentile(residual, 90.0)),
        linear_residual_max=float(np.max(residual)),
        speed_ratio_s_p90=float(np.percentile(speed_ratio_s, 90.0)),
        speed_ratio_s_max=float(np.max(speed_ratio_s)),
        knudsen_over_radius_min=float(np.min(knudsen_over_radius)),
        finite_linear_difference_p90=float(np.percentile(finite_linear_difference, 90.0)),
        finite_linear_difference_max=float(np.max(finite_linear_difference)),
        low_speed_ratio_p90=float(np.percentile(low_speed_ratio, 90.0)),
        low_speed_ratio_max=float(np.max(low_speed_ratio)),
        low_speed_coverage=float(np.mean(low_speed_gate)),
        saved_row_p16_coverage=float(np.mean(p16_gate)),
    )


def _field_metadata(path: Path) -> tuple[str, tuple[str, ...]]:
    version = ""
    description: tuple[str, ...] = ()
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.startswith("%"):
                break
            parsed = next(csv.reader([line.removeprefix("% ").rstrip("\n")]))
            if parsed and parsed[0] == "Version":
                version = parsed[1]
            elif parsed and parsed[0] == "Description":
                description = tuple(item.strip() for item in parsed[1].split(","))
    return version, description


def _background_metrics(path: Path, radius_m: float, config: ClosureConfig) -> BackgroundMetrics:
    version, description = _field_metadata(path)
    if version != config.expected_comsol_version:
        raise ValueError(f"unexpected COMSOL version in {path}: {version!r}")
    missing = sorted(set(FIELD_COLUMNS) - set(description))
    if missing:
        raise ValueError(f"{path} lacks required background fields: {missing}")
    indices = tuple(description.index(name) for name in FIELD_COLUMNS)
    values = np.loadtxt(path, delimiter=",", comments="%", usecols=indices, ndmin=2)
    inside = values[:, 0] == 1.0
    finite = inside & np.isfinite(values).all(axis=1)
    selected = values[finite]
    if selected.size == 0:
        raise ValueError(f"no finite inside-domain background rows: {path}")
    temperature = selected[:, 1]
    conductivity = selected[:, 2]
    mean_free_path = selected[:, 3]
    gradient = selected[:, 4:6]
    if not bool(
        (temperature > 0.0).all() and (conductivity > 0.0).all() and (mean_free_path > 0.0).all()
    ):
        raise ValueError(f"background contains nonpositive neutral primitives: {path}")
    heat_flux = -conductivity[:, None] * gradient
    heat_flux_force = waldmann_heat_flux_force(
        radius_m,
        heat_flux,
        temperature,
        molar_mass_kg_per_mol=config.neutral_molar_mass_kg_per_mol,
    )
    gradient_force = waldmann_gradient_force(
        radius_m,
        conductivity,
        gradient,
        temperature,
        molar_mass_kg_per_mol=config.neutral_molar_mass_kg_per_mol,
    )
    equivalence_residual = _relative_vector_residual(heat_flux_force, gradient_force)
    fourier_parameter = mean_free_path * np.linalg.norm(gradient, axis=1) / temperature
    return BackgroundMetrics(
        finite_rows=int(selected.shape[0]),
        fourier_parameter_min=float(np.min(fourier_parameter)),
        fourier_parameter_p90=float(np.percentile(fourier_parameter, 90.0)),
        fourier_parameter_max=float(np.max(fourier_parameter)),
        waldmann_equivalence_residual_max=float(np.max(equivalence_residual)),
    )


def _read_dict(path: Path, key: str, value: str) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return {row[key]: row[value] for row in csv.DictReader(stream)}


def _verify_package_metadata(
    package: Path, spec: PackageSpec, config: ClosureConfig
) -> tuple[float, float, float]:
    variant, profile, diameter = _case_parts(spec.case_id)
    manifest = _read_dict(package / "manifest.csv", "key", "value")
    expected_manifest = {
        "model_variant": variant,
        "case": profile.removeprefix("case"),
        "particle_diameter_nm": diameter,
        "coordinate_system": "2D axisymmetric r-z; all normalized CSV coordinates are SI metres",
        "solver_or_study_run_by_this_export": "false",
        "background_recomputed": "false",
        "particle_recomputed": "false",
    }
    if any(manifest.get(key) != value for key, value in expected_manifest.items()):
        raise ValueError(f"manifest provenance mismatch: {spec.case_id}")
    with (package / "validation/package_validation.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        validation_rows = list(csv.DictReader(stream))
    if not validation_rows or any(row.get("status") != STATUS_PASS for row in validation_rows):
        raise ValueError(f"package validation is not PASS: {spec.case_id}")

    parameters = _read_dict(
        package / "config/global_parameters.csv", "parameter", "evaluated_SI_value"
    )
    expected_parameters = {
        "Mmix": config.neutral_molar_mass_kg_per_mol,
        "sigmaR_p": config.diffuse_reflection_fraction,
        "k_mix": config.thermal_conductivity_W_per_m_K,
    }
    for name, expected in expected_parameters.items():
        actual = float(parameters[name])
        if not math.isclose(actual, expected, rel_tol=2.0e-15, abs_tol=0.0):
            raise ValueError(f"{spec.case_id} has unexpected {name}: {actual}")

    profile_config = config.case_profiles[profile]
    physics_tag = str(profile_config["physics_tag"])
    expected_settings = _string_mapping(profile_config["settings"], f"{profile} settings")
    with (package / "config/particle_physics_feature_settings.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    observed = {
        f"{row['feature_tag']}|{row['property']}": (row["value"], row["selected_entities"])
        for row in rows
        if row["physics_tag"] == physics_tag
    }
    for key, expected in expected_settings.items():
        if observed.get(key) != (expected, "[3]"):
            raise ValueError(f"{spec.case_id} has unexpected setting {key}: {observed.get(key)}")
    return (
        float(parameters["Mmix"]),
        float(parameters["sigmaR_p"]),
        float(parameters["k_mix"]),
    )


def _verify_hashes(package: Path, spec: PackageSpec) -> None:
    for relative_path, expected in spec.hashes.items():
        path = package / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(
                f"locked artifact hash mismatch for {spec.case_id}/{relative_path}: {actual}"
            )


def _detected_case_ids(dataset: Path) -> tuple[str, ...]:
    return tuple(
        sorted(
            f"{path.parent.parent.name}/{path.parent.name}"
            for path in (dataset / "cases").glob("*/*/external_reproduction")
            if path.is_dir()
        )
    )


def _verify_source_models(dataset: Path, config: ClosureConfig) -> None:
    for relative_path, expected in config.source_models.items():
        path = dataset / relative_path
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"source model identity mismatch: {relative_path}")


def _package_path(dataset: Path, case_id: str) -> Path:
    variant, case_name = case_id.split("/", maxsplit=1)
    return dataset / "cases" / variant / case_name / "external_reproduction"


def _evaluate_package(dataset: Path, spec: PackageSpec, config: ClosureConfig) -> Record:
    package = _package_path(dataset, spec.case_id)
    _verify_hashes(package, spec)
    molar_mass, diffuse_fraction, conductivity = _verify_package_metadata(package, spec, config)
    expected_settings = (
        config.neutral_molar_mass_kg_per_mol,
        config.diffuse_reflection_fraction,
        config.thermal_conductivity_W_per_m_K,
    )
    if not all(
        math.isclose(actual, expected, rel_tol=2.0e-15, abs_tol=0.0)
        for actual, expected in zip(
            (molar_mass, diffuse_fraction, conductivity), expected_settings, strict=True
        )
    ):
        raise ValueError(f"neutral settings mismatch after validation: {spec.case_id}")
    history = _history_metrics(package / "results/particle_history_full_tidy.csv", config)
    if (history.rows, history.particles, history.times) != (
        config.expected_rows,
        config.expected_particles,
        config.expected_times,
    ):
        raise ValueError(f"history structure mismatch: {spec.case_id}")
    _, _, diameter_nm = _case_parts(spec.case_id)
    background = _background_metrics(
        package / "input_fields/background_fields_regular_grid_301x301.csv",
        float(diameter_nm) * 0.5e-9,
        config,
    )
    linear_status = (
        STATUS_PASS if history.linear_residual_max <= config.force_residual_limit else "FAIL"
    )
    equivalence_status = (
        STATUS_PASS
        if background.waldmann_equivalence_residual_max <= config.equivalence_residual_limit
        else "FAIL"
    )
    if linear_status != STATUS_PASS or equivalence_status != STATUS_PASS:
        raise ValueError(f"formula replay failed: {spec.case_id}")
    return {
        "case_id": spec.case_id,
        "identity_status": STATUS_PASS,
        "settings_provenance_status": STATUS_PASS,
        "active_rows": history.active_rows,
        "linear_epstein_replay_status": linear_status,
        "linear_force_relative_residual_p90": history.linear_residual_p90,
        "linear_force_relative_residual_max": history.linear_residual_max,
        "effective_mmix_sensitivity_status": STATUS_PASS,
        "p15e_speed_ratio_S_p90": history.speed_ratio_s_p90,
        "p15e_speed_ratio_S_max": history.speed_ratio_s_max,
        "mean_free_path_over_radius_min": history.knudsen_over_radius_min,
        "finite_vs_linear_coefficient_relative_difference_p90": (
            history.finite_linear_difference_p90
        ),
        "finite_vs_linear_coefficient_relative_difference_max": (
            history.finite_linear_difference_max
        ),
        "p15e_existing_revision_physical_status": STATUS_NOT_APPLICABLE,
        "p16_low_speed_ratio_p90": history.low_speed_ratio_p90,
        "p16_low_speed_ratio_max": history.low_speed_ratio_max,
        "p16_low_speed_saved_row_coverage": history.low_speed_coverage,
        "p16_kn_and_low_speed_saved_row_coverage": history.saved_row_p16_coverage,
        "p16_existing_revision_physical_status": STATUS_NOT_APPLICABLE,
        "waldmann_algebraic_equivalence_status": equivalence_status,
        "waldmann_algebraic_relative_residual_max": (background.waldmann_equivalence_residual_max),
        "trajectory_thermophoresis_force_replay_status": (
            trajectory_thermophoresis_replay_status(history.history_columns)
        ),
        "continuous_path_applicability_status": STATUS_NOT_TESTED,
        "background_gradient_characterization_status": STATUS_PASS,
        "background_finite_rows": background.finite_rows,
        "background_lambda_gradT_over_T_min": background.fourier_parameter_min,
        "background_lambda_gradT_over_T_p90": background.fourier_parameter_p90,
        "background_lambda_gradT_over_T_max": background.fourier_parameter_max,
    }


def _gate(gate: str, status: str, reason: str) -> dict[str, JsonValue]:
    if status not in ALLOWED_EVIDENCE_STATUSES:
        raise ValueError(f"unsupported evidence status: {status}")
    return {"gate": gate, "status": status, "reason": reason}


def _summary(rows: list[Record]) -> dict[str, JsonValue]:
    def values(name: str) -> FloatArray:
        return np.asarray([float(row[name]) for row in rows], dtype=np.float64)

    active_rows = values("active_rows")
    low_speed_coverage = values("p16_low_speed_saved_row_coverage")
    return {
        "package_count": len(rows),
        "active_rows": int(np.sum(active_rows)),
        "linear_force_relative_residual_max": float(
            np.max(values("linear_force_relative_residual_max"))
        ),
        "p15e_speed_ratio_S_p90_case_range": [
            float(np.min(values("p15e_speed_ratio_S_p90"))),
            float(np.max(values("p15e_speed_ratio_S_p90"))),
        ],
        "p15e_speed_ratio_S_max": float(np.max(values("p15e_speed_ratio_S_max"))),
        "mean_free_path_over_radius_min": float(np.min(values("mean_free_path_over_radius_min"))),
        "finite_vs_linear_coefficient_relative_difference_p90_case_range": [
            float(np.min(values("finite_vs_linear_coefficient_relative_difference_p90"))),
            float(np.max(values("finite_vs_linear_coefficient_relative_difference_p90"))),
        ],
        "finite_vs_linear_coefficient_relative_difference_max": float(
            np.max(values("finite_vs_linear_coefficient_relative_difference_max"))
        ),
        "p16_low_speed_saved_row_coverage_case_range": [
            float(np.min(low_speed_coverage)),
            float(np.max(low_speed_coverage)),
        ],
        "p16_low_speed_saved_row_coverage_weighted": float(
            np.sum(active_rows * low_speed_coverage) / np.sum(active_rows)
        ),
        "background_lambda_gradT_over_T_range": [
            float(np.min(values("background_lambda_gradT_over_T_min"))),
            float(np.max(values("background_lambda_gradT_over_T_max"))),
        ],
        "waldmann_algebraic_relative_residual_max": float(
            np.max(values("waldmann_algebraic_relative_residual_max"))
        ),
    }


def _gates(summary: dict[str, JsonValue]) -> list[JsonValue]:
    package_count_value = summary["package_count"]
    residual_value = summary["linear_force_relative_residual_max"]
    if isinstance(package_count_value, bool) or not isinstance(package_count_value, int):
        raise ValueError("summary package_count must be an integer")
    if isinstance(residual_value, bool) or not isinstance(residual_value, int | float):
        raise ValueError("summary linear residual must be numeric")
    package_count = package_count_value
    residual = float(residual_value)
    return [
        _gate(
            "P18R-01-package-and-source-identity",
            STATUS_PASS,
            f"exactly {package_count} locked packages and two source MPH hashes verified",
        ),
        _gate(
            "P18R-02-neutral-settings-and-export-provenance",
            STATUS_PASS,
            "Epstein/Waldmann settings, Mmix, sigmaR, k_mix, validation, and no-rerun flags verified",
        ),
        _gate(
            "P18R-03-native-linear-epstein-formula-replay",
            STATUS_PASS,
            f"all saved active rows replayed; maximum vector relative residual={residual:.6g}",
        ),
        _gate(
            "P18R-04-p15e-effective-mmix-numerical-sensitivity",
            STATUS_PASS,
            "S, Kn, and finite-versus-linear coefficient differences characterized with Mmix only",
        ),
        _gate(
            "P18R-05-p15e-existing-revision-physical-applicability",
            STATUS_NOT_APPLICABLE,
            "CF4/O2 mixture accommodation semantics and particle surface temperature are not established",
        ),
        _gate(
            "P18R-06-waldmann-gradient-heat-flux-equivalence",
            STATUS_PASS,
            "q=-k grad(T), m=M/N_A, and R=N_A k_B give the same (32/15) a^2 q/c_bar force",
        ),
        _gate(
            "P18R-07-trajectory-thermophoresis-force-replay",
            STATUS_NOT_TESTED,
            "saved trajectory rows contain neither translational q nor trajectory-local PPR temperature gradient",
        ),
        _gate(
            "P18R-08-p16-saved-row-low-speed-characterization",
            STATUS_PASS,
            "effective-Mmix saved-row low-speed and Kn coverage characterized without claiming mixture physics",
        ),
        _gate(
            "P18R-09-p16-existing-revision-physical-applicability",
            STATUS_NOT_APPLICABLE,
            "the existing P16 revision is single-species while the saved reference is an unresolved CF4/O2 mixture",
        ),
        _gate(
            "P18R-10-continuous-path-applicability",
            STATUS_NOT_TESTED,
            "saved output frames do not certify all integrator stages or the continuous accepted path",
        ),
        _gate(
            "P18R-11-background-fourier-gradient-scale",
            STATUS_PASS,
            "lambda*|grad(T)|/T characterized on finite inside-domain saved grid rows only",
        ),
    ]


def _write_csv(path: Path, rows: list[Record]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_readme(path: Path, manifest: dict[str, JsonValue]) -> None:
    summary = _mapping(manifest["summary"], "summary")
    p16_range = summary["p16_low_speed_saved_row_coverage_case_range"]
    gradient_range = summary["background_lambda_gradT_over_T_range"]
    text = "\n".join(
        [
            "# P18-R neutral-transport closure v1",
            "",
            "Status: `PASS` for the offline audit. This is not a physical mixture-truth or trajectory certificate.",
            "",
            "The tool verified the exact 12-package identity, locked hashes, neutral-force settings, export",
            "provenance, and source MPH identity without running COMSOL. It replayed native COMSOL linear",
            f"Epstein on {summary['active_rows']} active saved rows; the maximum vector relative residual was",
            f"`{float(summary['linear_force_relative_residual_max']):.6g}`.",
            "",
            "Using the saved effective mixture molar mass only as a numerical sensitivity, it characterized",
            "P15-E speed ratio, Kn, and finite-versus-linear coefficient differences. Existing P15-E is",
            "`NOT_APPLICABLE` as physical reference closure because CF4/O2 species/accommodation semantics",
            "and particle surface temperature are unknown.",
            "",
            "The Waldmann heat-flux form is algebraically identical to the gradient form after substituting",
            "`q=-k grad(T)`, `m=M/N_A`, and `R=N_A k_B`. Existing P16 saved-row low-speed coverage spans",
            f"`{float(p16_range[0]):.6g}` to `{float(p16_range[1]):.6g}` across cases, but its single-species",
            "physical applicability is `NOT_APPLICABLE`. The trajectory rows export neither translational",
            "heat flux nor the PPR temperature gradient, so numerical thermophoretic-force replay is",
            "`NOT_TESTED`.",
            "",
            "On finite inside-domain background-grid rows, the descriptive `lambda*|grad(T)|/T` range was",
            f"`{float(gradient_range[0]):.6g}` to `{float(gradient_range[1]):.6g}`. This does not replace",
            "trajectory-local PPR provenance or continuous-path certification.",
            "",
            "Decision: `epstein_linear_effective_gas_sensitivity_v1` and",
            "`waldmann_gallis_free_molecular_effective_gas_heat_flux_sensitivity_v1` are required for",
            "same-form reference comparison, with an explicit finite `maximum_speed_ratio` in `(0, 1]`.",
            "They are not species-resolved mixture truth. Continuous-path",
            "applicability remains runtime-certification work and is `NOT_TESTED` here.",
            "",
            "`case_matrix.csv` contains all per-package metrics and independent status columns.",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def build_evidence(
    config: ClosureConfig, repository_root: Path, output: Path
) -> dict[str, JsonValue]:
    require_new_output_path(output)
    dataset = repository_root / config.dataset_relative_path
    expected_ids = tuple(spec.case_id for spec in config.packages)
    validate_case_identity(expected_ids, _detected_case_ids(dataset), expected_count=12)
    _verify_source_models(dataset, config)
    rows = [_evaluate_package(dataset, spec, config) for spec in config.packages]
    summary = _summary(rows)
    gates = _gates(summary)
    manifest: dict[str, JsonValue] = {
        "schema_version": 1,
        "evaluation_id": "P18-R",
        "evaluation_revision": 1,
        "classification": "EXTERNAL_VV_NEUTRAL_TRANSPORT_CLOSURE",
        "execution_status": STATUS_PASS,
        "generated_utc": datetime.now(UTC).isoformat(),
        "configuration_sha256": _sha256(config.source),
        "tool_sha256": _sha256(Path(__file__).resolve()),
        "dataset_relative_path": config.dataset_relative_path,
        "dataset_role": "REFERENCE_INPUT_NOT_GOLDEN_TRUTH",
        "comsol_run_performed": False,
        "solver_core_called": False,
        "summary": summary,
        "gates": gates,
        "decision": {
            "epstein_linear_effective_gas_sensitivity_v1": (
                "REQUIRED_FOR_SAME_FORM_REFERENCE_COMPARISON"
            ),
            "waldmann_gallis_free_molecular_effective_gas_heat_flux_sensitivity_v1": (
                "REQUIRED_FOR_SAME_FORM_REFERENCE_COMPARISON"
            ),
            "required_runtime_applicability_input": "explicit maximum_speed_ratio in (0, 1]",
            "species_resolved_mixture_truth": "NOT_CLAIMED",
            "continuous_path_runtime_certification": STATUS_NOT_TESTED,
        },
        "limitations": [
            "Mmix is used only to quantify formula sensitivity; it does not resolve mixture moments",
            "particle surface temperature and species accommodation semantics are not exported",
            "trajectory-local translational heat flux and PPR temperature gradient are not exported",
            "saved frames do not certify integrator stages or continuous accepted paths",
            "background gradient ranges are descriptive saved-grid evidence, not a force replay",
        ],
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    _write_csv(output / "case_matrix.csv", rows)
    (output / "comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_readme(output / "README.md", manifest)
    return manifest


def _default_repository_root() -> Path:
    return Path(__file__).resolve().parents[5]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=_default_repository_root())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    manifest = build_evidence(config, args.repository_root.resolve(), args.output.resolve())
    print(
        json.dumps({"execution_status": manifest["execution_status"], "output": str(args.output)})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
