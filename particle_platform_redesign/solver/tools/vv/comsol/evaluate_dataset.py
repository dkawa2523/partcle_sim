"""Evaluate the M3-V COMSOL reference matrix without importing solver internals."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int64]
type Scalar = str | int | float | bool | None
type Record = dict[str, Scalar]

TOOL_REVISION = "m3v-v3"
ELEMENTARY_CHARGE_C = 1.602176634e-19
BOLTZMANN_J_K = 1.380649e-23
VACUUM_PERMITTIVITY_F_M = 8.8541878128e-12
ELECTRON_MASS_KG = 9.1093837139e-31
AVOGADRO_PER_MOL = 6.02214076e23

ACTIVE_COLUMNS = (
    "particle_id",
    "time_s",
    "particle_radius_m",
    "particle_mass_kg",
    "charge_number_e",
    "dynamic_charge_rate_dZdt_per_s",
    "velocity_r_m_per_s",
    "velocity_z_m_per_s",
    "local_gas_velocity_r_m_per_s",
    "local_gas_velocity_z_m_per_s",
    "local_gas_temperature_K",
    "local_absolute_pressure_Pa",
    "local_gas_density_kg_per_m3",
    "local_gas_mean_free_path_m",
    "local_electron_density_per_m3",
    "local_total_positive_ion_density_per_m3",
    "electron_temperature_eV_as_V",
    "effective_positive_ion_mass_kg",
    "local_ion_velocity_r_m_per_s",
    "local_ion_velocity_z_m_per_s",
    "local_ion_thermal_energy_eV_as_V",
    "local_single_charge_surface_potential_increment_V",
    "Epstein_drag_force_r_N",
    "Epstein_drag_force_z_N",
)
POSITIVE_ACTIVE_COLUMNS = (
    "particle_radius_m",
    "particle_mass_kg",
    "local_gas_temperature_K",
    "local_absolute_pressure_Pa",
    "local_gas_density_kg_per_m3",
    "local_gas_mean_free_path_m",
    "local_electron_density_per_m3",
    "local_total_positive_ion_density_per_m3",
    "electron_temperature_eV_as_V",
    "effective_positive_ion_mass_kg",
    "local_ion_thermal_energy_eV_as_V",
)
CASE_P_COLUMN_ALIASES = {
    "electron_temperature_eV_as_V": "local_electron_temperature_eV_as_V",
    "effective_positive_ion_mass_kg": "local_effective_positive_ion_mass_kg",
}
AUDIT_VARIANTS = frozenset(
    {
        "relative_flow_screened_collection_orbital_v1",
        "electric_field_aligned_image_sensitivity_v1",
    }
)
AUDIT_PARTICLE_FEATURES = frozenset(
    {
        "auxq",
        "idf",
        "pp1",
        "relg1",
        "df1",
        "bf1",
        "ef1",
        "liftfm",
        "depf",
        "thpf1",
        "wall1",
        "outin",
        "outpump",
        "axi1",
    }
)
AUDIT_ELECTROSTATIC_FEATURES = frozenset(
    {"rhoAS", "waferAS", "wallAS", "dielectricAS", "dielectricOuterAS", "bulkAS"}
)
AUDIT_PARAMETERS = frozenset(
    {
        "AS_Te",
        "AS_ne0",
        "AS_ni0",
        "AS_mi",
        "AS_mu_i",
        "AS_Vp",
        "AS_Vwall",
        "AS_Vwafer",
        "AS_Vdielectric",
        "AS_particle_dt",
        "AS_Z0",
        "particle_dt",
        "timestep",
        "AS_dV_smooth",
        "AS_exp_min",
        "AS_exp_max",
        "AS_flux_speed_floor",
        "AS_focus_r0",
        "AS_focus_transition",
        "AS_n_floor",
        "AS_sheath_ramp",
        "AS_u_floor",
        "AS_float_drop",
        "u_eps",
        "Ti_floor",
        "sigmaR_p",
        "Mmix",
        "brownian_seed",
        "AS_brownian_seed",
        "iondrag_scale_P",
        "iondrag_scale_A",
        "AS_iondrag_scale",
        "C_lift_fm",
        "sigma_in",
        "rho_p",
        "epsr_p",
        "d0",
        "E2floor",
    }
)
AUDIT_VARIABLES = frozenset(
    {
        "AS_ugr",
        "AS_ugz",
        "AS_Tg",
        "AS_pabs",
        "AS_psi_raw",
        "AS_sheath_gate",
        "AS_psi",
        "AS_uB",
        "AS_ne",
        "AS_ui_mag",
        "AS_ni",
        "AS_rhoq",
        "AS_Er",
        "AS_Ez",
        "AS_TiV",
        "AS_Di",
        "AS_Gir",
        "AS_Giz",
        "AS_Gimag",
        "AS_uir",
        "AS_uiz",
        "AS_Ge0",
        "AS_lambdaD",
        "AS_phi1",
        "pabs_d",
        "rho_g_d",
        "mu_g_d",
        "Er_P",
        "Ez_P",
        "ne_d",
        "ni_d",
        "nm_d",
        "Te_d",
        "mi_d",
        "uir_d",
        "uiz_d",
        "lambda_g_d",
        "lambdaD_d",
        "lambda_in_d",
        "TiV_d",
        "Ge0_d",
        "phi1_d",
    }
)


@dataclass(frozen=True, slots=True)
class M3VConfig:
    """Small typed view of the external evaluation configuration."""

    source: Path
    raw: dict[str, Any]
    expected_particles: int
    expected_times: int
    expected_rows: int
    expected_time_start_s: float
    expected_time_end_s: float
    internal_solver_step_s: float
    field_sources: dict[str, dict[str, str]]
    ion_drag_models: dict[str, str]
    diameters_nm: tuple[int, ...]
    force_columns: dict[str, str]
    force_components: dict[str, tuple[str, str]]
    oml_radius_limit: float
    oml_drift_limit: float
    epstein_lambda_limit: float
    epstein_speed_limit: float
    epstein_diffuse_fraction: float
    epstein_force_residual_limit: float
    charge_speed_regularization_m_s: float
    charge_minimum_ion_energy_V: float
    charge_exponent_min: float
    charge_exponent_max: float
    charge_rate_residual_limit: float
    neutral_molecular_mass_kg: float
    position_difference_threshold_m: float


@dataclass(frozen=True, slots=True)
class CaseDescriptor:
    """One point in the field-source, ion-drag, and size matrix."""

    variant_directory: str
    ion_drag_model: str
    legacy_case: str
    field_source_mode: str
    diameter_nm: int
    package: Path

    @property
    def case_id(self) -> str:
        return f"{self.variant_directory}/{self.legacy_case}_{self.diameter_nm}nm"


@dataclass(frozen=True, slots=True)
class Trace:
    """Minimal stored history needed for time-resolved model-form sensitivity."""

    particle_id: IntArray
    time_s: FloatArray
    position_m: FloatArray
    velocity_m_s: FloatArray
    charge_number: FloatArray
    status_code: IntArray
    active: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class CaseEvaluation:
    """Metrics and trace from one exported package."""

    descriptor: CaseDescriptor
    metrics: Record
    force_rows: tuple[Record, ...]
    trace: Trace


@dataclass(frozen=True, slots=True)
class ForceEvidence:
    """Per-particle deterministic impulse and sampled Brownian evidence."""

    absolute_impulse_N_s: dict[str, FloatArray]
    net_impulse_r_N_s: dict[str, FloatArray]
    net_impulse_z_N_s: dict[str, FloatArray]
    covered_duration_s: FloatArray
    unresolved_terminal_gap_s: FloatArray
    particle_mass_kg: FloatArray
    brownian_rz_rms_N: FloatArray


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return {str(key): item for key, item in value.items()}


def load_evaluation_config(path: Path) -> M3VConfig:
    raw_value = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw = _mapping(raw_value, "configuration")
    dataset = _mapping(raw["dataset"], "dataset")
    physics = _mapping(raw["physics_reference"], "physics_reference")
    oml = _mapping(physics["oml_stationary_maxwellian_debye_huckel_v1"], "OML")
    drift_charge = _mapping(
        physics["relative_drift_regularized_two_current_v1"], "relative-drift charge"
    )
    epstein = _mapping(physics["epstein_linear_v1"], "Epstein")
    mixture = _mapping(physics["neutral_mixture"], "neutral_mixture")
    field_values = _mapping(raw["field_sources"], "field_sources")
    field_sources = {
        key: {str(k): str(v) for k, v in _mapping(value, key).items()}
        for key, value in field_values.items()
    }
    ion_drag = {
        str(key): str(value)
        for key, value in _mapping(raw["ion_drag_models"], "ion_drag_models").items()
    }
    forces = {
        str(key): str(value)
        for key, value in _mapping(raw["force_columns"], "force_columns").items()
    }
    force_components = {
        str(key): (str(value[0]), str(value[1]))
        for key, value in _mapping(raw["force_components"], "force_components").items()
    }
    if set(forces) != set(force_components):
        raise ValueError("force_columns and force_components must have identical names")
    sensitivity = _mapping(raw["sensitivity"], "sensitivity")
    molar_mass = float(mixture["cf4_mole_fraction"]) * float(
        mixture["cf4_molar_mass_kg_per_mol"]
    ) + float(mixture["o2_mole_fraction"]) * float(mixture["o2_molar_mass_kg_per_mol"])
    return M3VConfig(
        source=path,
        raw=raw,
        expected_particles=int(dataset["expected_particles"]),
        expected_times=int(dataset["expected_times"]),
        expected_rows=int(dataset["expected_rows"]),
        expected_time_start_s=float(dataset["expected_time_start_s"]),
        expected_time_end_s=float(dataset["expected_time_end_s"]),
        internal_solver_step_s=float(dataset["internal_solver_step_s"]),
        field_sources=field_sources,
        ion_drag_models=ion_drag,
        diameters_nm=tuple(int(value) for value in raw["particle_diameters_nm"]),
        force_columns=forces,
        force_components=force_components,
        oml_radius_limit=float(oml["maximum_radius_over_debye"]),
        oml_drift_limit=float(oml["maximum_ion_drift_over_mean_thermal_speed"]),
        epstein_lambda_limit=float(epstein["minimum_mean_free_path_over_radius"]),
        epstein_speed_limit=float(epstein["maximum_relative_speed_over_mean_thermal_speed"]),
        epstein_diffuse_fraction=float(epstein["diffuse_reflection_fraction"]),
        epstein_force_residual_limit=float(epstein["force_vector_relative_residual_limit"]),
        charge_speed_regularization_m_s=float(
            drift_charge["relative_speed_regularization_m_per_s"]
        ),
        charge_minimum_ion_energy_V=float(drift_charge["minimum_ion_energy_eV_as_V"]),
        charge_exponent_min=float(drift_charge["exponential_argument_min"]),
        charge_exponent_max=float(drift_charge["exponential_argument_max"]),
        charge_rate_residual_limit=float(drift_charge["frozen_rate_current_scale_residual_limit"]),
        neutral_molecular_mass_kg=molar_mass / AVOGADRO_PER_MOL,
        position_difference_threshold_m=float(sensitivity["position_difference_threshold_m"]),
    )


def _descriptors(dataset_root: Path, config: M3VConfig) -> tuple[CaseDescriptor, ...]:
    result: list[CaseDescriptor] = []
    for variant_directory, ion_drag_model in config.ion_drag_models.items():
        for legacy_case, field_source in config.field_sources.items():
            for diameter_nm in config.diameters_nm:
                package = (
                    dataset_root
                    / "cases"
                    / variant_directory
                    / f"{legacy_case}_{diameter_nm}nm"
                    / "external_reproduction"
                )
                result.append(
                    CaseDescriptor(
                        variant_directory,
                        ion_drag_model,
                        legacy_case,
                        field_source["mode"],
                        diameter_nm,
                        package,
                    )
                )
    return tuple(result)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_float(row: dict[str, str], column: str) -> float:
    result = float(row[column])
    if not math.isfinite(result):
        raise ValueError(f"{column} is not finite")
    return result


def _float_value(row: dict[str, str], column: str) -> float:
    return float(row[column])


def _active_source_column(descriptor: CaseDescriptor, canonical_name: str) -> str:
    if descriptor.legacy_case == "caseP":
        return CASE_P_COLUMN_ALIASES.get(canonical_name, canonical_name)
    return canonical_name


def _package_metadata_pass(descriptor: CaseDescriptor) -> bool:
    validation = descriptor.package / "validation" / "package_validation.csv"
    manifest = descriptor.package / "manifest.csv"
    if not validation.is_file() or not manifest.is_file():
        return False
    with validation.open(encoding="utf-8-sig", newline="") as stream:
        if any(row.get("status") != "PASS" for row in csv.DictReader(stream)):
            return False
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        values = {row["key"]: row["value"] for row in csv.DictReader(stream)}
    return (
        values.get("model_variant") == descriptor.variant_directory
        and values.get("case") == descriptor.legacy_case.removeprefix("case")
        and values.get("particle_diameter_nm") == str(descriptor.diameter_nm)
        and values.get("coordinate_system")
        == "2D axisymmetric r-z; all normalized CSV coordinates are SI metres"
        and values.get("solver_or_study_run_by_this_export") == "false"
        and values.get("background_recomputed") == "false"
        and values.get("particle_recomputed") == "false"
    )


def _append_particle_force_evidence(
    *,
    absolute_storage: dict[str, list[float]],
    net_r_storage: dict[str, list[float]],
    net_z_storage: dict[str, list[float]],
    current_absolute: dict[str, float],
    current_net_r: dict[str, float],
    current_net_z: dict[str, float],
    deterministic_forces: tuple[str, ...],
) -> None:
    for name in deterministic_forces:
        absolute_storage[name].append(current_absolute[name])
        net_r_storage[name].append(current_net_r[name])
        net_z_storage[name].append(current_net_z[name])


class _HistoryAccumulator:
    """Single-pass reader state with particle-local force integration."""

    def __init__(self, descriptor: CaseDescriptor, config: M3VConfig) -> None:
        self.descriptor = descriptor
        self.config = config
        self.force_names = tuple(config.force_columns)
        self.deterministic_forces = tuple(
            name for name in self.force_names if name != "brownian_sample"
        )
        self.absolute_storage = {name: [] for name in self.deterministic_forces}
        self.net_r_storage = {name: [] for name in self.deterministic_forces}
        self.net_z_storage = {name: [] for name in self.deterministic_forces}
        self.covered_duration: list[float] = []
        self.unresolved_terminal_gaps: list[float] = []
        self.particle_masses: list[float] = []
        self.brownian_rms: list[float] = []
        self.trace_values: dict[str, list[float]] = {
            name: []
            for name in (
                "particle_id",
                "time_s",
                "r_m",
                "z_m",
                "velocity_r",
                "velocity_z",
                "charge",
                "status",
                "active",
            )
        }
        self.active_values: dict[str, list[float]] = {
            name: [] for name in (*ACTIVE_COLUMNS, "negative_ion_density_m3")
        }
        self.particle_ids: set[int] = set()
        self.output_times: set[float] = set()
        self.row_count = 0
        self.invalid_active_rows = 0
        self.monotonic = True
        self.current_particle: int | None = None
        self.previous_time = 0.0
        self.previous_active = False
        self.previous_magnitudes = dict.fromkeys(self.force_names, 0.0)
        self.previous_components = dict.fromkeys(self.force_names, (0.0, 0.0))
        self._reset_particle_values()

    def required_columns(self) -> set[str]:
        result = {
            "particle_id",
            "time_s",
            "r_m",
            "z_m",
            "charge_number_e",
            "current_status_code",
            "active_state_flag",
            "stop_or_event_time_s",
            *(_active_source_column(self.descriptor, name) for name in ACTIVE_COLUMNS),
            *self.config.force_columns.values(),
            *(column for pair in self.config.force_components.values() for column in pair),
        }
        if self.descriptor.legacy_case == "caseP":
            result.add("local_total_negative_ion_density_per_m3")
        return result

    def _reset_particle_values(self) -> None:
        self.current_absolute = dict.fromkeys(self.deterministic_forces, 0.0)
        self.current_net_r = dict.fromkeys(self.deterministic_forces, 0.0)
        self.current_net_z = dict.fromkeys(self.deterministic_forces, 0.0)
        self.current_covered_duration = 0.0
        self.current_event_time = 0.0
        self.current_last_active_time = 0.0
        self.current_mass = math.nan
        self.current_brownian_sum_square = 0.0
        self.current_brownian_samples = 0

    def _start_particle(self, row: dict[str, str], particle_id: int) -> None:
        self.current_particle = particle_id
        self.current_mass = _finite_float(row, "particle_mass_kg")
        self.current_event_time = _finite_float(row, "stop_or_event_time_s")

    def _finish_particle(self) -> None:
        if self.current_particle is None:
            return
        _append_particle_force_evidence(
            absolute_storage=self.absolute_storage,
            net_r_storage=self.net_r_storage,
            net_z_storage=self.net_z_storage,
            current_absolute=self.current_absolute,
            current_net_r=self.current_net_r,
            current_net_z=self.current_net_z,
            deterministic_forces=self.deterministic_forces,
        )
        self.covered_duration.append(self.current_covered_duration)
        terminal_gap = (
            max(0.0, self.current_event_time - self.current_last_active_time)
            if self.current_event_time > 0.0
            else 0.0
        )
        self.unresolved_terminal_gaps.append(terminal_gap)
        self.particle_masses.append(self.current_mass)
        rms = (
            math.sqrt(self.current_brownian_sum_square / self.current_brownian_samples)
            if self.current_brownian_samples
            else math.nan
        )
        self.brownian_rms.append(rms)
        self._reset_particle_values()

    def _row_forces(
        self, row: dict[str, str], active: bool
    ) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
        number = _finite_float if active else _float_value
        for column in self.config.force_columns.values():
            number(row, column)
        components = {
            name: (number(row, pair[0]), number(row, pair[1]))
            for name, pair in self.config.force_components.items()
        }
        magnitudes = {
            name: math.hypot(component[0], component[1]) for name, component in components.items()
        }
        return magnitudes, components

    def _advance_interval(
        self,
        time_s: float,
        active: bool,
        magnitudes: dict[str, float],
        components: dict[str, tuple[float, float]],
    ) -> None:
        delta_time = time_s - self.previous_time
        if delta_time <= 0.0:
            self.monotonic = False
            return
        if not (self.previous_active and active):
            return
        self.current_covered_duration += delta_time
        for name in self.deterministic_forces:
            self.current_absolute[name] += (
                0.5 * (self.previous_magnitudes[name] + magnitudes[name]) * delta_time
            )
            self.current_net_r[name] += (
                0.5 * (self.previous_components[name][0] + components[name][0]) * delta_time
            )
            self.current_net_z[name] += (
                0.5 * (self.previous_components[name][1] + components[name][1]) * delta_time
            )

    def _record_trace(
        self,
        row: dict[str, str],
        particle_id: int,
        time_s: float,
        active: bool,
    ) -> None:
        number = _finite_float if active else _float_value
        self.trace_values["particle_id"].append(float(particle_id))
        self.trace_values["time_s"].append(time_s)
        self.trace_values["r_m"].append(number(row, "r_m"))
        self.trace_values["z_m"].append(number(row, "z_m"))
        self.trace_values["velocity_r"].append(number(row, "velocity_r_m_per_s"))
        self.trace_values["velocity_z"].append(number(row, "velocity_z_m_per_s"))
        self.trace_values["charge"].append(number(row, "charge_number_e"))
        self.trace_values["status"].append(_finite_float(row, "current_status_code"))
        self.trace_values["active"].append(float(active))

    def _record_active_primitives(self, row: dict[str, str], active: bool) -> None:
        if not active:
            return
        try:
            values = {
                name: _finite_float(row, _active_source_column(self.descriptor, name))
                for name in ACTIVE_COLUMNS
            }
            values["negative_ion_density_m3"] = (
                _finite_float(row, "local_total_negative_ion_density_per_m3")
                if self.descriptor.legacy_case == "caseP"
                else 0.0
            )
            if any(values[name] <= 0.0 for name in POSITIVE_ACTIVE_COLUMNS):
                raise ValueError("positive active primitive is nonpositive")
        except (KeyError, ValueError):
            self.invalid_active_rows += 1
            return
        for name, value in values.items():
            self.active_values[name].append(value)

    def consume(self, row: dict[str, str]) -> None:
        self.row_count += 1
        particle_id = int(float(row["particle_id"]))
        time_s = _finite_float(row, "time_s")
        active = int(float(row["active_state_flag"])) == 1
        magnitudes, components = self._row_forces(row, active)
        if self.current_particle is None:
            self._start_particle(row, particle_id)
        elif particle_id != self.current_particle:
            self._finish_particle()
            self._start_particle(row, particle_id)
        else:
            self._advance_interval(time_s, active, magnitudes, components)

        self.previous_time = time_s
        self.previous_active = active
        self.previous_magnitudes = magnitudes
        self.previous_components = components
        if active:
            self.current_last_active_time = time_s
            brownian_r, brownian_z = components["brownian_sample"]
            self.current_brownian_sum_square += brownian_r**2 + brownian_z**2
            self.current_brownian_samples += 1
        self.particle_ids.add(particle_id)
        self.output_times.add(time_s)
        self._record_trace(row, particle_id, time_s, active)
        self._record_active_primitives(row, active)

    def finish(self) -> None:
        self._finish_particle()

    def build(
        self,
    ) -> tuple[Trace, dict[str, FloatArray], ForceEvidence, Record]:
        trace = Trace(
            particle_id=np.asarray(self.trace_values["particle_id"], dtype=np.int64),
            time_s=np.asarray(self.trace_values["time_s"], dtype=np.float64),
            position_m=np.column_stack((self.trace_values["r_m"], self.trace_values["z_m"])),
            velocity_m_s=np.column_stack(
                (self.trace_values["velocity_r"], self.trace_values["velocity_z"])
            ),
            charge_number=np.asarray(self.trace_values["charge"], dtype=np.float64),
            status_code=np.asarray(self.trace_values["status"], dtype=np.int64),
            active=np.asarray(self.trace_values["active"], dtype=np.bool_),
        )
        active_arrays = {
            name: np.asarray(values, dtype=np.float64)
            for name, values in self.active_values.items()
        }
        force_evidence = ForceEvidence(
            absolute_impulse_N_s={
                name: np.asarray(values, dtype=np.float64)
                for name, values in self.absolute_storage.items()
            },
            net_impulse_r_N_s={
                name: np.asarray(values, dtype=np.float64)
                for name, values in self.net_r_storage.items()
            },
            net_impulse_z_N_s={
                name: np.asarray(values, dtype=np.float64)
                for name, values in self.net_z_storage.items()
            },
            covered_duration_s=np.asarray(self.covered_duration, dtype=np.float64),
            unresolved_terminal_gap_s=np.asarray(self.unresolved_terminal_gaps, dtype=np.float64),
            particle_mass_kg=np.asarray(self.particle_masses, dtype=np.float64),
            brownian_rz_rms_N=np.asarray(self.brownian_rms, dtype=np.float64),
        )
        sorted_output_times = np.asarray(sorted(self.output_times), dtype="<f8")
        interval_values = np.unique(np.round(np.diff(sorted_output_times), decimals=12))
        structure: Record = {
            "rows": self.row_count,
            "particles": len(self.particle_ids),
            "stored_times": len(self.output_times),
            "time_start_s": (float(sorted_output_times[0]) if sorted_output_times.size else None),
            "time_end_s": (float(sorted_output_times[-1]) if sorted_output_times.size else None),
            "active_rows": int(np.count_nonzero(trace.active)),
            "valid_active_rows": len(next(iter(self.active_values.values()))),
            "invalid_active_rows": self.invalid_active_rows,
            "particle_time_monotonic": self.monotonic,
            "output_time_grid_sha256": hashlib.sha256(sorted_output_times.tobytes()).hexdigest(),
            "output_time_interval_levels_s": ";".join(f"{value:.12g}" for value in interval_values),
        }
        structure.update(_history_grid_metrics(trace))
        return trace, active_arrays, force_evidence, structure


def _read_history(
    descriptor: CaseDescriptor, config: M3VConfig
) -> tuple[Trace, dict[str, FloatArray], ForceEvidence, Record]:
    history = descriptor.package / "results" / "particle_history_full_tidy.csv"
    if not history.is_file():
        raise FileNotFoundError(history)
    state = _HistoryAccumulator(descriptor, config)
    with history.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = state.required_columns().difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{history} is missing columns: {sorted(missing)}")
        for row in reader:
            state.consume(row)
    state.finish()
    return state.build()


def _sampled_gate_summary(
    values: dict[str, FloatArray], applicable: NDArray[np.bool_], prefix: str
) -> Record:
    particle_ids = values["particle_id"].astype(np.int64)
    all_pass = [
        bool(np.all(applicable[particle_ids == particle_id]))
        for particle_id in np.unique(particle_ids)
    ]
    failed_times = values["time_s"][~applicable]
    return {
        f"{prefix}_particle_all_active_samples_applicable_fraction": float(np.mean(all_pass)),
        f"{prefix}_first_sampled_violation_time_s": (
            None if failed_times.size == 0 else float(np.min(failed_times))
        ),
    }


def replay_relative_drift_charge(
    values: dict[str, FloatArray], config: M3VConfig
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Replay the saved model's frozen-state mean collection law."""

    radius = values["particle_radius_m"]
    charge = values["charge_number_e"]
    phi1 = values["local_single_charge_surface_potential_increment_V"]
    ion_mass = values["effective_positive_ion_mass_kg"]
    ion_energy = values["local_ion_thermal_energy_eV_as_V"]
    electron_energy = values["electron_temperature_eV_as_V"]
    relative_speed_squared = (
        (values["local_ion_velocity_r_m_per_s"] - values["velocity_r_m_per_s"]) ** 2
        + (values["local_ion_velocity_z_m_per_s"] - values["velocity_z_m_per_s"]) ** 2
        + 8.0 * ELEMENTARY_CHARGE_C * ion_energy / (math.pi * ion_mass)
        + config.charge_speed_regularization_m_s**2
    )
    ion_speed = np.sqrt(relative_speed_squared)
    effective_ion_energy = np.maximum(
        ion_mass * relative_speed_squared / (2.0 * ELEMENTARY_CHARGE_C),
        config.charge_minimum_ion_energy_V,
    )
    ion_base = math.pi * radius**2 * values["local_total_positive_ion_density_per_m3"] * ion_speed
    electron_base = (
        math.pi
        * radius**2
        * values["local_electron_density_per_m3"]
        * np.sqrt(8.0 * ELEMENTARY_CHARGE_C * electron_energy / (math.pi * ELECTRON_MASS_KG))
    )
    particle_potential = charge * phi1
    nonpositive = particle_potential <= 0.0
    electron_argument = particle_potential / electron_energy
    ion_argument = -particle_potential / effective_ion_energy
    electron_factor = np.where(
        nonpositive,
        np.exp(
            np.clip(
                electron_argument,
                config.charge_exponent_min,
                config.charge_exponent_max,
            )
        ),
        1.0 + electron_argument,
    )
    ion_factor = np.where(
        nonpositive,
        1.0 + ion_argument,
        np.exp(np.clip(ion_argument, config.charge_exponent_min, config.charge_exponent_max)),
    )
    rate = ion_base * ion_factor - electron_base * electron_factor
    current_scale = ion_base * np.abs(ion_factor) + electron_base * np.abs(electron_factor)

    electron_unclipped = (electron_argument > config.charge_exponent_min) & (
        electron_argument < config.charge_exponent_max
    )
    ion_unclipped = (ion_argument > config.charge_exponent_min) & (
        ion_argument < config.charge_exponent_max
    )
    electron_derivative = np.where(
        nonpositive,
        np.where(electron_unclipped, electron_factor * phi1 / electron_energy, 0.0),
        phi1 / electron_energy,
    )
    ion_derivative = np.where(
        nonpositive,
        -phi1 / effective_ion_energy,
        np.where(ion_unclipped, -ion_factor * phi1 / effective_ion_energy, 0.0),
    )
    derivative = ion_base * ion_derivative - electron_base * electron_derivative
    return rate, current_scale, derivative


def charge_formula_metrics(values: dict[str, FloatArray], config: M3VConfig) -> Record:
    reconstructed, current_scale, derivative = replay_relative_drift_charge(values, config)
    exported = values["dynamic_charge_rate_dZdt_per_s"]
    residual = reconstructed - exported
    normalized = np.abs(residual) / np.maximum(current_scale, 1.0)
    global_relative_l2 = float(
        np.linalg.norm(residual) / max(float(np.linalg.norm(exported)), np.finfo(float).tiny)
    )
    finite = np.isfinite(reconstructed) & np.isfinite(derivative) & np.isfinite(normalized)
    parity_pass = (
        bool(finite.all()) and float(np.max(normalized)) <= config.charge_rate_residual_limit
    )
    local_stiffness = config.internal_solver_step_s * np.abs(derivative)
    potential = (
        values["charge_number_e"] * values["local_single_charge_surface_potential_increment_V"]
    )
    relative_speed_squared = (
        (values["local_ion_velocity_r_m_per_s"] - values["velocity_r_m_per_s"]) ** 2
        + (values["local_ion_velocity_z_m_per_s"] - values["velocity_z_m_per_s"]) ** 2
        + 8.0
        * ELEMENTARY_CHARGE_C
        * values["local_ion_thermal_energy_eV_as_V"]
        / (math.pi * values["effective_positive_ion_mass_kg"])
        + config.charge_speed_regularization_m_s**2
    )
    raw_ion_energy = (
        values["effective_positive_ion_mass_kg"]
        * relative_speed_squared
        / (2.0 * ELEMENTARY_CHARGE_C)
    )
    effective_ion_energy = np.maximum(raw_ion_energy, config.charge_minimum_ion_energy_V)
    electron_argument = potential / values["electron_temperature_eV_as_V"]
    ion_argument = -potential / effective_ion_energy
    electron_clipped = (electron_argument < config.charge_exponent_min) | (
        electron_argument > config.charge_exponent_max
    )
    ion_clipped = (ion_argument < config.charge_exponent_min) | (
        ion_argument > config.charge_exponent_max
    )
    return {
        "dataset_drift_aware_charge_rate_formula_parity": "PASS" if parity_pass else "FAIL",
        "dataset_charge_rate_global_relative_l2_residual": global_relative_l2,
        "dataset_charge_rate_current_scale_residual_p90": float(np.percentile(normalized, 90.0)),
        "dataset_charge_rate_current_scale_residual_p99": float(np.percentile(normalized, 99.0)),
        "dataset_charge_rate_current_scale_residual_max": float(np.max(normalized)),
        "dataset_charge_rate_formula_finite_fraction": float(np.mean(finite)),
        "dataset_drift_aware_local_h_abs_dR_dZ_p90": float(np.percentile(local_stiffness, 90.0)),
        "dataset_drift_aware_local_h_abs_dR_dZ_max": float(np.max(local_stiffness)),
        "dataset_drift_aware_local_h_abs_dR_dZ_over_half_fraction": float(
            np.mean(local_stiffness > 0.5)
        ),
        "dataset_charge_positive_potential_branch_fraction": float(np.mean(potential > 0.0)),
        "dataset_charge_ion_energy_floor_branch_fraction": float(
            np.mean(raw_ion_energy < config.charge_minimum_ion_energy_V)
        ),
        "dataset_charge_electron_exponent_clamp_branch_fraction": float(
            np.mean(electron_clipped & (potential <= 0.0))
        ),
        "dataset_charge_ion_exponent_clamp_branch_fraction": float(
            np.mean(ion_clipped & (potential > 0.0))
        ),
        "dataset_drift_aware_charge_stiffness": (
            "SAMPLED_LOCAL_CHARACTERIZATION_NOT_CONTINUOUS_PATH_CERTIFICATE"
        ),
    }


def _oml_metrics(values: dict[str, FloatArray], config: M3VConfig) -> Record:
    radius = values["particle_radius_m"]
    charge = values["charge_number_e"]
    electron_temperature_K = (
        values["electron_temperature_eV_as_V"] * ELEMENTARY_CHARGE_C / BOLTZMANN_J_K
    )
    ion_temperature_K = (
        values["local_ion_thermal_energy_eV_as_V"] * ELEMENTARY_CHARGE_C / BOLTZMANN_J_K
    )
    ion_mass = values["effective_positive_ion_mass_kg"]
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        inverse_debye_squared = (
            ELEMENTARY_CHARGE_C**2
            / (VACUUM_PERMITTIVITY_F_M * BOLTZMANN_J_K)
            * (
                values["local_electron_density_per_m3"] / electron_temperature_K
                + values["local_total_positive_ion_density_per_m3"] / ion_temperature_K
            )
        )
        debye = 1.0 / np.sqrt(inverse_debye_squared)
        radius_ratio = radius / debye
        ion_relative_speed = np.hypot(
            values["local_ion_velocity_r_m_per_s"] - values["velocity_r_m_per_s"],
            values["local_ion_velocity_z_m_per_s"] - values["velocity_z_m_per_s"],
        )
        ion_thermal_speed = np.sqrt(8.0 * BOLTZMANN_J_K * ion_temperature_K / (math.pi * ion_mass))
        drift_ratio = ion_relative_speed / ion_thermal_speed
    finite = np.isfinite(radius_ratio) & np.isfinite(drift_ratio)
    radius_drift_applicable = (
        (radius_ratio <= config.oml_radius_limit) & (drift_ratio <= config.oml_drift_limit) & finite
    )
    ion_mass_relative_span = float((np.max(ion_mass) - np.min(ion_mass)) / np.median(ion_mass))
    scalar_ion_mass = ion_mass_relative_span <= 1.0e-12
    negative_ion_ratio = values["negative_ion_density_m3"] / values["local_electron_density_per_m3"]
    applicable = radius_drift_applicable & (negative_ion_ratio == 0.0) & scalar_ion_mass
    absolute_exported_rate = np.abs(values["dynamic_charge_rate_dZdt_per_s"])
    fractional_charge_activity = (
        config.internal_solver_step_s * absolute_exported_rate / np.maximum(1.0, np.abs(charge))
    )
    particle_ids = values["particle_id"].astype(np.int64)
    maximum_excursions = np.asarray(
        [
            np.max(
                np.abs(charge[particle_ids == particle_id] - charge[particle_ids == particle_id][0])
            )
            for particle_id in np.unique(particle_ids)
        ],
        dtype=np.float64,
    )
    return {
        "charge_sample_rows": int(radius.size),
        "charge_applicability_primitives_finite_fraction": float(np.mean(finite)),
        "charge_debye_gate_fraction": float(np.mean(radius_ratio <= config.oml_radius_limit)),
        "charge_stationary_ion_gate_fraction": float(
            np.mean(drift_ratio <= config.oml_drift_limit)
        ),
        "charge_sampled_radius_drift_gate_fraction": float(np.mean(radius_drift_applicable)),
        "charge_sampled_applicable_fraction": float(np.mean(applicable)),
        "charge_radius_over_debye_p90": float(np.percentile(radius_ratio, 90.0)),
        "charge_ion_drift_ratio_p50": float(np.percentile(drift_ratio, 50.0)),
        "charge_ion_drift_ratio_p90": float(np.percentile(drift_ratio, 90.0)),
        "charge_ion_drift_ratio_max": float(np.max(drift_ratio)),
        "charge_positive_ion_mass_relative_span": ion_mass_relative_span,
        "charge_scalar_ion_mass_gate_fraction": 1.0 if scalar_ion_mass else 0.0,
        "charge_scalar_ion_mass_contract": "MATCH" if scalar_ion_mass else "MISMATCH",
        "charge_negative_ion_to_electron_ratio_p50": float(np.percentile(negative_ion_ratio, 50.0)),
        "charge_negative_ion_to_electron_ratio_p90": float(np.percentile(negative_ion_ratio, 90.0)),
        "charge_negative_ion_to_electron_ratio_max": float(np.max(negative_ion_ratio)),
        "charge_two_species_assumption": (
            "MATCH" if bool((negative_ion_ratio == 0.0).all()) else "NOT_APPLICABLE"
        ),
        "dataset_charge_rate_abs_p50_number_per_s": float(
            np.percentile(absolute_exported_rate, 50.0)
        ),
        "dataset_charge_rate_abs_p90_number_per_s": float(
            np.percentile(absolute_exported_rate, 90.0)
        ),
        "dataset_charge_rate_abs_max_number_per_s": float(np.max(absolute_exported_rate)),
        "dataset_internal_step_fractional_charge_activity_p90": float(
            np.percentile(fractional_charge_activity, 90.0)
        ),
        "dataset_internal_step_fractional_charge_activity_max": float(
            np.max(fractional_charge_activity)
        ),
        "dataset_particle_max_charge_excursion_p50": float(np.percentile(maximum_excursions, 50.0)),
        "dataset_particle_max_charge_excursion_p90": float(np.percentile(maximum_excursions, 90.0)),
        **charge_formula_metrics(values, config),
        **_sampled_gate_summary(values, applicable, "charge"),
    }


def epstein_diffuse_factor(diffuse_reflection_fraction: float) -> float:
    return 1.0 + diffuse_reflection_fraction * math.pi / 8.0


def _epstein_metrics(values: dict[str, FloatArray], config: M3VConfig) -> Record:
    radius = values["particle_radius_m"]
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        mean_thermal_speed = np.sqrt(
            8.0
            * BOLTZMANN_J_K
            * values["local_gas_temperature_K"]
            / (math.pi * config.neutral_molecular_mass_kg)
        )
        relative_speed = np.hypot(
            values["local_gas_velocity_r_m_per_s"] - values["velocity_r_m_per_s"],
            values["local_gas_velocity_z_m_per_s"] - values["velocity_z_m_per_s"],
        )
        lambda_ratio = values["local_gas_mean_free_path_m"] / radius
        speed_ratio = relative_speed / mean_thermal_speed
        inferred_molecular_mass = (
            values["local_gas_density_kg_per_m3"]
            * BOLTZMANN_J_K
            * values["local_gas_temperature_K"]
            / values["local_absolute_pressure_Pa"]
        )
        delta = epstein_diffuse_factor(config.epstein_diffuse_fraction)
        friction = (
            4.0
            * math.pi
            / 3.0
            * radius**2
            * values["local_gas_density_kg_per_m3"]
            * mean_thermal_speed
            * delta
        )
        reconstructed_force = np.column_stack(
            (
                friction * (values["local_gas_velocity_r_m_per_s"] - values["velocity_r_m_per_s"]),
                friction * (values["local_gas_velocity_z_m_per_s"] - values["velocity_z_m_per_s"]),
            )
        )
        exported_force = np.column_stack(
            (values["Epstein_drag_force_r_N"], values["Epstein_drag_force_z_N"])
        )
    finite = np.isfinite(lambda_ratio) & np.isfinite(speed_ratio)
    applicable = (
        (lambda_ratio >= config.epstein_lambda_limit)
        & (speed_ratio <= config.epstein_speed_limit)
        & finite
    )
    relative_mass_error = np.abs(inferred_molecular_mass / config.neutral_molecular_mass_kg - 1.0)
    force_residual = np.linalg.norm(reconstructed_force - exported_force, axis=1)
    force_scale = np.maximum(
        np.maximum(
            np.linalg.norm(reconstructed_force, axis=1),
            np.linalg.norm(exported_force, axis=1),
        ),
        np.finfo(float).tiny,
    )
    relative_force_residual = force_residual / force_scale
    force_parity_pass = (
        bool(np.isfinite(relative_force_residual).all())
        and float(np.max(relative_force_residual)) <= config.epstein_force_residual_limit
    )
    return {
        "epstein_sample_rows": int(radius.size),
        "epstein_lambda_gate_fraction": float(np.mean(lambda_ratio >= config.epstein_lambda_limit)),
        "epstein_low_speed_gate_fraction": float(
            np.mean(speed_ratio <= config.epstein_speed_limit)
        ),
        "epstein_sampled_applicable_fraction": float(np.mean(applicable)),
        "epstein_lambda_over_radius_min": float(np.min(lambda_ratio)),
        "epstein_speed_ratio_p50": float(np.percentile(speed_ratio, 50.0)),
        "epstein_speed_ratio_p90": float(np.percentile(speed_ratio, 90.0)),
        "epstein_speed_ratio_max": float(np.max(speed_ratio)),
        "epstein_inferred_neutral_mass_relative_error_p90": float(
            np.percentile(relative_mass_error, 90.0)
        ),
        "epstein_inferred_neutral_mass_relative_error_max": float(np.max(relative_mass_error)),
        "epstein_diffuse_reflection_fraction": config.epstein_diffuse_fraction,
        "epstein_delta": epstein_diffuse_factor(config.epstein_diffuse_fraction),
        "epstein_force_formula_parity": "PASS" if force_parity_pass else "FAIL",
        "epstein_force_vector_relative_residual_p90": float(
            np.percentile(relative_force_residual, 90.0)
        ),
        "epstein_force_vector_relative_residual_p99": float(
            np.percentile(relative_force_residual, 99.0)
        ),
        "epstein_force_vector_relative_residual_max": float(np.max(relative_force_residual)),
        **_sampled_gate_summary(values, applicable, "epstein"),
    }


def _force_metrics(
    descriptor: CaseDescriptor,
    evidence: ForceEvidence,
) -> tuple[Record, ...]:
    deterministic_total = np.sum(np.vstack(list(evidence.absolute_impulse_N_s.values())), axis=0)
    rows: list[Record] = []
    for name, values in evidence.absolute_impulse_N_s.items():
        net_impulse = np.hypot(evidence.net_impulse_r_N_s[name], evidence.net_impulse_z_N_s[name])
        fraction = np.divide(
            values,
            deterministic_total,
            out=np.zeros_like(values),
            where=deterministic_total > 0.0,
        )
        rows.append(
            {
                "case_id": descriptor.case_id,
                "field_source_mode": descriptor.field_source_mode,
                "ion_drag_model": descriptor.ion_drag_model,
                "diameter_nm": descriptor.diameter_nm,
                "force": name,
                "particle_count": int(values.size),
                "integral_abs_force_dt_median_N_s": float(np.median(values)),
                "integral_abs_force_dt_p90_N_s": float(np.percentile(values, 90.0)),
                "integral_abs_force_dt_p99_N_s": float(np.percentile(values, 99.0)),
                "integral_abs_force_dt_max_N_s": float(np.max(values)),
                "net_rz_impulse_median_N_s": float(np.median(net_impulse)),
                "net_rz_impulse_p90_N_s": float(np.percentile(net_impulse, 90.0)),
                "net_rz_impulse_p99_N_s": float(np.percentile(net_impulse, 99.0)),
                "integral_abs_force_dt_over_mass_p90_m_per_s": float(
                    np.percentile(values / evidence.particle_mass_kg, 90.0)
                ),
                "net_rz_impulse_over_mass_p90_m_per_s": float(
                    np.percentile(net_impulse / evidence.particle_mass_kg, 90.0)
                ),
                "median_fraction_of_deterministic_abs_force_time": float(np.median(fraction)),
                "active_active_covered_duration_median_s": float(
                    np.median(evidence.covered_duration_s)
                ),
                "unresolved_terminal_gap_p90_s": float(
                    np.percentile(evidence.unresolved_terminal_gap_s, 90.0)
                ),
                "interpretation": (
                    "deterministic_rz_active_active_saved_output_grid_trapezoid;"
                    "quadrature_resolution_unverified"
                ),
            }
        )
    brownian = evidence.brownian_rz_rms_N
    rows.append(
        {
            "case_id": descriptor.case_id,
            "field_source_mode": descriptor.field_source_mode,
            "ion_drag_model": descriptor.ion_drag_model,
            "diameter_nm": descriptor.diameter_nm,
            "force": "brownian_sample",
            "particle_count": int(brownian.size),
            "sampled_rz_force_rms_median_N": float(np.nanmedian(brownian)),
            "sampled_rz_force_rms_p90_N": float(np.nanpercentile(brownian, 90.0)),
            "sampled_rz_force_rms_p99_N": float(np.nanpercentile(brownian, 99.0)),
            "active_active_covered_duration_median_s": float(
                np.median(evidence.covered_duration_s)
            ),
            "unresolved_terminal_gap_p90_s": float(
                np.percentile(evidence.unresolved_terminal_gap_s, 90.0)
            ),
            "interpretation": "single_seed_timestep_dependent_noise_scale_not_impulse",
        }
    )
    return tuple(rows)


def _terminal_indices(trace: Trace) -> IntArray:
    if trace.particle_id.size == 0:
        return np.asarray([], dtype=np.int64)
    return np.concatenate(
        (
            np.flatnonzero(trace.particle_id[1:] != trace.particle_id[:-1]),
            np.asarray([trace.particle_id.size - 1], dtype=np.int64),
        )
    )


def _history_grid_metrics(trace: Trace) -> Record:
    keys = np.rec.fromarrays((trace.particle_id, trace.time_s), names=("particle", "time"))
    unique_keys = int(np.unique(keys).size)
    particle_ids, counts = np.unique(trace.particle_id, return_counts=True)
    starts = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.flatnonzero(trace.particle_id[1:] != trace.particle_id[:-1]) + 1,
        )
    )
    stops = np.concatenate((starts[1:], np.asarray([trace.particle_id.size], dtype=np.int64)))
    common_time_grid = len(starts) == len(particle_ids)
    if common_time_grid and len(starts):
        reference = trace.time_s[starts[0] : stops[0]]
        common_time_grid = all(
            np.array_equal(trace.time_s[start:stop], reference)
            for start, stop in zip(starts[1:], stops[1:], strict=True)
        )
    return {
        "unique_particle_time_keys": unique_keys,
        "duplicate_particle_time_keys": int(trace.particle_id.size - unique_keys),
        "particle_block_count": len(starts),
        "rows_per_particle_min": int(np.min(counts)),
        "rows_per_particle_max": int(np.max(counts)),
        "common_particle_time_grid": common_time_grid,
    }


def _last_active_indices(trace: Trace) -> IntArray:
    indices: list[int] = []
    starts = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.flatnonzero(trace.particle_id[1:] != trace.particle_id[:-1]) + 1,
        )
    )
    stops = np.concatenate((starts[1:], np.asarray([trace.particle_id.size], dtype=np.int64)))
    for start, stop in zip(starts, stops, strict=True):
        local = np.flatnonzero(trace.active[start:stop])
        if local.size:
            indices.append(int(start + local[-1]))
    return np.asarray(indices, dtype=np.int64)


def _structure_pass(structure: Record, config: M3VConfig, descriptor: CaseDescriptor) -> bool:
    return (
        structure["rows"] == config.expected_rows
        and structure["particles"] == config.expected_particles
        and structure["stored_times"] == config.expected_times
        and structure["time_start_s"] == config.expected_time_start_s
        and structure["time_end_s"] == config.expected_time_end_s
        and structure["invalid_active_rows"] == 0
        and structure["particle_time_monotonic"] is True
        and structure["unique_particle_time_keys"] == config.expected_rows
        and structure["duplicate_particle_time_keys"] == 0
        and structure["particle_block_count"] == config.expected_particles
        and structure["rows_per_particle_min"] == config.expected_times
        and structure["rows_per_particle_max"] == config.expected_times
        and structure["common_particle_time_grid"] is True
        and _package_metadata_pass(descriptor)
    )


def _evaluate_case(descriptor: CaseDescriptor, config: M3VConfig) -> CaseEvaluation:
    trace, active_values, force_evidence, structure = _read_history(descriptor, config)
    expected_structure = _structure_pass(structure, config, descriptor)
    terminal = _terminal_indices(trace)
    last_active = _last_active_indices(trace)
    status_values, status_counts = np.unique(trace.status_code[terminal], return_counts=True)
    status_summary = ";".join(
        f"{int(status)}:{int(count)}"
        for status, count in zip(status_values, status_counts, strict=True)
    )
    metrics: Record = {
        "case_id": descriptor.case_id,
        "legacy_case_profile": descriptor.legacy_case,
        "field_source_mode": descriptor.field_source_mode,
        "ion_drag_model": descriptor.ion_drag_model,
        "diameter_nm": descriptor.diameter_nm,
        "package_structure_status": "PASS" if expected_structure else "FAIL",
        **structure,
        **_oml_metrics(active_values, config),
        **_epstein_metrics(active_values, config),
        "charge_number_initial_median": float(
            np.median(trace.charge_number[trace.time_s == config.expected_time_start_s])
        ),
        "charge_number_last_active_saved_median": float(
            np.median(trace.charge_number[last_active])
        ),
        "charge_number_last_active_saved_p10": float(
            np.percentile(trace.charge_number[last_active], 10.0)
        ),
        "charge_number_last_active_saved_p90": float(
            np.percentile(trace.charge_number[last_active], 90.0)
        ),
        "last_active_charge_particle_count": int(last_active.size),
        "terminal_status_counts": status_summary,
    }
    return CaseEvaluation(
        descriptor,
        metrics,
        _force_metrics(descriptor, force_evidence),
        trace,
    )


def _integrated_rms_by_particle(
    trace: Trace, separation_m: FloatArray, valid: NDArray[np.bool_]
) -> FloatArray:
    values: list[float] = []
    starts = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.flatnonzero(trace.particle_id[1:] != trace.particle_id[:-1]) + 1,
        )
    )
    stops = np.concatenate((starts[1:], np.asarray([trace.particle_id.size], dtype=np.int64)))
    for start, stop in zip(starts, stops, strict=True):
        local_valid = valid[start:stop]
        time = trace.time_s[start:stop][local_valid]
        local_separation = separation_m[start:stop][local_valid]
        if time.size < 2:
            continue
        duration = float(time[-1] - time[0])
        if duration > 0.0:
            values.append(float(np.sqrt(np.trapezoid(local_separation**2, time) / duration)))
    return np.asarray(values, dtype=np.float64)


def _first_time(time_s: FloatArray, mask: NDArray[np.bool_]) -> float | None:
    selected = time_s[mask]
    return None if selected.size == 0 else float(np.min(selected))


def _outcome_confusion(first: IntArray, second: IntArray) -> str:
    pairs, counts = np.unique(np.column_stack((first, second)), axis=0, return_counts=True)
    return ";".join(
        f"{int(pair[0])}->{int(pair[1])}:{int(count)}"
        for pair, count in zip(pairs, counts, strict=True)
    )


def _percentile_or_none(values: FloatArray, percentile: float) -> float | None:
    return None if values.size == 0 else float(np.percentile(values, percentile))


def _time_history_rows(first: CaseEvaluation, second: CaseEvaluation) -> list[Record]:
    position_difference = np.linalg.norm(first.trace.position_m - second.trace.position_m, axis=1)
    velocity_difference = np.linalg.norm(
        first.trace.velocity_m_s - second.trace.velocity_m_s, axis=1
    )
    charge_difference = np.abs(first.trace.charge_number - second.trace.charge_number)
    position_finite = np.isfinite(position_difference)
    velocity_finite = np.isfinite(velocity_difference)
    charge_finite = np.isfinite(charge_difference)
    result: list[Record] = []
    for time_s in np.unique(first.trace.time_s):
        at_time = first.trace.time_s == time_s
        common_active = at_time & first.trace.active & second.trace.active
        position = position_difference[common_active & position_finite]
        velocity = velocity_difference[common_active & velocity_finite]
        charge = charge_difference[common_active & charge_finite]
        sample_count = int(np.count_nonzero(at_time))
        result.append(
            {
                "field_source_mode": first.descriptor.field_source_mode,
                "diameter_nm": first.descriptor.diameter_nm,
                "variant_a": first.descriptor.ion_drag_model,
                "variant_b": second.descriptor.ion_drag_model,
                "time_s": float(time_s),
                "particle_sample_count": sample_count,
                "active_count_a": int(np.count_nonzero(at_time & first.trace.active)),
                "active_count_b": int(np.count_nonzero(at_time & second.trace.active)),
                "common_active_finite_position_count": int(position.size),
                "common_active_finite_position_fraction": (
                    float(position.size / sample_count) if sample_count else 0.0
                ),
                "position_difference_p50_m": _percentile_or_none(position, 50.0),
                "position_difference_p90_m": _percentile_or_none(position, 90.0),
                "position_difference_p99_m": _percentile_or_none(position, 99.0),
                "velocity_difference_p50_m_per_s": _percentile_or_none(velocity, 50.0),
                "velocity_difference_p90_m_per_s": _percentile_or_none(velocity, 90.0),
                "velocity_difference_p99_m_per_s": _percentile_or_none(velocity, 99.0),
                "charge_number_difference_p50": _percentile_or_none(charge, 50.0),
                "charge_number_difference_p90": _percentile_or_none(charge, 90.0),
                "charge_number_difference_p99": _percentile_or_none(charge, 99.0),
                "status_mismatch_fraction": float(
                    np.mean(first.trace.status_code[at_time] != second.trace.status_code[at_time])
                ),
            }
        )
    return result


def _assert_paired_initial_state(first: CaseEvaluation, second: CaseEvaluation) -> None:
    initial_time = float(np.min(first.trace.time_s))
    first_initial = first.trace.time_s == initial_time
    second_initial = second.trace.time_s == initial_time
    equal = (
        np.array_equal(
            first.trace.particle_id[first_initial], second.trace.particle_id[second_initial]
        )
        and np.array_equal(
            first.trace.position_m[first_initial], second.trace.position_m[second_initial]
        )
        and np.array_equal(
            first.trace.velocity_m_s[first_initial], second.trace.velocity_m_s[second_initial]
        )
        and np.array_equal(
            first.trace.charge_number[first_initial], second.trace.charge_number[second_initial]
        )
        and np.array_equal(
            first.trace.status_code[first_initial], second.trace.status_code[second_initial]
        )
        and np.array_equal(first.trace.active[first_initial], second.trace.active[second_initial])
    )
    if not equal:
        raise ValueError(
            "variant sensitivity requires identical t=0 particle state: "
            f"{first.descriptor.case_id} vs {second.descriptor.case_id}"
        )


def _particle_crossing_summary(
    trace: Trace,
    separation_m: FloatArray,
    valid: NDArray[np.bool_],
    threshold_m: float,
) -> Record:
    crossing_times: list[float] = []
    particle_count = 0
    starts = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.flatnonzero(trace.particle_id[1:] != trace.particle_id[:-1]) + 1,
        )
    )
    stops = np.concatenate((starts[1:], np.asarray([trace.particle_id.size], dtype=np.int64)))
    for start, stop in zip(starts, stops, strict=True):
        particle_count += 1
        crossed = valid[start:stop] & (separation_m[start:stop] >= threshold_m)
        if bool(np.any(crossed)):
            crossing_times.append(float(np.min(trace.time_s[start:stop][crossed])))
    times = np.asarray(crossing_times, dtype=np.float64)
    return {
        "first_any_particle_sampled_crossing_time_s": (
            None if times.size == 0 else float(np.min(times))
        ),
        "particle_first_crossing_time_p50_s": _percentile_or_none(times, 50.0),
        "particle_first_crossing_time_p90_s": _percentile_or_none(times, 90.0),
        "no_sampled_crossing_before_pair_censoring_fraction": float(
            1.0 - times.size / particle_count
        ),
    }


def _variant_sensitivity(
    evaluations: tuple[CaseEvaluation, ...], config: M3VConfig
) -> tuple[tuple[Record, ...], tuple[Record, ...]]:
    by_axis = {
        (
            item.descriptor.field_source_mode,
            item.descriptor.diameter_nm,
            item.descriptor.variant_directory,
        ): item
        for item in evaluations
    }
    variants = sorted({item.descriptor.variant_directory for item in evaluations})
    if len(variants) != 2:
        raise ValueError("M3-V requires exactly two ion-drag variants")
    rows: list[Record] = []
    time_rows: list[Record] = []
    axes = sorted(
        {(item.descriptor.field_source_mode, item.descriptor.diameter_nm) for item in evaluations}
    )
    for field_source, diameter in axes:
        first = by_axis[(field_source, diameter, variants[0])]
        second = by_axis[(field_source, diameter, variants[1])]
        if not (
            np.array_equal(first.trace.particle_id, second.trace.particle_id)
            and np.array_equal(first.trace.time_s, second.trace.time_s)
        ):
            raise ValueError(f"variant histories are not aligned for {field_source}/{diameter}nm")
        _assert_paired_initial_state(first, second)
        position_finite = np.isfinite(first.trace.position_m).all(axis=1) & np.isfinite(
            second.trace.position_m
        ).all(axis=1)
        velocity_finite = np.isfinite(first.trace.velocity_m_s).all(axis=1) & np.isfinite(
            second.trace.velocity_m_s
        ).all(axis=1)
        charge_finite = np.isfinite(first.trace.charge_number) & np.isfinite(
            second.trace.charge_number
        )
        joint_active = first.trace.active & second.trace.active
        common_position = joint_active & position_finite
        common_velocity = joint_active & velocity_finite
        common_charge = joint_active & charge_finite
        separation = np.linalg.norm(first.trace.position_m - second.trace.position_m, axis=1)
        velocity_separation = np.linalg.norm(
            first.trace.velocity_m_s - second.trace.velocity_m_s, axis=1
        )
        charge_separation = np.abs(first.trace.charge_number - second.trace.charge_number)
        terminal_first = _terminal_indices(first.trace)
        terminal_second = _terminal_indices(second.trace)
        terminal_position_finite = position_finite[terminal_first]
        final_separation = np.linalg.norm(
            first.trace.position_m[terminal_first] - second.trace.position_m[terminal_second],
            axis=1,
        )[terminal_position_finite]
        integrated_rms = _integrated_rms_by_particle(first.trace, separation, common_position)
        state_divergence = first.trace.status_code != second.trace.status_code
        first_status = first.trace.status_code[terminal_first]
        second_status = second.trace.status_code[terminal_second]
        time_rows.extend(_time_history_rows(first, second))
        rows.append(
            {
                "field_source_mode": field_source,
                "diameter_nm": diameter,
                "variant_a": first.descriptor.ion_drag_model,
                "variant_b": second.descriptor.ion_drag_model,
                "common_active_position_sample_count": int(np.count_nonzero(common_position)),
                "common_active_position_sample_fraction": float(np.mean(common_position)),
                "common_active_position_difference_p50_m": float(
                    np.percentile(separation[common_position], 50.0)
                ),
                "common_active_position_difference_p90_m": float(
                    np.percentile(separation[common_position], 90.0)
                ),
                "common_active_position_difference_p99_m": float(
                    np.percentile(separation[common_position], 99.0)
                ),
                "common_active_position_difference_max_m": float(
                    np.max(separation[common_position])
                ),
                "common_active_velocity_difference_p90_m_per_s": float(
                    np.percentile(velocity_separation[common_velocity], 90.0)
                ),
                "common_active_velocity_difference_p99_m_per_s": float(
                    np.percentile(velocity_separation[common_velocity], 99.0)
                ),
                "common_active_charge_number_difference_p90": float(
                    np.percentile(charge_separation[common_charge], 90.0)
                ),
                "common_active_charge_number_difference_p99": float(
                    np.percentile(charge_separation[common_charge], 99.0)
                ),
                "particle_time_rms_position_difference_median_m": float(np.median(integrated_rms)),
                "particle_time_rms_position_difference_p90_m": float(
                    np.percentile(integrated_rms, 90.0)
                ),
                "position_difference_threshold_m": config.position_difference_threshold_m,
                **_particle_crossing_summary(
                    first.trace,
                    separation,
                    common_position,
                    config.position_difference_threshold_m,
                ),
                "first_state_divergence_time_s": _first_time(first.trace.time_s, state_divergence),
                "active_state_mismatch_fraction": float(
                    np.mean(first.trace.active != second.trace.active)
                ),
                "finite_terminal_position_pair_count": int(final_separation.size),
                "finite_terminal_position_difference_median_m": (
                    None if final_separation.size == 0 else float(np.median(final_separation))
                ),
                "finite_terminal_position_difference_p90_m": (
                    None
                    if final_separation.size == 0
                    else float(np.percentile(final_separation, 90.0))
                ),
                "terminal_status_mismatch_fraction": float(np.mean(first_status != second_status)),
                "terminal_outcome_confusion": _outcome_confusion(first_status, second_status),
                "comparison_scope": (
                    "ion_drag_only_configuration_diff_single_stochastic_realization"
                    if field_source == "reduced_electrostatic_reference"
                    else "ion_drag_plus_lift_expression_confounder"
                ),
                "interpretation": (
                    "censored_time_history_model_form_sensitivity_not_validation_error;"
                    "force_causality_not_inferred_after_first_divergence;"
                    "rng_stream_equivalence_not_certified"
                ),
            }
        )
    return tuple(rows), tuple(time_rows)


def _particle_setting_map(package: Path) -> dict[tuple[str, str, str], tuple[str, str]]:
    path = package / "config" / "particle_physics_feature_settings.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return {
            (row["physics_tag"], row["feature_tag"], row["property"]): (
                row["selected_entities"],
                row["value"],
            )
            for row in csv.DictReader(stream)
        }


def _variant_config_differences(evaluations: tuple[CaseEvaluation, ...]) -> tuple[Record, ...]:
    by_axis = {
        (
            item.descriptor.field_source_mode,
            item.descriptor.diameter_nm,
            item.descriptor.variant_directory,
        ): item
        for item in evaluations
    }
    variants = sorted({item.descriptor.variant_directory for item in evaluations})
    axes = sorted(
        {(item.descriptor.field_source_mode, item.descriptor.diameter_nm) for item in evaluations}
    )
    result: list[Record] = []
    for field_source, diameter_nm in axes:
        first = by_axis[(field_source, diameter_nm, variants[0])]
        second = by_axis[(field_source, diameter_nm, variants[1])]
        first_settings = _particle_setting_map(first.descriptor.package)
        second_settings = _particle_setting_map(second.descriptor.package)
        for key in sorted(set(first_settings) | set(second_settings)):
            first_value = first_settings.get(key)
            second_value = second_settings.get(key)
            if first_value == second_value:
                continue
            result.append(
                {
                    "field_source_mode": field_source,
                    "diameter_nm": diameter_nm,
                    "physics_tag": key[0],
                    "feature_tag": key[1],
                    "property": key[2],
                    "variant_a_selected_entities": (
                        None if first_value is None else first_value[0]
                    ),
                    "variant_a_value": None if first_value is None else first_value[1],
                    "variant_b_selected_entities": (
                        None if second_value is None else second_value[0]
                    ),
                    "variant_b_value": None if second_value is None else second_value[1],
                    "non_ion_drag_confounder": key[1] != "idf",
                }
            )
    return tuple(result)


def _audit_variant_contents(
    records: list[dict[str, str]], variant: str
) -> tuple[
    list[dict[str, str]], set[str | None], set[str | None], set[tuple[str | None, str | None]]
]:
    inventories: list[dict[str, str]] = []
    parameters: set[str | None] = set()
    variables: set[str | None] = set()
    features: set[tuple[str | None, str | None]] = set()
    for record in records:
        if record.get("variant") != variant:
            continue
        match record.get("type"):
            case "inventory":
                inventories.append(record)
            case "parameter":
                parameters.add(record.get("name"))
            case "variable":
                variables.add(record.get("name"))
            case "physics_feature":
                features.add((record.get("physics_tag"), record.get("feature_tag")))
    return inventories, parameters, variables, features


def _audit_variant_complete(records: list[dict[str, str]], variant: str) -> bool:
    inventories, parameters, variables, features = _audit_variant_contents(records, variant)
    if len(inventories) != 1:
        return False
    physics_tags = inventories[0].get("physics_tags", "")
    if not all(tag in physics_tags for tag in ("fpt", "fptas", "esass")):
        return False
    expected_particle = {
        (physics_tag, feature_tag)
        for physics_tag in ("fpt", "fptas")
        for feature_tag in AUDIT_PARTICLE_FEATURES
    }
    expected_electrostatic = {
        ("esass", feature_tag) for feature_tag in AUDIT_ELECTROSTATIC_FEATURES
    }
    return (
        AUDIT_PARAMETERS.issubset(parameters)
        and AUDIT_VARIABLES.issubset(variables)
        and expected_particle.issubset(features)
        and expected_electrostatic.issubset(features)
    )


def _audit_inventory_complete(records: list[dict[str, str]]) -> bool:
    return _audit_read_only_records_complete(records) and all(
        _audit_variant_complete(records, variant) for variant in AUDIT_VARIANTS
    )


def _audit_read_only_records_complete(records: list[dict[str, str]]) -> bool:
    for record_type in ("model", "model_pass"):
        selected = [record for record in records if record.get("type") == record_type]
        variants = {record.get("variant") for record in selected}
        if len(selected) != len(AUDIT_VARIANTS) or variants != AUDIT_VARIANTS:
            return False
        if any(record.get("read_only") != "true" for record in selected):
            return False
    return True


def _audit_completion_record_complete(records: list[dict[str, str]]) -> bool:
    selected = [record for record in records if record.get("type") == "audit_pass"]
    return len(selected) == 1 and {
        "model_count": selected[0].get("model_count"),
        "read_only": selected[0].get("read_only"),
        "study_run": selected[0].get("study_run"),
        "model_save": selected[0].get("model_save"),
    } == {
        "model_count": "2",
        "read_only": "true",
        "study_run": "false",
        "model_save": "false",
    }


def _parse_model_audit(path: Path) -> tuple[list[dict[str, str]], bool]:
    records: list[dict[str, str]] = []
    if not path.is_file():
        return records, False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        marker = "M3V_JSON|"
        if marker not in line:
            continue
        value = json.loads(line.split(marker, maxsplit=1)[1])
        if isinstance(value, dict):
            records.append({str(key): str(item) for key, item in value.items()})
    audit_pass = _audit_completion_record_complete(records)
    return records, audit_pass and _audit_inventory_complete(records)


def _mph_hash_evidence_pass(path: Path, expected_models: tuple[Path, ...]) -> bool:
    if not path.is_file() or len(expected_models) != 2:
        return False
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(expected_models):
        return False
    expected = {str(model.resolve()): _sha256(model) for model in expected_models}
    observed = {row.get("path", ""): row for row in rows}
    return set(observed) == set(expected) and all(
        observed[model_path].get("unchanged", "").casefold() == "true"
        and observed[model_path].get("sha256_before", "").casefold() == model_hash.casefold()
        and observed[model_path].get("sha256_after", "").casefold() == model_hash.casefold()
        for model_path, model_hash in expected.items()
    )


def _metric_float(evaluation: CaseEvaluation, name: str) -> float:
    value = evaluation.metrics[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"case metric {name} is not numeric")
    return float(value)


def _all_metric_equal(evaluations: tuple[CaseEvaluation, ...], name: str, expected: Scalar) -> bool:
    return all(item.metrics[name] == expected for item in evaluations)


def _variant_diff_scope_pass(differences: tuple[Record, ...], config: M3VConfig) -> bool:
    observed = {
        (
            row["field_source_mode"],
            row["diameter_nm"],
            row["physics_tag"],
            row["feature_tag"],
            row["property"],
        )
        for row in differences
    }
    expected = {
        (field_source, diameter_nm, physics_tag, feature_tag, "F")
        for diameter_nm in config.diameters_nm
        for field_source, physics_tag, feature_tag in (
            ("imported_external_plasma_fields", "fpt", "idf"),
            ("imported_external_plasma_fields", "fpt", "liftfm"),
            ("reduced_electrostatic_reference", "fptas", "idf"),
        )
    }
    return observed == expected


def _pass_fail(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def _applicability_status(minimum_coverage: float) -> str:
    return "NOT_APPLICABLE" if minimum_coverage < 1.0 else "NOT_TESTED"


def _gates(
    evaluations: tuple[CaseEvaluation, ...],
    sensitivities: tuple[Record, ...],
    config_differences: tuple[Record, ...],
    config: M3VConfig,
    model_audit_pass: bool,
) -> tuple[Record, ...]:
    structure_pass = all(item.metrics["package_structure_status"] == "PASS" for item in evaluations)
    charge_coverages = [
        _metric_float(item, "charge_sampled_applicable_fraction") for item in evaluations
    ]
    epstein_coverages = [
        _metric_float(item, "epstein_sampled_applicable_fraction") for item in evaluations
    ]
    charge_formula_pass = _all_metric_equal(
        evaluations, "dataset_drift_aware_charge_rate_formula_parity", "PASS"
    )
    epstein_formula_pass = _all_metric_equal(evaluations, "epstein_force_formula_parity", "PASS")
    relevance_pass = structure_pass and charge_formula_pass and epstein_formula_pass
    scalar_mass_mismatches = sum(
        item.metrics["charge_scalar_ion_mass_contract"] != "MATCH" for item in evaluations
    )
    species_mismatches = sum(
        item.metrics["charge_two_species_assumption"] != "MATCH" for item in evaluations
    )
    variant_scope_pass = len(sensitivities) == 6 and _variant_diff_scope_pass(
        config_differences, config
    )
    return (
        {
            "gate": "M3V-01-direct-model-inventory",
            "status": _pass_fail(model_audit_pass),
            "reason": "two MPH files loaded read-only; no study run and no save",
        },
        {
            "gate": "M3V-02-reference-package-structure",
            "status": _pass_fail(structure_pass),
            "reason": "12 reference-only packages have the declared rectangular history and provenance",
        },
        {
            "gate": "M3V-03-p15-charge-trajectory-applicability",
            "status": _applicability_status(min(charge_coverages)),
            "reason": (
                f"sampled P15 applicability range={min(charge_coverages):.6g}.."
                f"{max(charge_coverages):.6g}; scalar-ion-mass mismatch cases="
                f"{scalar_mass_mismatches}/12; species mismatch cases={species_mismatches}/12; "
                "continuous-path certification is also required"
            ),
        },
        {
            "gate": "M3V-03b-reference-charge-rate-formula-parity",
            "status": _pass_fail(charge_formula_pass),
            "reason": (
                "frozen-state relative-drift regularized two-current law reconstructed from "
                "saved primitives; parity is provenance evidence, not physical acceptance"
            ),
        },
        {
            "gate": "M3V-03c-reference-charge-local-stiffness-characterization",
            "status": _pass_fail(charge_formula_pass),
            "reason": (
                "analytic frozen-state dR/dZ sampled at saved rows; this is not a continuous-path "
                "or integrator stability certificate"
            ),
        },
        {
            "gate": "M3V-04-p15-epstein-trajectory-applicability",
            "status": _applicability_status(min(epstein_coverages)),
            "reason": (
                f"sampled linear-Epstein coverage range={min(epstein_coverages):.6g}.."
                f"{max(epstein_coverages):.6g}; continuous-path certification is also required"
            ),
        },
        {
            "gate": "M3V-04b-reference-epstein-formula-parity",
            "status": _pass_fail(epstein_formula_pass),
            "reason": (
                "COMSOL Epstein force reconstructed with delta=1+sigma_R*pi/8; formula parity "
                "does not extend the linear model applicability domain"
            ),
        },
        {
            "gate": "M3V-05-force-and-charge-relevance",
            "status": _pass_fail(relevance_pass),
            "reason": (
                "all derived values are finite and independently replayable formulas match; "
                "saved-grid force quadrature remains relevance-only"
            ),
        },
        {
            "gate": "M3V-06-reference-package-variant-sensitivity",
            "status": _pass_fail(variant_scope_pass),
            "reason": (
                "full time histories compared; Case A has an ion-drag-only configuration diff "
                "but only one stochastic realization; Case P also has a lift expression diff"
            ),
        },
        {
            "gate": "M3V-07-full-production-trajectory-comparison",
            "status": "NOT_APPLICABLE",
            "reason": "reference trajectories contain unmatched charge and force closures",
        },
        {
            "gate": "M3V-08-boundary-validation",
            "status": "NOT_TESTED",
            "reason": "reference releases are internal and contain no enabled reflection case",
        },
        {
            "gate": "M3V-09-stochastic-validation",
            "status": "NOT_TESTED",
            "reason": "one Brownian seed cannot validate a stochastic distribution",
        },
        {
            "gate": "M3V-10-reduced-electrostatic-builder-parity",
            "status": "NOT_TESTED",
            "reason": "MPH closure inventoried; independent canonical-field builder is a later slice",
        },
    )


def _write_csv(path: Path, rows: tuple[Record, ...] | list[Record]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _record_float(record: Record, name: str) -> float:
    value = record.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"record value {name} is not numeric")
    return float(value)


def _priority_decision(evaluations: tuple[CaseEvaluation, ...]) -> tuple[Record, ...]:
    minimum_charge = min(
        _metric_float(item, "charge_sampled_applicable_fraction") for item in evaluations
    )
    minimum_epstein = min(
        _metric_float(item, "epstein_sampled_applicable_fraction") for item in evaluations
    )
    scalar_mass_mismatches = sum(
        item.metrics["charge_scalar_ion_mass_contract"] != "MATCH" for item in evaluations
    )
    species_mismatches = sum(
        item.metrics["charge_two_species_assumption"] != "MATCH" for item in evaluations
    )
    deterministic_force_rows = [
        row for item in evaluations for row in item.force_rows if row["force"] != "brownian_sample"
    ]
    maximum_velocity_scale = {
        str(force): max(
            _record_float(row, "integral_abs_force_dt_over_mass_p90_m_per_s")
            for row in deterministic_force_rows
            if row["force"] == force
        )
        for force in {row["force"] for row in deterministic_force_rows}
    }
    return (
        {
            "workstream": "canonical_field_production",
            "workstream_priority": 1,
            "candidate": "independent_reduced_electrostatic_builder",
            "decision": "mandatory_first_party_preprocessor_with_canonical_output",
            "evidence": "Case-A equations and boundary groups inventoried; independent parity not tested",
        },
        {
            "workstream": "trajectory_physics",
            "workstream_priority": 1,
            "candidate": "relative_drift_regularized_two_current_charge_v1",
            "decision": "required_before_reference_trajectory_replay",
            "evidence": (
                f"minimum sampled stationary-OML coverage={minimum_charge:.6g}; "
                f"scalar-mass mismatches={scalar_mass_mismatches}/12; species mismatches="
                f"{species_mismatches}/12; reference formula omits negative-ion collection "
                "and is not a universal model"
            ),
        },
        {
            "workstream": "trajectory_physics",
            "workstream_priority": 2,
            "candidate": "finite_speed_epstein_drag",
            "decision": "required_for_uncovered_reference_states",
            "evidence": f"minimum sampled linear-Epstein coverage={minimum_epstein:.6g}",
        },
        {
            "workstream": "trajectory_physics",
            "workstream_priority": 3,
            "candidate": "versioned_relative_flow_ion_drag",
            "decision": "implement_as_explicit_model_not_case_profile",
            "evidence": (
                "reference-only p90 abs-impulse/mass reaches "
                f"{maximum_velocity_scale['ion_drag']:.6g} m/s; paired Case-A histories diverge"
            ),
        },
        {
            "workstream": "trajectory_physics",
            "workstream_priority": 4,
            "candidate": "waldmann_thermophoresis_p16",
            "decision": "retain_as_separate_deterministic_model",
            "evidence": (
                "reference-only p90 abs-impulse/mass reaches "
                f"{maximum_velocity_scale['thermophoresis']:.6g} m/s; gradient oracle remains controlled work"
            ),
        },
        {
            "workstream": "trajectory_physics",
            "workstream_priority": 5,
            "candidate": "brownian_multi_seed_campaign",
            "decision": "validate_distribution_before_production_acceptance",
            "evidence": "current reference has one seed and only a timestep-dependent RMS noise scale",
        },
        {
            "workstream": "state_dimension",
            "workstream_priority": 1,
            "candidate": "cartesian_3d_p17",
            "decision": "retain_independent_product_roadmap_item",
            "evidence": "NOT_TESTED by this axisymmetric reference matrix",
        },
    )


def _write_summary(
    output: Path,
    gates: tuple[Record, ...],
    priorities: tuple[Record, ...],
) -> None:
    gate_lines = "\n".join(
        f"| {row['gate']} | {row['status']} | {row['reason']} |" for row in gates
    )
    priority_lines = "\n".join(
        f"| {row['workstream']} | {row['workstream_priority']} | {row['candidate']} | "
        f"{row['decision']} |"
        for row in priorities
    )
    output.write_text(
        "# M3-V result\n\n"
        "M3-V completed the external applicability/relevance decision; it did not certify "
        "full trajectory equivalence. `NOT_APPLICABLE` and `NOT_TESTED` are explicit "
        "scope results, not hidden passes.\n\n"
        "## Gates\n\n| Gate | Status | Reason |\n|---|---|---|\n"
        f"{gate_lines}\n\n"
        "## Implementation priority by independent workstream\n\n"
        "| Workstream | Priority | Candidate | Decision |\n|---|---:|---|---|\n"
        f"{priority_lines}\n",
        encoding="utf-8",
    )


def _case_input_hashes(evaluation: CaseEvaluation) -> dict[str, str]:
    package = evaluation.descriptor.package
    relative_paths = (
        Path("manifest.csv"),
        Path("validation/package_validation.csv"),
        Path("config/global_parameters.csv"),
        Path("config/particle_physics_feature_settings.csv"),
        Path("results/particle_history_full_tidy.csv"),
    )
    return {path.as_posix(): _sha256(package / path) for path in relative_paths}


def _write_reports(
    output: Path,
    config: M3VConfig,
    dataset_root: Path,
    model_audit_log: Path,
    model_records: list[dict[str, str]],
    evaluations: tuple[CaseEvaluation, ...],
) -> tuple[Record, ...]:
    output.mkdir(parents=True, exist_ok=True)
    case_rows = tuple(item.metrics for item in evaluations)
    force_rows = tuple(row for item in evaluations for row in item.force_rows)
    sensitivities, time_history = _variant_sensitivity(evaluations, config)
    config_differences = _variant_config_differences(evaluations)
    _, model_pass = _parse_model_audit(model_audit_log)
    expected_models = tuple(sorted((dataset_root / "model").glob("*.mph")))
    direct_model_evidence_pass = model_pass and _mph_hash_evidence_pass(
        output / "mph_hashes.csv", expected_models
    )
    gates = _gates(
        evaluations,
        sensitivities,
        config_differences,
        config,
        direct_model_evidence_pass,
    )
    priorities = _priority_decision(evaluations)
    _write_csv(output / "case_metrics.csv", case_rows)
    _write_csv(output / "force_relevance.csv", force_rows)
    _write_csv(output / "variant_sensitivity.csv", sensitivities)
    _write_csv(output / "variant_time_history.csv", time_history)
    _write_csv(output / "variant_config_diff.csv", config_differences)
    _write_csv(output / "gates.csv", gates)
    _write_csv(output / "priority_decision.csv", priorities)
    (output / "model_inventory.json").write_text(
        json.dumps(model_records, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "evaluation_id": "M3-V",
        "evaluation_revision": int(config.raw["evaluation_revision"]),
        "evaluation_schema_version": int(config.raw["schema_version"]),
        "tool_revision": TOOL_REVISION,
        "generated_utc": datetime.now(UTC).isoformat(),
        "dataset_classification": "reference_only",
        "dataset_root": str(dataset_root),
        "config": str(config.source),
        "config_sha256": _sha256(config.source),
        "model_audit_log": str(model_audit_log),
        "model_audit_log_sha256": _sha256(model_audit_log),
        "comsol_version": (
            (output / "comsol_version.txt").read_text(encoding="utf-8").strip()
            if (output / "comsol_version.txt").is_file()
            else "unavailable"
        ),
        "comsol_version_file_sha256": (
            _sha256(output / "comsol_version.txt")
            if (output / "comsol_version.txt").is_file()
            else "unavailable"
        ),
        "comsol_raw_model_audit_sha256": (
            _sha256(output / "comsol_model_audit.txt")
            if (output / "comsol_model_audit.txt").is_file()
            else "unavailable"
        ),
        "mph_pre_post_hash_evidence": (
            _sha256(output / "mph_hashes.csv")
            if (output / "mph_hashes.csv").is_file()
            else "unavailable"
        ),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "model_inputs": {path.name: _sha256(path) for path in expected_models},
        "case_inputs": {item.descriptor.case_id: _case_input_hashes(item) for item in evaluations},
        "gates": list(gates),
        "execution_status": "FAIL" if any(row["status"] == "FAIL" for row in gates) else "PASS",
        "physics_certification_status": "NOT_CERTIFIED",
    }
    (output / "comparison_manifest.json").write_text(
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_summary(output / "README.md", gates, priorities)
    return gates


def _default_repository_root() -> Path:
    return Path(__file__).resolve().parents[5]


def main() -> int:
    repository_root = _default_repository_root()
    solver_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("cases") / "m3v.yaml"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=repository_root / "model_dataset" / "cf4_o2_etch_caseA_nonlinear_sass",
    )
    parser.add_argument("--output-dir", type=Path, default=solver_root / "evidence" / "m3v")
    parser.add_argument("--model-audit-log", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    arguments = parser.parse_args()

    config = load_evaluation_config(arguments.config.resolve())
    dataset_root = arguments.dataset_root.resolve()
    model_audit_log = arguments.model_audit_log.resolve()
    model_records, model_audit_pass = _parse_model_audit(model_audit_log)
    if not model_audit_pass:
        print("M3-V: COMSOL model audit is incomplete", file=sys.stderr)
    evaluations = tuple(
        _evaluate_case(descriptor, config) for descriptor in _descriptors(dataset_root, config)
    )
    gates = _write_reports(
        arguments.output_dir.resolve(),
        config,
        dataset_root,
        model_audit_log,
        model_records,
        evaluations,
    )
    failed = [str(row["gate"]) for row in gates if row["status"] == "FAIL"]
    print(
        json.dumps(
            {
                "evaluation_id": "M3-V",
                "status": "FAIL" if failed else "PASS",
                "failed_gates": failed,
                "output": str(arguments.output_dir.resolve()),
            },
            sort_keys=True,
        )
    )
    return 1 if arguments.strict and failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
