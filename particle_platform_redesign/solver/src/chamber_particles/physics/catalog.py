"""Resolve the small, explicit physics catalog into one immutable plan."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Literal, cast

from .forces import (
    EPSTEIN_MAX_SPEED_OVER_MEAN_THERMAL,
    WALDMANN_GALLIS_MAX_SPEED_OVER_MEAN_THERMAL,
)

PHYSICS_CATALOG_REVISION = "declared_primitive_physics_catalog_v25"

BROWNIAN_MIDPOINT_2D_REVISION = "inertial_langevin_fdt_epstein_linear_midpoint_2d_v2"

_EPSTEIN_LINEAR_REVISION = "epstein_linear_v1"
_EPSTEIN_LINEAR_EFFECTIVE_GAS_REVISION = "epstein_linear_effective_gas_sensitivity_v1"
_WALDMANN_GALLIS_SINGLE_SPECIES_REVISION = (
    "waldmann_gallis_free_molecular_single_species_heat_flux_v1"
)
_WALDMANN_GALLIS_EFFECTIVE_GAS_REVISION = (
    "waldmann_gallis_free_molecular_effective_gas_heat_flux_sensitivity_v1"
)
_TALBOT_REVISION = "talbot_cross_regime_radius_knudsen_v1"
_SAFFMAN_REVISION = "saffman_unbounded_creeping_shear_v1"
_AGGREGATE_TWO_CURRENT_REVISION = "aggregate_relative_drift_regularized_two_current_v1"
_AGGREGATE_THREE_CURRENT_REVISION = "aggregate_relative_drift_regularized_three_current_v1"

type ValueConstraint = Literal["finite", "positive", "nonnegative"]

type CoordinateSystem = Literal["cartesian_xy", "axisymmetric_rz"]


class PhysicsConfigurationError(ValueError):
    """A selected model or one of its declared inputs is unsupported."""


SUPPORTED_INTEGRATORS = ("rk4_fixed", "exponential_midpoint", "ou_langevin")
_DETERMINISTIC_INTEGRATORS = SUPPORTED_INTEGRATORS[:2]
_COORDINATE_SYSTEMS: tuple[CoordinateSystem, ...] = ("cartesian_xy", "axisymmetric_rz")


@dataclass(frozen=True, slots=True)
class ModelFieldDescriptor:
    """Canonical SI metadata for one model input role in one coordinate chart."""

    parameter: str
    coordinate_system: CoordinateSystem
    unit: str
    components: tuple[str, ...]
    stored_basis: str
    value_constraint: ValueConstraint
    zero_on_rz_axis: bool


@dataclass(frozen=True, slots=True)
class ModelCertificateRoles:
    """Inputs used by each path proof, distinct from stage inputs.

    Constant acceleration declares family support only. Preparation still
    proves constant fields and includes every selected force contribution.
    """

    local_applicability_fields: tuple[str, ...]
    local_additive_fields: tuple[str, ...]
    constant_acceleration: bool


@dataclass(frozen=True, slots=True)
class ModelDescriptor:
    """Input keys and capabilities owned by the selected scientific revision.

    Integrator names describe the model individually. The selected noise/drag
    pair, field support and runtime applicability still require case prepare.
    Field parameters are caller-supplied canonical field names, not fixed names.
    """

    category: str
    model: str
    revision: str
    coordinate_systems: tuple[CoordinateSystem, ...]
    integrators: tuple[str, ...]
    required_parameters: tuple[str, ...]
    optional_parameters: tuple[str, ...]
    field_parameters: tuple[str, ...]
    notes: tuple[str, ...]
    certificates: ModelCertificateRoles
    fields: tuple[ModelFieldDescriptor, ...] = ()


def _descriptor(
    category: str,
    model: str,
    revision: str,
    parameters: tuple[str, ...] = (),
    *,
    certificates: ModelCertificateRoles,
    optional: tuple[str, ...] = (),
    coordinates: tuple[CoordinateSystem, ...] = _COORDINATE_SYSTEMS,
    integrators: tuple[str, ...] = SUPPORTED_INTEGRATORS,
    notes: tuple[str, ...] = (),
) -> ModelDescriptor:
    selectors = ("model",) if (category, model) == ("charge", "fixed") else ("model", "revision")
    required = tuple(sorted((*selectors, *parameters)))
    return ModelDescriptor(
        category,
        model,
        revision,
        coordinates,
        integrators,
        required,
        tuple(sorted(optional)),
        tuple(name for name in required if name.endswith("_field")),
        notes,
        certificates,
    )


_OML_COLLECTION_PARAMETERS = (
    "applicability",
    "electron_number_density_field",
    "positive_ion_number_density_field",
    "electron_temperature_field",
    "positive_ion_temperature_field",
    "positive_ion_velocity_field",
    "positive_ion_mass_kg",
)
_AGGREGATE_COLLECTION_PARAMETERS = (
    "applicability",
    "electron_number_density_field",
    "positive_ion_number_density_field",
    "electron_thermal_voltage_field",
    "positive_ion_thermal_voltage_field",
    "positive_ion_velocity_field",
    "effective_positive_ion_mass_field",
    "screening_length_field",
    "maximum_relative_ion_speed_m_s",
)
_EPSTEIN_PARAMETERS = (
    "gas_velocity_field",
    "gas_density_field",
    "gas_temperature_field",
    "gas_mean_free_path_field",
    "gas_molecular_mass_kg",
    "applicability",
)
_WALDMANN_PARAMETERS = (
    "gas_velocity_field",
    "gas_temperature_field",
    "gas_translational_heat_flux_field",
    "gas_mean_free_path_field",
    "gas_molecular_mass_kg",
    "applicability",
)
_REFERENCE_NOTE = (
    "Explicit mathematical reference coefficients; not a physical-model certification.",
    "particle_properties must exactly match every realized particle and every selected reference model.",
)

_MODEL_DESCRIPTORS = (
    _descriptor(
        "charge",
        "fixed",
        "fixed_charge_v1",
        notes=("Source charge_number is constant; omit revision in the input selector.",),
        certificates=ModelCertificateRoles((), (), True),
    ),
    _descriptor(
        "charge",
        "plasma_continuous",
        "oml_stationary_maxwellian_debye_huckel_v1",
        _OML_COLLECTION_PARAMETERS,
        notes=("Electron and ion temperature fields use K; applicability must be error.",),
        certificates=ModelCertificateRoles(
            (
                "positive_ion_velocity_field",
                "electron_number_density_field",
                "positive_ion_number_density_field",
                "electron_temperature_field",
                "positive_ion_temperature_field",
            ),
            (),
            False,
        ),
    ),
    _descriptor(
        "charge",
        "plasma_continuous",
        "oml_shifted_maxwellian_single_ion_negative_debye_huckel_v1",
        (*_OML_COLLECTION_PARAMETERS, "maximum_ion_drift_ratio"),
        notes=("Electron and ion temperature fields use K; applicability must be error.",),
        certificates=ModelCertificateRoles(
            (
                "positive_ion_velocity_field",
                "electron_number_density_field",
                "positive_ion_number_density_field",
                "electron_temperature_field",
                "positive_ion_temperature_field",
            ),
            (),
            False,
        ),
    ),
    _descriptor(
        "charge",
        "plasma_continuous",
        _AGGREGATE_TWO_CURRENT_REVISION,
        _AGGREGATE_COLLECTION_PARAMETERS,
        notes=("Aggregate thermal voltage fields use V; applicability must be error.",),
        certificates=ModelCertificateRoles(("positive_ion_velocity_field",), (), False),
    ),
    _descriptor(
        "charge",
        "plasma_continuous",
        _AGGREGATE_THREE_CURRENT_REVISION,
        (
            *_AGGREGATE_COLLECTION_PARAMETERS,
            "negative_ion_number_density_field",
            "negative_ion_thermal_voltage_field",
            "negative_ion_velocity_field",
            "effective_negative_ion_mass_field",
        ),
        notes=("Aggregate thermal voltage fields use V; applicability must be error.",),
        certificates=ModelCertificateRoles(
            ("positive_ion_velocity_field", "negative_ion_velocity_field"), (), False
        ),
    ),
    _descriptor(
        "charge",
        "prescribed_exponential_linear",
        "prescribed_exponential_linear_collection_v1",
        (
            "electron_rate_reference_number_s",
            "electron_exponent_offset_field",
            "electron_charge_slope_field",
            "ion_rate_field",
            "ion_charge_slope_field",
            "particle_properties",
        ),
        notes=_REFERENCE_NOTE,
        certificates=ModelCertificateRoles((), (), False),
    ),
    _descriptor(
        "drag",
        "epstein_linear",
        _EPSTEIN_LINEAR_REVISION,
        (*_EPSTEIN_PARAMETERS, "delta"),
        certificates=ModelCertificateRoles(
            ("gas_velocity_field", "gas_mean_free_path_field", "gas_temperature_field"), (), False
        ),
    ),
    _descriptor(
        "drag",
        "epstein_linear",
        _EPSTEIN_LINEAR_EFFECTIVE_GAS_REVISION,
        (*_EPSTEIN_PARAMETERS, "delta", "maximum_speed_ratio"),
        notes=(
            "Producer-owned effective-gas sensitivity; no implicit single-species equivalence.",
        ),
        certificates=ModelCertificateRoles(
            ("gas_velocity_field", "gas_mean_free_path_field", "gas_temperature_field"), (), False
        ),
    ),
    _descriptor(
        "drag",
        "epstein_finite_speed",
        "epstein_finite_speed_maxwell_mixed_equal_temperature_v1",
        (*_EPSTEIN_PARAMETERS, "diffuse_reflection_fraction", "maximum_speed_ratio"),
        integrators=_DETERMINISTIC_INTEGRATORS,
        notes=("Nonlinear finite-speed drag cannot be paired with the current OU noise models.",),
        certificates=ModelCertificateRoles(
            ("gas_velocity_field", "gas_mean_free_path_field", "gas_temperature_field"), (), False
        ),
    ),
    _descriptor(
        "drag",
        "stokes_cunningham",
        "stokes_cunningham_allen_raabe_air_v1",
        (
            "gas_velocity_field",
            "gas_density_field",
            "gas_dynamic_viscosity_field",
            "gas_mean_free_path_field",
            "applicability",
        ),
        integrators=_DETERMINISTIC_INTEGRATORS,
        notes=("Air-slip validity gates are unchanged; no fallback to a rarefied-gas model.",),
        certificates=ModelCertificateRoles(
            (
                "gas_velocity_field",
                "gas_mean_free_path_field",
                "gas_density_field",
                "gas_dynamic_viscosity_field",
            ),
            (),
            False,
        ),
    ),
    _descriptor(
        "drag",
        "prescribed_linear_friction",
        "prescribed_linear_friction_v1",
        ("gas_velocity_field", "friction_coefficient_field", "particle_properties"),
        notes=_REFERENCE_NOTE,
        certificates=ModelCertificateRoles((), (), False),
    ),
    _descriptor(
        "noise",
        "inertial_langevin_fdt",
        BROWNIAN_MIDPOINT_2D_REVISION,
        ("interval_tree_depth",),
        optional=("adaptive_max_depth",),
        integrators=("ou_langevin",),
        notes=(
            "Requires a supported linear Epstein drag revision, not finite-speed or Stokes drag.",
            "Uses same-stage gas temperature and the existing joint position/velocity OU path.",
        ),
        certificates=ModelCertificateRoles((), (), False),
    ),
    _descriptor(
        "noise",
        "prescribed_force_diffusion",
        "prescribed_force_diffusion_midpoint_2d_v1",
        ("force_spectral_density_field", "interval_tree_depth", "particle_properties"),
        optional=("adaptive_max_depth",),
        integrators=("ou_langevin",),
        notes=(
            *_REFERENCE_NOTE,
            "Requires prescribed_linear_friction; diffusion is independent of FDT temperature.",
        ),
        certificates=ModelCertificateRoles((), (), False),
    ),
    _descriptor(
        "thermophoresis",
        "waldmann_gallis",
        _WALDMANN_GALLIS_SINGLE_SPECIES_REVISION,
        _WALDMANN_PARAMETERS,
        certificates=ModelCertificateRoles(
            ("gas_velocity_field", "gas_temperature_field", "gas_mean_free_path_field"), (), False
        ),
    ),
    _descriptor(
        "thermophoresis",
        "waldmann_gallis",
        _WALDMANN_GALLIS_EFFECTIVE_GAS_REVISION,
        (*_WALDMANN_PARAMETERS, "maximum_speed_ratio"),
        notes=("Producer-owned effective-gas sensitivity; applicability must be error.",),
        certificates=ModelCertificateRoles(
            ("gas_velocity_field", "gas_temperature_field", "gas_mean_free_path_field"), (), False
        ),
    ),
    _descriptor(
        "thermophoresis",
        "talbot",
        _TALBOT_REVISION,
        (
            "gas_temperature_field",
            "gas_temperature_gradient_field",
            "gas_density_field",
            "gas_dynamic_viscosity_field",
            "gas_thermal_conductivity_field",
            "gas_mean_free_path_field",
            "particle_thermal_conductivity_W_m_K",
            "thermal_slip_coefficient",
            "momentum_exchange_coefficient",
            "thermal_exchange_coefficient",
            "applicability",
        ),
        notes=(
            "Uses radius-based Knudsen number; coefficients are explicit, not universal defaults.",
        ),
        certificates=ModelCertificateRoles(
            (
                "gas_temperature_field",
                "gas_density_field",
                "gas_dynamic_viscosity_field",
                "gas_thermal_conductivity_field",
                "gas_mean_free_path_field",
                "gas_temperature_gradient_field",
            ),
            (),
            False,
        ),
    ),
    _descriptor(
        "ion_drag",
        "barnes_collisionless",
        "barnes_collisionless_effective_speed_single_positive_ion_negative_debye_huckel_v1",
        (
            *_OML_COLLECTION_PARAMETERS,
            "ion_neutral_mean_free_path_field",
            "maximum_ion_drift_ratio",
        ),
        certificates=ModelCertificateRoles(
            (
                "positive_ion_velocity_field",
                "electron_number_density_field",
                "positive_ion_number_density_field",
                "electron_temperature_field",
                "positive_ion_temperature_field",
                "ion_neutral_mean_free_path_field",
            ),
            (),
            False,
        ),
    ),
    _descriptor(
        "ion_drag",
        "screened_collection_orbital",
        "relative_flow_screened_collection_orbital_aggregate_ion_v1",
        (
            "positive_ion_number_density_field",
            "positive_ion_thermal_voltage_field",
            "positive_ion_velocity_field",
            "effective_positive_ion_mass_field",
            "screening_length_field",
            "ion_neutral_mean_free_path_field",
            "maximum_relative_ion_speed_m_s",
            "applicability",
        ),
        certificates=ModelCertificateRoles(
            ("positive_ion_velocity_field",),
            (
                "positive_ion_number_density_field",
                "positive_ion_thermal_voltage_field",
                "positive_ion_velocity_field",
                "effective_positive_ion_mass_field",
                "screening_length_field",
                "ion_neutral_mean_free_path_field",
            ),
            False,
        ),
    ),
    _descriptor(
        "ion_drag",
        "image_orbital_sensitivity",
        "electric_field_directed_image_orbital_sensitivity_v1",
        (
            "positive_ion_number_density_field",
            "electron_thermal_voltage_field",
            "positive_ion_thermal_voltage_field",
            "positive_ion_velocity_field",
            "effective_positive_ion_mass_field",
            "screening_length_field",
            "electric_field",
            "applicability",
        ),
        notes=("Sensitivity model; its E binding must agree with any selected Coulomb model.",),
        certificates=ModelCertificateRoles((), (), False),
    ),
    _descriptor(
        "ion_drag",
        "prescribed_charge_polynomial",
        "prescribed_charge_polynomial_v1",
        (
            "constant_force_field",
            "linear_charge_force_field",
            "quadratic_charge_force_field",
            "particle_properties",
        ),
        notes=_REFERENCE_NOTE,
        certificates=ModelCertificateRoles((), (), False),
    ),
    _descriptor(
        "dielectrophoresis",
        "quasistatic_spherical",
        "quasistatic_spherical_gradient_e2_v1",
        (
            "gradient_mean_e_squared_field",
            "medium_relative_permittivity",
            "real_clausius_mossotti_factor",
            "maximum_point_dipole_radius_m",
        ),
        notes=("Consumes grad(|E|^2), not E itself; the point-dipole radius bound is explicit.",),
        certificates=ModelCertificateRoles((), (), True),
    ),
    _descriptor(
        "lift",
        "rarefied_vorticity_sensitivity",
        "rarefied_vorticity_sensitivity_rz_v1",
        (
            "gas_velocity_field",
            "gas_density_field",
            "gas_mean_free_path_field",
            "azimuthal_gas_vorticity_field",
            "lift_coefficient",
            "applicability",
        ),
        coordinates=("axisymmetric_rz",),
        notes=("Rarefied-gas sensitivity; signed azimuthal vorticity is an explicit primitive.",),
        certificates=ModelCertificateRoles(("gas_mean_free_path_field",), (), False),
    ),
    _descriptor(
        "lift",
        "saffman",
        _SAFFMAN_REVISION,
        (
            "gas_velocity_field",
            "gas_density_field",
            "gas_dynamic_viscosity_field",
            "gas_mean_free_path_field",
            "out_of_plane_gas_vorticity_field",
            "applicability",
        ),
        notes=("Creeping/shear/slip validity gates must overlap all co-selected models.",),
        certificates=ModelCertificateRoles(
            (
                "gas_mean_free_path_field",
                "gas_velocity_field",
                "gas_density_field",
                "gas_dynamic_viscosity_field",
                "out_of_plane_gas_vorticity_field",
            ),
            (),
            False,
        ),
    ),
    _descriptor(
        "lift",
        "prescribed_rotational_lift",
        "prescribed_rotational_lift_v1",
        ("gas_velocity_field", "rotation_coefficient_field", "particle_properties"),
        notes=_REFERENCE_NOTE,
        certificates=ModelCertificateRoles((), (), False),
    ),
    _descriptor(
        "electric",
        "coulomb",
        "electric_coulomb_v1",
        ("electric_field",),
        certificates=ModelCertificateRoles((), ("electric_field",), True),
    ),
    _descriptor(
        "electric",
        "coulomb",
        "electric_coulomb_charge_cap_v1",
        ("electric_field", "maximum_abs_charge_number"),
        notes=(
            "Caps Coulomb force only; continuous charge and ion drag keep the uncapped stage charge.",
        ),
        certificates=ModelCertificateRoles((), ("electric_field",), True),
    ),
    _descriptor(
        "gravity_buoyancy",
        "standard",
        "gravity_buoyancy_standard_v1",
        ("gas_density_field", "gravity_m_s2"),
        notes=(
            "RZ requires zero radial gravity; displaced volume is an independent particle authority.",
        ),
        certificates=ModelCertificateRoles((), (), True),
    ),
)
SUPPORTED_PHYSICS_CATEGORIES = frozenset(model.category for model in _MODEL_DESCRIPTORS)


def describe_models() -> tuple[ModelDescriptor, ...]:
    """Return immutable catalog-owned selectors, parameter keys and capabilities."""

    return tuple(
        replace(
            descriptor,
            fields=tuple(
                field
                for coordinate in descriptor.coordinate_systems
                for field in _model_field_roles(descriptor, coordinate)
            ),
        )
        for descriptor in _MODEL_DESCRIPTORS
    )


def _model_descriptor(category: str, model: str, revision: str) -> ModelDescriptor | None:
    return next(
        (
            item
            for item in _MODEL_DESCRIPTORS
            if (item.category, item.model, item.revision) == (category, model, revision)
        ),
        None,
    )


def _validate_model_structure(
    category: str, model: Mapping[str, object], coordinate_system: CoordinateSystem
) -> ModelDescriptor:
    name = _text(model.get("model"), f"physics.{category}.model")
    revision = (
        "fixed_charge_v1"
        if (category, name) == ("charge", "fixed")
        else _text(model.get("revision"), f"physics.{category}.revision")
    )
    descriptor = _model_descriptor(category, name, revision)
    if descriptor is None:
        names = sorted({item.model for item in _MODEL_DESCRIPTORS if item.category == category})
        if name not in names:
            raise PhysicsConfigurationError(
                f"physics.{category} model must be {' or '.join(names)}"
            )
        revisions = (
            item.revision
            for item in _MODEL_DESCRIPTORS
            if item.category == category and item.model == name
        )
        raise PhysicsConfigurationError(f"{name} revision must be {' or '.join(revisions)}")
    expected = set(descriptor.required_parameters) | (
        set(descriptor.optional_parameters) & set(model)
    )
    _exact_keys(model, expected, f"physics.{category}")
    if coordinate_system not in descriptor.coordinate_systems:
        raise PhysicsConfigurationError(
            f"{revision} supports {', '.join(descriptor.coordinate_systems)} motion only"
        )
    return descriptor


@dataclass(frozen=True, slots=True)
class RequiredField:
    """The exact canonical metadata expected for one primitive field."""

    name: str
    unit: str
    components: tuple[str, ...]
    stored_basis: str
    value_constraint: ValueConstraint = "finite"
    zero_on_rz_axis: bool = False


@dataclass(frozen=True, slots=True)
class ReferenceParticleProperties:
    """Particle authorities used to produce a homogeneous reference coefficient set."""

    mass_kg: float
    drag_diameter_m: float
    electrostatic_radius_m: float
    displaced_volume_m3: float


@dataclass(frozen=True, slots=True)
class PrescribedLinearFrictionPlan:
    revision: str
    gas_velocity_field: str
    friction_coefficient_field: str
    particle_properties: ReferenceParticleProperties


@dataclass(frozen=True, slots=True)
class PrescribedForceDiffusionNoisePlan:
    revision: str
    force_spectral_density_field: str
    interval_tree_depth: int
    adaptive_max_depth: int
    particle_properties: ReferenceParticleProperties


@dataclass(frozen=True, slots=True)
class PrescribedExponentialLinearChargePlan:
    revision: str
    electron_rate_reference_number_s: float
    electron_exponent_offset_field: str
    electron_charge_slope_field: str
    ion_rate_field: str
    ion_charge_slope_field: str
    particle_properties: ReferenceParticleProperties


@dataclass(frozen=True, slots=True)
class PrescribedChargePolynomialIonDragPlan:
    revision: str
    constant_force_field: str
    linear_charge_force_field: str
    quadratic_charge_force_field: str
    particle_properties: ReferenceParticleProperties


@dataclass(frozen=True, slots=True)
class PrescribedRotationalLiftPlan:
    revision: str
    gas_velocity_field: str
    rotation_coefficient_field: str
    particle_properties: ReferenceParticleProperties


@dataclass(frozen=True, slots=True)
class PlasmaContinuousChargePlan:
    """Resolved inputs for one explicit continuous-plasma charge revision."""

    revision: str
    electron_number_density_field: str
    positive_ion_number_density_field: str
    electron_temperature_field: str
    positive_ion_temperature_field: str
    positive_ion_velocity_field: str
    positive_ion_mass_kg: float
    maximum_ion_drift_ratio: float | None


@dataclass(frozen=True, slots=True)
class AggregateRelativeDriftChargePlan:
    """Resolved fields for one aggregate regularized charge revision."""

    revision: str
    electron_number_density_field: str
    positive_ion_number_density_field: str
    electron_thermal_voltage_field: str
    positive_ion_thermal_voltage_field: str
    positive_ion_velocity_field: str
    effective_positive_ion_mass_field: str
    screening_length_field: str
    maximum_relative_ion_speed_m_s: float
    negative_ion_number_density_field: str | None = None
    negative_ion_thermal_voltage_field: str | None = None
    negative_ion_velocity_field: str | None = None
    effective_negative_ion_mass_field: str | None = None

    @property
    def has_negative_ion_current(self) -> bool:
        """Whether the selected revision includes the negative-ion current."""

        return self.revision == _AGGREGATE_THREE_CURRENT_REVISION


type ChargePlan = (
    PlasmaContinuousChargePlan
    | AggregateRelativeDriftChargePlan
    | PrescribedExponentialLinearChargePlan
)


@dataclass(frozen=True, slots=True)
class EpsteinDragPlan:
    """Resolved parameters and field names for one linear-Epstein revision."""

    revision: str
    gas_velocity_field: str
    gas_density_field: str
    gas_temperature_field: str
    gas_mean_free_path_field: str
    gas_molecular_mass_kg: float
    delta: float
    maximum_speed_ratio: float


@dataclass(frozen=True, slots=True)
class FiniteSpeedEpsteinDragPlan:
    """Resolved inputs for the equal-temperature Maxwell mixed sphere drag."""

    gas_velocity_field: str
    gas_density_field: str
    gas_temperature_field: str
    gas_mean_free_path_field: str
    gas_molecular_mass_kg: float
    diffuse_reflection_fraction: float
    maximum_speed_ratio: float


@dataclass(frozen=True, slots=True)
class StokesCunninghamDragPlan:
    """Resolved fields for ``stokes_cunningham_allen_raabe_air_v1``."""

    gas_velocity_field: str
    gas_density_field: str
    gas_dynamic_viscosity_field: str
    gas_mean_free_path_field: str


type DragPlan = (
    EpsteinDragPlan
    | FiniteSpeedEpsteinDragPlan
    | StokesCunninghamDragPlan
    | PrescribedLinearFrictionPlan
)


@dataclass(frozen=True, slots=True)
class InertialLangevinNoisePlan:
    """Resolved controls for the inertial Langevin FDT revision."""

    revision: str
    interval_tree_depth: int
    adaptive_max_depth: int


@dataclass(frozen=True, slots=True)
class ElectricPlan:
    """Resolved field name for ``electric_coulomb_v1``."""

    electric_field: str
    maximum_abs_charge_number: float | None = None

    @property
    def revision(self) -> str:
        return (
            "electric_coulomb_v1"
            if self.maximum_abs_charge_number is None
            else "electric_coulomb_charge_cap_v1"
        )


@dataclass(frozen=True, slots=True)
class QuasistaticSphericalDielectrophoresisPlan:
    """Resolved inputs for the first quasistatic spherical DEP revision."""

    gradient_mean_e_squared_field: str
    medium_relative_permittivity: float
    real_clausius_mossotti_factor: float
    maximum_point_dipole_radius_m: float


@dataclass(frozen=True, slots=True)
class WaldmannGallisThermophoresisPlan:
    """Resolved inputs for one free-molecular heat-flux revision."""

    revision: str
    gas_velocity_field: str
    gas_temperature_field: str
    gas_translational_heat_flux_field: str
    gas_mean_free_path_field: str
    gas_molecular_mass_kg: float
    maximum_speed_ratio: float


@dataclass(frozen=True, slots=True)
class TalbotThermophoresisPlan:
    """Resolved inputs for the explicitly selected Talbot cross-regime revision."""

    gas_temperature_field: str
    gas_temperature_gradient_field: str
    gas_density_field: str
    gas_dynamic_viscosity_field: str
    gas_thermal_conductivity_field: str
    gas_mean_free_path_field: str
    particle_thermal_conductivity_W_m_K: float
    thermal_slip_coefficient: float
    momentum_exchange_coefficient: float
    thermal_exchange_coefficient: float


type ThermophoresisPlan = WaldmannGallisThermophoresisPlan | TalbotThermophoresisPlan


@dataclass(frozen=True, slots=True)
class BarnesCollisionlessIonDragPlan:
    """Resolved inputs for the first collisionless Barnes ion-drag revision."""

    electron_number_density_field: str
    positive_ion_number_density_field: str
    electron_temperature_field: str
    positive_ion_temperature_field: str
    positive_ion_velocity_field: str
    ion_neutral_mean_free_path_field: str
    positive_ion_mass_kg: float
    maximum_ion_drift_ratio: float


@dataclass(frozen=True, slots=True)
class RelativeFlowScreenedIonDragPlan:
    """Resolved aggregate-ion inputs for the relative-flow sensitivity revision."""

    positive_ion_number_density_field: str
    positive_ion_thermal_voltage_field: str
    positive_ion_velocity_field: str
    effective_positive_ion_mass_field: str
    screening_length_field: str
    ion_neutral_mean_free_path_field: str
    maximum_relative_ion_speed_m_s: float


@dataclass(frozen=True, slots=True)
class ElectricFieldDirectedImageIonDragPlan:
    """Resolved aggregate-ion inputs for the electric-field-directed sensitivity."""

    positive_ion_number_density_field: str
    electron_thermal_voltage_field: str
    positive_ion_thermal_voltage_field: str
    positive_ion_velocity_field: str
    effective_positive_ion_mass_field: str
    screening_length_field: str
    electric_field: str


type IonDragPlan = (
    BarnesCollisionlessIonDragPlan
    | RelativeFlowScreenedIonDragPlan
    | ElectricFieldDirectedImageIonDragPlan
    | PrescribedChargePolynomialIonDragPlan
)


@dataclass(frozen=True, slots=True)
class GravityBuoyancyPlan:
    """Resolved inputs for ``gravity_buoyancy_standard_v1``."""

    gas_density_field: str
    gravity_m_s2: tuple[float, float]


@dataclass(frozen=True, slots=True)
class RarefiedVorticityLiftPlan:
    """Resolved inputs for the RZ free-molecular lift sensitivity revision."""

    gas_velocity_field: str
    gas_density_field: str
    gas_mean_free_path_field: str
    azimuthal_gas_vorticity_field: str
    lift_coefficient: float


@dataclass(frozen=True, slots=True)
class SaffmanLiftPlan:
    """Resolved inputs for unbounded creeping-flow Saffman lift."""

    coordinate_system: CoordinateSystem
    gas_velocity_field: str
    gas_density_field: str
    gas_dynamic_viscosity_field: str
    gas_mean_free_path_field: str
    out_of_plane_gas_vorticity_field: str


type LiftPlan = RarefiedVorticityLiftPlan | SaffmanLiftPlan | PrescribedRotationalLiftPlan


@dataclass(frozen=True, slots=True)
class SelectedModel:
    """Keep the selected descriptor and caller field names instead of re-inferring identity."""

    descriptor: ModelDescriptor
    field_names: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class PhysicsPlan:
    """One model per category, in the fixed contribution order."""

    charge: ChargePlan | None
    drag: DragPlan | None
    noise: InertialLangevinNoisePlan | PrescribedForceDiffusionNoisePlan | None
    thermophoresis: ThermophoresisPlan | None
    ion_drag: IonDragPlan | None
    dielectrophoresis: QuasistaticSphericalDielectrophoresisPlan | None
    lift: LiftPlan | None
    electric: ElectricPlan | None
    gravity_buoyancy: GravityBuoyancyPlan | None
    required_fields: tuple[RequiredField, ...]
    selected_models: tuple[SelectedModel, ...]

    @property
    def has_force(self) -> bool:
        """Whether velocity has a nonzero configured derivative contribution."""

        return any(
            model.descriptor.category not in ("charge", "noise") for model in self.selected_models
        )

    @property
    def evolves_continuous_state(self) -> bool:
        """Whether a configured model evolves resident state beyond position and velocity."""

        return self.charge is not None

    @property
    def requires_stage_evaluation(self) -> bool:
        """Whether the full state must be reevaluated at integrator stages."""

        return self.has_force or self.evolves_continuous_state

    def resolved_models(self) -> dict[str, dict[str, str | float]]:
        """Project the actual selected descriptors and resolved scientific parameters."""
        result: dict[str, dict[str, str | float]] = {
            item.descriptor.category: {
                "model": item.descriptor.model,
                "revision": item.descriptor.revision,
            }
            for item in self.selected_models
        }
        if isinstance(self.thermophoresis, TalbotThermophoresisPlan):
            model = self.thermophoresis
            result["thermophoresis"].update(
                particle_thermal_conductivity_W_m_K=model.particle_thermal_conductivity_W_m_K,
                thermal_slip_coefficient=model.thermal_slip_coefficient,
                momentum_exchange_coefficient=model.momentum_exchange_coefficient,
                thermal_exchange_coefficient=model.thermal_exchange_coefficient,
            )
        if isinstance(self.lift, RarefiedVorticityLiftPlan):
            result["lift"]["lift_coefficient"] = self.lift.lift_coefficient
        if self.electric is not None and self.electric.maximum_abs_charge_number is not None:
            result["electric"]["maximum_abs_charge_number"] = (
                self.electric.maximum_abs_charge_number
            )
        return result


def compatible_integrators(plan: PhysicsPlan) -> tuple[str, ...]:
    """Resolve method availability for this already selected noise/model set.

    Field support, stiffness and path certificates remain prepare-time checks.
    This function does not select a method or loosen any applicability gate.
    """

    allowed = set(("ou_langevin",) if plan.noise is not None else _DETERMINISTIC_INTEGRATORS)
    for model in plan.selected_models:
        allowed.intersection_update(model.descriptor.integrators)
    return tuple(method for method in SUPPORTED_INTEGRATORS if method in allowed)


def resolve_physics_plan(
    models: Mapping[str, Mapping[str, object]], coordinate_system: CoordinateSystem
) -> PhysicsPlan:
    """Validate selected model revisions and return their static execution plan."""

    unsupported = set(models) - SUPPORTED_PHYSICS_CATEGORIES
    if unsupported:
        raise PhysicsConfigurationError(
            f"physics catalog does not support enabled categories: {sorted(unsupported)}"
        )
    descriptors = {
        category: _validate_model_structure(category, model, coordinate_system)
        for category, model in models.items()
    }

    if "charge" not in descriptors:
        descriptors["charge"] = _validate_model_structure(
            "charge", {"model": "fixed"}, coordinate_system
        )
    selected_models = tuple(
        SelectedModel(
            descriptors[category],
            tuple(
                (parameter, cast(str, models[category][parameter]))
                for parameter in descriptors[category].field_parameters
            ),
        )
        for category in (
            "charge",
            "drag",
            "noise",
            "thermophoresis",
            "ion_drag",
            "dielectrophoresis",
            "lift",
            "electric",
            "gravity_buoyancy",
        )
        if category in descriptors
    )
    charge = _resolve_charge(models.get("charge"))
    drag = _resolve_drag(models.get("drag"))
    noise = _resolve_noise(models.get("noise"))
    thermophoresis = _resolve_thermophoresis(models.get("thermophoresis"))
    ion_drag = _resolve_ion_drag(models.get("ion_drag"))
    _require_matching_ion_species(charge, ion_drag)
    dielectrophoresis = _resolve_dielectrophoresis(models.get("dielectrophoresis"))
    lift = _resolve_lift(models.get("lift"), coordinate_system)
    electric = _resolve_electric(models.get("electric"))
    _require_matching_electric_field(ion_drag, electric)
    gravity = _resolve_gravity(models.get("gravity_buoyancy"), coordinate_system)
    _require_matching_neutral_gas(drag, thermophoresis, lift, gravity)
    _require_noise_compatibility(noise, drag)
    _require_matching_reference_particles(charge, drag, noise, ion_drag, lift)
    requirements = [
        RequiredField(
            cast(str, models[category][role.parameter]),
            role.unit,
            role.components,
            role.stored_basis,
            role.value_constraint,
            role.zero_on_rz_axis,
        )
        for category, descriptor in descriptors.items()
        for role in _model_field_roles(descriptor, coordinate_system)
    ]
    return PhysicsPlan(
        charge,
        drag,
        noise,
        thermophoresis,
        ion_drag,
        dielectrophoresis,
        lift,
        electric,
        gravity,
        _deduplicate_requirements(requirements),
        selected_models,
    )


def _model_field_roles(
    descriptor: ModelDescriptor, coordinate_system: CoordinateSystem
) -> tuple[ModelFieldDescriptor, ...]:
    vector_components, vector_basis = _vector_metadata(coordinate_system)
    requirements = _model_field_requirements(descriptor, vector_components, vector_basis)
    return tuple(
        ModelFieldDescriptor(
            field.name,
            coordinate_system,
            field.unit,
            field.components,
            field.stored_basis,
            field.value_constraint,
            field.zero_on_rz_axis,
        )
        for field in requirements
    )


def _model_field_requirements(
    descriptor: ModelDescriptor, vector_components: tuple[str, str], vector_basis: str
) -> list[RequiredField]:
    """Define field roles once for discovery and validated selector binding."""

    if descriptor.category == "charge":
        return _charge_requirements(descriptor, vector_components, vector_basis)
    if descriptor.category == "drag":
        return _drag_requirements(descriptor, vector_components, vector_basis)
    if descriptor.category == "ion_drag":
        return _ion_drag_requirements(descriptor, vector_components, vector_basis)
    if descriptor.category == "lift":
        return _lift_requirements(descriptor, vector_components, vector_basis)
    if descriptor.category == "thermophoresis":
        return _thermophoresis_requirements(descriptor, vector_components, vector_basis)
    if descriptor.category == "dielectrophoresis":
        return [
            _vector_field(
                "gradient_mean_e_squared_field", "V^2/m^3", vector_components, vector_basis
            )
        ]
    if descriptor.category == "electric":
        return [_vector_field("electric_field", "V/m", vector_components, vector_basis)]
    if descriptor.category == "gravity_buoyancy":
        return [_scalar_field("gas_density_field", "kg/m^3", value_constraint="positive")]
    if descriptor.model == "prescribed_force_diffusion":
        return [_scalar_field("force_spectral_density_field", "N^2*s", value_constraint="positive")]
    return []


def _thermophoresis_requirements(
    descriptor: ModelDescriptor,
    vector_components: tuple[str, str],
    vector_basis: str,
) -> list[RequiredField]:
    requirements: list[RequiredField] = []
    if descriptor.model == "waldmann_gallis":
        requirements.extend(
            (
                _vector_field(
                    "gas_velocity_field",
                    "m/s",
                    vector_components,
                    vector_basis,
                ),
                _scalar_field(
                    "gas_temperature_field",
                    "K",
                    value_constraint="positive",
                ),
                _vector_field(
                    "gas_translational_heat_flux_field",
                    "W/m^2",
                    vector_components,
                    vector_basis,
                ),
                _scalar_field(
                    "gas_mean_free_path_field",
                    "m",
                    value_constraint="positive",
                ),
            )
        )
    elif descriptor.model == "talbot":
        requirements.extend(
            (
                _scalar_field("gas_temperature_field", "K", value_constraint="positive"),
                _vector_field(
                    "gas_temperature_gradient_field",
                    "K/m",
                    vector_components,
                    vector_basis,
                ),
                _scalar_field("gas_density_field", "kg/m^3", value_constraint="positive"),
                _scalar_field(
                    "gas_dynamic_viscosity_field",
                    "Pa*s",
                    value_constraint="positive",
                ),
                _scalar_field(
                    "gas_thermal_conductivity_field",
                    "W/(m*K)",
                    value_constraint="positive",
                ),
                _scalar_field(
                    "gas_mean_free_path_field",
                    "m",
                    value_constraint="positive",
                ),
            )
        )
    return requirements


def _drag_requirements(
    descriptor: ModelDescriptor,
    vector_components: tuple[str, str],
    vector_basis: str,
) -> list[RequiredField]:
    requirements: list[RequiredField] = []
    if descriptor.model == "prescribed_linear_friction":
        requirements.extend(
            (
                _vector_field("gas_velocity_field", "m/s", vector_components, vector_basis),
                _scalar_field("friction_coefficient_field", "kg/s", value_constraint="positive"),
            )
        )
    elif descriptor.model in ("epstein_linear", "epstein_finite_speed"):
        requirements.extend(
            (
                _vector_field(
                    "gas_velocity_field",
                    "m/s",
                    vector_components,
                    vector_basis,
                ),
                _scalar_field("gas_density_field", "kg/m^3", value_constraint="positive"),
                _scalar_field("gas_temperature_field", "K", value_constraint="positive"),
                _scalar_field("gas_mean_free_path_field", "m", value_constraint="positive"),
            )
        )
    elif descriptor.model == "stokes_cunningham":
        requirements.extend(
            (
                _vector_field(
                    "gas_velocity_field",
                    "m/s",
                    vector_components,
                    vector_basis,
                ),
                _scalar_field("gas_density_field", "kg/m^3", value_constraint="positive"),
                _scalar_field(
                    "gas_dynamic_viscosity_field",
                    "Pa*s",
                    value_constraint="positive",
                ),
                _scalar_field("gas_mean_free_path_field", "m", value_constraint="positive"),
            )
        )
    return requirements


def _ion_drag_requirements(
    descriptor: ModelDescriptor,
    vector_components: tuple[str, str],
    vector_basis: str,
) -> list[RequiredField]:
    requirements: list[RequiredField] = []
    if descriptor.model == "prescribed_charge_polynomial":
        requirements.extend(
            _vector_field(name, "N", vector_components, vector_basis)
            for name in (
                "constant_force_field",
                "linear_charge_force_field",
                "quadratic_charge_force_field",
            )
        )
    elif descriptor.model != "prescribed_charge_polynomial":
        if descriptor.model == "barnes_collisionless":
            requirements.extend(
                (
                    _scalar_field(
                        "electron_number_density_field",
                        "1/m^3",
                        value_constraint="positive",
                    ),
                    _scalar_field(
                        "positive_ion_number_density_field",
                        "1/m^3",
                        value_constraint="positive",
                    ),
                    _scalar_field(
                        "electron_temperature_field",
                        "K",
                        value_constraint="positive",
                    ),
                    _scalar_field(
                        "positive_ion_temperature_field",
                        "K",
                        value_constraint="positive",
                    ),
                    _vector_field(
                        "positive_ion_velocity_field",
                        "m/s",
                        vector_components,
                        vector_basis,
                    ),
                    _scalar_field(
                        "ion_neutral_mean_free_path_field",
                        "m",
                        value_constraint="positive",
                    ),
                )
            )
        else:
            requirements.extend(
                (
                    _scalar_field(
                        "positive_ion_number_density_field",
                        "1/m^3",
                        value_constraint="positive",
                    ),
                    _scalar_field(
                        "positive_ion_thermal_voltage_field",
                        "V",
                        value_constraint="positive",
                    ),
                    _vector_field(
                        "positive_ion_velocity_field",
                        "m/s",
                        vector_components,
                        vector_basis,
                    ),
                    _scalar_field(
                        "effective_positive_ion_mass_field",
                        "kg",
                        value_constraint="positive",
                    ),
                )
            )
            if descriptor.model == "screened_collection_orbital":
                requirements.extend(
                    (
                        _scalar_field(
                            "screening_length_field",
                            "m",
                            value_constraint="positive",
                        ),
                        _scalar_field(
                            "ion_neutral_mean_free_path_field",
                            "m",
                            value_constraint="positive",
                        ),
                    )
                )
            else:
                requirements.extend(
                    (
                        _scalar_field(
                            "electron_thermal_voltage_field",
                            "V",
                            value_constraint="positive",
                        ),
                        _scalar_field(
                            "screening_length_field",
                            "m",
                            value_constraint="positive",
                        ),
                        _vector_field(
                            "electric_field",
                            "V/m",
                            vector_components,
                            vector_basis,
                        ),
                    )
                )
    return requirements


def _lift_requirements(
    descriptor: ModelDescriptor,
    vector_components: tuple[str, str],
    vector_basis: str,
) -> list[RequiredField]:
    requirements: list[RequiredField] = []
    if descriptor.model == "prescribed_rotational_lift":
        requirements.extend(
            (
                _vector_field("gas_velocity_field", "m/s", vector_components, vector_basis),
                _scalar_field(
                    "rotation_coefficient_field",
                    "kg/s",
                    value_constraint="finite",
                    zero_on_rz_axis=vector_basis == "axisymmetric_rz",
                ),
            )
        )
    elif descriptor.model == "rarefied_vorticity_sensitivity":
        requirements.extend(
            (
                _vector_field(
                    "gas_velocity_field",
                    "m/s",
                    vector_components,
                    vector_basis,
                ),
                _scalar_field("gas_density_field", "kg/m^3", value_constraint="positive"),
                _scalar_field("gas_mean_free_path_field", "m", value_constraint="positive"),
                _scalar_field(
                    "azimuthal_gas_vorticity_field",
                    "1/s",
                    value_constraint="finite",
                    zero_on_rz_axis=True,
                ),
            )
        )
    elif descriptor.model == "saffman":
        requirements.extend(
            (
                _vector_field(
                    "gas_velocity_field",
                    "m/s",
                    vector_components,
                    vector_basis,
                ),
                _scalar_field("gas_density_field", "kg/m^3", value_constraint="positive"),
                _scalar_field(
                    "gas_dynamic_viscosity_field",
                    "Pa*s",
                    value_constraint="positive",
                ),
                _scalar_field("gas_mean_free_path_field", "m", value_constraint="positive"),
                _scalar_field(
                    "out_of_plane_gas_vorticity_field",
                    "1/s",
                    value_constraint="finite",
                    zero_on_rz_axis=vector_basis == "axisymmetric_rz",
                ),
            )
        )
    return requirements


def _charge_requirements(
    descriptor: ModelDescriptor,
    vector_components: tuple[str, str],
    vector_basis: str,
) -> list[RequiredField]:
    """Return canonical primitive fields owned by the selected charge revision."""

    if descriptor.model == "prescribed_exponential_linear":
        return [
            _scalar_field("electron_exponent_offset_field", "1", value_constraint="finite"),
            _scalar_field("electron_charge_slope_field", "1", value_constraint="nonnegative"),
            _scalar_field("ion_rate_field", "1/s", value_constraint="nonnegative"),
            _scalar_field("ion_charge_slope_field", "1", value_constraint="nonnegative"),
        ]
    if descriptor.revision in (_AGGREGATE_TWO_CURRENT_REVISION, _AGGREGATE_THREE_CURRENT_REVISION):
        requirements = [
            _scalar_field("electron_number_density_field", "1/m^3", value_constraint="positive"),
            _scalar_field(
                "positive_ion_number_density_field", "1/m^3", value_constraint="positive"
            ),
            _scalar_field("electron_thermal_voltage_field", "V", value_constraint="positive"),
            _scalar_field("positive_ion_thermal_voltage_field", "V", value_constraint="positive"),
            _vector_field(
                "positive_ion_velocity_field",
                "m/s",
                vector_components,
                vector_basis,
            ),
            _scalar_field("effective_positive_ion_mass_field", "kg", value_constraint="positive"),
            _scalar_field("screening_length_field", "m", value_constraint="positive"),
        ]
        if descriptor.revision == _AGGREGATE_THREE_CURRENT_REVISION:
            requirements.extend(
                (
                    _scalar_field(
                        "negative_ion_number_density_field",
                        "1/m^3",
                        value_constraint="nonnegative",
                    ),
                    _scalar_field(
                        "negative_ion_thermal_voltage_field",
                        "V",
                        value_constraint="positive",
                    ),
                    _vector_field(
                        "negative_ion_velocity_field",
                        "m/s",
                        vector_components,
                        vector_basis,
                    ),
                    _scalar_field(
                        "effective_negative_ion_mass_field",
                        "kg",
                        value_constraint="positive",
                    ),
                )
            )
        return requirements
    if descriptor.model == "fixed":
        return []
    return [
        _scalar_field("electron_number_density_field", "1/m^3", value_constraint="positive"),
        _scalar_field("positive_ion_number_density_field", "1/m^3", value_constraint="positive"),
        _scalar_field("electron_temperature_field", "K", value_constraint="positive"),
        _scalar_field("positive_ion_temperature_field", "K", value_constraint="positive"),
        _vector_field(
            "positive_ion_velocity_field",
            "m/s",
            vector_components,
            vector_basis,
        ),
    ]


def _resolve_noise(
    model: Mapping[str, object] | None,
) -> InertialLangevinNoisePlan | PrescribedForceDiffusionNoisePlan | None:
    if model is None:
        return None
    if model.get("model") == "prescribed_force_diffusion":
        return _resolve_prescribed_noise(model)
    if _text(model["model"], "physics.noise.model") != "inertial_langevin_fdt":
        raise PhysicsConfigurationError("physics.noise model must be inertial_langevin_fdt")
    revision = _text(model["revision"], "physics.noise.revision")
    if revision != BROWNIAN_MIDPOINT_2D_REVISION:
        raise PhysicsConfigurationError(
            f"inertial_langevin_fdt revision must be {BROWNIAN_MIDPOINT_2D_REVISION}"
        )
    interval_tree_depth = _bounded_integer(
        model["interval_tree_depth"], "physics.noise.interval_tree_depth", minimum=0, maximum=10
    )
    adaptive_max_depth = _bounded_integer(
        model.get("adaptive_max_depth", interval_tree_depth),
        "physics.noise.adaptive_max_depth",
        minimum=interval_tree_depth,
        maximum=10,
    )
    return InertialLangevinNoisePlan(
        revision=revision,
        interval_tree_depth=interval_tree_depth,
        adaptive_max_depth=adaptive_max_depth,
    )


def _require_noise_compatibility(
    noise: InertialLangevinNoisePlan | PrescribedForceDiffusionNoisePlan | None,
    drag: DragPlan | None,
) -> None:
    if noise is None:
        return
    if isinstance(noise, PrescribedForceDiffusionNoisePlan):
        if not isinstance(drag, PrescribedLinearFrictionPlan):
            raise PhysicsConfigurationError(
                "prescribed_force_diffusion requires prescribed_linear_friction"
            )
        return
    if not isinstance(drag, EpsteinDragPlan) or drag.revision not in {
        _EPSTEIN_LINEAR_REVISION,
        _EPSTEIN_LINEAR_EFFECTIVE_GAS_REVISION,
    }:
        raise PhysicsConfigurationError(
            "the 2-D midpoint inertial_langevin_fdt revision requires a supported "
            "linear Epstein drag revision"
        )


def _resolve_charge(model: Mapping[str, object] | None) -> ChargePlan | None:
    if model is None:
        raise PhysicsConfigurationError("physics.charge must select a charge model")
    model_id = _text(model.get("model"), "physics.charge.model")
    if model_id == "prescribed_exponential_linear":
        return _resolve_prescribed_charge(model)
    if model_id == "fixed":
        return None
    if model_id != "plasma_continuous":
        raise PhysicsConfigurationError("physics.charge model must be fixed or plasma_continuous")
    revision = _text(model.get("revision"), "physics.charge.revision")
    aggregate_revisions = {_AGGREGATE_TWO_CURRENT_REVISION, _AGGREGATE_THREE_CURRENT_REVISION}
    if revision in aggregate_revisions:
        has_negative_ion_current = revision == _AGGREGATE_THREE_CURRENT_REVISION
        if _text(model["applicability"], "physics.charge.applicability") != "error":
            raise PhysicsConfigurationError(
                "aggregate charge revisions support physics.charge applicability=error only"
            )
        return AggregateRelativeDriftChargePlan(
            revision=revision,
            electron_number_density_field=_name(
                model["electron_number_density_field"], "electron_number_density_field"
            ),
            positive_ion_number_density_field=_name(
                model["positive_ion_number_density_field"], "positive_ion_number_density_field"
            ),
            electron_thermal_voltage_field=_name(
                model["electron_thermal_voltage_field"], "electron_thermal_voltage_field"
            ),
            positive_ion_thermal_voltage_field=_name(
                model["positive_ion_thermal_voltage_field"], "positive_ion_thermal_voltage_field"
            ),
            positive_ion_velocity_field=_name(
                model["positive_ion_velocity_field"], "positive_ion_velocity_field"
            ),
            effective_positive_ion_mass_field=_name(
                model["effective_positive_ion_mass_field"], "effective_positive_ion_mass_field"
            ),
            screening_length_field=_name(model["screening_length_field"], "screening_length_field"),
            maximum_relative_ion_speed_m_s=_positive_number(
                model["maximum_relative_ion_speed_m_s"],
                "physics.charge.maximum_relative_ion_speed_m_s",
            ),
            negative_ion_number_density_field=_name(
                model["negative_ion_number_density_field"], "negative_ion_number_density_field"
            )
            if has_negative_ion_current
            else None,
            negative_ion_thermal_voltage_field=_name(
                model["negative_ion_thermal_voltage_field"], "negative_ion_thermal_voltage_field"
            )
            if has_negative_ion_current
            else None,
            negative_ion_velocity_field=_name(
                model["negative_ion_velocity_field"], "negative_ion_velocity_field"
            )
            if has_negative_ion_current
            else None,
            effective_negative_ion_mass_field=_name(
                model["effective_negative_ion_mass_field"], "effective_negative_ion_mass_field"
            )
            if has_negative_ion_current
            else None,
        )
    if revision == "oml_stationary_maxwellian_debye_huckel_v1":
        maximum_ion_drift_ratio = None
    elif revision == "oml_shifted_maxwellian_single_ion_negative_debye_huckel_v1":
        maximum_ion_drift_ratio = _positive_number(
            model["maximum_ion_drift_ratio"], "physics.charge.maximum_ion_drift_ratio"
        )
    else:
        raise PhysicsConfigurationError(
            "physics.charge revision must be oml_stationary_maxwellian_debye_huckel_v1 or oml_shifted_maxwellian_single_ion_negative_debye_huckel_v1 or aggregate_relative_drift_regularized_two_current_v1 or aggregate_relative_drift_regularized_three_current_v1"
        )
    if _text(model["applicability"], "physics.charge.applicability") != "error":
        raise PhysicsConfigurationError("physics.charge applicability must be error")
    return PlasmaContinuousChargePlan(
        revision=revision,
        electron_number_density_field=_name(
            model["electron_number_density_field"], "electron_number_density_field"
        ),
        positive_ion_number_density_field=_name(
            model["positive_ion_number_density_field"], "positive_ion_number_density_field"
        ),
        electron_temperature_field=_name(
            model["electron_temperature_field"], "electron_temperature_field"
        ),
        positive_ion_temperature_field=_name(
            model["positive_ion_temperature_field"], "positive_ion_temperature_field"
        ),
        positive_ion_velocity_field=_name(
            model["positive_ion_velocity_field"], "positive_ion_velocity_field"
        ),
        positive_ion_mass_kg=_positive_number(
            model["positive_ion_mass_kg"], "physics.charge.positive_ion_mass_kg"
        ),
        maximum_ion_drift_ratio=maximum_ion_drift_ratio,
    )


def _resolve_drag(model: Mapping[str, object] | None) -> DragPlan | None:
    if model is None:
        return None
    model_id = _text(model.get("model"), "physics.drag.model")
    if model_id == "prescribed_linear_friction":
        return _resolve_prescribed_drag(model)
    if model_id == "stokes_cunningham":
        return _resolve_stokes_cunningham_drag(model)
    if model_id == "epstein_finite_speed":
        return _resolve_finite_speed_epstein_drag(model)
    if model_id != "epstein_linear":
        raise PhysicsConfigurationError(
            "physics.drag model must be epstein_linear, epstein_finite_speed, or stokes_cunningham"
        )
    revision = _text(model.get("revision"), "physics.drag.revision")
    if revision == _EPSTEIN_LINEAR_REVISION:
        maximum_speed_ratio = EPSTEIN_MAX_SPEED_OVER_MEAN_THERMAL
    elif revision == _EPSTEIN_LINEAR_EFFECTIVE_GAS_REVISION:
        maximum_speed_ratio = _maximum_speed_ratio(
            model.get("maximum_speed_ratio"), "physics.drag.maximum_speed_ratio"
        )
    else:
        raise PhysicsConfigurationError(
            "epstein_linear revision must be epstein_linear_v1 or epstein_linear_effective_gas_sensitivity_v1"
        )
    if _text(model["applicability"], "physics.drag.applicability") != "error":
        raise PhysicsConfigurationError(
            "linear Epstein drag supports physics.drag applicability=error only"
        )
    delta = _positive_number(model["delta"], "physics.drag.delta")
    if not 1.0 <= delta <= 13.0 / 9.0:
        raise PhysicsConfigurationError(
            "physics.drag.delta must be in [1, 13/9] for linear Epstein drag"
        )
    return EpsteinDragPlan(
        revision=revision,
        gas_velocity_field=_name(model["gas_velocity_field"], "gas_velocity_field"),
        gas_density_field=_name(model["gas_density_field"], "gas_density_field"),
        gas_temperature_field=_name(model["gas_temperature_field"], "gas_temperature_field"),
        gas_mean_free_path_field=_name(
            model["gas_mean_free_path_field"], "gas_mean_free_path_field"
        ),
        gas_molecular_mass_kg=_positive_number(
            model["gas_molecular_mass_kg"], "physics.drag.gas_molecular_mass_kg"
        ),
        delta=delta,
        maximum_speed_ratio=maximum_speed_ratio,
    )


def _resolve_finite_speed_epstein_drag(model: Mapping[str, object]) -> FiniteSpeedEpsteinDragPlan:
    revision = _text(model["revision"], "physics.drag.revision")
    if revision != "epstein_finite_speed_maxwell_mixed_equal_temperature_v1":
        raise PhysicsConfigurationError(
            "epstein_finite_speed revision must be epstein_finite_speed_maxwell_mixed_equal_temperature_v1"
        )
    if _text(model["applicability"], "physics.drag.applicability") != "error":
        raise PhysicsConfigurationError(
            "finite-speed Epstein supports physics.drag applicability=error only"
        )
    return FiniteSpeedEpsteinDragPlan(
        gas_velocity_field=_name(model["gas_velocity_field"], "gas_velocity_field"),
        gas_density_field=_name(model["gas_density_field"], "gas_density_field"),
        gas_temperature_field=_name(model["gas_temperature_field"], "gas_temperature_field"),
        gas_mean_free_path_field=_name(
            model["gas_mean_free_path_field"], "gas_mean_free_path_field"
        ),
        gas_molecular_mass_kg=_positive_number(
            model["gas_molecular_mass_kg"], "physics.drag.gas_molecular_mass_kg"
        ),
        diffuse_reflection_fraction=_unit_interval_number(
            model["diffuse_reflection_fraction"], "physics.drag.diffuse_reflection_fraction"
        ),
        maximum_speed_ratio=_positive_number(
            model["maximum_speed_ratio"], "physics.drag.maximum_speed_ratio"
        ),
    )


def _resolve_stokes_cunningham_drag(model: Mapping[str, object]) -> StokesCunninghamDragPlan:
    if _text(model["revision"], "physics.drag.revision") != "stokes_cunningham_allen_raabe_air_v1":
        raise PhysicsConfigurationError(
            "stokes_cunningham revision must be stokes_cunningham_allen_raabe_air_v1"
        )
    if _text(model["applicability"], "physics.drag.applicability") != "error":
        raise PhysicsConfigurationError("physics.drag applicability must be error")
    return StokesCunninghamDragPlan(
        gas_velocity_field=_name(model["gas_velocity_field"], "gas_velocity_field"),
        gas_density_field=_name(model["gas_density_field"], "gas_density_field"),
        gas_dynamic_viscosity_field=_name(
            model["gas_dynamic_viscosity_field"], "gas_dynamic_viscosity_field"
        ),
        gas_mean_free_path_field=_name(
            model["gas_mean_free_path_field"], "gas_mean_free_path_field"
        ),
    )


def _resolve_ion_drag(model: Mapping[str, object] | None) -> IonDragPlan | None:
    if model is None:
        return None
    model_id = _text(model.get("model"), "physics.ion_drag.model")
    if model_id == "prescribed_charge_polynomial":
        return _resolve_prescribed_ion_drag(model)
    if model_id == "screened_collection_orbital":
        return _resolve_relative_flow_screened_ion_drag(model)
    if model_id == "image_orbital_sensitivity":
        return _resolve_electric_field_directed_image_ion_drag(model)
    if model_id != "barnes_collisionless":
        raise PhysicsConfigurationError(
            "physics.ion_drag model must be barnes_collisionless, screened_collection_orbital, or image_orbital_sensitivity"
        )
    revision = _text(model["revision"], "physics.ion_drag.revision")
    expected_revision = (
        "barnes_collisionless_effective_speed_single_positive_ion_negative_debye_huckel_v1"
    )
    if revision != expected_revision:
        raise PhysicsConfigurationError(
            f"barnes_collisionless revision must be {expected_revision}"
        )
    if _text(model["applicability"], "physics.ion_drag.applicability") != "error":
        raise PhysicsConfigurationError(
            "collisionless Barnes ion drag supports applicability=error only"
        )
    return BarnesCollisionlessIonDragPlan(
        electron_number_density_field=_name(
            model["electron_number_density_field"], "electron_number_density_field"
        ),
        positive_ion_number_density_field=_name(
            model["positive_ion_number_density_field"], "positive_ion_number_density_field"
        ),
        electron_temperature_field=_name(
            model["electron_temperature_field"], "electron_temperature_field"
        ),
        positive_ion_temperature_field=_name(
            model["positive_ion_temperature_field"], "positive_ion_temperature_field"
        ),
        positive_ion_velocity_field=_name(
            model["positive_ion_velocity_field"], "positive_ion_velocity_field"
        ),
        ion_neutral_mean_free_path_field=_name(
            model["ion_neutral_mean_free_path_field"], "ion_neutral_mean_free_path_field"
        ),
        positive_ion_mass_kg=_positive_number(
            model["positive_ion_mass_kg"], "physics.ion_drag.positive_ion_mass_kg"
        ),
        maximum_ion_drift_ratio=_positive_number(
            model["maximum_ion_drift_ratio"], "physics.ion_drag.maximum_ion_drift_ratio"
        ),
    )


def _resolve_relative_flow_screened_ion_drag(
    model: Mapping[str, object],
) -> RelativeFlowScreenedIonDragPlan:
    expected = "relative_flow_screened_collection_orbital_aggregate_ion_v1"
    if _text(model["revision"], "physics.ion_drag.revision") != expected:
        raise PhysicsConfigurationError(f"screened_collection_orbital revision must be {expected}")
    _require_error_applicability(model, "screened_collection_orbital")
    return RelativeFlowScreenedIonDragPlan(
        positive_ion_number_density_field=_name(
            model["positive_ion_number_density_field"], "positive_ion_number_density_field"
        ),
        positive_ion_thermal_voltage_field=_name(
            model["positive_ion_thermal_voltage_field"], "positive_ion_thermal_voltage_field"
        ),
        positive_ion_velocity_field=_name(
            model["positive_ion_velocity_field"], "positive_ion_velocity_field"
        ),
        effective_positive_ion_mass_field=_name(
            model["effective_positive_ion_mass_field"], "effective_positive_ion_mass_field"
        ),
        screening_length_field=_name(model["screening_length_field"], "screening_length_field"),
        ion_neutral_mean_free_path_field=_name(
            model["ion_neutral_mean_free_path_field"], "ion_neutral_mean_free_path_field"
        ),
        maximum_relative_ion_speed_m_s=_positive_number(
            model["maximum_relative_ion_speed_m_s"],
            "physics.ion_drag.maximum_relative_ion_speed_m_s",
        ),
    )


def _resolve_electric_field_directed_image_ion_drag(
    model: Mapping[str, object],
) -> ElectricFieldDirectedImageIonDragPlan:
    expected = "electric_field_directed_image_orbital_sensitivity_v1"
    if _text(model["revision"], "physics.ion_drag.revision") != expected:
        raise PhysicsConfigurationError(f"image_orbital_sensitivity revision must be {expected}")
    _require_error_applicability(model, "image_orbital_sensitivity")
    return ElectricFieldDirectedImageIonDragPlan(
        positive_ion_number_density_field=_name(
            model["positive_ion_number_density_field"], "positive_ion_number_density_field"
        ),
        electron_thermal_voltage_field=_name(
            model["electron_thermal_voltage_field"], "electron_thermal_voltage_field"
        ),
        positive_ion_thermal_voltage_field=_name(
            model["positive_ion_thermal_voltage_field"], "positive_ion_thermal_voltage_field"
        ),
        positive_ion_velocity_field=_name(
            model["positive_ion_velocity_field"], "positive_ion_velocity_field"
        ),
        effective_positive_ion_mass_field=_name(
            model["effective_positive_ion_mass_field"], "effective_positive_ion_mass_field"
        ),
        screening_length_field=_name(model["screening_length_field"], "screening_length_field"),
        electric_field=_name(model["electric_field"], "electric_field"),
    )


def _require_error_applicability(model: Mapping[str, object], model_id: str) -> None:
    if _text(model["applicability"], "physics.ion_drag.applicability") != "error":
        raise PhysicsConfigurationError(f"{model_id} ion drag supports applicability=error only")


def _resolve_thermophoresis(model: Mapping[str, object] | None) -> ThermophoresisPlan | None:
    if model is None:
        return None
    model_id = _text(model.get("model"), "physics.thermophoresis.model")
    if model_id == "talbot":
        return _resolve_talbot_thermophoresis(model)
    if model_id != "waldmann_gallis":
        raise PhysicsConfigurationError(
            "physics.thermophoresis model must be waldmann_gallis or talbot"
        )
    revision = _text(model.get("revision"), "physics.thermophoresis.revision")
    if revision == _WALDMANN_GALLIS_SINGLE_SPECIES_REVISION:
        maximum_speed_ratio = WALDMANN_GALLIS_MAX_SPEED_OVER_MEAN_THERMAL
    elif revision == _WALDMANN_GALLIS_EFFECTIVE_GAS_REVISION:
        maximum_speed_ratio = _maximum_speed_ratio(
            model.get("maximum_speed_ratio"), "physics.thermophoresis.maximum_speed_ratio"
        )
    else:
        raise PhysicsConfigurationError(
            "waldmann_gallis revision must be waldmann_gallis_free_molecular_single_species_heat_flux_v1 or waldmann_gallis_free_molecular_effective_gas_heat_flux_sensitivity_v1"
        )
    if _text(model["applicability"], "physics.thermophoresis.applicability") != "error":
        raise PhysicsConfigurationError(
            "Waldmann--Gallis thermophoresis supports applicability=error only"
        )
    return WaldmannGallisThermophoresisPlan(
        revision=revision,
        gas_velocity_field=_name(model["gas_velocity_field"], "gas_velocity_field"),
        gas_temperature_field=_name(model["gas_temperature_field"], "gas_temperature_field"),
        gas_translational_heat_flux_field=_name(
            model["gas_translational_heat_flux_field"], "gas_translational_heat_flux_field"
        ),
        gas_mean_free_path_field=_name(
            model["gas_mean_free_path_field"], "gas_mean_free_path_field"
        ),
        gas_molecular_mass_kg=_positive_number(
            model["gas_molecular_mass_kg"], "physics.thermophoresis.gas_molecular_mass_kg"
        ),
        maximum_speed_ratio=maximum_speed_ratio,
    )


def _resolve_talbot_thermophoresis(model: Mapping[str, object]) -> TalbotThermophoresisPlan:
    revision = _text(model["revision"], "physics.thermophoresis.revision")
    if revision != _TALBOT_REVISION:
        raise PhysicsConfigurationError(f"talbot revision must be {_TALBOT_REVISION}")
    if _text(model["applicability"], "physics.thermophoresis.applicability") != "error":
        raise PhysicsConfigurationError("Talbot thermophoresis supports applicability=error only")
    return TalbotThermophoresisPlan(
        gas_temperature_field=_name(model["gas_temperature_field"], "gas_temperature_field"),
        gas_temperature_gradient_field=_name(
            model["gas_temperature_gradient_field"], "gas_temperature_gradient_field"
        ),
        gas_density_field=_name(model["gas_density_field"], "gas_density_field"),
        gas_dynamic_viscosity_field=_name(
            model["gas_dynamic_viscosity_field"], "gas_dynamic_viscosity_field"
        ),
        gas_thermal_conductivity_field=_name(
            model["gas_thermal_conductivity_field"], "gas_thermal_conductivity_field"
        ),
        gas_mean_free_path_field=_name(
            model["gas_mean_free_path_field"], "gas_mean_free_path_field"
        ),
        particle_thermal_conductivity_W_m_K=_positive_number(
            model["particle_thermal_conductivity_W_m_K"],
            "physics.thermophoresis.particle_thermal_conductivity_W_m_K",
        ),
        thermal_slip_coefficient=_positive_number(
            model["thermal_slip_coefficient"], "physics.thermophoresis.thermal_slip_coefficient"
        ),
        momentum_exchange_coefficient=_positive_number(
            model["momentum_exchange_coefficient"],
            "physics.thermophoresis.momentum_exchange_coefficient",
        ),
        thermal_exchange_coefficient=_positive_number(
            model["thermal_exchange_coefficient"],
            "physics.thermophoresis.thermal_exchange_coefficient",
        ),
    )


def _require_matching_neutral_gas(
    drag: DragPlan | None,
    thermophoresis: ThermophoresisPlan | None,
    lift: LiftPlan | None,
    gravity: GravityBuoyancyPlan | None,
) -> None:
    _require_compatible_neutral_gas_models(drag, thermophoresis, lift)
    neutral_models = tuple(
        model for model in (drag, thermophoresis, lift, gravity) if model is not None
    )
    shared_attributes = (
        "gas_velocity_field",
        "gas_density_field",
        "gas_temperature_field",
        "gas_dynamic_viscosity_field",
        "gas_mean_free_path_field",
        "gas_molecular_mass_kg",
    )
    mismatched = [
        name
        for name in shared_attributes
        if len({getattr(model, name) for model in neutral_models if hasattr(model, name)}) > 1
    ]
    if mismatched:
        raise PhysicsConfigurationError(
            "enabled physics models must use the same neutral-gas background; "
            f"mismatched={mismatched}"
        )


def _require_compatible_neutral_gas_models(
    drag: DragPlan | None,
    thermophoresis: ThermophoresisPlan | None,
    lift: LiftPlan | None,
) -> None:
    if drag is not None and isinstance(thermophoresis, WaldmannGallisThermophoresisPlan):
        if isinstance(drag, StokesCunninghamDragPlan):
            raise PhysicsConfigurationError(
                "Waldmann--Gallis thermophoresis has no applicability overlap with "
                "stokes_cunningham drag"
            )
        if isinstance(drag, EpsteinDragPlan) and (
            (drag.revision == _EPSTEIN_LINEAR_EFFECTIVE_GAS_REVISION)
            != (thermophoresis.revision == _WALDMANN_GALLIS_EFFECTIVE_GAS_REVISION)
        ):
            raise PhysicsConfigurationError(
                "physics.drag and physics.thermophoresis cannot mix single-species and "
                "effective-gas revisions"
            )
    if isinstance(lift, RarefiedVorticityLiftPlan) and isinstance(drag, StokesCunninghamDragPlan):
        raise PhysicsConfigurationError(
            "rarefied-vorticity lift has no applicability overlap with stokes_cunningham drag"
        )
    if isinstance(lift, SaffmanLiftPlan):
        if drag is not None and not isinstance(drag, StokesCunninghamDragPlan):
            raise PhysicsConfigurationError(
                "Saffman lift composes only with stokes_cunningham drag or no drag"
            )
        if isinstance(thermophoresis, WaldmannGallisThermophoresisPlan):
            raise PhysicsConfigurationError(
                "Saffman lift has no applicability overlap with Waldmann--Gallis thermophoresis"
            )


def _require_matching_ion_species(
    charge: ChargePlan | None,
    ion_drag: IonDragPlan | None,
) -> None:
    if charge is None or ion_drag is None:
        return
    if isinstance(charge, PrescribedExponentialLinearChargePlan) or isinstance(
        ion_drag, PrescribedChargePolynomialIonDragPlan
    ):
        if not isinstance(charge, PrescribedExponentialLinearChargePlan) or not isinstance(
            ion_drag, PrescribedChargePolynomialIonDragPlan
        ):
            raise PhysicsConfigurationError(
                "prescribed charge and ion drag must use their declared reference pair or fixed charge"
            )
        return
    if isinstance(charge, AggregateRelativeDriftChargePlan):
        _require_matching_aggregate_ion_background(charge, ion_drag)
        return
    _require_matching_single_ion_background(charge, ion_drag)


def _require_matching_single_ion_background(
    charge: PlasmaContinuousChargePlan,
    ion_drag: IonDragPlan,
) -> None:
    if not isinstance(ion_drag, BarnesCollisionlessIonDragPlan):
        raise PhysicsConfigurationError(
            "aggregate-ion ion drag requires fixed charge or an aggregate charge revision"
        )
    shared = (
        ("electron_number_density_field", charge.electron_number_density_field),
        (
            "positive_ion_number_density_field",
            charge.positive_ion_number_density_field,
        ),
        ("electron_temperature_field", charge.electron_temperature_field),
        ("positive_ion_temperature_field", charge.positive_ion_temperature_field),
        ("positive_ion_velocity_field", charge.positive_ion_velocity_field),
        ("positive_ion_mass_kg", charge.positive_ion_mass_kg),
    )
    mismatched = [name for name, value in shared if getattr(ion_drag, name) != value]
    if mismatched:
        raise PhysicsConfigurationError(
            "physics.charge and physics.ion_drag must use the same single-ion "
            f"background; mismatched={mismatched}"
        )


def _require_matching_aggregate_ion_background(
    charge: AggregateRelativeDriftChargePlan,
    ion_drag: IonDragPlan,
) -> None:
    if isinstance(ion_drag, BarnesCollisionlessIonDragPlan | PrescribedChargePolynomialIonDragPlan):
        raise PhysicsConfigurationError(
            "aggregate charge revisions do not compose with barnes_collisionless ion drag"
        )
    shared = _aggregate_ion_background_fields(charge, ion_drag)
    mismatched = [name for name, actual, expected in shared if actual != expected]
    if mismatched:
        raise PhysicsConfigurationError(
            "aggregate charge and ion drag must use the same aggregate-ion "
            f"background; mismatched={mismatched}"
        )


def _aggregate_ion_background_fields(
    charge: AggregateRelativeDriftChargePlan,
    ion_drag: IonDragPlan,
) -> tuple[tuple[str, object, object], ...]:
    if isinstance(ion_drag, BarnesCollisionlessIonDragPlan | PrescribedChargePolynomialIonDragPlan):
        raise PhysicsConfigurationError("Barnes ion drag has no aggregate-ion background")
    shared: tuple[tuple[str, object, object], ...] = (
        (
            "positive_ion_number_density_field",
            ion_drag.positive_ion_number_density_field,
            charge.positive_ion_number_density_field,
        ),
        (
            "positive_ion_thermal_voltage_field",
            ion_drag.positive_ion_thermal_voltage_field,
            charge.positive_ion_thermal_voltage_field,
        ),
        (
            "positive_ion_velocity_field",
            ion_drag.positive_ion_velocity_field,
            charge.positive_ion_velocity_field,
        ),
        (
            "effective_positive_ion_mass_field",
            ion_drag.effective_positive_ion_mass_field,
            charge.effective_positive_ion_mass_field,
        ),
        (
            "screening_length_field",
            ion_drag.screening_length_field,
            charge.screening_length_field,
        ),
    )
    if isinstance(ion_drag, RelativeFlowScreenedIonDragPlan):
        shared += (
            (
                "maximum_relative_ion_speed_m_s",
                ion_drag.maximum_relative_ion_speed_m_s,
                charge.maximum_relative_ion_speed_m_s,
            ),
        )
    else:
        shared += (
            (
                "electron_thermal_voltage_field",
                ion_drag.electron_thermal_voltage_field,
                charge.electron_thermal_voltage_field,
            ),
        )
    return shared


def _require_matching_electric_field(
    ion_drag: IonDragPlan | None,
    electric: ElectricPlan | None,
) -> None:
    if not isinstance(ion_drag, ElectricFieldDirectedImageIonDragPlan) or electric is None:
        return
    if ion_drag.electric_field != electric.electric_field:
        raise PhysicsConfigurationError(
            "image ion drag and Coulomb electric force must use the same electric field"
        )


def _resolve_dielectrophoresis(
    model: Mapping[str, object] | None,
) -> QuasistaticSphericalDielectrophoresisPlan | None:
    if model is None:
        return None
    if _text(model["model"], "physics.dielectrophoresis.model") != "quasistatic_spherical":
        raise PhysicsConfigurationError(
            "physics.dielectrophoresis model must be quasistatic_spherical"
        )
    expected = "quasistatic_spherical_gradient_e2_v1"
    if _text(model["revision"], "physics.dielectrophoresis.revision") != expected:
        raise PhysicsConfigurationError(
            f"quasistatic_spherical dielectrophoresis revision must be {expected}"
        )
    factor_value = model["real_clausius_mossotti_factor"]
    if isinstance(factor_value, bool) or not isinstance(factor_value, int | float):
        raise PhysicsConfigurationError(
            "physics.dielectrophoresis.real_clausius_mossotti_factor must be a finite number in [-0.5, 1]"
        )
    try:
        factor = float(factor_value)
    except OverflowError as exc:
        raise PhysicsConfigurationError(
            "physics.dielectrophoresis.real_clausius_mossotti_factor must be a finite number in [-0.5, 1]"
        ) from exc
    if not math.isfinite(factor) or not -0.5 <= factor <= 1.0:
        raise PhysicsConfigurationError(
            "physics.dielectrophoresis.real_clausius_mossotti_factor must be a finite number in [-0.5, 1]"
        )
    return QuasistaticSphericalDielectrophoresisPlan(
        gradient_mean_e_squared_field=_name(
            model["gradient_mean_e_squared_field"], "gradient_mean_e_squared_field"
        ),
        medium_relative_permittivity=_positive_number(
            model["medium_relative_permittivity"],
            "physics.dielectrophoresis.medium_relative_permittivity",
        ),
        real_clausius_mossotti_factor=factor,
        maximum_point_dipole_radius_m=_positive_number(
            model["maximum_point_dipole_radius_m"],
            "physics.dielectrophoresis.maximum_point_dipole_radius_m",
        ),
    )


def _resolve_lift(
    model: Mapping[str, object] | None, coordinate_system: CoordinateSystem
) -> LiftPlan | None:
    if model is None:
        return None
    model_id = _text(model.get("model"), "physics.lift.model")
    if model_id == "prescribed_rotational_lift":
        return _resolve_prescribed_lift(model)
    if model_id == "saffman":
        return _resolve_saffman_lift(model, coordinate_system)
    if model_id != "rarefied_vorticity_sensitivity":
        raise PhysicsConfigurationError(
            "physics.lift model must be rarefied_vorticity_sensitivity or saffman"
        )
    expected = "rarefied_vorticity_sensitivity_rz_v1"
    if _text(model["revision"], "physics.lift.revision") != expected:
        raise PhysicsConfigurationError(
            f"rarefied_vorticity_sensitivity revision must be {expected}"
        )
    if _text(model["applicability"], "physics.lift.applicability") != "error":
        raise PhysicsConfigurationError(
            "rarefied-vorticity lift supports physics.lift applicability=error only"
        )
    return RarefiedVorticityLiftPlan(
        gas_velocity_field=_name(model["gas_velocity_field"], "gas_velocity_field"),
        gas_density_field=_name(model["gas_density_field"], "gas_density_field"),
        gas_mean_free_path_field=_name(
            model["gas_mean_free_path_field"], "gas_mean_free_path_field"
        ),
        azimuthal_gas_vorticity_field=_name(
            model["azimuthal_gas_vorticity_field"], "azimuthal_gas_vorticity_field"
        ),
        lift_coefficient=_positive_number(
            model["lift_coefficient"], "physics.lift.lift_coefficient"
        ),
    )


def _resolve_saffman_lift(
    model: Mapping[str, object], coordinate_system: CoordinateSystem
) -> SaffmanLiftPlan:
    revision = _text(model["revision"], "physics.lift.revision")
    if revision != _SAFFMAN_REVISION:
        raise PhysicsConfigurationError(f"saffman revision must be {_SAFFMAN_REVISION}")
    if _text(model["applicability"], "physics.lift.applicability") != "error":
        raise PhysicsConfigurationError("Saffman lift supports applicability=error only")
    return SaffmanLiftPlan(
        coordinate_system=coordinate_system,
        gas_velocity_field=_name(model["gas_velocity_field"], "gas_velocity_field"),
        gas_density_field=_name(model["gas_density_field"], "gas_density_field"),
        gas_dynamic_viscosity_field=_name(
            model["gas_dynamic_viscosity_field"], "gas_dynamic_viscosity_field"
        ),
        gas_mean_free_path_field=_name(
            model["gas_mean_free_path_field"], "gas_mean_free_path_field"
        ),
        out_of_plane_gas_vorticity_field=_name(
            model["out_of_plane_gas_vorticity_field"], "out_of_plane_gas_vorticity_field"
        ),
    )


def _resolve_electric(model: Mapping[str, object] | None) -> ElectricPlan | None:
    if model is None:
        return None
    revision = _text(model.get("revision"), "physics.electric.revision")
    if _text(model["model"], "physics.electric.model") != "coulomb":
        raise PhysicsConfigurationError("physics.electric model must be coulomb")
    if revision not in {"electric_coulomb_v1", "electric_coulomb_charge_cap_v1"}:
        raise PhysicsConfigurationError("unsupported Coulomb model revision")
    cap = (
        _positive_number(model["maximum_abs_charge_number"], "maximum_abs_charge_number")
        if revision == "electric_coulomb_charge_cap_v1"
        else None
    )
    return ElectricPlan(_name(model["electric_field"], "electric_field"), cap)


def _resolve_gravity(
    model: Mapping[str, object] | None, coordinate_system: CoordinateSystem
) -> GravityBuoyancyPlan | None:
    if model is None:
        return None
    if _text(model["model"], "physics.gravity_buoyancy.model") != "standard":
        raise PhysicsConfigurationError("physics.gravity_buoyancy model must be standard")
    if (
        _text(model["revision"], "physics.gravity_buoyancy.revision")
        != "gravity_buoyancy_standard_v1"
    ):
        raise PhysicsConfigurationError(
            "physics.gravity_buoyancy revision must be gravity_buoyancy_standard_v1"
        )
    gravity = _pair(model["gravity_m_s2"], "physics.gravity_buoyancy.gravity_m_s2")
    if coordinate_system == "axisymmetric_rz" and gravity[0] != 0.0:
        raise PhysicsConfigurationError(
            "axisymmetric_rz gravity_buoyancy_standard_v1 requires gravity_m_s2[0] = 0; a nonzero radial body acceleration is not a spatially uniform gravity vector"
        )
    return GravityBuoyancyPlan(_name(model["gas_density_field"], "gas_density_field"), gravity)


def _reference_particle_properties(value: object) -> ReferenceParticleProperties:
    if not isinstance(value, Mapping):
        raise PhysicsConfigurationError(
            "particle_properties must bind the reference coefficient authorities"
        )
    _exact_keys(
        value,
        {"mass_kg", "drag_diameter_m", "electrostatic_radius_m", "displaced_volume_m3"},
        "particle_properties",
    )
    return ReferenceParticleProperties(
        _positive_number(value["mass_kg"], "particle_properties.mass_kg"),
        _positive_number(value["drag_diameter_m"], "particle_properties.drag_diameter_m"),
        _nonnegative_number(
            value["electrostatic_radius_m"], "particle_properties.electrostatic_radius_m"
        ),
        _nonnegative_number(
            value["displaced_volume_m3"], "particle_properties.displaced_volume_m3"
        ),
    )


def _reference_model_properties(
    model: Mapping[str, object],
    revision: str,
) -> ReferenceParticleProperties:
    if model["revision"] != revision:
        raise PhysicsConfigurationError(f"{model['model']} revision must be {revision}")
    return _reference_particle_properties(model["particle_properties"])


def _require_matching_reference_particles(*models: object) -> None:
    bindings = {
        model.particle_properties
        for model in models
        if isinstance(
            model,
            PrescribedLinearFrictionPlan
            | PrescribedForceDiffusionNoisePlan
            | PrescribedExponentialLinearChargePlan
            | PrescribedChargePolynomialIonDragPlan
            | PrescribedRotationalLiftPlan,
        )
    }
    if len(bindings) > 1:
        raise PhysicsConfigurationError("reference models must bind the same particle_properties")


def _resolve_prescribed_drag(model: Mapping[str, object]) -> PrescribedLinearFrictionPlan:
    revision = "prescribed_linear_friction_v1"
    properties = _reference_model_properties(model, revision)
    return PrescribedLinearFrictionPlan(
        revision,
        _name(model["gas_velocity_field"], "gas_velocity_field"),
        _name(model["friction_coefficient_field"], "friction_coefficient_field"),
        properties,
    )


def _resolve_prescribed_noise(model: Mapping[str, object]) -> PrescribedForceDiffusionNoisePlan:
    revision = "prescribed_force_diffusion_midpoint_2d_v1"
    properties = _reference_model_properties(model, revision)
    depth = _bounded_integer(
        model["interval_tree_depth"], "interval_tree_depth", minimum=0, maximum=10
    )
    maximum = _bounded_integer(
        model.get("adaptive_max_depth", depth), "adaptive_max_depth", minimum=depth, maximum=10
    )
    return PrescribedForceDiffusionNoisePlan(
        revision,
        _name(model["force_spectral_density_field"], "force_spectral_density_field"),
        depth,
        maximum,
        properties,
    )


def _resolve_prescribed_charge(
    model: Mapping[str, object],
) -> PrescribedExponentialLinearChargePlan:
    revision = "prescribed_exponential_linear_collection_v1"
    properties = _reference_model_properties(model, revision)
    return PrescribedExponentialLinearChargePlan(
        revision,
        _nonnegative_number(
            model["electron_rate_reference_number_s"], "electron_rate_reference_number_s"
        ),
        _name(model["electron_exponent_offset_field"], "electron_exponent_offset_field"),
        _name(model["electron_charge_slope_field"], "electron_charge_slope_field"),
        _name(model["ion_rate_field"], "ion_rate_field"),
        _name(model["ion_charge_slope_field"], "ion_charge_slope_field"),
        properties,
    )


def _resolve_prescribed_ion_drag(
    model: Mapping[str, object],
) -> PrescribedChargePolynomialIonDragPlan:
    revision = "prescribed_charge_polynomial_v1"
    properties = _reference_model_properties(model, revision)
    return PrescribedChargePolynomialIonDragPlan(
        revision,
        _name(model["constant_force_field"], "constant_force_field"),
        _name(model["linear_charge_force_field"], "linear_charge_force_field"),
        _name(model["quadratic_charge_force_field"], "quadratic_charge_force_field"),
        properties,
    )


def _resolve_prescribed_lift(model: Mapping[str, object]) -> PrescribedRotationalLiftPlan:
    revision = "prescribed_rotational_lift_v1"
    properties = _reference_model_properties(model, revision)
    return PrescribedRotationalLiftPlan(
        revision,
        _name(model["gas_velocity_field"], "gas_velocity_field"),
        _name(model["rotation_coefficient_field"], "rotation_coefficient_field"),
        properties,
    )


def _scalar_field(
    name: str,
    unit: str,
    *,
    value_constraint: ValueConstraint,
    zero_on_rz_axis: bool = False,
) -> RequiredField:
    return RequiredField(
        name,
        unit,
        ("value",),
        "scalar",
        value_constraint,
        zero_on_rz_axis,
    )


def _vector_field(
    name: str,
    unit: str,
    components: tuple[str, str],
    stored_basis: str,
) -> RequiredField:
    return RequiredField(name, unit, components, stored_basis)


def _vector_metadata(coordinate_system: CoordinateSystem) -> tuple[tuple[str, str], str]:
    if coordinate_system == "cartesian_xy":
        return ("x", "y"), "cartesian_xy"
    if coordinate_system == "axisymmetric_rz":
        return ("r", "z"), "axisymmetric_rz"
    raise PhysicsConfigurationError(f"unsupported coordinate system {coordinate_system!r}")


def _deduplicate_requirements(items: list[RequiredField]) -> tuple[RequiredField, ...]:
    by_name: dict[str, RequiredField] = {}
    for item in items:
        previous = by_name.get(item.name)
        if previous is not None and previous != item:
            raise PhysicsConfigurationError(
                f"field {item.name!r} is selected for incompatible primitive quantities"
            )
        by_name[item.name] = item
    return tuple(by_name[name] for name in sorted(by_name))


def _exact_keys(value: Mapping[str, object], expected: set[str], location: str) -> None:
    keys = set(value)
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        raise PhysicsConfigurationError(
            f"{location} keys do not match the model revision; missing={missing}, extra={extra}"
        )


def _text(value: object, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise PhysicsConfigurationError(f"{location} must be a nonempty string")
    return value


def _name(value: object, parameter: str) -> str:
    return _text(value, f"physics parameter {parameter}")


def _positive_number(value: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise PhysicsConfigurationError(f"{location} must be a positive finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise PhysicsConfigurationError(f"{location} must be a positive finite number") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise PhysicsConfigurationError(f"{location} must be a positive finite number")
    return number


def _nonnegative_number(value: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise PhysicsConfigurationError(f"{location} must be a nonnegative finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise PhysicsConfigurationError(f"{location} must be a nonnegative finite number") from exc
    if not math.isfinite(result) or result < 0.0:
        raise PhysicsConfigurationError(f"{location} must be a nonnegative finite number")
    return result


def _maximum_speed_ratio(value: object, location: str) -> float:
    number = _positive_number(value, location)
    if number > 1.0:
        raise PhysicsConfigurationError(f"{location} must be at most 1.0")
    return number


def _unit_interval_number(value: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise PhysicsConfigurationError(f"{location} must be a finite number in [0, 1]")
    try:
        number = float(value)
    except OverflowError as exc:
        raise PhysicsConfigurationError(f"{location} must be a finite number in [0, 1]") from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise PhysicsConfigurationError(f"{location} must be a finite number in [0, 1]")
    return number


def _bounded_integer(
    value: object,
    location: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PhysicsConfigurationError(f"{location} must be an integer in [{minimum}, {maximum}]")
    if not minimum <= value <= maximum:
        raise PhysicsConfigurationError(f"{location} must be an integer in [{minimum}, {maximum}]")
    return value


def _pair(value: object, location: str) -> tuple[float, float]:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise PhysicsConfigurationError(f"{location} must contain exactly two numbers")
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise PhysicsConfigurationError(f"{location} must contain finite numbers")
        try:
            number = float(item)
        except OverflowError as exc:
            raise PhysicsConfigurationError(f"{location} must contain finite numbers") from exc
        if not math.isfinite(number):
            raise PhysicsConfigurationError(f"{location} must contain finite numbers")
        result.append(number)
    return result[0], result[1]
