"""Resolve the small, explicit physics catalog into one immutable plan."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

from .forces import (
    EPSTEIN_MAX_SPEED_OVER_MEAN_THERMAL,
    WALDMANN_GALLIS_MAX_SPEED_OVER_MEAN_THERMAL,
)

PHYSICS_CATALOG_REVISION = "inertial_langevin_rz_catalog_v17"

BROWNIAN_FROZEN_START_REVISION = "inertial_langevin_fdt_epstein_linear_frozen_start_v1"
BROWNIAN_RZ_MIDPOINT_REVISION = "inertial_langevin_fdt_epstein_linear_rz_meridional_projected_v1"

_EPSTEIN_LINEAR_REVISION = "epstein_linear_v1"
_EPSTEIN_LINEAR_EFFECTIVE_GAS_REVISION = "epstein_linear_effective_gas_sensitivity_v1"
_WALDMANN_GALLIS_SINGLE_SPECIES_REVISION = (
    "waldmann_gallis_free_molecular_single_species_heat_flux_v1"
)
_WALDMANN_GALLIS_EFFECTIVE_GAS_REVISION = (
    "waldmann_gallis_free_molecular_effective_gas_heat_flux_sensitivity_v1"
)
_AGGREGATE_TWO_CURRENT_REVISION = "aggregate_relative_drift_regularized_two_current_v1"
_AGGREGATE_THREE_CURRENT_REVISION = "aggregate_relative_drift_regularized_three_current_v1"

type CoordinateSystem = Literal["cartesian_xy", "axisymmetric_rz"]


class PhysicsConfigurationError(ValueError):
    """A selected model or one of its declared inputs is unsupported."""


@dataclass(frozen=True, slots=True)
class RequiredField:
    """The exact canonical metadata expected for one primitive field."""

    name: str
    unit: str
    components: tuple[str, ...]
    stored_basis: str
    positive: bool = False


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


type ChargePlan = PlasmaContinuousChargePlan | AggregateRelativeDriftChargePlan


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


type DragPlan = EpsteinDragPlan | FiniteSpeedEpsteinDragPlan | StokesCunninghamDragPlan


@dataclass(frozen=True, slots=True)
class InertialLangevinNoisePlan:
    """Resolved controls for the inertial Langevin FDT revision."""

    revision: str
    interval_tree_depth: int


@dataclass(frozen=True, slots=True)
class ElectricPlan:
    """Resolved field name for ``electric_coulomb_v1``."""

    electric_field: str


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
class PhysicsPlan:
    """One model per category, in the fixed contribution order."""

    charge: ChargePlan | None
    drag: DragPlan | None
    noise: InertialLangevinNoisePlan | None
    thermophoresis: WaldmannGallisThermophoresisPlan | None
    ion_drag: IonDragPlan | None
    dielectrophoresis: QuasistaticSphericalDielectrophoresisPlan | None
    lift: RarefiedVorticityLiftPlan | None
    electric: ElectricPlan | None
    gravity_buoyancy: GravityBuoyancyPlan | None
    required_fields: tuple[RequiredField, ...]

    @property
    def has_force(self) -> bool:
        """Whether velocity has a nonzero configured derivative contribution."""

        return any(
            model is not None
            for model in (
                self.drag,
                self.thermophoresis,
                self.ion_drag,
                self.dielectrophoresis,
                self.lift,
                self.electric,
                self.gravity_buoyancy,
            )
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
        """Return the small manifest representation in deterministic category order."""

        result: dict[str, dict[str, str | float]]
        if self.charge is None:
            result = {"charge": {"model": "fixed", "revision": "fixed_charge_v1"}}
        else:
            result = {
                "charge": {
                    "model": "plasma_continuous",
                    "revision": self.charge.revision,
                }
            }
        if isinstance(self.drag, EpsteinDragPlan):
            result["drag"] = {
                "model": "epstein_linear",
                "revision": self.drag.revision,
            }
        elif isinstance(self.drag, FiniteSpeedEpsteinDragPlan):
            result["drag"] = {
                "model": "epstein_finite_speed",
                "revision": "epstein_finite_speed_maxwell_mixed_equal_temperature_v1",
            }
        elif isinstance(self.drag, StokesCunninghamDragPlan):
            result["drag"] = {
                "model": "stokes_cunningham",
                "revision": "stokes_cunningham_allen_raabe_air_v1",
            }
        if self.noise is not None:
            result["noise"] = {
                "model": "inertial_langevin_fdt",
                "revision": self.noise.revision,
            }
        if self.thermophoresis is not None:
            result["thermophoresis"] = {
                "model": "waldmann_gallis",
                "revision": self.thermophoresis.revision,
            }
        if isinstance(self.ion_drag, BarnesCollisionlessIonDragPlan):
            result["ion_drag"] = {
                "model": "barnes_collisionless",
                "revision": (
                    "barnes_collisionless_effective_speed_single_positive_ion_"
                    "negative_debye_huckel_v1"
                ),
            }
        elif isinstance(self.ion_drag, RelativeFlowScreenedIonDragPlan):
            result["ion_drag"] = {
                "model": "screened_collection_orbital",
                "revision": "relative_flow_screened_collection_orbital_aggregate_ion_v1",
            }
        elif isinstance(self.ion_drag, ElectricFieldDirectedImageIonDragPlan):
            result["ion_drag"] = {
                "model": "image_orbital_sensitivity",
                "revision": "electric_field_directed_image_orbital_sensitivity_v1",
            }
        if self.dielectrophoresis is not None:
            result["dielectrophoresis"] = {
                "model": "quasistatic_spherical",
                "revision": "quasistatic_spherical_gradient_e2_v1",
            }
        if self.lift is not None:
            result["lift"] = {
                "model": "rarefied_vorticity_sensitivity",
                "revision": "rarefied_vorticity_sensitivity_rz_v1",
                "lift_coefficient": self.lift.lift_coefficient,
            }
        if self.electric is not None:
            result["electric"] = {
                "model": "coulomb",
                "revision": "electric_coulomb_v1",
            }
        if self.gravity_buoyancy is not None:
            result["gravity_buoyancy"] = {
                "model": "standard",
                "revision": "gravity_buoyancy_standard_v1",
            }
        return result


def resolve_physics_plan(
    models: Mapping[str, Mapping[str, object]], coordinate_system: CoordinateSystem
) -> PhysicsPlan:
    """Validate selected model revisions and return their static execution plan."""

    unsupported = set(models) - {
        "charge",
        "drag",
        "noise",
        "thermophoresis",
        "ion_drag",
        "dielectrophoresis",
        "lift",
        "electric",
        "gravity_buoyancy",
    }
    if unsupported:
        raise PhysicsConfigurationError(
            f"physics catalog does not support enabled categories: {sorted(unsupported)}"
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
    _require_noise_compatibility(
        noise,
        coordinate_system,
        charge,
        drag,
        thermophoresis,
        ion_drag,
        dielectrophoresis,
        lift,
        electric,
        gravity,
    )
    vector_components, vector_basis = _vector_metadata(coordinate_system)

    requirements = _charge_requirements(charge, vector_components, vector_basis)
    if isinstance(drag, EpsteinDragPlan | FiniteSpeedEpsteinDragPlan):
        requirements.extend(
            (
                _vector_field(
                    drag.gas_velocity_field,
                    "m/s",
                    vector_components,
                    vector_basis,
                ),
                _scalar_field(drag.gas_density_field, "kg/m^3", positive=True),
                _scalar_field(drag.gas_temperature_field, "K", positive=True),
                _scalar_field(drag.gas_mean_free_path_field, "m", positive=True),
            )
        )
    elif isinstance(drag, StokesCunninghamDragPlan):
        requirements.extend(
            (
                _vector_field(
                    drag.gas_velocity_field,
                    "m/s",
                    vector_components,
                    vector_basis,
                ),
                _scalar_field(drag.gas_density_field, "kg/m^3", positive=True),
                _scalar_field(
                    drag.gas_dynamic_viscosity_field,
                    "Pa*s",
                    positive=True,
                ),
                _scalar_field(drag.gas_mean_free_path_field, "m", positive=True),
            )
        )
    if thermophoresis is not None:
        requirements.extend(
            (
                _vector_field(
                    thermophoresis.gas_velocity_field,
                    "m/s",
                    vector_components,
                    vector_basis,
                ),
                _scalar_field(
                    thermophoresis.gas_temperature_field,
                    "K",
                    positive=True,
                ),
                _vector_field(
                    thermophoresis.gas_translational_heat_flux_field,
                    "W/m^2",
                    vector_components,
                    vector_basis,
                ),
                _scalar_field(
                    thermophoresis.gas_mean_free_path_field,
                    "m",
                    positive=True,
                ),
            )
        )
    if ion_drag is not None:
        if isinstance(ion_drag, BarnesCollisionlessIonDragPlan):
            requirements.extend(
                (
                    _scalar_field(
                        ion_drag.electron_number_density_field,
                        "1/m^3",
                        positive=True,
                    ),
                    _scalar_field(
                        ion_drag.positive_ion_number_density_field,
                        "1/m^3",
                        positive=True,
                    ),
                    _scalar_field(
                        ion_drag.electron_temperature_field,
                        "K",
                        positive=True,
                    ),
                    _scalar_field(
                        ion_drag.positive_ion_temperature_field,
                        "K",
                        positive=True,
                    ),
                    _vector_field(
                        ion_drag.positive_ion_velocity_field,
                        "m/s",
                        vector_components,
                        vector_basis,
                    ),
                    _scalar_field(
                        ion_drag.ion_neutral_mean_free_path_field,
                        "m",
                        positive=True,
                    ),
                )
            )
        else:
            requirements.extend(
                (
                    _scalar_field(
                        ion_drag.positive_ion_number_density_field,
                        "1/m^3",
                        positive=True,
                    ),
                    _scalar_field(
                        ion_drag.positive_ion_thermal_voltage_field,
                        "V",
                        positive=True,
                    ),
                    _vector_field(
                        ion_drag.positive_ion_velocity_field,
                        "m/s",
                        vector_components,
                        vector_basis,
                    ),
                    _scalar_field(
                        ion_drag.effective_positive_ion_mass_field,
                        "kg",
                        positive=True,
                    ),
                )
            )
            if isinstance(ion_drag, RelativeFlowScreenedIonDragPlan):
                requirements.extend(
                    (
                        _scalar_field(
                            ion_drag.screening_length_field,
                            "m",
                            positive=True,
                        ),
                        _scalar_field(
                            ion_drag.ion_neutral_mean_free_path_field,
                            "m",
                            positive=True,
                        ),
                    )
                )
            else:
                requirements.extend(
                    (
                        _scalar_field(
                            ion_drag.electron_thermal_voltage_field,
                            "V",
                            positive=True,
                        ),
                        _scalar_field(
                            ion_drag.screening_length_field,
                            "m",
                            positive=True,
                        ),
                        _vector_field(
                            ion_drag.electric_field,
                            "V/m",
                            vector_components,
                            vector_basis,
                        ),
                    )
                )
    if dielectrophoresis is not None:
        requirements.append(
            _vector_field(
                dielectrophoresis.gradient_mean_e_squared_field,
                "V^2/m^3",
                vector_components,
                vector_basis,
            )
        )
    if lift is not None:
        requirements.extend(
            (
                _vector_field(
                    lift.gas_velocity_field,
                    "m/s",
                    vector_components,
                    vector_basis,
                ),
                _scalar_field(lift.gas_density_field, "kg/m^3", positive=True),
                _scalar_field(lift.gas_mean_free_path_field, "m", positive=True),
                _scalar_field(
                    lift.azimuthal_gas_vorticity_field,
                    "1/s",
                    positive=False,
                ),
            )
        )
    if electric is not None:
        requirements.append(
            _vector_field(
                electric.electric_field,
                "V/m",
                vector_components,
                vector_basis,
            )
        )
    if gravity is not None:
        requirements.append(_scalar_field(gravity.gas_density_field, "kg/m^3", positive=True))
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
    )


def _charge_requirements(
    charge: ChargePlan | None,
    vector_components: tuple[str, str],
    vector_basis: str,
) -> list[RequiredField]:
    """Return canonical primitive fields owned by the selected charge revision."""

    if isinstance(charge, AggregateRelativeDriftChargePlan):
        requirements = [
            _scalar_field(charge.electron_number_density_field, "1/m^3", positive=True),
            _scalar_field(charge.positive_ion_number_density_field, "1/m^3", positive=True),
            _scalar_field(charge.electron_thermal_voltage_field, "V", positive=True),
            _scalar_field(charge.positive_ion_thermal_voltage_field, "V", positive=True),
            _vector_field(
                charge.positive_ion_velocity_field,
                "m/s",
                vector_components,
                vector_basis,
            ),
            _scalar_field(charge.effective_positive_ion_mass_field, "kg", positive=True),
            _scalar_field(charge.screening_length_field, "m", positive=True),
        ]
        if charge.has_negative_ion_current:
            requirements.extend(
                (
                    _scalar_field(
                        cast(str, charge.negative_ion_number_density_field),
                        "1/m^3",
                        positive=False,
                    ),
                    _scalar_field(
                        cast(str, charge.negative_ion_thermal_voltage_field),
                        "V",
                        positive=True,
                    ),
                    _vector_field(
                        cast(str, charge.negative_ion_velocity_field),
                        "m/s",
                        vector_components,
                        vector_basis,
                    ),
                    _scalar_field(
                        cast(str, charge.effective_negative_ion_mass_field),
                        "kg",
                        positive=True,
                    ),
                )
            )
        return requirements
    if charge is None:
        return []
    return [
        _scalar_field(charge.electron_number_density_field, "1/m^3", positive=True),
        _scalar_field(charge.positive_ion_number_density_field, "1/m^3", positive=True),
        _scalar_field(charge.electron_temperature_field, "K", positive=True),
        _scalar_field(charge.positive_ion_temperature_field, "K", positive=True),
        _vector_field(
            charge.positive_ion_velocity_field,
            "m/s",
            vector_components,
            vector_basis,
        ),
    ]


def _resolve_noise(
    model: Mapping[str, object] | None,
) -> InertialLangevinNoisePlan | None:
    if model is None:
        return None
    _exact_keys(
        model,
        {"model", "revision", "interval_tree_depth"},
        "physics.noise",
    )
    if _text(model["model"], "physics.noise.model") != "inertial_langevin_fdt":
        raise PhysicsConfigurationError("physics.noise model must be inertial_langevin_fdt")
    revision = _text(model["revision"], "physics.noise.revision")
    supported_revisions = {
        BROWNIAN_FROZEN_START_REVISION,
        BROWNIAN_RZ_MIDPOINT_REVISION,
    }
    if revision not in supported_revisions:
        raise PhysicsConfigurationError(
            f"inertial_langevin_fdt revision must be one of {sorted(supported_revisions)}"
        )
    return InertialLangevinNoisePlan(
        revision=revision,
        interval_tree_depth=_bounded_integer(
            model["interval_tree_depth"],
            "physics.noise.interval_tree_depth",
            minimum=0,
            maximum=10,
        ),
    )


def _require_noise_compatibility(
    noise: InertialLangevinNoisePlan | None,
    coordinate_system: CoordinateSystem,
    charge: ChargePlan | None,
    drag: DragPlan | None,
    thermophoresis: WaldmannGallisThermophoresisPlan | None,
    ion_drag: IonDragPlan | None,
    dielectrophoresis: QuasistaticSphericalDielectrophoresisPlan | None,
    lift: RarefiedVorticityLiftPlan | None,
    electric: ElectricPlan | None,
    gravity: GravityBuoyancyPlan | None,
) -> None:
    if noise is None:
        return
    if noise.revision == BROWNIAN_FROZEN_START_REVISION:
        if coordinate_system != "cartesian_xy":
            raise PhysicsConfigurationError(
                "the frozen-start inertial_langevin_fdt revision supports cartesian_xy motion only"
            )
        if charge is not None:
            raise PhysicsConfigurationError(
                "the frozen-start inertial_langevin_fdt revision requires fixed charge"
            )
        if not isinstance(drag, EpsteinDragPlan) or drag.revision != _EPSTEIN_LINEAR_REVISION:
            raise PhysicsConfigurationError(
                "the frozen-start inertial_langevin_fdt revision requires epstein_linear_v1 drag"
            )
        if any(
            model is not None
            for model in (
                thermophoresis,
                ion_drag,
                dielectrophoresis,
                lift,
                electric,
                gravity,
            )
        ):
            raise PhysicsConfigurationError(
                "the frozen-start inertial_langevin_fdt revision does not compose "
                "with additional force models"
            )
        return
    if coordinate_system != "axisymmetric_rz":
        raise PhysicsConfigurationError(
            "the RZ projected inertial_langevin_fdt revision requires axisymmetric_rz motion"
        )
    if not isinstance(drag, EpsteinDragPlan) or drag.revision not in {
        _EPSTEIN_LINEAR_REVISION,
        _EPSTEIN_LINEAR_EFFECTIVE_GAS_REVISION,
    }:
        raise PhysicsConfigurationError(
            "the RZ projected inertial_langevin_fdt revision requires a supported "
            "linear Epstein drag revision"
        )


def _resolve_charge(
    model: Mapping[str, object] | None,
) -> ChargePlan | None:
    if model is None:
        raise PhysicsConfigurationError("physics.charge must select a charge model")
    model_id = _text(model.get("model"), "physics.charge.model")
    if model_id == "fixed":
        _exact_keys(model, {"model"}, "physics.charge")
        return None
    if model_id != "plasma_continuous":
        raise PhysicsConfigurationError("physics.charge model must be fixed or plasma_continuous")
    base = {
        "model",
        "revision",
        "applicability",
    }
    revision = _text(model.get("revision"), "physics.charge.revision")
    aggregate_revisions = {
        _AGGREGATE_TWO_CURRENT_REVISION,
        _AGGREGATE_THREE_CURRENT_REVISION,
    }
    if revision in aggregate_revisions:
        has_negative_ion_current = revision == _AGGREGATE_THREE_CURRENT_REVISION
        aggregate = base | {
            "electron_number_density_field",
            "positive_ion_number_density_field",
            "electron_thermal_voltage_field",
            "positive_ion_thermal_voltage_field",
            "positive_ion_velocity_field",
            "effective_positive_ion_mass_field",
            "screening_length_field",
            "maximum_relative_ion_speed_m_s",
        }
        if has_negative_ion_current:
            aggregate |= {
                "negative_ion_number_density_field",
                "negative_ion_thermal_voltage_field",
                "negative_ion_velocity_field",
                "effective_negative_ion_mass_field",
            }
        _exact_keys(model, aggregate, "physics.charge")
        if _text(model["applicability"], "physics.charge.applicability") != "error":
            raise PhysicsConfigurationError(
                "aggregate charge revisions support physics.charge applicability=error only"
            )
        return AggregateRelativeDriftChargePlan(
            revision=revision,
            electron_number_density_field=_name(
                model["electron_number_density_field"],
                "electron_number_density_field",
            ),
            positive_ion_number_density_field=_name(
                model["positive_ion_number_density_field"],
                "positive_ion_number_density_field",
            ),
            electron_thermal_voltage_field=_name(
                model["electron_thermal_voltage_field"],
                "electron_thermal_voltage_field",
            ),
            positive_ion_thermal_voltage_field=_name(
                model["positive_ion_thermal_voltage_field"],
                "positive_ion_thermal_voltage_field",
            ),
            positive_ion_velocity_field=_name(
                model["positive_ion_velocity_field"],
                "positive_ion_velocity_field",
            ),
            effective_positive_ion_mass_field=_name(
                model["effective_positive_ion_mass_field"],
                "effective_positive_ion_mass_field",
            ),
            screening_length_field=_name(
                model["screening_length_field"],
                "screening_length_field",
            ),
            maximum_relative_ion_speed_m_s=_positive_number(
                model["maximum_relative_ion_speed_m_s"],
                "physics.charge.maximum_relative_ion_speed_m_s",
            ),
            negative_ion_number_density_field=(
                _name(
                    model["negative_ion_number_density_field"],
                    "negative_ion_number_density_field",
                )
                if has_negative_ion_current
                else None
            ),
            negative_ion_thermal_voltage_field=(
                _name(
                    model["negative_ion_thermal_voltage_field"],
                    "negative_ion_thermal_voltage_field",
                )
                if has_negative_ion_current
                else None
            ),
            negative_ion_velocity_field=(
                _name(
                    model["negative_ion_velocity_field"],
                    "negative_ion_velocity_field",
                )
                if has_negative_ion_current
                else None
            ),
            effective_negative_ion_mass_field=(
                _name(
                    model["effective_negative_ion_mass_field"],
                    "effective_negative_ion_mass_field",
                )
                if has_negative_ion_current
                else None
            ),
        )
    common = base | {
        "electron_number_density_field",
        "positive_ion_number_density_field",
        "electron_temperature_field",
        "positive_ion_temperature_field",
        "positive_ion_velocity_field",
        "positive_ion_mass_kg",
    }
    if revision == "oml_stationary_maxwellian_debye_huckel_v1":
        _exact_keys(model, common, "physics.charge")
        maximum_ion_drift_ratio = None
    elif revision == "oml_shifted_maxwellian_single_ion_negative_debye_huckel_v1":
        _exact_keys(model, common | {"maximum_ion_drift_ratio"}, "physics.charge")
        maximum_ion_drift_ratio = _positive_number(
            model["maximum_ion_drift_ratio"],
            "physics.charge.maximum_ion_drift_ratio",
        )
    else:
        raise PhysicsConfigurationError(
            "physics.charge revision must be "
            "oml_stationary_maxwellian_debye_huckel_v1 or "
            "oml_shifted_maxwellian_single_ion_negative_debye_huckel_v1 or "
            "aggregate_relative_drift_regularized_two_current_v1 or "
            "aggregate_relative_drift_regularized_three_current_v1"
        )
    if _text(model["applicability"], "physics.charge.applicability") != "error":
        raise PhysicsConfigurationError("P15 supports physics.charge applicability=error only")
    return PlasmaContinuousChargePlan(
        revision=revision,
        electron_number_density_field=_name(
            model["electron_number_density_field"],
            "electron_number_density_field",
        ),
        positive_ion_number_density_field=_name(
            model["positive_ion_number_density_field"],
            "positive_ion_number_density_field",
        ),
        electron_temperature_field=_name(
            model["electron_temperature_field"],
            "electron_temperature_field",
        ),
        positive_ion_temperature_field=_name(
            model["positive_ion_temperature_field"],
            "positive_ion_temperature_field",
        ),
        positive_ion_velocity_field=_name(
            model["positive_ion_velocity_field"],
            "positive_ion_velocity_field",
        ),
        positive_ion_mass_kg=_positive_number(
            model["positive_ion_mass_kg"],
            "physics.charge.positive_ion_mass_kg",
        ),
        maximum_ion_drift_ratio=maximum_ion_drift_ratio,
    )


def _resolve_drag(model: Mapping[str, object] | None) -> DragPlan | None:
    if model is None:
        return None
    model_id = _text(model.get("model"), "physics.drag.model")
    if model_id == "stokes_cunningham":
        return _resolve_stokes_cunningham_drag(model)
    if model_id == "epstein_finite_speed":
        return _resolve_finite_speed_epstein_drag(model)
    if model_id != "epstein_linear":
        raise PhysicsConfigurationError(
            "physics.drag model must be epstein_linear, epstein_finite_speed, or stokes_cunningham"
        )
    required = {
        "model",
        "revision",
        "gas_velocity_field",
        "gas_density_field",
        "gas_temperature_field",
        "gas_mean_free_path_field",
        "gas_molecular_mass_kg",
        "delta",
        "applicability",
    }
    revision = _text(model.get("revision"), "physics.drag.revision")
    if revision == _EPSTEIN_LINEAR_REVISION:
        maximum_speed_ratio = EPSTEIN_MAX_SPEED_OVER_MEAN_THERMAL
    elif revision == _EPSTEIN_LINEAR_EFFECTIVE_GAS_REVISION:
        required.add("maximum_speed_ratio")
        maximum_speed_ratio = _maximum_speed_ratio(
            model.get("maximum_speed_ratio"),
            "physics.drag.maximum_speed_ratio",
        )
    else:
        raise PhysicsConfigurationError(
            "epstein_linear revision must be epstein_linear_v1 or "
            "epstein_linear_effective_gas_sensitivity_v1"
        )
    _exact_keys(model, required, "physics.drag")
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


def _resolve_finite_speed_epstein_drag(
    model: Mapping[str, object],
) -> FiniteSpeedEpsteinDragPlan:
    required = {
        "model",
        "revision",
        "gas_velocity_field",
        "gas_density_field",
        "gas_temperature_field",
        "gas_mean_free_path_field",
        "gas_molecular_mass_kg",
        "diffuse_reflection_fraction",
        "maximum_speed_ratio",
        "applicability",
    }
    _exact_keys(model, required, "physics.drag")
    revision = _text(model["revision"], "physics.drag.revision")
    if revision != "epstein_finite_speed_maxwell_mixed_equal_temperature_v1":
        raise PhysicsConfigurationError(
            "epstein_finite_speed revision must be "
            "epstein_finite_speed_maxwell_mixed_equal_temperature_v1"
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
            model["gas_mean_free_path_field"],
            "gas_mean_free_path_field",
        ),
        gas_molecular_mass_kg=_positive_number(
            model["gas_molecular_mass_kg"],
            "physics.drag.gas_molecular_mass_kg",
        ),
        diffuse_reflection_fraction=_unit_interval_number(
            model["diffuse_reflection_fraction"],
            "physics.drag.diffuse_reflection_fraction",
        ),
        maximum_speed_ratio=_positive_number(
            model["maximum_speed_ratio"],
            "physics.drag.maximum_speed_ratio",
        ),
    )


def _resolve_stokes_cunningham_drag(
    model: Mapping[str, object],
) -> StokesCunninghamDragPlan:
    required = {
        "model",
        "revision",
        "gas_velocity_field",
        "gas_density_field",
        "gas_dynamic_viscosity_field",
        "gas_mean_free_path_field",
        "applicability",
    }
    _exact_keys(model, required, "physics.drag")
    if _text(model["revision"], "physics.drag.revision") != "stokes_cunningham_allen_raabe_air_v1":
        raise PhysicsConfigurationError(
            "stokes_cunningham revision must be stokes_cunningham_allen_raabe_air_v1"
        )
    if _text(model["applicability"], "physics.drag.applicability") != "error":
        raise PhysicsConfigurationError("P06-S supports physics.drag applicability=error only")
    return StokesCunninghamDragPlan(
        gas_velocity_field=_name(model["gas_velocity_field"], "gas_velocity_field"),
        gas_density_field=_name(model["gas_density_field"], "gas_density_field"),
        gas_dynamic_viscosity_field=_name(
            model["gas_dynamic_viscosity_field"],
            "gas_dynamic_viscosity_field",
        ),
        gas_mean_free_path_field=_name(
            model["gas_mean_free_path_field"],
            "gas_mean_free_path_field",
        ),
    )


def _resolve_ion_drag(
    model: Mapping[str, object] | None,
) -> IonDragPlan | None:
    if model is None:
        return None
    model_id = _text(model.get("model"), "physics.ion_drag.model")
    if model_id == "screened_collection_orbital":
        return _resolve_relative_flow_screened_ion_drag(model)
    if model_id == "image_orbital_sensitivity":
        return _resolve_electric_field_directed_image_ion_drag(model)
    if model_id != "barnes_collisionless":
        raise PhysicsConfigurationError(
            "physics.ion_drag model must be barnes_collisionless, "
            "screened_collection_orbital, or image_orbital_sensitivity"
        )
    required = {
        "model",
        "revision",
        "electron_number_density_field",
        "positive_ion_number_density_field",
        "electron_temperature_field",
        "positive_ion_temperature_field",
        "positive_ion_velocity_field",
        "ion_neutral_mean_free_path_field",
        "positive_ion_mass_kg",
        "maximum_ion_drift_ratio",
        "applicability",
    }
    _exact_keys(model, required, "physics.ion_drag")
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
            model["electron_number_density_field"],
            "electron_number_density_field",
        ),
        positive_ion_number_density_field=_name(
            model["positive_ion_number_density_field"],
            "positive_ion_number_density_field",
        ),
        electron_temperature_field=_name(
            model["electron_temperature_field"],
            "electron_temperature_field",
        ),
        positive_ion_temperature_field=_name(
            model["positive_ion_temperature_field"],
            "positive_ion_temperature_field",
        ),
        positive_ion_velocity_field=_name(
            model["positive_ion_velocity_field"],
            "positive_ion_velocity_field",
        ),
        ion_neutral_mean_free_path_field=_name(
            model["ion_neutral_mean_free_path_field"],
            "ion_neutral_mean_free_path_field",
        ),
        positive_ion_mass_kg=_positive_number(
            model["positive_ion_mass_kg"],
            "physics.ion_drag.positive_ion_mass_kg",
        ),
        maximum_ion_drift_ratio=_positive_number(
            model["maximum_ion_drift_ratio"],
            "physics.ion_drag.maximum_ion_drift_ratio",
        ),
    )


def _resolve_relative_flow_screened_ion_drag(
    model: Mapping[str, object],
) -> RelativeFlowScreenedIonDragPlan:
    required = {
        "model",
        "revision",
        "positive_ion_number_density_field",
        "positive_ion_thermal_voltage_field",
        "positive_ion_velocity_field",
        "effective_positive_ion_mass_field",
        "screening_length_field",
        "ion_neutral_mean_free_path_field",
        "maximum_relative_ion_speed_m_s",
        "applicability",
    }
    _exact_keys(model, required, "physics.ion_drag")
    expected = "relative_flow_screened_collection_orbital_aggregate_ion_v1"
    if _text(model["revision"], "physics.ion_drag.revision") != expected:
        raise PhysicsConfigurationError(f"screened_collection_orbital revision must be {expected}")
    _require_error_applicability(model, "screened_collection_orbital")
    return RelativeFlowScreenedIonDragPlan(
        positive_ion_number_density_field=_name(
            model["positive_ion_number_density_field"],
            "positive_ion_number_density_field",
        ),
        positive_ion_thermal_voltage_field=_name(
            model["positive_ion_thermal_voltage_field"],
            "positive_ion_thermal_voltage_field",
        ),
        positive_ion_velocity_field=_name(
            model["positive_ion_velocity_field"],
            "positive_ion_velocity_field",
        ),
        effective_positive_ion_mass_field=_name(
            model["effective_positive_ion_mass_field"],
            "effective_positive_ion_mass_field",
        ),
        screening_length_field=_name(
            model["screening_length_field"],
            "screening_length_field",
        ),
        ion_neutral_mean_free_path_field=_name(
            model["ion_neutral_mean_free_path_field"],
            "ion_neutral_mean_free_path_field",
        ),
        maximum_relative_ion_speed_m_s=_positive_number(
            model["maximum_relative_ion_speed_m_s"],
            "physics.ion_drag.maximum_relative_ion_speed_m_s",
        ),
    )


def _resolve_electric_field_directed_image_ion_drag(
    model: Mapping[str, object],
) -> ElectricFieldDirectedImageIonDragPlan:
    required = {
        "model",
        "revision",
        "positive_ion_number_density_field",
        "electron_thermal_voltage_field",
        "positive_ion_thermal_voltage_field",
        "positive_ion_velocity_field",
        "effective_positive_ion_mass_field",
        "screening_length_field",
        "electric_field",
        "applicability",
    }
    _exact_keys(model, required, "physics.ion_drag")
    expected = "electric_field_directed_image_orbital_sensitivity_v1"
    if _text(model["revision"], "physics.ion_drag.revision") != expected:
        raise PhysicsConfigurationError(f"image_orbital_sensitivity revision must be {expected}")
    _require_error_applicability(model, "image_orbital_sensitivity")
    return ElectricFieldDirectedImageIonDragPlan(
        positive_ion_number_density_field=_name(
            model["positive_ion_number_density_field"],
            "positive_ion_number_density_field",
        ),
        electron_thermal_voltage_field=_name(
            model["electron_thermal_voltage_field"],
            "electron_thermal_voltage_field",
        ),
        positive_ion_thermal_voltage_field=_name(
            model["positive_ion_thermal_voltage_field"],
            "positive_ion_thermal_voltage_field",
        ),
        positive_ion_velocity_field=_name(
            model["positive_ion_velocity_field"],
            "positive_ion_velocity_field",
        ),
        effective_positive_ion_mass_field=_name(
            model["effective_positive_ion_mass_field"],
            "effective_positive_ion_mass_field",
        ),
        screening_length_field=_name(
            model["screening_length_field"],
            "screening_length_field",
        ),
        electric_field=_name(model["electric_field"], "electric_field"),
    )


def _require_error_applicability(model: Mapping[str, object], model_id: str) -> None:
    if _text(model["applicability"], "physics.ion_drag.applicability") != "error":
        raise PhysicsConfigurationError(f"{model_id} ion drag supports applicability=error only")


def _resolve_thermophoresis(
    model: Mapping[str, object] | None,
) -> WaldmannGallisThermophoresisPlan | None:
    if model is None:
        return None
    required = {
        "model",
        "revision",
        "gas_velocity_field",
        "gas_temperature_field",
        "gas_translational_heat_flux_field",
        "gas_mean_free_path_field",
        "gas_molecular_mass_kg",
        "applicability",
    }
    revision = _text(model.get("revision"), "physics.thermophoresis.revision")
    if revision == _WALDMANN_GALLIS_SINGLE_SPECIES_REVISION:
        maximum_speed_ratio = WALDMANN_GALLIS_MAX_SPEED_OVER_MEAN_THERMAL
    elif revision == _WALDMANN_GALLIS_EFFECTIVE_GAS_REVISION:
        required.add("maximum_speed_ratio")
        maximum_speed_ratio = _maximum_speed_ratio(
            model.get("maximum_speed_ratio"),
            "physics.thermophoresis.maximum_speed_ratio",
        )
    else:
        raise PhysicsConfigurationError(
            "waldmann_gallis revision must be "
            "waldmann_gallis_free_molecular_single_species_heat_flux_v1 or "
            "waldmann_gallis_free_molecular_effective_gas_heat_flux_sensitivity_v1"
        )
    _exact_keys(model, required, "physics.thermophoresis")
    if _text(model["model"], "physics.thermophoresis.model") != "waldmann_gallis":
        raise PhysicsConfigurationError("physics.thermophoresis model must be waldmann_gallis")
    if _text(model["applicability"], "physics.thermophoresis.applicability") != "error":
        raise PhysicsConfigurationError(
            "Waldmann--Gallis thermophoresis supports applicability=error only"
        )
    return WaldmannGallisThermophoresisPlan(
        revision=revision,
        gas_velocity_field=_name(model["gas_velocity_field"], "gas_velocity_field"),
        gas_temperature_field=_name(
            model["gas_temperature_field"],
            "gas_temperature_field",
        ),
        gas_translational_heat_flux_field=_name(
            model["gas_translational_heat_flux_field"],
            "gas_translational_heat_flux_field",
        ),
        gas_mean_free_path_field=_name(
            model["gas_mean_free_path_field"],
            "gas_mean_free_path_field",
        ),
        gas_molecular_mass_kg=_positive_number(
            model["gas_molecular_mass_kg"],
            "physics.thermophoresis.gas_molecular_mass_kg",
        ),
        maximum_speed_ratio=maximum_speed_ratio,
    )


def _require_matching_neutral_gas(
    drag: DragPlan | None,
    thermophoresis: WaldmannGallisThermophoresisPlan | None,
    lift: RarefiedVorticityLiftPlan | None,
    gravity: GravityBuoyancyPlan | None,
) -> None:
    if drag is not None and thermophoresis is not None:
        if isinstance(drag, StokesCunninghamDragPlan):
            raise PhysicsConfigurationError(
                "Waldmann--Gallis thermophoresis has no applicability overlap with "
                "stokes_cunningham drag"
            )
        shared = (
            ("gas_velocity_field", drag.gas_velocity_field),
            ("gas_temperature_field", drag.gas_temperature_field),
            ("gas_mean_free_path_field", drag.gas_mean_free_path_field),
            ("gas_molecular_mass_kg", drag.gas_molecular_mass_kg),
        )
        mismatched = [name for name, value in shared if getattr(thermophoresis, name) != value]
        if mismatched:
            raise PhysicsConfigurationError(
                "physics.drag and physics.thermophoresis must use the same neutral-gas "
                f"background; mismatched={mismatched}"
            )
    _require_matching_lift_background(drag, thermophoresis, lift, gravity)


def _require_matching_lift_background(
    drag: DragPlan | None,
    thermophoresis: WaldmannGallisThermophoresisPlan | None,
    lift: RarefiedVorticityLiftPlan | None,
    gravity: GravityBuoyancyPlan | None,
) -> None:
    if lift is None:
        return
    if isinstance(drag, StokesCunninghamDragPlan):
        raise PhysicsConfigurationError(
            "rarefied-vorticity lift has no applicability overlap with stokes_cunningham drag"
        )
    shared: list[tuple[str, str]] = []
    if drag is not None:
        shared.extend(
            (
                ("gas_velocity_field", drag.gas_velocity_field),
                ("gas_density_field", drag.gas_density_field),
                ("gas_mean_free_path_field", drag.gas_mean_free_path_field),
            )
        )
    if thermophoresis is not None:
        shared.extend(
            (
                ("gas_velocity_field", thermophoresis.gas_velocity_field),
                ("gas_mean_free_path_field", thermophoresis.gas_mean_free_path_field),
            )
        )
    if gravity is not None:
        shared.append(("gas_density_field", gravity.gas_density_field))
    mismatched = [name for name, value in shared if getattr(lift, name) != value]
    if mismatched:
        raise PhysicsConfigurationError(
            "physics.lift must use the same neutral-gas background as other enabled models; "
            f"mismatched={sorted(set(mismatched))}"
        )


def _require_matching_ion_species(
    charge: ChargePlan | None,
    ion_drag: IonDragPlan | None,
) -> None:
    if charge is None or ion_drag is None:
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
    if isinstance(ion_drag, BarnesCollisionlessIonDragPlan):
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
    if isinstance(ion_drag, BarnesCollisionlessIonDragPlan):
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
    _exact_keys(
        model,
        {
            "model",
            "revision",
            "gradient_mean_e_squared_field",
            "medium_relative_permittivity",
            "real_clausius_mossotti_factor",
            "maximum_point_dipole_radius_m",
        },
        "physics.dielectrophoresis",
    )
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
            "physics.dielectrophoresis.real_clausius_mossotti_factor must be a finite "
            "number in [-0.5, 1]"
        )
    factor = float(factor_value)
    if not math.isfinite(factor) or not -0.5 <= factor <= 1.0:
        raise PhysicsConfigurationError(
            "physics.dielectrophoresis.real_clausius_mossotti_factor must be a finite "
            "number in [-0.5, 1]"
        )
    return QuasistaticSphericalDielectrophoresisPlan(
        gradient_mean_e_squared_field=_name(
            model["gradient_mean_e_squared_field"],
            "gradient_mean_e_squared_field",
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
    model: Mapping[str, object] | None,
    coordinate_system: CoordinateSystem,
) -> RarefiedVorticityLiftPlan | None:
    if model is None:
        return None
    _exact_keys(
        model,
        {
            "model",
            "revision",
            "gas_velocity_field",
            "gas_density_field",
            "gas_mean_free_path_field",
            "azimuthal_gas_vorticity_field",
            "lift_coefficient",
            "applicability",
        },
        "physics.lift",
    )
    if _text(model["model"], "physics.lift.model") != "rarefied_vorticity_sensitivity":
        raise PhysicsConfigurationError("physics.lift model must be rarefied_vorticity_sensitivity")
    expected = "rarefied_vorticity_sensitivity_rz_v1"
    if _text(model["revision"], "physics.lift.revision") != expected:
        raise PhysicsConfigurationError(
            f"rarefied_vorticity_sensitivity revision must be {expected}"
        )
    if coordinate_system != "axisymmetric_rz":
        raise PhysicsConfigurationError(
            "rarefied_vorticity_sensitivity_rz_v1 supports axisymmetric_rz motion only"
        )
    if _text(model["applicability"], "physics.lift.applicability") != "error":
        raise PhysicsConfigurationError(
            "rarefied-vorticity lift supports physics.lift applicability=error only"
        )
    return RarefiedVorticityLiftPlan(
        gas_velocity_field=_name(model["gas_velocity_field"], "gas_velocity_field"),
        gas_density_field=_name(model["gas_density_field"], "gas_density_field"),
        gas_mean_free_path_field=_name(
            model["gas_mean_free_path_field"],
            "gas_mean_free_path_field",
        ),
        azimuthal_gas_vorticity_field=_name(
            model["azimuthal_gas_vorticity_field"],
            "azimuthal_gas_vorticity_field",
        ),
        lift_coefficient=_positive_number(
            model["lift_coefficient"],
            "physics.lift.lift_coefficient",
        ),
    )


def _resolve_electric(model: Mapping[str, object] | None) -> ElectricPlan | None:
    if model is None:
        return None
    _exact_keys(model, {"model", "revision", "electric_field"}, "physics.electric")
    if _text(model["model"], "physics.electric.model") != "coulomb":
        raise PhysicsConfigurationError("P06 supports only electric model=coulomb")
    if _text(model["revision"], "physics.electric.revision") != "electric_coulomb_v1":
        raise PhysicsConfigurationError("physics.electric revision must be electric_coulomb_v1")
    return ElectricPlan(_name(model["electric_field"], "electric_field"))


def _resolve_gravity(
    model: Mapping[str, object] | None,
    coordinate_system: CoordinateSystem,
) -> GravityBuoyancyPlan | None:
    if model is None:
        return None
    _exact_keys(
        model,
        {"model", "revision", "gas_density_field", "gravity_m_s2"},
        "physics.gravity_buoyancy",
    )
    if _text(model["model"], "physics.gravity_buoyancy.model") != "standard":
        raise PhysicsConfigurationError("P06 supports only gravity_buoyancy model=standard")
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
            "axisymmetric_rz gravity_buoyancy_standard_v1 requires gravity_m_s2[0] = 0; "
            "a nonzero radial body acceleration is not a spatially uniform gravity vector"
        )
    return GravityBuoyancyPlan(_name(model["gas_density_field"], "gas_density_field"), gravity)


def _scalar_field(name: str, unit: str, *, positive: bool) -> RequiredField:
    return RequiredField(name, unit, ("value",), "scalar", positive)


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
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise PhysicsConfigurationError(f"{location} must be a positive finite number")
    return number


def _maximum_speed_ratio(value: object, location: str) -> float:
    number = _positive_number(value, location)
    if number > 1.0:
        raise PhysicsConfigurationError(f"{location} must be at most 1.0")
    return number


def _unit_interval_number(value: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise PhysicsConfigurationError(f"{location} must be a finite number in [0, 1]")
    number = float(value)
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
        number = float(item)
        if not math.isfinite(number):
            raise PhysicsConfigurationError(f"{location} must contain finite numbers")
        result.append(number)
    return result[0], result[1]
