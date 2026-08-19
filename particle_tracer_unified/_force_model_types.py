"""Immutable semantic force types and their canonical names."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar

DRAG_MODELS = frozenset(
    {"none", "stokes", "stokes_cunningham", "schiller_naumann", "epstein"}
)
OPTIONAL_FORCE_NAMES = (
    "electric",
    "gravity",
    "thermophoresis",
    "dielectrophoresis",
    "lift",
    "pressure_gradient",
    "virtual_mass",
)
FORCE_NAMES = ("drag", *OPTIONAL_FORCE_NAMES)

THERMOPHORESIS_GAS_CONDUCTIVITY_DEFAULT = 0.026
THERMOPHORESIS_PARTICLE_CONDUCTIVITY_DEFAULT = 1.4
THERMOPHORESIS_CS_DEFAULT = 1.17
THERMOPHORESIS_CM_DEFAULT = 1.14
THERMOPHORESIS_CT_DEFAULT = 2.18
DEP_MEDIUM_REL_PERMITTIVITY_DEFAULT = 1.0006
DEP_ELECTRIC_FIELD_AMPLITUDES = frozenset({"rms", "peak"})
LIFT_COEFFICIENT_DEFAULT = 6.46
VIRTUAL_MASS_COEFFICIENT_DEFAULT = 0.5


class ForceModelError(ValueError):
    """One invalid value in the canonical force contract."""


@dataclass(frozen=True, slots=True)
class DragForce:
    name: ClassVar[str] = "drag"
    status: ClassVar[str] = "implemented"
    enabled: bool = False
    model: str = "none"


@dataclass(frozen=True, slots=True)
class ElectricForce:
    name: ClassVar[str] = "electric"
    status: ClassVar[str] = "implemented"
    enabled: bool = False
    model: str = "particle_charge"


@dataclass(frozen=True, slots=True)
class GravityForce:
    name: ClassVar[str] = "gravity"
    status: ClassVar[str] = "implemented"
    enabled: bool = False
    model: str = "constant_acceleration"
    acceleration_mps2: tuple[float, ...] = ()
    buoyancy: bool = False


@dataclass(frozen=True, slots=True)
class ThermophoresisForce:
    name: ClassVar[str] = "thermophoresis"
    status: ClassVar[str] = "experimental"
    enabled: bool = False
    model: str = "talbot"
    gas_thermal_conductivity_W_mK: float = THERMOPHORESIS_GAS_CONDUCTIVITY_DEFAULT
    particle_thermal_conductivity_W_mK: float = (
        THERMOPHORESIS_PARTICLE_CONDUCTIVITY_DEFAULT
    )
    Cs: float = THERMOPHORESIS_CS_DEFAULT
    Cm: float = THERMOPHORESIS_CM_DEFAULT
    Ct: float = THERMOPHORESIS_CT_DEFAULT


@dataclass(frozen=True, slots=True)
class DielectrophoresisForce:
    name: ClassVar[str] = "dielectrophoresis"
    status: ClassVar[str] = "experimental"
    enabled: bool = False
    model: str = "dc"
    medium_rel_permittivity: float = DEP_MEDIUM_REL_PERMITTIVITY_DEFAULT
    particle_rel_permittivity: float | None = None
    medium_conductivity_Sm: float = 0.0
    particle_conductivity_Sm: float = 0.0
    frequency_Hz: float = 0.0
    electric_field_amplitude: str = "rms"


@dataclass(frozen=True, slots=True)
class LiftForce:
    name: ClassVar[str] = "lift"
    status: ClassVar[str] = "experimental"
    enabled: bool = False
    model: str = "saffman"
    coefficient: float = LIFT_COEFFICIENT_DEFAULT


@dataclass(frozen=True, slots=True)
class PressureGradientForce:
    name: ClassVar[str] = "pressure_gradient"
    status: ClassVar[str] = "experimental"
    enabled: bool = False
    model: str = "fluid_material_acceleration"


@dataclass(frozen=True, slots=True)
class VirtualMassForce:
    name: ClassVar[str] = "virtual_mass"
    status: ClassVar[str] = "experimental"
    enabled: bool = False
    model: str = "particle_material_acceleration"
    coefficient: float = VIRTUAL_MASS_COEFFICIENT_DEFAULT


SemanticForce = (
    DragForce
    | ElectricForce
    | GravityForce
    | ThermophoresisForce
    | DielectrophoresisForce
    | LiftForce
    | PressureGradientForce
    | VirtualMassForce
)


@dataclass(frozen=True, slots=True)
class ForceModel:
    """Complete force selection, with one typed value per supported equation."""

    drag: DragForce
    electric: ElectricForce = ElectricForce()
    gravity: GravityForce = GravityForce()
    thermophoresis: ThermophoresisForce = ThermophoresisForce()
    dielectrophoresis: DielectrophoresisForce = DielectrophoresisForce()
    lift: LiftForce = LiftForce()
    pressure_gradient: PressureGradientForce = PressureGradientForce()
    virtual_mass: VirtualMassForce = VirtualMassForce()
    declared: frozenset[str] = frozenset({"drag"})

    def definition(self, name: str) -> SemanticForce:
        if name not in FORCE_NAMES:
            raise KeyError(name)
        return getattr(self, name)

    def definitions(self) -> tuple[SemanticForce, ...]:
        return tuple(self.definition(name) for name in FORCE_NAMES)

    def enabled_names(self) -> tuple[str, ...]:
        return tuple(
            force.name
            for force in self.definitions()
            if force.enabled and not (force.name == "drag" and force.model == "none")
        )


FORCE_MODELS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "electric": frozenset({"particle_charge"}),
        "gravity": frozenset({"constant_acceleration"}),
        "thermophoresis": frozenset({"talbot", "continuum"}),
        "dielectrophoresis": frozenset({"dc", "ac_clausius_mossotti"}),
        "lift": frozenset({"saffman"}),
        "pressure_gradient": frozenset({"fluid_material_acceleration"}),
        "virtual_mass": frozenset({"particle_material_acceleration"}),
    }
)
_DEFAULT_MODELS = {name: next(iter(models)) for name, models in FORCE_MODELS.items()}
_DEFAULT_MODELS.update(
    {
        "thermophoresis": "talbot",
        "dielectrophoresis": "dc",
    }
)
FORCE_PARAMETER_KEYS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "electric": frozenset(),
        "gravity": frozenset({"acceleration_mps2", "buoyancy"}),
        "thermophoresis": frozenset(
            {
                "gas_thermal_conductivity_W_mK",
                "particle_thermal_conductivity_W_mK",
                "Cs",
                "Cm",
                "Ct",
            }
        ),
        "dielectrophoresis": frozenset(
            {
                "medium_rel_permittivity",
                "particle_rel_permittivity",
                "medium_conductivity_Sm",
                "particle_conductivity_Sm",
                "frequency_Hz",
                "electric_field_amplitude",
            }
        ),
        "lift": frozenset({"coefficient"}),
        "pressure_gradient": frozenset(),
        "virtual_mass": frozenset({"coefficient"}),
    }
)
