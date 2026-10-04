"""Flat force parameters consumed by numerical/JIT kernels.

This is a pure projection of :mod:`particle_tracer_unified.force_models`.
Validation, model defaults, and user mappings deliberately do not live here.
"""

from __future__ import annotations

from dataclasses import dataclass

from particle_tracer_unified.force_models import (
    DEP_MEDIUM_REL_PERMITTIVITY_DEFAULT,
    LIFT_COEFFICIENT_DEFAULT,
    THERMOPHORESIS_CM_DEFAULT,
    THERMOPHORESIS_CS_DEFAULT,
    THERMOPHORESIS_CT_DEFAULT,
    THERMOPHORESIS_GAS_CONDUCTIVITY_DEFAULT,
    THERMOPHORESIS_PARTICLE_CONDUCTIVITY_DEFAULT,
    VIRTUAL_MASS_COEFFICIENT_DEFAULT,
    DielectrophoresisForce,
    ForceModel,
    LiftForce,
    PressureGradientForce,
    ThermophoresisForce,
    VirtualMassForce,
)


@dataclass(frozen=True)
class ForceRuntimeParameters:
    """Stable scalar ABI for the solver hot path and Numba kernels."""

    thermophoresis_enabled: bool = False
    thermophoresis_model: str = ThermophoresisForce().model
    gas_thermal_conductivity_W_mK: float = THERMOPHORESIS_GAS_CONDUCTIVITY_DEFAULT
    particle_thermal_conductivity_W_mK: float = (
        THERMOPHORESIS_PARTICLE_CONDUCTIVITY_DEFAULT
    )
    thermophoresis_Cs: float = THERMOPHORESIS_CS_DEFAULT
    thermophoresis_Cm: float = THERMOPHORESIS_CM_DEFAULT
    thermophoresis_Ct: float = THERMOPHORESIS_CT_DEFAULT

    dielectrophoresis_enabled: bool = False
    dielectrophoresis_model: str = DielectrophoresisForce().model
    dep_medium_rel_permittivity: float = DEP_MEDIUM_REL_PERMITTIVITY_DEFAULT
    dep_particle_rel_permittivity: float = float("nan")
    dep_medium_conductivity_Sm: float = 0.0
    dep_particle_conductivity_Sm: float = 0.0
    dep_frequency_Hz: float = 0.0

    lift_enabled: bool = False
    lift_model: str = LiftForce().model
    lift_coefficient: float = LIFT_COEFFICIENT_DEFAULT

    pressure_gradient_enabled: bool = False
    pressure_gradient_model: str = PressureGradientForce().model

    virtual_mass_enabled: bool = False
    virtual_mass_model: str = VirtualMassForce().model
    virtual_mass_coefficient: float = VIRTUAL_MASS_COEFFICIENT_DEFAULT

    gravity_buoyancy_enabled: bool = False
    dep_electric_field_amplitude: str = "rms"

    def enabled_evaluator_names(self) -> tuple[str, ...]:
        """Return enabled non-body-force evaluators in execution order."""

        flags = (
            ("pressure_gradient", self.pressure_gradient_enabled),
            ("virtual_mass", self.virtual_mass_enabled),
            ("thermophoresis", self.thermophoresis_enabled),
            ("dielectrophoresis", self.dielectrophoresis_enabled),
            ("lift", self.lift_enabled),
        )
        return tuple(name for name, enabled in flags if bool(enabled))


def compile_force_runtime_parameters(model: ForceModel) -> ForceRuntimeParameters:
    """Compile validated semantic values without parsing or filling defaults."""

    if not isinstance(model, ForceModel):
        raise TypeError("compile_force_runtime_parameters requires a typed ForceModel")
    thermo = model.thermophoresis
    dep = model.dielectrophoresis
    lift = model.lift
    pressure = model.pressure_gradient
    virtual_mass = model.virtual_mass
    return ForceRuntimeParameters(
        thermophoresis_enabled=thermo.enabled,
        thermophoresis_model=thermo.model,
        gas_thermal_conductivity_W_mK=thermo.gas_thermal_conductivity_W_mK,
        particle_thermal_conductivity_W_mK=thermo.particle_thermal_conductivity_W_mK,
        thermophoresis_Cs=thermo.Cs,
        thermophoresis_Cm=thermo.Cm,
        thermophoresis_Ct=thermo.Ct,
        dielectrophoresis_enabled=dep.enabled,
        dielectrophoresis_model=dep.model,
        dep_medium_rel_permittivity=dep.medium_rel_permittivity,
        dep_particle_rel_permittivity=(
            float("nan")
            if dep.particle_rel_permittivity is None
            else dep.particle_rel_permittivity
        ),
        dep_medium_conductivity_Sm=dep.medium_conductivity_Sm,
        dep_particle_conductivity_Sm=dep.particle_conductivity_Sm,
        dep_frequency_Hz=dep.frequency_Hz,
        dep_electric_field_amplitude=dep.electric_field_amplitude,
        lift_enabled=lift.enabled,
        lift_model=lift.model,
        lift_coefficient=lift.coefficient,
        pressure_gradient_enabled=pressure.enabled,
        pressure_gradient_model=pressure.model,
        virtual_mass_enabled=virtual_mass.enabled,
        virtual_mass_model=virtual_mass.model,
        virtual_mass_coefficient=virtual_mass.coefficient,
        gravity_buoyancy_enabled=model.gravity.enabled and model.gravity.buoyancy,
    )


def force_runtime_parameters_summary(
    params: ForceRuntimeParameters | None,
) -> dict[str, object]:
    p = params or ForceRuntimeParameters()
    return {
        "thermophoresis_enabled": int(bool(p.thermophoresis_enabled)),
        "thermophoresis_model": str(p.thermophoresis_model),
        "dielectrophoresis_enabled": int(bool(p.dielectrophoresis_enabled)),
        "dielectrophoresis_model": str(p.dielectrophoresis_model),
        "dep_electric_field_amplitude": str(p.dep_electric_field_amplitude),
        "lift_enabled": int(bool(p.lift_enabled)),
        "lift_model": str(p.lift_model),
        "pressure_gradient_enabled": int(bool(p.pressure_gradient_enabled)),
        "pressure_gradient_model": str(p.pressure_gradient_model),
        "virtual_mass_enabled": int(bool(p.virtual_mass_enabled)),
        "virtual_mass_model": str(p.virtual_mass_model),
        "virtual_mass_coefficient": float(p.virtual_mass_coefficient),
        "gravity_buoyancy_enabled": int(bool(p.gravity_buoyancy_enabled)),
        "rejected_model_policy": "fail_fast",
        "implemented_equations": {
            "thermophoresis": "talbot_like_with_continuum_kn_zero"
            if p.thermophoresis_enabled
            else "",
            "dielectrophoresis": (
                "clausius_mossotti_gradient_e_squared_"
                f"{p.dep_electric_field_amplitude}_electric_field"
            )
            if p.dielectrophoresis_enabled
            else "",
            "lift": "saffman_vorticity_form" if p.lift_enabled else "",
            "pressure_gradient": "rho_g_over_rho_p_fluid_material_acceleration"
            if p.pressure_gradient_enabled
            else "",
            "virtual_mass": (
                "coefficient_rho_g_over_rho_p_particle_path_fluid_acceleration"
            )
            if p.virtual_mass_enabled
            else "",
            "gravity_buoyancy": "body_acceleration_scaled_by_one_minus_rho_g_over_rho_p"
            if p.gravity_buoyancy_enabled
            else "",
        },
    }


__all__ = (
    "ForceRuntimeParameters",
    "compile_force_runtime_parameters",
    "force_runtime_parameters_summary",
)
