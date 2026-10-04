from __future__ import annotations

import pytest

from particle_tracer_unified.force_models import parse_native_force_model
from particle_tracer_unified.solvers.charge_model import ChargeModelConfig
from particle_tracer_unified.solvers.forces import (
    ForceRuntimeParameters,
    resolve_force_catalog,
)
from particle_tracer_unified.solvers.runtime_plan import (
    StageFieldPlan,
    build_stage_field_plan,
)
from particle_tracer_unified.solvers.stochastic_motion import StochasticMotionConfig


def _catalog(
    *,
    drag_model: str = "none",
    forces: dict[str, object] | None = None,
):
    model = parse_native_force_model(
        {"model": drag_model},
        forces or {},
        spatial_dim=2,
    )
    return resolve_force_catalog(model, spatial_dim=2)


def _stage_plan(
    *,
    drag_model: str,
    force_catalog=None,
    charge: ChargeModelConfig | None = None,
    stochastic: StochasticMotionConfig | None = None,
    force_runtime: ForceRuntimeParameters | None = None,
) -> StageFieldPlan:
    return build_stage_field_plan(
        drag_model=drag_model,
        force_catalog=force_catalog,
        charge_model=charge or ChargeModelConfig(),
        stochastic_motion=stochastic or StochasticMotionConfig(),
        force_runtime=force_runtime or ForceRuntimeParameters(),
    )


@pytest.mark.parametrize(
    ("drag_model", "expected"),
    [
        (
            "none",
            StageFieldPlan(
                need_flow=False,
                need_electric=False,
                need_gas_density=False,
                need_gas_mu=False,
                need_gas_temperature=False,
            ),
        ),
        ("stokes", StageFieldPlan(need_flow=True, need_gas_mu=True)),
        (
            "epstein",
            StageFieldPlan(
                need_flow=True,
                need_gas_density=True,
                need_gas_temperature=True,
            ),
        ),
        (
            "stokes_cunningham",
            StageFieldPlan(
                need_flow=True,
                need_gas_density=True,
                need_gas_mu=True,
                need_gas_temperature=True,
            ),
        ),
        (
            "schiller_naumann",
            StageFieldPlan(
                need_flow=True,
                need_gas_density=True,
                need_gas_mu=True,
            ),
        ),
    ],
)
def test_drag_model_preserves_exact_stage_field_requirements(
    drag_model: str,
    expected: StageFieldPlan,
) -> None:
    assert _stage_plan(drag_model=drag_model) == expected


def test_catalog_and_runtime_force_flags_share_stage_requirements() -> None:
    catalog = _catalog(
        forces={
            "electric": {"enabled": True},
            "pressure_gradient": {"enabled": True},
        }
    )
    runtime = ForceRuntimeParameters(
        thermophoresis_enabled=True,
        dielectrophoresis_enabled=True,
        lift_enabled=True,
        virtual_mass_enabled=True,
        gravity_buoyancy_enabled=True,
    )

    plan = _stage_plan(
        drag_model="none",
        force_catalog=catalog,
        force_runtime=runtime,
    )

    assert plan == StageFieldPlan(
        need_flow=True,
        need_electric=True,
        need_gas_density=True,
        need_gas_mu=True,
        need_gas_temperature=True,
    )


def test_charge_and_stochastic_sources_preserve_field_authority() -> None:
    field_backed = _stage_plan(
        drag_model="none",
        charge=ChargeModelConfig(enabled=True, background_source="field"),
        stochastic=StochasticMotionConfig(
            enabled=True,
            temperature_source="field_T_then_gas",
        ),
    )
    configured = _stage_plan(
        drag_model="none",
        charge=ChargeModelConfig(enabled=True, background_source="configured"),
        stochastic=StochasticMotionConfig(
            enabled=True,
            temperature_source="gas",
        ),
    )

    assert field_backed == StageFieldPlan(
        need_flow=True,
        need_electric=True,
        need_gas_temperature=True,
    )
    assert configured == StageFieldPlan(
        need_flow=True,
        need_electric=True,
        need_gas_temperature=False,
    )


def test_catalog_force_remains_authoritative_when_runtime_projection_is_disabled() -> (
    None
):
    catalog = _catalog(forces={"virtual_mass": {"enabled": True}})

    plan = _stage_plan(
        drag_model="none",
        force_catalog=catalog,
        force_runtime=ForceRuntimeParameters(),
    )

    assert plan.need_flow
    assert plan.need_gas_density


@pytest.mark.parametrize(
    "runtime",
    [
        ForceRuntimeParameters(thermophoresis_enabled=True),
        ForceRuntimeParameters(lift_enabled=True),
    ],
)
def test_transport_forces_require_density_from_the_shared_authority(
    runtime: ForceRuntimeParameters,
) -> None:
    plan = _stage_plan(drag_model="none", force_runtime=runtime)

    assert plan.need_gas_density
    assert plan.need_gas_mu


def test_enabled_force_names_preserve_execution_and_catalog_order() -> None:
    runtime = ForceRuntimeParameters(
        thermophoresis_enabled=True,
        dielectrophoresis_enabled=True,
        lift_enabled=True,
        pressure_gradient_enabled=True,
        virtual_mass_enabled=True,
    )
    catalog = _catalog(
        forces={
            "electric": {"enabled": True},
            "pressure_gradient": {"enabled": True},
        }
    )

    assert runtime.enabled_evaluator_names() == (
        "pressure_gradient",
        "virtual_mass",
        "thermophoresis",
        "dielectrophoresis",
        "lift",
    )
    assert catalog.enabled_names() == ("electric", "pressure_gradient")
