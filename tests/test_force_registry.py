from __future__ import annotations

import numpy as np
import pytest

from particle_tracer_unified.core.datamodel import (
    FieldProviderND,
    QuantitySeriesND,
    RegularFieldND,
)
from particle_tracer_unified.domain import StageFields
from particle_tracer_unified.force_models import (
    ForceModelError,
    parse_native_force_model,
)
from particle_tracer_unified.solvers.force_runtime import (
    ForceBatchState,
    ForceBatchStatic,
    ForcePipeline,
    evaluate_force_pipeline,
)
from particle_tracer_unified.solvers.forces import (
    ForceRuntimeParameters,
    compile_force_runtime_parameters,
    force_catalog_summary,
    resolve_force_catalog,
)


def _series(name: str, values: np.ndarray) -> QuantitySeriesND:
    return QuantitySeriesND(
        name=name,
        unit="",
        times=np.asarray([0.0], dtype=np.float64),
        data=np.asarray(values, dtype=np.float64),
    )


def _field_provider() -> FieldProviderND:
    axes = (
        np.asarray([0.0, 1.0], dtype=np.float64),
        np.asarray([0.0, 1.0], dtype=np.float64),
    )
    shape = (1, 2, 2)
    return FieldProviderND(
        field=RegularFieldND(
            spatial_dim=2,
            coordinate_system="cartesian_xy",
            axis_names=("x", "y"),
            axes=axes,
            valid_mask=np.ones((2, 2), dtype=bool),
            quantities={
                "ux": _series("ux", np.zeros(shape)),
                "uy": _series("uy", np.zeros(shape)),
                "E_x": _series("E_x", np.ones(shape)),
                "E_y": _series("E_y", -2.0 * np.ones(shape)),
                "T": _series("T", 320.0 * np.ones(shape)),
            },
        )
    )


def _regular_exported_fluid_accel_provider() -> FieldProviderND:
    axes = tuple(np.asarray([0.0, 0.5, 1.0]) for _ in range(2))
    shape = (1, 3, 3)
    return FieldProviderND(
        field=RegularFieldND(
            spatial_dim=2,
            coordinate_system="cartesian_xy",
            axis_names=("x", "y"),
            axes=axes,
            valid_mask=np.ones((3, 3), dtype=bool),
            quantities={
                "fluid_accel_x": _series("fluid_accel_x", np.ones(shape)),
                "fluid_accel_y": _series("fluid_accel_y", np.zeros(shape)),
                "rho_g": _series("rho_g", 2.0 * np.ones(shape)),
            },
        )
    )


def _catalog(
    forces: dict[str, object] | None = None,
    *,
    drag_model: str = "stokes",
    field_provider: FieldProviderND | None = None,
):
    model = parse_native_force_model(
        {"model": drag_model},
        forces or {},
        spatial_dim=2,
    )
    return resolve_force_catalog(
        model,
        field_provider=_field_provider() if field_provider is None else field_provider,
        spatial_dim=2,
    )


def test_native_force_model_requires_explicit_drag_contract() -> None:
    with pytest.raises(ForceModelError, match=r"physics\.drag: must be a mapping"):
        parse_native_force_model(None, {}, spatial_dim=2)


def test_catalog_does_not_infer_forces_from_available_fields() -> None:
    catalog = _catalog()
    assert catalog.enabled("drag")
    assert catalog.force_model_name("drag") == "stokes"
    assert not catalog.enabled("electric")
    assert not catalog.enabled("gravity")
    assert not catalog.enabled("thermophoresis")


def test_explicit_none_drag_is_bound_as_inactive() -> None:
    catalog = _catalog(drag_model="none")
    assert not catalog.enabled("drag")
    assert catalog.model.drag.model == "none"


def test_disabled_electric_has_no_field_binding() -> None:
    catalog = _catalog({"electric": {"enabled": False}})
    electric = catalog.by_name()["electric"]
    assert not electric.enabled
    assert electric.required_fields == ()
    assert electric.field_sources == ()


def test_all_implemented_models_compile_from_the_same_typed_model() -> None:
    catalog = _catalog(
        {
            "thermophoresis": {
                "enabled": True,
                "model": "continuum",
                "parameters": {
                    "gas_thermal_conductivity_W_mK": 0.031,
                    "particle_thermal_conductivity_W_mK": 2.4,
                },
            },
            "dielectrophoresis": {
                "enabled": True,
                "model": "ac_clausius_mossotti",
                "parameters": {
                    "medium_rel_permittivity": 1.2,
                    "medium_conductivity_Sm": 0.1,
                    "particle_conductivity_Sm": 0.2,
                    "frequency_Hz": 13.56e6,
                },
            },
            "lift": {"enabled": True, "model": "saffman"},
            "pressure_gradient": {
                "enabled": True,
                "model": "fluid_material_acceleration",
            },
            "virtual_mass": {
                "enabled": True,
                "model": "particle_material_acceleration",
            },
        }
    )
    params = compile_force_runtime_parameters(catalog.model)
    assert params.thermophoresis_model == "continuum"
    assert params.dielectrophoresis_model == "ac_clausius_mossotti"
    assert params.lift_model == "saffman"
    assert params.pressure_gradient_model == "fluid_material_acceleration"
    assert params.virtual_mass_model == "particle_material_acceleration"


def test_unknown_force_and_model_are_rejected_at_the_parser_boundary() -> None:
    with pytest.raises(ForceModelError, match="unknown force"):
        parse_native_force_model(
            {"model": "stokes"},
            {"magic_force": {"enabled": True}},
            spatial_dim=2,
        )
    with pytest.raises(ForceModelError, match=r"thermophoresis\.model"):
        parse_native_force_model(
            {"model": "stokes"},
            {"thermophoresis": {"enabled": False, "model": "waldmann"}},
            spatial_dim=2,
        )


def test_catalog_summary_uses_semantic_status_and_declaration() -> None:
    catalog = _catalog(
        {
            "electric": {"enabled": False},
            "gravity": {
                "enabled": True,
                "parameters": {"acceleration_mps2": [0.0, -9.81]},
            },
        }
    )
    summary = force_catalog_summary(catalog)
    assert summary["force_status"]["virtual_mass"] == "experimental"
    assert summary["force_enabled_reason"]["electric"] == "explicit_config"
    assert summary["force_enabled_reason"]["virtual_mass"] == "default_false"


def test_virtual_mass_and_pressure_gradient_bind_velocity_fields() -> None:
    catalog = _catalog(
        {
            "virtual_mass": {"enabled": True},
            "pressure_gradient": {"enabled": True},
        }
    )
    assert catalog.by_name()["virtual_mass"].required_fields == ("ux", "uy")
    assert catalog.by_name()["pressure_gradient"].required_fields == ("ux", "uy")


def _test_regular_pressure_gradient_without_velocity() -> None:
    catalog = _catalog(
        {"pressure_gradient": {"enabled": True}},
        field_provider=_regular_exported_fluid_accel_provider(),
    )
    assert catalog.by_name()["pressure_gradient"].required_fields == ()


test_regular_pressure_gradient_does_not_bind_exported_acceleration_without_velocity = (
    _test_regular_pressure_gradient_without_velocity
)


def test_force_pipeline_rejects_unknown_evaluator_name() -> None:
    pipeline = ForcePipeline(
        evaluator_names=("not_a_force",),
        params=ForceRuntimeParameters(),
    )
    with pytest.raises(ValueError, match="unknown force evaluator"):
        evaluate_force_pipeline(
            np.zeros((1, 2)),
            static=ForceBatchStatic(
                particle_diameter=np.ones(1),
                particle_density=np.ones(1),
                particle_mass=np.ones(1),
                dep_particle_rel_permittivity=np.ones(1),
                thermophoretic_coeff=np.ones(1),
            ),
            state=ForceBatchState(velocity=np.zeros((1, 2))),
            fields=StageFields(
                points_m=np.zeros((1, 2)),
                time_s=0.0,
                values={},
                supported=np.ones(1, dtype=bool),
            ),
            plan=pipeline,
        )


def test_force_runtime_parameters_default_to_disabled() -> None:
    params = ForceRuntimeParameters()
    assert not params.thermophoresis_enabled
    assert not params.dielectrophoresis_enabled
    assert not params.lift_enabled
    assert not params.pressure_gradient_enabled
    assert not params.virtual_mass_enabled
