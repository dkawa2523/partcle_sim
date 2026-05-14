from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from particle_tracer_unified.core.catalogs import build_physics_catalog
from particle_tracer_unified.core.datamodel import FieldProviderND, QuantitySeriesND, RegularFieldND
from particle_tracer_unified.solvers.forces import (
    SUPPORTED_FORCE_NAMES,
    ForceRuntimeParameters,
    apply_manifest_force_inventory_to_solver_config,
    build_force_catalog,
    force_catalog_summary,
    force_runtime_parameters_from_catalog,
    solver_cfg_with_force_overrides,
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
    field = RegularFieldND(
        spatial_dim=2,
        coordinate_system="cartesian_xy",
        axis_names=("x", "y"),
        axes=axes,
        valid_mask=np.ones((2, 2), dtype=bool),
        quantities={
            "ux": _series("ux", np.zeros(shape, dtype=np.float64)),
            "uy": _series("uy", np.zeros(shape, dtype=np.float64)),
            "E_x": _series("E_x", np.ones(shape, dtype=np.float64)),
            "E_y": _series("E_y", np.ones(shape, dtype=np.float64) * -2.0),
            "T": _series("T", np.ones(shape, dtype=np.float64) * 320.0),
        },
    )
    return FieldProviderND(field=field)


def _regular_exported_fluid_accel_provider() -> FieldProviderND:
    axes = (
        np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
        np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
    )
    shape = (1, 3, 3)
    field = RegularFieldND(
        spatial_dim=2,
        coordinate_system="cartesian_xy",
        axis_names=("x", "y"),
        axes=axes,
        valid_mask=np.ones((3, 3), dtype=bool),
        quantities={
            "fluid_accel_x": _series("fluid_accel_x", np.ones(shape, dtype=np.float64)),
            "fluid_accel_y": _series("fluid_accel_y", np.zeros(shape, dtype=np.float64)),
            "rho_g": _series("rho_g", np.ones(shape, dtype=np.float64) * 2.0),
        },
    )
    return FieldProviderND(field=field)


def test_force_catalog_defaults_use_available_electric_field() -> None:
    catalog = build_force_catalog({"solver": {}}, field_provider=_field_provider(), spatial_dim=2)

    assert catalog.enabled("drag")
    assert catalog.model("drag") == "stokes"
    assert catalog.enabled("electric")
    assert not catalog.enabled("gravity")
    assert not catalog.enabled("brownian")
    assert not catalog.enabled("thermophoresis")


def test_force_catalog_accepts_stokes_cunningham_drag_model() -> None:
    catalog = build_force_catalog(
        {"solver": {"drag_model": "stokes_cunningham", "forces": {"drag": {"model": "stokes_cunningham"}}}},
        field_provider=_field_provider(),
        spatial_dim=2,
    )

    assert catalog.model("drag") == "stokes_cunningham"


def test_force_catalog_can_disable_electric_field_sampling() -> None:
    catalog = build_force_catalog(
        {"solver": {"forces": {"electric": {"enabled": False}}}},
        field_provider=_field_provider(),
        spatial_dim=2,
    )

    electric = catalog.by_name()["electric"]
    assert not catalog.enabled("electric")
    assert electric.required_fields == ()
    assert electric.field_sources == {}


def test_thermophoresis_force_can_be_enabled_when_temperature_is_available() -> None:
    catalog = build_force_catalog(
        {"solver": {"forces": {"thermophoresis": {"enabled": True}}}},
        field_provider=_field_provider(),
        spatial_dim=2,
    )

    assert catalog.enabled("thermophoresis")
    assert catalog.model("thermophoresis") == "talbot"


def test_force_runtime_rejects_unsupported_force_model_name() -> None:
    catalog = build_force_catalog(
        {"solver": {"forces": {"thermophoresis": {"enabled": True, "model": "waldmann"}}}},
        field_provider=_field_provider(),
        spatial_dim=2,
    )

    with pytest.raises(ValueError, match="solver.forces.thermophoresis.model"):
        force_runtime_parameters_from_catalog(catalog)


def test_force_runtime_accepts_implemented_force_model_names() -> None:
    catalog = build_force_catalog(
        {
            "solver": {
                "forces": {
                    "thermophoresis": {"enabled": True, "model": "continuum"},
                    "dielectrophoresis": {"enabled": True, "model": "ac_clausius_mossotti"},
                    "lift": {"enabled": True, "model": "saffman"},
                    "pressure_gradient": {"enabled": True, "model": "fluid_material_acceleration"},
                    "virtual_mass": {"enabled": True, "model": "particle_material_acceleration"},
                }
            }
        },
        field_provider=_field_provider(),
        spatial_dim=2,
    )

    params = force_runtime_parameters_from_catalog(catalog)

    assert params.thermophoresis_model == "continuum"
    assert params.dielectrophoresis_model == "ac_clausius_mossotti"
    assert params.lift_model == "saffman"
    assert params.pressure_gradient_model == "fluid_material_acceleration"
    assert params.virtual_mass_model == "particle_material_acceleration"


def test_unknown_force_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown solver.forces entries"):
        build_force_catalog(
            {"solver": {"forces": {"magic_force": {"enabled": True}}}},
            field_provider=_field_provider(),
            spatial_dim=2,
        )


def test_known_comsol_force_contribution_gaps_are_reported_when_disabled() -> None:
    catalog = build_force_catalog({"solver": {}}, field_provider=_field_provider(), spatial_dim=2)
    summary = force_catalog_summary(catalog)

    assert summary["force_status"]["virtual_mass"] == "implemented"
    assert summary["force_status"]["pressure_gradient"] == "implemented"
    assert "virtual_mass" in summary["disabled_forces"]
    assert "pressure_gradient" in summary["disabled_forces"]


def test_force_catalog_summary_reports_enabled_reasons() -> None:
    catalog = build_force_catalog(
        {
            "solver": {
                "gravity_mps2": 9.81,
                "stochastic_motion": {"enabled": True},
                "forces": {
                    "electric": {"enabled": False},
                    "thermophoresis": {"enabled": True},
                },
            }
        },
        field_provider=_field_provider(),
        spatial_dim=2,
    )

    summary = force_catalog_summary(catalog)
    reasons = summary["force_enabled_reason"]

    assert reasons["drag"] == "required_solver"
    assert reasons["electric"] == "explicit_config"
    assert reasons["gravity"] == "legacy_solver_config"
    assert reasons["brownian"] == "stochastic_motion_default"
    assert reasons["thermophoresis"] == "explicit_config"
    assert reasons["virtual_mass"] == "default_false"


def test_virtual_mass_force_can_be_enabled_when_velocity_is_available() -> None:
    catalog = build_force_catalog(
        {"solver": {"forces": {"virtual_mass": {"enabled": True, "coefficient": 0.5}}}},
        field_provider=_field_provider(),
        spatial_dim=2,
    )

    assert catalog.enabled("virtual_mass")
    assert catalog.by_name()["virtual_mass"].required_fields == ("ux", "uy")


def test_pressure_gradient_force_can_be_enabled_when_pressure_is_available() -> None:
    catalog = build_force_catalog(
        {"solver": {"forces": {"pressure_gradient": {"enabled": True}}}},
        field_provider=_field_provider(),
        spatial_dim=2,
    )

    assert catalog.enabled("pressure_gradient")
    assert catalog.by_name()["pressure_gradient"].required_fields == ("ux", "uy")


def test_regular_grid_pressure_gradient_requires_velocity_not_exported_acceleration_only() -> None:
    with pytest.raises(ValueError, match="pressure_gradient"):
        build_force_catalog(
            {"solver": {"forces": {"pressure_gradient": {"enabled": True}}}},
            field_provider=_regular_exported_fluid_accel_provider(),
            spatial_dim=2,
        )


def test_force_gravity_config_controls_physics_catalog() -> None:
    disabled = build_physics_catalog(
        {"solver": {"gravity_mps2": 9.81, "forces": {"gravity": False}}},
        spatial_dim=2,
    )
    explicit = build_physics_catalog(
        {"solver": {"forces": {"gravity": {"enabled": True, "acceleration_mps2": [1.0, -3.0]}}}},
        spatial_dim=2,
    )

    assert disabled.body_acceleration == (0.0, 0.0)
    assert explicit.body_acceleration == (1.0, -3.0)


def test_force_overrides_keep_legacy_solver_keys_for_current_runtime() -> None:
    solver_cfg = {
        "drag_model": "stokes",
        "forces": {
            "drag": {"model": "epstein"},
            "brownian": {"enabled": True, "stride": 5, "seed": 123},
        },
    }
    catalog = build_force_catalog({"solver": solver_cfg}, field_provider=_field_provider(), spatial_dim=2)

    resolved = solver_cfg_with_force_overrides(solver_cfg, catalog)

    assert resolved["drag_model"] == "epstein"
    assert resolved["stochastic_motion"]["enabled"] is True
    assert resolved["stochastic_motion"]["stride"] == 5


def test_manifest_force_inventory_disables_every_supported_unlisted_force() -> None:
    solver_cfg: dict[str, object] = {}

    apply_manifest_force_inventory_to_solver_config(
        solver_cfg,
        ({"solver_force": "drag", "enabled": True, "law": "stokes"},),
    )

    disabled = {
        name
        for name, cfg in solver_cfg["forces"].items()
        if isinstance(cfg, dict) and cfg.get("enabled") is False
    }
    assert disabled == set(SUPPORTED_FORCE_NAMES) - {"drag"}


def test_run_config_schema_force_names_match_registry() -> None:
    schema_path = Path(__file__).resolve().parents[1] / "schemas" / "run_config.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    force_schema = schema["properties"]["solver"]["properties"]["forces"]["properties"]

    assert set(force_schema) == set(SUPPORTED_FORCE_NAMES)


def test_force_runtime_parameters_default_to_disabled() -> None:
    params = ForceRuntimeParameters()

    assert not params.thermophoresis_enabled
    assert not params.dielectrophoresis_enabled
    assert not params.lift_enabled
    assert not params.pressure_gradient_enabled
    assert not params.virtual_mass_enabled
