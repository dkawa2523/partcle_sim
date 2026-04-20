from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from particle_tracer_unified.core.catalogs import build_physics_catalog
from particle_tracer_unified.core.datamodel import FieldProviderND, QuantitySeriesND, RegularFieldND
from particle_tracer_unified.solvers.compiled_field_backend import (
    compile_runtime_backend,
    sample_compiled_acceleration_vectors,
)
from particle_tracer_unified.solvers.forces import (
    ForceRuntimeParameters,
    build_force_catalog,
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


def _varying_field_provider() -> FieldProviderND:
    axes = (
        np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
        np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
    )
    xx, yy = np.meshgrid(axes[0], axes[1], indexing="ij")
    shape = (1, 3, 3)
    field = RegularFieldND(
        spatial_dim=2,
        coordinate_system="cartesian_xy",
        axis_names=("x", "y"),
        axes=axes,
        valid_mask=np.ones((3, 3), dtype=bool),
        quantities={
            "ux": _series("ux", yy.reshape(shape)),
            "uy": _series("uy", np.zeros(shape, dtype=np.float64)),
            "E_x": _series("E_x", xx.reshape(shape)),
            "E_y": _series("E_y", np.zeros(shape, dtype=np.float64)),
            "T": _series("T", (300.0 + 20.0 * xx).reshape(shape)),
        },
    )
    return FieldProviderND(field=field)


def _runtime(provider: FieldProviderND) -> SimpleNamespace:
    field = provider.field
    return SimpleNamespace(
        geometry_provider=SimpleNamespace(
            geometry=SimpleNamespace(axes=field.axes, valid_mask=field.valid_mask)
        ),
        field_provider=provider,
        gas=SimpleNamespace(density_kgm3=1.0, dynamic_viscosity_Pas=1.8e-5, temperature=300.0),
    )


def test_force_catalog_defaults_use_available_electric_field() -> None:
    catalog = build_force_catalog({"solver": {}}, field_provider=_field_provider(), spatial_dim=2)

    assert catalog.enabled("drag")
    assert catalog.model("drag") == "stokes"
    assert catalog.enabled("electric")
    assert not catalog.enabled("gravity")
    assert not catalog.enabled("brownian")
    assert not catalog.enabled("thermophoresis")


def test_force_catalog_can_disable_electric_field_sampling() -> None:
    provider = _field_provider()
    catalog = build_force_catalog(
        {"solver": {"forces": {"electric": {"enabled": False}}}},
        field_provider=provider,
        spatial_dim=2,
    )

    backend = compile_runtime_backend(
        _runtime(provider),
        2,
        enable_electric=catalog.enabled("electric"),
    )

    assert not catalog.enabled("electric")
    assert backend.acceleration_source == "none"
    assert backend.electric_field_names == ()


def test_thermophoresis_force_can_be_enabled_when_temperature_is_available() -> None:
    catalog = build_force_catalog(
        {"solver": {"forces": {"thermophoresis": {"enabled": True}}}},
        field_provider=_field_provider(),
        spatial_dim=2,
    )

    assert catalog.enabled("thermophoresis")
    assert catalog.model("thermophoresis") == "talbot"


def test_unknown_force_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown solver.forces entries"):
        build_force_catalog(
            {"solver": {"forces": {"magic_force": {"enabled": True}}}},
            field_provider=_field_provider(),
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


def test_comsol_style_forces_add_expected_acceleration_directions() -> None:
    provider = _varying_field_provider()
    runtime = _runtime(provider)
    force_runtime = ForceRuntimeParameters(
        thermophoresis_enabled=True,
        dielectrophoresis_enabled=True,
        lift_enabled=True,
        dep_particle_rel_permittivity=3.9,
        dep_medium_rel_permittivity=1.0,
    )
    backend = compile_runtime_backend(runtime, 2, force_runtime=force_runtime)
    positions = np.asarray([[0.5, 0.5]], dtype=np.float64)
    velocity = np.asarray([[1.5, 0.0]], dtype=np.float64)
    diameter = np.asarray([1.0e-6], dtype=np.float64)
    density = np.asarray([2200.0], dtype=np.float64)
    mass = density * np.pi * diameter**3 / 6.0

    accel = sample_compiled_acceleration_vectors(
        backend,
        2,
        0.0,
        positions,
        force_runtime=force_runtime,
        particle_diameter=diameter,
        particle_density=density,
        particle_mass=mass,
        dep_particle_rel_permittivity=np.asarray([3.9], dtype=np.float64),
        thermophoretic_coeff=np.asarray([1.0], dtype=np.float64),
        velocity=velocity,
        gas_density_kgm3=1.0,
        gas_mu_pas=1.8e-5,
        gas_temperature_K=300.0,
    )

    assert accel.shape == (1, 2)
    assert accel[0, 0] < 0.0  # thermophoresis moves from hot to cold for T increasing in +x.
    assert accel[0, 1] > 0.0  # Saffman term is nonzero for slip in this shear field.
