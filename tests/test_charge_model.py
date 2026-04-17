from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from particle_tracer_unified.core.datamodel import (
    FieldProviderND,
    GasProperties,
    GeometryND,
    GeometryProviderND,
    ParticleTable,
    PreparedRuntime,
    QuantitySeriesND,
    RegularFieldND,
    RuntimeLike,
)
from particle_tracer_unified.solvers.charge_model import (
    E_CHARGE_C,
    EPS0_F_M,
    ChargeModelConfig,
    apply_charge_model_update,
)
from particle_tracer_unified.solvers.compiled_field_backend import (
    compile_runtime_backend,
    sample_compiled_acceleration_vectors,
)
from particle_tracer_unified.solvers.high_fidelity_runtime import run_prepared_runtime


def _series(name: str, times: np.ndarray, values: np.ndarray, unit: str = "") -> QuantitySeriesND:
    return QuantitySeriesND(
        name=name,
        unit=unit,
        times=np.asarray(times, dtype=np.float64),
        data=np.asarray(values, dtype=np.float64),
        metadata={},
    )


def _regular_charge_field(
    *,
    times: np.ndarray | None = None,
    te_values: np.ndarray | None = None,
    ne_values: np.ndarray | None = None,
) -> RegularFieldND:
    axes = (
        np.asarray([0.0, 1.0], dtype=np.float64),
        np.asarray([0.0, 1.0], dtype=np.float64),
    )
    t = np.asarray([0.0], dtype=np.float64) if times is None else np.asarray(times, dtype=np.float64)
    shape = (int(t.size), 2, 2)
    te = np.ones(shape, dtype=np.float64) * 3.0 if te_values is None else np.asarray(te_values, dtype=np.float64)
    ne = np.ones(shape, dtype=np.float64) * 1.0e16 if ne_values is None else np.asarray(ne_values, dtype=np.float64)
    quantities = {
        "ux": _series("ux", t, np.zeros(shape, dtype=np.float64)),
        "uy": _series("uy", t, np.zeros(shape, dtype=np.float64)),
        "E_x": _series("E_x", t, np.ones(shape, dtype=np.float64) * 4.0),
        "E_y": _series("E_y", t, np.ones(shape, dtype=np.float64) * -6.0),
        "Te": _series("Te", t, te, "eV"),
        "ne": _series("ne", t, ne, "1/m^3"),
        "ni": _series("ni", t, ne, "1/m^3"),
    }
    return RegularFieldND(
        spatial_dim=2,
        coordinate_system="cartesian_xy",
        axis_names=("x", "y"),
        axes=axes,
        quantities=quantities,
        valid_mask=np.ones((2, 2), dtype=bool),
        time_mode="steady" if t.size == 1 else "transient",
        metadata={"provider_kind": "precomputed_npz"},
    )


def _runtime_for_field(field: RegularFieldND):
    return SimpleNamespace(
        geometry_provider=SimpleNamespace(
            geometry=SimpleNamespace(axes=field.axes, valid_mask=field.valid_mask)
        ),
        field_provider=SimpleNamespace(field=field),
        gas=SimpleNamespace(density_kgm3=1.0, dynamic_viscosity_Pas=1.8e-5, temperature=300.0),
    )


def _geometry_provider_for_field(field: RegularFieldND) -> GeometryProviderND:
    valid_mask = np.asarray(field.valid_mask, dtype=bool)
    return GeometryProviderND(
        geometry=GeometryND(
            spatial_dim=2,
            coordinate_system="cartesian_xy",
            axes=field.axes,
            valid_mask=valid_mask,
            sdf=-np.ones_like(valid_mask, dtype=np.float64),
            normal_components=tuple(np.zeros_like(valid_mask, dtype=np.float64) for _ in range(2)),
            nearest_boundary_part_id_map=np.ones_like(valid_mask, dtype=np.int32),
            source_kind="synthetic",
            metadata={},
        ),
        kind="synthetic",
    )


def _particle_table(charge: float = -0.5, mass: float = 2.0) -> ParticleTable:
    return ParticleTable(
        spatial_dim=2,
        particle_id=np.asarray([1], dtype=np.int64),
        position=np.asarray([[0.5, 0.5]], dtype=np.float64),
        velocity=np.asarray([[0.0, 0.0]], dtype=np.float64),
        release_time=np.asarray([0.0], dtype=np.float64),
        mass=np.asarray([mass], dtype=np.float64),
        diameter=np.asarray([20.0e-9], dtype=np.float64),
        density=np.asarray([2200.0], dtype=np.float64),
        charge=np.asarray([charge], dtype=np.float64),
        source_part_id=np.asarray([1], dtype=np.int64),
        material_id=np.asarray([1], dtype=np.int64),
        source_event_tag=np.asarray([""], dtype=object),
        source_law_override=np.asarray([""], dtype=object),
        source_speed_scale_override=np.asarray([np.nan], dtype=np.float64),
        stick_probability=np.asarray([0.0], dtype=np.float64),
        dep_particle_rel_permittivity=np.asarray([np.nan], dtype=np.float64),
        thermophoretic_coeff=np.asarray([np.nan], dtype=np.float64),
    )


def test_te_relaxation_charge_model_samples_transient_temperature() -> None:
    times = np.asarray([0.0, 1.0], dtype=np.float64)
    te_values = np.stack(
        [
            np.ones((2, 2), dtype=np.float64) * 2.0,
            np.ones((2, 2), dtype=np.float64) * 4.0,
        ],
        axis=0,
    )
    field = _regular_charge_field(times=times, te_values=te_values)
    charge = np.asarray([0.0], dtype=np.float64)
    config = ChargeModelConfig(
        enabled=True,
        mode="te_relaxation",
        te_relaxation_alpha=2.5,
        relaxation_time_s=1.0e-9,
    )

    result = apply_charge_model_update(
        config=config,
        runtime=_runtime_for_field(field),
        spatial_dim=2,
        t_eval=0.5,
        delta_t_s=1.0e-6,
        active_mask=np.asarray([True]),
        x=np.asarray([[0.5, 0.5]], dtype=np.float64),
        charge=charge,
        particle_diameter=np.asarray([20.0e-9], dtype=np.float64),
    )

    expected = -4.0 * np.pi * EPS0_F_M * 10.0e-9 * 2.5 * 3.0
    assert result["applied"] is True
    assert charge[0] == pytest.approx(expected, rel=1.0e-6)


def test_density_temperature_flux_charge_model_produces_finite_negative_charge() -> None:
    field = _regular_charge_field()
    charge = np.asarray([0.0], dtype=np.float64)
    config = ChargeModelConfig(
        enabled=True,
        mode="density_temperature_flux_relaxation",
        relaxation_time_s=1.0e-6,
        max_abs_potential_V=100.0,
    )

    result = apply_charge_model_update(
        config=config,
        runtime=_runtime_for_field(field),
        spatial_dim=2,
        t_eval=0.0,
        delta_t_s=1.0e-3,
        active_mask=np.asarray([True]),
        x=np.asarray([[0.5, 0.5]], dtype=np.float64),
        charge=charge,
        particle_diameter=np.asarray([20.0e-9], dtype=np.float64),
    )

    assert result["applied"] is True
    assert np.isfinite(charge[0])
    assert charge[0] < 0.0
    assert abs(charge[0] / E_CHARGE_C) > 1.0


def test_dynamic_charge_electric_backend_ignores_stale_exported_acceleration() -> None:
    field = _regular_charge_field()
    stale = np.ones((1, 2, 2), dtype=np.float64) * 1.0e9
    quantities = dict(field.quantities)
    quantities["ax"] = _series("ax", np.asarray([0.0], dtype=np.float64), stale)
    quantities["ay"] = _series("ay", np.asarray([0.0], dtype=np.float64), stale)
    field = RegularFieldND(
        spatial_dim=field.spatial_dim,
        coordinate_system=field.coordinate_system,
        axis_names=field.axis_names,
        axes=field.axes,
        quantities=quantities,
        valid_mask=field.valid_mask,
        time_mode=field.time_mode,
        metadata=field.metadata,
    )
    runtime = _runtime_for_field(field)
    particles = _particle_table(charge=-0.5, mass=2.0)
    runtime.particles = particles

    backend = compile_runtime_backend(runtime, 2, particles=particles, dynamic_electric=True)
    accel = sample_compiled_acceleration_vectors(
        backend,
        2,
        0.0,
        np.asarray([[0.5, 0.5]], dtype=np.float64),
        electric_q_over_m=np.asarray([-0.25], dtype=np.float64),
    )

    assert backend.acceleration_source == "particle_charge_electric_field"
    np.testing.assert_allclose(accel, [[-1.0, 1.5]])


def test_runtime_charge_model_updates_report_without_enabling_by_default() -> None:
    field = _regular_charge_field()
    particles = _particle_table(charge=0.0, mass=1.0e-18)
    runtime = RuntimeLike(
        spatial_dim=2,
        coordinate_system="cartesian_xy",
        particles=particles,
        walls=None,
        materials=None,
        source_events=None,
        process_steps=None,
        compiled_source_events=None,
        geometry_provider=_geometry_provider_for_field(field),
        field_provider=FieldProviderND(field=field, kind="precomputed_npz"),
        gas=GasProperties(temperature=300.0, dynamic_viscosity_Pas=1.8e-5, density_kgm3=1.0),
        config_payload={
            "solver": {
                "dt": 1.0e-6,
                "t_end": 1.0e-6,
                "charge_model": {
                    "enabled": True,
                    "mode": "te_relaxation",
                    "electron_temperature_quantity": "Te",
                    "relaxation_time_s": 1.0e-9,
                },
            },
            "output": {"artifact_mode": "minimal"},
        },
    )

    report = run_prepared_runtime(PreparedRuntime(runtime=runtime), output_dir=None, spatial_dim=2)

    charge_report = report["charge_model"]
    assert charge_report["enabled"] == 1
    assert charge_report["update_event_count"] == 1
    assert charge_report["final_mean_charge_C"] < 0.0
