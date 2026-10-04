from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
import yaml

from particle_tracer_unified import load_case, simulate, validate_case
from particle_tracer_unified.core.datamodel import (
    FieldProviderND,
    ParticleTable,
    QuantitySeriesND,
    RegularFieldND,
)
from particle_tracer_unified.core.field_sampling import (
    VALID_MASK_STATUS_CLEAN,
    VALID_MASK_STATUS_HARD_INVALID,
)
from particle_tracer_unified.solvers import (
    _coupled_charge_motion as coupled_motion_module,
)
from particle_tracer_unified.solvers import charge_model as charge_model_module
from particle_tracer_unified.solvers import high_fidelity_runtime as runtime_module
from particle_tracer_unified.solvers._charge_background import (
    density_names,
    ion_density_names,
)
from particle_tracer_unified.solvers._charge_oml import oml_linearized_equilibrium
from particle_tracer_unified.solvers.charge_model import (
    EPS0_F_M,
    ChargeModelConfig,
    advance_charge_strang_segment,
    apply_charge_model_update,
    finalize_charge_model_diagnostics,
    merge_charge_model_diagnostics,
    record_terminal_charge_replay,
    validate_charge_model_support,
)
from particle_tracer_unified.solvers.collision_detection import TrialCollisionBatch
from particle_tracer_unified.solvers.field_compilation import compile_runtime_backend
from particle_tracer_unified.solvers.force_field_assembly import (
    sample_compiled_acceleration_vectors,
)
from particle_tracer_unified.solvers.high_fidelity_collision import (
    CollidingParticleAdvanceResult,
)


def _series(
    name: str, times: np.ndarray, values: np.ndarray, unit: str = ""
) -> QuantitySeriesND:
    return QuantitySeriesND(name=name, unit=unit, times=times, data=values)


def _field(
    *, times: np.ndarray | None = None, electron_temperature: np.ndarray | None = None
) -> RegularFieldND:
    axes = (np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0]))
    time_axis = (
        np.asarray([0.0]) if times is None else np.asarray(times, dtype=np.float64)
    )
    shape = (time_axis.size, 2, 2)
    temperature = (
        np.full(shape, 3.0)
        if electron_temperature is None
        else np.asarray(electron_temperature, dtype=np.float64)
    )
    return RegularFieldND(
        spatial_dim=2,
        coordinate_system="cartesian_xy",
        axis_names=("x", "y"),
        axes=axes,
        quantities={
            "ux": _series("ux", time_axis, np.zeros(shape), "m/s"),
            "uy": _series("uy", time_axis, np.zeros(shape), "m/s"),
            "E_x": _series("E_x", time_axis, np.full(shape, 4.0), "V/m"),
            "E_y": _series("E_y", time_axis, np.full(shape, -6.0), "V/m"),
            "Te": _series("Te", time_axis, temperature, "eV"),
        },
        valid_mask=np.ones((2, 2), dtype=bool),
        time_mode="steady" if time_axis.size == 1 else "transient",
    )


def _runtime(field: RegularFieldND) -> SimpleNamespace:
    return SimpleNamespace(
        geometry_provider=SimpleNamespace(
            geometry=SimpleNamespace(axes=field.axes, valid_mask=field.valid_mask)
        ),
        field_provider=FieldProviderND(field=field),
        gas=SimpleNamespace(
            density_kgm3=1.0, dynamic_viscosity_Pas=1.8e-5, temperature=300.0
        ),
    )


def _particles(charge: float, mass: float) -> ParticleTable:
    return ParticleTable(
        spatial_dim=2,
        particle_id=np.asarray([1], dtype=np.int64),
        position=np.asarray([[0.5, 0.5]]),
        velocity=np.asarray([[0.0, 0.0]]),
        release_time=np.asarray([0.0]),
        mass=np.asarray([mass]),
        diameter=np.asarray([20.0e-9]),
        density=np.asarray([2200.0]),
        charge=np.asarray([charge]),
        source_part_id=np.asarray([1], dtype=np.int64),
        material_id=np.asarray([1], dtype=np.int64),
        dep_particle_rel_permittivity=np.asarray([np.nan]),
        thermophoretic_coeff=np.asarray([np.nan]),
    )


def test_te_relaxation_samples_transient_temperature() -> None:
    times = np.asarray([0.0, 1.0])
    temperatures = np.stack((np.full((2, 2), 2.0), np.full((2, 2), 4.0)))
    charge = np.asarray([0.0])

    result = apply_charge_model_update(
        config=ChargeModelConfig(
            enabled=True,
            mode="te_relaxation",
            te_relaxation_alpha=2.5,
            relaxation_time_s=1.0e-9,
        ),
        runtime=_runtime(_field(times=times, electron_temperature=temperatures)),
        spatial_dim=2,
        t_eval=0.5,
        delta_t_s=1.0e-6,
        active_mask=np.asarray([True]),
        x=np.asarray([[0.5, 0.5]]),
        charge=charge,
        particle_diameter=np.asarray([20.0e-9]),
    )

    expected = -4.0 * np.pi * EPS0_F_M * 10.0e-9 * 2.5 * 3.0
    assert result["applied"] is True
    assert charge[0] == pytest.approx(expected, rel=1.0e-6)


def test_charge_field_temperature_unit_must_match_configuration() -> None:
    field = _field()
    field = replace(
        field,
        quantities={
            **field.quantities,
            "Te": replace(field.quantities["Te"], unit="K"),
        },
    )
    runtime = _runtime(field)

    with pytest.raises(
        ValueError,
        match=r"electron temperature field 'Te' unit must be 'eV', got 'K'",
    ):
        validate_charge_model_support(
            ChargeModelConfig(enabled=True, electron_temperature_unit="eV"),
            runtime,
            compile_runtime_backend(runtime, 2),
            2,
        )


def test_charge_field_temperature_uses_matching_kelvin_unit() -> None:
    field = _field()
    field = replace(
        field,
        quantities={
            **field.quantities,
            "Te": replace(field.quantities["Te"], unit="K"),
        },
    )
    charge = np.asarray([0.0])

    apply_charge_model_update(
        config=ChargeModelConfig(
            enabled=True,
            electron_temperature_unit="K",
            te_relaxation_alpha=1.0,
            relaxation_time_s=1.0e-12,
        ),
        runtime=_runtime(field),
        spatial_dim=2,
        t_eval=0.0,
        delta_t_s=1.0,
        active_mask=np.asarray([True]),
        x=np.asarray([[0.5, 0.5]]),
        charge=charge,
        particle_diameter=np.asarray([2.0e-6]),
    )

    expected_temperature_eV = 3.0 / 11604.51812155008
    expected = -4.0 * np.pi * EPS0_F_M * 1.0e-6 * expected_temperature_eV
    assert charge[0] == pytest.approx(expected, rel=1.0e-13)


def test_charge_update_preserves_array_contract_and_reports_finite_si_values() -> None:
    charge = np.asarray([0.0, 7.5e-19], dtype=np.float64)
    original_inactive = charge[1]

    result = apply_charge_model_update(
        config=ChargeModelConfig(
            enabled=True,
            mode="te_relaxation",
            te_relaxation_alpha=2.5,
            relaxation_time_s=1.0e-9,
        ),
        runtime=_runtime(_field()),
        spatial_dim=2,
        t_eval=0.0,
        delta_t_s=1.0e-6,
        active_mask=np.asarray([True, False], dtype=bool),
        x=np.asarray([[0.5, 0.5], [0.25, 0.25]], dtype=np.float64),
        charge=charge,
        particle_diameter=np.asarray([20.0e-9, 30.0e-9], dtype=np.float64),
        collect_diagnostics=True,
    )

    expected_coulomb = -4.0 * np.pi * EPS0_F_M * 10.0e-9 * 2.5 * 3.0
    assert charge.shape == (2,)
    assert charge.dtype == np.dtype(np.float64)
    assert charge[0] == pytest.approx(expected_coulomb, rel=1.0e-6)
    assert charge[1] == original_inactive
    assert result["particle_count"] == 1
    assert result["background_source"] == "field"
    assert result["mean_charge_C"] == pytest.approx(expected_coulomb, rel=1.0e-6)
    numeric_diagnostics = [
        value
        for key, value in result.items()
        if key.startswith("mean_") and isinstance(value, float)
    ]
    assert numeric_diagnostics
    assert np.all(np.isfinite(np.asarray(numeric_diagnostics, dtype=np.float64)))


def test_charge_update_rejects_invalid_array_and_time_contracts() -> None:
    config = ChargeModelConfig(enabled=True, mode="te_relaxation")
    runtime = _runtime(_field())
    common = {
        "config": config,
        "runtime": runtime,
        "spatial_dim": 2,
        "t_eval": 0.0,
        "active_mask": np.asarray([True]),
        "x": np.asarray([[0.5, 0.5]]),
        "charge": np.asarray([0.0]),
        "particle_diameter": np.asarray([20.0e-9]),
    }
    with pytest.raises(ValueError, match="delta_t_s"):
        apply_charge_model_update(delta_t_s=np.nan, **common)
    with pytest.raises(ValueError, match="supports only 2D"):
        apply_charge_model_update(delta_t_s=1.0, **{**common, "spatial_dim": 3})
    with pytest.raises(ValueError, match="positive particle diameter"):
        apply_charge_model_update(
            delta_t_s=1.0,
            **{**common, "particle_diameter": np.asarray([0.0])},
        )
    with pytest.raises(ValueError, match="finite initial charge"):
        apply_charge_model_update(
            delta_t_s=1.0,
            **{**common, "charge": np.asarray([np.nan])},
        )
    assert apply_charge_model_update(delta_t_s=0.0, **common) == {"applied": False}
    assert apply_charge_model_update(
        delta_t_s=1.0,
        **{**common, "active_mask": np.asarray([False])},
    ) == {"applied": False, "particle_count": 0}


def test_charge_support_segment_and_diagnostic_error_paths() -> None:
    config = ChargeModelConfig(enabled=True, mode="te_relaxation")
    runtime = _runtime(_field())
    compiled = compile_runtime_backend(runtime, 2)
    with pytest.raises(ValueError, match="supports 2D"):
        validate_charge_model_support(config, runtime, compiled, 3)
    with pytest.raises(ValueError, match="regular rectilinear"):
        validate_charge_model_support(
            config,
            runtime,
            cast(Any, SimpleNamespace()),
            2,
        )
    with pytest.raises(ValueError, match="regular field provider"):
        validate_charge_model_support(
            config,
            SimpleNamespace(field_provider=None),
            compiled,
            2,
        )
    segment = {
        "config": config,
        "runtime": runtime,
        "spatial_dim": 2,
        "t_start_s": 0.0,
        "x_start": np.zeros((1, 2)),
        "x_end": np.zeros((1, 2)),
        "charge_start": np.zeros(1),
        "particle_diameter": np.ones(1),
    }
    with pytest.raises(ValueError, match="duration_s"):
        advance_charge_strang_segment(duration_s=-1.0, **segment)
    with pytest.raises(ValueError, match="equally shaped"):
        advance_charge_strang_segment(
            duration_s=1.0,
            **{**segment, "x_end": np.zeros((2, 2))},
        )
    with pytest.raises(ValueError, match="match endpoint rows"):
        advance_charge_strang_segment(
            duration_s=1.0,
            **{**segment, "charge_start": np.zeros(2)},
        )

    diagnostics: dict[str, object] = {"charge_model": "invalid"}
    merge_charge_model_diagnostics(diagnostics, config, {"applied": False})
    diagnostics["charge_model"] = "invalid"
    record_terminal_charge_replay(diagnostics, config, age_s=-1.0)
    diagnostics["charge_model"] = "invalid"
    finalize_charge_model_diagnostics(diagnostics, config, np.asarray([1.0, np.nan]))
    assert isinstance(diagnostics["charge_model"], dict)


def test_oml_rejects_each_nonphysical_scalar_contract() -> None:
    config = ChargeModelConfig(enabled=True, mode="oml_linearized_relaxation")
    values = (
        np.asarray([1.0e-6]),
        np.asarray([3.0]),
        np.asarray([1.0e16]),
        np.asarray([1.0e16]),
        np.asarray([0.03]),
    )
    with pytest.raises(ValueError, match="broadcast-compatible"):
        oml_linearized_equilibrium(
            config,
            np.ones(2),
            np.ones(3),
            *values[2:],
        )
    with pytest.raises(ValueError, match="particle radius"):
        oml_linearized_equilibrium(
            config,
            np.asarray([0.0]),
            *values[1:],
        )
    cases = [
        (replace(config, ion_mass_amu=0.0), "ion mass"),
        (replace(config, ion_charge_number=0.0), "ion charge number"),
        (replace(config, electron_sticking=-1.0), "sticking coefficients"),
        (replace(config, bohm_velocity_factor=0.0), "Bohm velocity factor"),
        (replace(config, max_abs_potential_V=0.0), "maximum potential"),
        (replace(config, root_iterations=0), "root_iterations"),
    ]
    for invalid_config, message in cases:
        with pytest.raises(ValueError, match=message):
            oml_linearized_equilibrium(invalid_config, *values)


def test_oml_preserves_broadcast_shape_float64_and_finite_contracts() -> None:
    config = ChargeModelConfig(
        enabled=True,
        mode="oml_linearized_relaxation",
        ion_mass_amu=40.0,
        root_iterations=80,
    )
    charge, tau_q, potential = oml_linearized_equilibrium(
        config,
        np.asarray([[0.5e-6], [1.0e-6]], dtype=np.float32),
        np.asarray([[2.0, 3.0, 4.0]], dtype=np.float32),
        np.asarray(1.0e16, dtype=np.float32),
        np.asarray([[0.9e16, 1.0e16, 1.1e16]], dtype=np.float32),
        np.asarray(0.03, dtype=np.float32),
    )

    for values in (charge, tau_q, potential):
        assert values.shape == (2, 3)
        assert values.dtype == np.dtype(np.float64)
        assert np.all(np.isfinite(values))
    assert np.all(charge < 0.0)
    assert np.all(tau_q > 0.0)
    assert np.all((-config.max_abs_potential_V <= potential) & (potential <= 0.0))


def test_oml_rejects_nan_in_each_particle_or_plasma_input() -> None:
    config = ChargeModelConfig(enabled=True, mode="oml_linearized_relaxation")
    valid = [
        np.asarray([1.0e-6]),
        np.asarray([3.0]),
        np.asarray([1.0e16]),
        np.asarray([1.0e16]),
        np.asarray([0.03]),
    ]
    messages = (
        "particle radius",
        "electron temperature",
        "electron density",
        "ion density",
        "ion temperature",
    )

    for index, message in enumerate(messages):
        values = [item.copy() for item in valid]
        values[index][0] = np.nan
        with pytest.raises(ValueError, match=message):
            oml_linearized_equilibrium(config, *values)


def test_charge_disabled_and_response_regime_contracts() -> None:
    disabled = ChargeModelConfig(enabled=False)
    result = apply_charge_model_update(
        config=disabled,
        runtime=SimpleNamespace(),
        spatial_dim=2,
        t_eval=0.0,
        delta_t_s=1.0,
        active_mask=np.asarray([True]),
        x=np.zeros((1, 2)),
        charge=np.zeros(1),
        particle_diameter=np.ones(1),
    )
    assert result == {"applied": False}
    assert density_names(ChargeModelConfig()) == (
        "ne",
        "n_e",
        "electron_density",
        "electron_number_density",
    )
    assert ion_density_names(ChargeModelConfig()) == (
        "ni",
        "n_i",
        "ion_density",
        "ion_number_density",
    )
    assert charge_model_module._charge_response_regime(1.0, np.asarray([np.nan])) == (
        "unknown"
    )
    assert charge_model_module._charge_response_regime(0.1, np.asarray([1.0])) == (
        "explicit_transient"
    )


def test_charge_update_collects_debye_and_response_diagnostics_only_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    field = _oml_field(include_ion_density=True, include_ion_temperature=True)
    config = ChargeModelConfig(
        enabled=True,
        mode="oml_linearized_relaxation",
        background_source="field",
        electron_density_quantity="ne",
        ion_density_quantity="ni",
        ion_temperature_quantity="Ti_required",
        ion_mass_amu=40.0,
        ion_charge_number=1.0,
    )
    calls = {"debye": 0, "regime": 0}
    original_debye = charge_model_module.debye_length_m
    original_regime = charge_model_module._charge_response_regime

    def counted_debye(*args, **kwargs):
        calls["debye"] += 1
        return original_debye(*args, **kwargs)

    def counted_regime(*args, **kwargs):
        calls["regime"] += 1
        return original_regime(*args, **kwargs)

    monkeypatch.setattr(charge_model_module, "debye_length_m", counted_debye)
    monkeypatch.setattr(charge_model_module, "_charge_response_regime", counted_regime)

    common = {
        "config": config,
        "runtime": _runtime(field),
        "spatial_dim": 2,
        "t_eval": 0.0,
        "delta_t_s": 1.0e-6,
        "active_mask": np.asarray([True]),
        "x": np.asarray([[0.5, 0.5]]),
        "particle_diameter": np.asarray([1.0e-6]),
    }
    standard = apply_charge_model_update(charge=np.asarray([0.0]), **common)

    assert standard == {"applied": True}
    assert calls == {"debye": 0, "regime": 0}

    debug = apply_charge_model_update(
        charge=np.asarray([0.0]),
        collect_diagnostics=True,
        **common,
    )

    assert calls == {"debye": 1, "regime": 1}
    assert float(debug["mean_debye_length_m"]) > 0.0
    assert debug["charge_response_regime"] != "unknown"


def test_dynamic_charge_uses_particle_q_over_m_not_stale_exported_acceleration() -> (
    None
):
    field = _field()
    stale = np.full((1, 2, 2), 1.0e9)
    field = RegularFieldND(
        spatial_dim=field.spatial_dim,
        coordinate_system=field.coordinate_system,
        axis_names=field.axis_names,
        axes=field.axes,
        quantities={
            **field.quantities,
            "ax": _series("ax", np.asarray([0.0]), stale),
            "ay": _series("ay", np.asarray([0.0]), stale),
        },
        valid_mask=field.valid_mask,
        time_mode=field.time_mode,
    )
    runtime = _runtime(field)
    particles = _particles(charge=-0.5, mass=2.0)
    runtime.particles = particles

    backend = compile_runtime_backend(runtime, 2)
    acceleration = sample_compiled_acceleration_vectors(
        backend,
        2,
        0.0,
        np.asarray([[0.5, 0.5]]),
        electric_q_over_m=np.asarray([-0.25]),
    )

    assert backend.acceleration_source == "particle_charge_electric_field"
    np.testing.assert_allclose(acceleration, [[-1.0, 1.5]])


def _oml_field(
    *, include_ion_density: bool, include_ion_temperature: bool
) -> RegularFieldND:
    field = _field()
    shape = np.asarray(field.quantities["Te"].data).shape
    quantities = {
        **field.quantities,
        "ne": _series("ne", np.asarray([0.0]), np.full(shape, 1.0e16), "1/m^3"),
    }
    if include_ion_density:
        quantities["ni"] = _series(
            "ni", np.asarray([0.0]), np.full(shape, 1.0e16), "1/m^3"
        )
    if include_ion_temperature:
        quantities["Ti_required"] = _series(
            "Ti_required", np.asarray([0.0]), np.full(shape, 0.03), "eV"
        )
    return RegularFieldND(
        spatial_dim=field.spatial_dim,
        coordinate_system=field.coordinate_system,
        axis_names=field.axis_names,
        axes=field.axes,
        quantities=quantities,
        valid_mask=field.valid_mask,
        time_mode=field.time_mode,
    )


def _apply_field_oml(field: RegularFieldND) -> None:
    apply_charge_model_update(
        config=ChargeModelConfig(
            enabled=True,
            mode="oml_linearized_relaxation",
            background_source="field",
            electron_density_quantity="ne",
            ion_density_quantity="ni",
            ion_temperature_quantity="Ti_required",
            ion_mass_amu=40.0,
            ion_charge_number=1.0,
            relaxation_time_s=1.0e-6,
        ),
        runtime=_runtime(field),
        spatial_dim=2,
        t_eval=0.0,
        delta_t_s=1.0e-6,
        active_mask=np.asarray([True]),
        x=np.asarray([[0.5, 0.5]]),
        charge=np.asarray([0.0]),
        particle_diameter=np.asarray([1.0e-6]),
    )


def test_field_oml_requires_ion_density_instead_of_assuming_quasineutrality() -> None:
    with pytest.raises(ValueError, match=r"tried \['ni'\]"):
        _apply_field_oml(
            _oml_field(include_ion_density=False, include_ion_temperature=True)
        )


def test_field_oml_explicit_ion_temperature_quantity_cannot_fall_back_to_constant() -> (
    None
):
    with pytest.raises(ValueError, match=r"tried \['Ti_required'\]"):
        _apply_field_oml(
            _oml_field(include_ion_density=True, include_ion_temperature=False)
        )


@pytest.mark.parametrize(
    ("quantity", "unit", "message"),
    [
        ("ne", "kg/m^3", r"electron density field 'ne' unit must be '1/m\^3'"),
        ("ni", "kg/m^3", r"ion density field 'ni' unit must be '1/m\^3'"),
        ("Ti_required", "K", "ion temperature field 'Ti_required' unit must be 'eV'"),
    ],
)
def test_field_oml_rejects_mismatched_declared_plasma_units(
    quantity: str,
    unit: str,
    message: str,
) -> None:
    field = _oml_field(include_ion_density=True, include_ion_temperature=True)
    field = replace(
        field,
        quantities={
            **field.quantities,
            quantity: replace(field.quantities[quantity], unit=unit),
        },
    )

    with pytest.raises(ValueError, match=message):
        _apply_field_oml(field)


def _write_public_charge_case(
    root: Path,
    *,
    release_time_s: float,
    mass_kg: float = 1.0e-12,
    drag_diameter_m: float = 2.0e-6,
    particle_density_kgm3: float | None = None,
    velocity_x_mps: float = 0.0,
    velocity_y_mps: float = 0.0,
    wall_law: str = "specular",
    charge_mode: str = "te_relaxation",
    gravity_acceleration_mps2: tuple[float, float] | None = None,
    t_end_s: float = 1.0,
    electric_force_enabled: bool = False,
    electric_field_x_Vpm: float = 4.0,
    output_mode: str = "standard",
    field_valid_mask: np.ndarray | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    particle_row = {
        "particle_id": 1,
        "x_m": 0.0,
        "y_m": 0.0,
        "vx_mps": float(velocity_x_mps),
        "vy_mps": float(velocity_y_mps),
        "release_time_s": release_time_s,
        "mass_kg": float(mass_kg),
        "drag_diameter_m": float(drag_diameter_m),
        "charge_C": 0.0,
        "source_part_id": 1,
    }
    if particle_density_kgm3 is not None:
        particle_row["density_kgm3"] = float(particle_density_kgm3)
    pd.DataFrame([particle_row]).to_csv(root / "particles.csv", index=False)
    pd.DataFrame(
        [
            {
                "part_id": part_id,
                "part_name": f"wall_{part_id}",
                "role": "internal" if wall_law == "pass_through" else "wall",
                "material_id": part_id,
                "material_name": "test",
                "wall_law": str(wall_law),
                "wall_stick_probability": 0.0,
                "wall_restitution": 1.0,
                "wall_diffuse_fraction": 0.0,
                "wall_critical_sticking_velocity_mps": 0.0,
            }
            for part_id in (1, 2, 3, 4)
        ]
    ).to_csv(root / "boundaries.csv", index=False)
    axis = np.linspace(-2.0, 2.0, 9)
    shape = (axis.size, axis.size)
    valid_mask = (
        np.ones(shape, dtype=bool)
        if field_valid_mask is None
        else np.asarray(field_valid_mask, dtype=bool)
    )
    if valid_mask.shape != shape:
        raise ValueError(f"field_valid_mask must have shape {shape}")
    np.savez_compressed(
        root / "field.npz",
        axis_0=axis,
        axis_1=axis,
        times=np.asarray([0.0]),
        valid_mask=valid_mask,
        ux=np.zeros(shape),
        uy=np.zeros(shape),
        E_x=np.full(shape, float(electric_field_x_Vpm)),
        E_y=np.full(shape, -6.0),
        Te=np.full(shape, 4.0),
    )
    charge_config: dict[str, object]
    if charge_mode == "te_relaxation":
        charge_config = {
            "enabled": True,
            "mode": "te_relaxation",
            "parameters": {
                "background_source": "field",
                "electron_temperature_quantity": "Te",
                "electron_temperature_unit": "eV",
                "te_relaxation_alpha": 2.0,
                "relaxation_time_s": 0.4,
            },
        }
    else:
        charge_config = {
            "enabled": True,
            "mode": "oml_linearized_relaxation",
            "parameters": {
                "background_source": "plasma_background",
                "electron_temperature_unit": "eV",
            },
            "background": {
                "source": "saas_constant",
                "electron_density_m3": 1.0e15,
                "ion_density_m3": 1.0e15,
                "electron_temperature_eV": 4.0,
                "ion_temperature_eV": 0.03,
                "ion_mass_amu": 40.0,
                "ion_charge_number": 1.0,
            },
        }
    forces: dict[str, object] = {}
    if gravity_acceleration_mps2 is not None:
        forces["gravity"] = {
            "enabled": True,
            "model": "constant_acceleration",
            "parameters": {
                "acceleration_mps2": [
                    float(value) for value in gravity_acceleration_mps2
                ],
                "buoyancy": False,
            },
        }
    if electric_force_enabled:
        forces["electric"] = {"enabled": True}
    config = {
        "schema_version": 2,
        "case": {
            "spatial_dim": 2,
            "coordinate_system": "cartesian_xy",
            "adapter": "native",
        },
        "inputs": {
            "particles": "particles.csv",
            "boundaries": "boundaries.csv",
            "geometry": {
                "kind": "box",
                "parameters": {
                    "bounds": [-2.0, 2.0, -2.0, 2.0],
                    "grid_shape": [9, 9],
                    "boundary_part_ids": [1, 2, 3, 4],
                },
            },
            "field": {"kind": "precomputed_npz", "path": "field.npz"},
        },
        "physics": {
            "drag": {"model": "none"},
            "gas": {"temperature_K": 300.0},
            "forces": forces,
            "charge": charge_config,
            "seed": 19,
        },
        "time": {"dt": 1.0, "t_end": float(t_end_s)},
        "output": (
            {"mode": "debug", "trajectory_interval_steps": 1}
            if output_mode == "debug"
            else {"mode": "standard"}
        ),
    }
    path = root / "case.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_public_runtime_charge_half_steps_use_only_post_release_age(
    tmp_path: Path,
) -> None:
    case = load_case(
        _write_public_charge_case(tmp_path / "delayed", release_time_s=0.75)
    )
    report = validate_case(case)
    assert report.passed, report.errors

    result = simulate(case)

    radius = 1.0e-6
    equilibrium = 4.0 * np.pi * EPS0_F_M * radius * (-2.0 * 4.0)
    expected = equilibrium * (1.0 - np.exp(-0.25 / 0.4))
    assert result.state.charge_C[0] == pytest.approx(expected, rel=1.0e-13, abs=0.0)


@pytest.mark.parametrize(
    "charge_mode",
    ["te_relaxation", "oml_linearized_relaxation"],
)
def test_public_runtime_charge_uses_physical_not_aerodynamic_radius(
    tmp_path: Path,
    charge_mode: str,
) -> None:
    physical_diameter_m = 1.0e-6
    density_kgm3 = 1_600.0
    mass_kg = density_kgm3 * np.pi * physical_diameter_m**3 / 6.0
    case = load_case(
        _write_public_charge_case(
            tmp_path / "physical-radius",
            release_time_s=0.0,
            mass_kg=mass_kg,
            drag_diameter_m=4.0e-6,
            particle_density_kgm3=density_kgm3,
            charge_mode=charge_mode,
        )
    )

    result = simulate(case)

    physical_radius_m = 0.5 * physical_diameter_m
    if charge_mode == "te_relaxation":
        equilibrium = 4.0 * np.pi * EPS0_F_M * physical_radius_m * (-2.0 * 4.0)
        tau_q_s = 0.4
    else:
        equilibrium_arr, tau_arr, _potential = oml_linearized_equilibrium(
            ChargeModelConfig(
                enabled=True,
                mode="oml_linearized_relaxation",
                background_source="plasma_background",
            ),
            np.asarray([physical_radius_m]),
            np.asarray([4.0]),
            np.asarray([1.0e15]),
            np.asarray([1.0e15]),
            np.asarray([0.03]),
            ion_mass_amu=40.0,
            ion_charge_number=1.0,
        )
        equilibrium = float(equilibrium_arr[0])
        tau_q_s = float(tau_arr[0])
    expected = equilibrium * (1.0 - np.exp(-1.0 / tau_q_s))
    assert result.state.charge_C[0] == pytest.approx(expected, rel=2.0e-10, abs=1.0e-30)


def test_public_runtime_step_end_release_does_not_charge(tmp_path: Path) -> None:
    case = load_case(
        _write_public_charge_case(tmp_path / "step-end", release_time_s=1.0)
    )

    result = simulate(case)

    assert result.state.released.tolist() == [True]
    assert result.state.charge_C.tolist() == [0.0]


@pytest.mark.parametrize("charge_mode", ["te_relaxation", "oml_linearized_relaxation"])
def test_terminal_wall_charge_uses_hit_age_after_release_split(
    tmp_path: Path,
    charge_mode: str,
) -> None:
    case = load_case(
        _write_public_charge_case(
            tmp_path / charge_mode,
            release_time_s=0.2,
            velocity_x_mps=8.0,
            wall_law="absorb",
            charge_mode=charge_mode,
        )
    )

    result = simulate(case)

    assert result.state.terminal_state.tolist() == ["absorbed"]
    hit_age_s = 0.25
    radius_m = 1.0e-6
    if charge_mode == "te_relaxation":
        equilibrium = 4.0 * np.pi * EPS0_F_M * radius_m * (-2.0 * 4.0)
        tau_q_s = 0.4
    else:
        equilibrium_arr, tau_arr, _potential = oml_linearized_equilibrium(
            ChargeModelConfig(
                enabled=True,
                mode="oml_linearized_relaxation",
                background_source="plasma_background",
            ),
            np.asarray([radius_m]),
            np.asarray([4.0]),
            np.asarray([1.0e15]),
            np.asarray([1.0e15]),
            np.asarray([0.03]),
            ion_mass_amu=40.0,
            ion_charge_number=1.0,
        )
        equilibrium = float(equilibrium_arr[0])
        tau_q_s = float(tau_arr[0])
    expected = equilibrium * (1.0 - np.exp(-hit_age_s / tau_q_s))
    assert result.state.charge_C[0] == pytest.approx(expected, rel=2.0e-10, abs=1.0e-30)


def test_contact_endpoint_stop_keeps_full_active_charge_age(tmp_path: Path) -> None:
    case = load_case(
        _write_public_charge_case(
            tmp_path / "contact-endpoint",
            release_time_s=0.0,
            velocity_y_mps=2.0,
            wall_law="specular",
            gravity_acceleration_mps2=(400.0, 0.0),
            t_end_s=2.0,
        )
    )

    result = simulate(case)

    assert result.state.terminal_state.tolist() == ["contact_endpoint_stopped"]
    radius_m = 1.0e-6
    equilibrium = 4.0 * np.pi * EPS0_F_M * radius_m * (-2.0 * 4.0)
    expected = equilibrium * (1.0 - np.exp(-2.0 / 0.4))
    assert result.state.charge_C[0] == pytest.approx(expected, rel=1.0e-13, abs=1.0e-30)


def test_terminal_charge_age_includes_nonterminal_wall_replay(tmp_path: Path) -> None:
    path = _write_public_charge_case(
        tmp_path / "reflected-then-terminal",
        release_time_s=0.2,
        velocity_x_mps=8.0,
        wall_law="specular",
    )
    boundaries_path = path.parent / "boundaries.csv"
    boundaries = pd.read_csv(boundaries_path)
    # Synthetic-box order is bottom, right, top, left.  Reflect at x=+2,
    # then absorb at x=-2 after replaying the remaining segment.
    boundaries.loc[boundaries["part_id"] == 4, "wall_law"] = "absorb"
    boundaries.to_csv(boundaries_path, index=False)

    result = simulate(load_case(path))

    assert result.state.terminal_state.tolist() == ["absorbed"]
    radius_m = 1.0e-6
    equilibrium = 4.0 * np.pi * EPS0_F_M * radius_m * (-2.0 * 4.0)
    expected = equilibrium * (1.0 - np.exp(-0.75 / 0.4))
    assert result.state.charge_C[0] == pytest.approx(expected, rel=5.0e-10, abs=1.0e-30)


def test_terminal_charge_replay_has_compact_debug_provenance(tmp_path: Path) -> None:
    case = load_case(
        _write_public_charge_case(
            tmp_path / "debug-terminal",
            release_time_s=0.2,
            velocity_x_mps=8.0,
            wall_law="absorb",
            output_mode="debug",
        )
    )

    result = simulate(case)

    charge_diagnostics = result.debug["collision_diagnostics"]["charge_model"]
    assert (
        charge_diagnostics["operator_statistics_scope"]
        == "latest_global_half_step_evaluation"
    )
    assert charge_diagnostics["terminal_hit_replay_count"] == 1
    assert charge_diagnostics["terminal_hit_replay_age_total_s"] == pytest.approx(
        0.25, abs=3.0e-10
    )
    assert charge_diagnostics["terminal_hit_replay_age_max_s"] == pytest.approx(
        0.25, abs=3.0e-10
    )


def test_terminal_hit_integrates_charge_and_electric_motion_to_event_time(
    tmp_path: Path,
) -> None:
    case = load_case(
        _write_public_charge_case(
            tmp_path / "electric-terminal",
            release_time_s=0.0,
            mass_kg=1.0e-15,
            velocity_x_mps=1.0,
            wall_law="freeze",
            electric_force_enabled=True,
            electric_field_x_Vpm=-1.0e3,
        )
    )

    result = simulate(case)

    radius_m = 1.0e-6
    equilibrium = 4.0 * np.pi * EPS0_F_M * radius_m * (-2.0 * 4.0)
    tau_q_s = 0.4
    acceleration_limit = -1.0e3 * equilibrium / 1.0e-15

    def position_x(time_s: float) -> float:
        exponential = 1.0 - np.exp(-time_s / tau_q_s)
        return 1.0 * time_s + acceleration_limit * (
            0.5 * time_s**2 - tau_q_s * time_s + tau_q_s**2 * exponential
        )

    lower, upper = 0.0, 1.0
    for _ in range(80):
        midpoint = 0.5 * (lower + upper)
        if position_x(midpoint) < 2.0:
            lower = midpoint
        else:
            upper = midpoint
    hit_time_s = 0.5 * (lower + upper)
    expected_charge = equilibrium * (1.0 - np.exp(-hit_time_s / tau_q_s))
    expected_velocity_x = 1.0 + acceleration_limit * (
        hit_time_s - tau_q_s * (1.0 - np.exp(-hit_time_s / tau_q_s))
    )

    assert result.state.terminal_state.tolist() == ["frozen"]
    assert result.state.position_m[0, 0] == pytest.approx(2.0, abs=2.0e-10)
    assert result.state.velocity_mps[0, 0] == pytest.approx(
        expected_velocity_x,
        rel=2.0e-4,
    )
    assert result.state.charge_C[0] == pytest.approx(
        expected_charge,
        rel=2.0e-4,
        abs=1.0e-30,
    )


def _constant_te_charge_electric_state(
    time_s: float,
    *,
    initial_velocity_mps: float = 1.0,
    particle_mass_kg: float = 1.0e-15,
    electric_field_Vpm: float = -1.0e3,
) -> tuple[float, float, float]:
    equilibrium = 4.0 * np.pi * EPS0_F_M * 1.0e-6 * (-2.0 * 4.0)
    tau_q_s = 0.4
    acceleration_limit = electric_field_Vpm * equilibrium / particle_mass_kg
    exponential = 1.0 - np.exp(-float(time_s) / tau_q_s)
    charge = equilibrium * exponential
    velocity = initial_velocity_mps + acceleration_limit * (
        float(time_s) - tau_q_s * exponential
    )
    position = initial_velocity_mps * float(time_s) + acceleration_limit * (
        0.5 * float(time_s) ** 2 - tau_q_s * float(time_s) + tau_q_s**2 * exponential
    )
    return position, velocity, charge


def test_coupled_charge_electric_motion_advances_safe_endpoint(tmp_path: Path) -> None:
    duration_s = 0.1
    case = load_case(
        _write_public_charge_case(
            tmp_path / "electric-safe",
            release_time_s=0.0,
            mass_kg=1.0e-15,
            velocity_x_mps=1.0,
            wall_law="pass_through",
            electric_force_enabled=True,
            electric_field_x_Vpm=-1.0e3,
            t_end_s=duration_s,
        )
    )

    result = simulate(case)
    expected_position, expected_velocity, expected_charge = (
        _constant_te_charge_electric_state(duration_s)
    )

    assert result.state.terminal_state.tolist() == ["active_free_flight"]
    assert result.state.position_m[0, 0] == pytest.approx(expected_position, rel=2.0e-4)
    assert result.state.velocity_mps[0, 0] == pytest.approx(
        expected_velocity, rel=2.0e-4
    )
    assert result.state.charge_C[0] == pytest.approx(
        expected_charge, rel=2.0e-4, abs=1.0e-30
    )


def test_coupled_trace_is_continuous_at_saved_half_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = load_case(
        _write_public_charge_case(
            tmp_path / "electric-half-continuity",
            release_time_s=0.0,
            mass_kg=1.0e-15,
            velocity_x_mps=1.0,
            wall_law="pass_through",
            electric_force_enabled=True,
            electric_field_x_Vpm=-1.0e3,
            t_end_s=0.1,
        )
    )
    case = replace(
        case,
        _context=replace(
            case._context,
            plan=replace(case._context.plan, adaptive_substep_enabled=0),
        ),
    )
    half_states = []
    trace_batch = runtime_module.trace_coupled_charge_batch

    def capture_trace_batch(*args, **kwargs):
        batch = trace_batch(*args, **kwargs)
        trace = batch.traces[0]
        half_time = 0.5 * trace.request.duration_s / trace.substep_count
        epsilon = 1.0e-8 * trace.request.duration_s
        half_states.extend(
            (
                (
                    *trace.state_at(half_time - epsilon),
                    trace.charge_at(half_time - epsilon),
                ),
                (*trace.state_at(half_time), trace.charge_at(half_time)),
                (
                    *trace.state_at(half_time + epsilon),
                    trace.charge_at(half_time + epsilon),
                ),
            )
        )
        return batch

    monkeypatch.setattr(
        runtime_module,
        "trace_coupled_charge_batch",
        capture_trace_batch,
    )

    simulate(case)

    (
        (left_x, left_v, left_q),
        (center_x, center_v, center_q),
        (
            right_x,
            right_v,
            right_q,
        ),
    ) = half_states
    np.testing.assert_allclose(center_x, 0.5 * (left_x + right_x), atol=1.0e-12)
    np.testing.assert_allclose(center_v, 0.5 * (left_v + right_v), atol=1.0e-11)
    assert center_q == pytest.approx(0.5 * (left_q + right_q), abs=1.0e-28)


def test_coupled_trace_rejects_invalid_three_quarter_force_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = load_case(
        _write_public_charge_case(
            tmp_path / "electric-quarter-support",
            release_time_s=0.0,
            velocity_x_mps=1.0,
            wall_law="pass_through",
            electric_force_enabled=True,
            electric_field_x_Vpm=0.0,
        )
    )

    def narrow_probe_status(_backend, position_m: np.ndarray) -> int:
        x_position = float(np.asarray(position_m, dtype=np.float64)[0])
        if 0.74 < x_position < 0.76:
            return int(VALID_MASK_STATUS_HARD_INVALID)
        return int(VALID_MASK_STATUS_CLEAN)

    monkeypatch.setattr(
        coupled_motion_module,
        "sample_compiled_valid_mask_status",
        narrow_probe_status,
    )

    result = simulate(case)

    assert result.state.terminal_state.tolist() == ["invalid_mask_stopped"]


def test_spatial_charge_coefficients_converge_under_step_halving(
    tmp_path: Path,
) -> None:
    case_path = _write_public_charge_case(
        tmp_path / "electric-spatial-charge",
        release_time_s=0.0,
        mass_kg=1.0e-12,
        velocity_x_mps=1.0,
        wall_law="pass_through",
        electric_force_enabled=True,
        electric_field_x_Vpm=-1.0e3,
        t_end_s=0.1,
    )
    field_path = case_path.parent / "field.npz"
    with np.load(field_path) as source:
        payload = {name: source[name] for name in source.files}
    x_axis = np.asarray(payload["axis_0"], dtype=np.float64)
    payload["Te"] = (
        4.0 + x_axis[:, None] + np.zeros((x_axis.size, x_axis.size), dtype=np.float64)
    )
    np.savez_compressed(field_path, **payload)
    base = load_case(case_path)

    states: list[np.ndarray] = []
    for step_s in (0.1, 0.05, 0.0125):
        plan = replace(
            base._context.plan,
            dt=float(step_s),
            adaptive_substep_enabled=0,
        )
        case = replace(base, _context=replace(base._context, plan=plan))
        result = simulate(case)
        assert result.state.terminal_state.tolist() == ["active_free_flight"]
        states.append(
            np.asarray(
                [
                    result.state.position_m[0, 0],
                    result.state.velocity_mps[0, 0],
                    result.state.charge_C[0] / 1.0e-15,
                ],
                dtype=np.float64,
            )
        )

    coarse_error = float(np.linalg.norm(states[0] - states[2]))
    half_step_error = float(np.linalg.norm(states[1] - states[2]))
    assert half_step_error < 0.4 * coarse_error


def test_specular_remainder_keeps_charge_and_electric_motion_coupled(
    tmp_path: Path,
) -> None:
    duration_s = 0.25
    initial_velocity_mps = 10.0
    particle_mass_kg = 2.0e-8
    case = load_case(
        _write_public_charge_case(
            tmp_path / "electric-specular",
            release_time_s=0.0,
            mass_kg=particle_mass_kg,
            velocity_x_mps=initial_velocity_mps,
            wall_law="specular",
            electric_force_enabled=True,
            electric_field_x_Vpm=-1.0e3,
            t_end_s=duration_s,
        )
    )

    hit_lower, hit_upper = 0.0, duration_s
    for _ in range(80):
        midpoint = 0.5 * (hit_lower + hit_upper)
        position, _velocity, _charge = _constant_te_charge_electric_state(
            midpoint,
            initial_velocity_mps=initial_velocity_mps,
            particle_mass_kg=particle_mass_kg,
        )
        if position < 2.0:
            hit_lower = midpoint
        else:
            hit_upper = midpoint
    hit_time_s = 0.5 * (hit_lower + hit_upper)
    _hit_position, hit_velocity, _hit_charge = _constant_te_charge_electric_state(
        hit_time_s,
        initial_velocity_mps=initial_velocity_mps,
        particle_mass_kg=particle_mass_kg,
    )
    equilibrium = 4.0 * np.pi * EPS0_F_M * 1.0e-6 * (-2.0 * 4.0)
    acceleration_limit = -1.0e3 * equilibrium / particle_mass_kg
    tau_q_s = 0.4
    remainder_s = duration_s - hit_time_s
    hit_decay = np.exp(-hit_time_s / tau_q_s)
    remainder_decay = 1.0 - np.exp(-remainder_s / tau_q_s)
    expected_velocity = -hit_velocity + acceleration_limit * (
        remainder_s - tau_q_s * hit_decay * remainder_decay
    )
    expected_position = (
        2.0
        - hit_velocity * remainder_s
        + acceleration_limit
        * (
            0.5 * remainder_s**2
            - hit_decay * (tau_q_s * remainder_s - tau_q_s**2 * remainder_decay)
        )
    )
    _free_position, _free_velocity, expected_charge = (
        _constant_te_charge_electric_state(
            duration_s,
            initial_velocity_mps=initial_velocity_mps,
            particle_mass_kg=particle_mass_kg,
        )
    )

    result = simulate(case)

    assert result.state.terminal_state.tolist() == ["active_free_flight"]
    assert result.state.position_m[0, 0] == pytest.approx(expected_position, rel=3.0e-4)
    assert result.state.velocity_mps[0, 0] == pytest.approx(
        expected_velocity, rel=3.0e-4
    )
    assert result.state.charge_C[0] == pytest.approx(
        expected_charge, rel=2.0e-4, abs=1.0e-30
    )


def test_segment_endpoint_terminal_hit_allows_dynamic_charge_electric_split(
    tmp_path: Path,
) -> None:
    case = load_case(
        _write_public_charge_case(
            tmp_path / "electric-endpoint",
            release_time_s=0.2,
            velocity_x_mps=2.5,
            wall_law="absorb",
            electric_force_enabled=True,
            electric_field_x_Vpm=0.0,
        )
    )

    result = simulate(case)

    assert result.state.terminal_state.tolist() == ["absorbed"]
    radius_m = 1.0e-6
    equilibrium = 4.0 * np.pi * EPS0_F_M * radius_m * (-2.0 * 4.0)
    expected = equilibrium * (1.0 - np.exp(-0.8 / 0.4))
    assert result.state.charge_C[0] == pytest.approx(expected, rel=2.0e-10, abs=1.0e-30)


def test_geometry_refinement_stop_at_segment_start_has_zero_charge_age(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = load_case(
        _write_public_charge_case(tmp_path / "geometry-start-stop", release_time_s=0.0)
    )
    monkeypatch.setattr(
        runtime_module,
        "_refine_deterministic_stage_traces",
        lambda **_kwargs: {0: "geometry"},
    )

    result = simulate(case)

    assert result.state.terminal_state.tolist() == ["numerical_boundary_stopped"]
    assert result.state.charge_C.tolist() == [0.0]


def test_valid_mask_prefix_stop_charges_only_accepted_prefix(tmp_path: Path) -> None:
    valid_mask = np.ones((9, 9), dtype=bool)
    valid_mask[6:, :] = False
    case = load_case(
        _write_public_charge_case(
            tmp_path / "valid-prefix",
            release_time_s=0.0,
            velocity_x_mps=1.0,
            field_valid_mask=valid_mask,
        )
    )

    result = simulate(case)

    assert result.state.terminal_state.tolist() == ["invalid_mask_stopped"]
    # The retry accepts the first quarter-second prefix (x=0.25); charging
    # must stop at the same accepted physical age.
    assert result.state.position_m[0, 0] == pytest.approx(0.25, abs=1.0e-15)
    radius_m = 1.0e-6
    equilibrium = 4.0 * np.pi * EPS0_F_M * radius_m * (-2.0 * 4.0)
    expected = equilibrium * (1.0 - np.exp(-0.25 / 0.4))
    assert result.state.charge_C[0] == pytest.approx(expected, rel=1.0e-13, abs=1.0e-30)


def test_pass_through_bounding_box_endpoint_keeps_full_charge_age(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = load_case(
        _write_public_charge_case(
            tmp_path / "pass-through-endpoint",
            release_time_s=0.0,
            velocity_x_mps=0.0,
            wall_law="pass_through",
        )
    )
    # Isolate the endpoint commit made after a pass-through replay: the
    # collision primitive returns a still-active endpoint outside the box,
    # and the runtime owns the resulting bounding-box terminal transition.
    monkeypatch.setattr(
        runtime_module,
        "classify_trial_collisions",
        lambda *_args, **_kwargs: TrialCollisionBatch(
            colliders=np.asarray([0], dtype=np.int64),
            safe=np.zeros(0, dtype=np.int64),
            prefetched_hits={},
        ),
    )
    monkeypatch.setattr(
        runtime_module,
        "advance_colliding_particle",
        lambda **_kwargs: CollidingParticleAdvanceResult(
            position=np.asarray([3.0, 0.0]),
            velocity=np.asarray([1.0, 0.0]),
            total_hits=1,
            valid_mask_status=0,
            invalid_mask_stopped=False,
        ),
    )

    result = simulate(case)

    assert result.state.terminal_state.tolist() == ["escaped"]
    radius_m = 1.0e-6
    equilibrium = 4.0 * np.pi * EPS0_F_M * radius_m * (-2.0 * 4.0)
    expected = equilibrium * (1.0 - np.exp(-1.0 / 0.4))
    assert result.state.charge_C[0] == pytest.approx(expected, rel=2.0e-10, abs=1.0e-30)


@pytest.mark.parametrize("unresolved_kind", ["geometry", "local_error"])
def test_zero_time_terminal_restores_dynamic_charge_with_electric_force(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unresolved_kind: str,
) -> None:
    case = load_case(
        _write_public_charge_case(
            tmp_path / "electric-geometry-start-stop",
            release_time_s=0.0,
            electric_force_enabled=True,
        )
    )
    monkeypatch.setattr(
        runtime_module,
        "_refine_deterministic_stage_traces",
        lambda **_kwargs: {0: unresolved_kind},
    )

    result = simulate(case)

    assert result.state.terminal_state.tolist() == ["numerical_boundary_stopped"]
    assert result.state.charge_C.tolist() == [0.0]
