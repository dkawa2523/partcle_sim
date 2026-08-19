from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from particle_tracer_unified import load_case, validate_case
from particle_tracer_unified.solvers.contact_sliding import advance_contact_relaxation
from particle_tracer_unified.solvers.drag_models import (
    CONTINUUM_DRAG_SCHILLER_NAUMANN,
    CONTINUUM_DRAG_STOKES,
    DRAG_MODEL_SCHILLER_NAUMANN,
    DRAG_MODEL_STOKES,
    DRAG_MODEL_STOKES_CUNNINGHAM,
    RAREFACTION_CUNNINGHAM,
    RAREFACTION_EPSTEIN,
    RAREFACTION_NONE,
    drag_model_gas_requirements,
    drag_model_structure_from_name,
    effective_tau_from_drag_components,
    effective_tau_from_drag_model,
)
from particle_tracer_unified.solvers.drag_regime import (
    BOLTZMANN_J_K,
    classify_drag_regime,
    gas_mean_free_path_m,
    particle_reynolds_number,
    relative_knudsen_number,
    relative_mach_number,
)
from particle_tracer_unified.solvers.integrator_common import (
    _EPSTEIN_DEFAULT_ACCOMMODATION_DELTA,
    effective_tau_from_slip_speed,
    stokes_relaxation_time,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "v02_minimal"
AMU_KG = 1.66053906660e-27


def _copy_example(tmp_path: Path) -> Path:
    target = tmp_path / "case"
    shutil.copytree(EXAMPLE, target)
    return target / "run_config.yaml"


@pytest.mark.parametrize(
    ("name", "continuum", "rarefaction"),
    [
        ("stokes", "stokes", "none"),
        ("stokes_cunningham", "stokes", "cunningham"),
        ("schiller_naumann", "schiller_naumann", "none"),
        ("epstein", "stokes", "epstein"),
        ("none", "none", "none"),
    ],
)
def test_legacy_drag_names_resolve_to_orthogonal_internal_components(
    name: str,
    continuum: str,
    rarefaction: str,
) -> None:
    structure = drag_model_structure_from_name(name)

    assert structure.continuum_law == continuum
    assert structure.rarefaction_correction == rarefaction


def test_schiller_naumann_and_cunningham_factors_compose_independently() -> None:
    tau_stokes = 2.0
    slip_speed = 0.2
    diameter = 1.0e-3
    density = 1.0
    viscosity = 1.0e-3
    temperature = 300.0
    molecular_mass = 28.0 * AMU_KG
    mass = tau_stokes * 3.0 * np.pi * viscosity * diameter
    mean_free_path = gas_mean_free_path_m(
        np.asarray([viscosity]),
        np.asarray([density]),
        np.asarray([temperature]),
        np.asarray([molecular_mass]),
    )[0]
    knudsen = mean_free_path / diameter
    reynolds = density * diameter * slip_speed / viscosity
    cunningham = 1.0 + knudsen * (2.514 + 0.8 * np.exp(-0.55 / knudsen))
    schiller_naumann = 1.0 + 0.15 * reynolds**0.687

    actual = effective_tau_from_drag_components(
        tau_stokes,
        slip_speed,
        diameter,
        density,
        viscosity,
        CONTINUUM_DRAG_SCHILLER_NAUMANN,
        RAREFACTION_CUNNINGHAM,
        mass,
        temperature,
        molecular_mass,
        _EPSTEIN_DEFAULT_ACCOMMODATION_DELTA,
    )

    assert actual == pytest.approx(
        tau_stokes * cunningham / schiller_naumann,
        rel=2.0e-15,
    )


@pytest.mark.parametrize(
    "mode",
    [DRAG_MODEL_STOKES, DRAG_MODEL_SCHILLER_NAUMANN, DRAG_MODEL_STOKES_CUNNINGHAM],
)
def test_continuum_drag_uses_stage_viscosity_in_stokes_base(mode: int) -> None:
    mass = 2.0e-15
    diameter = 1.0e-6
    reference_viscosity = 1.8e-5
    stage_viscosity = 3.6e-5
    density = 1.2
    temperature = 300.0
    molecular_mass = 28.0 * AMU_KG
    slip_speed = 0.2
    tau_reference = stokes_relaxation_time(
        mass,
        reference_viscosity,
        diameter,
    )
    tau_stage = mass / (3.0 * np.pi * stage_viscosity * diameter)
    continuum_factor = 1.0
    rarefaction_factor = 1.0
    if mode == DRAG_MODEL_SCHILLER_NAUMANN:
        reynolds = density * diameter * slip_speed / stage_viscosity
        continuum_factor = 1.0 + 0.15 * reynolds**0.687
    elif mode == DRAG_MODEL_STOKES_CUNNINGHAM:
        mean_free_path = gas_mean_free_path_m(
            np.asarray([stage_viscosity]),
            np.asarray([density]),
            np.asarray([temperature]),
            np.asarray([molecular_mass]),
        )[0]
        knudsen = mean_free_path / diameter
        rarefaction_factor = 1.0 + knudsen * (2.514 + 0.8 * np.exp(-0.55 / knudsen))

    actual = effective_tau_from_drag_model(
        tau_reference,
        slip_speed,
        diameter,
        density,
        stage_viscosity,
        mode,
        mass,
        temperature,
        molecular_mass,
        _EPSTEIN_DEFAULT_ACCOMMODATION_DELTA,
    )

    assert actual == pytest.approx(
        tau_stage * rarefaction_factor / continuum_factor,
        rel=2.0e-15,
    )


def test_constant_viscosity_keeps_reference_stokes_tau_bit_exact() -> None:
    mass = 2.0e-15
    diameter = 1.0e-6
    viscosity = 1.8e-5
    tau_reference = stokes_relaxation_time(mass, viscosity, diameter)

    actual = effective_tau_from_drag_model(
        tau_reference,
        0.2,
        diameter,
        1.2,
        viscosity,
        DRAG_MODEL_STOKES,
        mass,
        300.0,
        28.0 * AMU_KG,
        _EPSTEIN_DEFAULT_ACCOMMODATION_DELTA,
    )

    assert actual == tau_reference


def test_drag_component_gas_requirements_preserve_legacy_inputs() -> None:
    assert drag_model_gas_requirements("stokes") == ("dynamic_viscosity_Pas",)
    assert drag_model_gas_requirements("stokes_cunningham") == (
        "temperature_K",
        "dynamic_viscosity_Pas",
        "density_kgm3",
        "molecular_mass_amu",
    )
    assert drag_model_gas_requirements("schiller_naumann") == (
        "dynamic_viscosity_Pas",
        "density_kgm3",
    )
    assert drag_model_gas_requirements("epstein") == (
        "temperature_K",
        "density_kgm3",
        "molecular_mass_amu",
    )
    assert drag_model_gas_requirements("none") == ()


def test_existing_public_models_use_only_their_declared_component_pair() -> None:
    assert drag_model_structure_from_name("stokes").mode_pair == (
        CONTINUUM_DRAG_STOKES,
        RAREFACTION_NONE,
    )
    assert drag_model_structure_from_name("stokes_cunningham").mode_pair == (
        CONTINUUM_DRAG_STOKES,
        RAREFACTION_CUNNINGHAM,
    )
    assert drag_model_structure_from_name("epstein").mode_pair == (
        CONTINUUM_DRAG_STOKES,
        RAREFACTION_EPSTEIN,
    )


def test_dimensionless_drag_metrics_use_declared_diameter_and_relative_velocity() -> (
    None
):
    slip = np.asarray([3.0])
    diameter = np.asarray([2.0e-6])
    density = np.asarray([0.4])
    viscosity = np.asarray([2.0e-5])
    temperature = np.asarray([400.0])
    molecular_mass = np.asarray([28.0 * AMU_KG])

    reynolds = particle_reynolds_number(slip, diameter, density, viscosity)
    mean_free_path = gas_mean_free_path_m(
        viscosity, density, temperature, molecular_mass
    )
    knudsen = relative_knudsen_number(mean_free_path, diameter)
    mach = relative_mach_number(slip, np.asarray([300.0]))

    assert reynolds[0] == pytest.approx(0.12)
    assert mean_free_path[0] == pytest.approx(
        (viscosity[0] / density[0])
        * np.sqrt(np.pi * molecular_mass[0] / (2.0 * BOLTZMANN_J_K * temperature[0]))
    )
    assert knudsen[0] == pytest.approx(mean_free_path[0] / diameter[0])
    assert mach[0] == pytest.approx(0.01)


@pytest.mark.parametrize(
    ("model", "reynolds", "knudsen", "mach", "error", "warning"),
    [
        ("stokes", 1.0, 1.0e-3, 0.0, "particle_reynolds_outside_creeping_flow", None),
        ("stokes", 1.0e-2, 2.0e-2, 0.0, None, "knudsen_requires_rarefaction_review"),
        (
            "stokes_cunningham",
            1.0e-2,
            12.0,
            0.0,
            None,
            "knudsen_free_molecular_epstein_review",
        ),
        (
            "schiller_naumann",
            800.0,
            1.0e-3,
            0.0,
            "particle_reynolds_outside_schiller_naumann",
            None,
        ),
        ("epstein", 0.0, 1.0, 0.0, "knudsen_outside_free_molecular_flow", None),
        (
            "epstein",
            0.0,
            5.0,
            0.5,
            None,
            "knudsen_transitional_not_asymptotic_free_molecular",
        ),
        (
            "stokes",
            1.0e-2,
            1.0e-3,
            1.0,
            "relative_mach_supersonic_drag_not_supported",
            None,
        ),
    ],
)
def test_drag_regime_policy_is_explicit_and_never_switches_model(
    model: str,
    reynolds: float,
    knudsen: float,
    mach: float,
    error: str | None,
    warning: str | None,
) -> None:
    decision = classify_drag_regime(
        model,
        reynolds=reynolds,
        knudsen=knudsen,
        relative_mach=mach,
    )

    if error is not None:
        assert error in decision.errors
    if warning is not None:
        assert warning in decision.warnings


def test_schiller_naumann_runtime_does_not_extend_past_published_range() -> None:
    with pytest.raises(ValueError, match="Reynolds number < 800"):
        effective_tau_from_slip_speed(
            1.0,
            800.0,
            1.0,
            1.0,
            1.0,
            DRAG_MODEL_SCHILLER_NAUMANN,
            1.0,
            300.0,
            39.948 * AMU_KG,
            _EPSTEIN_DEFAULT_ACCOMMODATION_DELTA,
        )


def test_contact_relaxation_keeps_physical_tau_below_one_picosecond() -> None:
    tau = 1.0e-15
    dt = 0.7e-15
    velocity0 = 2.0
    target = 0.5
    body = 3.0e12
    decay = np.exp(-dt / tau)
    one_minus_decay = 1.0 - decay
    response_integral = dt - tau * one_minus_decay
    expected_velocity = (
        target + (velocity0 - target) * decay + body * tau * one_minus_decay
    )
    expected_displacement = (
        velocity0 * dt
        + (target - velocity0) * response_integral
        + body * tau * response_integral
    )

    displacement, velocity = advance_contact_relaxation(
        np.asarray([velocity0]),
        np.asarray([target]),
        np.asarray([body]),
        np.asarray([tau]),
        dt,
    )

    assert velocity[0] == pytest.approx(expected_velocity, rel=1.0e-14)
    assert displacement[0] == pytest.approx(expected_displacement, rel=1.0e-14)


def test_contact_relaxation_has_explicit_ballistic_and_invalid_tau_contracts() -> None:
    displacement, velocity = advance_contact_relaxation(
        np.asarray([2.0]),
        np.asarray([999.0]),
        np.asarray([3.0]),
        np.asarray([np.inf]),
        0.4,
    )

    assert displacement[0] == pytest.approx(2.0 * 0.4 + 0.5 * 3.0 * 0.4**2)
    assert velocity[0] == pytest.approx(2.0 + 3.0 * 0.4)
    with pytest.raises(ValueError, match="relaxation time"):
        advance_contact_relaxation(
            np.asarray([0.0]),
            np.asarray([0.0]),
            np.asarray([0.0]),
            np.asarray([0.0]),
            1.0,
        )


def test_preflight_reports_release_state_scope_without_standard_history_diagnostics(
    tmp_path: Path,
) -> None:
    config_path = _copy_example(tmp_path)

    summary = validate_case(load_case(config_path), detail="summary")
    full = validate_case(load_case(config_path), detail="full")

    assert summary.passed
    assert summary.checks["drag_regime"]["scope"] == "initial_release_state"
    assert summary.checks["drag_regime"]["dynamic_history_assessed"] is False
    assert "violations" not in summary.checks["drag_regime"]
    assert full.checks["drag_regime"]["violations"]
    assert full.checks["drag_regime"]["relative_mach_assessed_count"] == 0


def test_preflight_rejects_unverifiable_drag_regime_instead_of_assuming_air(
    tmp_path: Path,
) -> None:
    config_path = _copy_example(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["physics"]["gas"].pop("molecular_mass_amu")
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    report = validate_case(load_case(config_path), detail="full")

    assert not report.passed
    issue = next(item for item in report.errors if item.code == "physics.drag.regime")
    assert issue.context["reason_counts"] == {"regime_inputs_unavailable": 2}
    assert report.checks["drag_regime"]["assessed_particle_count"] == 0


def test_preflight_requires_flow_velocity_for_active_drag(tmp_path: Path) -> None:
    config_path = _copy_example(tmp_path)
    field_path = config_path.parent / "field_without_velocity.npz"
    axis = np.linspace(-1.0, 1.0, 41)
    np.savez(
        field_path,
        axis_0=axis,
        axis_1=axis,
        times=np.asarray([0.0]),
        valid_mask=np.ones((41, 41), dtype=bool),
        pressure=np.ones((41, 41), dtype=np.float64),
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["inputs"]["field"] = {
        "kind": "precomputed_npz",
        "path": field_path.name,
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    report = validate_case(load_case(config_path))

    assert not report.passed
    missing_flow = [
        issue
        for issue in report.errors
        if issue.code == "physics.force.field.missing"
        and issue.context.get("feature") == "drag"
    ]
    assert len(missing_flow) == 1


def test_preflight_uses_only_explicit_sound_speed_for_relative_mach(
    tmp_path: Path,
) -> None:
    config_path = _copy_example(tmp_path)
    field_path = config_path.parent / "field_with_sound_speed.npz"
    axis = np.linspace(-1.0, 1.0, 41)
    shape = (41, 41)
    np.savez(
        field_path,
        axis_0=axis,
        axis_1=axis,
        times=np.asarray([0.0]),
        valid_mask=np.ones(shape, dtype=bool),
        ux=np.zeros(shape),
        uy=np.zeros(shape),
        sound_speed=np.full(shape, 0.05),
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["inputs"]["field"] = {
        "kind": "precomputed_npz",
        "path": field_path.name,
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    report = validate_case(load_case(config_path), detail="full")

    assert not report.passed
    drag_check = report.checks["drag_regime"]
    assert drag_check["relative_mach_assessed_count"] == 2
    assert drag_check["error_reason_counts"] == {
        "relative_mach_supersonic_drag_not_supported": 2
    }


def test_preflight_rejects_schiller_naumann_at_release_above_re_800(
    tmp_path: Path,
) -> None:
    config_path = _copy_example(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["physics"]["drag"]["model"] = "schiller_naumann"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    particles_path = config_path.parent / "particles.csv"
    particles = pd.read_csv(particles_path)
    particles.loc[:, "vx_mps"] = 20_000.0
    particles.to_csv(particles_path, index=False)

    report = validate_case(load_case(config_path), detail="full")

    assert not report.passed
    assert report.checks["drag_regime"]["error_reason_counts"] == {
        "particle_reynolds_outside_schiller_naumann": 2
    }


def test_preflight_samples_transient_drag_field_at_each_release_time(
    tmp_path: Path,
) -> None:
    config_path = _copy_example(tmp_path)
    particles_path = config_path.parent / "particles.csv"
    particles = pd.read_csv(particles_path)
    particles.loc[:, "release_time_s"] = [0.0, 0.5]
    particles.loc[:, "vx_mps"] = 0.1
    particles.to_csv(particles_path, index=False)

    axis = np.linspace(-1.0, 1.0, 41)
    shape = (2, axis.size, axis.size)
    ux = np.empty(shape, dtype=np.float64)
    ux[0] = 0.0
    ux[1] = 1.0
    field_path = config_path.parent / "transient_drag_field.npz"
    np.savez(
        field_path,
        axis_0=axis,
        axis_1=axis,
        times=np.asarray([0.0, 1.0]),
        valid_mask=np.ones((axis.size, axis.size), dtype=bool),
        ux=ux,
        uy=np.zeros(shape, dtype=np.float64),
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["inputs"]["field"] = {
        "kind": "precomputed_npz",
        "path": field_path.name,
    }
    config["time"] = {"dt": 0.1, "t_end": 1.0}
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    report = validate_case(load_case(config_path), detail="full")

    rows = report.checks["drag_regime"]["violations"]
    by_particle = {row["particle_id"]: row for row in rows}
    assert by_particle[1]["particle_reynolds"] == pytest.approx(
        1.2 * 1.0e-6 * 0.1 / 1.8e-5
    )
    assert by_particle[2]["particle_reynolds"] == pytest.approx(
        1.2 * 1.0e-6 * 0.4 / 1.8e-5
    )
    assert by_particle[2]["release_time_s"] == pytest.approx(0.5)
