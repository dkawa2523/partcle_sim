from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from field_backend_helpers import advance_motion_batch_into

from particle_tracer_unified.core.boundary_service import build_boundary_service
from particle_tracer_unified.core.datamodel import (
    FieldProviderND,
    GasProperties,
    GeometryND,
    GeometryProviderND,
    QuantitySeriesND,
    RegularFieldND,
    WallCatalog,
    WallPartModel,
)
from particle_tracer_unified.core.field_sampling import VALID_MASK_STATUS_CLEAN
from particle_tracer_unified.core.geometry2d import build_boundary_loops_2d
from particle_tracer_unified.solvers._charge_oml import (
    oml_linearized_equilibrium,
    te_relaxation_equilibrium,
)
from particle_tracer_unified.solvers._collision_particle import (
    advance_colliding_particle,
)
from particle_tracer_unified.solvers._runtime_preparation import (
    _require_particle_density_for_displaced_fluid_forces,
)
from particle_tracer_unified.solvers.charge_model import (
    AMU_KG,
    E_CHARGE_C,
    ELECTRON_MASS_KG,
    ChargeModelConfig,
    apply_charge_model_update,
)
from particle_tracer_unified.solvers.diagnostics import (
    increment_count,
    initial_collision_diagnostics,
)
from particle_tracer_unified.solvers.drag_regime import BOLTZMANN_J_K
from particle_tracer_unified.solvers.field_compilation import compile_runtime_backend
from particle_tracer_unified.solvers.forces import ForceRuntimeParameters
from particle_tracer_unified.solvers.integrator_common import (
    _EPSTEIN_DEFAULT_ACCOMMODATION_DELTA,
    DRAG_MODEL_EPSTEIN,
    DRAG_MODEL_NONE,
    DRAG_MODEL_SCHILLER_NAUMANN,
    DRAG_MODEL_STOKES,
    DRAG_MODEL_STOKES_CUNNINGHAM,
    advance_component,
    effective_tau_from_slip_speed,
    epstein_relaxation_time,
    stokes_relaxation_time,
)
from particle_tracer_unified.solvers.stochastic_motion import (
    PiecewiseLangevinPath,
)


def test_stokes_tau_uses_authoritative_mass_not_material_density() -> None:
    mass = 2.3e-15
    diameter = 1.7e-6
    viscosity = 1.81e-5
    expected = mass / (3.0 * np.pi * viscosity * diameter)

    assert stokes_relaxation_time(mass, viscosity, diameter) == pytest.approx(
        expected, rel=1e-15
    )


def test_standard_diagnostics_store_only_safety_counters() -> None:
    diagnostics = initial_collision_diagnostics(debug=False)
    increment_count(diagnostics, "primary_hit_count")
    increment_count(diagnostics, "unresolved_crossing_count")
    diagnostics["charge_model"] = {"enabled": 1}

    assert "primary_hit_count" not in diagnostics
    assert "charge_model" not in diagnostics
    assert diagnostics["unresolved_crossing_count"] == 1
    assert len(diagnostics) < 20


def test_displaced_fluid_forces_require_explicit_particle_density() -> None:
    with pytest.raises(
        ValueError, match=r"pressure_gradient.*invalid particle IDs: \[8\]"
    ):
        _require_particle_density_for_displaced_fluid_forces(
            np.asarray([1200.0, np.nan]),
            np.asarray([7, 8]),
            ForceRuntimeParameters(pressure_gradient_enabled=True),
        )

    _require_particle_density_for_displaced_fluid_forces(
        np.asarray([np.nan]),
        np.asarray([9]),
        ForceRuntimeParameters(),
    )


def test_etd_constant_coefficient_solution_is_analytic() -> None:
    velocity0 = 3.2
    target = -0.7
    acceleration = 1.4
    tau = 0.083
    dt = 0.37
    decay = np.exp(-dt / tau)
    terminal_velocity = target + acceleration * tau
    expected_velocity = terminal_velocity + (velocity0 - terminal_velocity) * decay
    expected_displacement = terminal_velocity * dt + (
        velocity0 - terminal_velocity
    ) * tau * (1.0 - decay)

    displacement, velocity = advance_component(velocity0, target, acceleration, tau, dt)

    assert velocity == pytest.approx(expected_velocity, rel=1e-13, abs=1e-15)
    assert displacement == pytest.approx(expected_displacement, rel=1e-13, abs=1e-15)


def _integrate_sinusoidal_target(
    dt: float, *, t_end: float, tau: float, velocity0: float
) -> tuple[float, float]:
    position = 0.0
    velocity = float(velocity0)
    time_s = 0.0
    while time_s < t_end - 1e-15:
        step = min(float(dt), t_end - time_s)
        target_midpoint = np.sin(time_s + 0.5 * step)
        displacement, velocity = advance_component(
            velocity,
            target_midpoint,
            0.0,
            tau,
            step,
        )
        position += displacement
        time_s += step
    return position, velocity


def test_etd2_midpoint_coefficients_show_second_order_convergence() -> None:
    tau = 0.37
    velocity0 = 0.23
    t_end = 1.0
    a = 1.0 / (1.0 + tau * tau)
    b = -tau / (1.0 + tau * tau)
    transient = velocity0 - b
    exact_velocity = (
        a * np.sin(t_end) + b * np.cos(t_end) + transient * np.exp(-t_end / tau)
    )
    exact_position = (
        a * (1.0 - np.cos(t_end))
        + b * np.sin(t_end)
        + transient * tau * (1.0 - np.exp(-t_end / tau))
    )
    coarse = np.asarray(
        _integrate_sinusoidal_target(0.1, t_end=t_end, tau=tau, velocity0=velocity0)
    )
    fine = np.asarray(
        _integrate_sinusoidal_target(0.05, t_end=t_end, tau=tau, velocity0=velocity0)
    )
    exact = np.asarray([exact_position, exact_velocity])
    coarse_error = float(np.linalg.norm(coarse - exact))
    fine_error = float(np.linalg.norm(fine - exact))

    assert coarse_error / fine_error >= 3.5


def test_drag_none_is_exact_ballistic() -> None:
    displacement, velocity = advance_component(2.0, 99.0, -3.0, np.inf, 0.4)

    assert displacement == pytest.approx(2.0 * 0.4 - 0.5 * 3.0 * 0.4**2, abs=1e-15)
    assert velocity == pytest.approx(2.0 - 3.0 * 0.4, abs=1e-15)
    assert (
        effective_tau_from_slip_speed(
            1.0,
            0.0,
            1e-6,
            1.0,
            1e-5,
            DRAG_MODEL_NONE,
            1e-15,
            300.0,
            4.65e-26,
            _EPSTEIN_DEFAULT_ACCOMMODATION_DELTA,
        )
        == np.inf
    )


def test_epstein_tau_uses_mass_directly_and_does_not_require_viscosity() -> None:
    mass = 2.7e-15
    diameter = 1.2e-6
    gas_density = 2.3e-5
    temperature = 420.0
    molecular_mass = 39.948 * AMU_KG
    thermal_speed = np.sqrt(
        8.0 * BOLTZMANN_J_K * temperature / (np.pi * molecular_mass)
    )
    expected = (
        3.0
        * mass
        / ((1.0 + np.pi / 8.0) * np.pi * diameter**2 * gas_density * thermal_speed)
    )
    public = epstein_relaxation_time(
        mass, gas_density, temperature, diameter, molecular_mass
    )
    deterministic = effective_tau_from_slip_speed(
        np.nan,
        13.0,
        diameter,
        gas_density,
        np.nan,
        DRAG_MODEL_EPSTEIN,
        mass,
        temperature,
        molecular_mass,
        _EPSTEIN_DEFAULT_ACCOMMODATION_DELTA,
    )
    assert public == pytest.approx(expected, rel=1e-15)
    assert deterministic == pytest.approx(expected, rel=1e-15)


def test_effective_tau_dispatch_preserves_regime_boundaries_and_formulas() -> None:
    tau_stokes = 0.37
    diameter = 1.2e-6
    gas_density = 0.8
    gas_mu = 1.9e-5
    temperature = 420.0
    molecular_mass = 39.948 * AMU_KG
    mass = tau_stokes * 3.0 * np.pi * gas_mu * diameter
    mean_free_path = (gas_mu / gas_density) * np.sqrt(
        np.pi * molecular_mass / (2.0 * BOLTZMANN_J_K * temperature)
    )
    knudsen = mean_free_path / diameter
    expected_cunningham = tau_stokes * (
        1.0 + knudsen * (2.514 + 0.8 * np.exp(-0.55 / knudsen))
    )

    assert (
        effective_tau_from_slip_speed(
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            DRAG_MODEL_NONE,
            np.nan,
            np.nan,
            np.nan,
            _EPSTEIN_DEFAULT_ACCOMMODATION_DELTA,
        )
        == np.inf
    )
    assert effective_tau_from_slip_speed(
        tau_stokes,
        np.nan,
        diameter,
        gas_density,
        gas_mu,
        DRAG_MODEL_STOKES,
        mass,
        temperature,
        molecular_mass,
        _EPSTEIN_DEFAULT_ACCOMMODATION_DELTA,
    ) == pytest.approx(tau_stokes, rel=2.0e-16)
    assert effective_tau_from_slip_speed(
        tau_stokes,
        1.0,
        diameter,
        gas_density,
        gas_mu,
        DRAG_MODEL_STOKES_CUNNINGHAM,
        mass,
        temperature,
        molecular_mass,
        _EPSTEIN_DEFAULT_ACCOMMODATION_DELTA,
    ) == pytest.approx(expected_cunningham, rel=1.0e-15)
    assert effective_tau_from_slip_speed(
        tau_stokes,
        -1.0,
        diameter,
        gas_density,
        gas_mu,
        DRAG_MODEL_SCHILLER_NAUMANN,
        mass,
        temperature,
        molecular_mass,
        _EPSTEIN_DEFAULT_ACCOMMODATION_DELTA,
    ) == pytest.approx(tau_stokes, rel=2.0e-16)


@pytest.mark.parametrize(
    ("drag_model_mode", "gas_density", "gas_mu", "temperature", "molecular_mass"),
    [
        (DRAG_MODEL_STOKES_CUNNINGHAM, np.nan, 1.8e-5, 300.0, 4.65e-26),
        (DRAG_MODEL_STOKES_CUNNINGHAM, 1.0, 1.8e-5, np.nan, 4.65e-26),
        (DRAG_MODEL_SCHILLER_NAUMANN, 1.0, np.nan, 300.0, 4.65e-26),
        (999, 1.0, 1.8e-5, 300.0, 4.65e-26),
    ],
)
def test_drag_tau_rejects_missing_gas_values_and_unknown_modes(
    drag_model_mode: int,
    gas_density: float,
    gas_mu: float,
    temperature: float,
    molecular_mass: float,
) -> None:
    tau = effective_tau_from_slip_speed(
        0.01,
        1.0,
        1.0e-6,
        gas_density,
        gas_mu,
        drag_model_mode,
        1.0e-15,
        temperature,
        molecular_mass,
        _EPSTEIN_DEFAULT_ACCOMMODATION_DELTA,
    )

    assert np.isnan(tau)


def _square_collision_runtime() -> SimpleNamespace:
    edges = np.asarray(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [1.0, 1.0]],
            [[1.0, 1.0], [0.0, 1.0]],
            [[0.0, 1.0], [0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    axes = (np.asarray([0.0, 0.5, 1.0]), np.asarray([0.0, 0.5, 1.0]))
    valid = np.ones((3, 3), dtype=bool)
    geometry = GeometryND(
        spatial_dim=2,
        coordinate_system="cartesian_xy",
        axes=axes,
        valid_mask=valid,
        sdf=-np.ones((3, 3), dtype=np.float64),
        normal_components=(np.zeros((3, 3)), np.ones((3, 3))),
        nearest_boundary_part_id_map=np.ones((3, 3), dtype=np.int32),
        boundary_edges=edges,
        boundary_edge_part_ids=np.asarray([1, 2, 3, 4], dtype=np.int32),
        boundary_loops_2d=build_boundary_loops_2d(edges),
    )
    models = tuple(
        WallPartModel(
            part_id=part_id,
            part_name=f"wall_{part_id}",
            material_id=part_id,
            material_name="test",
            law_name="specular",
            stick_probability=0.0,
            restitution=1.0,
            diffuse_fraction=0.0,
            critical_sticking_velocity_mps=0.0,
        )
        for part_id in (1, 2, 3, 4)
    )
    return SimpleNamespace(
        spatial_dim=2,
        coordinate_system="cartesian_xy",
        geometry_provider=GeometryProviderND(geometry=geometry, kind="test"),
        field_provider=None,
        wall_catalog=WallCatalog(part_models=models),
        gas=GasProperties(),
    )


def test_constant_electric_force_matches_ballistic_analytic_solution() -> None:
    runtime = _square_collision_runtime()
    axes = runtime.geometry_provider.geometry.axes
    zeros = np.zeros((3, 3), dtype=np.float64)
    times = np.asarray([0.0], dtype=np.float64)
    runtime.field_provider = FieldProviderND(
        field=RegularFieldND(
            spatial_dim=2,
            coordinate_system="cartesian_xy",
            axis_names=("x", "y"),
            axes=axes,
            quantities={
                "ux": QuantitySeriesND(
                    "ux", "m/s", times=times, data=zeros, metadata={}
                ),
                "uy": QuantitySeriesND(
                    "uy", "m/s", times=times, data=zeros, metadata={}
                ),
                "Ex": QuantitySeriesND(
                    "Ex", "V/m", times=times, data=np.full((3, 3), 2.0), metadata={}
                ),
                "Ey": QuantitySeriesND(
                    "Ey", "V/m", times=times, data=np.full((3, 3), -3.0), metadata={}
                ),
            },
            valid_mask=np.ones((3, 3), dtype=bool),
            time_mode="steady",
        ),
        kind="test",
    )
    compiled = compile_runtime_backend(runtime, spatial_dim=2, enable_electric=True)
    position0 = np.asarray([[0.5, 0.5]], dtype=np.float64)
    velocity0 = np.asarray([[0.1, -0.2]], dtype=np.float64)
    position1 = np.zeros_like(position0)
    velocity1 = np.zeros_like(velocity0)
    midpoint = np.zeros_like(position0)
    dt = 0.1
    q_over_m = 0.4
    acceleration = q_over_m * np.asarray([2.0, -3.0])
    advance_motion_batch_into(
        spatial_dim=2,
        compiled=compiled,
        x=position0,
        v=velocity0,
        active=np.asarray([True]),
        tau_p=np.asarray([np.inf]),
        particle_diameter=np.asarray([1.0e-6]),
        particle_mass=np.asarray([1.0e-15]),
        particle_density=np.asarray([100.0]),
        t=dt,
        dt_step=dt,
        phys={
            "gas_temperature_K": np.nan,
            "gas_molecular_mass_kg": np.nan,
        },
        body_accel=np.zeros(2),
        gas_density_kgm3=np.nan,
        gas_mu_pas=np.nan,
        drag_model_mode=DRAG_MODEL_NONE,
        adaptive_substep_enabled=0,
        adaptive_substep_max_splits=4,
        x_trial=position1,
        v_trial=velocity1,
        x_mid_trial=midpoint,
        substep_counts=np.ones(1, dtype=np.int32),
        valid_mask_status_flags=np.zeros(1, dtype=np.uint8),
        electric_q_over_m_particle=np.asarray([q_over_m]),
    )

    expected_velocity = velocity0[0] + acceleration * dt
    expected_position = position0[0] + velocity0[0] * dt + 0.5 * acceleration * dt**2
    np.testing.assert_allclose(velocity1[0], expected_velocity, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(position1[0], expected_position, rtol=1e-12, atol=1e-14)


def _run_saved_brownian_crossing() -> tuple[object, list[dict[str, object]]]:
    runtime = _square_collision_runtime()
    compiled = compile_runtime_backend(runtime, spatial_dim=2)
    service = build_boundary_service(
        runtime,
        spatial_dim=2,
        on_boundary_tol_m=1e-10,
        triangle_surface_3d=None,
    )
    path = PiecewiseLangevinPath(
        leaf_end_times_s=np.asarray([1.0]),
        tau_eff_s=np.asarray([0.05]),
        thermal_velocity_variance_m2s2=np.asarray([20.0]),
        z_velocity=np.asarray([[0.9775674511260357, 0.0]]),
        z_position=np.asarray([[-0.31055654665915255, 0.0]]),
        bridge_seeds=np.asarray([0], dtype=np.int64),
    )
    x_start = np.asarray([0.95, 0.5])
    v_start = np.zeros(2)
    fractions = np.linspace(0.125, 1.0, 8)
    stage_points = np.asarray(
        [x_start + path.state_at(float(fraction))[0] for fraction in fractions],
        dtype=np.float64,
    )
    initial_hit = service.polyline_hit(x_start, stage_points)
    assert initial_hit is not None
    wall_rows: list[dict[str, object]] = []
    result = advance_colliding_particle(
        runtime=runtime,
        particles=None,
        particle_index=0,
        rng=np.random.default_rng(44),
        t=1.0,
        x_start=x_start,
        v_start=v_start,
        dt_step=1.0,
        spatial_dim=2,
        compiled=compiled,
        base_adaptive_substep_enabled=0,
        adaptive_substep_max_splits=4,
        tau_p_i=0.05,
        particle_diameter_i=1e-6,
        particle_density_i=1000.0,
        particle_mass_i=0.05 * 3.0 * np.pi * 1.8e-5 * 1.0e-6,
        particle_id_i=1,
        body_accel=np.zeros(2),
        gas_density_kgm3=1.0,
        gas_mu_pas=1.8e-5,
        drag_model_mode=0,
        initial_x_next=stage_points[-1],
        initial_v_next=path.state_at(1.0)[1],
        initial_stage_points=stage_points,
        initial_valid_mask_status=int(VALID_MASK_STATUS_CLEAN),
        initial_primary_hit=initial_hit,
        initial_primary_hit_counted=False,
        inside_fn=service.inside,
        strict_inside_fn=service.inside_strict,
        primary_hit_fn=service.polyline_hit,
        nearest_projection_fn=service.nearest_projection,
        primary_hit_counter_key=service.primary_hit_counter_key,
        collision_diagnostics=initial_collision_diagnostics(debug=True),
        max_hit_rows=[],
        wall_rows=wall_rows,
        wall_summary_counts={},
        stuck=np.asarray([False]),
        absorbed=np.asarray([False]),
        escaped=np.asarray([False]),
        active=np.asarray([True]),
        max_wall_hits_per_step=5,
        epsilon_offset_m=1e-8,
        on_boundary_tol_m=1e-10,
        triangle_surface_3d=None,
        stochastic_path=path,
    )
    return result, wall_rows


def test_saved_brownian_path_detects_first_passage_and_replays_reproducibly() -> None:
    first, first_rows = _run_saved_brownian_crossing()
    second, second_rows = _run_saved_brownian_crossing()

    assert first.total_hits >= 1
    assert first_rows
    # The node-addressed bridge first reaches the upper wall before the right wall.
    assert first_rows[0]["part_id"] == 3
    assert 0.0 <= float(first.position[0]) <= 1.0 + 1e-10
    np.testing.assert_array_equal(first.position, second.position)
    np.testing.assert_array_equal(first.velocity, second.velocity)
    assert [(row["part_id"], row["outcome"]) for row in first_rows] == [
        (row["part_id"], row["outcome"]) for row in second_rows
    ]
    assert [row["hit_time_s"] for row in first_rows] == pytest.approx(
        [row["hit_time_s"] for row in second_rows],
        abs=0.0,
    )


def test_oml_equilibrium_uses_a_converged_bracketed_root() -> None:
    config = ChargeModelConfig(
        enabled=True,
        mode="oml_linearized_relaxation",
        ion_mass_amu=40.0,
        root_iterations=80,
    )
    radius = np.asarray([0.5e-6])
    te = np.asarray([3.0])
    ne = np.asarray([1.0e16])
    ni = np.asarray([1.0e16])
    ti = np.asarray([0.03])
    _charge, tau_q, potential = oml_linearized_equilibrium(
        config,
        radius,
        te,
        ne,
        ni,
        ti,
    )
    electron_speed = np.sqrt(E_CHARGE_C * te / (2.0 * np.pi * ELECTRON_MASS_KG))
    ion_mass = config.ion_mass_amu * AMU_KG
    ion_speed = np.maximum(
        np.sqrt(E_CHARGE_C * ti / (2.0 * np.pi * ion_mass)),
        np.sqrt(E_CHARGE_C * te / ion_mass),
    )
    electron_flux = ne * electron_speed * np.exp(potential / te)
    ion_flux = ni * ion_speed * (1.0 - potential / ti)
    relative_residual = np.abs(ion_flux - electron_flux) / np.maximum(
        ion_flux + electron_flux, 1.0
    )

    assert float(relative_residual[0]) < 1e-13
    assert -config.max_abs_potential_V <= float(potential[0]) <= 0.0
    assert float(tau_q[0]) > 0.0


def test_oml_linearized_tau_matches_current_derivative_without_configured_cap() -> None:
    config = ChargeModelConfig(
        enabled=True,
        mode="oml_linearized_relaxation",
        ion_mass_amu=40.0,
        ion_charge_number=2.0,
        relaxation_time_s=1.0e-12,
        root_iterations=80,
    )
    radius = np.asarray([0.5e-6])
    te = np.asarray([3.0])
    ne = np.asarray([1.0e16])
    ni = np.asarray([1.0e16])
    ti = np.asarray([0.03])
    charge, tau_q, potential = oml_linearized_equilibrium(
        config,
        radius,
        te,
        ne,
        ni,
        ti,
    )
    ion_mass = config.ion_mass_amu * AMU_KG
    electron_speed = np.sqrt(E_CHARGE_C * te / (2.0 * np.pi * ELECTRON_MASS_KG))
    ion_speed = np.maximum(
        np.sqrt(E_CHARGE_C * ti / (2.0 * np.pi * ion_mass)),
        np.sqrt(config.ion_charge_number * E_CHARGE_C * te / ion_mass),
    )

    def current_balance(phi: np.ndarray) -> np.ndarray:
        electron_flux = ne * electron_speed * np.exp(phi / te)
        ion_flux = (
            config.ion_charge_number
            * ni
            * ion_speed
            * (1.0 - config.ion_charge_number * phi / ti)
        )
        return ion_flux - electron_flux

    perturbation_V = 1.0e-6
    numerical_derivative = (
        current_balance(potential + perturbation_V)
        - current_balance(potential - perturbation_V)
    ) / (2.0 * perturbation_V)
    capacitance = charge / potential
    collection_area = 4.0 * np.pi * radius * radius
    expected_tau = -capacitance / (E_CHARGE_C * collection_area * numerical_derivative)

    np.testing.assert_allclose(tau_q, expected_tau, rtol=2.0e-10, atol=0.0)
    assert float(tau_q[0]) > 100.0 * config.relaxation_time_s


def test_oml_equilibrium_rejects_an_unbracketed_current_balance() -> None:
    config = ChargeModelConfig(
        enabled=True,
        mode="oml_linearized_relaxation",
        ion_mass_amu=40.0,
        electron_sticking=0.0,
        ion_sticking=1.0,
        root_iterations=64,
    )
    with pytest.raises(ValueError, match="not bracketed"):
        oml_linearized_equilibrium(
            config,
            np.asarray([0.5e-6]),
            np.asarray([3.0]),
            np.asarray([1.0e16]),
            np.asarray([1.0e16]),
            np.asarray([0.03]),
        )


def test_oml_equilibrium_rejects_an_unconverged_root() -> None:
    config = ChargeModelConfig(
        enabled=True,
        mode="oml_linearized_relaxation",
        ion_mass_amu=40.0,
        root_iterations=1,
    )
    with pytest.raises(ValueError, match="residual did not converge"):
        oml_linearized_equilibrium(
            config,
            np.asarray([0.5e-6]),
            np.asarray([3.0]),
            np.asarray([1.0e16]),
            np.asarray([1.0e16]),
            np.asarray([0.03]),
        )


def test_te_charge_relaxation_keeps_the_explicit_timescale_without_a_floor() -> None:
    config = ChargeModelConfig(
        enabled=True,
        mode="te_relaxation",
        te_relaxation_alpha=2.5,
        relaxation_time_s=1.0e-40,
    )
    _charge, tau_q, _potential = te_relaxation_equilibrium(
        config,
        np.asarray([0.5e-6]),
        np.asarray([3.0]),
    )
    assert tau_q.tolist() == [1.0e-40]


def test_te_charge_relaxation_is_exact_and_split_invariant() -> None:
    runtime = _square_collision_runtime()
    axes = runtime.geometry_provider.geometry.axes
    times = np.asarray([0.0], dtype=np.float64)
    shape = (3, 3)
    runtime.field_provider = FieldProviderND(
        field=RegularFieldND(
            spatial_dim=2,
            coordinate_system="cartesian_xy",
            axis_names=("x", "y"),
            axes=axes,
            quantities={
                "Te": QuantitySeriesND(
                    "Te", "eV", times=times, data=np.full(shape, 4.0), metadata={}
                )
            },
            valid_mask=np.ones(shape, dtype=bool),
            time_mode="steady",
        ),
        kind="test",
    )
    config = ChargeModelConfig(
        enabled=True,
        mode="te_relaxation",
        te_relaxation_alpha=2.0,
        relaxation_time_s=0.3,
    )
    diameter = np.asarray([2.0e-6])
    initial_charge = np.asarray([3.0e-16])
    radius = 0.5 * diameter
    equilibrium, _tau, _potential = te_relaxation_equilibrium(
        config,
        radius,
        np.asarray([4.0]),
    )
    duration = 0.7
    expected = equilibrium + (initial_charge - equilibrium) * np.exp(
        -duration / config.relaxation_time_s
    )

    one_step = initial_charge.copy()
    apply_charge_model_update(
        config=config,
        runtime=runtime,
        spatial_dim=2,
        t_eval=0.0,
        delta_t_s=duration,
        active_mask=np.asarray([True]),
        x=np.asarray([[0.5, 0.5]]),
        charge=one_step,
        particle_diameter=diameter,
    )
    split = initial_charge.copy()
    for _ in range(2):
        apply_charge_model_update(
            config=config,
            runtime=runtime,
            spatial_dim=2,
            t_eval=0.0,
            delta_t_s=0.5 * duration,
            active_mask=np.asarray([True]),
            x=np.asarray([[0.5, 0.5]]),
            charge=split,
            particle_diameter=diameter,
        )

    np.testing.assert_allclose(one_step, expected, rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(split, expected, rtol=1e-14, atol=0.0)
