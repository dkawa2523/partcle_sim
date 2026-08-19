from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from particle_tracer_unified._preflight_boundary import (
    boundary_field_support_report,
    runtime_boundary_coverage,
)
from particle_tracer_unified._preflight_comsol import (
    check_boundary_coverage,
    check_case,
    check_time_support,
)
from particle_tracer_unified._preflight_runtime import particle_issues
from particle_tracer_unified.core.geometry2d import (
    _points_inside_boundary_edges_2d_with_boundary_kernel,
)
from particle_tracer_unified.domain import StageFields
from particle_tracer_unified.preflight import validate_case_preflight
from particle_tracer_unified.solvers import _force_field_sources
from particle_tracer_unified.solvers._force_field_sources import (
    _electric_force_field,
    _preferred_flow_velocity,
)
from particle_tracer_unified.solvers._force_field_triangle import (
    _triangle_electric_magnitude_squared_gradient,
    _triangle_temperature_gradient,
)
from particle_tracer_unified.solvers.drag_models import (
    CONTINUUM_DRAG_SCHILLER_NAUMANN,
    _continuum_drag_force_multiplier,
    _cunningham_effective_tau,
    _epstein_effective_tau,
)
from particle_tracer_unified.solvers.drag_regime import gas_mean_free_path_scalar_m
from particle_tracer_unified.solvers.field_compilation_common import (
    backend_time_grid,
    common_quantity_times,
    curl_from_velocity_grids,
    gas_defaults,
    gas_property_quantity_names,
    gradient_time_grid,
    merge_optional_quantity_times,
    time_derivative_time_grid,
    vertex_time_grid,
)
from particle_tracer_unified.solvers.force_field_assembly import (
    _optional_charge_over_mass,
    _optional_flow_velocity_values,
    _particle_vector,
    _particle_velocity_values,
    _sample_backend_force_fields,
    _validate_positive_gas_fields,
    _validated_stage_points,
    sample_compiled_acceleration_vectors,
)
from particle_tracer_unified.solvers.forces import ForceRuntimeParameters
from particle_tracer_unified.solvers.integrator_common import (
    _EPSTEIN_DEFAULT_ACCOMMODATION_DELTA,
    DRAG_MODEL_EPSTEIN,
    DRAG_MODEL_NONE,
    DRAG_MODEL_SCHILLER_NAUMANN,
    DRAG_MODEL_STOKES,
    DRAG_MODEL_STOKES_CUNNINGHAM,
    _is_positive_finite,
    _maximum_substeps,
    advance_component,
    advance_state_2d,
    advance_state_3d,
    compose_stage_acceleration_2d,
    compose_stage_acceleration_3d,
    cunningham_slip_correction,
    drag_model_mode_from_name,
    drag_model_name_from_mode,
    effective_tau_from_slip_speed,
    epstein_relaxation_time,
    etd2_stage_schedule,
    schiller_naumann_drag_correction,
    stokes_relaxation_time,
    uniform_substep_schedule,
)

_GAS_DENSITY_KGM3 = 1.2
_GAS_VISCOSITY_PAS = 1.8e-5
_GAS_TEMPERATURE_K = 300.0
_GAS_MOLECULAR_MASS_KG = 4.65e-26
_REFERENCE_PARTICLE_MASS_KG = 0.2 * 3.0 * np.pi * _GAS_VISCOSITY_PAS * 1.0e-6


def _python(function: Any) -> Any:
    """Exercise the Python implementation behind a Numba dispatcher."""

    return function.py_func


def _effective_tau(mode: int, **overrides: float) -> float:
    values = {
        "tau_stokes": 0.2,
        "slip_speed": 0.4,
        "particle_diameter_m": 1.0e-6,
        "gas_density_kgm3": _GAS_DENSITY_KGM3,
        "gas_mu_pas": _GAS_VISCOSITY_PAS,
        "particle_mass_kg": _REFERENCE_PARTICLE_MASS_KG,
        "gas_temperature_K": _GAS_TEMPERATURE_K,
        "gas_molecular_mass_kg": _GAS_MOLECULAR_MASS_KG,
    }
    values.update(overrides)
    return float(
        effective_tau_from_slip_speed(
            values["tau_stokes"],
            values["slip_speed"],
            values["particle_diameter_m"],
            values["gas_density_kgm3"],
            values["gas_mu_pas"],
            mode,
            values["particle_mass_kg"],
            values["gas_temperature_K"],
            values["gas_molecular_mass_kg"],
            _EPSTEIN_DEFAULT_ACCOMMODATION_DELTA,
        )
    )


def test_drag_dispatch_is_bijective_and_rejects_unknown_values() -> None:
    modes = {
        "stokes": DRAG_MODEL_STOKES,
        "stokes_cunningham": DRAG_MODEL_STOKES_CUNNINGHAM,
        "schiller_naumann": DRAG_MODEL_SCHILLER_NAUMANN,
        "epstein": DRAG_MODEL_EPSTEIN,
        "none": DRAG_MODEL_NONE,
    }

    for name, mode in modes.items():
        assert drag_model_mode_from_name(f" {name.upper()} ") == mode
        assert drag_model_name_from_mode(mode) == name

    with pytest.raises(ValueError, match=r"solver\.drag_model"):
        drag_model_mode_from_name("implicit")
    with pytest.raises(ValueError, match="unknown drag model mode"):
        drag_model_name_from_mode(999)


@pytest.mark.parametrize("invalid", [0.0, -1.0, np.nan, np.inf])
def test_relaxation_time_inputs_must_be_positive_and_finite(invalid: float) -> None:
    with pytest.raises(ValueError, match="particle mass_kg"):
        stokes_relaxation_time(invalid, _GAS_VISCOSITY_PAS, 1.0e-6)
    with pytest.raises(ValueError, match="Epstein drag requires gas density"):
        epstein_relaxation_time(
            1.0e-15,
            invalid,
            _GAS_TEMPERATURE_K,
            1.0e-6,
            _GAS_MOLECULAR_MASS_KG,
        )


def test_effective_tau_dispatch_preserves_invalid_input_semantics() -> None:
    assert _effective_tau(DRAG_MODEL_NONE, tau_stokes=np.nan) == np.inf
    assert _effective_tau(DRAG_MODEL_STOKES) == pytest.approx(0.2)
    assert np.isfinite(_effective_tau(DRAG_MODEL_EPSTEIN))
    assert np.isfinite(_effective_tau(DRAG_MODEL_STOKES_CUNNINGHAM))
    assert np.isfinite(_effective_tau(DRAG_MODEL_SCHILLER_NAUMANN))
    assert np.isnan(_effective_tau(999))
    assert np.isnan(_effective_tau(DRAG_MODEL_STOKES, tau_stokes=0.0))

    for helper, arguments in (
        (
            _epstein_effective_tau,
            (
                0.0,
                _GAS_DENSITY_KGM3,
                1.0e-15,
                300.0,
                _GAS_MOLECULAR_MASS_KG,
                _EPSTEIN_DEFAULT_ACCOMMODATION_DELTA,
            ),
        ),
        (
            _cunningham_effective_tau,
            (0.2, 1.0e-6, np.nan, _GAS_VISCOSITY_PAS, 300.0, _GAS_MOLECULAR_MASS_KG),
        ),
        (
            _continuum_drag_force_multiplier,
            (
                CONTINUUM_DRAG_SCHILLER_NAUMANN,
                1.0,
                1.0e-6,
                _GAS_DENSITY_KGM3,
                0.0,
            ),
        ),
    ):
        assert np.isnan(_python(helper)(*arguments))

    assert _python(_epstein_effective_tau)(
        1.0e-6,
        _GAS_DENSITY_KGM3,
        _REFERENCE_PARTICLE_MASS_KG,
        _GAS_TEMPERATURE_K,
        _GAS_MOLECULAR_MASS_KG,
        _EPSTEIN_DEFAULT_ACCOMMODATION_DELTA,
    ) == pytest.approx(_effective_tau(DRAG_MODEL_EPSTEIN))
    assert _python(_cunningham_effective_tau)(
        0.2,
        1.0e-6,
        _GAS_DENSITY_KGM3,
        _GAS_VISCOSITY_PAS,
        _GAS_TEMPERATURE_K,
        _GAS_MOLECULAR_MASS_KG,
    ) == pytest.approx(_effective_tau(DRAG_MODEL_STOKES_CUNNINGHAM))
    assert 0.2 / _python(_continuum_drag_force_multiplier)(
        CONTINUUM_DRAG_SCHILLER_NAUMANN,
        0.4,
        1.0e-6,
        _GAS_DENSITY_KGM3,
        _GAS_VISCOSITY_PAS,
    ) == pytest.approx(_effective_tau(DRAG_MODEL_SCHILLER_NAUMANN))


def test_drag_correction_rejects_the_outside_correlation_boundary() -> None:
    assert _python(schiller_naumann_drag_correction)(-1.0) == 1.0
    with pytest.raises(ValueError, match="Reynolds number < 800"):
        _python(schiller_naumann_drag_correction)(800.0)


def test_substep_schedules_clamp_invalid_and_excessive_requests() -> None:
    count, dt, start = _python(uniform_substep_schedule)(
        1.0,
        3.0,
        2,
        3,
    )
    assert (count, dt, start) == pytest.approx((3, 1.0 / 3.0, 2.0))
    assert _python(uniform_substep_schedule)(1.0, 1.0, 2, 10) == (
        4,
        0.25,
        0.0,
    )
    assert _python(etd2_stage_schedule)(2.0, -1.0) == pytest.approx((2.0, 0.0, 0.0))


def test_stage_acceleration_and_state_wrappers_preserve_dimensions() -> None:
    acceleration_2d = _python(compose_stage_acceleration_2d)(
        10.0, -4.0, 2.0, 6.0, 1.0, 4.0, 1, 2.0
    )
    acceleration_3d = _python(compose_stage_acceleration_3d)(
        10.0, -4.0, 8.0, 2.0, 6.0, -2.0, 1.0, 4.0, 1, 2.0
    )
    assert acceleration_2d == pytest.approx((4.75, 1.5))
    assert acceleration_3d == pytest.approx((4.75, 1.5, 2.0))

    assert _python(advance_component)(1.0, 2.0, 3.0, 0.2, 0.0) == (0.0, 1.0)
    assert all(
        np.isnan(value)
        for value in _python(advance_component)(1.0, 2.0, 3.0, np.nan, 0.1)
    )
    state_2d = _python(advance_state_2d)(
        0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, np.inf, 0.1
    )
    state_3d = _python(advance_state_3d)(
        0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.inf, 0.1
    )
    assert state_2d == pytest.approx((0.1, 0.2, 1.0, 2.0))
    assert state_3d == pytest.approx((0.1, 0.2, 0.3, 1.0, 2.0, 3.0))


@pytest.mark.parametrize("tolerance", [0.15, np.nan])
def test_geometry_edge_kernel_python_path_matches_compiled_boundaries(
    tolerance: float,
) -> None:
    edges = np.asarray(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [1.0, 1.0]],
            [[1.0, 1.0], [0.0, 1.0]],
            [[0.0, 1.0], [0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    points = np.asarray(
        [[-0.1, -0.1], [0.0, 0.5], [0.5, 0.5], [1.2, 0.5]],
        dtype=np.float64,
    )

    python_result = _python(_points_inside_boundary_edges_2d_with_boundary_kernel)(
        points, edges, tolerance
    )
    compiled_result = _points_inside_boundary_edges_2d_with_boundary_kernel(
        points,
        edges,
        tolerance,
    )

    for python_values, compiled_values in zip(
        python_result,
        compiled_result,
        strict=True,
    ):
        np.testing.assert_array_equal(python_values, compiled_values)
    if np.isfinite(tolerance):
        assert python_result[0].tolist() == [True, True, True, False]
        assert python_result[1].tolist() == [True, True, False, False]
    else:
        assert python_result[0].tolist() == [False, True, True, False]
        assert not np.any(python_result[1])


@pytest.mark.parametrize(
    ("mode", "tau_stokes"),
    [
        (DRAG_MODEL_NONE, np.nan),
        (DRAG_MODEL_EPSTEIN, 0.2),
        (DRAG_MODEL_STOKES, 0.0),
        (DRAG_MODEL_STOKES, 0.2),
        (DRAG_MODEL_STOKES_CUNNINGHAM, 0.2),
        (DRAG_MODEL_SCHILLER_NAUMANN, 0.2),
        (999, 0.2),
    ],
)
def test_effective_tau_python_dispatch_matches_compiled_modes(
    mode: int,
    tau_stokes: float,
) -> None:
    arguments = (
        tau_stokes,
        0.4,
        1.0e-6,
        _GAS_DENSITY_KGM3,
        _GAS_VISCOSITY_PAS,
        mode,
        1.0e-15,
        _GAS_TEMPERATURE_K,
        _GAS_MOLECULAR_MASS_KG,
        _EPSTEIN_DEFAULT_ACCOMMODATION_DELTA,
    )

    python_value = _python(effective_tau_from_slip_speed)(*arguments)
    compiled_value = effective_tau_from_slip_speed(*arguments)

    if np.isnan(python_value):
        assert np.isnan(compiled_value)
    else:
        assert compiled_value == pytest.approx(python_value, rel=2.0e-15)


def test_numba_scalar_helpers_preserve_python_boundary_contracts() -> None:
    for value in (-1.0, 0.2):
        assert cunningham_slip_correction(value) == pytest.approx(
            _python(cunningham_slip_correction)(value),
            rel=2.0e-15,
        )
    for value in (1.0, 0.0, np.nan):
        assert bool(_is_positive_finite(value)) is bool(
            _python(_is_positive_finite)(value)
        )
    for splits in (-1, 3):
        assert _maximum_substeps(splits) == _python(_maximum_substeps)(splits)

    mean_free_path_arguments = (
        _GAS_VISCOSITY_PAS,
        _GAS_DENSITY_KGM3,
        _GAS_TEMPERATURE_K,
        _GAS_MOLECULAR_MASS_KG,
    )
    assert gas_mean_free_path_scalar_m(*mean_free_path_arguments) == pytest.approx(
        _python(gas_mean_free_path_scalar_m)(*mean_free_path_arguments),
        rel=2.0e-15,
    )
    assert np.isnan(
        _python(gas_mean_free_path_scalar_m)(
            0.0,
            _GAS_DENSITY_KGM3,
            _GAS_TEMPERATURE_K,
            _GAS_MOLECULAR_MASS_KG,
        )
    )


def test_integrator_python_boundaries_match_compiled_schedules_and_motion() -> None:
    assert _python(uniform_substep_schedule)(0.01, 1.0, 2, 3) == (
        3,
        pytest.approx(0.01 / 3.0),
        0.99,
    )
    assert _python(uniform_substep_schedule)(1.0, 1.0, 2, 10) == (
        4,
        0.25,
        0.0,
    )

    for tau_eff in (0.2, 1.0e6):
        arguments = (1.0, 2.0, 3.0, tau_eff, 0.1)
        python_result = np.asarray(_python(advance_component)(*arguments))
        compiled_result = np.asarray(advance_component(*arguments))
        np.testing.assert_array_max_ulp(python_result, compiled_result, maxulp=2)


def _series(times: list[float]) -> SimpleNamespace:
    return SimpleNamespace(times=np.asarray(times, dtype=np.float64))


def test_field_grid_derivatives_preserve_shape_dtype_and_boundaries() -> None:
    axes = (np.asarray([0.0, 1.0, 2.0]), np.asarray([0.0, 2.0, 4.0]))
    x, y = np.meshgrid(*axes, indexing="ij")
    steady = (x + 2.0 * y)[None, ...]
    repeated = backend_time_grid(steady[0], 2, np.asarray([0.0, 1.0]))
    assert repeated.shape == (2, 3, 3)
    assert repeated.dtype == np.float64

    gradient_x, gradient_y = gradient_time_grid(steady, axes)
    np.testing.assert_allclose(gradient_x, 1.0)
    np.testing.assert_allclose(gradient_y, 2.0)
    singleton = gradient_time_grid(steady[:, :1, :], (axes[0][:1], axes[1]))
    assert all(np.count_nonzero(component) == 0 for component in singleton)
    with pytest.raises(ValueError, match="time grid"):
        gradient_time_grid(steady[0], axes)

    transient = np.stack((steady[0], steady[0] + 2.0, steady[0] + 8.0))
    derivative = time_derivative_time_grid(transient, np.asarray([0.0, 1.0, 2.0]))
    assert derivative.shape == transient.shape
    assert np.all(np.isfinite(derivative))
    np.testing.assert_array_equal(
        time_derivative_time_grid(steady, np.asarray([0.0])),
        np.zeros_like(steady),
    )

    vertices = vertex_time_grid(np.asarray([1.0, 2.0]), np.asarray([0.0, 1.0]))
    np.testing.assert_array_equal(vertices, [[1.0, 2.0], [1.0, 2.0]])


def test_velocity_curl_covers_2d_3d_and_missing_component_contracts() -> None:
    axis = np.asarray([0.0, 1.0, 2.0])
    x, y = np.meshgrid(axis, axis, indexing="ij")
    ux = (-y)[None, ...]
    uy = x[None, ...]
    curl_x, curl_y, curl_z = curl_from_velocity_grids(ux, uy, None, (axis, axis))
    assert curl_x is None
    assert curl_y is None
    assert curl_z is not None
    np.testing.assert_allclose(curl_z, 2.0)

    assert curl_from_velocity_grids(ux, uy, None, (axis, axis, axis)) == (
        None,
        None,
        None,
    )
    x3, y3, z3 = np.meshgrid(axis, axis, axis, indexing="ij")
    curl_x, curl_y, curl_z = curl_from_velocity_grids(
        (-y3)[None, ...],
        x3[None, ...],
        np.zeros_like(z3)[None, ...],
        (axis, axis, axis),
    )
    assert curl_x is not None
    assert curl_y is not None
    assert curl_z is not None
    np.testing.assert_allclose(curl_x, 0.0)
    np.testing.assert_allclose(curl_y, 0.0)
    np.testing.assert_allclose(curl_z, 2.0)


def test_field_quantity_time_axes_have_one_transient_authority() -> None:
    field = SimpleNamespace(
        quantities={
            "empty": _series([]),
            "steady": _series([0.0]),
            "primary": _series([0.0, 1.0]),
            "matching": _series([0.0, 1.0]),
            "different": _series([0.0, 2.0]),
        }
    )
    np.testing.assert_array_equal(common_quantity_times(field, ("missing",)), [0.0])
    np.testing.assert_array_equal(
        common_quantity_times(field, ("empty", "steady")),
        [0.0],
    )
    with pytest.raises(ValueError, match="empty and primary differ"):
        common_quantity_times(field, ("empty", "primary"))

    np.testing.assert_array_equal(
        merge_optional_quantity_times(
            field,
            np.asarray([]),
            ("steady", "primary", "matching"),
        ),
        [0.0, 1.0],
    )
    with pytest.raises(ValueError, match="primary and different differ"):
        merge_optional_quantity_times(
            field,
            np.asarray([0.0]),
            ("primary", "different"),
        )

    aliases = SimpleNamespace(
        quantities={"rho": object(), "dynamic_viscosity_Pas": object(), "T": object()}
    )
    assert gas_property_quantity_names(aliases) == {
        "gas_density": "rho",
        "gas_mu": "dynamic_viscosity_Pas",
        "gas_temperature": "T",
    }
    defaults = gas_defaults(SimpleNamespace(gas=None))
    assert np.isnan(defaults.density_kgm3)
    assert defaults.density_source == "unavailable"


def test_preflight_particle_errors_are_complete_and_deterministically_ordered() -> None:
    assert [issue.code for issue in particle_issues(SimpleNamespace())] == [
        "input.particles.missing"
    ]
    particles = SimpleNamespace(
        particle_id=np.asarray([7, 7]),
        position=np.asarray([[np.nan, 0.0], [0.0, 0.0]]),
        velocity=np.asarray([[0.0, np.inf], [0.0, 0.0]]),
        release_time=np.asarray([-1.0, 0.0]),
        mass=np.asarray([0.0, 1.0]),
        diameter=np.asarray([-1.0, 1.0]),
        charge=np.asarray([np.nan, 0.0]),
    )

    assert [
        issue.code for issue in particle_issues(SimpleNamespace(particles=particles))
    ] == [
        "input.particles.position.non_finite",
        "input.particles.velocity.non_finite",
        "input.particles.charge_C.non_finite",
        "input.particles.release_time.negative",
        "input.particles.mass.non_positive",
        "input.particles.drag_diameter.non_positive",
        "input.particles.id.duplicate",
    ]


def test_preflight_boundary_support_uses_explicit_fallbacks_without_guessing() -> None:
    geometry = SimpleNamespace(
        boundary_edge_part_ids=None,
        boundary_triangle_part_ids=None,
        nearest_boundary_part_id_map=np.asarray([[0, 3, 3]]),
        boundary_edges=None,
    )
    runtime = SimpleNamespace(
        spatial_dim=2,
        geometry_provider=SimpleNamespace(geometry=geometry),
        wall_catalog=SimpleNamespace(
            part_models=(SimpleNamespace(part_id=3),),
        ),
    )
    coverage = runtime_boundary_coverage(runtime)
    assert coverage["passed"] is True
    assert coverage["geometry_part_source"] == "nearest_boundary_part_id_map"
    assert coverage["geometry_part_ids"] == [3]

    support = boundary_field_support_report(runtime, include_violations=True)
    assert support == {
        "mode": "strict",
        "passed": True,
        "applicable": False,
        "reason": "no explicit boundary",
    }


def test_comsol_preflight_reports_missing_and_insufficient_support(
    tmp_path: Path,
) -> None:
    missing_case = SimpleNamespace(
        config={"case": {"adapter": "comsol"}, "inputs": {}},
        config_path=tmp_path / "case.yaml",
    )
    issues: list[Any] = []
    check_case(cast(Any, missing_case), SimpleNamespace(), {}, issues)
    assert [issue.code for issue in issues] == ["comsol.manifest.missing"]

    manifest = SimpleNamespace(time_support_s=(0.0, 0.5), boundaries_path=lambda: None)
    runtime = SimpleNamespace(
        particles=SimpleNamespace(release_time=np.asarray([0.25, 2.0])),
        field_provider=SimpleNamespace(field=SimpleNamespace(time_mode="transient")),
    )
    checks: dict[str, Any] = {}
    issues = []
    timed_case = SimpleNamespace(config={"time": {"t_end": 1.0}})
    check_time_support(
        cast(Any, timed_case), runtime, cast(Any, manifest), checks, issues
    )
    assert checks["comsol_time_support"] == {
        "declared_support_s": [0.0, 0.5],
        "required_support_s": [0.25, 1.0],
        "field_time_mode": "transient",
        "passed": False,
    }
    assert [issue.code for issue in issues] == ["comsol.time_support"]

    # A steady field has one time sample and the same value at every instant,
    # so the same declared support constrains nothing.  ``simulate()`` has
    # always read it this way and preflight must agree, or ``check`` rejects
    # cases ``run`` integrates.
    steady_runtime = SimpleNamespace(
        particles=SimpleNamespace(release_time=np.asarray([0.25, 2.0])),
        field_provider=SimpleNamespace(field=SimpleNamespace(time_mode="steady")),
    )
    steady_checks: dict[str, Any] = {}
    steady_issues: list[Any] = []
    check_time_support(
        cast(Any, timed_case),
        steady_runtime,
        cast(Any, manifest),
        steady_checks,
        steady_issues,
    )
    assert steady_checks["comsol_time_support"]["passed"] is True
    assert steady_checks["comsol_time_support"]["field_time_mode"] == "steady"
    assert steady_issues == []

    check_boundary_coverage(
        SimpleNamespace(),
        cast(Any, manifest),
        ["manifest invalid"],
        {},
        [],
    )
    no_time_manifest = SimpleNamespace(time_support_s=None)
    check_time_support(
        cast(Any, timed_case),
        runtime,
        cast(Any, no_time_manifest),
        {},
        [],
    )

    invalid_case = SimpleNamespace(
        config={
            "case": {"adapter": "comsol"},
            "inputs": {"comsol_manifest": "missing.yaml"},
        },
        config_path=tmp_path / "case.yaml",
    )
    issues = []
    check_case(cast(Any, invalid_case), runtime, {}, issues)
    assert [issue.code for issue in issues] == ["comsol.manifest"]

    with pytest.raises(ValueError, match="detail must be"):
        validate_case_preflight(cast(Any, invalid_case), detail="diagnostic")


def _base_fields(*, points: np.ndarray, time_s: float = 0.0) -> StageFields:
    return StageFields(
        points_m=points,
        time_s=time_s,
        values={"flow_velocity": np.ones_like(points)},
        supported=np.ones(points.shape[0], dtype=bool),
    )


def test_force_stage_reuse_requires_exact_points_and_time() -> None:
    points = np.asarray([[0.25, 0.75]], dtype=np.float64)
    base = _base_fields(points=points)
    np.testing.assert_array_equal(
        _preferred_flow_velocity(None, base, np.zeros_like(points)),
        np.ones_like(points),
    )
    supplied = np.full_like(points, 2.0)
    np.testing.assert_array_equal(
        _preferred_flow_velocity(supplied, base, np.zeros_like(points)),
        supplied,
    )

    with pytest.raises(ValueError, match="points must exactly match"):
        _validated_stage_points(2, 0.0, points, _base_fields(points=points + 1.0))
    with pytest.raises(ValueError, match="time must exactly match"):
        _validated_stage_points(2, 0.0, points, _base_fields(points=points, time_s=1.0))
    with pytest.raises(ValueError, match="finite coordinates"):
        _validated_stage_points(2, 0.0, np.asarray([[np.nan, 0.0]]), None)


def test_force_particle_inputs_reject_ambiguous_shapes_and_nonfinite_values() -> None:
    with pytest.raises(ValueError, match=r"shape \(2,\)"):
        _particle_vector(np.ones(3), 2, "mass", default=np.nan, positive_required=True)
    with pytest.raises(ValueError, match=r"invalid rows: \[1\]"):
        _particle_vector(
            np.asarray([1.0, 0.0]),
            2,
            "mass",
            default=np.nan,
            positive_required=True,
        )
    with pytest.raises(ValueError, match="electric_q_over_m must be finite"):
        _optional_charge_over_mass(np.asarray([np.nan]), 1)
    with pytest.raises(ValueError, match="require particle velocity"):
        _particle_velocity_values(None, 1, 2, required=True)
    with pytest.raises(ValueError, match=r"shape \(1, 2\)"):
        _particle_velocity_values(np.asarray([[np.nan, 0.0]]), 1, 2, required=False)
    with pytest.raises(ValueError, match=r"shape \(1, 2\)"):
        _optional_flow_velocity_values(np.zeros((1, 3)), 1, 2)


def test_force_field_validation_ignores_unsupported_and_rejects_invalid() -> None:
    values = {"gas_density": np.asarray([np.nan, 1.0])}
    _validate_positive_gas_fields(values, np.asarray([False, True]))
    with pytest.raises(ValueError, match=r"invalid sample rows: \[0\]"):
        _validate_positive_gas_fields(values, np.asarray([True, True]))

    points = np.asarray([[0.0, 0.0]])
    regular_stub = SimpleNamespace(axes=(np.asarray([0.0]),))
    with pytest.raises(ValueError, match="requested dimension differ"):
        _sample_backend_force_fields(
            cast(Any, regular_stub),
            2,
            0.0,
            points,
            params=ForceRuntimeParameters(),
            include_electric=False,
            flow_velocity=None,
            fallback_density_kgm3=np.nan,
            fallback_mu_pas=np.nan,
            fallback_temperature_K=np.nan,
            base_fields=None,
        )


def test_force_field_missing_quantities_fail_at_the_semantic_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    points = np.asarray([[0.0, 0.0]])
    triangle_stub = SimpleNamespace(
        gas_property_names={},
        electric_field_names=(),
    )
    with pytest.raises(ValueError, match="temperature field quantity"):
        _triangle_temperature_gradient(cast(Any, triangle_stub), 0.0, points, {})
    with pytest.raises(ValueError, match="electric field components"):
        _triangle_electric_magnitude_squared_gradient(
            cast(Any, triangle_stub), 0.0, points, {}
        )

    monkeypatch.setattr(
        _force_field_sources,
        "sample_compiled_electric_vectors",
        lambda *_args, **_kwargs: None,
    )
    with pytest.raises(ValueError, match="exported electric field components"):
        _electric_force_field(cast(Any, object()), 2, 0.0, points, None)


def test_empty_force_batch_keeps_float64_shape_without_sampling() -> None:
    acceleration = sample_compiled_acceleration_vectors(
        cast(Any, object()),
        3,
        0.0,
        np.empty((0, 3), dtype=np.float64),
    )
    assert acceleration.shape == (0, 3)
    assert acceleration.dtype == np.float64
