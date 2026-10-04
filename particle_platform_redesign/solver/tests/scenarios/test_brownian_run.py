from __future__ import annotations

import math
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

import chamber_particles.engine as engine_module
import chamber_particles.output as output_module
from chamber_particles import SimulationError, load_case, open_result, simulate
from chamber_particles.case_format import RegularLayout, write
from chamber_particles.physics.charge import oml_stationary_maxwellian_debye_huckel_v1
from chamber_particles.physics.forces import BOLTZMANN_J_K
from chamber_particles.stochastic import joint_ou_covariance
from tests.verification.microcases import materialize_microcase

_MASS_KG = 4.0e-15
_TEMPERATURE_K = 300.0
_RATE_S_INV = 2.0
_TARGET_VELOCITY_M_S = np.asarray([0.1, -0.05])
_INITIAL_VELOCITY_M_S = np.asarray([0.2, -0.1])
_NOISE_REVISION = "inertial_langevin_fdt_epstein_linear_frozen_start_v1"
_RZ_NOISE_REVISION = "inertial_langevin_fdt_epstein_linear_rz_meridional_projected_v1"


def test_brownian_public_run_matches_joint_ou_mean_and_covariance(tmp_path: Path) -> None:
    count = 30_000
    end_s = 0.2
    case_path = _brownian_case(
        tmp_path / "ensemble",
        particle_count=count,
        end_s=end_s,
        dt_s=end_s,
        tree_depth=0,
        frame_times=None,
    )
    output = tmp_path / "ensemble-result"

    simulate(load_case(case_path), output)
    result = open_result(output)
    final = result.read_final()

    decay = math.exp(-_RATE_S_INV * end_s)
    displacement = -math.expm1(-_RATE_S_INV * end_s) / _RATE_S_INV
    expected_position = (
        _TARGET_VELOCITY_M_S * end_s + (_INITIAL_VELOCITY_M_S - _TARGET_VELOCITY_M_S) * displacement
    )
    expected_velocity = _TARGET_VELOCITY_M_S + decay * (
        _INITIAL_VELOCITY_M_S - _TARGET_VELOCITY_M_S
    )
    thermal = BOLTZMANN_J_K * _TEMPERATURE_K / _MASS_KG
    variance_x, covariance_xv, variance_v = joint_ou_covariance(
        np.asarray([_RATE_S_INV]),
        np.asarray([thermal]),
        np.asarray([end_s]),
    )
    expected_covariance = np.asarray(
        [[variance_x[0], covariance_xv[0]], [covariance_xv[0], variance_v[0]]]
    )
    for component in range(2):
        sample = np.column_stack((final.position_m[:, component], final.velocity_m_s[:, component]))
        expected_mean = np.asarray([expected_position[component], expected_velocity[component]])
        standard_error = np.sqrt(np.diag(expected_covariance) / count)
        assert np.all(np.abs(sample.mean(axis=0) - expected_mean) <= 7.0 * standard_error)
        np.testing.assert_allclose(
            np.cov(sample, rowvar=False, bias=True),
            expected_covariance,
            rtol=4.0e-2,
            atol=2.0e-9,
        )
    centered_position = final.position_m - expected_position
    np.testing.assert_allclose(
        np.mean(np.sum(centered_position**2, axis=1)),
        2.0 * variance_x[0],
        rtol=4.0e-2,
        atol=2.0e-9,
    )
    assert result.manifest["resolved"]["path_kind"] == "cubic_hermite"
    assert result.manifest["brownian_interval_tree_depth"] == 0
    assert result.manifest["resolved"]["brownian_coefficient_policy"] == (
        "macro_root_frozen_start_v1"
    )


def test_rz_brownian_constant_gravity_matches_joint_ou_mean_covariance_and_charge(
    tmp_path: Path,
) -> None:
    count = 20_000
    end_s = 0.2
    gravity = np.asarray([0.0, -0.3])
    initial_position = np.asarray([1.1, 0.0])
    case_path = _brownian_rz_case(
        tmp_path / "rz-ensemble",
        particle_count=count,
        initial_position_m=initial_position,
        initial_velocity_m_s=_INITIAL_VELOCITY_M_S,
        gas_velocity_m_s=_TARGET_VELOCITY_M_S,
        radial_shift_m=1.5,
        end_s=end_s,
        dt_s=end_s,
        tree_depth=0,
        gravity_m_s2=gravity,
        frame_times=None,
        memory_limit_mb=128,
    )

    simulate(load_case(case_path), tmp_path / "rz-ensemble-result")
    result = open_result(tmp_path / "rz-ensemble-result")
    final = result.read_final()

    effective_target = _TARGET_VELOCITY_M_S + gravity / _RATE_S_INV
    decay = math.exp(-_RATE_S_INV * end_s)
    displacement = -math.expm1(-_RATE_S_INV * end_s) / _RATE_S_INV
    expected_position = (
        initial_position
        + effective_target * end_s
        + (_INITIAL_VELOCITY_M_S - effective_target) * displacement
    )
    expected_velocity = effective_target + decay * (_INITIAL_VELOCITY_M_S - effective_target)
    thermal = BOLTZMANN_J_K * _TEMPERATURE_K / _MASS_KG
    variance_x, covariance_xv, variance_v = joint_ou_covariance(
        np.asarray([_RATE_S_INV]),
        np.asarray([thermal]),
        np.asarray([end_s]),
    )
    expected_covariance = np.asarray(
        [[variance_x[0], covariance_xv[0]], [covariance_xv[0], variance_v[0]]]
    )
    for component in range(2):
        sample = np.column_stack((final.position_m[:, component], final.velocity_m_s[:, component]))
        standard_error = np.sqrt(np.diag(expected_covariance) / count)
        expected_mean = np.asarray([expected_position[component], expected_velocity[component]])
        assert np.all(np.abs(sample.mean(axis=0) - expected_mean) <= 7.0 * standard_error)
        np.testing.assert_allclose(
            np.cov(sample, rowvar=False, bias=True),
            expected_covariance,
            rtol=5.0e-2,
            atol=2.0e-9,
        )
    np.testing.assert_array_equal(final.charge_number, np.full(count, 3.0))
    assert result.manifest["resolved"]["brownian_coefficient_policy"] == (
        "macro_root_frozen_midpoint_v1"
    )
    assert result.manifest["brownian_composition_revision"] == (
        "stochastic_exponential_midpoint_v1"
    )
    assert result.manifest["brownian_charge_dense_revision"] == ("macro_root_affine_exponential_v2")
    assert result.manifest["engine_algorithm_revision"] == "particle_engine_v36"
    assert result.manifest["step_proposal_revision"] == "coupled_fixed_step_proposal_v10"
    assert result.manifest["physics_catalog_revision"] == "inertial_langevin_rz_catalog_v17"
    resume_identity = result.manifest["resume_identity"]
    assert (
        resume_identity["brownian_coefficient_policy"],
        resume_identity["brownian_composition_revision"],
        resume_identity["brownian_charge_dense_revision"],
    ) == (
        "macro_root_frozen_midpoint_v1",
        "stochastic_exponential_midpoint_v1",
        "macro_root_affine_exponential_v2",
    )


def test_rz_brownian_axis_restart_is_output_and_slab_invariant(tmp_path: Path) -> None:
    common = {
        "particle_count": 1_200,
        "initial_position_m": np.asarray([0.05, 0.0]),
        "initial_velocity_m_s": np.asarray([-1.0, 0.0]),
        "gas_velocity_m_s": np.zeros(2),
        "radial_shift_m": 1.0,
        "end_s": 0.2,
        "dt_s": 0.2,
        "tree_depth": 2,
        "gravity_m_s2": None,
    }
    narrow_case = _brownian_rz_case(
        tmp_path / "axis-narrow",
        frame_times=None,
        memory_limit_mb=8,
        **common,
    )
    wide_case = _brownian_rz_case(
        tmp_path / "axis-wide",
        frame_times=[0.0, 0.031, 0.1, 0.2],
        memory_limit_mb=64,
        **common,
    )

    simulate(load_case(narrow_case), tmp_path / "axis-narrow-result")
    simulate(load_case(wide_case), tmp_path / "axis-wide-result")
    narrow = open_result(tmp_path / "axis-narrow-result")
    wide = open_result(tmp_path / "axis-wide-result")

    _assert_final_identity(narrow.read_final(), wide.read_final())
    assert narrow.manifest["boundary_interactions"]["axis_crossings"] == 1_200
    assert wide.manifest["boundary_interactions"]["axis_crossings"] == 1_200
    assert bool((wide.read_final().position_m[:, 0] >= 0.0).all())
    assert bool((wide.read_final().velocity_m_s[:, 0] > 0.0).all())
    assert len(list(wide.iter_frames())) == 4
    assert (
        narrow.manifest["memory_plan"]["slab_particles"]
        < wide.manifest["memory_plan"]["slab_particles"]
    )


def test_rz_brownian_dynamic_charge_dense_path_is_tree_depth_invariant(
    tmp_path: Path,
) -> None:
    end_s = 0.2
    frame_times = [0.0, 0.037, 0.08, 0.13, end_s]
    number_density_m3 = 1.0e10
    initial_charge_number = 0.0
    charge_inputs = {
        "electrostatic_radius_m": np.asarray([1.0e-6]),
        "electron_number_density_m3": np.asarray([number_density_m3]),
        "positive_ion_number_density_m3": np.asarray([number_density_m3]),
        "electron_temperature_K": np.asarray([20_000.0]),
        "positive_ion_temperature_K": np.asarray([300.0]),
        "particle_velocity_m_s": np.zeros((1, 2)),
        "positive_ion_velocity_m_s": np.zeros((1, 2)),
        "positive_ion_mass_kg": 6.6335209e-26,
    }

    def evaluate_charge(charge_number: float) -> tuple[float, float]:
        evaluation = oml_stationary_maxwellian_debye_huckel_v1(
            charge_number=np.asarray([charge_number]),
            **charge_inputs,
        )
        return (
            float(evaluation.charge_rate_number_s[0]),
            float(evaluation.charge_rate_derivative_s_inv[0]),
        )

    def multiplier(derivative_s_inv: float, elapsed_s: float) -> float:
        if derivative_s_inv == 0.0:
            return elapsed_s
        return math.expm1(derivative_s_inv * elapsed_s) / derivative_s_inv

    start_rate, start_derivative = evaluate_charge(initial_charge_number)
    assert -start_derivative * end_s > 4.0
    predicted_midpoint_charge = (
        initial_charge_number
        + multiplier(
            start_derivative,
            0.5 * end_s,
        )
        * start_rate
    )
    midpoint_rate, midpoint_derivative = evaluate_charge(predicted_midpoint_charge)
    affine_rate = midpoint_rate + midpoint_derivative * (
        initial_charge_number - predicted_midpoint_charge
    )
    expected_charge = np.asarray(
        [
            initial_charge_number + multiplier(midpoint_derivative, time_s) * affine_rate
            for time_s in frame_times
        ]
    )
    charge_paths = []
    for tree_depth in (0, 3):
        case_path = _brownian_rz_case(
            tmp_path / f"dynamic-charge-depth-{tree_depth}",
            particle_count=1,
            initial_position_m=np.asarray([1.1, 0.0]),
            initial_velocity_m_s=_INITIAL_VELOCITY_M_S,
            gas_velocity_m_s=_TARGET_VELOCITY_M_S,
            radial_shift_m=1.5,
            end_s=end_s,
            dt_s=end_s,
            tree_depth=tree_depth,
            gravity_m_s2=None,
            frame_times=frame_times,
            memory_limit_mb=8,
            initial_charge_number=initial_charge_number,
            continuous_charge=True,
            number_density_m3=number_density_m3,
        )
        output = tmp_path / f"dynamic-charge-result-{tree_depth}"
        simulate(load_case(case_path), output)
        result = open_result(output)
        frames = list(result.iter_frames())
        final_charge = float(result.read_final().charge_number[0])

        assert result.manifest["boundary_interactions"]["axis_crossings"] == 0
        assert final_charge != 0.0
        assert [frame.time_s for frame in frames] == frame_times
        actual_charge = np.asarray([frame.charge_number[0] for frame in frames])
        np.testing.assert_allclose(actual_charge, expected_charge, rtol=5.0e-14, atol=5.0e-14)
        assert bool((np.diff(actual_charge) < 0.0).all())
        assert expected_charge[-1] <= actual_charge.min() <= actual_charge.max() <= 0.0
        linear_charge = final_charge * np.asarray(frame_times) / end_s
        assert float(np.max(np.abs(actual_charge - linear_charge))) > 10.0
        assert result.manifest["brownian_charge_dense_revision"] == (
            "macro_root_affine_exponential_v2"
        )
        charge_paths.append(actual_charge)

    np.testing.assert_array_equal(charge_paths[0], charge_paths[1])


def test_brownian_output_schedule_does_not_change_physical_result(tmp_path: Path) -> None:
    common = {
        "particle_count": 96,
        "end_s": 0.4,
        "dt_s": 0.2,
        "tree_depth": 3,
    }
    plain_path = _brownian_case(tmp_path / "plain", frame_times=None, **common)
    framed_path = _brownian_case(
        tmp_path / "framed",
        frame_times=[0.0, 0.037, 0.1, 0.2, 0.311, 0.4],
        **common,
    )

    simulate(load_case(plain_path), tmp_path / "plain-result")
    simulate(load_case(framed_path), tmp_path / "framed-result")
    plain = open_result(tmp_path / "plain-result")
    framed = open_result(tmp_path / "framed-result")

    _assert_final_identity(plain.read_final(), framed.read_final())
    assert len(list(framed.iter_frames())) == 6
    assert plain.manifest["joint_ou_revision"] == "inertial_joint_ou_v1"
    assert plain.manifest["joint_ou_split_revision"] == ("conditional_gaussian_half_split_v1")
    assert plain.manifest["brownian_rng_revision"] == ("philox4x32_10_brownian_interval_tree_v1")
    memory_plan = plain.manifest["memory_plan"]
    assert memory_plan["stochastic_tree_work_bytes_per_particle"] == 32 * (3 + 4)


def test_brownian_decimal_macro_grid_has_no_roundoff_tail(tmp_path: Path) -> None:
    case_path = _brownian_case(
        tmp_path / "decimal-grid",
        particle_count=8,
        end_s=0.2,
        dt_s=0.02,
        tree_depth=3,
        frame_times=None,
    )

    summary = simulate(load_case(case_path), tmp_path / "decimal-grid-result")
    result = open_result(tmp_path / "decimal-grid-result")
    lifecycle = result.read_lifecycle_series()

    assert summary.macro_step_count == 10
    np.testing.assert_array_equal(lifecycle.time_s[-1:], [0.2])
    np.testing.assert_array_equal(lifecycle.active, np.full(10, 8))
    np.testing.assert_array_equal(result.read_final().lifecycle, np.full(8, 1, dtype="<u1"))
    assert result.manifest["failure_reason_counts"]["integrator_accuracy"] == 0


def test_brownian_checkpoint_resume_preserves_the_public_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(engine_module, "_DURABLE_COMMIT_MINIMUM_WORK", 16)
    monkeypatch.setattr(engine_module, "_DURABLE_COMMIT_WORK_PER_PARTICLE", 0)
    case_path = _brownian_case(
        tmp_path / "resume-case",
        particle_count=96,
        end_s=0.65,
        dt_s=0.01,
        tree_depth=2,
        frame_times=[0.0, 0.17, 0.33, 0.65],
    )
    case = load_case(case_path)
    uninterrupted_path = tmp_path / "resume-reference"
    resumed_path = tmp_path / "resume-interrupted"
    simulate(case, uninterrupted_path)

    real_replace = os.replace

    def replace_then_fail(source: object, destination: object) -> None:
        real_replace(source, destination)
        if Path(destination).name == "LATEST":
            raise OSError("injected Brownian checkpoint interruption")

    with monkeypatch.context() as patch:
        patch.setattr(output_module.os, "replace", replace_then_fail)
        with pytest.raises(SimulationError):
            simulate(case, resumed_path)

    assert open_result(resumed_path, recovery=True).read_lifecycle_series().time_s.size < 65
    simulate(case, resumed_path)
    expected = open_result(uninterrupted_path)
    actual = open_result(resumed_path)
    _assert_final_identity(expected.read_final(), actual.read_final())
    _assert_record_identity(expected.read_lifecycle_series(), actual.read_lifecycle_series())
    for first, second in zip(expected.iter_frames(), actual.iter_frames(), strict=True):
        _assert_record_identity(first, second)
    for key in (
        "brownian_rng_revision",
        "joint_ou_revision",
        "joint_ou_split_revision",
        "brownian_interval_tree_depth",
        "random_draw_kinds",
        "resolved",
    ):
        assert expected.manifest[key] == actual.manifest[key]


def test_rz_brownian_checkpoint_resume_is_bitwise_identical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(engine_module, "_DURABLE_COMMIT_MINIMUM_WORK", 16)
    monkeypatch.setattr(engine_module, "_DURABLE_COMMIT_WORK_PER_PARTICLE", 0)
    case_path = _brownian_rz_case(
        tmp_path / "rz-resume-case",
        particle_count=96,
        initial_position_m=np.asarray([1.1, 0.0]),
        initial_velocity_m_s=_INITIAL_VELOCITY_M_S,
        gas_velocity_m_s=_TARGET_VELOCITY_M_S,
        radial_shift_m=1.5,
        end_s=0.65,
        dt_s=0.01,
        tree_depth=2,
        gravity_m_s2=np.asarray([0.0, -0.3]),
        frame_times=[0.0, 0.17, 0.33, 0.65],
        memory_limit_mb=16,
        continuous_charge=True,
    )
    case = load_case(case_path)
    uninterrupted_path = tmp_path / "rz-resume-reference"
    resumed_path = tmp_path / "rz-resume-interrupted"
    simulate(case, uninterrupted_path)

    real_replace = os.replace

    def replace_then_fail(source: object, destination: object) -> None:
        real_replace(source, destination)
        if Path(destination).name == "LATEST":
            raise OSError("injected RZ Brownian checkpoint interruption")

    with monkeypatch.context() as patch:
        patch.setattr(output_module.os, "replace", replace_then_fail)
        with pytest.raises(SimulationError):
            simulate(case, resumed_path)

    assert open_result(resumed_path, recovery=True).read_lifecycle_series().time_s.size < 65
    simulate(case, resumed_path)
    expected = open_result(uninterrupted_path)
    actual = open_result(resumed_path)

    _assert_final_identity(expected.read_final(), actual.read_final())
    _assert_record_identity(expected.read_lifecycle_series(), actual.read_lifecycle_series())
    for first, second in zip(expected.iter_frames(), actual.iter_frames(), strict=True):
        _assert_record_identity(first, second)
    assert expected.manifest["resume_identity"] == actual.manifest["resume_identity"]
    assert actual.manifest["resume_identity"]["engine_algorithm_revision"] == (
        "particle_engine_v36"
    )


def test_brownian_unresolved_absolute_leaf_time_fails_the_particle(tmp_path: Path) -> None:
    start_s = float(2**53)
    case_path = _brownian_case(
        tmp_path / "unresolved-time",
        particle_count=1,
        end_s=0.2,
        dt_s=0.2,
        tree_depth=4,
        frame_times=None,
    )
    case = load_case(case_path)
    source = replace(
        case.data.sources[0],
        release_time_s=np.asarray([start_s], dtype="<f8"),
    )
    data_path = case_path.with_name("unresolved-time.h5")
    info = write(data_path, replace(case.data, sources=(source,)))
    document = yaml.safe_load(case_path.read_text(encoding="utf-8"))
    document["case"]["data_path"] = data_path.name
    document["case"]["expected_content_hash"] = info.content_hash
    document["time"] = {"start_s": start_s, "end_s": start_s + 16.0, "dt_s": 16.0}
    case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    simulate(load_case(case_path), tmp_path / "unresolved-time-result")
    result = open_result(tmp_path / "unresolved-time-result")

    np.testing.assert_array_equal(result.read_final().lifecycle, [4])
    assert result.manifest["failure_reason_counts"]["integrator_accuracy"] == 1


def test_brownian_rejects_an_unresolved_relaxation_scale_before_running(
    tmp_path: Path,
) -> None:
    case_path = _brownian_case(
        tmp_path / "unresolved-relaxation",
        particle_count=1,
        end_s=0.2,
        dt_s=0.2,
        tree_depth=1,
        frame_times=None,
    )
    case = load_case(case_path)
    source = replace(
        case.data.sources[0],
        mass_kg=np.asarray([4.0e-125], dtype="<f8"),
    )
    data_path = case_path.with_name("unresolved-relaxation.h5")
    info = write(data_path, replace(case.data, sources=(source,)))
    document = yaml.safe_load(case_path.read_text(encoding="utf-8"))
    document["case"]["data_path"] = data_path.name
    document["case"]["expected_content_hash"] = info.content_hash
    case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(SimulationError, match=r"gamma \* dt <= 1e\+06"):
        simulate(load_case(case_path), tmp_path / "unresolved-relaxation-result")


def test_brownian_covariance_failure_is_particle_local(tmp_path: Path) -> None:
    """One unrepresentable OU row must not discard its valid neighbour."""

    case_path = _brownian_case(
        tmp_path / "mixed-covariance",
        particle_count=2,
        end_s=10.0,
        dt_s=10.0,
        tree_depth=3,
        frame_times=None,
    )
    case = load_case(case_path)
    temperature_K = 1.0e307
    target_mean_thermal_speed_m_s = 100.0
    gas_molecular_mass_kg = (
        8.0 * BOLTZMANN_J_K * temperature_K / (math.pi * target_mean_thermal_speed_m_s**2)
    )
    bad_mass_kg = 2.0e-24
    good_mass_kg = BOLTZMANN_J_K * temperature_K
    bad_radius_m = 1.0e-6
    gas_density_kg_m3 = (
        0.1
        * bad_mass_kg
        / ((4.0 * math.pi / 3.0) * bad_radius_m**2 * target_mean_thermal_speed_m_s)
    )
    good_drag_diameter_m = 2.0 * bad_radius_m * math.sqrt(good_mass_kg / bad_mass_kg)
    bad_thermal_variance = BOLTZMANN_J_K * temperature_K / bad_mass_kg
    with pytest.raises(FloatingPointError, match="covariance is not finite"):
        joint_ou_covariance(
            np.asarray([0.1]),
            np.asarray([bad_thermal_variance]),
            np.asarray([10.0]),
        )

    fields = []
    for field in case.data.fields:
        if field.name == "gas_temperature":
            field = replace(field, values=np.full_like(field.values, temperature_K))
        elif field.name == "gas_density":
            field = replace(field, values=np.full_like(field.values, gas_density_kg_m3))
        elif field.name == "gas_velocity":
            field = replace(field, values=np.zeros_like(field.values))
        elif field.name == "gas_mean_free_path":
            field = replace(field, values=np.full_like(field.values, 1.0e150))
        fields.append(field)
    layout = replace(
        case.data.layouts[0],
        axis0_m=np.asarray([-100.0, 100.0]),
        axis1_m=np.asarray([-100.0, 100.0]),
    )
    source = replace(
        case.data.sources[0],
        velocity_m_s=np.zeros((2, 2), dtype="<f8"),
        mass_kg=np.asarray([bad_mass_kg, good_mass_kg], dtype="<f8"),
        drag_diameter_m=np.asarray([2.0e-6, good_drag_diameter_m], dtype="<f8"),
    )
    data_path = case_path.with_name("mixed-covariance.h5")
    info = write(
        data_path,
        replace(
            case.data,
            layouts=(layout,),
            fields=tuple(fields),
            sources=(source,),
        ),
    )
    document = yaml.safe_load(case_path.read_text(encoding="utf-8"))
    document["case"]["data_path"] = data_path.name
    document["case"]["expected_content_hash"] = info.content_hash
    document["physics"]["drag"]["gas_molecular_mass_kg"] = gas_molecular_mass_kg
    case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    simulate(load_case(case_path), tmp_path / "mixed-covariance-result")
    result = open_result(tmp_path / "mixed-covariance-result")
    final = result.read_final()
    failures = result.read_failure_events()

    np.testing.assert_array_equal(final.particle_id, [1, 2])
    np.testing.assert_array_equal(final.lifecycle, [4, 1])
    np.testing.assert_array_equal(final.kinematics_valid, [0, 1])
    np.testing.assert_array_equal(failures.particle_id, [1])
    np.testing.assert_array_equal(failures.time_s, [0.0])
    assert failures.reason_code[0] == result.manifest["failure_reason_codes"]["nonfinite_physics"]
    assert result.manifest["failure_reason_counts"]["nonfinite_physics"] == 1
    assert np.isfinite(final.position_m[1]).all()
    assert np.isfinite(final.velocity_m_s[1]).all()


@pytest.mark.parametrize(
    ("law", "expected_lifecycle", "expected_outcome"),
    [("stick", 2, "stuck"), ("escape", 3, "escaped"), ("hold", 5, "held")],
)
def test_brownian_terminal_wall_uses_the_certified_hermite_path(
    tmp_path: Path,
    law: str,
    expected_lifecycle: int,
    expected_outcome: str,
) -> None:
    case_path = _brownian_wall_case(tmp_path / law, law=law)
    output = tmp_path / f"{law}-result"

    simulate(load_case(case_path), output)
    result = open_result(output)
    final = result.read_final()
    events = result.read_boundary_events()

    np.testing.assert_array_equal(final.lifecycle, [expected_lifecycle])
    np.testing.assert_array_equal(events.outcome, [expected_outcome])
    assert events.time_s.size == 1
    assert 0.35 < events.time_s[0] < 1.5
    np.testing.assert_allclose(events.position_m[0, 0], 1.0, rtol=0.0, atol=2.0e-10)
    assert result.manifest["event_refinement"] is not None


def test_brownian_planar_first_passage_stabilizes_with_tree_depth(tmp_path: Path) -> None:
    event_tables = []
    count_per_seed = 400
    seeds = (8241, 8242, 8243)
    count = count_per_seed * len(seeds)
    for depth in (6, 7, 8):
        particle_ids = []
        event_times = []
        for seed in seeds:
            case_path = _brownian_first_passage_case(
                tmp_path / f"first-passage-{depth}-{seed}",
                particle_count=count_per_seed,
                tree_depth=depth,
                seed=seed,
            )
            output = tmp_path / f"first-passage-result-{depth}-{seed}"
            simulate(load_case(case_path), output)
            events = open_result(output).read_boundary_events()
            particle_ids.append(events.particle_id)
            event_times.append(events.time_s)
        event_tables.append((particle_ids, np.concatenate(event_times)))

    coarse_probability = event_tables[0][1].size / count
    medium_probability = event_tables[1][1].size / count
    fine_probability = event_tables[2][1].size / count
    assert 0.02 < fine_probability < 0.2
    assert abs(fine_probability - medium_probability) < 0.005
    assert abs(fine_probability - medium_probability) <= abs(
        medium_probability - coarse_probability
    )
    standard_error = math.sqrt(
        (
            medium_probability * (1.0 - medium_probability)
            + fine_probability * (1.0 - fine_probability)
        )
        / count
    )
    assert abs(fine_probability - medium_probability) <= 3.0 * standard_error
    for seed_index in range(len(seeds)):
        medium_ids = set(event_tables[1][0][seed_index].tolist())
        fine_ids = set(event_tables[2][0][seed_index].tolist())
        assert len(medium_ids.symmetric_difference(fine_ids)) <= 4
    evaluation_times = np.asarray([0.25, 0.5, 0.75, 1.0])
    medium_cdf = np.asarray(
        [np.count_nonzero(event_tables[1][1] <= time_s) / count for time_s in evaluation_times]
    )
    fine_cdf = np.asarray(
        [np.count_nonzero(event_tables[2][1] <= time_s) / count for time_s in evaluation_times]
    )
    assert float(np.max(np.abs(fine_cdf - medium_cdf))) < 0.005


@pytest.mark.parametrize(
    ("rz_b03", "particle_id", "wall_position_m"),
    [(False, 32, 0.002), (True, 285, 0.004)],
    ids=("cartesian", "axisymmetric-rz"),
)
def test_brownian_monotone_wall_approach_is_committed(
    tmp_path: Path,
    rz_b03: bool,
    particle_id: int,
    wall_position_m: float,
) -> None:
    case_path = _brownian_first_passage_case(
        tmp_path / "monotone-first-passage",
        particle_count=1,
        tree_depth=8,
        seed=8241,
        rz_b03=rz_b03,
        particle_id_start=particle_id,
    )

    simulate(load_case(case_path), tmp_path / "monotone-first-passage-result")
    result = open_result(tmp_path / "monotone-first-passage-result")
    final = result.read_final()
    events = result.read_boundary_events()
    failures = result.read_failure_events()

    np.testing.assert_array_equal(final.particle_id, [particle_id])
    np.testing.assert_array_equal(final.lifecycle, [2])
    np.testing.assert_array_equal(events.particle_id, [particle_id])
    np.testing.assert_allclose(
        events.position_m[:, 0],
        [wall_position_m],
        rtol=0.0,
        atol=1.0e-15,
    )
    assert failures.particle_id.size == 0


def test_brownian_slab_width_does_not_change_events_or_state(tmp_path: Path) -> None:
    case_path = _brownian_first_passage_case(
        tmp_path / "slab-case",
        particle_count=1_200,
        tree_depth=4,
        seed=812,
    )
    document = yaml.safe_load(case_path.read_text(encoding="utf-8"))
    results = []
    for memory_limit_mb in (8, 64):
        variant = {
            **document,
            "resources": {
                **document["resources"],
                "memory_limit_mb": memory_limit_mb,
            },
        }
        variant_path = case_path.with_name(f"slab-{memory_limit_mb}.yaml")
        variant_path.write_text(yaml.safe_dump(variant, sort_keys=False), encoding="utf-8")
        output = tmp_path / f"slab-result-{memory_limit_mb}"
        simulate(load_case(variant_path), output)
        results.append(open_result(output))

    assert (
        results[0].manifest["memory_plan"]["slab_particles"]
        < results[1].manifest["memory_plan"]["slab_particles"]
    )
    _assert_final_identity(results[0].read_final(), results[1].read_final())
    _assert_record_identity(
        results[0].read_boundary_events(),
        results[1].read_boundary_events(),
    )


def test_brownian_frame_replay_does_not_change_wall_events(tmp_path: Path) -> None:
    plain_case = _brownian_wall_case(tmp_path / "wall-plain", law="stick", frame_times=None)
    framed_case = _brownian_wall_case(
        tmp_path / "wall-framed",
        law="stick",
        frame_times=[0.0, 0.25, 1.0, 2.0],
    )
    simulate(load_case(plain_case), tmp_path / "wall-plain-result")
    simulate(load_case(framed_case), tmp_path / "wall-framed-result")
    plain = open_result(tmp_path / "wall-plain-result")
    framed = open_result(tmp_path / "wall-framed-result")

    _assert_final_identity(plain.read_final(), framed.read_final())
    _assert_record_identity(plain.read_boundary_events(), framed.read_boundary_events())


def test_brownian_rejects_nonterminal_wall_laws(tmp_path: Path) -> None:
    case_path = _brownian_wall_case(tmp_path / "reflection", law="specular")
    with pytest.raises(SimulationError, match="terminal stick/escape/hold"):
        simulate(load_case(case_path), tmp_path / "reflection-result")


@pytest.mark.parametrize("release_time_s", [0.1, 0.2])
@pytest.mark.parametrize(
    ("law", "expected_lifecycle", "expected_outcome", "expected_velocity_m_s"),
    [("stick", 2, "stuck", [0.0, 0.0]), ("hold", 5, "held", [-1.0, 0.0])],
)
def test_brownian_surface_release_is_right_continuous_at_macro_boundaries(
    tmp_path: Path,
    release_time_s: float,
    law: str,
    expected_lifecycle: int,
    expected_outcome: str,
    expected_velocity_m_s: list[float],
) -> None:
    case_path = _brownian_surface_case(
        tmp_path / f"surface-{law}-{release_time_s}",
        release_time_s,
        law=law,
    )
    output = tmp_path / f"surface-result-{law}-{release_time_s}"

    simulate(load_case(case_path), output)
    result = open_result(output)
    final = result.read_final()
    events = result.read_boundary_events()
    frame = next(iter(result.iter_frames()))

    np.testing.assert_array_equal(final.lifecycle, [expected_lifecycle])
    np.testing.assert_array_equal(events.time_s, [release_time_s])
    np.testing.assert_array_equal(events.outcome, [expected_outcome])
    np.testing.assert_array_equal(events.velocity_post_m_s, [expected_velocity_m_s])
    np.testing.assert_array_equal(frame.lifecycle, [expected_lifecycle])
    np.testing.assert_array_equal(frame.velocity_m_s, [expected_velocity_m_s])


def _brownian_rz_case(
    directory: Path,
    *,
    particle_count: int,
    initial_position_m: np.ndarray,
    initial_velocity_m_s: np.ndarray,
    gas_velocity_m_s: np.ndarray,
    radial_shift_m: float,
    end_s: float,
    dt_s: float,
    tree_depth: int,
    gravity_m_s2: np.ndarray | None,
    frame_times: list[float] | None,
    memory_limit_mb: int,
    initial_charge_number: float = 3.0,
    continuous_charge: bool = False,
    number_density_m3: float = 1.0e6,
) -> Path:
    paths = materialize_microcase("C02", directory)
    case = load_case(paths.case_path)
    count = particle_count
    geometry = replace(
        case.data.geometry,
        nodes_m=case.data.geometry.nodes_m + np.asarray([radial_shift_m, 0.0]),
    )
    layouts = []
    for layout in case.data.layouts:
        if not isinstance(layout, RegularLayout):
            raise TypeError("RZ Brownian scenario requires a regular field layout")
        layouts.append(replace(layout, axis0_m=layout.axis0_m + radial_shift_m))
    fields = []
    for field in case.data.fields:
        if field.name == "gas_velocity":
            field = replace(
                field,
                components=("r", "z"),
                stored_basis="axisymmetric_rz",
                values=np.broadcast_to(gas_velocity_m_s, field.values.shape).copy(),
            )
        fields.append(field)
    if continuous_charge:
        by_name = {field.name: field for field in fields}
        scalar_template = by_name["gas_temperature"]
        vector_template = by_name["gas_velocity"]
        fields.extend(
            (
                replace(
                    scalar_template,
                    name="electron_density",
                    unit="1/m^3",
                    values=np.full_like(scalar_template.values, number_density_m3),
                ),
                replace(
                    scalar_template,
                    name="ion_density",
                    unit="1/m^3",
                    values=np.full_like(scalar_template.values, number_density_m3),
                ),
                replace(
                    scalar_template,
                    name="electron_temperature",
                    values=np.full_like(scalar_template.values, 20_000.0),
                ),
                replace(
                    scalar_template,
                    name="ion_temperature",
                    values=np.full_like(scalar_template.values, 300.0),
                ),
                replace(
                    vector_template,
                    name="ion_velocity",
                    values=np.zeros_like(vector_template.values),
                ),
            )
        )
    source = replace(
        case.data.sources[0],
        particle_id=np.arange(1, count + 1, dtype="<i8"),
        release_time_s=np.zeros(count, dtype="<f8"),
        position_m=np.broadcast_to(initial_position_m, (count, 2)).copy(),
        velocity_m_s=np.broadcast_to(initial_velocity_m_s, (count, 2)).copy(),
        charge_number=np.full(count, initial_charge_number, dtype="<f8"),
        mass_kg=np.full(count, _MASS_KG, dtype="<f8"),
        drag_diameter_m=np.full(count, case.data.sources[0].drag_diameter_m[0], dtype="<f8"),
        electrostatic_radius_m=np.full(
            count,
            case.data.sources[0].electrostatic_radius_m[0],
            dtype="<f8",
        ),
        displaced_volume_m3=np.zeros(count, dtype="<f8"),
        model_weight=np.ones(count, dtype="<f8"),
        material_id=np.zeros(count, dtype="<i4"),
    )
    directory.mkdir(parents=True, exist_ok=True)
    data_path = directory / "brownian-rz.h5"
    info = write(
        data_path,
        replace(
            case.data,
            coordinate_system="axisymmetric_rz",
            geometry=geometry,
            layouts=tuple(layouts),
            fields=tuple(fields),
            sources=(source,),
        ),
    )
    document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
    document["case"]["data_path"] = data_path.name
    document["case"]["expected_content_hash"] = info.content_hash
    document["motion"]["mode"] = "axisymmetric_rz_meridional"
    document["time"] = {"start_s": 0.0, "end_s": end_s, "dt_s": dt_s}
    document["solver"]["integrator"] = "ou_langevin"
    document["solver"]["seed"] = 9_741
    document["resources"]["memory_limit_mb"] = memory_limit_mb
    charge_model: dict[str, object] = {"model": "fixed"}
    if continuous_charge:
        charge_model = {
            "model": "plasma_continuous",
            "revision": "oml_stationary_maxwellian_debye_huckel_v1",
            "electron_number_density_field": "electron_density",
            "positive_ion_number_density_field": "ion_density",
            "electron_temperature_field": "electron_temperature",
            "positive_ion_temperature_field": "ion_temperature",
            "positive_ion_velocity_field": "ion_velocity",
            "positive_ion_mass_kg": 6.6335209e-26,
            "applicability": "error",
        }
    physics = {
        "charge": charge_model,
        "drag": dict(case.spec.physics.models["drag"]),
        "noise": {
            "model": "inertial_langevin_fdt",
            "revision": _RZ_NOISE_REVISION,
            "interval_tree_depth": tree_depth,
        },
    }
    if gravity_m_s2 is not None:
        physics["gravity_buoyancy"] = {
            "model": "standard",
            "revision": "gravity_buoyancy_standard_v1",
            "gas_density_field": "gas_density",
            "gravity_m_s2": gravity_m_s2.tolist(),
        }
    document["physics"] = physics
    document["output"]["trajectories"] = (
        None
        if frame_times is None
        else {"selection": "all", "schedule": {"explicit_times_s": frame_times}}
    )
    case_path = directory / "brownian-rz.yaml"
    case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return case_path


def _brownian_case(
    directory: Path,
    *,
    particle_count: int,
    end_s: float,
    dt_s: float,
    tree_depth: int,
    frame_times: list[float] | None,
) -> Path:
    paths = materialize_microcase("C02", directory)
    case = load_case(paths.case_path)
    source = case.data.sources[0]
    count = particle_count
    data = replace(
        case.data,
        sources=(
            replace(
                source,
                particle_id=np.arange(1, count + 1, dtype="<i8"),
                release_time_s=np.zeros(count, dtype="<f8"),
                position_m=np.zeros((count, 2), dtype="<f8"),
                velocity_m_s=np.broadcast_to(_INITIAL_VELOCITY_M_S, (count, 2)).copy(),
                charge_number=np.zeros(count, dtype="<f8"),
                mass_kg=np.full(count, _MASS_KG, dtype="<f8"),
                drag_diameter_m=np.full(count, source.drag_diameter_m[0], dtype="<f8"),
                electrostatic_radius_m=np.full(
                    count,
                    source.electrostatic_radius_m[0],
                    dtype="<f8",
                ),
                displaced_volume_m3=np.zeros(count, dtype="<f8"),
                model_weight=np.ones(count, dtype="<f8"),
                material_id=np.zeros(count, dtype="<i4"),
            ),
        ),
    )
    data_path = directory / "brownian.h5"
    info = write(data_path, data)
    document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
    document["case"]["data_path"] = data_path.name
    document["case"]["expected_content_hash"] = info.content_hash
    document["time"] = {"start_s": 0.0, "end_s": end_s, "dt_s": dt_s}
    document["solver"]["integrator"] = "ou_langevin"
    document["solver"]["seed"] = 1741
    document["physics"]["noise"] = {
        "model": "inertial_langevin_fdt",
        "revision": _NOISE_REVISION,
        "interval_tree_depth": tree_depth,
    }
    document["output"]["trajectories"] = (
        None
        if frame_times is None
        else {"selection": "all", "schedule": {"explicit_times_s": frame_times}}
    )
    case_path = directory / "brownian.yaml"
    case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return case_path


def _brownian_surface_case(directory: Path, release_time_s: float, *, law: str) -> Path:
    surface_paths = materialize_microcase("C08", directory / "surface")
    field_paths = materialize_microcase("C02", directory / "field")
    surface_case = load_case(surface_paths.case_path)
    field_case = load_case(field_paths.case_path)
    data_path = directory / "brownian-surface.h5"
    directory.mkdir(parents=True, exist_ok=True)
    info = write(
        data_path,
        replace(
            surface_case.data,
            layouts=field_case.data.layouts,
            fields=field_case.data.fields,
        ),
    )
    document = yaml.safe_load(surface_paths.case_path.read_text(encoding="utf-8"))
    document["case"]["data_path"] = data_path.name
    document["case"]["expected_content_hash"] = info.content_hash
    document["time"] = {"start_s": 0.0, "end_s": 0.2, "dt_s": 0.1}
    document["solver"]["integrator"] = "ou_langevin"
    document["solver"]["seed"] = 414
    document["sources"][0]["velocity"]["value_m_s"] = [-1.0, 0.0]
    document["sources"][0]["release"]["time_s"] = release_time_s
    document["physics"] = {
        "charge": {"model": "fixed"},
        "drag": dict(field_case.spec.physics.models["drag"]),
        "noise": {
            "model": "inertial_langevin_fdt",
            "revision": _NOISE_REVISION,
            "interval_tree_depth": 2,
        },
    }
    document["boundaries"] = [
        {"boundary_group": group, "priority": priority, "law": law}
        for priority, group in enumerate(("collector", "mirror", "caps"), start=1)
    ]
    document["output"]["trajectories"] = {
        "selection": "all",
        "schedule": {"explicit_times_s": [release_time_s]},
    }
    case_path = directory / "brownian-surface.yaml"
    case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return case_path


def _brownian_first_passage_case(
    directory: Path,
    *,
    particle_count: int,
    tree_depth: int,
    seed: int,
    rz_b03: bool = False,
    particle_id_start: int = 1,
) -> Path:
    wall_paths = materialize_microcase("C07", directory / "wall")
    field_paths = materialize_microcase("C02", directory / "field")
    wall_case = load_case(wall_paths.case_path)
    field_case = load_case(field_paths.case_path)
    radial_shift_m = 0.002 if rz_b03 else 0.0
    geometry = replace(
        wall_case.data.geometry,
        nodes_m=(
            wall_case.data.geometry.nodes_m * np.asarray([0.002, 1.0])
            + np.asarray([radial_shift_m, 0.0])
        ),
    )
    source = wall_case.data.sources[0]
    count = particle_count
    source = replace(
        source,
        particle_id=np.arange(
            particle_id_start,
            particle_id_start + count,
            dtype="<i8",
        ),
        release_time_s=np.zeros(count, dtype="<f8"),
        position_m=np.broadcast_to(
            np.asarray([radial_shift_m + 0.001, 0.5]),
            (count, 2),
        ).copy(),
        velocity_m_s=np.zeros((count, 2), dtype="<f8"),
        charge_number=np.zeros(count, dtype="<f8"),
        mass_kg=np.full(count, _MASS_KG, dtype="<f8"),
        drag_diameter_m=np.full(count, source.drag_diameter_m[0], dtype="<f8"),
        electrostatic_radius_m=np.full(
            count,
            source.electrostatic_radius_m[0],
            dtype="<f8",
        ),
        displaced_volume_m3=np.zeros(count, dtype="<f8"),
        model_weight=np.ones(count, dtype="<f8"),
        material_id=np.zeros(count, dtype="<i4"),
    )
    fields = []
    for field in field_case.data.fields:
        if field.name == "gas_velocity":
            field = replace(field, values=np.zeros_like(field.values))
            if rz_b03:
                field = replace(
                    field,
                    components=("r", "z"),
                    stored_basis="axisymmetric_rz",
                )
        fields.append(field)
    layouts = field_case.data.layouts
    if rz_b03:
        layouts = tuple(
            replace(
                layout,
                axis0_m=np.asarray([0.0, 0.006], dtype="<f8"),
                axis1_m=np.asarray([0.0, 1.0], dtype="<f8"),
            )
            for layout in layouts
            if isinstance(layout, RegularLayout)
        )
        if len(layouts) != len(field_case.data.layouts):
            raise TypeError("RZ Brownian first-passage scenario requires regular layouts")
    data = replace(
        wall_case.data,
        coordinate_system=("axisymmetric_rz" if rz_b03 else "cartesian_xy"),
        geometry=geometry,
        layouts=layouts,
        fields=tuple(fields),
        sources=(source,),
    )
    directory.mkdir(parents=True, exist_ok=True)
    data_path = directory / "first-passage.h5"
    info = write(data_path, data)
    document = yaml.safe_load(wall_paths.case_path.read_text(encoding="utf-8"))
    document["case"]["data_path"] = data_path.name
    document["case"]["expected_content_hash"] = info.content_hash
    document["time"] = {"start_s": 0.0, "end_s": 1.0, "dt_s": 1.0}
    if rz_b03:
        document["motion"]["mode"] = "axisymmetric_rz_meridional"
    document["solver"]["integrator"] = "ou_langevin"
    document["solver"]["seed"] = seed
    document["physics"] = {
        "charge": {"model": "fixed"},
        "drag": dict(field_case.spec.physics.models["drag"]),
        "noise": {
            "model": "inertial_langevin_fdt",
            "revision": _RZ_NOISE_REVISION if rz_b03 else _NOISE_REVISION,
            "interval_tree_depth": tree_depth,
        },
    }
    document["boundaries"] = [{"boundary_group": "wall", "priority": 10, "law": "stick"}]
    document["output"]["trajectories"] = None
    case_path = directory / "first-passage.yaml"
    case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return case_path


def _brownian_wall_case(
    directory: Path,
    *,
    law: str,
    frame_times: list[float] | None = None,
) -> Path:
    wall_paths = materialize_microcase("C07", directory / "wall")
    field_paths = materialize_microcase("C02", directory / "field")
    wall_case = load_case(wall_paths.case_path)
    field_case = load_case(field_paths.case_path)
    source = replace(
        wall_case.data.sources[0],
        velocity_m_s=np.asarray([[2.0, 0.0]], dtype="<f8"),
        charge_number=np.asarray([0.0], dtype="<f8"),
    )
    data = replace(
        wall_case.data,
        layouts=field_case.data.layouts,
        fields=field_case.data.fields,
        sources=(source,),
    )
    directory.mkdir(parents=True, exist_ok=True)
    data_path = directory / "brownian-wall.h5"
    info = write(data_path, data)
    document = yaml.safe_load(wall_paths.case_path.read_text(encoding="utf-8"))
    document["case"]["data_path"] = data_path.name
    document["case"]["expected_content_hash"] = info.content_hash
    document["time"] = {"start_s": 0.0, "end_s": 2.0, "dt_s": 2.0}
    document["solver"]["integrator"] = "ou_langevin"
    document["solver"]["seed"] = 91
    document["physics"] = {
        "charge": {"model": "fixed"},
        "drag": dict(field_case.spec.physics.models["drag"]),
        "noise": {
            "model": "inertial_langevin_fdt",
            "revision": _NOISE_REVISION,
            "interval_tree_depth": 4,
        },
    }
    document["boundaries"] = [{"boundary_group": "wall", "priority": 10, "law": law}]
    document["output"]["trajectories"] = (
        {"selection": "all", "schedule": {"explicit_times_s": frame_times}}
        if frame_times is not None
        else None
    )
    case_path = directory / "brownian-wall.yaml"
    case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return case_path


def _assert_final_identity(first: object, second: object) -> None:
    for name in (
        "particle_id",
        "source_id",
        "time_s",
        "position_m",
        "velocity_m_s",
        "charge_number",
        "lifecycle",
        "kinematics_valid",
        "failure_reason_code",
    ):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))


def _assert_record_identity(first: Any, second: Any) -> None:
    field_names = first.__dataclass_fields__
    assert field_names.keys() == second.__dataclass_fields__.keys()
    for name in field_names:
        expected = getattr(first, name)
        actual = getattr(second, name)
        if isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
        else:
            assert actual == expected
