from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Literal

import numpy as np
import yaml

from chamber_particles import load_case, open_result, simulate
from chamber_particles.case_format import RegularLayout, write
from tests.verification.microcases import materialize_microcase

_MASS_KG = 4.0e-15
_DRAG_DIAMETER_M = 2.0e-6
_GAS_DENSITY_KG_M3 = 0.1
_GAS_DYNAMIC_VISCOSITY_PA_S = 1.8e-5
_GAS_MEAN_FREE_PATH_M = 1.0e-6
_GAS_VELOCITY_M_S = np.asarray([0.1, -0.05])
_INITIAL_POSITION_M = np.asarray([-0.4, 0.2])
_INITIAL_VELOCITY_M_S = np.asarray([0.9, 0.35])
_END_TIME_S = 1.0e-4


def test_stokes_cunningham_xy_has_fourth_order_and_schedule_independent_final(
    tmp_path: Path,
) -> None:
    position_exact, velocity_exact, rate_s_inv = _analytic_final()
    position_errors: list[float] = []
    velocity_errors: list[float] = []
    results = []

    for index, dt_s in enumerate((2.0e-5, 1.0e-5, 5.0e-6)):
        case_path = _stokes_case(tmp_path / f"xy-dt-{index}", dt_s=dt_s)
        output = tmp_path / f"xy-result-{index}"
        simulate(load_case(case_path), output)
        result = open_result(output)
        final = result.read_final()
        position_errors.append(float(np.max(np.abs(final.position_m[0] - position_exact))))
        velocity_errors.append(float(np.max(np.abs(final.velocity_m_s[0] - velocity_exact))))
        prepared_ratio = float(result.manifest["maximum_dt_over_tau"])
        exact_ratio = dt_s * rate_s_inv
        assert exact_ratio <= prepared_ratio <= exact_ratio * (1.0 + 1.0e-12)
        assert result.manifest["resolved"]["physics_models"]["drag"] == {
            "model": "stokes_cunningham",
            "revision": "stokes_cunningham_allen_raabe_air_v1",
        }
        assert result.manifest["resolved"]["path_kind"] == "rk4_dense"
        results.append(result)

    _assert_observed_order(position_errors, minimum_order=3.5)
    _assert_observed_order(velocity_errors, minimum_order=3.5)

    framed_path = _stokes_case(
        tmp_path / "xy-framed",
        dt_s=5.0e-6,
        frame_times=[0.0, 3.7e-5, _END_TIME_S],
    )
    simulate(load_case(framed_path), tmp_path / "xy-framed-result")
    framed = open_result(tmp_path / "xy-framed-result")
    unframed_final = results[-1].read_final()
    framed_final = framed.read_final()
    np.testing.assert_array_equal(framed_final.position_m, unframed_final.position_m)
    np.testing.assert_array_equal(framed_final.velocity_m_s, unframed_final.velocity_m_s)
    np.testing.assert_array_equal(framed_final.charge_number, unframed_final.charge_number)
    np.testing.assert_array_equal(framed_final.lifecycle, unframed_final.lifecycle)

    knudsen_radius = 2.0 * _GAS_MEAN_FREE_PATH_M / _DRAG_DIAMETER_M
    initial_reynolds = (
        _GAS_DENSITY_KG_M3
        * _DRAG_DIAMETER_M
        * float(np.linalg.norm(_INITIAL_VELOCITY_M_S - _GAS_VELOCITY_M_S))
        / _GAS_DYNAMIC_VISCOSITY_PA_S
    )
    assert 0.03 <= knudsen_radius <= 7.2
    assert initial_reynolds <= 0.1


def test_stokes_cunningham_exponential_midpoint_matches_the_closed_form(
    tmp_path: Path,
) -> None:
    """The second drag model uses the same native linear-relaxation strategy."""

    position_exact, velocity_exact, _ = _analytic_final()
    case_path = _stokes_case(tmp_path / "exponential", dt_s=_END_TIME_S)
    document = yaml.safe_load(case_path.read_text(encoding="utf-8"))
    document["solver"]["integrator"] = "exponential_midpoint"
    case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    output = tmp_path / "exponential-result"

    simulate(load_case(case_path), output)
    result = open_result(output)
    final = result.read_final()

    np.testing.assert_allclose(final.position_m[0], position_exact, rtol=0.0, atol=2.0e-15)
    np.testing.assert_allclose(final.velocity_m_s[0], velocity_exact, rtol=0.0, atol=2.0e-14)
    assert result.manifest["resolved"]["path_kind"] == "exponential_midpoint_reintegrated"
    assert result.manifest["resolved"]["physics_models"]["drag"] == {
        "model": "stokes_cunningham",
        "revision": "stokes_cunningham_allen_raabe_air_v1",
    }


def test_stokes_cunningham_rz_matches_xy_away_from_axis(tmp_path: Path) -> None:
    xy_path = _stokes_case(tmp_path / "xy-parity", dt_s=5.0e-6)
    rz_path = _stokes_case(
        tmp_path / "rz-parity",
        dt_s=5.0e-6,
        coordinate_system="axisymmetric_rz",
    )
    simulate(load_case(xy_path), tmp_path / "xy-parity-result")
    simulate(load_case(rz_path), tmp_path / "rz-parity-result")
    xy = open_result(tmp_path / "xy-parity-result")
    rz = open_result(tmp_path / "rz-parity-result")
    radial_shift = np.asarray([1.5, 0.0])

    np.testing.assert_allclose(
        rz.read_final().position_m - radial_shift,
        xy.read_final().position_m,
        rtol=0.0,
        atol=5.0e-16,
    )
    np.testing.assert_array_equal(rz.read_final().velocity_m_s, xy.read_final().velocity_m_s)
    np.testing.assert_array_equal(rz.read_final().charge_number, xy.read_final().charge_number)
    assert rz.manifest["boundary_interactions"]["axis_crossings"] == 0
    assert rz.manifest["resolved"]["path_kind"] == "rk4_dense"


def test_stokes_cunningham_out_of_range_knudsen_is_local_failure(tmp_path: Path) -> None:
    case_path = _stokes_case(tmp_path / "knudsen-failure", dt_s=5.0e-6)
    case = load_case(case_path)
    fields = tuple(
        replace(field, values=np.full_like(field.values, 1.0e-9))
        if field.name == "gas_mean_free_path"
        else field
        for field in case.data.fields
    )
    data_path = case_path.with_name("knudsen-failure.h5")
    info = write(data_path, replace(case.data, fields=fields))
    document = yaml.safe_load(case_path.read_text(encoding="utf-8"))
    document["case"]["data_path"] = data_path.name
    document["case"]["expected_content_hash"] = info.content_hash
    failing_case_path = case_path.with_name("knudsen-failure.yaml")
    failing_case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    output = tmp_path / "knudsen-failure-result"

    simulate(load_case(failing_case_path), output)
    result = open_result(output)
    final = result.read_final()
    failure = result.read_failure_events()
    reason = result.manifest["failure_reason_codes"]["model_applicability"]
    np.testing.assert_array_equal(final.lifecycle, [4])
    np.testing.assert_array_equal(final.failure_reason_code, [reason])
    np.testing.assert_array_equal(failure.reason_code, [reason])


def test_continuous_applicability_overflow_fails_only_the_affected_particle(
    tmp_path: Path,
) -> None:
    case_path = _stokes_case(tmp_path / "continuous-overflow", dt_s=1.0e-22)
    case = load_case(case_path)
    original_layout = case.data.layouts[0]
    if not isinstance(original_layout, RegularLayout):
        raise TypeError("Stokes-Cunningham scenario requires a regular field layout")
    layout = replace(
        original_layout,
        axis0_m=np.asarray([-1.0, 0.0, 1.0], dtype="<f8"),
        axis1_m=np.asarray([-1.0, 0.0, 1.0], dtype="<f8"),
        cell_support=np.ones((2, 2), dtype="<u1"),
    )
    fields = []
    for field in case.data.fields:
        if field.name == "gas_velocity":
            values = np.zeros((9, 2), dtype="<f8")
        elif field.name == "gas_density":
            values = np.full((9, 1), 1.0e155, dtype="<f8")
        elif field.name == "gas_dynamic_viscosity":
            values = np.full((9, 1), 1.0e-155, dtype="<f8")
        elif field.name == "gas_mean_free_path":
            values = np.full((9, 1), 1.0e-6, dtype="<f8")
        else:
            raise AssertionError(f"unexpected Stokes field: {field.name}")
        fields.append(replace(field, values=values))

    original = case.data.sources[0]
    source = replace(
        original,
        particle_id=np.asarray([201, 202], dtype="<i8"),
        release_time_s=np.zeros(2, dtype="<f8"),
        position_m=np.asarray([[0.75, 0.2], [-0.4, 0.2]], dtype="<f8"),
        velocity_m_s=np.asarray([[1.0e10, 0.0], [0.0, 0.0]], dtype="<f8"),
        charge_number=np.zeros(2, dtype="<f8"),
        mass_kg=np.asarray([1.0e100, 4.0e-15], dtype="<f8"),
        drag_diameter_m=np.full(2, 2.0e-6, dtype="<f8"),
        electrostatic_radius_m=np.full(2, 1.0e-6, dtype="<f8"),
        contact_radius_m=np.zeros(2, dtype="<f8"),
        displaced_volume_m3=np.zeros(2, dtype="<f8"),
        model_weight=np.ones(2, dtype="<f8"),
        material_id=np.zeros(2, dtype="<i4"),
    )
    data_path = case_path.with_name("continuous-overflow.h5")
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
    document["time"] = {"start_s": 0.0, "end_s": 1.0e-22, "dt_s": 1.0e-22}
    overflow_path = case_path.with_name("continuous-overflow.yaml")
    overflow_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    output = tmp_path / "continuous-overflow-result"

    simulate(load_case(overflow_path), output)
    result = open_result(output)
    final = result.read_final()
    failure = result.read_failure_events()
    reason = result.manifest["failure_reason_codes"]["nonfinite_physics"]

    np.testing.assert_array_equal(final.lifecycle, [4, 1])
    np.testing.assert_array_equal(final.failure_reason_code, [reason, 0])
    np.testing.assert_array_equal(failure.particle_id, [201])
    np.testing.assert_array_equal(failure.reason_code, [reason])


def test_dense_local_applicability_separates_remote_extrema_actual_failure_and_exhaustion(
    tmp_path: Path,
) -> None:
    """One mixed batch distinguishes physics evidence from an unproved range."""

    reference_path = _local_applicability_case(
        tmp_path / "local-reference",
        remote_velocity_m_s=0.0,
    )
    extreme_path = _local_applicability_case(
        tmp_path / "local-extreme",
        remote_velocity_m_s=100.0,
    )
    simulate(load_case(reference_path), tmp_path / "local-reference-result")
    simulate(load_case(extreme_path), tmp_path / "local-extreme-result")
    reference = open_result(tmp_path / "local-reference-result")
    extreme = open_result(tmp_path / "local-extreme-result")
    reference_final = reference.read_final()
    final = extreme.read_final()
    failure = extreme.read_failure_events()
    model_reason = extreme.manifest["failure_reason_codes"]["model_applicability"]
    certificate_reason = extreme.manifest["failure_reason_codes"][
        "indeterminate_applicability_certificate"
    ]

    # Particle 301 remains in the remote-extreme-free left cell and therefore
    # has exactly the same accepted endpoint as the all-safe reference run.
    np.testing.assert_array_equal(final.position_m[0], reference_final.position_m[0])
    np.testing.assert_array_equal(final.velocity_m_s[0], reference_final.velocity_m_s[0])
    assert final.lifecycle[0] == 1
    assert final.failure_reason_code[0] == 0

    # Particle 302 travels on the shared x=0 edge.  Every actual dense sample
    # is valid, while the cell-extrema certificate includes the extreme right
    # cell and cannot prove validity within the configured bounded split work.
    # Particle 303 samples the extreme cell itself, providing actual model
    # applicability evidence rather than a certificate-only miss.
    np.testing.assert_array_equal(final.lifecycle, [1, 4, 4])
    np.testing.assert_array_equal(
        final.failure_reason_code,
        [0, certificate_reason, model_reason],
    )
    np.testing.assert_array_equal(failure.particle_id, [302, 303])
    np.testing.assert_array_equal(
        failure.reason_code,
        [certificate_reason, model_reason],
    )


def test_dense_local_applicability_preserves_output_and_slab_identity(
    tmp_path: Path,
) -> None:
    """Output sampling and bounded slab width do not change certificate outcomes."""

    particle_count = 512
    case_path = _local_applicability_case(
        tmp_path / "local-identity",
        remote_velocity_m_s=100.0,
        particle_count=particle_count,
    )
    document = yaml.safe_load(case_path.read_text(encoding="utf-8"))
    results = []
    for label, memory_limit_mb, trajectories in (
        (
            "tight-framed",
            4,
            {"selection": "all", "schedule": {"explicit_times_s": [0.0, 5.0e-5, 1.0e-4]}},
        ),
        ("loose-unframed", 64, None),
    ):
        variant = {
            **document,
            "resources": {**document["resources"], "memory_limit_mb": memory_limit_mb},
            "output": {"trajectories": trajectories},
        }
        variant_path = case_path.with_name(f"local-identity-{label}.yaml")
        variant_path.write_text(yaml.safe_dump(variant, sort_keys=False), encoding="utf-8")
        output = tmp_path / f"local-identity-{label}-result"
        simulate(load_case(variant_path), output)
        results.append(open_result(output))

    tight, loose = results
    assert tight.manifest["memory_plan"]["slab_particles"] < particle_count
    assert loose.manifest["memory_plan"]["slab_particles"] == particle_count
    for name in (
        "particle_id",
        "time_s",
        "position_m",
        "velocity_m_s",
        "charge_number",
        "lifecycle",
        "failure_reason_code",
    ):
        np.testing.assert_array_equal(
            getattr(tight.read_final(), name),
            getattr(loose.read_final(), name),
        )
    for name in ("time_s", "particle_id", "event_ordinal", "reason_code"):
        np.testing.assert_array_equal(
            getattr(tight.read_failure_events(), name),
            getattr(loose.read_failure_events(), name),
        )
    np.testing.assert_array_equal(
        tight.read_boundary_events().particle_id,
        loose.read_boundary_events().particle_id,
    )
    assert tight.manifest["failure_reason_counts"] == loose.manifest["failure_reason_counts"]


def _local_applicability_case(
    directory: Path,
    *,
    remote_velocity_m_s: float,
    particle_count: int = 3,
) -> Path:
    case_path = _stokes_case(directory, dt_s=5.0e-6)
    case = load_case(case_path)
    original_layout = case.data.layouts[0]
    if not isinstance(original_layout, RegularLayout):
        raise TypeError("Stokes-Cunningham scenario requires a regular field layout")
    layout = replace(
        original_layout,
        axis0_m=np.asarray([-1.0, 0.0, 1.0], dtype="<f8"),
        axis1_m=np.asarray([-1.0, 1.0], dtype="<f8"),
        cell_support=np.ones((2, 1), dtype="<u1"),
    )
    fields = []
    for field in case.data.fields:
        if field.name == "gas_velocity":
            values = np.broadcast_to(np.asarray([0.0, 0.1]), (6, 2)).copy()
            values[4:, 0] = remote_velocity_m_s
        elif field.name == "gas_density":
            values = np.full((6, 1), _GAS_DENSITY_KG_M3, dtype="<f8")
        elif field.name == "gas_dynamic_viscosity":
            values = np.full((6, 1), _GAS_DYNAMIC_VISCOSITY_PA_S, dtype="<f8")
        elif field.name == "gas_mean_free_path":
            values = np.full((6, 1), _GAS_MEAN_FREE_PATH_M, dtype="<f8")
        else:
            raise AssertionError(f"unexpected Stokes field: {field.name}")
        fields.append(replace(field, values=values))

    original = case.data.sources[0]
    count = particle_count

    def repeated(values: np.ndarray) -> np.ndarray:
        return np.repeat(values[:1], count, axis=0)

    source = replace(
        original,
        particle_id=np.arange(301, 301 + count, dtype="<i8"),
        release_time_s=np.zeros(count, dtype="<f8"),
        position_m=np.tile(
            np.asarray([[-0.5, 0.2], [0.0, 0.2], [0.75, 0.2]], dtype="<f8"),
            ((count + 2) // 3, 1),
        )[:count],
        velocity_m_s=np.repeat(np.asarray([[0.0, 0.1]], dtype="<f8"), count, axis=0),
        charge_number=repeated(original.charge_number),
        mass_kg=repeated(original.mass_kg),
        drag_diameter_m=repeated(original.drag_diameter_m),
        electrostatic_radius_m=repeated(original.electrostatic_radius_m),
        contact_radius_m=repeated(original.contact_radius_m),
        displaced_volume_m3=repeated(original.displaced_volume_m3),
        model_weight=repeated(original.model_weight),
        material_id=repeated(original.material_id),
    )
    data_path = case_path.with_name("local-applicability.h5")
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
    document["solver"]["event"]["max_refinements"] = 2
    document["output"]["trajectories"] = None
    result_path = case_path.with_name("local-applicability.yaml")
    result_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return result_path


def _stokes_case(
    directory: Path,
    *,
    dt_s: float,
    coordinate_system: Literal["cartesian_xy", "axisymmetric_rz"] = "cartesian_xy",
    frame_times: list[float] | None = None,
) -> Path:
    paths = materialize_microcase("C02", directory)
    case = load_case(paths.case_path)
    radial_shift = np.asarray([1.5, 0.0]) if coordinate_system == "axisymmetric_rz" else np.zeros(2)
    geometry = replace(case.data.geometry, nodes_m=case.data.geometry.nodes_m + radial_shift)
    layouts = []
    for layout in case.data.layouts:
        if not isinstance(layout, RegularLayout):
            raise TypeError("Stokes-Cunningham scenario requires a regular field layout")
        layouts.append(replace(layout, axis0_m=layout.axis0_m + radial_shift[0]))

    fields = []
    for field in case.data.fields:
        if field.name == "gas_velocity":
            updated = replace(
                field,
                values=np.broadcast_to(_GAS_VELOCITY_M_S, field.values.shape).copy(),
            )
        elif field.name == "gas_density":
            updated = replace(field, values=np.full_like(field.values, _GAS_DENSITY_KG_M3))
        elif field.name == "gas_temperature":
            updated = replace(
                field,
                name="gas_dynamic_viscosity",
                unit="Pa*s",
                values=np.full_like(field.values, _GAS_DYNAMIC_VISCOSITY_PA_S),
            )
        elif field.name == "gas_mean_free_path":
            updated = replace(
                field,
                values=np.full_like(field.values, _GAS_MEAN_FREE_PATH_M),
            )
        else:
            raise AssertionError(f"unexpected C02 field: {field.name}")
        if coordinate_system == "axisymmetric_rz" and len(updated.components) == 2:
            updated = replace(
                updated,
                components=("r", "z"),
                stored_basis="axisymmetric_rz",
            )
        fields.append(updated)

    source = replace(
        case.data.sources[0], position_m=case.data.sources[0].position_m + radial_shift
    )
    data_path = paths.case_path.with_name("stokes-cunningham.h5")
    info = write(
        data_path,
        replace(
            case.data,
            coordinate_system=coordinate_system,
            geometry=geometry,
            layouts=tuple(layouts),
            fields=tuple(fields),
            sources=(source,),
        ),
    )
    document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
    document["case"]["data_path"] = data_path.name
    document["case"]["expected_content_hash"] = info.content_hash
    document["motion"]["mode"] = (
        "axisymmetric_rz_meridional" if coordinate_system == "axisymmetric_rz" else "cartesian_xy"
    )
    document["time"] = {"start_s": 0.0, "end_s": _END_TIME_S, "dt_s": dt_s}
    document["physics"] = {
        "charge": {"model": "fixed"},
        "drag": {
            "model": "stokes_cunningham",
            "revision": "stokes_cunningham_allen_raabe_air_v1",
            "gas_velocity_field": "gas_velocity",
            "gas_density_field": "gas_density",
            "gas_dynamic_viscosity_field": "gas_dynamic_viscosity",
            "gas_mean_free_path_field": "gas_mean_free_path",
            "applicability": "error",
        },
    }
    document["output"]["trajectories"] = (
        None
        if frame_times is None
        else {"selection": "all", "schedule": {"explicit_times_s": frame_times}}
    )
    case_path = paths.case_path.with_name("stokes-cunningham.yaml")
    case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return case_path


def _analytic_final() -> tuple[np.ndarray, np.ndarray, float]:
    knudsen_radius = 2.0 * _GAS_MEAN_FREE_PATH_M / _DRAG_DIAMETER_M
    slip_correction = 1.0 + knudsen_radius * (1.142 + 0.558 * math.exp(-0.999 / knudsen_radius))
    rate_s_inv = (
        3.0
        * math.pi
        * _GAS_DYNAMIC_VISCOSITY_PA_S
        * _DRAG_DIAMETER_M
        / (slip_correction * _MASS_KG)
    )
    decay = math.exp(-rate_s_inv * _END_TIME_S)
    relative_velocity = _INITIAL_VELOCITY_M_S - _GAS_VELOCITY_M_S
    position = (
        _INITIAL_POSITION_M
        + _GAS_VELOCITY_M_S * _END_TIME_S
        + relative_velocity * (1.0 - decay) / rate_s_inv
    )
    velocity = _GAS_VELOCITY_M_S + decay * relative_velocity
    return position, velocity, rate_s_inv


def _assert_observed_order(errors: list[float], *, minimum_order: float) -> None:
    assert errors[0] > errors[1] > errors[2]
    orders = np.log2(np.asarray(errors[:-1]) / np.asarray(errors[1:]))
    assert min(orders) >= minimum_order
