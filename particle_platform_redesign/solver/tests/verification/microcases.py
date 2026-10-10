"""Independent analytic inputs and oracles for the C01--C10 verification pack.

This module is test material, not a production solver.  It deliberately uses
only closed-form mathematics and the canonical case writer.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from chamber_particles.case_format import (
    BoundaryData,
    DataBundle,
    FieldData,
    GeometryData,
    P1TriLayout,
    Q1QuadLayout,
    RealizedSource,
    RealizedSurfaceSource,
    RealizedTableSource,
    RegularLayout,
    write,
)

MICROCASE_IDS = tuple(f"C{index:02d}" for index in range(1, 11))

_ELEMENTARY_CHARGE_C = 1.602176634e-19
_BOLTZMANN_J_K = 1.380649e-23
_GAS_MOLECULE_MASS_KG = 4.65e-26
_GAS_TEMPERATURE_K = 300.0
_COMMON_MASS_KG = 4.0e-15
_COMMON_DRAG_DIAMETER_M = 2.0e-6
_COMMON_ELECTROSTATIC_RADIUS_M = 1.0e-6


@dataclass(frozen=True, slots=True)
class MicrocaseDefinition:
    """One canonical input plus an independent expected-result document."""

    data: DataBundle
    spec: dict[str, Any]
    expected: dict[str, Any]


@dataclass(frozen=True, slots=True)
class MaterializedMicrocase:
    """Paths created for one temporary verification case."""

    case_path: Path
    expected_path: Path


def materialize_microcase(case_id: str, directory: Path) -> MaterializedMicrocase:
    """Write one canonical case and its separate analytic oracle."""

    definition = build_microcase(case_id)
    directory.mkdir(parents=True, exist_ok=False)
    data_path = directory / "case.h5"
    info = write(data_path, definition.data)
    spec = dict(definition.spec)
    spec["case"] = dict(spec["case"], expected_content_hash=info.content_hash)
    case_path = directory / "case.yaml"
    case_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    expected_path = directory / "expected.json"
    expected_path.write_text(
        json.dumps(definition.expected, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return MaterializedMicrocase(case_path, expected_path)


def build_microcase(case_id: str) -> MicrocaseDefinition:
    """Build a named microcase without consulting production numerics."""

    builders = {
        "C01": _c01,
        "C02": _c02,
        "C03": _c03,
        "C04": _c04,
        "C05": _c05,
        "C06": _c06,
        "C07": _c07,
        "C08": _c08,
        "C09": _c09,
        "C10": _c10,
    }
    try:
        return builders[case_id]()
    except KeyError as error:
        raise ValueError(f"unknown microcase: {case_id}") from error


def _c01() -> MicrocaseDefinition:
    times = [0.0, 0.05, 0.15, 0.35, 0.8]
    position = np.asarray([0.125, -0.275])
    velocity = np.asarray([0.4, -0.2])
    expected_positions = [(position + velocity * time).tolist() for time in times]
    source = _table_source([101], [position], [velocity], [0.0], [_COMMON_MASS_KG], [0.0])
    expected = _trajectory_expected(
        "C01",
        "ballistic_closed_form_v1",
        times,
        [101],
        [[item] for item in expected_positions],
        [[velocity.tolist()] for _ in times],
        [[0.0] for _ in times],
    )
    expected["activation_package"] = "P04"
    expected["method_acceptance"] = {
        "rk4_fixed": {
            "position_atol_m": 1.0e-13,
            "velocity_atol_m_s": 1.0e-13,
        }
    }
    return MicrocaseDefinition(
        _bundle("C01", _empty_square_geometry(), sources=(source,)),
        _spec("C01", 0.8, 0.2, _table_source_spec(), output_times=times),
        expected,
    )


def _c02() -> MicrocaseDefinition:
    tau = 0.5
    target = np.asarray([0.1, -0.05])
    position = np.asarray([-0.4, 0.2])
    velocity = np.asarray([0.9, 0.35])
    times = [0.0, 0.25, 0.5, 1.0]
    positions, velocities = _linear_relaxation(times, position, velocity, target, tau)
    fields = _epstein_fields(target, tau)
    source = _table_source([201], [position], [velocity], [0.0], [_COMMON_MASS_KG], [0.0])
    expected = _trajectory_expected(
        "C02",
        "linear_relaxation_closed_form_v1",
        times,
        [201],
        [[item] for item in positions],
        [[item] for item in velocities],
        [[0.0] for _ in times],
    )
    expected.update(
        {
            "activation_package": "P06",
            "derived": {"rate_s_inverse": 2.0, "tau_s": tau},
            "method_acceptance": {
                "rk4_fixed": {
                    "dt_s": [0.2, 0.1, 0.05, 0.025],
                    "comparison_time_s": 1.0,
                    "error_norm": "component_linf",
                    "minimum_observed_order": 3.5,
                    "minimum_order_on_dt_pairs": [[0.1, 0.05], [0.05, 0.025]],
                    "errors_strictly_decrease": True,
                    "finest_position_error_m": 1.0e-8,
                    "finest_velocity_error_m_s": 2.0e-8,
                }
            },
        }
    )
    physics = {"drag": _epstein_physics(), "charge": {"model": "fixed"}}
    return MicrocaseDefinition(
        _bundle("C02", _empty_square_geometry(), fields=fields, sources=(source,)),
        _spec("C02", 1.0, 0.2, _table_source_spec(), physics=physics, output_times=times),
        expected,
    )


def _c03() -> MicrocaseDefinition:
    tau = 0.25
    target = np.asarray([-0.2, 0.3])
    acceleration = np.asarray([0.8, -0.4])
    position = np.asarray([-0.25, -0.1])
    velocity = np.asarray([0.6, -0.5])
    times = [0.0, 0.125, 0.375, 0.75]
    positions, velocities = _relaxation_with_acceleration(
        times, position, velocity, target, acceleration, tau
    )
    fields = _epstein_fields(target, tau)
    source = _table_source([301], [position], [velocity], [0.0], [_COMMON_MASS_KG], [0.0])
    expected = _trajectory_expected(
        "C03",
        "linear_relaxation_constant_acceleration_v1",
        times,
        [301],
        [[item] for item in positions],
        [[item] for item in velocities],
        [[0.0] for _ in times],
    )
    expected.update(
        {
            "activation_package": "P06_rk4_and_P11_exponential_midpoint",
            "derived": {
                "rate_s_inverse": 4.0,
                "tau_s": tau,
                "acceleration_m_s2": acceleration.tolist(),
            },
            "method_acceptance": {
                "rk4_fixed": {
                    "dt_s": [0.125, 0.0625, 0.03125, 0.015625],
                    "comparison_time_s": 0.75,
                    "error_norm": "component_linf",
                    "minimum_observed_order": 3.5,
                    "minimum_order_on_dt_pairs": [
                        [0.0625, 0.03125],
                        [0.03125, 0.015625],
                    ],
                    "errors_strictly_decrease": True,
                    "finest_position_error_m": 5.0e-9,
                    "finest_velocity_error_m_s": 2.0e-8,
                },
                "exponential_midpoint": {
                    "dt_s": [0.75, 0.375, 0.125],
                    "comparison_time_s": 0.75,
                    "error_norm": "component_linf",
                    "position_atol_m": 1.0e-13,
                    "velocity_atol_m_s": 1.0e-13,
                },
            },
        }
    )
    physics = {
        "drag": _epstein_physics(),
        "charge": {"model": "fixed"},
        "gravity_buoyancy": {
            "model": "standard",
            "revision": "gravity_buoyancy_standard_v1",
            "gas_density_field": "gas_density",
            "gravity_m_s2": acceleration.tolist(),
        },
    }
    return MicrocaseDefinition(
        _bundle("C03", _empty_square_geometry(), fields=fields, sources=(source,)),
        _spec("C03", 0.75, 0.125, _table_source_spec(), physics=physics, output_times=times),
        expected,
    )


def _c04() -> MicrocaseDefinition:
    masses = [1.602176634e-18, 3.204353268e-18]
    particle_ids = [401, 402]
    initial_position = np.asarray([0.3, -0.2])
    initial_velocity = np.asarray([-0.1, 0.4])
    electric_field = np.asarray([4.0, -2.0])
    charge_number = -5.0
    force = charge_number * _ELEMENTARY_CHARGE_C * electric_field
    accelerations = [force / mass for mass in masses]
    times = [0.0, 0.25, 0.5]
    positions = [
        [
            (initial_position + initial_velocity * time + 0.5 * acceleration * time * time).tolist()
            for acceleration in accelerations
        ]
        for time in times
    ]
    velocities = [
        [(initial_velocity + acceleration * time).tolist() for acceleration in accelerations]
        for time in times
    ]
    source = _table_source(
        particle_ids,
        [initial_position, initial_position],
        [initial_velocity, initial_velocity],
        [charge_number, charge_number],
        masses,
        [7.0e-18, 7.0e-18],
    )
    fields = _uniform_fields(
        ("electric_field", electric_field, ("x", "y"), "V/m"),
    )
    expected = _trajectory_expected(
        "C04",
        "uniform_electric_fixed_charge_v1",
        times,
        particle_ids,
        positions,
        velocities,
        [[charge_number, charge_number] for _ in times],
    )
    expected.update(
        {
            "activation_package": "P06",
            "derived": {
                "force_N": [force.tolist(), force.tolist()],
                "acceleration_m_s2": [item.tolist() for item in accelerations],
            },
            "method_acceptance": {
                "rk4_fixed": {
                    "position_atol_m": 2.0e-13,
                    "velocity_atol_m_s": 2.0e-13,
                }
            },
        }
    )
    physics = {
        "charge": {"model": "fixed"},
        "electric": {
            "model": "coulomb",
            "revision": "electric_coulomb_v1",
            "electric_field": "electric_field",
        },
    }
    return MicrocaseDefinition(
        _bundle("C04", _empty_square_geometry(), fields=fields, sources=(source,)),
        _spec("C04", 0.5, 0.125, _table_source_spec(), physics=physics, output_times=times),
        expected,
    )


def _c05() -> MicrocaseDefinition:
    particle_ids = [501, 502, 503]
    masses = np.asarray([4.0e-15, 4.0e-15, 8.0e-15])
    volumes = np.asarray([0.0, 1.0e-15, 1.0e-15])
    density = 1.2
    gravity = np.asarray([0.0, -10.0])
    accelerations = [
        ((1.0 - density * volume / mass) * gravity)
        for mass, volume in zip(masses, volumes, strict=True)
    ]
    initial_positions = [np.asarray([-0.3, 0.0]), np.asarray([0.0, 0.0]), np.asarray([0.3, 0.0])]
    initial_velocity = np.asarray([0.2, 0.5])
    times = [0.0, 0.1, 0.2]
    positions = [
        [
            (position + initial_velocity * time + 0.5 * acceleration * time * time).tolist()
            for position, acceleration in zip(initial_positions, accelerations, strict=True)
        ]
        for time in times
    ]
    velocities = [
        [(initial_velocity + acceleration * time).tolist() for acceleration in accelerations]
        for time in times
    ]
    source = _table_source(
        particle_ids,
        initial_positions,
        [initial_velocity] * 3,
        [0.0, 0.0, 0.0],
        masses.tolist(),
        volumes.tolist(),
    )
    fields = _uniform_fields(
        ("gas_density", np.asarray([density]), ("value",), "kg/m^3"),
    )
    expected = _trajectory_expected(
        "C05",
        "gravity_buoyancy_constant_acceleration_v1",
        times,
        particle_ids,
        positions,
        velocities,
        [[0.0, 0.0, 0.0] for _ in times],
    )
    expected.update(
        {
            "activation_package": "P06",
            "derived": {"acceleration_m_s2": [item.tolist() for item in accelerations]},
            "method_acceptance": {
                "rk4_fixed": {
                    "position_atol_m": 2.0e-13,
                    "velocity_atol_m_s": 2.0e-13,
                }
            },
        }
    )
    physics = {
        "charge": {"model": "fixed"},
        "gravity_buoyancy": {
            "model": "standard",
            "revision": "gravity_buoyancy_standard_v1",
            "gas_density_field": "gas_density",
            "gravity_m_s2": gravity.tolist(),
        },
    }
    return MicrocaseDefinition(
        _bundle("C05", _empty_square_geometry(), fields=fields, sources=(source,)),
        _spec("C05", 0.2, 0.05, _table_source_spec(), physics=physics, output_times=times),
        expected,
    )


def _c06() -> MicrocaseDefinition:
    p1_nodes = _f8([[0.0, 0.0], [0.02, 0.0], [0.0, 0.01], [0.02, 0.01]])
    p1_connectivity = _i8([[0, 1, 2], [1, 3, 2]])
    q1_nodes = _f8([[0.0, 0.0], [0.02, 0.0], [0.03, 0.02], [0.0, 0.02]])
    layouts = (
        P1TriLayout("p1", p1_nodes, p1_connectivity, _u1([1, 1])),
        Q1QuadLayout("q1", q1_nodes, _i8([[0, 1, 2, 3]]), _u1([1])),
    )
    fields = (
        FieldData(
            "p1_affine",
            "p1",
            "node",
            ("x", "y"),
            "cartesian_xy",
            _f8([[1.0, -2.0], [5.0, -1.0], [-2.0, 2.0], [2.0, 3.0]]),
            "1",
        ),
        FieldData(
            "q1_bilinear",
            "q1",
            "node",
            ("x", "y"),
            "cartesian_xy",
            _f8([[0.5, -8.0], [3.5, -2.0], [2.5, 0.0], [-2.5, 2.0]]),
            "1",
        ),
    )
    source = _table_source([601], [[0.005, 0.0025]], [[0.0, 0.0]], [0.0], [_COMMON_MASS_KG], [0.0])
    expected = {
        "format_version": 2,
        "case_id": "C06",
        "oracle_revision": "p1_affine_and_mapped_q1_v1",
        "activation_package": "P03",
        "field_probes": [
            {
                "field": "p1_affine",
                "position_m": [0.005, 0.0025],
                "cell_id": 0,
                "support_inside": True,
                "value": [1.25, -0.75],
            },
            {
                "field": "p1_affine",
                "position_m": [0.01, 0.005],
                "candidate_cell_id": [0, 1],
                "support_inside": True,
                "value_from_each_cell": [[1.5, 0.5], [1.5, 0.5]],
            },
            {
                "field": "q1_bilinear",
                "reference_position": [0.2, -0.4],
                "position_m": [0.0138, 0.006],
                "shape_weights": [0.28, 0.42, 0.18, 0.12],
                "cell_id": 0,
                "support_inside": True,
                "value": [1.76, -2.84],
            },
            {
                "field": "q1_bilinear",
                "reference_position": [1.24, 0.0],
                "position_m": [0.028, 0.01],
                "support_inside": False,
            },
        ],
        "acceptance": {
            "field_rtol": 2.0e-12,
            "field_atol": 2.0e-12,
            "coordinate_atol_m": 2.0e-12,
        },
    }
    return MicrocaseDefinition(
        _bundle(
            "C06",
            GeometryData(
                nodes_m=p1_nodes,
                boundary=_empty_boundary(),
                group_names=(),
                tri3=p1_connectivity,
                tri3_domain_id=_i4([0, 0]),
            ),
            layouts=layouts,
            fields=fields,
            sources=(source,),
        ),
        _spec("C06", 0.1, 0.1, _table_source_spec()),
        expected,
    )


def _c07() -> MicrocaseDefinition:
    geometry = _square_boundary_geometry(("wall",), (0, 0, 0, 0))
    source = _table_source([701], [[0.25, 0.25]], [[0.5, 0.25]], [3.0], [_COMMON_MASS_KG], [0.0])
    expected = _event_expected(
        "C07",
        "plane_first_hit_v1",
        "P05",
        events=[
            {
                "particle_id": 701,
                "boundary_event_ordinal": 0,
                "event_type": "boundary",
                "time_s": 1.5,
                "position_m": [1.0, 0.625],
                "primary_facet_id": 1,
                "candidate_facet_id": [1],
                "boundary_id": 10,
                "material_id": 0,
                "normal": [1.0, 0.0],
                "velocity_pre_m_s": [0.5, 0.25],
                "velocity_post_m_s": [0.0, 0.0],
                "charge_number_pre": 3.0,
                "charge_number_post": 3.0,
                "model_weight": 1.0,
                "law_id": "stick",
                "outcome": "stuck",
            }
        ],
    )
    return MicrocaseDefinition(
        _bundle("C07", geometry, sources=(source,)),
        _spec(
            "C07",
            2.0,
            2.0,
            _table_source_spec(),
            boundaries=[_boundary("wall", 10, "stick")],
        ),
        expected,
    )


def _c08() -> MicrocaseDefinition:
    geometry = _square_boundary_geometry(
        ("caps", "mirror", "collector"),
        (0, 1, 0, 2),
    )
    source = RealizedSurfaceSource(
        name="surface_particles",
        particle_id=_i8([801]),
        release_time_s=_f8([0.0]),
        facet_id=_i8([3]),
        facet_parameter=_f8([0.5]),
        velocity_m_s=_f8([[1.0, 0.0]]),
        charge_number=_f8([0.0]),
        mass_kg=_f8([_COMMON_MASS_KG]),
        drag_diameter_m=_f8([_COMMON_DRAG_DIAMETER_M]),
        contact_radius_m=_f8([0.0]),
        electrostatic_radius_m=_f8([_COMMON_ELECTROSTATIC_RADIUS_M]),
        displaced_volume_m3=_f8([0.0]),
        model_weight=_f8([1.0]),
        material_id=_i4([0]),
    )
    source_spec = {
        "name": "surface_release",
        "type": "surface",
        "table": "surface_particles",
    }
    expected = _event_expected(
        "C08",
        "surface_departure_and_reimpact_v1",
        "P07",
        events=[
            {
                "particle_id": 801,
                "boundary_event_ordinal": 0,
                "event_type": "boundary",
                "time_s": 1.0,
                "position_m": [1.0, 0.5],
                "primary_facet_id": 1,
                "candidate_facet_id": [1],
                "boundary_id": 20,
                "material_id": 0,
                "normal": [1.0, 0.0],
                "velocity_pre_m_s": [1.0, 0.0],
                "velocity_post_m_s": [-1.0, 0.0],
                "charge_number_pre": 0.0,
                "charge_number_post": 0.0,
                "model_weight": 1.0,
                "law_id": "specular",
                "outcome": "reflected",
            },
            {
                "particle_id": 801,
                "boundary_event_ordinal": 1,
                "event_type": "boundary",
                "time_s": 2.0,
                "position_m": [0.0, 0.5],
                "primary_facet_id": 3,
                "candidate_facet_id": [3],
                "boundary_id": 30,
                "material_id": 0,
                "normal": [-1.0, 0.0],
                "velocity_pre_m_s": [-1.0, 0.0],
                "velocity_post_m_s": [0.0, 0.0],
                "charge_number_pre": 0.0,
                "charge_number_post": 0.0,
                "model_weight": 1.0,
                "law_id": "stick",
                "outcome": "stuck",
            },
        ],
    )
    expected["zero_time_departure_event_count"] = 0
    return MicrocaseDefinition(
        _bundle("C08", geometry, sources=(source,)),
        _spec(
            "C08",
            2.25,
            2.25,
            source_spec,
            boundaries=[
                _boundary("collector", 10, "stick"),
                _boundary("mirror", 20, "specular"),
                _boundary("caps", 30, "escape"),
            ],
        ),
        expected,
    )


def _c09() -> MicrocaseDefinition:
    gap = 2.0**-10
    height = 128.0 * gap
    geometry = _rectangle_boundary_geometry(
        gap,
        height,
        ("caps", "mirror"),
        (0, 1, 0, 1),
    )
    source = _table_source(
        [901],
        [[gap / 2.0, height / 2.0]],
        [[1.0, 0.0]],
        [0.0],
        [_COMMON_MASS_KG],
        [0.0],
    )
    hit_times = [gap / 2.0 + index * gap for index in range(5)]
    events = [
        {
            "particle_id": 901,
            "boundary_event_ordinal": index,
            "event_type": "boundary",
            "time_s": time,
            "position_m": [gap if index % 2 == 0 else 0.0, height / 2.0],
            "primary_facet_id": 1 if index % 2 == 0 else 3,
            "candidate_facet_id": [1 if index % 2 == 0 else 3],
            "boundary_id": 20,
            "material_id": 0,
            "normal": [1.0, 0.0] if index % 2 == 0 else [-1.0, 0.0],
            "velocity_pre_m_s": [1.0, 0.0] if index % 2 == 0 else [-1.0, 0.0],
            "velocity_post_m_s": [-1.0, 0.0] if index % 2 == 0 else [1.0, 0.0],
            "charge_number_pre": 0.0,
            "charge_number_post": 0.0,
            "model_weight": 1.0,
            "law_id": "specular",
            "outcome": "reflected",
        }
        for index, time in enumerate(hit_times)
    ]
    expected = _event_expected("C09", "thin_gap_residual_time_v1", "P07", events)
    expected["final"] = {
        "time_s": 21.0 * gap / 4.0,
        "position_m": [gap / 4.0, height / 2.0],
        "velocity_m_s": [-1.0, 0.0],
    }
    expected["final_acceptance"] = {
        "time_atol_s": 1.0e-15,
        "position_atol_m": 5.0e-14,
        "velocity_atol_m_s": 1.0e-13,
    }
    expected["success_probes"] = [
        {
            "name": "residual_work_subdivision_success",
            "spec_override": {
                "solver.event.max_interactions_per_step": 1,
                "solver.event.max_refinements": 2,
            },
            "expected_run_status": "complete",
            "expected_boundary_events": events,
            "expected_final": expected["final"],
        }
    ]
    expected["failure_probes"] = [
        {
            "name": "residual_work_budget_exhaustion",
            "spec_override": {
                "solver.event.max_interactions_per_step": 1,
                "solver.event.max_refinements": 1,
            },
            "expected_run_status": "failed",
            "failure_reason": "numerical_event_budget",
            "expected_processed_boundary_events": events[:2],
            "expected_failure_time_s": hit_times[1],
            "next_unprocessed_boundary_event_ordinal": 2,
            "next_unprocessed_boundary_event_time_s": hit_times[2],
        }
    ]
    return MicrocaseDefinition(
        _bundle("C09", geometry, sources=(source,)),
        _spec(
            "C09",
            21.0 * gap / 4.0,
            21.0 * gap / 4.0,
            _table_source_spec(),
            boundaries=[
                _boundary("mirror", 10, "specular"),
                _boundary("caps", 20, "escape"),
            ],
        ),
        expected,
    )


def _c10() -> MicrocaseDefinition:
    nodes = _f8([[0.0, 0.0], [3.0, 0.0], [1.0, 1.0], [4.0, 0.0], [7.0, 0.0], [5.0, 1.0]])
    boundary = BoundaryData(
        line2=_i8([[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3]]),
        boundary_id=_i4([10, 20, 20, 10, 30, 40]),
        group_id=_i4([0, 1, 1, 0, 2, 3]),
        material_id=_i4([0, 0, 0, 0, 0, 0]),
        owner_cell_type=_u1([1, 1, 1, 1, 1, 1]),
        owner_cell_local_index=_i8([0, 0, 0, 1, 1, 1]),
        orientation=_i1([1, 1, 1, 1, 1, 1]),
    )
    geometry = GeometryData(
        nodes_m=nodes,
        boundary=boundary,
        group_names=("base", "corner_mirror", "priority_collector", "priority_mirror"),
        tri3=_i8([[0, 1, 2], [3, 4, 5]]),
        tri3_domain_id=_i4([0, 0]),
    )
    source = _table_source(
        [1001, 1002],
        [[1.0, 0.25], [5.0, 0.25]],
        [[0.0, 1.0], [0.0, 1.0]],
        [0.0, 0.0],
        [_COMMON_MASS_KG, _COMMON_MASS_KG],
        [0.0, 0.0],
    )
    inverse_sqrt_ten = 1.0 / math.sqrt(10.0)
    combined_normal = [-0.1601822430069672, 0.9870874576374967]
    priority_normal = [1.0 / math.sqrt(5.0), 2.0 / math.sqrt(5.0)]
    expected = _event_expected(
        "C10",
        "nonorthogonal_corner_priority_and_combined_normal_v1",
        "P07",
        events=[
            {
                "particle_id": 1001,
                "boundary_event_ordinal": 0,
                "event_type": "boundary",
                "time_s": 0.75,
                "position_m": [1.0, 1.0],
                "primary_facet_id": 1,
                "candidate_facet_id": [1, 2],
                "boundary_id": 20,
                "material_id": 0,
                "normal": combined_normal,
                "velocity_pre_m_s": [0.0, 1.0],
                "velocity_post_m_s": [inverse_sqrt_ten, -3.0 * inverse_sqrt_ten],
                "charge_number_pre": 0.0,
                "charge_number_post": 0.0,
                "model_weight": 1.0,
                "law_id": "specular",
                "outcome": "reflected",
            },
            {
                "particle_id": 1002,
                "boundary_event_ordinal": 0,
                "event_type": "boundary",
                "time_s": 0.75,
                "position_m": [5.0, 1.0],
                "primary_facet_id": 4,
                "candidate_facet_id": [4, 5],
                "boundary_id": 30,
                "material_id": 0,
                "normal": priority_normal,
                "velocity_pre_m_s": [0.0, 1.0],
                "velocity_post_m_s": [0.0, 0.0],
                "charge_number_pre": 0.0,
                "charge_number_post": 0.0,
                "model_weight": 1.0,
                "law_id": "stick",
                "outcome": "stuck",
            },
        ],
    )
    expected["final"] = {
        "time_s": 1.0,
        "particle_id": [1001, 1002],
        "position_m": [
            [1.0 + 0.25 * inverse_sqrt_ten, 1.0 - 0.75 * inverse_sqrt_ten],
            [5.0, 1.0],
        ],
        "velocity_m_s": [[inverse_sqrt_ten, -3.0 * inverse_sqrt_ten], [0.0, 0.0]],
        "lifecycle": ["active", "stuck"],
    }
    expected["final_acceptance"] = {
        "time_atol_s": 1.0e-13,
        "position_atol_m": 2.0e-13,
        "velocity_atol_m_s": 2.0e-13,
        "lifecycle_exact": True,
    }
    expected["policy_failure_probes"] = [
        {
            "name": "same_priority_incompatible_laws",
            "candidates": [
                {"facet_id": 1, "boundary_id": 20, "priority": 10, "law_id": "specular"},
                {"facet_id": 2, "boundary_id": 21, "priority": 10, "law_id": "stick"},
            ],
            "failure_reason": "ambiguous_boundary_law",
        },
        {
            "name": "selected_response_exits_ignored_candidate",
            "velocity_pre_m_s": [1.0, 1.0],
            "candidates": [
                {
                    "facet_id": 10,
                    "boundary_id": 50,
                    "priority": 5,
                    "law_id": "specular",
                    "normal": [1.0, 0.0],
                },
                {
                    "facet_id": 11,
                    "boundary_id": 60,
                    "priority": 20,
                    "law_id": "specular",
                    "normal": [0.0, 1.0],
                },
            ],
            "selected_primary_facet_id": 10,
            "selected_velocity_post_m_s": [-1.0, 1.0],
            "failure_reason": "indeterminate_boundary_policy",
        },
    ]
    return MicrocaseDefinition(
        _bundle("C10", geometry, sources=(source,)),
        _spec(
            "C10",
            1.0,
            1.0,
            _table_source_spec(),
            boundaries=[
                _boundary("base", 30, "escape"),
                _boundary("corner_mirror", 10, "specular"),
                _boundary("priority_collector", 5, "stick"),
                _boundary("priority_mirror", 20, "specular"),
            ],
        ),
        expected,
    )


def _spec(
    case_id: str,
    end_s: float,
    dt_s: float,
    source: dict[str, Any],
    *,
    physics: dict[str, Any] | None = None,
    boundaries: list[dict[str, Any]] | None = None,
    output_times: list[float] | None = None,
) -> dict[str, Any]:
    trajectories = None
    if output_times is not None:
        trajectories = {
            "selection": "all",
            "schedule": {"explicit_times_s": output_times},
        }
    return {
        "format_version": 3,
        "case": {
            "name": case_id,
            "data_path": "case.h5",
            "expected_content_hash": f"sha256:{'0' * 64}",
        },
        "motion": {"mode": "cartesian_xy"},
        "time": {"start_s": 0.0, "end_s": end_s, "dt_s": dt_s},
        "solver": {
            "integrator": "rk4_fixed",
            "backend": "cpu",
            "seed": 0,
            "event": {
                "geometry_rtol": 1.0e-12,
                "roundoff_ulps": 64,
                "max_refinements": 48,
                "max_interactions_per_step": 8,
                "corner_policy": "priority_then_combined_normal_v1",
            },
        },
        "resources": {"memory_limit_mb": 128},
        "physics": physics or {"charge": {"model": "fixed"}},
        "sources": [source],
        "boundaries": boundaries or [],
        "output": {"trajectories": trajectories},
    }


def _bundle(
    case_id: str,
    geometry: GeometryData,
    *,
    layouts: tuple[RegularLayout | P1TriLayout | Q1QuadLayout, ...] = (),
    fields: tuple[FieldData, ...] = (),
    sources: tuple[RealizedSource, ...] = (),
) -> DataBundle:
    provenance = json.dumps(
        {
            "producer": "analytic-microcase-pack",
            "producer_version": "1",
            "source_sha256": "sha256:" + hashlib.sha256(f"{case_id}:input-v1".encode()).hexdigest(),
            "field_semantics_revision": "microcase-primitive-v1",
            "producer_metadata": {"case_id": case_id},
        }
    )
    if not layouts and fields:
        layouts = (_uniform_layout(),)
    return DataBundle("cartesian_xy", provenance, geometry, layouts, fields, sources)


def _trajectory_expected(
    case_id: str,
    revision: str,
    times: list[float],
    particle_ids: list[int],
    positions: list[list[list[float]]],
    velocities: list[list[list[float]]],
    charges: list[list[float]],
) -> dict[str, Any]:
    return {
        "format_version": 2,
        "case_id": case_id,
        "oracle_revision": revision,
        "time_s": times,
        "particle_id": particle_ids,
        "position_m": positions,
        "velocity_m_s": velocities,
        "charge_number": charges,
        "analytic_reference_precision": {
            "position_atol_m": 1.0e-13,
            "position_rtol": 1.0e-13,
            "velocity_atol_m_s": 1.0e-13,
            "velocity_rtol": 1.0e-13,
            "charge_number_exact": True,
        },
    }


def _event_expected(
    case_id: str,
    revision: str,
    activation: str,
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "format_version": 2,
        "case_id": case_id,
        "oracle_revision": revision,
        "activation_package": activation,
        "boundary_events": events,
        "acceptance": {
            "requested_geometry_rtol": 1.0e-12,
            "accepted_budget_multiplier": 4.0,
            "candidate_set_exact": True,
            "boundary_event_count_and_order_exact": True,
            "normal_atol": 2.0e-13,
            "velocity_atol_m_s": 2.0e-13,
            "charge_number_exact": True,
            "model_weight_exact": True,
            "law_id_and_outcome_exact": True,
        },
    }


def _linear_relaxation(
    times: list[float],
    position: np.ndarray,
    velocity: np.ndarray,
    target: np.ndarray,
    tau: float,
) -> tuple[list[list[float]], list[list[float]]]:
    positions: list[list[float]] = []
    velocities: list[list[float]] = []
    for time in times:
        decay = math.exp(-time / tau)
        positions.append(
            (position + target * time + tau * (1.0 - decay) * (velocity - target)).tolist()
        )
        velocities.append((target + decay * (velocity - target)).tolist())
    return positions, velocities


def _relaxation_with_acceleration(
    times: list[float],
    position: np.ndarray,
    velocity: np.ndarray,
    target: np.ndarray,
    acceleration: np.ndarray,
    tau: float,
) -> tuple[list[list[float]], list[list[float]]]:
    positions: list[list[float]] = []
    velocities: list[list[float]] = []
    for time in times:
        decay = math.exp(-time / tau)
        response = 1.0 - decay
        positions.append(
            (
                position
                + target * time
                + tau * response * (velocity - target)
                + tau * (time - tau * response) * acceleration
            ).tolist()
        )
        velocities.append(
            (target + decay * (velocity - target) + tau * response * acceleration).tolist()
        )
    return positions, velocities


def _epstein_fields(target_velocity_m_s: np.ndarray, tau_s: float) -> tuple[FieldData, ...]:
    mean_thermal_speed = math.sqrt(
        8.0 * _BOLTZMANN_J_K * _GAS_TEMPERATURE_K / (math.pi * _GAS_MOLECULE_MASS_KG)
    )
    radius = 0.5 * _COMMON_DRAG_DIAMETER_M
    density = (_COMMON_MASS_KG / tau_s) / (
        (4.0 * math.pi / 3.0) * radius * radius * mean_thermal_speed
    )
    return _uniform_fields(
        ("gas_velocity", target_velocity_m_s, ("x", "y"), "m/s"),
        ("gas_density", np.asarray([density]), ("value",), "kg/m^3"),
        (
            "gas_temperature",
            np.asarray([_GAS_TEMPERATURE_K]),
            ("value",),
            "K",
        ),
        ("gas_mean_free_path", np.asarray([1.0e-3]), ("value",), "m"),
    )


def _epstein_physics() -> dict[str, Any]:
    return {
        "model": "epstein_linear",
        "revision": "epstein_linear_v1",
        "gas_velocity_field": "gas_velocity",
        "gas_density_field": "gas_density",
        "gas_temperature_field": "gas_temperature",
        "gas_mean_free_path_field": "gas_mean_free_path",
        "gas_molecular_mass_kg": _GAS_MOLECULE_MASS_KG,
        "delta": 1.0,
        "applicability": "error",
    }


def _table_source(
    particle_ids: list[int],
    positions: list[Any],
    velocities: list[Any],
    charge_numbers: list[float],
    masses_kg: list[float],
    displaced_volumes_m3: list[float],
) -> RealizedTableSource:
    count = len(particle_ids)
    return RealizedTableSource(
        name="particles",
        particle_id=_i8(particle_ids),
        release_time_s=_f8([0.0] * count),
        position_m=_f8(positions),
        velocity_m_s=_f8(velocities),
        charge_number=_f8(charge_numbers),
        mass_kg=_f8(masses_kg),
        drag_diameter_m=_f8([_COMMON_DRAG_DIAMETER_M] * count),
        contact_radius_m=_f8([0.0] * count),
        electrostatic_radius_m=_f8([_COMMON_ELECTROSTATIC_RADIUS_M] * count),
        displaced_volume_m3=_f8(displaced_volumes_m3),
        model_weight=_f8([1.0] * count),
        material_id=_i4([0] * count),
    )


def _table_source_spec() -> dict[str, Any]:
    return {"name": "release", "type": "table", "table": "particles"}


def _boundary(group: str, priority: int, law: str, **parameters: Any) -> dict[str, Any]:
    return {"boundary_group": group, "priority": priority, "law": law, **parameters}


def _uniform_layout() -> RegularLayout:
    return RegularLayout("uniform", _f8([-1.0, 1.0]), _f8([-1.0, 1.0]), _u1([[1]]))


def _uniform_fields(
    *definitions: tuple[str, np.ndarray, tuple[str, ...], str],
) -> tuple[FieldData, ...]:
    fields = []
    for name, value, components, unit in definitions:
        row = np.asarray(value, dtype="<f8").reshape(1, len(components))
        values = np.repeat(row, 4, axis=0).astype("<f8", copy=False)
        basis = "scalar" if len(components) == 1 else "cartesian_xy"
        fields.append(FieldData(name, "uniform", "node", components, basis, values, unit))
    return tuple(fields)


def _empty_square_geometry() -> GeometryData:
    nodes = _f8([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    return GeometryData(
        nodes_m=nodes,
        boundary=_empty_boundary(),
        group_names=(),
        quad4=_i8([[0, 1, 2, 3]]),
        quad4_domain_id=_i4([0]),
    )


def _square_boundary_geometry(
    group_names: tuple[str, ...], edge_group_ids: tuple[int, int, int, int]
) -> GeometryData:
    return _rectangle_boundary_geometry(1.0, 1.0, group_names, edge_group_ids)


def _rectangle_boundary_geometry(
    width: float,
    height: float,
    group_names: tuple[str, ...],
    edge_group_ids: tuple[int, int, int, int],
) -> GeometryData:
    nodes = _f8([[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]])
    boundary = BoundaryData(
        line2=_i8([[0, 1], [1, 2], [2, 3], [3, 0]]),
        boundary_id=_i4([10 * (group_id + 1) for group_id in edge_group_ids]),
        group_id=_i4(edge_group_ids),
        material_id=_i4([0, 0, 0, 0]),
        owner_cell_type=_u1([2, 2, 2, 2]),
        owner_cell_local_index=_i8([0, 0, 0, 0]),
        orientation=_i1([1, 1, 1, 1]),
    )
    return GeometryData(
        nodes_m=nodes,
        boundary=boundary,
        group_names=group_names,
        quad4=_i8([[0, 1, 2, 3]]),
        quad4_domain_id=_i4([0]),
    )


def _empty_boundary() -> BoundaryData:
    return BoundaryData(
        line2=_i8([]).reshape(0, 2),
        boundary_id=_i4([]),
        group_id=_i4([]),
        material_id=_i4([]),
        owner_cell_type=_u1([]),
        owner_cell_local_index=_i8([]),
        orientation=_i1([]),
    )


def _f8(values: Any) -> np.ndarray:
    return np.asarray(values, dtype="<f8")


def _i8(values: Any) -> np.ndarray:
    return np.asarray(values, dtype="<i8")


def _i4(values: Any) -> np.ndarray:
    return np.asarray(values, dtype="<i4")


def _u1(values: Any) -> np.ndarray:
    return np.asarray(values, dtype="<u1")


def _i1(values: Any) -> np.ndarray:
    return np.asarray(values, dtype="<i1")
