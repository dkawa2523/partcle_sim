"""An unsupported provisional stage cannot corrupt an earlier physical hit."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from chamber_particles import load_case, open_result, simulate
from chamber_particles.case_format import (
    BoundaryData,
    DataBundle,
    FieldData,
    GeometryData,
    P1TriLayout,
    RealizedTableSource,
    write,
)

_GAMMA_S_INV = 0.001
_INITIAL_SPEED_M_S = 30.0
_DURATION_S = 1.4 / _INITIAL_SPEED_M_S


def test_exponential_midpoint_preserves_first_hit_before_unsupported_stage(
    tmp_path: Path,
) -> None:
    """Two constant patches isolate the enclosure premise from physical models.

    The particle must stick on the left patch at x=1. Its coarse frozen
    midpoint is in the unsupported gap and projects to the right patch,
    whose opposing electric acceleration exceeds the left patch range.
    Before the first hit the exact solution has only constant weak drag.
    """
    case_path = _write_disconnected_drag_case(tmp_path)
    summary = simulate(load_case(case_path), tmp_path / "result", workers=1)
    assert summary.failure_event_count == 0
    assert summary.boundary_event_count == 1
    result = open_result(summary.output_path)
    event = result.read_boundary_events()
    hit_time_s = -math.log1p(-_GAMMA_S_INV * 0.1 / _INITIAL_SPEED_M_S) / _GAMMA_S_INV
    assert event.time_s[0] == pytest.approx(hit_time_s, abs=2.0e-12)
    np.testing.assert_allclose(event.position_m, [[1.0, 0.5]], rtol=0.0, atol=2.0e-12)
    np.testing.assert_allclose(
        event.velocity_pre_m_s,
        [[_INITIAL_SPEED_M_S - _GAMMA_S_INV * 0.1, 0.0]],
        rtol=0.0,
        atol=2.0e-10,
    )
    np.testing.assert_array_equal(event.velocity_post_m_s, [[0.0, 0.0]])
    np.testing.assert_array_equal(event.charge_number_pre, [1.0])
    np.testing.assert_array_equal(event.charge_number_post, [1.0])
    np.testing.assert_array_equal(event.boundary_id, [1])
    final = result.read_final()
    np.testing.assert_allclose(final.position_m, [[1.0, 0.5]], rtol=0.0, atol=2.0e-12)
    np.testing.assert_array_equal(final.velocity_m_s, [[0.0, 0.0]])
    np.testing.assert_array_equal(final.charge_number, [1.0])
    for frame in result.iter_frames():
        if frame.time_s < hit_time_s:
            expected_x = (
                0.9 - _INITIAL_SPEED_M_S * math.expm1(-_GAMMA_S_INV * frame.time_s) / _GAMMA_S_INV
            )
            expected_v = _INITIAL_SPEED_M_S * math.exp(-_GAMMA_S_INV * frame.time_s)
            np.testing.assert_allclose(
                frame.position_m, [[expected_x, 0.5]], rtol=0.0, atol=2.0e-12
            )
            np.testing.assert_allclose(
                frame.velocity_m_s, [[expected_v, 0.0]], rtol=0.0, atol=2.0e-10
            )


def _write_disconnected_drag_case(directory: Path) -> Path:
    boltzmann_j_k, elementary_charge_c = 1.380649e-23, 1.602176634e-19
    temperature_k, mean_speed_m_s = 300.0, 400.0
    gas_mass_kg = 8.0 * boltzmann_j_k * temperature_k / (math.pi * mean_speed_m_s**2)
    mass_kg, diameter_m = 1.0e-21, 1.0e-8
    gas_density_kg_m3 = (
        _GAMMA_S_INV * mass_kg / ((4.0 * math.pi / 3.0) * (diameter_m / 2.0) ** 2 * mean_speed_m_s)
    )
    nodes = np.asarray(
        [[0, 0], [1, 0], [1, 1], [0, 1], [2, 0], [3, 0], [3, 1], [2, 1]], dtype="<f8"
    )
    triangles = np.asarray([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]], dtype="<i8")
    boundary = BoundaryData(
        line2=np.asarray(
            [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4]],
            dtype="<i8",
        ),
        boundary_id=np.arange(8, dtype="<i4"),
        group_id=np.zeros(8, dtype="<i4"),
        material_id=np.zeros(8, dtype="<i4"),
        owner_cell_type=np.ones(8, dtype="<u1"),
        owner_cell_local_index=np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype="<i8"),
        orientation=np.ones(8, dtype="<i1"),
    )
    geometry = GeometryData(
        nodes,
        boundary,
        ("terminal",),
        tri3=triangles,
        tri3_domain_id=np.asarray([0, 0, 1, 1], dtype="<i4"),
    )
    layout = P1TriLayout("gas", nodes, triangles, np.ones(4, dtype="<u1"))
    electric = np.zeros((8, 2), dtype="<f8")
    electric[4:, 0] = -300.0 * mass_kg / elementary_charge_c
    fields = [
        FieldData("electric", "gas", "node", ("x", "y"), "cartesian_xy", electric, "V/m"),
        FieldData("velocity", "gas", "node", ("x", "y"), "cartesian_xy", np.zeros((8, 2)), "m/s"),
    ]
    for name, value, unit in (
        ("density", gas_density_kg_m3, "kg/m^3"),
        ("temperature", temperature_k, "K"),
        ("mean_free_path", 1.0, "m"),
    ):
        fields.append(
            FieldData(name, "gas", "node", ("value",), "scalar", np.full((8, 1), value), unit)
        )
    source = RealizedTableSource(
        "particle",
        np.asarray([1], dtype="<i8"),
        np.asarray([0.0]),
        np.asarray([[0.9, 0.5]]),
        np.asarray([[_INITIAL_SPEED_M_S, 0.0]]),
        np.asarray([1.0]),
        np.asarray([mass_kg]),
        np.asarray([diameter_m]),
        np.asarray([0.0]),
        np.asarray([diameter_m / 2.0]),
        np.asarray([0.0]),
        np.asarray([1.0]),
        np.asarray([0], dtype="<i4"),
    )
    provenance = {
        "producer": "analytic_two_disconnected_constant_patches",
        "producer_version": "1",
        "source_sha256": "sha256:" + hashlib.sha256(b"analytic two patches").hexdigest(),
        "field_semantics_revision": "analytic_constant_patches_v1",
        "producer_metadata": {"purpose": "independent analytic first-hit verification"},
    }
    info = write(
        directory / "case.h5",
        DataBundle(
            "cartesian_xy", json.dumps(provenance), geometry, (layout,), tuple(fields), (source,)
        ),
    )
    document = {
        "format_version": 3,
        "case": {
            "name": "analytic_projected_midpoint_first_hit",
            "data_path": "case.h5",
            "expected_content_hash": info.content_hash,
        },
        "motion": {"mode": "cartesian_xy"},
        "time": {"start_s": 0.0, "end_s": _DURATION_S, "dt_s": _DURATION_S},
        "solver": {
            "integrator": "exponential_midpoint",
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
        "physics": {
            "charge": {"model": "fixed"},
            "drag": {
                "model": "epstein_linear",
                "revision": "epstein_linear_v1",
                "gas_velocity_field": "velocity",
                "gas_density_field": "density",
                "gas_temperature_field": "temperature",
                "gas_mean_free_path_field": "mean_free_path",
                "gas_molecular_mass_kg": gas_mass_kg,
                "delta": 1.0,
                "applicability": "error",
            },
            "electric": {
                "model": "coulomb",
                "revision": "electric_coulomb_v1",
                "electric_field": "electric",
            },
        },
        "sources": [{"name": "source", "type": "table", "table": "particle"}],
        "boundaries": [
            {
                "boundary_group": "terminal",
                "priority": 0,
                "law": "stick",
                "contact_geometry": "particle_center",
            }
        ],
        "output": {
            "trajectories": {
                "selection": "all",
                "schedule": {"explicit_times_s": [0.0, 0.001, 0.002, 0.003, _DURATION_S]},
            },
            "probes": None,
        },
    }
    case_path = directory / "case.yaml"
    case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return case_path
