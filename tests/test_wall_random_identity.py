from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from particle_tracer_unified import load_case, simulate
from particle_tracer_unified.core.datamodel import WallPartModel
from particle_tracer_unified.solvers import _stochastic_randomness
from particle_tracer_unified.solvers._stochastic_randomness import WallRandomContext
from particle_tracer_unified.solvers.wall_response import _apply_keyed_wall_response


def _model(law_name: str, *, stick_probability: float = 0.0) -> WallPartModel:
    return WallPartModel(
        part_id=1,
        part_name="wall",
        material_id=1,
        material_name="test",
        law_name=law_name,
        stick_probability=stick_probability,
        restitution=1.0,
        diffuse_fraction=0.5,
        critical_sticking_velocity_mps=0.0,
        metadata={"role": "wall"},
    )


def _context(**overrides: int) -> WallRandomContext:
    values = {
        "seed": 31415,
        "particle_id": 7,
        "macro_step_index": 3,
        "cohort_index": 2,
        "wall_event_ordinal": 1,
    }
    values.update(overrides)
    return WallRandomContext(**values)


@pytest.mark.parametrize(
    ("velocity", "normal"),
    [
        (np.asarray([2.0, 0.5]), np.asarray([1.0, 0.0])),
        (np.asarray([2.0, 0.5, -0.25]), np.asarray([1.0, 0.0, 0.0])),
    ],
)
def test_keyed_diffuse_response_is_reproducible_in_2d_and_3d(
    velocity: np.ndarray,
    normal: np.ndarray,
) -> None:
    first = _apply_keyed_wall_response(
        _context(),
        velocity,
        normal,
        _model("cosine_diffuse"),
    )
    second = _apply_keyed_wall_response(
        _context(),
        velocity,
        normal,
        _model("cosine_diffuse"),
    )

    assert first[0] == second[0] == "reflected_diffuse"
    np.testing.assert_array_equal(first[1], second[1])


@pytest.mark.parametrize(
    "override",
    [
        {"seed": 31416},
        {"particle_id": 8},
        {"macro_step_index": 4},
        {"cohort_index": 3},
        {"wall_event_ordinal": 2},
    ],
)
def test_each_wall_event_key_component_selects_a_distinct_response(
    override: dict[str, int],
) -> None:
    velocity = np.asarray([2.0, 0.5, -0.25])
    normal = np.asarray([1.0, 0.0, 0.0])
    baseline = _apply_keyed_wall_response(
        _context(),
        velocity,
        normal,
        _model("cosine_diffuse"),
    )[1]
    changed = _apply_keyed_wall_response(
        _context(**override),
        velocity,
        normal,
        _model("cosine_diffuse"),
    )[1]

    assert not np.array_equal(baseline, changed)


def test_wall_draw_kinds_select_independent_variates() -> None:
    values = [
        _stochastic_randomness.draw_wall_uniform(_context(), draw_kind)
        for draw_kind in (
            "stick",
            "diffuse_choice",
            "diffuse_polar",
            "diffuse_azimuth",
        )
    ]

    assert len(set(values)) == len(values)


def test_unknown_wall_draw_kind_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown wall random draw kind"):
        _stochastic_randomness.draw_wall_uniform(_context(), "unknown")


@pytest.mark.parametrize(
    ("stick_probability", "expected_outcome", "expected_velocity"),
    [
        (0.0, "reflected_specular", [-2.0, 0.5]),
        (1.0, "stuck", [0.0, 0.0]),
    ],
)
def test_keyed_deterministic_specular_response_does_not_draw(
    monkeypatch: pytest.MonkeyPatch,
    stick_probability: float,
    expected_outcome: str,
    expected_velocity: list[float],
) -> None:
    def fail_draw(*_args: object, **_kwargs: object) -> float:
        raise AssertionError("deterministic specular response must not draw")

    monkeypatch.setattr(_stochastic_randomness, "draw_wall_uniform", fail_draw)

    outcome, velocity = _apply_keyed_wall_response(
        _context(),
        np.asarray([2.0, 0.5]),
        np.asarray([1.0, 0.0]),
        _model("specular", stick_probability=stick_probability),
    )

    assert outcome == expected_outcome
    np.testing.assert_array_equal(velocity, expected_velocity)


def _write_wall_case(
    root: Path,
    *,
    particle_order: tuple[int, ...],
    law_name: str,
    stick_probability: float,
) -> Path:
    root.mkdir(parents=True)
    rows = {
        1: {
            "particle_id": 1,
            "x_m": 0.0,
            "y_m": -0.4,
            "vx_mps": 2.0,
            "vy_mps": 0.0,
            "release_time_s": 0.0,
            "mass_kg": 1.0e-12,
            "drag_diameter_m": 1.0e-6,
            "charge_C": 0.0,
            "source_part_id": 1,
        },
        2: {
            "particle_id": 2,
            "x_m": 0.0,
            "y_m": 0.4,
            "vx_mps": 2.0,
            "vy_mps": 0.0,
            "release_time_s": 0.0,
            "mass_kg": 1.0e-12,
            "drag_diameter_m": 1.0e-6,
            "charge_C": 0.0,
            "source_part_id": 1,
        },
    }
    pd.DataFrame([rows[particle_id] for particle_id in particle_order]).to_csv(
        root / "particles.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "part_id": part_id,
                "part_name": f"wall_{part_id}",
                "role": "wall",
                "material_id": part_id,
                "material_name": "test",
                "wall_law": law_name,
                "wall_stick_probability": stick_probability,
                "wall_restitution": 1.0,
                "wall_diffuse_fraction": 0.5,
                "wall_critical_sticking_velocity_mps": 0.0,
            }
            for part_id in (1, 2, 3, 4)
        ]
    ).to_csv(root / "boundaries.csv", index=False)
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
                    "bounds": [-1.0, 1.0, -1.0, 1.0],
                    "grid_shape": [21, 21],
                    "boundary_part_ids": [1, 2, 3, 4],
                },
            },
            "field": {
                "kind": "linear_shear",
                "parameters": {
                    "shear_rate": 0.0,
                    "dynamic_viscosity_Pas": 1.8e-5,
                },
            },
        },
        "physics": {
            "drag": {"model": "none"},
            "gas": {},
            "forces": {},
            "seed": 31415,
        },
        "time": {"dt": 1.0, "t_end": 1.0},
        "output": {"mode": "debug", "trajectory_interval_steps": 1},
    }
    path = root / "case.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _particle_result(
    path: Path,
    particle_id: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    result = simulate(load_case(path))
    index = int(np.flatnonzero(result.state.particle_id == particle_id)[0])
    events = [
        str(row["outcome"])
        for row in result.debug["wall_events"]
        if int(row["particle_id"]) == particle_id
    ]
    return result.state.position_m[index], result.state.velocity_mps[index], events


@pytest.mark.parametrize(
    ("law_name", "stick_probability"),
    [("cosine_diffuse", 0.0), ("specular", 0.5)],
)
def test_runtime_wall_randomness_ignores_unrelated_particle_and_row_order(
    tmp_path: Path,
    law_name: str,
    stick_probability: float,
) -> None:
    baseline = _particle_result(
        _write_wall_case(
            tmp_path / "baseline",
            particle_order=(1,),
            law_name=law_name,
            stick_probability=stick_probability,
        ),
        1,
    )
    for name, order in (("expanded", (2, 1)), ("reordered", (1, 2))):
        actual = _particle_result(
            _write_wall_case(
                tmp_path / name,
                particle_order=order,
                law_name=law_name,
                stick_probability=stick_probability,
            ),
            1,
        )
        np.testing.assert_array_equal(actual[0], baseline[0])
        np.testing.assert_array_equal(actual[1], baseline[1])
        assert actual[2] == baseline[2]
