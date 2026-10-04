from __future__ import annotations

from dataclasses import replace
from typing import cast

import numpy as np
import pytest

from particle_tracer_unified.core.datamodel import WallPartModel
from particle_tracer_unified.solvers.wall_response import apply_wall_response


class _TracingRng:
    def __init__(
        self,
        *,
        random_values: tuple[float, ...] = (),
        uniform_values: tuple[float, ...] = (),
    ) -> None:
        self.random_values = list(random_values)
        self.uniform_values = list(uniform_values)
        self.calls: list[str] = []

    def random(self) -> float:
        self.calls.append("random")
        return self.random_values.pop(0)

    def uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        self.calls.append("uniform")
        value = self.uniform_values.pop(0)
        return float(low + (high - low) * value)


def _rng(trace: _TracingRng) -> np.random.Generator:
    return cast(np.random.Generator, trace)


def _model(law_name: str, **overrides: float) -> WallPartModel:
    values = {
        "stick_probability": 0.0,
        "restitution": 0.5,
        "diffuse_fraction": 0.25,
        "critical_sticking_velocity_mps": 0.0,
    }
    values.update(overrides)
    return WallPartModel(
        part_id=1,
        part_name="wall",
        material_id=1,
        material_name="test",
        law_name=law_name,
        metadata={"role": "internal" if law_name == "pass_through" else "wall"},
        **values,
    )


@pytest.mark.parametrize(
    ("law_name", "expected_outcome", "keeps_velocity"),
    [
        ("stick", "stuck", False),
        ("freeze", "frozen", True),
        ("escape", "escaped", False),
        ("absorb", "absorbed", False),
        ("pass_through", "passed_through", True),
    ],
)
def test_terminal_wall_laws_do_not_consume_randomness(
    law_name: str,
    expected_outcome: str,
    keeps_velocity: bool,
) -> None:
    trace = _TracingRng()
    velocity = np.asarray([2.0, -3.0], dtype=np.float32)

    outcome, updated = apply_wall_response(
        _rng(trace),
        velocity,
        np.asarray([1.0, 0.0]),
        _model(law_name),
    )

    assert outcome == expected_outcome
    np.testing.assert_array_equal(updated, velocity if keeps_velocity else 0.0)
    assert updated.dtype == np.float64
    assert not np.shares_memory(updated, velocity)
    assert trace.calls == []


def test_wall_response_validates_law_before_normal_without_rng_draw() -> None:
    trace = _TracingRng()

    with pytest.raises(ValueError, match="Unsupported collision wall law"):
        apply_wall_response(
            _rng(trace),
            np.asarray([1.0, -2.0]),
            np.zeros(2),
            _model("unknown"),
        )

    assert trace.calls == []


@pytest.mark.parametrize("role", ["wall", "inlet", "outlet", "field_support"])
def test_exterior_pass_through_becomes_escape_without_changing_hit_velocity(
    role: str,
) -> None:
    trace = _TracingRng()
    velocity = np.asarray([2.0, -3.0], dtype=np.float64)

    outcome, updated = apply_wall_response(
        _rng(trace),
        velocity,
        np.asarray([1.0, 0.0]),
        replace(_model("pass_through"), metadata={"role": role}),
    )

    assert outcome == "escaped"
    np.testing.assert_array_equal(updated, velocity)
    assert trace.calls == []


@pytest.mark.parametrize(
    ("law_name", "overrides", "random_values", "expected", "calls"),
    [
        (
            "critical_sticking_velocity",
            {"critical_sticking_velocity_mps": 2.0},
            (),
            "stuck",
            [],
        ),
        (
            "critical_sticking_velocity",
            {"critical_sticking_velocity_mps": 0.5},
            (0.8,),
            "reflected_specular",
            ["random"],
        ),
        (
            "specular",
            {"stick_probability": 1.0},
            (0.8,),
            "stuck",
            ["random"],
        ),
    ],
)
def test_sticking_decisions_preserve_rng_draw_order(
    law_name: str,
    overrides: dict[str, float],
    random_values: tuple[float, ...],
    expected: str,
    calls: list[str],
) -> None:
    trace = _TracingRng(random_values=random_values)

    outcome, _ = apply_wall_response(
        _rng(trace),
        np.asarray([1.0, -2.0]),
        np.asarray([1.0, 0.0]),
        _model(law_name, **overrides),
    )

    assert outcome == expected
    assert trace.calls == calls


@pytest.mark.parametrize(
    (
        "law_name",
        "random_values",
        "uniform_values",
        "expected",
        "expected_speed",
        "calls",
    ),
    [
        (
            "cosine_diffuse",
            (0.8,),
            (0.25,),
            "reflected_diffuse",
            2.5,
            ["random", "uniform"],
        ),
        (
            "mixed_specular_diffuse",
            (0.8, 0.1),
            (0.25,),
            "reflected_diffuse",
            2.5,
            ["random", "random", "uniform"],
        ),
        (
            "mixed_specular_diffuse",
            (0.8, 0.9),
            (),
            "reflected_specular",
            float(np.hypot(1.5, 4.0)),
            ["random", "random"],
        ),
    ],
)
def test_reflection_choice_preserves_rng_draw_order_and_finite_speed(
    law_name: str,
    random_values: tuple[float, ...],
    uniform_values: tuple[float, ...],
    expected: str,
    expected_speed: float,
    calls: list[str],
) -> None:
    trace = _TracingRng(
        random_values=random_values,
        uniform_values=uniform_values,
    )
    velocity = np.asarray([3.0, -4.0], dtype=np.float64)

    outcome, updated = apply_wall_response(
        _rng(trace),
        velocity,
        np.asarray([1.0, 0.0]),
        _model(law_name),
    )

    assert outcome == expected
    assert updated.dtype == np.float64
    assert np.all(np.isfinite(updated))
    assert np.linalg.norm(updated) == pytest.approx(expected_speed)
    assert trace.calls == calls


@pytest.mark.parametrize(
    "normal",
    [
        np.asarray([0.0, 0.0, 1.0]),
        np.asarray([1.0, 0.0, 0.0]),
    ],
)
def test_diffuse_3d_draws_azimuth_after_polar_sample(normal: np.ndarray) -> None:
    trace = _TracingRng(
        random_values=(0.8,),
        uniform_values=(0.25, 0.75),
    )

    outcome, updated = apply_wall_response(
        _rng(trace),
        np.asarray([3.0, -4.0, 0.0]),
        normal,
        _model("cosine_diffuse"),
    )

    assert outcome == "reflected_diffuse"
    assert np.linalg.norm(updated) == pytest.approx(2.5)
    assert float(np.dot(updated, normal)) < 0.0
    assert trace.calls == ["random", "uniform", "uniform"]
