from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from particle_tracer_unified.solvers._stochastic_config import (
    StochasticMotionConfig as OwnedStochasticMotionConfig,
)
from particle_tracer_unified.solvers._stochastic_config import (
    merge_stochastic_motion_diagnostics,
)
from particle_tracer_unified.solvers._stochastic_path import (
    PiecewiseLangevinPath as OwnedPiecewiseLangevinPath,
)
from particle_tracer_unified.solvers.stochastic_motion import (
    PiecewiseLangevinPath,
    StochasticMotionConfig,
    compose_piecewise_langevin_trace,
    resolve_piecewise_valid_mask_prefix,
)


def _path() -> PiecewiseLangevinPath:
    return PiecewiseLangevinPath(
        leaf_end_times_s=np.asarray([1.0]),
        tau_eff_s=np.asarray([0.2]),
        thermal_velocity_variance_m2s2=np.asarray([1.0]),
        z_velocity=np.asarray([[0.1, -0.2]]),
        z_position=np.asarray([[0.3, 0.4]]),
        bridge_seeds=np.asarray([17], dtype=np.int64),
    )


def test_public_types_are_the_owner_types() -> None:
    assert StochasticMotionConfig is OwnedStochasticMotionConfig
    assert PiecewiseLangevinPath is OwnedPiecewiseLangevinPath


def test_diagnostic_merge_handles_disabled_unapplied_and_invalid_existing_state() -> (
    None
):
    class _DebugDisabled(dict[str, object]):
        debug = False

    config = StochasticMotionConfig(enabled=True)
    disabled: dict[str, object] = _DebugDisabled()
    merge_stochastic_motion_diagnostics(disabled, config, {"applied": True})
    assert disabled == {}

    diagnostics: dict[str, object] = {"stochastic_motion": "invalid"}
    merge_stochastic_motion_diagnostics(diagnostics, config, {"applied": False})
    summary = diagnostics["stochastic_motion"]
    assert isinstance(summary, dict)
    assert summary["enabled"] == 1
    assert summary["kick_event_count"] == 0


@pytest.mark.parametrize(
    ("duration_s", "times_s", "offset_s", "message"),
    [
        (0.0, np.asarray([0.0]), 0.0, "positive segment duration"),
        (0.5, np.asarray([0.5]), 0.6, "outside the saved path interval"),
        (0.5, np.asarray([]), 0.0, "no accepted stage nodes"),
    ],
)
def test_composition_rejects_invalid_segment_contracts(
    duration_s: float,
    times_s: np.ndarray,
    offset_s: float,
    message: str,
) -> None:
    trace = cast(
        Any,
        SimpleNamespace(
            request=SimpleNamespace(duration_s=duration_s),
            times_s=times_s,
        ),
    )

    with pytest.raises((ValueError, RuntimeError), match=message):
        compose_piecewise_langevin_trace(
            path=_path(),
            deterministic_trace=trace,
            stochastic_offset_s=offset_s,
        )


def test_zero_duration_prefix_keeps_the_initial_state() -> None:
    request = cast(
        Any,
        SimpleNamespace(
            position_m=np.asarray([1.0, 2.0]),
            velocity_mps=np.asarray([3.0, 4.0]),
            duration_s=0.0,
        ),
    )

    resolution = resolve_piecewise_valid_mask_prefix(
        request,
        _path(),
        max_halving_count=4,
    )

    assert resolution.found_valid_prefix is False
    assert resolution.retry_count == 0
    np.testing.assert_array_equal(resolution.position, request.position_m)
    np.testing.assert_array_equal(resolution.velocity, request.velocity_mps)
