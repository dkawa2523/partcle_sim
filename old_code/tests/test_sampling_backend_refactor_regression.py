from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from particle_tracer_unified.core.field_sampling import VALID_MASK_STATUS_CLEAN
from particle_tracer_unified.domain import FieldRequest
from particle_tracer_unified.solvers import sampling_backend as sampling_module
from particle_tracer_unified.solvers.compiled_backend_types import (
    RegularRectilinearCompiledBackend,
)
from particle_tracer_unified.solvers.sampling_backend import (
    DYNAMIC_VISCOSITY,
    ELECTRIC_FIELD,
    FLOW_VELOCITY,
    GAS_DENSITY,
    TEMPERATURE,
    VALID_MASK_STATUS,
    CompiledSamplingBackend,
)


def _compiled_backend() -> RegularRectilinearCompiledBackend:
    axes = (
        np.asarray([0.0, 1.0], dtype=np.float64),
        np.asarray([0.0, 1.0], dtype=np.float64),
    )
    scalar = np.zeros((1, 2, 2), dtype=np.float64)
    mask = np.ones((2, 2), dtype=bool)
    return RegularRectilinearCompiledBackend(
        axes=axes,
        times=np.asarray([0.0], dtype=np.float64),
        ux=scalar,
        uy=scalar,
        gas_density=scalar,
        gas_mu=scalar,
        gas_temperature=scalar,
        valid_mask=mask,
        core_valid_mask=mask,
        backend_kind="characterization_backend",
    )


def _sampling_backend(*, strict: bool = True) -> CompiledSamplingBackend:
    return CompiledSamplingBackend(
        compiled=_compiled_backend(),
        spatial_dim=2,
        fallback_density_kgm3=1.2,
        fallback_dynamic_viscosity_Pas=1.8e-5,
        fallback_temperature_K=300.0,
        strict=strict,
    )


def test_sample_preserves_callback_order_and_stage_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def statuses(_backend: object, points: np.ndarray) -> np.ndarray:
        calls.append("status")
        assert points.dtype == np.float64
        return np.asarray([VALID_MASK_STATUS_CLEAN, 2], dtype=np.uint8)

    def flow(
        _backend: object,
        spatial_dim: int,
        time_s: float,
        points: np.ndarray,
    ) -> np.ndarray:
        calls.append("flow")
        assert (spatial_dim, time_s, points.shape) == (2, 0.25, (2, 2))
        return np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    def electric(
        _backend: object,
        _spatial_dim: int,
        _time_s: float,
        _points: np.ndarray,
    ) -> np.ndarray:
        calls.append("electric")
        return np.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

    def gas(
        _backend: object,
        _spatial_dim: int,
        _time_s: float,
        _points: np.ndarray,
        **_fallback: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        calls.append("gas")
        return (
            np.asarray([1.0, 2.0], dtype=np.float32),
            np.asarray([3.0, 4.0], dtype=np.float32),
            np.asarray([300.0, 310.0], dtype=np.float32),
        )

    monkeypatch.setattr(
        sampling_module, "sample_compiled_valid_mask_statuses", statuses
    )
    monkeypatch.setattr(sampling_module, "sample_compiled_flow_vectors", flow)
    monkeypatch.setattr(sampling_module, "sample_compiled_electric_vectors", electric)
    monkeypatch.setattr(sampling_module, "sample_compiled_gas_properties_vectors", gas)

    request = FieldRequest(
        (
            TEMPERATURE,
            FLOW_VELOCITY,
            VALID_MASK_STATUS,
            ELECTRIC_FIELD,
            GAS_DENSITY,
            DYNAMIC_VISCOSITY,
        )
    )
    sampled = _sampling_backend().sample(
        np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        float(np.float32(0.25)),
        request,
    )

    assert calls == ["status", "flow", "electric", "gas"]
    assert sampled.points_m.dtype == np.float64
    assert sampled.supported.dtype == np.bool_
    assert sampled.supported.tolist() == [True, False]
    assert list(sampled.values) == [
        VALID_MASK_STATUS,
        FLOW_VELOCITY,
        ELECTRIC_FIELD,
        GAS_DENSITY,
        DYNAMIC_VISCOSITY,
        TEMPERATURE,
    ]
    assert all(value.dtype == np.float64 for value in sampled.values.values())
    assert {
        name: value
        for name, value in sampled.metadata.items()
        if name != "valid_mask_status"
    } == {
        "backend_kind": "characterization_backend",
        "interpolation": "linear",
        "sample_call_count": 4,
        "sample_point_count": 8,
    }
    np.testing.assert_array_equal(
        sampled.metadata["valid_mask_status"],
        np.asarray([VALID_MASK_STATUS_CLEAN, 2], dtype=np.uint8),
    )


def test_support_status_is_required_even_when_not_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def statuses(_backend: object, _points: np.ndarray) -> np.ndarray:
        calls.append("status")
        return np.asarray([2], dtype=np.uint8)

    def flow(
        _backend: object,
        _spatial_dim: int,
        _time_s: float,
        _points: np.ndarray,
    ) -> np.ndarray:
        calls.append("flow")
        return np.asarray([[1.0, 2.0]], dtype=np.float64)

    monkeypatch.setattr(
        sampling_module, "sample_compiled_valid_mask_statuses", statuses
    )
    monkeypatch.setattr(sampling_module, "sample_compiled_flow_vectors", flow)

    sampled = _sampling_backend().sample(
        np.asarray([[0.1, 0.2]], dtype=np.float64),
        0.0,
        FieldRequest((FLOW_VELOCITY,)),
    )

    assert calls == ["status", "flow"]
    assert sampled.supported.tolist() == [False]
    assert list(sampled.values) == [FLOW_VELOCITY]
    assert sampled.metadata["sample_call_count"] == 2
    assert sampled.metadata["sample_point_count"] == 2
    np.testing.assert_array_equal(
        sampled.metadata["valid_mask_status"], np.asarray([2], dtype=np.uint8)
    )


def test_unknown_quantity_is_rejected_before_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callback: Callable[..., np.ndarray] = pytest.fail
    monkeypatch.setattr(
        sampling_module,
        "sample_compiled_valid_mask_statuses",
        callback,
    )

    with pytest.raises(KeyError, match=r"unsupported field quantities \('pressure',\)"):
        _sampling_backend().sample(
            np.asarray([[0.1, 0.2]], dtype=np.float64),
            0.0,
            FieldRequest(("pressure",)),
        )


@pytest.mark.parametrize(
    ("points", "message"),
    [
        (np.asarray([0.1, 0.2]), "must have shape"),
        (np.zeros((1, 3), dtype=np.float64), "must have shape"),
        (np.asarray([[np.nan, 0.2]]), "finite coordinates"),
    ],
)
def test_sample_rejects_invalid_point_arrays(
    points: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _sampling_backend().sample(points, 0.0, FieldRequest((FLOW_VELOCITY,)))


def test_sample_rejects_non_finite_time() -> None:
    with pytest.raises(ValueError, match="time_s must be finite"):
        _sampling_backend().sample(
            np.asarray([[0.1, 0.2]], dtype=np.float64),
            np.inf,
            FieldRequest((FLOW_VELOCITY,)),
        )


def test_non_strict_missing_electric_field_preserves_support_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sampling_module,
        "sample_compiled_valid_mask_statuses",
        lambda _backend, _points: np.asarray([VALID_MASK_STATUS_CLEAN], dtype=np.uint8),
    )
    monkeypatch.setattr(
        sampling_module,
        "sample_compiled_electric_vectors",
        lambda _backend, _dim, _time, _points: None,
    )

    sampled = _sampling_backend(strict=False).sample(
        np.asarray([[0.1, 0.2]], dtype=np.float64),
        0.0,
        FieldRequest((ELECTRIC_FIELD,)),
    )

    assert sampled.values == {}
    assert sampled.supported.tolist() == [True]
    assert sampled.metadata["sample_call_count"] == 2
    assert sampled.metadata["sample_point_count"] == 2


def test_strict_missing_electric_field_stops_before_gas_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def statuses(_backend: object, _points: np.ndarray) -> np.ndarray:
        calls.append("status")
        return np.asarray([VALID_MASK_STATUS_CLEAN], dtype=np.uint8)

    def electric(
        _backend: object,
        _dim: int,
        _time: float,
        _points: np.ndarray,
    ) -> None:
        calls.append("electric")

    def gas(*_args: object, **_kwargs: float) -> tuple[np.ndarray, ...]:
        calls.append("gas")
        return (np.ones(1), np.ones(1), np.ones(1))

    monkeypatch.setattr(
        sampling_module, "sample_compiled_valid_mask_statuses", statuses
    )
    monkeypatch.setattr(sampling_module, "sample_compiled_electric_vectors", electric)
    monkeypatch.setattr(sampling_module, "sample_compiled_gas_properties_vectors", gas)

    with pytest.raises(
        ValueError,
        match="electric_field was requested but is unavailable",
    ):
        _sampling_backend().sample(
            np.asarray([[0.1, 0.2]], dtype=np.float64),
            0.0,
            FieldRequest((ELECTRIC_FIELD, TEMPERATURE)),
        )

    assert calls == ["status", "electric"]


def test_invalid_gas_value_is_ignored_only_outside_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sampling_module,
        "sample_compiled_valid_mask_statuses",
        lambda _backend, _points: np.asarray(
            [VALID_MASK_STATUS_CLEAN, 2], dtype=np.uint8
        ),
    )
    monkeypatch.setattr(
        sampling_module,
        "sample_compiled_gas_properties_vectors",
        lambda *_args, **_kwargs: (
            np.asarray([1.0, np.nan]),
            np.asarray([2.0, np.nan]),
            np.asarray([300.0, np.nan]),
        ),
    )

    sampled = _sampling_backend().sample(
        np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64),
        0.0,
        FieldRequest((GAS_DENSITY, DYNAMIC_VISCOSITY, TEMPERATURE)),
    )

    assert sampled.supported.tolist() == [True, False]
    assert np.isnan(sampled.require(GAS_DENSITY)[1])
