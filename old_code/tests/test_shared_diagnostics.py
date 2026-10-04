from __future__ import annotations

from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np

from particle_tracer_unified.compare._common import finite_json_safe, json_safe
from particle_tracer_unified.experimental_features import (
    enabled_experimental_features,
)
from particle_tracer_unified.force_models import (
    DielectrophoresisForce,
    DragForce,
    ForceModel,
    LiftForce,
    PressureGradientForce,
    ThermophoresisForce,
    VirtualMassForce,
)


def test_experimental_features_follow_force_status_and_stable_order() -> None:
    force_model = ForceModel(
        drag=DragForce(),
        thermophoresis=ThermophoresisForce(enabled=True),
        dielectrophoresis=DielectrophoresisForce(enabled=True),
        lift=LiftForce(enabled=True),
        pressure_gradient=PressureGradientForce(enabled=True),
        virtual_mass=VirtualMassForce(enabled=True),
    )
    physics = SimpleNamespace(
        charge=SimpleNamespace(enabled=True),
        stochastic=SimpleNamespace(enabled=True),
    )

    assert enabled_experimental_features(force_model, physics) == (
        "brownian_motion",
        "dielectrophoresis",
        "dynamic_charge",
        "lift",
        "pressure_gradient",
        "thermophoresis",
        "virtual_mass",
    )


def test_experimental_features_accept_absent_runtime_and_optional_models() -> None:
    assert enabled_experimental_features(None, None) == ()
    assert (
        enabled_experimental_features(ForceModel(drag=DragForce()), SimpleNamespace())
        == ()
    )


def test_json_safe_preserves_existing_compare_value_policy(tmp_path: Path) -> None:
    payload = {
        7: np.int64(3),
        "tuple": (np.float64(1.25), Path(tmp_path / "artifact.csv")),
        "numpy_nan": np.float64(np.nan),
        "python_inf": float("inf"),
    }

    converted = json_safe(payload)

    assert list(converted) == ["7", "tuple", "numpy_nan", "python_inf"]
    assert converted["7"] == 3
    assert converted["tuple"] == [1.25, str(tmp_path / "artifact.csv")]
    assert np.isnan(converted["numpy_nan"])
    assert np.isinf(converted["python_inf"])

    non_dict_mapping = MappingProxyType({"numpy_nan": np.float64(np.nan)})
    assert json_safe(non_dict_mapping) is non_dict_mapping


def test_finite_json_safe_handles_mapping_and_replaces_nonfinite_values(
    tmp_path: Path,
) -> None:
    payload = MappingProxyType(
        {
            "numpy_nan": np.float64(np.nan),
            "numpy_inf": np.float64(np.inf),
            "python_nan": float("nan"),
            "path": tmp_path / "summary.json",
            "nested": (np.int32(4),),
        }
    )

    converted = finite_json_safe(payload)

    assert list(converted) == [
        "numpy_nan",
        "numpy_inf",
        "python_nan",
        "path",
        "nested",
    ]
    assert converted == {
        "numpy_nan": None,
        "numpy_inf": None,
        "python_nan": None,
        "path": str(tmp_path / "summary.json"),
        "nested": [4],
    }
