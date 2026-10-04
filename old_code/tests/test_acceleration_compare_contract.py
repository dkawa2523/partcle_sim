from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from particle_tracer_unified.compare import acceleration_compare


def _runtime() -> SimpleNamespace:
    particles = SimpleNamespace(
        particle_id=np.asarray([7], dtype=np.int64),
        diameter=np.asarray([1.0e-6]),
        density=np.asarray([1000.0]),
        mass=np.asarray([1.0e-15]),
        charge=np.asarray([0.0]),
        dep_particle_rel_permittivity=np.asarray([2.5]),
        thermophoretic_coeff=np.asarray([1.0]),
    )
    return SimpleNamespace(
        particles=particles,
        spatial_dim=2,
        force_catalog=SimpleNamespace(model=object()),
        gas=SimpleNamespace(
            density_kgm3=1.0,
            dynamic_viscosity_Pas=1.0e-5,
            temperature=300.0,
            molecular_mass_amu=40.0,
        ),
    )


@pytest.mark.parametrize(
    ("particle_id", "message"),
    [
        (None, "acceleration comparison points must contain particle_id"),
        (99, "acceleration comparison point references unknown particle_id 99"),
    ],
)
def test_acceleration_comparison_rejects_implicit_particle_defaults(
    monkeypatch: pytest.MonkeyPatch,
    particle_id: int | None,
    message: str,
) -> None:
    monkeypatch.setattr(
        acceleration_compare,
        "compile_force_runtime_parameters",
        lambda _model: object(),
    )
    monkeypatch.setattr(
        acceleration_compare,
        "compile_runtime_backend",
        lambda *_args, **_kwargs: object(),
    )
    row = {"point_id": 1, "time": 0.0, "x": 0.0, "y": 0.0}
    if particle_id is not None:
        row["particle_id"] = particle_id

    with pytest.raises(ValueError, match=message):
        acceleration_compare._sample_acceleration(_runtime(), pd.DataFrame([row]))


def test_acceleration_comparison_uses_explicit_particle_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        acceleration_compare,
        "compile_force_runtime_parameters",
        lambda _model: object(),
    )
    monkeypatch.setattr(
        acceleration_compare,
        "compile_runtime_backend",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        acceleration_compare,
        "sample_compiled_acceleration_vector",
        lambda *_args, **_kwargs: np.asarray([1.5, -2.5]),
    )

    result = acceleration_compare._sample_acceleration(
        _runtime(),
        pd.DataFrame(
            [{"point_id": 4, "particle_id": 7, "time": 0.0, "x": 0.0, "y": 0.0}]
        ),
    )

    assert result["particle_id"].tolist() == [7, 7]
    assert result["python_value"].tolist() == [1.5, -2.5]
