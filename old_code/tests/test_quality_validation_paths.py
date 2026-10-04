from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from particle_tracer_unified.core.datamodel import FieldProviderND, RegularFieldND
from particle_tracer_unified.io import runtime_builder_support as runtime_support
from particle_tracer_unified.providers.synthetic import (
    build_synthetic_field,
    build_synthetic_geometry,
)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({}, "geometry.kind is required"),
        ({"kind": 1}, "geometry.kind must be a string"),
        ({"kind": " box"}, "leading or trailing whitespace"),
        ({"kind": "sphere"}, "Unsupported synthetic geometry"),
    ],
)
def test_synthetic_geometry_rejects_ambiguous_kinds(
    config: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_synthetic_geometry(config, 2, "cartesian_xy")


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({}, "field.kind is required"),
        ({"kind": 1}, "field.kind must be a string"),
        ({"kind": " linear_shear"}, "leading or trailing whitespace"),
        ({"kind": "uniform"}, "Unsupported synthetic field"),
        ({"kind": "linear_shear", "time_mode": 1}, "time_mode must be a string"),
        (
            {"kind": "linear_shear", "time_mode": " steady"},
            "leading or trailing whitespace",
        ),
        (
            {"kind": "linear_shear", "time_mode": "periodic"},
            "time_mode must be steady or transient",
        ),
        (
            {"kind": "linear_shear", "times": []},
            "times must be a non-empty 1D array",
        ),
        (
            {"kind": "linear_shear", "times": [float("nan")]},
            "times must contain only finite values",
        ),
        (
            {"kind": "linear_shear", "time_mode": "steady", "times": [0.0, 1.0]},
            "steady requires exactly one time value",
        ),
        (
            {"kind": "linear_shear", "time_mode": "transient", "times": [0.0]},
            "transient requires at least two time values",
        ),
    ],
)
def test_synthetic_field_rejects_invalid_time_contracts(
    config: dict[str, object],
    message: str,
) -> None:
    axes = (np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0]))
    with pytest.raises(ValueError, match=message):
        build_synthetic_field(config, 2, "cartesian_xy", axes)


def test_runtime_builder_rejects_invalid_paths_and_provider_alignment(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="must be a string"):
        runtime_support._resolve_path(tmp_path, cast(Any, 1), context="input")
    with pytest.raises(ValueError, match="must not be empty"):
        runtime_support._resolve_path(tmp_path, "", context="input")
    with pytest.raises(ValueError, match="particles_csv is required"):
        runtime_support.resolve_runtime_input_paths(
            tmp_path,
            {"boundaries_csv": "boundaries.csv"},
        )
    with pytest.raises(ValueError, match="boundaries_csv is required"):
        runtime_support.resolve_runtime_input_paths(
            tmp_path,
            {"particles_csv": "particles.csv"},
        )
    with pytest.raises(ValueError, match="unknown entries"):
        runtime_support.build_runtime_providers(
            config_dir=tmp_path,
            providers_cfg={"unknown": {}},
            spatial_dim=2,
            coordinate_system="cartesian_xy",
        )
    with pytest.raises(ValueError, match=r"requires providers\.geometry"):
        runtime_support.build_runtime_providers(
            config_dir=tmp_path,
            providers_cfg={"field": {"kind": "linear_shear"}},
            spatial_dim=2,
            coordinate_system="cartesian_xy",
        )

    geometry = build_synthetic_geometry(
        {"kind": "box", "grid_shape": [3, 3]},
        2,
        "cartesian_xy",
    )
    with pytest.raises(ValueError, match="spatial_dim"):
        runtime_support._align_field_provider_to_geometry(
            cast(Any, SimpleNamespace(field=SimpleNamespace(spatial_dim=3))),
            geometry,
        )
    with pytest.raises(TypeError, match="unsupported field data type"):
        runtime_support._align_field_provider_to_geometry(
            cast(Any, SimpleNamespace(field=SimpleNamespace(spatial_dim=2))),
            geometry,
        )
    field_provider = build_synthetic_field(
        {"kind": "linear_shear"},
        2,
        "cartesian_xy",
        geometry.geometry.axes,
    )
    assert isinstance(field_provider.field, RegularFieldND)
    shifted = replace(
        field_provider.field,
        axes=(field_provider.field.axes[0] + 1.0, field_provider.field.axes[1]),
    )
    with pytest.raises(ValueError, match="must exactly match"):
        runtime_support._align_field_provider_to_geometry(
            FieldProviderND(field=shifted),
            geometry,
        )
    bad_support = replace(
        field_provider.field,
        support_phi=np.zeros((1, 1), dtype=np.float64),
    )
    with pytest.raises(ValueError, match="support_phi shape mismatch"):
        runtime_support._align_field_provider_to_geometry(
            FieldProviderND(field=bad_support),
            geometry,
        )


@pytest.mark.parametrize(
    ("field_axis_count", "geometry_axis_count", "shift_common_axis"),
    [(1, 2, True), (3, 2, False), (2, 1, False)],
)
def test_runtime_builder_rejects_axis_count_mismatch_before_axis_values(
    field_axis_count: int,
    geometry_axis_count: int,
    shift_common_axis: bool,
) -> None:
    geometry = build_synthetic_geometry(
        {"kind": "box", "grid_shape": [3, 3]},
        2,
        "cartesian_xy",
    )
    field_provider = build_synthetic_field(
        {"kind": "linear_shear"},
        2,
        "cartesian_xy",
        geometry.geometry.axes,
    )
    assert isinstance(field_provider.field, RegularFieldND)
    axis = np.asarray([10.0, 11.0, 12.0], dtype=np.float64)
    field_axes = (*field_provider.field.axes[:field_axis_count],)
    if field_axis_count > len(field_axes):
        field_axes = (*field_axes, axis)
    if shift_common_axis:
        field_axes = (field_axes[0] + 1.0, *field_axes[1:])
    geometry_axes = geometry.geometry.axes[:geometry_axis_count]
    malformed_field = replace(field_provider.field, axes=field_axes)
    malformed_geometry = replace(
        geometry,
        geometry=replace(geometry.geometry, axes=geometry_axes),
    )

    with pytest.raises(
        ValueError,
        match="Field and geometry must each provide exactly spatial_dim axes",
    ):
        runtime_support._align_field_provider_to_geometry(
            FieldProviderND(field=malformed_field),
            malformed_geometry,
        )
