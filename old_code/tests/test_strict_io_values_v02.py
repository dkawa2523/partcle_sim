from __future__ import annotations

from pathlib import Path

import pytest

from particle_tracer_unified.io.runtime_builder_support import (
    build_runtime_providers,
    resolve_runtime_input_paths,
)


def _box_config() -> dict[str, object]:
    return {
        "kind": "box",
        "bounds": [0.0, 1.0, 0.0, 1.0],
        "grid_shape": [3, 3],
        "boundary_part_ids": [1, 1, 1, 1],
    }


@pytest.mark.parametrize("kind", [None, "Box", " box", "npz"])
def test_geometry_provider_kind_has_no_default_or_alias(
    tmp_path: Path, kind: str | None
) -> None:
    geometry = _box_config()
    if kind is None:
        geometry.pop("kind")
    else:
        geometry["kind"] = kind

    with pytest.raises(ValueError, match=r"providers\.geometry\.kind"):
        build_runtime_providers(
            config_dir=tmp_path,
            providers_cfg={"geometry": geometry},
            spatial_dim=2,
            coordinate_system="cartesian_xy",
        )


def test_field_provider_kind_is_required_at_adapter_edge(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"providers\.field\.kind is required"):
        build_runtime_providers(
            config_dir=tmp_path,
            providers_cfg={"geometry": _box_config(), "field": {}},
            spatial_dim=2,
            coordinate_system="cartesian_xy",
        )


@pytest.mark.parametrize("kind", ["Linear_Shear", " linear_shear", "npz"])
def test_field_provider_kind_has_no_case_or_alias_normalization(
    tmp_path: Path, kind: str
) -> None:
    with pytest.raises(ValueError, match=r"providers\.field\.kind"):
        build_runtime_providers(
            config_dir=tmp_path,
            providers_cfg={"geometry": _box_config(), "field": {"kind": kind}},
            spatial_dim=2,
            coordinate_system="cartesian_xy",
        )


@pytest.mark.parametrize("value", [" particles.csv", "particles.csv "])
def test_runtime_input_paths_reject_surrounding_whitespace(
    tmp_path: Path, value: str
) -> None:
    with pytest.raises(ValueError, match="leading or trailing whitespace"):
        resolve_runtime_input_paths(
            tmp_path,
            {"particles_csv": value, "boundaries_csv": "boundaries.csv"},
        )


def test_provider_entry_must_be_a_mapping(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"providers\.geometry must be a mapping"):
        build_runtime_providers(
            config_dir=tmp_path,
            providers_cfg={"geometry": "box"},
            spatial_dim=2,
            coordinate_system="cartesian_xy",
        )
