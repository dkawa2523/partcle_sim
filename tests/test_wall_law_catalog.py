from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from particle_tracer_unified.core.catalogs import build_wall_catalog
from particle_tracer_unified.core.datamodel import PartWallRow, PartWallTable
from particle_tracer_unified.solvers.wall_response import apply_wall_response


def _wall_row(
    *, part_id: int, part_name: str, wall_law: str, **overrides: object
) -> PartWallRow:
    values: dict[str, object] = {
        "part_id": part_id,
        "part_name": part_name,
        "role": "wall",
        "material_id": 1,
        "material_name": "test_material",
        "wall_law": wall_law,
        "wall_stick_probability": 0.0,
        "wall_restitution": 1.0,
        "wall_diffuse_fraction": 0.0,
        "wall_critical_sticking_velocity_mps": 0.0,
    }
    values.update(overrides)
    return PartWallRow(**values)


def _wall_model(law_name: str):
    catalog = build_wall_catalog(
        PartWallTable(
            rows=(
                _wall_row(
                    part_id=1,
                    part_name="wall",
                    wall_law=law_name,
                    role="internal" if law_name == "pass_through" else "wall",
                ),
            )
        ),
    )
    return catalog.model_for_part(1)


def test_production_explicit_known_wall_law_builds_catalog() -> None:
    walls = PartWallTable(
        rows=(
            _wall_row(
                part_id=987654,
                part_name="arbitrary_part",
                wall_law="mixed_specular_diffuse",
                wall_restitution=0.5,
                wall_diffuse_fraction=0.25,
            ),
        )
    )

    catalog = build_wall_catalog(walls)
    model = catalog.model_for_part(987654)

    assert model.law_name == "mixed_specular_diffuse"
    assert model.restitution == pytest.approx(0.5)
    assert model.diffuse_fraction == pytest.approx(0.25)


def test_production_unknown_wall_law_fails_instead_of_specular_fallback() -> None:
    walls = PartWallTable(
        rows=(_wall_row(part_id=7, part_name="wall", wall_law="teleport"),)
    )

    with pytest.raises(ValueError, match="Unsupported wall law"):
        build_wall_catalog(walls)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"wall_stick_probability": np.nan}, "non-finite wall coefficients"),
        (
            {
                "wall_stick_probability": 2.0,
                "wall_diffuse_fraction": 2.0,
                "wall_restitution": -1.0,
            },
            "stick probability",
        ),
        (
            {"wall_diffuse_fraction": 2.0, "wall_restitution": -1.0},
            "diffuse fraction",
        ),
        ({"wall_restitution": -1.0}, "wall coefficients must be non-negative"),
    ],
)
def test_wall_coefficient_validation_order_is_stable(
    overrides: dict[str, float],
    message: str,
) -> None:
    row = _wall_row(
        part_id=4,
        part_name="wall",
        wall_law="teleport",
        **overrides,
    )

    with pytest.raises(ValueError, match=message):
        build_wall_catalog(PartWallTable(rows=(row,)))


def test_duplicate_part_id_precedes_second_row_coefficient_validation() -> None:
    rows = (
        _wall_row(part_id=3, part_name="first", wall_law="stick"),
        _wall_row(
            part_id=3,
            part_name="duplicate",
            wall_law="stick",
            wall_stick_probability=np.nan,
        ),
    )

    with pytest.raises(ValueError, match="Duplicate boundary part_id=3"):
        build_wall_catalog(PartWallTable(rows=rows))


@pytest.mark.parametrize(
    "law_name",
    [
        "stick",
        "freeze",
        "absorb",
        "escape",
        "pass_through",
        "specular",
        "cosine_diffuse",
        "mixed_specular_diffuse",
        "critical_sticking_velocity",
    ],
)
def test_only_canonical_wall_law_names_are_accepted(law_name: str) -> None:
    assert _wall_model(law_name).law_name == law_name


@pytest.mark.parametrize("alias", ["bounce", "diffuse", "inactive", "open"])
def test_wall_law_aliases_are_rejected(alias: str) -> None:
    with pytest.raises(ValueError, match="Unsupported wall law"):
        _wall_model(alias)


def _test_unknown_collision_wall_law_has_no_specular_fallback() -> None:
    model = _wall_model("specular")
    object.__setattr__(model, "law_name", "teleport")

    with pytest.raises(ValueError, match="Unsupported collision wall law"):
        apply_wall_response(
            np.random.default_rng(1),
            np.asarray([1.0, -2.0], dtype=float),
            np.asarray([1.0, 0.0], dtype=float),
            model,
        )


test_collision_wall_interaction_rejects_unknown_wall_law_without_specular_fallback = (
    _test_unknown_collision_wall_law_has_no_specular_fallback
)


def test_collision_wall_interaction_rejects_zero_normal_without_numerical_floor() -> (
    None
):
    with pytest.raises(
        ValueError, match="wall normal must be a finite non-zero vector"
    ):
        apply_wall_response(
            np.random.default_rng(1),
            np.asarray([1.0, -2.0], dtype=float),
            np.zeros(2, dtype=float),
            _wall_model("specular"),
        )


def test_pass_through_wall_interaction_keeps_velocity_instead_of_reflecting() -> None:
    velocity = np.asarray([1.0, -2.0], dtype=float)

    outcome, updated = apply_wall_response(
        np.random.default_rng(1),
        velocity,
        np.asarray([1.0, 0.0], dtype=float),
        _wall_model("pass_through"),
    )

    assert outcome == "passed_through"
    assert updated.tolist() == pytest.approx(velocity.tolist())


def test_wall_role_pass_through_is_an_exterior_exit() -> None:
    row = _wall_row(part_id=1, part_name="wall", wall_law="pass_through")
    model = build_wall_catalog(PartWallTable(rows=(row,))).model_for_part(1)

    outcome, velocity = apply_wall_response(
        np.random.default_rng(1),
        np.asarray([1.0, -2.0]),
        np.asarray([1.0, 0.0]),
        model,
    )

    assert outcome == "escaped"
    np.testing.assert_array_equal(velocity, [1.0, -2.0])


def test_wall_law_core_has_no_vigus_specific_branches_or_ids() -> None:
    root = Path(__file__).resolve().parents[1]
    production_files = [
        root / "particle_tracer_unified" / "core" / "catalogs.py",
        root / "particle_tracer_unified" / "io" / "comsol_boundary_reader.py",
        root / "particle_tracer_unified" / "solvers" / "wall_response.py",
    ]

    for path in production_files:
        assert "vigus" not in path.read_text(encoding="utf-8").lower()
