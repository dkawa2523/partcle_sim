from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from particle_tracer_unified.core.catalogs import build_wall_catalog
from particle_tracer_unified.core.datamodel import MaterialTable, PartWallRow, PartWallTable
from particle_tracer_unified.solvers.high_fidelity_collision import _wall_interaction


def _wall_model(law_name: str):
    catalog = build_wall_catalog(
        PartWallTable(rows=(PartWallRow(part_id=1, part_name="wall", wall_law=law_name),)),
        MaterialTable(rows=()),
        {"wall": {"default_mode": "specular"}},
    )
    return catalog.model_for_part(1)


def test_production_explicit_known_wall_law_builds_catalog() -> None:
    walls = PartWallTable(
        rows=(
            PartWallRow(
                part_id=987654,
                part_name="arbitrary_part",
                wall_law="mixed diffuse/specular",
                wall_restitution=0.5,
                wall_diffuse_fraction=0.25,
            ),
        )
    )

    catalog = build_wall_catalog(walls, MaterialTable(rows=()), {"wall": {"default_mode": "specular"}})
    model = catalog.model_for_part(987654)

    assert model.law_name == "mixed_specular_diffuse"
    assert model.restitution == pytest.approx(0.5)
    assert model.diffuse_fraction == pytest.approx(0.25)


def test_production_unknown_wall_law_fails_instead_of_specular_fallback() -> None:
    walls = PartWallTable(rows=(PartWallRow(part_id=7, part_name="wall", wall_law="teleport"),))

    with pytest.raises(ValueError, match="Unsupported wall law"):
        build_wall_catalog(walls, MaterialTable(rows=()), {"wall": {"default_mode": "specular"}})


def test_collision_wall_interaction_rejects_unknown_wall_law_without_specular_fallback() -> None:
    model = _wall_model("specular")
    object.__setattr__(model, "law_name", "teleport")

    with pytest.raises(ValueError, match="Unsupported collision wall law"):
        _wall_interaction(
            np.random.default_rng(1),
            np.asarray([1.0, -2.0], dtype=float),
            np.asarray([1.0, 0.0], dtype=float),
            0.0,
            model,
        )


def test_pass_through_wall_interaction_keeps_velocity_instead_of_reflecting() -> None:
    velocity = np.asarray([1.0, -2.0], dtype=float)

    outcome, updated = _wall_interaction(
        np.random.default_rng(1),
        velocity,
        np.asarray([1.0, 0.0], dtype=float),
        0.0,
        _wall_model("pass_through"),
    )

    assert outcome == "passed_through"
    assert updated.tolist() == pytest.approx(velocity.tolist())


def test_wall_law_core_has_no_vigus_specific_branches_or_ids() -> None:
    root = Path(__file__).resolve().parents[1]
    production_files = [
        root / "particle_tracer_unified" / "core" / "catalogs.py",
        root / "particle_tracer_unified" / "io" / "comsol_boundary_reader.py",
        root / "particle_tracer_unified" / "solvers" / "high_fidelity_collision.py",
    ]

    for path in production_files:
        assert "vigus" not in path.read_text(encoding="utf-8").lower()
