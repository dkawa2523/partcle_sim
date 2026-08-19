from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from particle_tracer_unified import migration
from particle_tracer_unified._migration.forces import _legacy_drag_model
from particle_tracer_unified._migration.tables import (
    _canonical_boundaries,
    _canonical_particles,
)

ROOT = Path(__file__).resolve().parents[1]
LEGACY_CASE = ROOT / "tests" / "fixtures" / "legacy_minimal_2d" / "run_config.yaml"


def _particle_frame(**updates: object) -> pd.DataFrame:
    row: dict[str, object] = {
        "id": 7,
        "t0": 0.25,
        "mass": 1.0e-15,
        "d_eq": 2.0e-6,
        "q": -3.0e-18,
        "origin_part_id": 11,
        "x": 1.0,
        "y": 2.0,
        "z": 3.0,
        "vx": 4.0,
        "vy": 5.0,
        "vz": 6.0,
        "rho_p": 1200.0,
        "particle_material_id": 9,
        "epsr_particle": 2.5,
        "thermo_coeff": 0.75,
    }
    row.update(updates)
    return pd.DataFrame([row])


def test_public_migration_preserves_exact_canonical_artifacts(tmp_path: Path) -> None:
    result = migration.migrate_legacy_case(LEGACY_CASE, tmp_path / "migrated")

    assert result.particles_path is not None
    assert result.boundaries_path is not None
    particles = pd.read_csv(result.particles_path)
    boundaries = pd.read_csv(result.boundaries_path)
    assert particles.columns.tolist() == [
        "particle_id",
        "release_time_s",
        "mass_kg",
        "drag_diameter_m",
        "charge_C",
        "source_part_id",
        "x_m",
        "y_m",
        "vx_mps",
        "vy_mps",
        "density_kgm3",
        "material_id",
    ]
    assert particles[["particle_id", "source_part_id", "material_id"]].to_dict(
        "records"
    ) == [
        {"particle_id": 1, "source_part_id": 10, "material_id": 1},
        {"particle_id": 2, "source_part_id": 20, "material_id": 2},
        {"particle_id": 3, "source_part_id": 10, "material_id": 1},
    ]
    assert boundaries.columns.tolist() == [
        "part_id",
        "part_name",
        "role",
        "material_id",
        "material_name",
        "wall_law",
        "wall_stick_probability",
        "wall_restitution",
        "wall_diffuse_fraction",
        "wall_critical_sticking_velocity_mps",
    ]
    assert boundaries[["part_id", "role", "wall_law"]].to_dict("records") == [
        {"part_id": 10, "role": "wall", "wall_law": "specular"},
        {
            "part_id": 20,
            "role": "wall",
            "wall_law": "mixed_specular_diffuse",
        },
    ]
    assert result.warnings == (
        "legacy integrator was replaced by the v0.2 fixed ETD2 integrator",
        "solver.drag_model was absent; materialized the legacy stokes default",
        "dropped legacy particle sticking columns stick_probability; sticking is "
        "defined only by boundaries.wall_stick_probability in schema v2",
    )


@pytest.mark.parametrize(
    ("spatial_dim", "coordinate_system", "expected"),
    [
        (
            2,
            "cartesian_xy",
            {"x_m": 1.0, "y_m": 2.0, "vx_mps": 4.0, "vy_mps": 5.0},
        ),
        (
            2,
            "axisymmetric_rz",
            {"r_m": 1.0, "z_m": 3.0, "vr_mps": 4.0, "vz_mps": 6.0},
        ),
        (
            3,
            "cartesian_xyz",
            {
                "x_m": 1.0,
                "y_m": 2.0,
                "z_m": 3.0,
                "vx_mps": 4.0,
                "vy_mps": 5.0,
                "vz_mps": 6.0,
            },
        ),
    ],
)
def test_particle_canonicalization_preserves_coordinate_and_optional_aliases(
    spatial_dim: int,
    coordinate_system: str,
    expected: dict[str, float],
) -> None:
    warnings: list[str] = []

    result = _canonical_particles(
        _particle_frame(),
        spatial_dim=spatial_dim,
        coordinate_system=coordinate_system,
        warnings=warnings,
    )

    row = result.iloc[0].to_dict()
    assert {name: row[name] for name in expected} == expected
    assert result.loc[0, "density_kgm3"] == 1200.0
    assert result.loc[0, "material_id"] == 9
    assert result.loc[0, "dep_particle_rel_permittivity"] == 2.5
    assert result.loc[0, "thermophoretic_coeff"] == 0.75
    assert warnings == []


def test_particle_canonicalization_preserves_failure_and_warning_order() -> None:
    with pytest.raises(ValueError, match="at least one row"):
        _canonical_particles(
            pd.DataFrame(),
            spatial_dim=2,
            coordinate_system="cartesian_xy",
            warnings=[],
        )
    with pytest.raises(ValueError, match="missing particle_id"):
        _canonical_particles(
            _particle_frame().drop(columns="id"),
            spatial_dim=4,
            coordinate_system="unknown",
            warnings=[],
        )
    with pytest.raises(ValueError, match="unsupported coordinate system"):
        _canonical_particles(
            _particle_frame(),
            spatial_dim=4,
            coordinate_system="unknown",
            warnings=[],
        )

    warnings: list[str] = []
    invalid_source = _particle_frame(
        origin_part_id="missing",
        stick_probability=0.5,
        p_stick=0.25,
    )
    with pytest.raises(ValueError, match=r"CSV rows \[2\]"):
        _canonical_particles(
            invalid_source,
            spatial_dim=2,
            coordinate_system="cartesian_xy",
            warnings=warnings,
        )
    assert warnings == [
        "dropped legacy particle sticking columns stick_probability, p_stick; "
        "sticking is defined only by boundaries.wall_stick_probability in schema v2"
    ]


def test_boundary_canonicalization_preserves_precedence_roles_and_sorting() -> None:
    walls = pd.DataFrame(
        [
            {"part_id": 30, "material_id": 3, "wall_law": "passthrough"},
            {"part_id": 10, "material_id": 1, "wall_law": "field_support_exit"},
            {"part_id": 20, "material_id": 2, "wall_law": "open"},
        ]
    )
    materials = pd.DataFrame(
        [
            {
                "material_id": 1,
                "material_name": "support",
                "wall_restitution": 0.1,
            },
            {
                "material_id": 2,
                "material_name": "outlet",
                "wall_restitution": 0.2,
            },
            {
                "material_id": 3,
                "material_name": "interior",
                "wall_restitution": 0.3,
            },
        ]
    )

    result = _canonical_boundaries(
        walls,
        materials,
        {"stick_probability": 0.4, "diffuse_fraction": 0.5},
    )

    assert result[["part_id", "role", "wall_law"]].to_dict("records") == [
        {"part_id": 10, "role": "field_support", "wall_law": "escape"},
        {"part_id": 20, "role": "outlet", "wall_law": "escape"},
        {"part_id": 30, "role": "internal", "wall_law": "pass_through"},
    ]
    assert result["part_name"].tolist() == ["part_10", "part_20", "part_30"]
    assert result["material_name"].tolist() == ["support", "outlet", "interior"]
    assert result["wall_restitution"].tolist() == [0.1, 0.2, 0.3]
    assert result["wall_stick_probability"].tolist() == [0.4, 0.4, 0.4]
    assert result["wall_diffuse_fraction"].tolist() == [0.5, 0.5, 0.5]


def test_boundary_canonicalization_preserves_validation_order() -> None:
    with pytest.raises(ValueError, match="at least one row"):
        _canonical_boundaries(pd.DataFrame(), None, {})
    with pytest.raises(ValueError, match="invalid part_id at row 2"):
        _canonical_boundaries(
            pd.DataFrame([{"part_id": 0}, {"part_id": 1, "wall_law": "unknown"}]),
            None,
            {},
        )
    with pytest.raises(ValueError, match=r"duplicate part IDs: \[2\]"):
        _canonical_boundaries(
            pd.DataFrame(
                [
                    {"part_id": 2, "wall_law": "specular"},
                    {"part_id": 2, "wall_law": "bounce"},
                ]
            ),
            None,
            {},
        )


def test_legacy_drag_resolution_preserves_precedence_and_warning_order() -> None:
    warnings: list[str] = []
    assert _legacy_drag_model({}, warnings) == "stokes"
    assert warnings == [
        "solver.drag_model was absent; materialized the legacy stokes default"
    ]

    warnings = []
    assert (
        _legacy_drag_model(
            {"forces": {"drag": False}},
            warnings,
        )
        == "none"
    )
    assert warnings == [
        "solver.drag_model was absent; materialized the legacy stokes default"
    ]

    warnings = []
    assert (
        _legacy_drag_model(
            {"forces": {"drag": {"enabled": False}}},
            warnings,
        )
        == "none"
    )
    assert warnings == [
        "solver.drag_model was absent; materialized the legacy stokes default"
    ]

    warnings = []
    assert (
        _legacy_drag_model(
            {
                "drag_model": "cunningham",
                "forces": {"drag": {"active": "yes", "drag_law": "cunningham"}},
            },
            warnings,
        )
        == "stokes_cunningham"
    )
    assert warnings == []

    with pytest.raises(ValueError, match=r"drag\.model disagree"):
        _legacy_drag_model(
            {
                "drag_model": "epstein",
                "forces": {"drag": {"model": "stokes"}},
            },
            [],
        )
    with pytest.raises(ValueError, match=r"unknown legacy solver\.forces\.drag key"):
        _legacy_drag_model(
            {
                "drag_model": "unknown_solver_model",
                "forces": {"drag": {"unknown_force_key": True}},
            },
            [],
        )
