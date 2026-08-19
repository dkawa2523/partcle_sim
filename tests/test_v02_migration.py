from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from particle_tracer_unified import load_case
from particle_tracer_unified.configuration import load_run_config
from particle_tracer_unified.migration import (
    RemovedSourceGenerationError,
    migrate_legacy_case,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_CASE = REPO_ROOT / "tests" / "fixtures" / "legacy_minimal_2d" / "run_config.yaml"
LEGACY_NPZ_CASE = REPO_ROOT / "tests" / "fixtures" / "legacy_npz_2d" / "run_config.yaml"


def _legacy_value() -> dict[str, object]:
    value = yaml.safe_load(LEGACY_CASE.read_text(encoding="utf-8"))
    value["paths"]["particles_csv"] = str(
        (LEGACY_CASE.parent / "particles.csv").resolve()
    )
    value["paths"]["part_walls_csv"] = str(
        (LEGACY_CASE.parent / "part_walls.csv").resolve()
    )
    value["paths"]["materials_csv"] = str(
        (LEGACY_CASE.parent / "materials.csv").resolve()
    )
    return value


def _legacy_npz_value() -> dict[str, object]:
    value = yaml.safe_load(LEGACY_NPZ_CASE.read_text(encoding="utf-8"))
    value["paths"]["particles_csv"] = str(
        (LEGACY_NPZ_CASE.parent / "particles.csv").resolve()
    )
    value["paths"]["part_walls_csv"] = str(
        (LEGACY_NPZ_CASE.parent / "part_walls.csv").resolve()
    )
    value["paths"]["materials_csv"] = str(
        (LEGACY_NPZ_CASE.parent / "materials.csv").resolve()
    )
    for provider_name in ("geometry", "field"):
        value["providers"][provider_name]["npz_path"] = str(
            (
                LEGACY_NPZ_CASE.parent / value["providers"][provider_name]["npz_path"]
            ).resolve()
        )
    return value


def _legacy_plasma_background() -> dict[str, object]:
    return {
        "source": "SAAS",
        "ne_m3": 2.0e15,
        "ni_m3": 2.0e15,
        "Te_eV": 3.0,
        "Ti_eV": 0.03,
        "mi_amu": 39.948,
        "Zi": 1.0,
        "gas_pressure_Pa": 2.0,
        "conductivity_S_m": 0.0,
    }


def _write_legacy(tmp_path: Path, value: dict[str, object]) -> Path:
    config_path = tmp_path / "legacy.yaml"
    config_path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return config_path


def test_legacy_case_migrates_to_reopenable_canonical_tables(tmp_path: Path) -> None:
    result = migrate_legacy_case(LEGACY_CASE, tmp_path / "migrated")

    config = load_run_config(result.config_path)
    particles = pd.read_csv(result.particles_path)
    boundaries = pd.read_csv(result.boundaries_path)
    assert config.schema_version == 2
    assert config.physics.force_model is not None
    assert config.physics.force_model.drag.model == "stokes"
    assert {
        "particle_id",
        "x_m",
        "y_m",
        "vx_mps",
        "vy_mps",
        "release_time_s",
        "mass_kg",
        "drag_diameter_m",
        "charge_C",
        "source_part_id",
    }.issubset(particles.columns)
    assert "source_event_tag" not in particles.columns
    assert boundaries["part_id"].is_unique
    assert boundaries["wall_law"].notna().all()
    assert set(boundaries["role"]) == {"wall"}
    assert load_case(result.config_path).config.inputs.particles == "particles.csv"


def test_migration_refuses_removed_source_generation(tmp_path: Path) -> None:
    value = _legacy_value()
    value["paths"]["source_events_csv"] = "removed-source-events.csv"
    config_path = _write_legacy(tmp_path, value)

    with pytest.raises(RemovedSourceGenerationError, match=r"paths\.source_events_csv"):
        migrate_legacy_case(config_path, tmp_path / "out")

    assert not (tmp_path / "out").exists()


def test_migration_does_not_overwrite_without_explicit_permission(
    tmp_path: Path,
) -> None:
    output = tmp_path / "migrated"
    migrate_legacy_case(LEGACY_CASE, output)
    with pytest.raises(FileExistsError):
        migrate_legacy_case(LEGACY_CASE, output)


def test_migration_reports_targets_in_write_order_and_overwrites_explicitly(
    tmp_path: Path,
) -> None:
    output = tmp_path / "migrated"
    first = migrate_legacy_case(LEGACY_CASE, output)
    expected_targets = (
        output.resolve() / "run_config.yaml",
        output.resolve() / "particles.csv",
        output.resolve() / "boundaries.csv",
    )

    assert (first.config_path, first.particles_path, first.boundaries_path) == (
        expected_targets
    )
    with pytest.raises(FileExistsError) as exc_info:
        migrate_legacy_case(LEGACY_CASE, output)
    assert str(exc_info.value) == (
        "migration target(s) already exist; use overwrite=True to replace: "
        + ", ".join(str(path) for path in expected_targets)
    )

    expected_targets[1].write_text("not,a,canonical,particle,table\n", encoding="utf-8")
    second = migrate_legacy_case(LEGACY_CASE, output, overwrite=True)

    assert second == first
    assert "particle_id" in pd.read_csv(second.particles_path).columns


def test_comsol_migration_writes_only_relocated_config(tmp_path: Path) -> None:
    value = _legacy_value()
    manifest = tmp_path / "legacy" / "manifest.yaml"
    value["mode"] = "comsol_faithful"
    value["comsol"] = {"manifest": str(manifest)}
    output = tmp_path / "nested" / "migrated"

    result = migrate_legacy_case(_write_legacy(tmp_path, value), output)
    config = load_run_config(result.config_path)

    assert result.config_path == output.resolve() / "run_config.yaml"
    assert result.particles_path is None
    assert result.boundaries_path is None
    assert config.case.adapter == "comsol"
    assert (
        config.inputs.comsol_manifest
        == Path(Path("..") / ".." / "legacy" / "manifest.yaml").as_posix()
    )
    assert sorted(path.name for path in output.iterdir()) == ["run_config.yaml"]


def test_migration_explicitly_drops_particle_sticking_override(tmp_path: Path) -> None:
    value = _legacy_value()
    particles = pd.read_csv(value["paths"]["particles_csv"])
    particles["p_stick"] = 0.75
    particles_path = tmp_path / "legacy_particles.csv"
    particles.to_csv(particles_path, index=False)
    value["paths"]["particles_csv"] = str(particles_path)
    config_path = _write_legacy(tmp_path, value)

    result = migrate_legacy_case(config_path, tmp_path / "out")

    assert "p_stick" not in pd.read_csv(result.particles_path).columns
    assert any(
        "dropped legacy particle sticking columns" in warning and "p_stick" in warning
        for warning in result.warnings
    )


def test_migration_canonicalizes_provider_drag_force_names_models_and_keys(
    tmp_path: Path,
) -> None:
    value = _legacy_value()
    value["run"]["coordinate_system"] = "Cartesian XY"
    value["providers"]["geometry"]["kind"] = "SYNTHETIC-BOX"
    value["providers"]["field"]["kind"] = "SHEAR"
    value["providers"]["field"]["mu_Pas"] = value["providers"]["field"].pop(
        "dynamic_viscosity_Pas"
    )
    value["gas"]["molecular_mass_amu"] = 28.97
    value["solver"]["forces"] = {
        "drag": {"enabled": "YES", "model": "CUNNINGHAM-STOKES"},
        "Electric Force": {"enabled": "yes", "model": "Q-E"},
        "Gravity-Buoyancy": {
            "enabled": "on",
            "model": "BODY-ACCELERATION",
            "body_acceleration": [0.0, -9.81],
            "buoyancy_enabled": "NO",
        },
    }

    result = migrate_legacy_case(_write_legacy(tmp_path, value), tmp_path / "out")
    config = load_run_config(result.config_path)

    assert config.case.coordinate_system == "cartesian_xy"
    assert config.inputs.geometry.kind == "box"
    assert config.inputs.field.kind == "linear_shear"
    assert config.inputs.field.parameters["dynamic_viscosity_Pas"] == 1.8e-5
    assert config.physics.force_model is not None
    assert config.physics.force_model.drag.model == "stokes_cunningham"
    assert config.physics.force_model.declared == frozenset(
        {"drag", "electric", "gravity"}
    )
    assert config.physics.force_model.electric.model == "particle_charge"
    assert config.physics.force_model.gravity.model == "constant_acceleration"
    assert config.physics.force_model.gravity.acceleration_mps2 == (0.0, -9.81)
    assert config.physics.force_model.gravity.buoyancy is False


def test_migration_canonicalizes_stochastic_model_and_temperature_source(
    tmp_path: Path,
) -> None:
    value = _legacy_value()
    value["solver"]["stochastic_motion"] = {
        "enabled": "YES",
        "update_stride": 1,
        "model": "LANGEVIN",
        "temperature_source": "FIELD-T-THEN-GAS",
        "random_seed": 29,
    }

    result = migrate_legacy_case(_write_legacy(tmp_path, value), tmp_path / "out")
    stochastic = load_run_config(result.config_path).physics.stochastic

    assert stochastic is not None
    assert stochastic.enabled is True
    assert stochastic.model == "underdamped_langevin"
    assert stochastic.temperature_source == "field_T_then_gas"
    assert stochastic.seed == 29


def test_migration_canonicalizes_oml_and_plasma_without_te_relaxation_defaults(
    tmp_path: Path,
) -> None:
    value = _legacy_value()
    value["solver"]["charge_model"] = {
        "enabled": "YES",
        "mode": "FINITE-RATE-FLUX-BALANCE",
        "background_type": "PLASMA",
        "electron_temperature_unit": "EV",
    }
    value["solver"]["plasma_background"] = _legacy_plasma_background()

    result = migrate_legacy_case(_write_legacy(tmp_path, value), tmp_path / "out")
    charge = load_run_config(result.config_path).physics.charge

    assert charge is not None
    assert charge.enabled is True
    assert charge.mode == "oml_linearized_relaxation"
    assert charge.parameters["background_source"] == "plasma_background"
    assert charge.parameters["electron_temperature_unit"] == "eV"
    assert "te_relaxation_alpha" not in charge.parameters
    assert "relaxation_time_s" not in charge.parameters
    assert charge.background["source"] == "saas_constant"
    assert charge.background["electron_density_m3"] == 2.0e15
    assert charge.background["pressure_Pa"] == 2.0
    assert charge.background["conductivity_Sm"] == 0.0
    assert not any("te_relaxation_alpha" in warning for warning in result.warnings)
    assert not any("relaxation_time_s" in warning for warning in result.warnings)


def test_migration_materializes_relaxation_defaults_only_for_te_mode(
    tmp_path: Path,
) -> None:
    value = _legacy_value()
    value["solver"]["charge_model"] = {
        "enabled": "YES",
        "mode": "TE-RELAXATION",
    }

    result = migrate_legacy_case(_write_legacy(tmp_path, value), tmp_path / "out")
    charge = load_run_config(result.config_path).physics.charge

    assert charge is not None
    assert charge.mode == "te_relaxation"
    assert charge.parameters["te_relaxation_alpha"] == 2.5
    assert charge.parameters["relaxation_time_s"] == 1.0e-6
    assert any(
        "te_relaxation_alpha" in warning and "2.5" in warning
        for warning in result.warnings
    )
    assert any(
        "relaxation_time_s" in warning and "1e-6" in warning
        for warning in result.warnings
    )


def test_migration_does_not_materialize_te_defaults_when_charge_is_disabled(
    tmp_path: Path,
) -> None:
    value = _legacy_value()
    value["solver"]["charge_model"] = {
        "enabled": False,
        "mode": "te_relaxation",
    }

    result = migrate_legacy_case(_write_legacy(tmp_path, value), tmp_path / "out")
    charge = load_run_config(result.config_path).physics.charge

    assert charge is not None
    assert "te_relaxation_alpha" not in charge.parameters
    assert "relaxation_time_s" not in charge.parameters
    assert not any("te_relaxation_alpha" in warning for warning in result.warnings)
    assert not any("relaxation_time_s" in warning for warning in result.warnings)


def test_migration_accepts_scalar_charge_and_materializes_legacy_defaults(
    tmp_path: Path,
) -> None:
    value = _legacy_value()
    value["solver"]["charge_model"] = True

    result = migrate_legacy_case(_write_legacy(tmp_path, value), tmp_path / "out")
    charge = load_run_config(result.config_path).physics.charge

    assert charge is not None
    assert charge.enabled is True
    assert charge.mode == "te_relaxation"
    assert charge.parameters["te_relaxation_alpha"] == 2.5
    assert charge.parameters["relaxation_time_s"] == 1.0e-6


def test_migration_infers_charge_source_from_explicit_plasma_background(
    tmp_path: Path,
) -> None:
    value = _legacy_value()
    value["solver"]["charge_model"] = {"enabled": False, "mode": "OML"}
    value["solver"]["plasma_background"] = _legacy_plasma_background()

    result = migrate_legacy_case(_write_legacy(tmp_path, value), tmp_path / "out")
    charge = load_run_config(result.config_path).physics.charge

    assert charge is not None
    assert charge.parameters["background_source"] == "plasma_background"
    assert (
        "solver.charge_model.background_source was absent; selected "
        "plasma_background because solver.plasma_background was explicit"
    ) in result.warnings


@pytest.mark.parametrize(
    ("charge_model", "background", "expected"),
    [
        (
            {"enabled": False, "stride": 2},
            None,
            "charge update stride cannot be migrated",
        ),
        (
            None,
            _legacy_plasma_background(),
            "requires an explicit solver.charge_model",
        ),
        (
            {"enabled": False, "mode": "OML", "background_source": "field"},
            _legacy_plasma_background(),
            "background_source conflicts",
        ),
        (
            {"enabled": False},
            {"source": "disabled", "ne_m3": 2.0e15},
            "disabled legacy plasma background",
        ),
    ],
)
def test_migration_rejects_incompatible_charge_background_contracts(
    tmp_path: Path,
    charge_model: object,
    background: dict[str, object] | None,
    expected: str,
) -> None:
    value = _legacy_value()
    solver = value["solver"]
    assert isinstance(solver, dict)
    if charge_model is None:
        solver.pop("charge_model", None)
    else:
        solver["charge_model"] = charge_model
    if background is None:
        solver.pop("plasma_background", None)
    else:
        solver["plasma_background"] = background

    output = tmp_path / "out"
    with pytest.raises(ValueError, match=expected):
        migrate_legacy_case(_write_legacy(tmp_path, value), output)
    assert not output.exists()


@pytest.mark.parametrize("parameter", ["te_relaxation_alpha", "relaxation_time_s"])
def test_migration_rejects_te_relaxation_parameters_in_oml_mode(
    tmp_path: Path,
    parameter: str,
) -> None:
    value = _legacy_value()
    value["solver"]["charge_model"] = {
        "enabled": False,
        "mode": "OML",
        parameter: 1.0,
    }

    with pytest.raises(ValueError, match="cannot migrate Te-relaxation parameters"):
        migrate_legacy_case(_write_legacy(tmp_path, value), tmp_path / "out")


def test_migration_trims_legacy_paths_and_boundary_text_before_strict_load(
    tmp_path: Path,
) -> None:
    value = _legacy_npz_value()
    walls = pd.read_csv(value["paths"]["part_walls_csv"])
    expected_part_names = walls["part_name"].tolist()
    expected_material_names = walls["material_name"].tolist()
    walls["part_name"] = walls["part_name"].map(lambda item: f"  {item}  ")
    walls["material_name"] = walls["material_name"].map(lambda item: f"  {item}  ")
    walls["role"] = "  WALL  "
    walls["wall_law"] = walls["wall_law"].map(
        {
            "specular": "  BOUNCE  ",
            "mixed_specular_diffuse": "  MIXED-DIFFUSE-SPECULAR  ",
        }
    )
    walls_path = tmp_path / "legacy walls.csv"
    walls.to_csv(walls_path, index=False)
    value["paths"]["part_walls_csv"] = str(walls_path)
    for name in ("particles_csv", "part_walls_csv", "materials_csv"):
        value["paths"][name] = f"  {value['paths'][name]}  "
    for provider_name in ("geometry", "field"):
        provider = value["providers"][provider_name]
        provider["npz_path"] = f"  {provider['npz_path']}  "

    result = migrate_legacy_case(_write_legacy(tmp_path, value), tmp_path / "out")
    case = load_case(result.config_path)
    boundaries = pd.read_csv(result.boundaries_path)

    assert case.config.inputs.geometry.path is not None
    assert case.config.inputs.field.path is not None
    assert boundaries["part_name"].tolist() == expected_part_names
    assert boundaries["material_name"].tolist() == expected_material_names
    assert set(boundaries["role"]) == {"wall"}
    assert set(boundaries["wall_law"]) == {"specular", "mixed_specular_diffuse"}


@pytest.mark.parametrize("column", ["part_name", "material_name", "role", "wall_law"])
@pytest.mark.parametrize("blank_value", ["", "   "])
def test_migration_rejects_explicit_blank_boundary_text(
    tmp_path: Path,
    column: str,
    blank_value: str,
) -> None:
    value = _legacy_value()
    walls = pd.read_csv(value["paths"]["part_walls_csv"])
    walls.loc[0, column] = blank_value
    walls_path = tmp_path / "legacy_walls.csv"
    walls.to_csv(walls_path, index=False)
    value["paths"]["part_walls_csv"] = str(walls_path)

    with pytest.raises(ValueError, match=column):
        migrate_legacy_case(_write_legacy(tmp_path, value), tmp_path / "out")


@pytest.mark.parametrize(
    ("bad_case", "expected"),
    [
        ("provider_kind", "providers.geometry.kind"),
        ("stochastic_model", "stochastic model"),
        ("charge_model", "charge model"),
        ("plasma_source", "plasma background source"),
        ("force_name", "force name"),
        ("force_model", "electric force model"),
        ("force_parameter", "solver.forces.gravity"),
    ],
)
def test_migration_rejects_unknown_physics_instead_of_guessing(
    tmp_path: Path,
    bad_case: str,
    expected: str,
) -> None:
    value = _legacy_value()
    if bad_case == "provider_kind":
        value["providers"]["geometry"]["kind"] = "mystery_mesh"
    elif bad_case == "stochastic_model":
        value["solver"]["stochastic_motion"] = {
            "enabled": False,
            "stride": 1,
            "model": "random_walk",
        }
    elif bad_case == "charge_model":
        value["solver"]["charge_model"] = {"enabled": False, "mode": "magic_charge"}
    elif bad_case == "plasma_source":
        value["solver"]["charge_model"] = {"enabled": False, "mode": "te_relaxation"}
        value["solver"]["plasma_background"] = {"source": "mystery_background"}
    elif bad_case == "force_name":
        value["solver"]["forces"] = {"rocket": {"enabled": True}}
    elif bad_case == "force_model":
        value["solver"]["forces"] = {
            "electric": {"enabled": False, "model": "mystery_electric"}
        }
    elif bad_case == "force_parameter":
        value["solver"]["forces"] = {
            "gravity": {"enabled": False, "unknown_acceleration": [0.0, -9.81]}
        }
    output = tmp_path / "out"
    with pytest.raises(ValueError, match=expected):
        migrate_legacy_case(_write_legacy(tmp_path, value), output)

    assert not output.exists()
