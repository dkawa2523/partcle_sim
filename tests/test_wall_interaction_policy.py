"""Repeated same-wall contact is a declared policy, not a hidden default."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from test_comsol_mesh_native_field import (
    _write_boundaries,
    _write_mesh,
    _write_release,
    _write_uniform_node_samples,
)

from particle_tracer_unified import load_case, simulate
from particle_tracer_unified.comsol_case.builder import write_case_files
from particle_tracer_unified.configuration import WallInteractionConfig


def test_wall_interaction_defaults_and_rejections() -> None:
    default = WallInteractionConfig.from_mapping({})
    assert default.contact_sliding is True
    assert default.max_hits_per_step == 5
    assert default.to_mapping() == {}

    explicit = WallInteractionConfig.from_mapping(
        {"contact_sliding": False, "max_hits_per_step": 32}
    )
    assert explicit.to_mapping() == {
        "contact_sliding": False,
        "max_hits_per_step": 32,
    }

    with pytest.raises(ValueError, match="must be a YAML boolean"):
        WallInteractionConfig.from_mapping({"contact_sliding": "no"})
    with pytest.raises(ValueError, match="must be at most 64"):
        WallInteractionConfig.from_mapping({"max_hits_per_step": 999})
    with pytest.raises(ValueError, match="unknown key"):
        WallInteractionConfig.from_mapping({"sliding": True})


def _pressed_into_wall_case(tmp_path: Path) -> Path:
    """Build a case whose flow presses the particle into the x = 0 wall.

    The particle reflects, drag turns it around within microseconds, and it
    reflects again -- the repeated same-wall contact the policy governs.
    """

    out = tmp_path / "case"
    write_case_files(
        _write_mesh(tmp_path / "mesh.mphtxt"),
        out,
        field_node_samples_path=_write_uniform_node_samples(
            tmp_path / "field_samples_nodes.csv", -1.0
        ),
        release_table_path=_write_release(tmp_path / "release.csv"),
        boundaries_path=_write_boundaries(tmp_path / "walls.csv"),
        diagnostic_grid_spacing_m=0.25,
        coordinate_scale_m_per_model_unit=1.0,
        coordinate_system="cartesian_xy",
        model_name="pressed",
        study="std1",
        dataset="dset1",
        solution="sol1",
        solution_number=1,
        vacuum_domain_ids=(1,),
        drag_law="stokes",
        gas_dynamic_viscosity_Pas=1.8e-5,
        gas_temperature_K=300.0,
        gas_density_kgm3=1.2,
        gas_molecular_mass_amu=39.948,
        solver_dt_s=0.05,
        solver_t_end_s=0.7,
    )
    return out / "run_config.yaml"


def _run_with_policy(config_path: Path, **policy: object) -> tuple[str, int]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["physics"]["wall_interaction"] = dict(policy)
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    result = simulate(load_case(config_path))
    return (
        str(result.state.terminal_state[0]),
        sum(int(count) for count in result.wall_summary.values()),
    )


def test_comsol_cases_are_built_without_contact_sliding(tmp_path: Path) -> None:
    """COMSOL has no contact model, so a COMSOL-built case must not use one."""

    config_path = _pressed_into_wall_case(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert payload["physics"]["wall_interaction"] == {"contact_sliding": False}
    assert load_case(config_path).config.physics.wall_interaction.contact_sliding is (
        False
    )


def test_contact_sliding_policy_changes_the_terminal_state(tmp_path: Path) -> None:
    config_path = _pressed_into_wall_case(tmp_path)

    # Enabled: the particle latches onto the wall and slides along it.  That is
    # a regularization of repeated contact, and it has no COMSOL counterpart.
    sliding_state, _ = _run_with_policy(config_path, contact_sliding=True)
    assert sliding_state == "contact_sliding"

    # Disabled: individual bounces keep being resolved until the declared
    # budget runs out, and exhausting it is a visible numerical stop rather
    # than a silent switch to a different model.
    bouncing_state, bouncing_hits = _run_with_policy(
        config_path, contact_sliding=False, max_hits_per_step=5
    )
    assert bouncing_state == "numerical_boundary_stopped"
    assert bouncing_hits == 5

    # The budget is what bounds the bounce count, so raising it resolves more.
    _, more_hits = _run_with_policy(
        config_path, contact_sliding=False, max_hits_per_step=32
    )
    assert more_hits == 32
