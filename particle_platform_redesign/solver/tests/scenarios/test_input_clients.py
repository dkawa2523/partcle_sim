"""Installed clients create complete cases without native software or templates."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from chamber_particles import CaseError, check_case, open_result, simulate
from chamber_particles.case_format import RealizedSurfaceSource, RealizedTableSource, read
from chamber_particles_tools.examples import EXAMPLE_NAMES
from chamber_particles_tools.inputs import create_case, create_from_configuration, init_example
from chamber_particles_tools.source_builder import read_csv_source


@pytest.mark.parametrize("name", EXAMPLE_NAMES)
def test_packaged_examples_check_run_and_editable_recipe_rebuild(tmp_path: Path, name: str) -> None:
    case = init_example(name, tmp_path / "input")
    check_case(case)
    rebuilt = create_from_configuration(case.case_path.parent / "recipe.yaml", tmp_path / "rebuilt")
    assert isinstance(rebuilt.data.sources[0], RealizedTableSource)
    assert isinstance(case.data.sources[0], RealizedTableSource)
    np.testing.assert_array_equal(
        rebuilt.data.sources[0].position_m, case.data.sources[0].position_m
    )
    np.testing.assert_array_equal(rebuilt.data.sources[0].mass_kg, case.data.sources[0].mass_kg)
    output = tmp_path / "result"
    simulate(rebuilt, output)
    result = open_result(output)
    assert result.read_final().particle_id.tolist() == [1]
    assert result.read_failure_events().particle_id.size == 0
    if name == "xy-ballistic":
        final = result.read_final()
        np.testing.assert_allclose(final.position_m, [[0.01, 0.005]], rtol=0, atol=1e-14)
        np.testing.assert_allclose(
            result.read_boundary_events().time_s, [0.015], rtol=0, atol=1e-13
        )
        assert final.lifecycle.tolist() == [2]
    with pytest.raises(FileExistsError):
        init_example(name, tmp_path / "input")


def test_explicit_first_and_surface_csv_sources_need_no_existing_table(tmp_path: Path) -> None:
    example = init_example("xy-ballistic", tmp_path / "example")
    recipe = yaml.safe_load((example.case_path.parent / "recipe.yaml").read_text(encoding="utf-8"))
    config = recipe["sources"][0]["configuration"]
    empty = replace(read(example.data_path), sources=())
    source = read_csv_source(example.case_path.parent / "particles.csv", empty, config)
    assert source.particle_id.tolist() == [1]
    assert source.mass_kg.tolist() == [1e-15]
    config["kind"] = "surface"
    config["columns"].pop("position_m")
    config["units"].pop("position_m")
    config["columns"].update(facet_id="facet", facet_parameter="parameter")
    with (example.case_path.parent / "particles.csv").open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    row.update(facet=3, parameter=0.5, v0_m_s=0.5)
    path = tmp_path / "surface.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    surface = read_csv_source(path, empty, config)
    assert isinstance(surface, RealizedSurfaceSource)
    settings = recipe["settings"]
    settings["sources"][0]["type"] = "surface"
    surface_case = create_case(empty, settings, (surface,), tmp_path / "surface")
    check_case(surface_case)
    simulate(surface_case, tmp_path / "surface-result")
    result = open_result(tmp_path / "surface-result")
    assert result.read_failure_events().particle_id.size == 0
    assert result.read_boundary_events().particle_id.tolist() == [1]
    config["columns"].pop("mass_kg")
    config["units"].pop("mass_kg")
    with pytest.raises(ValueError, match="inheritance was not requested"):
        read_csv_source(path, example.data, config)


def test_rejected_case_never_publishes_partial_case(tmp_path: Path) -> None:
    example = init_example("xy-ballistic", tmp_path / "example")
    recipe = yaml.safe_load((example.case_path.parent / "recipe.yaml").read_text(encoding="utf-8"))
    settings = recipe["settings"]
    settings["physics"]["electric"] = {
        "model": "coulomb",
        "revision": "electric_coulomb_v1",
        "electric_field": "missing",
    }
    output = tmp_path / "rejected"
    with pytest.raises(CaseError):
        create_case(example.data, settings, example.data.sources, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".case-publication-*"))
