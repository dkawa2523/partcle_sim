"""Installed clients create complete cases without native software or templates."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from chamber_particles import CaseError, check_case, open_result, simulate
from chamber_particles.case_format import RealizedSurfaceSource, RealizedTableSource, read, write
from chamber_particles_tools.csv_import import adapt_from_configuration
from chamber_particles_tools.examples import EXAMPLE_NAMES
from chamber_particles_tools.field_generation import record_generation
from chamber_particles_tools.inputs import create_case, create_from_configuration, init_example
from chamber_particles_tools.source_builder import read_csv_source


def test_recipe_binds_the_same_source_bytes_that_it_parses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    example = init_example("xy-ballistic", tmp_path / "example")
    csv_path = example.case_path.parent / "particles.csv"
    original = Path.read_bytes
    snapshot = original(csv_path)

    def replace_after_read(path: Path) -> bytes:
        raw = original(path)
        if path == csv_path:
            path.write_bytes(b"no longer the original source\n")
        return raw

    monkeypatch.setattr(Path, "read_bytes", replace_after_read)
    rebuilt = create_from_configuration(
        example.case_path.parent / "recipe.yaml", tmp_path / "rebuilt"
    )
    np.testing.assert_array_equal(rebuilt.data.sources[0].particle_id, [1])
    metadata = json.loads(rebuilt.data.provenance_json)["producer_metadata"]
    assert metadata["sources"][0]["csv_sha256"] == "sha256:" + hashlib.sha256(snapshot).hexdigest()
    assert original(csv_path) != snapshot


def test_editable_charged_mesh_and_fields_rebuild_the_same_analytic_problem(tmp_path: Path) -> None:
    original = init_example("rz-charged", tmp_path / "editable")
    inputs = original.case_path.parent
    adapt_from_configuration(inputs / "field-mapping.yaml", inputs / "background")
    rebuilt = create_from_configuration(inputs / "field-recipe.yaml", tmp_path / "rebuilt")
    np.testing.assert_array_equal(rebuilt.data.geometry.nodes_m, original.data.geometry.nodes_m)
    np.testing.assert_array_equal(rebuilt.data.geometry.tri3, original.data.geometry.tri3)
    np.testing.assert_array_equal(
        rebuilt.data.geometry.boundary.boundary_id, original.data.geometry.boundary.boundary_id
    )
    np.testing.assert_array_equal(rebuilt.data.fields[0].values, original.data.fields[0].values)
    np.testing.assert_array_equal(rebuilt.data.fields[0].values, [[0, 1000]] * 4)
    simulate(rebuilt, tmp_path / "result")
    result = open_result(tmp_path / "result")
    source = rebuilt.data.sources[0]
    assert isinstance(source, RealizedTableSource)
    final = result.read_final()
    time = (
        np.max(rebuilt.data.geometry.nodes_m[:, 0]) - source.position_m[0, 0]
    ) / source.velocity_m_s[0, 0]
    acceleration = 1.602176634e-19 * source.charge_number[0] * 1000 / source.mass_kg[0]
    expected = source.position_m[0] + source.velocity_m_s[0] * time
    expected[1] += 0.5 * acceleration * time**2
    np.testing.assert_allclose(final.position_m[0], expected, rtol=0, atol=1e-13)
    np.testing.assert_allclose(result.read_boundary_events().time_s, [time], rtol=0, atol=1e-13)
    assert result.read_failure_events().particle_id.size == 0


def test_recipe_retains_declared_generation_and_rejects_changed_source_mass(tmp_path: Path) -> None:
    example = init_example("rz-charged", tmp_path / "editable")
    electric = example.data.fields[0]
    mass = float(example.data.sources[0].mass_kg[0])
    elementary_charge = 1.602176634e-19
    derived = replace(
        electric,
        name="acceleration_per_charge",
        values=electric.values * elementary_charge / mass,
        unit="m/s^2",
    )
    bundle = replace(example.data, fields=(*example.data.fields, derived))
    record = record_generation(
        bundle,
        primitive_names=(electric.name,),
        parameter_document={"elementary_charge_C": elementary_charge},
        particle_properties={"mass_kg": mass},
        output_names=(derived.name,),
        producer_revision="analytic_acceleration_per_charge_v1",
    )
    provenance = json.loads(bundle.provenance_json)
    provenance["derived_field_generations"] = [record]
    generated = tmp_path / "generated.h5"
    write(generated, replace(bundle, provenance_json=json.dumps(provenance)))
    recipe_path = example.case_path.parent / "recipe.yaml"
    recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    recipe["input_bundle"] = str(generated)
    recipe_path.write_text(yaml.safe_dump(recipe, sort_keys=False), encoding="utf-8")
    rebuilt = create_from_configuration(recipe_path, tmp_path / "rebuilt")
    assert json.loads(rebuilt.data.provenance_json)["derived_field_generations"] == [record]
    source_csv = recipe_path.parent / "particles.csv"
    with source_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["mass_kg"] = str(2 * mass)
    with source_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    destination = tmp_path / "stale"
    with pytest.raises(ValueError, match="homogeneous particle property 'mass_kg' changed"):
        create_from_configuration(recipe_path, destination)
    assert not destination.exists()


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
    surface_case = create_case(replace(empty, sources=(surface,)), settings, tmp_path / "surface")
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
        create_case(example.data, settings, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".case-publication-*"))


def test_supporting_input_failure_does_not_publish_case_and_can_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    example = init_example("xy-ballistic", tmp_path / "example")
    recipe = yaml.safe_load((example.case_path.parent / "recipe.yaml").read_text(encoding="utf-8"))
    output = tmp_path / "complete-input"
    original = Path.open

    def unavailable(path: Path, *args: object, **kwargs: object):
        if path.name == "editable.csv":
            raise OSError("editable input unavailable")
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", unavailable)
        with pytest.raises(OSError, match="editable input unavailable"):
            create_case(
                example.data,
                recipe["settings"],
                output,
                supporting_files={"editable.csv": b"id\n1\n"},
            )
    assert not output.exists()
    case = create_case(
        example.data, recipe["settings"], output, supporting_files={"editable.csv": b"id\n1\n"}
    )
    assert (case.case_path.parent / "editable.csv").read_bytes() == b"id\n1\n"


def test_recipe_declares_complete_table_set_and_rejects_implicit_overlay(tmp_path: Path) -> None:
    example = init_example("xy-ballistic", tmp_path / "example")
    source = example.data.sources[0]
    other = replace(source, name="other", particle_id=np.asarray([99], dtype="<i8"))
    path = tmp_path / "available.h5"
    write(path, replace(example.data, sources=(source, other)))
    recipe = yaml.safe_load((example.case_path.parent / "recipe.yaml").read_text(encoding="utf-8"))
    recipe.update(input_bundle="available.h5", sources=[{"existing_table": "particles"}])
    recipe_path = tmp_path / "selection.yaml"
    recipe_path.write_text(yaml.safe_dump(recipe), encoding="utf-8")
    selected = create_from_configuration(recipe_path, tmp_path / "selected")
    assert [source.name for source in selected.data.sources] == ["particles"]
    recipe["sources"] = [{"existing_table": "missing"}]
    recipe_path.write_text(yaml.safe_dump(recipe), encoding="utf-8")
    with pytest.raises(ValueError, match="no source table"):
        create_from_configuration(recipe_path, tmp_path / "missing")
    recipe["format_version"] = 1
    recipe_path.write_text(yaml.safe_dump(recipe), encoding="utf-8")
    with pytest.raises(ValueError, match="format_version 2"):
        create_from_configuration(recipe_path, tmp_path / "old")


def test_recipe_preserves_joint_rows_across_multiple_csv_sources(tmp_path: Path) -> None:
    example = init_example("xy-ballistic", tmp_path / "example")
    recipe = yaml.safe_load((example.case_path.parent / "recipe.yaml").read_text(encoding="utf-8"))
    recipe["input_bundle"] = str(example.data_path)
    with (example.case_path.parent / "particles.csv").open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    rows = [dict(row), dict(row)]
    rows[0].update(particle_id="3", model_weight="2.5", mass_kg="2e-15")
    rows[1].update(particle_id="8", release_time_s="0.001", model_weight="4", mass_kg="3e-15")
    csv_path = tmp_path / "joint.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(rows)
    recipe["sources"][0]["csv"] = str(csv_path)
    second = yaml.safe_load(yaml.safe_dump(recipe["sources"][0]))
    second["configuration"]["table"] = "other"
    second_path = tmp_path / "other.csv"
    with second_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(
            [{**item, "particle_id": str(int(item["particle_id"]) * 10)} for item in rows]
        )
    second["csv"] = str(second_path)
    recipe["sources"].append(second)
    recipe["settings"]["sources"].append({"name": "other", "type": "table", "table": "other"})
    recipe_path = tmp_path / "joint.yaml"
    recipe_path.write_text(yaml.safe_dump(recipe), encoding="utf-8")
    case = create_from_configuration(recipe_path, tmp_path / "joint-case")
    ids_by_table = {"particles": [3, 8], "other": [30, 80]}
    for source in case.data.sources:
        ids = ids_by_table[source.name]
        np.testing.assert_array_equal(source.particle_id, ids)
        np.testing.assert_array_equal(source.mass_kg, [2e-15, 3e-15])
        np.testing.assert_array_equal(source.model_weight, [2.5, 4])
    simulate(case, tmp_path / "joint-result")
    assert open_result(tmp_path / "joint-result").read_final().particle_id.tolist() == [
        3,
        8,
        30,
        80,
    ]
