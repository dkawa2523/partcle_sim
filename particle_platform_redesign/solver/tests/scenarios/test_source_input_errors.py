"""Source input errors are rejected before publishing an otherwise valid case."""

from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path

import numpy as np
import pytest
import yaml

from chamber_particles import open_result, simulate
from chamber_particles_tools.cli import main
from chamber_particles_tools.inputs import create_from_configuration, init_example


def _editable_source(tmp_path: Path) -> tuple[Path, dict, Path]:
    case = init_example("xy-ballistic", tmp_path / "example")
    recipe_path = case.case_path.parent / "recipe.yaml"
    recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    return recipe_path, recipe, case.case_path.parent / "particles.csv"


def _reject_recipe(
    path: Path, recipe: dict, output: Path, match: str, capsys: pytest.CaptureFixture[str]
) -> None:
    path.write_text(yaml.safe_dump(recipe), encoding="utf-8")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        with pytest.raises(ValueError, match=match):
            create_from_configuration(path, output)
        assert not output.exists()
        assert main(["create", str(path), "-o", str(output), "--json"]) == 2
    assert caught == []
    captured = capsys.readouterr()
    assert captured.out == ""
    error = json.loads(captured.err)
    assert error["status"] == "error" and error["error_type"] == "ValueError"
    assert not output.exists()
    assert not list(output.parent.glob(".case-publication-*"))


@pytest.mark.parametrize(
    ("name", "value", "match"),
    [
        ("columns", None, "columns"),
        ("units", None, "units"),
        ("filter_equals", None, "filter_equals"),
        ("filter_equals", {"missing": "initial"}, "missing"),
        ("velocity_columns", "v0_m_s", "velocity_columns"),
        ("components", None, "components"),
        ("metadata", None, "metadata"),
        ("inherit_from", {"table": "particles"}, "inherit_from"),
        ("basis_to_case", [[10**400, 0], [0, 1]], "basis_to_case"),
        ("basis_to_case", [[1e308, 0], [0, 1]], "basis_to_case"),
    ],
)
def test_nested_source_configuration_is_resolved_before_csv_consumers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], name: str, value: object, match: str
) -> None:
    path, recipe, _ = _editable_source(tmp_path)
    recipe["sources"][0]["configuration"][name] = value
    _reject_recipe(path, recipe, tmp_path / "rejected", match, capsys)


@pytest.mark.parametrize("kind", ["extra", "missing", "unselected-extra"])
def test_source_csv_row_width_is_checked_before_filtering(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], kind: str
) -> None:
    path, recipe, source_csv = _editable_source(tmp_path)
    with source_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if kind == "unselected-extra":
        rows[0].append("cohort")
        rows[1].append("initial")
        rows.append([*rows[1][:-1], "later", "extra"])
        recipe["sources"][0]["configuration"]["filter_equals"] = {"cohort": "initial"}
    elif kind == "extra":
        rows[1].append("extra")
    else:
        rows[1].pop()
    with source_csv.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)
    _reject_recipe(path, recipe, tmp_path / "rejected", "cells.*header", capsys)


@pytest.mark.parametrize(
    "token",
    ["not-an-integer", "NaN", "1.5", "9223372036854775808", "1e99999999999999999999999999999"],
)
def test_source_ids_reject_lexical_noninteger_and_out_of_range_values(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], token: str
) -> None:
    path, recipe, source_csv = _editable_source(tmp_path)
    with source_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    rows[1][rows[0].index("particle_id")] = token
    with source_csv.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)
    _reject_recipe(path, recipe, tmp_path / "rejected", "integer", capsys)


@pytest.mark.parametrize(
    ("mode", "unit"),
    [
        ("vector", "km/s"),
        ("momentum", "kg*m/s"),
        ("speed_direction", "km/s"),
        ("energy_direction", "J"),
    ],
)
def test_unrepresentable_converted_velocities_are_structured_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], mode: str, unit: str
) -> None:
    path, recipe, source_csv = _editable_source(tmp_path)
    with source_csv.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    config = recipe["sources"][0]["configuration"]
    config.update(velocity_mode=mode, velocity_unit=unit)
    if mode in ("vector", "momentum"):
        row[config["velocity_columns"][0]] = "1e308"
    else:
        config.update(velocity_columns=["magnitude"], direction_columns=["dx", "dy"])
        row.update(magnitude="1e308", dx="1", dy="0")
    if mode == "energy_direction":
        # sqrt(2E/m) itself exceeds float64, rather than just an intermediate.
        row["mass_kg"] = "5e-324"
    with source_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    _reject_recipe(path, recipe, tmp_path / "rejected", "converted velocity_m_s", capsys)


@pytest.mark.parametrize(
    ("column", "expected", "tokens"),
    [
        ("particle_id", 9007199254740992, ("9007199254740992", "9007199254740993")),
        ("selector", 0.1, ("0.1", "0.10000000000000001")),
    ],
)
def test_numeric_source_filter_preserves_exact_population(
    tmp_path: Path, column: str, expected: int | float, tokens: tuple[str, str]
) -> None:
    path, recipe, source_csv = _editable_source(tmp_path)
    with source_csv.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    rows = [{**row, "particle_id": str(i), column: token} for i, token in enumerate(tokens, 1)]
    with source_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    recipe["sources"][0]["configuration"]["filter_equals"] = {column: expected}
    path.write_text(yaml.safe_dump(recipe), encoding="utf-8")
    case = create_from_configuration(path, tmp_path / "input")
    assert case.data.sources[0].particle_id.tolist() == [int(rows[0]["particle_id"])]


def test_filtered_joint_source_preserves_exact_id_and_allows_declared_unused_columns(
    tmp_path: Path,
) -> None:
    path, recipe, source_csv = _editable_source(tmp_path)
    with source_csv.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        row = next(reader)
    row.update(particle_id="9007199254740993", cohort="initial", note="supplied metadata")
    later = {**row, "particle_id": "not-an-integer", "mass_kg": "not-numeric", "cohort": "later"}
    with source_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerows([row, later])
    recipe["sources"][0]["configuration"]["filter_equals"] = {"cohort": "initial"}
    path.write_text(yaml.safe_dump(recipe), encoding="utf-8")
    case = create_from_configuration(path, tmp_path / "input")
    source = case.data.sources[0]
    assert source.particle_id.tolist() == [9007199254740993]
    np.testing.assert_array_equal(source.mass_kg, [float(row["mass_kg"])])
    np.testing.assert_array_equal(source.model_weight, [float(row["model_weight"])])
    simulate(case, tmp_path / "result")
    result = open_result(tmp_path / "result")
    assert result.read_final().particle_id.tolist() == [9007199254740993]
    assert result.read_failure_events().particle_id.size == 0
    np.testing.assert_allclose(result.read_final().position_m, [[0.01, 0.005]], rtol=0, atol=1e-14)
    np.testing.assert_allclose(result.read_boundary_events().time_s, [0.015], rtol=0, atol=1e-13)
