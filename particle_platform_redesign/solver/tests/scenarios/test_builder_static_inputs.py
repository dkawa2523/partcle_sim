"""The static field producer resolves thermal input before check or build."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from chamber_particles import check_case, open_result, simulate
from chamber_particles.case_format import read, write
from chamber_particles_tools.cli import main
from chamber_particles_tools.electrostatic_builder.workflow import (
    build_from_configuration,
    check_from_configuration,
)
from chamber_particles_tools.inputs import create_from_configuration, init_example


def _configuration(tmp_path: Path, temporal_field: str | None = None) -> Path:
    case = init_example("reduced-electrostatic", tmp_path / "example")
    directory = case.case_path.parent
    configuration = yaml.safe_load((directory / "builder.yaml").read_text(encoding="utf-8"))
    thermal = read(directory / "thermal.h5")
    if temporal_field is not None:
        fields = tuple(
            replace(
                field,
                values=np.ascontiguousarray(np.stack((field.values, field.values * 2.0))),
                time_s=np.array([0.0, 1.0], dtype="<f8"),
            )
            if field.name == temporal_field
            else field
            for field in thermal.fields
        )
        thermal = replace(thermal, fields=fields)
    source = tmp_path / "thermal.h5"
    write(source, thermal)
    configuration["input"]["data_path"] = str(source)
    for condition in configuration["boundaries"]:
        condition["potential"]["value_V"] = 0.0
    path = tmp_path / "builder.yaml"
    path.write_text(yaml.safe_dump(configuration), encoding="utf-8")
    return path


@pytest.mark.parametrize("field_name", ["gas_velocity", "gas_temperature"])
def test_time_snapshot_thermal_fields_fail_at_public_check_and_build(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], field_name: str
) -> None:
    configuration = _configuration(tmp_path, field_name)
    output = tmp_path / "electric"
    with pytest.raises(ValueError, match=rf"{field_name}.*must be static"):
        check_from_configuration(configuration)
    with pytest.raises(ValueError, match=rf"{field_name}.*must be static"):
        build_from_configuration(configuration, output)
    assert not output.exists()
    for operation in ("check", "build"):
        arguments = ["inputs", "electrostatic", operation, str(configuration)]
        if operation == "build":
            arguments.append(str(output))
        assert main([*arguments, "--json"]) == 2
        captured = capsys.readouterr()
        assert captured.out == ""
        error = json.loads(captured.err)
        assert error["error_type"] == "ValueError"
        assert field_name in error["message"] and "must be static" in error["message"]
        assert not output.exists()


def test_static_builder_retains_temperature_and_supports_particle_case(tmp_path: Path) -> None:
    configuration = _configuration(tmp_path)
    output = tmp_path / "electric"
    checked = check_from_configuration(configuration)
    report = build_from_configuration(configuration, output)
    thermal = read(tmp_path / "thermal.h5")
    built = read(output / "bundle.h5")
    inputs = {field.name: field for field in thermal.fields}
    fields = {field.name: field for field in built.fields}
    assert checked["input_content_hash"] == report["input_content_hash"]
    np.testing.assert_array_equal(fields["gas_velocity"].values, inputs["gas_velocity"].values)
    np.testing.assert_array_equal(
        fields["positive_ion_temperature"].values, inputs["gas_temperature"].values
    )
    np.testing.assert_array_equal(fields["electric_potential"].values, 0.0)
    np.testing.assert_array_equal(fields["electric_field"].values, 0.0)
    np.testing.assert_array_equal(
        fields["electron_number_density"].values, fields["positive_ion_number_density"].values
    )
    recipe_path = tmp_path / "example/recipe.yaml"
    recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    recipe["input_bundle"] = str(output / "bundle.h5")
    recipe_path.write_text(yaml.safe_dump(recipe), encoding="utf-8")
    case = create_from_configuration(recipe_path, tmp_path / "case")
    check_case(case)
    simulate(case, tmp_path / "result")
    result = open_result(tmp_path / "result")
    assert result.read_final().particle_id.tolist() == [1]
    assert result.read_failure_events().particle_id.size == 0
