from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from particle_tracer_unified.comsol_case import (
    _case_contract,
    _contract_inputs,
    _raw_export_contract,
    contracts,
)
from particle_tracer_unified.comsol_case.profiles import BUILD_PROFILES
from particle_tracer_unified.integrity import sha256_file


def _write_inventory(path: Path, value: object) -> Path:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def test_force_inventory_preserves_entry_and_key_order(tmp_path: Path) -> None:
    path = _write_inventory(
        tmp_path / "forces.yaml",
        {
            "forces": [
                {
                    "enabled": True,
                    "solver_force": "electric",
                    "parameters": {"second": 2, "first": 1},
                },
                {"solver_force": "lift", "enabled": False},
            ]
        },
    )

    entries = _case_contract._load_force_inventory_entries(path)

    assert isinstance(entries, tuple)
    assert [entry["solver_force"] for entry in entries] == ["electric", "lift"]
    assert list(entries[0]) == ["enabled", "solver_force", "parameters"]
    parameters = entries[0]["parameters"]
    assert isinstance(parameters, dict)
    assert list(parameters) == ["second", "first"]


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (None, "force inventory YAML root must be a mapping with a forces list"),
        ([], "force inventory YAML root must be a mapping with a forces list"),
        ({}, "force inventory YAML forces must be a list"),
        ({"forces": {}}, "force inventory YAML forces must be a list"),
        (
            {"forces": [{"solver_force": "electric"}, 7]},
            "force inventory YAML forces[1] must be a mapping",
        ),
    ],
)
def test_force_inventory_rejects_missing_or_wrong_shapes(
    tmp_path: Path,
    value: object,
    message: str,
) -> None:
    path = _write_inventory(tmp_path / "forces.yaml", value)

    with pytest.raises(ValueError, match=re.escape(message)) as captured:
        _case_contract._load_force_inventory_entries(path)

    assert str(captured.value) == message


def test_force_inventory_reports_sorted_unknown_keys_before_forces_type(
    tmp_path: Path,
) -> None:
    path = _write_inventory(
        tmp_path / "forces.yaml",
        {"zeta": 1, "forces": "wrong", "alpha": 2},
    )
    message = "force inventory YAML has unknown keys: ['alpha', 'zeta']"

    with pytest.raises(ValueError, match=re.escape(message)) as captured:
        _case_contract._load_force_inventory_entries(path)

    assert str(captured.value) == message


def test_force_inventory_normalizes_read_and_yaml_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = Path("forces.yaml")

    def unreadable(_path: Path, *, encoding: str) -> str:
        assert encoding == "utf-8"
        raise OSError("unavailable")

    monkeypatch.setattr(Path, "read_text", unreadable)
    with pytest.raises(ValueError, match="cannot read force inventory") as captured:
        _case_contract._load_force_inventory_entries(path)
    assert str(captured.value) == "cannot read force inventory forces.yaml: unavailable"

    monkeypatch.setattr(Path, "read_text", lambda *_args, **_kwargs: "content")

    def invalid_yaml(_text: str) -> object:
        raise yaml.YAMLError("invalid syntax")

    monkeypatch.setattr(_case_contract.yaml, "safe_load", invalid_yaml)
    with pytest.raises(ValueError, match="invalid force inventory YAML") as captured:
        _case_contract._load_force_inventory_entries(path)
    assert str(captured.value) == (
        "invalid force inventory YAML forces.yaml: invalid syntax"
    )


def test_contract_facade_reexports_owner_objects_without_wrappers() -> None:
    expected = {
        "FIELD_STORAGE_MESH_NATIVE": _contract_inputs.FIELD_STORAGE_MESH_NATIVE,
        "FIELD_STORAGE_REGULAR_GRID": _contract_inputs.FIELD_STORAGE_REGULAR_GRID,
        "GeometryOnlyBuild": _contract_inputs.GeometryOnlyBuild,
        "RunnableBuild": _contract_inputs.RunnableBuild,
        "canonical_boundary_table": _contract_inputs.canonical_boundary_table,
        "canonical_release_table": _contract_inputs.canonical_release_table,
        "copy_explicit_input": _contract_inputs.copy_explicit_input,
        "load_json_mapping": _contract_inputs.load_json_mapping,
        "required_positive_float": _contract_inputs.required_positive_float,
        "resolve_force_inventory": _case_contract.resolve_force_inventory,
        "sha256": sha256_file,
        "validate_gas": _case_contract.validate_gas,
        "validate_raw_export": _raw_export_contract.validate_raw_export,
        "validate_runnable_inputs": _contract_inputs.validate_runnable_inputs,
        "write_case_contract": _case_contract.write_case_contract,
    }

    assert tuple(contracts.__all__) == tuple(expected)
    for name, owner in expected.items():
        assert getattr(contracts, name) is owner


def test_build_input_helpers_keep_missing_and_type_errors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"value.*positive and finite"):
        _contract_inputs.required_positive_float("not-a-number", context="value")

    with pytest.raises(ValueError, match="requires --field-bundle"):
        _contract_inputs.validate_runnable_inputs(
            geometry_only=False,
            field_bundle_path=None,
            release_table_path=None,
            boundaries_path=None,
            model_name=None,
            study=None,
            dataset=None,
            solution=None,
            solution_number=None,
            drag_law=None,
            solver_dt_s=None,
            solver_t_end_s=None,
        )

    missing = tmp_path / "missing.txt"
    with pytest.raises(FileNotFoundError) as captured:
        _contract_inputs.copy_explicit_input(missing, tmp_path / "copy.txt")
    assert captured.value.args == (missing.resolve(),)

    existing = tmp_path / "existing.txt"
    existing.write_text("unchanged", encoding="utf-8")
    _contract_inputs.copy_explicit_input(existing, existing)
    assert existing.read_text(encoding="utf-8") == "unchanged"

    assert _contract_inputs.load_json_mapping(tmp_path / "missing.json") == {}
    invalid_json = tmp_path / "list.json"
    invalid_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="expected a JSON object"):
        _contract_inputs.load_json_mapping(invalid_json)


def test_force_inventory_rejects_explicit_drag_before_model_parsing(
    tmp_path: Path,
) -> None:
    path = _write_inventory(
        tmp_path / "forces.yaml",
        {"forces": [{"solver_force": "drag", "enabled": True}]},
    )
    message = (
        "force inventory YAML must not declare drag; use --drag-law as its "
        "single source"
    )

    with pytest.raises(ValueError, match=re.escape(message)) as captured:
        _case_contract.resolve_force_inventory(
            drag_law="none",
            enabled_forces=(),
            force_inventory_path=path,
            coordinate_system="cartesian_xy",
        )

    assert str(captured.value) == message


def test_case_contract_rejects_missing_provenance_before_artifact_reads(
    tmp_path: Path,
) -> None:
    force_model = _case_contract.resolve_force_inventory(
        drag_law="none",
        enabled_forces=(),
        force_inventory_path=None,
        coordinate_system="cartesian_xy",
    )

    with pytest.raises(ValueError, match="COMSOL model provenance is missing") as error:
        _case_contract.write_case_contract(
            out_dir=tmp_path,
            geometry_npz=tmp_path / "missing-geometry.npz",
            field_npz=tmp_path / "missing-field.npz",
            particles_csv=tmp_path / "missing-particles.csv",
            boundaries_csv=tmp_path / "missing-boundaries.csv",
            coordinate_system="cartesian_xy",
            profile=BUILD_PROFILES["generic"],
            model_provenance={"name": "", "study": "", "dataset": "", "solution": ""},
            force_inventory=force_model,
            gas={},
            dt_s=0.1,
            t_end_s=1.0,
            output_mode="standard",
            trajectory_interval_steps=None,
            source_metadata={},
        )

    assert str(error.value) == (
        "COMSOL model provenance is missing: ['name', 'study', 'dataset', 'solution']"
    )


def test_raw_export_validation_reports_missing_quantity_unit_and_artifact(
    tmp_path: Path,
) -> None:
    profile = BUILD_PROFILES["generic"]
    expected_units = _raw_export_contract._profile_expression_units(profile)

    with pytest.raises(ValueError, match="missing required quantity 'uy'"):
        _raw_export_contract._validate_profile_quantities(
            {"ux": "u"},
            {"ux": "m/s"},
            profile,
            expected_units,
        )
    with pytest.raises(ValueError, match=r"expression_units.*quantity 'uy'"):
        _raw_export_contract._validate_profile_quantities(
            {"ux": "u", "uy": "v"},
            {"ux": "m/s"},
            profile,
            expected_units,
        )
    with pytest.raises(ValueError, match="must have identical keys"):
        _raw_export_contract._validate_expression_units(
            {"ux": "u"},
            {},
            expected_units,
        )
    with pytest.raises(ValueError, match=r"artifact is missing: .*mesh\.mphtxt"):
        _raw_export_contract._validated_artifact_hashes(
            tmp_path,
            {"mesh_sha256": "0" * 64},
        )
