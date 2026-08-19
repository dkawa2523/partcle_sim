from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

import particle_tracer_unified as particle_tracer
import particle_tracer_unified.writer as result_writer
from particle_tracer_unified.artifacts import STANDARD_ARTIFACTS
from particle_tracer_unified.comsol_case.contracts import validate_gas
from particle_tracer_unified.io.canonical_tables import validate_boundaries_csv

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = REPO_ROOT / "examples" / "v02_minimal" / "run_config.yaml"
BOUNDARY_COLUMNS = (
    "wall_law",
    "part_name",
    "wall_stick_probability",
    "part_id",
    "material_name",
    "role",
    "wall_restitution",
    "material_id",
    "wall_diffuse_fraction",
    "wall_critical_sticking_velocity_mps",
    "metadata_json",
)


@pytest.fixture(scope="module")
def standard_result():
    return particle_tracer.simulate(particle_tracer.load_case(EXAMPLE))


def _boundary_row(part_id: int = 1) -> dict[str, object]:
    return {
        "part_id": part_id,
        "part_name": f"part_{part_id}",
        "role": "wall",
        "material_id": 10,
        "material_name": "steel",
        "wall_law": "specular",
        "wall_stick_probability": 0.25,
        "wall_restitution": 0.75,
        "wall_diffuse_fraction": 0.0,
        "wall_critical_sticking_velocity_mps": 0.0,
        "metadata_json": '{"emissivity": 0.8}',
    }


def _write_boundaries(
    path: Path,
    rows: list[dict[str, object]],
    *,
    columns: tuple[str, ...] = BOUNDARY_COLUMNS,
) -> Path:
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    return path


def test_boundary_validation_preserves_input_column_and_row_order(
    tmp_path: Path,
) -> None:
    rows = [_boundary_row(2), _boundary_row(1)]

    frame = validate_boundaries_csv(
        _write_boundaries(tmp_path / "boundaries.csv", rows)
    )

    assert tuple(frame.columns) == BOUNDARY_COLUMNS
    assert frame["part_id"].tolist() == [2, 1]
    assert frame["metadata_json"].tolist() == [
        '{"emissivity": 0.8}',
        '{"emissivity": 0.8}',
    ]


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            [
                {
                    **_boundary_row(0),
                    "role": "mystery",
                    "wall_stick_probability": 2.0,
                }
            ],
            "boundaries.part_id must be > 0",
        ),
        (
            [
                _boundary_row(1),
                {**_boundary_row(1), "material_id": -1, "role": "mystery"},
            ],
            "boundaries.part_id values must be unique",
        ),
        (
            [
                {
                    **_boundary_row(),
                    "material_id": -1,
                    "part_name": " part",
                    "role": "mystery",
                }
            ],
            "boundaries.material_id must be >= 0",
        ),
        (
            [
                {
                    **_boundary_row(),
                    "role": "mystery",
                    "wall_law": "teleport",
                    "wall_stick_probability": 2.0,
                }
            ],
            "boundaries.role has unsupported values: mystery",
        ),
        (
            [
                {
                    **_boundary_row(),
                    "wall_law": "teleport",
                    "wall_stick_probability": 2.0,
                }
            ],
            "boundaries.wall_law has unsupported values: teleport",
        ),
        (
            [
                {
                    **_boundary_row(),
                    "wall_stick_probability": 2.0,
                    "wall_diffuse_fraction": -1.0,
                    "wall_restitution": -1.0,
                }
            ],
            "boundaries.wall_stick_probability must be in [0, 1]",
        ),
    ],
)
def test_boundary_validation_keeps_error_priority(
    tmp_path: Path,
    rows: list[dict[str, object]],
    message: str,
) -> None:
    path = _write_boundaries(tmp_path / "boundaries.csv", rows)

    with pytest.raises(ValueError, match=re.escape(message)) as exc_info:
        validate_boundaries_csv(path)

    assert str(exc_info.value) == message


def test_gas_validation_keeps_drag_requirements_and_error_order() -> None:
    with pytest.raises(ValueError, match="unsupported drag law") as unsupported:
        validate_gas("unknown", {"temperature_K": None})
    assert str(unsupported.value) == "unsupported drag law: 'unknown'"

    with pytest.raises(ValueError, match="requires explicit gas values") as missing:
        validate_gas(
            "stokes_cunningham",
            {
                "density_kgm3": None,
                "dynamic_viscosity_Pas": None,
                "molecular_mass_amu": None,
                "temperature_K": None,
            },
        )
    assert str(missing.value) == (
        "drag law 'stokes_cunningham' requires explicit gas values: "
        "['temperature_K', 'dynamic_viscosity_Pas', 'density_kgm3', "
        "'molecular_mass_amu']"
    )

    with pytest.raises(ValueError, match="positive and finite") as nonpositive:
        validate_gas(
            "none",
            {"density_kgm3": -1.0, "temperature_K": float("nan")},
        )
    assert str(nonpositive.value) == (
        "gas values must be positive and finite: ['density_kgm3', 'temperature_K']"
    )


def test_gas_validation_preserves_mapping_order_and_optional_values() -> None:
    gas = validate_gas(
        "none",
        {
            "density_kgm3": 1,
            "unused_positive_value": 2.5,
            "temperature_K": None,
        },
    )

    assert list(gas) == ["density_kgm3", "unused_positive_value"]
    assert gas == {"density_kgm3": 1.0, "unused_positive_value": 2.5}


def test_writer_publishes_through_sibling_staging_into_existing_empty_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    standard_result,
) -> None:
    output = tmp_path / "result"
    output.mkdir()
    observed_staging: list[Path] = []
    original = result_writer._write_result_files

    def observe_staging(result, staging: Path) -> None:
        assert output.is_dir()
        assert not list(output.iterdir())
        assert staging.parent == output.parent
        assert staging.name.startswith(".result.staging-")
        observed_staging.append(staging)
        original(result, staging)

    monkeypatch.setattr(result_writer, "_write_result_files", observe_staging)

    manifest = particle_tracer.write_result(standard_result, output)

    assert len(observed_staging) == 1
    assert sorted(path.name for path in output.iterdir()) == sorted(STANDARD_ARTIFACTS)
    assert [record.path.name for record in manifest.records] == list(STANDARD_ARTIFACTS)
    assert not list(tmp_path.glob(".result.staging-*"))


@pytest.mark.parametrize("precreate_empty_output", [False, True])
def test_writer_detects_destination_race_and_cleans_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    standard_result,
    precreate_empty_output: bool,
) -> None:
    output = tmp_path / "raced-result"
    if precreate_empty_output:
        output.mkdir()
    original = result_writer._write_result_files

    def race_after_staging(result, staging: Path) -> None:
        original(result, staging)
        if precreate_empty_output:
            (output / "concurrent.txt").write_text("owner", encoding="utf-8")
        else:
            output.mkdir()

    monkeypatch.setattr(result_writer, "_write_result_files", race_after_staging)

    expected = (
        "output directory changed while result was staged"
        if precreate_empty_output
        else "immutable result output already exists"
    )
    with pytest.raises(FileExistsError, match=expected):
        particle_tracer.write_result(standard_result, output)

    assert output.is_dir()
    assert not list(tmp_path.glob(".raced-result.staging-*"))


def test_writer_rejects_incomplete_staging_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    standard_result,
) -> None:
    output = tmp_path / "incomplete-result"
    original = result_writer._write_result_files

    def omit_summary(result, staging: Path) -> None:
        original(result, staging)
        (staging / "run_summary.json").unlink()

    monkeypatch.setattr(result_writer, "_write_result_files", omit_summary)

    with pytest.raises(RuntimeError) as exc_info:
        particle_tracer.write_result(standard_result, output)

    assert str(exc_info.value) == (
        "result writer did not create required artifact(s): run_summary.json"
    )
    assert not output.exists()
    assert not list(tmp_path.glob(".incomplete-result.staging-*"))
