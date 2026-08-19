from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from particle_tracer_unified.cli import main as cli_main

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MINIMAL_CASE = REPOSITORY_ROOT / "examples" / "v02_minimal" / "run_config.yaml"


def _run_compare(*arguments: object) -> int:
    return cli_main(["compare", *(str(argument) for argument in arguments)])


def test_field_compare_public_cli_writes_sample_artifact(tmp_path: Path) -> None:
    points = tmp_path / "points.csv"
    output = tmp_path / "field_validation_error.csv"
    pd.DataFrame([{"point_id": 7, "time": 0.0, "x": 0.25, "y": 0.5}]).to_csv(
        points,
        index=False,
    )

    return_code = _run_compare(
        "field",
        "--config",
        MINIMAL_CASE,
        "--points",
        points,
        "--quantities",
        "ux",
        "uy",
        "--output",
        output,
    )

    assert return_code == 0
    result = pd.read_csv(output)
    assert result.columns.tolist() == [
        "point_id",
        "time",
        "x",
        "y",
        "z",
        "field",
        "component",
        "quantity",
        "python_value",
        "provider_kind",
        "provider_status",
        "provider_reason",
        "cell_id",
        "valid",
    ]
    assert result["component"].tolist() == ["ux", "uy"]
    np.testing.assert_allclose(result["python_value"], [0.5, 0.0])


def test_field_compare_public_cli_merges_comsol_reference(tmp_path: Path) -> None:
    points = tmp_path / "points.csv"
    reference = tmp_path / "comsol.csv"
    output = tmp_path / "field_validation_error.csv"
    pd.DataFrame([{"point_id": 7, "time": 0.0, "x": 0.25, "y": 0.5}]).to_csv(
        points,
        index=False,
    )
    pd.DataFrame(
        [{"point_id": 7, "field": "u", "component": "x", "comsol_value": 0.5}]
    ).to_csv(reference, index=False)

    return_code = _run_compare(
        "field",
        "--config",
        MINIMAL_CASE,
        "--points",
        points,
        "--comsol",
        reference,
        "--output",
        output,
    )

    assert return_code == 0
    result = pd.read_csv(output)
    assert result.loc[0, "quantity"] == "ux"
    assert result.loc[0, "python_value"] == pytest.approx(0.5)
    assert result.loc[0, "comsol_value"] == pytest.approx(0.5)
    assert result.loc[0, "abs_error"] == pytest.approx(0.0)


def test_field_compare_matches_repeated_points_by_reference_time(
    tmp_path: Path,
) -> None:
    points = tmp_path / "points.csv"
    reference = tmp_path / "comsol.csv"
    output = tmp_path / "field_validation_error.csv"
    pd.DataFrame(
        [
            {"point_id": 7, "time": 0.0, "x": 0.25, "y": 0.5},
            {"point_id": 7, "time": 1.0, "x": 0.25, "y": 0.5},
        ]
    ).to_csv(points, index=False)
    pd.DataFrame(
        [
            {
                "point_id": 7,
                "t": 0.0,
                "field": "u",
                "component": "x",
                "comsol_value": 0.5,
            },
            {
                "point_id": 7,
                "t": 1.0,
                "field": "u",
                "component": "x",
                "comsol_value": 0.5,
            },
        ]
    ).to_csv(reference, index=False)

    return_code = _run_compare(
        "field",
        "--config",
        MINIMAL_CASE,
        "--points",
        points,
        "--comsol",
        reference,
        "--output",
        output,
    )

    assert return_code == 0
    result = pd.read_csv(output)
    assert result.shape[0] == 2
    assert result["time_s"].tolist() == [0.0, 1.0]
    assert result["abs_error"].tolist() == [0.0, 0.0]


def test_field_compare_public_cli_rejects_missing_coordinates(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    points = tmp_path / "points.csv"
    output = tmp_path / "field_validation_error.csv"
    pd.DataFrame([{"point_id": 7, "time": 0.0, "x": 0.25}]).to_csv(
        points,
        index=False,
    )

    return_code = _run_compare(
        "field",
        "--config",
        MINIMAL_CASE,
        "--points",
        points,
        "--output",
        output,
    )

    assert return_code == 2
    assert "points CSV is missing coordinate axis 1" in capsys.readouterr().err
    assert not output.exists()


def test_trajectory_compare_public_cli_writes_error_artifact(
    tmp_path: Path,
) -> None:
    python_csv = tmp_path / "python.csv"
    comsol_csv = tmp_path / "comsol.csv"
    output = tmp_path / "trajectory_error.csv"
    pd.DataFrame(
        {
            "particle_id": [3, 3],
            "time_s": [0.0, 1.0],
            "x": [1.0, 2.0],
            "y": [2.0, 4.0],
        }
    ).to_csv(python_csv, index=False)
    pd.DataFrame(
        {
            "particle_id": [3, 3],
            "time_s": [0.0, 1.0],
            "x": [0.0, 2.0],
            "y": [2.0, 3.0],
        }
    ).to_csv(comsol_csv, index=False)

    return_code = _run_compare(
        "trajectory",
        "--python",
        python_csv,
        "--comsol",
        comsol_csv,
        "--output",
        output,
    )

    assert return_code == 0
    result = pd.read_csv(output)
    assert result.columns.tolist() == [
        "particle_id",
        "time_s",
        "x_python",
        "y_python",
        "x_comsol",
        "y_comsol",
        "_merge",
        "dx",
        "dy",
        "position_error",
    ]
    np.testing.assert_allclose(result["position_error"], [1.0, 1.0])


def test_trajectory_compare_normalizes_time_aliases(tmp_path: Path) -> None:
    python_csv = tmp_path / "python.csv"
    comsol_csv = tmp_path / "comsol.csv"
    output = tmp_path / "trajectory_error.csv"
    pd.DataFrame({"particle_id": [1, 1], "time_s": [0.0, 1.0], "x": [1.0, 2.0]}).to_csv(
        python_csv, index=False
    )
    pd.DataFrame({"particle_id": [1, 1], "t": [0.0, 1.0], "x": [1.0, 2.0]}).to_csv(
        comsol_csv, index=False
    )

    return_code = _run_compare(
        "trajectory",
        "--python",
        python_csv,
        "--comsol",
        comsol_csv,
        "--output",
        output,
    )

    assert return_code == 0
    result = pd.read_csv(output)
    assert result.columns.tolist() == [
        "particle_id",
        "time_s",
        "x_python",
        "x_comsol",
        "_merge",
        "dx",
        "position_error",
    ]
    assert result["_merge"].tolist() == ["both", "both"]


@pytest.mark.parametrize(
    ("python_rows", "comsol_rows", "expected_error"),
    [
        (
            [{"particle_id": 1, "x": 0.0}],
            [{"particle_id": 1, "x": 0.0}],
            "must contain one of time_s/time/t or sample_index",
        ),
        (
            [
                {"particle_id": 1, "time_s": 0.0, "x": 0.0},
                {"particle_id": 1, "time_s": 0.0, "x": 1.0},
            ],
            [{"particle_id": 1, "time_s": 0.0, "x": 0.0}],
            "duplicate trajectory key",
        ),
        (
            [{"particle_id": 1, "time_s": 0.0, "x": 0.0}],
            [{"particle_id": 1, "time_s": 1.0, "x": 0.0}],
            "trajectory keys do not match",
        ),
        (
            [{"particle_id": 1, "time_s": 0.0, "x": np.nan}],
            [{"particle_id": 1, "time_s": 0.0, "x": 0.0}],
            "contains non-finite x",
        ),
        (
            [{"particle_id": 1, "time_s": 0.0, "time": 0.0, "x": 0.0}],
            [{"particle_id": 1, "time_s": 0.0, "x": 0.0}],
            "contains multiple time columns",
        ),
        (
            [{"particle_id": 1, "time_s": 0.0, "x": 0.0}],
            [{"particle_id": 1, "sample_index": 0, "x": 0.0}],
            "must use the same sample key",
        ),
        (
            [{"particle_id": 1, "time_s": 0.0, "x": 0.0}],
            [{"particle_id": 1, "time_s": 0.0, "y": 0.0}],
            "must share at least one x/y/z column",
        ),
        (
            [{"particle_id": 1, "time_s": "not-a-time", "x": 0.0}],
            [{"particle_id": 1, "time_s": 0.0, "x": 0.0}],
            "contains non-numeric time_s",
        ),
        (
            [{"particle_id": 1, "time_s": 0.0, "x": 1.0e308}],
            [{"particle_id": 1, "time_s": 0.0, "x": -1.0e308}],
            "produced non-finite position_error",
        ),
    ],
)
def test_trajectory_compare_rejects_unsafe_alignment(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    python_rows: list[dict[str, object]],
    comsol_rows: list[dict[str, object]],
    expected_error: str,
) -> None:
    python_csv = tmp_path / "python.csv"
    comsol_csv = tmp_path / "comsol.csv"
    output = tmp_path / "trajectory_error.csv"
    pd.DataFrame(python_rows).to_csv(python_csv, index=False)
    pd.DataFrame(comsol_rows).to_csv(comsol_csv, index=False)

    return_code = _run_compare(
        "trajectory",
        "--python",
        python_csv,
        "--comsol",
        comsol_csv,
        "--output",
        output,
    )

    assert return_code == 2
    assert expected_error in capsys.readouterr().err
    assert not output.exists()


def test_trajectory_compare_accepts_unique_sample_indices(tmp_path: Path) -> None:
    python_csv = tmp_path / "python.csv"
    comsol_csv = tmp_path / "comsol.csv"
    output = tmp_path / "trajectory_error.csv"
    rows = [
        {"particle_id": 2, "sample_index": 0, "x": 1.0},
        {"particle_id": 2, "sample_index": 1, "x": 2.0},
    ]
    pd.DataFrame(rows).to_csv(python_csv, index=False)
    pd.DataFrame(rows).to_csv(comsol_csv, index=False)

    return_code = _run_compare(
        "trajectory",
        "--python",
        python_csv,
        "--comsol",
        comsol_csv,
        "--output",
        output,
    )

    assert return_code == 0
    assert pd.read_csv(output)["position_error"].tolist() == [0.0, 0.0]


@pytest.mark.parametrize(
    ("python_input", "expected_error"),
    [
        ("missing_id.csv", "--python trajectory CSV must contain particle_id"),
        ("trajectory.npy", "expects a long-form trajectory CSV"),
    ],
)
def test_trajectory_compare_public_cli_rejects_invalid_python_artifact(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    python_input: str,
    expected_error: str,
) -> None:
    python_path = tmp_path / python_input
    comsol_csv = tmp_path / "comsol.csv"
    output = tmp_path / "trajectory_error.csv"
    if python_path.suffix == ".npy":
        np.save(python_path, np.zeros((1, 1, 2), dtype=np.float64))
    else:
        pd.DataFrame({"time_s": [0.0], "x": [0.0], "y": [0.0]}).to_csv(
            python_path,
            index=False,
        )
    pd.DataFrame({"particle_id": [1], "time_s": [0.0], "x": [0.0], "y": [0.0]}).to_csv(
        comsol_csv, index=False
    )

    return_code = _run_compare(
        "trajectory",
        "--python",
        python_path,
        "--comsol",
        comsol_csv,
        "--output",
        output,
    )

    assert return_code == 2
    assert expected_error in capsys.readouterr().err
    assert not output.exists()
