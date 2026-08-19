from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from particle_tracer_unified.cli import main
from tools import _reference_compare_inputs as comparison_inputs
from tools import _reference_compare_metrics as comparison_metrics
from tools import _reference_compare_runs as comparison_runs
from tools import compare_against_reference as reference_compare

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = REPO_ROOT / "examples" / "v02_minimal" / "run_config.yaml"


@pytest.mark.parametrize("disable_jit", ["0", "1"], ids=("jit_on", "jit_off"))
def test_unified_reference_compare_runs_canonical_case_with_jit_modes(
    tmp_path: Path,
    disable_jit: str,
) -> None:
    output_root = tmp_path / f"compare-jit-{disable_jit}"
    environment = os.environ.copy()
    environment["NUMBA_DISABLE_JIT"] = disable_jit

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "particle_tracer_unified.cli",
            "compare",
            "reference",
            "--reference-config",
            str(EXAMPLE),
            "--run",
            f"same={EXAMPLE}",
            "--output-root",
            str(output_root),
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    comparison_dir = Path(summary["comparison_dir"])
    assert comparison_dir.is_dir()
    assert summary["runs"][0]["class_match_ratio_vs_reference"] == 1.0
    assert (comparison_dir / "comparison_summary.json").is_file()
    assert (comparison_dir / "reference" / "run_summary.json").is_file()
    assert (comparison_dir / "same" / "run_summary.json").is_file()
    assert not list(output_root.glob(".*.staging-*"))


def test_unified_reference_compare_cleans_partial_staging_on_preflight_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    value = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    value["inputs"]["particles"] = str((EXAMPLE.parent / "particles.csv").resolve())
    value["inputs"]["boundaries"] = str((EXAMPLE.parent / "boundaries.csv").resolve())
    value["physics"]["gas"].pop("density_kgm3")
    value["physics"]["forces"] = {"lift": {"enabled": True}}
    invalid_config = tmp_path / "invalid.yaml"
    invalid_config.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    output_root = tmp_path / "failed-comparison"

    exit_code = main(
        [
            "compare",
            "reference",
            "--reference-config",
            str(EXAMPLE),
            "--run",
            f"invalid={invalid_config}",
            "--output-root",
            str(output_root),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "preflight failed" in captured.err
    assert not list(output_root.glob("compare_*"))
    assert not list(output_root.glob(".*.staging-*"))


def test_reference_compare_rejects_removed_timeout_option(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "particle_tracer_unified.cli",
            "compare",
            "reference",
            "--reference-config",
            str(EXAMPLE),
            "--run",
            f"same={EXAMPLE}",
            "--output-root",
            str(tmp_path / "comparison"),
            "--per-run-timeout-s",
            "1",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 2
    assert "unrecognized arguments: --per-run-timeout-s 1" in completed.stderr


def test_reference_compare_config_materialization_and_helper_contracts(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source" / "run.yaml"
    source.parent.mkdir()
    source.write_text(
        yaml.safe_dump(
            {
                "inputs": {
                    "particles": "particles.csv",
                    "boundaries": "boundaries.csv",
                    "geometry": {"path": "geometry.npz"},
                    "field": {"path": "field.npz"},
                },
                "output": {"mode": "debug", "trajectory_interval_steps": 9},
            }
        ),
        encoding="utf-8",
    )
    debug_path = comparison_inputs.write_config_variant(
        source_config=source,
        output_config=tmp_path / "debug.yaml",
        override_t_end=0.5,
        artifact_mode="debug",
    )
    debug = comparison_inputs.load_yaml_mapping(debug_path)
    assert debug["time"]["t_end"] == 0.5
    assert Path(debug["inputs"]["particles"]).is_absolute()
    assert Path(debug["inputs"]["geometry"]["path"]).is_absolute()
    assert debug["output"]["trajectory_interval_steps"] == 9
    standard_path = comparison_inputs.write_config_variant(
        source_config=source,
        output_config=tmp_path / "standard.yaml",
        override_t_end=None,
        artifact_mode="standard",
    )
    standard = comparison_inputs.load_yaml_mapping(standard_path)
    assert "trajectory_interval_steps" not in standard["output"]
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("- not-a-mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML root"):
        comparison_inputs.load_yaml_mapping(invalid)

    empty = pd.DataFrame({"particle_id": [], "final_state": []})
    assert reference_compare.class_transition_summary(empty, empty) == {
        "compared_particles": 0,
        "mismatch_count": 0,
        "top_transitions": [],
    }
    with pytest.raises(ValueError, match="missing position columns"):
        comparison_metrics.final_position_array(
            pd.DataFrame({"x_m": [0.0]}),
            2,
        )
    assert (
        comparison_metrics.pair_delta(
            {"run": "base", "runtime_s": 0.0},
            {"run": "candidate", "runtime_s": 1.0},
        )["runtime_increase_ratio"]
        == 0.0
    )
    with pytest.raises(Exception, match="Expected NAME=path"):
        comparison_inputs.parse_named_run("invalid")
    with pytest.raises(ValueError, match="duplicate"):
        comparison_inputs.validate_run_specs([("same", Path("a")), ("same", Path("b"))])
    relocated = comparison_runs.relocate_value(
        {"path": str(tmp_path / "old" / "file")},
        tmp_path / "old",
        tmp_path / "new",
    )
    assert relocated["path"] == str(tmp_path / "new" / "file")


def test_reference_compare_geometry_summary_handles_no_shared_particles() -> None:
    columns = {
        "particle_id": pd.Series(dtype="int64"),
        "final_state": pd.Series(dtype="object"),
        "x_m": pd.Series(dtype="float64"),
        "y_m": pd.Series(dtype="float64"),
    }
    empty = pd.DataFrame(columns)
    runtime = SimpleNamespace(spatial_dim=2, geometry_provider=None)

    assert reference_compare.geometry_feature_delta_summary(
        empty,
        empty,
        runtime,
    ) == {"compared_particles": 0}
