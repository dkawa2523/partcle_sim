"""The CLI stays a JSON-emitting shell around the public Python API."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tests.verification.microcases import materialize_microcase


def test_check_run_and_inspect_use_the_public_workflow(tmp_path: Path) -> None:
    paths = materialize_microcase("C01", tmp_path / "case")
    output = tmp_path / "result"

    checked = _run_cli("check", paths.case_path)
    assert checked.returncode == 0
    assert checked.stderr == ""
    check_payload = json.loads(checked.stdout)
    assert check_payload["operation"] == "check"
    assert check_payload["status"] == "valid"
    assert check_payload["case_name"] == "C01"
    assert check_payload["data_coordinate_system"] == "cartesian_xy"
    assert check_payload["canonical_numeric_array_bytes"] > 0
    assert check_payload["memory_limit_bytes"] == 128 * 1024 * 1024

    run = _run_cli("run", paths.case_path, "-o", output)
    assert run.returncode == 0
    assert run.stderr == ""
    run_payload = json.loads(run.stdout)
    assert run_payload["operation"] == "run"
    assert run_payload["status"] == "complete"
    assert run_payload["particle_count"] == 1
    assert Path(run_payload["output_path"]) == output.resolve()

    inspected = _run_cli("inspect", output)
    assert inspected.returncode == 0
    assert inspected.stderr == ""
    inspect_payload = json.loads(inspected.stdout)
    assert inspect_payload["operation"] == "inspect"
    assert inspect_payload["status"] == "complete"
    assert inspect_payload["manifest"]["case_name"] == "C01"
    assert inspect_payload["manifest"]["counts"]["particles"] == 1


def test_public_cli_failure_is_json_and_exit_two(tmp_path: Path) -> None:
    missing_case = tmp_path / "missing.yaml"

    result = _run_cli("check", missing_case)

    assert result.returncode == 2
    assert result.stdout == ""
    payload = json.loads(result.stderr)
    assert payload["status"] == "error"
    assert payload["error_type"] == "CaseError"
    assert "cannot load case" in payload["message"]


def _run_cli(command: str, *arguments: str | Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "chamber_particles", command, *(str(value) for value in arguments)],
        check=False,
        capture_output=True,
        text=True,
    )
