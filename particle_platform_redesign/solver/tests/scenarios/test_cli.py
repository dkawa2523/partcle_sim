"""The CLI stays a JSON-emitting shell around the public Python API."""

from __future__ import annotations

import json
import os
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


def test_unrepresentable_numeric_input_is_json_case_error(tmp_path: Path) -> None:
    paths = materialize_microcase("C01", tmp_path / "case")
    document = paths.case_path.read_text(encoding="utf-8")
    paths.case_path.write_text(
        document.replace("start_s: 0.0", f"start_s: {10**400}"), encoding="utf-8"
    )
    output = tmp_path / "result"
    result = _run_cli("run", paths.case_path, "-o", output)
    assert result.returncode == 2
    assert result.stdout == ""
    payload = json.loads(result.stderr)
    assert payload["error_type"] == "CaseError"
    assert "time.start_s must be finite" in payload["message"]
    assert not output.exists()


def test_unicode_case_name_and_error_survive_ascii_output_pipe(tmp_path: Path) -> None:
    paths = materialize_microcase("C01", tmp_path / "case")
    case_name = "粒子🧪"
    paths.case_path.write_text(
        paths.case_path.read_text(encoding="utf-8").replace("name: C01", f"name: {case_name}"),
        encoding="utf-8",
    )
    environment = {**os.environ, "PYTHONIOENCODING": "ascii"}
    for path, expected_code in ((paths.case_path, 0), (tmp_path / "不存在🧪.yaml", 2)):
        result = subprocess.run(
            [sys.executable, "-m", "chamber_particles", "check", str(path)],
            env=environment,
            check=False,
            capture_output=True,
        )
        assert result.returncode == expected_code
        payload = json.loads((result.stdout or result.stderr).decode("ascii"))
        if expected_code == 0:
            assert result.stderr == b""
            assert payload["case_name"] == case_name
        else:
            assert result.stdout == b""
            assert payload["error_type"] == "CaseError"
            assert str(path) in payload["message"]


def _run_cli(command: str, *arguments: str | Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "chamber_particles", command, *(str(value) for value in arguments)],
        check=False,
        capture_output=True,
        text=True,
    )
