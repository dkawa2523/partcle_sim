"""Analysis exposes only complete JSON and reports post-publication cleanup errors."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from chamber_particles import load_case, simulate
from chamber_particles_tools import analysis, cli
from tests.verification.microcases import materialize_microcase


@pytest.fixture(scope="module")
def completed_result(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("analysis-publication")
    case = materialize_microcase("C01", root / "case")
    result = root / "result"
    simulate(load_case(case.case_path), result)
    return result


def test_encoding_error_does_not_publish_and_can_retry(
    completed_result: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    destination = tmp_path / "summary.json"

    def invalid_summary(*args: object, **kwargs: object) -> dict[str, float]:
        return {"nonfinite": math.nan}

    with monkeypatch.context() as patch:
        patch.setattr(analysis, "summarize_result", invalid_summary)
        code = cli.main(["analyze", str(completed_result), "-o", str(destination), "--json"])
    captured = capfd.readouterr()
    assert code == 2
    assert captured.out == ""
    assert json.loads(captured.err)["error_type"] == "ValueError"
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []

    assert cli.main(["analyze", str(completed_result), "-o", str(destination), "--json"]) == 0
    published = json.loads(destination.read_text(encoding="utf-8"))
    assert published == analysis.summarize_result(completed_result)
    assert published["fate"] == [{"lifecycle": "active", "model_weight": 1.0, "particles": 1}]


def test_interrupted_write_does_not_publish_and_can_retry(
    completed_result: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    destination = tmp_path / "summary.json"
    original_open = Path.open

    def interrupted_open(path: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if path.parent == destination.parent and any(flag in mode for flag in "wax"):
            with original_open(path, mode, *args, **kwargs) as stream:
                stream.write('{"partial":')
            raise OSError("injected interrupted analysis write")
        return original_open(path, mode, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", interrupted_open)
        code = cli.main(["analyze", str(completed_result), "-o", str(destination), "--json"])
    captured = capfd.readouterr()
    assert code == 2
    assert captured.out == ""
    assert json.loads(captured.err)["error_type"] == "OSError"
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []

    assert cli.main(["analyze", str(completed_result), "-o", str(destination), "--json"]) == 0
    assert json.loads(destination.read_text(encoding="utf-8")) == analysis.summarize_result(
        completed_result
    )


@pytest.mark.parametrize("operation", ["fsync", "link"])
def test_sync_or_publication_error_leaves_no_output_and_can_retry(
    completed_result: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    operation: str,
) -> None:
    destination = tmp_path / "summary.json"

    def fail(*args: object, **kwargs: object) -> None:
        raise OSError(f"injected analysis {operation} error")

    with monkeypatch.context() as patch:
        patch.setattr(os, operation, fail)
        code = cli.main(["analyze", str(completed_result), "-o", str(destination), "--json"])
    captured = capfd.readouterr()
    assert code == 2
    assert captured.out == ""
    assert json.loads(captured.err)["error_type"] == "OSError"
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []

    assert cli.main(["analyze", str(completed_result), "-o", str(destination), "--json"]) == 0
    assert json.loads(destination.read_text(encoding="utf-8")) == analysis.summarize_result(
        completed_result
    )


def test_analysis_cli_publishes_the_complete_summary_without_changing_result(
    completed_result: Path, tmp_path: Path
) -> None:
    destination = tmp_path / "summary.json"
    original_manifest = (completed_result / "run.json").read_bytes()
    command = subprocess.run(
        [
            sys.executable,
            "-m",
            "chamber_particles",
            "analyze",
            str(completed_result),
            "-o",
            str(destination),
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert command.returncode == 0 and command.stderr == ""
    published = json.loads(destination.read_text(encoding="utf-8"))
    assert published == json.loads(command.stdout)
    assert published["population"]["selected_particles"] == 1
    assert published["fate"] == [{"lifecycle": "active", "model_weight": 1.0, "particles": 1}]
    assert list(tmp_path.iterdir()) == [destination]
    assert (completed_result / "run.json").read_bytes() == original_manifest


@pytest.mark.parametrize("concurrent", [False, True])
def test_existing_destination_is_never_replaced(
    completed_result: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    concurrent: bool,
) -> None:
    destination = tmp_path / "summary.json"
    existing = b"existing owner's content\n"
    original_link = os.link

    def competing_publish(source: Path, target: Path) -> None:
        target.write_bytes(existing)
        original_link(source, target)

    if not concurrent:
        destination.write_bytes(existing)
    with monkeypatch.context() as patch:
        if concurrent:
            patch.setattr(os, "link", competing_publish)
        code = cli.main(["analyze", str(completed_result), "-o", str(destination), "--json"])
    captured = capfd.readouterr()
    assert code == 2
    assert captured.out == ""
    assert json.loads(captured.err)["error_type"] == "FileExistsError"
    assert destination.read_bytes() == existing
    assert list(tmp_path.iterdir()) == [destination]


def test_analysis_cannot_publish_inside_its_source_result(
    completed_result: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    destination = completed_result / "derived" / "summary.json"
    code = cli.main(["analyze", str(completed_result), "-o", str(destination), "--json"])
    captured = capfd.readouterr()
    assert code == 2
    assert captured.out == ""
    assert json.loads(captured.err)["error_type"] == "ValueError"
    assert not destination.parent.exists()


def test_cleanup_error_after_publication_reports_the_complete_retained_file(
    completed_result: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    destination = tmp_path / "summary.json"
    original_manifest = (completed_result / "run.json").read_bytes()
    original_unlink = Path.unlink

    def deny_cleanup(path: Path, *args: Any, **kwargs: Any) -> None:
        if path.parent == destination.parent and path != destination:
            raise PermissionError("injected cleanup denial")
        original_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", deny_cleanup)
        code = cli.main(["analyze", str(completed_result), "-o", str(destination), "--json"])
    captured = capfd.readouterr()
    assert code == 2 and captured.out == ""
    error = json.loads(captured.err)
    assert error["error_type"] == "OSError"
    assert "JSON was published" in error["message"]
    assert str(destination) in error["message"]
    assert "temporary cleanup failed" in error["message"]
    assert json.loads(destination.read_text(encoding="utf-8")) == analysis.summarize_result(
        completed_result
    )
    assert (completed_result / "run.json").read_bytes() == original_manifest
    remaining = [path for path in tmp_path.iterdir() if path != destination]
    assert len(remaining) == 1
    assert remaining[0].read_bytes() == destination.read_bytes()
    remaining[0].unlink()
