from __future__ import annotations

import importlib
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
runner = importlib.import_module("quality_tools.runner")


def _completed(
    arguments: Sequence[str],
    *,
    returncode: int,
    stdout: str,
    stderr: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        arguments,
        returncode,
        stdout=stdout,
        stderr=stderr,
    )


def test_captured_machine_output_is_not_echoed(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def run_stub(
        arguments: Sequence[str], **_: Any
    ) -> subprocess.CompletedProcess[str]:
        return _completed(
            arguments,
            returncode=0,
            stdout='{"machine": "payload"}\n',
            stderr="collector detail\n",
        )

    monkeypatch.setattr(runner.subprocess, "run", run_stub)

    result = runner._run(["collector", "--json"], capture=True)

    captured = capsys.readouterr()
    assert captured.out == "+ collector --json\n"
    assert captured.err == ""
    assert result.stdout == '{"machine": "payload"}\n'
    assert result.stderr == "collector detail\n"


def test_failed_captured_command_reports_only_bounded_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = "\n".join(f"diagnostic {index}" for index in range(30))

    def run_stub(
        arguments: Sequence[str], **_: Any
    ) -> subprocess.CompletedProcess[str]:
        return _completed(
            arguments,
            returncode=2,
            stdout=output,
            stderr="final error",
        )

    monkeypatch.setattr(runner.subprocess, "run", run_stub)

    with pytest.raises(runner.GateFailure) as error:
        runner._run(["collector", "--json"], capture=True)

    message = str(error.value)
    assert "captured output truncated" in message
    assert "diagnostic 0" not in message
    assert "diagnostic 12" in message
    assert "diagnostic 29" in message
    assert "final error" in message
