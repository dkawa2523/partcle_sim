from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from quality_tools import _runner_tools, runner


def _completed(arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(arguments, 0, stdout="", stderr="")


def _install_common_session_stubs(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[str], list[list[str]]]:
    phases: list[str] = []
    commands: list[list[str]] = []
    monkeypatch.setattr(runner, "_load_baseline", lambda: {"baseline": True})
    monkeypatch.setattr(runner, "_tool", lambda name: name)
    monkeypatch.setattr(
        runner,
        "_run",
        lambda arguments, **kwargs: (
            commands.append(list(arguments)) or _completed(arguments)
        ),
    )
    monkeypatch.setattr(runner, "_collect_static", lambda: {"static": True})
    monkeypatch.setattr(
        runner, "_compare_static", lambda current, expected: phases.append("static")
    )
    monkeypatch.setattr(runner, "_run_pyrefly", lambda **kwargs: phases.append("types"))
    monkeypatch.setattr(
        runner,
        "_collect_tests",
        lambda: {"count": 1, "node_ids": ["test_a.py::test_a"]},
    )
    monkeypatch.setattr(
        runner, "_compare_tests", lambda current, expected: phases.append("tests")
    )
    return phases, commands


def test_fast_formats_only_changed_paths_without_updating_baseline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "changed.py"
    source.write_text("value=1\n", encoding="utf-8")
    phases, commands = _install_common_session_stubs(monkeypatch)
    monkeypatch.setattr(runner, "_changed_paths", lambda paths: [source])
    monkeypatch.setattr(
        runner,
        "_write_secret_baseline",
        lambda payload: pytest.fail("fast must not update a baseline"),
    )

    runner._quality_fast((str(source),))

    assert commands[0][:2] == ["ruff", "format"]
    assert commands[1][:3] == ["ruff", "check", "--fix"]
    assert commands[-1] == [sys.executable, "-m", "pytest", "-q"]
    assert phases == ["static", "types", "tests"]


def test_pr_preserves_static_type_architecture_test_coverage_security_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phases, _ = _install_common_session_stubs(monkeypatch)
    monkeypatch.setattr(
        runner, "_run_architecture", lambda: phases.append("architecture")
    )
    monkeypatch.setattr(
        runner,
        "_collect_coverage",
        lambda: phases.append("coverage") or {"covered": 9, "total": 10},
    )
    monkeypatch.setattr(
        runner,
        "_compare_coverage",
        lambda current, expected: phases.append("coverage gate"),
    )
    monkeypatch.setattr(runner, "_run_diff_coverage", lambda: phases.append("diff"))
    monkeypatch.setattr(runner, "_run_security", lambda: phases.append("security"))

    result = runner._quality_pr()

    assert result["coverage"] == {"covered": 9, "total": 10}
    assert phases == [
        "static",
        "types",
        "architecture",
        "tests",
        "coverage",
        "coverage gate",
        "diff",
        "security",
    ]


def test_baseline_is_the_only_session_that_writes_baseline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    baseline_path = tmp_path / "baseline.json"
    monkeypatch.setattr(runner, "BASELINE_PATH", baseline_path)
    monkeypatch.setattr(runner, "QUALITY_DIR", tmp_path)
    monkeypatch.setattr(runner, "PYREFLY_BASELINE_PATH", tmp_path / "pyrefly.json")
    monkeypatch.setattr(runner, "_relative", lambda path: Path(path).name)
    monkeypatch.setattr(runner, "_load_baseline", lambda: None)
    monkeypatch.setattr(runner, "_run_pyrefly", lambda **kwargs: None)
    monkeypatch.setattr(runner, "_run_architecture", lambda: None)
    monkeypatch.setattr(runner, "_run_code_and_dependency_security", lambda: None)
    monkeypatch.setattr(
        runner,
        "_collect_static",
        lambda: {"ruff": {}, "format": {}, "radon": {}, "vulture": {}},
    )
    monkeypatch.setattr(
        runner,
        "_collect_tests",
        lambda: {"count": 1, "node_ids": ["test_a.py::test_a"]},
    )
    monkeypatch.setattr(
        runner,
        "_collect_coverage",
        lambda: {"covered": 95, "total": 100, "percent": 95.0},
    )
    monkeypatch.setattr(runner, "_secret_scan", lambda: {"results": {}})
    monkeypatch.setattr(runner, "_write_secret_baseline", lambda payload: None)

    runner._quality_baseline()

    assert json.loads(baseline_path.read_text(encoding="utf-8"))["schema_version"] == 2


@pytest.mark.parametrize("command", ["fast", "pr", "nightly", "baseline"])
def test_main_routes_each_quality_command(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
) -> None:
    called: list[object] = []
    monkeypatch.setattr(
        runner, "_quality_fast", lambda paths: called.append(tuple(paths))
    )
    monkeypatch.setattr(runner, "_quality_pr", lambda: called.append("pr"))
    monkeypatch.setattr(runner, "_quality_nightly", lambda: called.append("nightly"))
    monkeypatch.setattr(runner, "_quality_baseline", lambda: called.append("baseline"))
    arguments = [command, "changed.py"] if command == "fast" else [command]

    assert runner.main(arguments) == 0
    assert called
    assert f"quality-{command}: passed" in capsys.readouterr().out


def test_main_propagates_gate_failure_as_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail() -> None:
        raise _runner_tools.GateFailure("security failed")

    monkeypatch.setattr(runner, "_quality_pr", fail)

    assert runner.main(["pr"]) == 1
    assert "QUALITY GATE FAILED: security failed" in capsys.readouterr().err


def test_heavy_session_gates_cover_performance_mutation_and_nightly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = {
        "config": "examples/v02_minimal/run_config.yaml",
        "tracemalloc_bytes": {"peak_median": 90_000},
        "wall_time_s": {"median": 0.001},
    }
    monkeypatch.setattr(runner, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(
        runner,
        "_run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            (), 0, stdout=json.dumps({"cases": [case]}), stderr=""
        ),
    )
    monkeypatch.delenv("QUALITY_PERF_TIMING_GATE", raising=False)
    runner._run_performance()
    case["tracemalloc_bytes"]["peak_median"] = 200_000
    with pytest.raises(_runner_tools.GateFailure, match="memory regression"):
        runner._run_performance()
    case["tracemalloc_bytes"]["peak_median"] = 90_000
    case["wall_time_s"]["median"] = 1.0
    monkeypatch.setenv("QUALITY_PERF_TIMING_GATE", "1")
    with pytest.raises(_runner_tools.GateFailure, match="timing regression"):
        runner._run_performance()

    monkeypatch.setattr(runner.os, "name", "nt")
    with pytest.raises(_runner_tools.GateFailure, match="WSL/Linux"):
        runner._run_mutation()
    monkeypatch.setattr(runner.os, "name", "posix")
    monkeypatch.setattr(runner, "_tool", lambda name: name)
    mutation = "0 survived\n0 suspicious\n0 timeout\n0 untested\n"
    monkeypatch.setattr(
        runner,
        "_run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            (), 0, stdout=mutation, stderr=""
        ),
    )
    runner._run_mutation()
    mutation = "1 survived\n"
    with pytest.raises(_runner_tools.GateFailure, match="non-killed mutants"):
        runner._run_mutation()

    calls: list[dict[str, str] | None] = []
    monkeypatch.setattr(runner, "_quality_pr", lambda: None)
    monkeypatch.setattr(
        runner,
        "_run",
        lambda *args, **kwargs: (
            calls.append(kwargs.get("environment"))
            or subprocess.CompletedProcess((), 0)
        ),
    )
    monkeypatch.setattr(runner, "_run_performance", lambda: None)
    monkeypatch.setattr(runner, "_run_mutation", lambda: None)
    runner._quality_nightly()
    assert [
        environment["NUMBA_DISABLE_JIT"] for environment in calls if environment
    ] == [
        "0",
        "1",
    ]
