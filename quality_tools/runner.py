from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Sequence
from typing import Any

from . import _runner_tools as _tools
from ._runner_baseline import (
    _baseline_static,
    _compare_coverage,
    _compare_static,
    _compare_tests,
    _load_baseline,
    _write_secret_baseline,
)
from ._runner_diff import (
    _changed_paths,
    _run_diff_coverage,
)
from ._runner_tools import (
    BASELINE_PATH,
    INITIAL_COVERED,
    INITIAL_TOTAL,
    PYREFLY_BASELINE_PATH,
    QUALITY_DIR,
    REPORT_DIR,
    GateFailure,
    _collect_coverage,
    _collect_static,
    _collect_tests,
    _python_module,
    _relative,
    _run,
    _run_architecture,
    _run_code_and_dependency_security,
    _run_pyrefly,
    _run_security,
    _secret_scan,
    _tool,
)

subprocess = _tools.subprocess


def _quality_fast(explicit_paths: Sequence[str]) -> None:
    baseline = _load_baseline()
    if baseline is None:
        raise GateFailure(
            "missing .quality/baseline.json; run quality-baseline explicitly"
        )
    paths = _changed_paths(explicit_paths)
    if paths:
        _run([_tool("ruff"), "format", *[str(path) for path in paths]])
        _run(
            [
                _tool("ruff"),
                "check",
                "--fix",
                "--exit-zero",
                *[str(path) for path in paths],
            ],
            capture=True,
        )
    else:
        print("No changed Python files to format or fix")
    current = _collect_static()
    _compare_static(current, baseline)
    _run_pyrefly()
    tests = _collect_tests()
    _compare_tests({"tests": tests}, baseline)
    _run(_python_module("pytest", "-q"))


def _quality_pr() -> dict[str, Any]:
    baseline = _load_baseline()
    if baseline is None:
        raise GateFailure(
            "missing .quality/baseline.json; run quality-baseline explicitly"
        )
    current = _collect_static()
    _compare_static(current, baseline)
    _run_pyrefly()
    _run_architecture()
    tests = _collect_tests()
    _compare_tests({"tests": tests}, baseline)
    coverage = _collect_coverage()
    _compare_coverage(coverage, baseline)
    _run_diff_coverage()
    _run_security()
    return {**current, "tests": tests, "coverage": coverage}


def _run_performance() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    result = _run(
        [
            sys.executable,
            "validation/measure_performance_v0_2.py",
            "--warmups",
            "2",
            "--repeats",
            "7",
            "--memory-repeats",
            "3",
        ],
        capture=True,
    )
    payload = json.loads(result.stdout)
    (REPORT_DIR / "performance.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    memory_limits = {
        "examples/v02_minimal/run_config.yaml": 90_954,
        "examples/v02_minimal_3d/run_config.yaml": 529_419,
    }
    timing_limits = {
        "examples/v02_minimal/run_config.yaml": 0.0015361001715064049,
        "examples/v02_minimal_3d/run_config.yaml": 0.03321920009329915,
    }
    for case in payload["cases"]:
        config = str(case["config"])
        measured = int(case["tracemalloc_bytes"]["peak_median"])
        limit = int(memory_limits[config] * 1.10)
        if measured > limit:
            raise GateFailure(
                f"performance memory regression for {config}: {measured} > {limit}"
            )
        if os.environ.get("QUALITY_PERF_TIMING_GATE") == "1":
            elapsed = float(case["wall_time_s"]["median"])
            timing_limit = timing_limits[config] * 1.10
            if elapsed > timing_limit:
                raise GateFailure(
                    f"performance timing regression for {config}: "
                    f"{elapsed:.6f}s > {timing_limit:.6f}s"
                )


def _run_mutation() -> None:
    if os.name == "nt":
        raise GateFailure("mutmut requires fork; run quality-nightly under WSL/Linux")
    uv = _tool("uv")
    _run([uv, "run", "--frozen", "--group", "nightly", "mutmut", "run"])
    result = _run(
        [uv, "run", "--frozen", "--group", "nightly", "mutmut", "results"],
        capture=True,
    )
    (REPORT_DIR / "mutation.txt").write_text(result.stdout, encoding="utf-8")
    forbidden = re.compile(r"(?im)^.*\b(survived|suspicious|timeout|untested)\b.*$")
    nonzero_findings = [
        match.group(0)
        for match in forbidden.finditer(result.stdout)
        if not re.search(
            r"\b0\s+(?:survived|suspicious|timeout|untested)\b",
            match.group(0),
        )
    ]
    if nonzero_findings:
        raise GateFailure("mutation report contains non-killed mutants")


def _quality_nightly() -> None:
    _quality_pr()
    selected = [
        "tests/test_reference_compare_cli.py",
        "tests/test_piecewise_brownian_v06.py",
        "tests/test_brownian_temperature_authority_v02.py",
        "tests/test_nightly_quality.py",
    ]
    for disabled in ("0", "1"):
        environment = dict(os.environ)
        environment["NUMBA_DISABLE_JIT"] = disabled
        environment["HYPOTHESIS_PROFILE"] = "nightly"
        _run(_python_module("pytest", "-q", *selected), environment=environment)
    _run_performance()
    _run_mutation()


def _quality_baseline() -> None:
    previous = _load_baseline()
    if previous is not None:
        current_static = _collect_static()
        _compare_static(current_static, previous)
        _run_pyrefly()
    _run_pyrefly(update=True)
    _run_architecture()
    _run_code_and_dependency_security()
    static = _collect_static()
    tests = _collect_tests()
    coverage = _collect_coverage()
    if previous is not None:
        _compare_tests({"tests": tests}, previous)
        _compare_coverage(coverage, previous)
    if coverage["covered"] * INITIAL_TOTAL < INITIAL_COVERED * coverage["total"]:
        raise GateFailure("refusing baseline below immutable initial coverage")
    secrets = _secret_scan()
    _write_secret_baseline(secrets)
    payload = {
        "schema_version": 2,
        **_baseline_static(static),
        "tests": tests,
        "coverage": coverage,
        "pyrefly_baseline": _relative(PYREFLY_BASELINE_PATH),
        "secret_baseline": ".secrets.baseline",
    }
    QUALITY_DIR.mkdir(parents=True, exist_ok=True)
    BASELINE_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Updated {_relative(BASELINE_PATH)}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fast", "pr", "nightly", "baseline"))
    parser.add_argument("paths", nargs="*")
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "fast":
            _quality_fast(arguments.paths)
        elif arguments.command == "pr":
            _quality_pr()
        elif arguments.command == "nightly":
            _quality_nightly()
        else:
            _quality_baseline()
    except GateFailure as exc:
        print(f"QUALITY GATE FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"quality-{arguments.command}: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
