"""Thin command-line access to the three public solver operations."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import fields
from pathlib import Path
from typing import TextIO

from .api import CaseError, SimulationError, load_case, open_result, simulate


def main(argv: Sequence[str] | None = None) -> int:
    """Run one CLI command and return a process exit status."""

    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "check":
            case = load_case(arguments.case)
            _write_json(
                {
                    "operation": "check",
                    "status": "valid",
                    "case_name": case.spec.name,
                    "case_path": str(case.case_path),
                    "data_path": str(case.data_path),
                    "case_file_hash": case.case_file_hash,
                    "data_content_hash": case.content_hash,
                    "data_coordinate_system": case.data.coordinate_system,
                    "motion_mode": case.spec.motion.mode,
                    "canonical_numeric_array_bytes": (case.data_footprint.numeric_array_bytes),
                    "memory_limit_bytes": case.spec.resources.memory_limit_mb * 1024 * 1024,
                }
            )
            return 0
        if arguments.command == "run":
            summary = simulate(load_case(arguments.case), arguments.output)
            payload = {field.name: getattr(summary, field.name) for field in fields(summary)}
            payload["output_path"] = str(Path(summary.output_path))
            _write_json({"operation": "run", "status": "complete", **payload})
            return 0

        result = open_result(arguments.result)
        _write_json(
            {
                "operation": "inspect",
                "status": "complete",
                "result_path": str(result.path),
                "manifest": dict(result.manifest),
            }
        )
        return 0
    except (CaseError, SimulationError) as error:
        _write_json(
            {
                "status": "error",
                "error_type": type(error).__name__,
                "message": str(error),
            },
            stream=sys.stderr,
        )
        return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chamber-particles",
        description="Run and inspect deterministic chamber-particle simulations.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check", help="validate one canonical case")
    check.add_argument("case", help="path to case.yaml")

    run = commands.add_parser("run", help="simulate one canonical case")
    run.add_argument("case", help="path to case.yaml")
    run.add_argument("-o", "--output", required=True, help="new result directory")

    inspect = commands.add_parser("inspect", help="inspect one completed result")
    inspect.add_argument("result", help="completed result directory")
    return parser


def _write_json(payload: object, *, stream: TextIO = sys.stdout) -> None:
    print(
        json.dumps(payload, allow_nan=False, ensure_ascii=False, sort_keys=True),
        file=stream,
    )
