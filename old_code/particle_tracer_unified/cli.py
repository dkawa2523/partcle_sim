"""The single v0.2 command-line entry point."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ._version import PACKAGE_VERSION
from .application import load_case, simulate, validate_case
from .artifacts import validate_artifacts
from .migration import migrate_legacy_case
from .writer import write_result

_COMPARE_COMMANDS = {
    "field": "particle_tracer_unified.compare.field_compare",
    "acceleration": "particle_tracer_unified.compare.acceleration_compare",
    "trajectory": "particle_tracer_unified.compare.trajectory_compare",
    "boundary": "particle_tracer_unified.compare.boundary_compare",
    "first-step": "particle_tracer_unified.compare.first_step_compare",
    "near-wall": "particle_tracer_unified.compare.near_wall_nohit",
    "comsol-full": "particle_tracer_unified.compare.comsol_full_diagnostics",
    "reference": "tools.compare_against_reference",
}
_COMSOL_COMMANDS = {
    "build-case": "particle_tracer_unified.comsol_case.cli",
}


def _json(value: Any) -> str:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    return json.dumps(value, indent=2, sort_keys=True, default=str)


def _module_main(module_name: str, arguments: Sequence[str]) -> int:
    """Route retained specialist tools below the one public console script."""

    module = importlib.import_module(module_name)
    entrypoint: Callable[[Sequence[str] | None], Any] | None = getattr(
        module,
        "main",
        None,
    )
    if entrypoint is None:
        raise ValueError(f"{module_name} does not expose main(argv)")
    result = entrypoint([str(argument) for argument in arguments])
    return int(result or 0)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="particle-tracer",
        description="Validate, run, migrate, and inspect particle trajectory cases.",
    )
    parser.add_argument(
        "--version", action="version", version=f"particle-tracer {PACKAGE_VERSION}"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="validate and run a canonical v0.2 case")
    run.add_argument("config", type=Path)
    run.add_argument("--output-dir", "-o", type=Path, default=None)

    check = commands.add_parser("check", help="run the side-effect-free case preflight")
    check.add_argument("config", type=Path)
    check.add_argument(
        "--full", action="store_true", help="include row/sample-level check detail"
    )

    migrate = commands.add_parser(
        "migrate", help="convert a legacy YAML/CSV case to v0.2"
    )
    migrate.add_argument("config", type=Path)
    migrate.add_argument("--output-dir", "-o", required=True, type=Path)
    migrate.add_argument("--overwrite", action="store_true")

    compare = commands.add_parser("compare", help="run a focused validation comparison")
    compare.add_argument("workflow", choices=tuple(_COMPARE_COMMANDS))
    compare.add_argument("arguments", nargs=argparse.REMAINDER)

    artifacts = commands.add_parser(
        "artifacts", help="validate the canonical artifact set"
    )
    artifacts.add_argument("root", type=Path)
    artifacts.add_argument("--require-debug", action="store_true")

    # Its complete option surface belongs to tools.export_visualizations and
    # is routed before this generic parser in ``main``.
    commands.add_parser("visualize", help="export optional visualizations")

    comsol = commands.add_parser("comsol", help="COMSOL case extraction/build tools")
    comsol.add_argument("workflow", choices=tuple(_COMSOL_COMMANDS))
    comsol.add_argument("arguments", nargs=argparse.REMAINDER)
    return parser


def _run(args: argparse.Namespace) -> int:
    case = load_case(args.config)
    report = validate_case(case, detail="summary")
    if not report.passed:
        print(_json(report), file=sys.stderr)
        return 2
    result = simulate(case)
    output_dir = args.output_dir or (Path(args.config).resolve().parent / "run_output")
    manifest = write_result(result, output_dir)
    print(
        _json(
            {
                "schema_version": manifest.schema_version,
                "output_dir": str(manifest.output_dir),
                "artifacts": {
                    record.artifact_type: {
                        "path": str(record.path),
                        "size_bytes": record.size_bytes,
                        "sha256": record.sha256,
                    }
                    for record in manifest.records
                },
            }
        )
    )
    return 0


def _check(args: argparse.Namespace) -> int:
    case = load_case(args.config)
    report = validate_case(case, detail="full" if args.full else "summary")
    print(_json(report))
    return 0 if report.passed else 1


def _migrate(args: argparse.Namespace) -> int:
    result = migrate_legacy_case(args.config, args.output_dir, overwrite=args.overwrite)
    print(
        _json(
            {
                "schema_version": 2,
                "config": str(result.config_path),
                "particles": str(result.particles_path)
                if result.particles_path
                else None,
                "boundaries": str(result.boundaries_path)
                if result.boundaries_path
                else None,
                "warnings": list(result.warnings),
            }
        )
    )
    return 0


def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "run":
        return _run(args)
    if args.command == "check":
        return _check(args)
    if args.command == "migrate":
        return _migrate(args)
    if args.command == "compare":
        return _module_main(_COMPARE_COMMANDS[args.workflow], args.arguments)
    if args.command == "artifacts":
        report = validate_artifacts(args.root, require_debug=bool(args.require_debug))
        print(_json(report))
        return 0 if bool(report["passed"]) else 1
    if args.command == "comsol":
        return _module_main(_COMSOL_COMMANDS[args.workflow], args.arguments)
    raise ValueError(f"unknown command {args.command!r}")


def main(argv: Sequence[str] | None = None) -> int:
    raw_arguments = list(argv) if argv is not None else sys.argv[1:]
    # ``visualize`` has no intermediate workflow selector.  Route it before
    # argparse so its real parser owns ``--help`` and error messages as well as
    # execution; the public process argument vector is only read, never
    # rewritten.
    if raw_arguments and raw_arguments[0] == "visualize":
        try:
            return _module_main("tools.export_visualizations", raw_arguments[1:])
        except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
            print(f"particle-tracer: {exc}", file=sys.stderr)
            return 2
    parser = _build_parser()
    args = parser.parse_args(raw_arguments)
    try:
        return _dispatch(args)
    except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
        print(f"particle-tracer: {exc}", file=sys.stderr)
        return 2


__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
