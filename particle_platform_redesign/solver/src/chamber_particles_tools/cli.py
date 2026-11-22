"""One command line for installed input preparation, computation, and results."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TextIO

from chamber_particles import (
    CaseError,
    SimulationError,
    check_case,
    load_case,
    open_result,
    simulate,
)
from chamber_particles_tools.electrostatic_builder.numerics import NonlinearSolveError


def main(argv: Sequence[str] | None = None) -> int:
    """Execute a public operation and return its process exit status."""
    arguments = _parser().parse_args(argv)
    try:
        payload = _dispatch(arguments)
    except (CaseError, SimulationError, NonlinearSolveError, OSError, ValueError) as error:
        _write(
            {
                "status": "error",
                "operation": arguments.command,
                "error_type": type(error).__name__,
                "message": str(error),
            },
            machine=arguments.json,
            stream=sys.stderr,
        )
        return 2
    _write(payload, machine=arguments.json)
    return 0


def _dispatch(arguments: argparse.Namespace) -> object:
    command = arguments.command
    if command == "inputs":
        return _input_operation(arguments)
    if command in {"models", "init", "create"}:
        return _preparation_operation(arguments)
    if command in {"check", "run", "inspect"}:
        return _simulation_operation(arguments)
    return _result_operation(arguments)


def _preparation_operation(arguments: argparse.Namespace) -> object:
    if arguments.command == "models":
        from chamber_particles.physics.catalog import describe_models

        descriptors = describe_models()
        selected = [
            item
            for item in descriptors
            if arguments.category is None or item.category == arguments.category
        ]
        if not selected:
            raise ValueError(f"unknown model category: {arguments.category}")
        return {"operation": "models", "status": "complete", "models": selected}
    from .inputs import create_from_configuration, init_example

    if arguments.command == "init":
        case = init_example(arguments.example, arguments.output)
    else:
        case = create_from_configuration(arguments.configuration, arguments.output)
    return {
        "operation": arguments.command,
        "status": "created",
        "case": str(case.case_path),
        "data": str(case.data_path),
        "content_hash": case.content_hash,
    }


def _simulation_operation(arguments: argparse.Namespace) -> object:
    if arguments.command == "check":
        report = check_case(load_case(arguments.case), workers=arguments.workers)
        return {"operation": "check", "status": "ready", "report": report}
    if arguments.command == "run":
        summary = simulate(load_case(arguments.case), arguments.output, workers=arguments.workers)
        return {"operation": "run", "status": "complete", "summary": summary}
    result = open_result(arguments.result)
    keys = (
        "case_name",
        "counts",
        "lifecycle_counts",
        "failure_reason_counts",
        "data_coordinate_system",
        "motion_mode",
        "result_schema_version",
    )
    summary = {key: result.manifest[key] for key in keys if key in result.manifest}
    payload: dict[str, object] = {
        "operation": "inspect",
        "status": "complete",
        "result_path": str(result.path),
        "summary": summary,
    }
    if arguments.full:
        payload["manifest"] = dict(result.manifest)
    return payload


def _input_operation(arguments: argparse.Namespace) -> object:
    if arguments.input_command == "import-csv":
        from .csv_import.workflow import adapt_from_configuration

        report = adapt_from_configuration(
            arguments.configuration, arguments.output, report_path=arguments.report
        )
        return {"operation": "inputs import-csv", "status": "created", "report": report}
    if arguments.input_command == "cache":
        from .field_preprocessor.workflow import preprocess_from_configuration

        report = preprocess_from_configuration(
            arguments.configuration,
            arguments.output,
            report_path=arguments.report,
            source_time_diagnostic_requested=arguments.source_time_diagnostic,
        )
        return {"operation": "inputs cache", "status": "created", "report": report}
    from .electrostatic_builder.workflow import build_from_configuration, check_from_configuration

    if arguments.electric_command == "check":
        return {
            "operation": "inputs electrostatic check",
            "status": "ready",
            "report": check_from_configuration(arguments.configuration),
        }
    report = build_from_configuration(
        arguments.configuration, arguments.output, report_path=arguments.report
    )
    return {"operation": "inputs electrostatic build", "status": "created", "report": report}


def _result_operation(arguments: argparse.Namespace) -> object:
    from .analysis import export_result, summarize_result

    selection = {
        "source_ids": _integer_selection(arguments.source_ids),
        "particle_ids": _integer_selection(arguments.particle_ids),
        "event_batch_rows": arguments.event_batch_rows,
    }
    if arguments.command == "analyze":
        summary = summarize_result(
            arguments.result,
            **selection,
            coordinate0_bin_edges_m=_number_selection(arguments.coordinate0_bin_edges_m),
            terminal_time_bin_edges_s=_number_selection(arguments.terminal_time_bin_edges_s),
        )
        if arguments.output is not None:
            _write_derived_json(arguments.result, arguments.output, summary)
        return summary
    if arguments.command == "export":
        return export_result(arguments.result, arguments.output, **selection)
    from .visualization import render_result

    return render_result(
        arguments.result,
        arguments.output,
        **selection,
        particle_id=arguments.particle_id,
        detail=arguments.detail,
        max_event_points=arguments.max_event_points,
        max_particle_points=arguments.max_particle_points,
        max_trajectory_points=arguments.max_trajectory_points,
    )


def _integer_selection(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    values = tuple(int(item) for item in value.split(","))
    if not values or any(item < 0 for item in values) or len(set(values)) != len(values):
        raise ValueError("ID selections must contain unique nonnegative comma-separated integers")
    return values


def _number_selection(value: str | None) -> tuple[float, ...] | None:
    return None if value is None else tuple(float(item) for item in value.split(","))


def _plain(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _plain(getattr(value, field.name)) for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _write(payload: object, *, machine: bool, stream: TextIO = sys.stdout) -> None:
    encoded = json.dumps(
        _plain(payload),
        allow_nan=False,
        ensure_ascii=machine,
        sort_keys=True,
        indent=None if machine else 2,
    )
    encoding = stream.encoding or "utf-8"
    try:
        encoded.encode(encoding)
    except UnicodeEncodeError:
        encoded = json.dumps(
            _plain(payload),
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            indent=None if machine else 2,
        )
    print(encoded, file=stream)


def _write_derived_json(result: Path, output: Path, payload: object) -> None:
    source, target = result.expanduser().resolve(), output.expanduser().resolve()
    if target == source or source in target.parents:
        raise ValueError("derived output must be outside the immutable source result directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(_plain(payload), allow_nan=False, indent=2, sort_keys=True) + "\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chamber-particles",
        description="Prepare, check, run, and interpret particle transport.",
    )
    parser.add_argument("--json", action="store_true", help="compact machine-readable output")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", default=argparse.SUPPRESS)
    commands = parser.add_subparsers(dest="command", required=True)
    _preparation_arguments(commands, common)
    _simulation_arguments(commands, common)
    _result_arguments(commands, common)
    return parser


def _preparation_arguments(
    commands: argparse._SubParsersAction, common: argparse.ArgumentParser
) -> None:
    models = commands.add_parser(
        "models", parents=[common], help="list models and input requirements"
    )
    models.add_argument("--category")
    init = commands.add_parser(
        "init", parents=[common], help="create an editable installed example"
    )
    init.add_argument("output", type=Path, help="new case directory")
    init.add_argument(
        "--example",
        default="xy-ballistic",
        choices=(
            "xy-ballistic",
            "rz-charged",
            "xy-time-field",
            "brownian-wall",
            "reduced-electrostatic",
        ),
    )
    create = commands.add_parser(
        "create", parents=[common], help="compose own bundle, sources, and settings"
    )
    create.add_argument("configuration", type=Path)
    create.add_argument("-o", "--output", type=Path, required=True)
    inputs = commands.add_parser(
        "inputs", parents=[common], help="import or generate canonical fields"
    )
    sub = inputs.add_subparsers(dest="input_command", required=True)
    for name in ("import-csv", "cache"):
        operation = sub.add_parser(name, parents=[common])
        _builder_arguments(operation)
        if name == "cache":
            operation.add_argument("--source-time-diagnostic", action="store_true")
    electric = sub.add_parser(
        "electrostatic", parents=[common], help="reduced static RZ electric model"
    )
    modes = electric.add_subparsers(dest="electric_command", required=True)
    check = modes.add_parser("check", parents=[common])
    check.add_argument("configuration", type=Path)
    _builder_arguments(modes.add_parser("build", parents=[common]))


def _builder_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("configuration", type=Path)
    parser.add_argument("output", type=Path, help="new canonical HDF5 bundle")
    parser.add_argument("--report", type=Path)


def _simulation_arguments(
    commands: argparse._SubParsersAction, common: argparse.ArgumentParser
) -> None:
    for name in ("check", "run"):
        parser = commands.add_parser(
            name,
            parents=[common],
            help="prepare without execution" if name == "check" else "run or resume one case",
        )
        parser.add_argument("case", type=Path)
        parser.add_argument("--workers", type=int, default=1)
        if name == "run":
            parser.add_argument("-o", "--output", type=Path, required=True)
    inspect = commands.add_parser(
        "inspect", parents=[common], help="summarize one completed result"
    )
    inspect.add_argument("result", type=Path)
    inspect.add_argument(
        "--full", action="store_true", help="include the complete immutable manifest"
    )


def _result_arguments(
    commands: argparse._SubParsersAction, common: argparse.ArgumentParser
) -> None:
    for name in ("analyze", "export", "view"):
        parser = commands.add_parser(name, parents=[common])
        parser.add_argument("result", type=Path)
        parser.add_argument("--source-ids", help="comma-separated resolved source IDs")
        parser.add_argument("--particle-ids", help="comma-separated stable particle IDs")
        parser.add_argument("--event-batch-rows", type=int, default=4096)
        parser.add_argument("-o", "--output", type=Path, required=name != "analyze")
        if name == "analyze":
            parser.add_argument("--coordinate0-bin-edges-m", help="comma-separated SI bin edges")
            parser.add_argument("--terminal-time-bin-edges-s", help="comma-separated SI bin edges")
        elif name == "view":
            parser.add_argument("--particle-id", type=int)
            parser.add_argument("--detail", action="store_true")
            parser.add_argument("--max-event-points", type=int, default=5000)
            parser.add_argument("--max-particle-points", type=int, default=5000)
            parser.add_argument("--max-trajectory-points", type=int, default=5000)
