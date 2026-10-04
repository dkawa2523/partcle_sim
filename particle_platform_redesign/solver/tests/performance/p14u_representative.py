"""P14-U representative-use accuracy, utility, and serial-performance gate.

This is an explicitly invoked external V&V harness.  Test-only builders and
the canonical writer construct analytic inputs; solver execution then uses the
three public APIs.  The harness records convergence, event work, memory,
output identity, and a separate external profile.  It is deliberately not a
second solver or a production diagnostic system.
"""

from __future__ import annotations

import argparse
import copy
import cProfile
import gc
import hashlib
import json
import math
import os
import pstats
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from statistics import median
from typing import Any, Literal

import numpy as np
import yaml

from chamber_particles import load_case, open_result, simulate
from chamber_particles.case_format import (
    BoundaryData,
    DataBundle,
    FieldData,
    GeometryData,
    P1TriLayout,
    Q1QuadLayout,
    RegularLayout,
    write,
)
from tests.performance.p09_memory import (
    _digest_dataclass,
    _directory_bytes,
    _machine_metadata,
    _process_memory_bytes,
)
from tests.verification.microcases import build_microcase

type _Suite = Literal["smoke", "release"]
type _LayoutKind = Literal["regular", "p1", "q1"]
type _OutputMode = Literal["none", "sample"]

_XY_SOURCE_Y_M = 0.301
_XY_EQUILIBRIUM_Y_M = 1.0 / 3.0
_XY_END_S = 4.0
_XY_SYNC_TIMES = (0.5, 1.5, 2.5, 3.5)
_RZ_END_S = 0.5
_RZ_SYNC_TIMES = (0.1, 0.2, 0.3, 0.4, 0.5)
_SAMPLE_PARTICLE_COUNT = 32
_RELEASE_PARTICLE_COUNTS = (10_000, 100_000, 1_000_000)
_TARGET_FACET_MIN_CLEARANCE = 0.2


def main() -> None:
    """Run the P14-U driver, or one isolated performance worker."""

    arguments = _arguments()
    if arguments.worker_spec is not None:
        _worker(arguments.worker_spec)
        return
    _driver(arguments)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("smoke", "release"), default="smoke")
    parser.add_argument(
        "--particles",
        type=_positive_integer,
        nargs="+",
        help="override the suite's representative performance particle counts",
    )
    parser.add_argument("--memory-limit-mb", type=_positive_integer, default=8192)
    parser.add_argument("--json", dest="json_path", type=Path)
    parser.add_argument("--worker-spec", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _json_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"P14-U report contains unsupported {type(value).__name__}")


def _driver(arguments: argparse.Namespace) -> None:
    suite: _Suite = arguments.suite
    counts = tuple(
        arguments.particles or ((256,) if suite == "smoke" else _RELEASE_PARTICLE_COUNTS)
    )
    if min(counts) < _SAMPLE_PARTICLE_COUNT:
        raise SystemExit(f"P14-U performance counts must be at least {_SAMPLE_PARTICLE_COUNT}")
    if tuple(sorted(set(counts))) != counts:
        raise SystemExit("P14-U performance counts must be unique and strictly increasing")
    repeat_count = 1 if suite == "smoke" else 3
    release_gate_complete = (
        suite == "release"
        and arguments.particles is None
        and counts == _RELEASE_PARTICLE_COUNTS
        and repeat_count == 3
    )

    with tempfile.TemporaryDirectory(prefix="chamber-particles-p14u-") as temporary:
        root = Path(temporary)
        convergence = _convergence_gate(root / "convergence", suite, arguments.memory_limit_mb)
        rz = _rz_gate(root / "rz", suite, arguments.memory_limit_mb)
        performance = _performance_gate(
            root / "performance",
            counts,
            repeat_count=repeat_count,
            memory_limit_mb=arguments.memory_limit_mb,
        )
        _validate_performance_identity(performance)
        report = {
            "benchmark": "p14u_representative_utility_v1",
            "captured_at_utc": datetime.now(UTC).isoformat(),
            "suite": suite,
            "machine": _machine_metadata(),
            "conditions": {
                "execution_mode": "single-thread compiled CPU",
                "solver_execution_uses_only_public_api": True,
                "harness_input_generation_uses_canonical_writer": True,
                "harness_uses_test_only_artifact_helpers": True,
                "absolute_seconds_are_gating": False,
                "memory_limit_mb": arguments.memory_limit_mb,
                "performance_particle_counts": list(counts),
                "performance_repeats_per_mode": repeat_count,
                "rss_note": (
                    "peak RSS is the process high-water since worker start, sampled after "
                    "load/simulate/open and before external result validation; additional "
                    "high-water is max(0, peak_after - peak_before), not instantaneous allocation"
                ),
                "release_gate_complete": release_gate_complete,
                "release_gate_reason": (
                    "complete canonical 10k/100k/1M matrix with three fresh-process repeats"
                    if release_gate_complete
                    else "smoke or custom particle counts are evidence only, not P14-U release completion"
                ),
            },
            "xy_convergence": convergence,
            "rz_convergence": rz,
            "performance": performance,
            "decisions": {
                "surface_source": (
                    "this Cartesian XY line_length workload validates edge_fraction and "
                    "uniform sampling on one line source group; it does not validate RZ "
                    "revolved_area sampling or justify a realized-surface-table format"
                ),
                "parallelism": (
                    "P14-P closed internal multithreading negatively; P14-U measures the one "
                    "serial engine and does not reopen scheduler design"
                ),
                "profile_interpretation": (
                    "owner-specific optimization conclusions are withheld when "
                    "runtime_or_dependency is the largest self-time owner"
                ),
                "rz_parity": (
                    "radial zero and standard-gravity radial zero are solver invariants; "
                    "scalar/axial smooth parity is an external producer-quality audit because "
                    "one-sided r>=0 samples cannot prove the underlying continuation"
                ),
            },
        }

    encoded = (
        json.dumps(
            report,
            allow_nan=False,
            default=_json_scalar,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if arguments.json_path is not None:
        arguments.json_path.parent.mkdir(parents=True, exist_ok=True)
        arguments.json_path.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


def _convergence_gate(root: Path, suite: _Suite, memory_limit_mb: int) -> dict[str, object]:
    root.mkdir(parents=True)
    # The coarsest 0.125 s row is useful for throughput but lies above the
    # representative event-localization asymptotic range.  The convergence
    # gate therefore starts at 0.0625 s and uses a separate finer reference.
    temporal_steps = (0.0625, 0.03125, 0.015625)
    temporal_reference_step = 0.0078125 if suite == "smoke" else 0.00390625
    # Keep the spatial mesh fixed across suites so this gate isolates time
    # refinement.  The independent mesh gate below owns the 128 x 128 reference.
    temporal_nx = 64
    temporal_reference = _run_xy_case(
        root / "time-reference",
        layout_kind="regular",
        nx=temporal_nx,
        dt_s=temporal_reference_step,
        particle_count=1,
        output_mode="sample",
        uniform_source=False,
        memory_limit_mb=memory_limit_mb,
    )
    temporal_rows = [
        _run_xy_case(
            root / f"time-{index}",
            layout_kind="regular",
            nx=temporal_nx,
            dt_s=step,
            particle_count=1,
            output_mode="sample",
            uniform_source=False,
            memory_limit_mb=memory_limit_mb,
        )
        for index, step in enumerate(temporal_steps)
    ]
    _require_successful_rows([*temporal_rows, temporal_reference], "time")
    temporal_errors = [_trajectory_errors(row, temporal_reference) for row in temporal_rows]
    temporal_self_differences = [
        _trajectory_errors(coarse, fine) for coarse, fine in pairwise(temporal_rows)
    ]
    _require_decreasing(
        temporal_self_differences,
        ("position_m", "velocity_m_s", "hit_time_s", "hit_position_m"),
        "time self-convergence",
    )
    _require_net_reduction(
        temporal_errors,
        ("position_m", "velocity_m_s", "hit_time_s", "hit_position_m"),
        "time reference",
    )
    hit_time = temporal_reference["hit_time_s"]
    if not isinstance(hit_time, float):
        raise RuntimeError("P14-U dense path audit requires one localized reference hit")
    dense_times = _dense_probe_times(temporal_reference_step, hit_time)
    dense_reference = _run_xy_case(
        root / "dense-path-reference",
        layout_kind="regular",
        nx=temporal_nx,
        dt_s=temporal_reference_step,
        particle_count=1,
        output_mode="sample",
        uniform_source=False,
        memory_limit_mb=memory_limit_mb,
        probe_times=dense_times,
    )
    _require_successful_rows([dense_reference], "dense path reference")

    mesh_levels = (4, 8, 16) if suite == "smoke" else (8, 16, 32)
    mesh_reference_nx = 64 if suite == "smoke" else 128
    mesh_step = temporal_reference_step
    mesh: dict[str, object] = {}
    for layout_kind in ("regular", "p1", "q1"):
        mesh_reference = _run_xy_case(
            root / f"mesh-{layout_kind}-reference",
            layout_kind=layout_kind,
            nx=mesh_reference_nx,
            dt_s=mesh_step,
            particle_count=1,
            output_mode="sample",
            uniform_source=False,
            memory_limit_mb=memory_limit_mb,
        )
        rows = [
            _run_xy_case(
                root / f"mesh-{layout_kind}-{nx}",
                layout_kind=layout_kind,
                nx=nx,
                dt_s=mesh_step,
                particle_count=1,
                output_mode="sample",
                uniform_source=False,
                memory_limit_mb=memory_limit_mb,
            )
            for nx in mesh_levels
        ]
        _require_successful_rows([*rows, mesh_reference], f"{layout_kind} mesh")
        errors = [
            _trajectory_errors(row, mesh_reference, require_same_facet_id=False) for row in rows
        ]
        _require_decreasing(
            errors,
            ("position_m", "velocity_m_s", "hit_time_s", "hit_position_m"),
            layout_kind,
        )
        mesh[layout_kind] = {
            "x_cells": list(mesh_levels),
            "errors": errors,
            "observed_orders": _observed_orders(errors),
            "target_boundary_id": [row["target_boundary_id"] for row in rows],
            "target_group": [row["target_group"] for row in rows],
            "target_facet_coordinate": [row["target_facet_coordinate"] for row in rows],
            "target_facet_clearance": [row["target_facet_clearance"] for row in rows],
            "same_layout_reference_x_y_cells": [mesh_reference_nx, mesh_reference_nx],
        }

    return {
        "case": (
            "surface release + Epstein drag + non-affine conservative electric field + "
            "gravity + material target + 32--512 macro steps"
        ),
        "time_steps_s": list(temporal_steps),
        "time_reference_step_s": temporal_reference_step,
        "time_errors": temporal_errors,
        "time_reference_error_orders": _observed_orders(temporal_errors),
        "time_self_differences": temporal_self_differences,
        "time_self_convergence_orders": _observed_orders(temporal_self_differences),
        "mesh_refinement": "fixed-aspect nx=ny with a same-layout fine reference",
        "mesh": mesh,
        "reference_hit": _hit_summary(temporal_reference),
        "global_to_dense_path_electric_acceleration_ratio": _global_to_path_ratio(dense_reference),
        "reference_event_work": temporal_reference["event_work"],
        "reference_failure_count": temporal_reference["failure_count"],
    }


def _rz_gate(root: Path, suite: _Suite, memory_limit_mb: int) -> dict[str, object]:
    root.mkdir(parents=True)
    steps = (0.05, 0.025, 0.0125)
    reference_step = 0.003125
    reference = _run_rz_case(root / "reference", reference_step, memory_limit_mb)
    rows = [
        _run_rz_case(root / f"step-{index}", step, memory_limit_mb)
        for index, step in enumerate(steps)
    ]
    _require_successful_rows([*rows, reference], "RZ time", expected_wall_events=0)
    errors = [_trajectory_errors(row, reference, compare_hit=False) for row in rows]
    self_differences = [
        _trajectory_errors(coarse, fine, compare_hit=False) for coarse, fine in pairwise(rows)
    ]
    _require_decreasing(self_differences, ("position_m", "velocity_m_s"), "RZ self")
    _require_net_reduction(errors, ("position_m", "velocity_m_s"), "RZ fine reference")
    if any(int(row["axis_crossings"]) != 1 for row in [*rows, reference]):
        raise RuntimeError("P14-U RZ convergence case did not cross the axis exactly once")
    parity_path = _materialize_rz_case(root / "parity-input", reference_step, memory_limit_mb)
    parity = _rz_parity_audit(load_case(parity_path))
    if not parity["passes"] or not parity["linear_counterexample_detected"]:
        raise RuntimeError("P14-U RZ input-parity audit did not separate smooth and cusp data")
    return {
        "case": "variable axis-regular Epstein field with one signed-chart axis crossing",
        "time_steps_s": list(steps),
        "reference_step_s": reference_step,
        "fine_reference_errors": errors,
        "fine_reference_trend": "coarse-to-fine net decrease",
        "self_differences": self_differences,
        "self_convergence_orders": _observed_orders(self_differences),
        "axis_crossings": [row["axis_crossings"] for row in rows],
        "event_work": [row["event_work"] for row in rows],
        "failure_count": [row["failure_count"] for row in rows],
        "input_parity_audit": parity,
        "suite_note": f"{suite} uses the same RZ convergence matrix",
    }


def _performance_gate(
    root: Path,
    particle_counts: Sequence[int],
    *,
    repeat_count: int,
    memory_limit_mb: int,
) -> dict[str, object]:
    root.mkdir(parents=True)
    warmup = _materialize_xy_case(
        root / "warmup",
        layout_kind="regular",
        nx=32,
        dt_s=0.125,
        particle_count=_SAMPLE_PARTICLE_COUNT,
        output_mode="none",
        uniform_source=True,
        memory_limit_mb=memory_limit_mb,
        ny=4,
    )
    observations: list[dict[str, object]] = []
    for count in particle_counts:
        for mode in ("none", "sample"):
            for repeat_index in range(repeat_count):
                stem = f"n-{count}-{mode}-r{repeat_index}"
                case_path = _materialize_xy_case(
                    root / "cases" / stem,
                    layout_kind="regular",
                    nx=32,
                    dt_s=0.125,
                    particle_count=int(count),
                    output_mode=mode,
                    uniform_source=True,
                    memory_limit_mb=memory_limit_mb,
                    ny=4,
                )
                worker_spec = root / "worker-specs" / f"{stem}.json"
                _write_worker_spec(
                    worker_spec,
                    operation="timed",
                    case_path=case_path,
                    warmup=warmup,
                    output_path=root / "results" / stem,
                    particle_count=int(count),
                    output_mode=mode,
                    repeat_index=repeat_index,
                )
                observations.append(_launch_worker(worker_spec, root / "numba-cache" / stem))

    profile_count = max(int(value) for value in particle_counts)
    profile_stem = f"n-{profile_count}-none-profile"
    profile_case = _materialize_xy_case(
        root / "cases" / profile_stem,
        layout_kind="regular",
        nx=32,
        dt_s=0.125,
        particle_count=profile_count,
        output_mode="none",
        uniform_source=True,
        memory_limit_mb=memory_limit_mb,
        ny=4,
    )
    profile_spec = root / "worker-specs" / f"{profile_stem}.json"
    _write_worker_spec(
        profile_spec,
        operation="profile",
        case_path=profile_case,
        warmup=warmup,
        output_path=root / "results" / profile_stem,
        particle_count=profile_count,
        output_mode="none",
        repeat_index=None,
    )
    profile = _launch_worker(profile_spec, root / "numba-cache" / profile_stem)
    return {
        "raw_observations": observations,
        "median_observations": _median_performance_observations(observations),
        "profile": profile,
    }


def _write_worker_spec(
    path: Path,
    *,
    operation: str,
    case_path: Path,
    warmup: Path,
    output_path: Path,
    particle_count: int,
    output_mode: _OutputMode,
    repeat_index: int | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "operation": operation,
                "case_path": str(case_path),
                "warmup_case_path": str(warmup),
                "output_path": str(output_path),
                "particle_count": particle_count,
                "output_mode": output_mode,
                "repeat_index": repeat_index,
            },
            allow_nan=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _launch_worker(specification: Path, cache_directory: Path) -> dict[str, object]:
    command = [
        sys.executable,
        "-m",
        "tests.performance.p14u_representative",
        "--worker-spec",
        str(specification),
    ]
    environment = os.environ.copy()
    environment["NUMBA_CACHE_DIR"] = str(cache_directory)
    environment["NUMBA_DISABLE_JIT"] = "0"
    environment["NUMBA_NUM_THREADS"] = "1"
    completed = subprocess.run(
        command, check=False, capture_output=True, text=True, env=environment
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"P14-U worker {specification.stem} failed with exit {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"P14-U worker {specification.stem} returned invalid JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError("P14-U worker returned a non-object")
    return value


def _worker(specification_path: Path) -> None:
    specification = json.loads(specification_path.read_text(encoding="utf-8"))
    warmup_case = load_case(Path(specification["warmup_case_path"]))
    warmup_output = Path(specification["output_path"]).with_name(
        f"{Path(specification['output_path']).name}-warmup"
    )
    simulate(warmup_case, warmup_output)
    del warmup_case
    gc.collect()

    if specification["operation"] == "profile":
        _profile_worker(specification)
        return
    if specification["operation"] != "timed":
        raise RuntimeError(f"unsupported P14-U worker operation: {specification['operation']}")

    peak_before, current_before, rss_source = _process_memory_bytes()
    total_started = time.perf_counter()
    load_started = time.perf_counter()
    case = load_case(Path(specification["case_path"]))
    load_seconds = time.perf_counter() - load_started
    simulate_started = time.perf_counter()
    summary = simulate(case, Path(specification["output_path"]))
    simulate_seconds = time.perf_counter() - simulate_started
    open_started = time.perf_counter()
    result = open_result(Path(specification["output_path"]))
    open_seconds = time.perf_counter() - open_started
    total_seconds = time.perf_counter() - total_started
    peak_after, current_after, _ = _process_memory_bytes()
    final = result.read_final()
    failures = result.read_failure_events()
    if summary.particle_count != int(specification["particle_count"]):
        raise RuntimeError("P14-U worker particle count changed")
    if failures.particle_id.size or not bool((final.lifecycle == 2).all()):
        raise RuntimeError("P14-U representative performance case did not end fully stuck")
    utility = _validate_output_utility(
        result,
        summary,
        particle_count=int(specification["particle_count"]),
        output_mode=str(specification["output_mode"]),
    )
    observation = {
        "particle_count": int(specification["particle_count"]),
        "output_mode": specification["output_mode"],
        "repeat_index": int(specification["repeat_index"]),
        "timing_s": {
            "load_case": load_seconds,
            "simulate": simulate_seconds,
            "open_result": open_seconds,
            "public_end_to_end": total_seconds,
        },
        "nominal_particle_macro_steps_per_simulate_s": (
            int(specification["particle_count"]) * summary.macro_step_count / simulate_seconds
        ),
        "peak_rss_before_bytes": peak_before,
        "peak_rss_after_bytes": peak_after,
        "additional_process_high_water_bytes": max(0, peak_after - peak_before),
        "current_rss_before_bytes": current_before,
        "current_rss_after_bytes": current_after,
        "rss_source": rss_source,
        "rss_semantics": (
            "process high-water since worker start through load/simulate/open; excludes "
            "external result validation; additional=max(0, after-before)"
        ),
        "solver_memory_plan": result.manifest["memory_plan"],
        "event_work": result.manifest["event_refinement"],
        "boundary_interactions": result.manifest["boundary_interactions"],
        "failure_reason_counts": result.manifest["failure_reason_counts"],
        "result_artifact_bytes": _directory_bytes(Path(specification["output_path"])),
        "core_payload_sha256": _core_payload_digest(result),
        "probe_payload_sha256": _probe_payload_digest(result),
        "output_utility": utility,
        "algorithm_revisions": {
            key: value for key, value in result.manifest.items() if key.endswith("_revision")
        },
    }
    print(json.dumps(observation, allow_nan=False, sort_keys=True))


def _profile_worker(specification: Mapping[str, object]) -> None:
    case = load_case(Path(str(specification["case_path"])))
    output_path = Path(str(specification["output_path"]))
    profiler = cProfile.Profile()
    profiler.enable()
    summary = simulate(case, output_path)
    profiler.disable()
    result = open_result(output_path)
    utility = _validate_output_utility(
        result,
        summary,
        particle_count=int(specification["particle_count"]),
        output_mode=str(specification["output_mode"]),
    )
    owners = _profile_owners(profiler)
    if owners is None:
        raise RuntimeError("P14-U profiler did not produce owner statistics")
    largest_owner = next(iter(owners)) if owners else None
    profile = {
        "operation": "separate_untimed_profile",
        "particle_count": int(specification["particle_count"]),
        "output_mode": specification["output_mode"],
        "profile_owner_self_seconds": owners,
        "profile_top_functions": _profile_top_functions(profiler),
        "largest_self_time_owner": largest_owner,
        "owner_specific_conclusion_allowed": (
            largest_owner is not None and largest_owner != "runtime_or_dependency"
        ),
        "core_payload_sha256": _core_payload_digest(result),
        "probe_payload_sha256": _probe_payload_digest(result),
        "algorithm_revisions": {
            key: value for key, value in result.manifest.items() if key.endswith("_revision")
        },
        "event_work": result.manifest["event_refinement"],
        "boundary_interactions": result.manifest["boundary_interactions"],
        "output_utility": utility,
    }
    print(json.dumps(profile, allow_nan=False, sort_keys=True))


def _run_xy_case(
    directory: Path,
    *,
    layout_kind: _LayoutKind,
    nx: int,
    dt_s: float,
    particle_count: int,
    output_mode: _OutputMode,
    uniform_source: bool,
    memory_limit_mb: int,
    probe_times: Sequence[float] | None = None,
) -> dict[str, object]:
    case_path = _materialize_xy_case(
        directory,
        layout_kind=layout_kind,
        nx=nx,
        dt_s=dt_s,
        particle_count=particle_count,
        output_mode=output_mode,
        uniform_source=uniform_source,
        memory_limit_mb=memory_limit_mb,
        probe_times=probe_times,
    )
    output = directory / "result"
    summary = simulate(load_case(case_path), output)
    case = load_case(case_path)
    result = open_result(output)
    return _result_observation(case, result, summary)


def _run_rz_case(directory: Path, dt_s: float, memory_limit_mb: int) -> dict[str, object]:
    case_path = _materialize_rz_case(directory, dt_s, memory_limit_mb)
    case = load_case(case_path)
    output = directory / "result"
    summary = simulate(case, output)
    return _result_observation(case, open_result(output), summary)


def _result_observation(case: Any, result: Any, summary: Any) -> dict[str, object]:
    probes = tuple(result.iter_probes())
    positions = np.stack([probe.position_m[0] for probe in probes])
    velocities = np.stack([probe.velocity_m_s[0] for probe in probes])
    events = result.read_boundary_events()
    target_boundary_id: int | None = None
    target_group: str | None = None
    target_facet_coordinate: float | None = None
    target_facet_clearance: float | None = None
    if events.primary_facet_id.size:
        facet = int(events.primary_facet_id[0])
        boundary = case.data.geometry.boundary
        target_boundary_id = int(boundary.boundary_id[facet])
        target_group = case.data.geometry.group_names[int(boundary.group_id[facet])]
        node_ids = boundary.line2[facet]
        start, end = case.data.geometry.nodes_m[node_ids]
        direction = end - start
        length_squared = float(np.dot(direction, direction))
        hit = np.asarray(events.position_m[0])
        target_facet_coordinate = float(np.dot(hit - start, direction) / length_squared)
        target_facet_clearance = min(target_facet_coordinate, 1.0 - target_facet_coordinate)
    return {
        "probe_times_s": [probe.time_s for probe in probes],
        "probe_position_m": positions,
        "probe_velocity_m_s": velocities,
        "hit_time_s": None if not events.time_s.size else float(events.time_s[0]),
        "hit_position_m": None if not events.time_s.size else events.position_m[0].copy(),
        "hit_facet_id": None if not events.time_s.size else int(events.primary_facet_id[0]),
        "target_boundary_id": target_boundary_id,
        "target_group": target_group,
        "target_facet_coordinate": target_facet_coordinate,
        "target_facet_clearance": target_facet_clearance,
        "event_outcome": None if not events.time_s.size else str(events.outcome[0]),
        "axis_crossings": int(result.manifest["boundary_interactions"]["axis_crossings"]),
        "boundary_event_count": int(events.time_s.size),
        "event_work": result.manifest["event_refinement"],
        "failure_count": summary.failure_event_count,
        "failure_reason_counts": result.manifest["failure_reason_counts"],
        "path_kind": result.manifest["resolved"]["path_kind"],
    }


def _trajectory_errors(
    value: Mapping[str, object],
    reference: Mapping[str, object],
    *,
    compare_hit: bool = True,
    require_same_facet_id: bool = True,
) -> dict[str, float]:
    position = np.asarray(value["probe_position_m"])
    reference_position = np.asarray(reference["probe_position_m"])
    velocity = np.asarray(value["probe_velocity_m_s"])
    reference_velocity = np.asarray(reference["probe_velocity_m_s"])
    error = {
        "position_m": float(np.max(np.abs(position - reference_position))),
        "velocity_m_s": float(np.max(np.abs(velocity - reference_velocity))),
    }
    if not compare_hit:
        return error
    if (
        value["target_boundary_id"] != reference["target_boundary_id"]
        or value["target_group"] != reference["target_group"]
    ):
        raise RuntimeError("P14-U refinement changed the physical hit boundary")
    if require_same_facet_id and value["hit_facet_id"] != reference["hit_facet_id"]:
        raise RuntimeError("P14-U same-mesh time refinement changed the target facet")
    if value["event_outcome"] != "stuck" or reference["event_outcome"] != "stuck":
        raise RuntimeError("P14-U representative hit did not use the target stick law")
    hit_time = value["hit_time_s"]
    reference_hit_time = reference["hit_time_s"]
    hit_position = value["hit_position_m"]
    reference_hit_position = reference["hit_position_m"]
    if not isinstance(hit_time, float) or not isinstance(reference_hit_time, float):
        raise RuntimeError("P14-U representative case did not produce one first hit")
    error["hit_time_s"] = abs(hit_time - reference_hit_time)
    error["hit_position_m"] = float(
        np.max(np.abs(np.asarray(hit_position) - np.asarray(reference_hit_position)))
    )
    return error


def _require_decreasing(
    errors: Sequence[Mapping[str, float]], names: Sequence[str], label: str
) -> None:
    for name in names:
        values = [float(item[name]) for item in errors]
        if not all(first > second > 0.0 for first, second in pairwise(values)):
            raise RuntimeError(f"P14-U {label} {name} errors are not strictly decreasing: {values}")


def _require_net_reduction(
    errors: Sequence[Mapping[str, float]], names: Sequence[str], label: str
) -> None:
    for name in names:
        first = float(errors[0][name])
        last = float(errors[-1][name])
        if not 0.0 < last < first:
            raise RuntimeError(
                f"P14-U {label} {name} did not decrease from coarse to fine: {first} -> {last}"
            )


def _require_successful_rows(
    rows: Sequence[Mapping[str, object]],
    label: str,
    *,
    expected_wall_events: int = 1,
) -> None:
    for row in rows:
        if int(row["failure_count"]) != 0:
            raise RuntimeError(
                f"P14-U {label} case produced a particle failure: "
                f"reasons={row['failure_reason_counts']}, event_work={row['event_work']}"
            )
        if int(row["boundary_event_count"]) != expected_wall_events:
            raise RuntimeError(
                f"P14-U {label} case produced {row['boundary_event_count']} wall events; "
                f"expected {expected_wall_events}"
            )
        if row["path_kind"] != "rk4_dense":
            raise RuntimeError(f"P14-U {label} case did not use the general RK4 path")
        if expected_wall_events:
            clearance = row["target_facet_clearance"]
            if not isinstance(clearance, float) or clearance < _TARGET_FACET_MIN_CLEARANCE:
                raise RuntimeError(
                    f"P14-U {label} hit is confounded by a target facet endpoint: "
                    f"normalized clearance={clearance}"
                )


def _observed_orders(errors: Sequence[Mapping[str, float]]) -> dict[str, list[float]]:
    return {
        name: [
            math.log2(float(first[name]) / float(second[name]))
            for first, second in pairwise(errors)
        ]
        for name in errors[0]
        if all(float(item[name]) > 0.0 for item in errors)
    }


def _hit_summary(value: Mapping[str, object]) -> dict[str, object]:
    return {
        "time_s": value["hit_time_s"],
        "position_m": np.asarray(value["hit_position_m"]).tolist(),
        "facet_id_within_reference_mesh": value["hit_facet_id"],
        "boundary_id": value["target_boundary_id"],
        "boundary_group": value["target_group"],
        "normalized_facet_coordinate": value["target_facet_coordinate"],
        "normalized_facet_endpoint_clearance": value["target_facet_clearance"],
        "outcome": value["event_outcome"],
    }


def _global_to_path_ratio(reference: Mapping[str, object]) -> dict[str, object]:
    x = np.linspace(0.0, 1.0, 257)
    y = np.linspace(0.0, 1.0, 129)
    grid_x, grid_y = np.meshgrid(x, y, indexing="ij")
    global_norm = np.linalg.norm(_xy_electric_acceleration(grid_x, grid_y), axis=-1)
    path = np.asarray(reference["probe_position_m"])
    path_norm = np.linalg.norm(_xy_electric_acceleration(path[:, 0], path[:, 1]), axis=-1)
    global_max = float(np.max(global_norm))
    path_max = float(np.max(path_norm))
    times = np.asarray(reference["probe_times_s"], dtype=np.float64)
    hit_time = reference["hit_time_s"]
    if (
        times.size < 100
        or times[0] != 0.0
        or not isinstance(hit_time, float)
        or not bool(np.any(times == hit_time))
    ):
        raise RuntimeError("P14-U path-field audit does not densely cover source and exact hit")
    ratio = global_max / path_max
    if ratio <= 1.1:
        raise RuntimeError("P14-U representative field is not materially stronger off path")
    return {
        "global_min_m_s2": float(np.min(global_norm)),
        "global_max_m_s2": global_max,
        "dense_path_max_m_s2": path_max,
        "global_max_over_dense_path_max": ratio,
        "dense_probe_count": int(times.size),
        "includes_source_time": True,
        "includes_exact_localized_hit_time": True,
    }


def _dense_probe_times(step_s: float, hit_time_s: float) -> tuple[float, ...]:
    count = round(_XY_END_S / step_s)
    regular = [index * step_s for index in range(count + 1)]
    return tuple(sorted({*regular, hit_time_s, _XY_END_S}))


def _materialize_xy_case(
    directory: Path,
    *,
    layout_kind: _LayoutKind,
    nx: int,
    dt_s: float,
    particle_count: int,
    output_mode: _OutputMode,
    uniform_source: bool,
    memory_limit_mb: int,
    probe_times: Sequence[float] | None = None,
    ny: int | None = None,
) -> Path:
    directory.mkdir(parents=True)
    y_cells = nx if ny is None else ny
    data, specification = _xy_case_material(
        layout_kind, nx, y_cells, particle_count, uniform_source
    )
    data_path = directory / "case.h5"
    info = write(data_path, data)
    specification["case"] = {
        "name": f"p14u_xy_{layout_kind}_{nx}x{y_cells}_{particle_count}",
        "data_path": data_path.name,
        "expected_content_hash": info.content_hash,
    }
    specification["time"] = {"start_s": 0.0, "end_s": _XY_END_S, "dt_s": dt_s}
    specification["resources"] = {"memory_limit_mb": memory_limit_mb}
    specification["output"] = _output_spec(
        output_mode,
        particle_count,
        _XY_SYNC_TIMES if probe_times is None else probe_times,
    )
    case_path = directory / "case.yaml"
    case_path.write_text(yaml.safe_dump(specification, sort_keys=False), encoding="utf-8")
    return case_path


def _xy_case_material(
    layout_kind: _LayoutKind,
    nx: int,
    ny: int,
    particle_count: int,
    uniform_source: bool,
) -> tuple[DataBundle, dict[str, Any]]:
    drag = build_microcase("C02")
    electric = build_microcase("C04")
    geometry, layout = _xy_geometry_and_layout(layout_kind, nx, ny)
    nodes = _layout_nodes(layout)
    drag_fields = {field.name: field for field in drag.data.fields}
    electric_particle = electric.data.sources[0]
    drag_particle = drag.data.sources[0]
    charge_number = float(electric_particle.charge_number[0])
    mass_kg = float(drag_particle.mass_kg[0])
    charge_to_acceleration = charge_number * 1.602176634e-19 / mass_kg
    acceleration = _xy_electric_acceleration(nodes[:, 0], nodes[:, 1])
    fields = (
        _field("gas_velocity", layout.name, ("x", "y"), "cartesian_xy", "m/s", nodes, [0.15, 0.0]),
        _copied_scalar_field(drag_fields["gas_density"], layout.name, nodes.shape[0]),
        _copied_scalar_field(drag_fields["gas_temperature"], layout.name, nodes.shape[0]),
        _copied_scalar_field(drag_fields["gas_mean_free_path"], layout.name, nodes.shape[0]),
        FieldData(
            "electric_field",
            layout.name,
            "node",
            ("x", "y"),
            "cartesian_xy",
            np.asarray(acceleration / charge_to_acceleration, dtype="<f8"),
            "V/m",
        ),
    )
    particle = {
        "charge_number": charge_number,
        "mass_kg": mass_kg,
        "drag_diameter_m": float(drag_particle.drag_diameter_m[0]),
        "electrostatic_radius_m": float(drag_particle.electrostatic_radius_m[0]),
        "displaced_volume_m3": 0.0,
        "model_weight": 1.0,
        "material_id": int(drag_particle.material_id[0]),
    }
    source_index = math.floor(_XY_SOURCE_Y_M * ny)
    position = (
        {"model": "uniform", "measure": "line_length"}
        if uniform_source
        else {
            "model": "edge_fraction",
            "fraction": source_index + 1.0 - _XY_SOURCE_Y_M * ny,
        }
    )
    source = {
        "name": "part_surface_release",
        "type": "surface",
        "boundary_group": "source",
        "count": particle_count,
        "particle_id_start": 1,
        "particle": particle,
        "position": position,
        "velocity": {"model": "fixed", "value_m_s": [0.2, 0.0]},
        "release": {"model": "fixed", "time_s": 0.0},
    }
    specification = copy.deepcopy(drag.spec)
    specification["motion"] = {"mode": "cartesian_xy"}
    specification["sources"] = [source]
    specification["boundaries"] = [
        {"boundary_group": "caps", "priority": 10, "law": "escape"},
        {"boundary_group": "target", "priority": 10, "law": "stick"},
        {"boundary_group": "source", "priority": 10, "law": "stick"},
    ]
    specification["physics"] = copy.deepcopy(drag.spec["physics"])
    specification["physics"]["electric"] = copy.deepcopy(electric.spec["physics"]["electric"])
    specification["physics"]["gravity_buoyancy"] = {
        "model": "standard",
        "revision": "gravity_buoyancy_standard_v1",
        "gas_density_field": "gas_density",
        "gravity_m_s2": [0.05, 0.0],
    }
    return (
        replace(
            drag.data,
            coordinate_system="cartesian_xy",
            geometry=geometry,
            layouts=(layout,),
            fields=fields,
            sources=(),
        ),
        specification,
    )


def _xy_electric_acceleration(x: Any, y: Any) -> np.ndarray:
    x_value = np.asarray(x, dtype=np.float64)
    y_value = np.asarray(y, dtype=np.float64)
    offset = y_value - _XY_EQUILIBRIUM_Y_M
    coupling = 1.0 + 0.6 * np.sin(math.pi * x_value)
    axial = 0.15 + 1.2 * np.exp((x_value - 1.0) / 0.12)
    acceleration_x = axial - 0.6 * math.pi * offset * offset * np.cos(math.pi * x_value)
    acceleration_y = -2.0 * offset * coupling
    return np.stack((acceleration_x, acceleration_y), axis=-1)


def _xy_geometry_and_layout(
    kind: _LayoutKind, nx: int, ny: int
) -> tuple[GeometryData, RegularLayout | P1TriLayout | Q1QuadLayout]:
    x = np.linspace(0.0, 1.0, nx + 1, dtype=np.float64)
    y = np.linspace(0.0, 1.0, ny + 1, dtype=np.float64)
    nodes = np.asarray([[x_value, y_value] for x_value in x for y_value in y], dtype="<f8")
    quads = _quad_connectivity(nx, ny)
    boundary = _xy_boundary(kind, nx, ny)
    if kind == "p1":
        triangles = _triangle_connectivity(nx, ny)
        geometry = GeometryData(
            nodes,
            boundary,
            ("caps", "target", "source"),
            tri3=triangles,
            tri3_domain_id=np.zeros(triangles.shape[0], dtype="<i4"),
        )
        return geometry, P1TriLayout(
            "field", nodes.copy(), triangles.copy(), np.ones(triangles.shape[0], dtype="<u1")
        )
    geometry = GeometryData(
        nodes,
        boundary,
        ("caps", "target", "source"),
        quad4=quads,
        quad4_domain_id=np.zeros(quads.shape[0], dtype="<i4"),
    )
    if kind == "q1":
        return geometry, Q1QuadLayout(
            "field", nodes.copy(), quads.copy(), np.ones(quads.shape[0], dtype="<u1")
        )
    return geometry, RegularLayout("field", x, y, np.ones((nx, ny), dtype="<u1"))


def _xy_boundary(kind: _LayoutKind, nx: int, ny: int) -> BoundaryData:
    facets: list[list[int]] = []
    groups: list[int] = []
    owners: list[int] = []
    for index_x in range(nx):
        facets.append([_node_id(index_x, 0, ny), _node_id(index_x + 1, 0, ny)])
        groups.append(0)
        owners.append(_owner_id(kind, index_x, 0, ny, 0))
    for index_y in range(ny):
        facets.append([_node_id(nx, index_y, ny), _node_id(nx, index_y + 1, ny)])
        groups.append(1)
        owners.append(_owner_id(kind, nx - 1, index_y, ny, 0))
    for index_x in range(nx - 1, -1, -1):
        facets.append([_node_id(index_x + 1, ny, ny), _node_id(index_x, ny, ny)])
        groups.append(0)
        owners.append(_owner_id(kind, index_x, ny - 1, ny, 1))
    source_index = math.floor(_XY_SOURCE_Y_M * ny)
    for index_y in range(ny - 1, -1, -1):
        facets.append([_node_id(0, index_y + 1, ny), _node_id(0, index_y, ny)])
        groups.append(2 if index_y == source_index else 0)
        owners.append(_owner_id(kind, 0, index_y, ny, 1))
    count = len(facets)
    group_array = np.asarray(groups, dtype="<i4")
    boundary_ids = np.choose(group_array, np.asarray([10, 20, 30], dtype="<i4"))
    return BoundaryData(
        np.asarray(facets, dtype="<i8"),
        boundary_ids,
        group_array,
        np.zeros(count, dtype="<i4"),
        np.full(count, 1 if kind == "p1" else 2, dtype="<u1"),
        np.asarray(owners, dtype="<i8"),
        np.ones(count, dtype="<i1"),
    )


def _owner_id(kind: _LayoutKind, index_x: int, index_y: int, ny: int, triangle: int) -> int:
    cell = index_x * ny + index_y
    return 2 * cell + triangle if kind == "p1" else cell


def _node_id(index_x: int, index_y: int, ny: int) -> int:
    return index_x * (ny + 1) + index_y


def _quad_connectivity(nx: int, ny: int) -> np.ndarray:
    cells = [
        [
            _node_id(index_x, index_y, ny),
            _node_id(index_x + 1, index_y, ny),
            _node_id(index_x + 1, index_y + 1, ny),
            _node_id(index_x, index_y + 1, ny),
        ]
        for index_x in range(nx)
        for index_y in range(ny)
    ]
    return np.asarray(cells, dtype="<i8")


def _triangle_connectivity(nx: int, ny: int) -> np.ndarray:
    cells: list[list[int]] = []
    for index_x in range(nx):
        for index_y in range(ny):
            lower_left = _node_id(index_x, index_y, ny)
            lower_right = _node_id(index_x + 1, index_y, ny)
            upper_right = _node_id(index_x + 1, index_y + 1, ny)
            upper_left = _node_id(index_x, index_y + 1, ny)
            cells.extend(
                ([lower_left, lower_right, upper_right], [lower_left, upper_right, upper_left])
            )
    return np.asarray(cells, dtype="<i8")


def _layout_nodes(layout: RegularLayout | P1TriLayout | Q1QuadLayout) -> np.ndarray:
    if isinstance(layout, RegularLayout):
        return np.asarray(
            [[first, second] for first in layout.axis0_m for second in layout.axis1_m],
            dtype="<f8",
        )
    return layout.nodes_m


def _field(
    name: str,
    layout: str,
    components: tuple[str, ...],
    basis: str,
    unit: str,
    nodes: np.ndarray,
    value: Sequence[float],
) -> FieldData:
    values = np.repeat(np.asarray(value, dtype="<f8")[None, :], nodes.shape[0], axis=0)
    return FieldData(name, layout, "node", components, basis, values, unit)


def _copied_scalar_field(field: FieldData, layout: str, count: int) -> FieldData:
    return replace(
        field, layout=layout, values=np.full((count, 1), float(field.values[0, 0]), dtype="<f8")
    )


def _materialize_rz_case(directory: Path, dt_s: float, memory_limit_mb: int) -> Path:
    directory.mkdir(parents=True)
    drag = build_microcase("C02")
    radial = np.linspace(0.0, 1.0, 65, dtype=np.float64)
    axial = np.linspace(-2.0, 2.0, 17, dtype=np.float64)
    layout = RegularLayout("field", radial, axial, np.ones((64, 16), dtype="<u1"))
    nodes = _layout_nodes(layout)
    base_fields = {field.name: field for field in drag.data.fields}
    density0 = float(base_fields["gas_density"].values[0, 0])
    temperature0 = float(base_fields["gas_temperature"].values[0, 0])
    path0 = float(base_fields["gas_mean_free_path"].values[0, 0])
    r = nodes[:, 0]
    z = nodes[:, 1]
    fields = (
        FieldData(
            "gas_velocity",
            "field",
            "node",
            ("r", "z"),
            "axisymmetric_rz",
            np.column_stack((0.1 * r, 0.02 * r * r - 0.03 * z)).astype("<f8"),
            "m/s",
        ),
        FieldData(
            "gas_density",
            "field",
            "node",
            ("value",),
            "scalar",
            (density0 * (1.0 + 0.2 * r * r))[:, None].astype("<f8"),
            "kg/m^3",
        ),
        FieldData(
            "gas_temperature",
            "field",
            "node",
            ("value",),
            "scalar",
            (temperature0 * (1.0 + 0.02 * r * r))[:, None].astype("<f8"),
            "K",
        ),
        FieldData(
            "gas_mean_free_path",
            "field",
            "node",
            ("value",),
            "scalar",
            (path0 * (1.0 + 0.05 * r * r))[:, None].astype("<f8"),
            "m",
        ),
    )
    geometry = GeometryData(
        np.asarray([[0.0, -2.0], [1.0, -2.0], [1.0, 2.0], [0.0, 2.0]], dtype="<f8"),
        _empty_boundary(),
        (),
        quad4=np.asarray([[0, 1, 2, 3]], dtype="<i8"),
        quad4_domain_id=np.zeros(1, dtype="<i4"),
    )
    original = drag.data.sources[0]
    source = replace(
        original,
        particle_id=np.asarray([1], dtype="<i8"),
        release_time_s=np.asarray([0.0], dtype="<f8"),
        position_m=np.asarray([[0.2, 0.0]], dtype="<f8"),
        velocity_m_s=np.asarray([[-0.8, 0.05]], dtype="<f8"),
        charge_number=np.asarray([0.0], dtype="<f8"),
    )
    data = replace(
        drag.data,
        coordinate_system="axisymmetric_rz",
        geometry=geometry,
        layouts=(layout,),
        fields=fields,
        sources=(source,),
    )
    data_path = directory / "case.h5"
    info = write(data_path, data)
    specification = copy.deepcopy(drag.spec)
    specification["case"] = {
        "name": "p14u_variable_axis_regular_rz",
        "data_path": data_path.name,
        "expected_content_hash": info.content_hash,
    }
    specification["motion"] = {"mode": "axisymmetric_rz_meridional"}
    specification["time"] = {"start_s": 0.0, "end_s": _RZ_END_S, "dt_s": dt_s}
    specification["sources"] = [{"name": "release", "type": "table", "table": source.name}]
    specification["boundaries"] = []
    specification["physics"] = copy.deepcopy(drag.spec["physics"])
    specification["physics"]["gravity_buoyancy"] = {
        "model": "standard",
        "revision": "gravity_buoyancy_standard_v1",
        "gas_density_field": "gas_density",
        "gravity_m_s2": [0.0, -9.81],
    }
    specification["resources"] = {"memory_limit_mb": memory_limit_mb}
    specification["output"] = _output_spec("sample", 1, _RZ_SYNC_TIMES)
    case_path = directory / "case.yaml"
    case_path.write_text(yaml.safe_dump(specification, sort_keys=False), encoding="utf-8")
    return case_path


def _empty_boundary() -> BoundaryData:
    return BoundaryData(
        np.empty((0, 2), dtype="<i8"),
        np.empty(0, dtype="<i4"),
        np.empty(0, dtype="<i4"),
        np.empty(0, dtype="<i4"),
        np.empty(0, dtype="<u1"),
        np.empty(0, dtype="<i8"),
        np.empty(0, dtype="<i1"),
    )


def _output_spec(
    mode: _OutputMode, particle_count: int, times: Sequence[float]
) -> dict[str, object]:
    probes: dict[str, object] | None = None
    if mode == "sample":
        probes = {
            "particle_ids": list(range(1, min(particle_count, _SAMPLE_PARTICLE_COUNT) + 1)),
            "schedule": {"explicit_times_s": list(times)},
        }
    return {"trajectories": None, "probes": probes}


def _rz_parity_audit(case: Any) -> dict[str, object]:
    required_names = {
        str(value)
        for model in case.spec.physics.models.values()
        for key, value in model.items()
        if key.endswith("_field")
    }
    fields = {field.name: field for field in case.data.fields}
    missing = sorted(required_names - fields.keys())
    if missing:
        raise RuntimeError(f"P14-U RZ parity audit is missing required fields: {missing}")
    layouts = {layout.name: layout for layout in case.data.layouts}
    audits: dict[str, object] = {}
    passes = True
    counterexample_values: np.ndarray | None = None
    counterexample_radial: np.ndarray | None = None
    counterexample_tolerance: float | None = None
    for name in sorted(required_names):
        field = fields[name]
        layout = layouts[field.layout]
        if not isinstance(layout, RegularLayout):
            raise RuntimeError(f"P14-U RZ parity audit requires regular layout for {name}")
        radial = np.asarray(layout.axis0_m)
        axial_count = int(layout.axis1_m.size)
        values = np.asarray(field.values).reshape(radial.size, axial_count, -1)
        scale = max(float(np.max(np.abs(values))), 1.0)
        tolerance = 256.0 * np.finfo(np.float64).eps * scale / float(radial[1] - radial[0])
        if field.stored_basis == "scalar":
            residual = _axis_derivative_residual(radial, values[:, :, 0])
            field_passes = residual <= tolerance
            audits[name] = {
                "kind": "scalar_even",
                "axis_derivative_max_abs": residual,
                "tolerance": tolerance,
                "passes": field_passes,
            }
            if counterexample_values is None:
                counterexample_values = values[:, :, 0]
                counterexample_radial = radial
                counterexample_tolerance = tolerance
        elif field.stored_basis == "axisymmetric_rz" and field.components == ("r", "z"):
            radial_axis = float(np.max(np.abs(values[0, :, 0])))
            axial_residual = _axis_derivative_residual(radial, values[:, :, 1])
            field_passes = radial_axis == 0.0 and axial_residual <= tolerance
            audits[name] = {
                "kind": "radial_odd_axial_even",
                "radial_component_axis_max_abs": radial_axis,
                "axial_component_axis_derivative_max_abs": axial_residual,
                "tolerance": tolerance,
                "passes": field_passes,
            }
        else:
            raise RuntimeError(
                f"P14-U RZ parity audit does not understand {name} basis/components "
                f"{field.stored_basis}/{field.components}"
            )
        passes = passes and field_passes

    gravity = case.spec.physics.models.get("gravity_buoyancy")
    if gravity is None:
        raise RuntimeError("P14-U RZ parity audit requires configured standard gravity")
    gravity_vector = gravity.get("gravity_m_s2")
    gravity_radial_zero = (
        isinstance(gravity_vector, tuple)
        and len(gravity_vector) == 2
        and float(gravity_vector[0]) == 0.0
    )
    passes = passes and gravity_radial_zero
    if (
        counterexample_values is None
        or counterexample_radial is None
        or counterexample_tolerance is None
    ):
        raise RuntimeError("P14-U RZ parity audit has no scalar counterexample base")
    counterexample = counterexample_values + 0.25 * counterexample_radial[:, None]
    counterexample_residual = _axis_derivative_residual(counterexample_radial, counterexample)
    return {
        "method": "loaded canonical required fields; three-node axis derivative",
        "required_field_names": sorted(required_names),
        "fields": audits,
        "standard_gravity_radial_component_m_s2": float(gravity_vector[0]),
        "standard_gravity_radial_zero": gravity_radial_zero,
        "linear_counterexample_derivative_max_abs": counterexample_residual,
        "linear_counterexample_tolerance": counterexample_tolerance,
        "passes": passes,
        "linear_counterexample_detected": bool(counterexample_residual > counterexample_tolerance),
    }


def _axis_derivative_residual(radial: np.ndarray, values: np.ndarray) -> float:
    h = float(radial[1] - radial[0])
    if not np.allclose(np.diff(radial[:3]), h, rtol=0.0, atol=0.0):
        raise RuntimeError("P14-U parity audit requires the first three uniform radial nodes")
    derivative = (-3.0 * values[0] + 4.0 * values[1] - values[2]) / (2.0 * h)
    return float(np.max(np.abs(derivative)))


def _validate_output_utility(
    result: Any,
    summary: Any,
    *,
    particle_count: int,
    output_mode: str,
) -> dict[str, object]:
    expected_ids = np.arange(1, particle_count + 1, dtype=np.int64)
    releases = result.read_release_events()
    boundaries = result.read_boundary_events()
    probes = tuple(result.iter_probes())
    if (
        summary.release_event_count != particle_count
        or releases.particle_id.size != particle_count
        or not np.array_equal(releases.particle_id, expected_ids)
        or not bool((releases.time_s == 0.0).all())
    ):
        raise RuntimeError("P14-U output utility release events are not exact")
    if (
        summary.boundary_event_count != particle_count
        or boundaries.particle_id.size != particle_count
        or not np.array_equal(np.sort(boundaries.particle_id), expected_ids)
        or not bool((boundaries.boundary_id == 20).all())
        or not bool((boundaries.law_id == "stick").all())
        or not bool((boundaries.outcome == "stuck").all())
    ):
        raise RuntimeError("P14-U output utility target stick events are not exact")

    expected_probe_ids = expected_ids[: min(particle_count, _SAMPLE_PARTICLE_COUNT)]
    if output_mode == "none":
        if probes or summary.probe_count != 0 or summary.probe_row_count != 0:
            raise RuntimeError("P14-U none output unexpectedly contains probe rows")
    elif output_mode == "sample":
        if (
            len(probes) != len(_XY_SYNC_TIMES)
            or summary.probe_count != len(_XY_SYNC_TIMES)
            or summary.probe_row_count != len(_XY_SYNC_TIMES) * expected_probe_ids.size
            or [frame.time_s for frame in probes] != list(_XY_SYNC_TIMES)
            or any(not np.array_equal(frame.particle_id, expected_probe_ids) for frame in probes)
        ):
            raise RuntimeError("P14-U sample output does not contain the exact requested grid")
    else:
        raise RuntimeError(f"unsupported P14-U output mode: {output_mode}")
    return {
        "release_events": int(releases.particle_id.size),
        "target_stick_events": int(boundaries.particle_id.size),
        "probe_frames": len(probes),
        "probe_rows": sum(int(frame.particle_id.size) for frame in probes),
        "exact_requested_ids_and_times": True,
    }


def _median_performance_observations(
    observations: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[int, str], list[Mapping[str, object]]] = {}
    for observation in observations:
        key = (int(observation["particle_count"]), str(observation["output_mode"]))
        grouped.setdefault(key, []).append(observation)
    rows: list[dict[str, object]] = []
    for (count, mode), group in sorted(grouped.items()):
        first = group[0]
        timing_names = tuple(_as_mapping(first["timing_s"]))
        rows.append(
            {
                "particle_count": count,
                "output_mode": mode,
                "repeat_count": len(group),
                "timing_s": {
                    name: median(float(_as_mapping(item["timing_s"])[name]) for item in group)
                    for name in timing_names
                },
                "nominal_particle_macro_steps_per_simulate_s": median(
                    float(item["nominal_particle_macro_steps_per_simulate_s"]) for item in group
                ),
                "peak_rss_after_bytes": median(int(item["peak_rss_after_bytes"]) for item in group),
                "additional_process_high_water_bytes": median(
                    int(item["additional_process_high_water_bytes"]) for item in group
                ),
                "current_rss_after_bytes": _median_optional_int(
                    [item["current_rss_after_bytes"] for item in group]
                ),
                "result_artifact_bytes": median(
                    int(item["result_artifact_bytes"]) for item in group
                ),
                "core_payload_sha256": first["core_payload_sha256"],
                "probe_payload_sha256": first["probe_payload_sha256"],
                "algorithm_revisions": first["algorithm_revisions"],
                "event_work": first["event_work"],
                "boundary_interactions": first["boundary_interactions"],
                "output_utility": first["output_utility"],
                "solver_memory_plan": first["solver_memory_plan"],
            }
        )
    return rows


def _as_mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RuntimeError("P14-U expected a mapping in an observation")
    return value


def _median_optional_int(values: Sequence[object]) -> float | None:
    present = [int(value) for value in values if value is not None]
    return None if not present else float(median(present))


def _validate_performance_identity(performance: Mapping[str, object]) -> None:
    raw_value = performance.get("raw_observations")
    profile = performance.get("profile")
    if not isinstance(raw_value, list) or not isinstance(profile, Mapping):
        raise RuntimeError("P14-U performance report is incomplete")
    observations = [_as_mapping(item) for item in raw_value]
    by_count: dict[int, dict[str, list[Mapping[str, object]]]] = {}
    for observation in observations:
        count = int(observation["particle_count"])
        mode = str(observation["output_mode"])
        by_count.setdefault(count, {}).setdefault(mode, []).append(observation)
    for count, modes in by_count.items():
        if set(modes) != {"none", "sample"}:
            raise RuntimeError(f"P14-U output comparison is incomplete for {count} particles")
        repeat_counts = {len(group) for group in modes.values()}
        if len(repeat_counts) != 1:
            raise RuntimeError(f"P14-U repeat matrix is unbalanced for {count} particles")
        expected_repeats = set(range(next(iter(repeat_counts))))
        for mode, group in modes.items():
            actual_repeats = {int(item["repeat_index"]) for item in group}
            if actual_repeats != expected_repeats:
                raise RuntimeError(f"P14-U {count}/{mode} repeat indices are incomplete")
            probe_digests = {str(item["probe_payload_sha256"]) for item in group}
            if len(probe_digests) != 1:
                raise RuntimeError(f"P14-U {count}/{mode} repeats changed the probe payload")
        all_rows = [*modes["none"], *modes["sample"]]
        for key in (
            "core_payload_sha256",
            "algorithm_revisions",
            "event_work",
            "boundary_interactions",
            "failure_reason_counts",
        ):
            fingerprints = {
                json.dumps(item[key], allow_nan=False, sort_keys=True) for item in all_rows
            }
            if len(fingerprints) != 1:
                raise RuntimeError(f"P14-U output mode/repeat changed {key} for {count} particles")

    profile_count = int(profile["particle_count"])
    if profile_count != max(by_count) or profile["output_mode"] != "none":
        raise RuntimeError("P14-U profile is not the explicit maximum-count none case")
    baseline = by_count[profile_count]["none"][0]
    for key in (
        "core_payload_sha256",
        "probe_payload_sha256",
        "algorithm_revisions",
        "event_work",
        "boundary_interactions",
        "output_utility",
    ):
        if json.dumps(profile[key], sort_keys=True) != json.dumps(baseline[key], sort_keys=True):
            raise RuntimeError(f"P14-U separate profile changed {key}")


def _core_payload_digest(result: Any) -> str:
    digest = hashlib.sha256()
    for name, reader in (
        ("final", result.read_final),
        ("release", result.read_release_events),
        ("boundary", result.read_boundary_events),
        ("failure", result.read_failure_events),
        ("series", result.read_lifecycle_series),
    ):
        _digest_dataclass(digest, name, reader())
    return f"sha256:{digest.hexdigest()}"


def _probe_payload_digest(result: Any) -> str:
    digest = hashlib.sha256()
    for index, probe in enumerate(result.iter_probes()):
        _digest_dataclass(digest, f"probe-{index}", probe)
    return f"sha256:{digest.hexdigest()}"


def _profile_owners(profile: cProfile.Profile | None) -> dict[str, float] | None:
    if profile is None:
        return None
    owners: dict[str, float] = {}
    for (filename, _line, _name), statistics in pstats.Stats(profile).stats.items():
        owner = _profile_owner(filename)
        owners[owner] = owners.get(owner, 0.0) + float(statistics[2])
    return dict(sorted(owners.items(), key=lambda item: item[1], reverse=True))


def _profile_top_functions(profile: cProfile.Profile, limit: int = 20) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for (filename, line, name), statistics in pstats.Stats(profile).stats.items():
        primitive_calls, total_calls, self_seconds, cumulative_seconds, _callers = statistics
        rows.append(
            {
                "source": _profile_source(filename),
                "line": line,
                "function": name,
                "primitive_calls": primitive_calls,
                "total_calls": total_calls,
                "self_seconds": float(self_seconds),
                "cumulative_seconds": float(cumulative_seconds),
            }
        )
    return {
        "top_by_self_seconds": sorted(
            rows,
            key=lambda item: (-float(item["self_seconds"]), str(item["source"]), int(item["line"])),
        )[:limit],
        "top_by_cumulative_seconds": sorted(
            rows,
            key=lambda item: (
                -float(item["cumulative_seconds"]),
                str(item["source"]),
                int(item["line"]),
            ),
        )[:limit],
    }


def _profile_source(filename: str) -> str:
    normalized = filename.replace("\\", "/")
    for marker in ("/chamber_particles/", "/tests/performance/"):
        if marker in normalized:
            return marker.strip("/") + "/" + normalized.split(marker, 1)[1]
    if normalized.startswith("{") or normalized.startswith("~"):
        return normalized
    return f"runtime_or_dependency/{normalized.rsplit('/', 1)[-1]}"


def _profile_owner(filename: str) -> str:
    normalized = filename.replace("\\", "/")
    marker = "/chamber_particles/"
    if marker not in normalized:
        return "runtime_or_dependency"
    relative = normalized.split(marker, 1)[1]
    if relative.startswith("physics/"):
        return "physics"
    return relative.split("/", 1)[0].removesuffix(".py")


if __name__ == "__main__":
    main()
