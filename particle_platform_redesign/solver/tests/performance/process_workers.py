"""Measure the current public engine on one generated representative input.

Input construction, untimed warmups, timed runs, independent outcome checks and
scientific identity are separate operations. The retained input is editable;
there is no historical-result reader or alternate execution policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import statistics
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, cast

import numpy as np

from chamber_particles import load_case, open_result, simulate
from tests.performance.p09_memory import _machine_metadata
from tests.performance.p14_matrix import _scientific_payload_digest
from tests.performance.p14u_representative import _materialize_xy_case, _validate_output_utility


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_identity() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    paths = [root / "uv.lock", *sorted((root / "src").rglob("*.py"))]
    paths.extend(
        root / "tests" / "performance" / name
        for name in (
            "process_workers.py",
            "p14u_representative.py",
            "p14_matrix.py",
            "p09_memory.py",
            "p06_baseline.py",
        )
    )
    paths.append(root / "tests" / "verification" / "microcases.py")
    return {path.relative_to(root).as_posix(): _sha256(path) for path in paths}


def _validate_result(
    result: Any, summary: Any, particles: int, macro_steps: int, output_mode: str
) -> None:
    """Check known physical target and conservation, independently of run equality."""
    if summary.particle_count != particles or summary.macro_step_count != macro_steps:
        raise RuntimeError("generated workload changed particle or macro count")
    if summary.failure_event_count:
        raise RuntimeError("representative workload produced a particle failure")
    _validate_output_utility(result, summary, particle_count=particles, output_mode=output_mode)
    final = result.read_final()
    events = result.read_boundary_events()
    # The generated square has source x=0 and the absorbing target x=1.
    # Positive x forcing carries each particle to that target, once, before 4s.
    if not (
        np.all(final.lifecycle == 2)
        and np.all(events.outcome == "stuck")
        and np.allclose(final.position_m[:, 0], 1.0, rtol=0.0, atol=1e-12)
        and np.all((final.position_m[:, 1] > 0.0) & (final.position_m[:, 1] < 1.0))
        and np.all((events.time_s > 0.0) & (events.time_s < 4.0))
    ):
        raise RuntimeError("representative outcome violated the known absorbing-target problem")
    if multiprocessing.active_children():
        raise RuntimeError("simulate returned with live worker children")


def _observation(
    case_path: Path, output: Path, workers: int, particles: int, steps: int, output_mode: str
) -> dict[str, Any]:
    started = time.perf_counter()
    case = load_case(case_path)
    loaded = time.perf_counter()
    summary = simulate(case, output, workers=workers)
    simulated = time.perf_counter()
    result = open_result(output)
    opened = time.perf_counter()
    roster = cast(Mapping[str, str], result.source_identity()["content_files"])
    for name, expected in roster.items():
        with (output / name).open("rb") as stream:
            actual = "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise RuntimeError(f"payload byte hash changed: {name}")
    byte_hashed = time.perf_counter()
    _validate_result(result, summary, particles, steps, output_mode)
    validated = time.perf_counter()
    scientific_hash = _scientific_payload_digest(result)
    science_hashed = time.perf_counter()
    verification = result.verify_contents()
    verified = time.perf_counter()
    manifest = result.manifest_document()
    return {
        "workers": workers,
        "timing_s": {
            "load_case": loaded - started,
            "simulate": simulated - loaded,
            "open_result": opened - simulated,
            "public_end_to_end": opened - started,
            "payload_sha256_io": byte_hashed - opened,
            "outcome_validation_and_reads": validated - byte_hashed,
            "scientific_payload_hashing_and_reads": science_hashed - validated,
            "verify_contents": verified - science_hashed,
        },
        "full_scientific_payload_sha256": scientific_hash,
        "content_verification": verification,
        "solver_memory_plan": manifest["memory_plan"],
        "boundary_events": summary.boundary_event_count,
        "failure_events": summary.failure_event_count,
        "live_children_after_return": 0,
        "known_target_and_lifecycle_checked": True,
        "algorithm_revisions": {
            key: value for key, value in manifest.items() if key.endswith("_revision")
        },
    }


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_before = _source_identity()
    case_path = _materialize_xy_case(
        output / "input",
        layout_kind="regular",
        nx=32,
        ny=4,
        dt_s=4.0 / arguments.macro_steps,
        particle_count=arguments.particles,
        output_mode=arguments.output_mode,
        uniform_source=True,
        memory_limit_mb=arguments.memory_limit_mb,
    )
    input_identity = {name: _sha256(case_path.with_name(name)) for name in ("case.yaml", "case.h5")}
    # One serial warmup populates the persistent disk cache. Every simulate still
    # creates and closes its own process team; worker startup remains timed.
    _observation(
        case_path,
        output / "warmup",
        1,
        arguments.particles,
        arguments.macro_steps,
        arguments.output_mode,
    )
    observations = []
    worker_counts = tuple(dict.fromkeys(arguments.workers))
    for repeat in range(arguments.repeats):
        ordered = worker_counts if repeat % 2 == 0 else tuple(reversed(worker_counts))
        for workers in ordered:
            row = _observation(
                case_path,
                output / f"repeat-{repeat}-workers-{workers}",
                workers,
                arguments.particles,
                arguments.macro_steps,
                arguments.output_mode,
            )
            row["repeat"] = repeat
            observations.append(row)
            print(
                f"repeat={repeat} workers={workers}: {row['timing_s']['public_end_to_end']:.3f}s",
                flush=True,
            )
    digests = {row["full_scientific_payload_sha256"] for row in observations}
    if len(digests) != 1:
        raise RuntimeError("worker count or repeat changed the public scientific payload")
    if source_before != _source_identity():
        raise RuntimeError("source changed during measurement")
    if input_identity != {name: _sha256(case_path.with_name(name)) for name in input_identity}:
        raise RuntimeError("input changed during measurement")
    return _measurement_report(
        arguments, source_before, input_identity, observations, worker_counts
    )


def _measurement_report(
    arguments: argparse.Namespace,
    source_identity: dict[str, str],
    input_identity: dict[str, str],
    observations: list[dict[str, Any]],
    worker_counts: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "benchmark": "public_process_workers_v2",
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "machine": _machine_metadata(),
        "runtime": {
            "numba": metadata.version("numba"),
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "NUMBA_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                )
            },
        },
        "source_sha256": source_identity,
        "input_sha256": input_identity,
        "conditions": {
            "particles": arguments.particles,
            "macro_steps": arguments.macro_steps,
            "interval_s": [0.0, 4.0],
            "workers": list(worker_counts),
            "repeats": arguments.repeats,
            "timed_scope": "load_case + simulate (spawn, IPC, writer, shutdown included) + open_result",
            "untimed": "input generation, one serial warmup, outcome validation, scientific hashing",
            "input": "editable input/case.yaml and case.h5; P14-U XY regular32x4 surface release, Epstein, electric, gravity",
            "cache": "persistent disk cache, one serial warmup; fresh process team for every simulate",
            "output_mode": arguments.output_mode,
            "output": "P14-U none/sample policy; complete final/release/boundary/failure/lifecycle payload",
            "payload_io": "separate full-file SHA-256 I/O, outcome reads, science hashing and explicit verify_contents; these are outside public_end_to_end",
        },
        "observations": observations,
        "median_public_seconds": {
            str(workers): statistics.median(
                row["timing_s"]["public_end_to_end"]
                for row in observations
                if row["workers"] == workers
            )
            for workers in worker_counts
        },
        "scientific_identity": True,
        "independent_reference": {
            "known_target": "one retained capture per ID at x=1, interior y, 0<t<4s, zero failure",
            "accuracy": "P14-U separate time/mesh refinement and RZ gates; worker equality is not accuracy certification",
        },
        "limits": "machine-local observations; no portable speed threshold, release-size matrix, whole-process RSS bound or COMSOL parity",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new directory for input, results and measurement.json",
    )
    parser.add_argument("--particles", type=_positive, default=10_000)
    parser.add_argument("--macro-steps", type=_positive, default=320)
    parser.add_argument("--workers", type=_positive, nargs="+", default=[1, 4])
    parser.add_argument("--repeats", type=_positive, default=3)
    parser.add_argument("--memory-limit-mb", type=_positive, default=8192)
    parser.add_argument("--output-mode", choices=("none", "sample"), default="none")
    arguments = parser.parse_args()
    report = run(arguments)
    path = arguments.output / "measurement.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
