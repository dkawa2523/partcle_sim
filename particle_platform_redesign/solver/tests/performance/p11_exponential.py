"""Manual fresh/warm check of the P11 exponential-midpoint production path.

The driver materializes one expanded C03 case outside every measured process.
Each observation then uses a fresh child and an initially empty Numba cache.
``fresh`` measures the first public run; ``warm`` performs the requested public
warm-up runs in the same child before measuring.  Results are descriptive,
non-gating evidence rather than absolute performance thresholds.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from chamber_particles.integrators import _compiled_exponential_update
from tests.performance.p06_baseline import _expanded_definition, _materialize
from tests.performance.p10_compiled import (
    _compiled_machine_metadata,
    _execute_run,
    _identity_summary,
    _summaries,
)
from tests.verification.microcases import build_microcase

_MODES = ("fresh", "warm")
_EXPECTED_REVISIONS: dict[str, str | None] = {
    "engine_algorithm_revision": "particle_engine_v36",
    "compiled_cpu_tile_revision": "compiled_cpu_tile_v18",
    "physics_runtime_revision": "signed_ion_compiled_physics_runtime_v19",
    "step_proposal_revision": "coupled_fixed_step_proposal_v10",
    "rk4_enclosure_revision": None,
    "exponential_midpoint_revision": "charge_stable_exponential_midpoint_v3",
    "exponential_midpoint_enclosure_revision": ("exponential_midpoint_global_abs_enclosure_v3"),
    "field_location_revision": "field_location_v4",
    "event_algorithm_revision": "line_quadratic_rk4_axis_first_hit_v16",
    "result_algorithm_revision": "durable_segmented_result_v5",
}
_EXPECTED_MEMORY_PLAN_REVISION = "solver_owned_memory_plan_v13"
_EXPECTED_RUNTIME_LAYOUT_REVISION = "resident_soa_serial_slab_v6"


def main() -> None:
    """Run the human-facing driver or one isolated measurement worker."""

    arguments = _arguments()
    if arguments.worker_case is not None:
        _worker(arguments)
        return
    _driver(arguments)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=_positive_integer, default=10_000)
    parser.add_argument("--repeats", type=_positive_integer, default=3)
    parser.add_argument("--warmups", type=_positive_integer, default=1)
    parser.add_argument("--json", dest="json_path", type=Path)
    parser.add_argument("--worker-case", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-mode", choices=_MODES, help=argparse.SUPPRESS)
    parser.add_argument("--worker-particles", type=_positive_integer, help=argparse.SUPPRESS)
    parser.add_argument("--worker-warmups", type=_nonnegative_integer, help=argparse.SUPPRESS)
    return parser.parse_args()


def _positive_integer(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _nonnegative_integer(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return result


def _driver(arguments: argparse.Namespace) -> None:
    with tempfile.TemporaryDirectory(prefix="chamber-particles-p11-") as temporary:
        root = Path(temporary)
        case_path = _materialize_case(root / "case", arguments.particles)
        observations: list[dict[str, Any]] = []
        for repeat in range(arguments.repeats):
            observations.extend(
                _launch_worker(
                    case_path=case_path,
                    output=root / "results" / mode / f"{repeat:03d}",
                    cache_dir=root / "numba-cache" / mode / f"{repeat:03d}",
                    mode=mode,
                    particle_count=arguments.particles,
                    warmups=arguments.warmups if mode == "warm" else 0,
                    repeat=repeat,
                )
                for mode in _MODES
            )
        _validate_observations(observations, arguments.repeats)
        summaries = _summaries(observations)
        report = {
            "benchmark": "p11_exponential_midpoint_fresh_warm_v1",
            "captured_at_utc": datetime.now(UTC).isoformat(),
            "non_gating": True,
            "conditions": {
                "scenario": "expanded C03 constant linear relaxation and additive force",
                "particle_count": arguments.particles,
                "macro_step_dt_s": 0.75,
                "maximum_dt_over_tau": 3.0,
                "repeats": arguments.repeats,
                "warmup_runs_for_warm_mode": arguments.warmups,
                "case_materialization_in_measured_scope": False,
                "timed_scope": "load_case + simulate + open_result",
                "jit_isolation": (
                    "each observation uses a fresh process and a unique initially empty "
                    "NUMBA_CACHE_DIR"
                ),
                "compiled_execution": (
                    "compiled_cpu_tile_v18 with NUMBA_DISABLE_JIT=0, fastmath=False, "
                    "serial compiled kernels, NUMBA_NUM_THREADS=1"
                ),
            },
            "machine": _compiled_machine_metadata(),
            "observations": observations,
            "summaries": summaries,
            "comparisons": _fresh_warm_comparison(summaries),
            "identity": _identity_summary(observations),
            "interpretation": {
                "fresh_warm_speedup": (
                    "fresh/warm characterizes same-process JIT amortization; it is not a "
                    "comparison against RK4 or another engine revision"
                ),
                "rss": (
                    "OS process high-water RSS includes Python and native libraries and is "
                    "separate from the solver-owned memory plan"
                ),
                "scope": "P14 owns the product-scale field/event/output matrix",
                "thresholds": "absolute seconds, RSS, and speedup have no pass/fail threshold",
            },
        }
    encoded = json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    if arguments.json_path is not None:
        arguments.json_path.parent.mkdir(parents=True, exist_ok=True)
        arguments.json_path.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


def _materialize_case(directory: Path, particle_count: int) -> Path:
    definition = _expanded_definition(build_microcase("C03"), particle_count)
    case_path = _materialize(
        directory,
        definition,
        name="p11_exponential_midpoint",
        frame_times_s=None,
    )
    document = yaml.safe_load(case_path.read_text(encoding="utf-8"))
    document["solver"]["integrator"] = "exponential_midpoint"
    document["time"]["dt_s"] = document["time"]["end_s"]
    case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return case_path


def _launch_worker(
    *,
    case_path: Path,
    output: Path,
    cache_dir: Path,
    mode: str,
    particle_count: int,
    warmups: int,
    repeat: int,
) -> dict[str, Any]:
    cache_dir.mkdir(parents=True)
    command = [
        sys.executable,
        "-m",
        "tests.performance.p11_exponential",
        "--worker-case",
        str(case_path),
        "--worker-output",
        str(output),
        "--worker-mode",
        mode,
        "--worker-particles",
        str(particle_count),
        "--worker-warmups",
        str(warmups),
    ]
    environment = os.environ.copy()
    environment["NUMBA_CACHE_DIR"] = str(cache_dir)
    environment["NUMBA_DISABLE_JIT"] = "0"
    environment["NUMBA_NUM_THREADS"] = "1"
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"P11 {mode} worker failed with exit {completed.returncode}: {completed.stderr.strip()}"
        )
    try:
        observation = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"P11 {mode} worker did not return JSON") from error
    if not isinstance(observation, dict):
        raise RuntimeError(f"P11 {mode} worker returned a non-object")
    observation["repeat"] = repeat
    return observation


def _worker(arguments: argparse.Namespace) -> None:
    required = (
        arguments.worker_case,
        arguments.worker_output,
        arguments.worker_mode,
        arguments.worker_particles,
        arguments.worker_warmups,
    )
    if any(value is None for value in required):
        raise SystemExit("P11 worker requires case, output, mode, particle count, and warmups")
    for index in range(arguments.worker_warmups):
        _execute_run(
            arguments.worker_case,
            arguments.worker_output / f"warmup-{index:03d}",
            "exponential",
            arguments.worker_mode,
            arguments.worker_particles,
            0,
            include_digest=False,
            expected_path_kind="exponential_midpoint_reintegrated",
        )
        gc.collect()
    observation = _execute_run(
        arguments.worker_case,
        arguments.worker_output / "measured",
        "exponential",
        arguments.worker_mode,
        arguments.worker_particles,
        0,
        include_digest=True,
        expected_path_kind="exponential_midpoint_reintegrated",
    )
    exponential_signature_count = len(_compiled_exponential_update.signatures)
    if exponential_signature_count == 0:
        raise RuntimeError("P11 compiled exponential update did not execute")
    compiled = observation["compiled_execution"]
    if not isinstance(compiled, dict):
        raise RuntimeError("P11 compiled execution record is not mutable")
    compiled["exponential_update_signature_count"] = exponential_signature_count
    print(json.dumps(observation, allow_nan=False, sort_keys=True))


def _validate_observations(observations: Sequence[Mapping[str, object]], repeats: int) -> None:
    if len(observations) != repeats * len(_MODES):
        raise RuntimeError("P11 observation count is incomplete")
    digests = {item["semantic_result_sha256"] for item in observations}
    if None in digests or len(digests) != 1:
        raise RuntimeError("P11 fresh/warm semantic result identity changed")
    identities = {
        json.dumps(item["case_identity"], sort_keys=True, separators=(",", ":"))
        for item in observations
    }
    if len(identities) != 1:
        raise RuntimeError("P11 case identity changed across observations")
    for item in observations:
        algorithms = _require_mapping(item, "algorithms")
        for key, expected in _EXPECTED_REVISIONS.items():
            if algorithms.get(key) != expected:
                raise RuntimeError(f"P11 expected {key}={expected!r}, got {algorithms.get(key)!r}")
        resolved = _require_mapping(item, "resolved_execution")
        if resolved != {
            "integrator": "exponential_midpoint",
            "backend": "cpu",
            "path_kind": "exponential_midpoint_reintegrated",
        }:
            raise RuntimeError(f"P11 unexpected resolved execution {dict(resolved)!r}")
        memory = _require_mapping(item, "memory")
        plan = _require_mapping(memory, "solver_memory_plan")
        if plan.get("revision") != _EXPECTED_MEMORY_PLAN_REVISION:
            raise RuntimeError("P11 unexpected memory-plan revision")
        if plan.get("runtime_layout_revision") != _EXPECTED_RUNTIME_LAYOUT_REVISION:
            raise RuntimeError("P11 unexpected runtime-layout revision")
        compiled = _require_mapping(item, "compiled_execution")
        if int(compiled["physics_tile_signature_count"]) < 1:
            raise RuntimeError("P11 compiled physics tile did not execute")
        if int(compiled["exponential_update_signature_count"]) < 1:
            raise RuntimeError("P11 compiled exponential update did not execute")
        if compiled.get("numba_disable_jit") != "0":
            raise RuntimeError("P11 Numba JIT was not enabled")


def _fresh_warm_comparison(
    summaries: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    fresh = next(item for item in summaries if item["mode"] == "fresh")
    warm = next(item for item in summaries if item["mode"] == "warm")
    fresh_timing = _require_mapping(fresh, "median_timing_s")
    warm_timing = _require_mapping(warm, "median_timing_s")
    return {
        "exponential": {
            "simulate_fresh_over_warm_speedup": (
                float(fresh_timing["simulate"]) / float(warm_timing["simulate"])
            ),
            "public_end_to_end_fresh_over_warm_speedup": (
                float(fresh_timing["public_end_to_end"]) / float(warm_timing["public_end_to_end"])
            ),
            "simulate_fresh_minus_warm_s": (
                float(fresh_timing["simulate"]) - float(warm_timing["simulate"])
            ),
        }
    }


def _require_mapping(value: Mapping[str, object], key: str) -> Mapping[str, object]:
    result = value[key]
    if not isinstance(result, Mapping):
        raise RuntimeError(f"P11 observation {key} is not a mapping")
    return result


if __name__ == "__main__":
    main()
