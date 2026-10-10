"""Manual cold/warm characterization of the P10 compiled CPU production path.

The driver creates cases outside every measured process.  Each observation is
run in a fresh child with its own empty Numba cache.  ``cold`` measures the
first production run in that child; ``warm`` performs an untimed production
run in the same child before measuring.  The output is descriptive evidence,
not a pytest test or an absolute performance gate.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from chamber_particles import load_case, open_result, simulate
from chamber_particles.physics.compiled import evaluate_physics_tile_into
from tests.performance.p06_baseline import _build_scenarios
from tests.performance.p09_memory import (
    _directory_bytes,
    _machine_metadata,
    _process_memory_bytes,
    _semantic_result_digest,
)

_SCENARIOS = ("field", "event")
_MODES = ("cold", "warm")
_EXPECTED_REVISIONS = {
    "engine_algorithm_revision": "particle_engine_v46",
    "compiled_cpu_tile_revision": "compiled_cpu_tile_v21",
    "physics_runtime_revision": "signed_ion_compiled_physics_runtime_v22",
    "step_proposal_revision": "coupled_fixed_step_proposal_v10",
    "field_location_revision": "field_location_v4",
    "event_algorithm_revision": "line_quadratic_curved_capsule_periodic_first_hit_v22",
    "result_algorithm_revision": "durable_segmented_result_v6",
}
_EXPECTED_MEMORY_PLAN_REVISION = "solver_owned_memory_plan_v16"
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
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=_SCENARIOS,
        default=list(_SCENARIOS),
        help="field-heavy event-light and/or small event-heavy production cases",
    )
    parser.add_argument("--field-particles", type=_positive_integer, default=10_000)
    parser.add_argument("--event-particles", type=_positive_integer, default=32)
    parser.add_argument("--repeats", type=_positive_integer, default=3)
    parser.add_argument("--warmups", type=_positive_integer, default=1)
    parser.add_argument("--json", dest="json_path", type=Path)
    parser.add_argument("--worker-case", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-scenario", choices=_SCENARIOS, help=argparse.SUPPRESS)
    parser.add_argument("--worker-mode", choices=_MODES, help=argparse.SUPPRESS)
    parser.add_argument("--worker-particles", type=_positive_integer, help=argparse.SUPPRESS)
    parser.add_argument("--worker-events", type=_nonnegative_integer, help=argparse.SUPPRESS)
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
    scenarios = tuple(dict.fromkeys(arguments.scenarios))
    with tempfile.TemporaryDirectory(prefix="chamber-particles-p10-") as temporary:
        root = Path(temporary)
        cases = _materialize_cases(
            root / "cases",
            scenarios,
            arguments.field_particles,
            arguments.event_particles,
        )
        observations: list[dict[str, Any]] = []
        for scenario in scenarios:
            case = cases[scenario]
            for repeat in range(arguments.repeats):
                observations.extend(
                    _launch_worker(
                        case_path=case["case_path"],
                        output_root=root / "results" / scenario / mode / f"{repeat:03d}",
                        cache_dir=root / "numba-cache" / scenario / mode / f"{repeat:03d}",
                        scenario=scenario,
                        mode=mode,
                        particle_count=case["particle_count"],
                        expected_events=case["expected_events"],
                        warmups=arguments.warmups if mode == "warm" else 0,
                        repeat=repeat,
                    )
                    for mode in _MODES
                )
        _validate_observations(observations, scenarios, arguments.repeats)
        summaries = _summaries(observations)
        report = {
            "benchmark": "p10_compiled_cpu_cold_warm_v1",
            "captured_at_utc": datetime.now(UTC).isoformat(),
            "non_gating": True,
            "conditions": {
                "scenarios": list(scenarios),
                "field_particles": arguments.field_particles,
                "event_particles": arguments.event_particles,
                "repeats": arguments.repeats,
                "warmup_runs_for_warm_mode": arguments.warmups,
                "case_materialization_in_measured_scope": False,
                "timed_scope": "load_case + simulate + open_result",
                "jit_isolation": (
                    "each observation uses a fresh process and a unique initially empty "
                    "NUMBA_CACHE_DIR"
                ),
                "compiled_execution": (
                    "fastmath=False, serial compiled kernels, NUMBA_DISABLE_JIT=0, "
                    "NUMBA_NUM_THREADS=1"
                ),
            },
            "machine": _compiled_machine_metadata(),
            "observations": observations,
            "summaries": summaries,
            "comparisons": _comparisons(summaries),
            "identity": _identity_summary(observations),
            "interpretation": {
                "cold_warm_speedup": (
                    "cold/warm is compilation-amortization evidence, not an inter-revision "
                    "engine speedup claim"
                ),
                "rss": (
                    "OS process high-water RSS includes Python and native libraries and is "
                    "reported separately from the solver-owned memory plan"
                ),
                "scope": (
                    "this focused P10 pair is not the P14 10k/100k/1M, field, event, output, "
                    "and full product matrix"
                ),
                "thresholds": "absolute seconds, RSS, and speedup have no pass/fail threshold",
            },
        }
    encoded = json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    if arguments.json_path is not None:
        arguments.json_path.parent.mkdir(parents=True, exist_ok=True)
        arguments.json_path.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


def _materialize_cases(
    root: Path,
    scenarios: tuple[str, ...],
    field_particles: int,
    event_particles: int,
) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    if "field" in scenarios:
        candidates = _build_scenarios(root / "field", field_particles)
        scenario = next(item for item in candidates if item.name == "rk4_enclosed_no_frames")
        selected["field"] = {
            "case_path": scenario.case_path,
            "particle_count": field_particles,
            "expected_events": 0,
        }
    if "event" in scenarios:
        candidates = _build_scenarios(root / "event", event_particles)
        scenario = next(item for item in candidates if item.name == "rk4_material_hits")
        selected["event"] = {
            "case_path": scenario.case_path,
            "particle_count": event_particles,
            "expected_events": event_particles,
        }
    return selected


def _launch_worker(
    *,
    case_path: Path,
    output_root: Path,
    cache_dir: Path,
    scenario: str,
    mode: str,
    particle_count: int,
    expected_events: int,
    warmups: int,
    repeat: int,
) -> dict[str, Any]:
    cache_dir.mkdir(parents=True)
    command = [
        sys.executable,
        "-m",
        "tests.performance.p10_compiled",
        "--worker-case",
        str(case_path),
        "--worker-output",
        str(output_root),
        "--worker-scenario",
        scenario,
        "--worker-mode",
        mode,
        "--worker-particles",
        str(particle_count),
        "--worker-events",
        str(expected_events),
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
            f"P10 {scenario}/{mode} worker failed with exit {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    try:
        observation = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"P10 {scenario}/{mode} worker did not return JSON") from error
    if not isinstance(observation, dict):
        raise RuntimeError(f"P10 {scenario}/{mode} worker returned a non-object")
    observation["repeat"] = repeat
    return observation


def _worker(arguments: argparse.Namespace) -> None:
    required = (
        arguments.worker_output,
        arguments.worker_scenario,
        arguments.worker_mode,
        arguments.worker_particles,
        arguments.worker_events,
        arguments.worker_warmups,
    )
    if any(value is None for value in required):
        raise SystemExit("P10 worker requires output, scenario, mode, counts, and warmups")
    for index in range(arguments.worker_warmups):
        _execute_run(
            arguments.worker_case,
            arguments.worker_output / f"warmup-{index:03d}",
            arguments.worker_scenario,
            arguments.worker_mode,
            arguments.worker_particles,
            arguments.worker_events,
            include_digest=False,
            expected_path_kind="rk4_dense",
        )
        gc.collect()
    observation = _execute_run(
        arguments.worker_case,
        arguments.worker_output / "measured",
        arguments.worker_scenario,
        arguments.worker_mode,
        arguments.worker_particles,
        arguments.worker_events,
        include_digest=True,
        expected_path_kind="rk4_dense",
    )
    print(json.dumps(observation, allow_nan=False, sort_keys=True))


def _execute_run(
    case_path: Path,
    output: Path,
    scenario: str,
    mode: str,
    expected_particle_count: int,
    expected_events: int,
    *,
    include_digest: bool,
    expected_path_kind: str,
) -> dict[str, Any]:
    peak_before, rss_before, rss_source = _process_memory_bytes()
    total_started = time.perf_counter()
    load_started = time.perf_counter()
    case = load_case(case_path)
    load_elapsed = time.perf_counter() - load_started
    simulate_started = time.perf_counter()
    summary = simulate(case, output)
    simulate_elapsed = time.perf_counter() - simulate_started
    open_started = time.perf_counter()
    result = open_result(output)
    open_elapsed = time.perf_counter() - open_started
    total_elapsed = time.perf_counter() - total_started
    peak_after, rss_after, _ = _process_memory_bytes()

    if summary.particle_count != expected_particle_count:
        raise RuntimeError(f"{scenario}: unexpected particle count")
    if summary.boundary_event_count != expected_events:
        raise RuntimeError(f"{scenario}: unexpected boundary-event count")
    if summary.frame_count != 0:
        raise RuntimeError(f"{scenario}: timing case unexpectedly wrote frames")
    path_kind = str(result.manifest["resolved"]["path_kind"])
    if path_kind != expected_path_kind:
        raise RuntimeError(f"{scenario}: unexpected path kind {path_kind!r}")
    memory_plan = result.manifest["memory_plan"]
    if not isinstance(memory_plan, Mapping):
        raise RuntimeError(f"{scenario}: result memory plan is not a mapping")
    algorithms = {key: value for key, value in result.manifest.items() if key.endswith("_revision")}
    physics_signature_count = len(evaluate_physics_tile_into.signatures)
    if physics_signature_count == 0:
        raise RuntimeError(f"{scenario}: compiled physics tile did not execute")
    return {
        "scenario": scenario,
        "mode": mode,
        "particle_count": summary.particle_count,
        "macro_step_count": summary.macro_step_count,
        "boundary_event_count": summary.boundary_event_count,
        "case_identity": {
            "case_file_hash": case.case_file_hash,
            "data_content_hash": case.content_hash,
        },
        "timing_s": {
            "load_case": load_elapsed,
            "simulate": simulate_elapsed,
            "open_result": open_elapsed,
            "public_end_to_end": total_elapsed,
        },
        "throughput": {
            "particles_per_simulate_s": summary.particle_count / simulate_elapsed,
            "nominal_particle_macro_steps_per_simulate_s": (
                summary.particle_count * summary.macro_step_count / simulate_elapsed
            ),
            "events_per_simulate_s": summary.boundary_event_count / simulate_elapsed,
        },
        "memory": {
            "measurement_source": rss_source,
            "peak_rss_before_measured_scope_bytes": peak_before,
            "peak_rss_after_measured_scope_bytes": peak_after,
            "additional_process_high_water_bytes": max(0, peak_after - peak_before),
            "rss_before_measured_scope_bytes": rss_before,
            "rss_after_measured_scope_bytes": rss_after,
            "solver_memory_plan": memory_plan,
        },
        "algorithms": algorithms,
        "resolved_execution": {
            "integrator": result.manifest["resolved"]["integrator"],
            "backend": result.manifest["resolved"]["backend"],
            "path_kind": path_kind,
        },
        "compiled_execution": {
            "physics_tile_signature_count": physics_signature_count,
            "numba_disable_jit": os.environ.get("NUMBA_DISABLE_JIT"),
            "numba_num_threads": os.environ.get("NUMBA_NUM_THREADS"),
        },
        "semantic_result_sha256": _semantic_result_digest(result) if include_digest else None,
        "case_artifact_bytes": _directory_bytes(case_path.parent),
        "result_artifact_bytes": _directory_bytes(output),
    }


def _validate_observations(
    observations: Sequence[Mapping[str, object]],
    scenarios: tuple[str, ...],
    repeats: int,
) -> None:
    for scenario in scenarios:
        selected = [item for item in observations if item["scenario"] == scenario]
        if len(selected) != repeats * len(_MODES):
            raise RuntimeError(f"P10 {scenario}: observation count is incomplete")
        digests = {item["semantic_result_sha256"] for item in selected}
        if None in digests or len(digests) != 1:
            raise RuntimeError(f"P10 {scenario}: cold/warm semantic result identity changed")
        identities = {
            json.dumps(item["case_identity"], sort_keys=True, separators=(",", ":"))
            for item in selected
        }
        if len(identities) != 1:
            raise RuntimeError(f"P10 {scenario}: case identity changed across observations")
        for item in selected:
            algorithms = _mapping_value(item, "algorithms")
            for key, expected in _EXPECTED_REVISIONS.items():
                if algorithms.get(key) != expected:
                    raise RuntimeError(
                        f"P10 {scenario}: expected {key}={expected!r}, got {algorithms.get(key)!r}"
                    )
            memory = _mapping_value(item, "memory")
            plan = _mapping_value(memory, "solver_memory_plan")
            if plan.get("revision") != _EXPECTED_MEMORY_PLAN_REVISION:
                raise RuntimeError(f"P10 {scenario}: unexpected memory-plan revision")
            if plan.get("runtime_layout_revision") != _EXPECTED_RUNTIME_LAYOUT_REVISION:
                raise RuntimeError(f"P10 {scenario}: unexpected runtime-layout revision")
            compiled = _mapping_value(item, "compiled_execution")
            if int(compiled["physics_tile_signature_count"]) < 1:
                raise RuntimeError(f"P10 {scenario}: compiled physics tile did not execute")
            if compiled.get("numba_disable_jit") != "0":
                raise RuntimeError(f"P10 {scenario}: Numba JIT was not enabled")


def _summaries(observations: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    keys = sorted({(str(item["scenario"]), str(item["mode"])) for item in observations})
    for scenario, mode in keys:
        selected = [
            item for item in observations if item["scenario"] == scenario and item["mode"] == mode
        ]
        timings = [_mapping_value(item, "timing_s") for item in selected]
        throughputs = [_mapping_value(item, "throughput") for item in selected]
        memories = [_mapping_value(item, "memory") for item in selected]
        representative = selected[0]
        summaries.append(
            {
                "scenario": scenario,
                "mode": mode,
                "observations": len(selected),
                "particle_count": representative["particle_count"],
                "macro_step_count": representative["macro_step_count"],
                "boundary_event_count": representative["boundary_event_count"],
                "median_timing_s": {
                    name: statistics.median(float(value[name]) for value in timings)
                    for name in ("load_case", "simulate", "open_result", "public_end_to_end")
                },
                "median_throughput": {
                    name: statistics.median(float(value[name]) for value in throughputs)
                    for name in (
                        "particles_per_simulate_s",
                        "nominal_particle_macro_steps_per_simulate_s",
                        "events_per_simulate_s",
                    )
                },
                "maximum_peak_rss_bytes": max(
                    int(value["peak_rss_after_measured_scope_bytes"]) for value in memories
                ),
                "algorithms": representative["algorithms"],
                "semantic_result_sha256": representative["semantic_result_sha256"],
            }
        )
    return summaries


def _comparisons(summaries: Sequence[Mapping[str, object]]) -> dict[str, object]:
    comparisons: dict[str, object] = {}
    scenarios = sorted({str(item["scenario"]) for item in summaries})
    for scenario in scenarios:
        cold = next(
            item for item in summaries if item["scenario"] == scenario and item["mode"] == "cold"
        )
        warm = next(
            item for item in summaries if item["scenario"] == scenario and item["mode"] == "warm"
        )
        cold_timing = _mapping_value(cold, "median_timing_s")
        warm_timing = _mapping_value(warm, "median_timing_s")
        comparisons[scenario] = {
            "simulate_cold_over_warm_speedup": (
                float(cold_timing["simulate"]) / float(warm_timing["simulate"])
            ),
            "public_end_to_end_cold_over_warm_speedup": (
                float(cold_timing["public_end_to_end"]) / float(warm_timing["public_end_to_end"])
            ),
            "simulate_cold_minus_warm_s": (
                float(cold_timing["simulate"]) - float(warm_timing["simulate"])
            ),
        }
    return comparisons


def _identity_summary(observations: Sequence[Mapping[str, object]]) -> dict[str, object]:
    by_scenario: dict[str, object] = {}
    for scenario in sorted({str(item["scenario"]) for item in observations}):
        selected = [item for item in observations if item["scenario"] == scenario]
        by_scenario[scenario] = {
            "semantic_result_sha256": selected[0]["semantic_result_sha256"],
            "observations_compared": len(selected),
            "exact_match": True,
        }
    return {
        "definition": (
            "stable manifest semantics plus final/events/series/frames/probes; timing and RSS excluded"
        ),
        "by_scenario": by_scenario,
    }


def _mapping_value(value: Mapping[str, object], key: str) -> Mapping[str, object]:
    result = value[key]
    if not isinstance(result, Mapping):
        raise RuntimeError(f"observation {key} is not a mapping")
    return result


def _compiled_machine_metadata() -> dict[str, object]:
    machine = _machine_metadata()
    machine["numba"] = metadata.version("numba")
    return machine


if __name__ == "__main__":
    main()
