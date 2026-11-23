"""Cold/warm compiled CPU measurements for field, event and relaxation workloads.

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

import yaml

from chamber_particles import load_case, open_result, simulate
from chamber_particles.cpu import CPU_RUNTIME_LAYOUT_REVISION, MEMORY_PLAN_REVISION
from chamber_particles.engine import COMPILED_CPU_TILE_REVISION, ENGINE_ALGORITHM_REVISION
from chamber_particles.events import EVENT_ALGORITHM_REVISION
from chamber_particles.fields import FIELD_LOCATION_REVISION
from chamber_particles.integrators import (
    EXPONENTIAL_MIDPOINT_ENCLOSURE_REVISION,
    EXPONENTIAL_MIDPOINT_REVISION,
    RK4_ENCLOSURE_REVISION,
    STEP_PROPOSAL_REVISION,
    _compiled_relaxation_advance_into,
    _compiled_rk4_advance_into,
)
from chamber_particles.output import RESULT_ALGORITHM_REVISION
from chamber_particles.physics.runtime import PHYSICS_RUNTIME_REVISION
from tests.performance.p06_baseline import _build_scenarios, _expanded_definition, _materialize
from tests.performance.p09_memory import (
    _directory_bytes,
    _machine_metadata,
    _process_memory_bytes,
    _semantic_result_digest,
)
from tests.verification.microcases import build_microcase

_SCENARIOS = ("field", "event", "relaxation")
_MODES = ("cold", "warm")
_EXPECTED_REVISIONS = {
    "engine_algorithm_revision": ENGINE_ALGORITHM_REVISION,
    "compiled_cpu_tile_revision": COMPILED_CPU_TILE_REVISION,
    "physics_runtime_revision": PHYSICS_RUNTIME_REVISION,
    "step_proposal_revision": STEP_PROPOSAL_REVISION,
    "field_location_revision": FIELD_LOCATION_REVISION,
    "event_algorithm_revision": EVENT_ALGORITHM_REVISION,
    "result_algorithm_revision": RESULT_ALGORITHM_REVISION,
}
_EXPECTED_MEMORY_PLAN_REVISION = MEMORY_PLAN_REVISION
_EXPECTED_RUNTIME_LAYOUT_REVISION = CPU_RUNTIME_LAYOUT_REVISION
_EXPECTED_EXECUTION = {
    "field": {"integrator": "rk4_fixed", "backend": "cpu", "path_kind": "rk4_dense"},
    "event": {"integrator": "rk4_fixed", "backend": "cpu", "path_kind": "rk4_dense"},
    "relaxation": {
        "integrator": "exponential_midpoint",
        "backend": "cpu",
        "path_kind": "exponential_midpoint_reintegrated",
    },
}
_METHOD_REVISIONS = {
    "rk4_fixed": {
        "rk4_enclosure_revision": RK4_ENCLOSURE_REVISION,
        "exponential_midpoint_revision": None,
        "exponential_midpoint_enclosure_revision": None,
    },
    "exponential_midpoint": {
        "rk4_enclosure_revision": None,
        "exponential_midpoint_revision": EXPONENTIAL_MIDPOINT_REVISION,
        "exponential_midpoint_enclosure_revision": EXPONENTIAL_MIDPOINT_ENCLOSURE_REVISION,
    },
}


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
        help="field-heavy, event-heavy, and/or constant-relaxation production cases",
    )
    parser.add_argument("--field-particles", type=_positive_integer, default=10_000)
    parser.add_argument("--event-particles", type=_positive_integer, default=32)
    parser.add_argument("--relaxation-particles", type=_positive_integer, default=10_000)
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
            arguments.relaxation_particles,
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
            "benchmark": "compiled_cpu_workloads_cold_warm_v3",
            "captured_at_utc": datetime.now(UTC).isoformat(),
            "non_gating": True,
            "conditions": {
                "scenarios": list(scenarios),
                "field_particles": arguments.field_particles,
                "event_particles": arguments.event_particles,
                "relaxation_particles": arguments.relaxation_particles,
                "relaxation_problem": {
                    "meaning": "expanded C03 constant linear relaxation and additive force",
                    "macro_step_dt_s": 0.75,
                    "configured_dt_over_tau": 3.0,
                    "independent_reference": "C03 closed-form verification, run separately",
                },
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
                    "these focused workloads are not the P14 10k/100k/1M, field, event, output, "
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
    relaxation_particles: int,
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
    if "relaxation" in scenarios:
        definition = _expanded_definition(build_microcase("C03"), relaxation_particles)
        case_path = _materialize(
            root / "relaxation",
            definition,
            name="constant_relaxation",
            frame_times_s=None,
        )
        document = yaml.safe_load(case_path.read_text(encoding="utf-8"))
        document["solver"]["integrator"] = "exponential_midpoint"
        document["time"]["dt_s"] = document["time"]["end_s"]
        case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        selected["relaxation"] = {
            "case_path": case_path,
            "particle_count": relaxation_particles,
            "expected_events": 0,
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
    if summary.failure_event_count != 0:
        raise RuntimeError(f"{scenario}: performance case produced particle failures")
    if summary.frame_count != 0:
        raise RuntimeError(f"{scenario}: timing case unexpectedly wrote frames")
    manifest = result.manifest_document()
    resolved = _mapping_value(manifest, "resolved")
    execution = {name: resolved[name] for name in ("integrator", "backend", "path_kind")}
    if execution != _EXPECTED_EXECUTION[scenario]:
        raise RuntimeError(f"{scenario}: unexpected resolved execution {execution!r}")
    path_kind = str(resolved["path_kind"])
    memory_plan = manifest["memory_plan"]
    if not isinstance(memory_plan, Mapping):
        raise RuntimeError(f"{scenario}: result memory plan is not a mapping")
    algorithms = {key: value for key, value in manifest.items() if key.endswith("_revision")}
    # A nested cached kernel can execute without registering a specialization
    # on its Python dispatcher. Observe the actual Python-entered advance;
    # its signature registration works for both compiled and disk-cached code.
    if path_kind == "rk4_dense":
        stage_dispatcher = _compiled_rk4_advance_into
    elif path_kind == "exponential_midpoint_reintegrated":
        stage_dispatcher = _compiled_relaxation_advance_into
    else:
        raise RuntimeError(f"{scenario}: unsupported compiled benchmark path {path_kind!r}")
    dispatcher_metadata: Any = stage_dispatcher
    stage_signature_count = len(dispatcher_metadata.signatures)
    if stage_signature_count == 0:
        raise RuntimeError(f"{scenario}: compiled stage advance did not execute")
    return {
        "scenario": scenario,
        "mode": mode,
        "particle_count": summary.particle_count,
        "macro_step_count": summary.macro_step_count,
        "boundary_event_count": summary.boundary_event_count,
        "failure_event_count": summary.failure_event_count,
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
            "integrator": resolved["integrator"],
            "backend": resolved["backend"],
            "path_kind": path_kind,
        },
        "compiled_execution": {
            "stage_advance_signature_count": stage_signature_count,
            "numba_disable_jit": os.environ.get("NUMBA_DISABLE_JIT"),
            "numba_num_threads": os.environ.get("NUMBA_NUM_THREADS"),
        },
        "semantic_result_sha256": _semantic_result_digest(result) if include_digest else None,
        "case_artifact_bytes": _directory_bytes(case_path.parent),
        "result_artifact_bytes": _directory_bytes(output),
    }


def _validate_observations(
    observations: Sequence[Mapping[str, Any]],
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
            integrator = _EXPECTED_EXECUTION[scenario]["integrator"]
            expected_revisions = _EXPECTED_REVISIONS | _METHOD_REVISIONS[integrator]
            for key, expected in expected_revisions.items():
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
            if int(compiled["stage_advance_signature_count"]) < 1:
                raise RuntimeError(f"P10 {scenario}: compiled stage advance did not execute")
            if compiled.get("numba_disable_jit") != "0":
                raise RuntimeError(f"P10 {scenario}: Numba JIT was not enabled")


def _summaries(observations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
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


def _comparisons(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
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


def _identity_summary(observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_scenario: dict[str, Any] = {}
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


def _mapping_value(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    result = value[key]
    if not isinstance(result, Mapping):
        raise RuntimeError(f"observation {key} is not a mapping")
    return result


def _compiled_machine_metadata() -> dict[str, Any]:
    machine = _machine_metadata()
    machine["numba"] = metadata.version("numba")
    return machine


if __name__ == "__main__":
    main()
