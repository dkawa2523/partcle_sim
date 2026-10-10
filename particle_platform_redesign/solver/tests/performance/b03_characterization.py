"""Machine-local, non-gating B03 public-API performance characterization.

The driver materializes cases outside the measured process and launches one
fresh worker for each observation.  Every worker warms the same public
``load_case -> simulate -> open_result`` path before recording elapsed time and
the operating-system process RSS high-water mark.  A separate traced run is
kept out of timing.  Absolute values are evidence, never portable gates.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import math
import os
import platform
import statistics
import struct
import subprocess
import sys
import tempfile
import time
import tracemalloc
from collections.abc import Mapping, Sequence
from dataclasses import fields
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

from chamber_particles import load_case, open_result, simulate
from tests.scenarios.test_brownian_run import _brownian_case, _brownian_rz_case

_B02 = "b02_xy_fixed_drag"
_B03_FIXED = "b03_rz_fixed_drag"
_B03_CONTINUOUS = "b03_rz_effective_drag_continuous_charge_gravity"
_B03_AXIS = "b03_rz_fixed_drag_axis_restart"
_SCENARIOS = (_B02, _B03_FIXED, _B03_CONTINUOUS, _B03_AXIS)

_NOISE_REVISION = "inertial_langevin_fdt_epstein_linear_midpoint_2d_v2"
_NATIVE_DRAG_REVISION = "epstein_linear_v1"
_EFFECTIVE_DRAG_REVISION = "epstein_linear_effective_gas_sensitivity_v1"
_CONTINUOUS_CHARGE_REVISION = "oml_stationary_maxwellian_debye_huckel_v1"
_ENGINE_REVISION = "particle_engine_v46"
_EVENT_REVISION = "line_quadratic_curved_capsule_periodic_first_hit_v22"
_TREE_POLICY_REVISION = "conditional_boundary_refinement_v1"
_MEMORY_PLAN_REVISION = "solver_owned_memory_plan_v16"

_DEFAULT_PARTICLES = (2_000, 20_000)
_DEFAULT_REPEATS = 3
_DEFAULT_WARMUPS = 1
_DEFAULT_MEMORY_LIMIT_MB = 512
_DEFAULT_END_S = 0.2
_DEFAULT_DT_S = 0.02
_DEFAULT_TREE_DEPTH = 3
_SEED = 1_741
_PAIR_RADIAL_SHIFT_M = 1.5
_PAIR_INITIAL_VELOCITY_M_S = np.asarray([0.2, -0.1])
_PAIR_GAS_VELOCITY_M_S = np.asarray([0.1, -0.05])


class _WindowsProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("page_fault_count", ctypes.c_ulong),
        ("peak_working_set_size", ctypes.c_size_t),
        ("working_set_size", ctypes.c_size_t),
        ("quota_peak_paged_pool_usage", ctypes.c_size_t),
        ("quota_paged_pool_usage", ctypes.c_size_t),
        ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
        ("quota_non_paged_pool_usage", ctypes.c_size_t),
        ("pagefile_usage", ctypes.c_size_t),
        ("peak_pagefile_usage", ctypes.c_size_t),
    ]


def main(argv: list[str] | None = None) -> None:
    """Run the human-facing driver or one isolated measurement worker."""

    arguments = _arguments(argv)
    if arguments.worker_case is not None:
        _worker(arguments)
        return
    _driver(arguments)


def _arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--particles", nargs="+", type=_positive_integer, default=_DEFAULT_PARTICLES
    )
    parser.add_argument("--repeats", type=_positive_integer, default=_DEFAULT_REPEATS)
    parser.add_argument("--warmups", type=_positive_integer, default=_DEFAULT_WARMUPS)
    parser.add_argument(
        "--memory-limit-mb", type=_positive_integer, default=_DEFAULT_MEMORY_LIMIT_MB
    )
    parser.add_argument("--end-s", type=_positive_float, default=_DEFAULT_END_S)
    parser.add_argument("--dt-s", type=_positive_float, default=_DEFAULT_DT_S)
    parser.add_argument("--tree-depth", type=_tree_depth, default=_DEFAULT_TREE_DEPTH)
    parser.add_argument(
        "--adaptive-max-depth",
        type=_tree_depth,
        help="maximum conditional refinement depth; defaults to --tree-depth",
    )
    parser.add_argument("--json", dest="json_path", type=Path)
    parser.add_argument("--worker-case", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-scenario", choices=_SCENARIOS, help=argparse.SUPPRESS)
    parser.add_argument("--worker-particle-count", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-warmups", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--worker-trace", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def _positive_integer(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return result


def _tree_depth(value: str) -> int:
    result = int(value)
    if result < 0 or result > 10:
        raise argparse.ArgumentTypeError("tree depth must be in [0, 10]")
    return result


def _driver(arguments: argparse.Namespace) -> None:
    particle_counts = tuple(dict.fromkeys(arguments.particles))
    if len(particle_counts) < 2:
        raise SystemExit("B03 characterization requires a small and representative particle count")
    adaptive_max_depth = (
        arguments.tree_depth
        if arguments.adaptive_max_depth is None
        else arguments.adaptive_max_depth
    )
    if adaptive_max_depth < arguments.tree_depth:
        raise SystemExit("--adaptive-max-depth must be at least --tree-depth")
    macro_steps = _macro_step_count(arguments.end_s, arguments.dt_s)
    with tempfile.TemporaryDirectory(prefix="chamber-particles-b03-") as temporary:
        root = Path(temporary)
        cases = {
            count: _materialize_cases(
                root / "cases" / str(count),
                count,
                arguments.memory_limit_mb,
                arguments.end_s,
                arguments.dt_s,
                arguments.tree_depth,
                adaptive_max_depth,
            )
            for count in particle_counts
        }
        observations: list[dict[str, Any]] = []
        outputs: dict[tuple[int, int, str], Path] = {}
        for count in particle_counts:
            for repeat in range(arguments.repeats):
                ordered = (
                    _SCENARIOS[repeat % len(_SCENARIOS) :] + _SCENARIOS[: repeat % len(_SCENARIOS)]
                )
                for scenario in ordered:
                    output_root = root / "runs" / str(count) / f"repeat-{repeat:02d}" / scenario
                    observation = _launch_worker(
                        scenario=scenario,
                        particle_count=count,
                        case_path=cases[count][scenario],
                        output_root=output_root,
                        warmups=arguments.warmups,
                        trace_memory=repeat == 0,
                    )
                    observation["repeat"] = repeat
                    observations.append(observation)
                    outputs[(count, repeat, scenario)] = output_root / "measured"

        _validate_observations(
            observations,
            particle_counts,
            arguments.repeats,
            arguments.tree_depth,
            adaptive_max_depth,
        )
        paired = _paired_characterization(outputs, particle_counts, arguments.repeats)
        summaries = _summaries(observations)
        array_accounting = _array_accounting(observations)
        report = {
            "benchmark": "b03_public_api_warm_performance_memory_characterization_v1",
            "captured_at_utc": datetime.now(UTC).isoformat(),
            "non_gating": True,
            "conditions": {
                "particle_counts": list(particle_counts),
                "small_particle_count": particle_counts[0],
                "representative_particle_count": particle_counts[-1],
                "repeats_per_row": arguments.repeats,
                "same_process_warmups_per_observation": arguments.warmups,
                "memory_limit_mb": arguments.memory_limit_mb,
                "end_s": arguments.end_s,
                "dt_s": arguments.dt_s,
                "macro_steps": macro_steps,
                "time_grid_note": (
                    "The original decimal 0.2/0.02 ten-step grid is retained to exercise the "
                    "indexed macro-grid endpoint handling; engine revision is read from each "
                    "result manifest."
                ),
                "brownian_interval_tree_depth": arguments.tree_depth,
                "brownian_adaptive_max_depth": adaptive_max_depth,
                "uniform_leaves_per_root": 1 << arguments.tree_depth,
                "maximum_leaves_per_root": 1 << adaptive_max_depth,
                "case_materialization_in_measured_scope": False,
                "output_schedule": "none for every row",
                "seed": _SEED,
            },
            "case_matrix": _case_matrix(),
            "observations": observations,
            "summaries": summaries,
            "paired_fixed_drag_characterization": paired,
            "memory_array_accounting": array_accounting,
            "machine": _machine_metadata(),
            "interpretation": {
                "timing": (
                    "warm public load_case + simulate + open_result wall time; absolute seconds "
                    "are machine-local and are not an acceptance threshold"
                ),
                "process_peak_rss": (
                    "OS process high-water read after the measured public pipeline; the counter "
                    "cannot be reset, so it includes worker import, warm-up, Python/native "
                    "libraries, and HDF5"
                ),
                "solver_memory_plan": (
                    "manifest prediction for solver-owned arrays, not process RSS"
                ),
                "tracemalloc": (
                    "separate untimed public-pipeline observation; it is not RSS and does not "
                    "replace the solver-owned plan"
                ),
                "axis_restarts": (
                    "B03 v1 starts one independent residual root for every recorded RZ axis "
                    "crossing; the report labels this equality as an implementation inference"
                ),
            },
            "claim_limits": [
                "No absolute performance or memory threshold is applied.",
                "No COMSOL result or timing is used.",
                "The RZ noise revision is a two-degree-of-freedom meridional projection, not isotropic 3-D Brownian motion.",
                "The paired constant-coefficient case is a characterization, not a universal equivalence proof for state-dependent coefficients.",
                "The continuous-charge plus gravity row proves one accepted configuration completes; it is not physical-validity evidence.",
            ],
        }

    encoded = json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    if arguments.json_path is not None:
        arguments.json_path.parent.mkdir(parents=True, exist_ok=True)
        arguments.json_path.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


def _macro_step_count(end_s: float, dt_s: float) -> int:
    count = round(end_s / dt_s)
    if count < 1 or not math.isclose(
        count * dt_s, end_s, rel_tol=0.0, abs_tol=8.0 * math.ulp(end_s)
    ):
        raise SystemExit("--end-s must be an integer multiple of --dt-s")
    return count


def _materialize_cases(
    directory: Path,
    particle_count: int,
    memory_limit_mb: int,
    end_s: float,
    dt_s: float,
    tree_depth: int,
    adaptive_max_depth: int,
) -> dict[str, Path]:
    b02 = _brownian_case(
        directory / _B02,
        particle_count=particle_count,
        end_s=end_s,
        dt_s=dt_s,
        tree_depth=tree_depth,
        frame_times=None,
    )
    _patch_case(
        b02,
        _B02,
        memory_limit_mb,
        adaptive_max_depth=adaptive_max_depth,
        effective_drag=False,
    )

    b03_fixed = _brownian_rz_case(
        directory / _B03_FIXED,
        particle_count=particle_count,
        initial_position_m=np.asarray([_PAIR_RADIAL_SHIFT_M, 0.0]),
        initial_velocity_m_s=_PAIR_INITIAL_VELOCITY_M_S,
        gas_velocity_m_s=_PAIR_GAS_VELOCITY_M_S,
        radial_shift_m=_PAIR_RADIAL_SHIFT_M,
        end_s=end_s,
        dt_s=dt_s,
        tree_depth=tree_depth,
        gravity_m_s2=None,
        frame_times=None,
        memory_limit_mb=memory_limit_mb,
        initial_charge_number=0.0,
    )
    _patch_case(
        b03_fixed,
        _B03_FIXED,
        memory_limit_mb,
        adaptive_max_depth=adaptive_max_depth,
        effective_drag=False,
    )

    b03_continuous = _brownian_rz_case(
        directory / _B03_CONTINUOUS,
        particle_count=particle_count,
        initial_position_m=np.asarray([_PAIR_RADIAL_SHIFT_M, 0.0]),
        initial_velocity_m_s=_PAIR_INITIAL_VELOCITY_M_S,
        gas_velocity_m_s=_PAIR_GAS_VELOCITY_M_S,
        radial_shift_m=_PAIR_RADIAL_SHIFT_M,
        end_s=end_s,
        dt_s=dt_s,
        tree_depth=tree_depth,
        gravity_m_s2=np.asarray([0.0, -0.3]),
        frame_times=None,
        memory_limit_mb=memory_limit_mb,
        initial_charge_number=0.0,
        continuous_charge=True,
    )
    _patch_case(
        b03_continuous,
        _B03_CONTINUOUS,
        memory_limit_mb,
        adaptive_max_depth=adaptive_max_depth,
        effective_drag=True,
    )

    b03_axis = _brownian_rz_case(
        directory / _B03_AXIS,
        particle_count=particle_count,
        initial_position_m=np.asarray([0.01, 0.0]),
        initial_velocity_m_s=np.asarray([-1.0, 0.0]),
        gas_velocity_m_s=np.zeros(2),
        radial_shift_m=1.0,
        end_s=end_s,
        dt_s=dt_s,
        tree_depth=tree_depth,
        gravity_m_s2=None,
        frame_times=None,
        memory_limit_mb=memory_limit_mb,
        initial_charge_number=0.0,
    )
    _patch_case(
        b03_axis,
        _B03_AXIS,
        memory_limit_mb,
        adaptive_max_depth=adaptive_max_depth,
        effective_drag=False,
    )
    return {
        _B02: b02,
        _B03_FIXED: b03_fixed,
        _B03_CONTINUOUS: b03_continuous,
        _B03_AXIS: b03_axis,
    }


def _patch_case(
    path: Path,
    scenario: str,
    memory_limit_mb: int,
    *,
    adaptive_max_depth: int,
    effective_drag: bool,
) -> None:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["case"]["name"] = scenario
    document["resources"]["memory_limit_mb"] = memory_limit_mb
    document["solver"]["seed"] = _SEED
    document["physics"]["noise"]["adaptive_max_depth"] = adaptive_max_depth
    if effective_drag:
        document["physics"]["drag"]["revision"] = _EFFECTIVE_DRAG_REVISION
        document["physics"]["drag"]["maximum_speed_ratio"] = 0.6
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def _launch_worker(
    *,
    scenario: str,
    particle_count: int,
    case_path: Path,
    output_root: Path,
    warmups: int,
    trace_memory: bool,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "tests.performance.b03_characterization",
        "--worker-case",
        str(case_path),
        "--worker-output",
        str(output_root),
        "--worker-scenario",
        scenario,
        "--worker-particle-count",
        str(particle_count),
        "--worker-warmups",
        str(warmups),
    ]
    if trace_memory:
        command.append("--worker-trace")
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"B03 {scenario}/{particle_count} worker failed with exit "
            f"{completed.returncode}: {completed.stderr.strip()}"
        )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"B03 {scenario}/{particle_count} worker did not return JSON") from error
    if not isinstance(result, dict):
        raise RuntimeError("B03 worker returned a non-object")
    return result


def _worker(arguments: argparse.Namespace) -> None:
    if (
        arguments.worker_output is None
        or arguments.worker_scenario is None
        or arguments.worker_particle_count is None
    ):
        raise SystemExit("worker mode requires case, output, scenario, and particle count")
    if arguments.worker_warmups < 1:
        raise SystemExit("worker mode requires at least one warm-up")
    for index in range(arguments.worker_warmups):
        case = load_case(arguments.worker_case)
        warm_output = arguments.worker_output / f"warmup-{index:02d}"
        simulate(case, warm_output)
        warm_result = open_result(warm_output)
        _ = warm_result.manifest
        del warm_result, case
        gc.collect()

    peak_before, rss_before, rss_source = _process_memory_bytes()
    started = time.perf_counter()
    load_started = time.perf_counter()
    case = load_case(arguments.worker_case)
    load_elapsed = time.perf_counter() - load_started
    simulate_started = time.perf_counter()
    simulate(case, arguments.worker_output / "measured")
    simulate_elapsed = time.perf_counter() - simulate_started
    open_started = time.perf_counter()
    result = open_result(arguments.worker_output / "measured")
    open_elapsed = time.perf_counter() - open_started
    total_elapsed = time.perf_counter() - started
    peak_after, rss_after, _ = _process_memory_bytes()
    observation = _observation(
        arguments.worker_scenario,
        arguments.worker_particle_count,
        arguments.worker_case,
        arguments.worker_output / "measured",
        result,
        {
            "load_case_s": load_elapsed,
            "simulate_s": simulate_elapsed,
            "open_result_s": open_elapsed,
            "public_end_to_end_s": total_elapsed,
        },
        peak_before,
        peak_after,
        rss_before,
        rss_after,
        rss_source,
    )
    measured_digest = observation["scientific_payload_sha256"]
    del result, case
    gc.collect()

    observation["tracemalloc"] = None
    if arguments.worker_trace:
        tracemalloc.start()
        try:
            baseline, _ = tracemalloc.get_traced_memory()
            tracemalloc.reset_peak()
            traced_case = load_case(arguments.worker_case)
            simulate(traced_case, arguments.worker_output / "traced")
            traced_result = open_result(arguments.worker_output / "traced")
            current, peak = tracemalloc.get_traced_memory()
            traced_digest = _scientific_payload_digest(traced_result)
        finally:
            tracemalloc.stop()
        if traced_digest != measured_digest:
            raise RuntimeError("traced B03 run changed the scientific payload")
        observation["tracemalloc"] = {
            "baseline_traced_bytes": baseline,
            "current_traced_bytes_after_public_pipeline": current,
            "peak_traced_bytes": peak,
            "additional_peak_over_baseline_bytes": max(0, peak - baseline),
            "semantics": (
                "separate untimed load_case + simulate + open_result run; visible Python/NumPy "
                "allocations, not process RSS"
            ),
        }
    print(json.dumps(observation, allow_nan=False, sort_keys=True))


def _observation(
    scenario: str,
    particle_count: int,
    case_path: Path,
    output_path: Path,
    result: Any,
    timing_s: dict[str, float],
    peak_before: int,
    peak_after: int,
    rss_before: int | None,
    rss_after: int | None,
    rss_source: str,
) -> dict[str, Any]:
    manifest = result.manifest
    if int(manifest["counts"]["particles"]) != particle_count:
        raise RuntimeError("B03 worker particle count does not match the requested case")
    memory_plan = _mapping(manifest, "memory_plan")
    components = _mapping(memory_plan, "components")
    event_refinement = manifest["event_refinement"]
    accepted: int | None = None
    candidate_queries: int | None = None
    refinements: int | None = None
    maximum_depth: int | None = None
    if isinstance(event_refinement, Mapping):
        accepted = int(event_refinement["accepted_particle_pieces"])
        candidate_queries = int(event_refinement["candidate_queries"])
        refinements = int(event_refinement["refinements"])
        maximum_depth = int(event_refinement["maximum_refinement_depth"])
    boundary = _mapping(manifest, "boundary_interactions")
    axis_crossings = int(boundary["axis_crossings"])
    depth = int(manifest["brownian_interval_tree_depth"])
    adaptive_max_depth = int(manifest["brownian_adaptive_max_depth"])
    macro_steps = _macro_step_count(
        float(manifest["time"]["end_s"]),
        float(manifest["time"]["dt_s"]),
    )
    declared_leaf_visits = particle_count * macro_steps * (1 << depth)
    return {
        "scenario": scenario,
        "particle_count": particle_count,
        "timing_s": timing_s,
        "memory": {
            "measurement_source": rss_source,
            "process_peak_rss_before_measured_pipeline_bytes": peak_before,
            "process_peak_rss_after_measured_pipeline_bytes": peak_after,
            "additional_process_high_water_bytes": max(0, peak_after - peak_before),
            "rss_before_measured_pipeline_bytes": rss_before,
            "rss_after_measured_pipeline_bytes": rss_after,
            "solver_planned_bytes": int(memory_plan["planned_bytes"]),
            "solver_run_peak_bytes": int(memory_plan["phase_peaks"]["run"]),
            "slab_particles": int(memory_plan["slab_particles"]),
            "scratch_bytes_per_particle": int(memory_plan["scratch_bytes_per_particle"]),
            "slab_proposal_scratch_bytes": int(components["slab_proposal_scratch"]),
            "stochastic_tree_work_bytes_per_particle": int(
                memory_plan["stochastic_tree_work_bytes_per_particle"]
            ),
            "slab_stochastic_tree_work_bytes": int(components["slab_stochastic_tree_work"]),
            "full_plan": memory_plan,
        },
        "stochastic_work": {
            "uniform_tree_depth": depth,
            "adaptive_max_depth": adaptive_max_depth,
            "uniform_leaves_per_root": 1 << depth,
            "maximum_leaves_per_root": 1 << adaptive_max_depth,
            "macro_steps": macro_steps,
            "derived_nominal_leaf_visits_without_restarts": declared_leaf_visits,
            "manifest_accepted_particle_pieces": accepted,
            "manifest_candidate_queries": candidate_queries,
            "manifest_refinements": refinements,
            "manifest_maximum_refinement_depth": maximum_depth,
            "manifest_axis_crossings": axis_crossings,
            "inferred_b03_axis_restarts": axis_crossings if scenario != _B02 else None,
        },
        "lifecycle_counts": manifest["lifecycle_counts"],
        "failure_reason_counts": manifest["failure_reason_counts"],
        "revisions": _revisions(manifest),
        "resolved": {
            "path_kind": manifest["resolved"]["path_kind"],
            "brownian_coefficient_policy": manifest["resolved"]["brownian_coefficient_policy"],
            "physics_models": manifest["resolved"]["physics_models"],
            "brownian_composition_revision": manifest["brownian_composition_revision"],
            "brownian_charge_dense_revision": manifest["brownian_charge_dense_revision"],
        },
        "case_artifact_bytes": _directory_bytes(case_path.parent),
        "result_artifact_bytes": _directory_bytes(output_path),
        "scientific_payload_sha256": _scientific_payload_digest(result),
    }


def _mapping(value: Mapping[str, object], key: str) -> Mapping[str, object]:
    result = value[key]
    if not isinstance(result, Mapping):
        raise RuntimeError(f"B03 manifest {key} is not a mapping")
    return result


def _revisions(manifest: Mapping[str, object]) -> dict[str, object]:
    names = (
        "engine_algorithm_revision",
        "compiled_cpu_tile_revision",
        "step_proposal_revision",
        "event_algorithm_revision",
        "physics_catalog_revision",
        "physics_runtime_revision",
        "field_location_revision",
        "geometry_algorithm_revision",
        "result_algorithm_revision",
        "brownian_tree_policy_revision",
    )
    result = {name: manifest.get(name) for name in names}
    result["memory_plan_revision"] = manifest["memory_plan"]["revision"]
    result["runtime_layout_revision"] = manifest["memory_plan"]["runtime_layout_revision"]
    return result


def _validate_observations(
    observations: Sequence[Mapping[str, object]],
    particle_counts: tuple[int, ...],
    repeats: int,
    tree_depth: int,
    adaptive_max_depth: int,
) -> None:
    expected_rows = len(particle_counts) * len(_SCENARIOS) * repeats
    if len(observations) != expected_rows:
        raise RuntimeError(f"B03 expected {expected_rows} observations, got {len(observations)}")
    expected_tree_bytes = 128 * (adaptive_max_depth + 4)
    for count in particle_counts:
        for scenario in _SCENARIOS:
            selected = [
                item
                for item in observations
                if item["particle_count"] == count and item["scenario"] == scenario
            ]
            if len(selected) != repeats:
                raise RuntimeError(f"B03 missing repeats for {scenario}/{count}")
            digests = {item["scientific_payload_sha256"] for item in selected}
            if len(digests) != 1:
                raise RuntimeError(f"B03 payload changed across repeats for {scenario}/{count}")
            plans = {json.dumps(item["memory"]["full_plan"], sort_keys=True) for item in selected}
            work = {json.dumps(item["stochastic_work"], sort_keys=True) for item in selected}
            if len(plans) != 1 or len(work) != 1:
                raise RuntimeError(f"B03 manifest memory/work changed for {scenario}/{count}")
            representative = selected[0]
            work_record = representative["stochastic_work"]
            if (
                work_record["uniform_tree_depth"] != tree_depth
                or work_record["adaptive_max_depth"] != adaptive_max_depth
            ):
                raise RuntimeError("B03 Brownian base/max depth differs from the request")
            memory = representative["memory"]
            if memory["scratch_bytes_per_particle"] != 2_048:
                raise RuntimeError("B03 stage scratch changed from the v13 2048-byte allowance")
            if memory["stochastic_tree_work_bytes_per_particle"] != expected_tree_bytes:
                raise RuntimeError("B03 stochastic-tree plan does not match the declared depth")
            if representative["revisions"]["memory_plan_revision"] != _MEMORY_PLAN_REVISION:
                raise RuntimeError("B03 characterization unexpectedly changed memory-plan revision")
            expected_revisions = {
                "engine_algorithm_revision": _ENGINE_REVISION,
                "event_algorithm_revision": _EVENT_REVISION,
                "brownian_tree_policy_revision": _TREE_POLICY_REVISION,
            }
            for name, expected in expected_revisions.items():
                if representative["revisions"][name] != expected:
                    raise RuntimeError(
                        f"B03 expected {name}={expected!r}, got "
                        f"{representative['revisions'][name]!r}"
                    )
            lifecycle = representative["lifecycle_counts"]
            if lifecycle["active"] != count or any(
                lifecycle[name] != 0 for name in ("pending", "stuck", "held", "escaped", "failed")
            ):
                raise RuntimeError(
                    f"B03 {scenario}/{count} did not finish entirely active: "
                    f"lifecycle={lifecycle}, failures={representative['failure_reason_counts']}"
                )
            if any(representative["failure_reason_counts"].values()):
                raise RuntimeError(f"B03 {scenario}/{count} recorded a failure")
            _validate_scenario_contract(representative)


def _validate_scenario_contract(observation: Mapping[str, object]) -> None:
    scenario = str(observation["scenario"])
    resolved = observation["resolved"]
    models = resolved["physics_models"]
    work = observation["stochastic_work"]
    if resolved["path_kind"] != "cubic_hermite":
        raise RuntimeError(f"B03 {scenario} did not resolve the stochastic Hermite path")
    if scenario == _B02:
        expected = (
            "macro_root_frozen_midpoint_v1",
            "stochastic_exponential_midpoint_v1",
            "macro_root_affine_exponential_v2",
            _NOISE_REVISION,
            _NATIVE_DRAG_REVISION,
        )
        actual = (
            resolved["brownian_coefficient_policy"],
            resolved["brownian_composition_revision"],
            resolved["brownian_charge_dense_revision"],
            models["noise"]["revision"],
            models["drag"]["revision"],
        )
        if actual != expected:
            raise RuntimeError(f"XY midpoint control changed: {actual}")
        return
    expected = (
        "macro_root_frozen_midpoint_v1",
        "stochastic_exponential_midpoint_v1",
        "macro_root_affine_exponential_v2",
        _NOISE_REVISION,
    )
    actual = (
        resolved["brownian_coefficient_policy"],
        resolved["brownian_composition_revision"],
        resolved["brownian_charge_dense_revision"],
        models["noise"]["revision"],
    )
    if actual != expected:
        raise RuntimeError(f"B03 resolved contract changed: {actual}")
    if scenario == _B03_CONTINUOUS:
        if models["drag"]["revision"] != _EFFECTIVE_DRAG_REVISION:
            raise RuntimeError("B03 continuous case did not resolve effective-gas linear drag")
        if models["charge"]["revision"] != _CONTINUOUS_CHARGE_REVISION:
            raise RuntimeError("B03 continuous case did not resolve continuous charge")
        if models["gravity_buoyancy"]["revision"] != "gravity_buoyancy_standard_v1":
            raise RuntimeError("B03 continuous case did not resolve gravity")
    elif models["drag"]["revision"] != _NATIVE_DRAG_REVISION:
        raise RuntimeError("B03 fixed case did not resolve native linear drag")
    axis_crossings = int(work["manifest_axis_crossings"])
    if scenario == _B03_AXIS and axis_crossings < 1:
        raise RuntimeError("B03 axis characterization did not exercise a root restart")
    if scenario != _B03_AXIS and axis_crossings != 0:
        raise RuntimeError(f"B03 non-axis case unexpectedly crossed the axis: {scenario}")


def _paired_characterization(
    outputs: Mapping[tuple[int, int, str], Path],
    particle_counts: tuple[int, ...],
    repeats: int,
) -> dict[str, object]:
    comparisons = []
    for count in particle_counts:
        for repeat in range(repeats):
            b02 = open_result(outputs[(count, repeat, _B02)]).read_final()
            b03 = open_result(outputs[(count, repeat, _B03_FIXED)]).read_final()
            if not np.array_equal(b02.particle_id, b03.particle_id):
                raise RuntimeError("B03 paired case changed particle identity")
            if not np.array_equal(b02.lifecycle, b03.lifecycle):
                raise RuntimeError("B03 paired case changed lifecycle")
            translated_position = b03.position_m.copy()
            translated_position[:, 0] -= _PAIR_RADIAL_SHIFT_M
            position_delta = translated_position - b02.position_m
            velocity_delta = b03.velocity_m_s - b02.velocity_m_s
            charge_delta = b03.charge_number - b02.charge_number
            comparisons.append(
                {
                    "particle_count": count,
                    "repeat": repeat,
                    "translated_position_bitwise_equal": np.array_equal(
                        translated_position, b02.position_m
                    ),
                    "velocity_bitwise_equal": np.array_equal(b03.velocity_m_s, b02.velocity_m_s),
                    "charge_bitwise_equal": np.array_equal(b03.charge_number, b02.charge_number),
                    "translated_position_rms_difference_m": _rms(position_delta),
                    "translated_position_maximum_difference_m": _maximum_abs(position_delta),
                    "velocity_rms_difference_m_s": _rms(velocity_delta),
                    "velocity_maximum_difference_m_s": _maximum_abs(velocity_delta),
                    "charge_rms_difference_number": _rms(charge_delta),
                    "charge_maximum_difference_number": _maximum_abs(charge_delta),
                }
            )
    return {
        "shared_physical_coefficients": {
            "particle_mass_kg": 4.0e-15,
            "linear_drag_rate_s_inverse": 2.0,
            "gas_temperature_K": 300.0,
            "gas_velocity_m_s": _PAIR_GAS_VELOCITY_M_S.tolist(),
            "initial_velocity_m_s": _PAIR_INITIAL_VELOCITY_M_S.tolist(),
            "initial_charge_number": 0.0,
            "noise_seed": _SEED,
            "native_drag_revision": _NATIVE_DRAG_REVISION,
            "only_coordinate_construction_difference": (
                f"B03 radial coordinate is B02 x + {_PAIR_RADIAL_SHIFT_M} m; uniform field "
                "values, particle properties, time grid, tree depth, output schedule, and seed match"
            ),
        },
        "comparisons": comparisons,
        "interpretation": (
            "Constant coefficients make frozen-start and frozen-midpoint coefficient values equal. "
            "Reported payload differences characterize floating-point/coordinate-path effects and "
            "are not an acceptance threshold."
        ),
    }


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64), dtype=np.float64)))


def _maximum_abs(values: np.ndarray) -> float:
    return float(np.max(np.abs(values)))


def _summaries(observations: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    keys = sorted({(str(item["scenario"]), int(item["particle_count"])) for item in observations})
    result: list[dict[str, object]] = []
    for scenario, count in keys:
        selected = [
            item
            for item in observations
            if item["scenario"] == scenario and item["particle_count"] == count
        ]
        timing = [item["timing_s"] for item in selected]
        memory = [item["memory"] for item in selected]
        representative = selected[0]
        traced = [item["tracemalloc"] for item in selected if item["tracemalloc"] is not None]
        result.append(
            {
                "scenario": scenario,
                "particle_count": count,
                "observations": len(selected),
                "public_end_to_end_repeat_seconds": [
                    float(item["public_end_to_end_s"]) for item in timing
                ],
                "median_public_end_to_end_s": statistics.median(
                    float(item["public_end_to_end_s"]) for item in timing
                ),
                "median_simulate_s": statistics.median(
                    float(item["simulate_s"]) for item in timing
                ),
                "maximum_process_peak_rss_bytes": max(
                    int(item["process_peak_rss_after_measured_pipeline_bytes"]) for item in memory
                ),
                "solver_planned_bytes": int(memory[0]["solver_planned_bytes"]),
                "slab_particles": int(memory[0]["slab_particles"]),
                "slab_proposal_scratch_bytes": int(memory[0]["slab_proposal_scratch_bytes"]),
                "slab_stochastic_tree_work_bytes": int(
                    memory[0]["slab_stochastic_tree_work_bytes"]
                ),
                "tracemalloc_additional_peak_bytes": (
                    int(traced[0]["additional_peak_over_baseline_bytes"]) if traced else None
                ),
                "stochastic_work": representative["stochastic_work"],
                "scientific_payload_sha256": representative["scientific_payload_sha256"],
            }
        )
    return result


def _array_accounting(observations: Sequence[Mapping[str, object]]) -> dict[str, object]:
    columns = {
        # particle id + position/velocity/charge + the three one-byte verdicts
        "predictor_return": {"float64": 5, "int64": 1, "byte": 3},
        # Predictor construction deliberately includes its returned state again.
        "predictor_construction_and_start_relaxation": {
            "float64": 13,
            "int64": 1,
            "byte": 3,
        },
        # Duration/midpoint, row selection, two failure-code buffers, and a
        # conservative twelve simultaneous one-byte classifier masks/verdicts.
        "root_guard_and_classification": {
            "float64": 2,
            "int64": 1,
            "uint16": 2,
            "byte": 12,
        },
        "full_midpoint_coefficient_table": {
            "float64": 7,
            "int64": 1,
            "byte": 3,
        },
        "effective_equilibrium_safety_and_selection": {
            "float64": 7,
            "int64": 1,
            "byte": 5,
        },
        "valid_coefficient_subset": {"float64": 7, "int64": 1, "byte": 3},
        "returned_root_batch_state_and_safe_values": {
            "float64": 12,
            "int64": 1,
            "byte": 1,
        },
        # Start/end gathers, min/max interval, retained lower/upper interval,
        # and ordered/certified/status/temporary one-byte certificate columns.
        "cubic_dense_charge_invariant_certificate": {"float64": 6, "byte": 4},
        # Includes restart rows/particles/result selection, accepted time,
        # position/velocity copies, and one 2-vector advanced-index transient.
        "axis_restart_selected_copies": {"float64": 7, "int64": 3},
        # Per-row root/event/guard ordinals and the pending fresh-root wave are
        # owned by the separately reported stochastic-tree reserve.
        "fresh_root_wave_metadata": {},
    }
    item_sizes = {"float64": 8, "int64": 8, "uint16": 2, "byte": 1}
    groups = {
        name: {
            "columns_per_particle": counts,
            "raw_bytes_per_particle": sum(
                item_sizes[dtype] * count for dtype, count in counts.items()
            ),
        }
        for name, counts in columns.items()
    }
    raw = sum(int(group["raw_bytes_per_particle"]) for group in groups.values())
    rounded = (raw + 7) // 8 * 8
    plan_revisions = {str(item["revisions"]["memory_plan_revision"]) for item in observations}
    scratch_values = {int(item["memory"]["scratch_bytes_per_particle"]) for item in observations}
    if len(plan_revisions) != 1 or len(scratch_values) != 1:
        raise RuntimeError("B03 observations disagree on memory-plan revision or stage scratch")
    plan_revision = next(iter(plan_revisions))
    scratch = next(iter(scratch_values))
    checks = {
        "conservative_named_array_bound_fits_stage_scratch": rounded <= scratch,
        "all_rows_use_memory_plan_v16": all(
            item["revisions"]["memory_plan_revision"] == _MEMORY_PLAN_REVISION
            for item in observations
        ),
        "all_rows_keep_2048_bytes_per_particle": scratch == 2_048,
        "all_scratch_components_match_slab_formula": all(
            item["memory"]["slab_proposal_scratch_bytes"]
            == item["memory"]["slab_particles"] * scratch
            for item in observations
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"B03 memory-array accounting failed: {checks}")
    return {
        "memory_plan_revision_from_manifests": plan_revision,
        "general_stage_scratch_bytes_per_particle": scratch,
        "conservative_named_b03_path_array_bytes_per_particle": raw,
        "rounded_conservative_bytes_per_particle": rounded,
        "remaining_stage_scratch_headroom_bytes_per_particle": scratch - rounded,
        "groups": groups,
        "dtype_item_sizes_bytes": item_sizes,
        "method": (
            "Raw NumPy data-buffer bytes. The audit deliberately counts phase-separated root "
            "preparation and axis-restart arrays together, double-counts the returned predictor, "
            "and includes arrays shared with B02, so it over-bounds B03's named additions. "
            "Generic field/physics workspaces and the pre-existing Hermite/event arrays remain "
            "in the established general stage allowance."
        ),
        "root_ordinal_note": (
            "root_interval and restart counters are per-row counter-addressing state; their "
            "current and pending waves are included in stochastic_tree_work_bytes_per_particle, "
            "not this general stage-scratch subtotal"
        ),
        "checks": checks,
        "measurement_note": (
            "The separate tracemalloc and process-RSS observations include broader runtime and "
            "I/O allocations; they are consistency observations, not a direct meter for this "
            "per-row data-buffer proof."
        ),
    }


def _case_matrix() -> list[dict[str, object]]:
    return [
        {
            "scenario": _B02,
            "coordinate": "cartesian_xy",
            "charge": "fixed",
            "drag": _NATIVE_DRAG_REVISION,
            "additive_force": "none",
            "role": "frozen-start legacy-semantics control",
        },
        {
            "scenario": _B03_FIXED,
            "coordinate": "axisymmetric_rz meridional projected",
            "charge": "fixed",
            "drag": _NATIVE_DRAG_REVISION,
            "additive_force": "none",
            "role": "constant-coefficient paired B03 row; no axis crossing",
        },
        {
            "scenario": _B03_CONTINUOUS,
            "coordinate": "axisymmetric_rz meridional projected",
            "charge": _CONTINUOUS_CHARGE_REVISION,
            "drag": _EFFECTIVE_DRAG_REVISION,
            "additive_force": "gravity_buoyancy_standard_v1",
            "role": "one supported continuous-charge plus force composition",
        },
        {
            "scenario": _B03_AXIS,
            "coordinate": "axisymmetric_rz meridional projected",
            "charge": "fixed",
            "drag": _NATIVE_DRAG_REVISION,
            "additive_force": "none",
            "role": "nonzero axis-crossing/root-restart work observation",
        },
    ]


def _process_memory_bytes() -> tuple[int, int | None, str]:
    if sys.platform == "win32":
        counters = _windows_process_memory()
        return (
            int(counters.peak_working_set_size),
            int(counters.working_set_size),
            "Windows GetProcessMemoryInfo",
        )
    import resource

    maximum = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    peak_bytes = maximum if sys.platform == "darwin" else maximum * 1_024
    if sys.platform.startswith("linux"):
        fields_value = Path("/proc/self/statm").read_text(encoding="ascii").split()
        resident_pages = int(fields_value[1])
        return peak_bytes, resident_pages * int(os.sysconf("SC_PAGE_SIZE")), "getrusage + /proc"
    return peak_bytes, None, "getrusage"


def _windows_process_memory() -> _WindowsProcessMemoryCounters:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    get_current_process = kernel32.GetCurrentProcess
    get_current_process.restype = ctypes.c_void_p
    get_process_memory_info = psapi.GetProcessMemoryInfo
    get_process_memory_info.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_WindowsProcessMemoryCounters),
        ctypes.c_ulong,
    ]
    get_process_memory_info.restype = ctypes.c_int
    counters = _WindowsProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    if not get_process_memory_info(
        get_current_process(),
        ctypes.byref(counters),
        counters.cb,
    ):
        raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
    return counters


def _scientific_payload_digest(result: Any) -> str:
    digest = hashlib.sha256()
    for name, reader in (
        ("final", result.read_final),
        ("release_events", result.read_release_events),
        ("boundary_events", result.read_boundary_events),
        ("failure_events", result.read_failure_events),
        ("lifecycle_series", result.read_lifecycle_series),
    ):
        value = reader()
        digest.update(name.encode("utf-8"))
        for field in fields(value):
            array = np.asarray(getattr(value, field.name))
            digest.update(field.name.encode("utf-8"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(struct.pack("<Q", array.ndim))
            for extent in array.shape:
                digest.update(struct.pack("<Q", extent))
            if array.dtype.kind in "OSU":
                digest.update(
                    json.dumps(array.tolist(), ensure_ascii=False, separators=(",", ":")).encode(
                        "utf-8"
                    )
                )
            else:
                digest.update(np.ascontiguousarray(array).tobytes())
    return f"sha256:{digest.hexdigest()}"


def _machine_metadata() -> dict[str, object]:
    repository = Path(__file__).resolve().parents[2]
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unreported",
        "logical_cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "h5py": h5py.__version__,
        "chamber_particles": metadata.version("chamber-particles"),
        "uv_lock_sha256": _file_sha256(repository / "uv.lock"),
        "driver_sha256": _file_sha256(Path(__file__).resolve()),
        "peak_rss_source": _process_memory_bytes()[2],
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1_024 * 1_024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


if __name__ == "__main__":
    main()
