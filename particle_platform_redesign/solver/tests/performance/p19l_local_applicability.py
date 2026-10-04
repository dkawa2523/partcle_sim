"""Machine-local P19-L global-first versus local-certificate observation.

The pair uses the public ``load_case -> simulate -> open_result`` workflow.  It
changes only unused remote field values: one case is globally certifiable and
the other requires the bounded local-cell certificate for the same safe
left-cell trajectories.  Timings are descriptive and never form a portable
acceptance threshold.
"""

from __future__ import annotations

import argparse
import cProfile
import gc
import hashlib
import json
import os
import platform
import pstats
import statistics
import sys
import tempfile
import time
import tracemalloc
from dataclasses import fields, replace
from importlib import metadata
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

from chamber_particles import load_case, open_result, simulate
from chamber_particles.case_format import write
from tests.scenarios.test_stokes_cunningham_run import _local_applicability_case

_DEFAULT_PARTICLES = 4_096
_DEFAULT_REPEATS = 7
_REMOTE_SAFE_VELOCITY_M_S = 0.0
_REMOTE_EXTREME_VELOCITY_M_S = 100.0
_LOCAL_CELL_MAXIMUM = 64
_DENSE_PATH_BYTES_PER_ROW = 176


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=int, default=_DEFAULT_PARTICLES)
    parser.add_argument("--repeats", type=int, default=_DEFAULT_REPEATS)
    parser.add_argument("--json", type=Path)
    arguments = parser.parse_args(argv)
    if arguments.particles < 1:
        parser.error("--particles must be positive")
    if arguments.repeats < 1:
        parser.error("--repeats must be positive")

    report = _run_benchmark(arguments.particles, arguments.repeats)
    rendered = json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    if arguments.json is None:
        print(rendered, end="")
        return
    arguments.json.parent.mkdir(parents=True, exist_ok=True)
    arguments.json.write_text(rendered, encoding="utf-8")


def _run_benchmark(particle_count: int, repeats: int) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="p19l-local-applicability-") as temporary:
        root = Path(temporary)
        case_paths = {
            "global_fast": _materialize_safe_cohort(
                root / "global-fast",
                particle_count,
                remote_velocity_m_s=_REMOTE_SAFE_VELOCITY_M_S,
            ),
            "local_fallback": _materialize_safe_cohort(
                root / "local-fallback",
                particle_count,
                remote_velocity_m_s=_REMOTE_EXTREME_VELOCITY_M_S,
            ),
        }
        cases = {name: load_case(path) for name, path in case_paths.items()}
        warm_results = {
            name: _simulate_and_open(case, root / "warm" / name) for name, case in cases.items()
        }
        timings, execution_order = _timed_observations(cases, root, repeats)
        profiles = {
            name: _profile_simulate(case, root / "profile" / name) for name, case in cases.items()
        }
        _validate_profile_paths(profiles)
        traced_memory = {
            name: _trace_simulate_allocations(case, root / "trace" / name)
            for name, case in cases.items()
        }
        identity = _compare_scientific_payloads(warm_results)
        if (
            identity["active_final_rows"] != particle_count
            or identity["failure_event_rows"] != 0
            or identity["boundary_event_rows"] != 0
        ):
            raise RuntimeError(f"P19-L safe cohort produced an unexpected outcome: {identity}")
        memory = _memory_accounting(warm_results, particle_count)
        revisions = _revisions(warm_results["global_fast"].manifest)

    medians = {name: statistics.median(values) for name, values in timings.items()}
    return {
        "benchmark_role": "machine_local_non_gating_public_api_global_fast_vs_local_fallback",
        "particle_count": particle_count,
        "macro_steps": 20,
        "repeats_per_mode": repeats,
        "execution_order": execution_order,
        "case_pair": {
            "shared": (
                "safe left-cell Stokes-Cunningham XY cohort; dt=5e-6 s; end=1e-4 s; "
                "fixed-step general RK4; no trajectory output"
            ),
            "only_input_difference": (
                "unused far-right gas-velocity x component is 0 m/s for global_fast and "
                "100 m/s for local_fallback"
            ),
            "global_fast_expected_path": "global applicability proof; no local cell ranges",
            "local_fallback_expected_path": (
                "global proof inconclusive; bounded left-cell applicability proof"
            ),
        },
        "timing": {
            name: {
                "repeat_seconds": values,
                "median_seconds": medians[name],
                "particle_steps_per_second": particle_count * 20 / medians[name],
            }
            for name, values in timings.items()
        }
        | {
            "local_fallback_to_global_fast_median_ratio": medians["local_fallback"]
            / medians["global_fast"]
        },
        "separate_cprofile_observation": profiles,
        "separate_tracemalloc_observation": traced_memory,
        "scientific_payload_identity": identity,
        "memory_accounting": memory,
        "revisions": revisions,
        "environment": _environment(),
        "scope_limit": (
            "Current-tree fast-versus-local observation only; not a pre-P19 comparison, "
            "portable speed threshold, RSS measurement, or COMSOL/physics validation."
        ),
    }


def _materialize_safe_cohort(
    directory: Path,
    particle_count: int,
    *,
    remote_velocity_m_s: float,
) -> Path:
    base_path = _local_applicability_case(
        directory,
        remote_velocity_m_s=remote_velocity_m_s,
    )
    case = load_case(base_path)
    original = case.data.sources[0]

    def repeated(value: np.ndarray) -> np.ndarray:
        return np.repeat(value[:1], particle_count, axis=0)

    source = replace(
        original,
        particle_id=np.arange(1, particle_count + 1, dtype="<i8"),
        release_time_s=np.zeros(particle_count, dtype="<f8"),
        position_m=repeated(original.position_m),
        velocity_m_s=repeated(original.velocity_m_s),
        charge_number=repeated(original.charge_number),
        mass_kg=repeated(original.mass_kg),
        drag_diameter_m=repeated(original.drag_diameter_m),
        electrostatic_radius_m=repeated(original.electrostatic_radius_m),
        displaced_volume_m3=repeated(original.displaced_volume_m3),
        model_weight=repeated(original.model_weight),
        material_id=repeated(original.material_id),
    )
    data_path = base_path.with_name("p19l-performance.h5")
    info = write(data_path, replace(case.data, sources=(source,)))
    document = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    document["case"]["data_path"] = data_path.name
    document["case"]["expected_content_hash"] = info.content_hash
    case_path = base_path.with_name("p19l-performance.yaml")
    case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return case_path


def _simulate_and_open(case: Any, output_path: Path) -> Any:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    simulate(case, output_path)
    return open_result(output_path)


def _timed_observations(
    cases: dict[str, Any],
    root: Path,
    repeats: int,
) -> tuple[dict[str, list[float]], list[str]]:
    values = {name: [] for name in cases}
    execution_order: list[str] = []
    for repeat in range(repeats):
        names = ("global_fast", "local_fallback")
        if repeat % 2:
            names = tuple(reversed(names))
        for name in names:
            output_path = root / "timed" / f"{repeat:02d}-{name}"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            simulate(cases[name], output_path)
            values[name].append(time.perf_counter() - started)
            execution_order.append(name)
    return values, execution_order


def _profile_simulate(case: Any, output_path: Path) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile = cProfile.Profile()
    profile.enable()
    simulate(case, output_path)
    profile.disable()
    call_counts = _profile_call_counts(profile)
    return {
        "local_component_bounds_calls": call_counts.get("local_component_bounds", 0),
        "local_continuous_applicability_batch_calls": call_counts.get(
            "local_continuous_applicability_batch", 0
        ),
        "global_continuous_applicability_batch_calls": call_counts.get(
            "global_continuous_applicability_batch", 0
        ),
        "rk4_dense_interval_range_certificate_calls": call_counts.get(
            "_rk4_dense_interval_range_certificate", 0
        ),
        "interpretation": "separate untimed observational profile; no monkeypatch or mock",
    }


def _profile_call_counts(profile: cProfile.Profile) -> dict[str, int]:
    result: dict[str, int] = {}
    for (_filename, _line, name), statistics_row in pstats.Stats(profile).stats.items():
        total_calls = int(statistics_row[1])
        result[name] = result.get(name, 0) + total_calls
    return result


def _validate_profile_paths(profiles: dict[str, dict[str, object]]) -> None:
    expected_steps = 20
    fast = profiles["global_fast"]
    local = profiles["local_fallback"]
    if fast["local_component_bounds_calls"] != 0:
        raise RuntimeError("P19-L global-fast profile unexpectedly constructed local ranges")
    if local["local_component_bounds_calls"] != expected_steps:
        raise RuntimeError("P19-L local-fallback profile did not construct one range per step")
    for profile in profiles.values():
        if profile["global_continuous_applicability_batch_calls"] != expected_steps:
            raise RuntimeError("P19-L profile did not attempt one global proof per step")
        if profile["rk4_dense_interval_range_certificate_calls"] != expected_steps:
            raise RuntimeError("P19-L profile did not exercise one dense certificate per step")


def _trace_simulate_allocations(case: Any, output_path: Path) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gc.collect()
    tracemalloc.start()
    try:
        baseline_bytes, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        simulate(case, output_path)
        current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return {
        "baseline_traced_bytes": baseline_bytes,
        "current_traced_bytes_after_simulate": current_bytes,
        "peak_traced_bytes": peak_bytes,
        "additional_peak_over_baseline_bytes": max(0, peak_bytes - baseline_bytes),
        "semantics": (
            "Python tracemalloc scope around public simulate; includes visible NumPy/Python "
            "allocations and result writing; not process RSS or solver-owned-plan replacement"
        ),
    }


def _compare_scientific_payloads(results: dict[str, Any]) -> dict[str, object]:
    payloads = {name: _scientific_payload(result) for name, result in results.items()}
    names = tuple(payloads)
    identical = _payloads_equal(payloads[names[0]], payloads[names[1]])
    if not identical:
        raise RuntimeError("P19-L public pair changed the scientific output payload")
    digests = {name: _payload_digest(payload) for name, payload in payloads.items()}
    if len(set(digests.values())) != 1:
        raise RuntimeError("P19-L equal payloads produced different digests")
    final = payloads["global_fast"]["final"]
    failures = payloads["global_fast"]["failure_events"]
    boundaries = payloads["global_fast"]["boundary_events"]
    return {
        "bitwise_identical": True,
        "sha256_by_mode": digests,
        "included": [
            "final particles",
            "release events",
            "boundary events",
            "failure events",
            "lifecycle series",
        ],
        "active_final_rows": int(np.count_nonzero(final.lifecycle == 1)),
        "failure_event_rows": int(failures.particle_id.size),
        "boundary_event_rows": int(boundaries.particle_id.size),
        "manifest_excluded_because_input_content_hashes_intentionally_differ": True,
    }


def _scientific_payload(result: Any) -> dict[str, Any]:
    return {
        "final": result.read_final(),
        "release_events": result.read_release_events(),
        "boundary_events": result.read_boundary_events(),
        "failure_events": result.read_failure_events(),
        "lifecycle_series": result.read_lifecycle_series(),
    }


def _payloads_equal(first: dict[str, Any], second: dict[str, Any]) -> bool:
    for section, first_value in first.items():
        second_value = second[section]
        for field in fields(first_value):
            if not np.array_equal(
                np.asarray(getattr(first_value, field.name)),
                np.asarray(getattr(second_value, field.name)),
            ):
                return False
    return True


def _payload_digest(payload: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for section, value in payload.items():
        digest.update(section.encode("utf-8"))
        for field in fields(value):
            array = np.asarray(getattr(value, field.name))
            digest.update(field.name.encode("utf-8"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(json.dumps(array.shape).encode("ascii"))
            if array.dtype.kind in "OSU":
                digest.update(
                    json.dumps(array.tolist(), ensure_ascii=False, separators=(",", ":")).encode(
                        "utf-8"
                    )
                )
            else:
                digest.update(np.ascontiguousarray(array).tobytes())
    return f"sha256:{digest.hexdigest()}"


def _memory_accounting(results: dict[str, Any], particle_count: int) -> dict[str, object]:
    plans = {name: result.manifest["memory_plan"] for name, result in results.items()}
    if plans["global_fast"] != plans["local_fallback"]:
        raise RuntimeError("P19-L pair resolved different solver-owned memory plans")
    plan = plans["global_fast"]
    slab_particles = int(plan["slab_particles"])
    split_budget = 2
    stack_capacity = split_budget + 1
    actual_stack_bytes_per_row = 16 * stack_capacity + 18
    planned_stack_bytes_per_row = 16 * stack_capacity + 24
    candidate_actual_at_slab = 536 * slab_particles + 8
    candidate_planned_at_slab = 544 * slab_particles
    certificate_bytes_per_row = planned_stack_bytes_per_row + 544
    components = plan["components"]
    checks = {
        "dense_path_manifest_is_176_bytes_per_row": (
            int(plan["dense_path_bytes_per_particle"]) == _DENSE_PATH_BYTES_PER_ROW
        ),
        "dense_path_component_matches_slab": (
            int(components["slab_dense_path"]) == slab_particles * _DENSE_PATH_BYTES_PER_ROW
        ),
        "certificate_manifest_matches_stack_plus_candidate": (
            int(plan["certificate_work_bytes_per_particle"]) == certificate_bytes_per_row
        ),
        "certificate_component_matches_slab": (
            int(components["slab_certificate_work"]) == slab_particles * certificate_bytes_per_row
        ),
        "candidate_arena_contains_maximum_arrays": (
            candidate_actual_at_slab <= candidate_planned_at_slab
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"P19-L memory accounting check failed: {checks}")
    return {
        "same_plan_for_both_modes": True,
        "manifest": plan,
        "audit": {
            "rk4_dense_path": {
                "actual_and_planned_bytes_per_row": _DENSE_PATH_BYTES_PER_ROW,
                "arrays": (
                    "start/target time; 4x2 position controls; 4x2 velocity controls; "
                    "4 charge controls; all float64"
                ),
            },
            "certificate_interval_stack": {
                "split_budget": split_budget,
                "stack_capacity": stack_capacity,
                "actual_bytes_per_row": actual_stack_bytes_per_row,
                "planned_bytes_per_row": planned_stack_bytes_per_row,
                "headroom_bytes_per_row": (
                    planned_stack_bytes_per_row - actual_stack_bytes_per_row
                ),
            },
            "local_candidate_arena": {
                "maximum_cells_per_row": _LOCAL_CELL_MAXIMUM,
                "actual_peak_formula_bytes": "536*N+8",
                "actual_peak_at_slab_bytes": candidate_actual_at_slab,
                "planned_formula_bytes": "544*N",
                "planned_at_slab_bytes": candidate_planned_at_slab,
                "headroom_at_slab_bytes": (candidate_planned_at_slab - candidate_actual_at_slab),
                "arrays": (
                    "int64 counts, bounded counts, N+1 CSR offsets, and at most 64*N "
                    "int64 candidate IDs"
                ),
            },
            "primitive_range_temporaries": {
                "accounting_owner": "general stage scratch",
                "general_stage_scratch_bytes_per_row": int(plan["scratch_bytes_per_particle"]),
                "benchmark_persistent_lower_upper_bytes_per_row": 80,
            },
        },
        "checks": checks,
        "particle_count_matches_single_slab": slab_particles == particle_count,
    }


def _revisions(manifest: Any) -> dict[str, object]:
    keys = (
        "engine_algorithm_revision",
        "compiled_cpu_tile_revision",
        "step_proposal_revision",
        "rk4_enclosure_revision",
        "rk4_dense_path_revision",
        "physics_catalog_revision",
        "physics_runtime_revision",
        "field_location_revision",
        "event_algorithm_revision",
    )
    revisions = {key: manifest[key] for key in keys}
    revisions["memory_plan_revision"] = manifest["memory_plan"]["revision"]
    return revisions


def _environment() -> dict[str, object]:
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
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


if __name__ == "__main__":
    main()
