"""Manual, non-gating P20 cadence and charge-step efficiency evidence.

Both workloads execute through ``load_case -> simulate -> open_result``.  Case
materialization, case loading, result opening, validation, and hashing are
outside the timed ``simulate`` scope.  The fixed-64 comparison temporarily
changes only the two internal cadence constants in this benchmark process; it
does not expose or exercise a second production policy.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import statistics
import struct
import sys
import tempfile
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import fields, replace
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

import chamber_particles.engine as engine_module
from chamber_particles import load_case, open_result, simulate
from chamber_particles.case_format import write
from tests.performance.p06_baseline import _expanded_definition, _materialize, _repeat_source
from tests.scenarios.test_force_coupled_run import _continuous_charge_case
from tests.verification.microcases import build_microcase

_CADENCE_DT_S = 1.0e-3
_CHARGE_DENSITY_M3 = 1.0e9
_FIXED_64_MACRO_STEPS = 64
_CHARGE_REVISION = "charge_stable_exponential_midpoint_v3"
_EXPECTED_PRODUCTION_REVISIONS = {
    "engine_algorithm_revision": "particle_engine_v36",
    "compiled_cpu_tile_revision": "compiled_cpu_tile_v18",
    "step_proposal_revision": "coupled_fixed_step_proposal_v10",
    "physics_runtime_revision": "signed_ion_compiled_physics_runtime_v19",
    "result_algorithm_revision": "durable_segmented_result_v5",
    "event_algorithm_revision": "line_quadratic_rk4_axis_first_hit_v16",
    "memory_plan_revision": "solver_owned_memory_plan_v13",
    "runtime_layout_revision": "resident_soa_serial_slab_v6",
}


def main(argv: list[str] | None = None) -> None:
    """Run the focused benchmark and optionally persist its JSON report."""

    arguments = _arguments(argv)
    report = _run(arguments)
    rendered = json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    if arguments.json_path is None:
        print(rendered, end="")
        return
    arguments.json_path.parent.mkdir(parents=True, exist_ok=True)
    arguments.json_path.write_text(rendered, encoding="utf-8")


def _arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cadence-particles", type=_positive_int, default=8)
    parser.add_argument("--cadence-macro-steps", type=_positive_int, default=4_096)
    parser.add_argument("--charge-particles", type=_positive_int, default=128)
    parser.add_argument("--charge-end-s", type=_positive_float, default=32.0)
    parser.add_argument("--charge-old-dt-s", type=_positive_float, default=0.25)
    parser.add_argument("--charge-large-dt-s", type=_positive_float, default=2.0)
    parser.add_argument("--warmups", type=_nonnegative_int, default=1)
    parser.add_argument("--repeats", type=_positive_int, default=3)
    parser.add_argument("--json", dest="json_path", type=Path)
    result = parser.parse_args(argv)
    if result.charge_large_dt_s <= result.charge_old_dt_s:
        parser.error("--charge-large-dt-s must exceed --charge-old-dt-s")
    return result


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _nonnegative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return result


def _positive_float(value: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return result


def _run(arguments: argparse.Namespace) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="chamber-particles-p20-") as temporary:
        root = Path(temporary)
        cadence_path = _materialize_cadence_case(
            root / "cases" / "cadence",
            arguments.cadence_particles,
            arguments.cadence_macro_steps,
        )
        charge_paths = _materialize_charge_cases(
            root / "cases" / "charge",
            arguments.charge_particles,
            arguments.charge_end_s,
            arguments.charge_old_dt_s,
            arguments.charge_large_dt_s,
        )
        # Loading is deliberately setup, not part of the simulate-only timing.
        cadence_case = load_case(cadence_path)
        charge_cases = {name: load_case(path) for name, path in charge_paths.items()}

        cadence_observations = _measure_pair(
            cases={"current_work_scaled": cadence_case, "fixed_64_emulation": cadence_case},
            output_root=root / "results" / "cadence",
            warmups=arguments.warmups,
            repeats=arguments.repeats,
            fixed_64_particles={"fixed_64_emulation": arguments.cadence_particles},
        )
        charge_observations = _measure_pair(
            cases=charge_cases,
            output_root=root / "results" / "charge",
            warmups=arguments.warmups,
            repeats=arguments.repeats,
            fixed_64_particles={},
        )

        cadence = _cadence_report(
            cadence_observations,
            arguments.cadence_particles,
            arguments.cadence_macro_steps,
        )
        charge = _charge_report(
            charge_observations,
            arguments.charge_particles,
            arguments.charge_end_s,
        )

    return {
        "benchmark": "p20_efficiency_v1",
        "benchmark_role": "manual_machine_local_non_gating",
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "conditions": {
            "warmups_per_mode": arguments.warmups,
            "timed_repeats_per_mode": arguments.repeats,
            "timed_scope": "simulate only",
            "outside_timed_scope": (
                "case materialization, load_case, open_result, validation, artifact accounting, "
                "and payload hashing"
            ),
            "execution_order": (
                "one warmup per mode, then alternating mode order by timed repeat in one process"
            ),
            "jit": "warm in-process compiled execution; cold compilation is excluded",
            "comsol_dependency": False,
        },
        "machine": _machine_metadata(),
        "cadence_efficiency": cadence,
        "continuous_charge_operational_utility": charge,
        "interpretation": {
            "portable_threshold": "none; elapsed times and ratios are machine-local observations",
            "cadence_claim": (
                "the cadence pair has identical public scientific payload bytes; only durable "
                "partitioning and its manifest identity differ"
            ),
            "charge_accuracy_claim": (
                "none between the two benchmark step sizes; the larger step demonstrates stable "
                "operational execution and relies on separate convergence and stiff-limit tests"
            ),
        },
    }


def _materialize_cadence_case(directory: Path, particles: int, macro_steps: int) -> Path:
    definition = _expanded_definition(build_microcase("C01"), particles)
    source = definition.data.sources[0]
    stationary = replace(source, velocity_m_s=np.zeros_like(source.velocity_m_s))
    definition = replace(definition, data=replace(definition.data, sources=(stationary,)))
    case_path = _materialize(
        directory,
        definition,
        name="P20 small-N long-run cadence",
        frame_times_s=None,
    )
    document = yaml.safe_load(case_path.read_text(encoding="utf-8"))
    document["time"] = {
        "start_s": 0.0,
        "end_s": macro_steps * _CADENCE_DT_S,
        "dt_s": _CADENCE_DT_S,
    }
    case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return case_path


def _materialize_charge_cases(
    directory: Path,
    particles: int,
    end_s: float,
    old_dt_s: float,
    large_dt_s: float,
) -> dict[str, Path]:
    base_path = _continuous_charge_case(
        directory,
        integrator="exponential_midpoint",
        number_density_m3=_CHARGE_DENSITY_M3,
        dt_s=old_dt_s,
        electric=False,
        output=False,
    )
    base = load_case(base_path)
    source = _repeat_source(base.data.sources[0], particles)
    source = replace(source, velocity_m_s=np.zeros_like(source.velocity_m_s))
    data_path = directory / "p20-charge.h5"
    info = write(data_path, replace(base.data, sources=(source,)))
    base_document = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    base_document["case"]["data_path"] = data_path.name
    base_document["case"]["expected_content_hash"] = info.content_hash
    base_document["time"]["end_s"] = end_s

    paths: dict[str, Path] = {}
    for name, dt_s in (
        ("old_admissible_step", old_dt_s),
        ("stable_larger_step", large_dt_s),
    ):
        document = yaml.safe_load(yaml.safe_dump(base_document, sort_keys=False))
        document["case"]["name"] = f"P20 continuous charge {name}"
        document["time"]["dt_s"] = dt_s
        path = directory / f"{name}.yaml"
        path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        paths[name] = path
    return paths


def _measure_pair(
    *,
    cases: Mapping[str, Any],
    output_root: Path,
    warmups: int,
    repeats: int,
    fixed_64_particles: Mapping[str, int],
) -> dict[str, list[dict[str, object]]]:
    observations = {name: [] for name in cases}
    for name, case in cases.items():
        for index in range(warmups):
            _execute(
                case,
                output_root / name / f"warmup-{index:03d}",
                fixed_64_particle_count=fixed_64_particles.get(name),
                measured=False,
            )
    names = list(cases)
    for repeat in range(repeats):
        ordered = names if repeat % 2 == 0 else list(reversed(names))
        for name in ordered:
            observation = _execute(
                cases[name],
                output_root / name / f"measured-{repeat:03d}",
                fixed_64_particle_count=fixed_64_particles.get(name),
                measured=True,
            )
            observation["repeat"] = repeat
            observations[name].append(observation)
    return observations


def _execute(
    case: Any,
    output: Path,
    *,
    fixed_64_particle_count: int | None,
    measured: bool,
) -> dict[str, object]:
    gc.collect()
    with _fixed_64_cadence(fixed_64_particle_count):
        started = time.perf_counter()
        summary = simulate(case, output)
        elapsed_s = time.perf_counter() - started
    result = open_result(output)
    if summary.failure_event_count != 0 or summary.boundary_event_count != 0:
        raise RuntimeError("P20 benchmark case produced an unexpected event or failure")
    manifest = result.manifest
    memory_plan = _mapping(manifest, "memory_plan")
    cadence = _mapping(manifest, "durable_commit_cadence")
    resolved = _mapping(manifest, "resolved")
    production_revisions = _production_revision_tuple(manifest, memory_plan, resolved)
    final = result.read_final()
    files = [path for path in output.rglob("*") if path.is_file()]
    segments = list(output.joinpath("segments").glob("epoch-*.h5"))
    checkpoints = list(output.joinpath("checkpoints").glob("*.h5"))
    segment_macro_steps = _segment_macro_steps(segments)
    checkpoint_work = _latest_checkpoint_work(output)
    if fixed_64_particle_count is not None:
        expected_accepted = summary.particle_count * summary.macro_step_count
        expected_total = (summary.particle_count + 1) * summary.macro_step_count
        if (
            summary.particle_count != fixed_64_particle_count
            or manifest["event_refinement"] is not None
            or resolved["path_kind"] != "linear_exact"
            or resolved["boundary_laws"]
            or checkpoint_work["macro_step_count"] != summary.macro_step_count
            or checkpoint_work["accepted_particle_pieces"] != expected_accepted
            or checkpoint_work["candidate_queries"] != 0
            or checkpoint_work["refinements"] != 0
            or checkpoint_work["total"] != expected_total
            or cadence["work_threshold"] != _FIXED_64_MACRO_STEPS * (fixed_64_particle_count + 1)
        ):
            raise RuntimeError(
                "P20 fixed-64 emulation requires W=(N+1)*macro on an exact boundaryless path"
            )
    observation: dict[str, object] = {
        "elapsed_s": elapsed_s if measured else None,
        "particle_count": summary.particle_count,
        "macro_step_count": summary.macro_step_count,
        "segment_count": len(segments),
        "segment_macro_step_min": min(segment_macro_steps),
        "segment_macro_step_max": max(segment_macro_steps),
        "artifact_file_count": len(files),
        "checkpoint_count": len(checkpoints),
        "artifact_bytes": sum(path.stat().st_size for path in files),
        "segment_bytes": sum(path.stat().st_size for path in segments),
        "checkpoint_bytes": sum(path.stat().st_size for path in checkpoints),
        "planned_memory_bytes": int(memory_plan["planned_bytes"]),
        "planned_run_peak_bytes": int(_mapping(memory_plan, "phase_peaks")["run"]),
        "payload_sha256": _scientific_payload_digest(result),
        "durable_commit_cadence": dict(cadence),
        "latest_checkpoint_work": checkpoint_work,
        "production_revision_tuple": production_revisions,
        "maximum_dt_charge_lipschitz": float(manifest["maximum_dt_charge_lipschitz"]),
        "integrator": resolved["integrator"],
        "exponential_midpoint_revision": manifest["exponential_midpoint_revision"],
        "lifecycle_counts": manifest["lifecycle_counts"],
        "failure_reason_counts": manifest["failure_reason_counts"],
        "final_charge_number_min": float(np.min(final.charge_number)),
        "final_charge_number_max": float(np.max(final.charge_number)),
    }
    return observation


@contextmanager
def _fixed_64_cadence(particle_count: int | None) -> Iterator[None]:
    if particle_count is None:
        yield
        return
    previous = (
        engine_module._DURABLE_COMMIT_MINIMUM_WORK,
        engine_module._DURABLE_COMMIT_WORK_PER_PARTICLE,
    )
    engine_module._DURABLE_COMMIT_MINIMUM_WORK = _FIXED_64_MACRO_STEPS * (particle_count + 1)
    engine_module._DURABLE_COMMIT_WORK_PER_PARTICLE = 0
    try:
        yield
    finally:
        (
            engine_module._DURABLE_COMMIT_MINIMUM_WORK,
            engine_module._DURABLE_COMMIT_WORK_PER_PARTICLE,
        ) = previous


def _cadence_report(
    observations: Mapping[str, list[dict[str, object]]],
    particles: int,
    macro_steps: int,
) -> dict[str, object]:
    current = _summarize(observations["current_work_scaled"])
    fixed = _summarize(observations["fixed_64_emulation"])
    work_per_macro = particles + 1
    expected_total_work = macro_steps * work_per_macro
    current_threshold = int(_mapping(current, "durable_commit_cadence")["work_threshold"])
    current_macros_per_segment = (current_threshold + work_per_macro - 1) // work_per_macro
    expected_current_segments = (
        macro_steps + current_macros_per_segment - 1
    ) // current_macros_per_segment
    expected_fixed_segments = (macro_steps + _FIXED_64_MACRO_STEPS - 1) // _FIXED_64_MACRO_STEPS
    for name, summary in (("current", current), ("fixed-64", fixed)):
        work = _mapping(summary, "latest_checkpoint_work")
        if (
            int(work["macro_step_count"]) != macro_steps
            or int(work["accepted_particle_pieces"]) != particles * macro_steps
            or int(work["candidate_queries"]) != 0
            or int(work["refinements"]) != 0
            or int(work["total"]) != expected_total_work
        ):
            raise RuntimeError(f"P20 {name} cadence observation violated W=(N+1)*macro")
    if int(current["segment_count"]) != expected_current_segments:
        raise RuntimeError("P20 current cadence produced an unexpected segment count")
    if int(fixed["segment_count"]) != expected_fixed_segments:
        raise RuntimeError("P20 fixed-64 emulation did not commit every 64 macro steps")
    if int(fixed["segment_macro_step_max"]) > _FIXED_64_MACRO_STEPS:
        raise RuntimeError("P20 fixed-64 emulation produced an oversized segment")
    digests = {item["payload_sha256"] for values in observations.values() for item in values}
    if len(digests) != 1:
        raise RuntimeError("P20 cadence policy changed the scientific payload")
    return {
        "workload": {
            "particles": particles,
            "macro_steps": macro_steps,
            "work_per_macro": work_per_macro,
            "cumulative_work": expected_total_work,
            "motion": "stationary boundaryless ballistic",
            "output": "no trajectory frames or probes",
        },
        "legacy_emulation": (
            "benchmark-process-only cadence constants floor=64*(N+1) and per_particle=0; "
            "the guarded exact ballistic workload contributes N accepted pieces plus one macro "
            "count per step, so this threshold commits every 64 macro steps"
        ),
        "current_work_scaled": current,
        "fixed_64_emulation": fixed,
        "comparisons": {
            "payload_digest_exact_match": True,
            "current_median_speedup_over_fixed_64": (
                float(fixed["median_elapsed_s"]) / float(current["median_elapsed_s"])
            ),
            "segment_count_reduction_factor": (
                int(fixed["segment_count"]) / int(current["segment_count"])
            ),
            "artifact_file_count_reduction": (
                int(fixed["artifact_file_count"]) - int(current["artifact_file_count"])
            ),
            "artifact_byte_reduction": (
                int(fixed["artifact_bytes"]) - int(current["artifact_bytes"])
            ),
            "artifact_byte_reduction_fraction": (
                1.0 - int(current["artifact_bytes"]) / int(fixed["artifact_bytes"])
            ),
            "planned_memory_exact_match": (
                int(current["planned_memory_bytes"]) == int(fixed["planned_memory_bytes"])
            ),
        },
    }


def _charge_report(
    observations: Mapping[str, list[dict[str, object]]],
    particles: int,
    end_s: float,
) -> dict[str, object]:
    old = _summarize(observations["old_admissible_step"])
    large = _summarize(observations["stable_larger_step"])
    old_hl = float(old["maximum_dt_charge_lipschitz"])
    large_hl = float(large["maximum_dt_charge_lipschitz"])
    if old_hl > 0.5:
        raise RuntimeError("P20 old-admissible charge step exceeds hL=0.5")
    if large_hl <= 0.5:
        raise RuntimeError("P20 larger charge step did not exercise hL>0.5")
    for name, summary in (("old", old), ("large", large)):
        if summary["exponential_midpoint_revision"] != _CHARGE_REVISION:
            raise RuntimeError(f"P20 {name} charge run used an unexpected method revision")
        lifecycle = _mapping(summary, "lifecycle_counts")
        if int(lifecycle["active"]) != particles or int(lifecycle["failed"]) != 0:
            raise RuntimeError(f"P20 {name} charge run did not keep the full cohort active")
    return {
        "workload": {
            "particles": particles,
            "physical_interval_s": end_s,
            "motion": "stationary XY cohort; continuous stationary-Maxwellian charge",
            "integrator": "exponential_midpoint",
            "output": "no trajectory frames or probes",
        },
        "old_admissible_step": old,
        "stable_larger_step": large,
        "comparisons": {
            "macro_step_reduction": (int(old["macro_step_count"]) - int(large["macro_step_count"])),
            "macro_step_reduction_factor": (
                int(old["macro_step_count"]) / int(large["macro_step_count"])
            ),
            "median_simulate_speedup": (
                float(old["median_elapsed_s"]) / float(large["median_elapsed_s"])
            ),
            "both_completed_without_particle_failure": True,
            "payload_digest_compared_for_accuracy": False,
        },
        "accuracy_scope": {
            "claim": (
                "no equal-accuracy claim: the two step sizes intentionally produce different "
                "discrete payloads and only demonstrate operational utility"
            ),
            "separate_tests": [
                (
                    "tests/verification/test_integrators.py::"
                    "test_exponential_midpoint_charge_is_affine_exact_stiff_and_monotone"
                ),
                (
                    "tests/verification/test_integrators.py::"
                    "test_exponential_midpoint_couples_nonlinear_charge_and_motion_at_second_order"
                ),
                (
                    "tests/scenarios/test_force_coupled_run.py::"
                    "test_continuous_charge_stability_gate_is_owned_by_explicit_rk4"
                ),
                (
                    "tests/scenarios/test_force_coupled_run.py::"
                    "test_aggregate_charge_coupled_time_refinement"
                ),
            ],
        },
    }


def _summarize(observations: list[dict[str, object]]) -> dict[str, object]:
    if not observations:
        raise RuntimeError("P20 observation group is empty")
    stable_keys = (
        "particle_count",
        "macro_step_count",
        "segment_count",
        "segment_macro_step_min",
        "segment_macro_step_max",
        "artifact_file_count",
        "checkpoint_count",
        "artifact_bytes",
        "segment_bytes",
        "checkpoint_bytes",
        "planned_memory_bytes",
        "planned_run_peak_bytes",
        "payload_sha256",
        "durable_commit_cadence",
        "latest_checkpoint_work",
        "production_revision_tuple",
        "maximum_dt_charge_lipschitz",
        "integrator",
        "exponential_midpoint_revision",
        "lifecycle_counts",
        "failure_reason_counts",
        "final_charge_number_min",
        "final_charge_number_max",
    )
    first = observations[0]
    for observation in observations[1:]:
        for key in stable_keys:
            if json.dumps(observation[key], sort_keys=True) != json.dumps(
                first[key], sort_keys=True
            ):
                raise RuntimeError(f"P20 repeated observation changed {key}")
    elapsed = [float(item["elapsed_s"]) for item in observations]
    return {
        "timed_repeats": len(observations),
        "elapsed_s": elapsed,
        "median_elapsed_s": statistics.median(elapsed),
        **{key: first[key] for key in stable_keys},
    }


def _segment_macro_steps(paths: list[Path]) -> list[int]:
    counts: list[int] = []
    for path in paths:
        with h5py.File(path, "r") as segment:
            counts.append(int(segment["series"]["time_s"].shape[0]))
    if not counts:
        raise RuntimeError("P20 result has no durable segment")
    return counts


def _latest_checkpoint_work(output: Path) -> dict[str, int]:
    latest = json.loads(output.joinpath("LATEST").read_text(encoding="utf-8"))
    checkpoint_name = latest.get("checkpoint")
    if checkpoint_name not in {"A.h5", "B.h5"}:
        raise RuntimeError("P20 LATEST does not identify a canonical checkpoint")
    counters: dict[str, int] = {}
    with h5py.File(output / "checkpoints" / checkpoint_name, "r") as checkpoint:
        for name in (
            "macro_step_count",
            "accepted_particle_pieces",
            "candidate_queries",
            "refinements",
        ):
            counters[name] = int(checkpoint.attrs[name])
    counters["total"] = sum(counters.values())
    return counters


def _production_revision_tuple(
    manifest: Mapping[str, object],
    memory_plan: Mapping[str, object],
    resolved: Mapping[str, object],
) -> dict[str, object]:
    integrator = resolved["integrator"]
    if integrator == "exponential_midpoint":
        expected_exponential: str | None = _CHARGE_REVISION
    elif integrator == "rk4_fixed":
        expected_exponential = None
    else:
        raise RuntimeError(f"P20 benchmark resolved unexpected integrator {integrator!r}")
    expected: dict[str, object] = {
        **_EXPECTED_PRODUCTION_REVISIONS,
        "exponential_midpoint_revision": expected_exponential,
    }
    actual: dict[str, object] = {
        name: manifest[name]
        for name in (
            "engine_algorithm_revision",
            "compiled_cpu_tile_revision",
            "step_proposal_revision",
            "physics_runtime_revision",
            "result_algorithm_revision",
            "event_algorithm_revision",
            "exponential_midpoint_revision",
        )
    }
    actual["memory_plan_revision"] = memory_plan["revision"]
    actual["runtime_layout_revision"] = memory_plan["runtime_layout_revision"]
    resume_identity = _mapping(manifest, "resume_identity")
    resume_matches = all(
        resume_identity[name] == expected[name]
        for name in (
            "engine_algorithm_revision",
            "compiled_cpu_tile_revision",
            "step_proposal_revision",
            "physics_runtime_revision",
            "result_algorithm_revision",
            "event_algorithm_revision",
            "memory_plan_revision",
            "exponential_midpoint_revision",
        )
    )
    resume_matches &= (
        resume_identity["cpu_runtime_layout_revision"] == expected["runtime_layout_revision"]
    )
    if actual != expected or not resume_matches:
        raise RuntimeError(
            f"P20 production revision tuple changed: expected {expected!r}, got {actual!r}"
        )
    return actual


def _scientific_payload_digest(result: Any) -> str:
    digest = hashlib.sha256()
    for name, reader in (
        ("final", result.read_final),
        ("release", result.read_release_events),
        ("boundary", result.read_boundary_events),
        ("failure", result.read_failure_events),
        ("series", result.read_lifecycle_series),
    ):
        _digest_dataclass(digest, name, reader())
    for index, frame in enumerate(result.iter_frames()):
        _digest_dataclass(digest, f"frame/{index}", frame)
    for index, probe in enumerate(result.iter_probes()):
        _digest_dataclass(digest, f"probe/{index}", probe)
    return f"sha256:{digest.hexdigest()}"


def _digest_dataclass(digest: Any, name: str, value: Any) -> None:
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


def _mapping(value: Mapping[str, object], key: str) -> Mapping[str, object]:
    result = value[key]
    if not isinstance(result, Mapping):
        raise RuntimeError(f"P20 value {key} is not a mapping")
    return result


def _machine_metadata() -> dict[str, object]:
    project = Path(__file__).resolve().parents[2]
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unreported",
        "logical_cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "h5py": h5py.__version__,
        "numba": metadata.version("numba"),
        "chamber_particles": metadata.version("chamber-particles"),
        "uv_lock_sha256": _file_sha256(project / "uv.lock"),
        "driver_sha256": _file_sha256(Path(__file__).resolve()),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1_048_576), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


if __name__ == "__main__":
    main()
