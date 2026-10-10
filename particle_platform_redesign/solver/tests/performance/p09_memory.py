"""Non-gating load, prepare, and end-to-end memory characterization for P09.

The driver materializes each case outside the measured process, then launches a
fresh Python child for every observation.  Absolute timings and RSS values are
machine-local evidence, not pytest or CI thresholds.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import gc
import hashlib
import json
import os
import platform
import statistics
import struct
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, fields, is_dataclass, replace
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

from chamber_particles import load_case, open_result, simulate
from chamber_particles.case_format import RealizedTableSource, write
from tests.verification.microcases import build_microcase

_SCOPES = ("load", "prepare", "run")
_MODES = ("fresh", "warm")
_STABLE_MANIFEST_KEYS = (
    "status",
    "result_schema_version",
    "result_algorithm_revision",
    "case_name",
    "case_file_hash",
    "data_content_hash",
    "case_schema_version",
    "data_coordinate_system",
    "motion_mode",
    "random_draw_kinds",
    "requested",
    "resolved",
    "time",
    "event",
    "event_refinement",
    "boundary_interactions",
    "source_id_to_name",
    "lifecycle_counts",
    "failure_reason_codes",
    "failure_reason_counts",
    "maximum_dt_over_tau",
    "counts",
)


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


def main() -> None:
    """Run either the human-facing driver or one isolated worker."""

    arguments = _arguments()
    if arguments.worker_scope is not None:
        _worker(arguments)
        return
    _driver(arguments)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--particles",
        nargs="+",
        type=_positive_integer,
        default=[10_000],
        help="particle counts; pass 10000 100000 1000000 for the full P09 matrix",
    )
    parser.add_argument("--modes", nargs="+", choices=_MODES, default=list(_MODES))
    parser.add_argument("--scopes", nargs="+", choices=_SCOPES, default=list(_SCOPES))
    parser.add_argument("--repeats", type=_positive_integer, default=1)
    parser.add_argument("--warmup-runs", type=_nonnegative_integer, default=1)
    parser.add_argument("--memory-limit-mb", type=_positive_integer, default=4096)
    parser.add_argument("--json", dest="json_path", type=Path)
    parser.add_argument("--worker-scope", choices=_SCOPES, help=argparse.SUPPRESS)
    parser.add_argument("--worker-case", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-warmups", type=_nonnegative_integer, default=0, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--worker-particle-count",
        type=_positive_integer,
        help=argparse.SUPPRESS,
    )
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
    particle_counts = tuple(dict.fromkeys(arguments.particles))
    modes = tuple(dict.fromkeys(arguments.modes))
    scopes = tuple(dict.fromkeys(arguments.scopes))
    with tempfile.TemporaryDirectory(prefix="chamber-particles-p09-") as temporary:
        root = Path(temporary)
        cases = {
            count: _materialize_case(root / "cases" / str(count), count, arguments.memory_limit_mb)
            for count in particle_counts
        }
        observations: list[dict[str, Any]] = []
        for count in particle_counts:
            for mode in modes:
                warmups = arguments.warmup_runs if mode == "warm" else 0
                for repeat in range(arguments.repeats):
                    observations.extend(
                        _launch_worker(
                            scope=scope,
                            case_path=cases[count],
                            output_root=(
                                root / "results" / str(count) / mode / f"{repeat:03d}" / scope
                            ),
                            particle_count=count,
                            warmups=warmups,
                            mode=mode,
                            repeat=repeat,
                        )
                        for scope in scopes
                    )
        _validate_observations(observations, particle_counts, scopes)
        report = {
            "benchmark": "p09_load_prepare_runtime_memory_v1",
            "captured_at_utc": datetime.now(UTC).isoformat(),
            "non_gating": True,
            "conditions": {
                "particle_counts": list(particle_counts),
                "modes": list(modes),
                "scopes": list(scopes),
                "repeats": arguments.repeats,
                "warmup_runs_for_warm_mode": arguments.warmup_runs,
                "memory_limit_mb": arguments.memory_limit_mb,
                "case_materialization_in_measured_scope": False,
                "fresh_process_note": (
                    "fresh means a new Python process; filesystem caches are not flushed"
                ),
            },
            "machine": _machine_metadata(),
            "observations": observations,
            "summaries": _summaries(observations),
            "identity": _identity_summary(observations),
            "interpretation": {
                "rss": (
                    "OS process high-water RSS includes Python, native libraries, and HDF5; "
                    "it is distinct from the solver-owned array memory plan"
                ),
                "prepare_scope": (
                    "prepare is an internal characterization of the production engine, not a "
                    "public API or compatibility contract"
                ),
                "timing": "absolute seconds are descriptive and never a CI threshold",
            },
        }
    encoded = json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    if arguments.json_path is not None:
        arguments.json_path.parent.mkdir(parents=True, exist_ok=True)
        arguments.json_path.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


def _materialize_case(directory: Path, particle_count: int, memory_limit_mb: int) -> Path:
    directory.mkdir(parents=True)
    definition = build_microcase("C01")
    source = definition.data.sources[0]
    expanded = _repeat_source(source, particle_count)
    data = replace(definition.data, sources=(expanded,))
    data_path = directory / "case.h5"
    info = write(data_path, data)
    specification = copy.deepcopy(definition.spec)
    specification["case"] = {
        "name": f"P09_ballistic_{particle_count}",
        "data_path": data_path.name,
        "expected_content_hash": info.content_hash,
    }
    specification["time"] = {"start_s": 0.0, "end_s": 0.2, "dt_s": 0.2}
    specification["resources"]["memory_limit_mb"] = memory_limit_mb
    specification["output"]["trajectories"] = None
    case_path = directory / "case.yaml"
    case_path.write_text(yaml.safe_dump(specification, sort_keys=False), encoding="utf-8")
    return case_path


def _repeat_source(source: RealizedTableSource, count: int) -> RealizedTableSource:
    def scalar(values: np.ndarray) -> np.ndarray:
        return np.full(count, values[0], dtype=values.dtype)

    return RealizedTableSource(
        name=source.name,
        particle_id=np.arange(1, count + 1, dtype="<i8"),
        release_time_s=scalar(source.release_time_s),
        position_m=np.repeat(source.position_m[:1], count, axis=0),
        velocity_m_s=np.repeat(source.velocity_m_s[:1], count, axis=0),
        charge_number=scalar(source.charge_number),
        mass_kg=scalar(source.mass_kg),
        drag_diameter_m=scalar(source.drag_diameter_m),
        contact_radius_m=scalar(source.contact_radius_m),
        electrostatic_radius_m=scalar(source.electrostatic_radius_m),
        displaced_volume_m3=scalar(source.displaced_volume_m3),
        model_weight=scalar(source.model_weight),
        material_id=scalar(source.material_id),
    )


def _launch_worker(
    *,
    scope: str,
    case_path: Path,
    output_root: Path,
    particle_count: int,
    warmups: int,
    mode: str,
    repeat: int,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "tests.performance.p09_memory",
        "--worker-scope",
        scope,
        "--worker-case",
        str(case_path),
        "--worker-output",
        str(output_root),
        "--worker-warmups",
        str(warmups),
        "--worker-particle-count",
        str(particle_count),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"P09 {scope} worker failed with exit {completed.returncode}: {completed.stderr.strip()}"
        )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"P09 {scope} worker did not return JSON") from error
    if not isinstance(result, dict):
        raise RuntimeError(f"P09 {scope} worker returned a non-object")
    result["mode"] = mode
    result["repeat"] = repeat
    return result


def _worker(arguments: argparse.Namespace) -> None:
    if (
        arguments.worker_case is None
        or arguments.worker_output is None
        or arguments.worker_particle_count is None
    ):
        raise SystemExit("worker scope requires case, output, and particle count")
    for index in range(arguments.worker_warmups):
        _execute_scope(
            arguments.worker_scope,
            arguments.worker_case,
            arguments.worker_output / f"warmup-{index:03d}",
            arguments.worker_particle_count,
            include_digest=False,
        )
        gc.collect()
    observation = _execute_scope(
        arguments.worker_scope,
        arguments.worker_case,
        arguments.worker_output / "measured",
        arguments.worker_particle_count,
        include_digest=True,
    )
    print(json.dumps(observation, allow_nan=False, sort_keys=True))


def _execute_scope(
    scope: str,
    case_path: Path,
    output: Path,
    expected_particle_count: int,
    *,
    include_digest: bool,
) -> dict[str, Any]:
    peak_before, rss_before, rss_source = _process_memory_bytes()
    load_started = time.perf_counter()
    case = load_case(case_path)
    load_elapsed = time.perf_counter() - load_started
    identity = {
        "case_file_hash": case.case_file_hash,
        "data_content_hash": case.content_hash,
    }
    timings: dict[str, float] = {"load_case_s": load_elapsed}
    memory_plan: object | None = None
    algorithms: dict[str, object] = {}
    artifact_bytes = 0
    semantic_digest: str | None = None
    particle_count = expected_particle_count

    if scope == "prepare":
        from chamber_particles.engine import _prepare

        prepare_started = time.perf_counter()
        prepared = _prepare(case)
        timings["prepare_s"] = time.perf_counter() - prepare_started
        particle_count = prepared.schedule.particle_count
        memory_plan = _prepared_memory_plan(prepared)
    elif scope == "run":
        simulate_started = time.perf_counter()
        summary = simulate(case, output)
        timings["simulate_s"] = time.perf_counter() - simulate_started
        open_started = time.perf_counter()
        result = open_result(output)
        timings["open_result_s"] = time.perf_counter() - open_started
        particle_count = summary.particle_count
        artifact_bytes = _directory_bytes(output)
        algorithms = {
            key: value for key, value in result.manifest.items() if key.endswith("_revision")
        }
        memory_plan = _manifest_memory_plan(result.manifest)
        peak_after, rss_after, _ = _process_memory_bytes()
        if include_digest:
            semantic_digest = _semantic_result_digest(result)
        observation = _observation(
            scope,
            expected_particle_count,
            particle_count,
            case_path,
            identity,
            timings,
            peak_before,
            peak_after,
            rss_before,
            rss_after,
            rss_source,
            memory_plan,
            algorithms,
            artifact_bytes,
            semantic_digest,
        )
        return observation
    elif scope != "load":
        raise RuntimeError(f"unknown P09 worker scope: {scope}")

    peak_after, rss_after, _ = _process_memory_bytes()
    return _observation(
        scope,
        expected_particle_count,
        particle_count,
        case_path,
        identity,
        timings,
        peak_before,
        peak_after,
        rss_before,
        rss_after,
        rss_source,
        memory_plan,
        algorithms,
        artifact_bytes,
        semantic_digest,
    )


def _observation(
    scope: str,
    expected_particle_count: int,
    particle_count: int,
    case_path: Path,
    identity: dict[str, str],
    timings: dict[str, float],
    peak_before: int,
    peak_after: int,
    rss_before: int | None,
    rss_after: int | None,
    rss_source: str,
    memory_plan: object | None,
    algorithms: dict[str, object],
    artifact_bytes: int,
    semantic_digest: str | None,
) -> dict[str, Any]:
    if particle_count != expected_particle_count:
        raise RuntimeError(
            f"{scope}: expected {expected_particle_count} particles, got {particle_count}"
        )
    return {
        "scope": scope,
        "particle_count": particle_count,
        "identity": identity,
        "timing_s": timings,
        "memory": {
            "measurement_source": rss_source,
            "peak_rss_before_measured_scope_bytes": peak_before,
            "peak_rss_after_measured_scope_bytes": peak_after,
            "additional_process_high_water_bytes": max(0, peak_after - peak_before),
            "rss_before_measured_scope_bytes": rss_before,
            "rss_after_measured_scope_bytes": rss_after,
            "solver_memory_plan": memory_plan,
        },
        "case_artifact_bytes": _directory_bytes(case_path.parent),
        "result_artifact_bytes": artifact_bytes,
        "result_artifact_bytes_per_particle": artifact_bytes / particle_count,
        "algorithms": algorithms,
        "semantic_result_sha256": semantic_digest,
    }


def _prepared_memory_plan(prepared: Any) -> object:
    return _validated_memory_plan(_json_value(prepared.memory_plan.as_manifest()))


def _manifest_memory_plan(manifest: Mapping[str, object]) -> object:
    try:
        plan = manifest["memory_plan"]
    except KeyError as error:
        raise RuntimeError("completed P09 result is missing memory_plan") from error
    return _validated_memory_plan(_json_value(plan))


def _validated_memory_plan(value: object) -> object:
    if not isinstance(value, Mapping):
        raise RuntimeError("P09 memory plan must be a mapping")
    required = {
        "revision",
        "semantics",
        "limit_bytes",
        "planned_bytes",
        "slab_particles",
        "scratch_bytes_per_particle",
        "event_work_bytes_per_particle",
        "release_work_bytes_per_particle",
        "event_candidate_capacity",
        "event_staging_capacity",
        "event_staging_bytes_per_row",
        "event_staging_fixed_bytes",
        "failure_staging_bytes_per_particle",
        "phase_peaks",
        "components",
    }
    missing = required - set(value)
    if missing:
        raise RuntimeError(f"P09 memory plan is missing keys: {sorted(missing)}")
    return value


def _json_value(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return repr(value)


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
    peak_bytes = maximum if sys.platform == "darwin" else maximum * 1024
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
    succeeded = get_process_memory_info(
        get_current_process(),
        ctypes.byref(counters),
        counters.cb,
    )
    if not succeeded:
        raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
    return counters


def _semantic_result_digest(result: Any) -> str:
    manifest = result.manifest
    stable_manifest = {key: manifest[key] for key in _STABLE_MANIFEST_KEYS if key in manifest}
    stable_manifest.update(
        {key: value for key, value in manifest.items() if key.endswith("_revision")}
    )
    digest = hashlib.sha256()
    _digest_json(digest, "manifest", stable_manifest)
    readers = (
        ("final", result.read_final),
        ("release_events", result.read_release_events),
        ("boundary_events", result.read_boundary_events),
        ("failure_events", result.read_failure_events),
        ("lifecycle_series", result.read_lifecycle_series),
    )
    for name, reader in readers:
        value = reader()
        _digest_dataclass(digest, name, value)
        del value
    for index, frame in enumerate(result.iter_frames()):
        _digest_dataclass(digest, f"frame/{index}", frame)
    for index, probe in enumerate(result.iter_probes()):
        _digest_dataclass(digest, f"probe/{index}", probe)
    return f"sha256:{digest.hexdigest()}"


def _digest_dataclass(digest: Any, name: str, value: object) -> None:
    _digest_bytes(digest, name.encode("utf-8"))
    for field in fields(value):
        field_value = getattr(value, field.name)
        label = f"{name}.{field.name}"
        if isinstance(field_value, np.ndarray):
            _digest_array(digest, label, field_value)
        else:
            _digest_json(digest, label, field_value)


def _digest_array(digest: Any, name: str, array: np.ndarray) -> None:
    _digest_json(
        digest,
        f"{name}.metadata",
        {"dtype": array.dtype.str, "shape": list(array.shape)},
    )
    if not array.size:
        _digest_bytes(digest, b"")
        return
    if array.dtype.hasobject or array.dtype.kind in "US":
        _digest_json(digest, name, array.tolist())
        return
    contiguous = array if array.flags.c_contiguous else np.ascontiguousarray(array)
    _digest_bytes(digest, memoryview(contiguous).cast("B"))


def _digest_json(digest: Any, name: str, value: object) -> None:
    _digest_bytes(digest, name.encode("utf-8"))
    encoded = json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    _digest_bytes(digest, encoded)


def _digest_bytes(digest: Any, value: bytes | memoryview) -> None:
    digest.update(struct.pack("<Q", len(value)))
    digest.update(value)


def _validate_observations(
    observations: Sequence[Mapping[str, object]],
    particle_counts: tuple[int, ...],
    scopes: tuple[str, ...],
) -> None:
    for count in particle_counts:
        selected = [item for item in observations if item["particle_count"] == count]
        identities = {
            json.dumps(item["identity"], sort_keys=True, separators=(",", ":")) for item in selected
        }
        if len(identities) != 1:
            raise RuntimeError(f"P09 case identity changed across scopes for {count} particles")
        if "run" in scopes:
            digests = {
                item["semantic_result_sha256"] for item in selected if item["scope"] == "run"
            }
            if None in digests or len(digests) != 1:
                raise RuntimeError(
                    f"P09 semantic result identity changed across runs for {count} particles"
                )


def _summaries(observations: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    keys = sorted(
        {
            (int(item["particle_count"]), str(item["mode"]), str(item["scope"]))
            for item in observations
        }
    )
    summaries: list[dict[str, object]] = []
    for particle_count, mode, scope in keys:
        selected = [
            item
            for item in observations
            if item["particle_count"] == particle_count
            and item["mode"] == mode
            and item["scope"] == scope
        ]
        timing_names = sorted(
            {name for item in selected for name in _mapping_value(item, "timing_s")}
        )
        memory = [_mapping_value(item, "memory") for item in selected]
        summaries.append(
            {
                "particle_count": particle_count,
                "mode": mode,
                "scope": scope,
                "observations": len(selected),
                "median_timing_s": {
                    name: statistics.median(
                        float(_mapping_value(item, "timing_s")[name]) for item in selected
                    )
                    for name in timing_names
                },
                "maximum_peak_rss_bytes": max(
                    int(item["peak_rss_after_measured_scope_bytes"]) for item in memory
                ),
                "result_artifact_bytes_per_particle": statistics.median(
                    float(item["result_artifact_bytes_per_particle"]) for item in selected
                ),
            }
        )
    return summaries


def _identity_summary(observations: Sequence[Mapping[str, object]]) -> dict[str, object]:
    run_observations = [item for item in observations if item["scope"] == "run"]
    by_count: dict[str, object] = {}
    for count in sorted({int(item["particle_count"]) for item in run_observations}):
        selected = [item for item in run_observations if item["particle_count"] == count]
        by_count[str(count)] = {
            "semantic_result_sha256": selected[0]["semantic_result_sha256"],
            "observations_compared": len(selected),
            "exact_match": True,
        }
    return {
        "definition": (
            "stable manifest semantics plus final/events/series/frames/probes; timing and RSS excluded"
        ),
        "by_particle_count": by_count,
    }


def _mapping_value(value: Mapping[str, object], key: str) -> Mapping[str, object]:
    result = value[key]
    if not isinstance(result, Mapping):
        raise RuntimeError(f"observation {key} is not a mapping")
    return result


def _machine_metadata() -> dict[str, object]:
    lock_path = Path(__file__).resolve().parents[2] / "uv.lock"
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unreported",
        "logical_cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "h5py": h5py.__version__,
        "chamber_particles": metadata.version("chamber-particles"),
        "uv_lock_sha256": _file_sha256(lock_path),
        "peak_rss_source": _process_memory_bytes()[2],
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


if __name__ == "__main__":
    main()
