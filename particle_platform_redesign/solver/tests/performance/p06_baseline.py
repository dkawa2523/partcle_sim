"""Reproducible, non-gating end-to-end baseline for the P06 reference engine.

Run this module directly through uv; it is intentionally not a pytest test.  The
baseline exercises only the three public APIs and reports machine-local numbers
as JSON.  Absolute values are not acceptance thresholds.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import platform
import statistics
import sys
import tempfile
import time
import tracemalloc
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from chamber_particles import load_case, open_result, simulate
from chamber_particles.case_format import BoundaryData, DataBundle, RealizedTableSource, write
from tests.verification.microcases import MicrocaseDefinition, build_microcase


@dataclass(frozen=True, slots=True)
class _Scenario:
    name: str
    case_path: Path
    particle_count: int
    expected_frames: int
    expected_boundary_events: int
    expected_path_kind: str


@dataclass(frozen=True, slots=True)
class _Observation:
    elapsed_s: float
    macro_step_count: int
    planned_memory_bytes: int
    artifact_bytes: int
    boundary_events: int
    recorded_event_facet_ids: int
    engine_algorithm_revision: str
    step_proposal_revision: str
    rk4_enclosure_revision: str
    event_algorithm_revision: str
    path_kind: str
    accepted_particle_pieces: int | None
    candidate_queries: int | None
    refinements: int | None
    maximum_refinement_depth: int | None


@dataclass(frozen=True, slots=True)
class _TraceMemoryObservation:
    peak_bytes: int
    current_bytes_after_simulate: int


def main() -> None:
    """Run the requested warm measurements and print one JSON document."""

    arguments = _arguments()
    with tempfile.TemporaryDirectory(prefix="chamber-particles-p06-") as temporary:
        root = Path(temporary)
        scenarios = _build_scenarios(root / "cases", arguments.particles)
        observations = {
            scenario.name: _measure(
                scenario,
                root / "results" / scenario.name,
                arguments.warmups,
                arguments.repeats,
            )
            for scenario in scenarios
        }
        traced_memory = (
            {
                scenario.name: _measure_traced_memory(
                    scenario,
                    root / "trace-results" / scenario.name,
                )
                for scenario in scenarios
                if scenario.name in {"rk4_material_hits", "rk4_material_hits_interior_frame"}
            }
            if arguments.trace_memory
            else {}
        )
        numpy_calibration_bytes = _numpy_trace_calibration() if arguments.trace_memory else None
    report = _report(arguments, observations, traced_memory, numpy_calibration_bytes)
    encoded = json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    if arguments.json_path is not None:
        arguments.json_path.parent.mkdir(parents=True, exist_ok=True)
        arguments.json_path.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=_positive_integer, default=256)
    parser.add_argument("--repeats", type=_positive_integer, default=3)
    parser.add_argument("--warmups", type=_nonnegative_integer, default=1)
    parser.add_argument(
        "--trace-memory",
        action="store_true",
        help="measure simulate-time Python/NumPy allocations for the RK4 material pair",
    )
    parser.add_argument("--json", dest="json_path", type=Path)
    return parser.parse_args()


def _positive_integer(value: str) -> int:
    integer = int(value)
    if integer <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return integer


def _nonnegative_integer(value: str) -> int:
    integer = int(value)
    if integer < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return integer


def _build_scenarios(root: Path, particle_count: int) -> tuple[_Scenario, ...]:
    root.mkdir(parents=True)
    force = build_microcase("C04")
    exact = _expanded_definition(force, particle_count)
    enclosed = replace(exact, data=_nonuniform_electric_data(exact.data))
    boundary = _expanded_boundary_definition(build_microcase("C07"), particle_count)
    rk4_material = _rk4_material_definition(force, boundary, particle_count)
    boundaryless = replace(
        boundary,
        data=_without_material_boundary(boundary.data),
        spec=_without_boundary_laws(boundary.spec),
    )
    definitions = (
        ("quadratic_no_frames", exact, None, 0, 0, "quadratic_exact"),
        ("rk4_enclosed_no_frames", enclosed, None, 0, 0, "rk4_dense"),
        (
            "rk4_enclosed_macro_end_frames",
            enclosed,
            [0.125, 0.25, 0.375, 0.5],
            4,
            0,
            "rk4_dense",
        ),
        (
            "rk4_enclosed_interior_frames",
            enclosed,
            [0.0625, 0.1875, 0.3125, 0.4375],
            4,
            0,
            "rk4_dense",
        ),
        ("ballistic_boundaryless", boundaryless, None, 0, 0, "linear_exact"),
        (
            "ballistic_material_hits",
            boundary,
            None,
            0,
            particle_count,
            "linear_exact",
        ),
        (
            "rk4_material_hits",
            rk4_material,
            None,
            0,
            particle_count,
            "rk4_dense",
        ),
        (
            "rk4_material_hits_interior_frame",
            rk4_material,
            [0.28],
            1,
            particle_count,
            "rk4_dense",
        ),
    )
    scenarios = []
    for name, definition, frame_times, frame_count, boundary_events, path_kind in definitions:
        case_path = _materialize(
            root / name,
            definition,
            name=name,
            frame_times_s=frame_times,
        )
        scenarios.append(
            _Scenario(
                name,
                case_path,
                particle_count,
                frame_count,
                boundary_events,
                path_kind,
            )
        )
    return tuple(scenarios)


def _expanded_definition(
    definition: MicrocaseDefinition, particle_count: int
) -> MicrocaseDefinition:
    source = definition.data.sources[0]
    expanded = _repeat_source(source, particle_count)
    return replace(definition, data=replace(definition.data, sources=(expanded,)))


def _expanded_boundary_definition(
    definition: MicrocaseDefinition, particle_count: int
) -> MicrocaseDefinition:
    source = _repeat_source(definition.data.sources[0], particle_count)
    source = replace(
        source,
        position_m=np.column_stack(
            (
                np.full(particle_count, 0.25, dtype="<f8"),
                np.linspace(0.1, 0.9, particle_count, dtype="<f8"),
            )
        ),
        velocity_m_s=np.column_stack(
            (
                np.full(particle_count, 0.5, dtype="<f8"),
                np.zeros(particle_count, dtype="<f8"),
            )
        ),
    )
    return replace(definition, data=replace(definition.data, sources=(source,)))


def _repeat_source(source: RealizedTableSource, count: int) -> RealizedTableSource:
    def repeat(name: str) -> np.ndarray:
        values = getattr(source, name)
        return np.repeat(values[:1], count, axis=0).astype(values.dtype, copy=False)

    return replace(
        source,
        particle_id=np.arange(1, count + 1, dtype="<i8"),
        release_time_s=repeat("release_time_s"),
        position_m=repeat("position_m"),
        velocity_m_s=repeat("velocity_m_s"),
        charge_number=repeat("charge_number"),
        mass_kg=repeat("mass_kg"),
        drag_diameter_m=repeat("drag_diameter_m"),
        electrostatic_radius_m=repeat("electrostatic_radius_m"),
        contact_radius_m=repeat("contact_radius_m"),
        displaced_volume_m3=repeat("displaced_volume_m3"),
        model_weight=repeat("model_weight"),
        material_id=repeat("material_id"),
    )


def _nonuniform_electric_data(data: DataBundle) -> DataBundle:
    values = np.asarray([[1.0, 0.0], [1.0, 0.0], [3.0, 0.0], [3.0, 0.0]], dtype="<f8")
    fields = tuple(
        replace(field, values=values) if field.name == "electric_field" else field
        for field in data.fields
    )
    return replace(data, fields=fields)


def _rk4_material_definition(
    force: MicrocaseDefinition,
    boundary: MicrocaseDefinition,
    particle_count: int,
) -> MicrocaseDefinition:
    expanded = _expanded_definition(force, particle_count)
    source = expanded.data.sources[0]
    source = replace(
        source,
        position_m=np.column_stack(
            (
                np.full(particle_count, 0.5, dtype="<f8"),
                np.linspace(0.1, 0.9, particle_count, dtype="<f8"),
            )
        ),
        velocity_m_s=np.column_stack(
            (
                np.full(particle_count, 2.0, dtype="<f8"),
                np.zeros(particle_count, dtype="<f8"),
            )
        ),
    )
    charge_to_acceleration = (
        float(source.charge_number[0]) * 1.602176634e-19 / float(source.mass_kg[0])
    )
    node_x_m = np.asarray([-1.0, -1.0, 1.0, 1.0], dtype="<f8")
    electric_x_v_m = -4.0 * node_x_m / charge_to_acceleration
    electric_values = np.column_stack((electric_x_v_m, np.zeros(electric_x_v_m.size, dtype="<f8")))
    fields = tuple(
        replace(field, values=electric_values) if field.name == "electric_field" else field
        for field in expanded.data.fields
    )
    spec = copy.deepcopy(expanded.spec)
    spec["boundaries"] = copy.deepcopy(boundary.spec["boundaries"])
    data = replace(
        expanded.data,
        geometry=boundary.data.geometry,
        fields=fields,
        sources=(source,),
    )
    return replace(expanded, spec=spec, data=data)


def _without_material_boundary(data: DataBundle) -> DataBundle:
    boundary = BoundaryData(
        line2=np.empty((0, 2), dtype="<i8"),
        boundary_id=np.empty(0, dtype="<i4"),
        group_id=np.empty(0, dtype="<i4"),
        material_id=np.empty(0, dtype="<i4"),
        owner_cell_type=np.empty(0, dtype="<u1"),
        owner_cell_local_index=np.empty(0, dtype="<i8"),
        orientation=np.empty(0, dtype="<i1"),
    )
    geometry = replace(data.geometry, boundary=boundary, group_names=())
    return replace(data, geometry=geometry)


def _without_boundary_laws(spec: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(spec)
    result["boundaries"] = []
    return result


def _materialize(
    directory: Path,
    definition: MicrocaseDefinition,
    *,
    name: str,
    frame_times_s: list[float] | None,
) -> Path:
    directory.mkdir(parents=True)
    data_path = directory / "case.h5"
    info = write(data_path, definition.data)
    spec = copy.deepcopy(definition.spec)
    spec["case"] = {
        "name": name,
        "data_path": data_path.name,
        "expected_content_hash": info.content_hash,
    }
    spec["resources"]["memory_limit_mb"] = 1024
    if frame_times_s is None:
        spec["output"]["trajectories"] = None
    else:
        spec["output"]["trajectories"] = {
            "selection": "all",
            "schedule": {"explicit_times_s": frame_times_s},
        }
    case_path = directory / "case.yaml"
    case_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    return case_path


def _measure(
    scenario: _Scenario, result_root: Path, warmups: int, repeats: int
) -> tuple[_Observation, ...]:
    observations = []
    for index in range(warmups + repeats):
        output = result_root / f"run-{index:03d}"
        started = time.perf_counter()
        case = load_case(scenario.case_path)
        summary = simulate(case, output)
        result = open_result(output)
        elapsed_s = time.perf_counter() - started
        if summary.particle_count != scenario.particle_count:
            raise RuntimeError(f"{scenario.name}: unexpected particle count")
        if summary.frame_count != scenario.expected_frames:
            raise RuntimeError(f"{scenario.name}: unexpected frame count")
        if summary.boundary_event_count != scenario.expected_boundary_events:
            raise RuntimeError(f"{scenario.name}: unexpected boundary-event count")
        path_kind = str(result.manifest["resolved"]["path_kind"])
        if path_kind != scenario.expected_path_kind:
            raise RuntimeError(f"{scenario.name}: unexpected path kind {path_kind!r}")
        boundary_events = result.read_boundary_events()
        event_refinement = result.manifest["event_refinement"]
        if event_refinement is not None and not isinstance(event_refinement, dict):
            raise RuntimeError(f"{scenario.name}: invalid event_refinement manifest value")
        observation = _Observation(
            elapsed_s=elapsed_s,
            macro_step_count=summary.macro_step_count,
            planned_memory_bytes=int(result.manifest["memory_plan"]["planned_bytes"]),
            artifact_bytes=_directory_bytes(output),
            boundary_events=summary.boundary_event_count,
            recorded_event_facet_ids=int(boundary_events.candidate_facet_id.size),
            engine_algorithm_revision=str(result.manifest["engine_algorithm_revision"]),
            step_proposal_revision=str(result.manifest["step_proposal_revision"]),
            rk4_enclosure_revision=str(result.manifest["rk4_enclosure_revision"]),
            event_algorithm_revision=str(result.manifest["event_algorithm_revision"]),
            path_kind=path_kind,
            accepted_particle_pieces=_optional_metric(
                event_refinement,
                "accepted_particle_pieces",
            ),
            candidate_queries=_optional_metric(event_refinement, "candidate_queries"),
            refinements=_optional_metric(event_refinement, "refinements"),
            maximum_refinement_depth=_optional_metric(
                event_refinement,
                "maximum_refinement_depth",
            ),
        )
        if index >= warmups:
            observations.append(observation)
    return tuple(observations)


def _optional_metric(value: object, name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, dict) or not isinstance(value.get(name), int):
        raise RuntimeError(f"event_refinement.{name} is not an integer")
    return int(value[name])


def _measure_traced_memory(
    scenario: _Scenario,
    output: Path,
) -> _TraceMemoryObservation:
    """Measure temporary Python and NumPy allocations without distorting timings."""

    case = load_case(scenario.case_path)
    gc.collect()
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        summary = simulate(case, output)
        current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    if summary.particle_count != scenario.particle_count:
        raise RuntimeError(f"{scenario.name}: unexpected particle count in memory pass")
    if summary.frame_count != scenario.expected_frames:
        raise RuntimeError(f"{scenario.name}: unexpected frame count in memory pass")
    if summary.boundary_event_count != scenario.expected_boundary_events:
        raise RuntimeError(f"{scenario.name}: unexpected boundary events in memory pass")
    return _TraceMemoryObservation(peak_bytes, current_bytes)


def _numpy_trace_calibration() -> int:
    """Confirm that this NumPy runtime reports data-buffer allocations to tracemalloc."""

    calibration_size = 1_000_000
    gc.collect()
    tracemalloc.start()
    try:
        allocation = np.empty(calibration_size, dtype=np.uint8)
        current_bytes, _ = tracemalloc.get_traced_memory()
        if allocation.nbytes != calibration_size:
            raise RuntimeError("unexpected NumPy trace calibration allocation size")
        if current_bytes < calibration_size:
            raise RuntimeError("NumPy data-buffer allocations are not visible to tracemalloc")
    finally:
        tracemalloc.stop()
    return current_bytes


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _report(
    arguments: argparse.Namespace,
    observations: dict[str, tuple[_Observation, ...]],
    traced_memory: dict[str, _TraceMemoryObservation],
    numpy_calibration_bytes: int | None,
) -> dict[str, Any]:
    measurements = {
        name: _scenario_report(values, arguments.particles) for name, values in observations.items()
    }
    exact_elapsed = measurements["quadratic_no_frames"]["elapsed_s"]["median"]
    enclosed_elapsed = measurements["rk4_enclosed_no_frames"]["elapsed_s"]["median"]
    no_frame_elapsed = enclosed_elapsed
    macro_frame_elapsed = measurements["rk4_enclosed_macro_end_frames"]["elapsed_s"]["median"]
    interior_frame_elapsed = measurements["rk4_enclosed_interior_frames"]["elapsed_s"]["median"]
    boundaryless_elapsed = measurements["ballistic_boundaryless"]["elapsed_s"]["median"]
    material_elapsed = measurements["ballistic_material_hits"]["elapsed_s"]["median"]
    rk4_material_elapsed = measurements["rk4_material_hits"]["elapsed_s"]["median"]
    rk4_material_frame_elapsed = measurements["rk4_material_hits_interior_frame"]["elapsed_s"][
        "median"
    ]
    return {
        "baseline": "p06_reference_end_to_end_v4",
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "non_gating": True,
        "conditions": {
            "particles": arguments.particles,
            "repeats": arguments.repeats,
            "warmups": arguments.warmups,
            "timed_scope": "load_case + simulate + open_result",
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "processor": platform.processor() or "unreported",
        },
        "measurements": measurements,
        "comparisons": {
            "rk4_enclosure_stack_elapsed_ratio_vs_quadratic": (enclosed_elapsed / exact_elapsed),
            "macro_end_frame_elapsed_ratio_vs_no_frames": (macro_frame_elapsed / no_frame_elapsed),
            "interior_frame_elapsed_ratio_vs_no_frames": (
                interior_frame_elapsed / no_frame_elapsed
            ),
            "material_hit_stack_elapsed_ratio_vs_boundaryless": (
                material_elapsed / boundaryless_elapsed
            ),
            "rk4_material_hit_elapsed_ratio_vs_rk4_boundaryless": (
                rk4_material_elapsed / enclosed_elapsed
            ),
            "rk4_material_interior_frame_elapsed_ratio_vs_no_frames": (
                rk4_material_frame_elapsed / rk4_material_elapsed
            ),
        },
        "revision_3b_metrics": {
            "available": True,
            **measurements["rk4_material_hits"]["event_refinement"],
        },
        "trace_memory": _trace_memory_report(
            arguments.particles,
            traced_memory,
            numpy_calibration_bytes,
        ),
        "interpretation": {
            "memory": (
                "planned_memory_bytes_per_particle is a solver-owned estimate, not peak RSS"
            ),
            "enclosure_ratio": (
                "includes RK4 stage sampling, interpolation, and enclosure; it is not an isolated "
                "certificate microbenchmark"
            ),
            "material_hit_ratio": (
                "includes BVH/event localization, stick response, and boundary-event output; "
                "the RK4 material scenario also reports accepted pieces and refinement work"
            ),
        },
    }


def _trace_memory_report(
    particle_count: int,
    observations: dict[str, _TraceMemoryObservation],
    numpy_calibration_bytes: int | None,
) -> dict[str, Any]:
    if not observations:
        return {"measured": False}
    return {
        "measured": True,
        "allocator": "tracemalloc with NumPy data-buffer tracking",
        "scope": "simulate only; load_case, process RSS, HDF5/native memory excluded",
        "numpy_calibration_requested_bytes": 1_000_000,
        "numpy_calibration_traced_bytes": numpy_calibration_bytes,
        "measurements": {
            name: {
                "peak_bytes": value.peak_bytes,
                "peak_bytes_per_particle": value.peak_bytes / particle_count,
                "current_bytes_after_simulate": value.current_bytes_after_simulate,
            }
            for name, value in observations.items()
        },
    }


def _scenario_report(values: tuple[_Observation, ...], particle_count: int) -> dict[str, Any]:
    elapsed = [value.elapsed_s for value in values]
    median_elapsed = statistics.median(elapsed)
    representative = values[0]
    recorded_facet_count = statistics.median(value.recorded_event_facet_ids for value in values)
    event_count = statistics.median(value.boundary_events for value in values)
    return {
        "algorithm": {
            "engine_algorithm_revision": representative.engine_algorithm_revision,
            "step_proposal_revision": representative.step_proposal_revision,
            "rk4_enclosure_revision": representative.rk4_enclosure_revision,
            "event_algorithm_revision": representative.event_algorithm_revision,
            "path_kind": representative.path_kind,
        },
        "elapsed_s": {
            "minimum": min(elapsed),
            "median": median_elapsed,
            "maximum": max(elapsed),
        },
        "particles_per_s": particle_count / median_elapsed,
        "nominal_particle_macro_steps_per_s": (
            particle_count * representative.macro_step_count / median_elapsed
        ),
        "macro_step_count": representative.macro_step_count,
        "planned_memory_bytes_per_particle": (representative.planned_memory_bytes / particle_count),
        "result_artifact_bytes_per_particle": representative.artifact_bytes / particle_count,
        "boundary_events_per_particle": event_count / particle_count,
        "recorded_event_facet_ids_per_particle": recorded_facet_count / particle_count,
        "recorded_event_facet_ids_per_boundary_event": (
            recorded_facet_count / event_count if event_count else 0.0
        ),
        "event_refinement": {
            "accepted_particle_pieces": representative.accepted_particle_pieces,
            "candidate_queries": representative.candidate_queries,
            "refinements": representative.refinements,
            "maximum_refinement_depth": representative.maximum_refinement_depth,
        },
    }


if __name__ == "__main__":
    main()
