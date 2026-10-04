"""P14 v0.1 end-to-end performance matrix.

The harness deliberately uses an orthogonal set of rows rather than the full
Cartesian product of every size, field layout, event count, output volume, and
JIT state.  Every measured row runs the three public APIs in an
isolated child process.  Absolute timings are descriptive; scientific identity
and result-shape checks are mandatory.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

from chamber_particles import load_case, open_result, simulate
from chamber_particles.case_format import (
    BoundaryData,
    DataBundle,
    GeometryData,
    P1TriLayout,
    Q1QuadLayout,
    write,
)
from chamber_particles.physics.compiled import evaluate_physics_tile_into
from tests.performance.p06_baseline import (
    _expanded_definition,
    _nonuniform_electric_data,
    _repeat_source,
)
from tests.performance.p09_memory import (
    _digest_dataclass,
    _directory_bytes,
    _machine_metadata,
    _process_memory_bytes,
)
from tests.verification.microcases import build_microcase

type _Suite = Literal["smoke", "release"]
type _Mode = Literal["cold", "warm"]
type _Output = Literal["none", "sample", "all"]
type _Layout = Literal["regular", "p1", "q1"]
type _Motion = Literal["initial", "cross"]

_PROBE_PARTICLE_IDS = tuple(range(1, 33))
_EXPECTED_REVISIONS = {
    "engine_algorithm_revision": "particle_engine_v36",
    "compiled_cpu_tile_revision": "compiled_cpu_tile_v18",
    "physics_runtime_revision": "signed_ion_compiled_physics_runtime_v19",
    "step_proposal_revision": "coupled_fixed_step_proposal_v10",
    "field_location_revision": "field_location_v4",
    "event_algorithm_revision": "line_quadratic_rk4_axis_first_hit_v16",
    "result_algorithm_revision": "durable_segmented_result_v5",
    "geometry_algorithm_revision": "line_boundary_stackless_volume_cell_bvh_v5",
}
_EXPECTED_MEMORY_PLAN_REVISION = "solver_owned_memory_plan_v13"
_EXPECTED_RUNTIME_LAYOUT_REVISION = "resident_soa_serial_slab_v6"


@dataclass(frozen=True, slots=True)
class _MatrixRow:
    row_id: str
    family: Literal["field", "event"]
    particle_count: int
    layout: _Layout
    motion: _Motion
    hit_count: int
    output: _Output
    mode: _Mode
    cell_count: int

    @property
    def data_key(self) -> str:
        if self.family == "event":
            return f"{self.family}-n{self.particle_count}-h{self.hit_count}"
        return f"field-{self.layout}-{self.motion}-n{self.particle_count}-c{self.cell_count}"

    @property
    def science_key(self) -> str:
        return (
            f"{self.family}/{self.layout}/{self.motion}/n={self.particle_count}/"
            f"cells={self.cell_count}/hits={self.hit_count}"
        )


@dataclass(frozen=True, slots=True)
class _MaterializedRow:
    row: _MatrixRow
    case_path: Path
    expected_macro_steps: int
    expected_frames: int
    expected_probes: int
    expected_path_kind: str


def main() -> None:
    """Run the matrix driver or one isolated child observation."""

    arguments = _arguments()
    if arguments.worker_spec is not None:
        _worker(arguments)
        return
    _driver(arguments)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("smoke", "release"), default="smoke")
    parser.add_argument(
        "--repeats",
        type=_positive_integer,
        help="observations per row (default: smoke=1, release=3)",
    )
    parser.add_argument(
        "--warmups",
        type=_positive_integer,
        default=1,
        help="same-process public runs before each warm observation",
    )
    parser.add_argument("--memory-limit-mb", type=_positive_integer, default=8192)
    parser.add_argument("--json", dest="json_path", type=Path)
    parser.add_argument("--worker-spec", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
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
    suite: _Suite = arguments.suite
    repeats = arguments.repeats if arguments.repeats is not None else (1 if suite == "smoke" else 3)
    rows = _matrix_rows(suite)
    if suite == "release":
        _validate_release_coverage(rows)

    with tempfile.TemporaryDirectory(prefix="chamber-particles-p14-") as temporary:
        root = Path(temporary)
        materialized = _materialize_rows(
            root / "cases",
            rows,
            suite=suite,
            memory_limit_mb=arguments.memory_limit_mb,
        )
        observations: list[dict[str, Any]] = []
        for item in materialized:
            worker_spec = root / "worker-specs" / f"{item.row.row_id}.json"
            worker_spec.parent.mkdir(parents=True, exist_ok=True)
            worker_spec.write_text(
                json.dumps(
                    _worker_spec(item),
                    allow_nan=False,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            observations.extend(
                _launch_worker(
                    worker_spec=worker_spec,
                    output_root=root / "results" / item.row.row_id / f"repeat-{repeat:03d}",
                    cache_dir=root / "numba-cache" / item.row.row_id / f"repeat-{repeat:03d}",
                    warmups=arguments.warmups if item.row.mode == "warm" else 0,
                    repeat=repeat,
                )
                for repeat in range(repeats)
            )
        _validate_observations(observations, materialized, repeats)
        summaries = _summaries(observations)
        report = {
            "benchmark": "p14_v0_1_orthogonal_matrix_v1",
            "captured_at_utc": datetime.now(UTC).isoformat(),
            "non_gating_seconds": True,
            "suite": suite,
            "conditions": {
                "rows": len(rows),
                "repeats_per_row": repeats,
                "warmup_runs_for_warm_mode": arguments.warmups,
                "execution_mode": "single-thread compiled CPU",
                "memory_limit_mb": arguments.memory_limit_mb,
                "case_materialization_in_measured_scope": False,
                "timed_scope": "load_case + simulate + open_result",
                "process_isolation": (
                    "each observation uses a fresh process and private initially empty "
                    "NUMBA_CACHE_DIR"
                ),
                "compiled_execution": (
                    "NUMBA_DISABLE_JIT=0; NUMBA_NUM_THREADS=1; the case schema has no "
                    "thread-count setting"
                ),
                "matrix_policy": (
                    "orthogonal coverage; output=all is limited to one short regular-field "
                    "row per suite"
                ),
            },
            "machine": _p14_machine_metadata(),
            "coverage": _coverage(rows),
            "observations": observations,
            "summaries": summaries,
            "comparisons": _comparisons(summaries),
            "identity": {
                "core_definition": (
                    "final/release/boundary/failure/series payload; output frames, probes, "
                    "execution partition, timing, RSS, and manifest are excluded"
                ),
                "exact_within_science_key": True,
                "exact_repeats": True,
                "exact_global_algorithm_revisions": True,
            },
            "interpretation": {
                "hard_checks": (
                    "completion, exact frame/probe shape and row counts, pinned revisions, "
                    "memory-plan fit, field-path compiled-dispatcher evidence, and scientific "
                    "identity"
                ),
                "timing": "absolute seconds and scaling ratios are descriptive machine-local evidence",
                "rss": (
                    "process high-water RSS includes Python/native libraries and is separate "
                    "from the solver-owned memory plan"
                ),
                "artifact_throughput": (
                    "artifact bytes divided by simulate time is effective end-to-end result "
                    "throughput, not pure background-writer bandwidth"
                ),
            },
        }

    encoded = json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    if arguments.json_path is not None:
        arguments.json_path.parent.mkdir(parents=True, exist_ok=True)
        arguments.json_path.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


def _matrix_rows(suite: _Suite) -> tuple[_MatrixRow, ...]:
    if suite == "smoke":
        return (
            _row("regular-cold", "field", 64, "regular", "initial", 0, "none", "cold", 1),
            _row(
                "regular-warm-all",
                "field",
                64,
                "regular",
                "initial",
                0,
                "all",
                "warm",
                1,
            ),
            _row("p1-initial", "field", 64, "p1", "initial", 0, "none", "warm", 32),
            _row(
                "p1-cross",
                "field",
                64,
                "p1",
                "cross",
                0,
                "none",
                "warm",
                32,
            ),
            _row("q1-initial", "field", 64, "q1", "initial", 0, "none", "warm", 16),
            _row(
                "q1-cross",
                "field",
                64,
                "q1",
                "cross",
                0,
                "none",
                "warm",
                16,
            ),
            _row("event-20", "event", 64, "regular", "initial", 20, "none", "warm", 1),
        )

    rows: list[_MatrixRow] = [
        _row(
            f"regular-n{count}",
            "field",
            count,
            "regular",
            "initial",
            0,
            "none",
            "warm",
            1,
        )
        for count in (10_000, 100_000, 1_000_000)
    ]
    rows.extend(
        (
            _row(
                "regular-n10000-cold",
                "field",
                10_000,
                "regular",
                "initial",
                0,
                "none",
                "cold",
                1,
            ),
            _row(
                "regular-n10000-sample",
                "field",
                10_000,
                "regular",
                "initial",
                0,
                "sample",
                "warm",
                1,
            ),
            _row(
                "regular-n10000-all",
                "field",
                10_000,
                "regular",
                "initial",
                0,
                "all",
                "warm",
                1,
            ),
        )
    )
    for layout, cells in (("p1", 10_368), ("q1", 2_500)):
        rows.extend(
            _row(
                f"{layout}-{motion}",
                "field",
                10_000,
                layout,
                motion,
                0,
                "none",
                "warm",
                cells,
            )
            for motion in ("initial", "cross")
        )
        rows.append(
            _row(
                f"{layout}-initial-cold",
                "field",
                10_000,
                layout,
                "initial",
                0,
                "none",
                "cold",
                cells,
            )
        )
    rows.extend(
        _row(
            f"event-h{hits}-n10000",
            "event",
            10_000,
            "regular",
            "initial",
            hits,
            "none",
            "warm",
            1,
        )
        for hits in (0, 1, 5, 20)
    )
    rows.extend(
        _row(
            f"event-h1-n{count}",
            "event",
            count,
            "regular",
            "initial",
            1,
            "none",
            "warm",
            1,
        )
        for count in (100_000, 1_000_000)
    )
    return tuple(rows)


def _row(
    row_id: str,
    family: Literal["field", "event"],
    particle_count: int,
    layout: _Layout,
    motion: _Motion,
    hit_count: int,
    output: _Output,
    mode: _Mode,
    cell_count: int,
) -> _MatrixRow:
    return _MatrixRow(
        row_id,
        family,
        particle_count,
        layout,
        motion,
        hit_count,
        output,
        mode,
        cell_count,
    )


def _materialize_rows(
    root: Path,
    rows: Sequence[_MatrixRow],
    *,
    suite: _Suite,
    memory_limit_mb: int,
) -> tuple[_MaterializedRow, ...]:
    root.mkdir(parents=True)
    shared: dict[str, tuple[Path, str, dict[str, Any], int, str]] = {}
    materialized = []
    for row in rows:
        if row.data_key not in shared:
            directory = root / row.data_key
            directory.mkdir(parents=True)
            data, specification, macro_steps, path_kind = _case_material(row, suite)
            data_path = directory / "case.h5"
            info = write(data_path, data)
            shared[row.data_key] = (
                data_path,
                info.content_hash,
                specification,
                macro_steps,
                path_kind,
            )
        data_path, content_hash, base_spec, macro_steps, path_kind = shared[row.data_key]
        specification = copy.deepcopy(base_spec)
        specification["case"] = {
            "name": f"p14_{row.row_id.replace('-', '_')}",
            "data_path": data_path.name,
            "expected_content_hash": content_hash,
        }
        specification["resources"] = {
            "memory_limit_mb": memory_limit_mb,
        }
        frame_times, probe_times = _configure_output(specification, row.output)
        case_path = data_path.parent / f"{row.row_id}.yaml"
        case_path.write_text(yaml.safe_dump(specification, sort_keys=False), encoding="utf-8")
        materialized.append(
            _MaterializedRow(
                row,
                case_path,
                macro_steps,
                len(frame_times),
                len(probe_times),
                path_kind,
            )
        )
    return tuple(materialized)


def _case_material(row: _MatrixRow, suite: _Suite) -> tuple[DataBundle, dict[str, Any], int, str]:
    if row.family == "event":
        definition = _expanded_definition(build_microcase("C09"), row.particle_count)
        specification = copy.deepcopy(definition.spec)
        gap_m = 2.0**-10
        end_s = gap_m / 4.0 if row.hit_count == 0 else (row.hit_count - 0.25) * gap_m
        specification["time"] = {"start_s": 0.0, "end_s": end_s, "dt_s": end_s}
        return definition.data, specification, 1, "linear_exact"

    definition = _expanded_definition(build_microcase("C04"), row.particle_count)
    if row.layout == "regular":
        return (
            _nonuniform_electric_data(definition.data),
            copy.deepcopy(definition.spec),
            4,
            "rk4_dense",
        )
    data, specification = _unstructured_field_case(definition.data, definition.spec, row, suite)
    macro_steps = 1 if row.motion == "initial" else 2
    return data, specification, macro_steps, "rk4_dense"


def _unstructured_field_case(
    base_data: DataBundle,
    base_specification: Mapping[str, Any],
    row: _MatrixRow,
    suite: _Suite,
) -> tuple[DataBundle, dict[str, Any]]:
    nx, ny = _mesh_dimensions(row.layout, row.cell_count, suite)
    nodes = _grid_nodes(nx, ny)
    connectivity = _grid_connectivity(row.layout, nx, ny)
    boundary = _grid_boundary(row.layout, nx, ny)
    if row.layout == "p1":
        geometry = GeometryData(
            nodes,
            boundary,
            ("wall",),
            tri3=connectivity,
            tri3_domain_id=np.zeros(connectivity.shape[0], dtype="<i4"),
        )
        layout = P1TriLayout(
            "unstructured",
            nodes.copy(),
            connectivity.copy(),
            np.ones(connectivity.shape[0], dtype="<u1"),
        )
    else:
        geometry = GeometryData(
            nodes,
            boundary,
            ("wall",),
            quad4=connectivity,
            quad4_domain_id=np.zeros(connectivity.shape[0], dtype="<i4"),
        )
        layout = Q1QuadLayout(
            "unstructured",
            nodes.copy(),
            connectivity.copy(),
            np.ones(connectivity.shape[0], dtype="<u1"),
        )

    source = _repeat_source(base_data.sources[0], row.particle_count)
    position, velocity = _unstructured_initial_state(row, nx, ny)
    source = replace(source, position_m=position, velocity_m_s=velocity)
    charge_to_acceleration = (
        float(source.charge_number[0]) * 1.602176634e-19 / float(source.mass_kg[0])
    )
    acceleration_x = -0.25 * (nodes[:, 0] - 0.5)
    electric_values = np.column_stack(
        (acceleration_x / charge_to_acceleration, np.zeros(nodes.shape[0], dtype="<f8"))
    )
    fields = tuple(
        replace(field, layout="unstructured", values=electric_values)
        if field.name == "electric_field"
        else field
        for field in base_data.fields
    )
    specification = copy.deepcopy(dict(base_specification))
    dt_s = 0.01
    macro_steps = 1 if row.motion == "initial" else 2
    specification["time"] = {
        "start_s": 0.0,
        "end_s": macro_steps * dt_s,
        "dt_s": dt_s,
    }
    specification["boundaries"] = [{"boundary_group": "wall", "priority": 10, "law": "stick"}]
    data = replace(
        base_data,
        geometry=geometry,
        layouts=(layout,),
        fields=fields,
        sources=(source,),
    )
    return data, specification


def _mesh_dimensions(layout: _Layout, requested_cells: int, suite: _Suite) -> tuple[int, int]:
    if suite == "smoke":
        return (4, 4)
    if layout == "p1":
        side = math.isqrt(requested_cells // 2)
        if 2 * side * side != requested_cells:
            raise RuntimeError("P1 release cell count must be twice a square")
        return side, side
    side = math.isqrt(requested_cells)
    if side * side != requested_cells:
        raise RuntimeError("Q1 release cell count must be a square")
    return side, side


def _grid_nodes(nx: int, ny: int) -> np.ndarray:
    x = np.linspace(0.0, 1.0, nx + 1, dtype="<f8")
    y = np.linspace(0.0, 1.0, ny + 1, dtype="<f8")
    return np.asarray(
        [[x[index_x], y[index_y]] for index_x in range(nx + 1) for index_y in range(ny + 1)],
        dtype="<f8",
    )


def _node_id(index_x: int, index_y: int, ny: int) -> int:
    return index_x * (ny + 1) + index_y


def _grid_connectivity(layout: _Layout, nx: int, ny: int) -> np.ndarray:
    cells: list[list[int]] = []
    for index_x in range(nx):
        for index_y in range(ny):
            node00 = _node_id(index_x, index_y, ny)
            node10 = _node_id(index_x + 1, index_y, ny)
            node11 = _node_id(index_x + 1, index_y + 1, ny)
            node01 = _node_id(index_x, index_y + 1, ny)
            if layout == "p1":
                cells.extend(([node00, node10, node11], [node00, node11, node01]))
            else:
                cells.append([node00, node10, node11, node01])
    return np.asarray(cells, dtype="<i8")


def _grid_boundary(layout: _Layout, nx: int, ny: int) -> BoundaryData:
    facets: list[list[int]] = []
    owners: list[int] = []
    for index_x in range(nx):
        facets.append([_node_id(index_x, 0, ny), _node_id(index_x + 1, 0, ny)])
        owners.append((2 * (index_x * ny)) if layout == "p1" else index_x * ny)
    for index_y in range(ny):
        facets.append([_node_id(nx, index_y, ny), _node_id(nx, index_y + 1, ny)])
        owners.append(
            (2 * ((nx - 1) * ny + index_y)) if layout == "p1" else (nx - 1) * ny + index_y
        )
    for index_x in range(nx - 1, -1, -1):
        facets.append([_node_id(index_x + 1, ny, ny), _node_id(index_x, ny, ny)])
        owners.append(
            (2 * (index_x * ny + ny - 1) + 1) if layout == "p1" else index_x * ny + ny - 1
        )
    for index_y in range(ny - 1, -1, -1):
        facets.append([_node_id(0, index_y + 1, ny), _node_id(0, index_y, ny)])
        owners.append((2 * index_y + 1) if layout == "p1" else index_y)
    count = len(facets)
    return BoundaryData(
        line2=np.asarray(facets, dtype="<i8"),
        boundary_id=np.full(count, 10, dtype="<i4"),
        group_id=np.zeros(count, dtype="<i4"),
        material_id=np.zeros(count, dtype="<i4"),
        owner_cell_type=np.full(count, 1 if layout == "p1" else 2, dtype="<u1"),
        owner_cell_local_index=np.asarray(owners, dtype="<i8"),
        orientation=np.ones(count, dtype="<i1"),
    )


def _unstructured_initial_state(row: _MatrixRow, nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    ordinal = np.arange(row.particle_count, dtype=np.int64)
    usable_x = max(1, nx - 4)
    usable_y = max(1, ny - 2)
    cell_x = 1 + (ordinal % usable_x)
    cell_y = 1 + ((ordinal // usable_x) % usable_y)
    position = np.column_stack(((cell_x + 0.37) / nx, (cell_y + 0.43) / ny)).astype("<f8")
    velocity = np.zeros((row.particle_count, 2), dtype="<f8")
    if row.motion == "cross":
        velocity[:, 0] = 1.25 / (nx * 0.01)
    return position, velocity


def _configure_output(
    specification: dict[str, Any], output: _Output
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    time_spec = specification["time"]
    start_s = float(time_spec["start_s"])
    end_s = float(time_spec["end_s"])
    dt_s = float(time_spec["dt_s"])
    macro_steps = math.ceil((end_s - start_s) / dt_s)
    times = tuple(min(end_s, start_s + (index + 1) * dt_s) for index in range(macro_steps))
    if output == "none":
        specification["output"] = {"trajectories": None, "probes": None}
        return (), ()
    if output == "all":
        specification["output"] = {
            "trajectories": {
                "selection": "all",
                "schedule": {"explicit_times_s": list(times)},
            },
            "probes": None,
        }
        return times, ()
    specification["output"] = {
        "trajectories": None,
        "probes": {
            "particle_ids": list(_PROBE_PARTICLE_IDS),
            "schedule": {"explicit_times_s": list(times)},
        },
    }
    return (), times


def _worker_spec(item: _MaterializedRow) -> dict[str, object]:
    boundary_events = item.row.particle_count * item.row.hit_count
    return {
        "row": asdict(item.row),
        "case_path": str(item.case_path),
        "expected_macro_steps": item.expected_macro_steps,
        "expected_frames": item.expected_frames,
        "expected_frame_rows": item.expected_frames * item.row.particle_count,
        "expected_probes": item.expected_probes,
        "expected_probe_rows": item.expected_probes * len(_PROBE_PARTICLE_IDS),
        "expected_path_kind": item.expected_path_kind,
        "expected_boundary_events": boundary_events,
    }


def _launch_worker(
    *,
    worker_spec: Path,
    output_root: Path,
    cache_dir: Path,
    warmups: int,
    repeat: int,
) -> dict[str, Any]:
    cache_dir.mkdir(parents=True)
    command = [
        sys.executable,
        "-m",
        "tests.performance.p14_matrix",
        "--worker-spec",
        str(worker_spec),
        "--worker-output",
        str(output_root),
        "--worker-warmups",
        str(warmups),
    ]
    environment = os.environ.copy()
    environment["NUMBA_CACHE_DIR"] = str(cache_dir)
    environment["NUMBA_DISABLE_JIT"] = "0"
    environment["NUMBA_NUM_THREADS"] = "1"
    completed = subprocess.run(
        command, check=False, capture_output=True, text=True, env=environment
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"P14 worker {worker_spec.stem} failed with exit {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    try:
        observation = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("P14 worker did not return JSON") from error
    if not isinstance(observation, dict):
        raise RuntimeError("P14 worker returned a non-object")
    observation["repeat"] = repeat
    return observation


def _worker(arguments: argparse.Namespace) -> None:
    if (
        arguments.worker_spec is None
        or arguments.worker_output is None
        or arguments.worker_warmups is None
    ):
        raise SystemExit("P14 worker requires spec, output, and warmup count")
    specification = json.loads(arguments.worker_spec.read_text(encoding="utf-8"))
    if not isinstance(specification, dict):
        raise SystemExit("P14 worker specification must be an object")
    for index in range(arguments.worker_warmups):
        _execute_run(specification, arguments.worker_output / f"warmup-{index:03d}", digest=False)
        gc.collect()
    observation = _execute_run(specification, arguments.worker_output / "measured", digest=True)
    print(json.dumps(observation, allow_nan=False, sort_keys=True))


def _execute_run(
    specification: Mapping[str, object], output: Path, *, digest: bool
) -> dict[str, Any]:
    row = _mapping_value(specification, "row")
    case_path = Path(str(specification["case_path"]))
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

    physics_signature_count = len(evaluate_physics_tile_into.signatures)
    _validate_run(summary, result, row, specification, physics_signature_count)
    manifest = result.manifest
    memory_plan = _mapping_value(manifest, "memory_plan")
    components = _mapping_value(memory_plan, "components")
    event_work = manifest.get("event_refinement")
    all_revisions = {key: value for key, value in manifest.items() if key.endswith("_revision")}
    conditional_revision_names = {
        "boundary_algorithm_revision",
        "exponential_midpoint_enclosure_revision",
        "exponential_midpoint_revision",
        "rk4_dense_path_revision",
        "rk4_enclosure_revision",
    }
    revisions = {
        key: value for key, value in all_revisions.items() if key not in conditional_revision_names
    }
    conditional_revisions = {
        key: value for key, value in all_revisions.items() if key in conditional_revision_names
    }
    artifact_bytes = _directory_bytes(output)
    core_digest = _core_payload_digest(result) if digest else None
    full_digest = _scientific_payload_digest(result) if digest else None
    return {
        "row_id": row["row_id"],
        "science_key": _science_key(row),
        "mode": row["mode"],
        "particle_count": summary.particle_count,
        "layout": row["layout"],
        "motion": row["motion"],
        "hit_count_per_particle": row["hit_count"],
        "output_mode": row["output"],
        "execution_mode": "serial",
        "cell_count": row["cell_count"],
        "macro_step_count": summary.macro_step_count,
        "boundary_event_count": summary.boundary_event_count,
        "frame_count": summary.frame_count,
        "frame_row_count": summary.frame_row_count,
        "probe_count": summary.probe_count,
        "probe_row_count": summary.probe_row_count,
        "timing_s": {
            "load_case": load_elapsed,
            "simulate": simulate_elapsed,
            "open_result": open_elapsed,
            "public_end_to_end": total_elapsed,
        },
        "throughput": {
            "particles_per_simulate_s": summary.particle_count / simulate_elapsed,
            "particle_macro_steps_per_simulate_s": (
                summary.particle_count * summary.macro_step_count / simulate_elapsed
            ),
            "events_per_simulate_s": summary.boundary_event_count / simulate_elapsed,
            "effective_result_artifact_bytes_per_simulate_s": artifact_bytes / simulate_elapsed,
        },
        "memory": {
            "measurement_source": rss_source,
            "peak_rss_before_measured_scope_bytes": peak_before,
            "peak_rss_after_measured_scope_bytes": peak_after,
            "additional_process_high_water_bytes": max(0, peak_after - peak_before),
            "rss_before_measured_scope_bytes": rss_before,
            "rss_after_measured_scope_bytes": rss_after,
            "solver_memory_plan": dict(memory_plan),
            "planned_field_runtime_bytes": int(components.get("field_runtime", 0)),
        },
        "event_work": event_work,
        "algorithm_revisions": revisions,
        "conditional_algorithm_revisions": conditional_revisions,
        "resolved_path_kind": _mapping_value(manifest, "resolved")["path_kind"],
        "compiled_execution": {
            "physics_tile_signature_count": physics_signature_count,
            "numba_disable_jit": os.environ.get("NUMBA_DISABLE_JIT"),
            "numba_num_threads": os.environ.get("NUMBA_NUM_THREADS"),
        },
        "case_identity": {
            "case_file_hash": case.case_file_hash,
            "data_content_hash": case.content_hash,
        },
        "core_payload_sha256": core_digest,
        "full_payload_sha256": full_digest,
        "case_artifact_bytes": _directory_bytes(case_path.parent),
        "result_artifact_bytes": artifact_bytes,
        "result_bytes_per_particle": artifact_bytes / summary.particle_count,
    }


def _validate_run(
    summary: Any,
    result: Any,
    row: Mapping[str, object],
    specification: Mapping[str, object],
    physics_signature_count: int,
) -> None:
    particle_count = int(row["particle_count"])
    hit_count = int(row["hit_count"])
    if summary.particle_count != particle_count:
        raise RuntimeError("P14 worker returned an unexpected particle count")
    if summary.macro_step_count != int(specification["expected_macro_steps"]):
        raise RuntimeError("P14 worker returned an unexpected macro-step count")
    expected_boundary_events = int(
        specification.get("expected_boundary_events", particle_count * hit_count)
    )
    if summary.boundary_event_count != expected_boundary_events:
        raise RuntimeError("P14 worker returned an unexpected boundary-event count")
    if summary.failure_event_count != 0:
        raise RuntimeError("P14 performance case produced particle failures")
    if summary.frame_count != int(specification["expected_frames"]):
        raise RuntimeError("P14 performance case returned an unexpected frame count")
    if summary.frame_row_count != int(specification["expected_frame_rows"]):
        raise RuntimeError("P14 performance case returned an unexpected frame row count")
    if summary.probe_count != int(specification["expected_probes"]):
        raise RuntimeError("P14 performance case returned an unexpected probe count")
    if summary.probe_row_count != int(specification["expected_probe_rows"]):
        raise RuntimeError("P14 performance case returned an unexpected probe row count")
    resolved = _mapping_value(result.manifest, "resolved")
    if resolved.get("path_kind") != specification["expected_path_kind"]:
        raise RuntimeError("P14 performance case resolved an unexpected path kind")
    requested = _mapping_value(result.manifest, "requested")
    if "threads" in requested or "threads" in resolved or "thread_team" in resolved:
        raise RuntimeError("P14 serial result unexpectedly records a thread-count setting")
    memory_plan = _mapping_value(result.manifest, "memory_plan")
    if int(memory_plan["planned_bytes"]) > int(memory_plan["limit_bytes"]):
        raise RuntimeError("P14 solver memory plan exceeds its configured limit")
    if memory_plan.get("revision") != _EXPECTED_MEMORY_PLAN_REVISION:
        raise RuntimeError("P14 result recorded an unexpected memory-plan revision")
    if memory_plan.get("runtime_layout_revision") != _EXPECTED_RUNTIME_LAYOUT_REVISION:
        raise RuntimeError("P14 result recorded an unexpected runtime-layout revision")
    for key, expected in _EXPECTED_REVISIONS.items():
        if result.manifest.get(key) != expected:
            raise RuntimeError(f"P14 expected {key}={expected!r}, got {result.manifest.get(key)!r}")
    if row["family"] == "field" and physics_signature_count < 1:
        raise RuntimeError("P14 field worker did not compile the physics tile")
    if os.environ.get("NUMBA_DISABLE_JIT") != "0" or os.environ.get("NUMBA_NUM_THREADS") != "1":
        raise RuntimeError("P14 worker did not isolate compiled execution")


def _core_payload_digest(result: Any) -> str:
    digest = hashlib.sha256()
    readers = (
        ("final", result.read_final),
        ("release_events", result.read_release_events),
        ("boundary_events", result.read_boundary_events),
        ("failure_events", result.read_failure_events),
        ("lifecycle_series", result.read_lifecycle_series),
    )
    for name, reader in readers:
        _digest_dataclass(digest, name, reader())
    return f"sha256:{digest.hexdigest()}"


def _scientific_payload_digest(result: Any) -> str:
    digest = hashlib.sha256()
    readers = (
        ("final", result.read_final),
        ("release_events", result.read_release_events),
        ("boundary_events", result.read_boundary_events),
        ("failure_events", result.read_failure_events),
        ("lifecycle_series", result.read_lifecycle_series),
    )
    for name, reader in readers:
        _digest_dataclass(digest, name, reader())
    for index, frame in enumerate(result.iter_frames()):
        _digest_dataclass(digest, f"frame/{index}", frame)
    for index, probe in enumerate(result.iter_probes()):
        _digest_dataclass(digest, f"probe/{index}", probe)
    return f"sha256:{digest.hexdigest()}"


def _validate_observations(
    observations: Sequence[Mapping[str, object]],
    materialized: Sequence[_MaterializedRow],
    repeats: int,
) -> None:
    for item in materialized:
        selected = [value for value in observations if value["row_id"] == item.row.row_id]
        if len(selected) != repeats:
            raise RuntimeError(f"P14 {item.row.row_id}: observation count is incomplete")
        full_digests = {value["full_payload_sha256"] for value in selected}
        if None in full_digests or len(full_digests) != 1:
            raise RuntimeError(f"P14 {item.row.row_id}: repeated scientific payload changed")
        revisions = {
            json.dumps(value["algorithm_revisions"], sort_keys=True, separators=(",", ":"))
            for value in selected
        }
        if len(revisions) != 1:
            raise RuntimeError(f"P14 {item.row.row_id}: algorithm revisions changed")

    global_revisions = {
        json.dumps(value["algorithm_revisions"], sort_keys=True, separators=(",", ":"))
        for value in observations
    }
    if len(global_revisions) != 1:
        raise RuntimeError("P14 globally applicable algorithm revisions differ across rows")

    for science_key in sorted({str(value["science_key"]) for value in observations}):
        selected = [value for value in observations if value["science_key"] == science_key]
        core_digests = {value["core_payload_sha256"] for value in selected}
        if None in core_digests or len(core_digests) != 1:
            raise RuntimeError(f"P14 core scientific payload changed for {science_key}")


def _summaries(observations: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    summaries = []
    for row_id in sorted({str(value["row_id"]) for value in observations}):
        selected = [value for value in observations if value["row_id"] == row_id]
        timings = [_mapping_value(value, "timing_s") for value in selected]
        throughputs = [_mapping_value(value, "throughput") for value in selected]
        memories = [_mapping_value(value, "memory") for value in selected]
        representative = selected[0]
        summaries.append(
            {
                "row_id": row_id,
                "science_key": representative["science_key"],
                "mode": representative["mode"],
                "particle_count": representative["particle_count"],
                "layout": representative["layout"],
                "motion": representative["motion"],
                "hit_count_per_particle": representative["hit_count_per_particle"],
                "output_mode": representative["output_mode"],
                "execution_mode": representative["execution_mode"],
                "cell_count": representative["cell_count"],
                "observations": len(selected),
                "median_timing_s": {
                    name: statistics.median(float(value[name]) for value in timings)
                    for name in ("load_case", "simulate", "open_result", "public_end_to_end")
                },
                "median_throughput": {
                    name: statistics.median(float(value[name]) for value in throughputs)
                    for name in (
                        "particles_per_simulate_s",
                        "particle_macro_steps_per_simulate_s",
                        "events_per_simulate_s",
                        "effective_result_artifact_bytes_per_simulate_s",
                    )
                },
                "maximum_peak_rss_bytes": max(
                    int(value["peak_rss_after_measured_scope_bytes"]) for value in memories
                ),
                "solver_memory_plan": _mapping_value(memories[0], "solver_memory_plan"),
                "result_artifact_bytes": representative["result_artifact_bytes"],
                "result_bytes_per_particle": representative["result_bytes_per_particle"],
                "event_work": representative["event_work"],
                "algorithm_revisions": representative["algorithm_revisions"],
                "conditional_algorithm_revisions": representative[
                    "conditional_algorithm_revisions"
                ],
                "core_payload_sha256": representative["core_payload_sha256"],
                "full_payload_sha256": representative["full_payload_sha256"],
            }
        )
    return summaries


def _comparisons(summaries: Sequence[Mapping[str, object]]) -> dict[str, object]:
    comparisons: dict[str, object] = {
        "cold_over_warm": {},
        "regular_particle_scaling": _regular_particle_scaling(summaries),
    }
    for science_key in sorted({str(value["science_key"]) for value in summaries}):
        selected = [value for value in summaries if value["science_key"] == science_key]
        cold = next((value for value in selected if value["mode"] == "cold"), None)
        warm = next(
            (
                value
                for value in selected
                if value["mode"] == "warm"
                and cold is not None
                and value["output_mode"] == cold["output_mode"]
            ),
            None,
        )
        if cold is not None and warm is not None:
            cold_time = float(_mapping_value(cold, "median_timing_s")["simulate"])
            warm_time = float(_mapping_value(warm, "median_timing_s")["simulate"])
            _mapping_value(comparisons, "cold_over_warm")[science_key] = cold_time / warm_time
    return comparisons


def _regular_particle_scaling(
    summaries: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    selected = sorted(
        (
            value
            for value in summaries
            if str(value["science_key"]).startswith("field/regular/initial/")
            and value["layout"] == "regular"
            and value["mode"] == "warm"
            and value["output_mode"] == "none"
            and int(value["particle_count"]) in {10_000, 100_000, 1_000_000}
        ),
        key=lambda value: int(value["particle_count"]),
    )
    if len(selected) != 3:
        return {"status": "not_available_in_this_preset"}

    points = []
    for value in selected:
        plan = _mapping_value(value, "solver_memory_plan")
        particle_count = int(value["particle_count"])
        planned_bytes = int(plan["planned_bytes"])
        peak_rss_bytes = int(value["maximum_peak_rss_bytes"])
        simulate_s = float(_mapping_value(value, "median_timing_s")["simulate"])
        points.append(
            {
                "particle_count": particle_count,
                "simulate_s": simulate_s,
                "planned_bytes": planned_bytes,
                "peak_rss_bytes": peak_rss_bytes,
                "planned_bytes_per_particle": planned_bytes / particle_count,
                "peak_rss_bytes_per_particle": peak_rss_bytes / particle_count,
            }
        )
    first = points[0]
    last = points[-1]
    particle_ratio = float(last["particle_count"]) / float(first["particle_count"])
    return {
        "status": "measured",
        "points": points,
        "end_to_end_log_slopes": {
            "simulate_s": _log_slope(first["simulate_s"], last["simulate_s"], particle_ratio),
            "planned_bytes": _log_slope(
                first["planned_bytes"], last["planned_bytes"], particle_ratio
            ),
            "peak_rss_bytes": _log_slope(
                first["peak_rss_bytes"], last["peak_rss_bytes"], particle_ratio
            ),
        },
    }


def _log_slope(first: object, last: object, particle_ratio: float) -> float:
    return math.log(float(last) / float(first)) / math.log(particle_ratio)


def _coverage(rows: Sequence[_MatrixRow]) -> dict[str, object]:
    return {
        "particle_counts": sorted({row.particle_count for row in rows}),
        "layouts": sorted({row.layout for row in rows}),
        "unstructured_motion": sorted({row.motion for row in rows if row.layout in {"p1", "q1"}}),
        "hit_counts_per_particle": sorted({row.hit_count for row in rows if row.family == "event"}),
        "output_modes": sorted({row.output for row in rows}),
        "jit_modes": sorted({row.mode for row in rows}),
    }


def _validate_release_coverage(rows: Sequence[_MatrixRow]) -> None:
    coverage = _coverage(rows)
    expectations = {
        "particle_counts": {10_000, 100_000, 1_000_000},
        "layouts": {"regular", "p1", "q1"},
        "unstructured_motion": {"initial", "cross"},
        "hit_counts_per_particle": {0, 1, 5, 20},
        "output_modes": {"none", "sample", "all"},
        "jit_modes": {"cold", "warm"},
    }
    for name, expected in expectations.items():
        if not expected.issubset(set(coverage[name])):
            raise RuntimeError(f"P14 release matrix does not cover {name}")


def _science_key(row: Mapping[str, object]) -> str:
    return (
        f"{row['family']}/{row['layout']}/{row['motion']}/n={row['particle_count']}/"
        f"cells={row['cell_count']}/hits={row['hit_count']}"
    )


def _mapping_value(value: Mapping[str, object], key: str) -> Mapping[str, object]:
    result = value[key]
    if not isinstance(result, Mapping):
        raise RuntimeError(f"P14 value {key!r} is not a mapping")
    return result


def _p14_machine_metadata() -> dict[str, object]:
    machine = _machine_metadata()
    machine["numba"] = metadata.version("numba")
    machine["os_reported_logical_cpu_count"] = os.cpu_count()
    return machine


if __name__ == "__main__":
    main()
