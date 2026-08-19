"""Artifact writer for the public v0.2 result model.

The writer depends only on the public result value model.  It deliberately has no
access to solver runtimes, COMSOL adapters, geometry samplers, or mutable solver
arrays.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._application_types import ArtifactManifest, ArtifactRecord, SimulationResult
from .artifacts import DEBUG_ARTIFACTS, STANDARD_ARTIFACTS
from .integrity import sha256_file

SCHEMA_VERSION = 2
RUN_SUMMARY_ARTIFACT_TYPE = "particle_tracer.run_summary"
DEBUG_DIAGNOSTICS_ARTIFACT_TYPE = "particle_tracer.debug_diagnostics"

WALL_SUMMARY_COLUMNS = (
    "schema_version",
    "part_id",
    "outcome",
    "wall_mode",
    "count",
)
TRAJECTORY_FRAME_COLUMNS = (
    "schema_version",
    "save_index",
    "time_s",
    "step_name",
    "segment_name",
)
WALL_EVENT_COLUMNS = (
    "schema_version",
    "time_s",
    "hit_time_s",
    "particle_id",
    "part_id",
    "boundary_primitive_id",
    "boundary_primitive_kind",
    "boundary_hit_ambiguous",
    "step_name",
    "segment_name",
    "outcome",
    "wall_mode",
    "alpha_hit",
    "material_id",
    "material_name",
    "particle_mass_kg",
    "particle_diameter_m",
    "impact_speed_mps",
    "impact_normal_speed_mps",
    "impact_tangential_speed_mps",
    "impact_angle_deg_from_normal",
    "hit_x_m",
    "hit_y_m",
    "hit_z_m",
    "normal_x",
    "normal_y",
    "normal_z",
    "v_hit_x_mps",
    "v_hit_y_mps",
    "v_hit_z_mps",
)
STEP_SUMMARY_COLUMNS = (
    "schema_version",
    "time_s",
    "step_name",
    "segment_name",
    "released_count",
    "active_count",
    "stuck_count",
    "absorbed_count",
    "contact_sliding_count",
    "escaped_count",
    "valid_mask_violation_count_step",
    "invalid_mask_stopped_count_step",
    "frozen_count",
)
FORCE_CONTRIBUTION_COLUMNS = (
    "schema_version",
    "name",
    "enabled",
    "model",
    "status",
    "physical_quantity",
    "required_fields",
    "optional_fields",
    "field_sources",
    "parameters",
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_ready(value), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _versioned_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{"schema_version": SCHEMA_VERSION, **dict(row)} for row in rows]


def _write_rows(
    path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]
) -> None:
    pd.DataFrame(_versioned_rows(rows), columns=list(columns)).to_csv(path, index=False)


def _final_particles_frame(result: SimulationResult) -> pd.DataFrame:
    state = result.state
    count = int(result.stats.particle_count)
    values: dict[str, Any] = {
        "schema_version": np.full(count, SCHEMA_VERSION, dtype=np.int16),
        "particle_id": state.particle_id,
        "release_time_s": state.release_time_s,
        "released": np.asarray(state.released, dtype=np.int8),
        "final_state": state.terminal_state,
        "invalid_stop_reason": state.invalid_stop_reason,
        "source_part_id": state.source_part_id,
        "material_id": state.material_id,
        "mass_kg": state.mass_kg,
        "drag_diameter_m": state.drag_diameter_m,
        "charge_C": state.charge_C,
        "contact_part_id": state.contact_part_id,
        "final_step_name": np.full(count, result.final_step_name, dtype=object),
        "final_segment_name": np.full(count, result.final_segment_name, dtype=object),
    }
    for index, axis in enumerate(result.axis_names):
        values[f"{axis}_m"] = state.position_m[:, index]
        values[f"v{axis}_mps"] = state.velocity_mps[:, index]
        values[f"contact_normal_{axis}"] = state.contact_normal[:, index]
    return pd.DataFrame(values)


def _wall_summary_rows(result: SimulationResult) -> list[dict[str, Any]]:
    return [
        {
            "part_id": int(part_id),
            "outcome": str(outcome),
            "wall_mode": str(wall_mode),
            "count": int(count),
        }
        for (part_id, outcome, wall_mode), count in sorted(
            result.wall_summary.items(),
            key=lambda item: (int(item[0][0]), str(item[0][1]), str(item[0][2])),
        )
    ]


def _run_summary(result: SimulationResult) -> dict[str, Any]:
    terminal = result.stats.terminal_counts
    artifacts = STANDARD_ARTIFACTS + (
        DEBUG_ARTIFACTS if result.plan.output_mode == "debug" else ()
    )
    report: dict[str, Any] = {
        "artifact_type": RUN_SUMMARY_ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "particle_count": int(result.stats.particle_count),
        "released_count": int(result.stats.released_count),
        "unreleased_count": int(
            result.stats.particle_count - result.stats.released_count
        ),
        "active_count": int(terminal.get("active_free_flight", 0)),
        "stuck_count": int(terminal.get("stuck", 0)),
        "frozen_count": int(terminal.get("frozen", 0)),
        "absorbed_count": int(terminal.get("absorbed", 0)),
        "escaped_count": int(terminal.get("escaped", 0)),
        "invalid_mask_stopped_count": int(terminal.get("invalid_mask_stopped", 0)),
        "numerical_boundary_stopped_count": int(
            terminal.get("numerical_boundary_stopped", 0)
        ),
        "wall_outcome_counts": dict(result.stats.wall_outcome_counts),
        "final_state_counts": dict(result.stats.terminal_counts),
        "coordinate_system": result.plan.coordinate_system,
        "axis_names": list(result.axis_names),
        "integrator": result.plan.integrator,
        "drag_model": result.drag_model,
        "experimental_features": list(result.experimental_features),
        "output_mode": result.plan.output_mode,
        "execution": dict(result.execution_metadata),
        "timing_s": dict(result.stats.timing_s),
        "memory_estimate_bytes": dict(result.stats.memory_estimate_bytes),
        "artifacts": list(artifacts),
    }
    report.update(
        {str(key): int(value) for key, value in result.stats.safety_counters.items()}
    )
    return report


def _force_rows(result: SimulationResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in result.debug.get("force_contributions", ()):
        row = dict(raw)
        row["required_fields"] = ";".join(map(str, row.get("required_fields", ())))
        row["optional_fields"] = ";".join(map(str, row.get("optional_fields", ())))
        row["field_sources"] = json.dumps(
            _json_ready(row.get("field_sources", {})), sort_keys=True
        )
        row["parameters"] = json.dumps(
            _json_ready(row.get("parameters", {})), sort_keys=True
        )
        rows.append(row)
    return rows


def _write_debug(result: SimulationResult, output_dir: Path) -> None:
    np.save(
        output_dir / "trajectory.npy",
        np.asarray(result.debug.get("trajectory_m"), dtype=np.float64),
    )
    _write_rows(
        output_dir / "trajectory_frames.csv",
        result.debug.get("save_frames", ()),
        TRAJECTORY_FRAME_COLUMNS,
    )
    _write_rows(
        output_dir / "wall_events.csv",
        result.debug.get("wall_events", ()),
        WALL_EVENT_COLUMNS,
    )
    _write_rows(
        output_dir / "step_summary.csv",
        result.debug.get("step_summary", ()),
        STEP_SUMMARY_COLUMNS,
    )
    _write_rows(
        output_dir / "force_contributions.csv",
        _force_rows(result),
        FORCE_CONTRIBUTION_COLUMNS,
    )
    _write_json(
        output_dir / "debug_diagnostics.json",
        {
            "artifact_type": DEBUG_DIAGNOSTICS_ARTIFACT_TYPE,
            "schema_version": SCHEMA_VERSION,
            "collision": result.debug.get("collision_diagnostics", {}),
            "max_hit_events": result.debug.get("max_hit_events", ()),
        },
    )


def _artifact_record(artifact_type: str, path: Path) -> ArtifactRecord:
    return ArtifactRecord(
        artifact_type=artifact_type,
        path=path.resolve(),
        size_bytes=int(path.stat().st_size),
        sha256=sha256_file(path),
    )


def _require_publishable_output_directory(
    root: Path,
    expected: Sequence[str],
    *,
    allow_empty: bool,
) -> bool:
    if not root.exists():
        return False
    if not root.is_dir():
        raise FileExistsError(
            f"output path already exists and is not a directory: {root}"
        )
    names = sorted(path.name for path in root.iterdir())
    unexpected = sorted(name for name in names if name not in expected)
    if unexpected:
        raise ValueError(
            "output directory contains files outside the declared artifact contract: "
            + ", ".join(unexpected)
        )
    if not names and allow_empty:
        return True
    contents = ", ".join(names) if names else "an existing empty directory"
    raise FileExistsError(
        f"immutable result output already exists ({contents}): {root}"
    )


def _write_result_files(result: SimulationResult, root: Path) -> None:
    _final_particles_frame(result).to_csv(root / "final_particles.csv", index=False)
    _write_rows(
        root / "wall_summary.csv", _wall_summary_rows(result), WALL_SUMMARY_COLUMNS
    )
    if result.plan.output_mode == "debug":
        _write_debug(result, root)
    _write_json(root / "run_summary.json", _run_summary(result))


def _publish_staged_result(
    staging: Path,
    root: Path,
    expected: Sequence[str],
    *,
    root_was_empty: bool,
) -> None:
    missing = [name for name in expected if not (staging / name).is_file()]
    if missing:
        raise RuntimeError(
            f"result writer did not create required artifact(s): {', '.join(missing)}"
        )
    if root_was_empty:
        if not root.is_dir() or any(root.iterdir()):
            raise FileExistsError(
                f"output directory changed while result was staged: {root}"
            )
        root.rmdir()
    else:
        _require_publishable_output_directory(root, expected, allow_empty=False)
    staging.rename(root)


def write_result(result: SimulationResult, output_dir: str | Path) -> ArtifactManifest:
    """Atomically publish one immutable result into a new or empty directory.

    The directory is staged beside its destination and renamed only after all
    declared artifacts exist.  Existing destinations are never overwritten.
    """

    root = Path(output_dir).resolve()
    expected = STANDARD_ARTIFACTS + (
        DEBUG_ARTIFACTS if result.plan.output_mode == "debug" else ()
    )
    root_was_empty = _require_publishable_output_directory(
        root,
        expected,
        allow_empty=True,
    )
    root.parent.mkdir(parents=True, exist_ok=True)
    if not root_was_empty:
        _require_publishable_output_directory(root, expected, allow_empty=False)

    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{root.name}.staging-",
            dir=root.parent,
        )
    )
    try:
        _write_result_files(result, staging)
        _publish_staged_result(
            staging,
            root,
            expected,
            root_was_empty=root_was_empty,
        )
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    records = tuple(_artifact_record(Path(name).stem, root / name) for name in expected)
    return ArtifactManifest(output_dir=root, records=records)


__all__ = ("write_result",)
