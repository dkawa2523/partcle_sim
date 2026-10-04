"""Focused checks for the Case-P pre-final geometry-tolerance bridge."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
from tools.vv.comsol.evaluate_m3c2_caseP_event_tolerance_pre_final import (
    OUTPUT_COUNT,
    PARTICLE_COUNT,
    compare_results,
)

from chamber_particles.output import FinalParticles, TrajectoryFrame


def _solver_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _empty_events() -> SimpleNamespace:
    return SimpleNamespace(
        particle_id=np.empty(0, dtype=np.int64),
        event_ordinal=np.empty(0, dtype=np.uint32),
        candidate_offset=np.asarray([0], dtype=np.int64),
        candidate_facet_id=np.empty(0, dtype=np.int64),
        primary_facet_id=np.empty(0, dtype=np.int64),
        boundary_id=np.empty(0, dtype=np.int32),
        material_id=np.empty(0, dtype=np.int32),
        law_id=np.empty(0, dtype="<U1"),
        outcome=np.empty(0, dtype="<U1"),
        position_m=np.empty((0, 2), dtype=np.float64),
        time_s=np.empty(0, dtype=np.float64),
        position_budget_m=np.empty(0, dtype=np.float64),
        time_budget_s=np.empty(0, dtype=np.float64),
        normal=np.empty((0, 2), dtype=np.float64),
        velocity_pre_m_s=np.empty((0, 2), dtype=np.float64),
    )


def _final(position_offset: float = 0.0) -> FinalParticles:
    particle_id = np.arange(1, PARTICLE_COUNT + 1, dtype=np.int64)
    zeros = np.zeros(PARTICLE_COUNT, dtype=np.float64)
    position = np.column_stack((particle_id.astype(np.float64), zeros))
    position[10, 0] += position_offset
    return FinalParticles(
        particle_id=particle_id,
        source_id=np.zeros(PARTICLE_COUNT, dtype=np.int32),
        time_s=np.full(PARTICLE_COUNT, 0.03),
        position_m=position,
        velocity_m_s=np.zeros((PARTICLE_COUNT, 2), dtype=np.float64),
        charge_number=zeros.copy(),
        lifecycle=np.ones(PARTICLE_COUNT, dtype=np.uint8),
        kinematics_valid=np.ones(PARTICLE_COUNT, dtype=np.uint8),
        failure_reason_code=np.zeros(PARTICLE_COUNT, dtype=np.uint16),
        mass_kg=np.ones(PARTICLE_COUNT, dtype=np.float64),
        drag_diameter_m=np.full(PARTICLE_COUNT, 1.0e-7),
        electrostatic_radius_m=np.full(PARTICLE_COUNT, 5.0e-8),
        displaced_volume_m3=np.ones(PARTICLE_COUNT, dtype=np.float64),
        model_weight=np.ones(PARTICLE_COUNT, dtype=np.float64),
        material_id=np.zeros(PARTICLE_COUNT, dtype=np.int32),
    )


def _frames(position_offset: float = 0.0) -> list[TrajectoryFrame]:
    particle_id = np.arange(1, PARTICLE_COUNT + 1, dtype=np.int64)
    frames: list[TrajectoryFrame] = []
    for index in range(OUTPUT_COUNT):
        position = np.column_stack(
            (particle_id.astype(np.float64) + index, np.zeros(PARTICLE_COUNT))
        )
        if index == OUTPUT_COUNT - 1:
            position[10, 0] += position_offset
        frames.append(
            TrajectoryFrame(
                time_s=float(index),
                particle_id=particle_id,
                position_m=position,
                velocity_m_s=np.zeros((PARTICLE_COUNT, 2), dtype=np.float64),
                charge_number=np.zeros(PARTICLE_COUNT, dtype=np.float64),
                lifecycle=np.ones(PARTICLE_COUNT, dtype=np.uint8),
            )
        )
    return frames


def _result(*, position_offset: float = 0.0) -> Any:
    events = _empty_events()
    final = _final(position_offset)
    frames = _frames(position_offset)
    return SimpleNamespace(
        read_boundary_events=lambda: events,
        read_failure_events=lambda: SimpleNamespace(particle_id=np.empty(0, dtype=np.int64)),
        iter_frames=lambda: iter(frames),
        read_final=lambda: final,
    )


def test_zero_event_bridge_requires_exact_scheduled_and_final_state() -> None:
    report = compare_results(_result(), _result())
    scheduled = cast(dict[str, object], report["scheduled_state_bitwise_comparison"])
    final = cast(dict[str, object], report["final_state_bitwise_comparison"])

    assert report["status"] == "PASS"
    assert report["classification"] == "PASS_NO_EVENT_OPERATIONAL_BRIDGE"
    assert report["validity_condition"] == "zero_candidate_boundary_events"
    assert scheduled["exact"] is True
    assert final["exact"] is True


def test_zero_event_bridge_fails_on_one_bitwise_state_difference() -> None:
    report = compare_results(_result(), _result(position_offset=1.0e-12))
    scheduled = cast(dict[str, object], report["scheduled_state_bitwise_comparison"])
    final = cast(dict[str, object], report["final_state_bitwise_comparison"])

    assert report["status"] == "FAIL"
    assert report["validity_condition"] == "not_qualified"
    assert scheduled["exact"] is False
    assert final["exact"] is False


def test_v2_recipe_preserves_v1_science_matrix_and_locks_formatted_evaluator() -> None:
    root = _solver_root()
    cases = root / "tools" / "vv" / "comsol" / "cases"
    v1_path = cases / "m3c2_caseP_100nm_event_tolerance_pre_final_v1.json"
    v2_path = cases / "m3c2_caseP_100nm_event_tolerance_pre_final_v2.json"
    v1 = cast(dict[str, Any], json.loads(v1_path.read_text(encoding="utf-8")))
    v2 = cast(dict[str, Any], json.loads(v2_path.read_text(encoding="utf-8")))
    supersedes = cast(dict[str, Any], v2.pop("supersedes"))

    v1["recipe_revision"] = 2
    v1["evaluator"] = v2["evaluator"]

    assert v1 == v2
    assert supersedes["sha256"] == _sha256(v1_path)
    evaluator = root.parents[1] / cast(dict[str, str], v2["evaluator"])["path"]
    assert _sha256(evaluator) == cast(dict[str, str], v2["evaluator"])["sha256"]
