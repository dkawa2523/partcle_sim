from __future__ import annotations

import json
from dataclasses import fields as dataclass_fields
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from chamber_particles.output import (
    RESULT_ALGORITHM_REVISION,
    RESULT_SCHEMA_VERSION,
    BoundaryEvents,
    CheckpointState,
    FailureEvents,
    FinalParticles,
    LifecycleSeries,
    ProbeFrame,
    ResultWriteError,
    ResultWriter,
    open_result_store,
)


def _final_particles() -> FinalParticles:
    return FinalParticles(
        particle_id=np.asarray([10, 20, 30], dtype="<i8"),
        source_id=np.asarray([0, 1, 2], dtype="<i4"),
        time_s=np.asarray([1.0, 1.0, 1.0], dtype="<f8"),
        position_m=np.asarray([[0.5, 0.25], [0.75, 0.5], [0.25, 0.75]], dtype="<f8"),
        velocity_m_s=np.asarray([[0.1, 0.0], [0.2, -0.1], [0.0, 0.0]], dtype="<f8"),
        charge_number=np.asarray([1.0, 4.0, -2.0], dtype="<f8"),
        lifecycle=np.asarray([1, 5, 4], dtype="<u1"),
        kinematics_valid=np.asarray([1, 1, 0], dtype="<u1"),
        failure_reason_code=np.asarray([0, 0, 3], dtype="<u2"),
        mass_kg=np.asarray([1.0e-18, 1.5e-18, 2.0e-18], dtype="<f8"),
        drag_diameter_m=np.asarray([1.0e-7, 1.5e-7, 2.0e-7], dtype="<f8"),
        contact_radius_m=np.asarray([0.0, 2.5e-8, 7.5e-8], dtype="<f8"),
        electrostatic_radius_m=np.asarray([5.0e-8, 7.5e-8, 1.0e-7], dtype="<f8"),
        displaced_volume_m3=np.asarray([5.0e-22, 1.5e-21, 4.0e-21], dtype="<f8"),
        model_weight=np.asarray([1.0, 1.5, 2.0], dtype="<f8"),
        material_id=np.asarray([2, 4, 3], dtype="<i4"),
    )


def _checkpoint_state(*, macro_step_count: int) -> CheckpointState:
    final = _final_particles()
    return CheckpointState(
        macro_time_s=1.0,
        macro_step_count=macro_step_count,
        release_cursor=3,
        frame_cursor=0,
        probe_cursor=1,
        position_m=final.position_m.copy(),
        velocity_m_s=final.velocity_m_s.copy(),
        charge_number=final.charge_number.copy(),
        lifecycle=final.lifecycle.copy(),
        failure_reason_code=final.failure_reason_code.copy(),
        terminal_time_s=np.asarray([np.inf, 0.75, 0.5], dtype="<f8"),
        event_ordinal=np.asarray([0, 1, 1], dtype="<u4"),
        physical_boundary_event_ordinal=np.zeros(3, dtype="<u4"),
        exact_origin_time_s=np.zeros(3, dtype="<f8"),
        exact_origin_position_m=final.position_m.copy(),
        exact_origin_velocity_m_s=final.velocity_m_s.copy(),
        start_contact_state=np.zeros(3, dtype="<u1"),
        active_particle_index=np.asarray([0], dtype="<i8"),
        last_field_cell=None,
        accepted_particle_pieces=4,
        candidate_queries=0,
        refinements=0,
        maximum_refinement_depth=0,
        wall_interactions=0,
        residual_splits=0,
        axis_crossings=0,
    )


def _boundary_events(
    time_s: list[float],
    particle_id: list[int],
    event_ordinal: list[int],
    primary_facet_id: list[int],
    candidate_rows: list[list[int]],
    *,
    interaction_kind: list[str] | None = None,
    destination_facet_id: list[int] | None = None,
    position_m: list[list[float]] | None = None,
    position_post_m: list[list[float]] | None = None,
    law_id: list[str] | None = None,
    outcome: list[str] | None = None,
) -> BoundaryEvents:
    count = len(time_s)
    offsets = np.zeros(count + 1, dtype="<i8")
    offsets[1:] = np.cumsum([len(row) for row in candidate_rows], dtype="<i8")
    positions = np.asarray(
        np.zeros((count, 2), dtype="<f8") if position_m is None else position_m,
        dtype="<f8",
    )
    return BoundaryEvents(
        time_s=np.asarray(time_s, dtype="<f8"),
        particle_id=np.asarray(particle_id, dtype="<i8"),
        event_ordinal=np.asarray(event_ordinal, dtype="<u4"),
        interaction_kind=np.asarray(
            ["wall"] * count if interaction_kind is None else interaction_kind,
            dtype=np.str_,
        ),
        primary_facet_id=np.asarray(primary_facet_id, dtype="<i8"),
        destination_facet_id=np.asarray(
            [-1] * count if destination_facet_id is None else destination_facet_id,
            dtype="<i8",
        ),
        boundary_id=np.full(count, 10, dtype="<i4"),
        material_id=np.zeros(count, dtype="<i4"),
        contact_radius_m=np.asarray(primary_facet_id, dtype="<f8") * 1.0e-10,
        position_m=positions,
        position_post_m=np.asarray(
            positions.copy() if position_post_m is None else position_post_m,
            dtype="<f8",
        ),
        normal=np.tile(np.asarray([[1.0, 0.0]], dtype="<f8"), (count, 1)),
        velocity_pre_m_s=np.zeros((count, 2), dtype="<f8"),
        velocity_post_m_s=np.zeros((count, 2), dtype="<f8"),
        charge_number_pre=np.zeros(count, dtype="<f8"),
        charge_number_post=np.zeros(count, dtype="<f8"),
        model_weight=np.ones(count, dtype="<f8"),
        law_id=np.asarray(["specular"] * count if law_id is None else law_id, dtype=np.str_),
        outcome=np.asarray(["reflected"] * count if outcome is None else outcome, dtype=np.str_),
        localization_residual_m=np.zeros(count, dtype="<f8"),
        position_budget_m=np.full(count, 1.0e-9, dtype="<f8"),
        time_budget_s=np.full(count, 1.0e-9, dtype="<f8"),
        candidate_offset=offsets,
        candidate_facet_id=np.asarray(
            [facet for row in candidate_rows for facet in row], dtype="<i8"
        ),
    )


def test_boundary_batches_rebase_ragged_candidates_and_bulk_reader_sorts(tmp_path: Path) -> None:
    output = tmp_path / "ragged-boundary-result"
    first = _boundary_events(
        [2.0, 1.0],
        [20, 10],
        [2, 1],
        [4, 8],
        [[3, 4], [8]],
    )
    second = _boundary_events([0.5], [10], [0], [6], [[5, 6, 7]])
    with ResultWriter(
        output,
        particle_capacity=3,
        resume_identity={"case": "ragged-boundary-test"},
    ) as writer:
        writer.begin_epoch()
        writer.write_boundary_events(first)
        writer.write_boundary_events(second)
        writer.commit_epoch(_checkpoint_state(macro_step_count=1))
        writer.finalize(_final_particles(), {}, macro_step_count=1)

    result = open_result_store(output)
    batches = list(result.iter_boundary_event_batches(batch_rows=1))
    assert [float(batch.time_s[0]) for batch in batches] == [2.0, 1.0, 0.5]
    np.testing.assert_allclose(
        [float(batch.contact_radius_m[0]) for batch in batches],
        [4.0e-10, 8.0e-10, 6.0e-10],
    )
    assert [batch.candidate_facet_id.tolist() for batch in batches] == [[3, 4], [8], [5, 6, 7]]
    assert all(
        batch.candidate_offset.tolist() == [0, batch.candidate_facet_id.size] for batch in batches
    )

    bulk = result.read_boundary_events()
    np.testing.assert_array_equal(bulk.time_s, [0.5, 1.0, 2.0])
    np.testing.assert_array_equal(bulk.interaction_kind, ["wall", "wall", "wall"])
    np.testing.assert_array_equal(bulk.primary_facet_id, [6, 8, 4])
    np.testing.assert_array_equal(bulk.destination_facet_id, [-1, -1, -1])
    np.testing.assert_allclose(bulk.contact_radius_m, [6.0e-10, 8.0e-10, 4.0e-10])
    np.testing.assert_array_equal(bulk.position_post_m, bulk.position_m)
    np.testing.assert_array_equal(bulk.candidate_offset, [0, 3, 4, 6])
    np.testing.assert_array_equal(bulk.candidate_facet_id, [5, 6, 7, 8, 3, 4])


def test_periodic_translation_boundary_event_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "periodic-boundary-result"
    events = _boundary_events(
        [0.25, 0.5],
        [10, 20],
        [1, 2],
        [4, 6],
        [[4], [6, 7]],
        interaction_kind=["wall", "periodic_translation"],
        destination_facet_id=[-1, 12],
        position_m=[[0.0, 0.25], [1.0, 0.75]],
        position_post_m=[[0.0, 0.25], [0.0, 0.75]],
        law_id=["specular", ""],
        outcome=["reflected", "transferred"],
    )
    with ResultWriter(
        output,
        particle_capacity=3,
        resume_identity={"case": "periodic-output-test"},
    ) as writer:
        writer.begin_epoch()
        writer.write_boundary_events(events)
        writer.commit_epoch(_checkpoint_state(macro_step_count=1))
        writer.finalize(_final_particles(), {}, macro_step_count=1)

    actual = open_result_store(output).read_boundary_events()
    np.testing.assert_array_equal(actual.interaction_kind, ["wall", "periodic_translation"])
    np.testing.assert_array_equal(actual.destination_facet_id, [-1, 12])
    np.testing.assert_array_equal(actual.position_m, [[0.0, 0.25], [1.0, 0.75]])
    np.testing.assert_array_equal(actual.position_post_m, [[0.0, 0.25], [0.0, 0.75]])
    assert not actual.interaction_kind.flags.writeable
    assert not actual.destination_facet_id.flags.writeable
    assert not actual.position_post_m.flags.writeable


def test_extended_result_columns_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "result"
    with ResultWriter(
        output,
        particle_capacity=3,
        resume_identity={"case": "output-test"},
    ) as writer:
        writer.begin_epoch()
        writer.write_failure_events(
            FailureEvents(
                time_s=np.asarray([0.5], dtype="<f8"),
                particle_id=np.asarray([30], dtype="<i8"),
                event_ordinal=np.asarray([1], dtype="<u4"),
                reason_code=np.asarray([3], dtype="<u2"),
            )
        )
        writer.write_lifecycle_series(
            LifecycleSeries(
                time_s=np.asarray([0.0, 0.5, 1.0], dtype="<f8"),
                pending=np.asarray([2, 0, 0], dtype="<u8"),
                active=np.asarray([0, 1, 1], dtype="<u8"),
                stuck=np.asarray([0, 0, 0], dtype="<u8"),
                held=np.asarray([0, 1, 1], dtype="<u8"),
                escaped=np.asarray([0, 0, 0], dtype="<u8"),
                failed=np.asarray([0, 1, 1], dtype="<u8"),
            )
        )
        writer.write_probe(
            ProbeFrame(
                time_s=0.25,
                particle_id=np.asarray([10], dtype="<i8"),
                position_m=np.asarray([[0.25, 0.25]], dtype="<f8"),
                velocity_m_s=np.asarray([[0.1, 0.0]], dtype="<f8"),
                charge_number=np.asarray([1.0], dtype="<f8"),
                lifecycle=np.asarray([1], dtype="<u1"),
            )
        )
        writer.commit_epoch(_checkpoint_state(macro_step_count=4))
        summary = writer.finalize(
            _final_particles(), {"case_name": "output-test"}, macro_step_count=4
        )

    assert summary.failure_event_count == 1
    assert summary.series_count == 3
    assert summary.probe_count == 1
    assert summary.probe_row_count == 1

    manifest = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert manifest["result_schema_version"] == RESULT_SCHEMA_VERSION == 3
    assert manifest["result_algorithm_revision"] == RESULT_ALGORITHM_REVISION
    assert "epoch_macro_steps" not in manifest
    assert manifest["counts"]["failure_events"] == 1
    assert manifest["counts"]["series"] == 3
    assert manifest["counts"]["probes"] == 1
    assert manifest["counts"]["probe_rows"] == 1

    result = open_result_store(output)
    failures = result.read_failure_events()
    np.testing.assert_array_equal(failures.particle_id, [30])
    np.testing.assert_array_equal(failures.reason_code, [3])
    assert not failures.reason_code.flags.writeable

    series = result.read_lifecycle_series()
    np.testing.assert_array_equal(series.time_s, [0.0, 0.5, 1.0])
    np.testing.assert_array_equal(series.failed, [0, 1, 1])
    np.testing.assert_array_equal(series.held, [0, 1, 1])
    assert not series.failed.flags.writeable

    probes = list(result.iter_probes())
    assert len(probes) == 1
    assert probes[0].time_s == 0.25
    np.testing.assert_array_equal(probes[0].particle_id, [10])
    np.testing.assert_array_equal(probes[0].position_m, [[0.25, 0.25]])
    assert not probes[0].position_m.flags.writeable

    final = result.read_final()
    np.testing.assert_array_equal(final.lifecycle, [1, 5, 4])
    np.testing.assert_array_equal(final.failure_reason_code, [0, 0, 3])
    np.testing.assert_allclose(final.contact_radius_m, [0.0, 2.5e-8, 7.5e-8])
    assert final.failure_reason_code.dtype == np.dtype("<u2")


def test_failure_reason_codes_are_nonzero_and_match_failed_final_state(tmp_path: Path) -> None:
    with ResultWriter(
        tmp_path / "failure-result",
        particle_capacity=3,
        resume_identity={"case": "failure-test"},
    ) as writer:
        writer.begin_epoch()
        with pytest.raises(ResultWriteError, match="nonzero uint16"):
            writer.write_failure_events(
                FailureEvents(
                    time_s=np.asarray([0.5], dtype="<f8"),
                    particle_id=np.asarray([20], dtype="<i8"),
                    event_ordinal=np.asarray([1], dtype="<u4"),
                    reason_code=np.asarray([0], dtype="<u2"),
                )
            )

        invalid_final = _final_particles()
        invalid_final.failure_reason_code[0] = np.uint16(7)
        with pytest.raises(ResultWriteError, match="nonzero only for failed"):
            writer.finalize(invalid_final, {}, macro_step_count=1)


def test_result_contact_radius_must_be_finite_and_nonnegative(tmp_path: Path) -> None:
    with ResultWriter(
        tmp_path / "contact-radius-result",
        particle_capacity=3,
        resume_identity={"case": "contact-radius-test"},
    ) as writer:
        writer.begin_epoch()
        invalid_events = _boundary_events([0.5], [10], [1], [4], [[4]])
        invalid_events.contact_radius_m[0] = -1.0
        with pytest.raises(ResultWriteError, match="contact_radius_m"):
            writer.write_boundary_events(invalid_events)

        invalid_final = _final_particles()
        invalid_final.contact_radius_m[0] = np.nan
        with pytest.raises(ResultWriteError, match="floating-point columns"):
            writer.finalize(invalid_final, {}, macro_step_count=1)


@pytest.mark.parametrize("damage", ("shortened", "negative", "duplicate", "reversed"))
def test_final_publication_requires_capacity_and_canonical_particle_ids(
    tmp_path: Path, damage: str
) -> None:
    output = tmp_path / f"invalid-final-{damage}"
    final = _final_particles()
    if damage == "shortened":
        final = replace(
            final,
            **{field.name: getattr(final, field.name)[:-1] for field in dataclass_fields(final)},
        )
    else:
        ids = {"negative": [-1, 20, 30], "duplicate": [10, 10, 30], "reversed": [30, 20, 10]}
        final = replace(final, particle_id=np.asarray(ids[damage], dtype="<i8"))
    with ResultWriter(
        output, particle_capacity=3, resume_identity={"case": "invalid-final"}
    ) as writer:
        writer.begin_epoch()
        writer.commit_epoch(_checkpoint_state(macro_step_count=1))
        with pytest.raises(ResultWriteError):
            writer.finalize(final, {}, macro_step_count=1)
    assert not output.exists()
    assert not (output.with_name(f"{output.name}.partial") / "_SUCCESS").exists()


def test_final_ids_accept_sparse_values_up_to_int64_limit(tmp_path: Path) -> None:
    output = tmp_path / "sparse-final"
    final = replace(
        _final_particles(),
        particle_id=np.asarray([0, 2, np.iinfo(np.int64).max], dtype="<i8"),
    )
    with ResultWriter(
        output, particle_capacity=3, resume_identity={"case": "sparse-final"}
    ) as writer:
        writer.begin_epoch()
        writer.commit_epoch(_checkpoint_state(macro_step_count=1))
        writer.finalize(final, {}, macro_step_count=1)
    np.testing.assert_array_equal(
        open_result_store(output).read_final().particle_id, final.particle_id
    )


@pytest.mark.parametrize(
    ("interaction_kind", "destination_facet_id", "position_post_m", "law_id", "outcome"),
    [
        ("unknown", -1, [[0.0, 0.0]], ["specular"], ["reflected"]),
        ("periodic_translation", -1, [[1.0, 0.0]], [""], ["transferred"]),
        ("wall", -1, [[1.0, 0.0]], ["specular"], ["reflected"]),
        ("periodic_translation", 8, [[1.0, 0.0]], ["specular"], ["transferred"]),
    ],
)
def test_boundary_interaction_requires_canonical_encoding(
    tmp_path: Path,
    interaction_kind: str,
    destination_facet_id: int,
    position_post_m: list[list[float]],
    law_id: list[str],
    outcome: list[str],
) -> None:
    events = _boundary_events(
        [0.5],
        [10],
        [1],
        [4],
        [[4]],
        interaction_kind=[interaction_kind],
        destination_facet_id=[destination_facet_id],
        position_post_m=position_post_m,
        law_id=law_id,
        outcome=outcome,
    )
    with ResultWriter(
        tmp_path / f"invalid-{interaction_kind}-{destination_facet_id}-{law_id[0]}",
        particle_capacity=1,
        resume_identity={"case": "invalid-boundary-interaction"},
    ) as writer:
        writer.begin_epoch()
        with pytest.raises(ResultWriteError, match="interaction columns"):
            writer.write_boundary_events(events)
