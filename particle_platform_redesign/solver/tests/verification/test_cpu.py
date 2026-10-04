"""Verification of the bounded resident CPU execution plan."""

from chamber_particles.cpu import (
    early_memory_requirement_bytes,
    plan_cpu_memory,
)


def test_early_memory_gate_keeps_prepare_and_run_phase_peaks_separate() -> None:
    """An early lower bound must not reject a feasible final microtile plan."""

    particle_count = 100_000
    limit_bytes = 50 * 1024 * 1024
    canonical_data_bytes = 10_000_104
    schedule_bytes = particle_count * 120
    physics_runtime_bytes = particle_count * 16

    early_bytes = early_memory_requirement_bytes(
        canonical_data_bytes=canonical_data_bytes,
        particle_count=particle_count,
        requires_stage_evaluation=False,
        writer_reserve_bytes=3 * 1024 * 1024,
    )
    plan = plan_cpu_memory(
        limit_bytes=limit_bytes,
        particle_count=particle_count,
        canonical_data_bytes=canonical_data_bytes,
        prepared_geometry_bytes=0,
        particle_schedule_bytes=schedule_bytes,
        physics_runtime_bytes=physics_runtime_bytes,
        probe_index_bytes=0,
        output_buffer_bytes=0,
        writer_reserve_bytes=3 * 1024 * 1024,
        requires_stage_evaluation=False,
    )

    assert early_bytes <= plan.planned_bytes <= limit_bytes
    assert 0 < plan.slab_particles < particle_count


def test_stage_evaluated_early_gate_remains_a_lower_bound() -> None:
    """Model-specific runtime bytes belong to the authoritative final plan."""

    particle_count = 1_000_000
    limit_bytes = 300 * 1024 * 1024
    canonical_data_bytes = 1_000_000

    early_bytes = early_memory_requirement_bytes(
        canonical_data_bytes=canonical_data_bytes,
        particle_count=particle_count,
        requires_stage_evaluation=True,
        writer_reserve_bytes=3 * 1024 * 1024,
    )
    plan = plan_cpu_memory(
        limit_bytes=limit_bytes,
        particle_count=particle_count,
        canonical_data_bytes=canonical_data_bytes,
        prepared_geometry_bytes=1_000,
        particle_schedule_bytes=particle_count * 120,
        physics_runtime_bytes=particle_count * 32,
        probe_index_bytes=0,
        output_buffer_bytes=0,
        writer_reserve_bytes=3 * 1024 * 1024,
        requires_stage_evaluation=True,
    )

    assert early_bytes <= plan.planned_bytes <= limit_bytes
    assert 0 < plan.slab_particles < particle_count


def test_memory_plan_owns_one_bounded_slab() -> None:
    plan = plan_cpu_memory(
        limit_bytes=128 * 1024 * 1024,
        particle_count=100,
        canonical_data_bytes=1_000,
        prepared_geometry_bytes=2_000,
        particle_schedule_bytes=12_000,
        physics_runtime_bytes=3_200,
        probe_index_bytes=0,
        output_buffer_bytes=0,
        writer_reserve_bytes=3 * 1024 * 1024,
        requires_stage_evaluation=True,
        geometry_query_scratch_bytes=3 * 256 * 8,
        replay_work_bytes=2_048,
        dense_path_bytes_per_particle=176,
        event_work_bytes_per_particle=64,
        certificate_work_bytes_per_particle=1_320,
        release_work_bytes_per_particle=96,
        event_candidate_capacity=256,
        event_staging_capacity=128,
        event_staging_bytes_per_row=490,
        event_staging_fixed_bytes=2 * 256 * 8 + 16,
        failure_staging_bytes_per_particle=52,
    )

    manifest = plan.as_manifest()
    components = manifest["components"]
    assert isinstance(components, dict)
    assert plan.slab_particles == manifest["slab_particles"] == 100
    assert components["slab_proposal_scratch"] == (
        plan.slab_particles * plan.scratch_bytes_per_particle
    )
    assert components["slab_event_work"] == (
        plan.slab_particles * plan.event_work_bytes_per_particle
    )
    assert components["slab_dense_path"] == (
        plan.slab_particles * plan.dense_path_bytes_per_particle
    )
    assert plan.dense_path_bytes_per_particle == 176
    assert plan.event_work_bytes_per_particle == 64
    assert components["slab_certificate_work"] == (
        plan.slab_particles * plan.certificate_work_bytes_per_particle
    )
    assert plan.certificate_work_bytes_per_particle == 1_320
    assert components["slab_release_work"] == (
        plan.slab_particles * plan.release_work_bytes_per_particle
    )
    assert plan.release_work_bytes_per_particle == 96
    assert plan.event_candidate_capacity == manifest["event_candidate_capacity"] == 256
    assert plan.event_staging_capacity == manifest["event_staging_capacity"] == 128
    assert plan.event_staging_bytes_per_row == 490
    assert plan.event_staging_fixed_bytes == 2 * plan.event_candidate_capacity * 8 + 16
    assert plan.failure_staging_bytes_per_particle == 52
    assert components["slab_event_staging"] == (
        plan.event_staging_fixed_bytes + plan.slab_particles * plan.event_staging_bytes_per_row
    )
    assert components["slab_failure_staging"] == (
        plan.slab_particles * plan.failure_staging_bytes_per_particle
    )
    assert components["geometry_query_scratch"] == 3 * plan.event_candidate_capacity * 8
    assert plan.replay_work_bytes == components["replay_work"] == 2_048
    assert plan.run_peak_bytes == sum(components.values()) <= plan.limit_bytes


def test_memory_plan_counts_spatial_index_build_work_only_in_prepare_peak() -> None:
    baseline = plan_cpu_memory(
        limit_bytes=128 * 1024 * 1024,
        particle_count=100,
        canonical_data_bytes=1_000,
        prepared_geometry_bytes=2_000,
        particle_schedule_bytes=12_000,
        physics_runtime_bytes=3_200,
        field_runtime_bytes=4_000,
        probe_index_bytes=0,
        output_buffer_bytes=0,
        writer_reserve_bytes=0,
        requires_stage_evaluation=False,
    )
    geometry_transient_bytes = 51_200
    field_transient_bytes = 25_600
    indexed = plan_cpu_memory(
        limit_bytes=128 * 1024 * 1024,
        particle_count=100,
        canonical_data_bytes=1_000,
        prepared_geometry_bytes=2_000,
        particle_schedule_bytes=12_000,
        physics_runtime_bytes=3_200,
        field_runtime_bytes=4_000,
        geometry_preparation_transient_bytes=geometry_transient_bytes,
        field_preparation_transient_bytes=field_transient_bytes,
        probe_index_bytes=0,
        output_buffer_bytes=0,
        writer_reserve_bytes=0,
        requires_stage_evaluation=False,
    )

    assert indexed.prepare_peak_bytes > baseline.prepare_peak_bytes
    assert indexed.run_peak_bytes == baseline.run_peak_bytes
    assert indexed.geometry_preparation_transient_bytes == geometry_transient_bytes
    assert indexed.field_preparation_transient_bytes == field_transient_bytes
    assert indexed.as_manifest()["geometry_preparation_transient_bytes"] == (
        geometry_transient_bytes
    )
    assert indexed.as_manifest()["field_preparation_transient_bytes"] == field_transient_bytes


def test_memory_plan_reports_when_even_one_slab_row_does_not_fit() -> None:
    arguments = {
        "limit_bytes": 65_536,
        "particle_count": 10,
        "canonical_data_bytes": 0,
        "prepared_geometry_bytes": 0,
        "particle_schedule_bytes": 0,
        "physics_runtime_bytes": 0,
        "probe_index_bytes": 0,
        "output_buffer_bytes": 0,
        "writer_reserve_bytes": 0,
        "requires_stage_evaluation": False,
    }

    plan = plan_cpu_memory(**arguments)

    assert plan.slab_particles == 0
