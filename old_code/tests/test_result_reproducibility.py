from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import particle_tracer_unified as particle_tracer
import particle_tracer_unified.application as application
import particle_tracer_unified.writer as result_writer
from particle_tracer_unified.configuration import load_run_config
from particle_tracer_unified.io._runtime_adapter import resolve_adapter_inputs
from particle_tracer_unified.io._runtime_context import assemble_solver_context

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = REPO_ROOT / "examples" / "v02_minimal" / "run_config.yaml"


@pytest.fixture(scope="module")
def standard_result():
    return particle_tracer.simulate(particle_tracer.load_case(EXAMPLE))


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_standard_summary_records_resolved_execution_metadata(
    tmp_path: Path,
    standard_result,
) -> None:
    output = tmp_path / "result"
    particle_tracer.write_result(standard_result, output)

    assert sorted(path.name for path in output.iterdir()) == [
        "final_particles.csv",
        "run_summary.json",
        "wall_summary.csv",
    ]
    summary = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
    execution = summary["execution"]
    assert execution["adapter"] == "native"
    assert execution["dt_s"] == pytest.approx(0.01)
    assert execution["t_end_s"] == pytest.approx(0.05)
    assert execution["rng_seed"] == 12345
    assert execution["stochastic_seed"] == 12345
    assert execution["forces"] == [
        {"name": "drag", "model": "stokes", "parameters": {}}
    ]
    assert execution["gas"]["dynamic_viscosity_Pas"] == pytest.approx(1.8e-5)
    numerics = execution["numerics"]
    boundary = numerics.pop("boundary")
    assert numerics == {
        "adaptive_substep_enabled": True,
        "adaptive_substep_max_splits": 4,
        "boundary_broad_phase_enabled": False,
        "integrator": "etd2",
        "max_wall_hits_per_step": 5,
        "policy_version": "etd2-affine-lte-v3",
    }
    assert boundary["policy_version"] == "geometry-scaled-float64-v1"
    assert boundary["reference_length_m"] == pytest.approx(2.0)
    assert boundary["resolution_length_m"] == pytest.approx(0.05)
    assert boundary["classification_tolerance_m"] == pytest.approx(5.0e-12)
    assert boundary["contact_offset_m"] == pytest.approx(5.0e-10)
    assert boundary["radial_axis_tolerance_m"] == pytest.approx(
        boundary["classification_tolerance_m"]
    )
    assert boundary["coordinate_roundoff_m"] > 0.0
    assert execution["software"]["package"] == "particle-tracer-unified"
    assert execution["software"]["package_version"] == particle_tracer.__version__

    provenance = execution["provenance"]
    assert provenance["config"]["sha256"] == _digest(EXAMPLE)
    assert len(provenance["config"]["canonical_sha256"]) == 64
    for logical_name, filename in (
        ("particles", "particles.csv"),
        ("boundaries", "boundaries.csv"),
    ):
        source = EXAMPLE.parent / filename
        assert provenance["inputs"][logical_name]["sha256"] == _digest(source)
        assert provenance["inputs"][logical_name]["size_bytes"] == source.stat().st_size


def test_execution_metadata_is_resolved_once_when_loading_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = application._build_execution_metadata

    def counted(case):
        nonlocal calls
        calls += 1
        return original(case)

    monkeypatch.setattr(application, "_build_execution_metadata", counted)
    case = particle_tracer.load_case(EXAMPLE)
    first = particle_tracer.simulate(case)
    second = particle_tracer.simulate(case)

    assert calls == 1
    assert first.execution_metadata is second.execution_metadata


def test_loaded_case_rejects_input_and_provenance_mutation() -> None:
    case = particle_tracer.load_case(EXAMPLE)
    context = case.solver_context
    field = context.field_provider.field
    quantity = next(iter(field.quantities.values()))
    arrays = (
        context.particles.position,
        context.geometry_provider.geometry.axes[0],
        context.geometry_provider.geometry.valid_mask,
        quantity.times,
        quantity.data,
    )

    assert all(not value.flags.writeable for value in arrays)
    for value in arrays:
        with pytest.raises(ValueError, match="read-only"):
            value.flat[0] = value.flat[0]

    for mapping in (
        context.particles.metadata,
        context.geometry_provider.geometry.metadata,
        field.metadata,
        field.quantities,
        case._provenance,
        case._provenance["inputs"],
        case._execution,
        case._execution["gas"],
    ):
        with pytest.raises(TypeError):
            cast(Any, mapping)["mutation"] = True

    bounds = context.geometry_provider.geometry.metadata["bounds"]
    assert bounds == [-1.0, 1.0, -1.0, 1.0]
    with pytest.raises(TypeError, match="read-only"):
        cast(Any, bounds).append(2.0)

    result = particle_tracer.simulate(case)
    assert result.stats.particle_count == context.particles.count
    assert result.execution_metadata["provenance"]["config"]["sha256"] == _digest(
        EXAMPLE
    )


def test_simulation_result_mappings_are_read_only(standard_result) -> None:
    mappings = (
        standard_result.stats.timing_s,
        standard_result.stats.memory_estimate_bytes,
        standard_result.stats.terminal_counts,
        standard_result.stats.wall_outcome_counts,
        standard_result.stats.safety_counters,
        standard_result.wall_summary,
        standard_result.execution_metadata,
        standard_result.debug,
    )

    for mapping in mappings:
        with pytest.raises(TypeError):
            cast(Any, mapping)["mutation"] = True


def test_solver_context_owns_adapter_arrays_and_metadata() -> None:
    config = load_run_config(EXAMPLE)
    adapter = resolve_adapter_inputs(config, EXAMPLE.parent)
    source_particles = adapter.runtime_inputs.particles
    source_geometry = adapter.providers.geometry_provider
    source_field = adapter.providers.field_provider
    assert source_geometry is not None
    assert source_field is not None

    source_quantity = next(iter(source_field.field.quantities.values()))
    position_before = source_particles.position.copy()
    axis_before = source_geometry.geometry.axes[0].copy()
    quantity_before = source_quantity.data.copy()
    bounds_before = list(source_geometry.geometry.metadata["bounds"])
    edge_count_before = source_geometry.geometry.metadata["boundary_edge_topology"][
        "edge_count"
    ]
    context = assemble_solver_context(config, adapter)

    source_particles.position.flat[0] += 1.0
    source_geometry.geometry.axes[0].flat[0] += 1.0
    source_quantity.data.flat[0] += 1.0
    source_particles.metadata["mutation"] = True
    source_geometry.geometry.metadata["mutation"] = True
    source_field.field.metadata["mutation"] = True
    cast(Any, source_geometry.geometry.metadata["bounds"])[0] += 1.0
    cast(Any, source_geometry.geometry.metadata["boundary_edge_topology"])[
        "edge_count"
    ] = -1

    np.testing.assert_array_equal(context.particles.position, position_before)
    np.testing.assert_array_equal(
        context.geometry_provider.geometry.axes[0], axis_before
    )
    context_quantity = next(iter(context.field_provider.field.quantities.values()))
    np.testing.assert_array_equal(context_quantity.data, quantity_before)
    assert "mutation" not in context.particles.metadata
    assert "mutation" not in context.geometry_provider.geometry.metadata
    assert "mutation" not in context.field_provider.field.metadata
    assert context.geometry_provider.geometry.metadata["bounds"] == bounds_before
    assert (
        context.geometry_provider.geometry.metadata["boundary_edge_topology"][
            "edge_count"
        ]
        == edge_count_before
    )
    assert not np.shares_memory(context.particles.position, source_particles.position)
    assert not np.shares_memory(
        context.geometry_provider.geometry.axes[0],
        source_geometry.geometry.axes[0],
    )
    assert not np.shares_memory(context_quantity.data, source_quantity.data)


def test_writer_never_overwrites_existing_expected_artifact(
    tmp_path: Path,
    standard_result,
) -> None:
    output = tmp_path / "existing-result"
    output.mkdir()
    existing = output / "final_particles.csv"
    existing.write_text("do-not-overwrite\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="immutable result output already exists"):
        particle_tracer.write_result(standard_result, output)

    assert existing.read_text(encoding="utf-8") == "do-not-overwrite\n"
    assert sorted(path.name for path in output.iterdir()) == ["final_particles.csv"]


@pytest.mark.parametrize("precreate_empty_output", [False, True])
def test_writer_failure_does_not_publish_partial_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    standard_result,
    precreate_empty_output: bool,
) -> None:
    output = tmp_path / "atomic-result"
    if precreate_empty_output:
        output.mkdir()

    def fail_summary_write(_path: Path, _value) -> None:
        raise RuntimeError("injected writer failure")

    monkeypatch.setattr(result_writer, "_write_json", fail_summary_write)
    with pytest.raises(RuntimeError, match="injected writer failure"):
        particle_tracer.write_result(standard_result, output)

    if precreate_empty_output:
        assert output.is_dir()
        assert not list(output.iterdir())
    else:
        assert not output.exists()
    assert not list(tmp_path.glob(".atomic-result.staging-*"))
