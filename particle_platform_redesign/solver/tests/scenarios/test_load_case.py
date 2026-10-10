from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from chamber_particles import CaseError, load_case
from chamber_particles.case_format import (
    BoundaryData,
    DataBundle,
    GeometryData,
    RealizedSurfaceSource,
    RealizedTableSource,
    write,
)


def _bundle() -> DataBundle:
    boundary = BoundaryData(
        line2=np.asarray([[0, 1], [1, 2], [2, 0]], dtype="<i8"),
        boundary_id=np.asarray([10, 11, 12], dtype="<i4"),
        group_id=np.asarray([0, 1, 0], dtype="<i4"),
        material_id=np.asarray([4, 5, 4], dtype="<i4"),
        owner_cell_type=np.asarray([1, 1, 1], dtype="<u1"),
        owner_cell_local_index=np.asarray([0, 0, 0], dtype="<i8"),
        orientation=np.asarray([1, 1, 1], dtype="<i1"),
    )
    geometry = GeometryData(
        nodes_m=np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype="<f8"),
        boundary=boundary,
        group_names=("wall", "outlet"),
        tri3=np.asarray([[0, 1, 2]], dtype="<i8"),
        tri3_domain_id=np.asarray([0], dtype="<i4"),
    )
    source = RealizedTableSource(
        name="releases",
        particle_id=np.asarray([42], dtype="<i8"),
        release_time_s=np.asarray([0.25], dtype="<f8"),
        position_m=np.asarray([[0.25, 0.25]], dtype="<f8"),
        velocity_m_s=np.asarray([[0.1, 0.0]], dtype="<f8"),
        charge_number=np.asarray([0.0], dtype="<f8"),
        mass_kg=np.asarray([1.0e-18], dtype="<f8"),
        drag_diameter_m=np.asarray([1.0e-7], dtype="<f8"),
        contact_radius_m=np.asarray([0.0], dtype="<f8"),
        electrostatic_radius_m=np.asarray([5.0e-8], dtype="<f8"),
        displaced_volume_m3=np.asarray([5.0e-22], dtype="<f8"),
        model_weight=np.asarray([1.0], dtype="<f8"),
        material_id=np.asarray([2], dtype="<i4"),
    )
    provenance = json.dumps(
        {
            "producer": "scenario",
            "producer_version": "1.0",
            "source_sha256": f"sha256:{'0' * 64}",
            "field_semantics_revision": "primitive-v1",
            "producer_metadata": {},
        }
    )
    return DataBundle("cartesian_xy", provenance, geometry, sources=(source,))


def _case_yaml(data_path: str, expected_hash: str, *, complete_boundaries: bool = True) -> str:
    outlet = (
        "\n  - boundary_group: outlet\n    priority: 20\n    law: escape"
        if complete_boundaries
        else ""
    )
    return f"""\
format_version: 3
case:
  name: public_load_case
  data_path: {data_path}
  expected_content_hash: {expected_hash}
motion:
  mode: cartesian_xy
time:
  start_s: 0.0
  end_s: 1.0
  dt_s: 0.1
solver:
  integrator: rk4_fixed
  backend: cpu
  seed: 1234
  event:
    geometry_rtol: 1.0e-12
    roundoff_ulps: 64
    max_refinements: 48
    max_interactions_per_step: 8
    corner_policy: priority_then_combined_normal_v1
resources:
  memory_limit_mb: 128
physics:
  charge:
    model: fixed
sources:
  - name: input_particles
    type: table
    table: releases
boundaries:
  - boundary_group: wall
    priority: 10
    law: stick{outlet}
output:
  trajectories: null
"""


def _surface_source_yaml(name: str, particle_id_start: int, count: int) -> str:
    return f"""\
  - name: {name}
    type: surface
    boundary_group: wall
    count: {count}
    particle_id_start: {particle_id_start}
    particle:
      charge_number: 0.0
      mass_kg: 1.0e-18
      drag_diameter_m: 1.0e-7
      electrostatic_radius_m: 5.0e-8
      displaced_volume_m3: 0.0
      model_weight: 1.0
      material_id: 0
    position: {{model: uniform, measure: line_length}}
    velocity: {{model: fixed, value_m_s: [0.0, 0.0]}}
    release: {{model: fixed, time_s: 0.0}}
"""


def test_load_case_resolves_data_relative_to_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_dir = tmp_path / "case"
    data_dir = case_dir / "data"
    data_dir.mkdir(parents=True)
    data_path = data_dir / "case.h5"
    info = write(data_path, _bundle())
    yaml_path = case_dir / "run.yaml"
    yaml_path.write_text(
        _case_yaml("data/case.h5", info.content_hash),
        encoding="utf-8",
    )
    unrelated_directory = tmp_path / "elsewhere"
    unrelated_directory.mkdir()
    monkeypatch.chdir(unrelated_directory)

    case = load_case(yaml_path)

    assert case.data_path == data_path.resolve()
    assert case.content_hash == info.content_hash
    assert case.case_file_hash.startswith("sha256:")
    assert case.spec.name == "public_load_case"
    assert case.spec.motion.mode == "cartesian_xy"
    assert case.spec.output.probes is None
    assert case.data_footprint == info.footprint
    assert case.data_footprint.numeric_array_bytes > 0


@pytest.mark.parametrize(
    ("replacement", "error"),
    [
        ("  start_s: 0.0\n  start_s: 0.1", "duplicate YAML key"),
        ("  <<: {start_s: 0.0}\n  start_s: 0.0", "merge keys"),
        (f"  start_s: {10**400}", "time.start_s must be finite"),
    ],
)
def test_load_case_rejects_ambiguous_or_unrepresentable_yaml(
    tmp_path: Path, replacement: str, error: str
) -> None:
    yaml_path = tmp_path / "invalid.yaml"
    yaml_path.write_text(
        _case_yaml("missing.h5", f"sha256:{'0' * 64}").replace("  start_s: 0.0", replacement),
        encoding="utf-8",
    )
    with pytest.raises(CaseError, match=error):
        load_case(yaml_path)


def test_load_case_accepts_scalar_aliases_and_hashes_original_yaml(tmp_path: Path) -> None:
    data_path = tmp_path / "case.h5"
    info = write(data_path, _bundle())
    document = (
        _case_yaml("case.h5", info.content_hash)
        .replace("  start_s: 0.0", "  start_s: &origin 0.0")
        .replace("  dt_s: 0.1", "  dt_s: *origin")
    )
    # A valid alias still receives the domain's positive time-step validation.
    yaml_path = tmp_path / "aliased.yaml"
    yaml_path.write_text(document, encoding="utf-8")
    with pytest.raises(CaseError, match=r"time\.dt_s must be positive"):
        load_case(yaml_path)
    yaml_path.write_text(document.replace("&origin 0.0", "&origin 0.1"), encoding="utf-8")
    case = load_case(yaml_path)
    assert case.spec.time.start_s == case.spec.time.dt_s == 0.1
    assert case.case_file_hash == "sha256:" + hashlib.sha256(yaml_path.read_bytes()).hexdigest()


def test_load_case_reports_invalid_utf8_as_case_error(tmp_path: Path) -> None:
    yaml_path = tmp_path / "invalid-encoding.yaml"
    yaml_path.write_bytes(b"format_version: 3\ncase: \xff")
    with pytest.raises(CaseError, match="not valid UTF-8"):
        load_case(yaml_path)


def test_load_case_separates_periodic_topology_from_material_wall_laws(tmp_path: Path) -> None:
    data_path = tmp_path / "case.h5"
    info = write(data_path, _bundle())
    document = yaml.safe_load(_case_yaml("case.h5", info.content_hash))
    document["boundaries"] = []
    document["topology"] = {
        "model": "translation_periodic_xy_v1",
        "field_match_rtol": 1.0e-12,
        "pairs": [
            {
                "first_boundary_group": "wall",
                "second_boundary_group": "outlet",
                "first_to_second_m": [1.0, 0.0],
            }
        ],
    }
    yaml_path = tmp_path / "periodic.yaml"
    yaml_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    case = load_case(yaml_path)

    assert case.spec.topology is not None
    assert case.spec.topology.model == "translation_periodic_xy_v1"
    assert case.spec.topology.pairs[0].first_to_second_m == (1.0, 0.0)
    assert case.spec.boundaries == ()

    document["boundaries"] = [{"boundary_group": "wall", "priority": 10, "law": "stick"}]
    conflict_path = tmp_path / "periodic-wall-conflict.yaml"
    conflict_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    with pytest.raises(CaseError, match="boundary laws do not match data groups"):
        load_case(conflict_path)


def test_load_case_rejects_unsupported_periodic_coordinates_and_reused_groups(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "case.h5"
    info = write(data_path, _bundle())
    document = yaml.safe_load(_case_yaml("case.h5", info.content_hash))
    document["boundaries"] = []
    pair = {
        "first_boundary_group": "wall",
        "second_boundary_group": "outlet",
        "first_to_second_m": [1.0, 0.0],
    }
    document["topology"] = {
        "model": "translation_periodic_xy_v1",
        "field_match_rtol": 1.0e-12,
        "pairs": [pair],
    }
    document["motion"]["mode"] = "axisymmetric_rz_meridional"
    coordinate_path = tmp_path / "periodic-rz.yaml"
    coordinate_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    with pytest.raises(CaseError, match="requires Cartesian XY"):
        load_case(coordinate_path)

    document["motion"]["mode"] = "cartesian_xy"
    document["topology"]["pairs"] = [pair, pair]
    duplicate_path = tmp_path / "periodic-duplicate.yaml"
    duplicate_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    with pytest.raises(CaseError, match="used more than once"):
        load_case(duplicate_path)


def test_load_case_rejects_surface_release_from_periodic_seam(tmp_path: Path) -> None:
    bundle = _bundle()
    table = bundle.sources[0]
    assert isinstance(table, RealizedTableSource)
    surface = RealizedSurfaceSource(
        name="releases",
        particle_id=table.particle_id,
        release_time_s=table.release_time_s,
        facet_id=np.asarray([0], dtype="<i8"),
        facet_parameter=np.asarray([0.5], dtype="<f8"),
        velocity_m_s=table.velocity_m_s,
        charge_number=table.charge_number,
        mass_kg=table.mass_kg,
        drag_diameter_m=table.drag_diameter_m,
        contact_radius_m=table.contact_radius_m,
        electrostatic_radius_m=table.electrostatic_radius_m,
        displaced_volume_m3=table.displaced_volume_m3,
        model_weight=table.model_weight,
        material_id=table.material_id,
    )
    data_path = tmp_path / "case.h5"
    info = write(data_path, replace(bundle, sources=(surface,)))
    document = yaml.safe_load(_case_yaml("case.h5", info.content_hash))
    document["sources"][0]["type"] = "surface"
    document["boundaries"] = []
    document["topology"] = {
        "model": "translation_periodic_xy_v1",
        "field_match_rtol": 1.0e-12,
        "pairs": [
            {
                "first_boundary_group": "wall",
                "second_boundary_group": "outlet",
                "first_to_second_m": [1.0, 0.0],
            }
        ],
    }
    yaml_path = tmp_path / "periodic-surface-source.yaml"
    yaml_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(CaseError, match="cannot release from a periodic seam"):
        load_case(yaml_path)


def test_load_case_rejects_retired_v1_thread_configuration(tmp_path: Path) -> None:
    """The serial v3 schema must not silently accept the removed thread API."""

    data_path = tmp_path / "case.h5"
    info = write(data_path, _bundle())
    canonical = _case_yaml("case.h5", info.content_hash)

    v1_path = tmp_path / "v1.yaml"
    v1_path.write_text(
        canonical.replace("format_version: 3", "format_version: 1"), encoding="utf-8"
    )
    with pytest.raises(CaseError, match="unsupported YAML format_version"):
        load_case(v1_path)

    threaded_path = tmp_path / "threaded-v2.yaml"
    threaded_path.write_text(
        canonical.replace("resources:\n", "resources:\n  threads: 2\n"),
        encoding="utf-8",
    )
    with pytest.raises(CaseError, match="resources keys are invalid"):
        load_case(threaded_path)


def test_load_case_rejects_oversize_numeric_arrays_before_materializing_them(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "case.h5"
    info = write(data_path, _bundle())
    with h5py.File(data_path, "r+") as handle:
        source = handle["sources/releases"]
        del source["position_m"]
        source.create_dataset(
            "position_m",
            shape=(100_000, 2),
            dtype="<f8",
            chunks=(1024, 2),
        )
    yaml_path = tmp_path / "run.yaml"
    yaml_path.write_text(
        _case_yaml("case.h5", info.content_hash).replace(
            "memory_limit_mb: 128", "memory_limit_mb: 1"
        ),
        encoding="utf-8",
    )

    with pytest.raises(CaseError, match=r"canonical numeric arrays.*resources\.memory_limit_mb"):
        load_case(yaml_path)


def test_load_case_rejects_hash_mismatch(tmp_path: Path) -> None:
    data_path = tmp_path / "case.h5"
    info = write(data_path, _bundle())
    stale_hash = f"sha256:{'f' * 64}"
    assert stale_hash != info.content_hash
    yaml_path = tmp_path / "run.yaml"
    yaml_path.write_text(_case_yaml("case.h5", stale_hash), encoding="utf-8")

    with pytest.raises(CaseError):
        load_case(yaml_path)


def test_load_case_rejects_missing_data(tmp_path: Path) -> None:
    yaml_path = tmp_path / "run.yaml"
    yaml_path.write_text(
        _case_yaml("missing.h5", f"sha256:{'0' * 64}"),
        encoding="utf-8",
    )

    with pytest.raises(CaseError):
        load_case(yaml_path)


def test_load_case_requires_one_law_for_every_boundary_group(tmp_path: Path) -> None:
    data_path = tmp_path / "case.h5"
    info = write(data_path, _bundle())
    yaml_path = tmp_path / "run.yaml"
    yaml_path.write_text(
        _case_yaml("case.h5", info.content_hash, complete_boundaries=False),
        encoding="utf-8",
    )

    with pytest.raises(CaseError):
        load_case(yaml_path)


def test_fixed_charge_uses_each_source_initial_value(tmp_path: Path) -> None:
    data_path = tmp_path / "case.h5"
    bundle = _bundle()
    bundle.sources[0].charge_number[0] = -3.0
    info = write(data_path, bundle)
    yaml_path = tmp_path / "run.yaml"
    yaml_path.write_text(_case_yaml("case.h5", info.content_hash), encoding="utf-8")

    case = load_case(yaml_path)

    assert case.spec.physics.models["charge"] == {"model": "fixed"}
    assert case.data.sources[0].charge_number.tolist() == [-3.0]


def test_fixed_charge_rejects_a_second_global_initial_value(tmp_path: Path) -> None:
    data_path = tmp_path / "case.h5"
    info = write(data_path, _bundle())
    yaml_path = tmp_path / "run.yaml"
    yaml_path.write_text(
        _case_yaml("case.h5", info.content_hash).replace(
            "    model: fixed\n", "    model: fixed\n    charge_number: 0\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(CaseError):
        load_case(yaml_path)


def test_typed_trajectory_schedule_is_strict_and_normalized(tmp_path: Path) -> None:
    data_path = tmp_path / "case.h5"
    info = write(data_path, _bundle())
    yaml_path = tmp_path / "run.yaml"
    document = yaml.safe_load(_case_yaml("case.h5", info.content_hash))
    document["output"]["trajectories"] = {
        "selection": "all",
        "schedule": {"explicit_times_s": [0, 0.25, 1.0]},
    }
    yaml_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    case = load_case(yaml_path)

    assert case.spec.output.trajectories is not None
    assert case.spec.output.trajectories.selection == "all"
    assert case.spec.output.trajectories.explicit_times_s == (0.0, 0.25, 1.0)


@pytest.mark.parametrize(
    "trajectories",
    [
        {"selection": "all", "schedule": {"times_s": [0.0]}},
        {"selection": "all", "schedule": {"explicit_times_s": [0.25, 0.25]}},
        {"selection": "all", "schedule": {"explicit_times_s": [-0.1, 0.5]}},
        {"selection": "ids", "schedule": {"explicit_times_s": [0.5]}},
    ],
    ids=("old-key", "duplicate", "outside-interval", "unsupported-selection"),
)
def test_invalid_trajectory_schedule_is_rejected(
    tmp_path: Path, trajectories: dict[str, object]
) -> None:
    data_path = tmp_path / "case.h5"
    info = write(data_path, _bundle())
    document = yaml.safe_load(_case_yaml("case.h5", info.content_hash))
    document["output"]["trajectories"] = trajectories
    yaml_path = tmp_path / "run.yaml"
    yaml_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(CaseError):
        load_case(yaml_path)


def test_typed_probe_schedule_is_strict_and_normalized(tmp_path: Path) -> None:
    data_path = tmp_path / "case.h5"
    info = write(data_path, _bundle())
    yaml_path = tmp_path / "run.yaml"
    document = yaml.safe_load(_case_yaml("case.h5", info.content_hash))
    document["output"]["probes"] = {
        "particle_ids": [17, 42],
        "schedule": {"explicit_times_s": [0, 0.25, 1.0]},
    }
    yaml_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    case = load_case(yaml_path)

    assert case.spec.output.probes is not None
    assert case.spec.output.probes.particle_ids == (17, 42)
    assert case.spec.output.probes.explicit_times_s == (0.0, 0.25, 1.0)


def test_null_probe_schedule_is_disabled(tmp_path: Path) -> None:
    data_path = tmp_path / "case.h5"
    info = write(data_path, _bundle())
    yaml_path = tmp_path / "run.yaml"
    document = yaml.safe_load(_case_yaml("case.h5", info.content_hash))
    document["output"]["probes"] = None
    yaml_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    case = load_case(yaml_path)

    assert case.spec.output.probes is None


@pytest.mark.parametrize(
    "probes",
    [
        {"particle_ids": [], "schedule": {"explicit_times_s": [0.5]}},
        {"particle_ids": [42, 42], "schedule": {"explicit_times_s": [0.5]}},
        {"particle_ids": [43, 42], "schedule": {"explicit_times_s": [0.5]}},
        {"particle_ids": [-1], "schedule": {"explicit_times_s": [0.5]}},
        {"particle_ids": [2**63], "schedule": {"explicit_times_s": [0.5]}},
        {"particle_ids": [42], "schedule": {"explicit_times_s": [0.5, 0.5]}},
        {"particle_ids": [42], "schedule": {"explicit_times_s": [1.1]}},
        {"particle_ids": [42], "schedule": {"times_s": [0.5]}},
    ],
    ids=(
        "empty-ids",
        "duplicate-ids",
        "unsorted-ids",
        "negative-id",
        "id-overflow",
        "duplicate-time",
        "outside-interval",
        "old-time-key",
    ),
)
def test_invalid_probe_schedule_is_rejected(tmp_path: Path, probes: dict[str, object]) -> None:
    data_path = tmp_path / "case.h5"
    info = write(data_path, _bundle())
    document = yaml.safe_load(_case_yaml("case.h5", info.content_hash))
    document["output"]["probes"] = probes
    yaml_path = tmp_path / "run.yaml"
    yaml_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(CaseError):
        load_case(yaml_path)


def test_old_optional_result_flags_and_missing_motion_are_rejected(tmp_path: Path) -> None:
    data_path = tmp_path / "case.h5"
    info = write(data_path, _bundle())
    document = yaml.safe_load(_case_yaml("case.h5", info.content_hash))
    document["output"]["events"] = True
    with_flags = tmp_path / "with-flags.yaml"
    with_flags.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    with pytest.raises(CaseError):
        load_case(with_flags)

    document = yaml.safe_load(_case_yaml("case.h5", info.content_hash))
    del document["motion"]
    without_motion = tmp_path / "without-motion.yaml"
    without_motion.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    with pytest.raises(CaseError):
        load_case(without_motion)


def test_negative_release_time_is_valid_inside_a_negative_run_interval(tmp_path: Path) -> None:
    bundle = _bundle()
    source = replace(bundle.sources[0], release_time_s=np.asarray([-0.25], dtype="<f8"))
    data_path = tmp_path / "case.h5"
    info = write(data_path, replace(bundle, sources=(source,)))
    document = _case_yaml("case.h5", info.content_hash).replace("  start_s: 0.0", "  start_s: -0.5")
    yaml_path = tmp_path / "run.yaml"
    yaml_path.write_text(document, encoding="utf-8")

    case = load_case(yaml_path)

    np.testing.assert_array_equal(case.data.sources[0].release_time_s, [-0.25])


def test_retired_surface_distribution_keys_are_rejected(tmp_path: Path) -> None:
    data_path = tmp_path / "case.h5"
    info = write(data_path, _bundle())
    table_source = """\
sources:
  - name: input_particles
    type: table
    table: releases
"""
    surface_source = "sources:\n" + _surface_source_yaml("wall_release", 42, 1)
    yaml_path = tmp_path / "run.yaml"
    document = _case_yaml("case.h5", info.content_hash).replace(table_source, surface_source)
    yaml_path.write_text(document, encoding="utf-8")

    with pytest.raises(CaseError, match=r"sources\[0\] keys are invalid"):
        load_case(yaml_path)


@pytest.mark.parametrize(
    "declared_type",
    ["table", "surface"],
)
def test_source_type_must_match_the_canonical_schedule(tmp_path: Path, declared_type: str) -> None:
    bundle = _bundle()
    internal = bundle.sources[0]
    if declared_type == "surface":
        actual = internal
    else:
        actual = RealizedSurfaceSource(
            name="releases",
            particle_id=internal.particle_id.copy(),
            release_time_s=internal.release_time_s.copy(),
            facet_id=np.asarray([0], dtype="<i8"),
            facet_parameter=np.asarray([0.5], dtype="<f8"),
            velocity_m_s=internal.velocity_m_s.copy(),
            charge_number=internal.charge_number.copy(),
            mass_kg=internal.mass_kg.copy(),
            drag_diameter_m=internal.drag_diameter_m.copy(),
            contact_radius_m=internal.contact_radius_m.copy(),
            electrostatic_radius_m=internal.electrostatic_radius_m.copy(),
            displaced_volume_m3=internal.displaced_volume_m3.copy(),
            model_weight=internal.model_weight.copy(),
            material_id=internal.material_id.copy(),
        )
    data_path = tmp_path / "case.h5"
    info = write(data_path, replace(bundle, sources=(actual,)))
    table_source = """sources:
  - name: input_particles
    type: table
    table: releases
"""
    declared = table_source.replace("type: table", f"type: {declared_type}")
    yaml_path = tmp_path / "run.yaml"
    document = _case_yaml("case.h5", info.content_hash).replace(table_source, declared)
    yaml_path.write_text(document, encoding="utf-8")

    with pytest.raises(CaseError, match="type does not match canonical table"):
        load_case(yaml_path)
