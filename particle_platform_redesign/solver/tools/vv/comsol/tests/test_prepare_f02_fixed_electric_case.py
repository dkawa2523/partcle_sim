from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from tests.verification.microcases import build_microcase
from tools.vv.comsol.prepare_f02_fixed_electric_case import prepare_f02_fixed_electric_case

from chamber_particles import check_case, open_result, simulate
from chamber_particles.case_format import RealizedSurfaceSource, read, write
from chamber_particles.geometry import prepare_geometry
from chamber_particles_tools.inputs import create_case


def _write_field_bundle(input_path: Path) -> None:
    base = build_microcase("C07").data
    geometry = replace(
        base.geometry,
        nodes_m=base.geometry.nodes_m + np.asarray([1.0, 0.0]),
        group_names=("wafer",),
    )
    write(
        input_path,
        replace(
            base,
            coordinate_system="axisymmetric_rz",
            geometry=geometry,
            sources=(),
        ),
    )


def test_f02_fixture_writes_explicit_equal_area_surface_rows(tmp_path: Path) -> None:
    input_path = tmp_path / "fields.h5"
    output_path = tmp_path / "fields-with-source"
    _write_field_bundle(input_path)
    input_bundle = read(input_path)

    report = prepare_f02_fixed_electric_case(input_path, output_path)
    output = read(output_path / "bundle.h5")

    assert report["particle_count"] == 32
    assert len(output.sources) == 1
    source = output.sources[0]
    assert isinstance(source, RealizedSurfaceSource)
    np.testing.assert_array_equal(source.particle_id, np.arange(1000, 1032))
    assert bool(((0.0 < source.facet_parameter) & (source.facet_parameter < 1.0)).all())
    prepared = prepare_geometry(output.geometry, output.coordinate_system)
    np.testing.assert_array_less(
        np.sum(source.velocity_m_s * prepared.facet_normal[source.facet_id], axis=1),
        np.zeros(32),
    )
    provenance = json.loads(output.provenance_json)
    assert provenance["producer"] == "tools.vv.comsol.prepare_f02_fixed_electric_case"
    assert provenance["producer_version"] == "f02_realized_wafer_schedule_v1"
    assert provenance["source_sha256"] == report["input_content_hash"]
    assert (
        provenance["field_semantics_revision"]
        == json.loads(input_bundle.provenance_json)["field_semantics_revision"]
    )
    metadata = provenance["producer_metadata"]
    assert (
        metadata["input_provenance_sha256"]
        == "sha256:" + hashlib.sha256(input_bundle.provenance_json.encode("utf-8")).hexdigest()
    )
    assert metadata["source_realization"]["quantile_rule"] == "equal_area_midpoint"


def test_f02_fixture_publishes_bundle_and_report_together(tmp_path: Path) -> None:
    input_path = tmp_path / "fields.h5"
    output = tmp_path / "fields-with-source"
    _write_field_bundle(input_path)

    report = prepare_f02_fixed_electric_case(input_path, output)
    assert json.loads((output / "report.json").read_text(encoding="utf-8")) == report
    assert report["output_path"] == str((output / "bundle.h5").resolve())
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(FileExistsError):
        prepare_f02_fixed_electric_case(input_path, output)
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before


def test_f02_report_write_failure_does_not_publish_and_can_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path, output = tmp_path / "fields.h5", tmp_path / "prepared"
    _write_field_bundle(input_path)
    original = input_path.read_bytes()
    write_text = Path.write_text

    def fail_report(path: Path, *args, **kwargs):
        if path.name == "report.json":
            raise OSError("injected report write failure")
        return write_text(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "write_text", fail_report)
        with pytest.raises(OSError, match="injected report"):
            prepare_f02_fixed_electric_case(input_path, output)
    assert not output.exists()
    assert input_path.read_bytes() == original
    prepare_f02_fixed_electric_case(input_path, output)
    assert read(output / "bundle.h5").sources
    assert (output / "report.json").is_file()


def test_f02_fixture_rejects_resolved_path_collisions_before_writing(tmp_path: Path) -> None:
    input_path, output = tmp_path / "fields.h5", tmp_path / "prepared"
    _write_field_bundle(input_path)
    original = input_path.read_bytes()
    with pytest.raises(FileNotFoundError):
        prepare_f02_fixed_electric_case(tmp_path / "missing-fields.h5", output)
    with pytest.raises(ValueError, match="must be different"):
        prepare_f02_fixed_electric_case(input_path, input_path)
    assert input_path.read_bytes() == original
    assert not output.exists()


def test_f02_published_bundle_can_build_and_run_a_checked_public_case(tmp_path: Path) -> None:
    input_path, output = tmp_path / "fields.h5", tmp_path / "prepared"
    _write_field_bundle(input_path)
    prepare_f02_fixed_electric_case(input_path, output)
    settings = copy.deepcopy(build_microcase("C07").spec)
    settings["motion"] = {"mode": "axisymmetric_rz_meridional"}
    settings["time"] = {"start_s": 0.0, "end_s": 1e-3, "dt_s": 1e-4}
    settings["sources"] = [
        {"name": "wafer_surface_release", "type": "surface", "table": "wafer_surface_particles"}
    ]
    settings["boundaries"] = [{"boundary_group": "wafer", "priority": 10, "law": "stick"}]
    settings["output"] = {
        "trajectories": {"selection": "all", "schedule": {"explicit_times_s": [0.0, 1e-3]}},
        "probes": None,
    }
    case = create_case(read(output / "bundle.h5"), settings, tmp_path / "case")
    check_case(case)
    summary = simulate(case, tmp_path / "result")
    result = open_result(tmp_path / "result")
    assert summary.particle_count == 32 and summary.failure_event_count == 0
    np.testing.assert_array_equal(result.read_final().particle_id, np.arange(1000, 1032))
    assert list(result.iter_frames())[-1].time_s == 1e-3
