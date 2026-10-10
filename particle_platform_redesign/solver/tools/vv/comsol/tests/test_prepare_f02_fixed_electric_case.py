from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from tests.verification.microcases import build_microcase
from tools.vv.comsol.prepare_f02_fixed_electric_case import prepare_f02_fixed_electric_case

from chamber_particles.case_format import RealizedSurfaceSource, read, write
from chamber_particles.geometry import prepare_geometry


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
    output_path = tmp_path / "fields-with-source.h5"
    _write_field_bundle(input_path)
    input_bundle = read(input_path)

    report = prepare_f02_fixed_electric_case(input_path, output_path)
    output = read(output_path)

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


def test_f02_fixture_publishes_report_exclusively(tmp_path: Path) -> None:
    input_path = tmp_path / "fields.h5"
    output_path = tmp_path / "fields-with-source.h5"
    report_path = tmp_path / "source-report.json"
    _write_field_bundle(input_path)

    report = prepare_f02_fixed_electric_case(
        input_path,
        output_path,
        report_path=report_path,
    )

    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    second_output = tmp_path / "second-output.h5"
    with pytest.raises(FileExistsError, match=r"source-report\.json"):
        prepare_f02_fixed_electric_case(
            input_path,
            second_output,
            report_path=report_path,
        )
    assert not second_output.exists()


def test_f02_fixture_rejects_resolved_path_collisions_before_writing(tmp_path: Path) -> None:
    input_path = tmp_path / "fields.h5"
    missing_input_path = tmp_path / "missing-fields.h5"
    output_path = tmp_path / "fields-with-source.h5"
    alias_directory = tmp_path / "alias"
    alias_directory.mkdir()
    _write_field_bundle(input_path)
    original_input = input_path.read_bytes()

    with pytest.raises(FileNotFoundError, match=r"missing-fields\.h5"):
        prepare_f02_fixed_electric_case(missing_input_path, output_path)
    with pytest.raises(ValueError, match="must be different"):
        prepare_f02_fixed_electric_case(input_path, input_path)
    with pytest.raises(ValueError, match="must be different"):
        prepare_f02_fixed_electric_case(
            input_path,
            output_path,
            report_path=alias_directory / ".." / input_path.name,
        )
    with pytest.raises(ValueError, match="must be different"):
        prepare_f02_fixed_electric_case(
            input_path,
            output_path,
            report_path=output_path,
        )

    assert input_path.read_bytes() == original_input
    assert not output_path.exists()

    output_path.write_bytes(b"existing-output")
    with pytest.raises(FileExistsError, match=r"fields-with-source\.h5"):
        prepare_f02_fixed_electric_case(input_path, output_path)
    assert output_path.read_bytes() == b"existing-output"
