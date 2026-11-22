from __future__ import annotations

import multiprocessing
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
import yaml

from chamber_particles import CaseError, check_case, load_case
from chamber_particles.case_format import RealizedTableSource
from tests.verification.microcases import materialize_microcase


def test_public_check_prepares_without_output_workers_or_particle_updates(tmp_path: Path) -> None:
    paths = materialize_microcase("C01", tmp_path / "case")
    case = load_case(paths.case_path)
    source = case.data.sources[0]
    assert isinstance(source, RealizedTableSource)
    initial_position = source.position_m.copy()
    files_before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    children_before = {child.pid for child in multiprocessing.active_children()}

    report = check_case(case)

    assert report.status == "valid"
    assert report.particle_count == source.particle_id.size
    assert report.data_content_hash == case.content_hash
    assert report.case_file_hash == case.case_file_hash
    assert report.memory_limit_bytes >= report.planned_memory_bytes > 0
    assert report.workers == 1
    assert report.models[0].category == "charge"
    assert report.models[0].model == "fixed"
    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == files_before
    assert {child.pid for child in multiprocessing.active_children()} == children_before
    np.testing.assert_array_equal(source.position_m, initial_position)
    with pytest.raises(FrozenInstanceError):
        type(report).__setattr__(report, "particle_count", 0)


def test_public_check_rejects_a_statically_valid_unsupported_execution_request(
    tmp_path: Path,
) -> None:
    paths = materialize_microcase("C01", tmp_path / "case")
    document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
    document["solver"]["integrator"] = "unsupported_integrator"
    paths.case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    case = load_case(paths.case_path)

    with pytest.raises(CaseError, match="integrator"):
        check_case(case)


@pytest.mark.parametrize("workers", [0, -1, True])
def test_public_check_rejects_invalid_execution_resources(tmp_path: Path, workers: int) -> None:
    paths = materialize_microcase("C01", tmp_path / "case")
    with pytest.raises(CaseError, match="workers must be a positive integer"):
        check_case(load_case(paths.case_path), workers=workers)
