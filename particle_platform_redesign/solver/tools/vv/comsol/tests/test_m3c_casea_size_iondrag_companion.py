from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml
from tools.vv.comsol import prepare_m3c_casea_companion as companion_module
from tools.vv.comsol.prepare_m3c_casea_companion import prepare
from tools.vv.comsol.tests.common_p1_fixture import (
    sha256,
    write_current_common_p1_template,
    write_manufactured_common_p1,
)

from chamber_particles.case_format import read_with_info

TOOLS = Path(__file__).resolve().parents[1]
SHARED_JAVA = TOOLS / "comsol" / "RunM3C1CaseA100CommonP1.java"
SHARED_RUNNER = TOOLS / "run_m3c1_common_p1_reference.ps1"
CAMPAIGN_RUNNER = TOOLS / "run_m3c_casea_size_iondrag_companion.ps1"
CAMPAIGN_CONFIG = TOOLS / "cases" / "m3c_casea_size_iondrag_companion_v1.json"
REPOSITORY_ROOT = Path(__file__).resolve().parents[6]


def test_campaign_accepts_only_the_three_explicit_case_tuples() -> None:
    java = SHARED_JAVA.read_text(encoding="utf-8")
    powershell = SHARED_RUNNER.read_text(encoding="utf-8")
    expected = (
        "caseA_10nm_relative_flow",
        "caseA_30nm_relative_flow",
        "caseA_100nm_image",
        "relative_flow_screened_collection_orbital_aggregate_ion_v1",
        "electric_field_directed_image_orbital_sensitivity_v1",
        "relative_flow_screened_collection_orbital_ion_drag",
        "electric_field_directed_image_orbital_ion_drag",
    )
    for value in expected:
        assert value in java
    assert "System.getenv" not in java
    assert "FileInputStream" not in java
    assert "run_spec.properties" not in java
    assert "static void runCampaign(" in java
    assert "campaignSpec(caseId, diameterNm, revision, contribution)" in java
    assert "Read-CampaignRunSpec $StagedRunSpec" in powershell
    assert "New-CampaignEntrySource $RunSpec" in powershell
    assert "Unsupported common-P1 campaign case tuple" in powershell
    assert "[System.IO.File]::WriteAllText(" in powershell
    assert "SetEnvironmentVariable(" not in powershell
    assert "campaign_run_spec = $(" in powershell


def test_image_ion_drag_is_built_only_from_common_p1_primitives() -> None:
    source = SHARED_JAVA.read_text(encoding="utf-8")
    image_block = source.split("private static final String IMAGE_EPSILON", 1)[1].split(
        "private static final String[] STATE_COLUMNS", 1
    )[0]
    assert '"8.8541878128e-12[F/m]"' in image_block
    assert '"sqrt(m3c1_uir(r,z)^2+m3c1_uiz(r,z)^2)"' in image_block
    assert '"max(d0/2,m3c1_lambdaD(r,z))"' in image_block
    assert "m3c1_Te(r,z)" in image_block
    assert "m3c1_TiV(r,z)" in image_block
    assert "m3c1_ni(r,z)" in image_block
    assert "m3c1_mi(r,z)" in image_block
    assert "ZAS" in image_block
    assert "(1[m/s])^2" in image_block
    assert "(1[V/m])^2" in image_block
    assert "AS_ui_mag" not in source
    assert "fptas.vr" not in image_block
    assert "fptas.vz" not in image_block


def test_campaign_reuses_the_legacy_runner_without_changing_legacy_entry_points() -> None:
    shared = SHARED_JAVA.read_text(encoding="utf-8")
    runner = SHARED_RUNNER.read_text(encoding="utf-8")
    wrapper = CAMPAIGN_RUNNER.read_text(encoding="utf-8")
    assert "run(PRE_EVENT_PROFILE);" in shared
    assert "run(MATERIAL_EVENT_PROFILE);" in shared
    assert 'RunM3C1CaseA100CommonP1.runCampaign("caseA_10nm_relative_flow"' in runner
    assert 'RunM3C1CaseA100CommonP1.runCampaign("caseA_30nm_relative_flow"' in runner
    assert 'RunM3C1CaseA100CommonP1.runCampaign("caseA_100nm_image"' in runner
    assert "CaseSpec" not in shared
    assert "CaseSpec" not in runner
    assert "-RunProfile campaign" in wrapper
    assert "-CandidateInput" not in wrapper
    assert '"reference_run_config.json"' in wrapper
    assert '"run_spec.properties"' in wrapper
    assert (
        "ModelUtil"
        not in runner.split("function New-CampaignEntrySource", 1)[1].split("$SolverRoot", 1)[0]
    )
    assert "ModelUtil" not in wrapper


def test_campaign_runner_keeps_the_fixed_three_step_common_p1_schedule() -> None:
    java = SHARED_JAVA.read_text(encoding="utf-8")
    runner = SHARED_RUNNER.read_text(encoding="utf-8")
    assert "private static final int[] PRE_EVENT_STEP_CODES = {625, 3125, 15625};" in java
    assert '"range(0[s],1e-5[s],4.5e-4[s])"' in java
    assert '@("dt_0p625us", "dt_0p3125us", "dt_0p15625us")' in runner
    assert "time_end_s" not in CAMPAIGN_RUNNER.read_text(encoding="utf-8")
    assert "fixed_rk4_steps_s" not in CAMPAIGN_RUNNER.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def prepared_campaign(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, dict[str, object]]:
    output = tmp_path_factory.mktemp("casea-companion") / "prepared"
    repository = output.parent / "repository"
    repository.mkdir()
    config = json.loads(CAMPAIGN_CONFIG.read_text(encoding="utf-8"))
    input_path = repository / "input.h5"
    info = write_manufactured_common_p1(input_path)
    template = repository / "template.yaml"
    write_current_common_p1_template(
        REPOSITORY_ROOT / config["candidate_template"]["relative_path"], template, info.content_hash
    )
    source = repository / "identity_only_source.mph"
    source.write_bytes(b"manufactured source identity fixture; never opened by COMSOL")
    config["shared_field_input"] = {
        "relative_path": input_path.name,
        "sha256": sha256(input_path),
        "content_hash": info.content_hash,
    }
    config["candidate_template"] = {"relative_path": template.name, "sha256": sha256(template)}
    config["source_models"]["theory_common"] = {
        "relative_path": source.name,
        "sha256": sha256(source),
    }
    for case in config["cases"]:
        release = case["release_table"]
        release_path = repository / f"{case['case_id']}_release.csv"
        shutil.copyfile(REPOSITORY_ROOT / release["relative_path"], release_path)
        release.update(relative_path=release_path.name, sha256=sha256(release_path))
    config_path = repository / "campaign.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(companion_module, "_repository_root", lambda: repository)
        report = prepare(config_path, output)
    return output, report


def test_python_preparer_records_one_primitive_field_authority(
    prepared_campaign: tuple[Path, dict[str, object]],
) -> None:
    output, report = prepared_campaign

    assert report["status"] == "PREPARED"
    assert report["background_field_tables"] == "NOT_READ"
    authority = report["shared_field_authority"]
    assert isinstance(authority, dict)
    input_path = output.parent / "repository" / "input.h5"
    assert authority["sha256"] == sha256(input_path)
    assert authority["content_hash"] == read_with_info(input_path)[1].content_hash
    assert authority["field_count"] == 17
    assert authority["component_count"] == 22


def test_python_preparer_reuses_identical_primitive_arrays(
    prepared_campaign: tuple[Path, dict[str, object]],
) -> None:
    output, _report = prepared_campaign
    bundles = [
        read_with_info(output / "candidate_inputs" / f"{case_id}.h5")[0]
        for case_id in (
            "caseA_10nm_relative_flow",
            "caseA_30nm_relative_flow",
            "caseA_100nm_image",
        )
    ]
    authority = {field.name: field.values for field in bundles[0].fields}
    for bundle in bundles[1:]:
        for field in bundle.fields:
            np.testing.assert_array_equal(field.values, authority[field.name])


@pytest.mark.parametrize(
    ("case_id", "diameter_nm", "ion_model"),
    [
        ("caseA_10nm_relative_flow", 10, "screened_collection_orbital"),
        ("caseA_30nm_relative_flow", 30, "screened_collection_orbital"),
        ("caseA_100nm_image", 100, "image_orbital_sensitivity"),
    ],
)
def test_python_preparer_emits_one_owned_size_model_cell(
    prepared_campaign: tuple[Path, dict[str, object]],
    case_id: str,
    diameter_nm: int,
    ion_model: str,
) -> None:
    output, report = prepared_campaign
    cases = report["cases"]
    assert isinstance(cases, dict)
    case_root = output / "cases" / case_id
    bundle, _info = read_with_info(output / "candidate_inputs" / f"{case_id}.h5")
    np.testing.assert_array_equal(
        bundle.sources[0].drag_diameter_m,
        np.full(287, diameter_nm * 1.0e-9),
    )
    spec_lines = (case_root / "run_spec.properties").read_text(encoding="ascii").splitlines()
    assert [line.split("=", 1)[0] for line in spec_lines] == [
        "case_id",
        "diameter_nm",
        "ion_drag_revision",
        "deterministic_contribution_name",
    ]
    reference = json.loads((case_root / "reference_run_config.json").read_text(encoding="utf-8"))
    assert reference["source_model"]["sha256"] == sha256(
        output.parent / "repository" / "identity_only_source.mph"
    )
    assert reference["case"]["diameter_m"] == diameter_nm * 1.0e-9
    candidate = yaml.safe_load((case_root / "candidate_fine.yaml").read_text(encoding="utf-8"))
    assert candidate["physics"]["ion_drag"]["model"] == ion_model
    assert candidate["physics"]["drag"]["maximum_speed_ratio"] == 1.0
    assert candidate["physics"]["thermophoresis"]["maximum_speed_ratio"] == 1.0
    assert (
        candidate["physics"]["dielectrophoresis"]["maximum_point_dipole_radius_m"]
        == 0.5 * diameter_nm * 1.0e-9
    )
    case_report = cases[case_id]
    assert isinstance(case_report, dict)
    release = case_report["release"]
    assert isinstance(release, dict)
    assert release["derived_particle_columns_copied"] == []
    ignored = release["ignored_column_count"]
    assert isinstance(ignored, int) and ignored > 0
