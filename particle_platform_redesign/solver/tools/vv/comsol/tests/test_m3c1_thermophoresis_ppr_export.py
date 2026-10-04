from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

TOOL_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = TOOL_ROOT.parents[4]
CONFIG_PATH = TOOL_ROOT / "cases" / "m3c1_caseA_100nm_thermophoresis_ppr_v1.json"
JAVA_PATH = TOOL_ROOT / "comsol" / "RunM3C1CaseA100ThermophoresisPpr.java"
RUNNER_PATH = TOOL_ROOT / "run_m3c1_caseA_100nm_thermophoresis_ppr.ps1"
SOURCE_SETTINGS_PATH = (
    REPOSITORY_ROOT
    / "model_dataset"
    / "cf4_o2_etch_caseA_nonlinear_sass"
    / "cases"
    / "formal_iondrag_theory_consistent"
    / "caseA_100nm"
    / "external_reproduction"
    / "config"
    / "particle_physics_feature_settings.csv"
)
SOURCE_VARIABLES_PATH = SOURCE_SETTINGS_PATH.with_name("variable_definitions.csv")

PPR_R = "ppr(d(root.comp1.AS_Tg,r))"
PPR_Z = "ppr(d(root.comp1.AS_Tg,z))"
PPR_Q_R = f"-k_mix*{PPR_R}"
PPR_Q_Z = f"-k_mix*{PPR_Z}"


def _config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _unbracket(value: str) -> str:
    return value.removeprefix("[").removesuffix("]")


def _assert_config_source_contract(config: dict[str, Any]) -> None:
    assert config["classification"] == "external_comsol_reference_correction"
    assert config["source_model"] == {
        "relative_path": (
            "model_dataset/cf4_o2_etch_caseA_nonlinear_sass/model/"
            "icp_rf_bias_cf4_o2_si_etching_caseP_caseA_SASS_"
            "formal_iondrag_theory_consistent_10_30_100nm.mph"
        ),
        "sha256": "3bbf08e3469758313eac5de473a7a0dd4cc9a6f72c9722393229f0b856e9b524",
        "load_mode": "ModelUtil.loadCopy",
        "saved": False,
    }


def _assert_config_case_contract(config: dict[str, Any]) -> None:
    assert config["case"]["fixed_rk4_step_s"] == 1.5625e-7
    assert config["case"]["output_times"] == 46
    assert config["case"]["particle_count"] == 287
    assert config["scope"]["fine_step_only"] is True
    assert config["scope"]["modifies_m3c0b_v6_evidence"] is False
    assert config["preserved_reference"]["evaluation_revision"] == 6
    assert config["preserved_reference"]["modified_by_this_run"] is False


def _assert_config_feature_contract(config: dict[str, Any]) -> None:
    feature = config["thermophoretic_feature"]
    assert feature["model"] == "Waldmann"
    assert feature["UsePPR"] is True
    assert feature["temperature_input"] == "root.comp1.AS_Tg"
    assert feature["recovered_temperature_gradient"] == {"r": PPR_R, "z": PPR_Z}
    assert feature["recovered_heat_flux"] == {"r": PPR_Q_R, "z": PPR_Q_Z}


def _assert_config_mesh_contract(config: dict[str, Any]) -> None:
    mesh = config["raw_export"]["native_mesh_nodes"]
    assert mesh["dataset"] == "dset_AS_field"
    assert mesh["location"] == "fromdataset"
    assert mesh["export_recover"] == "off"
    assert mesh["operator_recovery"] == "explicit ppr expressions"


def _assert_java_ppr_expressions(source: str) -> None:
    for expression in (PPR_R, PPR_Z, PPR_Q_R, PPR_Q_Z):
        assert f'"{expression}"' in source
    assert "pprint(" not in source
    assert '"unrecovered_temperature_gradient_r_K_per_m"' in source
    assert '"ppr_temperature_gradient_r_K_per_m"' in source
    assert '"ppr_heat_flux_r_W_per_m2"' in source
    assert '"fptas.thpf1.Ftfr"' in source


def _assert_java_export_contract(source: str) -> None:
    assert 'private static final String STEP = "0.15625[us]";' in source
    assert "STEP_CODES" not in source
    assert 'private static final String BACKGROUND_DATASET = "dset_AS_field";' in source
    assert 'export.set("location", "fromdataset");' in source
    assert 'export.set("level", "surface");' in source
    assert 'export.set("smooth", "material");' in source
    assert 'export.set("recover", "off");' in source


def _assert_java_source_isolation(source: str) -> None:
    assert 'ModelUtil.loadCopy("M3C1ThermophoresisPpr", SOURCE)' in source
    assert ".save(" not in source


def _assert_runner_no_clobber(source: str) -> None:
    assert "if (Test-Path -LiteralPath $OutputDirectory)" in source
    assert 'throw "M3-C1 PPR output already exists: $OutputDirectory"' in source
    assert "$OutputFullPath.StartsWith(" in source
    assert "outside the locked M3-C0b v6 evidence tree" in source
    assert '"_out_m3c1\\caseA_100nm_thermophoresis_ppr_v1"' in source
    assert "dt_0p625us" not in source
    assert "dt_0p3125us" not in source
    assert "$Config.case.step_directory" in source


def _assert_runner_hash_guards(source: str) -> None:
    assert "Get-FileHash -Algorithm SHA256 -LiteralPath $SourceModel" in source
    assert "Get-FileHash -Algorithm SHA256 -LiteralPath $PreservedState" in source
    assert "Get-FileHash -Algorithm SHA256 -LiteralPath $PreservedForce" in source
    assert "$SourceHashAfter -ne $SourceHashBefore" in source
    assert "$V6StateHashAfter -ne $V6StateHashBefore" in source
    assert "$V6ForceHashAfter -ne $V6ForceHashBefore" in source


def _assert_runner_execution_contract(source: str) -> None:
    assert "-nosave -np 1" in source
    assert '"steps_run=1"' in source
    assert 'thermophoresis_closure_status = "NOT_EVALUATED"' in source
    assert "Remove-Item -LiteralPath $Temporary" in source


def test_archived_source_metadata_requires_ppr_for_case_a_thermophoresis() -> None:
    with SOURCE_SETTINGS_PATH.open(newline="", encoding="utf-8") as stream:
        settings = list(csv.DictReader(stream))
    feature = {
        row["property"]: _unbracket(row["value"])
        for row in settings
        if row["physics_tag"] == "fptas" and row["feature_tag"] == "thpf1"
    }

    assert feature["ThermophoreticForceModel"] == "Waldmann"
    assert feature["UsePPR"] == "1"
    assert feature["minput_temperature"] == "root.comp1.AS_Tg"
    assert feature["k"] == "k_mix"
    assert feature["mg"] == "Mmix"

    with SOURCE_VARIABLES_PATH.open(newline="", encoding="utf-8") as stream:
        variables = list(csv.DictReader(stream))
    temperature = [row for row in variables if row["variable"] == "AS_Tg"]
    assert len(temperature) == 1
    assert temperature[0]["expression"] == "withsol('sol2',T,setval(Vrf,100[V]))"


def test_config_pins_exact_ppr_expressions_and_preserves_v6() -> None:
    config = _config()
    _assert_config_source_contract(config)
    _assert_config_case_contract(config)
    _assert_config_feature_contract(config)
    _assert_config_mesh_contract(config)


def test_java_exports_expression_level_ppr_without_export_recovery() -> None:
    source = JAVA_PATH.read_text(encoding="utf-8")
    _assert_java_ppr_expressions(source)
    _assert_java_export_contract(source)
    _assert_java_source_isolation(source)


def test_runner_is_fine_step_only_no_clobber_and_hashes_protected_inputs() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    _assert_runner_no_clobber(source)
    _assert_runner_hash_guards(source)
    _assert_runner_execution_contract(source)
