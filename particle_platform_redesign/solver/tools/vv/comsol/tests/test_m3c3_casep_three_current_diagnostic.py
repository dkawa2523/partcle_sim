from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHARED_JAVA = ROOT / "comsol" / "RunM3C3CasePThreeCurrent.java"
DIAGNOSTIC_JAVA = ROOT / "comsol" / "RunM3C3CasePThreeCurrentDiagnostic.java"
RUNNER = ROOT / "run_m3c3_caseP_three_current.ps1"


def _assert_fragments(
    text: str,
    *,
    present: tuple[str, ...],
    absent: tuple[str, ...] = (),
) -> None:
    for fragment in present:
        assert fragment in text
    for fragment in absent:
        assert fragment not in text


def test_diagnostic_entry_is_fine_step_only_without_changing_formal_dispatch() -> None:
    shared = SHARED_JAVA.read_text(encoding="utf-8")
    diagnostic = DIAGNOSTIC_JAVA.read_text(encoding="utf-8")

    assert 'ModelUtil.loadCopy("M3C3CasePThreeCurrent", SOURCE)' in shared
    assert "ModelUtil.save" not in shared
    assert 'private static final String DIAGNOSTIC_FIXED_STEP_SECONDS = "1.25e-6";' in shared
    assert "run(fixedStepSeconds, false);" in shared
    assert "run(DIAGNOSTIC_FIXED_STEP_SECONDS, true);" in shared
    assert "Diagnostic export is restricted to the 1.25 us fine rerun" in shared
    assert "RunM3C3CasePThreeCurrent.runFineDiagnostic();" in diagnostic


def test_diagnostic_exports_required_state_force_and_charge_rate_columns() -> None:
    shared = SHARED_JAVA.read_text(encoding="utf-8")

    state_columns = (
        "r_m",
        "z_m",
        "velocity_r_m_per_s",
        "velocity_z_m_per_s",
        "charge_number_e",
        "current_status_code",
    )
    force_columns = (
        "charge_rate_number_s",
        "positive_collection_rate_number_s",
        "electron_collection_rate_number_s",
        "negative_collection_rate_number_s",
        "electric_force_r_N",
        "electric_force_z_N",
        "ion_drag_force_r_N",
        "ion_drag_force_z_N",
        "epstein_drag_force_r_N",
        "epstein_drag_force_z_N",
        "thermophoretic_force_r_N",
        "thermophoretic_force_z_N",
        "lift_force_r_N",
        "lift_force_z_N",
        "dep_force_r_N",
        "dep_force_z_N",
        "gravity_buoyancy_force_r_N",
        "gravity_buoyancy_force_z_N",
        "total_force_r_N",
        "total_force_z_N",
    )
    _assert_fragments(
        shared,
        present=(
            *(f'"{column}"' for column in (*state_columns, *force_columns)),
            'configuredForce(physics, "idf")',
            "configuredForce(physics, EPSTEIN_TAG)",
            "configuredForce(physics, THERMO_TAG)",
            'configuredForce(physics, "liftfm")',
            'configuredForce(physics, "depf")',
            'PHYSICS + ".ef1.Fer"',
            'PHYSICS + ".gf1.Fgr"',
            "configuredChargeRate(physics)",
            '"positive_collection_rate", rates[0]',
            '"electron_collection_rate", rates[1]',
            '"negative_collection_rate", rates[2]',
            '"diagnostic_state_raw_wide.csv"',
            '"diagnostic_force_raw_wide.csv"',
        ),
        absent=('PHYSICS + ".df1.FDr"', "diagnostic_primitive"),
    )


def test_reference_uses_one_explicit_p1_epstein_force_owner() -> None:
    shared = SHARED_JAVA.read_text(encoding="utf-8")

    assert 'private static final String EPSTEIN_TAG = "m3c3Epstein"' in shared
    assert 'physics.feature("df1").active(false)' in shared
    assert 'physics.create(EPSTEIN_TAG, "Force", 2)' in shared
    assert "configureForce(physics.feature(EPSTEIN_TAG)" in shared
    assert "CommonP1Epstein.force(PHYSICS)" in shared
    assert '"drag_implementation", "explicit_custom_force"' in shared
    assert "emitDiagnosticDragConfiguration" not in shared


def test_formal_powershell_has_isolated_external_vv_diagnostic_option() -> None:
    runner = RUNNER.read_text(encoding="utf-8")

    _assert_fragments(
        runner,
        present=(
            '"diagnostic_dt_1p25us"',
            "Join-Path $PreparedRoot $DefaultOutputName",
            "[switch]$DiagnosticForceExport",
            "$DiagnosticForceExport -and $FixedStepS -ne 1.25e-6",
            "$StagedJavaText.Replace($StepToken, $FixedStepText)",
            "RunM3C3CasePThreeCurrentDiagnostic.class",
            "& $Compiler -classpathadd $OutputDirectory $StagedDiagnosticJava",
            "-nosave",
            "source_copy.mph",
            "source_mph_sha256_before",
            "source_mph_sha256_after",
            '$NumericalRun["comparison_scope"] = "active_rows_only"',
            "diagnostic_state_raw_wide.csv",
            "diagnostic_force_raw_wide.csv",
            "tools.vv.comsol.normalize_m3c3_caseP_three_current $OutputDirectory",
            "M3-C3 normalized reference did not pass its structural checks",
        ),
        absent=("if (-not $DiagnosticForceExport)",),
    )
