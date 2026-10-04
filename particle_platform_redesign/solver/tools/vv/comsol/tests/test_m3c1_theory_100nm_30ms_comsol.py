from __future__ import annotations

import json
import re
from pathlib import Path

COMSOL_ROOT = Path(__file__).resolve().parents[1]
COMMON_CONFIG = COMSOL_ROOT / "cases" / "m3c1_theory_100nm_30ms_v2.json"
COMSOL_CONFIG = COMSOL_ROOT / "cases" / "m3c1_theory_100nm_30ms_comsol_v1.json"
RUNNER = COMSOL_ROOT / "run_m3c1_theory_100nm_30ms_reference.ps1"
JAVA = COMSOL_ROOT / "comsol" / "RunM3C1Theory100nm30ms.java"
INSPECTOR = COMSOL_ROOT / "comsol" / "InspectM3C1Theory100nm30ms.java"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _scheduled_times(matrix: dict[str, object]) -> list[float]:
    values: list[float] = []
    for raw_segment in matrix["output_schedule_segments"]:  # type: ignore[index]
        segment = raw_segment  # type: ignore[assignment]
        start = float(segment["start_s"])
        stop = float(segment["stop_s"])
        step = float(segment["step_s"])
        count = round((stop - start) / step)
        values.extend(start + index * step for index in range(count + 1))
    return values


def _assert_contains(text: str, fragments: tuple[str, ...]) -> None:
    for fragment in fragments:
        assert fragment in text


def _assert_excludes(text: str, fragments: tuple[str, ...]) -> None:
    for fragment in fragments:
        assert fragment not in text


def _assert_workflow_steps(
    common: dict[str, object],
    expected: dict[str, tuple[list[float], float]],
    limit: float,
) -> None:
    for name, (steps, lipschitz) in expected.items():
        workflow = common["workflows"][name]  # type: ignore[index]
        assert workflow["fixed_rk4_steps_s"] == steps
        assert workflow["charge_lipschitz_s_inv"] == lipschitz
        assert steps[0] == 2.0 * steps[1]
        assert steps[1] == 2.0 * steps[2]
        assert all(step * lipschitz <= limit for step in steps)
        assert all(abs(0.03 / step - round(0.03 / step)) <= 1.0e-9 for step in steps)


def test_comsol_contract_reuses_the_preregistered_matrix_without_redefining_it() -> None:
    common = _load(COMMON_CONFIG)
    comsol = _load(COMSOL_CONFIG)
    matrix = common["matrix"]

    assert comsol["shared_contract"]["relative_path"].endswith(  # type: ignore[index, union-attr]
        "m3c1_theory_100nm_30ms_v2.json"
    )
    assert common["evaluation_revision"] == comsol["evaluation_revision"] == 2
    assert comsol["shared_contract"]["required_evaluation_revision"] == 2  # type: ignore[index]
    assert matrix["particle_diameter_m"] == 1.0e-7  # type: ignore[index]
    assert matrix["particle_count"] == 287  # type: ignore[index]
    assert matrix["run_keys"] == ["coarse", "medium", "fine"]  # type: ignore[index]
    expected = {
        "caseA": ([6.25e-7, 3.125e-7, 1.5625e-7], 287013.1640482939),
        "caseP": ([4.6875e-8, 2.34375e-8, 1.171875e-8], 10403456.669581516),
    }
    limit = common["acceptance"]["maximum_dt_charge_lipschitz"]  # type: ignore[index]
    assert limit == 0.5
    _assert_workflow_steps(common, expected, limit)  # type: ignore[arg-type]
    times = _scheduled_times(matrix)  # type: ignore[arg-type]
    assert len(times) == matrix["output_count"] == 121  # type: ignore[index]
    assert times[0] == 0.0
    assert abs(times[-1] - 0.03) < 1.0e-15


def test_verified_case_tags_and_boundary_semantics_are_explicit() -> None:
    config = _load(COMSOL_CONFIG)
    workflows = config["workflows"]

    assert workflows["caseP"] == {  # type: ignore[index]
        "physics_tag": "fpt",
        "background_study": "std2",
        "background_solution": "sol2",
        "source_dataset": "part_P_100nm",
        "particle_geometry": "pgeom_fpt",
        "position_dofs": ["qr", "qz"],
        "charge_state": "ZP",
        "source_release_charge": "Z0_P",
    }
    assert workflows["caseA"] == {  # type: ignore[index]
        "physics_tag": "fptas",
        "background_study": "stdASf",
        "background_solution": "sol26",
        "source_dataset": "part_AS_100nm",
        "particle_geometry": "pgeom_fptas",
        "position_dofs": ["q3r", "q3z"],
        "charge_state": "ZAS",
        "source_release_charge": "AS_Z0",
    }
    boundaries = config["inventory_evidence"]["verified_boundary_selections"]  # type: ignore[index]
    assert boundaries["material_stick"] == [6, 8, 28, 29, 32, 33, 34, 36, 38, 40, 41, 45, 46, 47]  # type: ignore[index]
    assert boundaries["gas_inlet_freeze_hold"] == [37]  # type: ignore[index]
    assert boundaries["pump_disappear_escape"] == [35]  # type: ignore[index]
    assert boundaries["axis_not_a_material_event"] == [5]  # type: ignore[index]


def test_java_rhs_uses_only_canonical_p1_primitives() -> None:
    source = JAVA.read_text(encoding="utf-8")
    config = _load(COMSOL_CONFIG)

    _assert_contains(
        source,
        (
            "ModelUtil.loadCopy",
            "m3c1_ne(r,z)",
            "m3c1_gradE2r(r,z)",
            "m3c1_qr(r,z)",
            'brownian_active", "false',
            'saffman_active", "false',
            "range(6e-3[s],1e-3[s],3e-2[s])",
            'private static final String[] RUN_KEYS = {"coarse", "medium", "fine"};',
            "__M3C1_STEP_COARSE_S__",
            "__M3C1_CHARGE_LIPSCHITZ_S_INV__",
            "step violates the charge Lipschitz limit",
        ),
    )
    _assert_excludes(source, ("ModelUtil.save", "java.io.", "java.nio.file"))
    assert len(re.findall(r"\bclass\s+\w+", source)) == 1
    assert config["same_field_contract"]["primitive_component_count"] == 22  # type: ignore[index]
    _assert_excludes(
        source,
        ("root.comp1.AS_", "root.comp1.ne_d", "root.comp1.ni_d", "withsol("),
    )


def test_runner_is_no_clobber_and_attests_required_outputs() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    inspector = INSPECTOR.read_text(encoding="utf-8")

    _assert_contains(
        runner,
        (
            "if (Test-Path -LiteralPath $OutputDirectory)",
            "-nosave -np 1",
            "m3c1_theory_100nm_30ms_v2.json",
            "caseP = @(4.6875e-8, 2.34375e-8, 1.171875e-8)",
            "caseP = 10403456.669581516",
            "maximum_dt_charge_lipschitz = $Maximum",
            "violates the charge Lipschitz limit",
            '"__M3C1_STEP_COARSE_S__"',
            "source_sha256_before",
            "source_sha256_after",
            'field_representation = "canonical_exact_connectivity_p1"',
            'primitive_source = "prepared_common_p1_tables"',
            'status = "COMPLETE"',
            "workflow = $CaseName",
            "bytes = $_.Length",
            "reference_run_report.json",
            "artifact_hashes.csv",
            "No COMSOL trajectory study was started",
        ),
    )
    _assert_contains(inspector, ('study_run", "false', 'model_saved", "false'))


def test_runner_absolutizes_relative_cli_roots_before_changing_location() -> None:
    runner = RUNNER.read_text(encoding="utf-8")

    invocation = runner.index("$InvocationDirectory = (Get-Location).ProviderPath")
    comsol = runner.index("$ComsolRoot = [IO.Path]::GetFullPath($ComsolRoot, $InvocationDirectory)")
    candidate = runner.index(
        "$CandidateRoot = [IO.Path]::GetFullPath($CandidateRoot, $InvocationDirectory)"
    )
    output = runner.index(
        "$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory, $InvocationDirectory)"
    )
    first_location_change = runner.index("Push-Location")

    assert invocation < comsol < candidate < output < first_location_change
    assert 'Write-Output "Resolved candidate root: $CandidateRoot"' in runner
    assert 'Write-Output "Resolved output root: $OutputDirectory"' in runner


def test_escape_hit_position_is_never_claimed_when_comsol_only_exposes_nan() -> None:
    config = _load(COMSOL_CONFIG)
    raw = config["raw_export"]

    assert raw["include_nan"] is True  # type: ignore[index]
    assert raw["status_authority"] == "particlestatus plus physics.fs"  # type: ignore[index]
    assert raw["event_time_authority"] == "physics.st"  # type: ignore[index]
    assert str(raw["escape_hit_position_policy"]).startswith("NOT_DIRECTLY_OBSERVED")  # type: ignore[index]
