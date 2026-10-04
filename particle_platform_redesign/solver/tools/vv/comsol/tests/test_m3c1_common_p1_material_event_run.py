from __future__ import annotations

from pathlib import Path

import pytest
from tools.vv.comsol import validate_m3c1_common_p1_material_event_run as event_run

CONFIG = Path(__file__).parents[1] / "cases" / "m3c1_common_p1_material_event_v1.json"


def _configuration_line(*, profile: str = "material_event") -> str:
    fields = {
        "run_profile": profile,
        "step_s": "1.5625e-7",
        "time_start_s": "0.0",
        "time_end_s": "4.5875e-4",
        "brownian_active": "false",
        "saffman_active": "false",
        "dynamic_charge_active": "true",
        "native_thermophoresis_active": "false",
        "common_heat_flux_force_active": "true",
        "field_source": "canonical_exact_connectivity_P1_sectionwise",
        "initial_state_source": "candidate_realized_source_table",
        "primitive_function_count": "22",
        "deterministic_contributions": (
            "electric,relative_flow_ion_drag,epstein_drag,"
            "waldmann_heat_flux_thermophoresis,free_molecular_lift_sensitivity,"
            "dielectrophoresis,gravity_buoyancy"
        ),
        "integrator": "classical_rk4",
        "integrator_order": "4",
        "relative_tolerance": "1e-8",
        "wall_accuracy_order": "1",
        "store_particle_status": "true",
        "store_extra": "false",
        "physics": "fptas",
        "study": "stdM3C1P115625",
        "solution": "solM3C1P115625",
        "output_times": "48",
        "particle_rows": "287",
        "source_model": "source_copy.mph",
        "model_saved": "false",
    }
    return "M3C1_COMMON_P1|configuration|" + "|".join(
        f"{key}={value}" for key, value in fields.items()
    )


def _run_pass_line() -> str:
    return (
        "M3C1_COMMON_P1|run_pass|case=caseA_100nm|run_profile=material_event|"
        "steps=1.5625e-7|common_field=canonical_exact_connectivity_P1_sectionwise|"
        "model_saved=false"
    )


def _raw_tables(root: Path, *, malformed_width: bool = False) -> None:
    directory = root / event_run.STEP_DIRECTORY
    directory.mkdir()
    for name, values_per_frame in event_run.RAW_TABLE_EXPRESSIONS.items():
        expressions = 48 * values_per_frame
        row = ",".join(
            "0" for _ in range(expressions - (malformed_width and name == "force_raw_wide.csv"))
        )
        text = f"% Nodes,287\n% Expressions,{expressions}\n" + f"{row}\n" * 287
        (directory / name).write_text(text, encoding="utf-8")


def test_validates_exact_material_event_receipt_and_raw_shapes(tmp_path: Path) -> None:
    _raw_tables(tmp_path)
    (tmp_path / "comsol_process.log").write_text(
        f"{_configuration_line()}\n{_run_pass_line()}\n", encoding="utf-16"
    )

    summary = event_run.validate(tmp_path, CONFIG)

    assert summary["status"] == "PASS"
    assert summary["configuration_receipt"]["run_profile"] == "material_event"
    assert set(summary["raw_tables"]) == set(event_run.RAW_TABLE_EXPRESSIONS)
    assert (tmp_path / "material_event_raw_validation.json").is_file()


def test_rejects_a_pre_event_receipt_for_the_material_event_run(tmp_path: Path) -> None:
    _raw_tables(tmp_path)
    (tmp_path / "comsol_process.log").write_text(
        f"{_configuration_line(profile='pre_event')}\n{_run_pass_line()}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="configuration receipt differs"):
        event_run.validate(tmp_path, CONFIG)


def test_rejects_a_raw_row_that_does_not_match_the_expression_width(
    tmp_path: Path,
) -> None:
    _raw_tables(tmp_path, malformed_width=True)
    (tmp_path / "comsol_process.log").write_text(
        f"{_configuration_line()}\n{_run_pass_line()}\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="data row 1 has"):
        event_run.validate(tmp_path, CONFIG)
