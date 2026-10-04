from __future__ import annotations

from pathlib import Path

from tools.vv.comsol import run_m3c1_material_event_candidate as candidate

CONFIG = Path(__file__).parents[1] / "cases" / "m3c1_common_p1_material_event_v1.json"


def test_locked_material_event_configuration_is_narrow() -> None:
    payload = candidate._load_config(CONFIG)

    assert payload["case"]["fixed_rk4_step_s"] == 1.5625e-7
    assert payload["case"]["event_window_end_s"] == 4.5875e-4
    assert len(payload["case"]["output_times_s"]) == 48
    assert payload["expected_first_event"] == {
        "particle_id": 57,
        "candidate_boundary_id": 6,
        "candidate_external_id": 134,
        "boundary_group": "wafer",
        "comsol_status_code": 3,
        "candidate_lifecycle": "stuck",
        "candidate_law": "stick",
        "wafer_z_m": 0.022,
    }
    assert payload["claim_policy"]["certifies_native_field_agreement"] is False
    assert payload["claim_policy"]["certifies_terminal_velocity_parity"] is False
