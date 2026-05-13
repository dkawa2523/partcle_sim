from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "external" / "comsol_particle_export"
sys.path.insert(0, str(EXTERNAL))

from comsol_particle_export.promotion import (  # noqa: E402
    canonicalize_wall_event_table,
    is_wall_event_table,
    particle_property_defaults,
    promote_particle_status_truth,
    promote_reextract_outputs,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_probe(path: Path, expression: str, unit: str, values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = ["% Index," + f"{expression} ({unit}) @ t=0"]
    for idx, value in enumerate(values, start=1):
        rows.append(f"{idx},{value}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_particle_property_defaults_derive_mass_from_density_and_diameter(tmp_path: Path) -> None:
    inventory = tmp_path / "particle_release_inventory.json"
    _write_json(
        inventory,
        {
            "features": [
                {
                    "release_kind": "particle_properties",
                    "feature_tag": "pp1",
                    "property_values": {
                        "ParticlePropertySpec": "SpecifyDensityAndDiameter",
                        "dp": "10[um]",
                        "rhop": "2200[kg/m^3]",
                        "mp": "1[mg]",
                        "Z": "0",
                    },
                }
            ]
        },
    )

    defaults = particle_property_defaults(inventory)

    assert defaults["values"]["diameter"] == pytest.approx(1.0e-5)
    assert defaults["values"]["density"] == pytest.approx(2200.0)
    assert defaults["values"]["mass"] == pytest.approx(2200.0 * 3.141592653589793 / 6.0 * 1.0e-15)
    assert defaults["sources"]["mass"] == "derived_from_density_diameter"


def test_promote_reextract_outputs_writes_canonical_release_and_wall_events(tmp_path: Path) -> None:
    reextract = tmp_path / "reextract"
    baseline = tmp_path / "comsol_release_particles.csv"
    pd.DataFrame(
        [
            {"particle_id": 1, "release_time": 0.0, "x": 0.001, "y": 0.002, "v_x": 0.0, "v_y": 0.0},
            {"particle_id": 2, "release_time": 0.0, "x": 0.003, "y": 0.004, "v_x": 0.0, "v_y": 0.0},
        ]
    ).to_csv(baseline, index=False)
    _write_probe(reextract / "release_property_probe_fpt_dp" / "probe_fpt_dp.csv", "fpt.dp", "um", [10.0, 11.0])
    _write_probe(
        reextract / "release_property_probe_fpt_source" / "probe_fpt_source.csv",
        "fpt.source",
        "",
        [5.0, 7.0],
    )
    inventory = tmp_path / "particle_release_inventory.json"
    _write_json(
        inventory,
        {
            "features": [
                {
                    "release_kind": "particle_properties",
                    "property_values": {
                        "ParticlePropertySpec": "SpecifyDensityAndDiameter",
                        "dp": "10[um]",
                        "rhop": "2200[kg/m^3]",
                        "Z": "0",
                    },
                }
            ]
        },
    )
    pd.DataFrame(
        [
            {"particle_id": 1, "hit_time_s": 0.1, "comsol_entity_id": 20, "outcome": "bounce"},
            {"particle_id": 2, "hit_time_s": 0.2, "comsol_entity_id": 21, "outcome": "freeze"},
        ]
    ).to_csv(reextract / "reviewed_wall_hits.csv", index=False)

    summary = promote_reextract_outputs(
        reextract_root=reextract,
        baseline_release_csv=baseline,
        out_dir=tmp_path / "canonical",
        particle_release_inventory_json=inventory,
    )

    release = pd.read_csv(tmp_path / "canonical" / "comsol_release_particles_canonical.csv")
    assert release["diameter"].tolist() == pytest.approx([1.0e-5, 1.1e-5])
    assert release["density"].tolist() == pytest.approx([2200.0, 2200.0])
    assert release["source_entity"].tolist() == pytest.approx([5.0, 7.0])
    assert summary["ready_inputs"]["wall_event_truth_ready"] is True
    wall = pd.read_csv(tmp_path / "canonical" / "comsol_wall_events.csv")
    assert wall["hit_time_s"].tolist() == pytest.approx([0.1, 0.2])
    assert wall["outcome"].tolist() == ["bounce", "freeze"]


def test_promote_release_assigns_source_from_release_inventory_and_boundary_map(tmp_path: Path) -> None:
    reextract = tmp_path / "reextract"
    baseline = tmp_path / "comsol_release_particles.csv"
    pd.DataFrame(
        [
            {"particle_id": 1, "release_time": 0.0, "x": -0.0039, "y": 0.0, "v_x": 0.0, "v_y": 0.0},
            {"particle_id": 2, "release_time": 0.0, "x": 0.0, "y": -0.0039, "v_x": 0.0, "v_y": 0.0},
        ]
    ).to_csv(baseline, index=False)
    inventory = tmp_path / "particle_release_inventory.json"
    _write_json(
        inventory,
        {
            "features": [
                {"release_kind": "release", "feature_tag": "inl1", "selection_entities": [1]},
                {"release_kind": "release", "feature_tag": "inl2", "selection_entities": [5]},
            ]
        },
    )
    boundary_map = tmp_path / "comsol_boundary_entity_mapping.csv"
    pd.DataFrame(
        [
            {
                "solver_part_id": 2,
                "comsol_api_selection_entity_id": 1,
                "x_min_m": -0.0039,
                "x_max_m": -0.0039,
                "y_min_m": -0.00025,
                "y_max_m": 0.00025,
            },
            {
                "solver_part_id": 6,
                "comsol_api_selection_entity_id": 5,
                "x_min_m": -0.00025,
                "x_max_m": 0.00025,
                "y_min_m": -0.0039,
                "y_max_m": -0.0039,
            },
        ]
    ).to_csv(boundary_map, index=False)

    summary = promote_reextract_outputs(
        reextract_root=reextract,
        baseline_release_csv=baseline,
        out_dir=tmp_path / "canonical",
        particle_release_inventory_json=inventory,
        boundary_map_csv=boundary_map,
    )

    release = pd.read_csv(tmp_path / "canonical" / "comsol_release_particles_canonical.csv")
    assert release["source_entity"].tolist() == pytest.approx([1.0, 5.0])
    assert release["source_part_id"].tolist() == pytest.approx([2.0, 6.0])
    assert summary["release"]["source_assignment"]["assigned_count"] == 2


def test_promote_particle_status_from_status_stop_time_probe(tmp_path: Path) -> None:
    reextract = tmp_path / "reextract"
    baseline = tmp_path / "comsol_release_particles.csv"
    pd.DataFrame(
        [
            {"particle_id": 1, "release_time": 0.0, "x": 0.001, "y": 0.002, "v_x": 0.0, "v_y": 0.0},
            {"particle_id": 2, "release_time": 0.0, "x": 0.003, "y": 0.004, "v_x": 0.0, "v_y": 0.0},
        ]
    ).to_csv(baseline, index=False)
    inventory = tmp_path / "particle_release_inventory.json"
    _write_json(
        inventory,
        {
            "features": [
                {
                    "release_kind": "particle_properties",
                    "property_values": {
                        "ParticlePropertySpec": "SpecifyDensityAndDiameter",
                        "dp": "10[um]",
                        "rhop": "2200[kg/m^3]",
                        "Z": "0",
                    },
                }
            ]
        },
    )
    _write_probe(reextract / "wall_event_probe_fpt_st" / "probe_fpt_st.csv", "fpt.st", "s", [0.25, float("nan")])
    _write_probe(reextract / "wall_event_probe_fpt_fs" / "probe_fpt_fs.csv", "fpt.fs", "", [3.0, 1.0])

    summary = promote_reextract_outputs(
        reextract_root=reextract,
        baseline_release_csv=baseline,
        out_dir=tmp_path / "canonical",
        particle_release_inventory_json=inventory,
    )

    assert summary["wall_events"]["promoted"] is False
    assert summary["particle_status"]["promoted"] is True
    assert summary["particle_status"]["promotion_kind"] == "status_stop_time_probe"
    status = pd.read_csv(tmp_path / "canonical" / "comsol_particle_status.csv")
    assert status["particle_id"].tolist() == [1]
    assert status["stop_time_s"].tolist() == pytest.approx([0.25])
    assert status["final_status"].tolist() == ["stuck"]


def test_promote_particle_status_from_multi_expression_status_export(tmp_path: Path) -> None:
    reextract = tmp_path / "reextract"
    baseline = tmp_path / "comsol_release_particles.csv"
    pd.DataFrame(
        [{"particle_id": 1, "release_time": 0.0, "x": 0.001, "y": 0.002, "v_x": 0.0, "v_y": 0.0}]
    ).to_csv(baseline, index=False)
    inventory = tmp_path / "particle_release_inventory.json"
    _write_json(
        inventory,
        {
            "features": [
                {
                    "release_kind": "particle_properties",
                    "property_values": {
                        "ParticlePropertySpec": "SpecifyDensityAndDiameter",
                        "dp": "10[um]",
                        "rhop": "2200[kg/m^3]",
                        "Z": "0",
                    },
                }
            ]
        },
    )
    path = reextract / "wall_status_recomputed" / "comsol_wall_status_recomputed.csv"
    path.parent.mkdir(parents=True)
    path.write_text(
        "\n".join(
            [
                "% Index,fpt.st (s) @ t=0,fpt.fs @ t=0,fpt.bnd @ t=0.5",
                "1,0.5,4,18",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = promote_reextract_outputs(
        reextract_root=reextract,
        baseline_release_csv=baseline,
        out_dir=tmp_path / "canonical",
        particle_release_inventory_json=inventory,
    )

    assert summary["wall_events"]["promoted"] is True
    assert summary["particle_status"]["promoted"] is True
    status = pd.read_csv(tmp_path / "canonical" / "comsol_particle_status.csv")
    assert status.loc[0, "stop_time_s"] == pytest.approx(0.5)
    wall = pd.read_csv(tmp_path / "canonical" / "comsol_wall_events.csv")
    assert wall.loc[0, "comsol_entity_id"] == pytest.approx(18.0)


def test_wall_event_detection_uses_schema_not_filename(tmp_path: Path) -> None:
    path = tmp_path / "arbitrary_name.csv"
    pd.DataFrame([{"particle_id": 1, "hit_time_s": 0.1, "comsol_entity_id": 3}]).to_csv(path, index=False)
    assert is_wall_event_table(path)
    canonical = canonicalize_wall_event_table(path)
    assert canonical["comsol_entity_id"].iloc[0] == pytest.approx(3.0)
