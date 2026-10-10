"""Create a small producer-neutral canonical case for the Quick Start."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from chamber_particles.case_format import (
    BoundaryData,
    DataBundle,
    GeometryData,
    RealizedTableSource,
    write,
)


def _bundle() -> DataBundle:
    nodes_m = np.asarray(
        [[0.0, 0.0], [0.01, 0.0], [0.01, 0.01], [0.0, 0.01]],
        dtype="<f8",
    )
    boundary = BoundaryData(
        line2=np.asarray([[0, 1], [1, 2], [2, 3], [3, 0]], dtype="<i8"),
        boundary_id=np.asarray([30, 20, 30, 10], dtype="<i4"),
        group_id=np.asarray([2, 1, 2, 0], dtype="<i4"),
        material_id=np.zeros(4, dtype="<i4"),
        owner_cell_type=np.full(4, 2, dtype="<u1"),
        owner_cell_local_index=np.zeros(4, dtype="<i8"),
        orientation=np.ones(4, dtype="<i1"),
    )
    geometry = GeometryData(
        nodes_m=nodes_m,
        boundary=boundary,
        group_names=("collector", "mirror", "escape"),
        quad4=np.asarray([[0, 1, 2, 3]], dtype="<i8"),
        quad4_domain_id=np.zeros(1, dtype="<i4"),
    )
    source = RealizedTableSource(
        name="particles",
        particle_id=np.asarray([1], dtype="<i8"),
        release_time_s=np.asarray([0.0], dtype="<f8"),
        position_m=np.asarray([[0.0025, 0.005]], dtype="<f8"),
        velocity_m_s=np.asarray([[1.0, 0.0]], dtype="<f8"),
        charge_number=np.asarray([0.0], dtype="<f8"),
        mass_kg=np.asarray([1.0e-15], dtype="<f8"),
        drag_diameter_m=np.asarray([1.0e-6], dtype="<f8"),
        contact_radius_m=np.asarray([0.0], dtype="<f8"),
        electrostatic_radius_m=np.asarray([5.0e-7], dtype="<f8"),
        displaced_volume_m3=np.asarray([5.235987755982988e-19], dtype="<f8"),
        model_weight=np.asarray([1.0], dtype="<f8"),
        material_id=np.zeros(1, dtype="<i4"),
    )
    source_definition = b"quickstart_ballistic_xy_v1:10mm-square:one-table-particle"
    provenance = json.dumps(
        {
            "producer": "quickstart-generator",
            "producer_version": "1",
            "source_sha256": "sha256:" + hashlib.sha256(source_definition).hexdigest(),
            "field_semantics_revision": "no_fields_ballistic_v1",
            "producer_metadata": {"description": "analytic XY ballistic demonstration"},
        }
    )
    return DataBundle(
        coordinate_system="cartesian_xy",
        provenance_json=provenance,
        geometry=geometry,
        sources=(source,),
    )


def _case_document(content_hash: str) -> dict[str, Any]:
    return {
        "format_version": 3,
        "case": {
            "name": "quickstart_ballistic_xy",
            "data_path": "case.h5",
            "expected_content_hash": content_hash,
        },
        "motion": {"mode": "cartesian_xy"},
        "time": {"start_s": 0.0, "end_s": 0.02, "dt_s": 0.004},
        "solver": {
            "integrator": "rk4_fixed",
            "backend": "cpu",
            "seed": 0,
            "event": {
                "geometry_rtol": 1.0e-12,
                "roundoff_ulps": 64,
                "max_refinements": 48,
                "max_interactions_per_step": 8,
                "corner_policy": "priority_then_combined_normal_v1",
            },
        },
        "resources": {"memory_limit_mb": 128},
        "physics": {"charge": {"model": "fixed"}},
        "sources": [{"name": "release", "type": "table", "table": "particles"}],
        "boundaries": [
            {"boundary_group": "collector", "priority": 10, "law": "stick"},
            {"boundary_group": "mirror", "priority": 20, "law": "specular"},
            {"boundary_group": "escape", "priority": 30, "law": "escape"},
        ],
        "output": {
            "trajectories": {
                "selection": "all",
                "schedule": {"explicit_times_s": [0.0, 0.005, 0.01, 0.015, 0.02]},
            }
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="new or empty input directory")
    arguments = parser.parse_args()
    directory = arguments.directory.expanduser().resolve()
    data_path = directory / "case.h5"
    case_path = directory / "case.yaml"
    directory.mkdir(parents=True, exist_ok=True)
    if data_path.exists() or case_path.exists():
        parser.error(f"refusing to replace an existing Quick Start case in {directory}")

    info = write(data_path, _bundle())
    case_path.write_text(
        yaml.safe_dump(_case_document(info.content_hash), sort_keys=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "case": str(case_path),
                "content_hash": info.content_hash,
                "data": str(data_path),
                "status": "created",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
