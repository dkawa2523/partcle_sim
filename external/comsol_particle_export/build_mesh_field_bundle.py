from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from comsol_particle_export.field_bundle import (
    build_triangle_mesh_field_bundle_from_samples,
    write_field_bundle,
)


def _load_mesh_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        if "mesh_vertices" not in payload or "mesh_triangles" not in payload:
            raise ValueError(f"geometry NPZ must contain mesh_vertices and mesh_triangles: {path}")
        return np.asarray(payload["mesh_vertices"], dtype=np.float64), np.asarray(payload["mesh_triangles"], dtype=np.int32)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a solver triangle-mesh field NPZ from COMSOL mesh_field_samples.csv."
    )
    parser.add_argument("--mesh-field-samples-csv", type=Path, required=True)
    parser.add_argument("--geometry-npz", type=Path, required=True)
    parser.add_argument("--out-npz", type=Path, required=True)
    parser.add_argument("--axis-names", nargs="+", default=["x", "y"])
    parser.add_argument("--quantities", nargs="+", required=True)
    parser.add_argument("--coordinate-scale-m-per-model-unit", type=float, default=1.0)
    parser.add_argument("--coordinate-model-unit", default="m")
    parser.add_argument("--coordinate-tolerance-m", type=float, default=1.0e-9)
    parser.add_argument("--support-tolerance-m", type=float, default=None)
    parser.add_argument("--metadata-json", type=Path, default=None)
    args = parser.parse_args(argv)

    metadata = {}
    if args.metadata_json is not None:
        with args.metadata_json.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError("metadata-json must contain a JSON object")
        metadata = payload
    if args.support_tolerance_m is not None:
        metadata["support_tolerance_m"] = float(args.support_tolerance_m)

    vertices, triangles = _load_mesh_arrays(args.geometry_npz)
    bundle = build_triangle_mesh_field_bundle_from_samples(
        args.mesh_field_samples_csv,
        mesh_vertices=vertices,
        mesh_triangles=triangles,
        axis_names=args.axis_names,
        quantities=args.quantities,
        coordinate_scale_m_per_model_unit=float(args.coordinate_scale_m_per_model_unit),
        coordinate_model_unit=str(args.coordinate_model_unit),
        coordinate_tolerance_m=float(args.coordinate_tolerance_m),
        metadata=metadata,
    )
    write_field_bundle(bundle, args.out_npz)
    print(json.dumps({"out_npz": str(args.out_npz), "keys": sorted(bundle.keys())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
