from __future__ import annotations

import argparse
import json
from pathlib import Path

from comsol_particle_export.field_bundle import build_field_bundle_from_samples, write_field_bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a solver NPZ field bundle from COMSOL field_samples.csv.")
    parser.add_argument("--field-samples-csv", type=Path, required=True)
    parser.add_argument("--out-npz", type=Path, required=True)
    parser.add_argument("--axis-names", nargs="+", required=True)
    parser.add_argument("--quantities", nargs="+", required=True)
    parser.add_argument("--coordinate-scale-m-per-model-unit", type=float, default=1.0)
    parser.add_argument("--coordinate-model-unit", default="m")
    parser.add_argument("--metadata-json", type=Path, default=None)
    args = parser.parse_args(argv)

    metadata = {}
    if args.metadata_json is not None:
        with args.metadata_json.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError("metadata-json must contain a JSON object")
        metadata = payload

    bundle = build_field_bundle_from_samples(
        args.field_samples_csv,
        axis_names=args.axis_names,
        quantities=args.quantities,
        coordinate_scale_m_per_model_unit=float(args.coordinate_scale_m_per_model_unit),
        coordinate_model_unit=str(args.coordinate_model_unit),
        metadata=metadata,
    )
    write_field_bundle(bundle, args.out_npz)
    print(json.dumps({"out_npz": str(args.out_npz), "keys": sorted(bundle.keys())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
