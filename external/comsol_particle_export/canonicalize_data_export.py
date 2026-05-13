from __future__ import annotations

import argparse
import json
from pathlib import Path

from comsol_particle_export.data_export import (
    canonicalize_particle_wide_data_export,
    write_canonical_particle_trajectory,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Canonicalize a COMSOL Data export CSV into particle trajectory CSV.")
    parser.add_argument("--data-export-csv", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--axis-names", nargs="+", default=["x", "y"])
    parser.add_argument(
        "--expression-map",
        nargs="*",
        default=None,
        help="Optional raw=output mappings, for example x=x y=y fpt.vx=v_x fpt.vy=v_y.",
    )
    parser.add_argument("--required-output-columns", nargs="*", default=None)
    parser.add_argument("--fallback-coordinate-scale-m-per-unit", type=float, default=1.0)
    parser.add_argument("--report-json", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.expression_map:
        mapping: dict[str, str] = {}
        for item in args.expression_map:
            if "=" not in item:
                raise ValueError(f"expression-map item must be raw=output: {item}")
            raw, output = item.split("=", 1)
            mapping[raw] = output
        frame, report = canonicalize_particle_wide_data_export(
            args.data_export_csv,
            expression_map=mapping,
            required_output_columns=args.required_output_columns,
            fallback_unit_scale=float(args.fallback_coordinate_scale_m_per_unit),
        )
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.out_csv, index=False)
        report["output_csv"] = str(args.out_csv)
        if args.report_json is not None:
            args.report_json.parent.mkdir(parents=True, exist_ok=True)
            args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    else:
        report = write_canonical_particle_trajectory(
            args.data_export_csv,
            args.out_csv,
            axis_names=args.axis_names,
            fallback_coordinate_scale_m_per_unit=float(args.fallback_coordinate_scale_m_per_unit),
            report_json=args.report_json,
        )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
