from __future__ import annotations

import argparse
import json
from pathlib import Path

from comsol_particle_export.data_export import derive_particle_tables_from_trajectory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Derive COMSOL release/final particle tables from canonical trajectory CSV.")
    parser.add_argument("--trajectory-csv", type=Path, required=True)
    parser.add_argument("--release-csv", type=Path, default=None)
    parser.add_argument("--final-csv", type=Path, default=None)
    parser.add_argument("--initial-vx", type=float, default=None)
    parser.add_argument("--initial-vy", type=float, default=None)
    parser.add_argument("--initial-vz", type=float, default=None)
    parser.add_argument("--final-state", default="unknown")
    parser.add_argument("--report-json", type=Path, default=None)
    args = parser.parse_args(argv)

    initial_velocity = {}
    if args.initial_vx is not None:
        initial_velocity["v_x"] = float(args.initial_vx)
    if args.initial_vy is not None:
        initial_velocity["v_y"] = float(args.initial_vy)
    if args.initial_vz is not None:
        initial_velocity["v_z"] = float(args.initial_vz)

    report = derive_particle_tables_from_trajectory(
        args.trajectory_csv,
        release_csv=args.release_csv,
        final_csv=args.final_csv,
        initial_velocity=initial_velocity or None,
        final_state=args.final_state,
        report_json=args.report_json,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
