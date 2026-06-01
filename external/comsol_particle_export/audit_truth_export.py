from __future__ import annotations

import argparse
import json
from pathlib import Path

from comsol_particle_export.truth_audit import build_truth_audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit existing COMSOL export artifacts and normalize them into a "
            "truth manifest for parity/debug work."
        )
    )
    parser.add_argument("--case-name", default="micromixer_particle_tracing")
    parser.add_argument(
        "--field-raw-dir",
        default="_external_exports/micromixer_particle_tracing_field_probe",
        help="Raw COMSOL export directory that contains field/model inventories.",
    )
    parser.add_argument(
        "--particle-raw-dir",
        default="_external_exports/micromixer_particle_tracing_xy_velocity_probe",
        help="Raw COMSOL export directory that contains particle trajectory/release tables.",
    )
    parser.add_argument(
        "--solver-case-dir",
        default="_external_exports/micromixer_particle_tracing_solver_case",
        help="Solver case directory containing particles.csv, walls, and generated mapping files.",
    )
    parser.add_argument(
        "--out-dir",
        default="_external_exports/micromixer_particle_tracing_truth_audit",
        help="Directory for manifest, release, boundary, and field replay diagnostics.",
    )
    parser.add_argument(
        "--field-npz",
        default="_external_exports/micromixer_particle_tracing_solver_case/generated/comsol_field_mesh_2d.npz",
        help="Preferred mesh-native field bundle for COMSOL trajectory replay.",
    )
    parser.add_argument(
        "--regular-field-npz",
        default="_external_exports/micromixer_particle_tracing_solver_case/generated/comsol_field_2d.npz",
        help="Regular-grid bundle recorded as diagnostic-only context.",
    )
    parser.add_argument(
        "--solver-output-dir",
        default=None,
        help="Optional solver output directory recorded for future audit extensions.",
    )
    parser.add_argument(
        "--comparison-dir",
        default=None,
        help="Optional comparison output directory containing comparison_summary.json for root-cause ranking.",
    )
    parser.add_argument(
        "--run-config",
        default=None,
        help="Optional run config used to compare COMSOL force settings with solver force settings.",
    )
    parser.add_argument(
        "--skip-field-replay",
        action="store_true",
        help="Write the truth manifest without replaying the field on COMSOL trajectory points.",
    )
    args = parser.parse_args(argv)

    summary = build_truth_audit(
        case_name=args.case_name,
        field_raw_dir=Path(args.field_raw_dir),
        particle_raw_dir=Path(args.particle_raw_dir),
        solver_case_dir=Path(args.solver_case_dir),
        out_dir=Path(args.out_dir),
        field_npz=Path(args.field_npz),
        regular_field_npz=Path(args.regular_field_npz),
        solver_output_dir=Path(args.solver_output_dir) if args.solver_output_dir else None,
        comparison_dir=Path(args.comparison_dir) if args.comparison_dir else None,
        run_config=Path(args.run_config) if args.run_config else None,
        compare_field_replay=not args.skip_field_replay,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
