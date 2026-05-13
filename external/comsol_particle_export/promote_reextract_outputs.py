from __future__ import annotations

import argparse
import json
from pathlib import Path

from comsol_particle_export.promotion import promote_reextract_outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Promote reviewed COMSOL re-extraction outputs into canonical release "
            "and wall-event truth artifacts."
        )
    )
    parser.add_argument("--reextract-root", type=Path, required=True)
    parser.add_argument("--baseline-release-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--particle-release-inventory-json",
        type=Path,
        default=None,
        help="Optional particle_release_inventory.json used for global particle-property defaults.",
    )
    parser.add_argument(
        "--boundary-map-csv",
        type=Path,
        default=None,
        help="Optional COMSOL boundary mapping used to assign release source_entity/source_part_id from release selections.",
    )
    args = parser.parse_args(argv)

    summary = promote_reextract_outputs(
        reextract_root=args.reextract_root,
        baseline_release_csv=args.baseline_release_csv,
        out_dir=args.out_dir,
        particle_release_inventory_json=args.particle_release_inventory_json,
        boundary_map_csv=args.boundary_map_csv,
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
