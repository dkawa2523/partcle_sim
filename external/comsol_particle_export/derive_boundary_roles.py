from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from comsol_particle_export.boundary_roles import derive_boundary_roles


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Derive solver boundary roles from COMSOL physics feature inventory.")
    parser.add_argument("--raw-export-dir", type=Path, required=True)
    parser.add_argument("--boundary-map-csv", type=Path, default=None)
    parser.add_argument("--part-walls-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--write-part-walls-csv", type=Path, default=None)
    parser.add_argument("--write-materials-csv", type=Path, default=None)
    args = parser.parse_args(argv)

    summary = derive_boundary_roles(
        raw_export_dir=args.raw_export_dir,
        boundary_map_csv=args.boundary_map_csv,
        part_walls_csv=args.part_walls_csv,
        out_dir=args.out_dir,
        write_part_walls_csv=args.write_part_walls_csv,
        write_materials_csv=args.write_materials_csv,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
