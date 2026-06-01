from __future__ import annotations

import argparse
import json
from pathlib import Path

from comsol_particle_export.release_alignment import compare_release_tables


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare solver particles.csv with a canonical COMSOL release table.")
    parser.add_argument("--solver-particles-csv", type=Path, required=True)
    parser.add_argument("--comsol-release-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    summary = compare_release_tables(args.solver_particles_csv, args.comsol_release_csv, out_dir=args.out_dir)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
