from __future__ import annotations

import argparse
from pathlib import Path

from particle_tracer_unified.io.runtime_builder import build_prepared_runtime_from_yaml, prepared_runtime_summary
from particle_tracer_unified.solvers.solver_entrypoints import run_solver_for_dim


def main() -> int:
    ap = argparse.ArgumentParser(description='Build prepared runtime and run the step-aware prepared-runtime solver.')
    ap.add_argument('config', type=Path)
    ap.add_argument('--output-dir', type=Path, default=None)
    ap.add_argument('--prepare-only', action='store_true')
    args = ap.parse_args()

    prepared = build_prepared_runtime_from_yaml(args.config)
    if args.prepare_only:
        out = args.output_dir or (args.config.parent / 'prepared_runtime_output')
        out.mkdir(parents=True, exist_ok=True)
        import json
        from particle_tracer_unified.core.source_materials import write_source_summary
        (out / 'prepared_runtime_summary.json').write_text(json.dumps(prepared_runtime_summary(prepared), indent=2), encoding='utf-8')
        if prepared.source_preprocess is not None:
            write_source_summary(prepared.source_preprocess, out)
        return 0

    out = args.output_dir or (args.config.parent / ('run_output_2d' if int(prepared.runtime.spatial_dim) == 2 else 'run_output_3d'))
    if int(prepared.runtime.spatial_dim) not in {2, 3}:
        raise ValueError('run.spatial_dim must be 2 or 3')
    run_solver_for_dim(prepared, output_dir=out, spatial_dim=int(prepared.runtime.spatial_dim))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
