# Visualization Workflow

Visualization is an operator aid, not a production-output requirement. Standard
solver output should remain compact and suitable for validation runs; animation
is requested only when a report needs it.

## Recommended Report Artifacts

Use these artifacts first:

- `solver_report.json`
- `prepared_runtime_summary.json`
- `wall_summary_by_part.csv`
- `comparison_summary.json` when a reference comparison was run
- `first_step_compare_summary.json`, `first_step_error.csv`, and
  `force_contributions.csv` when first-step parity was run
- `visualizations/graphs/graph_summary.json`
- `visualizations/graphs/02_final_state_bar_and_pie.png`
- `visualizations/graphs/03_final_state_scatter_geometry.png` when position
  columns are available

For COMSOL comparison reports, the preferred animation artifact is a sampled
side-by-side comparison GIF built from sampled solver and COMSOL/reference
trajectories. Full-solver GIFs are secondary debugging aids because they can be
large and slow.

## Graphs

Graphs are safe to run from compact standard output:

```powershell
particle-tracer-export-visualizations --output-dir <solver_out> --modules graphs
```

If trajectory arrays are unavailable, graph export writes compact final-state
plots and records `graph_mode: compact_final_state` in
`visualizations/graphs/graph_summary.json`. When debug/full trajectory artifacts
are present, it writes the fuller trajectory-aware graph set and records
`graph_mode: trajectory_full`.

## Animations

Animations require trajectory output such as `positions_2d.npy` or
`positions_3d.npy` plus `save_frames.csv`. Generate them from a debug run or a
run that explicitly enables trajectory saving.

For large cases, prefer sampled trails and set limits:

```powershell
particle-tracer-export-visualizations `
  --output-dir <solver_out> `
  --case-dir <case_dir> `
  --modules animations `
  --animation-max-particles 1000 `
  --animation-max-frames 180 `
  --animation-downsample-mode uniform `
  --skip-all-particles-animation `
  --animation-progress
```

`--animation-max-particles` caps particles drawn in GIFs.
`--animation-max-frames` caps frames after interpolation.
`--skip-all-particles-animation` writes sampled-trails GIFs only.
By default, optional animation failures are recorded in
`visualizations/reports/visualization_index.json` and do not prevent graphs from
being written. Add `--strict-visualizations` only when debugging visualization
generation itself.

## Validation Policy

Do not make GIF generation a validation gate. If animation cannot be generated,
use compare summaries, first-step artifacts, wall summaries, and graph exports
as the official validation evidence.
