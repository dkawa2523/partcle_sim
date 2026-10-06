---
name: visualize-particle-trajectories
description: Create trustworthy spatial plots and animations for chamber-particles results, including geometry-overlaid trajectories, boundary events, many-particle summaries, and same-space COMSOL comparisons. Use when an agent must visualize a solver run or V&V result without changing or recomputing the simulation.
---

# Visualize particle trajectories

Produce derived, inspectable figures from completed artifacts. Visualization
must not change physics, rerun trajectories implicitly, or become a dependency
of the production engine.

## Establish the view

Read the repository guidance and inspect the result with `open_result` or
`chamber-particles inspect`. For a geometry overlay, require the corresponding
case YAML/HDF5: a result stores case hashes, not the geometry itself. Load the
case with the public `load_case` interface and confirm its hashes match the
result. For a comparison, use normalized COMSOL tables prepared by
`$validate-comsol-trajectories`.

Confirm coordinate system and SI units, simulated time range, particle-ID
correspondence and release times, lifecycle/event meanings, and any rendering
sampling. Write every figure to a derived output directory outside the source
result. Use `ResultView`, the public case interface, or normalized tables; do
not import private engine modules.

## Choose the smallest useful output

For a normal run, make a physical-space plot with chamber boundaries behind
the trajectories:

- use `r` horizontally and `z` vertically for axisymmetric cases, otherwise
  `x` and `y`;
- keep equal geometric aspect and label units;
- show starts, terminal states, and boundary-event locations distinctly;
- distinguish boundary groups/materials when that answers the question;
- use stable color meaning such as time, fate, source group, or particle ID.

For many particles, avoid an unreadable opaque bundle. Use light trajectory
lines plus the most useful density/occupancy, deposition-by-boundary, hit map,
or arrival-time summary. Rendering may select a reproducible set of particle
IDs, but computed summaries must use the full requested population and the
sampling rule must be reported.

For deterministic COMSOL comparison, align stable particle IDs and common
observed times. Draw candidate and reference in the same coordinate frame,
geometry, axis limits, aspect, and units. Prefer paired overlays; add
error-versus-time or event summaries only when they answer the evaluation
question. Do not present independently auto-scaled panels as proof of agreement.

For stochastic/Brownian comparison, show ensemble distributions, occupancy,
fate/arrival curves, and confidence intervals. Individual paths are illustrative
and are not a pathwise agreement gate.

For numerical evaluation, select only relevant plots: `h/h2/h4` difference or
order, mesh convergence, position/velocity/charge or force histories, boundary
and fate counts, or ensemble confidence bands. Do not claim an observed order
from only two resolutions.

## Reuse existing external tools

Run from `particle_platform_redesign/solver/` with uv.

- `uv run --locked python -m tools.visualization RESULT --output-directory
  DERIVED` creates the current lightweight one-particle SVG and boundary-event
  view. It does not currently provide geometry overlay or a general
  all-particle renderer.
- `uv run --locked python -m tools.analysis RESULT --output
  DERIVED/summary.json` creates the current full-population summary.
- `tools/vv/comsol/plot_matched_trajectories.py` renders normalized solver/
  COMSOL R-Z overlays when its input format fits.

If the requested geometry overlay, population plot, XY comparison, field view,
or animation is not supported, extend the single external visualization path
or add a narrow task-specific renderer. Do not build a generic plotting
framework or add plotting libraries to the solver runtime for one figure.

## Animation rules

Stream saved `iter_frames()` data rather than retaining all frames; use probes
when only selected particles are needed. Keep geometry, axis range, aspect,
color scale, and frame clock fixed. Respect release times and distinguish
pending, active, stuck/held, escaped, and failed particles. Never recompute an
unsaved trajectory or interpolate across a release, boundary event, or missing
escaped suffix. If visual interpolation is used between safe saved states,
label it as visual only. Provide a static companion figure.

An R-Z result is a meridional 2-D trajectory, not a reconstructed 3-D orbit;
do not draw the symmetry axis as a wall unless it is actually a material
boundary.

## Verify and deliver

Open at least one representative rendered frame or static figure and verify
geometry alignment, coordinate orientation, complete requested particle/time
coverage, event placement, legends, and readable scale. Prefer SVG/PNG for
static output and HTML/MP4/GIF only when animation is requested and available.

Deliver the files with a short receipt giving source hashes, tool revision,
coordinates/units, axis limits, particle/time coverage, sampling, and known
limits. A visualization is evidence presentation, not a substitute for numeric
comparison or proof of general COMSOL equivalence.
