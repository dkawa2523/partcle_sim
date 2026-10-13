---
name: visualize-particle-trajectories
description: Present saved particle trajectories, geometry, events and population outcomes, including all-particle solver/COMSOL panels side by side. Use for trustworthy spatial comparison or animation without changing or recomputing the simulation; validation owns agreement judgments.
---

# Visualize particle trajectories

Produce derived, inspectable figures from completed artifacts. Visualization
must not change physics, rerun trajectories implicitly, or become a dependency
of the production engine.

## Establish the view

Select the smallest output implied by the request; a plot-only request does not
start inspection, simulation, or a new comparison campaign.

| Scope | Minimum input | Deliverable and completion |
|---|---|---|
| Spatial paths/events | Result and matching case geometry; normalized reference only for a requested overlay | Same-space figure with starts, valid path segments, outcomes and observed events; complete after visual QA |
| Population or evaluation figures | Saved population/states/events with identities, coverage and the relevant metric definition | Primary arrival/fate/occupancy distribution and its denominator/weight meaning; report quantities that are unobserved instead of recomputing them |
| Animation | Saved frames or probes, matching static geometry and requested time/particle coverage | Streaming animation plus static companion and sampling receipt; no reconstruction of unsaved simulation states |

Read the repository guidance and inspect the result with `open_result` or
`chamber-particles inspect`. For a geometry overlay, require the corresponding
case YAML/HDF5: a result stores case hashes, not the geometry itself. Load the
case with the public `load_case` interface and confirm its hashes match the
result. For a comparison, use normalized COMSOL tables prepared by
`$validate-comsol-trajectories` or reuse already verified normalized exports.

Confirm coordinate system and SI units, simulated time range, particle-ID
correspondence and release times, lifecycle/event meanings, and any rendering
sampling. Write every figure to a derived output directory outside the source
result. Use `ResultView`, the public case interface, or normalized tables; do
not import private engine modules.

Reuse a matching saved record or normalized table; invoke another skill only
for the missing fact the requested figure needs. Figure settings and its short
receipt can be sections of the existing comparison record, rather than a new
report bundle. A short request such as 「この結果の軌跡をチャンバー形状と重ねて」
needs a spatial figure; it does not require a convergence report.

For an evaluation, read the registered primary observable, scope, row coverage,
weights, acceptance budget, and first unresolved discrepancy from the comparison
manifest when available. Use
[evaluation-and-attribution.md](../validate-comsol-trajectories/references/evaluation-and-attribution.md)
to choose evidence that answers that question. Native final status reused as
timewise status and unobserved boundary identity cannot become semantic
deposition evidence through a convenient mask. If physical weights are unknown,
computational particle counts/fractions remain useful with a stated cohort and
denominator; physical concentration, deposited mass or flux remains unqualified.

## Choose the smallest useful output

For a normal run, make a physical-space plot with chamber boundaries behind
the trajectories:

- use `r` horizontally and `z` vertically for axisymmetric cases, otherwise
  `x` and `y`;
- keep equal geometric aspect and label units;
- show starts, terminal states, and boundary-event locations distinctly;
- distinguish boundary groups/materials when that answers the question;
- use stable color meaning such as time, fate, source group, or particle ID.

For an all-particle comparison, show the solver's full trajectory bundle on
the left and the reference's full bundle on the right, per case, unless the
user requests another arrangement. Use shared limits, equal aspect, units,
time horizon and line width/opacity/color meaning. Show all requested IDs;
do not replace this with selected paths, one mixed overlay or a per-ID gallery.
Use light lines and add a population summary only when it answers the question.
For other large-population requests a reported reproducible sample may help;
computed summaries still use the full requested population.

For chamber delivery/deposition, prefer arrival fractions and time distributions
by the actual semantic boundary, with released-population denominators, failed
and unobserved coverage, and stated weight meaning. For transport diagnostics,
use geometry-overlaid paths and the earliest relevant field/force/event
discrepancy. Mean final coordinates over only finite survivors can hide removal
or failed trajectories and are descriptive, not a primary agreement gate.
RZ occupancy needs a stated measure: meridional area and revolved volume
(`2*pi*r` measure) differ. Do not turn computational particle counts into
physical concentration without the source weight and normalization.
Keep surface arrival/incidence distinct from physical capture: Freeze/hold is
a numerical contact outcome and does not itself prove adhesion. State whether
counts represent computational particles or weighted physical particles.

For a paired deterministic comparison, align verified particle/release identities
and common physical observation times. Draw both in the same coordinate frame,
axis limits, aspect and units. Use whole-population panels for the overall view
and selected-ID overlays/time histories for diagnosis. If geometries differ,
show each bound geometry in that frame and identify transfer/faceting changes.
A shared canonical outline is a plotting reference, not proof of native geometry
equality. Do not present independently auto-scaled panels as agreement evidence.

For stochastic/Brownian comparison, show ensemble distributions, occupancy,
fate/arrival curves, and confidence intervals. Individual paths are illustrative
and are not a pathwise agreement gate. Whole-population spatial panels remain
useful illustrative views; different realizations or particle counts need their
own labels and cannot be compared by line density as physical concentration.

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
- `uv run --locked python -m tools.vv.comsol.plot_matched_trajectories` renders
  normalized solver/COMSOL XY or RZ overlays, including canonical geometry
  and supplied precontact flags. It expects identical ID sets and has no
  event-ledger or general unpaired-population evaluation. Inspect its CLI and
  CSV contract; a narrow side-by-side renderer may use the same bound tables.

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

The current external `tools.vv.comsol.plot_matched_trajectories` can bind a
canonical geometry overlay and an explicit per-row `precontact_eligible`
column. Read its current CLI before invoking it. When supplied, numerical
metrics use both participants' eligible finite active rows while observed
paths remain visible. Without such evidence label common-active metrics as
descriptive; do not call them precontact agreement. Preserve native disappearance
with unknown cause separately from candidate escape/failure and break paths
at missing observations.

Normalize omitted observations onto the expected saved clock as explicit
missing rows before using a line renderer, or split segments using a recorded
gap rule. An absent CSV row is not automatically detected as a gap by the
current plotter. Preserve the raw export and report this normalization.
Static polylines connect saved positions; they do not reconstruct a curved
flight or an unsaved bounce. Use available events to split paths/mark hits;
when these are unavailable, label this limit. A chord crossing a wall in such
a figure is a reason to inspect sampling/events, not proof of a wall defect.

Open at least one representative rendered frame or static figure and verify
geometry alignment, coordinate orientation, complete requested particle/time
coverage, event placement, legends, and readable scale. Prefer SVG/PNG for
static output and HTML/MP4/GIF only when animation is requested and available.

Deliver the files with a short receipt giving source hashes, tool revision,
coordinates/units, axis limits, particle/time coverage, sampling, and known
limits. A visualization is evidence presentation, not a substitute for numeric
comparison or proof of general COMSOL equivalence.
