# Field preprocessor

This external tool publishes a separately named cache while preserving original
fields, geometry, particle sources, component order, basis, and units. The solver
core never imports or invokes it. Configuration format v2 and producer revision
v4 certify **nodal P1, regular, or affine Q1 source → fully supported regular
target**, for static fields and fixed-topology, linearly interpolated time
snapshots. Unstructured targets, partial regular targets, and warped Q1 sources
fail before publication. There is no legacy sampling-gate fallback.

An affine Q1 cell must be an exact parallelogram in the input binary float64
coordinates: opposite vertex sums agree in rational arithmetic. Even a one-ULP
warp is refused as `validation_unavailable_for_warped_q1`. Warped Q1 would need
certified inverse-map, Jacobian, derivative and integration bounds; those are
not implemented, and coarse/fine sampling disagreement is not substituted for
a guarantee. The runtime can sample warped Q1 directly; this limitation applies
only to cache publication.

The production sampler generates cache node values. Validation streams source
cell/target-cell intersections, so source features cannot disappear between
target-only sample points. Exact rational arithmetic on the input binary float64
coordinates rejects source overlap and target gaps separately; positive slivers
are never discarded by epsilon. Collapsed float64 integration patches are
`unresolved_validation`, not accepted support. A patch that the production
locator cannot distinguish inside its roundoff tie region is also unresolved;
independently rounded near-coincident grid lines can trigger this conservative
refusal. Regular axes already define
nonoverlapping target cells. A nonrectangular or holed source must cover the
entire supported target rectangle; this tool does not fill its missing support.

On each common patch, positive 4×4 Duffy Gauss quadrature integrates the squared
polynomial differences through total degree 5. Values use the production sampler;
true P1, regular and Q1 gradients opt in to the same sampled cell IDs/basis through
`PreparedFieldSet.spatial_gradient`. There is no second locator or inverse map,
least-squares gradient fit, or production-stage gradient allocation.

The three relative L2 gates use physical measures: `dx dy` in XY and `2πr dr dz`
in RZ. Gradients differentiate stored components in the canonical two coordinates,
not a full 3D covariant vector gradient. Boundary values use the target **support
boundary**, with `ds` or `2πr ds`; they do not certify chamber material-wall values.
The RZ axis contributes zero boundary measure. `sample_absolute_max` is a sampled
report, not a certified L∞ bound.

Each report gives the weighted error/reference integrals, estimated relative L2,
and the upper relative L2 used for publication. The upper value includes nodal,
basis-conditioning, coordinate, and accumulation roundoff allowances. Rounded
quadrature coordinates also affect a bilinear gradient; its constant mixed
second derivative supplies the coordinate-displacement allowance. P1 gradients
are constant within each certified source cell and have no such Hessian term.
For affine Q1 the mixed reference derivative is transformed by the same affine
map, with conditioning and coordinate allowances. These bounds use the same
cell's stored nodal values without locating again.
A zero
reference receives zero relative error only when error and its allowance are
both zero. A nonrepresentable integral, unavailable lower reference norm, or an
upper error above the declared limit prevents publication. Tight/zero budgets
may therefore refuse an otherwise nearly exact affine cache. These polynomial
checks do not certify warped Q1.

Time knots and every cache snapshot are retained. Each adjacent pair contributes
three spatial Gram moments for error and reference, so both squared norms are
quadratic over the entire normalized time interval. Publication encloses their
ratio on bounded subintervals using Bernstein convex-hull bounds, propagated
spatial allowances, and an absolute-product accumulation allowance. It does
not accept only the snapshot endpoints. A reference that cancels between
snapshots can therefore reject a cache that would pass both endpoint checks.
Stationary-point roots provide a report-only estimate; their rounded positions
are not the acceptance proof. The three `relative_l2_upper` values are maxima
over all time intervals. Temporal `squared_*_integral` fields are duration-weighted
means of the spatial integrals, including nonuniform time spacing.

`--source-time-diagnostic` additionally reports sensitivity to the supplied time
knots. For each interior knot it omits that snapshot and interpolates its two
neighbors using the actual, possibly nonuniform time spacing. The difference
from the saved interior snapshot uses the same production sampler, common
partition, and weighted value, true-gradient and support-boundary norms. Scalar
and vector components retain their stored basis and units. The spatial scope
is the target support; it is not a whole-source-domain or chamber-wall claim.
Each `source_time_diagnostic.omitted_knots` item records the three times, right
neighbor weight and norms. These are **report-only coarsening sensitivities**,
not cache acceptance limits or a source time-accuracy bound. A zero diagnostic
cannot detect a short pulse missed by every saved snapshot. Continuous-time
fidelity remains `NOT_TESTED`; static sources are `NOT_APPLICABLE`, and two
snapshots are `INSUFFICIENT_SNAPSHOTS`.

Without the option the diagnostic is `NOT_REQUESTED` and performs no additional
sampling. With it, all selected cache norm gates run first. The diagnostic uses
only the remaining geometric-work budget and a separate conservative memory
preflight. `RESOURCE_LIMITED` or `UNRESOLVED` may include completed omissions
and the next unevaluated knot. Neither status vetoes an otherwise certified
cache, nor becomes a diagnostic `PASS`. The report records diagnostic resource
usage separately; the cache proof's resources and limits keep their meaning.

Example configuration:

```yaml
format_version: 2
input:
  data_path: source.h5
validation:
  memory_limit_mb: 128
  workspace_rows: 256
  max_patch_work: 100000000
target:
  kind: regular
  name: fast_grid
  axis0: {start_m: 0.0, stop_m: 0.02, count: 81}
  axis1: {start_m: 0.0, stop_m: 0.01, count: 41}
fields:
  - source: gas_velocity
    output: gas_velocity_cache
    limits:
      value_relative_l2: 0.001
      gradient_relative_l2: 0.01
      boundary_value_relative_l2: 0.001
```

An existing fully supported regular layout may instead be selected with
`target: {kind: existing, layout: target_grid}`. Run from the solver project:

```console
uv run --locked python -m tools.field_preprocessor config.yaml cache.h5 --report cache.json
```

Request the additional saved-time sensitivity report explicitly:

```console
uv run --locked python -m tools.field_preprocessor config.yaml cache.h5 --report cache.json --source-time-diagnostic
```

`validation` is required and has no implicit resource defaults. Workspace rows
must be at least 16. One shared work counter covers AABB rows examined and exact
intersection operations across coverage checks and all selected fields. Budget
exhaustion stops without an output bundle or report. Temporal subinterval checks
consume this same counter. Depth-first temporal refinement retains at most 53
pending intervals and refuses unresolved intervals at depth 52 or float64
midpoint collapse. Reference-zero uncertainty is refused without an epsilon
denominator. Memory preflight occurs
before target/cache allocation; canonical reading uses the same declared numeric
array limit. The report accounts for source/candidate resident arrays, layout/index
and writer transients, all retained cache snapshots, per-interval reports, fixed
sampled/gradient rows, and bounded patch scratch.
This owned-array plan is not a process RSS hard limit. Only one small rational
polygon is retained; no all-pair partition or global quadrature-point array is
built. Original source arrays and all generated cache values remain resident
until canonical publication; bounded scratch does not imply out-of-core data.
Cache validation samples only two adjacent snapshots per spatial side at a time;
the optional source diagnostic samples three source snapshots at a time.

Input YAML uses the shared strict parser: duplicate keys and merge keys fail.
Format v1 is rejected rather than silently retaining its fixed-stencil meaning.
The producer identity is v4 and the saved-knot diagnostic identity is v1.
Validation/support/field-semantics identities remain v3 and the optional
gradient identity remains v2; configuration format remains v2 because its keys
and limits did not change. Canonical
schema and production value-sampler identities are unchanged.

Before publication, selected caches also pass the production required-field
preflight, including coordinate/axis rules and coverage of the particle domain.
A run selects source or cache by explicit field name, without a runtime fallback.
Cache gate completion is distinct from product adoption: compare the enabled
physics with source/cache at the same time step and measure the required
trajectory, charge, arrival/fate, memory, and cost before adopting a cache for
that scientific use. Field-norm agreement alone does not certify those results.
