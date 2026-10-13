# Evaluate evidence and locate the first cause

Read only sections needed for metrics, numerical qualification or diagnosis.
This reference owns comparison procedure; repository authorities own equations,
schema and invariants. It adds no production API, validator, runner or status.
Method/step/resolution planning belongs to
[numerical-plan.md](../../prepare-comsol-field-case/references/numerical-plan.md).

## One decision record, proportional evidence

Use a single comparison manifest/summary or add linked comparison sections to
the existing extraction record. Reuse tables and schemas. A plan may refer to
future evidence; a completed claim must refer to actual artifacts. Keep:

| Record | Decision-relevant content |
|---|---|
| Identity | Actual original/altered MPH, solution/dataset and external dependency identities; candidate case/field/result, executor/evaluator and conversion lineage. New runs add versions, commands, completion and resolved revisions. |
| Scope | Mode; full/slice/companion/analogous intent; compared/excluded physics, particles, time, domains and semantic surfaces; relevant crosswalk and preflight-question flags. |
| Observable | Physical estimand, SI units/fixed scale, released population, weights, common observation times and metric. |
| Valid rows | Stable identity/release alignment, support, finite rows, active/terminal masks, event continuity, resampling/censoring and exclusions with reasons/counts. |
| Qualification | Numerical plan/levels and uncertainty actually available. Confirmatory agreement also needs predeclared budgets and statistical margins. Unknown evidence remains unknown. |
| Investigation | Representative pilot, bounded ordered hypotheses/probes and stop/handoff condition. |

Choose a few observables that answer the chamber question: delivery, wall
arrival, survival/residence, charge evolution or transport. Use
[agreement-and-observables.md](agreement-and-observables.md) for the separate
claims, purpose-specific metrics, denominators and missing-data treatment.
Time-resolved trajectories can themselves be primary transport observables;
mean/final coordinates do not replace them or the relevant physical outcome.
If acceptance budgets are absent, state a provisional target tied to the
objective and give sensitivities. Do not choose thresholds from observed
cross-solver differences. Saved-data exploration remains useful without a
registered target; label it descriptive.

Reused evidence must bind to the actual evaluated bytes. Revalidate changed/
unresolved bindings, including external tables/libraries: an unchanged MPH is
insufficient when dependencies changed. Apply inspection's
[dependency/provenance record](../../inspect-comsol-model/references/extraction-record.md).
The existing v2 preflight binds required/direct items by expected/observed
JSON artifacts, SHA-256 and pointers in seven layers. A copied status is not
observation. For a new gated claim re-open selected artifacts and check actual
source-model, field and run-readback identity. Do not create a generic schema
or require every conceivable readback for a limited descriptive question.

## Numerical qualification differs from raw difference

Before attributing a trajectory difference, check the joint realized source,
including full velocity vector, basis/frame and release time as well as position,
mass, charge, independent radii and weight. Speed-only matching misses direction
errors. Consume preparation's
[initial-conditions.md](../../prepare-comsol-field-case/references/initial-conditions.md).
For independently sampled sources compare conditional distributions and weighted
observables; ID-wise path equality is meaningful only for matched realizations.
Initial thermal sampling and trajectory Brownian forcing are separate mechanisms.

Before implementing nonlinear Brownian dynamics, independently qualify the mean
and covariance rate from the same collision assumptions. The external
`tools/vv/gas_collision` reference is useful for the supported isolated sphere,
Maxwell accommodation, one Maxwellian gas and declared particle/gas temperature
conditions. Read its README and limits; species/heat-flux/nonlinear trajectory/
3-D event validation do not follow from moment agreement. Scalar instantaneous
FDT, PSD or equilibrium variance alone cannot establish high-drift tails or
arrival statistics. The necessary increment/time/velocity scale and collision
sample uncertainty belong to the claim, not a case-independent cutoff.

Consume preparation's independent resolution plan when numerical accuracy is
claimed. Different valid methods/steps can approximate the same equations;
COMSOL saved times are not necessarily internal steps. Dense output alone is
no convergence proof; the candidate need not mimic a poor reference step.

For deterministic qualification consume sufficient independent candidate and
reference levels, normally the plan's three levels. Valid bound qualified
levels can be reused; no native rerun is necessary merely to follow the skill.
Report position, velocity and charge in separate SI norms or predeclared
physical normalization. Interpret order against regularity: interfaces, P1
gradients, model switches and impacts can reduce smooth-problem order.
Exact/roundoff floors differ from failed convergence. Richardson estimates need
demonstrated asymptotic behavior; otherwise fine-pair differences are empirical
sensitivity bands, not certified bounds.

Check event identity, time/location, normal, response and remaining flight
separately. Do not estimate RK order from final coordinates after event mismatch.
End pointwise parity aggregation at first event divergence; report population
fate/arrival differences with the mismatch still visible.

Native FE, exported P1/Q1, regular caches, recovered derivatives and faceted
geometry have distinct errors. Consume spatial/temporal/geometry evidence as
the question requires. Element/facet IDs do not match across refined meshes;
compare semantic surfaces and physical hits. Smaller dt cannot remove field
transfer, interface ownership, recovery, faceting or model bias. Preparation
owns admissible discontinuous-field representation and relevant alternatives.

For stochastic qualification consume the plan's root-step/path-depth pilot
and independent final ensemble. Adaptive event refinement does not prove
sufficient base depth. A noisy two-level population check establishes neither
deterministic order nor continuous SDE/first-passage accuracy. If finite-algorithm
reproduction is intended, explicitly limit the result to that finite policy.

Estimate combined comparison uncertainty from candidate/reference time/event error,
export/resampling error and, for native end-to-end reproduction, field/geometry
representation error. Use a conservative sum unless independence supports
root-sum-square. Give units and a fixed physical scale; distinguish empirical
sensitivity, statistical confidence and certified bounds. This uncertainty is
separate from a fixed purpose acceptance allowance; increasing uncertainty must
not relax that allowance.

Unknown reference uncertainty leaves numerically qualified parity `NOT_TESTED`.
Raw SI differences, descriptive arrival/fate and a supported local diagnosis
are still reportable, and purpose-level observable agreement can be assessed
with a stated limitation. Unknown uncertainty does not license arbitrary tolerances
or prohibit quantitative output. A supported meaning and tested registered
acceptance can yield `FAIL`; an inadequate/unknown reference cannot yield a
qualified core-parity verdict. Same-step stages diagnose a narrow algorithm
compatibility microcase, not independent accuracy or different valid integrators.

## Select essential measurements

### Sources, fields and frozen RHS

Match stable particle/release identity, never CSV row order. Check release time,
realized position/velocity, `mass_kg`, drag diameter, electrostatic/contact radius,
displaced volume, `q = Z e`, weights and relevant equations. Do not reconstruct
mass from diameter or interchange radii. Shared realized sources support
deterministic path comparisons; source-distribution realization is separate.

At identical `(x,t)` compare support/domain independently of values. Select
points that resolve the question: releases, early divergence, dominant-force/
charge extrema, gradients, interface/axis/boundary neighborhoods or time knots.
A pilot need not cover every category. Bind units, basis, temperature-energy
convention, pressure reference, species/effective-gas meaning and one-sided
ownership to preparation's crosswalk. Compare primitives before particle-dependent
Kn, drag rates or coefficients. Mean-free-path and thermal-conductivity definitions
belong to that crosswalk; equal field names do not prove equivalent physics.

At identical `(x,v,Z,t,properties)` compare relevant forces, charge rates,
Brownian/FDT coefficients and total acceleration. Full formula claims cover
all enabled terms; local diagnoses probe only terms needed to falsify their
alternatives. Report component SI differences and direction only for meaningful
nonzero magnitude. Near-zero direction is undefined. Use a fixed physical scale
for relative metrics; matching canceled totals does not prove term parity.

Separate reconstructed configured formula, native per-force readback and native
assembled acceleration/charge RHS. Reconstruction does not prove that a term
was active on the intended selection or assembled into the solved ODE. Missing
assembled RHS remains untested; it is not required for every descriptive
metric. A controlled microcase can resolve it when consequential to the claim.

### Deterministic paths and events

When path parity answers the question, report RMS/max SI and fixed-scale
position/velocity/charge differences at common valid times before event
divergence, early growth and first particle/time/layer beyond budget. Include
whole released-population fate, failures and unobserved counts. Mean coordinates
and final survivors are descriptive subsets.

Keep pending/active/stuck/held/escaped/failed distinct. Held tails are not active
velocity samples; Freeze/hold does not establish adhesion. At event-coincident
times state left/right continuity or an explicit small exclusion. Do not
interpolate across impact/charge discontinuity. Bind resampling method and
uncertainty to the evaluator.

Compare event identity/order and semantic boundary/outcome before time,
center/contact point, normal and pre/post state. Center-wall and finite-radius
contact are different problems; RZ axis crossing is not a wall. Arrival/incidence
differs from capture/deposition. Exercise a positive event for each law in a
boundary-parity claim; zero events leave that law `NOT_TESTED`. Missing native
boundary IDs can support terminal-time/fate observations, but semantic attribution
requires actual IDs or the repository's complete-response-map uniqueness rule.
Do not drop unknown/overlapping groups to manufacture uniqueness.

### Stochastic and analogous outcomes

Prefer full-population survival/fate curves, semantic arrival fractions,
first-arrival CDF, residence time, occupancy or charge distributions. Keep a
small confirmatory set; mark other paths/moments descriptive. Survivor-conditioned
transport is a subset; report terminal categories and censoring. Establish
whether weight means particle count or represented population.

Declare independent sampling units; repeated times are correlated. Shared global
randomness or interacting particles need seed/replica clustering; particle-level
bounds need justified independent noninteracting streams. Copy no historical
seed counts, bins, alpha or margins. Confirmatory agreement uses independent
final samples and predeclared equivalence margins. Failure to detect a difference
is not equivalence; a wide confidence interval failing the gate does not prove
different physics. Equal seed integers do not establish shared RNG paths.

For analogous sensitivity state changed/omitted terms, primitive conventions and
approximation regime. Brackets/ablations are optional when useful. Independent
candidate verification is separate. Subtracting a force from original saved
paths cannot produce a matched slice: charge, drag, fields and events depend on
the altered path. Remove it from both actual equations and recompute the
reference, or report model-form sensitivity. Do not tune another coefficient
to hide omitted physics.

## First-divergence packet and attribution

Keep one packet in the existing record: particle/release ID; suspect time/
interval/layer; artifact and row pointers; matched state/units; support/domain/
boundary/contact meaning; relevant native/canonical primitives and RHS;
earlier layers ruled out; uncertainty and one exact next probe. Preserve known
and unknown evidence rather than a permanent full-stage trace.

| First supported observation | Owner hypothesis and discriminating probe |
|---|---|
| Wrong actual model/selection, release, seed, contact/wall, study or solution | **Configuration**: correct external recipe, read back affected settings; no core patch. |
| Native typed value/ID/unit is correct, raw/normalized/canonical data differ | **Extraction/adapter**: getter/expression, unit/basis conversion, identity/selection and post-import value; re-export smallest artifact. |
| Correct provenance, native FE versus canonical field differs before RHS | **Field production/transfer/representation**: same `(x,t)` support/values, known interpolants and topology/side/recovery/spatial convergence; distinguish producer equations from conversion. |
| Claimed common field, actual sampling/coefficient assembly differs | **Adapter or solver field owner**: prove shared representation/evaluation and independently evaluate interpolant before naming core. |
| Identical state/primitives/parameters, force or charge rate differs | **Equation binding or physics candidate**: configured reconstruction versus assembled RHS; dimensions/limits/original-equation microcase before assigning `physics`. |
| Field/RHS agree, candidate fails analytic evidence/self-convergence | **Integrator/coupling candidate**: public-API reproducer and stage state/time/charge; different valid methods alone are not a defect. |
| Adequate pre-contact paths, first hit differs | **Geometry producer or events candidate**: faceting/contact/normal versus localization, analytic geometry and event refinement. |
| Hit agrees, wall response or remaining flight differs | **Configuration/boundaries/event remainder**: actual law, one collision's independent pre/post state and residual flight. |
| Raw outputs agree, metric/plot does not | **Evaluator/visualization**: small row/event check of masks, joins, weights, continuity and resampling; preserve raw results. |
| Stochastic paths differ or ensemble precision is insufficient | **RNG/statistical/numerical evidence**: actual coefficients/streams, independence, path/root refinement and confidence; no automatic core defect. |
| Missing equation/DOF or verified out-of-regime physics | **Unsupported/model-form limit**: defensible slice/companion/analogy or exact capability request; no silent fallback. |
| Adequate candidate, unknown/insufficient reference resolution | **Reference uncertainty**: needed source-preserving reference level, or raw/descriptive findings with qualified parity `NOT_TESTED`. |
| Alternatives cannot be distinguished | **Unresolved**: smallest missing observation plus usable scoped result. |

Attribution stays a hypothesis until the probe supports it. `SimulationError`
or a particle failure identifies the first rejected invariant, not necessarily
the cause. Inspect extraction/model selection before proposing an applicability
core fix; a conservative certificate limitation differs from observed physical
violation. A proposed core change needs independent analytic/manufactured/
convergence/geometry evidence through the public API and one `AGENTS.md` owner.
Producer-specific changes stay outside core.

Example: verified native and canonical `E_r` have opposite signs at the same
point, and each electric force correctly equals its own `q*E_r`. This supports
a local field-transfer diagnosis. Correct the external transform, recheck the
one primitive/force, then rerun affected paths. Unknown reference accuracy/event
coverage still leaves whole-trajectory parity `NOT_TESTED`. Unauthenticated
supplied numbers yield a conditional finding, not verified native evidence.

## Bound retries and complete the handoff

Default to one representative pilot and at most three discriminating hypotheses
for its first unexplained layer, then reassess. Accuracy levels belong to
preparation; a full force matrix or all historical ensembles are not defaults.
Adjust budget to objective and authorized resources before expensive work.

For each hypothesis record observation, predicted probe outcome, actual result
and supported owner/next action. Change one cause at a time. After an evidenced
fix repeat the affected probe and essential comparison once; broaden only for
new evidence. Unchanged failures without discriminating observations stop
further step halving, coefficient tuning, API calls or repeated prompts.

Stop when the requested scope is decided, needed bindings/equations/primitives/
event meaning are unavailable, no defensible scope answers the question, or
budget reaches an unresolved layer/error floor. Deliver usable results in each
case. Handoff links objective/mode/intent, crosswalk and actual artifacts,
supported/untested scopes, available levels, metrics/budgets, first-divergence
packet and eliminated hypotheses. Give one exact next experiment/export and
owner. A bounded unresolved conclusion can complete a diagnostic request; it
cannot certify a full reproduction that remains untested.
