# Decide whether particle results are close enough for their use

Use this reference for metric selection and the comparison conclusion.
[evaluation-and-attribution.md](evaluation-and-attribution.md) owns diagnostic
probes and attribution; preparation's numerical plan owns resolution choices.
These procedures are independent of chamber names, particle counts, boundary
numbers, gas species and a particular evaluator. They do not add a schema or
production subsystem. Keep the selected decisions in the existing comparison
record rather than requiring every measurement below.

## 1. State the question before selecting a norm

Name the intended use, population/release cohorts, physical window, region or
semantic surface, compared equations/approximations, and available observations.
An unspecified transport comparison starts with whole-population spatial views,
time-resolved position differences when pairing is justified, and coverage/fate.
Include velocity, charge, arrival or residence when they affect the question.
A field-only model supports field assessment, not particle agreement until
particle results exist. An unsupported third DOF is not removed by plotting
two components.

Keep four claims separate, using the existing statuses only where a defined
test applies:

| Claim | Evidence and allowed conclusion |
|---|---|
| Spatial appearance | Same-space whole-population panels with bound units/geometry, coverage and horizon. State observed transport envelopes, branches or concentration regions as qualitative similarities/differences; do not turn visual judgment into a numeric PASS. |
| Practical observable agreement | The selected physical results and their differences are sufficiently small for the stated use, within independently justified allowances and available sampling/measurement uncertainty. This can be assessed for an explicit analogue or saved results even when strict parity is untested. Scope it to observed quantities and cohorts. |
| Numerically qualified agreement | Meaning-matched problem, independent verification/resolution on both sides, controlled representation/event error and adequate comparison data. The relevant preflight questions and numerical gates apply. A visually close or purpose-adequate result alone does not qualify this claim. |
| Physical validity | Equations/closures are suitable for the intended regime and predictions are checked against independent physical evidence, with uncertainty. COMSOL agreement alone does not provide this evidence. |

`NOT_TESTED` for the third claim does not erase the first two. An outcome
discrepancy can fail its practical criterion without proving a core bug. An
unavailable velocity/charge law leaves that claim untested, not every position
comparison. A plot-only task completes with verified figures; new simulations
or a full audit are not implied.

Use supplied purpose allowances or derive them from an independent decision:
resolvable spatial feature/clearance, arrival-time requirement, allowable
delivery-fraction change, relevant force/charge response or measurement
resolution. Record units, absolute scales near zero, norm, cohort, horizon
and required coverage. A fixed percentage of chamber size can hide a narrow
opening error; select the length scale that matters to the use. Numerical
self-sensitivity, export error and a practical acceptance margin are different
quantities. A larger uncertainty does not automatically make a result adequate.

With no defensible allowance, report measured SI differences and coverage plus
a scoped qualitative judgment. Do not invent a universal millimetre/percentage
threshold, enlarge a margin to fit the discrepancy, or call already inspected
data a blind confirmation. A user's acceptance of one bundle is useful context
for that view, not another model's tolerance. Exploration can guide a later
fixed criterion; disclose selection on viewed data.

## 2. Decide what can be paired

For deterministic path error, require an evidence-backed joint realized source
and identity map: scope/member, release/cohort, position, vector velocity/frame,
release time and relevant property/charge authorities. Equal ID integers across
files, equal speeds, nearest paths or equal seeds do not establish pairing.
Compare initial differences first and retain them in the conclusion. If the
source differs intentionally, a same-ID response sensitivity can be reported
as such, but it is not same-problem integration error.

Different stochastic paths or independent release samples are compared by
population distributions, weighted outcomes and uncertainty. Individual paired
plots can illustrate a path only when its pairing is justified. A deterministic
analogue versus a Brownian reference can have similar transport trends; one
native realization does not establish equivalence of population laws.

Align physical time and frame explicitly. Absolute time is usually needed for
time-dependent fields; time since release is useful for a stated cohort-age
question. Do not shift, rotate or time-warp paths to optimize agreement. A known
coordinate transform is an input conversion with evidence, not fitted alignment.
If grids differ, use already observed common times or a documented resampling
with error/continuity limits. Preserve raw observations. Never interpolate
across release, impact, charge jump, disappearance or a missing span.

## 3. Choose a small set of observables

Coverage is always reported; other rows are selected by purpose. Keep primary
decision metrics separate from exploratory diagnostics.

| Purpose | Useful measurements | Meaning that must accompany them |
|---|---|---|
| Transport path | Whole-population side-by-side spatial view; timewise and per-particle position RMS/quantiles/max; vector/component velocity differences; first material/event divergence | Spatial shape versus timing, paired cohort, active/pre-event or stated post-event segments, contributing IDs/weights at every time; all mismatched fates remain visible |
| Charging or force-driven deflection | Z or q histories, sign/equilibrium/transient differences and relevant v/x response; matched-state primitive/RHS probes for diagnosis | Charge units/authority, live versus held auxiliary state, coupling; a matched total force can conceal canceled term errors |
| Delivery, capture or escape | Released-population fractions and transition curves by semantic destination/outcome; first-arrival time CDF/quantiles and hit-location distributions | First arrival differs from repeated incidence and from capture. Freeze/hold is not proof of adhesion; unknown Disappear is not automatically escape |
| Residence and occupancy | Restricted active residence over a fixed horizon; time-weighted spatial occupancy; survival/active fraction with other fates | Entry/exit/reentry and selected region, active time versus time held on a wall, dimensional measure and pending/terminal handling |
| Stochastic or unpaired populations | Appropriate position/velocity/charge moments or distributions, fate/delivery probabilities, arrival distributions and uncertainty | Independent sampling unit, weights, replica dependence, physical source/noise law; repeated observation times are correlated |

For a fixed sparse observation grid, spatial envelope or a stated shape
distance can support geometric similarity, but discarding time loses transit
speed and arrival timing. A Hausdorff/nearest-curve distance is optional shape
evidence, never a replacement for timewise paired error or destination outcomes.
Mean endpoints or a dense-looking bundle alone can hide a minority taking a
wrong branch. Show relevant quantiles/outliers and count the affected particles.
Do not demand exact same facet IDs across independently refined meshes; compare
the physical hit and semantic surface using verified maps.

## 4. Make aggregation and missingness explicit

For a valid paired row define `d_i(t)=norm(x_candidate-x_reference)` in metres.
At each time, a weighted RMS is `sqrt(sum(w_i*d_i^2)/sum(w_i))` over that stated
valid cohort. Report its coverage and denominator separately from the full
released population. Use equivalent separate definitions for v and Z/q; do
not mix metres, m/s and charge in one unscaled state norm. Relative position
error based on absolute distance to the coordinate origin is translation
dependent; use a fixed physical length or declared release displacement scale.

Select equal-observation or time-integrated aggregation explicitly. The current
deterministic evaluator uses equal retained particle-observation rows; a
nonuniform clock can overemphasize densely sampled intervals. For a time-RMS,
integrate squared error with declared physical time weights, only across valid
continuous segments, and divide by the contributing duration/weight. This is
a quadrature over observed data, not proof of unsaved continuous-path accuracy.
Report unobserved durations; do not bridge gaps or replace missing error with
zero. A zero contributing denominator is unavailable, never zero error/PASS.

List planned/released/pending IDs, matched/unmatched IDs, valid rows/durations,
failures, observed terminal categories and unknown/missing states. At time t,
a cumulative delivery fraction uses the specified released cohort/weight,
not only surviving finite trajectories. If sources continue releasing, say
whether the denominator is the complete fixed cohort or those released by t;
do not change it silently. Repeated hits count incidences, not distinct arriving
particles. Disclose source/fate subgroups when a pooled value hides a relevant
difference; do not search many subgroups for a favorable gate.

For an unambiguous fixed released cohort, bound an outcome fraction by
`observed_outcome_weight/total_weight` and
`(observed_outcome_weight+unknown_weight)/total_weight` when unknown outcomes
could all fall into that category. If this range crosses the acceptance
criterion, report insufficiency rather than dropping unknowns. A numerical
failure cannot be silently treated as a physical escape or independent censoring.

An unconditional first-arrival CDF means the fraction of the stated released
cohort that has arrived by time t; it need not reach one. Quantiles conditional
on arrival have a different denominator and may hide lost particles. Distinguish
administrative end-of-run censoring, competing terminal outcomes, unknown
removal, and sparse event-time brackets. Do not apply noninformative-censoring
estimators to failures or competing fates without the assumptions they need.
Saved status transitions can bound an arrival interval; they are not exact
native event times.

Restricted residence is the integral of the declared in-region active
indicator up to a fixed horizon, with release and terminal conventions stated.
A histogram of saved point counts is not residence on a nonuniform clock.
RZ meridional area and revolved volume/surface measures differ (`2*pi*r`).
Computational counts/normalized fractions are useful when physical weights
are unknown, but do not supply physical concentration, deposited mass or flux.
Bind source rate/exposure and appropriate measure before comparing those units.

## 5. Assess precision and decide what to improve

For stochastic metrics declare independent sampling units and uncertainty of
the difference, with corresponding weights/replica clustering. A deterministic
fixed population can be described exactly as sampled; it is not automatically
an independent random sample of a physical population. Do not multiply sample
size by saved times. An insignificant difference is not equivalence. For a
confirmatory equivalence claim use fixed purpose margins and a justified
confidence/equivalence procedure, including simultaneous scope when several
times/metrics define the gate. Require its justified difference interval to lie
inside the fixed acceptance region; an interval crossing a margin is insufficient
evidence. Do not accept by adding uncertainty to the margin. Empirical step
sensitivity is not automatically such an interval. A wide interval is inconclusive, not evidence
of different physics. Weighted samples need suitable uncertainty calculations;
equal-weight binomial formulas are not a general replacement.

For numerical qualification consume independent temporal/event and relevant
field/geometry/path-depth studies, plus independent verification. Unknown
reference resolution leaves that qualification untested; available practical
differences still stand with that limitation. Empirical fine-pair sensitivity
is not an absolute error bound. Smaller dt cannot cure different field meaning,
initial vectors, absent DOF or a model-form error floor.

Deliver the conclusion first: which transport/outcome aspects are close or
different, in what units/scope, and whether that changes the intended decision.
Then give coverage/precision, model differences and the separate untested
claims. Keep a compact row per requested claim with evidence, criterion and
status or qualitative judgment. Do not collapse everything into one PASS.

Investigate a failed or consequential criterion through the first-divergence
procedure. An improvement proposal states the supported cause/owner, proposed
change, expected observable benefit, independent analytic/manufactured or
geometry check, and affected-case before/after comparison with unchanged
criteria. Compare one changed cause at a time and retain model differences.
External extraction/units/layout/field production stay outside core; a physical
or numerical core change needs evidence independent of COMSOL resemblance.
If criteria hold, no invariant fails and remaining differences are explained
by declared approximations/uncertainty, a scoped acceptance with limitations
is a useful completion. Do not start an endless parameter-tuning campaign.

## Primary guidance and limits

- [NASA-STD-7009B](https://standards.nasa.gov/standard/NASA/NASA-STD-7009)
  supports intended-use acceptance criteria and separate credibility evidence;
  its NASA organizational approval process is not a workflow requirement here.
- [COMSOL time stepping](https://www.comsol.com/support/knowledgebase/1254)
  explains why requested output times and actual solver steps differ.
- [COMSOL boundary theory](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_math.06.38.html)
  defines the native Freeze/Stick/Disappear distinctions; inspect the selected
  interface/custom law and version before applying them.
- [COMSOL domain accumulator](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_math.06.15.html)
  distinguishes count/density and elements versus elements-and-time; its
  exported quantities need their actual selected normalization and source.
- [NIST censoring](https://itl.nist.gov/div898/handbook/apr/section1/apr131.htm)
  describes censored observations. It does not justify treating numerical
  failures as independent censoring in a particle comparison.
