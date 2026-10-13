# Plan resolution independently of COMSOL output

Read the selected solver revision in
[numerics.md](../../../../particle_platform_redesign/solver/docs/numerics.md),
[engine.py](../../../../particle_platform_redesign/solver/src/chamber_particles/engine.py),
and [integrators.py](../../../../particle_platform_redesign/solver/src/chamber_particles/integrators.py).
Some document sections are historical. The currently checked method gate and
resolved `run.json` are the evidence for the executed numerical policy.

Apply only the requested scope. Planning/preparation records the proposed method,
pilot step, scales and unmeasured budgets. A smoke records actual preparation,
execution and failures; it needs no three-level convergence campaign. Run at
least three genuine integration resolutions when estimating self-convergence
or qualifying general time-resolution accuracy, and the applicable depth/
ensemble studies for a stochastic numerical claim. A narrowly exact or analytic
reference-backed case may justify its accuracy without an artificial three-run
matrix; identify that scope and evidence. Visual/practical saved-data agreement
does not require numerical qualification. An operational run without adequate
independent evidence remains resolution `NOT_TESTED`, while its measured
trajectories/outcomes can still be assessed for the intended use.

## 1. Keep the time axes separate

Record these independently in `numerical_plan.json` or a short equivalent table:

| Quantity | Meaning and authority |
|---|---|
| Run start/end | Physical experiment duration, including nonzero time origin |
| Particle release times | Realized canonical source rows; may lie inside a macro step |
| Solver `time.dt_s` | Candidate fixed macro grid, chosen for its equations and field variation |
| Field snapshots | Exported background time resolution; fixed topology and linear interpolation; internal snapshot knots split the macro partition |
| Requested observation times | Output/probe and comparison sampling; accepted proposals evaluate state without changing production integration |
| COMSOL requested output times | Saved/observed solution schedule; not evidence of every internal integrator step |
| COMSOL solver controls | Actual time-stepping method, tolerances, max-step/step policy and, when available, actual accepted times; native source controls remain a separate refinement axis |
| Worker count | Execution resource through CLI/API; no physics or time-step role |
| Event localization/refinement | Numerical first-hit tolerance and path subdivisions; separate from saved output and physical wall parameters |
| Brownian root and tree resolution | Actual root intervals, uniform base depth and candidate-only maximum depth; do not identify a saved frame with a stochastic root |

The source and candidate can use different numerical methods and internal
steps. Compare at common physical observation times after each system is
independently resolved. Matching an output spacing does not produce equivalent
ODE accuracy. A parameter sweep is not a time axis; do not create snapshot
times from row order.

## 2. Select the integrator from the mapped equations

| Integrator | Appropriate use and limits |
|---|---|
| `rk4_fixed` | Smooth nonstochastic coupled motion/charge with acceptable explicit stability cost. Fields, force, and charge are evaluated at the actual stage positions/times. Ballistic or proven constant-acceleration cases can resolve to an exact specialization internally. |
| `exponential_midpoint` | Linear relaxation or locally frozen relaxation dominates. The drag update is exact for frozen coefficients; source/midpoint evaluation controls the error when coefficients or additive force vary. Current charge-stable affine exponential coupling handles continuous charge in the same proposal; do not apply an obsolete charge-only prohibition from historical P11 text. |
| `ou_langevin` | Only with `inertial_langevin_fdt_epstein_linear_midpoint_2d_v2` and compatible linear Epstein. Midpoint-frozen joint OU position/velocity plus coupled charge uses a finite-depth Hermite numerical path. Noise is a stochastic model choice, not a stabilizer for deterministic physics. |

Never substitute adaptive stepping, an unregistered integrator, charge-only
subcycling, frozen-charge history replay, or an equilibrium charge replacement
without registering the resulting physical/numerical change. The production
engine remains the same.

The following source preparation gates were checked for this skill revision.
Verify them against the active engine/runtime and selected model before use;
they are not portable constants to impose on another solver revision:

- RK4: `dt * drag_velocity_lipschitz < 2.5` and, when charge evolves,
  `dt * charge_lipschitz <= 0.5`.
- Exponential midpoint: these explicit drag/charge gates are not its stability
  constraints. Invariant, finite arithmetic, support, applicability, and
  midpoint approximation checks still apply.
- OU: `gamma * dt <= 1e6` and every root uses positive finite `gamma*h`; valid
  coefficient/charge invariants and compatible physics are also required.

These are acceptance/stability gates, not desired accuracy, default steps,
or permission to use their limiting value. For finite-speed drag the velocity
Jacobian bound is not merely the frozen linear relaxation coefficient.

For a fixed-step accuracy problem, compare an existing supported integrator
when its stability/cost permits before adding a controller or reducing h without
limit. Keep the coupled source/field/force/charge/wall problem fixed. A higher
formal order does not guarantee higher accuracy through P1 derivative jumps or
impacts; measure x/v/Z and event prefixes. An exhausted field/model/reference
error floor is a handoff condition, not a reason to tune a physics parameter.

## 3. Choose a defensible pilot step

Calculate characteristic scales from the actual particle properties, field
range, and intended trajectory envelope. Keep the estimates and units in the
plan. Useful examples are:

- Relaxation `tau_v=1/L_v` from the selected drag velocity Jacobian bound; for
  linear drag `gamma=beta/mass_kg`, so `tau_v=1/gamma`.
- Charging `tau_Z=1/L_Z` and the time for a material change in charge, using
  actual current/rate and a declared charge scale. Near equilibrium `Z/G`
  alone is misleading; use rate derivative and relevant transient scales.
- Field advection `tau_adv=L_field/V_bound`, where `L_field` resolves physically
  important changes in flow, E, thermal/plasma primitives and narrow geometry.
  A local cell width is a cautious pilot proxy, not a universal CFL requirement.
- Background variation from the actual time-dependence and snapshot spacing;
  sampling more particle output frames cannot repair coarse field snapshots.
- Curvature/turning and event scales, for example `V_scale/A_bound` or
  `sqrt(L_scale/A_bound)` when acceleration sets the motion from rest. Use a
  physically stated nonzero scale; do not add arbitrary epsilon denominators.

Select an initial step comfortably inside applicable explicit stability gates
and small enough to sample the changing coefficients and event path. An
explicitly declared pilot fraction such as 0.05 of the fastest relevant
accuracy timescale is a starting heuristic only. Verify it by measured
refinement; do not preserve it as a chamber default. For exponential/OU methods
the exactly treated frozen relaxation need not set the accuracy step by itself,
but coefficient variation, charge/force coupling, and boundary statistics do.

If conservative source preparation rejects a physically plausible case,
inspect whether the issue is a true regime violation, unresolved envelope, or
input resolution. A smaller step may improve the envelope; it cannot turn an
invalid model into a valid one. Record the reason and owner rather than changing
thresholds or physical parameters to force a PASS.

## 4. Deterministic self-convergence

Register observables and SI error allowances before inspecting differences.
For trajectory comparisons these usually include position, active velocity,
charge where relevant, and boundary outcome/time/location. Derive tolerances
from the scientific question and declared input/reference accuracy, not the
size of the discrepancy. Define near-zero handling with an absolute SI scale.

1. Keep canonical fields, geometry, initial states/releases, model revisions,
   boundaries, output times, and observation windows fixed. Run independent
   cases at `h`, `h/2`, `h/4` in fresh output directories; retain hashes and
   all failure/lifecycle counts.
2. Confirm these are different integration resolutions. Inspect
   `time.field_snapshot_splits_s`, `counts.macro_steps`, and where available
   `event_refinement.accepted_particle_pieces` / `refinements`. A changed
   nominal h that produces the same intervals because of dense snapshot knots
   or event subdivisions is precision repeatability evidence, not a new
   convergence level.
   Record actual interval ranges/counts, including release-local intervals,
   field knots and event subdivisions when they affect the observable. For
   Brownian experiments also record the generated root intervals and numerical
   tree policy where available. Treat a missing diagnostic as unavailable;
   do not invent a readback or infer root count from output frame count.
3. Compare aligned states at fixed physical times in the same lifecycle
   domain. Separate pre-event motion from event timing/response. Do not
   evaluate active velocity against a source's held/frozen/undefined velocity.
   Track particles with differing event identities explicitly instead of
   dropping them from the error summary.
4. For a smooth deterministic observable define
   `D_h=norm(Q_h-Q_h/2)` and `D_h/2=norm(Q_h/2-Q_h/4)` in a stated norm.
   If differences are resolved above roundoff and monotone, report
   `p_obs=log2(D_h/D_h/2)`. Richardson error estimation
   `D_h/2/(2^p-1)` requires demonstrated asymptotic behavior; label its assumptions.
   Smooth RK4 commonly approaches order 4 and smooth exponential midpoint
   order 2, but event, field-knot, or nonsmooth model transitions need their own
   convergence evidence. Do not force an expected order onto an event CDF.
5. Accept the finest step only if the observable changes fit the registered
   numerical budget and event/lifecycle conclusions are stable in the required
   scope. If the result is roundoff-limited or exact-specialized, state that
   directly. Refine selectively only when it resolves an observed uncertainty.

Refine COMSOL independently using its actual tolerance and accepted/max-step
controls. Do not merely add saved output times and call it time refinement.
Reuse identical field/physics/releases across its refinement levels. If only
one stored native solution is available, candidate self-convergence is still
useful, but source numerical error is `NOT_TESTED`; do not certify a two-sided
agreement whose reference uncertainty is unknown.

## 5. Geometry and field convergence are separate experiments

First assess particle time convergence on one fixed canonical input. Then,
when the accuracy question requires it, refine source/export spatial mesh,
curved `line2` approximation, field snapshots, or gradient recovery with an
already resolved particle step. Change one numerical source at a time and
record its own hashes. A direct P1 export and a native high-order FE solution
are different interpolation models even at the same source nodes.

Use primitive errors at matched off-node points and the resulting change in
the registered observables to assess field uncertainty. For wall geometry
record displacement/normal errors and hit timing/location/outcome changes.
Common-field/common-geometry companions isolate the particle integrator;
native-field comparison includes field representation error. Do not credit a
smaller particle step with correcting the latter.

## 6. Brownian accuracy and uncertainty

The current noise model uses inertial FDT OU with midpoint-frozen coefficients.
Its Hermite path at finite depth is not exact continuous OU first passage and
does not certify zero missed wall crossings. Different RNGs/seeds between
COMSOL and candidate do not define matched stochastic trajectories.

Treat four controls independently:

- Macro `h` resolves changing coefficients and coupled force/charge.
- `interval_tree_depth=D_base` resolves the numerical path uniformly for every
  stochastic root. Test at least three depths when wall/axis arrival matters.
- `adaptive_max_depth=D_max` refines only wall/axis/uncertain candidates.
  Current bounds are `0<=D_base<=D_max<=10`; omission sets max to base.
  Increasing max does not replace base-depth convergence or reproduce a
  globally uniform max-depth path.
- Ensemble N, seed batches, and particle weights control sampling uncertainty.
  Record all of them. Do not call a reduced population a time refinement.

Use a small crossed pilot, for example three h levels at a fixed sufficient
base depth and three base depths at the finest pilot h, to distinguish sources
of error. Keep the physical model and release distribution fixed. Choose
depth/N from measured changes, confidence widths, failure counts, and memory
plan, not a stored reference-case value. If an active wall or RZ axis occurs,
the current method commits the prefix and starts a fresh stochastic root after
the response; do not externally reuse or reflect an unused root tail.

Compare distributions relevant to the physics: position/velocity means and
covariances, charge statistics, outcome probabilities, and first-arrival CDF
or quantiles. Report temporal/depth changes separately from Monte Carlo
confidence intervals and reference uncertainty. Deterministic paired runs with
noise disabled are a separate supported question, never validation of the
noise-enabled source. Weighted outcomes require corresponding uncertainty
estimates; simple binomial formulas only apply to independent equal-weight
samples. Use common random numbers for candidate sensitivity only when the
current RNG/root policy justifies the coupling; same seed across different
algorithms is not the same Brownian path.

## 7. Stop with a scoped result instead of an unbounded tuning loop

For general time-resolution qualification by self-convergence, prepare a bounded
run matrix: initial three genuine
resolutions, the scientific budget, and the next refinement that would resolve
a demonstrated uncertainty. Planning and smoke stop at their scoped artifacts.
After each batch record PASS/FAIL/NOT_TESTED for numerical resolution separately
from semantic mapping and external agreement. If more levels do not clarify
the responsible layer, hold the finest valid artifacts and hand off the
unresolved discrepancy to
[validate-comsol-trajectories](../../validate-comsol-trajectories/SKILL.md).
Do not keep changing physical parameters, tolerances, plots, or evaluators
until a favorable curve appears. A justified convergence limit and the
specific missing evidence are acceptable conclusions.

Three declared step sizes are not three actual refinements if the physical
window executes the same partition (for example every step exceeds the run
span and all runs take one macro step). Check completed manifests and existing
time/field-knot evidence. Distinct labels or files cannot supply this evidence.
Select each participant's empirical self-sensitivity allowance independently
of cross-solver differences, using the physical/numerical objective. Missing
or exceeded allowances leave resolution unqualified even when finest results
agree. Passing those checks is an empirical resolution statement, not an
absolute error bound or proof of an asymptotic convergence regime.
