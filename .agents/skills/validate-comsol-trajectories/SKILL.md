---
name: validate-comsol-trajectories
description: Reproduce and evaluate a specified COMSOL particle-tracing case with chamber-particles. Use when an agent must inspect an MPH containing fields and particle trajectories, make physically equivalent solver settings, extract time-resolved COMSOL states and events, compare matched trajectories, and issue a scoped V&V conclusion without coupling COMSOL logic into the solver core.
---

# Validate against a COMSOL particle-tracing case

Build a meaning-matched external comparison. COMSOL is reference evidence for
the stated case, not the product architecture or a reason to add special cases
to the production engine.

## Fix the comparison meaning first

Read the repository guidance, `particle_platform_redesign/vv_methodology.md`,
`particle_platform_redesign/solver/docs/physics_models.md`,
`particle_platform_redesign/solver/docs/support_and_errors.md`, and only the
relevant workflow in
`particle_platform_redesign/solver/tools/vv/comsol/README.md`. Write one compact
comparison-conditions table:

- source MPH hash and COMSOL version; component, study, solution, and dataset;
- geometry, mesh/field source, coordinate system, units, and domain side;
- force, charge, Brownian, and wall models with their resolved parameters;
- particle properties, stable particle IDs, release times, initial state, and
  random-seed semantics;
- integration/output settings and lifecycle/event meanings.

Represent that table with the seven producer-neutral layers documented under
**Generic meaning preflight** in
`particle_platform_redesign/solver/tools/vv/comsol/README.md`, and run
`tools/vv/comsol/meaning_preflight.py` before numerical comparison. Resolve
`ADAPTER_REQUIRED` items externally; stop on `AMBIGUOUS`. A question may receive
`PASS` or `FAIL` only when its own preflight condition is `SUPPORTED`.

Map equations by physical meaning, not by similarly named variables. Identify
the supported solver revision for every enabled COMSOL model. If a model is not
supported, compare a clearly named reduced slice or report it as out of scope;
do not approximate it silently.

Choose which question is being tested:

- **trajectory/physics parity**: feed both solvers the same canonical field;
- **end-to-end reproduction**: retain the COMSOL-native field and include field
  production/import error in the conclusion.

Do not mix these meanings in one error number.

## Extract authoritative reference data

Use a read-only copy or COMSOL `loadCopy`/`-nosave` runner, verify the source
hash before and after, and record the exact COMSOL version. Re-run COMSOL only
when the needed reference was not saved or when the comparison conditions were
intentionally changed. Extract, at minimum:

- geometry, connectivity, boundary IDs/material meanings, and primitive fields;
- resolved model expressions, selections, and parameter values;
- per-particle time, position, velocity, charge, and lifecycle while observed;
- boundary-event time, hit position, boundary ID, and outcome.

Existing Java/PowerShell runners under
`particle_platform_redesign/solver/tools/vv/comsol/` are useful examples but
are often tied to one model's tags and expressions. For an unfamiliar MPH,
inspect its semantics first and add only the narrow external exporter needed.

Keep raw exports, normalized SI tables, and their hashes outside the production
package. Never invent an escaped suffix or treat a held/stuck tail as an active
pre-event state.

## Run the candidate through public interfaces

Apply the background-field preparation rules from
`$prepare-comsol-field-case`, then use only `load_case`, `simulate`,
`open_result`, or the corresponding `chamber-particles check/run/inspect` CLI.
Do not import private engine modules from the V&V tool.

Never tune production coefficients or tolerances solely to fit one MPH. Each
solver should be numerically converged under its own valid controls; matching a
COMSOL internal step is useful only for a targeted integration-stage diagnosis.

## Make four decisive evaluations

Normally evaluate these four items and add a deeper probe only at the first
unexplained mismatch.

Before inspecting the differences, record the compared rows and exclusions,
metrics, acceptance tolerances, and relevant numerical/export/stochastic
uncertainty. Do not choose a gate after seeing the answer.

1. **Initial conditions**: particle IDs/count, release, position, velocity,
   diameter, mass, charge, enabled models, and boundary mapping.
2. **Fields and right-hand side**: at matched states compare support, primitive
   values, charge rate, each enabled force component, and total force.
3. **Boundary behavior**: compare first-hit time/location/semantic boundary,
   pre/post velocity, outcome, remaining substep, and axis handling. Zero events
   or unavailable event data means `NOT_TESTED`, not boundary agreement; do not
   infer hits from missing saved coordinates.
4. **Time trajectory**: compare common observed times for position, velocity,
   charge, lifecycle, and the first event. Do not judge by final coordinates
   alone.

For stochastic physics, compare independent ensembles, confidence intervals,
fates, occupancy, and arrival/deposition distributions. Do not require pathwise
equality unless RNG algorithm, seed, and draw allocation are genuinely shared.
Use `h` and `h/2` only for a bounded sensitivity check. Use `h`, `h/2`, and
`h/4` when asserting self-convergence or an observed order.

Report RMS and maximum state error on the common valid lifecycle, with a stated
physical normalization scale, and event/fate discrepancies separately. Create
shared-axis spatial figures with `$visualize-particle-trajectories`; figures aid
interpretation but numeric outputs remain authoritative.

## Deliver and stop

Place the external V&V package under `tools/vv/comsol/` or a dedicated evidence
output, never under `src/chamber_particles/`. Include:

- raw and normalized references, candidate outputs, hashes, and exact commands;
- `comparison_manifest.json`, normalized trajectory/event tables in the
  repository's existing CSV or Parquet form, and `comparison_summary.json`;
- critical probe data only when the field/right-hand-side comparison used it;
- a spatial plot and event summary;
- `PASS`, `FAIL`, `NOT_TESTED`, or `NOT_APPLICABLE` for the four evaluations;
- a conclusion stating the exact tested scope and broader equivalence not
  claimed.

On mismatch, check settings/provenance, field sampling, individual force/charge
terms, integration, then event handling in that order. Stop and request a new
export when identity, units, field ownership, event semantics, or RNG meaning
is ambiguous. Do not add permanent core diagnostics, COMSOL-specific branches,
fallback interpolation, or weaker gates to force agreement.
