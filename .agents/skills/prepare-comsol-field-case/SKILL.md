---
name: prepare-comsol-field-case
description: Convert a specified COMSOL model that contains chamber geometry and background-field solutions, but no particle-tracing study, into a checked input case for chamber-particles. Use when an agent must inspect an MPH or its exports, select the required primitive fields and boundary semantics, run the COMSOL adapter or reduced electrostatic builder, and produce solver YAML/HDF5 inputs with provenance.
---

# Prepare a COMSOL field case

Create a runnable solver input from the requested COMSOL field model. Keep all
COMSOL-specific work in external adapters; do not add a COMSOL branch to the
solver core.

## Work from the requested physics

1. Read the repository `AGENTS.md`, `particle_platform_redesign/AGENTS.md`,
   `particle_platform_redesign/solver/docs/physics_models.md`,
   `particle_platform_redesign/solver/docs/support_and_errors.md`, and the
   relevant part of `particle_platform_redesign/solver/docs/case_format_v3.md`.
2. List the particle models the user intends to enable. Derive the smallest
   set of required canonical primitive fields from those models. Prefer gas,
   thermal, electric, magnetic, and plasma primitives over particle-size-
   specific quantities already derived by COMSOL.
3. Classify the source before extracting data:
   - **direct-field case**: COMSOL supplies the required field primitives;
   - **reduced-field case**: COMSOL supplies geometry and flow/thermal fields,
     while `tools.electrostatic_builder` constructs the explicitly requested
     reduced electrostatic/plasma field.

Do not export every variable merely because it exists. Do not silently select
a force law, charge law, plasma closure, wall potential, or boundary response.

## Inspect and extract

Inspect the specified MPH read-only with COMSOL batch/API when available. Use
`loadCopy`/`-nosave`, do not overwrite the source model, and verify its hash is
unchanged. Record:

- source hash and COMSOL version;
- coordinate system, length unit, selected component, study, solution, dataset,
  domain, and time when applicable;
- mesh vertices and connectivity, element ordering, and the external domain ID;
- boundary entity IDs/names and their intended material and wall semantics;
- each exported expression, COMSOL unit, canonical SI name, basis/layout, and
  domain side;
- the exact export command or runner revision.

Export geometry/topology and boundary IDs as authoritative data. Never infer a
wall solely from a plot, a NaN mask, or a field discontinuity. Export stable
node, element, and boundary IDs with the required primitives. Do not fill,
nudge, or interpolate missing source values without an explicitly justified
conversion.

If model access or licensing is unavailable, or no stored solution exists and
re-solving is not authorized, state the exact missing export instead of
fabricating it.

Before writing or changing an adapter, map the selected scope into the seven
producer-neutral layers documented under **Generic meaning preflight** in
`particle_platform_redesign/solver/tools/vv/comsol/README.md`, then run
`tools/vv/comsol/meaning_preflight.py`. For a field-only source, mark both
trajectory comparison questions as unrequested and `NOT_APPLICABLE`; use the
layer classifications to decide whether direct conversion is supported, an
adapter is needed, or the source meaning is still ambiguous. Do not treat a
successful preflight as field-accuracy or trajectory evidence.

## Build the canonical case

Run commands from `particle_platform_redesign/solver/` with its locked uv
environment.

- The current `tools.comsol_adapter` is a narrow static axisymmetric CSV
  converter, not a generic MPH reader. Use it only when the source satisfies
  `particle_platform_redesign/solver/tools/comsol_adapter/README.md`.
- For a thermal/flow input that deliberately uses the reduced electrostatic
  model, first create the canonical base case, then follow
  `particle_platform_redesign/solver/tools/electrostatic_builder/README.md`.
  Keep the chosen plasma parameters, closure, and boundary potentials in the
  builder configuration.
- For a genuinely unsupported producer layout, extend or add a narrow adapter
  under `tools/`; write the same canonical schema and keep COMSOL imports out of
  `src/chamber_particles/`.

After adaptation or field building, compare the produced inventory with every
selected model's required-field table. A builder result is not a complete case
if a selected model still lacks a primitive such as a DEP gradient, signed
vorticity, heat flux, screening length, or ion transport property.

Create `case.yaml` with particle population and release, selected model
revisions, integrator and time/output settings, boundary groups and laws, and
resource settings. Put the returned logical content hash in the case reference
and use SI units in canonical artifacts. Do not infer particle properties,
release conditions, or wall laws from an MPH that contains no particles. Ask
the user only when an unresolved choice changes the physical problem.

The adapter and builder write HDF5/reports, not the solver YAML. Author the YAML
explicitly from `particle_platform_redesign/solver/docs/case_format_v3.md` and
`particle_platform_redesign/solver/examples/quickstart/create_case.py`; do not
hide physical choices in an adapter default.

## Check and finish

Run, in order:

```powershell
uv run --locked chamber-particles check PATH/TO/case.yaml
uv run --locked chamber-particles check PATH/TO/case.smoke.yaml
uv run --locked chamber-particles run PATH/TO/case.smoke.yaml -o PATH/TO/smoke-result
uv run --locked chamber-particles inspect PATH/TO/smoke-result
```

Create `case.smoke.yaml` as a separately named derivative with only a smaller
population/time/output burden; never overwrite the requested scientific case.
If release particles are realized in HDF5 rather than generated from YAML,
create a separately named reduced smoke artifact instead of pretending YAML can
subset it. Use the smoke case for wiring only. Add time-step or mesh convergence
work only when the intended scientific run requires it; do not grow a general
diagnostic suite around one import.

Return these concrete outputs:

- canonical HDF5 and solver YAML;
- adapter/builder report with hashes and boundary coverage;
- a short mapping of `COMSOL expression [unit] -> canonical primitive [SI]`;
- the reproducible commands and any remaining physical assumptions.

Stop rather than guess when units, coordinate convention, domain side, node
mapping, boundary semantics, geometry support, or a required primitive is
ambiguous. A successful field import does not claim COMSOL trajectory
agreement because this source has no particle-tracing reference.
