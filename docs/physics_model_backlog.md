# Physics Model Backlog

This backlog is a product-scope map, not a new numerical specification. The
solver advances supplied particles through supplied fields, gas properties,
source tables, wall tables, and configured force models. It must not infer
missing physical source populations or chamber-specific plasma and wall effects.

`docs/numerics_contract.md` remains the source of current numerical behavior.
This document classifies what is product scope now, what is optional or limited,
and what should stay outside solver inference.

## Classifications

- `current product scope`: core model behavior expected in normal production
  and comparison runs.
- `current supported with limitations`: implemented, but constrained by
  dimensionality, backend, required fields, empirical range, or explicit config.
- `comparison-only diagnostic`: useful for parity analysis, but not a production
  model claim by itself.
- `production optional`: available when explicitly configured and supplied with
  required inputs.
- `future extension`: useful product direction, but not implemented as a
  current solver model.
- `should not be inferred by solver`: must come from explicit input data,
  upstream adapters, fields, wall laws, or future named models.

## Model Matrix

| Model item | Classification | Current implementation status | Required inputs, fields, or config | COMSOL-faithful notes | Production notes and next action |
| --- | --- | --- | --- | --- | --- |
| Stokes drag | `current product scope` | Implemented as the default drag relaxation model. | `solver.drag_model: stokes` or omitted; particle mass/diameter/density and flow/gas inputs used by runtime. | Manifest force inventory may map drag law to `stokes`; faithful mode should fail if required drag metadata is missing. | Keep as the default baseline and first synthetic verification case. |
| Schiller-Naumann | `current supported with limitations` | Implemented as `schiller_naumann` finite-Re correction from local slip speed. | `solver.drag_model: schiller_naumann`; gas density and dynamic viscosity from fields or scalar fallbacks. | Use only when COMSOL force inventory explicitly identifies this drag law or accepted equivalent. | Document empirical range limits in case notes when used for high-Re particles. |
| Epstein / low pressure drag | `current supported with limitations` | Implemented as `epstein` low-pressure/free-molecular relaxation. | `solver.drag_model: epstein`; particle density/diameter, gas density, gas temperature, and gas molecular mass. | Faithful mode must receive an explicit force inventory and coordinate/unit scaling; do not infer pressure regime from endpoint behavior. | Product-ready for supplied rarefaction inputs; do not use as an automatic correction for missing gas fields. |
| Cunningham slip correction | `current supported with limitations` | Implemented as `stokes_cunningham`, with aliases `cunningham`, `cunningham_millikan_davies`, and `cmd`. | Gas density, dynamic viscosity, temperature, molecular mass, and particle diameter. | COMSOL rarefaction mappings should be explicit in the manifest or export audit. | Keep as a supported drag option; validate gas-unit assumptions in imported cases. |
| Brownian / Langevin motion | `production optional` | Implemented as opt-in `underdamped_langevin` stochastic velocity kicks. | `solver.stochastic_motion` or `solver.forces.brownian`; seed, stride, and temperature source `field_T_then_gas` or `gas`. | Disable or seed for deterministic parity and first-step comparison. Do not make stochastic comparison the default. | Keep optional. Use only when the case owner wants stochastic ensemble behavior. |
| Charge evolution | `production optional`, `current supported with limitations` | Implemented with `te_relaxation`, `density_temperature_flux_relaxation`, and `finite_rate_flux_balance`. | `solver.charge_model`; 2D regular rectilinear electric field plus electron/plasma fields or scalar plasma background, depending on mode. | COMSOL flux vectors are not consumed directly by the charge model. Manifest force inventory remains authoritative. | Next useful improvement is 3D and triangle-mesh support without weakening explicit input requirements. |
| `qE/m` electric force | `current product scope` | Implemented as acceleration from particle charge and sampled electric field. | Electric field quantities and particle charge, either static from particle table or updated by charge model. | Field bundles should provide electric field, not pre-baked fixed-charge acceleration, unless used only as a diagnostic reference. | Keep as core product behavior and first-step parity target. |
| Thermophoresis | `production optional`, `current supported with limitations` | Implemented with `talbot` and `continuum` model options. | Explicit `solver.forces.thermophoresis`; temperature field/gradient support plus gas density, gas viscosity, gas temperature, and thermal conductivity settings. | Enable only when COMSOL force inventory includes a matching force and required exported quantities exist. | Keep opt-in and fail fast when required field quantities are absent. |
| Dielectrophoresis | `production optional`, `current supported with limitations` | Implemented with DC and AC Clausius-Mossotti style options. | Explicit `solver.forces.dielectrophoresis`; electric field support and gradient of electric-field magnitude squared; particle/medium permittivity and conductivity settings. | Compare only against cases where COMSOL DEP force definition and field export are explicitly mapped. | Keep optional; do not infer DEP from electric field presence alone. |
| Ion drag (`ion_drag`) | `future extension`; sheath-derived behavior is `should not be inferred by solver` | Not implemented as a supported force catalog entry. | Future model should require explicit ion/plasma fields and named model parameters. | Missing COMSOL ion-drag or sheath physics is an export/model gap, not a solver repair opportunity. | Add only as a field-driven extension that fails closed when required plasma inputs are absent. |
| Lift | `production optional`, `current supported with limitations` | Implemented as a Saffman-style lift option in the force pipeline. | Explicit `solver.forces.lift`; velocity field, gas density, gas viscosity, and vorticity support. | Use only when COMSOL force inventory and exported quantities map to the implemented model. | Keep opt-in; near-wall lift corrections remain separate future work. |
| Pressure-gradient force | `production optional`, `current supported with limitations` | Implemented as `fluid_material_acceleration`. | Explicit `solver.forces.pressure_gradient`; fluid acceleration or velocity-derived material acceleration plus gas density. | Must be present in the manifest force inventory and available from exported fields for faithful comparison. | Keep optional; record field-source provenance in case summaries. |
| Virtual mass | `production optional`, `current supported with limitations` | Implemented as `particle_material_acceleration` with configurable coefficient. | Explicit `solver.forces.virtual_mass`; flow time derivative, flow velocity gradient, gas density, and particle velocity state. | Compare only when COMSOL force definition maps to the same material-acceleration semantics. | Keep optional; avoid silently enabling from flow fields alone. |
| Near-wall drag/lift/sheath corrections | `future extension` when explicitly modeled; otherwise `should not be inferred by solver` | Not automatically applied; summaries report no near-wall force correction. | Future model should consume explicit wall metadata, sheath/plasma fields, or named correction fields. | Do not repair COMSOL near-wall discrepancies with hidden wall-distance tuning. | Highest-value future interface if it consumes explicit supplied fields and leaves collision physics unchanged. |
| Deposition/resuspension/detachment population models | Supplied source/wall input behavior only; failure/fracture/detachment probabilities are `should not be inferred by solver` | Source laws and wall/material tables can encode explicit release/resuspension behavior, but the solver does not predict when a deposit fails or flakes. | Particle/source tables, material/wall source parameters, optional process steps, and explicit event timing. | COMSOL faithful mode preserves release tables and must not generate missing particles. | Upstream adapters or user models should generate release populations; solver should only preprocess explicitly supplied releases when configured. |

## COMSOL-Faithful Policy

- The COMSOL manifest force inventory is authoritative.
- Missing or unknown required force, field, wall-law, coordinate-scale, or
  release mappings should fail clearly before runtime comparison.
- Stochastic/Brownian behavior should be disabled or seeded for deterministic
  parity, especially in first-step comparisons.
- The solver must not add hidden repair for missing COMSOL sheath, ion-drag,
  wall, or export physics.
- `comsol_faithful` compares exported model behavior; production defaults should
  not silently substitute for missing COMSOL metadata.

## Production Policy

- Optional forces must be enabled explicitly under `solver.forces` or their
  established legacy config path.
- Required field quantities must be supplied by providers. Field support gaps
  should be fixed in exports/providers, not hidden in solver hot paths.
- Scalar gas fallbacks are allowed only where the current model already permits
  them, such as configured gas properties for supported drag or stochastic
  temperature fallback.
- Source population, detachment, fracture, flake generation, deposit failure,
  sheath assumptions, and chamber-specific release probability must come from
  inputs or future explicit adapters/models.
- Production surface-release preprocessing may classify/project/offset supplied
  near-boundary releases only when explicitly configured; it is not a population
  generator.

## Recommended Next Physics Features

1. Add a field-driven ion-drag extension that requires explicit ion/plasma
   inputs and fails closed when absent.
2. Add a near-wall/sheath correction interface that consumes explicit supplied
   fields or wall metadata, with no automatic wall-distance tuning.
3. Expand charge-model support to 3D and triangle-mesh backends while
   preserving the current explicit input requirements.
