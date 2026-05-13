# Micromixer Particle Tracing Verification

Status: truth-export audit and root-cause diagnostics are reproducible. The
current solver pipeline is useful for debugging and trend comparison, but it is
not COMSOL-equivalent yet.

Past ICP `_out_*` folders are reference-only and are not truth for this case.

## Clean Case Roots

- Field/inventory export: `_external_exports/micromixer_particle_tracing_field_probe/`
- Particle result export: `_external_exports/micromixer_particle_tracing_xy_velocity_probe/`
- Solver-ready case: `_external_exports/micromixer_particle_tracing_solver_case/`
- Truth audit: regenerated locally under
  `_external_exports/micromixer_particle_tracing_truth_audit/`
- Exact debug run: `_out_micromixer_particle_tracing_exact_pass_through_warn/`
- Trend run: `_out_micromixer_particle_tracing_inward_clean_trend/`

## Current Truth Inventory

- COMSOL model: `data/micromixer_particle_tracing.mph`, COMSOL 6.4 Build 378,
  2D micromixer particle tracing.
- Flow field: single-context `dset1` export, scaled from model `mm` to SI
  metres. The mesh-native bundle `generated/comsol_field_mesh_2d.npz` is the
  standard field truth for replay. The regular-grid bundle is diagnostic-only.
- Time context: the audit detects non-particle `RotatingFrameFD/rtfr1`
  `alpha=2[1/s]*pi*t`. A time-resolved mesh field export is required before
  claiming full time-varying field parity.
- Particle result dataset: `part1`, 3150 particles, 401 output times, 687786
  finite trajectory rows, `0..2 s`.
- Release features: source boundary selections `1`, `5`, and `12`, 50
  particles per release time, `rt=range(0,0.05,1)`, `v0=0`.
- Particle properties used by solver: `dp=10 um`, `rhop=2200 kg/m^3`,
  density/diameter-implied mass `1.1519173063162575e-12 kg`, charge `0`.
- Force inventory: COMSOL particle tracing uses Drag Force 1 with Stokes plus
  `CunninghamMillikanDavies`, and enables virtual mass/pressure-gradient
  forces. The current trend solver config still uses `stokes`, so force parity
  has explicit gaps and must not be tuned away.

## Implemented Cleanup

- COMSOL boundary roles are derived from `physics_feature_inventory.json`.
- `PairContinuity` selections `[15,16,17,18,33,34,35,36]` map to solver parts
  `[16,17,18,19,34,35,36,37]` and are `pass_through`.
- Outlet selection maps to solver part `8` and remains `stick`.
- Wall bounce entities remain `specular`.
- Collision search now filters `pass_through`/continuity/inactive wall laws out
  of boundary hit surfaces.
- `wall_catalog_alignment` now reports pass-through as non-colliding diagnostic
  state instead of active wall noise.
- `compare_particle_results.py` now writes field replay, boundary role
  alignment, trend alignment, and divergence-vs-wall-event alignment alongside
  the existing particle-wise outputs.
- `audit_truth_export.py` now writes one truth manifest from the existing
  COMSOL exports and separates exact release, inward-clean diagnostic release,
  boundary roles, mesh replay, force gaps, and missing COMSOL export items.
- `particles_inward_clean.csv` is the formal trend input. The previous boundary
  offset smoke files and PairContinuity-before-fix outputs were removed from
  the micromixer standard path.

## Current Diagnostic Results

Exact release alignment is exact:

- matched particles: 3150
- max release time error: `0`
- max release position error: `0`
- max release velocity error: `0`

Exact release is not a clean solver input:

- strict input status: 1432 clean, 1718 mixed-stencil
- exact pass-through run initial geometry diagnostic: 1174 outside/ambiguous
  geometry points

Inward-clean release is a clean trend input:

- moved particles: 3150
- failed moves: 0
- final support: 3150 clean
- max displacement: `7.905694150420969e-05 m`
- mean displacement: `3.9021628862244665e-05 m`

Truth audit summary:

- committed summary: `case_summary.json`
- exact release position/time/velocity max error: `0`
- inward-clean release max displacement from COMSOL release:
  `7.905694150428234e-05 m`
- boundary role mismatches: `0`
- missing COMSOL exports:
  - time-resolved mesh field export
  - wall-hit/outcome export
  - row-level release source and particle property columns
- force parity gaps:
  - Cunningham-Millikan-Davies rarefaction is not enabled in the current solver
    run config
  - virtual mass/pressure-gradient force parity has not been re-baselined after
    solver implementation
- parity readiness:
  - `ready_for_exact_solver_comparison=false`
  - blocker count: `6`
- next action source: regenerate the local audit output when fresh COMSOL
  exports are available

Mesh-native field replay on COMSOL trajectory:

- samples: 687786
- inside/support/clean fraction: 1.0
- velocity residual mean: `2.327e-4 m/s`, p90: `7.821e-4 m/s`,
  max: `7.528e-3 m/s`
- source residual mean: part 2 = `3.51e-6 m/s`, part 6 = `3.52e-4 m/s`,
  part 13 = `2.01e-4 m/s`

Historical regular-grid field replay on COMSOL trajectory:

- samples: 687786
- inside grid fraction: 1.0
- clean stencil fraction: 0.35961040207273776
- source clean fractions: part 2 = 0.7412, part 6 = 0.4035, part 13 = 0.2024
- velocity residual mean: `4.039e-4 m/s`, p90: `1.134e-3 m/s`,
  max: `9.430e-3 m/s`

Exact release + pass-through boundary run:

- trajectory position error mean: `1.772e-3 m`
- final position error mean: `2.058e-3 m`
- final distribution centroid error: `1.913e-3 m`
- final states: 28 active, 593 contact sliding, 1752 contact endpoint stopped,
  777 stuck
- wall events: 77820

Inward-clean trend run:

- trajectory position error mean: `1.535e-3 m`
- final position error mean: `1.943e-3 m`
- final distribution centroid error: `2.104e-3 m`
- final states: 28 active, 697 contact sliding, 1648 contact endpoint stopped,
  777 stuck
- wall events: 83180
- divergence-vs-wall alignment:
  - at `0.1 mm`, 2721 particles diverge; 1683 before first solver wall event,
    486 near it, and 552 after it
  - at `1 mm`, 1666 particles diverge; 133 before first solver wall event,
    364 near it, and 1169 after it

## Root-Cause Ranking

1. Missing direct COMSOL boundary-event truth.
   Wall-hit/outcome, first-hit entity, normal, and outcome are not exported yet.
   Without that table, boundary-event parity can only be inferred from solver
   wall events and trajectory divergence.

2. Incomplete row-level release truth.
   The current COMSOL release table has position, time, and velocity, but not
   source entity, mass, diameter, density, or charge. Solver `particles.csv`
   has these values from inventory/packing. That is useful, but exact
   particle-condition parity needs them in the canonical COMSOL release table.

3. Boundary contact/sliding/endpoint handling.
   Mesh-native replay removes the regular-grid support blocker, but full runs
   still produce many contact sliding and endpoint stops. This is now the most
   important solver-core debug area after COMSOL wall events are available.

4. Boundary-release support mismatch.
   COMSOL releases particles on boundary selections, while the solver's regular
   field provider needs clean interpolation support. Exact release is therefore
   a debug input, not a clean validation input.

5. Field time-context and force contribution gaps.
    Mesh-native steady replay has full support, but the COMSOL model contains a
    time-dependent rotating-frame expression. COMSOL drag also uses Cunningham
    rarefaction and virtual mass/pressure-gradient forces. These require a
    fresh solver-config/audit baseline, not tuning knobs.

6. Regular-grid field replay loses FEM boundary behavior.
   Only 35.96% of COMSOL trajectory samples have a clean solver stencil. Source
   part 13 is especially poor at 20.24%. Keep regular-grid replay as a failure
   diagnostic only, not the faithful path.

7. PairContinuity was a real boundary-role bug and is fixed.
   It was not sufficient by itself to reproduce COMSOL, but keeping it as
   pass-through is required before any further comparison is meaningful.

## Next Work

- Export or reconstruct COMSOL wall-hit/outcome data if available.
- Re-export row-level release source and particle property values, using the
  one-expression probe configs under `required_comsol_exports/`.
- Export mesh field samples for the required COMSOL time contexts if `rtfr1`
  actually changes the flow field used by particle tracing.
- Move the solver micromixer force config to `stokes_cunningham` and re-baseline
  COMSOL virtual mass/pressure-gradient parity.
- Simplify contact sliding/endpoint handling after comparing against COMSOL
  wall events.
- Keep solver core generic. COMSOL-specific extraction, role mapping, release
  placement, and comparison stay in `external/` and `tools/`.
