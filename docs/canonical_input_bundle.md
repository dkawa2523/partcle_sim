# Canonical Input Bundle

This document defines the boundary between source-specific import code and the
solver runtime. COMSOL is one adapter into this boundary, not the model source
that the solver is built around.

The canonical runtime input is the data consumed by
`particle_tracer_unified/io/runtime_builder.py`: normalized particle/source
tables, optional material and wall tables, optional process timing tables, and
provider-backed geometry and fields. New import sources should adapt their own
native data into this bundle before the solver is called.

## Bundle Shape

A normal file-backed bundle is a `run_config.yaml` plus referenced CSV/NPZ
files:

- `paths.particles_csv`: required particle/release table.
- `paths.materials_csv`: optional material defaults.
- `paths.part_walls_csv`: optional boundary part to wall/material mapping.
- `paths.source_events_csv`: optional source timing/gain events.
- `paths.process_steps_csv`: optional named process intervals.
- `providers.geometry`: optional geometry provider configuration.
- `providers.field`: optional field provider configuration; requires geometry
  when using regular rectilinear fields.
- `gas`: optional gas properties in SI units.

The in-memory equivalent is a `RuntimeLike` containing:

- `ParticleTable`
- optional `MaterialTable`
- optional `PartWallTable`
- optional `SourceEventTable` and compiled source events
- optional `ProcessStepTable`
- optional `GeometryProviderND`
- optional `FieldProviderND`

Adapters may either emit the file bundle above, or add a focused `io/` loader
that returns these canonical objects. Keep format-specific parsing out of the
solver loop.

## Coordinates And Units

Supported coordinate systems are:

- `cartesian_xy`: 2D axes `x`, `y`.
- `axisymmetric_rz`: 2D axes `r`, `z`; `r` is axis 0 and must be non-negative.
- `cartesian_xyz`: 3D axes `x`, `y`, `z`.

All canonical coordinates, distances, diameters, geometry axes, signed-distance
values, offsets, and boundary coordinates are SI metres. Particle velocities
are m/s. Release times and process times are seconds. Forces, accelerations,
gas properties, and material/wall coefficients should use the units named in
the corresponding CSV column or field quantity metadata.

If an upstream format uses model units, the adapter must scale into SI before
creating canonical tables/providers. The solver must not guess coordinate scale.

## Geometry Provider Expectations

Geometry is represented by `GeometryND` behind `GeometryProviderND`.

Regular geometry providers must supply:

- `spatial_dim`
- `coordinate_system`
- strictly increasing axes named by position as `axis_0`, `axis_1`, and
  optionally `axis_2`
- `valid_mask` with the same grid shape as the axes
- signed distance `sdf`
- one normal component array per spatial axis
- `nearest_boundary_part_id_map`

Boundary topology should be supplied when wall collisions or surface release
classification are needed:

- 2D: `boundary_edges` with `boundary_edge_part_ids`, and preferably decoded
  boundary loops.
- 3D: `boundary_triangles` with `boundary_triangle_part_ids`.

Part IDs are source data identifiers. Do not encode case-specific meanings in
solver code. If a future adapter needs special interpretation, it should produce
explicit part/wall/material rows.

## Field Provider Expectations

Fields are represented by `FieldProviderND`.

Regular rectilinear field bundles use NPZ arrays:

- `axis_0`, `axis_1`, and optionally `axis_2`
- optional `times`; absent means steady at `t=0`
- `valid_mask`
- optional `support_phi`
- one array per field quantity
- optional `metadata_json`

For regular fields, field axes must exactly match geometry axes after import.
This is deliberate: adapters may resample/canonicalize before writing the
bundle, but the runtime should not silently reinterpret mismatched axes.

Triangle-mesh 2D field bundles use:

- `mesh_vertices`
- `mesh_triangles`
- optional `times`
- one array per mesh quantity
- optional `metadata_json`

Quantity names should use solver-recognized canonical names or established
aliases. Common quantities include:

- flow velocity: `ux`, `uy`, `uz`, or RZ aliases such as `ur`, `vz` where
  supported by the adapter
- electric field: `Ex`, `Ey`, `Ez`
- gas density: `rho_g`
- dynamic viscosity: `mu`
- temperature: `T`
- pressure: `p`
- wall shear: `tauw`
- friction velocity: `u_tau`

Adapters should record source names and units in metadata when translating from
upstream names. In `comsol_faithful` mode, the manifest force inventory is the
authority for which field quantities are expected.

## Boundary And Wall Law Tables

`part_walls.csv` maps boundary `part_id` values to wall behavior and optional
source overrides. `materials.csv` provides material-scoped defaults. Part rows
override material rows; particle rows can override selected source properties.

Wall laws must be explicit. A faithful imported case should fail if an upstream
wall behavior cannot be mapped to a supported solver law. A production case may
use supported custom/explicit wall settings, but unknown wall laws must not be
silently approximated.

Use `docs/wall_law_catalog.md` for the current wall-law support matrix.

## Particle And Source Semantics

`particles.csv` is the canonical particle/release table. Required position
columns are:

- `x`, `y` for `cartesian_xy`
- `r`, `z` or `x`, `y` aliases for `axisymmetric_rz`
- `x`, `y`, `z` for `cartesian_xyz`

Important optional columns include:

- `particle_id`
- velocity components: `vx`, `vy`, `vz`, or RZ aliases `vr`, `vz`
- `release_time`
- `mass`
- `diameter`
- `density`
- `charge`
- `source_part_id`
- `material_id`
- `source_event_tag`
- `source_law_override`
- `source_speed_scale_override`
- `stick_probability`

`source_part_id > 0` means the release has known source boundary provenance.
`source_part_id == 0` means unknown source provenance and must remain unknown.
Unknown source particles must not receive same-source release grace and must not
be repaired into a nearest-wall source in faithful imports.

Production surface-release cases may classify/project/offset near-boundary
releases only when `source.preprocess.boundary_release: true` is explicitly
configured. `boundary_capture_tolerance_m` controls classification;
`boundary_inward_offset_m` controls the small inward displacement. These are
separate concepts.

COMSOL faithful cases preserve raw sampled release coordinates and reject source
preprocessing, boundary snapping, and solver-side repair.

## COMSOL Manifest Mapping

`particle_tracer_unified/io/comsol.py` maps a COMSOL manifest into the canonical
bundle:

- `coordinates.coordinate_system` and
  `coordinates.coordinate_scale_m_per_model_unit` set canonical coordinates and
  SI scaling.
- `particles.release_table` becomes the canonical `ParticleTable`.
- `boundaries.map_file` and `boundaries.wall_law_file` become
  `PartWallTable` and `MaterialTable`.
- `fields` and force inventory validate required field mappings and solver
  force configuration.
- optional `paths.source_events_csv` and `paths.process_steps_csv` are loaded
  through the same canonical table loaders as non-COMSOL cases.

`tools/build_comsol_case.py` is a case builder/export adapter. It may read
COMSOL-exported mesh/text/bundle files and write canonical CSV/NPZ/YAML
artifacts. The solver package must not read `.mph` files or call COMSOL APIs.

## Adding A Future Adapter

A future non-COMSOL adapter author should implement the following:

1. Parse the source format in a focused module under `particle_tracer_unified/io/`
   or in `tools/` when it is an offline case builder.
2. Normalize coordinates into SI metres and choose one canonical
   `coordinate_system`.
3. Produce canonical geometry and field providers, or write precomputed NPZ
   bundles accepted by `providers.precomputed`.
4. Produce `particles.csv` or a `ParticleTable` with explicit source
   provenance. Keep unknown source as `source_part_id = 0`.
5. Produce `part_walls.csv` and `materials.csv` for every boundary part that
   can participate in wall handling.
6. Produce optional source events and process steps only when the source format
   has real timing semantics.
7. Add one validation point near loading that fails on missing coordinate scale,
   unsupported wall laws, axis mismatch, missing required field quantities, or
   ambiguous source provenance.
8. Pass canonical objects to `RuntimeLike` through `runtime_builder.py` without
   adding source-specific branches to solver core.

Adapters should be small and conservative. If source data are ambiguous, fail
with an actionable import error rather than adding solver-side repair.
