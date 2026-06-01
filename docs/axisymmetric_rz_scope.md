# Axisymmetric RZ Scope

This document defines the current product meaning of `axisymmetric_rz`. It is a
coordinate-system contract, not a promise that the solver has full cylindrical
particle dynamics.

## Current Meaning

`axisymmetric_rz` means a 2D `(r, z)` runtime:

- `spatial_dim` is 2.
- axis 0 is radial `r`.
- axis 1 is axial `z`.
- the radial axis must be finite and non-negative.
- summaries and preflight reports expose `coordinate_system: axisymmetric_rz`
  and `axis_names: ["r", "z"]`.
- `r = 0` axis-boundary detection is reported when visible in the geometry.

The coordinate mode is normalized through
`particle_tracer_unified/core/coordinate_systems.py`. Supported aliases include
`axisymmetric`, `axisymmetric_2d`, `cylindrical_rz`, `r_z`, `rz`, and
`rz_axisymmetric`; all normalize to `axisymmetric_rz`.

This mode is not treated as a synonym for `cartesian_xy` at import, provider
summary, preflight, first-step comparison, or compare-summary boundaries.

## What Is Implemented

Implemented and product-supported:

- RZ parsing and preservation from config and COMSOL manifests.
- manifest/config coordinate mismatch rejection for faithful COMSOL cases.
- canonical RZ axis names in runtime summaries and provider summaries.
- non-negative radial-axis validation for synthetic and precomputed geometry.
- `r = 0` report fields, including edge count, edge indices, and part IDs where
  boundary edges are available.
- `ring_area_weight(r) = 2*pi*r` utility for reporting and future source
  sampling.
- COMSOL case summaries include coordinate system, axis names, RZ axis-boundary
  report, and ring-weight summary.
- first-step comparison artifacts use RZ columns such as `r`, `z`, `vr`, and
  `vz`; reference parsing also accepts legacy `x/y` aliases.
- ensemble comparison summaries carry `coordinate_system`, `axis_names`, and
  the RZ report from solver reports.

Existing legacy solver CSV outputs such as `final_particles.csv` keep `x/y`
column names for compatibility. Use `solver_report.json`, preflight reports,
first-step compare artifacts, and comparison summaries to determine coordinate
semantics.

## What Is Not Implemented

The current solver does not implement full cylindrical dynamics:

- no `theta` coordinate is tracked;
- no `v_theta` state is advanced;
- no centrifugal, swirl, or azimuthal-coupling terms are added automatically;
- no special `r = 0` symmetry collision/reflection law is applied;
- no automatic ring-area-weighted source sampling is applied.

The current RZ axis-boundary policy is:

```text
report_only_collision_unchanged
```

That means `r = 0` is identified and reported, but wall-collision behavior is
not rewritten by coordinate mode alone. A case that needs special axis symmetry
behavior must provide explicit geometry/wall semantics or wait for a future
named axis-boundary policy.

## COMSOL Faithful Comparison

`axisymmetric_rz` is comparison-safe for import and diagnostic comparison when
the exported COMSOL data provide:

- manifest `coordinates.coordinate_system: axisymmetric_rz`;
- positive coordinate scale;
- RZ-compatible release table coordinates;
- geometry and field axes that preserve `r/z` semantics;
- explicit wall law and force inventory mappings.

Faithful mode keeps COMSOL release coordinates machine-comparable. It does not
snap, repair, or reinterpret an axisymmetric release as Cartesian. If a COMSOL
case depends on azimuthal velocity, swirl, sheath behavior, or special axial
symmetry wall behavior, that dependency must be represented by exported fields,
explicit wall/source inputs, or a future named model. The solver should not
infer it from `axisymmetric_rz` alone.

## Production Surface-Release Use

`axisymmetric_rz` is production-safe for 2D RZ trajectory studies where the
model can be represented as radial/axial motion in supplied fields and explicit
walls:

- particles are supplied with `(r, z)` or accepted aliases;
- fields and geometry are already canonicalized to RZ axes;
- optional surface-release preprocessing is explicitly configured;
- wall behavior is explicit in `part_walls.csv` / `materials.csv`;
- source populations and release timing are supplied by the user or adapter.

It is limited, not complete cylindrical physics. Do not use it as evidence that
swirl, `v_theta`, ring-weighted population generation, or special `r = 0`
symmetry collision behavior is modeled.

## Ring-Area Weighting

`ring_area_weight(r)` exists for reporting and future sampling support. It
returns `2*pi*r`, rejects negative radii, and rejects non-finite radii.

Automatic ring-weighted source sampling is a future extension, not current
product behavior. This is intentional: enabling it would change source
population semantics and could break comparison parity unless the source
measure is explicitly declared by the case or adapter.

Future ring-weighted sampling should require an explicit source-sampling mode,
record the sampling measure in source diagnostics, and remain disabled in
COMSOL faithful comparison unless the COMSOL reference uses the same measure.

## Product Status

- COMSOL comparison-safe: yes, for coordinate-aware import/report/diagnostic
  comparison, with explicit exported inputs and no hidden repair.
- Production-safe: yes, for 2D radial/axial trajectory cases that do not require
  unresolved cylindrical dynamics.
- Limited: yes. Full cylindrical state, `v_theta`, automatic ring-weighted
  source sampling, and special `r = 0` symmetry collision behavior are future
  extensions.
