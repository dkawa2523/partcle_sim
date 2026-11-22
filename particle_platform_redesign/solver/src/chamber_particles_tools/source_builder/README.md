# Explicit realized particle sources

This installed client converts correlated numeric CSV rows into canonical internal or surface sources.
It does not sample distributions, evaluate expressions, select a native frame, or reconstruct particle properties.
The CSV can supply the first source in a geometry/field bundle with no existing particle tables.

## Installed entry points

```console
uv run --locked python -m chamber_particles create recipe.yaml -o input
uv run --locked python -m chamber_particles check input/case.yaml
```

The same dispatcher is available as `chamber-particles` after wheel installation. No source checkout is needed.
A format-1 case recipe contains `input_bundle`, `sources: [{csv, configuration}]`, and normal case `settings`.
Paths are relative to the recipe. The shared [publisher](../inputs.py) fills data path/hash, writes through
`case_format`, publicly loads the result and uses the same `check_case` prepare as execution before publication.
A rejected case is not published. This readiness check does not execute particles or qualify accuracy.

Python services:

- `chamber_particles_tools.source_builder.convert_velocity(...)` converts supplied row-wise velocity data.
- `read_csv_source(path, bundle, configuration)` returns `RealizedTableSource` or `RealizedSurfaceSource` without writes.
- `chamber_particles_tools.inputs.create_case(bundle, settings, sources, output_directory)` publishes the complete intended source tuple.
- `create_from_configuration(recipe_path, output_directory)` is CLI `create`.
- `prepare_case(template_yaml, rows_csv, configuration, output_directory)` derives a checked case while retaining
  the template's settings and other sources. It uses the same publisher; it is not necessary for a first case.

## Source configuration format 2

`table` names the destination canonical source, independently of the YAML release name. `kind` is required.
For an internal source, the complete numeric mapping can be:

```yaml
format_version: 2
kind: table
table: particles
columns:
  particle_id: id
  release_time_s: t
  position_m: [x, y]
  charge_number: Z
  mass_kg: mass
  drag_diameter_m: d_drag
  contact_radius_m: a_contact
  electrostatic_radius_m: a_electric
  displaced_volume_m3: volume
  model_weight: weight
  material_id: material
units:
  particle_id: '1'
  release_time_s: s
  position_m: m
  charge_number: '1'
  mass_kg: kg
  drag_diameter_m: m
  contact_radius_m: m
  electrostatic_radius_m: m
  displaced_volume_m3: m3
  model_weight: '1'
  material_id: '1'
velocity_mode: vector
velocity_columns: [vx, vy]
velocity_unit: m/s
components: [x, y]
basis_to_case: [[1, 0], [0, 1]]
frame: inertial
metadata:
  scope: supplied correlated numeric initial rows
  unresolved: []
```

The CSV has a unique header and one selected row per particle ID:

```csv
id,t,x,y,vx,vy,Z,mass,d_drag,a_contact,a_electric,volume,weight,material
1,0,0.0025,0.005,0.5,0,0,1e-15,1e-7,0,5e-8,5.235987755982989e-22,1,0
```

All physical properties are explicit by default, including independent mass, drag/contact/electric dimensions,
displaced volume, weight, material and charge. No property is inferred from particle size or another source.
Only an explicit `inherit_from: {table: existing_particles, join: particle_id}` permits missing property columns
from that exact existing table. New unmatched IDs still require every property. Equal ID integers do not establish
physical identity: verify the same cohort and property authorities before requesting inheritance.

A CSV containing several times per ID must explicitly select its initial rows with `filter_equals`, such as
`{time_s: 0}`. No time filter is implied. IDs and material IDs must be exact integers; selected IDs must be unique.
Metadata records assumptions, rather than authenticating a native source law.

## Surface rows

Use `kind: surface`, replace `position_m` with `facet_id` and `facet_parameter` mappings, and remove the position unit.
Facet IDs are zero-based **canonical boundary row indices**, not native boundary IDs. The parameter lies strictly
inside the facet, excluding both endpoints. All velocity and property columns retain the same meaning.
YAML selects it with `type: surface` and its destination table name. Core prepare resolves contact geometry,
containment and supported inward departure; no epsilon position nudge is added.

## Velocity, units and frames

`vector` uses velocity components; `momentum` uses nonrelativistic `v=p/m` with the same row's actual mass.
`speed_direction` normalizes an explicit nonzero direction. `energy_direction` uses `sqrt(2E/m)` and that direction.
Scalar speed/energy needs one `velocity_columns` item and matching `direction_columns`.
Supported velocity units are m/s, cm/s, km/s; momentum kg*m/s or g*cm/s; energy J, eV, keV.
Lengths accept m/cm/mm/um/nm, release times s/ms/us/ns, and masses kg/g/mg/ug/ng.
Volume conversion is declared independently; dimensionless quantities use `1`.

Components are `[x,y]` or `[x,y,z]` for XY; `[r,z]` or `[r,phi,z]` for RZ.
The omitted third component must be exactly zero. No 3D or azimuthal motion is silently projected.
`basis_to_case` is an oriented orthonormal two-component basis, fixed or row-wise through Python.
It rotates vectors, not positions; the supplied positions already use the canonical origin/basis.
`frame: inertial` is a caller declaration. Moving/rotating/unknown frames require external physical resolution.

## Identity and verification

The recipe publisher records CSV/config identities and upstream provenance. Derived-template preparation also
binds template YAML and canonical identity. Canonical validation remains with the writer; model/support/event
readiness remains with `check_case`. Conversion alone establishes neither native-frame equivalence nor trajectory agreement.

```console
uv run --locked pytest tests/verification/test_source_builder.py tests/scenarios/test_input_clients.py -q
```
