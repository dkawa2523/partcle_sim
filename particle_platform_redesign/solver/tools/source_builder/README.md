# Realized initial source builder

This external producer converts supplied joint initial rows into one existing
canonical **internal table** source. It uses `case_format.write()` and verifies
the new case through public `load_case()`. Geometry, fields, other sources and
the template's models, walls, numerical settings and output schedule are retained.
The output is a fresh `case.h5` and `case.yaml`; existing destinations are refused.
This does not run the simulation or certify a physical/native-frame correspondence.

Run from the independent solver project:

```console
uv run --locked python -m tools.source_builder TEMPLATE.yaml INITIAL.csv SOURCE.json OUTPUT_DIRECTORY
```

The three Python entry points in `tools.source_builder` are:

- `convert_velocity(values, mass_kg, *, mode, unit, components,
  coordinate_system, basis_to_case, frame, direction=None)` returns SI `[N,2]`
  initial velocity vectors. All values and masses are realized per-particle rows.
- `read_csv_source(path, data_bundle, configuration)` returns a
  `RealizedTableSource`, with no file writes. Canonical schema validation remains
  with the existing canonical writer.
- `prepare_case(template_yaml, rows_csv, configuration_json, output_directory)`
  writes a derived case, publicly loads it, and returns paths plus logical hash.

## Input meaning

`vector` converts velocity components. `speed_direction` uses
`v=s*direction/|direction|`. `momentum` uses nonrelativistic inertial `v=p/m`.
`energy_direction` uses `s=sqrt(2*E/m)` with the **same row's actual mass** and
the normalized direction. Energy accepts J, eV and keV; momentum accepts kg*m/s
and g*cm/s. Velocity/speed accept m/s, cm/s and km/s. These formulas assume a
nonrelativistic inertial particle; the producer does not establish that physical
regime from the numeric input or implement relativistic equations.

Positions must already use the template's canonical coordinates and origin;
their declared length unit alone is converted. Velocity/direction component
orders are `[x,y]` or `[x,y,z]` for Cartesian XY, and `[r,z]` or `[r,phi,z]` for
RZ. A third component must be **exactly zero**. Three-dimensional tracking and
nonzero RZ azimuthal motion are rejected rather than projected.

`basis_to_case` contains the source vector basis columns expressed in the case
basis. It is a right-handed orthonormal `[2,2]` matrix or, in the Python API,
`[N,2,2]` matrices for explicitly realized local bases. It transforms velocity
and direction, not positions. Its orthonormality budget is float64 roundoff;
it is not a physical angle tolerance. Zero directions are rejected.

`frame: "inertial"` is an explicit **caller declaration**. Moving, rotating and
unknown frames require external physical resolution. A native dataset's frame
label or `SubtractMovingFrameVelocity` value does not supply that evidence.
An exact same-component saved realization can carry an unresolved native-frame
qualification in metadata; this only claims transfer of its numeric initial rows.

There are no native expression aliases, equation DSLs, distributions, source RNG
or automatic mass/size reconstruction. An external producer evaluates complex
native expressions at the release positions and physical times before writing
rows. Source rows remain correlated by particle ID, including mass, independent
drag/contact/electrical sizes, volume, charge, release time and weight.

## CSV configuration

This example reads the existing A/P `*_all_states.csv` initial saved realization:

```json
{
  "format_version": 1,
  "table": "particles",
  "columns": {
    "particle_id": "particle_id",
    "release_time_s": "time_s",
    "position_m": ["r_m", "z_m"],
    "charge_number": "Z"
  },
  "units": {"release_time_s": "s", "position_m": "m", "charge_number": "1"},
  "velocity_mode": "vector",
  "velocity_columns": ["vr_m_s", "vz_m_s"],
  "velocity_unit": "m/s",
  "components": ["r", "z"],
  "basis_to_case": [[1, 0], [0, 1]],
  "frame": "inertial",
  "filter_equals": {"participant": "native10", "time_s": 0},
  "metadata": {
    "scope": "same-component-basis realization of the saved native initial rows",
    "unresolved": ["native laboratory-frame meaning is not established by the saved inventory"],
    "property_authority": "missing properties use the checked template's exact particle IDs"
  }
}
```

The selected IDs must be unique. Numeric filters compare parsed numeric values
exactly; string filters compare exact text. No filter is implied: a CSV with
multiple states per ID must explicitly select its initial cohort/time.

`particle_id`, `release_time_s` and `position_m` are mandatory supplied columns.
The selected existing table provides only **missing properties**, joined by
exact particle ID. It does not infer IDs from coordinates or row order. Unknown
IDs are accepted only when all properties below are supplied, allowing a new
population and count to reuse the template geometry and fields:

```text
charge_number, mass_kg, drag_diameter_m, contact_radius_m,
electrostatic_radius_m, displaced_volume_m3, model_weight, material_id
```

Map each supplied scalar property to one CSV column in `columns`. Add explicit
units for dimensionful supplied quantities: length m/cm/mm/um/nm; time
s/ms/us/ns; mass kg/g/mg/ug/ng; volume m3/cm3/mm3/um3. Charge number and model
weight are dimensionless (`1`), as are particle/material IDs. Diameter, contact
radius and electrostatic radius remain independent authorities. Position and
velocity CSV columns are ordered vectors; scalar speed/energy instead use one
`velocity_columns` entry plus `direction_columns` matching `components`.

For example, for energy change the velocity portion to:

```json
{
  "velocity_mode": "energy_direction",
  "velocity_columns": ["initial_energy_eV"],
  "velocity_unit": "eV",
  "direction_columns": ["direction_r", "direction_z"]
}
```

Provide the remaining configuration keys from the full example. `metadata.scope`
states the preparation claim; `metadata.unresolved` must be an explicit list,
possibly empty. Keep native feature/member/time/expression evidence and physical
qualifications here. They are preserved, not interpreted as core settings.

## Provenance and verification

The existing `producer_metadata` carries the upstream provenance and one
`source_builder` record: CSV/config/YAML file SHA-256, template logical content
hash, conversion revision, explicit configuration, inheritance rule and absence
of sampling. No additional manifest or result format is introduced.
The CSV and configuration are parsed and hashed from the same byte snapshots;
template bytes are checked unchanged across public canonical loading. The shared
configuration syntax reader rejects duplicate and merge keys, including ambiguous
JSON unit declarations, before publishing an input case.

```console
uv run --locked ruff check tools/source_builder tests/verification/test_source_builder.py
uv run --locked pyrefly check tools/source_builder --summarize-errors
uv run --locked python -m pytest tests/verification/test_source_builder.py -q
uv run --locked python -m radon cc tools/source_builder -s -a
```

Tests use independent vector, momentum, energy/mass, eV and rotated-basis
expectations, unsupported DOF/frame/direction cases, ID-correlated property
inheritance, explicit new populations, and a public canonical round trip.
The A/P source replay can use each case's current `prepared/full_h0p25us.yaml`
and the corresponding saved CSV with the configuration above; compare every ID's
realized source arrays after public load before any scientific execution.
