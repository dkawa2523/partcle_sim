# COMSOL adapter and V&V contract

COMSOL cases use `case.adapter: comsol` and `inputs.comsol_manifest`. The schema
version 2 manifest is the only source for exported artifacts, field component
mapping, coordinate conversion, time support, force inventory, and boundary
laws. A run config must not repeat geometry, field, force, or wall settings.

## Manifest version 2

The manifest declares four hashed artifacts: `release`, `geometry`, `field`,
and `boundaries`. Every entry contains `path`, `format`, and a lowercase
SHA-256 digest; `size_bytes` is optional. Validation checks the digest before a
case is accepted.

`coordinates` contains the coordinate system, exact axis order, and
`coordinate_scale_m_per_model_unit`. `time` contains `interpolation: linear`
and `support_s: [start, end]`. `model` records the model name, study, dataset,
and solution. Metadata must also identify the positive source solution number,
the non-empty explicit vacuum-domain ID set, and
`geometry_source: explicit_comsol_vacuum_domain_selection`. These resolved
values are copied into the standard run provenance; a textual solution label
or geometry artifact without its originating selection is not sufficient.
Time-support endpoints are compared at float64 ULP resolution rather than with
a fixed tolerance in seconds.

The generic case builder has no physical/time defaults. Every invocation
declares `diagnostic_grid_spacing_m`; a direct mphtxt build also declares
`coordinate_scale_m_per_model_unit`, while `--raw-export-dir` obtains that
scale only from its verified export manifest. Runnable case generation
requires explicit `dt` and `t_end`; geometry-only generation does not request
unused solver times. Missing, non-finite, or non-positive values fail before
the output directory is created. `--geometry-only` cannot be combined with a
field bundle or raw runnable export; the builder rejects that contradictory
state before writing any artifact.

`fields` is keyed by semantic quantity, and that key is the only semantic
identity. Each value contains only `artifact`, `components`, `unit`, and
positive `scale_to_si`; the redundant field-level `name`,
`physical_quantity`, `mesh`, and `interpolation` keys are rejected. Temporal
interpolation is declared once as `time.interpolation: linear`. Built-in keys
(`velocity`, `electric_field`, `density`, and so on) are exact and
case-sensitive. Additional profile quantities such as `ne`, `Te`, and `phi`
must be identifier-shaped scalar keys with exactly one `value` component;
`scalar` itself is not a semantic name. NPZ array names are therefore not
solver API names. The adapter renames and scales them once while constructing
the field provider. v0.2 loads every mapped quantity from the single declared
`field` artifact; a field entry that names another artifact is rejected until
true multi-artifact sampling exists.

The force inventory is explicit: mapping an electric-field quantity does not
enable electric force, and enabling dynamic charge does not enable it either.
Each force entry has a discriminated contract. `drag` uses `enabled` and an
enabled-only required `law`; `electric` uses only `enabled` and always selects
`particle_charge`. Gravity and experimental forces use optional `model` plus a
named `parameters` block validated by the same coefficient rules as native
cases. Enabled gravity requires a coordinate-dimension-matched
`acceleration_mps2`; thermophoresis requires both thermal conductivities; AC
DEP requires both conductivities and a positive frequency. DEP also records
`electric_field_amplitude: rms|peak`; omitted legacy input resolves explicitly
to `rms`, while `peak` is converted to the RMS-equivalent squared-field
coefficient. A non-drag `law`,
unknown parameter, string boolean, alias, or case-normalized model is rejected.
The inventory is parsed into the same immutable semantic force types used by
native YAML. Resolved coefficients and model defaults live in that one model;
the catalog only adds field bindings and the JIT runtime block is a pure scalar
projection with no second validation or default pass. Disabled inventory
entries should contain only `solver_force` and `enabled` unless a retained
model declaration is needed for provenance; unused coefficient payloads should
be omitted. An enabled non-`none` drag requires a semantic velocity field,
electric and DEP require an electric-field mapping, and thermophoresis requires
a temperature mapping. Missing force inputs fail manifest validation instead
of becoming zero flow or another implicit field.

The COMSOL case builder keeps `--drag-law` as the drag source and permits only
the coefficient-free electric force through the shorthand `--force electric`.
Gravity and experimental forces must be supplied with `--force-inventory`
pointing to strict YAML of the form `forces: [...]`; the same typed manifest
validation runs before the output directory is created. Field support does not
define physical geometry. The builder requires one or more explicit
COMSOL `--vacuum-domain-id` values, filters the exported domain elements to
that selection, and constructs walls from the selected domain boundary. This
preserves vacuum-solid interfaces that disappear when the exterior of the
entire multiphysics model is used. `valid_mask` remains only a field-support
contract: leaving it terminates field-backed integration but never creates a
staircase wall or assigns a wall part ID by nearest-grid inference. The v2
manifest metadata records the selected vacuum domain IDs and source solution
number.

Boundary topology is identified before coordinates are written. Surface
elements and edge-entity elements in one mphtxt export share the global COMSOL
mesh-node table, so the builder counts undirected integer node-ID pairs and
assigns each selected boundary edge from the exact edge-entity node pair. It
does not round SI coordinates to a fixed decimal count. Consequently a true
shared edge is removed exactly, while two geometrically close edges with
different node IDs remain distinct at both nanometre-scale and enlarged
similarity scales. Once an NPZ artifact contains coordinates only, 2D loop
validation records an identity tolerance derived from its shortest positive
edge and 64 float64 coordinate ULPs under the same versioned boundary-numerics
policy; no fixed-decimal fallback is used.

The diagnostic SDF grid follows the explicitly requested positive spacing at
the same physical scale; it is never raised to a fixed one-micrometre floor.
An impractically dense request fails before allocating the grid. Point-in-loop
classification and release inward-direction checks use the shortest boundary
edge plus coordinate ULPs, while the manifest's release projection offset and
detection tolerance remain explicit physical input values.

The model-specific external exporter uses one reviewed expression and one
explicit unit per semantic quantity. Model, study, dataset, solution, solution
number, parameter value, mesh tag, and vacuum domains are mandatory; COMSOL
references are resolved before export. The raw export manifest hashes the
source model, exporter configuration, mesh, and sample table. The generic
builder verifies these hashes and provenance and rejects duplicate CLI
provenance instead of silently overriding the export manifest.

Case construction is separate from the runtime adapter. `comsol_case.mesh`
owns MPHTXT and topology, `comsol_case.fields` owns regular field artifacts,
and `comsol_case.contracts` owns exporter and canonical manifest contracts.
`comsol_case.builder` only sequences these operations, while
`comsol_case.cli` is the sole command adapter.

For field-backed OML charging, ion density must be mapped as its own quantity;
it is never inferred from electron density. If an ion-temperature quantity is
named, that exact quantity must exist and cannot fall back to a configured
constant.

## Field storage: mesh-native or resampled grid

A COMSOL case declares exactly one field source, and the manifest's field
artifact `format` records which one the solver samples.

`precomputed_triangle_mesh_npz` (mesh-native) keeps the solution on the mesh
COMSOL solved it on. The exporter evaluates each declared expression at the
mesh vertex coordinates and writes `field_samples_nodes.csv` keyed by
`node_index`, the mphtxt global vertex index; the builder joins on that index,
never on rounded coordinates. Boundary-layer refinement therefore survives
unchanged, and field support is the mesh itself, so it ends exactly at the
vacuum-domain boundary. There is no `valid_mask` and no interpolation stencil
that can straddle the wall, which removes the band of terminated particles a
lattice leaves along every boundary. Every mesh vertex of the selected domains
must carry a finite value; a missing or non-finite node fails the build rather
than becoming a support hole. The exporter restricts every `Interp` feature to
the explicit vacuum-domain selection so a vertex on a vacuum/solid interface is
answered from the vacuum side.

`precomputed_npz` (regular grid) resamples the solution onto the diagnostic
lattice. The lattice must then resolve the thinnest physical layer in the
model on its own, and a cell whose stencil touches a node outside the vacuum
domain is a `MIXED_STENCIL` that stops the particle. Both limits come from the
lattice, not from the solution, so this form is appropriate for reference and
diagnostics rather than for reproducing near-wall physics.

Support classification and value sampling answer different questions on a
mesh. Classification stays strict: a point outside the mesh is unsupported.
Value sampling clamps to the nearest element with barycentric weights clamped
to the simplex, because a trial step that crosses a wall necessarily lands
outside the mesh and the wall hit that replaces it can only be localized from
a finite trajectory. This is the same endpoint-clamp semantics a regular grid
already applies outside its axes; the clamped weights stay a convex
combination, so no value is extrapolated and no physical state is committed
outside the mesh.

`particle-tracer comsol build-case` selects the form from the declared source:
`--field-node-samples` for mesh-native, `--field-bundle` for the grid. They are
mutually exclusive. A `--raw-export-dir` build uses the mesh-native table when
the export manifest declares `field_node_samples_sha256`; presence on disk
alone is never sufficient.

## Release positions

`release_boundary_projection` declares one value, `tolerance_m`: how far from
a boundary an exported release still counts as being on it. Points within that
distance of their declared entity are snapped onto it and left there, which is
where COMSOL puts an inlet particle -- the inlet feature overrides the wall
condition on that boundary. There is no inward displacement: a manifest still
declaring `inward_offset_m` is rejected, because that key asked a previous
solver to move the release off the wall to dodge a spurious self-hit, and that
hit no longer happens. The solver reproduces the COMSOL behaviour
geometrically: a segment that starts on a boundary primitive within the
resolved geometry tolerance and does not cross to the other side is not a hit
on that primitive. The same predicate covers restarts after a reflection and
persistent contact, so no release, reflection, or contact state has to be
displaced to avoid a spurious zero-length hit. A particle arriving from the
interior starts further than the tolerance from the wall line, so a real
crossing is never rejected.

Three layers had to agree for an on-surface release to behave like a COMSOL
inlet, and each answers a different question.

- The hit queries decide what counts as a crossing. The departure predicate
  applies to every one of them through the boundary service, not only to the
  batch prefetch, so a segment leaving its own wall is never a hit no matter
  which path examines it.
- The trace-refinement clearance criterion asks how much room a curved path
  has before its chord could hide a crossing. The segment start is not such a
  place: it is the committed state and its position is exact. A segment that
  begins on a wall has zero clearance there by construction, which would
  otherwise demand a refinement no curvature can satisfy, so the start is
  excluded from that measure while every other probe point still counts.
- Preflight accepts a release that sits on the boundary entity its own
  `source_part_id` declares, and only that entity. A position on any other
  boundary, or outside the geometry, remains a violation.

## Repeated wall contact

COMSOL's particle tracing has no contact model for point particles: a
particle pressed into a surface keeps having its individual bounces
resolved. This solver otherwise regularizes that Zeno behaviour by pinning
a particle to a wall it reflects off twice within one macro step and
advancing it along the tangent, which changes the terminal state from a
bounce sequence to `contact_sliding`. `comsol build-case` therefore writes
`physics.wall_interaction.contact_sliding: false`, so a COMSOL case resolves
bounces until `max_hits_per_step` is spent and then stops visibly instead of
silently applying a model the reference run does not have. Raise
`max_hits_per_step` when a case legitimately needs more bounces per step.

## Steady fields

A steady export has one time sample, so its declared `time.support_s`
collapses to a single instant. That is where the sample sits, not a limit on
when the field is valid: a steady field has the same value at every time and
supports the whole integration window. `simulate()` has always read it that
way, and preflight now does too. Requiring a steady field's support to reach
`t_end` made `check` reject cases `run` integrates without complaint, and a
preflight contradicted by the runtime stops being read. A genuinely transient
field -- two or more distinct time samples -- still has to cover from the
earliest integrated release to `t_end`.

## Canonical tables

Release CSV columns carry SI units in their names. Required common columns are
`particle_id`, `release_time_s`, `mass_kg`, `drag_diameter_m`, `charge_C`, and
`source_part_id`. Cartesian and RZ coordinate/velocity columns are selected by
the manifest coordinate system. `density_kgm3` is optional and is not used as
an alternative source of particle mass. When a COMSOL release row does provide
`density_kgm3`, preflight verifies
`mass_kg = density_kgm3 * pi * drag_diameter_m^3 / 6` within 0.1%. Native cases
do not inherit this COMSOL-only check because their drag diameter may be an
aerodynamic diameter. Missing density is never inferred from mass and diameter.

`boundaries.csv` is the canonical boundary authority. Every row includes the
part and COMSOL entity IDs, role, explicit wall law and
`wall_*` coefficients, material identity, and `metadata_json`. These columns
extend the same canonical boundary contract used by native cases. Geometry boundary
part IDs, boundary rows, and wall models must match exactly; no default
specular wall participates in coverage.

For 2D COMSOL releases, points explicitly identified as boundary releases are
projected to their declared boundary entity and shifted by the configured
inward offset. The adapter never repairs velocity or provenance. Equivalent
projection for arbitrary 3D surface meshes is not implemented in v0.2, so a
3D case requesting boundary-release projection is rejected instead of being
silently approximated.

## Validation workflow

`validate_case(case, detail="summary")` performs the normal, side-effect-free
safety preflight. `detail="full"` retains particle and boundary violation rows
for investigation. For active drag it also samples every integrated release
state and reports the particle Reynolds and relative Knudsen ranges. Relative
Mach is reported only if the field artifact explicitly maps a scalar named
`sound_speed`, `speed_of_sound`, or `c_sound`; no standard-gas sound speed is
invented. This check is explicitly initial-state scope, while the published
Schiller--Naumann limit `Re_p < 800` is also enforced at each runtime drag
evaluation. Validation never writes output files. V&V should additionally
compare field samples, force contributions, trajectories, and wall events with
the recorded COMSOL study/dataset/solution provenance.
Acceleration sample points must name a known `particle_id`; the comparison tool
never borrows mass, diameter, density, charge, or coefficients from the first
release row when an ID is absent or unknown.
Field comparison points must provide every spatial coordinate. Trajectory
comparison accepts long-form CSV tables keyed by `particle_id` and a shared
time column; debug `trajectory.npy` is not silently reinterpreted as that table.
Transient field comparison preserves `time_s` while converting wide references
to long form and requires a unique `(point_id, time_s, field, component)` match.

Trajectory V&V uses the complete ordered half-stage/endpoint trace from every
accepted substep. Motion is first refined by the position/velocity LTE from one
full step versus two half steps; an unresolved maximum schedule is stopped
before collision classification or state commit. A candidate accepted trace is
then refined when its sagitta is not resolved
against geometry SDF clearance, or when a segment is too long to probe an
internal valid-mask island at the backend grid spacing. After a nonterminal
wall hit, the remaining time is replayed through the same segment primitive;
part ID, outcome, terminal state, and multiple-hit order are therefore exact
comparison targets.

Regular-grid motion requires a fully clean interpolation stencil. A mixed
stencil is not permitted to blend invalid-node fill values into COMSOL fields;
if no physical wall crossing explains it, the particle exits field support.
Reaching the trace-refinement limit with unresolved geometry or support risk is
also a terminal safety result, not a silently accepted segment.

OML V&V requires a bracketed current-balance root, normalized current residual
at most `1e-10`, and a positive relaxation time derived from the current
derivative without a configured cap or numerical floor. Dynamic charging is
currently accepted only for 2D regular rectilinear fields; 3D and triangle
charging cases are rejected pending V&V. RZ comparisons are limited to
deterministic no-swirl `(r,z,vr,vz)` cases. Axis crossings use the canonical
map `(r,v_r) -> (-r,-v_r)` for a negative signed-chart radius. Geometry
primitives whose endpoints both lie on `r=0` are treated as coordinate-axis
primitives rather than wall collisions, while material boundaries at `r>0`
retain their wall law. Brownian motion and Cartesian lift remain outside the
v0.2 RZ contract. Persistent material-wall contact reaching an axis endpoint
is not symmetry-continued until contact motion can be split and re-integrated
at axis time; the existing endpoint hold remains in force.

Standard comparison starts from exactly `final_particles.csv`,
`run_summary.json`, and `wall_summary.csv`. Trajectories, wall-event rows, and
force contributions require an explicitly generated debug result.

Large `.mph` files remain in place. Their current byte sizes and SHA-256 values
are recorded in `data/assets.yaml`; removal is allowed only after an external
URI is supplied and a re-download reproduces the recorded digest.
