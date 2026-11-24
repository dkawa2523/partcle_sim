# External COMSOL input and comparison tools

The installed `chamber-particles` CLI owns normal input creation, checking,
simulation, analysis, export and visualization. These development tools add
native evidence, explicit coefficient preparation and comparisons against
independent observations. They are not another solver or an installation
requirement.

Run commands from `particle_platform_redesign/solver` with the locked uv
environment. Use a new output path for each preparation or evaluation.
[The methodology](../../../../vv_methodology.md) owns the meaning of
practical agreement, qualified numerical agreement and physical validity.

## Choose the required route

| Task | Entry and actual input | Result |
|---|---|---|
| Bind native equation/settings observations | `meaning_preflight.py`; explicit semantic inventory with artifact hashes/pointers | Readiness and unresolved meaning; no trajectory verdict |
| Build coefficient fields for the inspected Vigus equations | `prepare_vigus_transport.py`; primitive canonical case and explicit parameters | Current canonical case and build manifest |
| Compare deterministic saved x/v/Z or qualify refined levels | `evaluate_matched_trajectory.py`; explicit descriptors and policy | Raw SI differences, coverage and separately qualified verdicts |
| Describe saved stochastic/native populations | `evaluate_saved_population.py`; normalized native CSV and optional public result | Status/time/position/velocity/charge summaries; no pathwise or statistical-equivalence claim |
| Plot actual saved populations/trajectories | `plot_matched_trajectories.py`; explicit inputs and geometry | Geometry-overlaid observations and coverage |
| Compare local charge / ion equations | `evaluate_aggregate_charge_parity.py`, `evaluate_aggregate_ion_drag_parity.py`; one explicit CSV, current model and chosen allowance | Frozen-state residuals and separately declared model correspondence |
| Compare independently generated reduced fields | `compare_reduced_fields.py`; canonical field artifact and native node export | Field differences; no particle or all-material qualification |
| Import the selected native gas-field example | Installed `inputs import-csv`; `cases/f02_selected_csv.yaml` maps supplied native geometry/field files | Current field bundle and source/selection receipt |
| Export exact P1 connectivity for a native companion | `prepare_comsol_p1_tables.py`; static RZ P1 canonical input | Sectionwise native tables and probe values |
| Exercise native + candidate motion against analytic references | XY and RZ boundary benchmarks below | Current execution and bounded scientific checks |

Native extraction remains an external task guided by the inspection skill.
Bind the actual MPH, solution/member, selection, units and before/after hashes.
The retained Java producers create the explicit XY/RZ benchmark observations;
they are not a generic inspector for arbitrary chamber models.

## Verify the tools

All retained tests use temporary synthetic data or the small tracked native
benchmark reference inputs:

```console
uv run --locked python -m pytest tools/vv/comsol/tests -q
```

This command includes current public-API replay against saved native XY
observations and analytic trajectories. It performs no licensed solve.
The RZ native benchmark is run explicitly when its native reference must be
regenerated. Core tests independently cover charge/force algebra, stochastic
moments, event geometry, walls and axis behavior.

## Meaning and actual observations

For a newly qualified comparison, use `semantic_inventory.json` format 2 and:

```console
uv run --locked python -m tools.vv.comsol.meaning_preflight INVENTORY --output NEW_PREFLIGHT
```

`same_canonical_field_solver_parity` requires the actual participants to
execute the same field and equation meaning. `native_fe_end_to_end_reproduction`
includes native-to-canonical field/geometry transfer error. Only a supported
requested question can receive a qualified numerical PASS/FAIL. Readiness is
not accuracy, and practical saved-data comparisons can proceed while these
qualified claims remain `NOT_TESTED`.

Native Freeze, Stick and Disappear are separate observed outcomes. The saved
population evaluator keeps their status and terminal-time meaning separate and
does not infer inlet/outlet, axis or wall cause from the last finite point.
Boundary-law correspondence requires actual native observations bound through
the selected semantic inventory; the population CSV does not carry a native
boundary-event ledger. A requested native rerun uses an isolated model copy.

## Vigus coefficient preparation

`prepare_vigus_transport.py` prepares the explicit Vigus coefficient equations
from an inspected static canonical primitive case. It preserves realized
particles, geometry, wall laws, time grid and observation clocks, adds the
reference force/charge/noise bindings and selects the existing OU integrator.
It reads no MPH and replays no saved force or charge history.

```console
uv run --locked python -m tools.vv.comsol.prepare_vigus_transport PRIMITIVE_CASE NEW_DIRECTORY --parameters PARAMETERS_YAML --interval-tree-depth BASE_DEPTH --adaptive-max-depth MAX_DEPTH
uv run --locked chamber-particles check NEW_DIRECTORY/case.yaml
uv run --locked chamber-particles run NEW_DIRECTORY/case.yaml -o NEW_RESULT
uv run --locked chamber-particles inspect NEW_RESULT
```

Depths and `time.dt_s` belong to the independent numerical plan; the command
does not inherit a historical Vigus step or use the saved output spacing.
An unused output directory is required. `check` uses the same complete
model/bounds/support preparation as `run`, without starting workers or advancing particles.

The primitive case contains homogeneous particle properties and these static
nodal fields on one layout, with canonical SI units/basis:

| Fields | Unit |
|---|---|
| `gas_velocity`, `ion_speed` | m/s (vector, scalar respectively) |
| `electric_field` | V/m, vector |
| `gas_density`, `gas_viscosity`, `gas_pressure` | kg/m³, Pa s, Pa |
| `gas_temperature`, `ion_temperature` | K |
| `electron_density`, `ion_density` | 1/m³ |
| `electron_temperature_V`, `ion_mass` | V, kg |
| `rotation_rate` | 1/s, specifically d(ug_z)/dr − d(ug_r)/dz in RZ |

The initial YAML separately retains the inspected Talbot, DEP and gravity
bindings, when active. Their PPR gradients, thermal conductivity and mean-free
path are required by those models. The producer does not infer missing forces
or wall laws from filenames. The source-dependent near-zero Saffman
regularization and native-to-P1 transfer require explicit correspondence
evidence; the signed-rotation formula alone does not prove native parity.

The parameter YAML uses exactly `format_version: 2`,
`cmd_slip_coefficients: [C1, C2, C3]`,
`native_particle_density_kg_m3`, `native_capacitance_F`,
`maximum_abs_charge_number`, `brownian`, and a nonempty `native_binding` mapping linking
the inspected source/solution/export receipts. Supply actual native values.
The producer computes friction, independent force diffusion, finite electron
log offset, collection coefficients, ion force polynomial and signed rotation.
Zero E is an error for the literal ion direction; there is no density floor
or substitute direction. Coulomb alone uses the explicit charge cap.

Brownian interpretation is mandatory. For the source force's continuous
white-noise interpretation, use:

```yaml
brownian:
  interpretation: continuous_white_force
```

This generates `D = 6*pi*mu*d*kB*T`. A time-keyed random force evaluated by
classical fixed-step RK4 is a different numerical process. If the same random
value is reused at identical physical times, the velocity increment has
weights `(1/6, 2/3, 1/6)` and adjacent increments share an endpoint value.
For `h = force_normalization_step`, its N-step undamped variance is
`D*h/m^2 * (5*N/9 - 1/18)`, so the long-time white-force limit is `5*D/9`.
Select that external mathematical correspondence explicitly, in a separate case:

```yaml
brownian:
  interpretation: fixed_rk4_white_limit
  step_s: 1.0e-6
  force_normalization_step_s: 1.0e-6
  evidence: "Actual method/time-key readback and constant-coefficient covariance receipt with hashes"
```

The numbers are illustrative; supply inspected values. Unequal normalization
and native steps are rejected. The independent candidate `time.dt_s` need
not equal the native step. The multiplier is derived, never fitted to final
counts or trajectories; the primitive inputs and original continuous case
remain available. The manifest records the interpretation, derivation and
evidence. This limit does not reproduce finite-step joint x/v covariance,
native random paths, shortened event intervals, variable steps or arbitrary
state-dependent noise. It is neither an FDT repair nor physical validation.
In particular, a native parameter named `timestep` must not be assumed to be
the integrator's actual step, and changing only that step can change the
native stochastic problem.

`build_manifest.json` binds source/parameter bytes, primitive provenance,
particle properties, coefficient inventory and canonical logical hash.
Producer revision `vigus_static_coefficient_transport_v3` also declares one
`derived_field_generations` group in canonical provenance. Its primitive and
output identities include actual layout topology/support, coordinate basis,
units and field clocks. Parameters and the homogeneous mass/dimension authority
are bound with the entire ten-field output group. The normal `create_case`
publisher rejects stale dependencies or a missing/modified output before
publication. Source IDs and model weights are not coefficient dependencies.
Record-free prescribed fields remain legitimate independent input; the
publisher does not infer dependencies from their names. Regeneration starts
from the fresh primitive case, rather than partly replacing an existing group.
The field cache, reduced builder and source recipe preserve declared groups
when they retain the fields. This is a freshness check of declared inputs,
not verification of the producer's physical equations.
The installed `create_case` owner checks the full runnable case and publishes
`case.yaml`, `case.h5` and this manifest in the same directory commit.
A preparation or manifest-write failure exposes no incomplete case; retry
uses the same unused destination.
The reference models check their particle-property bindings during run
preparation. A descriptive native comparison and physically validated
transport remain separate claims.


## Deterministic pre-event comparison

`evaluate_matched_trajectory.py` consumes explicit series descriptors. Its scope is Cartesian
XY or RZ meridional **two-component deterministic pre-event** propagation.
Third DOF, stochastic population inference and event/post-event accuracy need
their corresponding external evaluations.

Each descriptor selects stable IDs and an explicit, strictly increasing common
observation clock. Absolute times may have a nonzero origin and nonuniform
spacing. Integration steps belong to the three actual input/run levels, not to
the observation clock. For example (choose real paths, IDs, times and steps):

```json
{
  "label": "requested deterministic case",
  "coordinate_mode": "axisymmetric_rz_meridional",
  "particle_ids": [7, 101, 3099],
  "observation_times_s": [3.0, 3.003, 3.014, 3.031],
  "levels": {
    "coarse": {"step_s": 0.004, "case_path": "coarse.yaml", "result_path": "coarse_result"},
    "medium": {"step_s": 0.002, "case_path": "medium.yaml", "result_path": "medium_result"},
    "fine": {"step_s": 0.001, "case_path": "fine.yaml", "result_path": "fine_result"}
  }
}
```

Paths resolve relative to the descriptor. Public `open_result` readback binds
the actual completed manifest, case/data hashes, integration step, motion mode,
algorithm revisions and selected states. Frames/probes must already contain
the requested times; the evaluator neither interpolates missing observations
nor starts a solve. Initial x/v/Z, ID and time equality, unchanged physical
problem/method across levels, lifecycle, coverage and pre-event scope are
reported. The first observation must be the actual common release state for
the supported deterministic slice. Declaring a step, a label or `deterministic`
in external metadata is not execution proof.

An external normalized CSV level instead specifies `csv_path` and an explicit
`columns` mapping: `particle_id`, `time`, two-name `position`, two-name `velocity`,
`charge`, `lifecycle`, and optional `support_inside`. Position is in metres,
velocity in m/s, charge in elementary-charge numbers; lifecycle transport rows
use `active`. Support accepts `true`/`false`/`unknown` or `1`/`0`. Missing and
nonfinite rows remain in the denominator. An optional `metadata` record with
`path`, `sha256` and `integration_step_pointer` binds an actual JSON readback;
it is explicitly labelled `BOUND_METADATA_NOT_EXECUTION_PROOF`. CSV-only native
authenticity, deterministic/event-free execution and support meaning remain
unqualified. Useful raw finite-active differences are still emitted.

```console
uv run --locked python -m tools.vv.comsol.evaluate_matched_trajectory characterize --series candidate_series.json --output candidate_self.json
uv run --locked python -m tools.vv.comsol.evaluate_matched_trajectory characterize --series reference_series.json --output reference_self.json
uv run --locked python -m tools.vv.comsol.evaluate_matched_trajectory register --candidate candidate_self.json --reference reference_self.json --policy physical_policy.json --output budget.json
uv run --locked python -m tools.vv.comsol.evaluate_matched_trajectory compare --budget budget.json --output comparison.json
```

The policy contains `physical_budgets` for `position`, `velocity` and `charge`,
each with nonnegative `rms` and `maximum` values in the units above. Select these
allowances from the physical decision before inspecting cross differences;
the tool does not derive them from step changes or tune them to the reference.
Optional `meaning_preflight_inventory: {path, sha256}` and `comparison_question`
reuse the existing meaning owner, rechecking its supported conditions against
the actual fine artifact identities. Missing/unsupported meaning leaves
`qualified_solver_parity=NOT_TESTED`, separately from the descriptive
physical-budget PASS/FAIL. CSV metadata alone cannot promote that claim.

Qualified parity also needs optional, independently selected
`self_sensitivity_budgets` in the registered policy. This record has
`candidate` and `reference` entries, each containing position/velocity/charge
`rms` and `maximum` absolute SI allowances. The comparator checks each genuine
medium/fine change against its own allowance. An omitted or exceeded
prerequisite keeps qualified parity `NOT_TESTED` without suppressing raw cross
differences. Do not derive these allowances from observed changes or aim to
match reference precision. Meeting them is empirical resolution evidence,
never a guaranteed absolute error bound.

For public results, the prerequisite consumes a known `macro_grid_revision`,
the actual selected release rows and the shared `time_grid.MacroGrid` policy.
It verifies the completed run count, then reconstructs only the macro intervals
intersecting the selected window. The first interval is shortened to release;
the last retains its actual target even when the last observation is earlier.
Both refinement pairs must have nested interior boundaries, more intervals and
a smaller maximum interval, with at least two coarse intervals and a common
run span. This is a conservative macro prerequisite, not a universal convergence
criterion. A field-knot-dominated window with identical partitions cannot pass
because extra steps elsewhere increase global counts. Reports retain bounded
partition summaries and hashes, not a full particle step ledger.

### Fixed-root execution prerequisite

Missing/unknown clock policy (including CSV-only metadata) leaves macro
qualification `NOT_TESTED` while raw differences remain available. A supported
macro refinement alone does not establish the particle integration intervals.
`fixed_root_execution` therefore adds a separate prerequisite: a reviewed
engine/proposal/event and deterministic method/path policy, with existing
run-wide refinement/depth, residual split, axis crossing, wall interaction,
boundary/topology event and failure counters all zero. The known exact paths
and Cartesian paths without boundary facets need no curved-refinement table;
their other counters must still be present and zero. Domain/support geometry
may exist in a Cartesian case without boundary facets.

This global-zero condition is sufficient to link macro intervals, shortened
by actual release, to particle integration intervals. Positive global work
does not locate it within the selected cohort/window, so even work after the
last observation leaves this prerequisite `NOT_TESTED`. General refined paths
require additional evidence; a dense RK4 path can still be physically
reintegrated by the outer transport loop. A small medium/fine change from the
same accepted partition is repeatability, not independent resolution. Both
macro and fixed-root prerequisites are needed before self-resolution receives
`PASS` or `FAIL`; raw changes, per-side empirical target verdicts and descriptive
cross-budget results remain available. No production step history is added,
and `accepted_particle_path_resolution` still does not certify a general
accepted-path ledger or stochastic strong convergence.
Regenerate comparison reports when the
evaluator or policy changes; completed historical reports do not qualify a
changed implementation.

Reports include component and Euclidean-vector x/v differences, scalar charge
differences, per-ID/per-time coverage and statistics, exact initial-state
checks and explicit coarse/medium/fine step ratios. RMS uses equal retained
particle-observation rows, not time weighting. Observed order is reported only
for bound genuine uniform macro refinements, known fixed-root execution, equal
step ratios, complete pre-event meaning/coverage and nonzero changes. Nonuniform macro grids retain
their measured changes without assigning an order from nominal step labels.
It is not a smooth-order gate for nonsmooth transferred fields. Neither finest-pair difference nor a
comparison PASS is a rigorous error bound, physical validation, universal
COMSOL accuracy or a reason to match COMSOL integration steps. Source hashes
are refreshed at consumption, output reports are created without overwriting,
and CLI exit zero means that the requested report was produced, not numerical
or physical qualification.

For original saved data or any two available participants, use `compare-pair`.
It needs **two actual levels**, and does not require a third convergence level
or new solves. Each participant descriptor uses the same label/mode/ID/clock
fields as above and a single `level` instead of `levels`. That level can be a
public result or mapped CSV. The policy must name an explicit
`comparison_question`, and may record `comparison_intent` and
`known_model_differences` (including noise, force law and native FE versus
transferred-field differences). Its physical budgets have the same structure.

```console
uv run --locked python -m tools.vv.comsol.evaluate_matched_trajectory compare-pair --candidate candidate_pair.json --reference original_pair.json --policy physical_policy.json --output raw_pair.json
```

This saved-pair route always records `qualified_solver_parity=NOT_TESTED`;
raw same-ID x/v/Z differences remain useful for a stochastic original versus
deterministic analogue. Equal IDs or seeds do not pair stochastic histories.
Missing/terminal rows keep their coverage denominator and prevent a full-cohort
descriptive PASS, even if retained rows lie within the budgets. A CSV lifecycle
mapping can specify `active_value` explicitly when the producer uses a value
other than `active`. Neither route evaluates event/post-event accuracy or
requires an artificial reference precision target.


## Saved populations and figures

Use `--help` for the exact column and CLI contracts:

```console
uv run --locked python -m tools.vv.comsol.evaluate_saved_population --help
uv run --locked python -m tools.vv.comsol.plot_matched_trajectories --help
```

Population CSV columns are particle ID, observation/release/stop times, native
current/final status and RZ x/v/Z in SI. Current status owns each observation;
a final status must not be painted backward through time. Stopped coordinates
repeated at many frames count once for terminal capture. Disappeared status
has an unknown cause unless a native event observation supplies it.

The required header names are `particle_id,time_s,r_m,z_m,vr_m_s,vz_m_s,charge_number,`
`native_status,release_time_s,stop_time_s,native_final_status`; extra columns are
retained in the source evidence but are not interpreted. IDs use exact
nonnegative int64 integer text. Revision `saved_rz_population_description_v2`
requires **ascending time groups**, with each ID appearing at most once per
group. ID order inside a group is arbitrary. Older unsorted tables must be
reordered by `time_s` at their external conversion owner before evaluation;
there is no hidden all-history sort or temporary database.

The native input is scanned twice using one opened file. Both passes hash the
bytes they parse and must agree. The first pass binds all identities and
terminal brackets; the second streams frames and exact quantiles. Candidate
frames and boundary-event batches use public result iterators. Memory is
O(N + T + B): IDs/release/terminal information and a few frames/quantile scratch
are O(N), report curves are O(T), and boundary batches are O(B). Exact quantiles
are not constant-memory. Comparison uses exact common stored times, preserves
unmatched clock counts and weights paired differences equally per finite-active
ID/time row. The active-fraction integral separately uses the nonuniform clock.
Each participant's speed/charge quantiles use its own survivors; unknown native
disappearance remains distinct from candidate escape.

For independent Brownian paths compare time-dependent populations, source and
destination counts, stopping-time/active-time distributions and conditional
speed/charge with uncertainty. Same IDs or seeds do not pair the random
histories. Describe missing rows and survivor conditioning. Practical agreement
does not certify continuous SDE bias, local deposition, adhesion or physical flux.
Normal result-only reuse uses the installed `analyze/export/view` clients.

## Frozen charge and ion-force comparisons

These tools consume one explicitly supplied CSV. Run `--help` to read the exact
required columns from the implementation's input declaration. Values use the
units in their column suffixes; charge is measured in elementary-charge numbers.
Particle IDs are exact int64 integers, stored separately from physical floats;
numerically integral spellings such as `1`, `1.0`, and `1e0` denote the same ID.
Each tool requires a dimensionless residual allowance chosen for the reference
precision before inspecting the residual. It is not a trajectory error budget.

```console
uv run --locked python -m tools.vv.comsol.evaluate_aggregate_charge_parity STATES_CSV NEW_REPORT_JSON --current-scale-residual-limit RELATIVE_ALLOWANCE
uv run --locked python -m tools.vv.comsol.evaluate_aggregate_ion_drag_parity STATES_CSV NEW_REPORT_JSON --model-revision relative_flow_screened_collection_orbital_aggregate_ion_v1 --correspondence matched --force-scale-residual-limit RELATIVE_ALLOWANCE
```

Replace the uppercase placeholders with actual inputs and independently chosen
allowances. Ion force comparison selects either the relative-flow revision above
or `electric_field_directed_image_orbital_sensitivity_v1`. Select `matched` only
when the reference uses the selected current equation and primitive meaning;
select `analogous` for a different equation. An analogous comparison emits raw
residuals with `frozen_force_comparison_status: NOT_TESTED`, even when the residual
is small. The declaration is not native-authenticity evidence. Model selection
does not depend on directory names. Only fields used by the selected model are
required.

Charge comparison separates the production capacitance/constants from a second
calculation using the exported single-charge potential increment. This can
expose a constant-convention difference at the supplied states. The exported
potential increment must be positive. The normalized charge residual is
`abs(predicted-exported) / max(abs(ion_rate)+abs(electron_rate), 1 s^-1)`;
all rates count elementary charges per second, so the allowance includes an
absolute scale at low current. Neither tool
starts a solve, fits a coefficient, or establishes integrated trajectory,
statistical, numerical-convergence or physical validity. Reports go to a new
JSON file and bind the input bytes and explicit choices.

## Producer outputs are completed inputs

The selected native CSV mapping is an external input asset consumed by the
installed importer. It refers to real geometry/primitive exports in
`model_dataset`, not to an accepted candidate result:

```console
uv run --locked chamber-particles inputs import-csv tools/vv/comsol/cases/f02_selected_csv.yaml NEW_BUNDLE_DIRECTORY
```

This publishes `bundle.h5` and `report.json` together. The report binds source
hashes, selected cells, coordinate joins and the mapping's explicit axis
correction. Unrecorded native solution/member observations stay unqualified.
Prepare the user's realized sources and choose physical/wall/numerical settings
through the installed current input workflow; the imported field bundle supplies
no default particle size, fixed cohort or release distribution.

Exact-P1 table export publishes the nodal audit CSV, sectionwise tables,
release probes and `table_receipt.json` only after all validation and writes
succeed:

```console
uv run --locked python -m tools.vv.comsol.prepare_comsol_p1_tables CANDIDATE_H5 NEW_TABLE_DIRECTORY
```

Its current native companion scope requires one static RZ triangular P1
layout, the five named static nodal SI primitive fields, and one nonempty
realized internal table. Its probe count follows that table. Connectivity is
preserved in the sectionwise function files; a point-cloud retriangulation is
not the canonical P1 interpolant. This is not an unpaired CSV exporter. A validation/table/
receipt error leaves the public table destination absent and retryable.
These publishers do not overwrite completed input directories.

## Independent native/analytic benchmarks

These fixed small physical inputs are verification examples, not settings for
another chamber. Their native raw measurements are independent of the new
solver; closed-form trajectories supply an additional reference.

* `run_xy_minimal_suite.ps1 -OutputDirectory NEW_DIRECTORY` creates seven
  unsaved Cartesian native models: ballistic, electric, linear drag, exact
  surface departure, specular, stick and hold. The evaluator runs the public
  candidate API at three actual steps and compares both participants with
  closed-form motion. Saved native inputs are in
  `references/xy`; replay with:
  `uv run --locked python -m tools.vv.comsol.evaluate_xy_minimal_suite tools/vv/comsol/references/xy/raw NEW_EVALUATION --run-receipt tools/vv/comsol/references/xy/run_receipt.json`.
* `run_m3c_critical_boundaries.ps1 -OutputDirectory NEW_RAW -EvidenceDirectory NEW_EVALUATION`
  creates one unsaved RZ rectangle with three force-free particles. It checks
  exact surface departure, specular same-macro remaining flight, and axis fold
  against native observations and analytic piecewise trajectories at three steps.
* `uv run --locked python -m tools.vv.comsol.evaluate_curved_wall_line2_convergence NEW_DIRECTORY`
  checks first hit, point, normal and reflected velocity against a unit circle
  with 16/32/64/128 inscribed facets. This measures line2 geometry error, not
  native curved-element parity.

The PowerShell runners accept `-ComsolRoot`; a licensed COMSOL installation
is required only for fresh native observations. Saved-native replay revalidates
raw hashes and observed configuration, records the current evaluator hash and
candidate resolved revisions, and does not require its current source to equal
the source that originally generated the reference. Results go to new destinations
and are distinct from the reference inputs. Each new evaluation records its
actual input/evaluator hashes, resolved candidate revisions and scientific limits.

At an event-coincident XY frame native and candidate velocities have opposite
left/right continuity. That named row is excluded for the affected wall cases;
pre/post states, terminal tails and actual candidate event payload remain
checked. These benchmarks do not cover arbitrary native FE fields, general
3D, all walls/corners, finite-radius contacts or physical validation.
