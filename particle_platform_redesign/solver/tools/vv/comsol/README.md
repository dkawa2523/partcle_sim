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
| Resolve observed native wall/terminal meaning | `actual_run_receipt.py`, `boundary_response_mapping.py`; observed readback | Terminal cause only where it was observed |
| Build coefficient fields for the inspected Vigus equations | `prepare_vigus_transport.py`; primitive canonical case and explicit parameters | Current canonical case and build manifest |
| Compare deterministic saved x/v/Z or qualify refined levels | `evaluate_matched_trajectory.py`; explicit descriptors and policy | Raw SI differences, coverage and separately qualified verdicts |
| Describe saved stochastic/native populations | `evaluate_saved_population.py`; normalized native CSV and optional public result | Status/time/position/velocity/charge summaries; no pathwise or statistical-equivalence claim |
| Plot actual saved populations/trajectories | `plot_matched_trajectories.py`; explicit inputs and geometry | Geometry-overlaid observations and coverage |
| Check saved local charge / ion equations | `evaluate_aggregate_charge_parity.py`, `evaluate_aggregate_ion_drag_parity.py`; named native primitive columns | Independent formula differences and applicable coverage |
| Compare independently generated reduced fields | `compare_reduced_fields.py`; canonical field artifact and native node export | Field differences; no particle or all-material qualification |
| Prepare the explicit F02 fixed-electric example | `prepare_f02_fixed_electric_case.py` and `cases/f02_*.yaml` | Example-specific current canonical input |
| Export exact P1 connectivity for a native companion | `prepare_comsol_p1_tables.py`; static RZ P1 canonical input | Sectionwise native tables and probe values |
| Exercise native + candidate motion against analytic references | XY and RZ boundary benchmarks below | Current execution and bounded scientific checks |

Native extraction remains an external task guided by the inspection skill.
Bind the actual MPH, solution/member, selection, units and before/after hashes.
The retained Java producers create the explicit XY/RZ benchmark observations;
they are not a generic inspector for arbitrary chamber models.

The old fixed Case-A/P campaign recipes, seed allocation locks, reporting
repair/projection routes and source-hash snapshot tests are retired. Their
removal record is in
[`evidence/retirement_2026_11_23/vv`](../../../evidence/retirement_2026_11_23/vv/README.md);
historical experiment data are not a current execution route.

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

Native Freeze, Stick and Disappear are separate observed outcomes. A missing
boundary ID never proves inlet/outlet escape. `actual_run_receipt.py` consumes
the observed native readback and keeps status, terminal time, response law and
identified boundary cause separate. It does not infer cause from the last
finite point. The source-preserving readback uses no solve/save of the original
MPH; a requested native rerun must use an isolated copy.

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
native stochastic problem. Parameter format v1 is historical execution
evidence; rebuild current inputs with an explicit v2 interpretation.

`build_manifest.json` binds source/parameter bytes, primitive provenance,
particle properties, coefficient inventory and canonical logical hash.
The reference models check their particle-property bindings during run
preparation. A descriptive native comparison and physically validated
transport remain separate claims.


## Deterministic pre-event comparison

`evaluate_matched_trajectory.py` now consumes explicit series descriptors. The
former 287-particle, 41-frame, fixed-charge evaluator and its precision-parity
policy have been replaced; historical v1 evidence is retained unchanged and
cannot be passed through a compatibility branch. The current scope is Cartesian
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

For public results, the prerequisite also checks actual accepted macro counts,
common run start/end, and more than one coarsest base step in the selected
observation window. Three dt labels with the same single macro partition do
not establish refinement. Missing partition evidence (including CSV-only
metadata) remains `NOT_TESTED`; no production step history is added for this
check. Regenerate comparison reports when the evaluator or policy changes;
completed historical reports do not qualify a changed implementation.

Reports include component and Euclidean-vector x/v differences, scalar charge
differences, per-ID/per-time coverage and statistics, exact initial-state
checks and explicit coarse/medium/fine step ratios. RMS uses equal retained
particle-observation rows, not time weighting. Observed order is reported only
for equal refinement ratios and nonzero changes; it is not a smooth-order gate
for nonsmooth transferred fields. Neither finest-pair difference nor a
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

For independent Brownian paths compare time-dependent populations, source and
destination counts, stopping-time/active-time distributions and conditional
speed/charge with uncertainty. Same IDs or seeds do not pair the random
histories. Describe missing rows and survivor conditioning. Practical agreement
does not certify continuous SDE bias, local deposition, adhesion or physical flux.
Normal result-only reuse uses the installed `analyze/export/view` clients.

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
a historical evaluator. Results go to new destinations and are distinct from
the reference inputs. The latest executed comparison and limits are recorded
under `evidence/retirement_2026_11_23/vv`.

At an event-coincident XY frame native and candidate velocities have opposite
left/right continuity. That named row is excluded for the affected wall cases;
pre/post states, terminal tails and actual candidate event payload remain
checked. These benchmarks do not cover arbitrary native FE fields, general
3D, all walls/corners, finite-radius contacts or physical validation.
