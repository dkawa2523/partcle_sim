# M3-V: external COMSOL reference-case evaluation

M3-V is an external target-applicability and relevance gate. It does **not**
make `model_dataset/` a golden truth, and it is not imported by
`chamber_particles`. Its purpose is to decide which production physics should
be implemented next before attempting a trajectory match.

The reference matrix has three independent axes:

| Workflow profile | Generic field-production responsibility |
|---|---|
| legacy `caseP` | `imported_external_plasma_fields`: consume fields produced by a plasma solver |
| legacy `caseA` | `reduced_electrostatic`: combine canonical thermal/fluid fields, plasma parameters, geometry, and semantic electrostatic boundary groups |

| Dataset variant | Generic ion-drag selection |
|---|---|
| `formal_iondrag_theory_consistent` | relative-flow screened collection plus orbital scattering |
| `formal_iondrag_image_minimal_corrected` | electric-field-aligned image-style sensitivity closure |

The particle solver sees neither `caseP` nor `caseA`; both routes must produce
the same canonical fields. Ion drag is a separate physics choice. The second
ion-drag model is retained only to measure model-form sensitivity.

## What the run does

1. COMSOL 6.4 loads both MPH files with `ModelUtil.loadCopy`.
2. A read-only Java inspector records model tags, saved studies/solutions,
   particle and electrostatic feature settings, Case-A closure variables, and
   semantic boundary selections. It never runs a study and never saves a model.
3. The Python evaluator checks all 12 exported packages and evaluates sampled
   P15 charge applicability, sampled Epstein applicability, deterministic
   force-time scales, charge histories, outcomes, and the full time-history
   sensitivity between the two ion-drag variants. It also reconstructs the
   saved regularized two-current charge rate and COMSOL Epstein force from
   exported primitives as independent formula-parity checks.
4. Reports use `PASS`, `FAIL`, `NOT_TESTED`, or `NOT_APPLICABLE`. Missing
   production physics is never relabelled as agreement.

The inspector also hashes both MPH files before and after loading. The raw
COMSOL server log is preserved as `comsol_model_audit.txt`; the filtered JSONL
is the machine-readable inventory. `execution_status=PASS` means this external
evaluation ran without a failed gate. It does not override the separate
`physics_certification_status=NOT_CERTIFIED`.

Run from the solver directory:

```powershell
powershell -ExecutionPolicy Bypass -File tools/vv/comsol/run_m3v.ps1
```

The default COMSOL root is
`C:\Program Files\COMSOL\COMSOL64\Multiphysics_copy1`. Paths can be overridden
with script parameters. Results are written to `evidence/m3v/` and contain a
manifest with input hashes.

To rerun only the dataset evaluation:

```powershell
uv run --locked python tools/vv/comsol/evaluate_dataset.py `
  --dataset-root ..\..\model_dataset\cf4_o2_etch_caseA_nonlinear_sass `
  --output-dir evidence\m3v `
  --model-audit-log evidence\m3v\comsol_model_inventory.jsonl
```

The complete external-tool suite includes formula checks, synthetic canonical
fixtures, and historical evidence audits. Run it locally with the reference
assets required by the case configurations:

```powershell
uv run --locked python -m pytest tools/vv/comsol/tests -q
```

Run the relevant modules when an evaluator, configuration, or reference
formula replay changes. No test in this command performs a licensed COMSOL
solve. Several modules nevertheless require historical metadata, templates,
or tables in `model_dataset/` and local `evidence/`; these assets are not all
tracked in Git. The complete suite is therefore an asset-dependent local
gate. The portable CI subset uses synthetic temporary inputs and tracked
source/configuration files; see [the external-tools index](../README.md).
Neither suite certifies native-model agreement.

## Generic meaning preflight

Run `meaning_preflight.py` before writing a case-specific exporter or comparing
numbers from an unfamiliar MPH. It consumes a producer-neutral JSON inventory;
it does not infer physics from COMSOL feature tags, import COMSOL, or import the
solver core:

```powershell
uv run --locked python tools/vv/comsol/meaning_preflight.py `
  path\to\semantic_inventory.json `
  --output path\to\new_preflight_directory
```

`source` must identify the model SHA-256, exact COMSOL version, component,
study, solution, and dataset. Producer-specific extra provenance may remain in
that mapping.

Current inventories use `schema_version: 2`; declaration-only version 1 cannot
certify a current comparison. The `layers` mapping has seven fixed keys: `coordinate_dof`, `formulation`,
`field_representation_owner_recovery`, `source`, `boundaries`, `models`, and
`integration`. Each selected inventory item records an `id`, `scope`
(`required`, `excluded`, or `unresolved`), its source and canonical meanings,
its `mapping` (`direct`, `adapter`, `unsupported`, or `unresolved`), an optional
adapter action, evidence, a `binding` list, and a reason. Each binding contains
`expected` and `observed` references with `path`, `sha256`, and an RFC 6901 JSON
`pointer`. Paths resolve relative to the inventory. The checker reads both
artifacts, verifies their hashes, and compares the selected values and JSON
types exactly. Evidence strings are annotations; an empty binding cannot make
a required direct mapping supported. Missing files, mismatches, nonfinite JSON,
and status declarations, including nested wrappers, stop before publication.
The tool derives only these four
classifications:

- `SUPPORTED`: the required meaning is canonical and its selected artifact
  values match the locked expectation;
- `ADAPTER_REQUIRED`: the meaning is explicit but an external conversion,
  realization, or recovery step remains;
- `NOT_APPLICABLE`: the item is excluded from the registered scope or a
  required feature has no supported canonical counterpart;
- `AMBIGUOUS`: ownership, meaning, or evidence is unresolved. A missing layer
  is ambiguous, never silently irrelevant.

The output directory is no-clobber and contains `comparison_conditions.json`
and `comparison_summary.json`. Both always keep two questions separate:

1. `same_canonical_field_solver_parity` requires the reference and candidate
   to name the identical canonical field identity. It excludes field
   production, import, and recovery error.
2. `native_fe_end_to_end_reproduction` requires a COMSOL-native FE reference,
   a canonical candidate field, and explicit adapter lineage. Its result
   includes field representation and import/recovery error.

The first question excludes the `field_representation_owner_recovery` layer
and instead requires identical canonical field identities. The second question
includes all seven layers.

Each question has its own `classification`, `outcome`, evidence, and allowed
claim. A `PASS` or `FAIL` outcome is rejected unless that question's condition
is `SUPPORTED`; the two questions cannot share one aggregate error number.
Exit code zero means every requested question is independently `SUPPORTED`;
an unrequested question does not block it. Other requested-question states
return code 2 after writing the reports.

Hash and value equality establish artifact binding. The producer owns native
readback authenticity, and the scientific evaluator owns numerical probes and
tolerances. This checker does not turn a declared `PASS` into an observation or
certify an unexecuted trajectory, force assembly, or physical model.
Current comparison consumers call `require_supported_comparison` on a
hash-bound inventory reference. It reopens the inventory and selected artifacts
and requires the named question to be supported, rather than trusting a saved
summary. Current candidate execution cannot gain a new parity verdict merely
by pairing with a historical COMSOL participant.

Terminal boundary semantics use `boundary_response_mapping.py`. Actual native
boundary IDs and response settings are preferred. With no observed IDs, a
terminal action can identify only a unique matching semantic group, after other
terminal causes have been excluded and the actual response map is complete.
A partial map cannot establish uniqueness by dropping unknown or overlapping
features. Mixed groups, unknown IDs, action mismatch,
or unexcluded causes remain `AMBIGUOUS`; the axis is not inferred to be a wall
from a lifecycle code.

## Deterministic matched-case trajectory slice

The historical blocked manifest `cases/m3v_matched_caseA_100nm.yaml` describes
why the original saved trajectory could not be used as a match; it does not
specify the later accepted companion and is not the current M3-C0b readiness
authority. The original saved Case-A 100 nm
trajectory has Brownian enabled, advances a
COMSOL-specific dynamic charge law, and includes ion drag, thermophoresis,
lift, and DEP. It remains relevance and formula-provenance evidence and is not
reused as a deterministic trajectory reference.

The first matched companion is deliberately narrower:

- Case-A 100 nm, axisymmetric RZ no-swirl, fixed classical RK4 at 10, 5, and
  2.5 us with a common 10 us output grid;
- fixed charge number `-1`;
- Brownian, ion drag, thermophoresis, lift, and DEP disabled;
- common deterministic forces limited to Coulomb electric,
  `epstein_linear_v1` with `delta=1+0.9*pi/8`, and standard
  gravity/buoyancy;
- the same release IDs and initial state;
- trajectory rows compared only on continuous segments, with boundary events
  carried by a separate sparse event ledger.

This does not redefine the production physics catalog. It isolates the first
layer at which the two independently implemented calculations diverge.

Two questions are kept separate. `same_canonical_field_solver_parity` asks
whether both solvers advance the identical canonical field and matched model
meaning. `native_fe_end_to_end_reproduction` keeps COMSOL's native
finite-element field and therefore includes field production/import/recovery
error. A metric from one question is never reported as the answer to the other.

The native-field companion is reproducibly generated by:

```powershell
powershell -ExecutionPolicy Bypass -File `
  tools/vv/comsol/run_comsol_matched_reference.ps1
```

That runner copies the audited theory-consistent MPH into an isolated output
directory, loads that copy with `ModelUtil.loadCopy`, and never saves a model.
It verifies the source and copy SHA-256 before running and verifies the source
again afterwards. The temporary 545 MB copy is removed after success. The
source hash remained
`3bbf08e3469758313eac5de473a7a0dd4cc9a6f72c9722393229f0b856e9b524`.

The first reference deliberately stops at `4e-4 s`, before the first positive
terminal/event time in the audited saved run (about `4.5783e-4 s`). Each step
size therefore contains 287 particles at 41 common output times, 11,767 rows,
all active and finite. The default output is
`_out_m3v_matched/caseA_100nm_common_v1/`. Each step-size directory contains
the raw COMSOL-wide export plus normalized trajectory, force, field-probe, and
event-observation CSVs. Root receipts record provenance, all artifact hashes,
and COMSOL self-convergence.

Its observed self-differences are:

| refinement | position RMS / max | velocity RMS / max |
|---|---:|---:|
| 10 us to 5 us | `1.404e-9 / 3.456e-8 m` | `9.430e-6 / 1.736e-4 m/s` |
| 5 us to 2.5 us | `6.020e-10 / 1.405e-8 m` | `4.128e-6 / 9.370e-5 m/s` |

The effective observed orders are only about 1.22 for position and 1.19 for
velocity. The native-field trajectory comparison failed: at 2.5 us, position
RMS/max differences were `1.513e-6 / 9.014e-6 m` and velocity RMS/max
differences were `9.526e-3 / 4.422e-2 m/s`. Frozen-state force replay showed
that the force formulas agreed, while native COMSOL electric-field sampling
differed from canonical P1 by relative L2 `2.903e-2`. This is retained as
field-production evidence, not used to certify the trajectory integrator.

Feeding the same P1 node values through COMSOL's ordinary scattered-linear
interpolator also failed the final trajectory gate. COMSOL constructed its own
Delaunay triangulation, which differed from the canonical triangle
connectivity. The resulting electric-field relative L2 difference was
`8.287e-4`; position RMS/max differences were
`5.046e-8 / 1.673e-6 m`, and velocity RMS/max differences were
`2.887e-4 / 6.402e-3 m/s`. This run is retained only as diagnostic history.

### Exact-connectivity integrator/physics companion

For the parity question, COMSOL's
[documented sectionwise interpolation format](https://doc.comsol.com/6.4/doc/com.comsol.help.comsol/comsol_api_fileformats.53.06.html)
is used. Each field file contains `%Coordinates`, `%Elements`, and `%Data`, so
COMSOL and the candidate use the same node values **and the same triangular P1
connectivity**. A small external test verifies that coordinates and values are
preserved and that zero-based canonical connectivity is written one-based for
COMSOL.

Run a fresh isolated companion from the solver directory (the output directory
must not already exist):

```powershell
$freshOutput = "_out_m3v_matched/caseA_100nm_canonical_p1_sectionwise_" + `
  (Get-Date -Format "yyyyMMdd_HHmmss")
powershell -ExecutionPolicy Bypass -File `
  tools/vv/comsol/run_comsol_matched_p1_reference.ps1 `
  -OutputDirectory $freshOutput
```

The runner prepares the sectionwise files from the hashed canonical candidate,
copies the audited MPH, verifies the copy hash, compiles the external Java
companion, runs COMSOL with `-nosave -np 1`, normalizes all three time-step
exports, deletes the isolated MPH and generated class/status files, verifies
that deletion, and only then writes provenance and recursive artifact hashes.
The receipt records the source hash before and after, `ModelUtil.loadCopy`,
`model_saved=false`, COMSOL version, candidate hash, and hashes of the runner,
Java source, preparer, and normalizer.

The exact-connectivity companion also reconstructs the ideal-gas pressure used
by COMSOL's Epstein implementation from the same P1 density and temperature,
using the audited 80/20 CF4/O2 mixture molecular mass. Leaving pressure on the
native COMSOL field would mix two field owners and produces a measurable drag
mismatch. The retained deterministic physics is therefore exactly:

- fixed charge number `-1` and Coulomb electric force;
- linear Epstein drag with `delta=1+0.9*pi/8`;
- standard gravity and gas-density buoyancy;
- Brownian, dynamic charging, ion drag, thermophoresis, lift, and DEP disabled.

### External trajectory figures

`plot_matched_trajectories.py` reads only the normalized candidate/reference
CSVs.  It does not import `chamber_particles`, COMSOL, or `model_dataset`.
Before rendering, it requires exact particle-ID identity. Each side's finite
R-Z path is drawn independently; missing escape suffixes are excluded without
reconstruction, and receipt metrics use only common finite
`(particle_id, time_s)` keys. Both input SHA-256 values are recorded in
`receipt.json`.

For the accepted exact-P1 fine-step pair:

```powershell
uv run --locked python tools/vv/comsol/plot_matched_trajectories.py `
  --candidate _out_m3v_matched/candidate_caseA_100nm_v1/candidate_trajectory_2p5us.csv `
  --reference _out_m3v_matched/caseA_100nm_canonical_p1_sectionwise_v4/dt_2p5us/trajectory_reference.csv `
  --output-directory evidence/m3v/trajectory_figures_caseA_100nm_p1_2p5us_v1 `
  --label "Case A 100 nm exact-P1, dt=2.5 us"
```

The immutable output directory contains one all-particle R-Z overview SVG and
optional PNG. `particles/` contains one R-Z-only SVG for every aligned
particle ID, while `index.html` is a lazy-loading atlas over all of them.
It intentionally creates no difference, `r(t)`, or `z(t)` plots. COMSOL is a
solid orange line and the candidate is a dashed blue line so the encoding
remains distinguishable without relying on color. PNG creation uses an
installed Chrome/Edge executable and adds no Python plotting dependency; use
`--skip-png` for SVG-only environments.

The same command can be applied to 10 us and 5 us accepted exact-P1 pairs, or
to retained native-field/scattered-P1 diagnostic pairs, but each pair receives
its own output directory and receipt.  A figure is descriptive evidence only:
it does not alter the comparison gate or turn an unaccepted diagnostic pair
into a pass.

For the completed 287-particle, 41-frame, pre-event slice, candidate P1 versus
COMSOL sectionwise field differences were at floating-point roundoff: electric
field relative L2 `7.550e-16`, gas velocity `5.562e-16`, density
`1.039e-16`, and temperature `1.169e-16`. COMSOL-exported force replay gave
relative L2 residuals `1.665e-15` for electric force, `7.022e-16` for Epstein
drag, `1.036e-16` for gravity/buoyancy, and `1.756e-15` for total
acceleration.

The canonical `v4` layer report uses neutral candidate/reference field names and
classifies the first detected discrepancy as `NONE_WITHIN_ROUNDOFF` at the
declared relative-L2 limit `1e-12`. The older `v2` JSON remains immutable
historical evidence, but its native-field-oriented key and fixed
`field_representation_or_sampling` label are superseded. The normalized
sectionwise v2/v3/v4 scientific CSVs are byte-identical; v4 also records hashes
matching the current runner, Java source, table preparer, and normalizer.

Observed sectionwise-reference self-differences are:

| refinement | position RMS / max | velocity RMS / max |
|---|---:|---:|
| 10 us to 5 us | `4.953e-12 / 1.388e-10 m` | `4.246e-8 / 7.457e-7 m/s` |
| 5 us to 2.5 us | `1.176e-12 / 1.727e-11 m` | `1.030e-8 / 9.938e-8 m/s` |

Tolerances were registered from the independent candidate and reference
5-to-2.5 us self-differences before inspecting their fine-grid difference.
The 2.5 us cross comparison then passed with identical initial states:
position RMS/max `8.643e-16 / 2.172e-15 m`, and velocity RMS/max
`1.999e-14 / 1.413e-13 m/s`. The registered limits were respectively
`2.807e-12 / 3.499e-11 m` and `2.060e-8 / 1.988e-7 m/s`.

This PASS supports only the hashed deterministic Case-A 100 nm, exact-P1,
pre-event slice. It establishes parity of the common field interpolation,
force subset, and fixed-step RK4 trajectory advance. It does not establish
native COMSOL field reproduction, Brownian or dynamic-charge parity, boundary
behavior, other particle sizes, or universal COMSOL-equivalent accuracy.
Boundary behavior remains `NOT_TESTED_PRE_EVENT_WINDOW`, and COMSOL RK stages
remain `NOT_EXPORTED`.

The separate legacy `compare_matched_case.py` validates a fully configured
manifest, verifies every registered artifact hash, aligns rows by semantic
keys, and compares five layers in this order:

1. primitive field/support probes;
2. frozen-state force and acceleration probes;
3. RK stage or one-step probes;
4. continuous trajectory rows;
5. sparse boundary-event rows.

The first failed layer is reported as field/geometry sampling, force/charge
RHS, integrator/stage coupling, accumulated trajectory, or boundary/wall
mapping. A later layer never overwrites an earlier cause. Header-only event
files are `NOT_TESTED`, not a boundary pass. Missing rows and key differences
are alignment failures rather than numeric error.

The earlier generic manifest at
`cases/m3v_matched_caseA_100nm.yaml` is retained as a historical fail-closed
readiness example for the unmatched saved trajectory. It is **not** the
accepted exact-connectivity certification path. Its blocked status applies
only to that historical unmatched input. Its safe
preflight can still be inspected from the solver directory:

```powershell
uv run --locked python tools/vv/comsol/compare_matched_case.py preflight `
  tools/vv/comsol/cases/m3v_matched_caseA_100nm.yaml `
  --artifact-root _out_m3v_matched/caseA_100nm `
  --output _out_m3v_matched/preflight.json
```

Do not unblock that manifest by pointing it at the later companion. The
exact-connectivity result is owned by `prepare_matched_candidate.py`,
`run_comsol_matched_p1_reference.ps1`, and the three-phase
`evaluate_matched_trajectory.py` workflow used for the result above. It keeps the
  historical decision and the separately generated reference identities intact.
At this historical matched-slice checkpoint Freeze/event mapping was untested.
The later isolated M3-C0 boundary probe below characterizes COMSOL semantics;
production boundary parity remains untested.

The legacy guarded runner remains available only for future manifests that
fully register their own inputs and tolerances:

```powershell
powershell -ExecutionPolicy Bypass -File tools/vv/comsol/run_matched_case.ps1 `
  -Manifest tools/vv/comsol/cases/m3v_matched_caseA_100nm.yaml `
  -ArtifactRoot _out_m3v_matched/caseA_100nm `
  -OutputDirectory evidence/m3v/matched_caseA_100nm_v1
```

That runner performs preflight first and does not launch or control COMSOL. It
refuses an existing output directory and writes no comparison when preflight is
blocked. It was not used to produce the exact-P1 PASS reported here.

Results produced by that legacy comparator carry
`comsol_equal_accuracy=NOT_CLAIMED` and `golden_truth=NOT_CLAIMED`. The accepted
three-phase evaluator instead records the narrower
`same_accuracy_for_hashed_case_and_time_window` claim. In either workflow, a
PASS means only “within the preregistered tolerances for these hashed
artifacts.” COMSOL and candidate time-step convergence, field/export
uncertainty, and boundary-event convergence remain separate accuracy evidence;
absent evidence stays `NOT_TESTED`.

## F02 historical field-production closeout and schema-v2 fixture

F02 is a separate external integration track. It converts one provider-specific
mixed-mesh CSV package to canonical P1, runs the first-party reduced
electrostatic builder, compares generated fields on the same exported nodes,
and feeds the completed canonical field bundle to the existing trajectory
solver. Neither the adapter nor this comparison module is imported by the
solver core.

After running the adapter and builder commands documented in their READMEs,
the following is the regeneration workflow from the solver directory. All
output and report paths must be absent; the tools do not overwrite artifacts.
Executing the schema-v2 commands does not by itself promote a new F02 closeout.

```powershell
$reference = "../../model_dataset/cf4_o2_etch_caseA_nonlinear_sass/cases/formal_iondrag_theory_consistent/caseA_100nm/external_reproduction"
uv run --locked python tools/vv/comsol/compare_reduced_fields.py `
  _out_f02_acceptance/case_a_fields.h5 `
  "$reference/input_fields/background_fields_mesh_points.csv" `
  "$reference/config/background_field_column_dictionary.csv" `
  _out_f02_acceptance/field_comparison.json `
  --coordinate-tolerance-m 1.5e-14
uv run --locked python tools/vv/comsol/prepare_f02_fixed_electric_case.py `
  _out_f02_acceptance/case_a_fields.h5 `
  _out_f02_acceptance/case_a_fields_with_source_v2.h5 `
  --report _out_f02_acceptance/source_report_schema_v2.json
uv run --locked chamber-particles check tools/vv/comsol/cases/f02_fixed_electric.yaml
uv run --locked chamber-particles run tools/vv/comsol/cases/f02_fixed_electric.yaml `
  -o _out_f02_acceptance/fixed_electric_result
uv run --locked chamber-particles inspect _out_f02_acceptance/fixed_electric_result
```

The historical `f02_closeout_v1` representative run used 1,987 nodes and 3,779
P1 cells. Its builder used 1,826 free nodes, 2,821 total linear iterations,
final relative residual `1.8441e-13`, and charge-balance error `5.8498e-21 C`.
That historical trajectory smoke used the former schema/source path, completed
10 macro steps and three frames (96 rows), and produced no wall or failure
event.

The schema-v2 fixture now writes 32 explicit equal-revolved-area wafer rows,
including each facet, strict-interior facet coordinate, velocity, release time,
and particle properties, to a new canonical HDF5 file. The fixture owns new
provenance linking the input content/provenance hashes and realization policy.
Fixture generation and `chamber-particles check` have been exercised, but the
schema-v2 trajectory has not received a formal F02 rerun or closeout. It must
not inherit the v1 trajectory result or acceptance status.

Field norms use `2 pi r` axisymmetric lumped P1 volume weights. Shared boundary
corners are reported separately from exclusive group nodes. This is a
same-exported-node descriptive comparison, not a pass/fail fit to COMSOL;
independent mesh convergence remains `NOT_TESTED_SINGLE_REFERENCE_MESH`.
Trajectory parity and COMSOL `Freeze` parity are not established. In
particular, the unexercised `gas_inlet: escape` law in the historical smoke case
is not a translation of COMSOL `Freeze`. The immutable historical hashes and
coverage matrix are in `evidence/f02/f02_closeout_v1.json`; they certify only
the v1 artifacts named there, not the schema-v2 fixture.

The focused external-tool test gate is:

```powershell
uv run --locked python -m pytest `
  tools/comsol_adapter/tests `
  tools/electrostatic_builder/tests `
  tools/vv/comsol/tests
```

## M3-C0a offline reference lock

`lock_m3c0_reference.py` is the first Stage 3 gate.  It reads the existing 12
reference packages without launching COMSOL or importing the production solver.
It refuses an existing output directory and records the two MPH hashes, all
required package artifact hashes, package dimensions, selected exact charge and
force expressions, required field names/units, paired-variant differences, the
candidate step series, and 32 preregistered seeds per package.

From the solver directory:

```powershell
uv run --locked python tools/vv/comsol/lock_m3c0_reference.py `
  --config tools/vv/comsol/cases/m3c0_reference_lock.yaml `
  --output evidence/m3c0/reference_lock_v1
```

The accepted offline artifact is deliberately
`PARTIAL_RERUN_REQUIRED`.  Model identity, the 12-package shape, formula and
parameter locking, required primitive names/units, existing Disappear status,
and the 384-row future seed cohort pass.  The current histories are Brownian-on
single-seed references, contain no positive Freeze sample or RK-stage export,
and do not establish an admissible full-physics time-step series.  The Case-A
variant pairs differ only in the selected ion-drag force.  The Case-P pairs also
contain a lift-expression difference and independently re-exported mesh-point
fields that are not byte-identical, so the ion-drag-only gate fails closed.

This result does not change M3-V history and does not certify any trajectory.
`PARTIAL_RERUN_REQUIRED` is the immutable decision for this offline v1 snapshot,
not the current M3-C0b program status.
At the offline-lock checkpoint, the next M3-C0 slice required one purpose-built
no-clobber runner and Java exporter:
use the audited theory MPH as the common isolated base, inject only the audited
alternative ion-drag expression, disable Brownian, retain the 121 output frames,
and add bounded accepted-step/RHS probes plus separate Freeze/Disappear
microcases.  The original MPH files must remain unsaved.

## M3-C0b deterministic pre-event step pilot

The external M3-C0b runner now executes one live, no-clobber protocol. It uses
the audited theory MPH through `ModelUtil.loadCopy`, invokes COMSOL with
`-nosave -np 1`, stages the exact Java/configuration/normalizer sources into the
output, verifies the source MPH hash before and after, and treats cleanup
failure as an incomplete run.

From the solver directory:

```powershell
powershell -ExecutionPolicy Bypass `
  -File tools/vv/comsol/run_m3c0b_caseA_100nm_pilot.ps1 `
  -OutputDirectory _out_m3c0b\caseA_100nm_theory_pre_event_v6_<new>
```

The v6 sequential-confirmation protocol covers the Case-A 100 nm theory case
only. Brownian and Saffman
are disabled while dynamic charge, electric, relative-flow ion drag, Epstein
drag, Waldmann thermophoresis, free-molecular lift sensitivity, DEP, and
gravity/buoyancy remain enabled. It evaluates all 287 particles at 46 output
times through 450 microseconds, before the first boundary event seen in the
preceding full-time characterization. The configuration and execution receipt
both fix classical RK4, relative tolerance `1e-8`, `WallAccuracyOrder=1`, status
storage on, and extra storage off.

The initial full 30 ms v3 run at 10/5/2.5 microseconds was `CHARACTERIZED`, not
accepted: observed orders were position `0.659`, velocity `0.951`, and charge
`0.241`. The first pre-event v4 refinement failed its preregistered charge-order
gate (`0.716 < 0.75`), and the threshold was not changed. The planned v5
extension was formerly reported as passing all gates, but its position relative
L2 divided by the norm of absolute global R-Z coordinates. That metric was
origin-dependent: an independent +10 m translation changed it by about 98.86%.
The former v5 compact PASS/admission is therefore `INVALIDATED`; the retained v5
raw artifact is historical `CHARACTERIZED` evidence, not current step authority.
On the identical 0.625 -> 0.3125 microsecond payload, the old absolute-position
value was `6.145964082982306e-7`, while the corrected displacement-normalized
value is `5.8064307113296814e-5` (`94.48x` larger).

v6 replaces that position criterion with the origin-invariant displacement
metric

```text
||x_h - x_h/2||_2 / ||x_h/2 - x_h/2(t0, particle)||_2
```

The disclosed corrected v5 overlap (`5.8064307113296814e-5`) was used to calibrate the
operational `1e-4` limit. v6 then sequentially confirmed it on the previously
unseen 0.15625 microsecond result. Under the same +10 m translation, the corrected
metric is invariant up to floating-point roundoff; this property is also fixed by
an automated regression test. The
0.625/0.3125 microsecond normalized trajectory, force, RHS, and event outputs
shared with v5 are bitwise identical; only provenance and receipt summaries
differ.

All 13,202 records in each of the 0.625/0.3125/0.15625 microsecond runs remain
active. The v6 fine-pair result is:

| fine comparison | position displacement relative L2 | velocity relative L2 | charge relative L2 |
|---|---:|---:|---:|
| 0.3125 -> 0.15625 us | `3.102727085428027e-5` | `3.92483251084038e-5` | `1.3511393490811483e-6` |

The v6 position, velocity, and charge observed orders are respectively
`0.9041136`, `0.944312`, and `1.123838`. The PASS is only the one-case,
287-particle, 46-frame, 0--450 microsecond pre-event operational fixed-step
selection gate for 0.15625 microseconds. It is not formal RK4-order evidence,
absolute physical accuracy, solver agreement, 30 ms/event convergence, another
size/case/ion-drag variant, Freeze/Disappear, or Brownian certification. The
limit is a sequential operational calibration, not an independently derived
universal tolerance. Current compact evidence is in
`evidence/m3c0/deterministic_pilot_v6/`. The full raw
`_out_m3c0b/...pre_event_v5...` run directory remains unchanged; compact v5
evidence is now marked `INVALIDATED`/`CHARACTERIZED`, never accepted or current
position-adequacy evidence.

## M3-C1 frozen-state checkpoint and integrated comparison

The first M3-C1 layer replays producer formulas at the locked Case-A 100 nm
pre-event saved states. The original v6 export did not contain the PPR heat-flux
primitive needed to test thermophoresis, so one minimal COMSOL supplement was
run. It loaded the unchanged source MPH with `ModelUtil.loadCopy`, used
`-nosave -np 1`, and exported only the fine-step PPR thermophoresis evidence.
At that frozen-state checkpoint, no common-field or additional trajectory
reference had been rerun.

The supplement contains 13,202 finite active saved rows with exactly the same
particle/time keys and R-Z coordinates as v6. Replaying the Waldmann producer
form gives a maximum component-scale normalized residual of
`4.4046499933294035e-16` and a global relative L2 of
`1.0992667494449471e-16`. Together with the existing frozen-state report, this
closes all 8/8 producer-form replay groups. The compact evidence is in
`evidence/m3c1/pre_event_frozen_rhs_v1/` and
`evidence/m3c1/thermophoresis_ppr_v1/`. This is saved-row formula parity only;
it does not certify physical applicability, the continuous accepted path, or
an integrated trajectory.

The integrated solver candidate is an **exported exact-connectivity P1** field.
Only the COMSOL reference uses its native finite-element field. The candidate
must therefore be described as `exported-P1 candidate vs COMSOL native-field
reference`, never as a candidate native-field run. The original strict run was
`BLOCKED`: every sampled initial state was locally admissible, but the global
continuous-applicability enclosure included remote field extrema and rejected
the rows at time zero. That blocker record remains historical evidence; a
relaxed or nonrestrictive gate is not an acceptable comparison result.

P19-L is complete. At that milestone it preserved the original fixed-step RK4
proposal, endpoint, and rev3b event ordering, while an integrator-owned dense
path plus bounded local-cell ranges restricted only the applicability certificate. An actual
`model_applicability` violation remains distinct from
`indeterminate_applicability_certificate`. Its then-current global event-query
ownership is historical; the later M3-C1 event v14 narrowed eligible dense broad-phase
queries while retaining the global safety authority. No second solver or COMSOL
branch was added.

The strict Case-A 100 nm pre-event exported-P1/native-field comparison is now
complete. The 0.625, 0.3125, and 0.15625 us candidate runs each contain 287
particles and 46 frames with no boundary event or particle failure. Each
manifest honors its requested macro step. Under event v13, event/path certification nevertheless
reintegrates dyadic leaves: all three runs contain exactly 8,450,307 accepted
particle pieces, with maximum depths 8/7/6 and the same deepest nominal leaf
width of 2.44140625 ns. The fine-pair relative-L2 changes are
`2.6773e-14`, `1.4486e-14`, and `5.4562e-15` for position, velocity, and
charge. They remain valid historical evidence of macro-step-halving
production-output stability below the configured float64 representation-scale
floor. Artificial subdivision made them unsuitable as evidence of independent
temporal convergence, RK4 order, error below that floor, or three distinct
effective grids. The later v14 three-step result is reported below as historical M3-C1 evidence.

The narrow pre-event operational COMSOL native-field reference self-convergence
gates pass. The fine
cross-representation comparison fails all six preregistered gates:

| quantity | RMS | maximum | status |
|---|---:|---:|---|
| position | `3.437485727086595e-4 m` | `2.660729292906687e-3 m` | `FAIL` |
| velocity | `1.919910730173562 m/s` | `11.668961060303342 m/s` | `FAIL` |
| charge | `17.58471490428266 e` | `157.15163693423065 e` | `FAIL` |

The compact decision, source hashes, and gate rows are in
[`evidence/m3c1/case_a_100nm_pre_event_v6/`](../../../evidence/m3c1/case_a_100nm_pre_event_v6/).

This is a comparison between the solver's exported-node exact-connectivity P1
representation and COMSOL's native finite-element fields. It is not a same-field
solver-agreement result and does not certify the physical applicability of the
effective-gas, point-dipole, lift-coefficient, heat-flux, or other benchmark
assumptions. The failure activates the preregistered conditional next step: an
isolated COMSOL diagnostic that uses the same exact-connectivity common field
and the full deterministic physics composition. The older M3-V common-field
runner covers only fixed charge and three common forces; it cannot stand in for
the dynamic-charge, ion-drag, thermophoresis, DEP, and lift diagnostic required
here. No core equation, applicability gate, or threshold is changed to reduce
the residual.

That conditional diagnostic is now complete. `prepare_m3c1_common_p1_tables.py`
materializes the candidate's 17 canonical nodal fields (22 scalar components),
exact TRI3 connectivity, 287 initial states, and field probes as external COMSOL
tables. `RunM3C1CaseA100CommonP1.java` evaluates the same dynamic charge and
seven deterministic force contributions from those tables. The runner uses an
isolated `loadCopy` model, `-nosave -np 1`, verifies 26 staged inputs and the
source MPH before and after execution, and does not modify the solver core.

Before any post-t0 cross difference was read, the evaluator fixed the
self-convergence requirements, a 4096-ULP initial-state criterion, and nine
position/velocity/charge RMS, maximum, and relative-L2 gates. All 1,435 t0
state values pass. The fine common-field comparison covers 287 particles and 46
frames from 0 to 450 us with no boundary event:

| quantity | RMS | maximum | relative L2 | status |
|---|---:|---:|---:|---|
| position | `4.06807283316903e-13 m` | `1.2035778717837921e-12 m` | `2.0316001855367802e-10` | `PASS` |
| velocity | `2.10847522277453e-9 m/s` | `3.844306466969233e-9 m/s` | `1.9630813983926987e-10` | `PASS` |
| charge | `1.1818881019499895e-7 e` | `2.3758877887303242e-7 e` | `4.670220207738041e-10` | `PASS` |

The current eval-v3 registered budget SHA-256 is
`ab3713fb54aba02f7a208920e377b1fb90daaba3e129a04e57da8b4b3048b7b5`;
the 9/9-PASS comparison result SHA-256 is
`9fa50b0481ea662c07a64ba258977d0004831a50636c10d9ba5c96f1d94f90bf`.

The compact authority is
[`evidence/m3c1/case_a_100nm_common_p1_v1/`](../../../evidence/m3c1/case_a_100nm_common_p1_v1/).
This certifies same-field solver agreement only for this Case-A 100 nm,
Brownian-off, pre-event slice. It does not certify native-field equivalence,
field-production accuracy, physical applicability, boundaries, Brownian motion,
30 ms trajectories, other cases/sizes/variants, or universal COMSOL accuracy.

## M3-C1 common-P1 material event and event v14

The common-P1 Case-A 100 nm comparison was extended through its first natural
wafer stick. The scope is Brownian off, 287 particles, common canonical
exact-connectivity P1, and the interval through 458.75 us only. Material-event
evaluation v5 passes 20/20 material gates and its embedded pre-event prefix
passes 9/9. Event-time, hit-position, and terminal-charge absolute differences
are `2.157542807607049e-13 s`, `2.683964162031316e-14 m`, and
`4.7283812421028415e-09 e`. Terminal velocity is not a cross-solver gate because
COMSOL Freeze and solver stick persist different velocity semantics.

The v13 baseline reused the global absolute safety enclosure as the event BVH
query bound. Its query/refinement/accepted/depth counters were
`16,427,517 / 7,792,306 / 8,635,211 / 16`; 7,623,460 of 7,792,306 refinements
(`97.8331703092769%`) had already accumulated at the 450-us checkpoint. Event
v14 uses current dense Bernstein position/velocity bounds as the event
broad-phase query authority only for valid `rk4_dense` rows whose global field
support was independently proven. The global enclosure remains authoritative
for shortened-stage, field-support, applicability, and acceptance safety.
Invalid dense bounds or unproven global support use the global query fallback.
The v14 material candidate reports `842,927 / 11 / 842,916 / 11` and zero
failures.

The operator-observed shell wall-time on the same local machine changed from
approximately 14m13s (about 853 s) to 36.5 s, about 23.4x. These are approximate,
machine-local, non-gating observations, not solver-reported or manifest timings.

The solver-only v14 0.625/0.3125/0.15625-us rerun reports
query/refinement/accepted counts of `206,927/0/206,927`,
`413,567/0/413,567`, and `826,847/0/826,847`. Candidate self-convergence passes;
all three quantities are `ORDER_EVALUATED`. Position, velocity, and charge RMS
orders are `2.029875353701904 / 2.0816971911764033 / 2.044084026475049`, and
fine-pair relative-L2 changes are
`6.099791486063973e-8 / 8.321356016032579e-8 / 1.3796067752988052e-8`.
The report SHA-256 is
`ca0ef84cda1f1ddfc2b5eecc357eced4cb3648b48d582de10195c825bbbf38c7`.
The approximately second-order observation is empirical for this piecewise-P1,
mesh-crossing case; it neither proves nor disproves formal fourth-order RK4.

No COMSOL study was rerun for v14. The change is confined to the solver's event
BVH broad phase, while the common-P1 COMSOL input, reference, and source MPH are
hash-locked and unchanged. The existing reference was reused and only the v14
candidate comparison was recomputed. Compact evidence is in
[`evidence/m3c1/case_a_100nm_material_event_v1/`](../../../evidence/m3c1/case_a_100nm_material_event_v1/).
This does not certify native-field parity, physical validity, Brownian behavior,
30-ms behavior, other cases/sizes/variants, or universal COMSOL accuracy.

## M3-C0 isolated boundary-semantics probe

The external `run_m3c_boundary_semantics.ps1` workflow isolates terminal wall
meaning from fields and forces. It loads an unchanged copy of the audited theory
MPH, disables every force and dynamic charge, and launches one particle normally
at boundary 37 (`Freeze`) and one normally at boundary 35 (`Disappear`). It uses
classical fixed-step RK4 at 10, 5, and 2.5 microseconds, exports 61 frames from
0 to 150 microseconds at 2.5-microsecond intervals, and never imports the
production solver:

```powershell
./tools/vv/comsol/run_m3c_boundary_semantics.ps1
```

The current v2 normalizer first requires the exact six configuration receipts
for the two scenarios at all three step sizes from the COMSOL process log. A
missing, duplicate, malformed, or mismatched receipt fails closed before the
scientific gates are evaluated. It then verifies every active frame against
`x = x0 + v0*t` and `v = v0`, the analytic normal-impact event, the single
active-to-terminal transition, the terminal status, and the saved post-event
payload. The maximum active-frame position and velocity errors over all runs
are `4.726604209672303e-16 m` and `1.7763568394002505e-15 m/s`, respectively,
both below their `1e-12` limits. Both analytic impacts occur at 73 microseconds;
because the output grid is discrete, the first terminal frame is 75
microseconds.

| scenario | terminal observation | h/h2/h4 event-time spread |
|---|---|---:|
| boundary 37 `Freeze` | status 2; R-Z remains fixed at the hit point; saved velocity exactly retains the pre-hit value | `3.07371315899641e-17 s` |
| boundary 35 `Disappear` | status 4; saved position and velocity are NaN after the event | `2.71050543121376e-20 s` |

The report contains 56 `PASS`, zero `FAIL`, and six
`CHARACTERIZED_NOT_GATED` velocity rows. Velocity is deliberately characterized
rather than made an acceptance criterion. The Disappear hit coordinate is an
explicitly labelled analytic reconstruction because COMSOL no longer exposes a
finite post-event coordinate; it is not presented as a directly observed event
point. The source hash remains
`3bbf08e3469758313eac5de473a7a0dd4cc9a6f72c9722393229f0b856e9b524`.
Compact evidence is in
[`evidence/m3c0/boundary_semantics_v2/`](../../../evidence/m3c0/boundary_semantics_v2/).
The v1 evidence remains as history: its scientific observations were not
invalidated, but v2 supersedes it because v1 did not fail closed on exact
configuration receipts or gate the complete active free-flight path.

This probe certifies only isolated COMSOL Freeze and Disappear semantics. It does
not certify production boundary parity, grazing or corner handling, multiple
hits, full-physics trajectories, or native-field/P1 equivalence. No production
source was changed for it. Freeze is now known to differ from both deposition
and escape. The producer-neutral `hold`/`held` P18-H model has since been
completed through analytic/public regression first and this external candidate
second.

The P18-H candidate reuses the locked fine Freeze run
`freeze_inlet_37/dt_2p5us`; COMSOL was not rerun. It compares one force-free
particle released at `(r,z)=(0.23927,0.115) m` with `(v_r,v_z)=(10,0) m/s`
against the production schema-2 result. All 15 preregistered gates pass. The
candidate event time is `7.2999999999998075e-05 s`, the locked COMSOL event
time is `7.3000000000041497e-05 s`, and the aligned held-position maximum
difference is `2.776e-17 m`. The candidate contains 30 active and 31 held
saved frames, with zero held-position spread. Compact evidence is in
[`evidence/p18h/hold_freeze_v1/`](../../../evidence/p18h/hold_freeze_v1/).
This closes only force-free normal-impact terminal semantics. It does not make
COMSOL a golden truth or certify full physics, grazing/corner impacts, or a
paused particle that can later resume. COMSOL charge was not exported, and the
zero-charge external gate is only candidate self-consistency; nonzero retained
charge is verified separately by the public core scenario.

The no-clobber candidate and evaluator can be reproduced into disposable
external directories with:

```powershell
uv run --locked python -m tools.vv.comsol.run_p18h_hold_candidate tools/vv/comsol/cases/p18h_hold_freeze_v1.json _out_p18h/reproduction_candidate
uv run --locked python -m tools.vv.comsol.evaluate_p18h_hold_freeze tools/vv/comsol/cases/p18h_hold_freeze_v1.json _out_p18h/reproduction_candidate _out_p18h/reproduction_evidence
```

The evaluator verifies locked-reference and candidate hashes, schema and
algorithm revisions, the frame grid, event cardinality, and all 15 gates before
publishing evidence. Existing output directories are never overwritten.

The material-event anchor, native-field/exported-P1 difference localization,
P18-H, and B03 core are now closed. B03 did not require a COMSOL rerun. The
later 100 nm, 30 ms candidate-first study supersedes the former requirement to
run a full 12-case, three-step COMSOL matrix next. Charge-stable coupling and a
work-scaled durable commit cadence now precede any broader external campaign.
The old estimate of roughly `3e8` accepted pieces per long run was based on v13
artificial subdivision and is historical, not a current planning basis.

## Existing 100 nm, 30 ms reference characterization

`evaluate_existing_30ms_reference.py` reads the completed Case A/P candidate
`h,h/2,h/4` results and the already-saved COMSOL histories. It does not run
COMSOL, import the solver package, or alter either source. The candidate's own
self-convergence is the primary numerical result; the cross-representation
comparison is descriptive.

The saved COMSOL configuration is explicit fixed RK4 at 10 microseconds with
manual stepping and no time adaptation. It also has Brownian forcing enabled,
whereas this candidate matrix is Brownian off, and it uses COMSOL native FE
fields rather than the candidate's exported P1 representation. Therefore the
tool reports particle-wise path parity as not valid and does not tune the core
to this reference.

Reproduce into a new output directory:

```powershell
uv run --locked python tools/vv/comsol/evaluate_existing_30ms_reference.py `
  --config tools/vv/comsol/cases/m3c1_theory_100nm_30ms_v2.json `
  --candidate-root _out_m3c1/theory_100nm_30ms_candidate_v3 `
  --candidate-characterization _out_m3c1/theory_100nm_30ms_eval_v4/candidate_characterization.json `
  --output-directory <new-output-directory>
```

The compact authority is
[`evidence/m3c1/existing_30ms_reference_v1/`](../../../evidence/m3c1/existing_30ms_reference_v1/).

## External R-Z trajectory overlays

`plot_matched_trajectories.py` is an external V&V renderer. It imports no
solver package and reads only normalized long-form trajectory CSVs. It first
requires exact particle-ID identity, then draws each side's finite observed
path independently: COMSOL as a solid orange line and the solver as a dashed
blue line in physical R-Z space. Missing escape suffix coordinates are never
fabricated. Metrics use only the intersection of finite
`(particle_id, time_s)` keys. Coordinates are displayed in millimetres while
recorded metrics remain in SI.

Each invocation creates one all-particle R-Z SVG/PNG, one R-Z-only SVG for
every particle ID, an HTML atlas, and a hash-bearing receipt. It intentionally
does not create difference, `r(t)`, or `z(t)` figures:

```powershell
uv run --locked python tools/vv/comsol/plot_matched_trajectories.py `
  --candidate <solver_trajectory.csv> `
  --reference <normalized_comsol_trajectory.csv> `
  --output-directory <new_output_directory> `
  --label "<matched-case label>"
```

Scientifically matched inputs include the separate reduced Case-A 100 nm
deterministic companion and the full-physics common-P1 pre-event diagnostic.
The later Case A/P 100 nm, 30 ms candidate is deliberately Brownian off and
uses exported P1 fields, so it is not a same-setting candidate for the original
Brownian-on native-field histories. Those single-seed stochastic paths do not
support particle-wise identity. Core model availability and candidate
self-convergence must not be reported as cross-representation agreement.

## Interpretation limits

### P18-C saved-primitive charge-rate parity

`evaluate_aggregate_charge_parity.py` calls the production
`aggregate_relative_drift_regularized_two_current_v1` model at every saved
active primitive state in the 12 reference packages. It compares the returned
`dZ/dt` with the already exported rate using the sum of the ion/electron
collection magnitudes as the normalization scale. It does not rerun COMSOL:

```powershell
uv run --locked python tools/vv/comsol/evaluate_aggregate_charge_parity.py `
  ../../model_dataset/cf4_o2_etch_caseA_nonlinear_sass `
  evidence/p18c/frozen_rate_parity_v2
```

The output directory is no-clobber. The report keeps two gates separate. The
strict production-core gate derives `phi1` from screening with the core's
`epsilon0`; it remains `FAIL` at the preregistered `1e-10` limit. Replaying the
same two-current formula with the exported `phi1` is `PASS` and diagnoses the
strict residual as the `6.8066e-10` epsilon-constant convention difference,
not as evidence of a different collection formula. Neither gate certifies
integrated charge, trajectory accuracy, continuous-path applicability, or the
provider dataset as golden truth.

### P18-I saved-primitive ion-drag force parity

`evaluate_aggregate_ion_drag_parity.py` reads the same 12 saved histories and
separates producer-native equation replay from the production-canonical model.
It does not run or save COMSOL:

```powershell
uv run --locked python tools/vv/comsol/evaluate_aggregate_ion_drag_parity.py `
  ../../model_dataset/cf4_o2_etch_caseA_nonlinear_sass `
  evidence/p18i/frozen_force_parity_v1
```

The output directory is no-clobber. Exit status gates only the native saved
equations, which replay all 397,820 active rows at a maximum normalized
residual of `1.7901e-15`. The production relative-flow comparison is a separate
strict gate and preserves the known COMSOL/core `epsilon0` convention mismatch
as `FAIL`. The image comparison is always reported as
`DOCUMENTED_MODEL_DEFINITION_DIFFERENCE`: saved Case P and Case A use different
scalar ion-speed authorities, while production deliberately uses one
producer-independent vector norm. It is not thresholded into false agreement.
The report does not certify integrated trajectories, model applicability,
boundaries, or stochastic behavior.

- Applicability fractions are sampled-row diagnostics. Production acceptance
  still requires continuous-path certification by the solver.
- `integral_abs_force_dt_{median,p90,p99,max}_N_s` approximates
  `integral ||F|| dt` by trapezoids on the nonuniform saved output grid
  (10 us, 100 us, then 1 ms intervals), not the internal 10 us solver grid.
  It is relevance evidence, not a verified impulse quadrature or the norm of
  net vector impulse. Brownian output is reported separately as a single-seed
  sampled noise scale and receives no pass/fail interpretation.
- Charge-rate and Epstein-force formula parity proves data/model provenance
  only. It does not prove that either closure is physically adequate or
  applicable along a production trajectory. The charge dataset does not
  exercise its exponent-clamp or ion-energy-floor branches.
- The saved trajectories include ion drag, thermophoresis, R-Z Brownian
  forcing, dielectrophoresis, and lift in one coupled composition. P18-D now
  implements the producer-neutral quasistatic spherical DEP family, but the
  saved DEP primitive lacks the complete solution/averaging/recovery and
  point-dipole certification required to identify that production revision.
  P18-L now implements the producer-neutral R-Z/no-swirl
  `rarefied_vorticity_sensitivity_rz_v1`, but the saved packages lack the
  trajectory-local signed vorticity and solver-step provenance needed to
  certify its isolated pointwise or native-field trajectory parity. The later
  M3-C1 common-P1 composite slice passes, but does not close that item or the
  lift model's physical validity. B02 also does not support the saved R-Z
  all-force stochastic composition. Full replay of the original trajectories
  therefore remains `NOT_APPLICABLE`. The later deterministic exact-P1
  companion is a separate artifact with a narrower common-physics scope; it
  does not relabel the originals as applicable.
- The reference release grid is internal rather than surface-originated, and
  the models contain no positive reflection example. Boundary and stochastic
  validation remain `NOT_TESTED`.

### P18-R neutral-transport closure audit

`evaluate_neutral_transport_closure.py` performs the no-rerun P18-R audit from
the locked 12 packages and source identities. It does not import solver core,
run COMSOL, or modify the source MPH files:

```powershell
uv run --locked python tools/vv/comsol/evaluate_neutral_transport_closure.py `
  --config tools/vv/comsol/cases/p18r_neutral_transport_closure_v1.json `
  --output evidence/p18r/neutral_transport_closure_v1
```

The output directory is no-clobber. The accepted artifact records 12 packages
and 397,820 active rows. Native COMSOL linear Epstein replay is `PASS` with a
maximum vector relative residual of `1.05859e-15`. The existing P15-E and P16
physical-applicability columns are `NOT_APPLICABLE` for all 12 packages because
the saved CF4/O2 mixture does not establish their molecular-species, surface,
and accommodation authority. The Waldmann gradient/heat-flux algebraic identity
passes, but trajectory thermophoretic-force replay is `NOT_TESTED`: saved PPR
rows contain neither translational conductive heat flux nor trajectory-local
PPR temperature gradient.

The audit uses effective mixture mass only as a numerical sensitivity. It does
not establish species-resolved mixture truth. Saved output frames also do not
certify all integrator stages or the continuous accepted path. The resulting
`epstein_linear_effective_gas_sensitivity_v1` and
`waldmann_gallis_free_molecular_effective_gas_heat_flux_sensitivity_v1` enable
producer-certified pseudogas reference/sensitivity runs with explicit
`maximum_speed_ratio in (0,1]`; they are not COMSOL branches or trajectory
certificates. No COMSOL study was rerun for this audit.

P18-R itself remains a historical offline audit. The later M3-C0b v6 run
sequentially confirmed only the Case-A 100 nm pre-event operational fixed-step
selection, and the M3-C1 PPR supplement subsequently closed frozen saved-state
producer-form replay 8/8. P19-L then removed the global-certificate false
rejection without relaxing the model gates. The exported-P1/native-field
comparison is complete and its six failed gates have activated the full-physics
exact-connectivity common-field COMSOL diagnostic. That diagnostic subsequently
passed all nine preregistered trajectory gates for the narrow Case-A 100 nm
pre-event scope. The later isolated boundary probe closed only COMSOL-side
Freeze/Disappear meaning. The common-P1 first material event and field
representation localization, P18-H, B03 core, charge-stable coupling, and
work-scaled durable I/O cadence are now closed. The current order is limited to
external cases whose physics, fields, boundaries, and stochastic meaning can
be matched. M3-C2 uses independent-seed ensembles rather than
single-path equality. The saved M3-C1 v14 evidence and evaluator remain
historical. A completed core model, matching formula shape, or v6
step-selection PASS must not be reported as native-field, production-boundary,
stochastic, or universal COMSOL trajectory agreement.

## M3-C2 anchor inventory preflight

The first stochastic comparison is gated by a fail-closed inventory preflight:

```powershell
uv run --locked python tools/vv/comsol/m3c2_anchor_preflight.py `
  --config tools/vv/comsol/cases/m3c2_theory_caseA_100nm_anchor_v1.json `
  --repository-root ../.. `
  --output evidence/m3c2/anchor_preflight_v1
```

Exit code zero means only that the report was generated; readiness is the
manifest's `overall_status`. The accepted report is
`BLOCKED_MISSING_MEANING_MATCHED_COHORT / NOT_AUTHORIZED`. It SHA-checks the
source model and selected Case A/P 100 nm artifacts, confirms the two disjoint
32-seed reservations, and records that the saved histories are one native-field
Brownian run per case whose effective random stream is unverified. Active Case
A/P samples with finite Brownian values number 28,354/34,727. The exported
feature table reports `r/phi/z` values and nonzero `Fbphi`, but the particle
interface has out-of-plane motion disabled and solves only R-Z motion; the phi
column is not evidence of a third motion DOF. The saved configured parameter
values 21/1 do not prove the stream used under `GenerateUnique` mode.
The read-only source-model authority is
[`../../../evidence/m3c2/model_semantics_probe_v1/`](../../../evidence/m3c2/model_semantics_probe_v1/README.md).

`run_matrix.csv` is only a provisional final-cohort seed allocation: 32 common-P1
RZ COMSOL companion rows and 32 B03 rows for Case A. It does not lock the 30 ms
run, input hash, release, boundary, physics revisions, COMSOL companion recipe,
candidate case, or the disjoint step-sensitivity pilot. Those belong to the next
compact execution contract. No current COMSOL result is promoted to golden truth.
That contract retains built-in `bf1` with the model's Epstein-equivalent effective
viscosity, explicitly selects `UserDefined` random-number mode, and gives `bf1.i`
the sole replica-seed authority. COMSOL and B03 use independent step/tolerance
convergence studies at common output times; equal fixed steps are not required.

The input and semantic portion of that contract is now locked with:

```powershell
uv run --locked python tools/vv/comsol/lock_m3c2_caseA_100nm_pilot_contract.py `
  --config tools/vv/comsol/cases/m3c2_caseA_100nm_stochastic_pilot_v1.json `
  --repository-root ../.. `
  --output evidence/m3c2/caseA_100nm_pilot_contract_v1
```

The receipt is
[`../../../evidence/m3c2/caseA_100nm_pilot_contract_v1/`](../../../evidence/m3c2/caseA_100nm_pilot_contract_v1/README.md).
It validates nine source artifacts, the canonical HDF5 content hash, release,
geometry, boundary/physics revisions, R-Z Brownian target, and disjoint pilot
seeds. Its status is `PASS_INPUT_IDENTITY_AND_SEMANTICS_LOCKED`, but execution
remains `NOT_AUTHORIZED`, the pilot is `NOT_RUN`, and accuracy is
`NOT_EVALUATED`. Executable COMSOL/candidate recipes and the independent
observable-convergence pilot are the next unit.

The paragraph above records the input-lock milestone. It is superseded for the
current status by the completed final campaign below.

## M3-C2 Case-A 100 nm final campaign

The numerical closeouts below are immutable historical evidence. New current
comparisons require the hash-bound source/model/field/actual-receipt association
described in Generic meaning preflight. Current recertification is recorded in
[`../../../evidence/comsol_binding_recert_2026_10_09_v1/`](../../../evidence/comsol_binding_recert_2026_10_09_v1/README.md).
Historical scientific PASS does not certify a changed producer, axis recovery,
or post-interpolation drag/Brownian coefficient.

The validated runners now execute the common-P1 companion without changing the
production solver core. COMSOL uses `ModelUtil.loadCopy`, `-nosave`, `-error on`,
an isolated preference directory, one process, `UserDefined` Brownian mode, and
`bf1.i` as the replica-seed authority. Every final replica is checked to ensure
that only `fptas` is enabled in the particle study. The candidate runner uses the
production Brownian engine and rejects legacy or unregistered final settings.
Runner v6 treats the recipe's `physics.noise_revision` as execution authority
and accepts only `inertial_langevin_fdt_epstein_linear_midpoint_2d_v2`. It
checks that field before reading the locked contract/input or creating output,
then copies the same value into generated cases and preparation/final manifests.
The checked-in M3-C2 recipe JSON files remain immutable historical v1 snapshots;
they are not executable inputs to runner v6 and must not be edited in place. A
future run needs a new recipe file that explicitly declares v2 and uses current
case-format-3 inputs and a format-3 template. The evaluator
binds historical runner v4 to the historical RZ revision and runners v5/v6 to v2,
while rejecting a mismatched or mixed cohort. Historical runner-v4 campaigns
retain evaluator revision v5 only in their immutable evidence. Every new
revision-5-policy evaluation emits evaluator revision v6, including a rerun of
a historical cohort, so the changed acceptance contract cannot reuse a
hash-locked historical tool identity. Historical recipe JSON and evidence
hashes are not rewritten.

For a current candidate replay, `execution.expected_executor` records exactly
`source_sha256`, `uv_lock_sha256`, `python_version`, and `distribution_version`.
The supported execution form is the installed uv source project. The source
digest covers the sorted relative names and raw bytes of all package `.py`
files, including uncommitted edits; package version or Git HEAD alone is
insufficient. Print that record before registering a new recipe:

```powershell
uv run --locked python tools/vv/comsol/run_m3c2_candidate_pilot.py executor-identity
```

`execution.expected_revisions` names case/result schema and result, engine,
compiled tile, physics catalog/runtime, RNG, boundary, Brownian RNG, joint OU,
OU split, Brownian composition, charge-dense, and tree-policy revisions.
`prepare` verifies input hashes, strict YAML syntax, current template format,
the installed identity, and static loading of the template and generated
level settings before creating its output directory. YAML duplicate keys at
any depth and merge keys are rejected by the core `yaml_input.parse_document`
owner. `run-cell` accepts only a v6 preparation and rechecks the installed
identity before execution and after opening the result. Its receipt records
the actual manifest revisions after the ordinary
`load_case -> simulate -> open_result` path and rejects a mismatch before
writing derived projections.

`recover-cell` and `renormalize-cell` do not run the solver and do not require
the current source digest to equal the original executor. They use the
original prepared lock and saved manifest identity/revisions, current reader
compatibility, and projection rules. Renormalization verifies the original
artifact hashes and preserves them in the supersession record. New receipts
name runner v6 and retain `source_runner_tool_revision`. Historical input and
normalized evidence remain hash-audit material; their old canonical schema is
not passed to the current input reader, and historical preparations never
authorize a new `simulate`. Unsupported saved-result schemas still fail in
the current result reader. These checks establish replay integrity, not a
COMSOL or physical-accuracy certification.

Policy revision 3 treats the disjoint pilot seeds only as configuration
screening. It selected COMSOL classical RK4 at 20 us and candidate 20 us with
Brownian interval-tree depth 3 and geometry rtol 1e-8. The independent final
cohort contains 32 COMSOL and 32 candidate seeds, 287 fixed source particles,
121 common output times, and 30 ms of simulated time.

The final evaluator returned `PASS`. Its sole confirmatory gate is the four
terminal-population curves (`stuck`, `held`, `escaped`, and `any_terminal`) over
all 121 times. The maximum observed difference is 0.00598868 and the 95%
simultaneous Hoeffding radius is 0.0327841, giving an upper bound of 0.0387728
inside the preregistered 0.05 margin. Mean/covariance/quantile position summaries,
fixed-bin R-Z occupancy, and the 287-source seed-mean R-Z overlay are descriptive
and cannot change the confirmatory result.

The durable authority is
[`../../../evidence/m3c2/caseA_100nm_final_campaign_v1/`](../../../evidence/m3c2/caseA_100nm_final_campaign_v1/README.md).
This certifies only the locked common-P1 Case-A 100 nm terminal-population
accuracy. That Case-A result does not itself certify pathwise equality, native
fields, Case P, the second ion-drag model, other sizes, 3-D, or universal COMSOL
equivalence.

Performance remains a separate decision. Candidate timing from this final cohort
is marked non-authoritative because part of the run overlapped external COMSOL
diagnosis. The current performance order is superseded by the completed Case-P
campaign below.

## M3-C2 Case-P 100 nm final campaign

The meaning-matched common-P1 Case-P campaign completed at 20 us with 32
independent COMSOL seeds and 32 independent candidate seeds. Every replica has
287 fixed source particles, 30 ms of simulated time, and 121 common frames.

The revision-5 evaluator returned `PASS`. The preregistered 83-category
full-population R-Z/fate gate had maximum empirical total variation
`0.010670731707317093`; its simultaneous confidence bound was
`0.13119968456545308`, inside the `0.15` margin. The terminal-population gate
also passed, but no terminal event occurred in any of the 64 replicas, so that
gate is uninformative for Case-P boundary parity.

The durable authority is
[`../../../evidence/m3c2/caseP_100nm_final_campaign_v1/`](../../../evidence/m3c2/caseP_100nm_final_campaign_v1/README.md).
The source Case-P COMSOL `auxq` intentionally uses electron and positive-ion
currents while keeping negative-ion density diagnostic; the candidate selected
the same two-current model. The result therefore certifies the registered
same-form population observable, not a species-resolved charging truth or the
later optional three-current extension. It also does not certify pathwise RNG
equality, eventful boundary parity, native-field parity, or universal COMSOL
equivalence. The comparison remains external V&V and does not add a COMSOL path
to production.

The candidate final used four overlapping external processes. Its timing is
therefore `NON_AUTHORITATIVE_EXTERNAL_WORKLOAD_OVERLAP` and is not a performance
baseline. Owner discovery is now complete for the accepted candidate seeds
`319032`, `319047`, and `319063` at 287 particles. Every rerun exactly preserved
the accepted scientific payload, work, case identity, and algorithm revisions.
`integrators` dominated all three profiles with 42.58--42.86% self-time shares,
but it was not a preregistered bounded owner, so
`optimization_authorized=false` and production remains unchanged. The durable
profile is [`../../../evidence/m3c2/caseP_100nm_owner_profile_v1/`](../../../evidence/m3c2/caseP_100nm_owner_profile_v1/README.md).

The Case-A/Case-P 100 nm anchor benchmark is
`CLOSED_ACCEPTED_WITH_LIMITATIONS`. Additional COMSOL runs, seeds, packages,
10,000-particle-or-larger scales, and process scaling are not benchmark exit
criteria. Product-scale performance may be opened as a separate work package
only after its target hardware, particle count, output mode, wall-time limit,
and memory limit are specified. Its bounded procedure is owned by
[`../../../docs/parallel_execution_plan.md`](../../../docs/parallel_execution_plan.md).

## P21 / M3-C3 critical 2-D boundary microcase

Priority 2 is complete and `PASS`. One from-scratch 10 mm by 20 mm
axisymmetric rectangle runs three force-free particles from identical initial
states in COMSOL 6.4 and the production public API:

- departure from exact lower-surface contact into the domain;
- one specular outer-wall reflection, including the same-step residual flight;
- passage through the R-Z axis without a material-wall event.

Fixed RK4 steps of 1, 0.5, and 0.25 ms produced 132/132 passing registered
checks against the analytic piecewise trajectories. The largest direct
COMSOL/API position difference was `1.61339e-17 m`. The durable authority is
[`../../../evidence/m3c0/critical_boundaries_v1/`](../../../evidence/m3c0/critical_boundaries_v1/README.md).

That v1 evidence remains an immutable historical snapshot. The current
reproducer writes the surface-start particle as a canonical realized surface
row (`facet_id` plus a strict-interior facet parameter); its YAML contains only
the standard `type: surface` / `table:` reference. A rerun must use new output
and evidence destinations and does not rewrite the v1 artifact.

This result does not cover grazing or corners, multiple material hits,
probabilistic laws, finite-radius contact, forces, fields, or native-field
equivalence. It is the single boundary artifact for this closeout, not the
start of a boundary-case matrix.

P21 priority 1 is complete. The explicitly selected aggregate
singly-negative-ion collection revision uses
`dZ/dt = Gamma_+ - Gamma_e - Gamma_-`. Its inputs are
explicit negative-ion density, thermal voltage, velocity, and effective mass;
the existing screening-length field remains the only screening authority.
Zero negative-ion density reproduces the existing two-current rate, Jacobian,
and global bounds exactly. The selected three-current revision still validates
its negative-ion fields and relative-speed applicability at zero density, so
full public-run identity is conditional on both revisions remaining applicable.
This is not a species-resolved current model and does not alter the
original Case-P anchor. It is integrated as catalog v17, runtime
`signed_ion_compiled_physics_runtime_v19`, and compiled tile v18 in the same
single engine; the standard verification/scenario suite and all quality gates
passed.

Priority 3's input blocker is resolved. A producer-owned domain cache plus
one-sided domain-3 boundary cache generated the finite canonical five-primitive
negative-ion field on all 1987 common-P1 nodes. The source MPH remained hash
unchanged; no coordinate nudge, missing-value imputation, or alternate-domain
fallback was used. Priority 4 is also complete: the shared three-current initial
charge drove a common-P1, Brownian-off, 100 nm, 287-particle, 30 ms comparison.
Candidate and COMSOL three-level self-convergence, position/charge over common
finite lifecycle states, common-active velocity comparison, and 141 terminal
event identities/times all pass. The original two-current
Case-P anchor is unchanged. The compact authority and current status are
[`../../../evidence/m3c3/caseP_three_current_companion_v1/`](../../../evidence/m3c3/caseP_three_current_companion_v1/README.md).
This optional-model external coverage is separated from the P21 exit. P21 and
the explicitly scoped two-current common-P1 2-D benchmark are
`CLOSED_ACCEPTED_WITH_LIMITATIONS` / `2D_CRITICAL_VV_COMPLETE`. The result does
not certify native FE equivalence, Brownian pathwise agreement, species-resolved
physics, arbitrary geometries/conditions, physical-model validity, or universal
COMSOL equivalence.

## Current native binding and batch completion

`CommonP1Epstein.java` forms the single coefficient from interpolated density
and temperature. The native metadata control observed that `importData()` clears
previous argument units. C2/C3 set function and argument units after import,
then preserve actual API getter values and independently evaluate initial
coefficient/FDT numbers. SI metadata and numerical agreement are separate facts.

For the registered COMSOL 6.4 class-input profile, `.class.status=Error` also
appears for successful empty-model keep/remove and returning-model controls.
It remains raw evidence. The wrapper requires process completion, no fatal
native log, exactly one registered completion record, the expected native
artifacts, unchanged source/request/producer hashes, and successful normalization.
A real adversarial class printed a valid completion record and then threw;
COMSOL returned exit 0, and the current fatal-log guard rejected it. The
[compact native closure](../../../evidence/comsol_binding_recert_2026_10_09_v1/native_unit_and_completion_closeout.json)
binds the controls and successful source-preserving campaigns. This does not
infer COMSOL's internal class-status implementation.

Actual boundary response and canonical geometry meaning remain separate.
Retained Freeze/Stick entities can be exported with `bndenv(dom)`; the native
Disappear control loses that ID. A `terminal_status` row may therefore have
an empty semantic group. The population evaluator consumes its observed
status/time without assigning an inlet/pump cause; `terminal_boundary` rows
still require their semantic group. Population gates do not certify semantic
boundary-event parity.

Choose canonical `contact_geometry` from the actual comparison profile. Its
`center` option preserves body/drag/electrostatic radius and changes the contact
detector only. Do not infer this choice from Stick/Freeze/Bounce. Current C2/C3
canonical sources have zero contact radius, so their frozen existing settings
are equivalent and are not modified during the campaign.

Current C3 candidate event writing uses actual boundary ID plus the canonical
group map. `reproject-events` repairs only saved reporting metadata and writes
a separate raw-result/input/original-receipt/producer hash receipt; it never
claims that the original executor used repaired code. Its evaluator verifies
that receipt before selecting the repaired CSV. Native total Ftr/Ftz from the
same fine solve is separate from configured per-force reconstruction and does
not certify individual contributions, auxiliary-charge assembly, or all stages.

Current C2 policies use fresh independent pilot/final seed arms. Four-seed
screening selects the first passing macro level in the registered descending
step order. Final Hoeffding/TV gates assume standard PRNG sampling, particle
stream separation, a fixed source, and one-way noninteracting dynamics. They
count 32×287 units and apply a union bound across 121 times. The two anchors
share alpha 0.05; each receives 0.025, split equally between terminal curves
and the 83-category RZ/fate gate. A failed upper-bound gate means registered
equivalence was not established. Continuous OU noise time-law and continuous
discretization bias remain separate, unproved scopes.

The current registered CaseA and CaseP confirmation completed all 128 fresh
final seed runs. Both fixed population gates passed: terminal simultaneous
bounds are 0.040128262937 / 0.035010667118 against 0.05, and RZ/fate TV bounds
are 0.130735400519 / 0.129537665328 against 0.15. Independent aggregation of
128 CSV files agrees with the public evaluator. Every native replica has a
unique actual readback with the requested seed, 20 microsecond SI timestep,
and UserDefined RNG getter. The
[current evidence](../../../evidence/comsol_binding_recert_2026_10_09_v1/README.md)
preserves original metadata views and hash tables alongside explicit meaning
attachment receipts. CaseP has no observed terminal events; CaseA Disappear
is only a population status/time observation. No boundary cause is inferred.

## Cartesian XY meaning suite and curved-wall line2 convergence

`run_xy_minimal_suite.ps1` creates seven fresh, unsaved COMSOL 6.4 models and
compares them with public-API candidate runs and independent analytic motion.
The cases isolate ballistic motion, constant electric acceleration, linear
drag, exact surface-origin departure, specular reflection, Stick, and
Freeze/`hold`.  Each case runs fixed RK4 steps of 20, 10, and 5 ms.  The runner
binds all raw CSV hashes to the registered Java, evaluator, runner, compiled
class, COMSOL version, and 21 configuration records before a scientific PASS
can be issued.

```powershell
uv sync --locked
.\tools\vv\comsol\run_xy_minimal_suite.ps1 `
  -OutputDirectory <fresh-output-directory>
```

The checked run is
[`../../../evidence/xy/cartesian_minimal_suite_v1/`](../../../evidence/xy/cartesian_minimal_suite_v1/evaluation/README.md).
All four meaning judgments pass.  The linear-drag candidate errors converge at
observed order 4.02 for position and 4.94 for velocity; at 5 ms the direct
COMSOL/candidate differences are below `3.0e-15 m` and `1.0e-15 m/s`.
COMSOL's event-coincident state export and the candidate frame use opposite
velocity continuity at exactly 0.25 s, so that single velocity row is excluded;
pre/post trajectory rows, terminal tails, and the candidate event payload are
still gated.

`evaluate_curved_wall_line2_convergence.py` separately compares first-hit time,
point, normal, and reflected velocity with an analytic unit circle for
16/32/64/128 inscribed line2 facets:

```powershell
uv run --locked python tools/vv/comsol/evaluate_curved_wall_line2_convergence.py `
  <fresh-output-directory>
```

The checked result is
[`../../../evidence/xy/curved_wall_line2_v1/`](../../../evidence/xy/curved_wall_line2_v1/README.md).
Time/point converge at order 2.003, normal at 0.998, and reflected velocity at
0.997.  These two workflows do not certify COMSOL event-table values, native
curved elements, finite-radius contact, grazing/corner/multiple-hit cases, or
arbitrary COMSOL models.
