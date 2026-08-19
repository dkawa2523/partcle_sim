# CLAUDE.md

Guidance for coding agents working in this repository.

## What this is

`particle_tracer_unified` computes dust-particle trajectories through a field
that was solved elsewhere — normally a COMSOL vacuum-chamber model. It is a
one-way-coupled Lagrangian point-particle solver: particles read the field and
never change it. It does not embed or re-run COMSOL.

Supported coordinate systems: `cartesian_xy`, `axisymmetric_rz` (no swirl),
`cartesian_xyz`.

## Commands

```console
python -m pip install -e .                      # runtime deps only
python -m pip install -e ".[viz,validation]"    # plus visualisation / V&V

particle-tracer check CONFIG --full             # preflight, writes nothing
particle-tracer run CONFIG -o OUT_DIR           # preflight then run
particle-tracer artifacts OUT_DIR               # verify written artifacts
particle-tracer comsol build-case ...           # COMSOL export -> case
particle-tracer compare WORKFLOW ...            # field/trajectory/... V&V
```

Tests and quality gates:

```console
python -m pytest tests/ -q                      # full suite, ~10 min
python -m pytest tests/test_NAME.py -q          # one file
python -m ruff check particle_tracer_unified/ tests/
python -m ruff format --check particle_tracer_unified/ tests/
uv run --frozen nox -s quality-pr               # the full PR gate
```

Numba compiles on first use, so a single solver test can take ~40 s.

## Public API

Four operations, in this order. Nothing else is a supported entry point.

```python
from particle_tracer_unified import load_case, validate_case, simulate, write_result

case = load_case("run_config.yaml")   # parse + resolve, no side effects
report = validate_case(case, detail="summary")   # preflight, writes nothing
result = simulate(case)               # numerics only, writes nothing
manifest = write_result(result, "out_dir")       # the only writer
```

## Where things live

`docs/architecture.md` has the full module-ownership table. The short version:

| Path | Owns |
|---|---|
| `configuration.py`, `_configuration_*.py` | strict parsing of the run config |
| `core/` | geometry, boundary queries, field sampling primitives |
| `io/` | canonical tables, NPZ, COMSOL manifest → solver types |
| `providers/` | field and geometry provider construction |
| `solvers/` | the integrator, forces, collisions, charge, Brownian |
| `comsol_case/` | turning a COMSOL export into a case |
| `preflight*.py` | pre-run validation and its report schema |
| `compare/` | V&V workflows that read cases and artifacts |
| `external/comsol_icp_export/` | the Java exporter that runs inside COMSOL |

Reading order for the physics: `docs/physics_numerics.md`, then
`solvers/integrator_common.py` (ETD2), `solvers/drag_models.py`,
`solvers/_force_evaluators.py`.

For anything COMSOL-facing, read `docs/comsol_workflow.md` first — it is the
procedure — and `docs/comsol_vv.md` for the contract it enforces.

## Invariants

These are load-bearing. Breaking one turns a visible failure into a silent
wrong answer, which is the failure mode this codebase is built to avoid.

- **No silent fallback.** Missing, non-finite, or non-positive inputs raise.
  Do not substitute a default, clamp to an epsilon, or switch to another model
  when a value is unusable. A field quantity that is declared must be used; a
  quantity that is absent is a different case, not the same case with a guess.
- **No tolerance-shaped repair.** Do not nudge a position, widen a tolerance,
  or blend a fill value to make a check pass. Boundary tolerances are resolved
  once from geometry scale and float64 roundoff
  (`core/boundary_numerics.py`); do not introduce a fixed metre or second
  constant anywhere.
- **Safety and accuracy are different verdicts.** An unproven wall crossing or
  an unresolved support island must stop the particle. An accuracy budget
  running out must not. Keep the two apart when touching
  `solvers/segment_trace.py` or `_runtime_trace_refinement.py`.
- **Values and support answer different questions.** Field sampling outside a
  mesh clamps to the nearest element so a trial step stays finite; the support
  classification stays strict so the particle still stops. Do not merge them.
- **Preflight must agree with the runtime.** If `check` rejects what `run`
  integrates, the preflight stops being read. Fix the disagreement, not the
  message.
- **SI everywhere, units in the name.** Canonical CSV columns carry their unit
  (`mass_kg`, `drag_diameter_m`). `mass_kg` is the authority for inertia; it is
  never reconstructed from density and diameter.
- **Physics changes are not refactors.** A change to a numerical expression or
  a physical result needs its intent pinned by a contract test first.

## Options and their defaults

Defaults are chosen for COMSOL agreement. Change one only for a stated reason;
`README.md` has the table with those reasons.

| Setting | Default | Meaning |
|---|---|---|
| field storage | `--field-node-samples` (mesh-native) | keeps the solution on the COMSOL mesh; `--field-bundle` resamples onto a lattice |
| `physics.wall_interaction.contact_sliding` | `false` for COMSOL cases | COMSOL has no point-particle contact model |
| `physics.wall_interaction.max_hits_per_step` | `5` | wall events resolved per macro step |
| `time.max_substep_splits` | `4` (16 substeps) | adaptive refinement budget |

## Known limits

- Second-order COMSOL meshes (`tri2`/`quad2`) are rejected: P2 sampling is not
  implemented, and the mid-side node ordering has not been fixed against a real
  export. Do not implement it from a guessed ordering.
- No ion drag force and no near-wall drag correction.
- Dynamic charge coupled to electric force is Cartesian-2D regular-grid only.
- Brownian motion and Cartesian lift are rejected in `axisymmetric_rz`.
- 3D boundary-release projection is not implemented.
