# Certified canonical field cache

This installed client creates an explicitly selected field cache while preserving the original canonical
geometry, fields and realized sources. It uses the production sampler; it does not add a runtime fallback,
change physics, integrate particles or repair geometry.

```console
uv run --locked python -m chamber_particles inputs cache cache.yaml cached.h5 --report cache.json
uv run --locked python -m chamber_particles inputs cache cache.yaml cached.h5 --report cache.json --source-time-diagnostic
```

The same CLI ships in the wheel. Python uses
`chamber_particles_tools.field_preprocessor.preprocess_from_configuration(configuration_path, output_path, *, report_path=None, source_time_diagnostic_requested=False)`.
Outputs/reports are fresh paths. Use `cached.h5` as a format-1 recipe's `input_bundle`, explicitly select the new
field names in physics settings, then use the common `create` publisher and `check`. No YAML path/hash is hand-maintained.

## Format 2

```yaml
format_version: 2
input:
  data_path: source.h5
validation:
  memory_limit_mb: 128
  workspace_rows: 256
  max_patch_work: 100000000
target:
  kind: regular
  name: fast_grid
  axis0: {start_m: 0.0, stop_m: 0.02, count: 81}
  axis1: {start_m: 0.0, stop_m: 0.01, count: 41}
fields:
  - source: gas_velocity
    output: gas_velocity_cache
    limits:
      value_relative_l2: 0.001
      gradient_relative_l2: 0.01
      boundary_value_relative_l2: 0.001
```

Values here are an example, not default accuracy/resource budgets. Paths are relative to the configuration.
An existing fully supported regular target can be selected with `target: {kind: existing, layout: target_grid}`.
`validation` is required; workspace rows are at least 16. Field output names are explicit and collision-free.

## What the acceptance establishes

Supported sources are nodal P1 triangles, regular tensor layouts and affine Q1 quadrilaterals, static or
fixed-topology piecewise-linear time snapshots. Selected fields retain their units, components, basis and time knots.
The target is regular and fully supported. Warped Q1, holes/partial targets, unsupported association or unresolved
coverage are rejected; direct runtime capability does not imply this producer supports the same conversion.

The comparison uses common source/target cell partitions, exact coverage/multiplicity checks, positive quadrature,
true spatial-gradient norms and support-boundary values. XY uses area measure; RZ uses revolved volume measure.
Temporal norm gates cover the piecewise-linear intervals, not just snapshot values. Bounds include the explicit
roundoff/uncertainty treatment; a near-zero reference is not divided by an arbitrary epsilon.
RZ axis regularity is checked before publishing a cache. This is a spatial/linear-time representation certificate,
not an estimate of unsaved native temporal behavior or particle integration error.

Geometric work and memory have explicit bounds. One shared work counter covers selected fields, coverage and
intersection work; exhaustion or unresolved temporal subdivision refuses publication. Memory preflight precedes
large cache allocation and includes canonical reads, snapshots, layout/index/writer transients and bounded scratch.
The owned-array budget is not a cap on Python/JIT/OS RSS. There is no alternative interpolation, padding or
nearest-neighbor fallback when the declared certificate fails.

## Optional saved-time diagnostic

`--source-time-diagnostic` separately measures the sensitivity to omitting stored source knots.
Without the option it is `NOT_REQUESTED` and performs no extra sampling. It runs after the actual cache norm gates
with only remaining work and a separate memory preflight. `RESOURCE_LIMITED`/`UNRESOLVED` diagnostics do not
veto an otherwise certified cache and are not diagnostic PASS. The report keeps their resources and claim separate.
A saved-knot diagnostic cannot discover a pulse between observations or certify the particle macro step.

## Identity and use

The producer binds source/config/content identity, field lineage, layouts, selected norms and resources.
Unselected time fields and original source rows remain unchanged. A generated cache is used only when the case
explicitly selects it; runtime never switches automatically. Recheck particle sensitivity where field error matters.

```console
uv run --locked pytest tests/verification/test_field_preprocessor.py -q
```
