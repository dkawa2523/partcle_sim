# Skill 09: Run Solver for COMSOL Comparison

## Purpose

preflight済みcaseを本solverで実行する。比較目的に応じて、minimal/standard/debug出力、deterministic設定、sharded実行を使い分ける。

## Before running

Confirm:

```text
preflight passed
mode selected correctly
seeds recorded
output mode chosen
reference data available for intended comparison
```

## Run types

### 1. First-step run

Use for force/integrator parity.

```powershell
py -3 -m particle_tracer_unified.compare.first_step_compare --config <case>/run_config.yaml --output-dir <out>/first_step --stochastic off
```

### 2. Short deterministic probe

Use for quick wall/field sanity.

```powershell
py -3 run_from_yaml.py <case>/run_config.yaml --output-dir <out_short>
```

Set short `t_end` through a copied config or supported override tool.

### 3. Full sampled run

Use sampled release for iteration. Record sampled scope.

### 4. Full reference or sharded run

Use when particle count is too large. Ensure root artifacts are collected.

## Output modes

```yaml
output:
  mode: standard
```

Use debug only when a comparison requires deep artifacts:

```yaml
output:
  mode: debug
```

## Required run artifacts

```text
solver_report.json
final_particles.csv
wall_summary_by_part.csv
prepared_runtime_summary.json
collision_diagnostics.json only if debug/requested
wall_events.csv only if debug/requested
```

## Pass criteria

- Solver completes without numerical boundary failure unless expected.
- Output artifacts match selected mode.
- Seeds and config are preserved.

## Fail criteria

- Debug output is always enabled for large runs without need.
- Sharded comparison is attempted without root artifact aggregation.
- Runtime improvement is accepted without checking wall/ensemble metrics.
